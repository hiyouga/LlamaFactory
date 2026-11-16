# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments

from llamafactory.hparams import FinetuningArguments
from llamafactory.train.kto.trainer import CustomKTOTrainer


def _check_metrics(rank, port, output_dir, use_cpu=True, world_size=2):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.manual_seed(42)
    config = GPT2Config(n_layer=1, n_head=1, n_embd=8, vocab_size=16)
    trainer = CustomKTOTrainer(
        model=GPT2LMHeadModel(config),
        ref_model=GPT2LMHeadModel(config),
        finetuning_args=FinetuningArguments(stage="kto"),
        processor=None,
        tokenizer=None,
        args=TrainingArguments(
            output_dir=str(Path(output_dir) / str(rank)), use_cpu=use_cpu, report_to=[], disable_tqdm=True
        ),
    )
    assert trainer.accelerator.num_processes == world_size
    device = trainer.accelerator.device
    try:
        for train_eval in ("train", "eval"):
            prefix = "eval_" if train_eval == "eval" else ""
            for case in ("opposite", "mixed", "chosen", "rejected"):
                totals = {}
                for step in range(2):
                    torch.manual_seed(100 + rank * 10 + step)
                    ids = torch.randint(1, 16, (2, 4), device=device)
                    if case == "opposite":
                        tags = [rank == 0, rank == 0]
                    elif case == "mixed":
                        tags = [True, step == rank]
                    else:
                        tags = [case == "chosen"] * 2
                    tags = torch.tensor(tags, device=device)
                    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids, "kto_tags": tags}
                    batch.update(
                        {f"kl_{key}": batch[key].clone() for key in ("input_ids", "attention_mask", "labels")}
                    )
                    with torch.no_grad():
                        logits, logps, _ = trainer.forward(trainer.model, batch)
                        _, ref_logps, _ = trainer.forward(trainer.ref_model, batch)
                    # Build the reference from per-example outputs, independently of metric packing/reduction.
                    local = {}
                    for split, mask in (("chosen", tags), ("rejected", ~tags)):
                        local[f"count/{split}"] = mask.sum().item()
                        local[f"rewards/{split}"] = (trainer.beta * (logps[mask] - ref_logps[mask])).sum().item()
                        local[f"logps/{split}"] = logps[mask].sum().item()
                        local[f"logits/{split}"] = logits[mask].sum().item()
                    gathered = [local]
                    if world_size > 1:
                        gathered = [None] * world_size
                        dist.all_gather_object(gathered, local)
                    for metrics in gathered:
                        for key, value in metrics.items():
                            totals[key] = totals.get(key, 0.0) + value
                    if train_eval == "train":
                        loss = trainer.compute_loss(trainer.model, batch)
                        loss.backward()
                        trainer.model.zero_grad()
                    else:
                        trainer.prediction_step(trainer.model, batch, prediction_loss_only=True)
                trainer.log({f"{prefix}loss": 0.5})
                if rank == 0:
                    actual = trainer.state.log_history[-1]
                    for split in ("chosen", "rejected"):
                        for key in ("rewards", "logps", "logits"):
                            name = f"{prefix}{key}/{split}"
                            if totals[f"count/{split}"]:
                                expected = totals[f"{key}/{split}"] / totals[f"count/{split}"]
                                assert actual[name] == pytest.approx(expected, rel=1e-5, abs=1e-6), (case, actual)
                            else:
                                assert name not in actual
                    if totals["count/chosen"] and totals["count/rejected"]:
                        expected = totals["rewards/chosen"] / totals["count/chosen"]
                        expected -= totals["rewards/rejected"] / totals["count/rejected"]
                        assert actual[f"{prefix}rewards/margins"] == pytest.approx(expected, abs=1e-6)
                    else:
                        assert f"{prefix}rewards/margins" not in actual
                    assert not any(key.startswith(("dummy_", "count/")) or key.endswith("_sum") for key in actual)
                    print(f"PASS: KTO {train_eval} {case} two-batch global metrics on {device.type}")
                assert not trainer._stored_metrics[train_eval]
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2])
def test_kto_metrics_with_different_rank_local_keys(tmp_path, world_size):
    """Training and evaluation must include both ranks when their class counts differ."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_check_metrics, args=(port, str(tmp_path), True, world_size), nprocs=world_size, join=True)
