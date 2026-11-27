# Copyright 2025 the LlamaFactory team.
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

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from llamafactory.v1.accelerator.interface import DistributedInterface
from llamafactory.v1.config import TrainingArguments
from llamafactory.v1.core import base_trainer
from llamafactory.v1.core.base_trainer import BaseTrainer
from llamafactory.v1.core.rendering import Renderer
from llamafactory.v1.plugins.trainer_plugins.distributed import interface as plugin_interface
from llamafactory.v1.trainers.sft_trainer import SFTTrainer
from llamafactory.v1.utils import helper


def test_fit_finalizes_accumulated_gradients_before_clipping_and_step(monkeypatch):
    """Exercise the real optimizer boundary with two micro-batches per update."""
    events = []
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.ones_(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.register_step_post_hook(lambda *_: events.append("step"))

    def loss(batch):
        result = model.weight.square().sum() * batch["scale"]
        result.register_hook(lambda grad: events.append("backward"))
        return result

    def synchronize(candidate):
        assert candidate is model
        assert events[-2:] == ["backward", "backward"]
        events.append("synchronize")
        # Simulate the final reduction's effect; real CP reduction is tested with Gloo meshes.
        model.weight.grad.div_(2)

    def clip(candidate, limit):
        assert candidate is model
        assert events[-1] == "synchronize"
        events.append("clip")
        return torch.nn.utils.clip_grad_norm_(candidate.parameters(), limit).item()

    plugin = SimpleNamespace(synchronize_gradients=synchronize, clip_grad_norm=clip)
    monkeypatch.setattr(plugin_interface, "DistributedPlugin", lambda _: plugin)
    context = SimpleNamespace(
        current_device=torch.device("cpu"), all_reduce=lambda value, **_: value, sync=lambda: None
    )
    monkeypatch.setattr(base_trainer, "DistributedInterface", lambda: context)
    monkeypatch.setattr(helper, "DistributedInterface", lambda: context)

    class Batches(list):
        def set_epoch(self, epoch):
            pass

    micro_batches = [{"labels": torch.ones(1, 1), "scale": scale} for scale in (1.0, 3.0)]

    class Trainer(SimpleNamespace, BaseTrainer):
        pass

    trainer = Trainer(
        model=model,
        args=SimpleNamespace(
            dist_config=SimpleNamespace(name="fsdpturbo"), max_grad_norm=1.0, logging_steps=1, save_steps=None
        ),
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
        callback_handler=Mock(),
        state=SimpleNamespace(),
        global_step=0,
        num_training_steps=2,
        _resume_epoch=0,
        _deepspeed_engine=None,
        dp_size=1,
        compute_loss=loss,
        train_batch_generator=Batches([micro_batches, micro_batches]),
    )
    trainer.fit()
    assert events == ["backward", "backward", "synchronize", "clip", "step"] * 2
    assert model.weight.item() == pytest.approx(0.8, abs=1e-6)
    assert model.weight.grad is None


@pytest.mark.parametrize("weights", [(0.125, 1.0), (2e-6, 4e-6)])
@pytest.mark.parametrize("chunk_loss_size", [None, 2])
def test_sft_accumulation_matches_one_weighted_batch(tmp_path, chunk_loss_size, weights):
    tokenizer = Tokenizer(WordLevel({"?": 0, "question": 1, "yes": 2, "no": 3}, unk_token="?"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="?")
    tokenizer.chat_template = "{% for m in messages %}{{ m.content + ' ' }}{% endfor %}"
    renderer = Renderer(tokenizer)
    samples = [
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "value": "question"}]},
                {"role": "assistant", "content": [{"type": "text", "value": answer}], "loss_weight": weight},
            ]
        }
        for answer, weight in zip(["yes", "no no"], weights, strict=True)
    ]
    torch.manual_seed(42)
    model = LlamaForCausalLM(
        LlamaConfig(vocab_size=4, hidden_size=8, intermediate_size=16, num_attention_heads=2, num_hidden_layers=1)
    ).to(DistributedInterface().current_device)

    def train(micro_batch_size):
        args = TrainingArguments(
            output_dir=str(tmp_path),
            global_batch_size=2,
            micro_batch_size=micro_batch_size,
            max_steps=1,
            cutoff_len=8,
            batching_workers=0,
            enable_activation_checkpointing=False,
            chunk_loss_size=chunk_loss_size,
            max_grad_norm=100.0,
        )
        trainer = SFTTrainer(args, deepcopy(model), renderer, samples)
        # A single SGD update exposes differences in gradient magnitude as well as direction.
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
        trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
        trainer.fit()
        return trainer.state.loss, trainer.model

    full_loss, full_model = train(2)
    accumulated_loss, accumulated_model = train(1)
    assert accumulated_loss == pytest.approx(full_loss, abs=1e-6)
    for full, accumulated in zip(full_model.parameters(), accumulated_model.parameters(), strict=True):
        torch.testing.assert_close(accumulated, full, atol=1e-6, rtol=1e-5)
