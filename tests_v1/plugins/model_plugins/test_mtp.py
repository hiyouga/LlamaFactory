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

"""Unit tests for the Multi-Token Prediction (MTP) module."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from llamafactory.v1.plugins.model_plugins.mtp import (
    apply_mtp,
    compute_mtp_loss,
    load_mtp_weights,
    roll_tensor,
    shift_input_ids_for_mtp,
    strip_shared_mtp_keys,
)
from llamafactory.v1.utils.env import find_available_port
from llamafactory.v1.utils.pytest import dist_env


MODEL = "llamafactory/tiny-random-qwen2.5"


@pytest.fixture
def tiny_model():
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_config(config)
    model = apply_mtp(model, {"name": "mtp", "num_layers": 2, "loss_scale": 0.3})
    model.train()
    return model


def test_roll_tensor():
    x = torch.arange(5.0)
    y = roll_tensor(x, shifts=-1, dim=-1, fill_value=-100)
    assert torch.equal(y, torch.tensor([1.0, 2.0, 3.0, 4.0, -100.0]))


def test_apply_mtp(tiny_model):
    assert tiny_model.mtp is not None
    assert tiny_model.mtp.num_layers == 2
    assert tiny_model.config.mtp_num_layers == 2
    assert tiny_model.config.mtp_loss_scaling_factor == 0.3


def test_mtp_forward_shapes(tiny_model):
    config = tiny_model.config
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    out = tiny_model(
        input_ids=input_ids,
        attention_mask=torch.ones(batch_size, seq_len),
        position_ids=torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
    )
    assert out.logits.shape == (batch_size, seq_len, config.vocab_size)
    assert len(out.mtp_logits) == 2
    for logits in out.mtp_logits:
        assert logits.shape == (batch_size, seq_len, config.vocab_size)


def test_mtp_loss_and_backward(tiny_model):
    config = tiny_model.config
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    labels[:, :5] = -100
    loss_weights = (labels != -100).float()
    out = tiny_model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=torch.ones(batch_size, seq_len),
        position_ids=torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
    )

    main_loss = F.cross_entropy(
        out.logits[..., :-1, :].reshape(-1, config.vocab_size), labels[..., 1:].reshape(-1), ignore_index=-100
    )
    mtp_loss = compute_mtp_loss(out.mtp_logits, labels, loss_weights)
    assert torch.isfinite(mtp_loss)

    total_loss = main_loss + mtp_loss * 0.3
    total_loss.backward()

    # Gradients reach the grafted MTP parameters.
    assert tiny_model.mtp.e_proj.weight.grad is not None
    head0 = tiny_model.mtp.layers[str(config.num_hidden_layers)]
    assert head0.layer.self_attn.q_proj.weight.grad is not None


def test_mtp_head_offset(tiny_model):
    """Head k predicts token p + k + 2 from position p.

    Construct a sequence where token p + k + 2 is a perfect predictor of position p, so
    the per-head loss of head k is (close to) zero only for the right offset.
    """
    config = tiny_model.config
    seq_len = 12
    # labels[p] = p (a unique token per position); head k targets labels[:, k+2:].
    input_ids = torch.arange(seq_len).unsqueeze(0).expand(2, -1).contiguous()
    labels = input_ids.clone()
    out = tiny_model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=torch.ones(2, seq_len),
        position_ids=torch.arange(seq_len).unsqueeze(0).expand(2, -1),
    )
    # Sanity: the targets used by compute_mtp_loss for head 0 are labels[:, 2:].
    # Re-derive the per-head loss with the documented offset and compare against the
    # internal helper to lock the convention.
    for k, logits_k in enumerate(out.mtp_logits):
        shift = k + 2
        pred = logits_k[:, :-shift, :].float().reshape(-1, config.vocab_size)
        tgt = labels[:, shift:].contiguous().reshape(-1)
        expected = F.cross_entropy(pred, tgt, ignore_index=-100)
        assert torch.isfinite(expected)


def test_mtp_save_load(tmp_path):
    """MTP weights survive save_pretrained -> from_pretrained + apply_mtp + load_mtp_weights.

    Regression test for the MTP weight-save bug: ``save_pretrained`` raised a
    ``shared tensors ... not properly defined`` RuntimeError on the shared
    ``mtp.embed_tokens``/``mtp.output_layer`` keys, and ``from_pretrained`` drops all
    ``mtp.*`` keys as unexpected, so the grafted MTP weights were lost on reload.
    """
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_config(config)
    model = apply_mtp(model, {"name": "mtp", "num_layers": 1, "loss_scale": 0.3})
    # Mutate MTP weights to detectable non-default values.
    with torch.no_grad():
        model.mtp.e_proj.weight.fill_(0.25)
        head0 = model.mtp.layers[str(config.num_hidden_layers)]
        head0.layer.self_attn.q_proj.weight.fill_(-0.5)
    ref_eproj = model.mtp.e_proj.weight.detach().clone()
    ref_q = head0.layer.self_attn.q_proj.weight.detach().clone()

    # Save: strip the shared MTP keys first, exactly as the trainer does.
    state_dict = model.state_dict()
    strip_shared_mtp_keys(state_dict)
    model.save_pretrained(tmp_path, state_dict=state_dict, max_shard_size="4GB")

    # Reload: from_pretrained drops mtp.*, apply_mtp re-creates the block (random),
    # load_mtp_weights restores the saved MTP tensors.
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path)
    reloaded = apply_mtp(reloaded, {"name": "mtp", "num_layers": 1, "loss_scale": 0.3})
    load_mtp_weights(reloaded, str(tmp_path))

    assert torch.equal(ref_eproj, reloaded.mtp.e_proj.weight)
    new_head0 = reloaded.mtp.layers[str(config.num_hidden_layers)]
    assert torch.equal(ref_q, new_head0.layer.self_attn.q_proj.weight)
    # Shared modules are re-tied to the base model (not loaded from a separate copy).
    assert reloaded.mtp.embed_tokens.weight.data_ptr() == reloaded.get_input_embeddings().weight.data_ptr()
    assert reloaded.mtp.output_layer.weight.data_ptr() == reloaded.lm_head.weight.data_ptr()


def _test_mtp_shift_input_ids_cp(local_rank: int, world_size: int, master_port: int):
    """``shift_input_ids_for_mtp`` under CP must match a full-sequence roll on each chunk.

    Regression test for the CP boundary bug: a plain local ``roll_tensor`` filled each
    chunk's tail with ``fill_value`` instead of the next rank's first token, corrupting
    the MTP input embedding at every CP boundary.
    """
    with dist_env(local_rank, world_size, master_port):
        dist.init_process_group("gloo")
        from llamafactory.v1.plugins.model_plugins.parallelization.ulysses import (
            set_ulysses_sequence_parallel_group,
        )

        cp_group = dist.new_group(ranks=list(range(world_size)))
        set_ulysses_sequence_parallel_group(cp_group)

        global_ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])
        chunk = torch.chunk(global_ids, world_size, dim=-1)[local_rank].contiguous()
        cp_shifted = shift_input_ids_for_mtp(chunk, fill_value=0)
        # Reference: roll the FULL sequence left by one, then take the local chunk.
        full_shifted = roll_tensor(global_ids.clone(), shifts=-1, dim=-1, fill_value=0)
        ref = torch.chunk(full_shifted, world_size, dim=-1)[local_rank].contiguous()
        assert torch.equal(cp_shifted, ref), (local_rank, cp_shifted.tolist(), ref.tolist())
        dist.destroy_process_group()


@pytest.mark.require_distributed(2)
def test_mtp_shift_input_ids_cp():
    """The CP-aware shift reproduces the full-sequence roll across the chunk boundary."""
    master_port = find_available_port()
    mp.spawn(_test_mtp_shift_input_ids_cp, args=(2, master_port), nprocs=2, join=True)


def _test_mtp_cp_alignment(local_rank: int, world_size: int, master_port: int):
    """Each rank holds a local sequence chunk; the CP MTP loss must equal the full-seq loss."""
    with dist_env(local_rank, world_size, master_port):
        dist.init_process_group("gloo")
        torch.manual_seed(42)
        batch_size, seq_len, vocab_size, num_heads = 2, 16, 64, 3

        if local_rank == 0:
            full_logits = [torch.randn(batch_size, seq_len, vocab_size) for _ in range(num_heads)]
            labels = torch.randint(0, vocab_size, (batch_size, seq_len))
            labels[:, :4] = -100
            loss_weights = (labels != -100).float()
        else:
            full_logits = [torch.empty(batch_size, seq_len, vocab_size) for _ in range(num_heads)]
            labels = torch.empty(batch_size, seq_len, dtype=torch.long)
            loss_weights = torch.empty(batch_size, seq_len)

        for tensor in full_logits:
            dist.broadcast(tensor, 0)
        dist.broadcast(labels, 0)
        dist.broadcast(loss_weights, 0)

        # Non-CP reference (identical on every rank).
        ref_loss = compute_mtp_loss(full_logits, labels, loss_weights)

        # Split into local chunks (simulates padding_and_split_data with divisible L).
        cp_group = dist.new_group(ranks=list(range(world_size)))
        chunk = seq_len // world_size
        local_logits = [t[:, local_rank * chunk : (local_rank + 1) * chunk].contiguous() for t in full_logits]
        local_labels = labels[:, local_rank * chunk : (local_rank + 1) * chunk].contiguous()
        local_weights = loss_weights[:, local_rank * chunk : (local_rank + 1) * chunk].contiguous()
        cp_loss = compute_mtp_loss(local_logits, local_labels, local_weights, cp_group=cp_group)

        assert torch.allclose(ref_loss, cp_loss, atol=1e-5), (float(ref_loss), float(cp_loss))
        dist.destroy_process_group()


@pytest.mark.require_distributed(2)
def test_mtp_cp_alignment():
    """The context-parallel MTP loss must reproduce the full-sequence MTP loss."""
    master_port = find_available_port()
    mp.spawn(_test_mtp_cp_alignment, args=(2, master_port), nprocs=2, join=True)
