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

from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

import llamafactory.v1.plugins.model_plugins.parallelization.hook as hook_module
from llamafactory.v1.accelerator.interface import DistributedInterface
from llamafactory.v1.config.model_args import ModelArguments
from llamafactory.v1.config.training_args import TrainingArguments
from llamafactory.v1.core.model_engine import ModelEngine
from llamafactory.v1.plugins.model_plugins.parallelization import ulysses
from llamafactory.v1.plugins.model_plugins.parallelization.batch import prepare_sequence_parallel_batch
from llamafactory.v1.plugins.model_plugins.parallelization.sequence_parallel import (
    SequenceParallelModelPlugin,
    sequence_parallel_loss,
)
from llamafactory.v1.utils.constants import IGNORE_INDEX
from llamafactory.v1.utils.env import find_available_port
from llamafactory.v1.utils.pytest import dist_env


def test_qwen3_5_broadcast_position_ids_keep_packed_boundaries(monkeypatch: pytest.MonkeyPatch):
    local_position_ids = torch.tensor([[0, 1, 0]])
    remote_position_ids = torch.tensor([[1, 2, 3]])
    mrope_position_ids = local_position_ids.unsqueeze(0).expand(3, -1, -1)
    captured = {}

    monkeypatch.setattr(ulysses.SeqAllToAll4D, "apply", lambda _, tensor, *__: tensor)
    monkeypatch.setattr(ulysses, "get_ulysses_sequence_parallel_world_size", lambda _: 2)

    def fake_all_gather(outputs, tensor, **_):
        outputs[0].copy_(tensor)
        outputs[1].copy_(remote_position_ids if tensor.shape == local_position_ids.shape else tensor)

    def fake_attention(query, _key, _value, _attention_mask, **kwargs):
        captured["position_ids"] = kwargs["position_ids"]
        return query

    monkeypatch.setattr(ulysses.dist, "all_gather", fake_all_gather)
    attention = ulysses.UlyssesAttention(sequence_process_group=object(), attn_fn=fake_attention)
    hidden_states = torch.zeros(1, 3, 2, 4)

    attention(hidden_states, hidden_states, hidden_states, None, 6, position_ids=mrope_position_ids)

    assert captured["position_ids"].tolist() == [[0, 1, 0, 1, 2, 3]]
    assert captured["position_ids"].is_contiguous()


def test_true_mrope_position_ids_are_not_used_as_packed_boundaries():
    mrope_position_ids = torch.tensor([[[0, 1, 2]], [[0, 1, 1]], [[0, 1, 0]]])

    assert ulysses._get_text_position_ids(mrope_position_ids) is None


def _test_sequence_parallel_loss(
    local_rank: int, world_size: int, master_port: int, cp_size: int, dp_size: int, batch_size: int
):
    with dist_env(local_rank, world_size, master_port):
        model_args = ModelArguments(model="llamafactory/tiny-random-qwen3")

        training_args = TrainingArguments(cp_mode="ulysses", cp_size=cp_size, dp_size=dp_size)
        DistributedInterface(training_args)

        # Now create model engine
        model_engine = ModelEngine(model_args=model_args)

        # Apply sequence parallel plugin
        SequenceParallelModelPlugin(training_args.cp_mode)(model_engine.model, training_args.cp_size)

        input_ids = torch.arange(1, batch_size * 5 + 1, dtype=torch.long).view(batch_size, 5)
        model_inputs = {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
            "position_ids": torch.arange(1, 6, dtype=torch.long).repeat(batch_size, 1),
            "loss_weights": torch.ones(batch_size, 5),
        }

        loss = sequence_parallel_loss(model_engine.model, model_inputs)
        assert loss is not None


@pytest.mark.runs_on(["cuda", "npu", "xpu"])
@pytest.mark.require_distributed(2)
@pytest.mark.parametrize(("cp_size", "dp_size", "batch_size"), [(2, 1, 1), (2, 1, 2)])
def test_sequence_parallel_loss(cp_size, dp_size, batch_size):
    master_port = find_available_port()
    world_size = cp_size * dp_size
    mp.spawn(
        _test_sequence_parallel_loss, args=(world_size, master_port, cp_size, dp_size, batch_size), nprocs=world_size
    )


def test_non_causal_multimodal_encoder_attention_bypasses_ulysses():
    captured_is_causal = None

    def fake_native_attention(query, _key, _value, _attention_mask, **kwargs):
        nonlocal captured_is_causal
        captured_is_causal = kwargs["is_causal"]
        return query + 1

    query = torch.zeros(1, 4, 2, 8)
    output = ulysses.new_flash_attn_forward(query, query, query, None, is_causal=False, attn_fn=fake_native_attention)

    torch.testing.assert_close(output, query + 1)
    assert captured_is_causal is False


def _device_mesh(rank=0, size=2):
    return {"cp": SimpleNamespace(size=lambda: size, get_local_rank=lambda: rank)}


class _RecordingLanguageModel(nn.Module):
    def forward(self, **kwargs):
        return kwargs


def test_multimodal_sequence_parallel_hook(monkeypatch):
    # One shard gets non-contiguous visual rows while the next shard is empty.
    cp_rank = [1]
    device_mesh = {"cp": SimpleNamespace(size=lambda: 3, get_local_rank=lambda: cp_rank[0])}
    distributed = SimpleNamespace(get_device_mesh=lambda _dim: device_mesh)
    monkeypatch.setattr(hook_module, "DistributedInterface", lambda: distributed)

    model = nn.Module()
    model.model = core = nn.Module()
    boundary = _RecordingLanguageModel()
    core.visual = nn.Identity()
    core.language_model = boundary
    hook_module.install_sequence_parallel_hook(SimpleNamespace(get_base_model=lambda: model))

    fused_inputs = torch.arange(24, dtype=torch.float32).view(2, 6, 2)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]])
    position_ids = torch.arange(36).view(3, 2, 6)
    visual_mask = torch.tensor([[True, False, False, True, False, False], [True, True, True, False, False, False]])
    visual_embeds = torch.arange(10, dtype=torch.float32).view(5, 2).requires_grad_()
    model_inputs = {
        "input_ids": None,
        "inputs_embeds": fused_inputs,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "visual_pos_masks": visual_mask,
        "deepstack_visual_embeds": [visual_embeds],
    }

    outputs = boundary(**model_inputs)
    torch.testing.assert_close(outputs["inputs_embeds"], fused_inputs[:, 2:4])
    torch.testing.assert_close(outputs["attention_mask"], attention_mask[:, 2:4])
    torch.testing.assert_close(outputs["position_ids"], position_ids[..., 2:4])
    torch.testing.assert_close(outputs["visual_pos_masks"], visual_mask[:, 2:4])
    torch.testing.assert_close(outputs["deepstack_visual_embeds"][0], visual_embeds[[1, 4]])
    assert outputs["use_cache"] is False

    outputs["deepstack_visual_embeds"][0].sum().backward()
    expected_grad = torch.zeros_like(visual_embeds)
    expected_grad[[1, 4]] = 1
    torch.testing.assert_close(visual_embeds.grad, expected_grad)

    cp_rank[0] = 2
    visual_embeds.grad = None
    empty_visual_embeds = boundary(**model_inputs)["deepstack_visual_embeds"][0]
    assert empty_visual_embeds.shape == (0, 2)
    empty_visual_embeds.sum().backward()
    torch.testing.assert_close(visual_embeds.grad, torch.zeros_like(visual_embeds))


def test_prepare_multimodal_sequence_parallel_batch_preserves_encoder_inputs_and_shifts_targets():
    pixel_values = torch.arange(12, dtype=torch.float32).view(3, 4)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "position_ids": torch.tensor([[0, 1, 2]]),
        "mm_token_type_ids": torch.tensor([[0, 1, 1]]),
        "labels": torch.tensor([[1, 2, 3]]),
        "loss_weights": torch.tensor([[9.0, 0.5, 2.0]]),
        "pixel_values": pixel_values,
    }
    for rank in range(2):
        prepared = prepare_sequence_parallel_batch(
            batch, device=torch.device("cpu"), device_mesh=_device_mesh(rank), uses_mrope=True
        )
        assert prepared.model_inputs["input_ids"].tolist() == [[1, 2, 3, 0]]
        assert prepared.model_inputs["mm_token_type_ids"].tolist() == [[0, 1, 1, 0]]
        assert "labels" not in prepared.model_inputs and "loss_weights" not in prepared.model_inputs
        assert "position_ids" not in prepared.model_inputs
        assert prepared.model_inputs["attention_mask"].tolist() == [[1, 1, 1, 0]]
        torch.testing.assert_close(prepared.model_inputs["pixel_values"], pixel_values)
        assert prepared.local_shift_labels.tolist() == ([[2, 3]] if rank == 0 else [[IGNORE_INDEX, IGNORE_INDEX]])
        assert prepared.local_shift_loss_weights.tolist() == ([[0.5, 2.0]] if rank == 0 else [[0.0, 0.0]])
        assert prepared.global_loss_weight_sum.item() == 2.5
