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

import pytest
import torch
import torch.multiprocessing as mp

from llamafactory.v1.accelerator.interface import DistributedInterface
from llamafactory.v1.config.model_args import ModelArguments
from llamafactory.v1.config.training_args import TrainingArguments
from llamafactory.v1.core.model_engine import ModelEngine
from llamafactory.v1.plugins.model_plugins.parallelization import ulysses
from llamafactory.v1.plugins.model_plugins.parallelization.sequence_parallel import (
    SequenceParallelModelPlugin,
    sequence_parallel_loss,
)
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


def test_kv_heads_fewer_than_cp_size_are_repeated_before_the_all2all(monkeypatch: pytest.MonkeyPatch):
    # apply_sequence_parallel() accepts models whose KV head count divides cp_size, so a
    # 2-KV-head model may run on 4 ranks. tensor_split() would then hand ranks 2 and 3
    # empty shards and all_to_all rejects them, so the heads have to be repeated first.
    captured = []

    def fake_all2all(_group, tensor, *_args):
        captured.append(tensor)
        return tensor

    monkeypatch.setattr(ulysses.SeqAllToAll4D, "apply", fake_all2all)
    monkeypatch.setattr(ulysses, "get_ulysses_sequence_parallel_world_size", lambda _: 4)
    monkeypatch.setattr(ulysses.dist, "all_gather", lambda outputs, tensor, **_: [o.copy_(tensor) for o in outputs])

    attention = ulysses.UlyssesAttention(sequence_process_group=object(), attn_fn=lambda query, *_, **__: query)
    query = torch.zeros(1, 4, 8, 2)
    key = torch.arange(2, dtype=torch.float32).view(1, 1, 2, 1).expand(1, 4, 2, 2).contiguous()

    attention(query, key, key.clone(), None, 16)

    query_sent, key_sent, value_sent = captured[:3]
    assert [query_sent.shape[2], key_sent.shape[2], value_sent.shape[2]] == [8, 4, 4]
    # Each rank owns two of the eight query heads, so it needs the KV head backing that
    # group: ranks 0-1 take KV head 0 and ranks 2-3 take KV head 1.
    assert key_sent[0, 0, :, 0].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert value_sent[0, 0, :, 0].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_kv_heads_matching_cp_size_are_left_alone(monkeypatch: pytest.MonkeyPatch):
    captured = []

    monkeypatch.setattr(ulysses.SeqAllToAll4D, "apply", lambda _g, tensor, *_a: captured.append(tensor) or tensor)
    monkeypatch.setattr(ulysses, "get_ulysses_sequence_parallel_world_size", lambda _: 2)
    monkeypatch.setattr(ulysses.dist, "all_gather", lambda outputs, tensor, **_: [o.copy_(tensor) for o in outputs])

    attention = ulysses.UlyssesAttention(sequence_process_group=object(), attn_fn=lambda query, *_, **__: query)
    key = torch.zeros(1, 4, 4, 2)

    attention(torch.zeros(1, 4, 8, 2), key, key.clone(), None, 8)

    assert [captured[1].shape[2], captured[2].shape[2]] == [4, 4]


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


@pytest.mark.runs_on(["cuda", "npu"])
@pytest.mark.require_distributed(2)
@pytest.mark.parametrize(("cp_size", "dp_size", "batch_size"), [(2, 1, 1), (2, 1, 2)])
def test_sequence_parallel_loss(cp_size, dp_size, batch_size):
    master_port = find_available_port()
    world_size = cp_size * dp_size
    mp.spawn(
        _test_sequence_parallel_loss, args=(world_size, master_port, cp_size, dp_size, batch_size), nprocs=world_size
    )
