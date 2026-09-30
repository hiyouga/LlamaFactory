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

from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.modeling_outputs import CausalLMOutput

from llamafactory.v1.plugins.model_plugins.chunk_loss import LossPlugin, _ChunkedLinearCrossEntropy
from llamafactory.v1.trainers.sft_trainer import SFTTrainer
from llamafactory.v1.utils.constants import IGNORE_INDEX
from llamafactory.v1.utils.env import find_available_port
from llamafactory.v1.utils.pytest import dist_env


class _TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(31, 16)
        self.lm_head = nn.Linear(16, 31, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, **_):
        return CausalLMOutput(logits=self.lm_head(self.embed_tokens(input_ids)))


def _make_model():
    return _TinyCausalLM()


def _assert_gradients_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    error = torch.linalg.vector_norm(actual.float() - expected.float())
    reference = torch.linalg.vector_norm(expected.float())
    assert error <= 2 * torch.finfo(expected.dtype).eps * reference


def _weighted_cross_entropy(logits, labels, loss_weights):
    losses = F.cross_entropy(logits.flatten(0, -2).float(), labels.flatten(), reduction="none")
    return (losses * loss_weights.flatten()).sum()


@pytest.mark.parametrize("frozen_head", [False, True])
def test_chunk_loss_matches_eager_loss_and_gradients(frozen_head):
    torch.manual_seed(0)
    eager_head = nn.Linear(4, 7).to(torch.bfloat16)
    eager_head.requires_grad_(not frozen_head)
    chunk_head = deepcopy(eager_head)
    eager_hidden = torch.randn(2, 5, 4, dtype=torch.bfloat16, requires_grad=True)
    chunk_hidden = eager_hidden.detach().clone().requires_grad_()
    labels = torch.tensor([[0, 1, IGNORE_INDEX, 3, 4], [5, 6, 0, 1, 2]])
    loss_weights = torch.tensor([[0.0, 0.25, 1.0, 0.75, 1.0], [1.0, 0.5, 0.0, 0.25, 1.0]])

    eager_loss = _weighted_cross_entropy(eager_head(eager_hidden), labels, loss_weights)
    chunk_loss = _ChunkedLinearCrossEntropy.apply(
        chunk_hidden, chunk_head.weight, chunk_head.bias, labels, loss_weights, 3
    )
    scale = 0.07 / (loss_weights.sum() + 1e-6)
    (eager_loss * scale).backward()
    (chunk_loss * scale).backward()

    torch.testing.assert_close(chunk_loss, eager_loss)
    _assert_gradients_close(chunk_hidden.grad, eager_hidden.grad)
    for actual, expected in zip(chunk_head.parameters(), eager_head.parameters()):
        if frozen_head:
            assert actual.grad is None
        else:
            _assert_gradients_close(actual.grad, expected.grad)


@pytest.mark.parametrize("zero_supervision", [False, True])
def test_chunk_sft_loss_matches_reference(zero_supervision):
    model = _make_model()
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    labels = input_ids.clone()
    labels[0, 2] = IGNORE_INDEX
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "position_ids": torch.arange(5).expand(2, -1),
        "labels": labels,
        "loss_weights": torch.tensor([[0.0, 0.25, 0.0, 1.0, 0.5], [0.0, 0.0, 0.75, 0.0, 1.0]]),
    }
    if zero_supervision:
        batch["labels"].fill_(IGNORE_INDEX)
        batch["loss_weights"].zero_()
    original_batch = {key: value.clone() for key, value in batch.items()}

    trainer = object.__new__(SFTTrainer)
    trainer.model = model
    trainer.device = torch.device("cpu")
    trainer.cp_size = 1
    trainer._uses_mrope = False
    trainer._chunk_loss = None
    eager_loss = trainer.compute_loss(batch)

    chunk_model = deepcopy(model)
    trainer.model = chunk_model
    trainer._chunk_loss = LossPlugin("chunk_loss")(chunk_model, chunk_size=3)
    chunk_loss = trainer.compute_loss(batch)

    torch.testing.assert_close(chunk_loss, eager_loss)
    for key in batch:
        torch.testing.assert_close(batch[key], original_batch[key])
    torch.testing.assert_close(chunk_model(input_ids=input_ids).logits, model(input_ids=input_ids).logits)


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Requires the CPU Gloo backend.")
def test_chunk_loss_preserves_ddp_output_backward_hooks():
    torch.manual_seed(7)
    eager_model = _make_model()
    chunk_model = deepcopy(eager_model)
    loss_fn = LossPlugin("chunk_loss")(chunk_model, chunk_size=2)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    model_inputs = {"input_ids": input_ids, "use_cache": False}
    labels = torch.tensor([[2, 3, 4, 5, IGNORE_INDEX]])
    loss_weights = torch.tensor([[0.0, 0.25, 1.0, 0.5, 0.0]])
    outer_scale = 0.3 / loss_weights.sum()
    outer_outputs = []

    def capture_outer_output(_model, _args, output):
        outer_outputs.append(output.logits)

    with dist_env(master_port=find_available_port()):
        dist.init_process_group("gloo")
        wrapped_model = DDP(chunk_model, find_unused_parameters=True)
        wrapped_model.register_forward_hook(capture_outer_output)
        eager_loss = _weighted_cross_entropy(eager_model(**model_inputs).logits, labels, loss_weights)
        chunk_loss = loss_fn(wrapped_model, model_inputs, labels, loss_weights)

        assert loss_fn._active_state is None
        assert chunk_loss is outer_outputs.pop()
        torch.testing.assert_close(chunk_loss, eager_loss)
        (eager_loss * outer_scale).backward()
        (chunk_loss * outer_scale).backward()
        for expected, actual in zip(eager_model.parameters(), chunk_model.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad)
