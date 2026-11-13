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
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import CausalLMOutput

from llamafactory.v1.plugins.model_plugins.chunk_loss import ChunkLoss, _ChunkedLinearCrossEntropy
from llamafactory.v1.utils.constants import IGNORE_INDEX


@pytest.mark.parametrize("num_tokens", [8192, 131072])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("outer_scale", [0.0, 0.125, 1.0])
def test_head_gradient_is_scaled_before_fp16_cast(num_tokens, bias, outer_scale):
    reference_head = nn.Linear(4, 8, bias=bias, dtype=torch.float16)
    nn.init.zeros_(reference_head.weight)
    if bias:
        nn.init.zeros_(reference_head.bias)
    actual_head = deepcopy(reference_head)
    reference_hidden = torch.ones(num_tokens, 4, dtype=torch.float16, requires_grad=True)
    actual_hidden = reference_hidden.detach().clone().requires_grad_()
    labels = torch.zeros(num_tokens, dtype=torch.long)
    weights = torch.ones(num_tokens)

    reference_loss = F.cross_entropy(reference_head(reference_hidden).float(), labels) * outer_scale
    actual_sum = _ChunkedLinearCrossEntropy.apply(
        actual_hidden, actual_head.weight, actual_head.bias, labels, weights, 4096
    )
    actual_loss = actual_sum * (outer_scale / num_tokens)
    reference_loss.backward()
    actual_loss.backward()

    torch.testing.assert_close(actual_loss, reference_loss)
    torch.testing.assert_close(actual_hidden.grad, reference_hidden.grad, rtol=0, atol=0)
    for actual, reference in zip(actual_head.parameters(), reference_head.parameters(), strict=True):
        assert actual.grad.dtype == actual.dtype
        assert torch.isfinite(actual.grad).all()
        torch.testing.assert_close(actual.grad, reference.grad, rtol=0, atol=0)

    # Uniform logits give a closed-form normalized gradient, independent of the chunking.
    expected = torch.full_like(actual_head.weight, outer_scale / 8)
    expected[0].fill_(-7 * outer_scale / 8)
    torch.testing.assert_close(actual_head.weight.grad, expected, rtol=0, atol=0)


class _LinearCausalLM(nn.Module):
    def __init__(self, dtype, bias):
        super().__init__()
        self.embed_tokens = nn.Embedding(131073, 4, dtype=dtype)
        self.lm_head = nn.Linear(4, 8, bias=bias, dtype=dtype)
        nn.init.ones_(self.embed_tokens.weight)
        nn.init.zeros_(self.lm_head.weight)
        if bias:
            nn.init.zeros_(self.lm_head.bias)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, **kwargs):
        hidden = self.embed_tokens(input_ids)
        if getattr(self, "wide_head_reference", False):
            # A single double-precision projection independently aggregates the long
            # sequence without rounding per-token mean gradients to FP16 first.
            bias = self.lm_head.bias.double() if self.lm_head.bias is not None else None
            return CausalLMOutput(logits=F.linear(hidden.double(), self.lm_head.weight.double(), bias).float())
        return CausalLMOutput(logits=self.lm_head(hidden))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("bias", [False, True])
def test_chunk_plugin_long_sequence_optimizer_matches_high_precision_reference(dtype, bias):
    reference = _LinearCausalLM(dtype, bias)
    actual = deepcopy(reference)
    reference.wide_head_reference = True
    chunk_loss = ChunkLoss(actual, chunk_size=4096)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05, momentum=0.9)
    actual_optimizer = torch.optim.SGD(actual.parameters(), lr=0.05, momentum=0.9)
    input_ids = torch.arange(131073).unsqueeze(0)
    labels = torch.ones_like(input_ids)
    weights = torch.ones_like(input_ids, dtype=torch.float32)
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "loss_weights": weights,
    }
    tolerance = 4 * torch.finfo(dtype).eps
    for _ in range(3):
        reference_optimizer.zero_grad()
        actual_optimizer.zero_grad()
        logits = reference(input_ids).logits[:, :-1].float()
        expected_loss = F.cross_entropy(logits.reshape(-1, 8), labels[:, 1:].reshape(-1))
        actual_loss = chunk_loss.compute_loss(actual, batch, device=torch.device("cpu"), uses_mrope=False)
        expected_loss.backward()
        actual_loss.backward()
        torch.testing.assert_close(actual_loss, expected_loss, rtol=tolerance, atol=tolerance)

        for actual_parameter, reference_parameter in zip(actual.parameters(), reference.parameters(), strict=True):
            assert torch.isfinite(actual_parameter.grad).all()
            assert actual_parameter.grad.dtype == actual_parameter.dtype
            torch.testing.assert_close(actual_parameter.grad, reference_parameter.grad, rtol=tolerance, atol=tolerance)
        actual_optimizer.step()
        reference_optimizer.step()
        for actual_parameter, reference_parameter in zip(actual.parameters(), reference.parameters(), strict=True):
            assert torch.isfinite(actual_parameter).all()
            torch.testing.assert_close(actual_parameter, reference_parameter, rtol=tolerance, atol=tolerance)
            torch.testing.assert_close(
                actual_optimizer.state[actual_parameter]["momentum_buffer"],
                reference_optimizer.state[reference_parameter]["momentum_buffer"],
                rtol=tolerance,
                atol=tolerance,
            )
    assert actual.lm_head.weight.count_nonzero() > 0
    assert torch.equal(batch["input_ids"], input_ids)
    assert torch.equal(batch["labels"], labels)
    assert torch.equal(batch["loss_weights"], weights)


@pytest.mark.parametrize("num_tokens", [0, 64])
def test_no_supervision_preserves_zero_head_gradients(num_tokens):
    head = nn.Linear(4, 8, dtype=torch.float16)
    hidden = torch.ones(num_tokens, 4, dtype=torch.float16, requires_grad=True)
    labels = torch.full((num_tokens,), IGNORE_INDEX, dtype=torch.long)
    loss = _ChunkedLinearCrossEntropy.apply(hidden, head.weight, head.bias, labels, torch.ones(num_tokens), 16)
    loss.backward()
    assert loss.item() == 0
    for parameter in head.parameters():
        assert parameter.grad.dtype == parameter.dtype
        assert torch.count_nonzero(parameter.grad) == 0
    assert torch.count_nonzero(hidden.grad) == 0
