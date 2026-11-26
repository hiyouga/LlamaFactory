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

from llamafactory.train.trainer_utils import dft_loss_func, eaft_loss_func


@pytest.mark.parametrize("loss_func", [dft_loss_func, eaft_loss_func])
@pytest.mark.parametrize("seq_len", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fully_masked_finetuning_loss_supports_backward(loss_func, seq_len, dtype):
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 24, dtype=dtype)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs = torch.randn(2, seq_len, 3, dtype=dtype)
    logits = model(inputs)
    labels = torch.full((2, seq_len), -100)
    weights_before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = loss_func({"logits": logits}, labels)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    optimizer.step()

    for parameter, previous in zip(model.parameters(), weights_before):
        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter), rtol=0, atol=0)
        torch.testing.assert_close(parameter, previous, rtol=0, atol=0)


@pytest.mark.parametrize("loss_func", [dft_loss_func, eaft_loss_func])
def test_masked_sequences_do_not_change_valid_loss_or_gradient(loss_func):
    torch.manual_seed(11)
    logits = torch.randn(1, 4, 24, requires_grad=True)
    labels = torch.tensor([[-100, 2, 5, 7]])
    expected = loss_func({"logits": logits}, labels)
    expected_gradient = torch.autograd.grad(expected, logits)[0]

    padded_logits = torch.cat([logits.detach(), torch.randn_like(logits)], dim=0).requires_grad_()
    padded_labels = torch.cat([labels, torch.full_like(labels, -100)], dim=0)
    actual = loss_func({"logits": padded_logits}, padded_labels)
    actual.backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(padded_logits.grad[:1], expected_gradient)
    torch.testing.assert_close(padded_logits.grad[1:], torch.zeros_like(padded_logits.grad[1:]))
