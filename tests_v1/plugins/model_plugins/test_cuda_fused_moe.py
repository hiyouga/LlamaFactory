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
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from llamafactory.v1.plugins.model_plugins.kernels.ops.mlp.cuda_fused_moe import TritonFusedMoeFunction


pytestmark = pytest.mark.runs_on(["cuda"])


def _inputs(dtype, frozen=False, unbalanced=False):
    torch.manual_seed(123)
    n, h, intermediate, experts, top_k = 97, 64, 96, 4, 2
    x = torch.randn(n, h, device="cuda", dtype=dtype).requires_grad_()
    w1 = (torch.randn(experts, 2 * intermediate, h, device="cuda", dtype=dtype) / h**0.5).requires_grad_(not frozen)
    w2 = (torch.randn(experts, h, intermediate, device="cuda", dtype=dtype) / intermediate**0.5).requires_grad_(
        not frozen
    )
    gates = torch.rand(n, top_k, device="cuda", dtype=dtype).requires_grad_()
    indices = torch.rand(n, experts, device="cuda").topk(top_k, dim=-1).indices
    if unbalanced:
        indices[:] = torch.tensor([0, 2], device="cuda")
        indices[0, 1] = 1
    return experts, gates, indices, x, w1, w2


def _reference(experts, gates, indices, x, w1, w2):
    output = torch.zeros_like(x)
    for expert in range(experts):
        token, slot = torch.where(indices == expert)
        gate, up = F.linear(x[token], w1[expert]).chunk(2, dim=-1)
        weighted = (F.silu(gate) * up) * gates[token, slot, None]
        output.index_add_(0, token, F.linear(weighted, w2[expert]))
    return output


def test_saved_activation_memory():
    inputs = _inputs(torch.bfloat16)
    saved = {}
    input_storage = {t.untyped_storage().data_ptr() for t in inputs if isinstance(t, torch.Tensor)}

    def pack(tensor):
        storage = tensor.untyped_storage()
        if storage.data_ptr() not in input_storage:
            saved[storage.data_ptr()] = storage.nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = TritonFusedMoeFunction.apply(*inputs)
    # Only gate/up activations plus routing metadata should survive the forward.
    tokens, top_k = inputs[1].shape
    intermediate = inputs[-1].shape[-1]
    budget = tokens * top_k * 2 * intermediate * inputs[3].element_size() + tokens * top_k * 16 + 128
    assert sum(saved.values()) <= budget
    output.sum().backward()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("frozen,unbalanced", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_forward_backward(dtype, frozen, unbalanced, checkpointed):
    inputs = _inputs(dtype, frozen, unbalanced)
    reference_inputs = tuple(
        t.detach().clone().requires_grad_(t.requires_grad) if isinstance(t, torch.Tensor) else t for t in inputs
    )
    expected = _reference(*reference_inputs)
    if checkpointed:
        actual = checkpoint(TritonFusedMoeFunction.apply, *inputs, use_reentrant=False)
    else:
        actual = TritonFusedMoeFunction.apply(*inputs)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    tolerance = {torch.float16: 0.006, torch.bfloat16: 0.05}[dtype]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    for actual_input, reference_input in zip(inputs[1:], reference_inputs[1:]):
        if actual_input.requires_grad:
            torch.testing.assert_close(actual_input.grad, reference_input.grad, atol=tolerance, rtol=tolerance)
        else:
            assert actual_input.grad is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("intermediate", [1, 96, 768, 1536, 14336])
def test_pointwise_matches_eager(dtype, intermediate):
    # Test FP32 directly here: the unchanged grouped GEMMs use TF32 for FP32 inputs.
    from llamafactory.v1.plugins.model_plugins.kernels.ops.mlp.triton_moe_activation import (
        moe_activation,
        moe_activation_backward,
    )

    torch.manual_seed(42)
    fc1 = torch.randn(17, 2 * intermediate, device="cuda", dtype=dtype).requires_grad_()
    weights = torch.rand(17, 1, device="cuda", dtype=dtype).requires_grad_()
    gate, up = fc1.chunk(2, dim=-1)
    expected = F.silu(gate) * up * weights
    grad = torch.randn_like(expected)
    expected.backward(grad)
    actual = moe_activation(fc1.detach(), weights.detach())
    grad_fc1, grad_weights = moe_activation_backward(fc1.detach(), weights.detach(), grad)
    tolerance = {torch.float16: 0.004, torch.bfloat16: 0.03, torch.float32: 1e-5}[dtype]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(grad_fc1, fc1.grad, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(grad_weights, weights.grad.flatten(), atol=tolerance, rtol=tolerance)
