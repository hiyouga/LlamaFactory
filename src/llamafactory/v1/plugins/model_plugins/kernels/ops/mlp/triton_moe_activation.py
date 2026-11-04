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

"""Fused SiLU and routing for the CUDA MoE autograd function."""

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_activation_kernel(fc1, weights, output, I: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    dtype = fc1.dtype.element_ty
    gate = tl.load(fc1 + row * (2 * I) + col, col < I, other=0).to(tl.float32)
    up = tl.load(fc1 + row * (2 * I) + I + col, col < I, other=0).to(tl.float32)
    weight = tl.load(weights + row).to(tl.float32)
    # Preserve the eager implementation's rounding between pointwise operations.
    silu = (gate * tl.sigmoid(gate)).to(dtype).to(tl.float32)
    activation = (silu * up).to(dtype).to(tl.float32)
    tl.store(output + row * I + col, activation * weight, col < I)


@triton.jit
def _moe_activation_backward_kernel(fc1, weights, grad, grad_fc1, grad_weights, I: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    dtype = fc1.dtype.element_ty
    gate = tl.load(fc1 + row * (2 * I) + col, col < I, other=0).to(tl.float32)
    up = tl.load(fc1 + row * (2 * I) + I + col, col < I, other=0).to(tl.float32)
    weight = tl.load(weights + row).to(tl.float32)
    dy = tl.load(grad + row * I + col, col < I, other=0).to(tl.float32)
    sigmoid = tl.sigmoid(gate)
    silu = (gate * sigmoid).to(dtype).to(tl.float32)
    activation = (silu * up).to(dtype).to(tl.float32)
    dweight = (activation * dy).to(dtype).to(tl.float32)
    tl.store(grad_weights + row, tl.sum(tl.where(col < I, dweight, 0), axis=0))
    dactivation = (dy * weight).to(dtype).to(tl.float32)
    dsilu = (dactivation * up).to(dtype).to(tl.float32)
    dgate = dsilu * (sigmoid * (1 + gate * (1 - sigmoid)))
    tl.store(grad_fc1 + row * (2 * I) + col, dgate, col < I)
    tl.store(grad_fc1 + row * (2 * I) + I + col, silu * dactivation, col < I)


def moe_activation(fc1: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute rounded SiLU(gate) * up * routing weight in one kernel."""
    rows, twice_intermediate = fc1.shape
    intermediate = twice_intermediate // 2
    output = fc1.new_empty((rows, intermediate))
    _moe_activation_kernel[(rows,)](
        fc1, weights, output, intermediate, triton.next_power_of_2(intermediate), enable_fp_fusion=False
    )
    return output


def moe_activation_backward(
    fc1: torch.Tensor, weights: torch.Tensor, grad: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute activations and return gate/up and routing gradients."""
    rows, twice_intermediate = fc1.shape
    intermediate = twice_intermediate // 2
    grad_fc1 = torch.empty_like(fc1)
    grad_weights = weights.new_empty((rows,))
    _moe_activation_backward_kernel[(rows,)](
        fc1,
        weights,
        grad,
        grad_fc1,
        grad_weights,
        intermediate,
        triton.next_power_of_2(intermediate),
        enable_fp_fusion=False,
    )
    return grad_fc1, grad_weights
