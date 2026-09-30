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

"""Pure-Triton Fused MoE Kernel for NVIDIA GPUs.

Replaces the HuggingFace per-expert Python loop with a fully fused Triton pipeline:
- Forward: scatter → grouped GEMM fc1 → SiLU·gate → apply routing → grouped GEMM fc2 → gather
- Backward: all dX via grouped GEMM, all dW via grouped GEMM (no Python loops)

Supported models: Mixtral, Qwen3-MoE, Qwen3.5-MoE.
"""

import logging
import types

import torch
import torch.nn.functional as F

from ......accelerator.helper import DeviceType, get_current_accelerator
from ......utils.types import HFModel
from ...base import BaseKernel, KernelPlugin


try:
    from .triton_grouped_gemm import (
        group_gemm_same_mn,
        group_gemm_same_nk,
        moe_gather,
        moe_scatter,
    )
except ImportError as exc:
    _TRITON_IMPORT_ERROR = exc
else:
    _TRITON_IMPORT_ERROR = None


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Autograd Function: Full Triton MoE forward + backward
# ---------------------------------------------------------------------------


class TritonFusedMoeFunction(torch.autograd.Function):
    """Fused MoE expert computation using Triton grouped GEMMs.

    Forward: scatter → fc1 (group GEMM) → SiLU·gate → weight → fc2 (group GEMM) → gather
    Backward: all gradients computed via grouped GEMMs in single kernel launches.
    """

    @staticmethod
    def forward(
        ctx,
        num_experts,
        gate_weights,
        expert_index,
        hidden_states,
        fc1_weight,
        fc2_weight,
    ):
        """Forward pass.

        Args:
            ctx: autograd context
            num_experts: int
            gate_weights: (num_tokens, top_k) routing weights
            expert_index: (num_tokens, top_k) expert assignments
            hidden_states: (num_tokens, hidden_dim)
            fc1_weight: (E, 2*inter, hidden) merged gate+up weight
            fc2_weight: (E, hidden, inter) down projection weight
        """
        # Compute scatter index: maps (token, topk) → position in sorted buffer
        scatter_index = expert_index.flatten().argsort(stable=True).argsort().int().view(expert_index.shape)

        # Token counts per expert and cumulative boundaries
        splits = torch.zeros(num_experts, dtype=torch.int32, device=hidden_states.device)
        flat_experts = expert_index.flatten().int()
        splits.scatter_add_(0, flat_experts.long(), torch.ones_like(flat_experts))
        cumsum_t = torch.cumsum(splits, dim=0)

        # Scatter hidden states to sorted expert buffer
        scatter_output = moe_scatter(hidden_states, scatter_index)

        # FC1: grouped GEMM (scatter_output @ fc1_weight.T)
        max_M = int(splits.max().item())
        fc1_output = group_gemm_same_nk(
            a=scatter_output,
            b=fc1_weight,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=True,
        )

        # SiLU gate activation
        fc1_1_output, fc1_2_output = fc1_output.chunk(2, dim=-1)
        fc1_1_activation = torch.nn.functional.silu(fc1_1_output)
        fc1_activation = fc1_1_activation * fc1_2_output

        # Apply routing weights before fc2 (mathematically equivalent to after)
        reshaped_gate_weight = gate_weights.reshape(-1, 1)
        scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
        scattered_gate_weight[scatter_index.flatten().long()] = reshaped_gate_weight
        fc1_weighted_output = fc1_activation * scattered_gate_weight

        # FC2: grouped GEMM (fc1_weighted @ fc2_weight.T)
        fc2_output = group_gemm_same_nk(
            a=fc1_weighted_output,
            b=fc2_weight,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=True,
        )

        # Gather back to original token positions (sum over topk)
        expert_output = moe_gather(fc2_output, scatter_index)

        ctx.num_experts = num_experts
        ctx.save_for_backward(
            gate_weights,
            fc1_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            scatter_output,
            cumsum_t,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
        )

        return expert_output

    @staticmethod
    def backward(ctx, grad_output):
        (
            gate_weights,
            fc1_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            scatter_output,
            cumsum_t,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
        ) = ctx.saved_tensors
        num_experts = ctx.num_experts
        hidden_dim = grad_output.shape[-1]
        grad_output = grad_output.reshape(-1, hidden_dim).contiguous()

        # Recompute max_M from cumsum
        splits = torch.zeros(num_experts, dtype=cumsum_t.dtype, device=cumsum_t.device)
        splits[0] = cumsum_t[0]
        splits[1:] = cumsum_t[1:] - cumsum_t[:-1]
        max_M = int(splits.max().item())

        # Step 1: Scatter grad_output to expert buffer
        grad_fc2_output = moe_scatter(grad_output, scatter_index)

        # Step 2: FC2 backward
        # dX for fc2: grad_fc2_output @ fc2_weight (transpose_b=False since fc2 is (E, hidden, inter))
        grad_fc1_weighted_output = group_gemm_same_nk(
            a=grad_fc2_output,
            b=fc2_weight,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=False,
        )

        # dW for fc2: grad_fc2_output.T @ fc1_weighted_output
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = torch.empty_like(fc2_weight)
            group_gemm_same_mn(
                a=grad_fc2_output,
                b=fc1_weighted_output,
                c=grad_fc2_weight,
                cumsum_K=cumsum_t,
            )

        # Step 3: Routing weight backward
        grad_fc1_activation = grad_fc1_weighted_output * scattered_gate_weight
        grad_scattered_gate_weight = torch.sum(fc1_activation * grad_fc1_weighted_output, dim=-1)
        grad_gate_weight = grad_scattered_gate_weight[scatter_index.flatten().long()]
        grad_gate_weight = grad_gate_weight.reshape(gate_weights.shape)

        # Recompute silu activation for backward
        fc1_1_activation = torch.nn.functional.silu(fc1_1_output)

        # Step 4: SiLU gate backward
        grad_fc1_1_activation = grad_fc1_activation * fc1_2_output
        grad_fc1_2_output = fc1_1_activation * grad_fc1_activation

        # SiLU backward: d/dx[x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)

        # Merge fc1 gradients back to (total_M, 2*inter)
        grad_fc1_output = torch.cat([grad_fc1_1_output, grad_fc1_2_output], dim=-1)

        # Step 5: FC1 backward
        # dX for fc1: grad_fc1_output @ fc1_weight (transpose_b=False)
        grad_scatter_output = group_gemm_same_nk(
            a=grad_fc1_output,
            b=fc1_weight,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=False,
        )

        # dW for fc1: grad_fc1_output.T @ scatter_output
        grad_fc1_weight = None
        if fc1_weight.requires_grad:
            grad_fc1_weight = torch.empty_like(fc1_weight)
            group_gemm_same_mn(
                a=grad_fc1_output,
                b=scatter_output,
                c=grad_fc1_weight,
                cumsum_K=cumsum_t,
            )

        # Step 6: Gather gradients back to original positions
        grad_hidden_states = moe_gather(grad_scatter_output, scatter_index)
        grad_hidden_states = grad_hidden_states.reshape(hidden_states.shape)

        return (
            None,  # num_experts
            grad_gate_weight,  # gate_weights
            None,  # expert_index
            grad_hidden_states,  # hidden_states
            grad_fc1_weight,  # fc1_weight
            grad_fc2_weight,  # fc2_weight
        )


# ---------------------------------------------------------------------------
# Patched forward functions
# ---------------------------------------------------------------------------


def _triton_moe_experts_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Replacement forward for v5+ MoE expert modules with stacked 3D weights."""
    return TritonFusedMoeFunction.apply(
        self.num_experts,
        top_k_weights.to(hidden_states.dtype),
        top_k_index,
        hidden_states,
        self.gate_up_proj,
        self.down_proj,
    )


# ---------------------------------------------------------------------------
# Legacy (transformers < 5.0) support: weight stacking + SparseMoeBlock patch
# ---------------------------------------------------------------------------


class _StackedExpertWeights(torch.nn.Module):
    """Lightweight container holding stacked 3D expert weight tensors."""

    def __init__(self, gate_up_proj: torch.Tensor, down_proj: torch.Tensor, num_experts: int):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(gate_up_proj)
        self.down_proj = torch.nn.Parameter(down_proj)
        self.num_experts = num_experts


def _stack_expert_weights(module: torch.nn.Module) -> None:
    """Replace nn.ModuleList of individual experts with stacked 3D parameter tensors."""
    experts = module.experts
    num_experts = len(experts)

    gate_up_list = []
    for expert in experts:
        gate_w = expert.gate_proj.weight.data  # (inter, hidden)
        up_w = expert.up_proj.weight.data  # (inter, hidden)
        gate_up_list.append(torch.cat([gate_w, up_w], dim=0))  # (2*inter, hidden)
    gate_up_proj = torch.stack(gate_up_list, dim=0)  # (E, 2*inter, hidden)

    down_proj = torch.stack([e.down_proj.weight.data for e in experts], dim=0)  # (E, hidden, inter)

    module.experts = _StackedExpertWeights(gate_up_proj, down_proj, num_experts)
    logger.info(
        f"cuda_fused_moe: Stacked {num_experts} expert weights into "
        f"gate_up_proj {tuple(gate_up_proj.shape)}, down_proj {tuple(down_proj.shape)}"
    )


def _triton_moe_sparse_block_forward(self, hidden_states: torch.Tensor):
    """Replacement forward for legacy SparseMoeBlock with inline routing."""
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    router_logits = self.gate(hidden_states)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = TritonFusedMoeFunction.apply(
        self.num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        self.experts.gate_up_proj,
        self.experts.down_proj,
    )

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


# ---------------------------------------------------------------------------
# Module mapping
# ---------------------------------------------------------------------------

_TRITON_MOE_MAPPING: dict[str, dict[str, object]] = {
    "MixtralForCausalLM": {
        "MixtralExperts": _triton_moe_experts_forward,
    },
    "Qwen3MoeForCausalLM": {
        "Qwen3MoeExperts": _triton_moe_experts_forward,
        "Qwen3MoeSparseMoeBlock": _triton_moe_sparse_block_forward,
    },
    "Qwen3_5MoeForCausalLM": {
        "Qwen3_5MoeExperts": _triton_moe_experts_forward,
    },
    "Qwen3_5MoeForConditionalGeneration": {
        "Qwen3_5MoeExperts": _triton_moe_experts_forward,
    },
}


# ---------------------------------------------------------------------------
# Kernel registration
# ---------------------------------------------------------------------------


@KernelPlugin("cuda_fused_moe").register()
class CudaFusedMoEKernel(BaseKernel):
    """Pure-Triton fused MoE kernel for NVIDIA CUDA GPUs.

    Replaces HuggingFace per-expert Python loops with a fully fused Triton pipeline:
    - Forward: scatter + grouped GEMMs + gather (single kernel per GEMM)
    - Backward: all dX and dW via grouped GEMMs (no Python loops)

    Requires: CUDA GPU + Triton
    """

    @staticmethod
    def check_device() -> None:
        current = get_current_accelerator().type
        if current != DeviceType.CUDA:
            raise RuntimeError(f"CudaFusedMoEKernel requires CUDA, current accelerator is {current}.")

    @staticmethod
    def check_deps() -> None:
        if _TRITON_IMPORT_ERROR is not None:
            raise RuntimeError("cuda_fused_moe requires Triton.") from _TRITON_IMPORT_ERROR

    @staticmethod
    def _apply(**kwargs) -> HFModel:
        model = kwargs.get("model")

        archs = getattr(model.config, "architectures", None) or []
        target_mapping = None
        for arch in archs:
            if arch in _TRITON_MOE_MAPPING:
                target_mapping = _TRITON_MOE_MAPPING[arch]
                break

        if target_mapping is None:
            logger.info(
                f"cuda_fused_moe: Model architecture {archs} not supported. "
                f"Supported: {list(_TRITON_MOE_MAPPING.keys())}"
            )
            return model

        patched_count = 0
        for module in model.modules():
            class_name = module.__class__.__name__
            if class_name not in target_mapping:
                continue

            target_fn = target_mapping[class_name]

            if hasattr(module, "gate_up_proj") and hasattr(module, "down_proj"):
                module.forward = types.MethodType(target_fn, module)
                patched_count += 1
            elif (
                hasattr(module, "experts")
                and isinstance(module.experts, torch.nn.ModuleList)
                and hasattr(module, "gate")
            ):
                _stack_expert_weights(module)
                module.forward = types.MethodType(target_fn, module)
                patched_count += 1

        if patched_count > 0:
            logger.info(f"cuda_fused_moe: Patched {patched_count} MoE expert modules with pure Triton pipeline.")
        else:
            logger.warning("cuda_fused_moe: No MoE expert modules found to patch.")

        return model
