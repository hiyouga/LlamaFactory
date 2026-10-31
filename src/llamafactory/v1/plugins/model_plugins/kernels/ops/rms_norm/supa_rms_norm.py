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

"""The definition of the SUPA fused RMSNorm kernel."""

import types
from functools import lru_cache

import torch

from ......accelerator.helper import DeviceType, get_current_accelerator
from ......utils.logging import get_logger
from ......utils.types import HFModel
from ...base import BaseKernel, KernelPlugin


logger = get_logger(__name__)


# Standard-convention RMSNorm classes: ``y = x * rsqrt(mean(x^2) + eps) * weight``.
# The fused ``sudnn.rms_norm_func`` matches this convention, so only these classes are patched.
# Variants with a different weight convention (e.g. Gemma's ``1 + weight``, or gated/residual
# RMSNorm) are intentionally excluded and fall back to the eager path. Add new class names here
# after verifying they follow the standard convention.
_SUPPORTED_RMSNORM_CLASSES = frozenset(
    {
        "LlamaRMSNorm",
        "MistralRMSNorm",
        "MixtralRMSNorm",
        "Phi3RMSNorm",
        "InternLM2RMSNorm",
        "GlmRMSNorm",
        "DeepseekV2RMSNorm",
        "DeepseekV3RMSNorm",
        "Qwen2RMSNorm",
        "Qwen2MoeRMSNorm",
        "Qwen2VLRMSNorm",
        "Qwen2_5_VLRMSNorm",
        "Qwen2_5_VLTextRMSNorm",
        "Qwen3RMSNorm",
        "Qwen3MoeRMSNorm",
        "Qwen3VLTextRMSNorm",
        "Qwen3VLMoeTextRMSNorm",
    }
)



@lru_cache
def _is_sudnn_rms_norm_available() -> bool:
    """Return whether the fused SUPA RMSNorm operator is registered."""
    try:
        import torch_supa  # noqa: F401
        import torch_supa_ext  # noqa: F401
    except Exception:
        return False

    return hasattr(torch.ops, "sudnn") and hasattr(torch.ops.sudnn, "rms_norm_func")


class SupaRMSNormFunction(torch.autograd.Function):
    """Autograd wrapper around the fused SUPA RMSNorm forward operator."""

    @staticmethod
    def forward(ctx, hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        weight = weight.to(dtype=hidden_states.dtype)
        bias = torch.zeros_like(weight)
        output = torch.ops.sudnn.rms_norm_func(hidden_states, weight, bias, eps, False, False)
        ctx.save_for_backward(hidden_states, weight)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden_states, weight = ctx.saved_tensors
        eps = ctx.eps
        hidden_size = hidden_states.shape[-1]

        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        rrms = torch.rsqrt(variance + eps)
        x_norm = hidden_states * rrms

        grad_input = grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_weight = (grad_output * x_norm).reshape(-1, hidden_size).sum(0).to(weight.dtype)

        if ctx.needs_input_grad[0]:
            grad_norm = grad_output * weight
            dot = (grad_norm * hidden_states).sum(-1, keepdim=True)
            grad_input = rrms * grad_norm - (rrms.pow(3) / hidden_size) * hidden_states * dot
            grad_input = grad_input.to(hidden_states.dtype)

        return grad_input, grad_weight, None


def supa_rms_norm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Run the fused kernel for standard RMSNorm modules."""
    if hidden_states.device.type != DeviceType.SUPA:
        return self._supa_rmsnorm_original_forward(hidden_states)

    return SupaRMSNormFunction.apply(hidden_states, self.weight, self.variance_epsilon)


@KernelPlugin("supa_fused_rmsnorm").register()
class SupaRMSNormKernel(BaseKernel):
    """SUPA fused RMSNorm kernel for trainable models."""

    @staticmethod
    def check_device() -> None:
        current = get_current_accelerator().type
        if current != DeviceType.SUPA:
            raise RuntimeError(f"SupaRMSNormKernel requires SUPA, current accelerator is {current}.")

    @staticmethod
    def _apply(**kwargs) -> HFModel:
        model = kwargs["model"]

        if not _is_sudnn_rms_norm_available():
            logger.warning_rank0(
                "Fused SUPA RMSNorm (torch.ops.sudnn.rms_norm_func) is unavailable; "
                "the eager RMSNorm path will be used. Ensure torch_supa_ext is installed."
            )
            return model

        patched_count = 0
        patched_classes = set()
        for module in model.modules():
            cls = module.__class__
            if getattr(module, "_supa_rmsnorm_patched", False):
                continue
            if cls.__name__ not in _SUPPORTED_RMSNORM_CLASSES:
                continue

            module._supa_rmsnorm_original_forward = module.forward
            module.forward = types.MethodType(supa_rms_norm_forward, module)
            module._supa_rmsnorm_patched = True
            patched_count += 1
            patched_classes.add(cls)

        if patched_count:
            logger.info_rank0(
                "Applied fused SUPA RMSNorm to {} modules: {}.".format(
                    patched_count, ", ".join(sorted(cls.__name__ for cls in patched_classes))
                )
            )

        return model
