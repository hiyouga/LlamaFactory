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

"""The definition of NPU fused RMSNorm kernels.

Init Phase:
1. Define RMSNorm forward function.
2. Register NPU fused RMSNorm kernel.

"""

import types

import torch
import torch.nn.functional as F

from ......accelerator.helper import DeviceType, get_current_accelerator
from ......utils.logging import get_logger
from ......utils.types import HFModel
from ...base import BaseKernel, KernelPlugin


logger = get_logger(__name__)

try:
    import torch_npu
except ImportError as exc:
    _TORCH_NPU_IMPORT_ERROR = exc
else:
    _TORCH_NPU_IMPORT_ERROR = None


def npu_rms_norm_forward(self, hidden_states):
    """NPU forward implementation for standard RMSNorm.

    Args:
        self (nn.Module): The RMSNorm module instance with ``weight`` and either ``variance_epsilon`` or ``eps``.
        hidden_states (Tensor): Input hidden states tensor.

    Returns:
        Tensor: Normalized tensor consistent with the baseline RMSNorm behavior.
    """
    _eps = getattr(self, "variance_epsilon", None) or getattr(self, "eps", 1e-6)

    weight = getattr(self, "weight", None)
    if weight is None:
        raise RuntimeError(f"{self.__class__.__name__} has no RMSNorm weight for NPU RMSNorm kernel.")

    effective_weight = weight.float()

    return torch_npu.npu_rms_norm(hidden_states, effective_weight.to(hidden_states.dtype), epsilon=_eps)[0]


def npu_residual_rms_norm_forward(self, hidden_states):
    """NPU forward implementation for residual RMSNorm.

    Residual RMSNorm uses ``scale = 1.0 + weight`` where ``weight`` is initialized
    to 0 in the original transformers implementation.

    Args:
        self (nn.Module): The residual RMSNorm module with ``weight`` and either ``variance_epsilon`` or ``eps``.
        hidden_states (Tensor): Input hidden states tensor.

    Returns:
        Tensor: Normalized tensor consistent with residual RMSNorm behavior.
    """
    _eps = getattr(self, "variance_epsilon", None) or getattr(self, "eps", 1e-6)

    weight = getattr(self, "weight", None)
    if weight is None:
        raise RuntimeError(f"{self.__class__.__name__} has no RMSNorm weight for NPU RMSNorm kernel.")

    effective_weight = 1.0 + weight.float()

    return torch_npu.npu_rms_norm(hidden_states, effective_weight.to(hidden_states.dtype), epsilon=_eps)[0]


def npu_gated_rms_norm_forward(self, hidden_states, gate=None):
    """NPU forward implementation for Gated RMSNorm with high-precision FP32 computation.

    This function performs RMSNorm and gated SiLU multiplication in FP32 for numerical
    stability. The supported gated RMSNorm modules use ``scale = weight`` with weight
    initialized to 1, unlike the residual RMSNorm variants that use ``1.0 + weight``.

    Args:
        self (nn.Module): The Gated RMSNorm module instance.
        hidden_states (Tensor): Input hidden states tensor.
        gate (Tensor): Gate tensor for SiLU activation.

    Returns:
        Tensor: Output tensor cast back to the original input dtype.

    Raises:
        ValueError: If the gate tensor is not provided.
    """
    if gate is None:
        raise ValueError(f"{self.__class__.__name__} requires a gate tensor for NPU Gated RMSNorm.")

    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    _eps = getattr(self, "variance_epsilon", None) or getattr(self, "eps", 1e-6)

    hidden_states = torch_npu.npu_rms_norm(hidden_states, self.weight.float(), epsilon=_eps)[0]
    hidden_states = hidden_states * F.silu(gate.to(torch.float32))

    return hidden_states.to(input_dtype)


_MODEL_TYPE_TO_PATCHES = {
    "qwen3": {
        "Qwen3RMSNorm": npu_rms_norm_forward,
    },
    "qwen3_moe": {
        "Qwen3MoeRMSNorm": npu_rms_norm_forward,
    },
    "qwen3_next": {
        "Qwen3NextRMSNorm": npu_residual_rms_norm_forward,
        "Qwen3NextRMSNormGated": npu_gated_rms_norm_forward,
    },
    "qwen3_omni_moe": {
        "Qwen3OmniMoeThinkerTextRMSNorm": npu_rms_norm_forward,
        "Qwen3OmniMoeTextRMSNorm": npu_rms_norm_forward,
        "Qwen3OmniMoeRMSNorm": npu_rms_norm_forward,
        "Qwen3OmniMoeCode2WavRMSNorm": npu_rms_norm_forward,
    },
    "qwen3_omni_moe_thinker": {
        "Qwen3OmniMoeThinkerTextRMSNorm": npu_rms_norm_forward,
        "Qwen3OmniMoeTextRMSNorm": npu_rms_norm_forward,
    },
    "qwen3_vl": {
        "Qwen3VLTextRMSNorm": npu_rms_norm_forward,
    },
    "qwen3_vl_moe": {
        "Qwen3VLMoeTextRMSNorm": npu_rms_norm_forward,
    },
    "qwen3_5": {
        "Qwen3_5RMSNorm": npu_residual_rms_norm_forward,
        "Qwen3_5RMSNormGated": npu_gated_rms_norm_forward,
    },
    "qwen3_5_text": {
        "Qwen3_5RMSNorm": npu_residual_rms_norm_forward,
        "Qwen3_5RMSNormGated": npu_gated_rms_norm_forward,
    },
    "qwen3_5_moe": {
        "Qwen3_5MoeRMSNorm": npu_residual_rms_norm_forward,
        "Qwen3_5MoeRMSNormGated": npu_gated_rms_norm_forward,
    },
    "qwen3_5_moe_text": {
        "Qwen3_5MoeRMSNorm": npu_residual_rms_norm_forward,
        "Qwen3_5MoeRMSNormGated": npu_gated_rms_norm_forward,
    },
}


@KernelPlugin("npu_fused_rmsnorm").register()
class NpuRMSNormKernel(BaseKernel):
    """NPU kernel wrapper for RMSNorm that applies the replacement within a model."""

    @staticmethod
    def check_device() -> None:
        current = get_current_accelerator().type
        if current != DeviceType.NPU:
            raise RuntimeError(f"NpuRMSNormKernel requires NPU, current accelerator is {current}.")

    @staticmethod
    def check_deps() -> None:
        if _TORCH_NPU_IMPORT_ERROR is not None:
            raise RuntimeError("NpuRMSNormKernel requires torch_npu.") from _TORCH_NPU_IMPORT_ERROR

    @staticmethod
    def _get_patch_forward(model_type: str, module: torch.nn.Module):
        """Return the NPU forward function for a matched RMSNorm module."""
        model_patches = _MODEL_TYPE_TO_PATCHES.get(model_type, {})
        return model_patches.get(module.__class__.__name__)

    @staticmethod
    def _apply(**kwargs) -> "HFModel":
        """Iterate the model and apply NPU-optimized forward to matched RMSNorm modules.

        Matches modules configured for the current model type, then binds the corresponding
        NPU-optimized forward function as an instance method via ``types.MethodType`` to
        replace the original ``forward``.

        Args:
            **kwargs: Keyword arguments containing the model.

        Returns:
            HFModel: The model with NPU fused RMSNorm.
        """
        model = kwargs["model"]

        model_type = getattr(model.config, "model_type", None)
        if model_type not in _MODEL_TYPE_TO_PATCHES:
            return model

        patched_count = 0
        for module in model.modules():
            patch_forward = NpuRMSNormKernel._get_patch_forward(model_type, module)
            if patch_forward is not None:
                module.forward = types.MethodType(patch_forward, module)
                patched_count += 1

        if patched_count:
            logger.info_rank0(f"Applied NPU RMSNorm kernel to {patched_count} modules for model type: {model_type}.")

        return model
