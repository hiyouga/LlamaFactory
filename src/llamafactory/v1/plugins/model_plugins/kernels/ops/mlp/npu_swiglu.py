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

"""The definition of NPU fused SwiGLU kernels.

Init Phase:
1. Define SwiGLU forward functions.
2. Register NPU fused SwiGLU kernel.

"""

import types

import torch

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


def npu_swiglu_forward(self, hidden_state):
    """SwiGLU forward pass for NPU.

    Args:
        self: The MLP layer instance.
        hidden_state (Tensor): Input hidden state.

    Returns:
        Tensor: Output of SwiGLU.
    """
    return self.down_proj(
        torch_npu.npu_swiglu(torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1), dim=-1)
    )


_MODEL_TYPE_TO_PATCHES = {
    "qwen3": {
        "Qwen3MLP": npu_swiglu_forward,
    },
    "qwen3_moe": {
        "Qwen3MoeMLP": npu_swiglu_forward,
    },
    "qwen3_next": {
        "Qwen3NextMLP": npu_swiglu_forward,
    },
    "qwen3_omni_moe": {
        "Qwen3OmniMoeThinkerTextMLP": npu_swiglu_forward,
        "Qwen3OmniMoeMLP": npu_swiglu_forward,
        "Qwen3OmniMoeTalkerTextMLP": npu_swiglu_forward,
        "Qwen3OmniMoeCode2WavMlp": npu_swiglu_forward,
    },
    "qwen3_omni_moe_thinker": {
        "Qwen3OmniMoeThinkerTextMLP": npu_swiglu_forward,
    },
    "qwen3_vl": {
        "Qwen3VLTextMLP": npu_swiglu_forward,
    },
    "qwen3_vl_moe": {
        "Qwen3VLMoeTextMLP": npu_swiglu_forward,
    },
    "qwen3_5": {
        "Qwen3_5MLP": npu_swiglu_forward,
    },
    "qwen3_5_text": {
        "Qwen3_5MLP": npu_swiglu_forward,
    },
    "qwen3_5_moe": {
        "Qwen3_5MoeMLP": npu_swiglu_forward,
    },
    "qwen3_5_moe_text": {
        "Qwen3_5MoeMLP": npu_swiglu_forward,
    },
}


@KernelPlugin("npu_fused_swiglu").register()
class NpuSwiGluKernel(BaseKernel):
    """NPU Kernel for fused SwiGLU activation."""

    @staticmethod
    def check_device() -> None:
        current = get_current_accelerator().type
        if current != DeviceType.NPU:
            raise RuntimeError(f"NpuSwiGluKernel requires NPU, current accelerator is {current}.")

    @staticmethod
    def check_deps() -> None:
        if _TORCH_NPU_IMPORT_ERROR is not None:
            raise RuntimeError("NpuSwiGluKernel requires torch_npu.") from _TORCH_NPU_IMPORT_ERROR

    @staticmethod
    def _get_patch_forward(model_type: str, module: torch.nn.Module):
        """Return the NPU forward function for a matched SwiGLU MLP module."""
        model_patches = _MODEL_TYPE_TO_PATCHES.get(model_type, {})
        patch_forward = model_patches.get(module.__class__.__name__)
        if patch_forward is None:
            return None

        config = getattr(module, "config", None)
        if getattr(config, "hidden_act", None) != "silu":
            return None

        return patch_forward

    @staticmethod
    def _apply(**kwargs) -> "HFModel":
        """Applies the NPU fused SwiGLU kernel to the model.

        Args:
            **kwargs: Keyword arguments containing the model.

        Returns:
            HFModel: The model with patched SwiGLU forward functions.
        """
        model = kwargs["model"]

        model_type = getattr(model.config, "model_type", None)
        if model_type not in _MODEL_TYPE_TO_PATCHES:
            return model

        patched_count = 0
        for module in model.modules():
            patch_forward = NpuSwiGluKernel._get_patch_forward(model_type, module)
            if patch_forward is not None:
                module.forward = types.MethodType(patch_forward, module)
                patched_count += 1

        if patched_count:
            logger.info_rank0(f"Applied NPU SwiGLU kernel to {patched_count} modules for model type: {model_type}.")

        return model
