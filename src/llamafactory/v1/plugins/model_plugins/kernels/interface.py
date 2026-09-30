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

from typing import Any

from ....accelerator.helper import DeviceType, get_current_accelerator
from ....utils.types import HFModel
from .base import KernelPlugin

# Import built-in implementations so their class decorators populate the registry.
from .liger_kernel_ops import LigerKernel  # noqa: F401
from .ops.linear_attention.fla import FlashLinearAttentionKernel  # noqa: F401
from .ops.mlp.cuda_fused_moe import CudaFusedMoEKernel  # noqa: F401
from .ops.mlp.npu_fused_moe import NpuFusedMoEKernel  # noqa: F401
from .ops.mlp.npu_swiglu import NpuSwiGluKernel  # noqa: F401
from .ops.rms_norm.npu_rms_norm import NpuRMSNormKernel  # noqa: F401
from .ops.rope.npu_rope import NpuRoPEKernel  # noqa: F401


_AUTO_KERNELS = {
    DeviceType.NPU: ("npu_fused_moe", "npu_fused_rmsnorm", "npu_fused_rope", "npu_fused_swiglu"),
}


def _apply_auto_kernels(model: HFModel, **kwargs) -> HFModel:
    device_type = get_current_accelerator().type
    for kernel_name in _AUTO_KERNELS.get(device_type, ()):
        model = KernelPlugin(kernel_name).apply(model=model, **kwargs)

    return model


def apply_kernels(model: HFModel, config: dict[str, Any], require_logits: bool = False) -> HFModel:
    """Apply the comma-separated kernel names selected by ``kernel_config.name``."""
    kernel_names = config.get("name")
    if not isinstance(kernel_names, str):
        raise TypeError("kernel_config.name must be a string.")

    names = [name.strip() for name in kernel_names.split(",") if name.strip()]
    if not names:
        raise ValueError("kernel_config.name must contain at least one kernel name.")

    for name in names:
        if name == "auto":
            model = _apply_auto_kernels(model=model, config=config, require_logits=require_logits)
        else:
            model = KernelPlugin(name).apply(model=model, config=config, require_logits=require_logits)

    return model


def apply_v1_kernels(model: HFModel, use_v1_kernels: bool) -> HFModel:
    """Apply v1 automatic kernels for the transitional v0 ``use_v1_kernels`` option."""
    if not use_v1_kernels:
        return model

    return apply_kernels(model, {"name": "auto"})


def apply_kernel(kernel_id: str, **kwargs) -> HFModel:
    if kernel_id == "auto":
        return _apply_auto_kernels(**kwargs)

    return KernelPlugin(kernel_id).apply(**kwargs)
