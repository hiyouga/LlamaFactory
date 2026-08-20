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

"""Flash Linear Attention kernel plugin backed by FSDPTurbo's operator registry."""

from functools import partial

from ......accelerator.helper import DeviceType, get_current_accelerator
from ......utils import logging
from ......utils.types import HFModel
from ...base import BaseKernel, KernelPlugin


logger = logging.get_logger(__name__)

CHUNK_GATED_DELTA_RULE = "chunk_gated_delta_rule"
FUSED_RECURRENT_GATED_DELTA_RULE = "fused_recurrent_gated_delta_rule"
FLASH_LINEAR_ATTENTION_KERNELS = (
    CHUNK_GATED_DELTA_RULE,
    FUSED_RECURRENT_GATED_DELTA_RULE,
)
FLA_MODULE_ATTRIBUTES = {
    CHUNK_GATED_DELTA_RULE: "chunk_gated_delta_rule",
    FUSED_RECURRENT_GATED_DELTA_RULE: "recurrent_gated_delta_rule",
}
SUPPORTED_CHUNK_SIZES = (16, 32, 64)


@KernelPlugin("flash-linear-attention").register()
class FlashLinearAttentionKernel(BaseKernel):
    """Install selected FLA callables through FSDPTurbo's device operator registry."""

    @staticmethod
    def check_device() -> None:
        current = get_current_accelerator().type
        if current not in (DeviceType.CUDA, DeviceType.NPU):
            raise RuntimeError(f"FlashLinearAttentionKernel requires CUDA or NPU, current accelerator is {current}.")

    @staticmethod
    def check_deps() -> None:
        try:
            import fla.ops.gated_delta_rule  # noqa: F401
            import fsdp_turbo.ops.fla  # noqa: F401
            from fsdp_turbo.ops.registry import get_op  # noqa: F401
            from fsdp_turbo.utils.patch import patch_model_members  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Flash Linear Attention and FSDPTurbo are required for this kernel.") from exc

    @staticmethod
    def _apply(**kwargs) -> HFModel:
        model = kwargs["model"]
        config = kwargs.get("config") or {}
        include_kernels = config.get("include_kernels", "auto")
        chunk_size = config.get("chunk_size", 64)

        if include_kernels == "auto" or include_kernels is True:
            selected = list(FLASH_LINEAR_ATTENTION_KERNELS)
        elif isinstance(include_kernels, str):
            selected = [name.strip() for name in include_kernels.split(",") if name.strip()]
        else:
            raise TypeError("kernel_config.include_kernels must be 'auto' or a comma-separated string.")

        if not selected:
            raise ValueError("kernel_config.include_kernels must select at least one FLA kernel.")

        unsupported = set(selected).difference(FLASH_LINEAR_ATTENTION_KERNELS)
        if unsupported:
            raise ValueError(f"Unsupported Flash Linear Attention kernels: {sorted(unsupported)}")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size not in SUPPORTED_CHUNK_SIZES:
            raise ValueError(f"chunk_size must be one of {SUPPORTED_CHUNK_SIZES}, got {chunk_size!r}.")

        from fsdp_turbo.ops.registry import get_op
        from fsdp_turbo.utils.patch import patch_model_members

        patched = 0
        named_modules = tuple(model.named_modules())
        for op_name in selected:
            module_attribute = FLA_MODULE_ATTRIBUTES[op_name]
            op = get_op(op_name)
            configured_op = partial(op, chunk_size=chunk_size) if op_name == CHUNK_GATED_DELTA_RULE else op
            targets = {
                f"{type(module).__module__}.{type(module).__name__}.{module_attribute}"
                for _, module in named_modules
                if callable(getattr(module, module_attribute, None))
            }
            matched = patch_model_members(model, sorted(targets), configured_op) if targets else 0
            if matched == 0:
                raise RuntimeError(f"FLA operator `{op_name}` did not match any model module attributes.")
            patched += matched

        logger.info_rank0(f"Flash Linear Attention kernels updated {patched} module callables: {selected}.")
        return model
