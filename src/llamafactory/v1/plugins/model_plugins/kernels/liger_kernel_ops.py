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

"""The definition of Liger Kernel.

Init Phase:
1. Define LigerKernel class.
2. Register Liger kernel.

"""

import inspect

from ....accelerator.helper import DeviceType, get_current_accelerator
from ....utils.logging import get_logger
from ....utils.types import HFModel
from .base import BaseKernel, KernelPlugin


logger = get_logger(__name__)

_LIGER_FN_BY_MODEL_TYPE: dict[str, str] = {
    "qwen3": "apply_liger_kernel_to_qwen3",
    "qwen3_moe": "apply_liger_kernel_to_qwen3_moe",
    "qwen3_next": "apply_liger_kernel_to_qwen3_next",
    "qwen3_5": "apply_liger_kernel_to_qwen3_5",
    "qwen3_5_text": "apply_liger_kernel_to_qwen3_5_text",
    "qwen3_5_moe": "apply_liger_kernel_to_qwen3_5_moe",
    "qwen3_5_moe_text": "apply_liger_kernel_to_qwen3_5_moe_text",
}


@KernelPlugin("liger_kernel").register()
class LigerKernel(BaseKernel):
    """Liger Kernel for optimized model training."""

    @staticmethod
    def check_device() -> None:
        current = get_current_accelerator().type
        if current not in (DeviceType.CUDA, DeviceType.NPU):
            raise RuntimeError(f"LigerKernel requires CUDA or NPU, current accelerator is {current}.")

    @staticmethod
    def check_deps() -> None:
        """Checks if the required dependencies for the kernel are available."""
        try:
            import liger_kernel  # noqa: F401
        except ImportError:
            raise RuntimeError("Liger kernel is not installed.") from None

    @staticmethod
    def _apply(**kwargs) -> "HFModel":
        """Applies the Liger kernel to the model.

        Args:
            **kwargs: Must include ``model``. Optional ``use_kernels`` is a list of Liger op
                names to enable exclusively, or the string ``"auto"`` to use each
                ``apply_liger_kernel_to_*`` function's signature defaults (same as calling
                upstream with only ``model``). Optional ``require_logits`` forces non-fused
                cross entropy when supported.

        Returns:
            HFModel: The model with Liger kernel applied.

        Raises:
            ValueError: If the model is not provided.
            RuntimeError: If dependencies are not met.
        """
        model = kwargs.get("model")
        config = kwargs.get("config")
        use_kernels = kwargs.get("use_kernels", "auto")

        require_logits = kwargs.get("require_logits", False)
        if config is not None:
            require_logits = config.get("require_logits", require_logits)

        model_type = getattr(model.config, "model_type", None)

        if model_type not in _LIGER_FN_BY_MODEL_TYPE:
            logger.warning_rank0("Current model does not support liger kernel.")
            return model

        import liger_kernel.transformers as liger_transformers

        apply_liger_kernel = getattr(liger_transformers, _LIGER_FN_BY_MODEL_TYPE[model_type])

        sig = inspect.signature(apply_liger_kernel).parameters
        togglable = [name for name in sig if name != "model"]

        def _normalize_op_name(raw: str) -> str:
            key = raw.strip().lower().replace("-", "_")
            aliases = {
                "rmsnorm": "rms_norm",
                "flce": "fused_linear_cross_entropy",
                "lce": "fused_linear_cross_entropy",
                "fused_ce": "fused_linear_cross_entropy",
            }
            return aliases.get(key, key)

        if use_kernels is not None and len(use_kernels) == 0:
            return model

        if use_kernels != "auto":
            selected = {_normalize_op_name(k) for k in use_kernels}
            ops = selected - set(togglable)
            if ops:
                raise ValueError(
                    f"Unknown Liger op(s) {sorted(ops)} for model_type={model_type}. Valid: {sorted(togglable)}"
                )
            if "cross_entropy" in selected and "fused_linear_cross_entropy" in selected:
                raise ValueError("cross_entropy and fused_linear_cross_entropy cannot both be enabled.")
            call_kwargs = {name: (name in selected) for name in togglable}
            call_kwargs["model"] = model
        else:
            # Mirror ``liger_kernel`` signature defaults so patches match upstream defaults
            # and logging reflects enabled ops (omitted kwargs only live in the callee).
            call_kwargs = {"model": model}
            for name in togglable:
                param = sig[name]
                if param.default is not inspect.Parameter.empty:
                    call_kwargs[name] = param.default

        if require_logits and "fused_linear_cross_entropy" in sig:
            logger.warning_rank0("Current training stage does not support chunked cross entropy.")
            call_kwargs["fused_linear_cross_entropy"] = False
            call_kwargs["cross_entropy"] = True

        apply_liger_kernel(**call_kwargs)

        applied = sorted(name for name, on in call_kwargs.items() if name != "model" and on)
        logger.info_rank0(f"These Liger ops are applied to the model: {applied}")

        return model
