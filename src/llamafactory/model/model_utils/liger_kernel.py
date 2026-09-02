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

import inspect
from typing import TYPE_CHECKING

from ...extras import logging
from ...extras.misc import get_device_name


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def apply_liger_kernel(
    config: "PretrainedConfig",
    model_args: "ModelArguments",
    is_trainable: bool,
    require_logits: bool,
) -> None:
    if not is_trainable or not model_args.enable_liger_kernel:
        return

    model_type = getattr(config, "model_type", None)
    if model_type == "gemma":
        from liger_kernel.transformers import apply_liger_kernel_to_gemma as apply_liger_kernel
    elif model_type == "gemma2":
        from liger_kernel.transformers import apply_liger_kernel_to_gemma2 as apply_liger_kernel
    elif model_type == "gemma3":
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3 as apply_liger_kernel
    elif model_type == "gemma3_text":
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3_text as apply_liger_kernel
    elif model_type in ["glm", "glm4"]:  # for glm4-9b, glm4-32B respectively
        from liger_kernel.transformers import apply_liger_kernel_to_glm4 as apply_liger_kernel
    elif model_type == "glm4v":
        from liger_kernel.transformers import apply_liger_kernel_to_glm4v as apply_liger_kernel
    elif model_type == "granite":
        from liger_kernel.transformers import apply_liger_kernel_to_granite as apply_liger_kernel
    elif model_type == "llama":
        from liger_kernel.transformers import apply_liger_kernel_to_llama as apply_liger_kernel
    elif model_type == "llava":
        from liger_kernel.transformers import apply_liger_kernel_to_llava as apply_liger_kernel
    elif model_type == "mistral":
        from liger_kernel.transformers import apply_liger_kernel_to_mistral as apply_liger_kernel
    elif model_type == "mixtral":
        from liger_kernel.transformers import apply_liger_kernel_to_mixtral as apply_liger_kernel
    elif model_type == "mllama":
        from liger_kernel.transformers import apply_liger_kernel_to_mllama as apply_liger_kernel
    elif model_type == "olmo2":
        from liger_kernel.transformers import apply_liger_kernel_to_olmo2 as apply_liger_kernel
    elif model_type == "paligemma":
        from liger_kernel.transformers import apply_liger_kernel_to_paligemma as apply_liger_kernel
    elif model_type == "phi3":
        from liger_kernel.transformers import apply_liger_kernel_to_phi3 as apply_liger_kernel
    elif model_type == "qwen2":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2 as apply_liger_kernel
    elif model_type == "qwen2_vl":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl as apply_liger_kernel
    elif model_type == "qwen2_5_vl":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl as apply_liger_kernel
    elif model_type == "qwen3":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3 as apply_liger_kernel
    elif model_type == "qwen3_moe":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe as apply_liger_kernel
    elif model_type == "qwen3_next":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_next as apply_liger_kernel
    elif model_type == "qwen3_5":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5 as apply_liger_kernel
    elif model_type == "qwen3_5_text":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5_text as apply_liger_kernel
    elif model_type == "qwen3_5_moe":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5_moe as apply_liger_kernel
    elif model_type == "qwen3_5_moe_text":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5_moe_text as apply_liger_kernel
    elif model_type == "gpt_oss":
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_gpt_oss as apply_liger_kernel
        except ImportError:
            logger.warning_rank0("Please install liger-kernel from https://github.com/Comet0322/Liger-Kernel.")
            return
    else:
        logger.warning_rank0("Current model does not support liger kernel.")
        return

    if require_logits and "fused_linear_cross_entropy" in inspect.signature(apply_liger_kernel).parameters:
        logger.info_rank0("Current training stage does not support chunked cross entropy.")
        kwargs = {"fused_linear_cross_entropy": False, "cross_entropy": True}
    else:
        kwargs = {}

    if get_device_name() == "npu":
        import torch

        if "Ascend910" not in torch.npu.get_device_name(0):
            kwargs["swiglu"] = False
            kwargs["fused_linear_cross_entropy"] = False

    apply_liger_kernel(**kwargs)
    logger.info_rank0("Liger kernel has been applied to the model.")
