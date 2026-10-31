# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HyperParallel-specific meta model construction and sharded weight loading."""

import os
from typing import TYPE_CHECKING

import torch
from hyper_parallel import init_empty_weights
from hyper_parallel.integration.llamafactory import parallelize_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForTextToWaveform,
)
from transformers.modeling_utils import ContextManagers

from ...extras import logging
from ...extras.misc import count_parameters
from ...extras.packages import is_torch_version_greater_than
from ...model.adapter import init_adapter
from ...model.loader import _get_init_kwargs, load_config
from ...model.model_utils.liger_kernel import apply_liger_kernel
from ...model.model_utils.misc import register_autoclass
from ...model.patcher import patch_config, patch_model


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


def _get_no_init_weights():
    """Resolve the Transformers no-init context across supported versions."""
    try:
        from transformers.modeling_utils import no_init_weights
    except ImportError:
        from transformers.initialization import no_init_weights

    return no_init_weights


def _get_model_class(config):
    """Select the same Transformers AutoModel class as the native loader."""
    if type(config) in AutoModelForImageTextToText._model_mapping.keys():
        return AutoModelForImageTextToText
    if type(config) in AutoModelForSeq2SeqLM._model_mapping.keys():
        return AutoModelForSeq2SeqLM
    if type(config) in AutoModelForTextToWaveform._model_mapping.keys():
        return AutoModelForTextToWaveform
    return AutoModelForCausalLM


def _validate_conv3d_compatibility(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    """Retain the native loader's torch 2.9 Conv3D compatibility guard."""
    if not is_torch_version_greater_than("2.9.0") or is_torch_version_greater_than("2.10.0"):
        return

    conv3d_modules = [module for module in model.modules() if isinstance(module, torch.nn.Conv3d)]
    kt_conv3d_ready = (
        model_args.use_kt
        and bool(conv3d_modules)
        and all(getattr(module, "_kt_conv3d_compatible", False) for module in conv3d_modules)
    )
    if conv3d_modules and not kt_conv3d_ready:
        raise ValueError(
            "Unsupported torch version detected: torch 2.9.x with Conv3D. "
            "This combination is known to cause severe performance regression. "
            "Please downgrade torch to <2.9 or remove Conv3D. "
            "See https://github.com/pytorch/pytorch/issues/166122"
        )
    if kt_conv3d_ready:
        logger.info_rank0("Using KTransformers instance-scoped Conv3D fallback for torch 2.9.x VLM training.")


def load_hyper_parallel_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    distributed_setup,
    hp_args,
    is_trainable: bool = False,
) -> "PreTrainedModel":
    """Build on meta, apply HyperParallel layouts, then load local weight shards."""
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    apply_liger_kernel(config, model_args, is_trainable, require_logits=False)

    load_class = _get_model_class(config)
    no_init_weights = _get_no_init_weights()
    with ContextManagers([no_init_weights(), init_empty_weights()]):
        model = load_class.from_config(
            config,
            trust_remote_code=model_args.trust_remote_code,
            dtype=model_args.compute_dtype,
        )

    if getattr(model.config, "model_type", None) in ["qwen2_5_omni", "qwen3_omni_moe"]:
        model = getattr(model, "thinker")

    patch_model(model, tokenizer, model_args, is_trainable, add_valuehead=False)
    register_autoclass(config, model, tokenizer)
    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if model_args.use_v1_kernels and is_trainable:
        logger.warning_rank0(
            "You are try to using future feature about kernels, please note that this feature "
            "is not supported for all models. If get any error, please disable this feature, or report the issue."
        )
        from ...v1.plugins.model_plugins.kernels.interface import apply_v1_kernels

        model = apply_v1_kernels(model, use_v1_kernels=model_args.use_v1_kernels)

    activation_mode = hp_args.activation_mode
    model = parallelize_model(
        model,
        distributed_setup,
        pretrained_path=None if model_args.train_from_scratch else model_args.model_name_or_path,
        activation_checkpoint="full" if activation_mode in {"recompute", "swap"} else None,
        swap_inputs=activation_mode == "swap" and hp_args.activation_swap_inputs,
    )

    _validate_conv3d_compatibility(model, model_args)
    if not is_trainable:
        model.requires_grad_(False)
        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = (
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}"
        )
    else:
        param_stats = f"all params: {all_param:,}"
    logger.info_rank0(param_stats)

    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        for name, param in model.named_parameters():
            print(f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}")

    return model
