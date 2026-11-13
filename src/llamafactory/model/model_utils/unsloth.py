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

from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.misc import get_current_device


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


def _coerce_unsloth_target_modules(
    model: "PreTrainedModel", target_modules: list[str] | set[str] | str
) -> list[str]:
    r"""Coerce full VLM module paths into leaf names for Unsloth.

    Unsloth builds LoRA regexes from leaf module names, but LlamaFactory's
    ``patch_target_modules()`` returns full dotted paths for composite VLMs.
    Convert those full paths back to leaf names while preserving the set of
    targeted layers (forbidden modules were already filtered out upstream).
    Only language-model paths are kept; vision-tower paths are dropped.
    """
    from .visual import COMPOSITE_MODELS

    if isinstance(target_modules, str):
        target_modules = [target_modules]
    elif isinstance(target_modules, set):
        target_modules = list(target_modules)

    model_type = getattr(model.config, "model_type", None)
    if model_type not in COMPOSITE_MODELS or not target_modules:
        return target_modules

    if all("." not in name for name in target_modules):
        return target_modules

    language_model_keys = COMPOSITE_MODELS[model_type].language_model_keys
    target_modules = [
        name
        for name in target_modules
        if "." not in name
        or any(
            name == key or name.startswith(f"{key}.") or f".{key}." in name or name.endswith(f".{key}")
            for key in language_model_keys
        )
    ]
    leaf_names = sorted({name.rsplit(".", 1)[-1] for name in target_modules})
    logger.info_rank0(
        f"Unsloth expects leaf LoRA targets; converting {model_type} composite target_modules "
        f"to leaf names: {leaf_names}"
    )
    return leaf_names


def _get_unsloth_kwargs(
    config: "PretrainedConfig",
    model_name_or_path: str,
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
) -> dict[str, Any]:
    return {
        "model_name": model_name_or_path,
        "max_seq_length": model_args.model_max_length or 4096,
        "dtype": model_args.compute_dtype,
        "load_in_4bit": model_args.quantization_bit == 4,
        "token": model_args.hf_hub_token,
        "full_finetuning": finetuning_args.finetuning_type == "full",
        "device_map": {"": get_current_device()},
        "rope_scaling": getattr(config, "rope_scaling", None),
        "fix_tokenizer": False,
        "trust_remote_code": model_args.trust_remote_code,
        "use_gradient_checkpointing": "unsloth",
    }


def load_unsloth_pretrained_model(
    config: "PretrainedConfig", model_args: "ModelArguments", finetuning_args: "FinetuningArguments"
) -> Optional["PreTrainedModel"]:
    r"""Optionally load pretrained model with unsloth. Used in training."""
    from unsloth import FastLanguageModel  # type: ignore

    unsloth_kwargs = _get_unsloth_kwargs(config, model_args.model_name_or_path, model_args, finetuning_args)
    try:
        model, _ = FastLanguageModel.from_pretrained(**unsloth_kwargs)
    except NotImplementedError:
        logger.warning_rank0("Unsloth does not support model type {}.".format(getattr(config, "model_type", None)))
        model = None
        model_args.use_unsloth = False

    return model


def get_unsloth_peft_model(
    model: "PreTrainedModel", model_args: "ModelArguments", peft_kwargs: dict[str, Any]
) -> "PreTrainedModel":
    r"""Get the peft model for the pretrained model with unsloth. Used in training."""
    from unsloth import FastLanguageModel  # type: ignore

    if "target_modules" in peft_kwargs:
        peft_kwargs["target_modules"] = _coerce_unsloth_target_modules(model, peft_kwargs["target_modules"])

    unsloth_peft_kwargs = {
        "model": model,
        "max_seq_length": model_args.model_max_length,
        "use_gradient_checkpointing": "unsloth",
    }
    return FastLanguageModel.get_peft_model(**peft_kwargs, **unsloth_peft_kwargs)


def load_unsloth_peft_model(
    config: "PretrainedConfig",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
) -> Optional["PreTrainedModel"]:
    r"""Load peft model with unsloth. Used in both training and inference.

    Returns None if unsloth does not support the model type, and sets
    model_args.use_unsloth = False so callers can fall back to standard loading.
    """
    from unsloth import FastLanguageModel  # type: ignore

    unsloth_kwargs = _get_unsloth_kwargs(config, model_args.adapter_name_or_path[0], model_args, finetuning_args)
    try:
        if not is_trainable:
            unsloth_kwargs["use_gradient_checkpointing"] = False

        model, _ = FastLanguageModel.from_pretrained(**unsloth_kwargs)
    except NotImplementedError:
        logger.warning_rank0("Unsloth does not support model type {}.".format(getattr(config, "model_type", None)))
        model_args.use_unsloth = False
        return None

    if not is_trainable:
        FastLanguageModel.for_inference(model)

    return model
