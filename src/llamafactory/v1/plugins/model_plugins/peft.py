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

import re
from dataclasses import dataclass, field
from typing import Literal

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from ...config import InputArgument, get_args
from ...core.model_engine import ModelEngine
from ...utils import logging
from ...utils.plugin import BasePlugin
from ...utils.types import HFModel


logger = logging.get_logger(__name__)


@dataclass
class LoraParams:
    """Typed configuration for the LoRA PEFT plugin."""

    name: Literal["lora"] = "lora"
    """Plugin name."""
    r: int = 8
    """LoRA rank."""
    lora_alpha: int = 16
    """LoRA alpha."""
    lora_dropout: float = 0.05
    """LoRA dropout."""
    target_modules: list[str] | str = "all"
    """Target modules."""
    use_rslora: bool = False
    """Use RS-LoRA."""
    use_dora: bool = False
    """Use DoRA."""
    modules_to_save: list[str] | None = None
    """Modules to save."""
    adapter_name_or_path: list[str] | str | None = None
    """Path to the adapter(s)."""
    export_dir: str | None = None
    """Path to the export directory."""
    export_size: int = 5
    """Shard size for the exported model, in GB."""
    export_hub_model_id: str | None = None
    """Hub model ID for the exported model."""
    infer_dtype: Literal["auto", "float16", "float32", "bfloat16"] = "auto"
    """Inference data type for the exported model."""
    export_legacy_format: bool = False
    """Use legacy format for the exported model."""


@dataclass
class FreezeParams:
    """Typed configuration for the freeze PEFT plugin."""

    name: Literal["freeze"] = "freeze"
    """Plugin name."""
    freeze_trainable_layers: int = 2
    """Number of trainable layers."""
    freeze_trainable_modules: list[str] | str = "all"
    """Trainable modules in the selected layers."""
    freeze_extra_modules: list[str] | str | None = field(default_factory=list)
    """Extra non-hidden modules to train."""
    cast_trainable_params_to_fp32: bool = True
    """Cast trainable parameters to float32."""


class PeftPlugin(BasePlugin):
    def __call__(self, model: HFModel, peft_config: dict, is_train: bool) -> HFModel:
        return super().__call__(model, peft_config, is_train)


def _find_all_linear_modules(model: HFModel) -> list[str]:
    r"""Find all available modules to apply LoRA."""
    forbidden_modules = {"lm_head", "output_layer", "output", "score", "classifier"}
    module_names = set()
    for name, module in model.named_modules():
        if any(forbidden_module in name for forbidden_module in forbidden_modules):
            continue

        if "Linear" in module.__class__.__name__ and "Embedding" not in module.__class__.__name__:
            module_names.add(name.split(".")[-1])

    return list(module_names)


def merge_adapters(model: HFModel, adapter_name_or_path: list[str] | str) -> HFModel:
    if not isinstance(adapter_name_or_path, list):
        adapter_name_or_path = [adapter_name_or_path]

    for adapter_path in adapter_name_or_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        logger.info_rank0(f"Merged adapter from {adapter_path}")

    return model


def load_adapter(model: HFModel, adapter_name_or_path: list[str] | str, is_train: bool) -> HFModel:
    r"""Loads adapter(s) into the model.

    Determine adapter usage based on mode:
    - Training: Load the single adapter for continued training.
    - Inference: Merge all adapters to clean up the model.
    - Unmergeable: Keep the single adapter active without merging.
    """
    if not isinstance(adapter_name_or_path, list):
        adapter_name_or_path = [adapter_name_or_path]

    # TODO
    # Adapters fix for deepspeed and quant
    # Adapters fix for vision

    if is_train and len(adapter_name_or_path) > 1:
        raise ValueError(
            "When `adapter_name_or_path` is provided for training, only a single LoRA adapter is supported. "
            "Training will continue on the specified adapter. "
            "Please merge multiple adapters before starting a new LoRA adapter."
        )

    if is_train:
        adapter_to_merge = []
        adapter_to_resume = adapter_name_or_path[0]
    else:
        adapter_to_merge = adapter_name_or_path
        adapter_to_resume = None

    if adapter_to_merge:
        model = merge_adapters(model, adapter_to_merge)

    if adapter_to_resume is not None:
        model = PeftModel.from_pretrained(model, adapter_to_resume, is_trainable=is_train)
        if is_train:
            logger.info_rank0(
                f"Resuming training from existing LoRA adapter at {adapter_to_resume}. "
                "LoRA hyperparameters will be loaded from the adapter itself; "
                "the current LoRA configuration will be ignored. "
                "Merge the adapter into the base model before training if you want to start a new adapter."
            )

    return model


@PeftPlugin("lora").register()
def get_lora_model(model: HFModel, peft_config: dict | LoraParams, is_train: bool = False) -> HFModel:
    peft_config = PeftPlugin.parse_params(peft_config, LoraParams)
    adapter_name_or_path = peft_config.adapter_name_or_path

    if adapter_name_or_path:
        return load_adapter(model, adapter_name_or_path, is_train)

    logger.info_rank0("Fine-tuning method: LoRA")

    target_modules = peft_config.target_modules

    # Handle target modules
    if target_modules == "all":
        target_modules = _find_all_linear_modules(model)
    elif isinstance(target_modules, str):
        target_modules = [target_modules]

    logger.info_rank0(f"LoRA target modules: {target_modules}")

    cls_name = model.__class__.__name__
    if cls_name.endswith("ForTokenClassification"):
        task_type = TaskType.TOKEN_CLS
    elif cls_name.endswith("ForSequenceClassification"):
        task_type = TaskType.SEQ_CLS
    else:
        task_type = TaskType.CAUSAL_LM

    lora_config = LoraConfig(
        task_type=task_type,
        inference_mode=not is_train,
        r=peft_config.r,
        lora_alpha=peft_config.lora_alpha,
        lora_dropout=peft_config.lora_dropout,
        use_rslora=peft_config.use_rslora,
        use_dora=peft_config.use_dora,
        target_modules=target_modules,
        modules_to_save=peft_config.modules_to_save,
    )

    model = get_peft_model(model, lora_config)

    if is_train:
        model.print_trainable_parameters()

    return model


@PeftPlugin("freeze").register()
def get_freeze_model(model: HFModel, peft_config: dict | FreezeParams, is_train: bool = False) -> HFModel:
    peft_config = PeftPlugin.parse_params(peft_config, FreezeParams)
    logger.info_rank0("Fine-tuning method: Freeze")

    if not is_train:
        return model

    freeze_trainable_layers = peft_config.freeze_trainable_layers
    freeze_trainable_modules = peft_config.freeze_trainable_modules
    freeze_extra_modules = peft_config.freeze_extra_modules
    cast_trainable_params_to_fp32 = peft_config.cast_trainable_params_to_fp32

    if isinstance(freeze_trainable_modules, str):
        freeze_trainable_modules = [module.strip() for module in freeze_trainable_modules.split(",")]

    if isinstance(freeze_extra_modules, str):
        freeze_extra_modules = [module.strip() for module in freeze_extra_modules.split(",")]

    # Get number of layers
    num_layers = (
        getattr(model.config, "num_hidden_layers", None)
        or getattr(model.config, "num_layers", None)
        or getattr(model.config, "n_layer", None)
    )

    if not num_layers:
        raise ValueError("Current model does not support freeze tuning.")

    if freeze_trainable_layers > 0:
        # last n layers
        trainable_layer_ids = range(max(0, num_layers - freeze_trainable_layers), num_layers)
    else:
        # first n layers
        trainable_layer_ids = range(min(-freeze_trainable_layers, num_layers))

    # Identify hidden and non-hidden modules
    hidden_modules = set()
    non_hidden_modules = set()
    for name, _ in model.named_parameters():
        if ".0." in name:
            hidden_modules.add(name.split(".0.")[-1].split(".")[0])
        elif ".1." in name:
            hidden_modules.add(name.split(".1.")[-1].split(".")[0])

        if re.search(r"\.\d+\.", name) is None:
            non_hidden_modules.add(name.split(".")[-2])

    # Build list of trainable layer patterns
    trainable_layers = []
    for module_name in freeze_trainable_modules:
        if module_name == "all":
            for idx in trainable_layer_ids:
                trainable_layers.append(f".{idx:d}.")
        elif module_name in hidden_modules:
            for idx in trainable_layer_ids:
                trainable_layers.append(f".{idx:d}.{module_name}")
        else:
            raise ValueError(f"Module {module_name} not found in hidden modules: {hidden_modules}")

    # Add extra modules
    if freeze_extra_modules:
        for module_name in freeze_extra_modules:
            if module_name in non_hidden_modules:
                trainable_layers.append(module_name)
            else:
                raise ValueError(f"Module {module_name} not found in non-hidden modules: {non_hidden_modules}")

    # TODO
    # Multi-modal special handling

    # Set requires_grad
    forbidden_modules = {"quant_state", "quantization_weight", "qweight", "qzeros", "scales"}
    for name, param in model.named_parameters():
        if any(trainable_layer in name for trainable_layer in trainable_layers) and not any(
            forbidden_module in name for forbidden_module in forbidden_modules
        ):
            param.requires_grad_(True)
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)  # Cast to fp32 for stability
        else:
            param.requires_grad_(False)

    logger.info_rank0(f"Set trainable layers: {trainable_layers}")

    # Count trainable params for verification
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info_rank0(
        f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.4f}"
    )

    return model


def merge_and_export_model(args: InputArgument = None):
    model_args, _, _, _ = get_args(args)

    raw_config = model_args.peft_config
    if raw_config is None:
        raise ValueError("Please specify peft_config to merge and export model.")
    if raw_config.name != "lora":
        raise ValueError("Currently merge and export model function is only supported for lora.")

    export_peft_config = PeftPlugin.parse_params(raw_config, LoraParams)
    if export_peft_config.export_dir is None:
        raise ValueError("Please specify export_dir.")
    if export_peft_config.adapter_name_or_path is None:
        raise ValueError("Please set adapter_name_or_path to merge adapters into base model.")

    logger.info_rank0("Loading model for export...")
    model_engine = ModelEngine(model_args, is_train=False)
    model = model_engine.model
    tokenizer = model_engine.processor

    if export_peft_config.infer_dtype == "auto":
        if model.config.torch_dtype == torch.float32 and torch.cuda.is_bf16_supported():
            model = model.to(torch.bfloat16)
            logger.info_rank0("Converted model to bfloat16.")
    else:
        target_dtype = getattr(torch, export_peft_config.infer_dtype)
        model = model.to(target_dtype)
        logger.info_rank0(f"Converted model to {export_peft_config.infer_dtype}.")

    logger.info_rank0(f"Exporting model to {export_peft_config.export_dir}...")
    model.save_pretrained(
        export_peft_config.export_dir,
        max_shard_size=f"{export_peft_config.export_size}GB",
        safe_serialization=not export_peft_config.export_legacy_format,
    )
    if tokenizer is not None:
        try:
            if hasattr(tokenizer, "padding_side"):
                tokenizer.padding_side = "left"
            tokenizer.save_pretrained(export_peft_config.export_dir)
        except Exception as e:
            logger.warning(f"Failed to save tokenizer: {e}")

    if export_peft_config.export_hub_model_id:
        logger.info_rank0(f"Pushing to hub: {export_peft_config.export_hub_model_id}...")
        model.push_to_hub(export_peft_config.export_hub_model_id)
        if tokenizer is not None:
            tokenizer.push_to_hub(export_peft_config.export_hub_model_id)

    logger.info_rank0("Model exported successfully.")
