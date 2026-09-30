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

"""DeepSpeed integration via accelerate's built-in capabilities.

Instead of manually calling deepspeed.initialize() and syncing config,
this module leverages accelerate's Accelerator + DeepSpeedPlugin to handle
initialization, backward, gradient accumulation, and model saving.
"""

import os
from typing import Any, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin

from ....utils.logging import get_logger
from ....utils.types import HFModel, Processor
from ...model_plugins.deepspeed_utils import infer_deepspeed_mixed_precision


logger = get_logger(__name__)


# ZeRO-3 bucket sizes that accelerate derives from the model's hidden size
_ZERO3_BUCKET_FORMULAS = {
    "reduce_bucket_size": lambda hidden: hidden * hidden,
    "stage3_prefetch_bucket_size": lambda hidden: int(0.9 * hidden * hidden),
    "stage3_param_persistence_threshold": lambda hidden: 10 * hidden,
}


class DeepSpeedEngine:
    """DeepSpeed integration using accelerate's built-in capabilities.

    This replaces the manual DeepSpeedConfigHelper / DeepSpeedEngine approach
    with accelerate's Accelerator + DeepSpeedPlugin, which handles:
    - Config syncing (auto values, batch size, lr, etc.)
    - deepspeed.initialize() call
    - Optimizer / LR scheduler wrapping
    - Backward + gradient accumulation boundary
    - ZeRO-3 parameter gathering for saving
    """

    def __init__(self, dist_config: dict[str, Any], num_micro_batch: int = 1, micro_batch_size: int = 1):
        config_file = dist_config.get("config_file")
        if not config_file:
            raise ValueError("DeepSpeed config_file is required in dist_config")

        ds_plugin = DeepSpeedPlugin(hf_ds_config=config_file)
        ds_plugin.set_mixed_precision(infer_deepspeed_mixed_precision(ds_plugin.deepspeed_config))

        self.accelerator = Accelerator(
            deepspeed_plugin=ds_plugin,
            gradient_accumulation_steps=num_micro_batch,
        )

        # Resolve "auto" for train_micro_batch_size_per_gpu so that
        # accelerate.prepare() does not require a DataLoader to infer it.
        ds_config = self.accelerator.state.deepspeed_plugin.deepspeed_config
        if ds_config.get("train_micro_batch_size_per_gpu") in (None, "auto"):
            ds_config["train_micro_batch_size_per_gpu"] = micro_batch_size

        logger.info_rank0(f"DeepSpeedEngine initialized with config: {config_file}")

    def shard_model(self, model: HFModel) -> "DeepSpeedEngine":
        """No-op shard — actual model wrapping happens in prepare().

        Returns self so the caller gets the engine instance via the hub interface.
        """
        return self

    def prepare(
        self,
        model: HFModel,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Any] = None,
    ) -> tuple[HFModel, torch.optim.Optimizer, Any]:
        """Prepare model, optimizer, and lr_scheduler using accelerate.

        Internally calls deepspeed.initialize() and wraps the returned objects.
        """
        self._fill_zero3_bucket_sizes(model)
        if lr_scheduler is not None:
            model, optimizer, lr_scheduler = self.accelerator.prepare(model, optimizer, lr_scheduler)
        else:
            model, optimizer = self.accelerator.prepare(model, optimizer)

        model._accelerator = self.accelerator  # type: ignore[assignment]

        logger.info_rank0("Model, optimizer, and lr_scheduler prepared via accelerate")
        return model, optimizer, lr_scheduler

    def _fill_zero3_bucket_sizes(self, model: HFModel) -> None:
        """Fill ZeRO-3 ``auto`` bucket sizes that accelerate cannot infer for multimodal models."""
        zero_config = self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {})
        auto_keys = [key for key in _ZERO3_BUCKET_FORMULAS if zero_config.get(key) == "auto"]
        if not auto_keys:
            return

        config = model.config
        text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
        hidden_size = getattr(text_config, "hidden_size", None)
        if hidden_size is None:
            return

        for key in auto_keys:
            zero_config[key] = _ZERO3_BUCKET_FORMULAS[key](hidden_size)
        logger.info_rank0(f"Resolved ZeRO-3 {auto_keys} from text-config hidden_size={hidden_size}.")

    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass using accelerate.

        Delegates to DeepSpeedEngineWrapper.backward() which respects
        sync_gradients to control gradient accumulation boundaries.
        When sync_gradients=True: engine.backward(loss) + engine.step()
        When sync_gradients=False: engine.backward(loss) only
        """
        self.accelerator.backward(loss)

    def get_grad_norm(self) -> float:
        """Get the global gradient norm from the DeepSpeed engine."""
        engine_wrapper = getattr(self.accelerator, "deepspeed_engine_wrapped", None)
        if engine_wrapper is not None:
            return float(engine_wrapper.engine.get_global_grad_norm() or 0.0)
        return 0.0


def save_model(model: HFModel, output_dir: str, processor: Processor) -> None:
    """Save model using accelerate's built-in ZeRO-aware utilities.

    Expects model._accelerator to be set during prepare().
    Handles ZeRO-3 parameter gathering automatically via
    accelerator.get_state_dict().
    """
    accelerator: Accelerator = model._accelerator  # type: ignore[union-attr]

    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)

    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(output_dir, state_dict=state_dict, max_shard_size="4GB")
        processor.save_pretrained(output_dir, max_shard_size="4GB")

    accelerator.wait_for_everyone()
    logger.info_rank0(f"Model saved to {output_dir}")


def save_checkpoint(model: HFModel, optimizer: torch.optim.Optimizer, ckpt_dir: str, **kwargs) -> None:
    save_ckpt_as_hf = kwargs.get("save_ckpt_as_hf", False)
    processor = kwargs.get("processor", None)

    # Always save DeepSpeed state for resume capability
    accelerator: Accelerator = model._accelerator  # type: ignore[union-attr]
    accelerator.save_state(ckpt_dir)

    # Additionally save HF format if requested
    if save_ckpt_as_hf:
        hf_dir = os.path.join(ckpt_dir, "hf_model")
        save_model(model, hf_dir, processor)


def load_checkpoint(model: HFModel, optimizer: torch.optim.Optimizer, ckpt_dir: str, **kwargs) -> None:
    accelerator: Accelerator = model._accelerator  # type: ignore[union-attr]
    accelerator.load_state(ckpt_dir)
