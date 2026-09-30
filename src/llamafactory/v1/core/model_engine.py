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

"""The definition of model engine.

How to use:
model_engine = ModelEngine(model_args, is_train=True)
model_engine.processor: Get the tokenizer or multi-modal processor.
model_engine.renderer: Get the renderer.
model_engine.model_config: Get the model configuration.
model_engine.model: Get the HF model.

Init workflow:
1. Init processor.
2. Init render.
2. Init model config.
3. Init model.
4. Init adapter.
"""

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoProcessor

from ..accelerator.helper import DeviceType
from ..accelerator.interface import DistributedInterface
from ..config.model_args import ModelArguments, ModelClass
from ..utils import logging
from ..utils.helper import get_tokenizer, is_tokenizer
from ..utils.types import HFConfig, HFModel, Processor
from .rendering import Renderer


logger = logging.get_logger(__name__)


class ModelEngine:
    """Model engine.

    Args:
        model_args: Model arguments.
        is_train: Whether to train the model.
    """

    def __init__(
        self,
        model_args: ModelArguments,
        is_train: bool = False,
    ) -> None:
        self.args = model_args
        """Model arguments."""
        self.is_train = is_train
        """Whether to train the model."""
        self.processor = self._init_processor()
        """Tokenizer or multi-modal processor."""
        self._sync_chat_template()
        self.model_config = self._init_model_config()
        """Model configuration."""
        self.renderer = Renderer(self.processor)
        """Renderer."""
        self._deepspeed_zero3_enabled = False

        try:
            from ..plugins.model_plugins.deepspeed_utils import (
                is_deepspeed_zero3_enabled,
                setup_deepspeed_zero3_model_loading,
                teardown_deepspeed_zero3_model_loading,
            )

            self._deepspeed_zero3_enabled = self.is_train and is_deepspeed_zero3_enabled()
        except ImportError:
            pass

        if self._deepspeed_zero3_enabled:
            plugin = setup_deepspeed_zero3_model_loading()
            try:
                self.model = self._init_model()
            finally:
                teardown_deepspeed_zero3_model_loading(plugin)
        else:
            self.model = self._init_model()

    def _init_processor(self) -> Processor:
        """Init processor.

        NOTE: Transformers v5 always use fast tokenizer.
        https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/auto/tokenization_auto.py#L642
        """
        return AutoProcessor.from_pretrained(
            self.args.model,
            trust_remote_code=self.args.trust_remote_code,
        )

    def _sync_chat_template(self) -> None:
        """Sync chat_template and inject custom_chat_template."""
        tokenizer = get_tokenizer(self.processor)
        if not is_tokenizer(self.processor) and not getattr(self.processor, "chat_template", None):
            if getattr(tokenizer, "chat_template", None):
                self.processor.chat_template = tokenizer.chat_template

        if self.args.custom_chat_template:
            if not is_tokenizer(self.processor):
                self.processor.chat_template = self.args.custom_chat_template
            tokenizer.chat_template = self.args.custom_chat_template

    def _init_model_config(self) -> HFConfig:
        """Init model config."""
        return AutoConfig.from_pretrained(
            self.args.model,
            trust_remote_code=self.args.trust_remote_code,
        )

    def _init_model(self) -> HFModel:
        """Init model.

        Transformers can choose the proper model init context.
        https://github.com/huggingface/transformers/blob/v5.0.0rc0/src/transformers/modeling_utils.py#L3538
        """
        if self.args.init_config is not None:
            from ..plugins.model_plugins.initialization import InitPlugin

            init_device = InitPlugin(self.args.init_config.name)()
        else:
            init_device = DistributedInterface().current_device

        init_kwargs = {} if self._deepspeed_zero3_enabled else {"device_map": init_device}
        logger.info_rank0(f"Using attention implementation: {self.args.flash_attn}.")

        if self.args.quant_config is not None:
            from ..plugins.model_plugins.quantization import QuantizationPlugin

            init_kwargs = QuantizationPlugin(self.args.quant_config.name)(
                init_kwargs=init_kwargs,
                quant_config=self.args.quant_config,
                is_trainable=self.is_train,
            )

        if self.args.model_class == ModelClass.LLM:
            from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

            # AutoModelForMultimodalLM (audio / other multimodal LMs, e.g. Qwen2-Audio) was added in
            # a newer transformers; fall back gracefully when it is absent (e.g. 4.57.1).
            try:
                from transformers import AutoModelForMultimodalLM
            except ImportError:
                AutoModelForMultimodalLM = None

            cfg_type = type(self.model_config)
            if cfg_type in AutoModelForImageTextToText._model_mapping.keys():
                AutoClass = AutoModelForImageTextToText
            elif AutoModelForMultimodalLM is not None and cfg_type in AutoModelForMultimodalLM._model_mapping.keys():
                # Audio / other multimodal LMs (e.g. Qwen2-Audio) live here, not in CausalLM.
                AutoClass = AutoModelForMultimodalLM
            else:
                AutoClass = AutoModelForCausalLM

        elif self.args.model_class == ModelClass.CLS:
            from transformers import AutoModelForTokenClassification

            self.model_config.num_labels = 1
            self.model_config.classifier_dropout = 0.0
            text_config = getattr(self.model_config, "text_config", None)
            if text_config is not None:
                text_config.num_labels = 1
                text_config.classifier_dropout = 0.0
            AutoClass = AutoModelForTokenClassification
        else:
            from transformers import AutoModel

            AutoClass = AutoModel

        if init_device.type == DeviceType.META:
            assert self.args.quant_config is None, "Quantization is not supported with meta device."
            with init_empty_weights():
                model = AutoClass.from_config(self.model_config, attn_implementation=self.args.flash_attn)
        else:
            model = AutoClass.from_pretrained(
                self.args.model,
                config=self.model_config,
                dtype="auto",
                attn_implementation=self.args.flash_attn,
                trust_remote_code=self.args.trust_remote_code,
                **init_kwargs,
            )

        init_mode = self.args.init_config.name if self.args.init_config is not None else "init_on_default"
        model._init_mode = init_mode

        if hasattr(model, "thinker"):
            model = model.thinker
            model._init_mode = init_mode

        if self.args.peft_config is None:
            if self.is_train:
                logger.info_rank0("Fine-tuning mode: full tuning")
                model = model.to(torch.float32)
            else:
                logger.info_rank0("Inference the original model")
        else:
            if self.args.peft_config.name == "lora" and init_mode == "init_on_meta":
                raise ValueError("Currently lora stage does not support loading model by meta.")

            from ..plugins.model_plugins.peft import PeftPlugin

            model = PeftPlugin(self.args.peft_config.name)(
                model,
                peft_config=self.args.peft_config,
                is_train=self.is_train,
            )

        if self.args.kernel_config is not None:
            from ..plugins.model_plugins.kernels.interface import apply_kernels

            model = apply_kernels(model, self.args.kernel_config, require_logits=self.is_train)

        return model


if __name__ == "__main__":
    """
    python -m llamafactory.v1.core.model_engine --model llamafactory/tiny-random-qwen2.5
    """
    from ..config.arg_parser import get_args

    model_args, *_ = get_args()
    model_engine = ModelEngine(model_args=model_args)
    print(model_engine.processor)
    print(model_engine.model_config)
    print(model_engine.model)
