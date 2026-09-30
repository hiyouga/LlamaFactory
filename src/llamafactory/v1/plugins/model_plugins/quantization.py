# Copyright 2025 HuggingFace Inc., the KVCache.AI team, Approaching AI, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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

from dataclasses import dataclass
from typing import Any, Literal

from ...utils import logging
from ...utils.plugin import BasePlugin


logger = logging.get_logger(__name__)


class QuantizationPlugin(BasePlugin):
    def __call__(
        self,
        init_kwargs: dict[str, Any] | None = None,
        quant_config=None,
        is_trainable: bool = False,
    ) -> dict[str, Any]:
        return super().__call__(init_kwargs, quant_config=quant_config, is_trainable=is_trainable)


@dataclass
class BnbParams:
    name: Literal["bnb", "auto"] = "bnb"
    quantization_bit: int | None = None
    compute_dtype: str | Any = "float16"
    double_quantization: bool = True
    quantization_type: str = "nf4"

    def __post_init__(self) -> None:
        import torch

        if isinstance(self.compute_dtype, str):
            dtype = getattr(torch, self.compute_dtype, None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f"compute_dtype={self.compute_dtype!r} is not a torch dtype name.")
            self.compute_dtype = dtype
        elif not isinstance(self.compute_dtype, torch.dtype):
            raise TypeError(f"compute_dtype must be str or torch.dtype, got {type(self.compute_dtype).__name__}.")


@QuantizationPlugin("auto").register()
def quantization_auto(
    init_kwargs: dict[str, Any],
    quant_config: dict | BnbParams,
    is_trainable: bool = False,
) -> dict[str, Any]:
    quant_config = QuantizationPlugin.parse_params(quant_config, BnbParams)
    if quant_config.quantization_bit is None:
        logger.warning_rank0("No quantization method applied.")
        return init_kwargs
    if quant_config.quantization_bit not in (4, 8):
        raise ValueError(f"Unsupported quantization bit: {quant_config.quantization_bit} for auto quantization.")

    logger.info_rank0(f"Loading {quant_config.quantization_bit}-bit quantized model.")
    return QuantizationPlugin("bnb")(init_kwargs, quant_config=quant_config, is_trainable=is_trainable)


@QuantizationPlugin("bnb").register()
def quantization_with_bnb(
    init_kwargs: dict[str, Any],
    quant_config: dict | BnbParams,
    is_trainable: bool = False,
) -> dict[str, Any]:
    from transformers import BitsAndBytesConfig

    from ...accelerator.helper import get_current_device
    from ...utils.packages import check_version

    quant_config = QuantizationPlugin.parse_params(quant_config, BnbParams)
    quantization_bit = quant_config.quantization_bit
    if quantization_bit is None:
        logger.warning_rank0("quantization_bit is not specified, default to 4-bit quantization.")
        quantization_bit = 4
    if quantization_bit not in (4, 8):
        raise ValueError("Bitsandbytes only accepts 4-bit or 8-bit quantization.")

    logger.info_rank0("Using Bitsandbytes quantization.")
    if quantization_bit == 8:
        check_version("bitsandbytes>=0.37.0", mandatory=True)
        init_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        check_version("bitsandbytes>=0.39.0", mandatory=True)
        init_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=quant_config.compute_dtype,
            bnb_4bit_use_double_quant=quant_config.double_quantization,
            bnb_4bit_quant_type=quant_config.quantization_type,
            bnb_4bit_quant_storage=quant_config.compute_dtype,
        )

    if is_trainable:
        logger.info_rank0("Detected inference mode, setting device_map for bitsandbytes quantization.")
        init_kwargs["device_map"] = {"": get_current_device()}
    else:
        logger.info_rank0("Detected training mode, skip setting device_map for bitsandbytes quantization.")
        if quantization_bit != 4:
            raise ValueError("Only 4-bit quantized model can use fsdp+qlora or auto device map.")
        check_version("bitsandbytes>=0.43.0", mandatory=True)

    logger.info_rank0(f"Quantizing model to {quantization_bit} bit with bitsandbytes.")
    return init_kwargs
