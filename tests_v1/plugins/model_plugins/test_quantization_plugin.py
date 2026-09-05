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

import pytest

from llamafactory.v1.config.model_args import ModelArguments
from llamafactory.v1.core.model_engine import ModelEngine


bitsandbytes = pytest.importorskip("bitsandbytes")


def check_quantization_status(model):
    quantized_info = {"bnb": []}

    for name, module in model.named_modules():
        # check BitsAndBytes quantization
        if isinstance(module, bitsandbytes.nn.modules.Linear8bitLt) or isinstance(
            module, bitsandbytes.nn.modules.Linear4bit
        ):
            quantized_info["bnb"].append(name)

    return quantized_info


@pytest.mark.runs_on(["cuda", "xpu"])
@pytest.mark.parametrize("name, quantization_bit", [("bnb", 4), ("auto", 4)])
def test_quantization_plugin(name, quantization_bit):
    model_args = ModelArguments(
        model="llamafactory/tiny-random-qwen3",
        quant_config={
            "name": name,
            "quantization_bit": quantization_bit,
        },
    )

    model_engine = ModelEngine(model_args=model_args)
    quantized_info = check_quantization_status(model_engine.model)
    print(f"Quantized weights for method {name} with {quantization_bit} bit: {quantized_info}")
    assert any(v for v in quantized_info.values()), "model is not quantized properly."
