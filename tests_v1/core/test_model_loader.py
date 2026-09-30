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

import torch

from llamafactory.v1.config.model_args import ModelArguments
from llamafactory.v1.core.model_engine import ModelEngine


def test_tiny_qwen():
    model_args = ModelArguments(model="llamafactory/tiny-random-qwen3")
    model_engine = ModelEngine(model_args)
    assert "Qwen2Tokenizer" in model_engine.processor.__class__.__name__
    assert "Qwen3Config" in model_engine.model_config.__class__.__name__
    assert "Qwen3ForCausalLM" in model_engine.model.__class__.__name__
    assert model_engine.model.dtype == torch.bfloat16


def test_tiny_qwen_with_kernel_plugin():
    from llamafactory.v1.plugins.model_plugins.kernels.ops.rms_norm.npu_rms_norm import npu_rms_norm_forward

    model_args = ModelArguments(model="llamafactory/tiny-random-qwen3", kernel_config={"name": "auto"})
    model_engine = ModelEngine(model_args)
    # test enable apply kernel plugin
    if hasattr(torch, "npu"):
        assert model_engine.model.model.layers[0].input_layernorm.forward.__code__ == npu_rms_norm_forward.__code__
    else:
        assert model_engine.model.model.layers[0].input_layernorm.forward.__code__ != npu_rms_norm_forward.__code__

    assert "Qwen3ForCausalLM" in model_engine.model.__class__.__name__
