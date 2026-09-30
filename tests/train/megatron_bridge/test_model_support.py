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

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llamafactory.extras.constants import MEGATRON_BRIDGE_SUPPORTED_MODELS
from llamafactory.train.megatron_bridge.workflow import _check_model_support


# Keep in sync with MEGATRON_BRIDGE_SUPPORTED_MODELS (text LLM gpt_step path only).
EXPECTED_MEGATRON_BRIDGE_SUPPORTED_MODELS = {
    "deepseek_v3",
    "deepseek_v4",
    "llama",
    "mistral",
    "qwen2",
    "qwen3",
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen3_5_text",
    "qwen3_moe",
    "qwen3_next",
}

# Formerly allowlisted text models, plus multimodal types excluded in v0.
UNSUPPORTED_MEGATRON_BRIDGE_MODELS = (
    "bailing_moe_v2",
    "deepseek_v2",
    "ernie4_5_moe",
    "falcon_h1",
    "gemma",
    "gemma2",
    "gemma3",
    "gemma4",
    "glm4_moe",
    "glm4_moe_lite",
    "glm_moe_dsa",
    "gpt_oss",
    "kimi_k2",
    "mimo",
    "mimo_v2_flash",
    "minimax_m2",
    "mixtral",
    "nemotron",
    "nemotron_h",
    "olmoe",
    "qwen2_5_vl",
    "qwen2_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
    "step3p5",
)


def test_megatron_bridge_supported_models_match_expected():
    assert MEGATRON_BRIDGE_SUPPORTED_MODELS == EXPECTED_MEGATRON_BRIDGE_SUPPORTED_MODELS


@pytest.mark.parametrize("model_type", sorted(EXPECTED_MEGATRON_BRIDGE_SUPPORTED_MODELS))
def test_check_model_support_accepts_supported(model_type: str):
    with patch(
        "llamafactory.train.megatron_bridge.workflow.HfAutoConfig.from_pretrained",
        return_value=SimpleNamespace(model_type=model_type),
    ):
        _check_model_support(SimpleNamespace(model_name_or_path="dummy", trust_remote_code=False))


@pytest.mark.parametrize("model_type", UNSUPPORTED_MEGATRON_BRIDGE_MODELS)
def test_check_model_support_rejects_unsupported(model_type: str):
    assert model_type not in MEGATRON_BRIDGE_SUPPORTED_MODELS
    with patch(
        "llamafactory.train.megatron_bridge.workflow.HfAutoConfig.from_pretrained",
        return_value=SimpleNamespace(model_type=model_type),
    ):
        with pytest.raises(ValueError, match="not supported by the Megatron Bridge"):
            _check_model_support(SimpleNamespace(model_name_or_path="dummy", trust_remote_code=False))
