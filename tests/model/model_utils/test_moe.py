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

import pytest

from llamafactory.model.model_utils.moe import configure_moe


def test_configure_moe_sets_qwen3_5_moe_text_aux_loss_config():
    """The released MoE checkpoint reports `qwen3_5_moe_text` as its top-level model type."""
    qwen3_5_moe = pytest.importorskip("transformers.models.qwen3_5_moe.configuration_qwen3_5_moe")
    config = qwen3_5_moe.Qwen3_5MoeTextConfig()
    assert config.model_type == "qwen3_5_moe_text"
    # both fields exist on the real config, so setting them is honoured rather than invented
    assert config.output_router_logits is False
    assert config.router_aux_loss_coef == pytest.approx(0.001)

    configure_moe(config, SimpleNamespace(moe_aux_loss_coef=0.01), is_trainable=True)

    assert config.output_router_logits is True
    assert config.router_aux_loss_coef == pytest.approx(0.01)


def test_configure_moe_sets_qwen3_5_moe_aux_loss_on_text_config():
    """`qwen3_5_moe` is the vision+text wrapper; the router fields live in its text_config."""
    qwen3_5_moe = pytest.importorskip("transformers.models.qwen3_5_moe.configuration_qwen3_5_moe")
    config = qwen3_5_moe.Qwen3_5MoeConfig()
    assert config.model_type == "qwen3_5_moe"
    assert config.text_config.model_type == "qwen3_5_moe_text"
    # the wrapper itself carries neither field, so setting them there would be a no-op
    assert not hasattr(config, "output_router_logits")
    assert not hasattr(config, "router_aux_loss_coef")

    configure_moe(config, SimpleNamespace(moe_aux_loss_coef=0.01), is_trainable=True)

    assert config.text_config.output_router_logits is True
    assert config.text_config.router_aux_loss_coef == pytest.approx(0.01)
