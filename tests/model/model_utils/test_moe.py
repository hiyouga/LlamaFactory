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

from llamafactory.model.model_utils.moe import configure_moe


def test_configure_moe_sets_qwen3_5_moe_aux_loss_config():
    config = SimpleNamespace(model_type="qwen3_5_moe")
    model_args = SimpleNamespace(moe_aux_loss_coef=0.01)

    configure_moe(config, model_args, is_trainable=True)

    assert config.output_router_logits is True
    assert config.router_aux_loss_coef == 0.01
