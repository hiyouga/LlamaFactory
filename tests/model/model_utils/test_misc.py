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

import os

import pytest
import torch
from peft import LoraConfig, inject_adapter_in_model
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig

from llamafactory.model.model_utils.misc import find_all_linear_modules, find_expanded_modules


HF_TOKEN = os.getenv("HF_TOKEN")


@pytest.mark.skipif(not HF_TOKEN, reason="Gated model.")
def test_expanded_modules():
    config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    expanded_modules = find_expanded_modules(model, ["q_proj", "v_proj"], num_layer_trainable=4)
    assert expanded_modules == [
        "model.layers.7.self_attn.q_proj",
        "model.layers.7.self_attn.v_proj",
        "model.layers.15.self_attn.q_proj",
        "model.layers.15.self_attn.v_proj",
        "model.layers.23.self_attn.q_proj",
        "model.layers.23.self_attn.v_proj",
        "model.layers.31.self_attn.q_proj",
        "model.layers.31.self_attn.v_proj",
    ]


@pytest.mark.parametrize("ownership", [("unwrapped",), ("peft",), ("forced",), ("packed",), ("forced", "peft")])
def test_deepseek_v3_auto_lora_targets(ownership):
    model = torch.nn.Module()
    model.config = PretrainedConfig()
    model.config.model_type = "deepseek_v3"
    model.layers = torch.nn.ModuleList()
    model.lm_head = torch.nn.Linear(4, 4)
    projections = ("gate_proj", "up_proj", "down_proj")
    for owner in ownership:
        layer = torch.nn.Module()
        layer.experts = torch.nn.ModuleList(
            [torch.nn.ModuleDict({name: torch.nn.Linear(4, 4) for name in projections}) for _ in range(2)]
        )
        layer.shared_experts = torch.nn.ModuleDict({name: torch.nn.Linear(4, 4) for name in projections})
        layer._use_fused_expert_lora = owner == "forced"
        layer._fused_experts = owner == "packed"
        model.layers.append(layer)

    if "unwrapped" not in ownership:
        model._kt_wrappers = list(model.layers)

    assert set(find_all_linear_modules(model, False)) == set(projections)
    targets = find_all_linear_modules(model, False, for_kt_lora=True)
    inject_adapter_in_model(LoraConfig(r=2, target_modules=targets), model)
    for layer, owner in zip(model.layers, ownership):
        for expert in layer.experts:
            assert all(hasattr(proj, "lora_A") == (owner in ("peft", "unwrapped")) for proj in expert.values())
        assert all(hasattr(proj, "lora_A") for proj in layer.shared_experts.values())

    assert not hasattr(model.lm_head, "lora_A")
