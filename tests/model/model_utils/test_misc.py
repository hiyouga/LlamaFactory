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
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

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


@pytest.mark.parametrize("kt_wrapped", [False, True])
def test_deepseek_v3_auto_lora_targets(kt_wrapped):
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type="deepseek_v3")
    model.mlp = torch.nn.Module()
    model.mlp.experts = torch.nn.ModuleList([torch.nn.ModuleDict({"gate_proj": torch.nn.Linear(4, 4)})])
    model.mlp.shared_experts = torch.nn.ModuleDict({"gate_proj": torch.nn.Linear(4, 4)})
    model.lm_head = torch.nn.Linear(4, 4)
    if kt_wrapped:
        model._kt_wrappers = [object()]

    targets = find_all_linear_modules(model, freeze_vision_tower=False, for_kt_lora=True)
    assert set(targets) == ({"mlp.shared_experts.gate_proj"} if kt_wrapped else {"gate_proj"})
    assert set(find_all_linear_modules(model, freeze_vision_tower=False)) == {"gate_proj"}
