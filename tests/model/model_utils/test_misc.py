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


class _DiscoveryFixture(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="test")
        self.q_proj = torch.nn.Linear(4, 4)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.SiLU(),
            torch.nn.Linear(4, 4),
        )


class _ExpansionFixture(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="test", num_hidden_layers=2)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "mlp": torch.nn.ModuleDict(
                            {
                                "0": torch.nn.Linear(4, 4),
                                "01": torch.nn.SiLU(),
                            }
                        )
                    }
                )
                for _ in range(2)
            ]
        )


def test_find_all_linear_modules_preserves_numeric_paths():
    model = _DiscoveryFixture()

    target_modules = set(find_all_linear_modules(model, freeze_vision_tower=False))

    assert target_modules == {"q_proj", "mlp.0", "mlp.2"}


def test_find_expanded_modules_keeps_numeric_paths_exact():
    model = _ExpansionFixture()
    target_modules = find_all_linear_modules(model, freeze_vision_tower=False)

    expanded_modules = find_expanded_modules(model, target_modules, num_layer_trainable=2, strict_numeric=True)

    assert expanded_modules == ["layers.0.mlp.0", "layers.1.mlp.0"]


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
