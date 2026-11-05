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

import types

import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import PretrainedConfig, PreTrainedModel

from llamafactory.v1.accelerator.interface import DistributedInterface
from llamafactory.v1.plugins.trainer_plugins.distributed.fsdp2 import FSDP2Engine


NUM_EXPERTS = 11
HIDDEN_SIZE = 3
INTERMEDIATE_SIZE = 2


class Holder(nn.Module):
    pass


class FakeQwen3MoeConfig(PretrainedConfig):
    model_type = "qwen3_moe"


# transformers v5 only collects model-specific weight conversions from `PreTrainedModel` submodules.
class FakeFusedExpertsModel(PreTrainedModel):
    config_class = FakeQwen3MoeConfig
    base_model_prefix = "model"

    def __init__(self):
        super().__init__(FakeQwen3MoeConfig())

        self.model = Holder()
        self.model.layers = nn.ModuleList([Holder()])
        self.model.layers[0].mlp = Holder()
        self.model.layers[0].mlp.experts = Holder()
        self.model.layers[0].mlp.experts.gate_up_proj = nn.Parameter(
            torch.zeros(NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
        )
        self.model.layers[0].mlp.experts.down_proj = nn.Parameter(
            torch.zeros(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
        )


class FakeLegacyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False)


class FakeLegacyExpertsModel(nn.Module):
    base_model_prefix = "model"

    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(model_type="qwen3_moe")

        self.model = Holder()
        self.model.layers = nn.ModuleList([Holder()])
        self.model.layers[0].mlp = Holder()
        self.model.layers[0].mlp.experts = nn.ModuleList([FakeLegacyExpert() for _ in range(NUM_EXPERTS)])


def build_engine():
    DistributedInterface()
    return FSDP2Engine({"name": "fsdp2"})


def build_checkpoint():
    ckpt = {}
    gates, ups, downs = [], [], []

    for i in range(NUM_EXPERTS):
        # Use distinct values per expert so ordering bugs are easy to catch.
        gate = torch.full((INTERMEDIATE_SIZE, HIDDEN_SIZE), float(i))
        up = torch.full((INTERMEDIATE_SIZE, HIDDEN_SIZE), float(i) + 100.0)
        down = torch.full((HIDDEN_SIZE, INTERMEDIATE_SIZE), float(i) + 200.0)

        ckpt[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = gate
        ckpt[f"model.layers.0.mlp.experts.{i}.up_proj.weight"] = up
        ckpt[f"model.layers.0.mlp.experts.{i}.down_proj.weight"] = down

        gates.append(gate)
        ups.append(up)
        downs.append(down)

    return ckpt, gates, ups, downs


def test_fsdp2_gate_up_proj_loading(tmp_path):
    engine = build_engine()
    ckpt, gates, ups, downs = build_checkpoint()
    save_file(ckpt, str(tmp_path / "model.safetensors"))

    fused_model = FakeFusedExpertsModel()
    conversion_ctx = engine._try_build_hf_weight_conversion_context(fused_model)

    if conversion_ctx is not None:
        # In transformers v5-style environments, legacy expert weights should be fused.
        engine._load_weights_from_hf_checkpoint(fused_model, str(tmp_path))

        expected_gate_up = torch.cat(
            [torch.stack(gates, dim=0), torch.stack(ups, dim=0)],
            dim=1,
        )
        expected_down = torch.stack(downs, dim=0)

        experts = fused_model.model.layers[0].mlp.experts
        assert torch.allclose(experts.gate_up_proj, expected_gate_up)
        assert torch.allclose(experts.down_proj, expected_down)

        # Check a double-digit expert index to ensure natural ordering is preserved.
        assert torch.allclose(experts.gate_up_proj[2], expected_gate_up[2])
        assert torch.allclose(experts.gate_up_proj[10], expected_gate_up[10])
        assert torch.allclose(experts.down_proj[2], expected_down[2])
        assert torch.allclose(experts.down_proj[10], expected_down[10])

    else:
        # In pre-v5 environments, the loader should fall back to direct copy.
        legacy_model = FakeLegacyExpertsModel()
        engine._load_weights_from_hf_checkpoint(legacy_model, str(tmp_path))

        experts = legacy_model.model.layers[0].mlp.experts
        for i in range(NUM_EXPERTS):
            assert torch.allclose(experts[i].gate_proj.weight, gates[i])
            assert torch.allclose(experts[i].up_proj.weight, ups[i])
            assert torch.allclose(experts[i].down_proj.weight, downs[i])
