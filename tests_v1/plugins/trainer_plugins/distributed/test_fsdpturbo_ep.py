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
import torch

from llamafactory.v1.plugins.trainer_plugins.distributed import fsdpturbo as fsdpturbo_module
from llamafactory.v1.plugins.trainer_plugins.distributed.fsdpturbo import (
    FSDPTurboEPModelSpec,
    FSDPTurboFSDP2Engine,
    FSDPTurboParallelState,
)
from llamafactory.v1.plugins.trainer_plugins.distributed.interface import (
    DistributedPlugin,
    FSDPTurboParams,
)


class _Model(torch.nn.Module):
    def __init__(self, model_type: str):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type)


def test_qwen35_ep_model_spec():
    spec = FSDPTurboEPModelSpec.get(_Model("qwen3_5_moe"))

    assert spec is not None
    assert spec.ep_modules == ["model.language_model.layers.{*}.mlp.experts"]
    assert spec.ep_fsdp_modules == ["model.language_model.layers.{*}.mlp"]


def test_fsdpturbo_uses_class_plugin_and_strict_backend_params():
    plugin = DistributedPlugin("fsdpturbo")
    params = plugin.parse_params({"name": "fsdpturbo", "ep_size": 4}, FSDPTurboParams)

    assert params.ep_size == 4
    assert callable(plugin.shard_model)
    assert callable(plugin.clip_grad_norm)
    with pytest.raises(ValueError, match="Unknown params"):
        plugin.parse_params({"name": "fsdpturbo", "cp_size": 2}, FSDPTurboParams)
    for key in ("ep_modules", "ep_fsdp_modules"):
        with pytest.raises(ValueError, match="Unknown params"):
            plugin.parse_params({"name": "fsdpturbo", key: ["model.layers.*.mlp"]}, FSDPTurboParams)


def test_fsdpturbo_sets_storage_dtype_inside_backend(monkeypatch):
    from llamafactory.v1.plugins.trainer_plugins.distributed.fsdp2 import FSDP2Engine

    monkeypatch.setattr(FSDP2Engine, "shard_model", lambda self, model: model)
    engine = object.__new__(FSDPTurboFSDP2Engine)
    engine.mixed_precision = "bf16"
    model = torch.nn.Linear(2, 2, dtype=torch.float32)

    assert engine.shard_model(model).weight.dtype == torch.bfloat16


def test_fsdpturbo_sets_public_efsdp_gradient_divide_factor(monkeypatch):
    expert_parallel_module = pytest.importorskip("fsdp_turbo.distributed.expert_parallel.expert_parallel")
    expert_fully_shard_module = pytest.importorskip(
        "fsdp_turbo.distributed.expert_parallel.expert_fully_shard_parallel"
    )
    captured = {}
    monkeypatch.setattr(expert_parallel_module, "expert_parallelize_modules", lambda model, mesh, plan: model)

    def _expert_fully_shard_modules(model, mesh, ep_plan, fsdp_plan):
        captured["gradient_divide_factor"] = ep_plan.gradient_divide_factor
        return model

    monkeypatch.setattr(expert_fully_shard_module, "expert_fully_shard_modules", _expert_fully_shard_modules)

    engine = object.__new__(FSDPTurboFSDP2Engine)
    engine.dist_config = {"ep_dispatcher": "eager"}
    engine.ep_size = 4
    engine.ep_fsdp_size = 2
    engine.parallel_state = SimpleNamespace(efsdp_size=2, ep_mesh=object(), efsdp_mesh=object())
    engine.rank = 0

    engine.prepare_model_ep(_Model("qwen3_5_moe"))

    assert captured["gradient_divide_factor"] == 8.0


def test_fsdpturbo_owns_expert_mesh_topology(monkeypatch):
    calls = []

    class _Mesh:
        def __init__(self, name="expert"):
            self.name = name

        def __getitem__(self, name):
            return _Mesh(name)

    def _init_device_mesh(**kwargs):
        calls.append(kwargs)
        return _Mesh()

    class _DistributedInterface:
        current_device = torch.device("cpu")
        strategy = SimpleNamespace(cp_size=1)

        def get_world_size(self, dim):
            return 16

        def get_device_mesh(self, dim):
            return _Mesh("dp")

    monkeypatch.setattr(fsdpturbo_module, "init_device_mesh", _init_device_mesh)
    state = FSDPTurboParallelState()
    state.initialize(_DistributedInterface(), {"ep_size": 8})

    assert calls == [
        {
            "device_type": "cpu",
            "mesh_shape": (1, 2, 8, 1),
            "mesh_dim_names": ("edp", "efsdp", "ep", "expert_cp"),
        }
    ]
    assert state.ep_mesh.name == "ep"
    assert state.efsdp_mesh.name == "efsdp"
    assert state.expert_cp_mesh.name == "expert_cp"
