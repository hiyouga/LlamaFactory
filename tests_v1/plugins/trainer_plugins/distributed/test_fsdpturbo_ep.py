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
    assert spec.ep_fsdp_modules == ["model.language_model.layers.{*}.mlp.experts"]
    assert spec.moe_modules == ["model.language_model.layers.{*}.mlp"]


@pytest.mark.parametrize(
    ("model_type", "experts_path"),
    [
        ("qwen3_moe", "model.layers.{*}.mlp.experts"),
        ("qwen3_5_moe", "model.language_model.layers.{*}.mlp.experts"),
    ],
)
def test_fsdpturbo_efsdp_targets_only_routed_experts(model_type, experts_path):
    spec = FSDPTurboEPModelSpec.get(_Model(model_type))
    assert spec is not None
    assert FSDPTurboFSDP2Engine._get_ep_fsdp_modules(spec) == [experts_path]
    assert spec.moe_modules == [experts_path.removesuffix(".experts")]


def test_fsdpturbo_default_efsdp_preserves_ep_module_paths():
    paths = ["model.layers.{*}.mlp.experts", "model.layers.{*}.routed_experts"]
    spec = FSDPTurboEPModelSpec(ep_modules=paths)
    assert FSDPTurboFSDP2Engine._get_ep_fsdp_modules(spec) == paths


@pytest.mark.parametrize("override", [[], ["model.layers.{*}.custom_experts"]])
def test_fsdpturbo_efsdp_preserves_explicit_override(override):
    spec = FSDPTurboEPModelSpec(ep_modules=["model.layers.{*}.mlp.experts"], ep_fsdp_modules=override)
    assert FSDPTurboFSDP2Engine._get_ep_fsdp_modules(spec) == override


def test_fsdpturbo_uses_class_plugin_and_strict_backend_params():
    plugin = DistributedPlugin("fsdpturbo")
    params = plugin.parse_params({"name": "fsdpturbo", "ep_size": 4}, FSDPTurboParams)

    assert params.ep_size == 4
    assert params.fp32_moe is False
    assert plugin.parse_params({"name": "fsdpturbo", "fp32_moe": True}, FSDPTurboParams).fp32_moe
    assert callable(plugin.shard_model)
    assert callable(plugin.clip_grad_norm)
    with pytest.raises(ValueError, match="Unknown params"):
        plugin.parse_params({"name": "fsdpturbo", "cp_size": 2}, FSDPTurboParams)
    for key in ("ep_modules", "ep_fsdp_modules", "moe_modules"):
        with pytest.raises(ValueError, match="Unknown params"):
            plugin.parse_params({"name": "fsdpturbo", key: ["model.layers.*.mlp"]}, FSDPTurboParams)


@pytest.mark.parametrize("precision", ["bf16", "fp32"])
@pytest.mark.parametrize("dispatcher", ["eager", "fused"])
def test_fsdpturbo_sets_public_efsdp_plan(monkeypatch, precision, dispatcher):
    expert_parallel_module = pytest.importorskip("fsdp_turbo.distributed.expert_parallel.expert_parallel")
    expert_fully_shard_module = pytest.importorskip(
        "fsdp_turbo.distributed.expert_parallel.expert_fully_shard_parallel"
    )
    captured = {}
    monkeypatch.setattr(expert_parallel_module, "expert_parallelize_modules", lambda model, mesh, plan: model)

    def _expert_fully_shard_modules(model, mesh, ep_plan, fsdp_plan):
        captured["gradient_divide_factor"] = ep_plan.gradient_divide_factor
        captured["ep_plan"] = ep_plan
        captured["fsdp_plan"] = fsdp_plan
        return model

    monkeypatch.setattr(expert_fully_shard_module, "expert_fully_shard_modules", _expert_fully_shard_modules)

    engine = object.__new__(FSDPTurboFSDP2Engine)
    engine.dist_config = {"ep_dispatcher": dispatcher}
    engine.mixed_precision = precision
    engine.ep_size = 4
    engine.ep_fsdp_size = 2
    engine.parallel_state = SimpleNamespace(efsdp_size=2, ep_mesh=object(), efsdp_mesh=object())
    engine.rank = 0

    engine.prepare_model_ep(_Model("qwen3_5_moe"))

    assert captured["gradient_divide_factor"] == 8.0
    ep_plan = captured["ep_plan"]
    if hasattr(ep_plan, "use_fused_ops"):
        assert ep_plan.dispatcher == "all2all"
        assert ep_plan.use_fused_ops == (dispatcher == "fused")
        assert callable(expert_parallel_module.get_experts_forward_fn(None, ep_plan, ep_plan.dispatcher))
    else:
        assert ep_plan.dispatcher == dispatcher
    from fsdp_turbo.distributed.fully_shard_parallel.fully_shard_parallel import get_mixprecision_policy

    policy = get_mixprecision_policy(captured["fsdp_plan"])
    assert policy.param_dtype == (torch.bfloat16 if precision == "bf16" else torch.float32)
    assert policy.reduce_dtype == torch.float32
    assert policy.cast_forward_inputs


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


@pytest.mark.parametrize("model_type", ["qwen3_moe", "qwen3_5_moe"])
def test_fsdpturbo_real_efsdp_selection_excludes_dense_moe_parameters(model_type):
    from torch.distributed.fsdp._fully_shard._fsdp_init import _get_managed_modules, _get_managed_states

    backend = pytest.importorskip("fsdp_turbo.distributed.expert_parallel.expert_fully_shard_parallel")
    from fsdp_turbo.fsdp_turbo_config import EPPlanConfig

    if model_type == "qwen3_5_moe":
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig as Config
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock as Block
    else:
        from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig as Config
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock as Block

    config = Config(hidden_size=8, moe_intermediate_size=4, shared_expert_intermediate_size=4, num_experts=4)
    model = _Model(model_type)
    model.model = torch.nn.Module()
    parent = model.model
    if model_type == "qwen3_5_moe":
        parent.language_model = torch.nn.Module()
        parent = parent.language_model
    layer = torch.nn.Module()
    layer.mlp = Block(config)
    parent.layers = torch.nn.ModuleList([layer])
    spec = FSDPTurboEPModelSpec.get(model)
    plan = EPPlanConfig(
        apply_modules=spec.ep_modules, apply_efsdp_modules=FSDPTurboFSDP2Engine._get_ep_fsdp_modules(spec)
    )
    targets = backend.get_efsdp_modules(model, plan)
    parameters, _ = _get_managed_states(_get_managed_modules(tuple(targets)))
    assert set(parameters) == set(layer.mlp.experts.parameters())
    assert not set(parameters).intersection(layer.mlp.gate.parameters())
    if model_type == "qwen3_5_moe":
        assert not set(parameters).intersection(layer.mlp.shared_expert.parameters())
        assert not set(parameters).intersection(layer.mlp.shared_expert_gate.parameters())


@pytest.mark.parametrize("enabled", [False, True])
def test_fp32_moe_preserves_outer_precision_and_expert_ownership(monkeypatch, enabled):
    backend_config = pytest.importorskip("fsdp_turbo.fsdp_turbo_config")

    from llamafactory.v1.plugins.trainer_plugins.distributed.fsdp2 import FSDP2Engine

    if enabled and not hasattr(backend_config.EPPlanConfig, "use_fused_ops"):
        pytest.skip("FP32 MoE requires configurable EFSDP precision.")
    model = _Model("qwen3_moe")
    model.embed = torch.nn.Embedding(4, 4, dtype=torch.bfloat16)
    model.model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.mlp = torch.nn.Module()
    layer.mlp.gate = torch.nn.Linear(4, 4, dtype=torch.bfloat16)
    layer.mlp.experts = torch.nn.Linear(4, 4, dtype=torch.bfloat16)
    model.model.layers = torch.nn.ModuleList([layer])
    engine = object.__new__(FSDPTurboFSDP2Engine)
    engine.dist_config = {"fp32_moe": enabled}
    engine.mixed_precision = "bf16"
    engine.ep_fsdp_size = 2
    engine.fsdp_mesh = object()
    engine.reshard_after_forward = True
    engine.offload_params = False
    ignored = set(layer.mlp.experts.parameters())
    calls = []
    monkeypatch.setattr(engine, "prepare_model_ep", lambda model: (model, ignored))
    monkeypatch.setattr(fsdpturbo_module, "fully_shard", lambda module, **kwargs: calls.append((module, kwargs)))
    monkeypatch.setattr(FSDP2Engine, "prepare_model", lambda self, model, ignored_params: model)

    assert engine.prepare_model(model) is model
    assert model.embed.weight.dtype == torch.bfloat16
    if enabled:
        assert all(p.dtype == torch.float32 for p in layer.mlp.parameters())
        assert len(calls) == 1 and calls[0][0] is layer.mlp
        config = calls[0][1]
        assert config["mesh"] is engine.fsdp_mesh
        assert config["ignored_params"] == ignored
        assert config["mp_policy"].param_dtype == torch.float32
        assert config["mp_policy"].reduce_dtype == torch.float32
        assert config["mp_policy"].output_dtype == torch.bfloat16
    else:
        assert not calls
        assert all(p.dtype == torch.bfloat16 for p in layer.mlp.parameters())


@pytest.mark.parametrize("model_type", ["unregistered", "qwen3_moe"])
def test_fp32_moe_rejects_unregistered_or_unmatched_modules(model_type):
    pytest.importorskip("fsdp_turbo.utils.str_match")
    with pytest.raises(ValueError, match="fp32_moe"):
        FSDPTurboFSDP2Engine._get_fp32_moe_modules(_Model(model_type))


def test_fp32_moe_rejects_lora_before_wrapping():
    pytest.importorskip("fsdp_turbo.fsdp_turbo_config")
    from peft.tuners.lora.layer import Linear

    model = _Model("qwen3_moe")
    model.adapter = Linear(torch.nn.Linear(4, 4), "default", r=2, lora_alpha=2)
    engine = object.__new__(FSDPTurboFSDP2Engine)
    engine.dist_config = {"fp32_moe": True}
    with pytest.raises(ValueError, match="not LoRA"):
        engine.prepare_model(model)


def test_fp32_moe_rejects_legacy_efsdp_precision_before_casting(monkeypatch):
    backend_config = pytest.importorskip("fsdp_turbo.fsdp_turbo_config")

    monkeypatch.setattr(backend_config, "EPPlanConfig", type("LegacyEPPlanConfig", (), {}))
    model = _Model("qwen3_moe")
    model.weight = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    engine = object.__new__(FSDPTurboFSDP2Engine)
    engine.dist_config = {"fp32_moe": True}
    engine.ep_fsdp_size = 2
    with pytest.raises(ValueError, match="configurable EFSDP precision"):
        engine.prepare_model(model)
    assert model.weight.dtype == torch.bfloat16
