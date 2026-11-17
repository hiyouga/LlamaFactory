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

import copy
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor

from llamafactory.v1.plugins.trainer_plugins.distributed.fsdpturbo import (
    FSDPTurboEPModelSpec,
    FSDPTurboFSDP2Engine,
    FSDPTurboParallelState,
)
from llamafactory.v1.plugins.trainer_plugins.distributed.fsdpturbo import parallel_state as fsdpturbo_module
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
    assert callable(plugin.shard_model)
    assert callable(plugin.clip_grad_norm)
    with pytest.raises(ValueError, match="Unknown params"):
        plugin.parse_params({"name": "fsdpturbo", "cp_size": 2}, FSDPTurboParams)
    for key in ("ep_modules", "ep_fsdp_modules"):
        with pytest.raises(ValueError, match="Unknown params"):
            plugin.parse_params({"name": "fsdpturbo", key: ["model.layers.*.mlp"]}, FSDPTurboParams)


@pytest.mark.parametrize("precision", ["bf16", "fp32"])
@pytest.mark.parametrize("dispatcher", ["eager", "fused"])
@pytest.mark.parametrize("efsdp_size", [1, 2])
def test_fsdpturbo_sets_public_efsdp_plan(monkeypatch, precision, dispatcher, efsdp_size):
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
    engine.ep_fsdp_size = efsdp_size
    engine.parallel_state = SimpleNamespace(efsdp_size=efsdp_size, ep_mesh=object(), efsdp_mesh=object())
    engine.rank = 0

    engine.prepare_model_ep(_Model("qwen3_5_moe"))

    assert captured["gradient_divide_factor"] == 4 * efsdp_size
    ep_plan = captured["ep_plan"]
    assert ep_plan.dispatcher == "all2all"
    assert ep_plan.use_fused_ops == (dispatcher == "fused")
    assert callable(expert_parallel_module.get_experts_forward_fn(None, ep_plan, ep_plan.dispatcher))
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
    parameters = {param for module in targets for param in module.parameters()}
    assert set(parameters) == set(layer.mlp.experts.parameters())
    assert not set(parameters).intersection(layer.mlp.gate.parameters())
    if model_type == "qwen3_5_moe":
        assert not set(parameters).intersection(layer.mlp.shared_expert.parameters())
        assert not set(parameters).intersection(layer.mlp.shared_expert_gate.parameters())


def test_deepseek_v4_ep_model_spec():
    spec = FSDPTurboEPModelSpec.get(_Model("deepseek_v4"))
    assert spec is not None
    assert spec.ep_modules == ["model.layers.{*}.mlp.experts"]
    assert spec.ep_fsdp_modules == spec.ep_modules


def test_ep_dispatcher_rejects_unknown_mode():
    with pytest.raises(ValueError, match="ep_dispatcher"):
        FSDPTurboParams(ep_dispatcher="unknown")


def _qwen_expert_contract(model_type, ep_mesh, device):
    from fsdp_turbo.distributed.expert_parallel.expert_parallel import expert_parallelize_modules
    from fsdp_turbo.fsdp_turbo_config import EPPlanConfig

    if model_type == "qwen3_moe":
        from transformers.models.qwen3_moe import modeling_qwen3_moe as hf

        config = hf.Qwen3MoeConfig(hidden_size=16, moe_intermediate_size=8, num_experts=8)
        block = hf.Qwen3MoeSparseMoeBlock(config)
    else:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf

        config = hf.Qwen3_5MoeTextConfig(hidden_size=16, moe_intermediate_size=8, num_experts=8)
        block = hf.Qwen3_5MoeSparseMoeBlock(config)
    config._experts_implementation = "eager"
    for parameter in block.experts.parameters():
        torch.nn.init.normal_(parameter, mean=0.1, std=0.02)
    block.to(device)
    reference = copy.deepcopy(block.experts)
    had_alias = hasattr(block.experts, "hidden_size")
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type=model_type)
    model.model = torch.nn.Module()
    parent = model.model
    if model_type == "qwen3_5_moe":
        parent.language_model = torch.nn.Module()
        parent = parent.language_model
    layer = torch.nn.Module()
    layer.mlp = block
    parent.layers = torch.nn.ModuleList([layer])
    spec = FSDPTurboEPModelSpec.get(model)
    spec.prepare(model)
    plan = EPPlanConfig(apply_modules=spec.ep_modules, use_fused_ops=False)
    expert_parallelize_modules(model, ep_mesh, plan)
    assert hasattr(block.experts, "hidden_size") == had_alias

    hidden = torch.randn(6, 16, device="cpu").to(device).requires_grad_()
    original = hidden.detach().clone().requires_grad_()
    indices = (torch.arange(12, device="cpu").reshape(6, 2) % 8).to(device)
    weights = torch.tensor([0.4, 0.6], device=device).expand(6, -1)
    expected = reference(original, indices, weights)
    actual = block.experts(hidden, indices, weights)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(hidden.grad, original.grad, rtol=1e-5, atol=1e-6)
    for (name, parameter), (_, ref) in zip(
        block.experts.named_parameters(), reference.named_parameters(), strict=True
    ):
        oracle = distribute_tensor(ref.grad, parameter.device_mesh, parameter.placements, src_data_rank=None)
        # EP-only dispatch accumulates this pair's inputs; EFSDP normally supplies the DP average.
        torch.testing.assert_close(
            parameter.grad.to_local() / ep_mesh.size(), oracle.to_local(), rtol=1e-5, atol=1e-6, msg=name
        )


def _qwen_accelerator_worker(rank, rendezvous, device_type):
    from fsdp_turbo.utils.device import set_accelerator_compatible

    set_accelerator_compatible()
    torch.get_device_module(device_type).set_device(rank)
    torch.set_float32_matmul_precision("highest")
    if device_type == "npu":
        torch.get_device_module(device_type).matmul.allow_hf32 = False
    dist.init_process_group(
        "hccl" if device_type == "npu" else "nccl", init_method="file://" + rendezvous, rank=rank, world_size=2
    )
    try:
        mesh = init_device_mesh(device_type, (2,), mesh_dim_names=("ep",))
        for model_type in ("qwen3_moe", "qwen3_5_moe"):
            torch.manual_seed(43)
            _qwen_expert_contract(model_type, mesh, torch.device(device_type, rank))
    finally:
        dist.destroy_process_group()


def test_qwen_native_backend_ep_forward_backward_without_legacy_alias(monkeypatch):
    from llamafactory.v1.accelerator.helper import get_current_accelerator, get_device_count
    from llamafactory.v1.utils.packages import is_transformers_version_greater_than

    device = get_current_accelerator()
    if device.type not in ("npu", "cuda") or get_device_count() < 2:
        pytest.skip("Requires two admitted NPU/CUDA devices")
    if not is_transformers_version_greater_than("5.16.0"):
        pytest.skip("The pinned backend's native expert contract requires Transformers >=5.16")
    monkeypatch.setenv("EP_BINCOUNT_DISPATCH", "1")
    pytest.importorskip("fsdp_turbo.distributed.expert_parallel.expert_parallel")
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_qwen_accelerator_worker, args=(directory + "/group", device.type), nprocs=2, join=True)
