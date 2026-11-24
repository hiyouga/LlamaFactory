# Copyright 2026 the LlamaFactory team.
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

"""Real multi-rank loading and existing-model contracts against the optional backend."""

import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, distribute_tensor

from llamafactory.v1.plugins.trainer_plugins.distributed.fsdp2 import _save_pretrained
from llamafactory.v1.plugins.trainer_plugins.distributed.fsdpturbo import (
    FSDPTurboFSDP2Engine,
    FSDPTurboParallelState,
)
from llamafactory.v1.utils.packages import is_transformers_version_greater_than


def _engine(rank, mesh):
    interface = SimpleNamespace(
        strategy=SimpleNamespace(cp_size=1),
        current_device=torch.device("cpu"),
        get_world_size=lambda *_: 4,
        get_device_mesh=lambda *_: mesh,
    )
    state = FSDPTurboParallelState()
    state.initialize(interface, {"ep_size": 2})
    engine = object.__new__(FSDPTurboFSDP2Engine)
    engine.rank, engine.ep_size, engine.ep_fsdp_size = rank, 2, 2
    engine.parallel_state = state
    engine.mixed_precision = "fp32"
    engine.dist_config = {"ep_size": 2, "ep_dispatcher": "eager", "fsdp_implementation": "native"}
    return engine


def _assert_full_weights(model, expected):
    for name, value in model.state_dict().items():
        full = value.full_tensor() if isinstance(value, DTensor) else value
        torch.testing.assert_close(full, expected[name], rtol=0, atol=0, msg=name)


def _worker(rank, rendezvous, checkpoint):
    from transformers import DeepseekV4Config, DeepseekV4ForCausalLM

    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=4)
    try:
        torch.manual_seed(42)
        config = DeepseekV4Config(
            vocab_size=64,
            hidden_size=16,
            head_dim=16,
            q_lora_rank=8,
            num_attention_heads=4,
            o_groups=2,
            o_lora_rank=8,
            moe_intermediate_size=8,
            n_routed_experts=8,
            num_experts_per_tok=2,
            num_hidden_layers=1,
            num_hash_layers=0,
            compress_ratios=[0],
        )
        config._attn_implementation = "eager"
        model = DeepseekV4ForCausalLM(config)
        expected = {name: value.clone() for name, value in model.state_dict().items()}
        for name, value in expected.items():
            if ".experts." in name:
                # Every element identifies its expert and feature coordinates; shape checks cannot hide a permutation.
                value.copy_(torch.arange(value.numel()).reshape(value.shape) / 1000)
        mesh = init_device_mesh("cpu", (4,), mesh_dim_names=("dp",))
        engine = _engine(rank, mesh)
        use_compute_stream = callable(getattr(FSDPModule, "_set_unshard_async_op", None))
        engine.dist_config["efsdp_all_gather_on_compute_stream"] = use_compute_stream
        engine.prepare_model_ep(model)
        expert = model.model.layers[0].mlp.experts
        assert isinstance(expert, FSDPModule)
        group = expert._get_fsdp_state()._fsdp_param_group
        if use_compute_stream:
            assert group.unshard_async_op
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                engine._copy_weights(parameter, expected[name])
                if isinstance(parameter, DTensor):
                    oracle = distribute_tensor(
                        expected[name], parameter.device_mesh, parameter.placements, src_data_rank=None
                    )
                    torch.testing.assert_close(parameter.to_local(), oracle.to_local(), rtol=0, atol=0, msg=name)
        _assert_full_weights(model, expected)
        full_state = {
            name: value.full_tensor() if isinstance(value, DTensor) else value
            for name, value in model.state_dict().items()
        }
        if rank == 0:
            _save_pretrained(model, checkpoint, full_state)
            reloaded = DeepseekV4ForCausalLM.from_pretrained(
                checkpoint, dtype=torch.float32, attn_implementation="eager"
            )
            _assert_full_weights(reloaded, expected)
        dist.barrier()
        target = DeepseekV4ForCausalLM(config)
        engine.prepare_model_ep(target)
        with torch.no_grad():
            engine._load_weights_from_hf_checkpoint(target, checkpoint)
        _assert_full_weights(target, expected)
    finally:
        dist.destroy_process_group()


def test_native_four_rank_expert_loading_roundtrip():
    pytest.importorskip("fsdp_turbo.distributed.expert_parallel.expert_parallel")
    pytest.importorskip("transformers.models.deepseek_v4.modeling_deepseek_v4")
    if not is_transformers_version_greater_than("5.16.0"):
        pytest.skip("The pinned backend's native model contract requires Transformers >=5.16")
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Requires Gloo")
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_worker, args=(directory + "/group", directory + "/checkpoint"), nprocs=4, join=True)
