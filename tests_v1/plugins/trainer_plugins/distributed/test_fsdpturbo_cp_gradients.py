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

"""Execute production gradient-clipping functions with real Gloo/DTensor meshes.

Collectives, tensors, norm, gradients and clipping are real PyTorch operations.
"""

import math
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard


def worker(rank, store_path):
    dist.init_process_group(
        "gloo", store=dist.FileStore(store_path, 4), rank=rank, world_size=4, timeout=timedelta(seconds=60)
    )
    try:
        check_cp_gradients()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def check_cp_gradients():
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "cp"))
    expert = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep", "expert_cp"))
    dp = mesh["dp"].get_local_rank()
    cp = mesh["cp"].get_local_rank()
    interface = SimpleNamespace(
        current_device=torch.device("cpu"),
        strategy=SimpleNamespace(cp_size=2),
        get_group=lambda dim: mesh["cp" if dim == "cp" else "dp"].get_group(),
        get_world_size=lambda dim: 2,
    )
    state = SimpleNamespace(
        initialized=True,
        ep_mesh=expert["ep"],
        efsdp_mesh=None,
        expert_cp_mesh=expert["expert_cp"],
        cp_size=2,
        EP="ep",
        EFSDP="efsdp",
    )
    from llamafactory.v1.plugins.trainer_plugins.distributed.fsdpturbo import gradients as backend
    from llamafactory.v1.plugins.trainer_plugins.distributed.interface import DistributedPlugin

    backend.DistributedInterface = lambda: interface
    backend.get_fsdpturbo_parallel_state = lambda: state
    model = torch.nn.Module()
    references = [torch.tensor([[3.0, 4.0], [5.0, 12.0]])[dp], torch.tensor([[1.0, 2.0], [2.0, 4.0]])[dp]]
    for i, (dm, ref) in enumerate(zip([mesh["dp"], expert["ep"]], references)):
        param = torch.nn.Parameter(DTensor.from_local(torch.zeros(2), dm, [Shard(0)], run_check=False))
        partial = torch.zeros(2)
        partial[cp] = 2 * ref[cp]  # SUM backward from replicated differentiable gather losses
        param.grad = DTensor.from_local(partial, dm, [Shard(0)], run_check=False)
        model.register_parameter(f"p{i}", param)
    unused = torch.nn.Parameter(DTensor.from_local(torch.ones(2), mesh["dp"], [Shard(0)], run_check=False))
    model.register_parameter("unused", unused)
    expected = math.sqrt(219.0)
    plugin = DistributedPlugin("fsdpturbo")
    plugin.synchronize_gradients(model)
    observed = plugin.clip_grad_norm(model, 1.0)
    assert abs(observed - expected) < 1e-5
    for param, reference in zip(model.parameters(), references):
        torch.testing.assert_close(param.grad.to_local(), reference / expected, atol=1e-6, rtol=1e-5)
    # A missing contribution on one CP rank must be treated as zero, not skip a collective.
    for param, reference in zip(model.parameters(), references):
        param.grad = (
            DTensor.from_local(2 * reference, param.device_mesh, [Shard(0)], run_check=False) if cp == 0 else None
        )
    plugin.synchronize_gradients(model)
    observed = plugin.clip_grad_norm(model, 1.0)
    assert abs(observed - expected) < 1e-5
    for param, reference in zip(model.parameters(), references):
        torch.testing.assert_close(param.grad.to_local(), reference / expected, atol=1e-6, rtol=1e-5)
    assert unused.grad is None

    # Clipping works on synchronized shards and must not run the CP collectives again.
    all_reduce = dist.all_reduce

    def reduce_only_parameter_shards(tensor, *args, **kwargs):
        assert kwargs.get("group") not in (mesh["cp"].get_group(), expert["expert_cp"].get_group())
        return all_reduce(tensor, *args, **kwargs)

    dist.all_reduce = reduce_only_parameter_shards
    try:
        plugin.clip_grad_norm(model, float("inf"))
    finally:
        dist.all_reduce = all_reduce


def test_cp_replica_gradients_are_averaged_before_norm_and_update(tmp_path):
    mp.spawn(worker, args=(str(tmp_path / "rendezvous"),), nprocs=4, join=True)
