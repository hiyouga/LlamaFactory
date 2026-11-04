"""Execute production gradient-clipping functions with real Gloo/DTensor meshes.

Collectives, tensors, norm, gradients and clipping are real PyTorch operations.
"""

import math
import socket
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard


def worker(rank, port):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=4)
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
    from llamafactory.v1.plugins.trainer_plugins.distributed import fsdpturbo as backend

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
    observed = backend.clip_grad_norm_(model, 1.0)
    assert abs(observed - expected) < 1e-5
    for param, reference in zip(model.parameters(), references):
        torch.testing.assert_close(param.grad.to_local(), reference / expected, atol=1e-6, rtol=1e-5)
    # A missing contribution on one CP rank must be treated as zero, not skip a collective.
    for param, reference in zip(model.parameters(), references):
        param.grad = (
            DTensor.from_local(2 * reference, param.device_mesh, [Shard(0)], run_check=False) if cp == 0 else None
        )
    observed = backend.clip_grad_norm_(model, 1.0)
    assert abs(observed - expected) < 1e-5
    for param, reference in zip(model.parameters(), references):
        torch.testing.assert_close(param.grad.to_local(), reference / expected, atol=1e-6, rtol=1e-5)
    assert unused.grad is None
    dist.destroy_process_group()


def test_cp_replica_gradients_are_averaged_before_norm_and_update():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(port,), nprocs=4, join=True)
