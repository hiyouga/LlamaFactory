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

import torch

from .....accelerator.interface import Dim, DistributedInterface
from .....utils.types import HFModel
from .parallel_state import get_fsdpturbo_parallel_state


def _grad_to_local_fp32(grad: torch.Tensor) -> torch.Tensor:
    from torch.distributed._tensor import DTensor

    local_grad = grad.to_local() if isinstance(grad, DTensor) else grad
    return local_grad.detach().to(torch.float32)


def _local_pth_sum(parameters: list[torch.nn.Parameter], norm_type: float, device: torch.device) -> torch.Tensor:
    total = torch.zeros((), device=device, dtype=torch.float32)
    for param in parameters:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        total = total + torch.norm(_grad_to_local_fp32(grad), p=norm_type).pow(norm_type)
    return total


def _allreduce_sum_(value: torch.Tensor, groups: list[object]) -> torch.Tensor:
    import torch.distributed as dist

    for group in groups:
        if group is not None:
            dist.all_reduce(value, op=dist.ReduceOp.SUM, group=group)
    return value


@torch.no_grad()
def _average_cp_gradients(params: list[torch.nn.Parameter], group, device: torch.device) -> None:
    """Average CP contributions to the same parameter shard before clipping or updating it."""
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    if group is None or not params:
        return
    cp_size = dist.get_world_size(group)
    if cp_size == 1:
        return

    # Preserve None for globally unused parameters; an unused replica of an active
    # parameter still participates with a zero gradient.
    present = torch.tensor([param.grad is not None for param in params], dtype=torch.int32, device=device)
    dist.all_reduce(present, op=dist.ReduceOp.MAX, group=group)
    for param, has_grad in zip(params, present.tolist(), strict=True):
        if not has_grad:
            continue
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        grad = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
        if grad.numel() == 0:
            continue
        contiguous = grad.contiguous()
        # Bound the FP32 communication buffer to 32 MiB, including for CPU-offloaded gradients.
        for chunk in contiguous.view(-1).split(8 * 1024 * 1024):
            reduced = chunk.to(device=device, dtype=torch.float32)
            dist.all_reduce(reduced, group=group)
            chunk.copy_(reduced.div_(cp_size))
        if not grad.is_contiguous():
            grad.copy_(contiguous)


def _parameter_groups(model: HFModel, parallel_state) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    from torch.distributed.tensor import DTensor

    ep_params: list[torch.nn.Parameter] = []
    non_ep_params: list[torch.nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        mesh_names = set(param.device_mesh.mesh_dim_names or ()) if isinstance(param, DTensor) else set()
        is_ep_side = bool(mesh_names & {parallel_state.EP, parallel_state.EFSDP})
        if is_ep_side:
            ep_params.append(param)
        else:
            non_ep_params.append(param)

    return ep_params, non_ep_params


def synchronize_gradients(model: HFModel) -> None:
    """Finalize CP replicas once after accumulated backward and before clipping/stepping."""
    interface = DistributedInterface()
    state = get_fsdpturbo_parallel_state()
    if not state.initialized:
        raise RuntimeError("FSDPTurbo parallel state must be initialized before synchronizing gradients.")
    if state.cp_size <= 1:
        return
    experts, dense = _parameter_groups(model, state)
    expert_group = state.expert_cp_mesh.get_group() if state.expert_cp_mesh is not None else None
    _average_cp_gradients(dense, interface.get_group(Dim.CP), interface.current_device)
    _average_cp_gradients(experts, expert_group, interface.current_device)


def clip_grad_norm_(model: HFModel, max_norm: float, **kwargs) -> float:
    """Clip already-synchronized mixed-mesh gradients; call synchronize_gradients first.

    Avoids torch.nn.utils.get_total_norm() since mixed DTensor meshes
    (`dp` vs `efsdp`/`ep`) may hit DTensor stack propagation failures.
    """
    norm_type = float(kwargs.get("norm_type", 2.0))
    dist_interface = DistributedInterface()
    parallel_state = get_fsdpturbo_parallel_state()
    if not parallel_state.initialized:
        raise RuntimeError("FSDPTurbo parallel state must be initialized before clipping gradients.")

    device = dist_interface.current_device
    dp_group = dist_interface.get_group(Dim.DP)
    ep_group = parallel_state.ep_mesh.get_group() if parallel_state.ep_mesh is not None else None
    efsdp_group = parallel_state.efsdp_mesh.get_group() if parallel_state.efsdp_mesh is not None else None
    ep_params, non_ep_params = _parameter_groups(model, parallel_state)
    if not ep_params and not non_ep_params:
        return 0.0

    total_pth = torch.zeros((), device=device, dtype=torch.float32)
    if non_ep_params:
        non_ep_pth = _local_pth_sum(non_ep_params, norm_type, device)
        total_pth = total_pth + _allreduce_sum_(non_ep_pth, [dp_group])
    if ep_params:
        ep_pth = _local_pth_sum(ep_params, norm_type, device)
        total_pth = total_pth + _allreduce_sum_(ep_pth, [efsdp_group, ep_group])

    total_norm = total_pth.pow(1.0 / norm_type)
    clip_coef = min(max_norm / (float(total_norm.item()) + 1e-6), 1.0)
    if clip_coef < 1.0:
        for param in ep_params + non_ep_params:
            grad = getattr(param, "grad", None)
            if grad is not None:
                grad.detach().mul_(clip_coef)

    return float(total_norm.item())
