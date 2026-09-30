# Copyright 2025 Bytedance Ltd. and the LlamaFactory team.
#
# This code is inspired by the Bytedance's VeOmni library.
# https://github.com/ByteDance-Seed/VeOmni/blob/v0.1.4/veomni/utils/dist_utils.py
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

"""Utility functions used by the distributed interface.

Including:
- Environment info (rank, world_size, local_rank, etc.)
- Accelerator info (device type, device count, etc.)
- Collective communication operations (all_gather, all_reduce, broadcast)
- Synchronize processes and ensure main-process-first execution order
"""

import os
from collections.abc import Callable
from contextlib import contextmanager
from enum import StrEnum, unique
from functools import lru_cache, wraps
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist

from ..utils.types import ProcessGroup, Tensor, TensorLike


@unique
class DeviceType(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    META = "meta"
    MPS = "mps"
    NPU = "npu"
    XPU = "xpu"


@unique
class ReduceOp(StrEnum):
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"


def requires_accelerator(fn):
    """Decorator to check if torch.accelerator is available.

    Note: this api requires torch>=2.7.0, otherwise it will raise an AttributeError or RuntimeError
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not hasattr(torch, "accelerator"):
            raise RuntimeError("torch.accelerator is not available, please upgrade torch to 2.7.0 or higher.")

        return fn(*args, **kwargs)

    return wrapper


def is_distributed() -> bool:
    """Check if distributed environment is available."""
    return os.getenv("RANK") is not None


def get_rank() -> int:
    """Get rank."""
    return int(os.getenv("RANK", "0"))


def get_world_size() -> int:
    """Get world size."""
    return int(os.getenv("WORLD_SIZE", "1"))


def get_local_rank() -> int:
    """Get local rank."""
    return int(os.getenv("LOCAL_RANK", "0"))


def get_local_world_size() -> int:
    """Get local world size."""
    return int(os.getenv("LOCAL_WORLD_SIZE", "1"))


@lru_cache
@requires_accelerator
def get_current_accelerator(check_available: bool = True) -> torch.device:
    """Get current accelerator."""
    accelerator = torch.accelerator.current_accelerator(check_available=check_available)
    return accelerator or torch.device(DeviceType.CPU.value)


@lru_cache
@requires_accelerator
def get_device_count() -> int:
    """Get the number of available devices."""
    return torch.accelerator.device_count()


@requires_accelerator
def synchronize() -> None:
    """Synchronize all processes."""
    torch.accelerator.synchronize()


@requires_accelerator
def set_device_index() -> None:
    """Set current accelerator index to local rank."""
    if get_current_accelerator().type != DeviceType.CPU:
        torch.accelerator.set_device_index(get_local_rank())


@requires_accelerator
def get_current_device() -> torch.device:
    """Get current accelerator device."""
    if get_current_accelerator().type == DeviceType.CPU:
        return torch.device(DeviceType.CPU.value)
    else:
        return torch.device(type=get_current_accelerator().type, index=torch.accelerator.current_device_index())


def is_torch_cuda_available():
    """Check if CUDA is available."""
    return get_current_accelerator().type == DeviceType.CUDA


def is_torch_mps_available():
    """Check if MPS is available."""
    return get_current_accelerator().type == DeviceType.MPS


def is_torch_npu_available():
    """Check if NPU is available."""
    return get_current_accelerator().type == DeviceType.NPU


def is_torch_xpu_available():
    """Check if XPU is available."""
    return get_current_accelerator().type == DeviceType.XPU


def operate_tensorlike(fn: Callable[[...], Tensor], data: TensorLike, **kwargs) -> TensorLike:
    """Operate tensorlike data on current accelerator."""
    device = get_current_accelerator()
    is_tensor = isinstance(data, torch.Tensor)
    is_ndarray = isinstance(data, np.ndarray)

    if is_tensor:
        orig_device = data.device
        data = data.to(device=device)
    elif is_ndarray:
        data = torch.from_numpy(data).to(device=device, dtype=torch.float)
    else:
        data = torch.tensor(data, dtype=torch.float, device=device)

    result = fn(data, **kwargs)

    if is_tensor:
        return result.to(orig_device)
    elif is_ndarray:
        return result.cpu().numpy()
    elif result.numel() == 1:
        return result.item()
    else:
        return result.tolist()


def get_process_group_backend() -> str:
    """Get backend for init process group."""
    if get_current_accelerator().type == DeviceType.NPU:
        return "hccl"
    elif get_current_accelerator().type == DeviceType.CUDA:
        return "nccl"
    elif get_current_accelerator().type == DeviceType.XPU:
        return "xccl"
    else:
        return "gloo"


def all_gather(tensor: Tensor, group: Optional[ProcessGroup] = None) -> Tensor:
    """Gathers the tensor from all ranks and stacks them at the first dim."""
    world_size = get_world_size()
    output_tensor = torch.empty(world_size * tensor.numel(), dtype=tensor.dtype, device=tensor.device)
    dist.all_gather_into_tensor(output_tensor, tensor, group=group)
    return output_tensor.view(-1, *tensor.size())


def all_reduce(tensor: Tensor, op: ReduceOp = ReduceOp.MEAN, group: Optional[ProcessGroup] = None) -> Tensor:
    """Performs all reduce in the given process group."""
    reduce_ops = {
        ReduceOp.MEAN: dist.ReduceOp.SUM,
        ReduceOp.SUM: dist.ReduceOp.SUM,
        ReduceOp.MAX: dist.ReduceOp.MAX,
        ReduceOp.MIN: dist.ReduceOp.MIN,
    }
    dist.all_reduce(tensor, op=reduce_ops[op], group=group)
    if op == ReduceOp.MEAN:  # ReduceOp.AVG is not supported by the NPU backend
        tensor /= dist.get_world_size(group=group)

    return tensor


def broadcast(tensor: Tensor, src: int = 0, group: Optional[ProcessGroup] = None) -> Tensor:
    """Broadcasts the tensor from the src process to all other processes."""
    dist.broadcast(tensor, src=src, group=group)
    return tensor


@contextmanager
def main_process_first(local_only: bool = True) -> None:
    """A context manager for torch distributed environment to do something on the main process firstly."""
    if get_world_size() > 1:
        is_main_process = get_local_rank() == 0 if local_only else get_rank() == 0
        try:
            if not is_main_process:
                dist.barrier()
            yield
        finally:
            if is_main_process:
                dist.barrier()
    else:
        yield
