# Copyright 2025 Bytedance Ltd. and the LlamaFactory team.
#
# This code is inspired by the Bytedance's VeOmni library.
# https://github.com/ByteDance-Seed/VeOmni/blob/v0.1.4/veomni/distributed/parallel_state.py
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

"""A unified interface for model parallelism and data parallelism.

Supports model parallelism types:
- mp_replicate: Replicate model across multiple devices.
- mp_shard: Shard model across multiple devices.

And data parallelism types:
- dp: Data parallelism.
- cp: Context parallelism.
"""

from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Optional

from torch.distributed import barrier, destroy_process_group, init_process_group
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from ..utils import logging
from ..utils.types import ProcessGroup, TensorLike
from . import helper


if TYPE_CHECKING:
    from ..config.training_args import TrainingArguments


logger = logging.get_logger(__name__)


class Dim(StrEnum):
    """Dimension names."""

    MP_REPLICATE = "mp_replicate"
    MP_SHARD = "mp_shard"
    DP = "dp"
    CP = "cp"


@dataclass
class DistributedStrategy:
    """Distributed strategy."""

    mp_replicate_size: int = 1
    """Model parallel replicate size, default to 1."""
    mp_shard_size: int | None = None
    """Model parallel shard size, default to world_size // mp_replicate_size."""
    dp_size: int | None = None
    """Data parallel size, default to world_size // cp_size."""
    cp_size: int = 1
    """Context parallel size, default to 1."""

    def __post_init__(self) -> None:
        if not helper.is_distributed():
            self.mp_shard_size = 1
        elif self.mp_shard_size is None:
            if helper.get_world_size() % self.mp_replicate_size != 0:
                raise ValueError(
                    f"world_size ({helper.get_world_size()}) must be divisible by "
                    f"mp_replicate_size ({self.mp_replicate_size})."
                )
            self.mp_shard_size = helper.get_world_size() // self.mp_replicate_size
        elif self.mp_replicate_size * self.mp_shard_size != helper.get_world_size():
            raise ValueError(
                f"mp_replicate_size * mp_shard_size must equal to world_size, "
                f"got {self.mp_replicate_size} * {self.mp_shard_size} != {helper.get_world_size()}."
            )

        if not helper.is_distributed():
            self.dp_size = 1
        elif self.dp_size is None:
            if helper.get_world_size() % self.cp_size != 0:
                raise ValueError(
                    f"world_size ({helper.get_world_size()}) must be divisible by cp_size ({self.cp_size})."
                )
            self.dp_size = helper.get_world_size() // self.cp_size
        elif self.dp_size * self.cp_size != helper.get_world_size():
            raise ValueError(
                f"dp_size * cp_size must equal to world_size, "
                f"got {self.dp_size} * {self.cp_size} != {helper.get_world_size()}."
            )

    @property
    def model_mesh_shape(self) -> tuple[int, int]:
        """Model parallel mesh shape."""
        return (self.mp_replicate_size, self.mp_shard_size)

    @property
    def model_mesh_dim_names(self) -> tuple[str, str]:
        """Model parallel mesh dimension names."""
        return (Dim.MP_REPLICATE.value, Dim.MP_SHARD.value)

    @property
    def data_mesh_shape(self) -> tuple[int, int]:
        """Data parallel mesh shape."""
        return (self.dp_size, self.cp_size)

    @property
    def data_mesh_dim_names(self) -> tuple[str, str]:
        """Data parallel mesh dimension names."""
        return (Dim.DP.value, Dim.CP.value)


class DistributedInterface:
    """Distributed interface."""

    _instance: Optional["DistributedInterface"] = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "DistributedInterface":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(
        self,
        training_args: "TrainingArguments | None" = None,
    ) -> None:
        if self._initialized:
            return

        helper.set_device_index()
        self._is_distributed = helper.is_distributed()
        self._rank = helper.get_rank()
        self._world_size = helper.get_world_size()
        self._local_rank = helper.get_local_rank()
        self._local_world_size = helper.get_local_world_size()
        self.current_device = helper.get_current_device()
        self.device_count = helper.get_device_count()

        if training_args is None:
            self.strategy = DistributedStrategy()
            timeout = 18000
        else:
            self.strategy = DistributedStrategy(
                mp_replicate_size=training_args.mp_replicate_size,
                mp_shard_size=training_args.mp_shard_size,
                dp_size=training_args.dp_size,
                cp_size=training_args.cp_size,
            )
            timeout = training_args.dist_timeout

        if self._is_distributed:
            init_process_group(timeout=timedelta(seconds=timeout), backend=helper.get_process_group_backend())
            self.model_device_mesh = init_device_mesh(
                device_type=self.current_device.type,
                mesh_shape=self.strategy.model_mesh_shape,
                mesh_dim_names=self.strategy.model_mesh_dim_names,
            )
            self.data_device_mesh = init_device_mesh(
                device_type=self.current_device.type,
                mesh_shape=self.strategy.data_mesh_shape,
                mesh_dim_names=self.strategy.data_mesh_dim_names,
            )
        else:
            self.model_device_mesh = None
            self.data_device_mesh = None

        self._initialized = True
        logger.info_rank0(f"DistributedInterface initialized: {self}.")

    def __str__(self) -> str:
        return (
            f"DistributedInterface(strategy={self.strategy}), is_distributed={self._is_distributed}, "
            f"current_device={self.current_device}, rank={self._rank}, world_size={self._world_size}, "
            f"model_device_mesh={self.model_device_mesh}, data_device_mesh={self.data_device_mesh}"
        )

    def get_device_mesh(self, dim: Dim | None = None) -> DeviceMesh | None:
        """Get device mesh for specified dimension."""
        if dim is None:
            raise ValueError("dim must be specified.")
        elif not self._is_distributed:
            return None
        elif dim in self.strategy.data_mesh_dim_names:
            return self.data_device_mesh[dim.value]
        else:
            return self.model_device_mesh[dim.value]

    def get_group(self, dim: Dim | None = None) -> Optional[ProcessGroup]:
        """Get process group for specified dimension."""
        if not self._is_distributed or dim is None:
            return None
        else:
            return self.get_device_mesh(dim).get_group()

    def get_rank(self, dim: Dim | None = None) -> int:
        """Get parallel rank for specified dimension."""
        if not self._is_distributed:
            return 0
        elif dim is None:
            return self._rank
        else:
            return self.get_device_mesh(dim).get_local_rank()

    def get_world_size(self, dim: Dim | None = None) -> int:
        """Get parallel size for specified dimension."""
        if not self._is_distributed:
            return 1
        elif dim is None:
            return self._world_size
        else:
            return self.get_device_mesh(dim).size()

    def get_local_rank(self) -> int:
        """Get parallel local rank."""
        return self._local_rank

    def get_local_world_size(self) -> int:
        """Get parallel local world size."""
        return self._local_world_size

    def all_gather(self, data: TensorLike, dim: Dim | None = Dim.DP) -> TensorLike:
        """Gather tensor across specified parallel group."""
        if self._is_distributed:
            return helper.operate_tensorlike(helper.all_gather, data, group=self.get_group(dim))
        else:
            return data

    def all_reduce(
        self, data: TensorLike, op: helper.ReduceOp = helper.ReduceOp.MEAN, dim: Dim | None = Dim.DP
    ) -> TensorLike:
        """Reduce tensor across specified parallel group."""
        if self._is_distributed:
            return helper.operate_tensorlike(helper.all_reduce, data, op=op, group=self.get_group(dim))
        else:
            return data

    def broadcast(self, data: TensorLike, src: int = 0, dim: Dim | None = Dim.DP) -> TensorLike:
        """Broadcast tensor across specified parallel group."""
        if self._is_distributed:
            return helper.operate_tensorlike(helper.broadcast, data, src=src, group=self.get_group(dim))
        else:
            return data

    def sync(self) -> None:
        """Synchronize all processes."""
        if self._is_distributed:
            helper.synchronize()

    def barrier(self) -> None:
        """Barrier all processes."""
        if self._is_distributed:
            barrier()

    def destroy(self) -> None:
        """Destroy all processes."""
        if self._is_distributed:
            destroy_process_group()


if __name__ == "__main__":
    """
    python -m llamafactory.v1.accelerator.interface
    """
    print(DistributedInterface())
