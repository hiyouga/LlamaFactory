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

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from .....accelerator.interface import Dim, DistributedInterface


class FSDPTurboParallelState:
    """Own FSDPTurbo's expert topology without extending LlamaFactory's global interface."""

    EDP = "edp"
    EFSDP = "efsdp"
    EP = "ep"
    EXPERT_CP = "expert_cp"

    def __init__(self) -> None:
        self._initialized = False
        self.dp_size = 1
        self.cp_size = 1
        self.ep_size = 1
        self.efsdp_size = 1
        self.edp_size = 1
        self.expert_mesh: DeviceMesh | None = None
        self.edp_mesh: DeviceMesh | None = None
        self.efsdp_mesh: DeviceMesh | None = None
        self.ep_mesh: DeviceMesh | None = None
        self.expert_cp_mesh: DeviceMesh | None = None

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self, dist_interface: DistributedInterface, dist_config: dict) -> None:
        dp_size = dist_interface.get_world_size(Dim.DP)
        cp_size = dist_interface.strategy.cp_size
        ep_size = int(dist_config.get("ep_size", 1))

        if ep_size < 1:
            raise ValueError(f"ep_size must be positive, got {ep_size}.")
        if dp_size % ep_size != 0:
            raise ValueError(f"dp_size must be divisible by ep_size, got {dp_size} % {ep_size} != 0.")

        topology = (dp_size, cp_size, ep_size)
        if self._initialized:
            current_topology = (self.dp_size, self.cp_size, self.ep_size)
            if topology != current_topology:
                raise RuntimeError(
                    f"FSDPTurbo parallel state is already initialized with {current_topology}, got {topology}."
                )
            return

        self.dp_size = dp_size
        self.cp_size = cp_size
        self.ep_size = ep_size

        if ep_size > 1:
            self.efsdp_size = dp_size // ep_size
            self.edp_size = dp_size // (ep_size * self.efsdp_size)
            if dist_interface.get_device_mesh(Dim.DP) is None:
                raise RuntimeError("FSDPTurbo expert parallelism requires an initialized distributed device mesh.")

            self.expert_mesh = init_device_mesh(
                device_type=dist_interface.current_device.type,
                mesh_shape=(self.edp_size, self.efsdp_size, self.ep_size, self.cp_size),
                mesh_dim_names=(self.EDP, self.EFSDP, self.EP, self.EXPERT_CP),
            )
            self.edp_mesh = self.expert_mesh[self.EDP]
            self.efsdp_mesh = self.expert_mesh[self.EFSDP]
            self.ep_mesh = self.expert_mesh[self.EP]
            self.expert_cp_mesh = self.expert_mesh[self.EXPERT_CP]

        self._initialized = True


_FSDPTURBO_PARALLEL_STATE = FSDPTurboParallelState()


def get_fsdpturbo_parallel_state() -> FSDPTurboParallelState:
    return _FSDPTURBO_PARALLEL_STATE
