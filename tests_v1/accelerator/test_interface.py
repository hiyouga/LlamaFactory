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

import os

import pytest
import torch.multiprocessing as mp

from llamafactory.v1.accelerator.helper import ReduceOp
from llamafactory.v1.accelerator.interface import DistributedInterface
from llamafactory.v1.utils.env import find_available_port
from llamafactory.v1.utils.pytest import dist_env


def _all_reduce_tests(local_rank: int, world_size: int, master_port: int):
    with dist_env(local_rank, world_size, master_port):
        rank = DistributedInterface().get_rank()
        world_size = DistributedInterface().get_world_size()
        assert world_size == 2

        y_sum = DistributedInterface().all_reduce(rank + 1.0, op=ReduceOp.SUM)
        assert y_sum == pytest.approx(3.0)

        y_mean = DistributedInterface().all_reduce(rank + 1.0, op=ReduceOp.MEAN)
        assert y_mean == pytest.approx(1.5)

        y_max = DistributedInterface().all_reduce(rank + 1.0, op=ReduceOp.MAX)
        assert y_max == pytest.approx(2.0)

        z = DistributedInterface().all_gather(rank + 1.0)
        assert z == pytest.approx([1.0, 2.0])

        z = DistributedInterface().broadcast(rank + 1.0)
        assert z == pytest.approx(1.0)


def test_all_device():
    assert DistributedInterface().get_rank() == int(os.getenv("RANK", "0"))
    assert DistributedInterface().get_world_size() == int(os.getenv("WORLD_SIZE", "1"))
    assert DistributedInterface().get_local_rank() == int(os.getenv("LOCAL_RANK", "0"))
    assert DistributedInterface().get_local_world_size() == int(os.getenv("LOCAL_WORLD_SIZE", "1"))


@pytest.mark.runs_on(["cuda", "npu", "xpu"])
@pytest.mark.require_distributed(2)
def test_multi_device():
    master_port = find_available_port()
    world_size = 2
    mp.spawn(_all_reduce_tests, args=(world_size, master_port), nprocs=world_size)
