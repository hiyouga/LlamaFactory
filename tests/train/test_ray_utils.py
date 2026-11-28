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

"""Unit tests for the Ray placement group helpers.

These tests stub out the Ray entry points so that they neither require a Ray installation nor a
running cluster, keeping the scheduling logic itself under test.
"""

import pytest
from pytest import MonkeyPatch

from llamafactory.extras.constants import DEFAULT_RAY_NUM_CPUS_PER_WORKER
from llamafactory.train import trainer_utils
from llamafactory.train.trainer_utils import (
    get_placement_group,
    get_ray_remote_config_for_worker,
    resolve_master_addr,
    sort_placement_group_by_node_ip,
)


HEAD_IP = "10.0.0.1"
WORKER_IP = "10.0.0.2"


@pytest.fixture
def captured_bundles(monkeypatch: MonkeyPatch) -> list[list[dict[str, int]]]:
    """Capture the bundles handed to `ray.util.placement_group` without touching Ray."""
    captured = []

    def _placement_group(bundles, strategy=None, **kwargs):
        captured.append(bundles)
        return object()

    monkeypatch.setattr(trainer_utils, "placement_group", _placement_group, raising=False)
    monkeypatch.setattr(trainer_utils, "get_device_name", lambda: "gpu")
    return captured


def test_placement_group_reserves_one_bundle_per_worker(captured_bundles):
    """`num_workers` bundles are reserved, not one per device in the whole cluster."""
    _, bundle = get_placement_group(num_workers=2)

    assert len(captured_bundles[0]) == 2
    assert bundle["GPU"] == 1


def test_placement_group_cpu_reservation_defaults_to_ten(captured_bundles):
    """The default reservation stays at 10 CPUs so existing clusters keep their placement."""
    _, bundle = get_placement_group(num_workers=2)

    assert DEFAULT_RAY_NUM_CPUS_PER_WORKER == 10
    assert bundle["CPU"] == 10


def test_placement_group_cpu_reservation_is_configurable(captured_bundles):
    """CPU constrained nodes can lower the reservation without patching the source."""
    _, bundle = get_placement_group(num_workers=2, num_cpus_per_worker=1)

    assert bundle["CPU"] == 1
    assert captured_bundles[0][0]["CPU"] == 1


def test_remote_config_cpus_match_the_bundle(monkeypatch: MonkeyPatch):
    """A worker must not request more CPUs than its bundle reserved."""
    monkeypatch.setattr(trainer_utils, "PlacementGroupSchedulingStrategy", lambda **kwargs: None, raising=False)
    monkeypatch.setattr(trainer_utils, "get_device_name", lambda: "gpu")

    remote_config = get_ray_remote_config_for_worker(
        placement_group=object(),
        bundle_idx=0,
        rank=0,
        world_size=2,
        master_addr=HEAD_IP,
        master_port="12345",
        env={},
        num_cpus=1,
    )

    assert remote_config["num_cpus"] == 1
    assert remote_config["num_gpus"] == 1
    assert remote_config["runtime_env"]["env_vars"]["MASTER_ADDR"] == HEAD_IP


def test_sort_moves_master_bundles_to_the_front():
    """Bundles on the master node come first so that rank 0 runs there."""
    bundle_node_ips = [WORKER_IP, HEAD_IP, WORKER_IP, HEAD_IP]

    sorted_bundle_indices = sort_placement_group_by_node_ip(
        placement_group=None, master_addr=HEAD_IP, bundle_node_ips=bundle_node_ips
    )

    assert sorted_bundle_indices[:2] == [1, 3]
    assert sorted(sorted_bundle_indices) == [0, 1, 2, 3]


def test_sort_without_any_bundle_on_the_master_node():
    """A device-less head node leaves no preferred bundle, the ip ordering is kept as is."""
    bundle_node_ips = [WORKER_IP, "10.0.0.3"]

    sorted_bundle_indices = sort_placement_group_by_node_ip(
        placement_group=None, master_addr=HEAD_IP, bundle_node_ips=bundle_node_ips
    )

    assert sorted_bundle_indices == [0, 1]


def test_sort_is_stable_for_bundles_sharing_a_node():
    """Duplicated ips must not drop or duplicate bundle indices."""
    bundle_node_ips = [HEAD_IP, HEAD_IP, WORKER_IP]

    sorted_bundle_indices = sort_placement_group_by_node_ip(
        placement_group=None, master_addr=HEAD_IP, bundle_node_ips=bundle_node_ips
    )

    assert sorted_bundle_indices == [0, 1, 2]


def test_master_addr_is_kept_when_rank0_lands_on_it():
    bundle_node_ips = [WORKER_IP, HEAD_IP]
    sorted_bundle_indices = [1, 0]

    assert resolve_master_addr(HEAD_IP, bundle_node_ips, sorted_bundle_indices) == HEAD_IP


def test_master_addr_falls_back_to_the_rank0_node():
    """Without this fallback rank 0 serves the TCPStore on a node nobody connects to."""
    bundle_node_ips = [WORKER_IP, "10.0.0.3"]
    sorted_bundle_indices = sort_placement_group_by_node_ip(
        placement_group=None, master_addr=HEAD_IP, bundle_node_ips=bundle_node_ips
    )

    assert resolve_master_addr(HEAD_IP, bundle_node_ips, sorted_bundle_indices) == WORKER_IP
