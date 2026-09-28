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

"""Tests for Ray distributed training utilities.

These tests mock Ray internals so they can run without a real Ray cluster.
"""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_fake_placement_group(bundle_count: int) -> MagicMock:
    """Return a mock PlacementGroup with a given bundle_count."""
    pg = MagicMock()
    pg.bundle_count = bundle_count
    return pg


# ---------------------------------------------------------------------------
# Tests for get_placement_group
# ---------------------------------------------------------------------------


class TestGetPlacementGroup:
    """Verify that get_placement_group creates bundles for *num_workers* only."""

    @patch("llamafactory.train.trainer_utils.placement_group")
    @patch("llamafactory.train.trainer_utils.get_device_name", return_value="gpu")
    def test_bundle_count_equals_num_workers(self, _mock_device, mock_pg_fn):
        """The number of bundles must equal num_workers, not total cluster GPUs."""
        from llamafactory.train.trainer_utils import get_placement_group

        mock_pg_fn.return_value = _make_fake_placement_group(2)

        pg, bundle = get_placement_group(num_workers=2)

        # placement_group() should be called with exactly 2 bundles
        args, kwargs = mock_pg_fn.call_args
        assert len(args[0]) == 2
        assert kwargs.get("strategy") == "PACK"

    @patch("llamafactory.train.trainer_utils.placement_group")
    @patch("llamafactory.train.trainer_utils.get_device_name", return_value="gpu")
    def test_default_cpu_per_bundle_is_one(self, _mock_device, mock_pg_fn):
        """Default CPU per bundle should be 1 (not 10) to avoid scheduling failures."""
        from llamafactory.train.trainer_utils import get_placement_group

        mock_pg_fn.return_value = _make_fake_placement_group(4)

        _, bundle = get_placement_group(num_workers=4)

        assert bundle["CPU"] == 1
        assert bundle["GPU"] == 1

    @patch("llamafactory.train.trainer_utils.placement_group")
    @patch("llamafactory.train.trainer_utils.get_device_name", return_value="gpu")
    def test_custom_cpu_per_bundle(self, _mock_device, mock_pg_fn):
        """Users can override num_cpus_per_worker if they have more CPUs available."""
        from llamafactory.train.trainer_utils import get_placement_group

        mock_pg_fn.return_value = _make_fake_placement_group(2)

        _, bundle = get_placement_group(num_workers=2, num_cpus_per_worker=4)

        assert bundle["CPU"] == 4

    @patch("llamafactory.train.trainer_utils.placement_group")
    @patch("llamafactory.train.trainer_utils.get_device_name", return_value="cpu")
    def test_cpu_only_no_gpu_in_bundle(self, _mock_device, mock_pg_fn):
        """When training on CPU, no GPU key should appear in the bundle."""
        from llamafactory.train.trainer_utils import get_placement_group

        mock_pg_fn.return_value = _make_fake_placement_group(2)

        _, bundle = get_placement_group(num_workers=2)

        assert "GPU" not in bundle
        assert bundle["CPU"] == 1


# ---------------------------------------------------------------------------
# Tests for get_ray_remote_config_for_worker
# ---------------------------------------------------------------------------


class TestGetRayRemoteConfigForWorker:
    """Verify worker remote config correctness."""

    @patch("llamafactory.train.trainer_utils.get_device_name", return_value="gpu")
    @patch("llamafactory.train.trainer_utils.PlacementGroupSchedulingStrategy")
    def test_num_cpus_defaults_to_one(self, _mock_strategy, _mock_device):
        """num_cpus in remote config should default to 1, matching bundle CPU."""
        from llamafactory.train.trainer_utils import get_ray_remote_config_for_worker

        pg = _make_fake_placement_group(2)
        config = get_ray_remote_config_for_worker(
            placement_group=pg,
            bundle_idx=0,
            rank=0,
            world_size=2,
            master_addr="10.0.0.1",
            master_port="29500",
            env={},
        )

        assert config["num_cpus"] == 1

    @patch("llamafactory.train.trainer_utils.get_device_name", return_value="gpu")
    @patch("llamafactory.train.trainer_utils.PlacementGroupSchedulingStrategy")
    def test_custom_num_cpus(self, _mock_strategy, _mock_device):
        """Users can pass a custom num_cpus."""
        from llamafactory.train.trainer_utils import get_ray_remote_config_for_worker

        pg = _make_fake_placement_group(2)
        config = get_ray_remote_config_for_worker(
            placement_group=pg,
            bundle_idx=0,
            rank=0,
            world_size=2,
            master_addr="10.0.0.1",
            master_port="29500",
            num_cpus=8,
            env={},
        )

        assert config["num_cpus"] == 8

    @patch("llamafactory.train.trainer_utils.get_device_name", return_value="gpu")
    @patch("llamafactory.train.trainer_utils.PlacementGroupSchedulingStrategy")
    def test_env_vars_set_correctly(self, _mock_strategy, _mock_device):
        """MASTER_ADDR and RANK should be propagated to env_vars."""
        from llamafactory.train.trainer_utils import get_ray_remote_config_for_worker

        pg = _make_fake_placement_group(4)
        config = get_ray_remote_config_for_worker(
            placement_group=pg,
            bundle_idx=1,
            rank=1,
            world_size=4,
            master_addr="192.168.1.10",
            master_port="12345",
            env={},
        )

        env_vars = config["runtime_env"]["env_vars"]
        assert env_vars["RANK"] == "1"
        assert env_vars["WORLD_SIZE"] == "4"
        assert env_vars["MASTER_ADDR"] == "192.168.1.10"
        assert env_vars["MASTER_PORT"] == "12345"


# ---------------------------------------------------------------------------
# Tests for sort_placement_group_by_node_ip
# ---------------------------------------------------------------------------


class TestSortPlacementGroupByNodeIp:
    """Verify bundle sorting and master_addr fallback logic."""

    @patch("llamafactory.train.trainer_utils.ray")
    @patch("llamafactory.train.trainer_utils.PlacementGroupSchedulingStrategy")
    def test_prefers_master_addr_bundles(self, _mock_strategy, mock_ray):
        """Bundles on master_addr node should come first."""
        from llamafactory.train.trainer_utils import sort_placement_group_by_node_ip

        pg = _make_fake_placement_group(4)
        # Simulate 4 bundles: indices 0,1 on 10.0.0.2; indices 2,3 on 10.0.0.1
        mock_ray.get.return_value = ["10.0.0.2", "10.0.0.2", "10.0.0.1", "10.0.0.1"]

        sorted_indices, effective_addr = sort_placement_group_by_node_ip(pg, master_addr="10.0.0.1")

        # Bundles 2,3 (on master) should come first
        assert sorted_indices[0] in (2, 3)
        assert sorted_indices[1] in (2, 3)
        assert effective_addr == "10.0.0.1"

    @patch("llamafactory.train.trainer_utils.ray")
    @patch("llamafactory.train.trainer_utils.PlacementGroupSchedulingStrategy")
    def test_fallback_when_head_has_no_gpu(self, _mock_strategy, mock_ray):
        """When master_addr has no bundles, fall back to rank 0's IP."""
        from llamafactory.train.trainer_utils import sort_placement_group_by_node_ip

        pg = _make_fake_placement_group(2)
        # Head node (10.0.0.1) has no GPU -> no bundles there
        # All bundles on worker nodes
        mock_ray.get.return_value = ["10.0.0.2", "10.0.0.3"]

        sorted_indices, effective_addr = sort_placement_group_by_node_ip(pg, master_addr="10.0.0.1")

        # master_addr should be updated to rank 0's node IP
        rank0_bundle = sorted_indices[0]
        expected_ip = ["10.0.0.2", "10.0.0.3"][rank0_bundle]
        assert effective_addr == expected_ip
        assert effective_addr != "10.0.0.1"

    @patch("llamafactory.train.trainer_utils.ray")
    @patch("llamafactory.train.trainer_utils.PlacementGroupSchedulingStrategy")
    def test_no_master_addr_returns_sorted(self, _mock_strategy, mock_ray):
        """When master_addr is None, return IP-sorted indices and None."""
        from llamafactory.train.trainer_utils import sort_placement_group_by_node_ip

        pg = _make_fake_placement_group(3)
        mock_ray.get.return_value = ["10.0.0.3", "10.0.0.1", "10.0.0.2"]

        sorted_indices, effective_addr = sort_placement_group_by_node_ip(pg, master_addr=None)

        # Should be sorted by IP: bundle 1 (10.0.0.1), bundle 2 (10.0.0.2), bundle 0 (10.0.0.3)
        assert sorted_indices == [1, 2, 0]
        assert effective_addr is None

    @patch("llamafactory.train.trainer_utils.ray")
    @patch("llamafactory.train.trainer_utils.PlacementGroupSchedulingStrategy")
    def test_single_worker_fallback(self, _mock_strategy, mock_ray):
        """Single worker with head node having no GPU should still work."""
        from llamafactory.train.trainer_utils import sort_placement_group_by_node_ip

        pg = _make_fake_placement_group(1)
        mock_ray.get.return_value = ["10.0.0.5"]

        sorted_indices, effective_addr = sort_placement_group_by_node_ip(pg, master_addr="10.0.0.1")

        assert sorted_indices == [0]
        assert effective_addr == "10.0.0.5"
