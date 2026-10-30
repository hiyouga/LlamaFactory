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

import pytest

from llamafactory.train import trainer_utils
from llamafactory.train.trainer_utils import get_placement_group, get_ray_resource_name


# (pytorch device name from get_device_name) -> (expected Ray resource name)
# Ray's accelerator managers report both CUDA and Intel XPU under the generic
# "GPU" resource, so XPU must map to "GPU" (not a custom "XPU" resource that no
# cluster declares by default). NPU keeps its own "NPU" resource.
@pytest.mark.parametrize(
    ("device_name", "expected_resource"),
    [
        ("gpu", "GPU"),
        ("xpu", "GPU"),
        ("npu", "NPU"),
        ("cpu", "CPU"),
    ],
)
def test_ray_resource_name_mapping(monkeypatch, device_name, expected_resource):
    monkeypatch.setattr(trainer_utils, "get_device_name", lambda: device_name)
    assert get_ray_resource_name() == expected_resource


@pytest.mark.parametrize("device_name", ["gpu", "xpu"])
@pytest.mark.skipif(not trainer_utils.is_ray_available(), reason="Ray is not installed")
def test_gpu_and_xpu_share_gpu_bundle(monkeypatch, device_name):
    """GPU and XPU must request the same "GPU" placement-group bundle resource."""
    monkeypatch.setattr(trainer_utils, "get_device_name", lambda: device_name)
    # Stub out the real ray.util.placement_group call: it needs a connected Ray
    # runtime and raises otherwise, so building a real placement group here would
    # pass in CI only because Ray isn't installed there, and fail for anyone who
    # has Ray installed locally but not connected to a cluster.
    monkeypatch.setattr(trainer_utils, "placement_group", lambda bundles, strategy: None)
    _, bundle = get_placement_group(num_workers=1)
    assert bundle.get("GPU") == 1
    assert "XPU" not in bundle
