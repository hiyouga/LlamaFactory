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

"""LlamaFactory test configuration.

Contains shared fixtures, pytest configuration, and custom markers.
"""

import os

import pytest
import torch
import torch.distributed as dist
from pytest import Config, FixtureRequest, Item, MonkeyPatch

from llamafactory.extras.misc import get_current_device, get_device_count, is_env_enabled
from llamafactory.extras.packages import is_transformers_version_greater_than
from llamafactory.train.test_utils import patch_valuehead_model


CURRENT_DEVICE = get_current_device().type


def pytest_configure(config: Config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"' or set RUN_SLOW=1 to run)",
    )
    config.addinivalue_line(
        "markers",
        "runs_on: test requires specific device type, e.g., @pytest.mark.runs_on(['cuda'])",
    )
    config.addinivalue_line(
        "markers",
        "require_distributed(num_devices): allow multi-device execution (default: 2)",
    )


def _handle_runs_on(items: list[Item]):
    """Skip tests on specified device TYPES (cpu/cuda/npu)."""
    for item in items:
        marker = item.get_closest_marker("runs_on")
        if not marker:
            continue

        devices = marker.args[0]
        if isinstance(devices, str):
            devices = [devices]

        if CURRENT_DEVICE not in devices:
            item.add_marker(pytest.mark.skip(reason=f"test requires one of {devices} (current: {CURRENT_DEVICE})"))


def _handle_slow_tests(items: list[Item]):
    """Skip slow tests unless RUN_SLOW is enabled."""
    if not is_env_enabled("RUN_SLOW"):
        skip_slow = pytest.mark.skip(reason="slow test (set RUN_SLOW=1 to run)")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def _get_visible_devices_env() -> str | None:
    """Return device visibility env var name."""
    if CURRENT_DEVICE == "cuda":
        return "CUDA_VISIBLE_DEVICES"
    elif CURRENT_DEVICE == "npu":
        return "ASCEND_RT_VISIBLE_DEVICES"
    elif CURRENT_DEVICE == "xpu":
        return "ZE_AFFINITY_MASK"
    else:
        return None


def _handle_device_visibility(items: list[Item]):
    """Handle device visibility based on test markers."""
    env_key = _get_visible_devices_env()
    if env_key is None or CURRENT_DEVICE in ("cpu", "mps"):
        return

    # Parse visible devices
    visible_devices_env = os.environ.get(env_key)
    if visible_devices_env is None:
        available = get_device_count()
    else:
        visible_devices = [v for v in visible_devices_env.split(",") if v != ""]
        available = len(visible_devices)

    for item in items:
        marker = item.get_closest_marker("require_distributed")
        if not marker:
            continue

        required = marker.args[0] if marker.args else 2
        if available < required:
            item.add_marker(pytest.mark.skip(reason=f"test requires {required} devices, but only {available} visible"))


def pytest_collection_modifyitems(config: Config, items: list[Item]):
    """Modify test collection based on markers and environment."""
    # Handle version compatibility (from HEAD)
    skip_bc = pytest.mark.skip(reason="Skip backward compatibility tests")
    for item in items:
        if "tests_v1" in str(item.fspath) and not is_transformers_version_greater_than("4.57.0"):
            item.add_marker(skip_bc)

    _handle_slow_tests(items)
    _handle_runs_on(items)
    _handle_device_visibility(items)


@pytest.fixture(autouse=True)
def _cleanup_distributed_state():
    """Cleanup distributed state after each test."""
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(autouse=True)
def _manage_distributed_env(request: FixtureRequest, monkeypatch: MonkeyPatch) -> None:
    """Set environment variables for distributed tests if specific devices are requested."""
    env_key = _get_visible_devices_env()
    if not env_key:
        return

    # Save old environment for logic checks, monkeypatch handles restoration
    old_value = os.environ.get(env_key)

    marker = request.node.get_closest_marker("require_distributed")
    if marker:  # distributed test
        required = marker.args[0] if marker.args else 2
        specific_devices = marker.args[1] if len(marker.args) > 1 else None

        if specific_devices:
            devices_str = ",".join(map(str, specific_devices))
        else:
            devices_str = ",".join(str(i) for i in range(required))

        monkeypatch.setenv(env_key, devices_str)
        monkeypatch.syspath_prepend(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    else:  # non-distributed test
        if old_value:
            visible_devices = [v for v in old_value.split(",") if v != ""]
            monkeypatch.setenv(env_key, visible_devices[0] if visible_devices else "0")
        else:
            monkeypatch.setenv(env_key, "0")

        if CURRENT_DEVICE == "cuda":
            monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        elif CURRENT_DEVICE == "npu":
            monkeypatch.setattr(torch.npu, "device_count", lambda: 1)
        elif CURRENT_DEVICE == "xpu":
            monkeypatch.setattr(torch.xpu, "device_count", lambda: 1)


@pytest.fixture
def fix_valuehead_cpu_loading():
    """Fix valuehead model loading."""
    patch_valuehead_model()


@pytest.fixture(scope="session", autouse=True)
def bypass_mistral_regex_check():
    """Disable Mistral regex network check.

    Monkey-patch TokenizersBackend._patch_mistral_regex into a no-op.
    """
    try:
        from transformers.tokenization_utils_fast import TokenizersBackend
    except ImportError:
        # Very old transformers, nothing to patch
        yield
        return

    if not hasattr(TokenizersBackend, "_patch_mistral_regex"):
        # Method does not exist in this version
        yield
        return

    # Backup original method
    original = TokenizersBackend._patch_mistral_regex

    # Replace with no-op
    TokenizersBackend._patch_mistral_regex = lambda cls, tokenizer, *args, **kwargs: tokenizer

    yield

    # Restore original method
    TokenizersBackend._patch_mistral_regex = original
