# Copyright 2026 the LlamaFactory team.
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
import subprocess
import sys
from unittest.mock import Mock

import pytest

from llamafactory import cli, launcher
from llamafactory.extras import misc


@pytest.fixture(params=["static", "elastic", "elastic_range"])
def torchrun_env(request, monkeypatch):
    for name in ("USE_V1", "USE_MCA", "USE_MEGATRON_BRIDGE", "RDZV_ID", "MIN_NNODES", "MAX_NNODES"):
        monkeypatch.delenv(name, raising=False)

    for name, value in {
        "FORCE_TORCHRUN": "1",
        "NNODES": "2",
        "NODE_RANK": "1",
        "NPROC_PER_NODE": "2",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29501",
        "MAX_RESTARTS": "3",
        "OPTIM_TORCH": "0",
        "LLAMAFACTORY_TEST_ENV": "preserved",
    }.items():
        monkeypatch.setenv(name, value)

    monkeypatch.setattr(misc, "get_device_count", lambda: 2)
    monkeypatch.setattr(misc, "find_available_port", lambda: 29501)
    if request.param == "static":
        return [
            "torchrun",
            "--nnodes",
            "2",
            "--node_rank",
            "1",
            "--nproc_per_node",
            "2",
            "--master_addr",
            "127.0.0.1",
            "--master_port",
            "29501",
        ]

    monkeypatch.setenv("RDZV_ID", "test-job")
    if request.param == "elastic_range":
        monkeypatch.setenv("MIN_NNODES", "2")
        monkeypatch.setenv("MAX_NNODES", "4")

    return [
        "torchrun",
        "--nnodes",
        "2:4" if request.param == "elastic_range" else "2",
        "--nproc-per-node",
        "2",
        "--rdzv-id",
        "test-job",
        "--rdzv-backend",
        "c10d",
        "--rdzv-endpoint",
        "127.0.0.1:29501",
        "--max-restarts",
        "3",
    ]


@pytest.mark.parametrize(
    "train_args,spaced_checkout",
    [
        pytest.param(["config.yaml"], False, id="plain"),
        pytest.param(["experiment files/train config.yaml"], False, id="config-path"),
        pytest.param(["config.yaml", "output_dir=training outputs/run 1"], False, id="override"),
        pytest.param(["--run_name", 'a "quoted" run', "--output_dir", r"C:\training\runs"], False, id="literal-args"),
        pytest.param(["--run_name", ""], False, id="empty-argument"),
        pytest.param(["config.yaml"], True, id="checkout-path"),
    ],
)
def test_torchrun_preserves_arguments(torchrun_env, monkeypatch, tmp_path, train_args, spaced_checkout):
    checkout = "checkout with spaces" if spaced_checkout else "checkout"
    launcher_path = str(tmp_path / checkout / "launcher.py")
    monkeypatch.setattr(launcher, "__file__", launcher_path)
    monkeypatch.setattr(sys, "argv", ["llamafactory-cli", "train", *train_args])
    run = Mock(return_value=subprocess.CompletedProcess(args=[], returncode=0))
    monkeypatch.setattr(launcher.subprocess, "run", run)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    run.assert_called_once()
    assert run.call_args.args == (torchrun_env + [launcher_path, *train_args],)
    assert set(run.call_args.kwargs) == {"env", "check"}
    assert run.call_args.kwargs["check"] is True
    assert run.call_args.kwargs["env"]["LLAMAFACTORY_TEST_ENV"] == "preserved"
    assert run.call_args.kwargs["env"] is not os.environ


@pytest.mark.parametrize("optimize", [False, True])
def test_torchrun_preserves_child_environment(torchrun_env, monkeypatch, optimize):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    monkeypatch.setenv("TORCH_NCCL_AVOID_RECORD_STREAMS", "0")
    if optimize:
        monkeypatch.delenv("OPTIM_TORCH")  # Optimization is enabled by default.

    monkeypatch.setattr(sys, "argv", ["llamafactory-cli", "train", "config.yaml"])
    run = Mock(return_value=subprocess.CompletedProcess(args=[], returncode=0))
    monkeypatch.setattr(launcher.subprocess, "run", run)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    child_env = run.call_args.kwargs["env"]
    assert child_env["PYTORCH_CUDA_ALLOC_CONF"] == (
        "expandable_segments:True" if optimize else "max_split_size_mb:128"
    )
    assert child_env["TORCH_NCCL_AVOID_RECORD_STREAMS"] == ("1" if optimize else "0")
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"
    assert os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] == "0"


def test_torchrun_propagates_subprocess_failure(torchrun_env, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["llamafactory-cli", "train", "config.yaml"])
    error = subprocess.CalledProcessError(7, ["torchrun"])
    monkeypatch.setattr(launcher.subprocess, "run", Mock(side_effect=error))

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        cli.main()

    assert exc_info.value is error
