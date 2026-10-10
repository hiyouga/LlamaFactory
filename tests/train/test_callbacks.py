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

import glob
import json
import os
from types import SimpleNamespace

import pytest
import torch

from llamafactory.extras.env import print_env
from llamafactory.extras.misc import get_current_device
from llamafactory.train.callbacks import ModuleProfilerCallback, TorchProfilerCallback


@pytest.mark.runs_on(["cuda", "npu", "xpu"])
def test_print_env(capsys: pytest.CaptureFixture):
    device_label = {"cuda": "GPU", "npu": "NPU", "xpu": "XPU"}[get_current_device().type]
    print_env()
    assert f"({device_label})" in capsys.readouterr().out


@pytest.mark.runs_on(["cuda", "npu", "xpu"])
def test_module_profiler_callback():
    device = get_current_device()
    model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16)).to(device)
    callback = ModuleProfilerCallback("0,2")
    assert callback.enabled

    callback.on_train_begin(None, None, None, model=model)
    model(torch.randn(4, 16, device=device, requires_grad=True)).sum().backward()
    callback.on_step_end(None, None, None)
    callback.on_train_end(None, None, None)

    assert set(callback._forward_times) == {"0", "2"}
    assert set(callback._backward_times) == {"0", "2"}


@pytest.mark.runs_on(["cuda", "xpu"])
def test_torch_profiler_callback(tmp_path: str):
    device = get_current_device()
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        profiler_output_dir=None,
        profiler_wait_steps=0,
        profiler_warmup_steps=1,
        profiler_active_steps=1,
        profiler_repeat=1,
        profiler_record_shapes=False,
        profiler_profile_memory=False,
        profiler_with_stack=False,
    )
    callback = TorchProfilerCallback(args)
    callback.on_train_begin(args, None, None)
    x = torch.randn(64, 64, device=device)
    for _ in range(3):
        (x @ x).sum().item()
        callback.on_step_end(args, None, None)

    callback.on_train_end(args, None, None)

    traces = glob.glob(os.path.join(tmp_path, "profiler", "rank_0", "*.pt.trace.json"))
    assert len(traces) == 1
    with open(traces[0]) as f:
        events = json.load(f)["traceEvents"]

    assert any(event.get("cat") == "kernel" for event in events)  # device kernels are recorded
