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

import pytest

from llamafactory.v1.core.utils.train_timer import TrainTimer, format_duration


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_format_duration():
    assert format_duration(0) == "0:00:00"
    assert format_duration(59.9) == "0:00:59"
    assert format_duration(3725) == "1:02:05"
    assert format_duration(-3) == "0:00:00"


def test_log_metrics_average_over_logging_interval():
    clock = FakeClock()
    timer = TrainTimer(start_step=0, total_steps=10, clock=clock)

    # Two steps of 1 s and 3 s, 100 and 300 tokens.
    clock.advance(1.0)
    timer.update(num_tokens=100, loss=2.0)
    clock.advance(3.0)
    timer.update(num_tokens=300, loss=4.0)
    metrics = timer.log_metrics(global_step=2)
    assert metrics == {
        "elapsed_time": "0:00:04",
        "remaining_time": "0:00:16",  # 8 steps left at 2 s per step
        "step_time": 2.0,
        "tokens_per_second": 100.0,
    }

    # The next interval only covers the steps after the previous log entry.
    clock.advance(6.0)
    timer.update(num_tokens=60, loss=1.0)
    metrics = timer.log_metrics(global_step=3)
    assert metrics["step_time"] == 6.0
    assert metrics["tokens_per_second"] == 10.0
    assert metrics["elapsed_time"] == "0:00:10"
    assert metrics["remaining_time"] == "0:00:23"  # 7 steps left at 10/3 s per step


def test_remaining_time_after_resume_uses_resumed_steps_only():
    clock = FakeClock()
    timer = TrainTimer(start_step=90, total_steps=100, clock=clock)
    clock.advance(5.0)
    timer.update(num_tokens=10, loss=1.0)
    metrics = timer.log_metrics(global_step=91)
    assert metrics["remaining_time"] == "0:00:45"  # 9 steps left at 5 s per step


def test_summary():
    clock = FakeClock()
    timer = TrainTimer(start_step=0, total_steps=4, clock=clock)
    for loss in (4.0, 3.0, 2.0, 1.0):
        clock.advance(0.5)
        timer.update(num_tokens=50, loss=loss)

    assert timer.summary(epoch=1, global_step=4) == {
        "epoch": 1,
        "global_step": 4,
        "train_loss": pytest.approx(2.5),
        "train_runtime": 2.0,
        "train_steps_per_second": 2.0,
        "train_tokens": 200,
        "train_tokens_per_second": 100.0,
    }


def test_no_steps_does_not_divide_by_zero():
    timer = TrainTimer(start_step=0, total_steps=4, clock=FakeClock())
    assert timer.log_metrics(global_step=0)["step_time"] == 0.0
    summary = timer.summary(epoch=0, global_step=0)
    assert summary["train_loss"] == 0.0
    assert summary["train_steps_per_second"] == 0.0
