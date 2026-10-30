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

import json
import os
from pathlib import Path

import pytest
from transformers import TrainerControl, TrainerState, TrainingArguments

from llamafactory.extras.constants import TRAINER_LOG
from llamafactory.train.callbacks import LogCallback


ELAPSED_SECONDS = 20.0

NEW_STEPS = 10

NEW_TOKENS = 20000


def _drive_log_callback(output_dir: Path, resumed_steps: int, resumed_tokens: int) -> dict[str, float]:
    r"""Run one logging round after `ELAPSED_SECONDS` of training and return the written log entry."""
    args = TrainingArguments(output_dir=str(output_dir), report_to=[])
    state = TrainerState()
    state.global_step = resumed_steps
    state.max_steps = 2000
    state.num_input_tokens_seen = resumed_tokens
    state.log_history = [{"loss": 1.0, "learning_rate": 5e-5, "epoch": 1.0}]

    callback = LogCallback()
    callback.on_train_begin(args, state, TrainerControl())
    callback.start_time -= ELAPSED_SECONDS  # avoid sleeping for real

    state.global_step += NEW_STEPS
    state.num_input_tokens_seen += NEW_TOKENS
    callback.on_log(args, state, TrainerControl())
    callback.on_train_end(args, state, TrainerControl())

    with open(os.path.join(output_dir, TRAINER_LOG), encoding="utf-8") as f:
        return json.loads(f.read().splitlines()[-1])


def _to_seconds(formatted: str) -> int:
    hours, minutes, seconds = (int(part) for part in formatted.split(":"))
    return hours * 3600 + minutes * 60 + seconds


@pytest.mark.parametrize(("resumed_steps", "resumed_tokens"), [(0, 0), (1000, 10_000_000)])
def test_throughput_counts_only_tokens_seen_since_the_resume_point(
    tmp_path: Path, resumed_steps: int, resumed_tokens: int
):
    """Transformers restores `num_input_tokens_seen` before `on_train_begin` fires.

    Dividing that whole-run total by the time since the resume inflates throughput by the ratio between
    the two, so a run resumed near its end reports a throughput orders of magnitude too high.
    """
    logs = _drive_log_callback(tmp_path, resumed_steps, resumed_tokens)

    assert logs["throughput"] == pytest.approx(NEW_TOKENS / ELAPSED_SECONDS, rel=0.05)
    assert logs["total_tokens"] == resumed_tokens + NEW_TOKENS


@pytest.mark.parametrize(("resumed_steps", "resumed_tokens"), [(0, 0), (1000, 10_000_000)])
def test_remaining_time_counts_only_steps_taken_since_the_resume_point(
    tmp_path: Path, resumed_steps: int, resumed_tokens: int
):
    """`global_step` is restored the same way, so averaging over it understates the seconds per step."""
    logs = _drive_log_callback(tmp_path, resumed_steps, resumed_tokens)

    remaining_steps = 2000 - (resumed_steps + NEW_STEPS)
    assert _to_seconds(logs["remaining_time"]) == pytest.approx(remaining_steps * ELAPSED_SECONDS / NEW_STEPS, abs=10)
