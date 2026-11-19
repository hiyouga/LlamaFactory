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

"""Wall-clock timing and throughput metrics for the v1 training loop."""

import time
from collections.abc import Callable
from datetime import timedelta
from typing import Any


def format_duration(seconds: float) -> str:
    """Format seconds as ``H:MM:SS``, the same format the v0 trainer log uses."""
    return str(timedelta(seconds=int(max(seconds, 0))))


class TrainTimer:
    """Tracks elapsed time, trained tokens and loss over one ``fit()`` call.

    A step is one optimizer step: with gradient accumulation it covers every
    micro-batch of that step. Token counts are the valid (non-padding, non-ignored)
    label tokens of the step summed over data-parallel ranks, so throughput is
    global, not per device.

    Only steps run in this process are timed. After resuming from a checkpoint,
    ``remaining_time`` is estimated from the speed of the resumed run.

    Args:
        start_step: Global step the run starts from (non-zero after a resume).
        total_steps: Total number of optimizer steps planned.
        clock: Monotonic clock in seconds, injectable for tests.
    """

    def __init__(
        self,
        start_step: int,
        total_steps: int,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.start_step = start_step
        self.total_steps = total_steps
        self._clock = clock
        self._start_time = clock()
        self._interval_start = self._start_time
        self._interval_steps = 0
        self._interval_tokens = 0
        self.num_steps = 0
        self.num_tokens = 0
        self.loss_sum = 0.0

    def update(self, num_tokens: int | float, loss: float) -> None:
        """Record one finished optimizer step."""
        num_tokens = int(num_tokens)
        self.num_steps += 1
        self.num_tokens += num_tokens
        self.loss_sum += float(loss)
        self._interval_steps += 1
        self._interval_tokens += num_tokens

    def log_metrics(self, global_step: int) -> dict[str, Any]:
        """Return timing metrics for a log entry and start a new logging interval.

        ``step_time`` and ``tokens_per_second`` average over the steps since the
        previous log entry, so with ``logging_steps > 1`` they are not just the
        speed of the last step.
        """
        now = self._clock()
        elapsed = now - self._start_time
        interval = now - self._interval_start
        avg_step_time = elapsed / self.num_steps if self.num_steps else 0.0
        remaining = max(self.total_steps - global_step, 0) * avg_step_time
        metrics = {
            "elapsed_time": format_duration(elapsed),
            "remaining_time": format_duration(remaining),
            "step_time": round(interval / self._interval_steps, 4) if self._interval_steps else 0.0,
            "tokens_per_second": round(self._interval_tokens / interval, 2) if interval > 0 else 0.0,
        }
        self._interval_start = now
        self._interval_steps = 0
        self._interval_tokens = 0
        return metrics

    def summary(self, epoch: int, global_step: int) -> dict[str, Any]:
        """Return final metrics for the run, named like the v0 ``train_results.json``."""
        runtime = self._clock() - self._start_time
        return {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": self.loss_sum / self.num_steps if self.num_steps else 0.0,
            "train_runtime": round(runtime, 4),
            "train_steps_per_second": round(self.num_steps / runtime, 3) if runtime > 0 else 0.0,
            "train_tokens": self.num_tokens,
            "train_tokens_per_second": round(self.num_tokens / runtime, 2) if runtime > 0 else 0.0,
        }
