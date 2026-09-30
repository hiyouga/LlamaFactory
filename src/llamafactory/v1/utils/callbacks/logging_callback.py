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

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from .. import logging
from .trainer_callback import TrainerCallback, TrainerState


if TYPE_CHECKING:
    from ...config import TrainingArguments


logger = logging.get_logger(__name__)


class LoggingCallback(TrainerCallback):
    """Logs training metrics to stdout on rank-0 and appends to ``state.log_history``.

    On each logging step the entry is also persisted as a JSON line in
    ``<output_dir>/trainer_log.jsonl`` so that training history survives crashes.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        logs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        # Persist in history regardless of rank
        state.log_history.append(dict(logs))

        # Everything below is rank-0 only
        from ...accelerator.interface import DistributedInterface  # lazy import

        if DistributedInterface().get_rank() != 0:
            return

        # Human-readable output to stdout
        display_logs = {**logs, "step": state.global_step, "total_steps": state.num_training_steps}

        def _fmt(k: str, v) -> str:
            if not isinstance(v, float):
                return f"{k}: {v}"
            # learning_rate is often < 1e-4 (e.g. 1e-5); :.4f would print "0.0000".
            # Use :.4g so small values show as "1e-05" while 1e-4 still shows "0.0001".
            if k == "learning_rate":
                return f"{k}: {v:.4g}"
            return f"{k}: {v:.4f}"

        parts = ", ".join(_fmt(k, v) for k, v in display_logs.items())
        logger.info_rank0(parts)

        # Append to JSONL log file in output_dir
        os.makedirs(args.output_dir, exist_ok=True)
        log_file = os.path.join(args.output_dir, "trainer_log.jsonl")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(display_logs, ensure_ascii=False) + "\n")
