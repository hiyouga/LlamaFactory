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

from typing import TYPE_CHECKING, Any

from ...core.utils.profiler import ProfilerController
from .. import logging
from .trainer_callback import TrainerCallback, TrainerState


if TYPE_CHECKING:
    from ...config import TrainingArguments


logger = logging.get_logger(__name__)


class ProfilerCallback(TrainerCallback):
    def __init__(self, args: TrainingArguments) -> None:
        self.profiler = ProfilerController(args, logger=logger)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, **kwargs: Any) -> None:
        self.profiler.start(args.output_dir, initial_step=state.global_step)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, **kwargs: Any) -> None:
        self.profiler.step()

    def on_train_end(self, args: TrainingArguments, state: TrainerState, **kwargs: Any) -> None:
        self.profiler.stop()
