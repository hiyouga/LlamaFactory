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

import os
from dataclasses import dataclass, field
from typing import Literal

from datasets import DownloadMode


@dataclass
class EvaluationArguments:
    r"""Arguments pertaining to specify the evaluation parameters."""

    task: str = field(
        metadata={"help": "Name of the evaluation task."},
    )
    task_dir: str = field(
        default="evaluation",
        metadata={"help": "Path to the folder containing the evaluation datasets."},
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "The batch size per GPU for evaluation."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed to be used with data loaders."},
    )
    lang: Literal["en", "zh"] = field(
        default="en",
        metadata={"help": "Language used at evaluation."},
    )
    n_shot: int = field(
        default=5,
        metadata={"help": "Number of exemplars for few-shot learning."},
    )
    save_dir: str | None = field(
        default=None,
        metadata={"help": "Path to save the evaluation results."},
    )
    download_mode: DownloadMode = field(
        default=DownloadMode.REUSE_DATASET_IF_EXISTS,
        metadata={"help": "Download mode used for the evaluation datasets."},
    )

    def __post_init__(self):
        if self.save_dir is not None and os.path.exists(self.save_dir):
            raise ValueError("`save_dir` already exists, use another one.")
