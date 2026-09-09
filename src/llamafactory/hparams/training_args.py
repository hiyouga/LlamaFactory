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
from dataclasses import dataclass, field
from typing import Optional

from transformers import Seq2SeqTrainingArguments
from transformers.training_args import _convert_str_dict

from ..extras.misc import is_env_enabled, use_ray
from ..extras.packages import is_mcore_adapter_available


if is_env_enabled("USE_MCA"):
    if not is_mcore_adapter_available():
        raise ImportError(
            "mcore_adapter is required when USE_MCA=1. Please install `mcore_adapter` and its dependencies."
        )

    from mcore_adapter import Seq2SeqTrainingArguments as McaSeq2SeqTrainingArguments

    BaseTrainingArguments = McaSeq2SeqTrainingArguments
else:
    BaseTrainingArguments = Seq2SeqTrainingArguments


@dataclass
class RayArguments:
    r"""Arguments pertaining to the Ray training."""

    ray_num_workers: int = field(
        default=1,
        metadata={"help": "The number of workers for Ray training. Default is 1 worker."},
    )
    ray_init_kwargs: dict | str | None = field(
        default=None,
        metadata={"help": "The arguments to pass to ray.init for Ray training. Default is None."},
    )
    master_addr: str | None = field(
        default=None,
        metadata={"help": "The master address for init_process_group"},
    )
    master_port: str | None = field(
        default=None,
        metadata={"help": "The master port for init_process_group"},
    )

    def __post_init__(self):
        self.use_ray = use_ray()

        if isinstance(self.ray_init_kwargs, str) and self.ray_init_kwargs.startswith("{"):
            self.ray_init_kwargs = _convert_str_dict(json.loads(self.ray_init_kwargs))


@dataclass
class ProfilerArguments:
    r"""Arguments for torch profiler configuration."""

    enable_torch_profiler: bool = field(
        default=False,
        metadata={"help": "Whether to enable torch profiler for collecting performance traces."},
    )
    profiler_output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to write profiler traces. Defaults to <output_dir>/profiler if not set."},
    )
    profiler_wait_steps: int = field(
        default=1,
        metadata={"help": "Number of steps to skip at the start of each profiling cycle."},
    )
    profiler_warmup_steps: int = field(
        default=1,
        metadata={"help": "Number of profiler warm-up steps per cycle."},
    )
    profiler_active_steps: int = field(
        default=1,
        metadata={"help": "Number of steps to actively record per cycle."},
    )
    profiler_repeat: int = field(
        default=1,
        metadata={"help": "Number of profiling cycles. Set to 0 for continuous profiling."},
    )
    profiler_record_shapes: bool = field(
        default=True,
        metadata={"help": "Whether to record tensor shapes during profiling."},
    )
    profiler_profile_memory: bool = field(
        default=True,
        metadata={"help": "Whether to profile memory usage."},
    )
    profiler_with_stack: bool = field(
        default=True,
        metadata={"help": "Whether to record stack traces during profiling."},
    )
    profile_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Comma-separated list of module name patterns to profile with CUDA events. "
                "Supports fnmatch wildcards (e.g. 'model.layers.0.self_attn,model.layers.*.mlp'). "
                "Reports per-module forward/backward timing statistics at each logging step."
            )
        },
    )


@dataclass
class Fp8Arguments:
    r"""Arguments pertaining to the FP8 training."""

    fp8: bool = field(
        default=False,
        metadata={
            "help": "Enable FP8 mixed precision training via HuggingFace Accelerate. "
            "Requires PyTorch 2.7+ and Hopper architecture GPUs."
        },
    )
    fp8_backend: str = field(
        default="auto",
        metadata={
            "help": "FP8 backend to use ('auto', 'torchao', 'te', 'msamp'). 'auto' selects best available backend."
        },
    )
    fp8_enable_fsdp_float8_all_gather: bool = field(
        default=False,
        metadata={"help": "Enable FP8 optimizations for FSDP2 all-gather operations."},
    )


@dataclass
class ParallelArguments:
    r"""Arguments pertaining to distributed parallelism."""

    ulysses_context_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of ranks in each Ulysses context-parallel group."},
    )


@dataclass
class TrainingArguments(ParallelArguments, ProfilerArguments, Fp8Arguments, RayArguments, BaseTrainingArguments):
    r"""Arguments pertaining to the trainer."""

    overwrite_output_dir: bool = field(
        default=False,
        metadata={"help": "deprecated"},
    )

    def __post_init__(self):
        RayArguments.__post_init__(self)
        BaseTrainingArguments.__post_init__(self)
        if self.ulysses_context_parallel_size < 1:
            raise ValueError("`ulysses_context_parallel_size` must be greater than 0.")
