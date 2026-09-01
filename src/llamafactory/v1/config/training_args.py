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
from uuid import uuid4

from ..utils.logging import get_logger
from .arg_utils import BatchingStrategy, PluginConfig, get_plugin_config


logger = get_logger(__name__)


@dataclass
class TrainingArguments:
    output_dir: str = field(
        default=os.path.join("outputs", str(uuid4().hex)),
        metadata={"help": "Path to the output directory."},
    )
    micro_batch_size: int = field(
        default=1,
        metadata={"help": "Micro batch size for training."},
    )
    global_batch_size: int | None = field(
        default=None,
        metadata={"help": "Global batch size for training, default to DP size * micro batch size."},
    )
    cutoff_len: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for training."},
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for training."},
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs."},
    )
    max_steps: int | None = field(
        default=None,
        metadata={"help": "Maximum number of training steps. If set, overrides num_train_epochs."},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm for training."},
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bf16 for training."},
    )
    batching_strategy: BatchingStrategy = field(
        default=BatchingStrategy.NORMAL,
        metadata={"help": "Batching strategy for training."},
    )
    batching_workers: int = field(
        default=16,
        metadata={"help": "Number of workers for batching."},
    )
    enable_activation_checkpointing: bool = field(
        default=True,
        metadata={"help": "Enable activation checkpointing for training."},
    )
    enable_torch_profiler: bool = field(
        default=False,
        metadata={"help": "Whether to enable torch profiler for collecting CPU/CUDA/NPU performance traces."},
    )
    profiler_output_dir: str | None = field(
        default=None,
        metadata={"help": "Directory to write profiler traces. Defaults to <output_dir>/profiler if not set."},
    )
    profiler_skip_first: int = field(
        default=0,
        metadata={"help": "Number of steps to skip before the first profiler wait/warmup/active cycle."},
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
    profiler_record_shapes: bool | None = field(
        default=None,
        metadata={"help": "Whether to record tensor shapes during profiling.", "nargs": "?", "const": True},
    )
    profiler_profile_memory: bool | None = field(
        default=None,
        metadata={"help": "Whether to profile memory usage.", "nargs": "?", "const": True},
    )
    profiler_with_stack: bool | None = field(
        default=None,
        metadata={"help": "Whether to record stack traces during profiling.", "nargs": "?", "const": True},
    )
    profiler_with_flops: bool = field(
        default=False,
        metadata={"help": "Whether to estimate FLOPs where supported by the profiler backend."},
    )
    profiler_with_modules: bool = field(
        default=False,
        metadata={"help": "Whether to record module hierarchy where supported by the profiler backend."},
    )
    profiler_activities: str = field(
        default="auto",
        metadata={"help": "Profiler activities to collect. Choices: auto, all, cpu, device."},
    )
    profiler_rank_mode: str = field(
        default="all",
        metadata={"help": "Profiler rank collection mode. Choices: all, rank0."},
    )
    profiler_level: str = field(
        default="level0",
        metadata={"help": "NPU profiler collection level. Choices: none, level0, level1, level2."},
    )
    profiler_aic_metrics: str = field(
        default="auto",
        metadata={
            "help": (
                "NPU AI Core metric: auto, none, pipe_utilization, arithmetic_utilization, memory, "
                "memory_l0, memory_ub, l2_cache, memory_access, or resource_conflict_ratio."
            )
        },
    )
    profiler_backend_options: dict | None = field(
        default=None,
        metadata={
            "help": (
                "Backend-specific profiler mapping. Currently supports npu.data_simplification, npu.host_sys, "
                "npu.sys_io, npu.sys_interconnection, and npu.gc_detect_threshold."
            )
        },
    )
    dist_config: PluginConfig | None = field(
        default=None,
        metadata={"help": "Distributed backend plugin configuration."},
    )
    dp_size: int | None = field(
        default=None,
        metadata={"help": "Data parallel size, default to world_size // cp_size."},
    )
    cp_size: int = field(
        default=1,
        metadata={"help": "Context parallel size."},
    )
    cp_mode: str = field(
        default="ulysses",
        metadata={"help": "Context parallel implementation."},
    )
    mp_replicate_size: int = field(
        default=1,
        metadata={"help": "Model parallel replicate size."},
    )
    mp_shard_size: int | None = field(
        default=None,
        metadata={"help": "Model parallel shard size, default to world_size // mp_replicate_size."},
    )
    dist_timeout: int = field(
        default=18000,
        metadata={"help": "Distributed process group initialization timeout in seconds."},
    )
    optim_config: PluginConfig | None = field(
        default=None,
        metadata={"help": "Optimizer configuration for training."},
    )
    lr_scheduler_config: PluginConfig | None = field(
        default=None,
        metadata={"help": "Learning rate scheduler configuration for training."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."},
    )
    full_determinism: bool = field(
        default=False,
        metadata={"help": "Enable full deterministic mode for reproducible distributed training."},
    )
    resume_from_checkpoint: str | None = field(
        default=None,
        metadata={"help": "Path to a checkpoint directory to resume training from, or 'auto' to find the latest."},
    )
    save_steps: int | None = field(
        default=None,
        metadata={"help": "Save a training checkpoint every N global steps."},
    )
    save_epochs: float | None = field(
        default=None,
        metadata={"help": "Save a training checkpoint every N epochs."},
    )
    save_ckpt_as_hf: bool = field(
        default=False,
        metadata={
            "help": "Save intermediate checkpoints in HuggingFace format instead of distributed format. Warning: doubles memory usage."
        },
    )
    save_total_limit: int | None = field(
        default=None,
        metadata={"help": "Maximum number of checkpoints to keep. Oldest checkpoints are deleted."},
    )
    logging_steps: int = field(
        default=1,
        metadata={"help": "Log metrics every N optimizer steps."},
    )
    pref_loss: Literal["sigmoid", "orpo", "simpo"] = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use."},
    )
    pref_beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter in the preference loss."},
    )
    pref_ftx: float = field(
        default=0.0,
        metadata={"help": "The supervised fine-tuning loss coefficient in DPO training."},
    )
    simpo_gamma: float = field(
        default=0.5,
        metadata={"help": "The target reward margin term in SimPO loss."},
    )
    dpo_label_smoothing: float = field(
        default=0.0,
        metadata={"help": "The robust DPO label smoothing parameter in cDPO that should be between 0 and 0.5."},
    )
    ld_alpha: float | None = field(
        default=None,
        metadata={"help": "Alpha parameter from LD-DPO, controls weighting of verbose token log-probabilities."},
    )

    def __post_init__(self) -> None:
        self.dist_config = get_plugin_config(self.dist_config)
        self.optim_config = get_plugin_config(self.optim_config)
        self.lr_scheduler_config = get_plugin_config(self.lr_scheduler_config)
        try:
            from ..plugins.model_plugins.deepspeed_utils import register_deepspeed_dist_config

            register_deepspeed_dist_config(self.dist_config)
        except ImportError:
            pass

        # The optimizer learning rate has a single source of truth: ``learning_rate``.
        # Propagate it into ``optim_config["lr"]`` so optimizer plugins (e.g. Muon) pick it up
        # via ``optim_config.get("lr")`` without each plugin needing a separate ``learning_rate`` arg.
        if self.optim_config is not None:
            if "lr" in self.optim_config:
                logger.warning_rank0(
                    "`optim_config.lr` is overridden by `learning_rate`; set the learning rate via "
                    "`learning_rate` instead and remove `lr` from `optim_config`."
                )
            self.optim_config["lr"] = self.learning_rate

        if str(self.batching_strategy) == str(BatchingStrategy.DYNAMIC_BATCHING):
            if self.max_steps is None or self.max_steps <= 0:
                raise ValueError("`dynamic_batching` requires `max_steps` because it is step-driven.")
            if self.save_epochs is not None:
                raise ValueError("`save_epochs` is not supported with `dynamic_batching`; use `save_steps` instead.")
