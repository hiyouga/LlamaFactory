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
from dataclasses import dataclass, field
from typing import Literal, Optional

from transformers.training_args import _convert_str_dict


@dataclass
class MegatronBridgeArguments:
    r"""Arguments for Megatron Bridge distributed training backend.

    Parallelism, optimizer overlap, checkpoint conversion, and selected Megatron
    model-provider knobs are exposed here because Megatron Bridge uses a
    standalone workflow outside the Hugging Face Trainer.
    """

    tensor_model_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor model parallel size for Megatron Bridge."},
    )
    pipeline_model_parallel_size: int = field(
        default=1,
        metadata={"help": "Pipeline model parallel size for Megatron Bridge."},
    )
    expert_model_parallel_size: int = field(
        default=1,
        metadata={"help": "Expert model parallel size for MoE models."},
    )
    context_parallel_size: int = field(
        default=1,
        metadata={"help": "Context parallel size for Megatron Bridge."},
    )
    virtual_pipeline_model_parallel_size: Optional[int] = field(
        default=None,
        metadata={"help": "Virtual pipeline (interleaved) parallel size. None keeps provider default."},
    )
    sequence_parallel: bool = field(
        default=False,
        metadata={"help": "Whether to enable sequence parallelism."},
    )
    recompute_granularity: Optional[str] = field(
        default=None,
        metadata={"help": "Activation recomputation granularity: 'full' or 'selective'."},
    )
    recompute_method: Optional[Literal["uniform", "block"]] = field(
        default=None,
        metadata={"help": "Activation recomputation method: 'uniform' or 'block'."},
    )
    recompute_num_layers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of layers per recompute unit when recompute_method is set."},
    )
    account_for_embedding_in_pipeline_split: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether pipeline split accounts for the embedding layer."},
    )
    account_for_loss_in_pipeline_split: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether pipeline split accounts for the loss layer."},
    )
    bias_activation_fusion: Optional[bool] = field(
        default=None,
        metadata={"help": "Enable bias+activation fusion. None keeps Megatron provider default."},
    )
    apply_rope_fusion: Optional[bool] = field(
        default=None,
        metadata={"help": "Enable RoPE fusion kernel. None keeps Megatron provider default."},
    )
    masked_softmax_fusion: Optional[bool] = field(
        default=None,
        metadata={"help": "Enable masked softmax fusion. None keeps Megatron provider default."},
    )
    cross_entropy_loss_fusion: Optional[bool] = field(
        default=None,
        metadata={"help": "Enable cross-entropy loss fusion. None keeps Megatron provider default."},
    )
    moe_grouped_gemm: Optional[bool] = field(
        default=None,
        metadata={"help": "Enable grouped GEMM for MoE experts. None keeps provider default."},
    )
    moe_token_dispatcher_type: Optional[Literal["allgather", "alltoall", "flex"]] = field(
        default=None,
        metadata={"help": "MoE token dispatcher type: allgather, alltoall, or flex."},
    )
    calculate_per_token_loss: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to compute per-token loss. When context_parallel_size > 1, "
                "this is forced to True regardless of this setting."
            )
        },
    )
    use_distributed_optimizer: bool = field(
        default=True,
        metadata={"help": "Whether to use Megatron distributed optimizer."},
    )
    overlap_param_gather: bool = field(
        default=True,
        metadata={"help": "Whether to overlap parameter all-gather with forward compute."},
    )
    overlap_grad_reduce: bool = field(
        default=True,
        metadata={"help": "Whether to overlap gradient all-reduce with backward compute."},
    )
    use_packed_sequences: bool = field(
        default=False,
        metadata={"help": "Whether to use packed sequences for SFT efficiency."},
    )
    mixed_precision: str = field(
        default="bf16_mixed",
        metadata={"help": "Mixed precision mode for Megatron Bridge, e.g. bf16_mixed or fp8."},
    )
    megatron_pretrained_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to a Megatron-format pretrained checkpoint. "
                "If unset, HF weights are converted automatically before training."
            )
        },
    )
    export_hf_on_finish: bool = field(
        default=False,
        metadata={"help": "Whether to export the final checkpoint to Hugging Face format after training."},
    )
    extra_config: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional JSON string or path to a JSON file with extra Megatron Bridge model/training overrides. "
                "Dot-paths are supported (e.g. train.train_iters or checkpoint.save_interval)."
            )
        },
    )

    def __post_init__(self) -> None:
        if self.tensor_model_parallel_size < 1:
            raise ValueError("`tensor_model_parallel_size` must be >= 1.")
        if self.pipeline_model_parallel_size < 1:
            raise ValueError("`pipeline_model_parallel_size` must be >= 1.")
        if self.expert_model_parallel_size < 1:
            raise ValueError("`expert_model_parallel_size` must be >= 1.")
        if self.context_parallel_size < 1:
            raise ValueError("`context_parallel_size` must be >= 1.")
        if self.virtual_pipeline_model_parallel_size is not None and self.virtual_pipeline_model_parallel_size < 1:
            raise ValueError("`virtual_pipeline_model_parallel_size` must be >= 1 when set.")
        if self.sequence_parallel and self.tensor_model_parallel_size <= 1:
            raise ValueError("`sequence_parallel` requires `tensor_model_parallel_size` > 1.")
        if self.recompute_granularity is not None and self.recompute_granularity not in ("full", "selective"):
            raise ValueError("`recompute_granularity` must be 'full' or 'selective'.")
        if self.recompute_method is not None and self.recompute_method not in ("uniform", "block"):
            raise ValueError("`recompute_method` must be 'uniform' or 'block'.")
        if self.recompute_num_layers is not None and self.recompute_num_layers < 1:
            raise ValueError("`recompute_num_layers` must be >= 1 when set.")
        if self.moe_token_dispatcher_type is not None and self.moe_token_dispatcher_type not in (
            "allgather",
            "alltoall",
            "flex",
        ):
            raise ValueError("`moe_token_dispatcher_type` must be 'allgather', 'alltoall', or 'flex'.")

        if isinstance(self.extra_config, str):
            config_str = self.extra_config.strip()
            if config_str.startswith("{"):
                self.extra_config = _convert_str_dict(json.loads(config_str))
            else:
                self.extra_config = config_str

    def load_extra_config(self) -> dict:
        if self.extra_config is None:
            return {}
        if isinstance(self.extra_config, dict):
            return self.extra_config
        if not os.path.isfile(self.extra_config):
            raise ValueError(f"`extra_config` file not found: {self.extra_config}")
        with open(self.extra_config, encoding="utf-8") as f:
            return json.load(f)
