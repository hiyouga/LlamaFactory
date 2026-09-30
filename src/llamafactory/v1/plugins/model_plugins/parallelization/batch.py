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

"""Explicit batch ownership and sequence layout helpers for sequence parallelism."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ....utils.constants import IGNORE_INDEX
from ....utils.types import BatchInput, Tensor


# These tensors belong to a multimodal encoder tower and remain replicated
# until the outer model has fused them into the global language sequence.
MULTIMODAL_ENCODER_INPUT_KEYS = frozenset(
    {
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
        "second_per_grid_ts",
        "video_second_per_grid",
        "input_features",
        "feature_attention_mask",
    }
)

# Only tensors in this list are padded along the global language sequence.
# They remain full while the outer model performs multimodal fusion and mRoPE setup.
SEQUENCE_PARALLEL_INPUT_KEYS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "position_ids",
        "mm_token_type_ids",
    }
)


@dataclass(frozen=True)
class PreparedSequenceParallelBatch:
    """Full model inputs plus CP-local next-token targets."""

    model_inputs: dict[str, Tensor]
    local_shift_labels: Tensor
    local_shift_loss_weights: Tensor
    global_loss_weight_sum: Tensor


def split_sequence_tensor(tensor: Tensor, device_mesh, dim: int = -1) -> Tensor:
    """Take the contiguous sequence shard owned by the local CP rank."""
    cp_mesh = device_mesh["cp"]
    cp_size = cp_mesh.size()
    sequence_length = tensor.shape[dim]
    if sequence_length == 0 or sequence_length % cp_size != 0:
        raise ValueError(f"Sequence length {sequence_length} must be positive and divisible by CP size {cp_size}.")

    cp_rank = cp_mesh.get_local_rank()
    return torch.chunk(tensor, chunks=cp_size, dim=dim)[cp_rank].contiguous()


def prepare_sequence_parallel_batch(
    batch: BatchInput,
    *,
    device: torch.device,
    device_mesh,
    uses_mrope: bool = False,
) -> PreparedSequenceParallelBatch:
    """Pad the global language sequence while preserving encoder-owned layouts."""
    model_inputs = {
        key: value.to(device, non_blocking=True) for key, value in batch.items() if isinstance(value, torch.Tensor)
    }
    labels = model_inputs.pop("labels")
    loss_weights = model_inputs.pop("loss_weights")

    sequence_length = model_inputs["input_ids"].shape[-1]
    cp_size = device_mesh["cp"].size()
    pad_size = -sequence_length % cp_size
    has_multimodal_inputs = bool(MULTIMODAL_ENCODER_INPUT_KEYS.intersection(model_inputs))
    if uses_mrope and has_multimodal_inputs:
        model_inputs.pop("position_ids", None)

    # Only language tensors are padded; encoder tensors retain their original layouts.
    for key in SEQUENCE_PARALLEL_INPUT_KEYS.intersection(model_inputs):
        model_inputs[key] = F.pad(model_inputs[key], (0, pad_size), value=0)

    shift_labels = F.pad(labels[..., 1:], (0, pad_size + 1), value=IGNORE_INDEX)
    shift_loss_weights = F.pad(loss_weights[..., 1:], (0, pad_size + 1), value=0.0)

    return PreparedSequenceParallelBatch(
        model_inputs=model_inputs,
        local_shift_labels=split_sequence_tensor(shift_labels, device_mesh),
        local_shift_loss_weights=split_sequence_tensor(shift_loss_weights, device_mesh),
        global_loss_weight_sum=shift_loss_weights.sum(),
    )
