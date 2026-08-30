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

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ...extras.constants import IGNORE_INDEX


if TYPE_CHECKING:
    from torch.utils.data import Sampler

    from ...hparams import DataArguments, FinetuningArguments, TrainingArguments


SEQUENCE_INPUT_NAMES = {
    "attention_mask",
    "input_ids",
    "labels",
    "mm_token_type_ids",
    "position_ids",
    "shift_labels",
    "token_type_ids",
}

UNSUPPORTED_MULTIMODAL_INPUT_NAMES = {
    "image_position_ids",
    "input_features",
    "input_features_mask",
    "pixel_values",
    "pixel_values_videos",
    "video_position_ids",
}


def validate_context_parallel_sft_args(
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
    world_size: Optional[int] = None,
) -> None:
    """Validate the constraints of the current Ulysses SFT implementation.

    For Gemma 4 context parallelism currently. Raises ValueError if any constraints are violated.
    """
    cp_size = training_args.ulysses_context_parallel_size
    if cp_size <= 1:
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1")) if world_size is None else world_size
    if world_size % cp_size != 0:
        raise ValueError(
            f"WORLD_SIZE ({world_size}) must be divisible by ulysses_context_parallel_size ({cp_size})."
        )
    if training_args.deepspeed is None:
        raise ValueError("`ulysses_context_parallel_size > 1` currently requires DeepSpeed.")
    if not training_args.bf16:
        raise ValueError("`ulysses_context_parallel_size > 1` currently requires BF16 training.")
    if training_args.per_device_train_batch_size != 1:
        raise ValueError("Context parallelism currently requires `per_device_train_batch_size: 1`.")
    if not training_args.dataloader_drop_last:
        raise ValueError("Context parallelism currently requires `dataloader_drop_last: true`.")
    if data_args.packing or data_args.neat_packing:
        raise ValueError("Context parallelism does not support packed SFT data yet.")
    if (
        getattr(training_args.eval_strategy, "value", training_args.eval_strategy) != "no"
        or training_args.do_eval
        or training_args.do_predict
    ):
        raise ValueError("Evaluation and prediction are not supported with context parallelism yet.")
    if training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` is not supported with context parallelism.")
    if training_args.label_smoothing_factor != 0.0:
        raise ValueError("Label smoothing is not supported with context parallelism.")
    if not training_args.average_tokens_across_devices:
        raise ValueError("Context parallelism requires `average_tokens_across_devices: true`.")
    if training_args.parallelism_config is not None:
        raise ValueError("Do not combine LLaMA Factory context parallelism with Transformers native parallelism.")
    if finetuning_args.use_dft_loss or finetuning_args.use_eaft_loss or finetuning_args.use_asft_loss:
        raise ValueError("Custom SFT losses are not supported with context parallelism yet.")


class ContextParallelSampler(torch.utils.data.Sampler):
    """Repeat each sample for the contiguous ranks in one CP group.

    Accelerate shards the resulting batch stream across the full distributed
    world. With a per-device batch size of one, repeating each base index
    ``cp_size`` times maps the same sample to each rank in a contiguous CP
    group while distinct groups receive distinct data-parallel samples.
    """

    def __init__(self, sampler: "Sampler", cp_size: int) -> None:
        if cp_size < 2:
            raise ValueError("ContextParallelSampler requires cp_size >= 2.")

        self.sampler = sampler
        self.cp_size = cp_size

    def __iter__(self) -> Iterator[int]:
        for index in self.sampler:
            yield from (index for _ in range(self.cp_size))

    def __len__(self) -> int:
        return len(self.sampler) * self.cp_size

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


def create_context_parallel_group(cp_size: int) -> dist.ProcessGroup:
    """Create contiguous CP groups and return the group for this rank."""
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Context parallelism requires an initialized torch.distributed process group.")

    world_size = dist.get_world_size()
    if cp_size < 2 or world_size % cp_size != 0:
        raise ValueError(f"cp_size={cp_size} must be at least 2 and divide world_size={world_size}.")

    rank = dist.get_rank()
    rank_group: Optional[dist.ProcessGroup] = None
    for first_rank in range(0, world_size, cp_size):
        ranks = list(range(first_rank, first_rank + cp_size))
        process_group = dist.new_group(ranks=ranks)
        if rank in ranks:
            rank_group = process_group

    if rank_group is None:
        raise RuntimeError(f"Rank {rank} was not assigned to a context-parallel group.")

    return rank_group


def make_shift_labels(labels: torch.Tensor) -> torch.Tensor:
    """Shift full-sequence labels before sharding so CP boundaries are exact."""
    if labels.ndim != 2:
        raise ValueError(f"Context parallelism expects 2D labels, got shape {tuple(labels.shape)}.")

    return F.pad(labels[..., 1:], (0, 1), value=IGNORE_INDEX).contiguous()


def context_parallel_cross_entropy(
    logits: torch.Tensor,
    shift_labels: torch.Tensor,
    num_items_in_batch: torch.Tensor | int,
    world_size: int,
) -> torch.Tensor:
    """Normalize a local CP loss for DeepSpeed's full-world gradient average."""
    if logits.ndim != 3 or shift_labels.shape != logits.shape[:2]:
        raise ValueError(
            f"Expected logits (batch, sequence, vocab) and matching labels, got {tuple(logits.shape)} and "
            f"{tuple(shift_labels.shape)}."
        )
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}.")

    token_count = torch.as_tensor(num_items_in_batch, device=logits.device)
    if token_count.numel() != 1 or token_count.item() <= 0:
        raise ValueError("Context parallelism requires a positive global token count.")

    loss = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        shift_labels.reshape(-1).to(logits.device),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    return loss / token_count * world_size


def split_sequence_inputs(inputs: dict[str, Any], cp_size: int, cp_rank: int) -> dict[str, Any]:
    """Pad and shard supported text inputs along their sequence dimension."""
    if cp_size < 2 or not 0 <= cp_rank < cp_size:
        raise ValueError(f"Invalid context-parallel coordinates: cp_size={cp_size}, cp_rank={cp_rank}.")
    if "input_ids" not in inputs:
        raise ValueError("Context parallelism requires `input_ids`.")

    input_ids = inputs["input_ids"]
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("Context parallelism currently requires 2D `input_ids` and text-only SFT batches.")

    sharded_inputs = dict(inputs)
    attention_mask = inputs.get("attention_mask")
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    if mm_token_type_ids is not None:
        if (
            not isinstance(mm_token_type_ids, torch.Tensor)
            or mm_token_type_ids.ndim != 2
            or mm_token_type_ids.shape[0] != input_ids.shape[0]
        ):
            raise ValueError("`mm_token_type_ids` must have the same 2D batch shape as `input_ids`.")
        if mm_token_type_ids.shape[1] > input_ids.shape[1]:
            raise ValueError("`mm_token_type_ids` cannot be longer than `input_ids`.")
        if mm_token_type_ids.shape[1] < input_ids.shape[1]:
            # The multimodal collator builds token types before the tokenizer
            # left-pads input tensors to a tensor-core-friendly multiple.
            mm_token_type_ids = F.pad(mm_token_type_ids, (input_ids.shape[1] - mm_token_type_ids.shape[1], 0), value=0)
            sharded_inputs["mm_token_type_ids"] = mm_token_type_ids
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape:
            raise ValueError("Gemma 4 context parallelism requires an attention mask for multimodal token types.")

        active_multimodal_tokens = mm_token_type_ids.ne(0) & attention_mask.bool()
        if active_multimodal_tokens.any().item():
            raise NotImplementedError("Gemma 4 context parallelism currently supports text-only SFT batches.")

    has_dummy_multimodal_inputs = False
    for name in UNSUPPORTED_MULTIMODAL_INPUT_NAMES:
        value = inputs.get(name)
        if value is None:
            continue
        if (isinstance(value, torch.Tensor) and value.numel() == 0) or (
            isinstance(value, (list, tuple)) and len(value) == 0
        ):
            sharded_inputs.pop(name, None)
            continue
        if mm_token_type_ids is None:
            raise NotImplementedError(
                f"Gemma 4 context parallelism currently supports text-only SFT batches, but `{name}` is populated."
            )

        # The multimodal collator appends an attention-masked fake image/audio
        # under ZeRO-3 so trainable vision parameters participate in the graph.
        # CP requires those parameters to be frozen, so discard this dummy
        # payload and trim its masked placeholder tokens before sharding.
        has_dummy_multimodal_inputs = True
        sharded_inputs.pop(name, None)

    if has_dummy_multimodal_inputs:
        if input_ids.shape[0] != 1:
            raise ValueError("Dummy multimodal trimming requires a per-device batch size of one.")
        active_positions = torch.nonzero(attention_mask[0], as_tuple=False).flatten()
        if active_positions.numel() == 0:
            raise ValueError("Context parallelism received an empty text sequence.")

        text_length = int(active_positions[-1].item()) + 1
        for name in SEQUENCE_INPUT_NAMES:
            value = sharded_inputs.get(name)
            if isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[1] == input_ids.shape[1]:
                sharded_inputs[name] = value[:, :text_length]

        input_ids = sharded_inputs["input_ids"]

    sequence_length = input_ids.shape[1]
    padding_length = (-sequence_length) % cp_size
    for name, value in sharded_inputs.items():
        if name not in SEQUENCE_INPUT_NAMES:
            if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[:2] == input_ids.shape:
                raise NotImplementedError(f"Sequence input `{name}` is not supported by context parallelism.")
            continue
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != sequence_length:
            raise ValueError(
                f"Sequence input `{name}` must have shape (batch, {sequence_length}), got "
                f"{getattr(value, 'shape', None)}."
            )

        padding_value = IGNORE_INDEX if name in ("labels", "shift_labels") else 0
        padded_value = F.pad(value, (0, padding_length), value=padding_value)
        sharded_inputs[name] = padded_value.chunk(cp_size, dim=1)[cp_rank].contiguous()

    return sharded_inputs
