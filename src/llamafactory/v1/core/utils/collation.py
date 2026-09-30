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

"""Batch collation utils: padding/truncation and multimodal-feature alignment.

These operate on already-rendered ``ModelInput`` dicts (token lists + pixel tensors) and produce
padded ``BatchInput`` tensors. They are pure batching concerns -- independent of how a sample was
rendered -- and are consumed by the batch generators in ``core/utils/batching.py`` and
``plugins/trainer_plugins/batching.py``. Kept out of ``rendering.py`` so that file is only about
turning messages into a single tokenized sample.
"""

import torch

from ...utils.constants import IGNORE_INDEX
from ...utils.types import BatchInput, ModelInput, Tensor


# Multimodal feature keys the processor emits per sample. They are NOT padded/stacked like text
# fields: pixel/audio-feature tensors are ragged (variable patch / frame counts), so the collators
# concatenate them along dim 0 instead. Shared by rendering (which copies them verbatim from the
# processor) and the collators (which merge them across a micro batch).
_MULTIMODAL_PASSTHROUGH_KEYS = frozenset(
    {
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
        "second_per_grid_ts",  # Qwen2.5-VL name for the video temporal grid spacing
        "video_second_per_grid",  # Qwen2.5-Omni name for the same (fed to get_rope_index)
        "input_features",
        "feature_attention_mask",
    }
)


def _pad_and_truncate(tensor: Tensor, max_seqlen: int, pad_value: int = 0) -> Tensor:
    if tensor.shape[-1] >= max_seqlen:
        return tensor[..., :max_seqlen]

    pad_shape = list(tensor.shape)
    pad_shape[-1] = max_seqlen - tensor.shape[-1]
    pad_tensor = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad_tensor], dim=-1)


def _align_grid_media(
    sample: ModelInput,
    mm_type_ids: list[int],
    max_length: int,
    *,
    target: int,
    grid_key: str,
    pixel_key: str,
) -> list[int]:
    """Trim and zero one modality's orphaned tokens for a single sample.

    Layout-agnostic: a media item's placeholder tokens may be a single contiguous run or split
    into per-frame sub-runs; completeness is decided per token *position*, so both are handled
    identically.

    Returns the (possibly updated) ``mm_token_type_ids`` so chained calls see earlier zeroing.
    """
    if grid_key not in sample or pixel_key not in sample:
        return mm_type_ids

    grid = sample[grid_key]
    n_items = len(grid)
    if n_items == 0:
        return mm_type_ids

    positions = [i for i, t in enumerate(mm_type_ids) if t == target]
    patches_per_item = [int(grid[i].prod()) for i in range(n_items)]
    total_patches = sum(patches_per_item)
    total_tokens = len(positions)

    # merge_size**2 = pixel patches per placeholder token, derived from the data. Bail out
    # untouched if the sample is inconsistent.
    if total_tokens == 0 or total_patches % total_tokens != 0:
        return mm_type_ids
    merge_sq = total_patches // total_tokens
    tokens_per_item = [p // merge_sq for p in patches_per_item]
    if sum(tokens_per_item) != total_tokens:
        return mm_type_ids

    # Each item owns a contiguous slice of `positions`; it is complete iff its last
    # placeholder token lands inside the kept window [0, max_length).
    n_complete = 0
    cum = 0
    for n_i in tokens_per_item:
        if positions[cum + n_i - 1] < max_length:
            n_complete += 1
            cum += n_i
        else:
            break

    if n_complete >= n_items:
        return mm_type_ids

    # Trim pixel features and grid to the complete prefix.
    keep_patches = sum(patches_per_item[:n_complete])
    sample[pixel_key] = sample[pixel_key][:keep_patches]
    sample[grid_key] = grid[:n_complete]

    # Zero out orphaned placeholder tokens that fall inside the kept window; tokens
    # beyond max_length are removed by truncation anyway (positions are sorted).
    input_ids = list(sample["input_ids"])
    mm_type_ids = list(mm_type_ids)
    labels = list(sample["labels"]) if "labels" in sample else None
    loss_weights = list(sample["loss_weights"]) if "loss_weights" in sample else None

    for pos in positions[cum:]:
        if pos >= max_length:
            break
        input_ids[pos] = 0
        mm_type_ids[pos] = 0
        if labels is not None:
            labels[pos] = IGNORE_INDEX
        if loss_weights is not None:
            loss_weights[pos] = 0.0

    sample["input_ids"] = input_ids
    sample["mm_token_type_ids"] = mm_type_ids
    if labels is not None:
        sample["labels"] = labels
    if loss_weights is not None:
        sample["loss_weights"] = loss_weights
    return mm_type_ids


def _align_audio(sample: ModelInput, mm_type_ids: list[int], max_length: int, *, target: int = 3) -> list[int]:
    """Trim and zero orphaned audio tokens for a single sample on truncation.

    Returns the (possibly updated) ``mm_token_type_ids``.
    """
    if "input_features" not in sample or "feature_attention_mask" not in sample:
        return mm_type_ids

    n_items = sample["input_features"].shape[0]
    if n_items == 0:
        return mm_type_ids

    positions = [i for i, t in enumerate(mm_type_ids) if t == target]
    if not positions:
        return mm_type_ids

    # Group the marked positions into maximal contiguous runs; each run is one audio's token span.
    runs: list[tuple[int, int]] = []
    run_start = prev = positions[0]
    for pos in positions[1:]:
        if pos != prev + 1:
            runs.append((run_start, prev))
            run_start = pos
        prev = pos
    runs.append((run_start, prev))

    # Layout must match the feature rows one-to-one, else bail rather than corrupt the mapping.
    if len(runs) != n_items:
        return mm_type_ids

    # An audio is complete iff its last placeholder token lands inside the kept window.
    n_complete = 0
    for _start, end in runs:
        if end < max_length:
            n_complete += 1
        else:
            break

    if n_complete >= n_items:
        return mm_type_ids

    # Trim feature rows to the complete prefix.
    sample["input_features"] = sample["input_features"][:n_complete]
    sample["feature_attention_mask"] = sample["feature_attention_mask"][:n_complete]

    # Zero out orphaned placeholder tokens that fall inside the kept window; tokens beyond
    # max_length are removed by truncation anyway.
    input_ids = list(sample["input_ids"])
    mm_type_ids = list(mm_type_ids)
    labels = list(sample["labels"]) if "labels" in sample else None
    loss_weights = list(sample["loss_weights"]) if "loss_weights" in sample else None

    for start, end in runs[n_complete:]:
        for pos in range(start, end + 1):
            if pos >= max_length:
                break
            input_ids[pos] = 0
            mm_type_ids[pos] = 0
            if labels is not None:
                labels[pos] = IGNORE_INDEX
            if loss_weights is not None:
                loss_weights[pos] = 0.0

    sample["input_ids"] = input_ids
    sample["mm_token_type_ids"] = mm_type_ids
    if labels is not None:
        sample["labels"] = labels
    if loss_weights is not None:
        sample["loss_weights"] = loss_weights
    return mm_type_ids


def _align_multimodal_on_truncation(sample: ModelInput, max_length: int) -> ModelInput:
    """Remove orphaned multimodal data when the sequence will be truncated.

    When cutoff_len truncates input_ids, media whose placeholder tokens are partially cut lose
    their token<->feature correspondence. Trims pixel_values/grid_thw (vision) and
    input_features/feature_attention_mask (audio) to the complete items and zeros out orphaned
    placeholder tokens so the model ignores them.
    """
    mm_type_ids = sample.get("mm_token_type_ids")
    if mm_type_ids is None:
        return sample

    sample = dict(sample)

    mm_type_ids = _align_grid_media(
        sample, mm_type_ids, max_length, target=1, grid_key="image_grid_thw", pixel_key="pixel_values"
    )
    mm_type_ids = _align_grid_media(
        sample, mm_type_ids, max_length, target=2, grid_key="video_grid_thw", pixel_key="pixel_values_videos"
    )
    mm_type_ids = _align_audio(sample, mm_type_ids, max_length, target=3)

    # Remove empty multimodal fields entirely
    if "image_grid_thw" in sample and len(sample["image_grid_thw"]) == 0:
        del sample["pixel_values"]
        del sample["image_grid_thw"]
    if "video_grid_thw" in sample and len(sample["video_grid_thw"]) == 0:
        del sample["pixel_values_videos"]
        del sample["video_grid_thw"]
    if "input_features" in sample and sample["input_features"].shape[0] == 0:
        del sample["input_features"]
        del sample["feature_attention_mask"]

    return sample


def pad_and_truncate(samples: list[ModelInput], max_seqlen: int) -> list[BatchInput]:
    max_length = min(max(len(sample["input_ids"]) for sample in samples), max_seqlen)
    padded_samples = []
    for sample in samples:
        # Align multimodal fields before truncation: remove images/videos whose
        # placeholder tokens would be partially cut, preventing pixel<->token mismatch.
        if len(sample["input_ids"]) > max_length and any(k in sample for k in _MULTIMODAL_PASSTHROUGH_KEYS):
            sample = _align_multimodal_on_truncation(sample, max_length)

        padded_sample = {}
        for key, value in sample.items():
            if key in _MULTIMODAL_PASSTHROUGH_KEYS:
                padded_sample[key] = value
                continue

            if "label" in key:
                pad_value = IGNORE_INDEX
            else:
                pad_value = 0

            if not isinstance(value, str):
                padded_sample[key] = _pad_and_truncate(torch.tensor(value), max_length, pad_value)
            else:
                padded_sample[key] = value

        padded_samples.append(padded_sample)

    return padded_samples
