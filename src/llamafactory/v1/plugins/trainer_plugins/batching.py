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

from abc import ABC, abstractmethod
from collections.abc import Callable
from math import ceil
from typing import Any

import torch
from torch.utils.data import default_collate

from ...core.utils.collation import pad_and_truncate
from ...utils.constants import IGNORE_INDEX
from ...utils.objects import StatefulBuffer
from ...utils.plugin import BasePlugin, ensure_methods_implemented
from ...utils.types import BatchInfo, BatchInput, DataLoader, ModelInput


class BatchingPlugin(BasePlugin):
    """Plugin family for batching strategy method groups."""


class BaseBatcher(ABC):
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        ensure_methods_implemented(cls)

    @staticmethod
    @abstractmethod
    def get_data_provider_batch_size(batch_info: BatchInfo) -> int: ...

    @staticmethod
    @abstractmethod
    def compute_length(data_provider: DataLoader, batch_info: BatchInfo) -> int: ...

    @staticmethod
    @abstractmethod
    def fill_buffer(
        buffer: StatefulBuffer,
        batch_info: BatchInfo,
        next_samples: Callable[[bool], list[ModelInput] | None],
    ) -> None: ...

    @staticmethod
    @abstractmethod
    def generate_batch(buffer: StatefulBuffer, batch_info: BatchInfo) -> list[BatchInput] | None: ...


def _get_dynamic_micro_batch_sizes(samples: list[ModelInput], batch_info: BatchInfo) -> list[int]:
    """Return sample counts for micro batches formed by one padded-token budget."""
    budget = batch_info["cutoff_len"] * batch_info["micro_batch_size"]
    cutoff_len = batch_info["cutoff_len"]
    sizes = []
    index = 0
    while index < len(samples) and len(sizes) < batch_info["num_micro_batch"]:
        max_sample_len = 0
        used = 0
        is_complete = False
        while index + used < len(samples):
            sample_len = min(len(samples[index + used]["input_ids"]), cutoff_len)
            padded_tokens = max(max_sample_len, sample_len) * (used + 1)
            if used > 0 and padded_tokens > budget:
                is_complete = True
                break

            max_sample_len = max(max_sample_len, sample_len)
            used += 1
            if max_sample_len * used >= budget:
                is_complete = True
                break

        if used == 0 or not is_complete:
            break

        sizes.append(used)
        index += used

    return sizes


def _get_dynamic_padding_free_micro_batch_sizes(samples: list[ModelInput], batch_info: BatchInfo) -> list[int]:
    budget = batch_info["cutoff_len"] * batch_info["micro_batch_size"]
    cutoff_len = batch_info["cutoff_len"]
    sizes = []
    index = 0

    while index < len(samples) and len(sizes) < batch_info["num_micro_batch"]:
        current_tokens = 0
        used = 0
        is_complete = False

        while index + used < len(samples):
            sample = samples[index + used]
            sample_len = min(len(sample["input_ids"]), cutoff_len)

            if current_tokens + sample_len > budget:
                is_complete = True
                break

            current_tokens += sample_len
            used += 1

        if used <= 0 or not is_complete:
            break

        sizes.append(used)
        index += used

    return sizes


def _pack_padding_free_samples(samples: list[ModelInput], cutoff_len: int) -> BatchInput | None:
    """Pack fixed samples into one padding-free sequence without a token budget."""
    packed: dict[str, list[Any]] = {}
    position_ids: list[int] = []

    for sample_index, sample in enumerate(samples):
        # Padding-free still truncates each sample by cutoff_len before packing
        # all samples into one contiguous sequence.
        sample_len = min(len(sample["input_ids"]), cutoff_len)
        if sample_len <= 0:
            continue

        for key, value in sample.items():
            if key in ("attention_mask", "position_ids") or isinstance(value, str):
                continue

            if key not in packed:
                packed[key] = []

            sliced_value = list(value[:sample_len])
            if sample_index > 0 and sliced_value:
                if key == "labels":
                    sliced_value[0] = IGNORE_INDEX
                elif key == "loss_weights":
                    sliced_value[0] = 0.0

            packed[key].extend(sliced_value)

        position_ids.extend(range(sample_len))

    if not position_ids:
        return None

    packed["position_ids"] = position_ids
    packed["attention_mask"] = None
    return {key: None if value is None else torch.tensor(value).unsqueeze(0) for key, value in packed.items()}


@BatchingPlugin("padding_free").register()
class PaddingFreeBatcher(BaseBatcher):
    @staticmethod
    def get_data_provider_batch_size(batch_info: BatchInfo) -> int:
        return batch_info["micro_batch_size"] * batch_info["num_micro_batch"]

    @staticmethod
    def compute_length(data_provider: DataLoader, batch_info: BatchInfo) -> int:
        return len(data_provider)

    @staticmethod
    def fill_buffer(
        buffer: StatefulBuffer,
        batch_info: BatchInfo,
        next_samples: Callable[[bool], list[ModelInput] | None],
    ) -> None:
        while len(buffer) < batch_info["micro_batch_size"] * batch_info["num_micro_batch"]:
            samples = next_samples(False)
            if samples is None:
                break

            buffer.put(samples)

    @staticmethod
    def generate_batch(buffer: StatefulBuffer, batch_info: BatchInfo) -> list[BatchInput] | None:
        micro_batch_size = batch_info["micro_batch_size"]
        num_micro_batch = batch_info["num_micro_batch"]
        cutoff_len = batch_info["cutoff_len"]
        batch_size = micro_batch_size * num_micro_batch
        if len(buffer) < batch_size:
            return None

        samples = buffer.get(batch_size)
        batch = []
        for i in range(num_micro_batch):
            micro_batch = samples[i * micro_batch_size : (i + 1) * micro_batch_size]
            packed_micro_batch = _pack_padding_free_samples(micro_batch, cutoff_len)
            if packed_micro_batch is None:
                return None

            batch.append(packed_micro_batch)

        return batch


@BatchingPlugin("dynamic_batching").register()
class DynamicBatcher(BaseBatcher):
    @staticmethod
    def get_data_provider_batch_size(batch_info: BatchInfo) -> int:
        return 1

    @staticmethod
    def compute_length(data_provider: DataLoader, batch_info: BatchInfo) -> int:
        batch_size = batch_info["micro_batch_size"] * batch_info["num_micro_batch"]
        return ceil(len(data_provider) / batch_size)

    @staticmethod
    def fill_buffer(
        buffer: StatefulBuffer,
        batch_info: BatchInfo,
        next_samples: Callable[[bool], list[ModelInput] | None],
    ) -> None:
        while len(_get_dynamic_micro_batch_sizes(buffer.samples, batch_info)) < batch_info["num_micro_batch"]:
            samples = next_samples(True)
            if samples is None:
                break

            buffer.put(samples)

    @staticmethod
    def generate_batch(buffer: StatefulBuffer, batch_info: BatchInfo) -> list[BatchInput] | None:
        micro_batch_sample_counts = _get_dynamic_micro_batch_sizes(buffer.samples, batch_info)
        if len(micro_batch_sample_counts) < batch_info["num_micro_batch"]:
            return None

        batch = []
        cutoff_len = batch_info["cutoff_len"]
        for num_samples in micro_batch_sample_counts:
            samples = buffer.get(num_samples)
            batch.append(default_collate(pad_and_truncate(samples, cutoff_len)))

        return batch


@BatchingPlugin("dynamic_padding_free").register()
class DynamicPaddingFreeBatcher(BaseBatcher):
    @staticmethod
    def get_data_provider_batch_size(batch_info: BatchInfo) -> int:
        return 1

    @staticmethod
    def compute_length(data_provider: DataLoader, batch_info: BatchInfo) -> int:
        batch_size = batch_info["micro_batch_size"] * batch_info["num_micro_batch"]
        return ceil(len(data_provider) / batch_size)

    @staticmethod
    def fill_buffer(
        buffer: StatefulBuffer,
        batch_info: BatchInfo,
        next_samples: Callable[[bool], list[ModelInput] | None],
    ) -> None:
        while (
            len(_get_dynamic_padding_free_micro_batch_sizes(buffer.samples, batch_info))
            < batch_info["num_micro_batch"]
        ):
            samples = next_samples(True)
            if samples is None:
                break
            buffer.put(samples)

    @staticmethod
    def generate_batch(buffer: StatefulBuffer, batch_info: BatchInfo) -> list[BatchInput] | None:
        micro_batch_sample_counts = _get_dynamic_padding_free_micro_batch_sizes(buffer.samples, batch_info)
        if len(micro_batch_sample_counts) < batch_info["num_micro_batch"]:
            return None

        batch = []
        cutoff_len = batch_info["cutoff_len"]

        for num_samples in micro_batch_sample_counts:
            samples = buffer.get(num_samples)
            packed_batch = _pack_padding_free_samples(samples, cutoff_len)
            if packed_batch is None:
                return None

            batch.append(packed_batch)

        return batch
