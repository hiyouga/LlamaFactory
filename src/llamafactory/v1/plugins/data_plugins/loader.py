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
import random
from typing import Any, Literal

from datasets import load_dataset

from ...utils.plugin import BasePlugin
from ...utils.types import DatasetInfo, HFDataset


class DataLoaderPlugin(BasePlugin):
    """Plugin for loading dataset."""

    def load(self, dataset_info: DatasetInfo) -> HFDataset:
        path = dataset_info["path"]
        split = dataset_info.get("split", "train")
        streaming = dataset_info.get("streaming", False)
        return super().__call__(path, split, streaming)


def _get_builder_name(path: str) -> Literal["arrow", "csv", "json", "parquet", "text"]:
    """Get dataset builder name.

    Args:
        path (str): Dataset path.

    Returns:
        Literal["arrow", "csv", "json", "parquet", "text"]: Dataset builder name.
    """
    filetype = os.path.splitext(path)[-1][1:]
    if filetype in ["arrow", "csv", "json", "jsonl", "parquet", "txt"]:
        return filetype.replace("jsonl", "json").replace("txt", "text")
    else:
        raise ValueError(f"Unknown dataset filetype: {filetype}.")


@DataLoaderPlugin("local").register()
def load_data_from_file(filepath: str, split: str, streaming: bool) -> HFDataset:
    if os.path.isdir(filepath):
        dataset = load_dataset(os.path.abspath(filepath), split=split, trust_remote_code=False)
    elif os.path.isfile(filepath):
        filetype = _get_builder_name(filepath)
        dataset = load_dataset(filetype, data_files=filepath, split=split)
    else:
        raise ValueError(f"Can not load dataset from {filepath}.")

    if streaming:  # faster when data is streamed from local files
        dataset = dataset.to_iterable_dataset()

    return dataset


def adjust_data_index(
    data_index: list[tuple[str, int]], size: int | None, weight: float | None
) -> list[tuple[str, int]]:
    """Adjust dataset index by size and weight.

    Args:
        data_index (list[tuple[str, int]]): List of (dataset_name, sample_index).
        size (Optional[int]): Desired dataset size.
        weight (Optional[float]): Desired dataset weight.

    Returns:
        list[tuple[str, int]]: Adjusted dataset index.
    """
    if size is not None:
        data_index = random.choices(data_index, k=size)

    if weight is not None:
        data_index = random.choices(data_index, k=int(len(data_index) * weight))

    return data_index


def select_data_sample(
    data_index: list[tuple[str, int]], index: slice | list[int] | Any
) -> tuple[str, int] | list[tuple[str, int]]:
    """Select dataset samples.

    Args:
        data_index (list[tuple[str, int]]): List of (dataset_name, sample_index).
        index (Union[slice, list[int], Any]): Index of dataset samples.

    Returns:
        Union[tuple[str, int], list[tuple[str, int]]]: Selected dataset samples.
    """
    if isinstance(index, slice):
        return [data_index[i] for i in range(*index.indices(len(data_index)))]
    elif isinstance(index, list):
        return [data_index[i] for i in index]
    else:
        raise ValueError(f"Invalid index type {type(index)}.")
