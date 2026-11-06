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
from enum import StrEnum, unique
from typing import TYPE_CHECKING, Any, Optional, TypedDict, Union

import fsspec
from datasets import DatasetDict, concatenate_datasets, interleave_datasets

from ..extras import logging


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from ..hparams import DataArguments


logger = logging.get_logger(__name__)


SLOTS = list[Union[str, set[str], dict[str, str]]]


@unique
class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


class DatasetModule(TypedDict):
    train_dataset: Optional[Union["Dataset", "IterableDataset"]]
    eval_dataset: Optional[Union["Dataset", "IterableDataset", dict[str, "Dataset"]]]


def merge_dataset(
    all_datasets: list[Union["Dataset", "IterableDataset"]], data_args: "DataArguments", seed: int
) -> Union["Dataset", "IterableDataset"]:
    r"""Merge multiple datasets to a unified dataset."""
    if len(all_datasets) == 1:
        return all_datasets[0]

    elif data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning_rank0_once("The samples between different datasets will not be mixed in streaming mode.")

        return concatenate_datasets(all_datasets)

    elif data_args.mix_strategy.startswith("interleave"):
        if not data_args.streaming:
            logger.warning_rank0_once("We recommend using `mix_strategy=concat` in non-streaming mode.")

        strategy_map: str = {
            "interleave_under": "first_exhausted",
            "interleave_over": "all_exhausted",
            "interleave_once": "all_exhausted_without_replacement",
        }[data_args.mix_strategy]

        return interleave_datasets(
            datasets=all_datasets,
            probabilities=data_args.interleave_probs,
            seed=seed,
            stopping_strategy=strategy_map,  # type: ignore
        )

    else:
        raise ValueError(f"Unknown mixing strategy: {data_args.mix_strategy}.")


def split_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    eval_dataset: Optional[Union["Dataset", "IterableDataset", dict[str, "Dataset"]]],
    data_args: "DataArguments",
    seed: int,
) -> tuple[dict, dict]:
    r"""Split the dataset and returns two dicts containing train set and validation set.

    Support both map dataset and iterable dataset.

    Returns:
        train_dict: Dictionary containing training data with key "train"
        eval_dict: Dictionary containing evaluation data with keys "validation" or "validation_{name}"
    """
    if eval_dataset is not None and data_args.val_size > 1e-6:
        raise ValueError("Cannot specify `val_size` if `eval_dataset` is not None.")

    # the train and eval better to in dict dtype and separately return for cpode clearly and good handle outside
    train_dict, eval_dict = {}, {}

    if dataset is not None:
        if data_args.streaming:
            dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)

        if data_args.val_size > 1e-6:
            if data_args.streaming:
                eval_dict["validation"] = dataset.take(int(data_args.val_size))
                train_dict["train"] = dataset.skip(int(data_args.val_size))
            else:
                val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
                split_result = dataset.train_test_split(test_size=val_size, seed=seed)
                train_dict["train"] = split_result["train"]
                eval_dict["validation"] = split_result["test"]
        else:
            train_dict["train"] = dataset

    if eval_dataset is not None:
        if isinstance(eval_dataset, dict):
            for name, data in eval_dataset.items():
                eval_dict[f"validation_{name}"] = data
        else:
            if data_args.streaming:
                eval_dataset = eval_dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)

            eval_dict["validation"] = eval_dataset

    return train_dict, eval_dict


def get_dataset_module(dataset: Union["Dataset", "DatasetDict"]) -> "DatasetModule":
    r"""Convert dataset or dataset dict to dataset module."""
    dataset_module: DatasetModule = {}
    if isinstance(dataset, DatasetDict):  # dataset dict
        if "train" in dataset:
            dataset_module["train_dataset"] = dataset["train"]

        if "validation" in dataset:
            dataset_module["eval_dataset"] = dataset["validation"]
        else:
            eval_dataset = {}
            for key in dataset.keys():
                if key.startswith("validation_"):
                    eval_dataset[key[len("validation_") :]] = dataset[key]

            if len(eval_dataset):
                dataset_module["eval_dataset"] = eval_dataset

    else:  # single dataset
        dataset_module["train_dataset"] = dataset

    return dataset_module


def setup_fs(path: str, anon: bool = False) -> "fsspec.AbstractFileSystem":
    r"""Set up a filesystem object based on the path protocol."""
    storage_options = {"anon": anon} if anon else {}
    if path.startswith("s3://"):
        fs = fsspec.filesystem("s3", **storage_options)
    elif path.startswith(("gs://", "gcs://")):
        fs = fsspec.filesystem("gcs", **storage_options)
    else:
        raise ValueError(f"Unsupported protocol in path: {path}. Use 's3://' or 'gs://'.")

    if not fs.exists(path):
        raise ValueError(f"Path does not exist: {path}.")

    return fs


def _read_json_with_fs(fs: "fsspec.AbstractFileSystem", path: str) -> list[Any]:
    r"""Helper function to read JSON/JSONL files using fsspec."""
    with fs.open(path, "r") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        else:
            return json.load(f)


def read_cloud_json(cloud_path: str) -> list[Any]:
    r"""Read a JSON/JSONL file from cloud storage (S3 or GCS).

    Args:
        cloud_path: str
            Cloud path in the format:
            - 's3://bucket-name/file.json' for AWS S3
            - 'gs://bucket-name/file.jsonl' or 'gcs://bucket-name/file.jsonl' for Google Cloud Storage
    """
    try:
        fs = setup_fs(cloud_path, anon=True)  # try with anonymous access first
    except Exception:
        fs = setup_fs(cloud_path)  # try again with credentials

    # filter out non-JSON files
    # "name" is the fsspec-standard field populated by every backend's listdir() (s3fs sets both
    # "Key" and "name"; gcsfs only sets "name"), unlike "Key" which is an S3-only convenience alias.
    files = [x["name"] for x in fs.listdir(cloud_path)] if fs.isdir(cloud_path) else [cloud_path]
    files = list(filter(lambda file: file.endswith(".json") or file.endswith(".jsonl"), files))
    if not files:
        raise ValueError(f"No JSON/JSONL files found in the specified path: {cloud_path}.")

    return sum([_read_json_with_fs(fs, file) for file in files], [])
