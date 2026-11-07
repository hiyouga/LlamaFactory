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

import pytest
from datasets import Dataset, DatasetDict, IterableDataset

from llamafactory.data.loader import get_dataset
from llamafactory.hparams import DataArguments
from llamafactory.train.test_utils import load_dataset_module


DEMO_DATA = os.getenv("DEMO_DATA", "llamafactory/demo_data")

TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TINY_DATA = os.getenv("TINY_DATA", "llamafactory/tiny-supervised-dataset")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "full",
    "template": "llama3",
    "dataset": TINY_DATA,
    "dataset_dir": "ONLINE",
    "cutoff_len": 8192,
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "fp16": True,
}


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
def test_load_train_only():
    dataset_module = load_dataset_module(**TRAIN_ARGS)
    assert dataset_module.get("train_dataset") is not None
    assert dataset_module.get("eval_dataset") is None


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
def test_load_val_size():
    dataset_module = load_dataset_module(val_size=0.1, **TRAIN_ARGS)
    assert dataset_module.get("train_dataset") is not None
    assert dataset_module.get("eval_dataset") is not None


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
def test_load_eval_data():
    dataset_module = load_dataset_module(eval_dataset=TINY_DATA, **TRAIN_ARGS)
    assert dataset_module.get("train_dataset") is not None
    assert dataset_module.get("eval_dataset") is not None


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "splits", [("validation",), ("validation_a", "validation_b"), ("train",), ("train", "validation")]
)
def test_load_tokenized_splits(tmp_path, streaming, splits):
    data = Dataset.from_dict({"input_ids": [[1, 2]], "attention_mask": [[1, 1]], "labels": [[1, 2]]})
    DatasetDict(dict.fromkeys(splits, data)).save_to_disk(str(tmp_path))
    data_args = DataArguments(tokenized_path=str(tmp_path), streaming=streaming)

    dataset_module = get_dataset(None, None, data_args, None, "sft", None)
    if "train" in splits:
        train_dataset = dataset_module["train_dataset"]
        assert isinstance(train_dataset, IterableDataset if streaming else Dataset)
        assert list(train_dataset) == list(data)
    else:
        assert "train_dataset" not in dataset_module

    eval_dataset = dataset_module.get("eval_dataset")
    if "validation" in splits:
        assert isinstance(eval_dataset, Dataset)
        assert list(eval_dataset) == list(data)
    elif "validation_a" in splits:
        assert set(eval_dataset) == {"a", "b"}
        assert all(isinstance(dataset, Dataset) and list(dataset) == list(data) for dataset in eval_dataset.values())
    else:
        assert eval_dataset is None
