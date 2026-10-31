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

import pytest
from datasets.exceptions import DatasetNotFoundError

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


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
@pytest.mark.parametrize(
    "revision,expected_length",
    [
        ("5047f1f9e08e770f3b7608b99fefbb1ebf0ca81e", 298),
        ("69a1b759c08f6ceba7bedb2f1309461db04c829b", 300),
    ],
)
def test_load_dataset_revision(tmp_path, revision, expected_length):
    dataset_info = {
        "pinned": {
            "hf_hub_url": "llamafactory/tiny-supervised-dataset",
            "revision": revision,
        }
    }
    (tmp_path / "dataset_info.json").write_text(json.dumps(dataset_info))
    args = TRAIN_ARGS | {"dataset": "pinned", "dataset_dir": str(tmp_path), "use_cpu": True, "fp16": False}
    dataset_module = load_dataset_module(**args)
    assert len(dataset_module["train_dataset"]) == expected_length


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
def test_load_missing_dataset_revision(tmp_path):
    revision = "nonexistent-dataset-revision"
    dataset_info = {"pinned": {"hf_hub_url": "llamafactory/tiny-supervised-dataset", "revision": revision}}
    (tmp_path / "dataset_info.json").write_text(json.dumps(dataset_info))
    args = TRAIN_ARGS | {
        "dataset": "pinned",
        "dataset_dir": str(tmp_path),
        "cache_dir": str(tmp_path / "cache"),
        "use_cpu": True,
        "fp16": False,
    }
    with pytest.raises(DatasetNotFoundError, match=revision):
        load_dataset_module(**args)
