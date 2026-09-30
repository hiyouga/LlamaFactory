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
