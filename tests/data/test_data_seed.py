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

"""Tests for data_seed support in the data loading pipeline.

Verifies that training_args.data_seed (from HuggingFace TrainingArguments)
is respected by LLaMA-Factory's data loading and splitting logic.
"""

import os

import pytest

from llamafactory.train.test_utils import load_dataset_module


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TINY_DATA = os.getenv("TINY_DATA", "llamafactory/tiny-supervised-dataset")

BASE_ARGS = {
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


@pytest.mark.runs_on(["cpu", "mps"])
def test_data_seed_not_set():
    """When data_seed is not set, data loading should work as before (uses seed)."""
    dataset_module = load_dataset_module(val_size=0.1, seed=42, **BASE_ARGS)
    assert dataset_module.get("train_dataset") is not None
    assert dataset_module.get("eval_dataset") is not None


@pytest.mark.runs_on(["cpu", "mps"])
def test_data_seed_explicit():
    """When data_seed is set, dataset loading should succeed."""
    dataset_module = load_dataset_module(val_size=0.1, seed=42, data_seed=123, **BASE_ARGS)
    assert dataset_module.get("train_dataset") is not None
    assert dataset_module.get("eval_dataset") is not None


@pytest.mark.runs_on(["cpu", "mps"])
def test_data_seed_reproducibility():
    """Same data_seed should produce identical train/eval splits."""
    module_a = load_dataset_module(val_size=0.1, seed=42, data_seed=99, **BASE_ARGS)
    module_b = load_dataset_module(val_size=0.1, seed=42, data_seed=99, **BASE_ARGS)

    train_a = list(module_a["train_dataset"])
    train_b = list(module_b["train_dataset"])
    assert len(train_a) == len(train_b)
    for a, b in zip(train_a, train_b):
        assert a["input_ids"] == b["input_ids"]


@pytest.mark.runs_on(["cpu", "mps"])
def test_different_data_seeds_differ():
    """Different data_seed values should produce different splits."""
    module_a = load_dataset_module(val_size=0.1, seed=42, data_seed=10, **BASE_ARGS)
    module_b = load_dataset_module(val_size=0.1, seed=42, data_seed=99, **BASE_ARGS)

    eval_a = list(module_a["eval_dataset"])
    eval_b = list(module_b["eval_dataset"])

    if len(eval_a) > 1 and len(eval_b) > 1:
        ids_a = [ex["input_ids"] for ex in eval_a]
        ids_b = [ex["input_ids"] for ex in eval_b]
        assert ids_a != ids_b, "Different data_seed values should produce different eval splits"


@pytest.mark.runs_on(["cpu", "mps"])
def test_data_seed_isolates_from_training_seed():
    """Changing training seed while keeping data_seed fixed should yield identical data splits."""
    module_a = load_dataset_module(val_size=0.1, seed=42, data_seed=77, **BASE_ARGS)
    module_b = load_dataset_module(val_size=0.1, seed=999, data_seed=77, **BASE_ARGS)

    train_a = list(module_a["train_dataset"])
    train_b = list(module_b["train_dataset"])
    assert len(train_a) == len(train_b)
    for a, b in zip(train_a, train_b):
        assert a["input_ids"] == b["input_ids"]
