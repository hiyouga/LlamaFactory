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
from datasets import Dataset, Features, Sequence, Value

from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
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


def test_sft_features_stabilize_mixed_modality_batches():
    source_features = Features(
        {
            "_prompt": Value("string"),
            "_images": Sequence(Value("string")),
            "_videos": Sequence(Value("string")),
            "_audios": Sequence(Value("string")),
        }
    )
    dataset = Dataset.from_dict(
        {
            "_prompt": ["text 1", "text 2", "image", "video"],
            "_images": [None, None, ["image.png"], None],
            "_videos": [None, None, None, ["video.mp4"]],
            "_audios": [None, None, None, None],
        },
        features=source_features,
    )

    def preprocess_dataset(examples):
        batch_size = len(examples["_prompt"])
        return {
            "input_ids": [[1] for _ in range(batch_size)],
            "attention_mask": [[1] for _ in range(batch_size)],
            "labels": [[1] for _ in range(batch_size)],
            "images": examples["_images"],
            "videos": examples["_videos"],
            "audios": examples["_audios"],
        }

    processed_dataset = dataset.map(
        preprocess_dataset,
        batched=True,
        batch_size=2,
        remove_columns=dataset.column_names,
        features=SupervisedDatasetProcessor.get_dataset_features(dataset),
    )

    assert processed_dataset.features["images"] == source_features["_images"]
    assert processed_dataset.features["videos"] == source_features["_videos"]
    assert processed_dataset.features["labels"] == Sequence(Value("int64"))
    assert processed_dataset["images"][0] is None
    assert processed_dataset["videos"][0] is None
    assert processed_dataset["images"][2] == ["image.png"]
    assert processed_dataset["videos"][3] == ["video.mp4"]


def test_sft_features_require_aligned_media_columns():
    dataset = Dataset.from_dict({"_prompt": ["text"]})

    with pytest.raises(ValueError, match="_images, _videos, _audios"):
        SupervisedDatasetProcessor.get_dataset_features(dataset)


@pytest.mark.runs_on(["cpu", "mps"])
def test_load_train_only():
    dataset_module = load_dataset_module(**TRAIN_ARGS)
    assert dataset_module.get("train_dataset") is not None
    assert dataset_module.get("eval_dataset") is None


@pytest.mark.runs_on(["cpu", "mps"])
def test_load_val_size():
    dataset_module = load_dataset_module(val_size=0.1, **TRAIN_ARGS)
    assert dataset_module.get("train_dataset") is not None
    assert dataset_module.get("eval_dataset") is not None


@pytest.mark.runs_on(["cpu", "mps"])
def test_load_eval_data():
    dataset_module = load_dataset_module(eval_dataset=TINY_DATA, **TRAIN_ARGS)
    assert dataset_module.get("train_dataset") is not None
    assert dataset_module.get("eval_dataset") is not None
