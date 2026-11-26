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

import random

import pytest
from datasets import load_dataset

from llamafactory.v1.config.data_args import DataArguments
from llamafactory.v1.core.data_engine import DataEngine


@pytest.mark.parametrize("num_samples", [16])
def test_map_dataset(num_samples: int):
    data_args = DataArguments(train_dataset="llamafactory/v1-sft-demo")
    data_engine = DataEngine(data_args.train_dataset)
    original_data = load_dataset("llamafactory/v1-sft-demo", split="train")
    indexes = random.choices(range(len(data_engine)), k=num_samples)
    for index in indexes:
        print(data_engine[index])
        assert data_engine[index] == {"_dataset_name": "default", **original_data[index]}


@pytest.mark.parametrize("setting", ["size: 0", "weight: 0.0"])
def test_zero_size_or_weight_drops_dataset(tmp_path, setting: str):
    data_file = tmp_path / "data.jsonl"
    data_file.write_text('{"text": "a"}\n{"text": "b"}\n', encoding="utf-8")
    dataset_info = tmp_path / "dataset_info.yaml"
    dataset_info.write_text(
        f"keep:\n  path: {data_file}\n  source: local\noff:\n  path: {data_file}\n  source: local\n  {setting}\n",
        encoding="utf-8",
    )
    data_engine = DataEngine(str(dataset_info))
    assert len(data_engine) == 2
    assert all(data_engine[i]["_dataset_name"] == "keep" for i in range(len(data_engine)))
