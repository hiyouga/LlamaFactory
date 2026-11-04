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

import pytest
from datasets import Dataset
from transformers import HfArgumentParser

from llamafactory.data.data_utils import split_dataset
from llamafactory.hparams import DataArguments


@pytest.mark.parametrize(("val_size", "expected_count"), [("1", 1), ("1.0", 1), ("2", 2), ("0.2", 2)])
def test_split_dataset_cli_validation_size(val_size, expected_count):
    data_args = HfArgumentParser(DataArguments).parse_args_into_dataclasses(
        ["--dataset", "local", "--val_size", val_size]
    )[0]
    dataset = Dataset.from_dict({"id": list(range(10))})

    train, evaluation = split_dataset(dataset, None, data_args, seed=42)

    assert len(train["train"]) == 10 - expected_count
    assert len(evaluation["validation"]) == expected_count
    assert set(train["train"]["id"]).isdisjoint(evaluation["validation"]["id"])
    assert set(train["train"]["id"]) | set(evaluation["validation"]["id"]) == set(range(10))


@pytest.mark.parametrize("val_size", [1, 1.0])
@pytest.mark.parametrize("streaming", [False, True])
def test_split_dataset_integer_validation_size(streaming, val_size):
    data_args = DataArguments(dataset="local", val_size=val_size, streaming=streaming)
    dataset = Dataset.from_dict({"id": list(range(10))})
    if streaming:
        dataset = dataset.to_iterable_dataset()

    train, evaluation = split_dataset(dataset, None, data_args, seed=42)

    assert len(list(train["train"])) == 9
    assert len(list(evaluation["validation"])) == 1
