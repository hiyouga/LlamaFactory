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

from llamafactory.data.data_utils import split_dataset
from llamafactory.hparams.data_args import DataArguments


@pytest.mark.parametrize(("val_size", "expected_count"), [(1.0, 1), (2.0, 2), (0.25, 3)])
def test_split_dataset_honors_counts_and_ratios(val_size, expected_count):
    dataset = Dataset.from_dict({"id": list(range(10))})
    train, validation = split_dataset(dataset, None, DataArguments(dataset="fixture", val_size=val_size), seed=42)

    train_ids = set(train["train"]["id"])
    eval_ids = set(validation["validation"]["id"])
    assert len(eval_ids) == expected_count
    assert train_ids.isdisjoint(eval_ids)
    assert train_ids | eval_ids == set(range(10))


@pytest.mark.parametrize("val_size", [-1.0, 1.5, float("nan"), float("inf")])
def test_rejects_invalid_validation_size(val_size):
    with pytest.raises(ValueError, match="val_size"):
        DataArguments(dataset="fixture", val_size=val_size)
