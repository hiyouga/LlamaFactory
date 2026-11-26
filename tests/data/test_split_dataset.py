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


@pytest.mark.parametrize("val_size", [3])
def test_streaming_val_split_stays_disjoint(val_size: int) -> None:
    source = Dataset.from_dict({"id": list(range(12))}).to_iterable_dataset()
    args = DataArguments(dataset="demo", streaming=True, val_size=val_size, buffer_size=12)
    train_dict, eval_dict = split_dataset(source, None, args, seed=42)
    train, validation = train_dict["train"], eval_dict["validation"]

    held_out = {row["id"] for row in validation}
    assert len(held_out) == val_size
    for epoch in range(3):
        train.set_epoch(epoch)
        train_ids = {row["id"] for row in train}
        assert held_out.isdisjoint(train_ids)


def test_map_split_unchanged() -> None:
    source = Dataset.from_dict({"id": list(range(12))})
    args = DataArguments(dataset="demo", streaming=False, val_size=0.25)
    train_dict, eval_dict = split_dataset(source, None, args, seed=42)
    assert len(train_dict["train"]) == 9
    assert len(eval_dict["validation"]) == 3
