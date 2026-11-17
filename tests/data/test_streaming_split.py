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
from llamafactory.hparams import DataArguments


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
def test_streaming_validation_stays_held_out_across_epochs():
    dataset = Dataset.from_dict({"id": list(range(12))}).to_iterable_dataset()
    data_args = DataArguments(dataset="demo", streaming=True, val_size=3, buffer_size=12)
    train_dict, eval_dict = split_dataset(dataset, None, data_args, seed=42)
    train = train_dict["train"]
    validation = eval_dict["validation"]

    held_out = {row["id"] for row in validation}
    first_pass = [row["id"] for row in train]
    train.set_epoch(1)
    second_pass = [row["id"] for row in train]

    assert len(held_out) == 3
    assert len(first_pass) == len(second_pass) == 9
    assert held_out == {row["id"] for row in validation}
    assert set(first_pass) == set(second_pass) == set(range(12)) - held_out
    assert first_pass != second_pass
