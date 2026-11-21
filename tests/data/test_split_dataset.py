# Copyright 2026 the LlamaFactory team.
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


@pytest.mark.runs_on(["cpu", "cuda", "mps"])
def test_streaming_val_size_no_epoch_overlap():
    total_samples = 20
    val_size = 5
    buffer_size = 20
    source = Dataset.from_dict({"id": list(range(total_samples))}).to_iterable_dataset()
    args = DataArguments(dataset="demo", streaming=True, val_size=val_size, buffer_size=buffer_size)

    train_dict, eval_dict = split_dataset(source, None, args, seed=42)
    train_ds = train_dict["train"]
    val_ds = eval_dict["validation"]

    val_ids = {row["id"] for row in val_ds}
    assert len(val_ids) == val_size, f"Expected {val_size} validation samples, got {len(val_ids)}"

    epoch_orders = []
    for epoch in range(3):
        train_ds.set_epoch(epoch)
        train_ids = [row["id"] for row in train_ds]
        epoch_orders.append(train_ids)

        assert len(train_ids) == total_samples - val_size
        overlap = val_ids & set(train_ids)
        assert len(overlap) == 0, f"Epoch {epoch} leaked validation samples into train: {overlap}"

    # Verify that different epochs produce different shuffle permutations within train partition
    assert epoch_orders[0] != epoch_orders[1], "Expected different shuffle orders across epochs"


@pytest.mark.runs_on(["cpu", "cuda", "mps"])
def test_streaming_no_val_size():
    total_samples = 10
    source = Dataset.from_dict({"id": list(range(total_samples))}).to_iterable_dataset()
    args = DataArguments(dataset="demo", streaming=True, val_size=0.0, buffer_size=10)

    train_dict, eval_dict = split_dataset(source, None, args, seed=42)
    assert "train" in train_dict
    assert "validation" not in eval_dict

    train_ids = [row["id"] for row in train_dict["train"]]
    assert len(train_ids) == total_samples
