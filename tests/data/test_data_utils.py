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

from unittest.mock import patch

import pytest
from datasets import Dataset

from llamafactory.data.data_utils import merge_dataset
from llamafactory.hparams.data_args import DataArguments


def test_merge_dataset_interleave_once():
    sources = [
        Dataset.from_dict({"text": ["a"]}),
        Dataset.from_dict({"text": ["b", "c"]}),
    ]
    args = DataArguments(dataset="a,b", mix_strategy="interleave_once")
    merged = merge_dataset(sources, args, seed=42)
    assert sorted([row["text"] for row in merged]) == ["a", "b", "c"]


def test_merge_dataset_interleave_once_streaming():
    sources = [
        Dataset.from_dict({"text": ["a"]}).to_iterable_dataset(),
        Dataset.from_dict({"text": ["b", "c"]}).to_iterable_dataset(),
    ]
    args = DataArguments(dataset="a,b", mix_strategy="interleave_once", streaming=True)
    merged = merge_dataset(sources, args, seed=42)
    assert sorted([row["text"] for row in merged]) == ["a", "b", "c"]


def test_merge_dataset_interleave_once_version_check():
    sources = [
        Dataset.from_dict({"text": ["a"]}),
        Dataset.from_dict({"text": ["b", "c"]}),
    ]
    args = DataArguments(dataset="a,b", mix_strategy="interleave_once")
    with patch("llamafactory.data.data_utils.check_version") as mock_check:
        mock_check.side_effect = ImportError("datasets>=4.6.0 is required")
        with pytest.raises(ImportError, match="datasets>=4.6.0 is required"):
            merge_dataset(sources, args, seed=42)
