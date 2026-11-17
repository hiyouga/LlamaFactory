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

from datasets import Dataset

from llamafactory.data.data_utils import merge_dataset
from llamafactory.hparams.data_args import DataArguments


def test_interleave_once_contract() -> None:
    r"""interleave_once merges every row exactly once on newer datasets, or raises an actionable error on older ones."""
    sources = [Dataset.from_dict({"text": ["a"]}), Dataset.from_dict({"text": ["b", "c"]})]
    args = DataArguments(dataset="x,y", mix_strategy="interleave_once")
    try:
        merged = merge_dataset(sources, args, seed=42)
        assert sorted(row["text"] for row in merged) == ["a", "b", "c"]
    except ValueError as ex:
        assert "upgrade" in str(ex), "older datasets must raise an actionable error"


def test_concat_merge() -> None:
    sources = [Dataset.from_dict({"text": ["a"]}), Dataset.from_dict({"text": ["b", "c"]})]
    args = DataArguments(dataset="x,y", mix_strategy="concat")
    merged = merge_dataset(sources, args, seed=42)
    assert sorted(row["text"] for row in merged) == ["a", "b", "c"]
