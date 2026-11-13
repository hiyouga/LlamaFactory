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
from datasets import __version__ as datasets_version
from packaging.version import Version

from llamafactory.data.data_utils import merge_dataset
from llamafactory.hparams import DataArguments


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("probabilities", [None, "0.7,0.3"])
def test_interleave_once(streaming, probabilities):
    sources = [Dataset.from_dict({"id": [0, 1]}), Dataset.from_dict({"id": [2, 3, 4]})]
    if streaming:
        sources = [dataset.to_iterable_dataset() for dataset in sources]

    data_args = DataArguments(
        dataset="a,b", mix_strategy="interleave_once", interleave_probs=probabilities, streaming=streaming
    )
    if Version(datasets_version) < Version("4.8.5"):
        with pytest.raises(ImportError, match=r"datasets>=4\.8\.5"):
            merge_dataset(sources, data_args, seed=42)
    else:
        result = merge_dataset(sources, data_args, seed=42)
        assert sorted(row["id"] for row in result) == list(range(5))


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
def test_interleave_once_rejects_zero_probability():
    sources = [Dataset.from_dict({"id": [0]}), Dataset.from_dict({"id": [1]})]
    data_args = DataArguments(dataset="a,b", mix_strategy="interleave_once", interleave_probs="1,0")

    with pytest.raises(ValueError, match="positive sampling probability"):
        merge_dataset(sources, data_args, seed=42)


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
def test_interleave_once_rejects_weighted_empty_map_style_dataset():
    source = Dataset.from_dict({"id": [0, 1]})
    data_args = DataArguments(dataset="a,b", mix_strategy="interleave_once", interleave_probs="0.7,0.3")

    with pytest.raises(ValueError, match="nonempty map-style datasets"):
        merge_dataset([source.select([]), source], data_args, seed=42)
