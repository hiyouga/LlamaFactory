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

from llamafactory.hparams.generating_args import GeneratingArguments


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize("top_p", [0.0, -0.1, 1.5])
def test_top_p_out_of_range_is_rejected(top_p: float):
    with pytest.raises(ValueError, match="top_p"):
        GeneratingArguments(top_p=top_p)


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize("top_p", [1e-6, 0.1, 0.7, 1.0])
def test_top_p_in_range_is_preserved(top_p: float):
    assert GeneratingArguments(top_p=top_p).top_p == top_p


@pytest.mark.runs_on(["cpu", "mps"])
def test_zero_temperature_is_preserved():
    r"""`temperature=0.0` requests greedy decoding and must not be rewritten."""
    assert GeneratingArguments(temperature=0.0).temperature == 0.0


@pytest.mark.runs_on(["cpu", "mps"])
def test_to_dict_preserves_valid_top_p():
    assert GeneratingArguments(top_p=0.1).to_dict()["top_p"] == 0.1
