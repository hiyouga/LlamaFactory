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

from dataclasses import dataclass

import pytest
from transformers import HfArgumentParser

from llamafactory.hparams import DataArguments
from llamafactory.hparams.parser import _parse_args


@dataclass
class V5Args:
    warmup_steps: float = 0


@dataclass
class V4Args:
    warmup_ratio: float = 0.0
    warmup_steps: int = 0


def test_warmup_ratio_maps_to_warmup_steps():
    (args,) = _parse_args(HfArgumentParser(V5Args), {"warmup_ratio": 0.1})
    assert args.warmup_steps == 0.1


def test_warmup_steps_wins_over_warmup_ratio():
    (args,) = _parse_args(HfArgumentParser(V5Args), {"warmup_ratio": 0.1, "warmup_steps": 50})
    assert args.warmup_steps == 50


def test_warmup_ratio_kept_when_supported():
    (args,) = _parse_args(HfArgumentParser(V4Args), {"warmup_ratio": 0.1})
    assert args.warmup_ratio == 0.1
    assert args.warmup_steps == 0


@pytest.mark.parametrize(
    ("dataset", "eval_dataset", "eval_on_each_dataset"),
    [
        ("wind,solar", None, False),
        ("wind,solar", "holdout", False),
        ("wind,solar", "holdout", True),
        ("wind,solar", "dev_a,dev_b,dev_c", True),
        ("wind,solar", "dev_a,dev_b", False),
        ("wind", "dev_a,dev_b", False),
    ],
)
def test_interleave_probs_for_merged_datasets(dataset, eval_dataset, eval_on_each_dataset):
    args = DataArguments(
        dataset=dataset,
        eval_dataset=eval_dataset,
        eval_on_each_dataset=eval_on_each_dataset,
        mix_strategy="interleave_under",
        interleave_probs="0.7,0.3",
    )
    assert args.interleave_probs == [0.7, 0.3]


@pytest.mark.parametrize(
    ("dataset", "eval_dataset", "eval_on_each_dataset", "mix_strategy", "error"),
    [
        ("wind,solar,tidal", "holdout", True, "interleave_under", "length of dataset"),
        ("wind,solar", "dev_a,dev_b,dev_c", False, "interleave_under", "length of eval dataset"),
        ("wind,solar", "holdout", True, "concat", "only valid for interleaved mixing"),
    ],
)
def test_invalid_interleave_probs(dataset, eval_dataset, eval_on_each_dataset, mix_strategy, error):
    with pytest.raises(ValueError, match=error):
        DataArguments(
            dataset=dataset,
            eval_dataset=eval_dataset,
            eval_on_each_dataset=eval_on_each_dataset,
            mix_strategy=mix_strategy,
            interleave_probs="0.7,0.3",
        )
