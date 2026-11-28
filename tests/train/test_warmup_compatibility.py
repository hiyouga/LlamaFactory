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

from transformers import HfArgumentParser

from llamafactory.hparams.training_args import TrainingArguments


def test_training_args_warmup_ratio_dict():
    parser = HfArgumentParser(TrainingArguments)
    (args,) = parser.parse_dict({"output_dir": "saves/test", "warmup_ratio": 0.1})
    assert args.warmup_ratio == 0.1
    assert args.warmup_steps == 0.1
    assert args.get_warmup_steps(100) == 10


def test_training_args_warmup_ratio_cli():
    parser = HfArgumentParser(TrainingArguments)
    (args,) = parser.parse_args_into_dataclasses(["--output_dir", "saves/test", "--warmup_ratio", "0.15"])
    assert args.warmup_ratio == 0.15
    assert args.warmup_steps == 0.15
    assert args.get_warmup_steps(100) == 15


def test_training_args_warmup_steps_float():
    parser = HfArgumentParser(TrainingArguments)
    (args,) = parser.parse_dict({"output_dir": "saves/test", "warmup_steps": 0.1})
    assert args.warmup_ratio == 0.1
    assert args.warmup_steps == 0.1
    assert args.get_warmup_steps(100) == 10


def test_training_args_warmup_steps_int():
    parser = HfArgumentParser(TrainingArguments)
    (args,) = parser.parse_dict({"output_dir": "saves/test", "warmup_steps": 25})
    assert args.warmup_steps == 25
    assert args.get_warmup_steps(100) == 25


def test_training_args_warmup_steps_precedence():
    parser = HfArgumentParser(TrainingArguments)
    (args,) = parser.parse_dict({"output_dir": "saves/test", "warmup_ratio": 0.1, "warmup_steps": 30})
    assert args.warmup_steps == 30
    assert args.get_warmup_steps(100) == 30
