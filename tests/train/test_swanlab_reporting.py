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

from types import SimpleNamespace

import pytest

from llamafactory.hparams.parser import _route_swanlab_reporting


def _args(report_to, use_swanlab=False):
    return SimpleNamespace(use_swanlab=use_swanlab), SimpleNamespace(report_to=report_to)


def test_report_to_swanlab_selects_the_native_callback():
    # The example configs offer `swanlab` as a report_to choice and none of them set
    # use_swanlab, so this is the path a user following them takes.
    finetuning_args, training_args = _args(["swanlab"])

    _route_swanlab_reporting(finetuning_args, training_args)

    assert finetuning_args.use_swanlab is True


def test_report_to_swanlab_as_a_bare_string_is_recognised():
    finetuning_args, training_args = _args("swanlab")

    _route_swanlab_reporting(finetuning_args, training_args)

    assert finetuning_args.use_swanlab is True


def test_swanlab_alongside_another_logger_still_selects_it():
    finetuning_args, training_args = _args(["tensorboard", "swanlab"])

    _route_swanlab_reporting(finetuning_args, training_args)

    assert finetuning_args.use_swanlab is True
    # The other logger is untouched here; report_to is trimmed later in get_train_args.
    assert training_args.report_to == ["tensorboard", "swanlab"]


@pytest.mark.parametrize("report_to", [["none"], ["tensorboard"], [], None])
def test_other_loggers_do_not_turn_swanlab_on(report_to):
    finetuning_args, training_args = _args(report_to)

    _route_swanlab_reporting(finetuning_args, training_args)

    assert finetuning_args.use_swanlab is False


def test_explicit_use_swanlab_is_left_alone():
    finetuning_args, training_args = _args(["tensorboard"], use_swanlab=True)

    _route_swanlab_reporting(finetuning_args, training_args)

    assert finetuning_args.use_swanlab is True
