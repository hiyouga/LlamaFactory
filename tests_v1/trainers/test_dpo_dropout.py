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

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers import PretrainedConfig

from llamafactory.v1.config import get_args
from llamafactory.v1.trainers.dpo_trainer import DPOTrainer
from llamafactory.v1.trainers.rm_trainer import RMTrainer


class _FunctionalDropout(torch.nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.attention_dropout = p

    def forward(self, inputs):
        return F.dropout(inputs, p=self.attention_dropout, training=self.training)


class _DropoutModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = PretrainedConfig(attention_dropout=0.5)
        self.config.text_config = PretrainedConfig(hidden_dropout=0.25)
        self.linear = torch.nn.Linear(4, 4)
        self.module_dropout = torch.nn.Dropout(p=0.75)
        self.functional_dropout = _FunctionalDropout(p=0.5)

    def forward(self, inputs):
        return self.functional_dropout(self.module_dropout(self.linear(inputs)))


def _get_args(disable_dropout: bool = True):
    return SimpleNamespace(
        cp_size=1,
        disable_dropout=disable_dropout,
        pref_loss="orpo",
        pref_beta=0.1,
        pref_ftx=0.0,
        simpo_gamma=0.0,
        ld_alpha=None,
        dpo_label_smoothing=0.0,
    )


def test_dpo_trainer_disables_dropout():
    model = _DropoutModel()
    model.train()

    with patch("llamafactory.v1.trainers.dpo_trainer.BaseTrainer.__init__", return_value=None):
        DPOTrainer(_get_args(), model, renderer=None, train_dataset=None)

    assert model.training
    assert model.module_dropout.p == 0.0
    assert model.functional_dropout.attention_dropout == 0.0
    assert model.config.attention_dropout == 0.5
    assert model.config.text_config.hidden_dropout == 0.25
    inputs = torch.ones(2, 4)
    torch.manual_seed(1)
    first = model(inputs)
    torch.manual_seed(2)
    second = model(inputs)
    torch.testing.assert_close(first, second)


def test_rm_trainer_disables_dropout():
    model = _DropoutModel()

    with patch("llamafactory.v1.trainers.rm_trainer.BaseTrainer.__init__", return_value=None):
        RMTrainer(_get_args(), model, renderer=None, train_dataset=None)

    assert model.module_dropout.p == 0.0
    assert model.functional_dropout.attention_dropout == 0.0


def test_preference_trainers_can_keep_dropout_enabled():
    dpo_model = _DropoutModel()
    rm_model = _DropoutModel()

    with (
        patch("llamafactory.v1.trainers.dpo_trainer.BaseTrainer.__init__", return_value=None),
        patch("llamafactory.v1.trainers.rm_trainer.BaseTrainer.__init__", return_value=None),
    ):
        DPOTrainer(_get_args(disable_dropout=False), dpo_model, renderer=None, train_dataset=None)
        RMTrainer(_get_args(disable_dropout=False), rm_model, renderer=None, train_dataset=None)

    assert dpo_model.module_dropout.p == 0.75
    assert dpo_model.functional_dropout.attention_dropout == 0.5
    assert rm_model.module_dropout.p == 0.75
    assert rm_model.functional_dropout.attention_dropout == 0.5


def test_training_arguments_parser_exposes_dropout_control():
    assert get_args({})[2].disable_dropout
    assert not get_args({"disable_dropout": False})[2].disable_dropout
