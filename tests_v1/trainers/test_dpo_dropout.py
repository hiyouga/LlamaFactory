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

from llamafactory.v1.trainers.dpo_trainer import DPOTrainer


def test_dpo_trainer_disables_dropout():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.Dropout(p=0.75),
        torch.nn.Sequential(torch.nn.Dropout(p=0.25)),
    )
    model.train()

    args = SimpleNamespace(
        cp_size=1,
        pref_loss="orpo",
        pref_beta=0.1,
        pref_ftx=0.0,
        simpo_gamma=0.0,
        ld_alpha=None,
        dpo_label_smoothing=0.0,
    )
    with patch("llamafactory.v1.trainers.dpo_trainer.BaseTrainer.__init__", return_value=None):
        DPOTrainer(args, model, renderer=None, train_dataset=None)

    assert model.training
    assert all(module.p == 0.0 for module in model.modules() if isinstance(module, torch.nn.Dropout))
    inputs = torch.ones(2, 4)
    torch.manual_seed(1)
    first = model(inputs)
    torch.manual_seed(2)
    second = model(inputs)
    torch.testing.assert_close(first, second)
