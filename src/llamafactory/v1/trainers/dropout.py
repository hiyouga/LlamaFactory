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


import torch

from ..utils.types import HFModel


_DROPOUT_ATTRIBUTES = (
    "activation_dropout",
    "attention_dropout",
    "attention_probs_dropout_prob",
    "attn_pdrop",
    "classifier_dropout",
    "dropout",
    "dropout_rate",
    "embd_pdrop",
    "hidden_dropout",
    "hidden_dropout_prob",
    "resid_pdrop",
    "summary_first_dropout",
)
_DROPOUT_MODULES = (
    torch.nn.Dropout,
    torch.nn.Dropout1d,
    torch.nn.Dropout2d,
    torch.nn.Dropout3d,
    torch.nn.AlphaDropout,
    torch.nn.FeatureAlphaDropout,
)


def _zero_dropout_attributes(obj) -> None:
    for name in _DROPOUT_ATTRIBUTES:
        value = getattr(obj, name, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            setattr(obj, name, 0.0)


def disable_dropout_in_model(model: HFModel) -> None:
    r"""Disable module and functional dropout paths used by Transformers models."""
    for module in model.modules():
        if isinstance(module, _DROPOUT_MODULES):
            module.p = 0.0

        _zero_dropout_attributes(module)
