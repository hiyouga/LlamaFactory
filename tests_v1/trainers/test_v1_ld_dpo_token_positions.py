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

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.v1.trainers.dpo_trainer import DPOTrainer


def _reference(logits, labels, token_types, alpha):
    logps = logits[:, :-1].log_softmax(-1)
    outputs = []
    lengths = []
    for row in range(len(labels)):
        tokens = [
            [
                logps[row, position, label]
                for position, (label, kind) in enumerate(zip(labels[row, 1:], token_types[row, 1:]))
                if label != IGNORE_INDEX and kind == token_type
            ]
            for token_type in (1, 2)
        ]
        shared = min(map(len, tokens))
        outputs.append(
            torch.stack(
                [
                    sum(
                        score * (1 if alpha is None or index < shared else alpha)
                        for index, score in enumerate(sequence)
                    )
                    for sequence in tokens
                ]
            )
        )
        lengths.append([len(sequence) for sequence in tokens])
    sums = torch.stack(outputs)
    means = sums / (torch.tensor(lengths, device=logits.device) + 1e-6)
    return sums[:, 0], sums[:, 1], means[:, 0], means[:, 1]


def _inputs(layout):
    if layout == "contiguous":
        labels = [-100, 1, 2, 3, 4, -100, -100, -100, 1, 4, -100]
    elif layout == "internal-masks":
        labels = [-100, 1, -100, 2, 3, -100, 4, -100, 1, -100, 4]
    else:
        labels = [-100, -100, 1, -100, 2, 3, 4, -100, -100, 1, 4]
    labels = torch.tensor([labels, labels])
    kinds = torch.tensor([[1] * 7 + [2] * 4, [2] * 7 + [1] * 4])
    return labels, kinds


@pytest.mark.parametrize("layout", ["contiguous", "internal-masks", "different-prefix-lengths"])
@pytest.mark.parametrize("alpha", [None, 0.0, 0.25, 1.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_v1_ld_dpo_counts_only_supervised_tokens(layout, alpha, dtype):
    labels, token_types = _inputs(layout)
    original_labels = labels.clone()
    original_types = token_types.clone()
    torch.manual_seed(17)
    logits = torch.randn(*labels.shape, 7, dtype=dtype, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_()
    settings = SimpleNamespace(ld_alpha=alpha, device=torch.device("cpu"))
    actual = DPOTrainer._extract_chosen_rejected_logps(settings, logits, labels, token_types)
    expected = _reference(reference_logits, labels, token_types, alpha)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)
    sum(value.sum() for value in actual).backward()
    sum(value.sum() for value in expected).backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)
    torch.testing.assert_close(labels, original_labels)
    torch.testing.assert_close(token_types, original_types)


@pytest.mark.parametrize("use_ld", [False, True])
def test_v1_ld_dpo_with_native_causal_lm_updates(use_ld):
    torch.manual_seed(29)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=7, n_positions=16, n_embd=8, n_layer=1, n_head=2, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0
        )
    )
    reference = deepcopy(model)
    settings = SimpleNamespace(ld_alpha=0.3, device=torch.device("cpu"))
    labels, token_types = _inputs("internal-masks")
    input_ids = labels.masked_fill(labels == IGNORE_INDEX, 0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    for _ in range(2):
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        scores = DPOTrainer._extract_chosen_rejected_logps(
            settings, model(input_ids).logits, labels, token_types, use_ld=use_ld
        )
        expected_scores = _reference(reference(input_ids).logits, labels, token_types, 0.3 if use_ld else None)
        actual_loss = -F.logsigmoid(0.1 * (scores[0] - scores[1])).mean()
        expected_loss = -F.logsigmoid(0.1 * (expected_scores[0] - expected_scores[1])).mean()
        torch.testing.assert_close(actual_loss, expected_loss)
        actual_loss.backward()
        expected_loss.backward()
        for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, ref_parameter.grad)
        optimizer.step()
        reference_optimizer.step()
    for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter, ref_parameter)
