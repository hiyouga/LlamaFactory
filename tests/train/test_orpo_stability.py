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

import copy
from decimal import Decimal, localcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel, Seq2SeqTrainingArguments

from llamafactory.hparams import FinetuningArguments
from llamafactory.train.dpo.trainer import CustomDPOTrainer
from llamafactory.train.trainer_utils import get_batch_logps
from llamafactory.v1.trainers.dpo_trainer import DPOTrainer


def _loss(implementation, chosen, rejected, beta=0.1):
    if implementation == "legacy":
        return CustomDPOTrainer.odds_ratio_loss(SimpleNamespace(beta=beta), chosen, rejected)
    return DPOTrainer._odds_ratio_loss(SimpleNamespace(pref_beta=beta), chosen, rejected)


def _decimal_reference(chosen, rejected, beta):
    # Evaluate the probability-domain definition and its derivatives independently
    # of the implementation's log-domain operations.
    with localcontext() as context:
        context.prec = 80
        chosen = Decimal.from_float(chosen)
        rejected = Decimal.from_float(rejected)
        beta = Decimal.from_float(beta)
        chosen_prob = chosen.exp()
        rejected_prob = rejected.exp()
        odds = chosen_prob / (1 - chosen_prob) / (rejected_prob / (1 - rejected_prob))
        loss = -chosen + beta * (1 + 1 / odds).ln()
        chosen_grad = -1 - beta / ((1 + odds) * (1 - chosen_prob))
        rejected_grad = beta / ((1 + odds) * (1 - rejected_prob))
        return [float(value) for value in (loss, chosen_grad, rejected_grad)]


@pytest.mark.parametrize("implementation", ["legacy", "v1"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("case", ["chosen_near_one", "rejected_near_one", "both_near_one", "ordinary", "small_probs"])
def test_orpo_loss_and_gradients_match_high_precision(implementation, dtype, case):
    epsilon = 1e-8 if dtype == torch.float32 else 1e-18
    chosen_value, rejected_value = {
        "chosen_near_one": (-epsilon, -2.0),
        "rejected_near_one": (-2.0, -epsilon),
        "both_near_one": (-epsilon, -2 * epsilon),
        "ordinary": (-1.0, -2.0),
        "small_probs": (-1000.0, -999.0),
    }[case]
    chosen = torch.tensor([chosen_value], dtype=dtype, requires_grad=True)
    rejected = torch.tensor([rejected_value], dtype=dtype, requires_grad=True)
    expected = _decimal_reference(chosen.item(), rejected.item(), 0.1)
    loss = _loss(implementation, chosen, rejected)
    loss.sum().backward()
    actual = torch.stack((loss.detach(), chosen.grad, rejected.grad)).flatten()
    assert torch.isfinite(actual).all()
    tolerance = 5e-6 if dtype == torch.float32 else 1e-13
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=dtype), rtol=tolerance, atol=tolerance)


def _double_precision_loss(chosen, rejected, beta):
    chosen, rejected = chosen.double(), rejected.double()
    log_odds = (chosen - rejected) - (torch.log1p(-chosen.exp()) - torch.log1p(-rejected.exp()))
    return -chosen - beta * F.logsigmoid(log_odds)


@pytest.mark.parametrize("implementation", ["legacy", "v1"])
@pytest.mark.parametrize("near_one", ["chosen", "rejected", "both"])
def test_orpo_training_with_near_one_sequence_probabilities(implementation, near_one):
    # Long, confident responses can have a nonzero average log probability that
    # is much closer to zero than any individual token's log probability.
    sequence_length = 129
    policy = torch.nn.Embedding(2 * sequence_length, 2)
    with torch.no_grad():
        policy.weight.zero_()
        weights = policy.weight.view(2, sequence_length, 2)
        if near_one in ("chosen", "both"):
            weights[0, :, 0] = 30.0
            weights[0, 0, 0] = 14.0
        if near_one in ("rejected", "both"):
            weights[1, :, 0] = 30.0
            weights[1, 0, 0] = 13.0
    reference = copy.deepcopy(policy)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    input_ids = torch.arange(2 * sequence_length).view(2, sequence_length)
    labels = torch.zeros_like(input_ids)
    labels[:, 0] = -100

    for _ in range(2):
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        logps, lengths = get_batch_logps(policy(input_ids), labels)
        averaged = logps / lengths
        assert (averaged < 0).all()
        assert (averaged.exp() == 1).any()  # Reproduce the float32 cancellation.
        loss = _loss(implementation, averaged[:1], averaged[1:]).mean()
        ref_logps, ref_lengths = get_batch_logps(reference(input_ids), labels)
        ref_averaged = ref_logps / ref_lengths
        reference_loss = _double_precision_loss(ref_averaged[:1], ref_averaged[1:], 0.1).mean()
        assert torch.isfinite(loss)
        torch.testing.assert_close(loss.double(), reference_loss, rtol=5e-6, atol=1e-7)
        loss.backward()
        reference_loss.backward()
        assert torch.isfinite(policy.weight.grad).all()
        torch.testing.assert_close(policy.weight.grad, reference.weight.grad, rtol=1e-5, atol=1e-7)
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(policy.weight, reference.weight, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("beta", [0.1, 1.0])
def test_orpo_trainer_preserves_normal_training(tmp_path, beta):
    torch.manual_seed(123)
    policy = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=7,
            n_positions=8,
            n_embd=8,
            n_layer=1,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )
    reference = copy.deepcopy(policy)
    trainer = CustomDPOTrainer(
        model=policy,
        ref_model=None,
        finetuning_args=FinetuningArguments(stage="dpo", pref_loss="orpo", pref_beta=beta),
        processor=None,
        tokenizer=None,
        args=Seq2SeqTrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none"),
    )
    input_ids = torch.tensor([[1, 2, 3, 4], [1, 3, 2, 5]])
    labels = input_ids.clone()
    labels[:, :2] = -100
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)

    for _ in range(2):
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        loss, metrics = trainer.get_batch_loss_metrics(
            policy, {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": labels.clone()}
        )
        logits = reference(input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False).logits
        logps, lengths = get_batch_logps(logits.float(), labels)
        averaged = logps / lengths
        expected_loss = _double_precision_loss(averaged[:1], averaged[1:], beta).mean()
        torch.testing.assert_close(loss.double(), expected_loss, rtol=1e-6, atol=1e-7)
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
        loss.backward()
        expected_loss.backward()
        for actual, expected in zip(policy.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7)
        optimizer.step()
        reference_optimizer.step()
        for actual, expected in zip(policy.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
