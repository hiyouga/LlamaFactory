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
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, TrainingArguments

from llamafactory.hparams import FinetuningArguments
from llamafactory.train.dpo.trainer import CustomDPOTrainer


@pytest.mark.parametrize("loss_type", ["sigmoid", "hinge", "ipo"])
def test_preference_loss_matches_requested_objective(loss_type, tmp_path):
    config = GPT2Config(
        vocab_size=16,
        n_positions=16,
        n_embd=8,
        n_layer=1,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = GPT2LMHeadModel(config).double()
        torch.manual_seed(23)
        reference = GPT2LMHeadModel(config).double()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(16)}, unk_token="0")),
        unk_token="0",
        pad_token="0",
        eos_token="2",
    )
    trainer = CustomDPOTrainer(
        model=model,
        ref_model=reference,
        finetuning_args=FinetuningArguments(stage="dpo", finetuning_type="full", pref_loss=loss_type, pref_beta=0.1),
        processor=None,
        tokenizer=tokenizer,
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[]),
    )
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 6, 7, 0], [1, 2, 8, 9, 0], [1, 2, 11, 12, 13]])
    attention_mask = input_ids.ne(0)
    labels = input_ids.masked_fill(~attention_mask, -100)
    labels[:, :2] = -100
    batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    actual = trainer.compute_loss(model, batch)
    actual_gradients = torch.autograd.grad(actual, tuple(model.parameters()))

    # Calculate the requested objective directly from token log probabilities.
    response_mask = labels[:, 1:].ne(-100)
    targets = labels[:, 1:].masked_fill(~response_mask, 0)

    def response_logps(current_model):
        logits = current_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()
        token_logps = logits[:, :-1].log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        logps = (token_logps * response_mask).sum(-1)
        if loss_type == "ipo":
            logps = logps / response_mask.sum(-1)
        return logps.chunk(2)

    chosen, rejected = response_logps(model)
    with torch.no_grad():
        ref_chosen, ref_rejected = response_logps(reference)
    logratios = (chosen - rejected) - (ref_chosen - ref_rejected)
    beta = trainer.beta
    if loss_type == "sigmoid":
        expected = -F.logsigmoid(beta * logratios).mean()
    elif loss_type == "hinge":
        expected = torch.relu(1 - beta * logratios).mean()
    else:
        expected = ((logratios - 1 / (2 * beta)) ** 2).mean()
    expected_gradients = torch.autograd.grad(expected, tuple(model.parameters()))

    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)
