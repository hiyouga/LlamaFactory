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

from copy import deepcopy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llamafactory.hparams import FinetuningArguments, TrainingArguments
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.train.trainer_utils import asft_loss_func


@pytest.fixture
def asft_trainer(tmp_path):
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            use_cache=False,
        )
    )
    ref_model = deepcopy(model).requires_grad_(False)
    return CustomSeq2SeqTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=None,
        processor=None,
        finetuning_args=FinetuningArguments(finetuning_type="full", use_asft_loss=True),
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none", disable_tqdm=True),
    )


@pytest.fixture
def asft_batch():
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4], [1, 5, 6, 7]]),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 3, 4], [-100, 5, 6, 7]]),
    }


@pytest.mark.parametrize("return_outputs", [False, True])
def test_asft_compute_loss(asft_trainer, asft_batch, return_outputs):
    trainer = asft_trainer
    with torch.no_grad():
        expected_outputs = trainer.model(**asft_batch)
        ref_outputs = trainer.ref_model(**asft_batch)
        expected_loss = asft_loss_func(expected_outputs, asft_batch["labels"], ref_outputs.logits)

    result = trainer.compute_loss(trainer.model, asft_batch, return_outputs=return_outputs)
    if return_outputs:
        loss, outputs = result
        torch.testing.assert_close(outputs.logits, expected_outputs.logits)
    else:
        loss = result

    torch.testing.assert_close(loss, expected_loss)
    assert not torch.isclose(loss, expected_outputs.loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in trainer.model.parameters())
    assert all(parameter.grad is None for parameter in trainer.ref_model.parameters())


@pytest.mark.parametrize("prediction_loss_only", [False, True])
def test_asft_prediction_step(asft_trainer, asft_batch, prediction_loss_only):
    trainer = asft_trainer
    expected_loss = trainer.compute_loss(trainer.model, asft_batch).detach()
    loss, logits, labels = trainer.prediction_step(trainer.model, asft_batch, prediction_loss_only)

    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(labels, asft_batch["labels"])
    if prediction_loss_only:
        assert logits is None
    else:
        assert logits.shape == (2, 4, 32)


@pytest.mark.parametrize("return_outputs", [False, True])
def test_non_asft_compute_loss(asft_trainer, asft_batch, return_outputs):
    trainer = asft_trainer
    trainer.finetuning_args.use_asft_loss = False
    trainer.compute_loss_func = None
    with torch.no_grad():
        expected_outputs = trainer.model(**asft_batch)

    result = trainer.compute_loss(trainer.model, asft_batch, return_outputs)
    if return_outputs:
        loss, outputs = result
        torch.testing.assert_close(outputs.logits, expected_outputs.logits)
    else:
        loss = result

    torch.testing.assert_close(loss, expected_outputs.loss)


def test_asft_evaluate(asft_trainer, asft_batch):
    trainer = asft_trainer
    expected_loss = trainer.compute_loss(trainer.model, asft_batch).detach().item()
    dataset = [{key: value[i] for key, value in asft_batch.items()} for i in range(2)]

    metrics = trainer.evaluate(eval_dataset=dataset)

    assert metrics["eval_loss"] == pytest.approx(expected_loss)
