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

"""Answer-token normalization must survive gradient accumulation for native OCR."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.loss.loss_utils import ForCausalLMLoss

from llamafactory.hparams import FinetuningArguments, TrainingArguments
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer


class TinyCausalModel(torch.nn.Module):
    def __init__(self, model_type):
        super().__init__()
        self.config = PretrainedConfig()
        self.config.model_type = model_type
        self.embedding = torch.nn.Embedding(16, 4)

    def forward(self, input_ids, labels=None, **kwargs):
        logits = self.embedding(input_ids)
        loss = ForCausalLMLoss(logits, labels, 4, **kwargs) if labels is not None else None
        return {"loss": loss, "logits": logits}


def make_trainer(tmp_path, model_type):
    torch.manual_seed(3407)
    model = TinyCausalModel(model_type)
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            gradient_accumulation_steps=2,
            report_to="none",
        ),
        finetuning_args=FinetuningArguments(finetuning_type="full"),
        processor=SimpleNamespace(),
        tokenizer=None,
    )
    return trainer, model


def test_ocr_accumulation_matches_token_weighted_cross_entropy(tmp_path):
    trainer, model = make_trainer(tmp_path, "minicpm_ocr")
    assert trainer.model_accepts_loss_kwargs
    batches = [
        {"input_ids": torch.tensor([[1, 2, 3]]), "labels": torch.tensor([[-100, -100, 2]])},
        {"input_ids": torch.tensor([[4, 5, 6, 7, 8]]), "labels": torch.tensor([[-100, -100, 1, 2, 3]])},
    ]
    batches, count = trainer.get_batch_samples(iter(batches), len(batches), torch.device("cpu"))
    assert count.item() == 4
    trainer.current_gradient_accumulation_steps = 2
    trainer.create_optimizer()
    accumulated = sum(trainer.training_step(model, batch, num_items_in_batch=count) for batch in batches)
    gradient = model.embedding.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    expected = (
        sum(
            F.cross_entropy(
                model.embedding(batch["input_ids"])[:, :-1].reshape(-1, 4),
                batch["labels"][:, 1:].reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            for batch in batches
        )
        / 4
    )
    expected.backward()
    assert accumulated.item() == pytest.approx(expected.item())
    assert torch.allclose(gradient, model.embedding.weight.grad)


def test_existing_v46_loss_guard_is_preserved(tmp_path):
    trainer, _ = make_trainer(tmp_path, "minicpmv4_6")
    assert not trainer.model_accepts_loss_kwargs
