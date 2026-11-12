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

import numpy as np
import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import DataCollatorForSeq2Seq, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, set_seed

from llamafactory.hparams import FinetuningArguments, TrainingArguments
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer


def make_trainer(tmp_path, **training_kwargs):
    set_seed(42)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[PAD]": 0, "[UNK]": 1, "a": 2, "b": 3, "c": 4, "d": 5}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=6,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
            pad_token_id=0,
            bos_token_id=None,
            eos_token_id=None,
            attn_implementation="eager",
        )
    )
    model.generation_config.max_new_tokens = 3
    model.generation_config.do_sample = False
    dataset = Dataset.from_list(
        [
            {"input_ids": [2, 3, 4], "attention_mask": [1, 1, 1], "labels": [2, 3, 4]},
            {"input_ids": [2], "attention_mask": [1], "labels": [2]},
        ]
    )
    generation_matches = []

    def compute_metrics(prediction):
        # Compare batched completions with actual, unpadded single-prompt generation.
        expected = []
        for row in dataset:
            output = model.generate(
                input_ids=torch.tensor([row["input_ids"]]),
                attention_mask=torch.tensor([row["attention_mask"]]),
            )
            expected.append(output[0, len(row["input_ids"]) :].tolist())

        matches = np.array_equal(prediction.predictions[:, 3:6], expected)
        generation_matches.append(matches)
        return {"generation_matches": float(matches)}

    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to=[],
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        max_steps=2,
        save_strategy="no",
        disable_tqdm=True,
        **{"predict_with_generate": True, **training_kwargs},
    )
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=args,
        finetuning_args=FinetuningArguments(finetuning_type="full"),
        processor=None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer),
        train_dataset=dataset,
        eval_dataset=dataset,
        compute_metrics=compute_metrics,
    )
    return trainer, generation_matches


@pytest.mark.parametrize("num_workers", [0, 1])
def test_scheduled_generation_preserves_training_padding(tmp_path, num_workers):
    # Regression for #10611: generation during training must match final evaluation.
    trainer, generation_matches = make_trainer(
        tmp_path,
        eval_strategy="steps",
        eval_steps=1,
        dataloader_num_workers=num_workers,
        dataloader_persistent_workers=bool(num_workers),
    )
    training_masks = []

    def record_training_mask(model, args, kwargs):
        if model.training:
            training_masks.append(kwargs["attention_mask"].clone())

    handle = trainer.model.register_forward_pre_hook(record_training_mask, with_kwargs=True)
    trainer.train()
    handle.remove()
    assert generation_matches == [True, True]
    assert trainer.processing_class.padding_side == "right"
    assert len(training_masks) == 2
    for mask in training_masks:
        assert mask[:, 0].all()  # Both training steps retain right padding.
        assert (mask[:, -1] == 0).sum() == 1


@pytest.mark.parametrize("method", ["evaluate", "predict"])
def test_generation_restores_padding(tmp_path, method):
    trainer, generation_matches = make_trainer(tmp_path)
    getattr(trainer, method)(trainer.eval_dataset)
    assert generation_matches == [True]
    assert trainer.processing_class.padding_side == "right"


def test_generation_with_multiple_eval_datasets_restores_padding(tmp_path):
    trainer, generation_matches = make_trainer(tmp_path)
    metrics = trainer.evaluate({"first": trainer.eval_dataset, "second": trainer.eval_dataset})
    assert generation_matches == [True, True]
    assert metrics["eval_first_generation_matches"] == metrics["eval_second_generation_matches"] == 1.0
    assert trainer.processing_class.padding_side == "right"


@pytest.mark.parametrize("method", ["evaluate", "predict"])
@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_generation_error_restores_padding(tmp_path, method, padding_side):
    trainer, _ = make_trainer(tmp_path)
    trainer.processing_class.padding_side = padding_side
    with pytest.raises(ValueError, match="max_new_tokens"):
        getattr(trainer, method)(trainer.eval_dataset, max_new_tokens=0)

    assert trainer.processing_class.padding_side == padding_side


def test_loss_only_evaluation_preserves_padding(tmp_path):
    trainer, _ = make_trainer(tmp_path, predict_with_generate=False)
    trainer.compute_metrics = None
    metrics = trainer.evaluate()
    assert np.isfinite(metrics["eval_loss"])
    assert trainer.processing_class.padding_side == "right"
