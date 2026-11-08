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

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import DataCollatorWithPadding, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from transformers.trainer_utils import PredictionOutput

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.hparams import FinetuningArguments, TrainingArguments, get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer


DEMO_DATA = os.getenv("DEMO_DATA", "llamafactory/demo_data")

TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "dataset": "llamafactory/tiny-supervised-dataset",
    "dataset_dir": "ONLINE",
    "template": "llama3",
    "cutoff_len": 1024,
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 1,
    "max_steps": 1,
    "report_to": "none",
}


@dataclass
class DataCollatorWithVerbose(DataCollatorWithPadding):
    verbose_list: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        features = [
            {k: v for k, v in feature.items() if k in ["input_ids", "attention_mask", "labels"]}
            for feature in features
        ]
        self.verbose_list.extend(features)
        batch = super().__call__(features)
        return {k: v[:, :1] for k, v in batch.items()}  # truncate input length


@pytest.mark.parametrize("disable_shuffling", [False, True])
def test_shuffle(disable_shuffling: bool):
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(
        {
            "output_dir": os.path.join("output", f"shuffle{str(disable_shuffling).lower()}"),
            "disable_shuffling": disable_shuffling,
            **TRAIN_ARGS,
        }
    )
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    data_collator = DataCollatorWithVerbose(tokenizer=tokenizer)
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        **dataset_module,
        **tokenizer_module,
    )
    trainer.train()
    if disable_shuffling:
        assert data_collator.verbose_list[0]["input_ids"] == dataset_module["train_dataset"][0]["input_ids"]
    else:
        assert data_collator.verbose_list[0]["input_ids"] != dataset_module["train_dataset"][0]["input_ids"]


@pytest.mark.parametrize("streaming", [False, True])
def test_save_predictions(tmp_path, streaming):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<pad>": 0, "hello": 1, "world": 2})), pad_token="<pad>"
    )
    trainer = CustomSeq2SeqTrainer(
        model=GPT2LMHeadModel(GPT2Config(vocab_size=3, n_layer=1, n_head=1, n_embd=8)),
        args=TrainingArguments(output_dir=str(tmp_path), report_to="none", use_cpu=True),
        tokenizer=tokenizer,
        finetuning_args=FinetuningArguments(),
        processor=None,
    )
    dataset = Dataset.from_dict({"input_ids": [[1], [2]]})
    if streaming:
        dataset = dataset.to_iterable_dataset()

    results = PredictionOutput(predictions=np.array([[0, 2], [0, 1]]), label_ids=np.array([[1], [2]]), metrics={})
    trainer.save_predictions(dataset, results)

    rows = [json.loads(line) for line in (tmp_path / "generated_predictions.jsonl").read_text().splitlines()]
    assert rows == [
        {"prompt": "hello", "predict": "world", "label": "hello"},
        {"prompt": "world", "predict": "hello", "label": "world"},
    ]
