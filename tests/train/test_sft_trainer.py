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

import math
import os
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch
from transformers import DataCollatorWithPadding

from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.train.trainer_utils import create_ref_model


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


def test_ambient_loss_func_at_initialization():
    # at initialization the model equals the initial model, so the matching loss is the scaled tempered entropy
    from llamafactory.train.trainer_utils import ambient_loss_func

    torch.manual_seed(0)
    logits = torch.randn(2, 6, 11)
    labels = torch.tensor([[-100, -100, 3, 4, 5, 6], [-100, 1, 2, -100, -100, -100]])
    loss = ambient_loss_func(logits, logits, labels, temperature=2.5)

    def entropy(temperature):
        log_probs = torch.log_softmax(logits[:, :-1] / temperature, dim=-1)
        return -(log_probs.exp() * log_probs).sum(-1)[labels[:, 1:] != -100]

    expected = entropy(2.5).sum() / entropy(1.0).sum() * entropy(2.5).mean()
    assert entropy(2.5).sum() > entropy(1.0).sum()  # a hotter distribution has more entropy
    assert torch.allclose(loss, expected, rtol=1e-5)


@pytest.mark.parametrize("finetuning_type", ["lora", "full"])
def test_ambient_sft(finetuning_type: str):
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(
        {
            **TRAIN_ARGS,
            "output_dir": os.path.join("output", f"ambient_{finetuning_type}"),
            "finetuning_type": finetuning_type,
            "use_ambient_loss": True,
            "ambient_clean_probability": 0.5,
            "max_steps": 4,
            "gradient_accumulation_steps": 2,
            "logging_steps": 1,
        }
    )
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    ref_model = create_ref_model(model_args, finetuning_args)
    assert (ref_model is None) == (finetuning_type == "lora")  # LoRA disables its adapters instead
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=SFTDataCollatorWith4DAttentionMask(template=template, model=model, **tokenizer_module),
        ref_model=ref_model,
        **dataset_module,
        **tokenizer_module,
    )
    trainer.train()
    losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    assert losses and all(math.isfinite(loss) for loss in losses)
