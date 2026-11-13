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

import json

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments
from trl import AutoModelForCausalLMWithValueHead

from llamafactory.hparams import FinetuningArguments
from llamafactory.train.rm.trainer import PairwiseTrainer


def collate_pairs(pairs):
    input_ids = torch.tensor([pair[key] for key in ("chosen", "rejected") for pair in pairs])
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


@pytest.fixture
def reward_trainer(tmp_path):
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        GPT2LMHeadModel(GPT2Config(vocab_size=24, n_embd=8, n_layer=1, n_head=1))
    )
    model.eval()
    return PairwiseTrainer(
        finetuning_args=FinetuningArguments(),
        processor=None,
        model=model,
        tokenizer=None,
        data_collator=collate_pairs,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            report_to="none",
            use_cpu=True,
            per_device_eval_batch_size=2,
            remove_unused_columns=False,
            disable_tqdm=True,
        ),
    )


@pytest.mark.parametrize("batch_size", [1, 3])
def test_pairwise_rewards_preserve_the_batch_dimension(reward_trainer, batch_size):
    pairs = [{"chosen": [2, 3, 4], "rejected": [2, 3, 5]} for _ in range(batch_size)]
    inputs = collate_pairs(pairs)
    loss, (_, chosen_scores, rejected_scores) = reward_trainer.compute_loss(
        reward_trainer.model, inputs, return_outputs=True
    )

    assert chosen_scores.shape == rejected_scores.shape == (batch_size,)
    expected = -torch.nn.functional.logsigmoid(chosen_scores.float() - rejected_scores.float()).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(reward_trainer.model.v_head.summary.weight.grad).all()


@pytest.mark.parametrize("num_pairs", [1, 3])
def test_reward_predictions_can_be_saved_for_singleton_batches(reward_trainer, num_pairs, tmp_path):
    pairs = [{"chosen": [2, 3, 4], "rejected": [2, 3, 5]} for _ in range(num_pairs)]

    predictions = reward_trainer.predict(pairs)
    reward_trainer.save_predictions(predictions)

    chosen_scores, rejected_scores = predictions.predictions
    assert chosen_scores.shape == rejected_scores.shape == (num_pairs,)
    records = [json.loads(line) for line in (tmp_path / "generated_predictions.jsonl").read_text().splitlines()]
    assert records == [
        {"chosen": round(float(chosen), 2), "rejected": round(float(rejected), 2)}
        for chosen, rejected in zip(chosen_scores, rejected_scores)
    ]
