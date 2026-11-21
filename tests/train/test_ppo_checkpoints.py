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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_callback import CallbackHandler, DefaultFlowCallback


@pytest.mark.parametrize(
    ("strategy", "interval", "limit", "saved", "retained"),
    [
        ("no", 1, 2, [], []),
        ("steps", 2, 2, [2, 4, 5], [4, 5]),
        ("steps", 0.6, None, [3, 5], [3, 5]),
        ("steps", 1, 1, [1, 2, 3, 4, 5], [5]),
        ("epoch", 1, 2, [2, 4, 5], [4, 5]),
    ],
)
def test_ppo_checkpoint_schedule_and_retention(tmp_path, strategy, interval, limit, saved, retained):
    from llamafactory.train.ppo.trainer import CustomPPOTrainer

    save_events = []

    class RecordSaves(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint = tmp_path / f"checkpoint-{state.global_step}" / "model.safetensors"
            assert checkpoint.is_file()
            save_events.append(state.global_step)

    trainer = object.__new__(CustomPPOTrainer)
    trainer.args = TrainingArguments(
        output_dir=str(tmp_path),
        max_steps=5,
        save_strategy=strategy,
        save_steps=interval,
        save_total_limit=limit,
        per_device_train_batch_size=1,
        report_to="none",
        use_cpu=True,
    )
    trainer.model = torch.nn.Linear(1, 1)
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda model, **kwargs: model)
    trainer.is_deepspeed_enabled = trainer.is_fsdp_enabled = False
    trainer.config = SimpleNamespace(batch_size=1, mini_batch_size=1, log_with=None)
    trainer.finetuning_args = SimpleNamespace(ppo_buffer_size=1, ppo_epochs=1)
    trainer.dataloader = [{"input_ids": torch.ones(1, 1), "attention_mask": torch.ones(1, 1)}] * 2
    trainer.processing_class = Mock()
    trainer.tokenizer = trainer.processing_class
    trainer.get_inputs = lambda batch: ([batch["input_ids"][0]], [batch["input_ids"][0]])
    trainer.get_rewards = lambda queries, responses: [torch.tensor(1.0)]
    trainer.step = lambda *args: {"ppo/loss/total": 1.0}
    trainer.state, trainer.control = TrainerState(), TrainerControl()
    trainer.callback_handler = CallbackHandler([DefaultFlowCallback, RecordSaves], trainer.model, None, None, None)

    trainer.ppo_train()
    assert save_events == saved
    assert sorted(path.name for path in tmp_path.glob("checkpoint-*")) == [f"checkpoint-{n}" for n in retained]
    trainer.save_model()
    assert (tmp_path / "model.safetensors").is_file()
