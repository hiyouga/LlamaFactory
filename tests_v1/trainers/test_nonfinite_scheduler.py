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

import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from llamafactory.v1.config import TrainingArguments
from llamafactory.v1.core.rendering import Renderer
from llamafactory.v1.trainers.sft_trainer import SFTTrainer


class _RecordingSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.batches = {}
        super().__init__(*args, **kwargs)

    def compute_loss(self, batch):
        self.batches.setdefault(self.global_step, []).append(
            {key: value.clone() for key, value in batch.items() if isinstance(value, torch.Tensor)}
        )
        return super().compute_loss(batch)


def _make_trainer(tmp_path):
    vocab = {
        token: i
        for i, token in enumerate(["[PAD]", "[UNK]", "user", "assistant", "ask", "one", "two", "answer", "yes", "no"])
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
    )
    torch.manual_seed(92)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocab),
            n_positions=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
    )
    samples = [
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "value": "ask one"}]},
                {"role": "assistant", "content": [{"type": "text", "value": "yes" if i % 2 else "no answer"}]},
            ]
        }
        for i in range(10)
    ]
    args = TrainingArguments(
        output_dir=str(tmp_path),
        micro_batch_size=1,
        global_batch_size=2,
        cutoff_len=16,
        batching_workers=0,
        bf16=False,
        enable_activation_checkpointing=False,
        max_steps=5,
        logging_steps=10,
    )
    return _RecordingSFTTrainer(args, model, Renderer(tokenizer), samples)


def _make_optimizer(model, name):
    parameters = list(model.parameters())
    groups = [{"params": parameters[:2], "lr": 0.04}, {"params": parameters[2:], "lr": 0.01}]
    if name == "sgd":
        return torch.optim.SGD(groups, momentum=0.9)
    return torch.optim.AdamW(groups, weight_decay=0.01)


def _reference_step(model, optimizer, batches, max_grad_norm):
    valid_tokens = sum((batch["labels"] != -100).sum().item() for batch in batches)
    for batch in batches:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            use_cache=False,
        ).logits.float()
        labels = batch["labels"][..., 1:]
        weights = batch["loss_weights"][..., 1:]
        token_losses = F.cross_entropy(
            logits[..., :-1, :].reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).reshape_as(labels)
        loss = (token_losses * weights).sum() / (weights.sum() + 1e-6)
        count = (batch["labels"] != -100).sum().item()
        (loss * count / (valid_tokens + 1e-6)).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()


@pytest.mark.runs_on(["cpu"])
@pytest.mark.parametrize("optimizer_name", ["sgd", "adamw"])
@pytest.mark.parametrize("bad_steps", [{}, {1: float("nan")}, {3: float("inf")}, {1: float("nan"), 2: float("inf")}])
def test_scheduler_tracks_successful_sft_updates(tmp_path, optimizer_name, bad_steps):
    trainer = _make_trainer(tmp_path)
    reference = copy.deepcopy(trainer.model).train()
    trainer.optimizer = _make_optimizer(trainer.model, optimizer_name)
    expected_optimizer = _make_optimizer(reference, optimizer_name)
    trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=1, gamma=0.5)
    expected_scheduler = torch.optim.lr_scheduler.StepLR(expected_optimizer, step_size=1, gamma=0.5)
    actual_rates = []
    hook = trainer.optimizer.register_step_pre_hook(
        lambda optimizer, _args, _kwargs: actual_rates.append([group["lr"] for group in optimizer.param_groups])
    )

    # Inject a real non-finite gradient after the native GPT-2 backward.
    parameter = next(trainer.model.parameters())
    fault = parameter.register_hook(
        lambda grad: (
            torch.full_like(grad, bad_steps[trainer.global_step]) if trainer.global_step in bad_steps else grad
        )
    )
    try:
        trainer.fit()
    finally:
        fault.remove()
        hook.remove()

    expected_rates = []
    for step, batches in trainer.batches.items():
        if step in bad_steps:
            continue
        expected_rates.append([group["lr"] for group in expected_optimizer.param_groups])
        _reference_step(reference, expected_optimizer, batches, trainer.args.max_grad_norm)
        expected_scheduler.step()

    assert trainer.global_step == 5
    assert actual_rates == expected_rates
    assert trainer.lr_scheduler.state_dict() == expected_scheduler.state_dict()
    assert trainer.lr_scheduler.last_epoch == 5 - len(bad_steps)
    for actual, expected in zip(trainer.model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert actual.grad is None
    torch.testing.assert_close(trainer.optimizer.state_dict(), expected_optimizer.state_dict(), rtol=1e-5, atol=1e-6)

    # Resume the model, optimizer, and schedule, then compare the next actual update.
    checkpoint = tmp_path / "state.pt"
    torch.save(
        {
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.lr_scheduler.state_dict(),
        },
        checkpoint,
    )
    saved = torch.load(checkpoint, weights_only=True)
    resumed = copy.deepcopy(trainer.model)
    resumed.load_state_dict(saved["model"])
    resumed_optimizer = _make_optimizer(resumed, optimizer_name)
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, step_size=1, gamma=0.5)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    resumed_scheduler.load_state_dict(saved["scheduler"])
    _reference_step(resumed, resumed_optimizer, trainer.batches[5], trainer.args.max_grad_norm)
    resumed_scheduler.step()
    _reference_step(reference, expected_optimizer, trainer.batches[5], trainer.args.max_grad_norm)
    expected_scheduler.step()
    for actual, expected in zip(resumed.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert resumed_scheduler.state_dict() == expected_scheduler.state_dict()
