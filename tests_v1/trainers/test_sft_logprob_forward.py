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
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

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


def _make_trainer(tmp_path, model_name, dtype=torch.float32):
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
    torch.manual_seed(73)
    if model_name == "gpt2":
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
    else:
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=len(vocab),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                attention_dropout=0.0,
                use_cache=False,
            )
        )
    model.to(dtype=dtype)
    samples = [
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "value": "ask one"}]},
                {"role": "assistant", "content": [{"type": "text", "value": "yes" if i % 2 else "no answer"}]},
            ]
        }
        for i in range(4)
    ]
    args = TrainingArguments(
        output_dir=str(tmp_path),
        micro_batch_size=1,
        global_batch_size=2,
        cutoff_len=16,
        batching_workers=0,
        bf16=False,
        enable_activation_checkpointing=False,
        max_steps=2,
        logging_steps=10,
    )
    return _RecordingSFTTrainer(args, model, Renderer(tokenizer), samples)


def _batch(sparse):
    input_ids = torch.tensor([[2, 4, 5, 3, 7, 8, 9], [2, 4, 6, 3, 8, 0, 0]])
    labels = input_ids.clone()
    labels[:, :2] = -100
    labels[1, -2:] = -100
    if sparse:
        labels[0, 4] = -100
    weights = (labels != -100).float()
    weights[0, 3] = 0.5
    weights[1, 4] = 2.0
    return {
        "input_ids": input_ids,
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]]),
        "position_ids": torch.arange(7).expand(2, -1),
        "labels": labels,
        "loss_weights": weights,
    }


def _reference_loss(model, batch):
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
    return (token_losses * weights).sum() / (weights.sum() + 1e-6)


@pytest.mark.runs_on(["cpu"])
@pytest.mark.parametrize("model_name", ["gpt2", "llama"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sparse", [False, True])
def test_sft_log_probs_compute_only_requested_cross_entropy(tmp_path, model_name, dtype, sparse):
    trainer = _make_trainer(tmp_path, model_name, dtype)
    reference = copy.deepcopy(trainer.model)
    batch = _batch(sparse)
    originals = {key: value.clone() for key, value in batch.items()}
    observed_losses = []
    hook = trainer.model.register_forward_hook(
        lambda _module, _args, output: observed_losses.append(output.loss is None)
    )
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as profile:
            actual = trainer.compute_loss(batch)
    finally:
        hook.remove()
    expected = _reference_loss(reference, batch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    expected.backward()
    for parameter, expected_parameter in zip(trainer.model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=0, atol=0)
    for key in batch:
        torch.testing.assert_close(batch[key], originals[key], rtol=0, atol=0)
    cross_entropy_calls = sum(
        event.count for event in profile.key_averages() if event.key == "aten::cross_entropy_loss"
    )
    softmax_bytes = sum(event.cpu_memory_usage for event in profile.events() if event.name == "aten::_log_softmax")
    print(
        {
            "model": model_name,
            "dtype": str(dtype),
            "cross_entropy_calls": cross_entropy_calls,
            "log_softmax_output_allocation_bytes": softmax_bytes,
        }
    )
    assert observed_losses == [True]
    assert cross_entropy_calls == 1


@pytest.mark.runs_on(["cpu"])
@pytest.mark.parametrize("model_name", ["gpt2", "llama"])
@pytest.mark.parametrize("optimizer_name", ["sgd", "adamw"])
def test_sft_updates_match_explicit_weighted_loss(tmp_path, model_name, optimizer_name):
    trainer = _make_trainer(tmp_path, model_name)
    reference = copy.deepcopy(trainer.model).train()
    optimizer_class = torch.optim.SGD if optimizer_name == "sgd" else torch.optim.AdamW
    trainer.optimizer = optimizer_class(trainer.model.parameters(), lr=0.01)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _step: 1.0)
    expected_optimizer = optimizer_class(reference.parameters(), lr=0.01)
    expected_scheduler = torch.optim.lr_scheduler.LambdaLR(expected_optimizer, lambda _step: 1.0)
    observed_losses = []
    hook = trainer.model.register_forward_hook(
        lambda _module, _args, output: observed_losses.append(output.loss is None)
    )
    try:
        trainer.fit()
    finally:
        hook.remove()
    for batches in trainer.batches.values():
        total = sum((batch["labels"] != -100).sum().item() for batch in batches)
        for batch in batches:
            count = (batch["labels"] != -100).sum().item()
            (_reference_loss(reference, batch) * count / (total + 1e-6)).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), trainer.args.max_grad_norm)
        expected_optimizer.step()
        expected_scheduler.step()
        expected_optimizer.zero_grad()
    for actual, expected in zip(trainer.model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(trainer.optimizer.state_dict(), expected_optimizer.state_dict(), rtol=1e-5, atol=1e-6)
    assert trainer.lr_scheduler.state_dict() == expected_scheduler.state_dict()
    assert trainer.global_step == 2
    assert observed_losses == [True] * 4


@pytest.mark.runs_on(["cpu"])
@pytest.mark.parametrize("model_name", ["gpt2", "llama"])
def test_direct_model_calls_keep_builtin_label_loss(tmp_path, model_name):
    trainer = _make_trainer(tmp_path, model_name)
    batch = _batch(False)
    output = trainer.model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]
    )
    assert output.loss is not None
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert all(
        torch.isfinite(parameter.grad).all() for parameter in trainer.model.parameters() if parameter.grad is not None
    )
