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
from transformers import GPT2Config, GPT2LMHeadModel, TrainerCallback

from llamafactory.hparams import FinetuningArguments, TrainingArguments
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.train.trainer_utils import asft_loss_func


def _model(seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return GPT2LMHeadModel(
            GPT2Config(
                vocab_size=23,
                n_positions=8,
                n_embd=16,
                n_layer=1,
                n_head=2,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
                use_cache=False,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
        )


def _batch():
    ids = (torch.arange(42).reshape(6, 7) % 21) + 1
    labels = ids.clone()
    labels[:, 0] = -100
    supervised = ([6], [1, 2, 4, 6], [2, 5], [1, 2, 3, 4, 6], [3], [1, 4, 5])
    for row, positions in enumerate(supervised):
        for position in range(1, 7):
            if position not in positions:
                labels[row, position] = -100
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def _oracle(policy_logits, labels, ref_logits, alpha, denominator=None):
    targets = labels[:, 1:]
    valid = targets != -100
    log_p = policy_logits.float()[:, :-1].log_softmax(-1)
    log_q = ref_logits[:, :-1].log_softmax(-1)
    log_target_p = log_p.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    dft = -log_target_p * log_target_p.exp().detach()
    kl = (log_q.exp() * (log_q - log_p)).sum(-1)
    count = valid.sum() if denominator is None else denominator
    return ((dft + alpha * kl) * valid).sum() / count


def _trainer(tmp_path, alpha, model=None, ref_model=None, micro_batch=1, accumulation=1, optimizer_name="sgd"):
    model = _model(7) if model is None else model
    ref_model = _model(29).requires_grad_(False) if ref_model is None else ref_model
    optimizer = _optimizer(model, optimizer_name)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    batch = _batch()
    dataset = [{key: value[index] for key, value in batch.items()} for index in range(6)]
    return CustomSeq2SeqTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=None,
        processor=None,
        finetuning_args=FinetuningArguments(
            finetuning_type="full", use_asft_loss=True, asft_alpha=alpha, disable_shuffling=True
        ),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            report_to="none",
            disable_tqdm=True,
            per_device_train_batch_size=micro_batch,
            gradient_accumulation_steps=accumulation,
            max_steps=2,
            max_grad_norm=0.0,
            save_strategy="no",
            logging_strategy="no",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            seed=42,
        ),
        train_dataset=dataset,
        optimizers=(optimizer, scheduler),
    )


def _optimizer(model, name):
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.6)
    return torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.02)


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.7])
@pytest.mark.parametrize("denominator_kind", ["local", "accumulated_int", "accumulated_tensor"])
@pytest.mark.parametrize("rows", [[0], [1, 2]])
def test_asft_compute_loss_uses_supplied_token_count(tmp_path, alpha, denominator_kind, rows):
    trainer = _trainer(tmp_path, alpha)
    full_batch = _batch()
    batch = {key: value[rows] for key, value in full_batch.items()}
    count = (full_batch["labels"][:, 1:] != -100).sum()
    denominator = None if denominator_kind == "local" else count.item()
    if denominator_kind == "accumulated_tensor":
        denominator = count
    expected_model = deepcopy(trainer.model)
    with torch.no_grad():
        ref_logits = trainer.ref_model(**batch).logits
    expected = _oracle(expected_model(**batch).logits, batch["labels"], ref_logits, alpha, denominator)
    actual = trainer.compute_loss(trainer.model, batch, num_items_in_batch=denominator)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    actual.backward()
    expected.backward()
    for parameter, reference in zip(trainer.model.parameters(), expected_model.parameters()):
        torch.testing.assert_close(parameter.grad, reference.grad, rtol=2e-5, atol=2e-7)
    assert all(parameter.grad is None for parameter in trainer.ref_model.parameters())


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.7])
def test_asft_without_explicit_count_preserves_local_token_mean(alpha):
    batch = _batch()
    policy = _model(3)
    reference = _model(11)
    with torch.no_grad():
        ref_logits = reference(**batch).logits
    outputs = policy(**batch)
    expected = _oracle(outputs.logits, batch["labels"], ref_logits, alpha)
    torch.testing.assert_close(asft_loss_func(outputs, batch["labels"], ref_logits, alpha), expected)


class _GradientTrace(TrainerCallback):
    def __init__(self):
        self.gradients = []

    def on_pre_optimizer_step(self, args, state, control, model, **kwargs):
        self.gradients.append([parameter.grad.detach().clone() for parameter in model.parameters()])


@pytest.mark.parametrize("alpha", [0.0, 0.7])
@pytest.mark.parametrize("optimizer_name", ["sgd", "adamw"])
@pytest.mark.parametrize("micro_batch,accumulation", [(1, 3), (3, 1)])
def test_native_asft_training_matches_full_token_objective(
    tmp_path, alpha, optimizer_name, micro_batch, accumulation
):
    trainer = _trainer(
        tmp_path, alpha, micro_batch=micro_batch, accumulation=accumulation, optimizer_name=optimizer_name
    )
    expected_model = deepcopy(trainer.model)
    expected_optimizer = _optimizer(expected_model, optimizer_name)
    torch.optim.lr_scheduler.LambdaLR(expected_optimizer, lambda _: 1.0)
    trace = _GradientTrace()
    trainer.add_callback(trace)
    full_batch = _batch()
    expected_losses = []
    expected_gradients = []
    for offset in (0, 3):
        batch = {key: value[offset : offset + 3] for key, value in full_batch.items()}
        expected_optimizer.zero_grad()
        with torch.no_grad():
            ref_logits = trainer.ref_model(**batch).logits
        loss = _oracle(expected_model(**batch).logits, batch["labels"], ref_logits, alpha)
        loss.backward()
        expected_losses.append(loss.detach())
        expected_gradients.append([parameter.grad.detach().clone() for parameter in expected_model.parameters()])
        expected_optimizer.step()

    result = trainer.train()

    assert result.global_step == 2
    assert len(trace.gradients) == 2
    assert result.training_loss == pytest.approx(torch.stack(expected_losses).mean().item(), rel=1e-5, abs=1e-7)
    for actual_step, expected_step in zip(trace.gradients, expected_gradients):
        for actual, expected in zip(actual_step, expected_step):
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-7)
    for actual, expected in zip(trainer.model.parameters(), expected_model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-7)
    actual_state = trainer.optimizer.state_dict()
    expected_state = expected_optimizer.state_dict()
    assert actual_state["param_groups"] == expected_state["param_groups"]
    assert actual_state["state"].keys() == expected_state["state"].keys()
    for parameter_id, state in actual_state["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, expected_state["state"][parameter_id][name], rtol=3e-5, atol=3e-7)
    assert all(parameter.grad is None for parameter in trainer.ref_model.parameters())
