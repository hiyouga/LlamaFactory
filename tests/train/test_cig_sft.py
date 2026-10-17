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

r"""Unit tests of the CIG-SFT loss.

Everything here runs on the CPU without a model, a tokenizer or a network connection, which keeps the numerical
contract of the loss checked on any machine. The tests are grouped by what they protect:

- the erased prompt presets, and the precedence of a dataset column over the dataset level prompt
- the shape of the supervised run that the two views are aligned on
- the per-token credit read from the reference model
- the weighted cross entropy that the trainer optimizes
- where the gradient flows, which is what keeps the reference view a frozen scorer
"""

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import Seq2SeqTrainer

from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask
from llamafactory.data.converter import align_dataset
from llamafactory.data.erased_prompt import (
    ERASED_PROMPT_COLUMN,
    ERASED_PROMPT_PRESETS,
    resolve_erased_prompt,
)
from llamafactory.data.parser import DatasetAttr
from llamafactory.data.processor.supervised import PackedSupervisedDatasetProcessor, SupervisedDatasetProcessor
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import DataArguments, FinetuningArguments
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.train.trainer_utils import (
    _response_spans,
    _supervised_extent,
    cig_sft_credit_func,
    cig_sft_loss_func,
)


VOCAB_SIZE = 32
EPS = 1e-8


class StubTrainingArgs:
    r"""`align_dataset` only reads `local_process_index` from the training arguments."""

    local_process_index = 0


def shifted_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    r"""Per-token cross entropy of shape (batch_size, seq_len - 1), written out independently of the loss."""
    shift_logits = logits[..., :-1, :].reshape(-1, logits.size(-1))
    shift_labels = labels[..., 1:].reshape(-1)
    per_token = F.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE_INDEX, reduction="none")
    return per_token.view(labels.size(0), -1)


def reference_credit(present_ce: torch.Tensor, erased_ce: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    r"""The closed form credit, clipped to [0, 1]."""
    gain = (-present_ce) - (-erased_ce)  # log p_present - log p_erased
    denominator = erased_ce.clamp(min=0.0) + eps  # -log p_erased + eps
    return (gain.clamp(min=0.0) / denominator).clamp(0.0, 1.0)


def reference_weights(probs: torch.Tensor, credit: torch.Tensor, cig_lambda: float) -> torch.Tensor:
    r"""The closed form weight of the anchored projection, written out independently of the loss."""
    exponent = torch.exp((credit / cig_lambda).clamp(max=60.0))
    projected = probs * (exponent - 1.0) / (1.0 - probs + probs * exponent + 1e-12)
    return projected


def build_case(
    present_boost: float = 8.0, erased_boost: float = -8.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""A training view of three prompt tokens and four supervised tokens, erased down to two prompt tokens.

    `present_boost` and `erased_boost` tilt the two views towards and away from the response tokens on purpose, so that
    every credit lands strictly inside `(0, 1)`. A uniform draw would leave most credits at exactly zero and hide the
    arithmetic behind a lucky random seed.
    """
    torch.manual_seed(0)
    present_ids = torch.tensor([11, 12, 13, 21, 22, 23, 24])
    erased_ids = torch.tensor([11, 12, 21, 22, 23, 24])
    present_labels = torch.tensor([[-100, -100, -100, 21, 22, 23, 24]])
    erased_labels = torch.tensor([[-100, -100, 21, 22, 23, 24]])
    present_logits = torch.randn(1, len(present_ids), VOCAB_SIZE)
    erased_logits = torch.randn(1, len(erased_ids), VOCAB_SIZE)
    for offset, token in enumerate((21, 22, 23, 24)):
        # a supervised token at position `p` is scored by the column `p - 1` of the logits
        present_logits[0, 2 + offset, token] += present_boost
        erased_logits[0, 1 + offset, token] += erased_boost

    return present_logits, present_labels, erased_logits, erased_labels


# ------------------------------------------------------------------ erased prompt


def test_presets_match_the_reference_implementation():
    # byte for byte the values of the helper that built every published run of the method
    assert ERASED_PROMPT_PRESETS["math"] == (
        "[PROBLEM_REMOVED]\nLet's think step by step and output the final answer within \\boxed{}."
    )
    assert ERASED_PROMPT_PRESETS["code"] == "[TASK_REMOVED]"
    assert ERASED_PROMPT_PRESETS["medical"] == (
        "Question:\n[QUESTION_REMOVED]\n\nOptions:\n[OPTIONS_REMOVED]\n\nAnswer:"
    )
    assert ERASED_PROMPT_PRESETS["default"] == "[CONTEXT_REMOVED]"


def test_resolve_erased_prompt():
    assert resolve_erased_prompt("math") == ERASED_PROMPT_PRESETS["math"]
    assert resolve_erased_prompt("  MATH ") == ERASED_PROMPT_PRESETS["math"]
    assert resolve_erased_prompt("erase this") == "erase this"  # a literal is passed through
    assert resolve_erased_prompt(None) is None
    assert resolve_erased_prompt("") is None
    assert resolve_erased_prompt(float("nan")) is None  # a missing entry of a dataset column


def test_cig_arguments_reject_invalid_combinations():
    with pytest.raises(ValueError, match="Cannot use DFT"):
        FinetuningArguments(use_cig_sft_loss=True, use_dft_loss=True)

    with pytest.raises(ValueError, match="cig_lambda"):
        FinetuningArguments(use_cig_sft_loss=True, cig_lambda=0.0)

    with pytest.raises(ValueError, match="train_on_prompt"):
        DataArguments(cig_erased_prompt="default", train_on_prompt=True)


def test_align_dataset_carries_the_erased_prompt_column():
    dataset = Dataset.from_dict(
        {
            "instruction": ["q1", "q2"],
            "input": ["", ""],
            "output": ["a1", "a2"],
            "erased_prompt": ["e1", "e2"],
            "unused": ["u1", "u2"],
        }
    )
    dataset_attr = DatasetAttr("file", dataset_name="tiny", formatting="alpaca", erased_prompt="erased_prompt")
    aligned = align_dataset(dataset, dataset_attr, DataArguments(preprocessing_num_workers=1), StubTrainingArgs())

    # `align_dataset` drops every raw column, so the erased prompt must be carried in its own `_` column instead
    assert "instruction" not in aligned.column_names
    assert ERASED_PROMPT_COLUMN in aligned.column_names
    assert aligned[ERASED_PROMPT_COLUMN] == ["e1", "e2"]
    assert aligned[0]["_prompt"] == [{"role": "user", "content": "q1"}]


def test_align_dataset_without_the_column_is_untouched():
    dataset = Dataset.from_dict({"instruction": ["q1"], "input": [""], "output": ["a1"]})
    dataset_attr = DatasetAttr("file", dataset_name="tiny", formatting="alpaca")
    data_args = DataArguments(preprocessing_num_workers=1)
    assert ERASED_PROMPT_COLUMN not in align_dataset(dataset, dataset_attr, data_args, StubTrainingArgs()).column_names


def build_processor(data_args: DataArguments, packed: bool = False) -> SupervisedDatasetProcessor:
    r"""A processor with the tokenizer stubbed out, so that only the erased prompt logic is exercised."""
    processor_class = PackedSupervisedDatasetProcessor if packed else SupervisedDatasetProcessor
    processor = processor_class(template=None, tokenizer=None, processor=None, data_args=data_args)
    seen: list[list[dict[str, str]]] = []

    def fake_encode(prompt, response, system, tools, images, videos, audios) -> tuple[list[int], list[int]]:
        seen.append(list(prompt))
        return [1, 2, 3], [-100, -100, -100]

    processor._encode_data_example = fake_encode  # type: ignore[method-assign]
    return processor, seen


def single_row_examples(**columns) -> dict[str, list]:
    r"""A one row batch with the columns `preprocess_dataset` always reads."""
    examples = {
        "_prompt": [[{"role": "user", "content": "q1"}]],
        "_response": [[{"role": "assistant", "content": "a1"}]],
        "_system": [None],
        "_tools": [None],
        "_images": [None],
        "_videos": [None],
        "_audios": [None],
    }
    examples.update(columns)
    return examples


def test_the_dataset_column_wins_over_the_dataset_level_prompt():
    processor, seen = build_processor(DataArguments(cig_erased_prompt="math"))
    examples = single_row_examples(**{ERASED_PROMPT_COLUMN: ["erase this one"]})
    examples["_prompt"].append([{"role": "user", "content": "q2"}])
    examples["_response"].append([{"role": "assistant", "content": "a2"}])
    for key in ("_system", "_tools", "_images", "_videos", "_audios"):
        examples[key].append(None)

    examples[ERASED_PROMPT_COLUMN].append(None)  # the second row falls back to the dataset level prompt
    processor.preprocess_dataset(examples)

    assert len(seen) == 4  # the training view and the erased view of both rows
    assert seen[1] == [{"role": "user", "content": "erase this one"}]
    assert seen[3] == [{"role": "user", "content": ERASED_PROMPT_PRESETS["math"]}]


def test_a_preset_name_in_a_dataset_column_is_resolved():
    processor, seen = build_processor(DataArguments())
    processor.preprocess_dataset(single_row_examples(**{ERASED_PROMPT_COLUMN: ["code"]}))

    assert seen[1] == [{"role": "user", "content": ERASED_PROMPT_PRESETS["code"]}]


def test_no_erased_prompt_builds_no_second_view():
    processor, seen = build_processor(DataArguments())
    model_inputs = processor.preprocess_dataset(single_row_examples())

    assert seen == [[{"role": "user", "content": "q1"}]]
    assert "erased_input_ids" not in model_inputs


def test_packing_rejects_the_erased_prompt_column():
    processor, _ = build_processor(DataArguments(), packed=True)
    try:
        processor.preprocess_dataset(single_row_examples(**{ERASED_PROMPT_COLUMN: ["e1"]}))
        raise AssertionError("packing accepted the erased prompt column")
    except ValueError as err:
        assert "packing" in str(err)


def test_a_batch_mixing_rows_with_and_without_the_erased_view_is_rejected():
    # the loss drops tokens without a credit, so such a batch would silently train on a subset of its rows
    collator = SFTDataCollatorWith4DAttentionMask(template=object(), model=None, tokenizer=None, use_cig_sft_loss=True)
    features = [{"erased_input_ids": [1, 2], "erased_labels": [1, 2]}, {}]
    try:
        collator(features)
        raise AssertionError("the collator accepted a half-erased batch")
    except ValueError as err:
        assert "erased view" in str(err)


def test_erased_view_is_not_truncated_to_the_ordinary_view_length():
    tokenizer = type("Tokenizer", (), {"pad_token_id": 0, "padding_side": "right"})()
    collator = SFTDataCollatorWith4DAttentionMask(
        template=object(), model=None, tokenizer=tokenizer, use_cig_sft_loss=True
    )
    erased_ids, erased_mask, erased_labels = collator._pad_erased(
        erased_input_ids=[[1, 2, 3, 4]],
        erased_labels=[[IGNORE_INDEX, IGNORE_INDEX, 3, 4]],
        seq_len=2,
    )

    assert erased_ids.tolist() == [[1, 2, 3, 4]]
    assert erased_mask.tolist() == [[1, 1, 1, 1]]
    assert erased_labels.tolist() == [[IGNORE_INDEX, IGNORE_INDEX, 3, 4]]


# ------------------------------------------------------------------ supervised run


def test_supervised_extent_ignores_right_padding():
    labels = torch.tensor([[-100, -100, 5, 6, 7, -100], [-100, 5, -100, -100, -100, -100]])
    assert _supervised_extent(labels).tolist() == [5, 2]


def test_response_spans():
    starts, counts = _response_spans(torch.tensor([[-100, -100, -100, 5, 6, 7, 8]]))
    assert starts.tolist() == [3]
    assert counts.tolist() == [4]


def test_response_spans_rejects_a_non_contiguous_run():
    # the loss pairs the two views token by token, so a gap in the supervised tokens has no defined alignment
    try:
        _response_spans(torch.tensor([[-100, 5, -100, 6]]))
        raise AssertionError("a non-contiguous run was accepted")
    except ValueError as err:
        assert "contiguous" in str(err)


def test_response_spans_accepts_an_empty_row():
    starts, counts = _response_spans(torch.tensor([[-100, -100, -100], [-100, 5, 6]]))
    assert starts.tolist() == [0, 1]
    assert counts.tolist() == [0, 2]


# ------------------------------------------------------------------ the credit


def test_credit_matches_the_closed_form():
    present_logits, present_labels, erased_logits, erased_labels = build_case()
    credit = cig_sft_credit_func(present_logits, present_labels, erased_logits, erased_labels)

    assert len(credit) == 1
    assert credit[0].numel() == 4  # the four supervised tokens, aligned across the two views
    present_ce = shifted_ce(present_logits, present_labels)[0, -4:]
    erased_ce = shifted_ce(erased_logits, erased_labels)[0, -4:]
    assert torch.allclose(credit[0], reference_credit(present_ce, erased_ce), atol=1e-6)


def test_credit_is_bounded_and_detached():
    present_logits, present_labels, erased_logits, erased_labels = build_case()
    credit = cig_sft_credit_func(present_logits, present_labels, erased_logits, erased_labels)

    assert bool((credit[0] >= 0.0).all() and (credit[0] <= 1.0).all())
    assert not credit[0].requires_grad
    assert credit[0].device.type == "cpu"


def test_credit_keeps_the_shorter_view():
    # the erased view holds one more response token, which has nothing to be compared against
    present_logits, present_labels, _, _ = build_case()
    erased_labels_long = torch.tensor([[-100, -100, 21, 22, 23, 24, 25]])
    erased_logits_long = torch.randn(1, 7, VOCAB_SIZE)

    credit = cig_sft_credit_func(present_logits, present_labels, erased_logits_long, erased_labels_long)
    assert credit[0].numel() == 4


def build_ragged_case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Rows whose responses hold six, three and two tokens, which makes the last column differ per row."""
    present_labels = torch.tensor(
        [
            [-100, -100, -100, 21, 22, 23, 24, 25, 26],
            [-100, 21, 22, 23, -100, -100, -100, -100, -100],
            [-100, -100, -100, -100, 21, 22, -100, -100, -100],
        ]
    )
    erased_labels = torch.tensor(
        [
            [-100, -100, 21, 22, 23, 24, 25, 26, -100],
            [-100, 21, 22, 23, -100, -100, -100, -100, -100],
            [-100, 21, 22, -100, -100, -100, -100, -100, -100],
        ]
    )
    return present_labels, erased_labels


def test_credit_of_a_ragged_batch():
    present_labels, erased_labels = build_ragged_case()
    credit = cig_sft_credit_func(
        torch.randn(3, 9, VOCAB_SIZE), present_labels, torch.randn(3, 9, VOCAB_SIZE), erased_labels
    )

    assert [row.numel() for row in credit] == [6, 3, 2]


def test_credit_of_an_empty_response():
    labels = torch.tensor([[-100, -100, -100]])
    credit = cig_sft_credit_func(torch.randn(1, 3, VOCAB_SIZE), labels, torch.randn(1, 3, VOCAB_SIZE), labels)

    assert len(credit) == 1
    assert credit[0].numel() == 0


# ------------------------------------------------------------------ the loss


def test_loss_matches_the_weighted_cross_entropy():
    present_logits, present_labels, erased_logits, erased_labels = build_case()
    credit = cig_sft_credit_func(present_logits, present_labels, erased_logits, erased_labels)
    present_ce = shifted_ce(present_logits, present_labels)[0, -4:]

    for cig_lambda in (0.1, 0.3, 1.0):
        loss = cig_sft_loss_func({"logits": present_logits}, present_labels, credit, cig_lambda=cig_lambda)
        weights = reference_weights(torch.exp(-present_ce).detach(), credit[0], cig_lambda)
        expected = (weights * present_ce).sum() / 4  # normalized by the number of credited tokens
        assert torch.allclose(loss, expected, atol=1e-6), (cig_lambda, float(loss))


def test_loss_of_two_equal_rows_is_split_invariant():
    # the weights are detached and the sum is normalized by the credited token count, so the mean of the per row
    # losses of equally sized responses reproduces the loss of the whole batch
    present_labels = torch.tensor(
        [
            [-100, -100, -100, 21, 22, 23, 24],
            [-100, -100, 11, 12, 13, 14, -100],
        ]
    )
    erased_labels = torch.tensor(
        [
            [-100, 21, 22, 23, 24, -100, -100],
            [-100, 11, 12, 13, 14, -100, -100],
        ]
    )
    present_logits = torch.randn(2, 7, VOCAB_SIZE)
    credit = cig_sft_credit_func(
        torch.randn(2, 7, VOCAB_SIZE), present_labels, torch.randn(2, 7, VOCAB_SIZE), erased_labels
    )
    kwargs = dict(cig_lambda=0.3)
    batched = cig_sft_loss_func({"logits": present_logits}, present_labels, credit, **kwargs)
    rows = [
        cig_sft_loss_func(
            {"logits": present_logits[index : index + 1]}, present_labels[index : index + 1], [credit[index]], **kwargs
        )
        for index in range(2)
    ]

    assert torch.allclose(batched, sum(rows) / 2, atol=1e-6), (float(batched), [float(row) for row in rows])


def test_loss_of_a_ragged_batch_is_finite():
    present_labels, erased_labels = build_ragged_case()
    present_logits = torch.randn(3, 9, VOCAB_SIZE, requires_grad=True)
    credit = cig_sft_credit_func(
        torch.randn(3, 9, VOCAB_SIZE), present_labels, torch.randn(3, 9, VOCAB_SIZE), erased_labels
    )
    loss = cig_sft_loss_func({"logits": present_logits}, present_labels, credit, cig_lambda=0.3)

    assert torch.isfinite(loss)
    loss.backward()
    assert float(present_logits.grad.abs().sum()) > 0


def test_loss_when_a_response_sits_on_the_last_column():
    # the single token row is masked out of the sum, but its column index is still built and must stay in range
    labels = torch.tensor([[-100, -100, 21, 22, 23, 24, 25, 26], [-100] * 7 + [26]])
    erased_labels = torch.tensor([[-100, 21, 22, 23, 24, 25, 26, -100], [-100] * 6 + [26, -100]])
    logits = torch.randn(2, 8, VOCAB_SIZE, requires_grad=True)
    credit = cig_sft_credit_func(torch.randn(2, 8, VOCAB_SIZE), labels, torch.randn(2, 8, VOCAB_SIZE), erased_labels)
    loss = cig_sft_loss_func({"logits": logits}, labels, credit, cig_lambda=0.3)

    assert torch.isfinite(loss)
    loss.backward()
    assert float(logits.grad.abs().sum()) > 0


def test_loss_without_a_credit_keeps_the_graph_alive():
    present_logits, present_labels, _, _ = build_case()
    present_logits.requires_grad_(True)
    loss = cig_sft_loss_func({"logits": present_logits}, present_labels, [torch.zeros(0)], cig_lambda=0.3)

    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()


def test_loss_falls_back_to_the_model_loss_without_logits():
    assert float(cig_sft_loss_func({"loss": torch.tensor(1.5)}, torch.zeros(1, 2, dtype=torch.long), [])) == 1.5


def test_cig_evaluation_uses_the_ordinary_sft_loss(monkeypatch):
    called = {}

    def base_compute_loss(self, model, inputs, *args, **kwargs):
        called["inputs"] = inputs.copy()
        return torch.tensor(1.5)

    monkeypatch.setattr(Seq2SeqTrainer, "compute_loss", base_compute_loss)
    trainer = object.__new__(CustomSeq2SeqTrainer)
    trainer.finetuning_args = type("FinetuningArguments", (), {"use_cig_sft_loss": True})()
    model = torch.nn.Linear(1, 1)
    model.eval()
    inputs = {
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "labels": torch.ones(1, 2, dtype=torch.long),
        "erased_input_ids": torch.ones(1, 2, dtype=torch.long),
        "erased_attention_mask": torch.ones(1, 2, dtype=torch.long),
        "erased_labels": torch.ones(1, 2, dtype=torch.long),
    }

    loss = trainer.compute_loss(model, inputs)

    assert float(loss) == 1.5
    assert "erased_input_ids" not in called["inputs"]


# ------------------------------------------------------------------ gradient routing


def test_gradient_only_reaches_the_online_logits():
    present_logits, present_labels, erased_logits, erased_labels = build_case()
    present_logits.requires_grad_(True)
    credit = cig_sft_credit_func(present_logits, present_labels, erased_logits, erased_labels)
    loss = cig_sft_loss_func({"logits": present_logits}, present_labels, credit, cig_lambda=0.3)
    loss.backward()

    assert present_logits.grad is not None and float(present_logits.grad.abs().sum()) > 0
    # a supervised token at position p is scored by the cross entropy column p - 1, so the first two prompt
    # positions only provide context and the third one already predicts the first response token
    start, _ = _response_spans(present_labels)
    assert float(present_logits.grad[0, : int(start) - 1, :].abs().sum()) == 0.0
    assert erased_logits.grad is None  # the erased view is a frozen scorer


def test_credit_is_a_deterministic_function_of_the_reference_views():
    # the credit is read from the frozen reference model over both views and carries no gradient, so it cannot change
    # during training. That is what makes reading it once and replaying it from the cache lossless.
    present_logits, present_labels, erased_logits, erased_labels = build_case()
    first = cig_sft_credit_func(present_logits, present_labels, erased_logits, erased_labels)
    second = cig_sft_credit_func(present_logits, present_labels, erased_logits, erased_labels)

    assert all(torch.equal(before, after) for before, after in zip(first, second))
    assert all(not credit.requires_grad for credit in first)
    assert all(float(credit.min()) > 0.0 and float(credit.max()) < 1.0 for credit in first)
