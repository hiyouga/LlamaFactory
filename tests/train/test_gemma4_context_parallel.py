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

import pytest
import torch
from transformers import HfArgumentParser

from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import TrainingArguments
from llamafactory.hparams.megatron_bridge_args import MegatronBridgeArguments
from llamafactory.train.sft.context_parallel import (
    ContextParallelSampler,
    context_parallel_cross_entropy,
    make_shift_labels,
    split_sequence_inputs,
    validate_context_parallel_sft_args,
)
from llamafactory.train.sft.context_parallel_trainer import ContextParallelSeq2SeqTrainer
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.train.sft.workflow import _get_sft_trainer_class


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_gemma4_context_parallel_sampler_matches_contiguous_rank_groups(cp_size: int) -> None:
    world_size = 8
    sampler = ContextParallelSampler(range(16), cp_size)
    rank_samples = [list(sampler)[rank::world_size] for rank in range(world_size)]

    for first_rank in range(0, world_size, cp_size):
        assert all(rank_samples[rank] == rank_samples[first_rank] for rank in range(first_rank, first_rank + cp_size))

    distinct_group_samples = [rank_samples[first_rank] for first_rank in range(0, world_size, cp_size)]
    assert len({tuple(samples) for samples in distinct_group_samples}) == world_size // cp_size


def test_gemma4_context_parallel_wraps_existing_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    base_sampler = torch.utils.data.SequentialSampler(range(4))
    monkeypatch.setattr(CustomSeq2SeqTrainer, "_get_train_sampler", lambda *_args, **_kwargs: base_sampler)
    trainer = object.__new__(ContextParallelSeq2SeqTrainer)
    trainer.context_parallel_size = 2

    sampler = trainer._get_train_sampler()

    assert sampler.sampler is base_sampler
    assert list(sampler) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_gemma4_context_parallel_trainer_selection_is_opt_in() -> None:
    assert _get_sft_trainer_class(SimpleNamespace(ulysses_context_parallel_size=1)) is CustomSeq2SeqTrainer
    assert _get_sft_trainer_class(SimpleNamespace(ulysses_context_parallel_size=2)) is ContextParallelSeq2SeqTrainer
    assert _get_sft_trainer_class(SimpleNamespace(context_parallel_size=2)) is CustomSeq2SeqTrainer
    assert _get_sft_trainer_class(SimpleNamespace()) is CustomSeq2SeqTrainer


def test_ulysses_context_parallel_argument_coexists_with_megatron_bridge(tmp_path) -> None:
    native_parser = HfArgumentParser(TrainingArguments)
    (native_args,) = native_parser.parse_args_into_dataclasses(
        ["--output_dir", str(tmp_path / "native"), "--ulysses_context_parallel_size", "4"]
    )
    assert native_args.ulysses_context_parallel_size == 4

    megatron_parser = HfArgumentParser((TrainingArguments, MegatronBridgeArguments))
    training_args, megatron_args = megatron_parser.parse_args_into_dataclasses(
        [
            "--output_dir",
            str(tmp_path / "megatron"),
            "--ulysses_context_parallel_size",
            "2",
            "--context_parallel_size",
            "4",
        ]
    )
    assert training_args.ulysses_context_parallel_size == 2
    assert megatron_args.context_parallel_size == 4


def test_gemma4_shift_labels_is_exact_across_cp_boundary() -> None:
    labels = torch.tensor([[10, 11, 12, 13]])
    shifted = make_shift_labels(labels)

    assert shifted.tolist() == [[11, 12, 13, IGNORE_INDEX]]
    assert split_sequence_inputs({"input_ids": labels, "shift_labels": shifted}, 2, 0)["shift_labels"].tolist() == [
        [11, 12]
    ]
    assert split_sequence_inputs({"input_ids": labels, "shift_labels": shifted}, 2, 1)["shift_labels"].tolist() == [
        [13, IGNORE_INDEX]
    ]


def test_gemma4_context_parallel_loss_matches_full_sequence_mean() -> None:
    logits = torch.tensor(
        [[[4.0, 1.0], [1.0, 3.0], [2.0, 0.0], [0.0, 5.0]]],
        requires_grad=True,
    )
    shift_labels = torch.tensor([[0, 1, 0, 1]])
    reference = torch.nn.functional.cross_entropy(logits.view(-1, 2), shift_labels.view(-1))
    shard_losses = [
        context_parallel_cross_entropy(
            shard,
            label_shard,
            num_items_in_batch=shift_labels.numel(),
            world_size=2,
        )
        for shard, label_shard in zip(logits.chunk(2, dim=1), shift_labels.chunk(2, dim=1))
    ]

    assert torch.testing.assert_close(torch.stack(shard_losses).mean(), reference) is None


def test_gemma4_split_sequence_inputs_pads_to_cp_size() -> None:
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "labels": torch.tensor([[1, 2, 3, 4, 5]]),
    }
    shard = split_sequence_inputs(inputs, cp_size=4, cp_rank=3)

    assert shard["input_ids"].tolist() == [[0, 0]]
    assert shard["attention_mask"].tolist() == [[0, 0]]
    assert shard["labels"].tolist() == [[IGNORE_INDEX, IGNORE_INDEX]]


def test_gemma4_split_sequence_inputs_rejects_multimodal_batch() -> None:
    with pytest.raises(NotImplementedError, match="text-only"):
        split_sequence_inputs(
            {
                "input_ids": torch.ones(1, 8, dtype=torch.long),
                "pixel_values": torch.ones(1, 3, 16, 16),
            },
            cp_size=2,
            cp_rank=0,
        )


def test_gemma4_split_sequence_inputs_discards_empty_multimodal_placeholders() -> None:
    shard = split_sequence_inputs(
        {
            "input_ids": torch.ones(1, 8, dtype=torch.long),
            "pixel_values": torch.empty(0),
            "input_features": [],
        },
        cp_size=2,
        cp_rank=0,
    )

    assert "pixel_values" not in shard
    assert "input_features" not in shard


def test_gemma4_split_sequence_inputs_discards_masked_dummy_multimodal_inputs() -> None:
    shard = split_sequence_inputs(
        {
            "input_ids": torch.tensor([[10, 11, 12, 99, 99]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
            "labels": torch.tensor([[10, 11, 12, IGNORE_INDEX, IGNORE_INDEX]]),
            "mm_token_type_ids": torch.tensor([[0, 0, 0, 1, 1]]),
            "pixel_values": torch.ones(1, 3, 16, 16),
            "image_position_ids": torch.ones(1, 4, 2, dtype=torch.long),
        },
        cp_size=2,
        cp_rank=1,
    )

    assert shard["input_ids"].tolist() == [[12, 0]]
    assert shard["attention_mask"].tolist() == [[1, 0]]
    assert shard["mm_token_type_ids"].tolist() == [[0, 0]]
    assert "pixel_values" not in shard
    assert "image_position_ids" not in shard


def test_gemma4_split_sequence_inputs_aligns_shorter_multimodal_token_types() -> None:
    shard = split_sequence_inputs(
        {
            "input_ids": torch.tensor([[0, 10, 11, 99, 99]]),
            "attention_mask": torch.tensor([[0, 1, 1, 0, 0]]),
            "labels": torch.tensor([[IGNORE_INDEX, 10, 11, IGNORE_INDEX, IGNORE_INDEX]]),
            "mm_token_type_ids": torch.tensor([[0, 0, 1, 1]]),
            "pixel_values": torch.ones(1, 3, 16, 16),
        },
        cp_size=2,
        cp_rank=0,
    )

    assert shard["input_ids"].tolist() == [[0, 10]]
    assert shard["mm_token_type_ids"].tolist() == [[0, 0]]


def _cp_arguments(**overrides):
    arguments = {
        "data_args": SimpleNamespace(packing=False, neat_packing=False),
        "training_args": SimpleNamespace(
            ulysses_context_parallel_size=4,
            deepspeed="ds_z3.json",
            bf16=True,
            per_device_train_batch_size=1,
            dataloader_drop_last=True,
            eval_strategy="no",
            do_eval=False,
            do_predict=False,
            predict_with_generate=False,
            label_smoothing_factor=0.0,
            average_tokens_across_devices=True,
            parallelism_config=None,
        ),
        "finetuning_args": SimpleNamespace(
            use_dft_loss=False,
            use_eaft_loss=False,
            use_asft_loss=False,
        ),
    }
    for argument_name, values in overrides.items():
        for name, value in values.items():
            setattr(arguments[argument_name], name, value)
    return arguments


def test_validate_gemma4_context_parallel_sft_args() -> None:
    arguments = _cp_arguments()
    validate_context_parallel_sft_args(**arguments, world_size=8)


def test_reject_gemma4_context_parallel_world_size_one() -> None:
    arguments = _cp_arguments()
    with pytest.raises(ValueError, match="WORLD_SIZE"):
        validate_context_parallel_sft_args(**arguments, world_size=1)


@pytest.mark.parametrize("finetuning_type", ["full", "lora"])
def test_gemma4_context_parallel_accepts_training_feature_overlap(finetuning_type: str) -> None:
    arguments = _cp_arguments(
        training_args={"gradient_checkpointing": True, "gradient_accumulation_steps": 4},
        finetuning_args={"finetuning_type": finetuning_type, "disable_shuffling": True},
    )

    validate_context_parallel_sft_args(**arguments, world_size=8)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"training_args": {"deepspeed": None}}, "requires DeepSpeed"),
        ({"training_args": {"bf16": False}}, "requires BF16"),
        ({"training_args": {"per_device_train_batch_size": 2}}, "batch_size: 1"),
        ({"data_args": {"packing": True}}, "packed SFT"),
        ({"data_args": {"neat_packing": True}}, "packed SFT"),
        ({"training_args": {"do_eval": True}}, "Evaluation and prediction"),
        ({"training_args": {"predict_with_generate": True}}, "predict_with_generate"),
        ({"training_args": {"label_smoothing_factor": 0.1}}, "Label smoothing"),
        ({"training_args": {"average_tokens_across_devices": False}}, "average_tokens_across_devices"),
        ({"training_args": {"parallelism_config": object()}}, "native parallelism"),
        ({"finetuning_args": {"use_dft_loss": True}}, "Custom SFT losses"),
    ],
)
def test_reject_unsupported_gemma4_context_parallel_args(overrides, match: str) -> None:
    arguments = _cp_arguments(**overrides)
    with pytest.raises(ValueError, match=match):
        validate_context_parallel_sft_args(**arguments, world_size=8)
