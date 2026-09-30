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

from types import SimpleNamespace

import pytest

from llamafactory.hparams.megatron_bridge_args import MegatronBridgeArguments
from llamafactory.train.megatron_bridge.config_builder import (
    _apply_extra_overrides,
    _apply_fusion_safety,
    _apply_model_parallelism,
    _compute_train_schedule,
    _resolve_warmup_steps,
    _should_resume_checkpoint,
    build_pretrain_config,
    build_sft_config,
)


class _DummyProvider:
    tensor_model_parallel_size = 1
    pipeline_model_parallel_size = 1
    context_parallel_size = 1
    sequence_parallel = False
    gradient_accumulation_fusion = True


def test_compute_train_schedule_tp1():
    training_args = SimpleNamespace(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        max_steps=-1,
        world_size=2,
    )
    mb_args = MegatronBridgeArguments(tensor_model_parallel_size=1)
    micro_batch, global_batch, train_iters = _compute_train_schedule(training_args, mb_args, num_train_samples=32)
    assert micro_batch == 2
    assert global_batch == 16
    assert train_iters == 2


def test_compute_train_schedule_tp2():
    training_args = SimpleNamespace(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=-1,
        world_size=2,
    )
    mb_args = MegatronBridgeArguments(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
    )
    micro_batch, global_batch, train_iters = _compute_train_schedule(training_args, mb_args, num_train_samples=10)
    assert micro_batch == 1
    assert global_batch == 1
    assert train_iters == 10


def test_compute_train_schedule_max_steps():
    training_args = SimpleNamespace(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=10,
        max_steps=5,
        world_size=1,
    )
    mb_args = MegatronBridgeArguments()
    _, _, train_iters = _compute_train_schedule(training_args, mb_args, num_train_samples=1000)
    assert train_iters == 5


def test_apply_model_parallelism():
    provider = _DummyProvider()
    mb_args = MegatronBridgeArguments(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=True,
        bias_activation_fusion=True,
        moe_token_dispatcher_type="alltoall",
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=2,
    )
    provider.bias_activation_fusion = False
    provider.moe_token_dispatcher_type = "allgather"
    provider.recompute_granularity = None
    provider.recompute_method = None
    provider.recompute_num_layers = None
    _apply_model_parallelism(provider, mb_args)
    assert provider.tensor_model_parallel_size == 2
    assert provider.sequence_parallel is True
    assert provider.bias_activation_fusion is True
    assert provider.moe_token_dispatcher_type == "alltoall"
    assert provider.recompute_granularity == "full"
    assert provider.recompute_method == "uniform"
    assert provider.recompute_num_layers == 2


def test_megatron_bridge_args_rejects_invalid_moe_dispatcher():
    with pytest.raises(ValueError, match="moe_token_dispatcher_type"):
        MegatronBridgeArguments(moe_token_dispatcher_type="invalid")  # type: ignore[arg-type]


def test_apply_fusion_safety_disables_missing_apex():
    provider = _DummyProvider()
    _apply_fusion_safety(provider)
    assert provider.gradient_accumulation_fusion is False


def test_megatron_bridge_args_sequence_parallel_requires_tp():
    with pytest.raises(ValueError, match="sequence_parallel"):
        MegatronBridgeArguments(sequence_parallel=True, tensor_model_parallel_size=1)


def test_resolve_warmup_steps_keeps_absolute_warmup():
    training_args = SimpleNamespace(warmup_steps=10, warmup_ratio=0.0)
    assert _resolve_warmup_steps(training_args, train_iters=5) == 10


def test_resolve_decay_iters_expands_for_warmup():
    from llamafactory.train.megatron_bridge.config_builder import _resolve_decay_iters

    assert _resolve_decay_iters(train_iters=5, warmup_steps=10) == 11
    assert _resolve_decay_iters(train_iters=20, warmup_steps=10) == 20
    assert _resolve_decay_iters(train_iters=5, warmup_steps=0) == 5


def test_megatron_bridge_args_extra_config_strips_whitespace():
    args = MegatronBridgeArguments(extra_config='  {"train.train_iters": 3}  ')
    assert args.extra_config == {"train.train_iters": 3}


def test_apply_extra_overrides_nested_path():
    cfg = SimpleNamespace(train=SimpleNamespace(nested=SimpleNamespace(value=1)))
    _apply_extra_overrides(cfg, {"train.nested.value": 9})
    assert cfg.train.nested.value == 9


def test_should_resume_checkpoint_respects_overwrite(tmp_path):
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("5")
    assert _should_resume_checkpoint(SimpleNamespace(output_dir=str(tmp_path), overwrite_output_dir=False))
    assert not _should_resume_checkpoint(SimpleNamespace(output_dir=str(tmp_path), overwrite_output_dir=True))
    assert not _should_resume_checkpoint(
        SimpleNamespace(output_dir=str(tmp_path / "missing"), overwrite_output_dir=False)
    )


@pytest.mark.runs_on(["cuda"])
def test_build_sft_config_on_gpu(mb_training_args_factory, mb_output_dir):
    model_args, data_args, training_args, finetuning_args, mb_args, num_train_samples = mb_training_args_factory(
        tensor_model_parallel_size=1,
        sequence_parallel=False,
    )
    cfg = build_sft_config(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        mb_args=mb_args,
        dataset_root=str(mb_output_dir / "dataset"),
        pretrained_checkpoint=str(mb_output_dir / "pretrained"),
        num_train_samples=num_train_samples,
    )
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.seq_length == data_args.cutoff_len
    assert cfg.train.train_iters == num_train_samples
    assert cfg.dataset.dataset_root == str(mb_output_dir / "dataset")
    assert cfg.tokenizer.tokenizer_model == model_args.model_name_or_path


@pytest.mark.runs_on(["cuda"])
def test_build_pretrain_config_on_gpu(mb_training_args_factory, mb_output_dir):
    model_args, data_args, training_args, finetuning_args, mb_args, num_train_samples = mb_training_args_factory(
        tensor_model_parallel_size=1,
    )
    finetuning_args.stage = "pt"
    dataset_path = str(mb_output_dir / "dataset" / "training.jsonl")
    cfg = build_pretrain_config(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        mb_args=mb_args,
        dataset_path=dataset_path,
        num_train_samples=num_train_samples,
    )
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.dataset.blend[0][0] == dataset_path
    assert cfg.train.train_iters == num_train_samples
