# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from hyper_parallel.core.optimizer import ChainedOptimizer

from llamafactory.train.hyper_parallel.loader import load_hyper_parallel_model
from llamafactory.train.hyper_parallel.model_registry import apply_model_parallel_plan
from llamafactory.train.hyper_parallel.trainer import (
    HyperParallelTrainer,
    _build_cp_shift_labels,
    _create_hyper_muon_optimizer,
    _pad_inputs_for_model_cp,
    _shard_inputs_for_cp,
)
from llamafactory.train.hyper_parallel.workflow import _prepare_hp_args


def test_create_hyper_muon_optimizer_preserves_llamafactory_parameter_split():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(8, 4)
            self.proj = torch.nn.Linear(4, 4)
            self.norm = torch.nn.LayerNorm(4)
            self.lm_head = torch.nn.Linear(4, 8, bias=False)

    model = _Model()
    training_args = SimpleNamespace(
        learning_rate=3.0e-4,
        weight_decay=0.1,
        adam_beta1=0.8,
        adam_beta2=0.95,
        adam_epsilon=1.0e-8,
    )
    expected_optimizer = ChainedOptimizer(
        model,
        {"adamw": torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)},
    )

    with patch(
        "llamafactory.train.hyper_parallel.trainer.get_hyper_optimizer",
        return_value=expected_optimizer,
    ) as get_optimizer:
        optimizer = _create_hyper_muon_optimizer(model, training_args)

    assert isinstance(optimizer, torch.optim.Optimizer)
    assert optimizer.optimizer is expected_optimizer
    call_kwargs = get_optimizer.call_args.kwargs
    assert call_kwargs["muon_params"][0]["params"] == [model.proj.weight]
    assert set(call_kwargs["adamw_params"][0]["params"]) == {
        model.embed_tokens.weight,
        model.proj.bias,
        model.norm.weight,
        model.norm.bias,
        model.lm_head.weight,
    }
    assert call_kwargs["muon_kwargs"] == {"muon_lr": 3.0e-4, "muon_weight_decay": 0.1}
    assert call_kwargs["adamw_kwargs"] == {
        "adamw_lr": 3.0e-4,
        "adamw_weight_decay": 0.1,
        "adamw_betas": (0.8, 0.95),
        "adamw_eps": 1.0e-8,
    }
    optimizer.param_groups[0]["lr"] = 1.0e-4
    assert expected_optimizer.chained_optimizers[0].param_groups[0]["lr"] == 1.0e-4


def test_cp_batch_adapter_keeps_global_mask_and_generates_local_positions():
    class _CPMesh:
        @staticmethod
        def size():
            return 2

        @staticmethod
        def get_local_rank():
            return 1

    inputs = {
        "input_ids": torch.arange(8).unsqueeze(0),
        "labels": torch.arange(8).unsqueeze(0),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
    }

    sharded = _shard_inputs_for_cp(inputs, _CPMesh())

    torch.testing.assert_close(sharded["input_ids"], torch.arange(4, 8).unsqueeze(0))
    torch.testing.assert_close(sharded["labels"], inputs["labels"])
    torch.testing.assert_close(sharded["position_ids"], torch.arange(4, 8).unsqueeze(0))
    torch.testing.assert_close(sharded["attention_mask"], inputs["attention_mask"])


def test_cp_shift_labels_preserve_cross_rank_causal_boundary():
    labels = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])

    rank0_labels = _build_cp_shift_labels(labels, local_seq_len=4, cp_rank=0, ignore_index=-100)
    rank1_labels = _build_cp_shift_labels(labels, local_seq_len=4, cp_rank=1, ignore_index=-100)

    torch.testing.assert_close(rank0_labels, torch.tensor([[11, 12, 13, 14]]))
    torch.testing.assert_close(rank1_labels, torch.tensor([[15, 16, 17, -100]]))


def test_model_cp_padding_preserves_multimodal_fields():
    class _CPMesh:
        @staticmethod
        def size():
            return 2

    pixel_values = torch.randn(6, 12)
    image_grid_thw = torch.tensor([[1, 2, 3]])
    inputs = {
        "input_ids": torch.arange(6).unsqueeze(0),
        "labels": torch.arange(6).unsqueeze(0),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
        "mm_token_type_ids": torch.zeros(1, 6, dtype=torch.long),
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }

    padded = _pad_inputs_for_model_cp(inputs, _CPMesh())

    assert padded["input_ids"].shape[-1] == 8
    torch.testing.assert_close(padded["labels"][..., -2:], torch.full((1, 2), -100))
    torch.testing.assert_close(padded["attention_mask"][..., -2:], torch.zeros(1, 2, dtype=torch.long))
    assert padded["pixel_values"] is pixel_values
    assert padded["image_grid_thw"] is image_grid_thw


def test_accelerate_patches_skip_torch_auto_wrap_for_prepared_hyper_model():
    class _FSDPPlugin:
        def __init__(self):
            self.calls = []

        def set_auto_wrap_policy(self, model):
            self.calls.append(model)

    plugin = _FSDPPlugin()
    original_clip = object()
    trainer = object.__new__(HyperParallelTrainer)
    trainer.accelerator = SimpleNamespace(
        state=SimpleNamespace(fsdp_plugin=plugin),
        clip_grad_norm_=original_clip,
    )
    trainer._orig_accelerator_clip_grad_norm = original_clip
    trainer._orig_fsdp2_prepare_model = None
    trainer._orig_fsdp2_set_auto_wrap_policy = None
    trainer._accelerator_patches_active = False
    prepared_models = []
    trainer._prepare_model_for_hyper_parallel = lambda model: prepared_models.append(model) or model

    trainer._activate_accelerator_patches()
    model = object()
    plugin.set_auto_wrap_policy(model)
    trainer._restore_accelerator_patches()

    assert prepared_models == [model]
    assert plugin.calls == []
    assert plugin.set_auto_wrap_policy == trainer._orig_fsdp2_set_auto_wrap_policy


def test_training_step_compensates_accelerate_gradient_accumulation_scaling():
    for model_accepts_loss_kwargs, expected_loss in ((True, 8.0), (False, 4.0)):
        model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.ones_(model.weight)
        trainer = object.__new__(HyperParallelTrainer)
        trainer._cp_size = 1
        trainer.args = SimpleNamespace(n_gpu=1, gradient_accumulation_steps=2)
        trainer.current_gradient_accumulation_steps = 2
        trainer.model_accepts_loss_kwargs = model_accepts_loss_kwargs
        trainer.compute_loss_func = None
        trainer._prepare_inputs = lambda inputs: inputs
        trainer.compute_loss_context_manager = nullcontext
        trainer.compute_loss = lambda model, inputs, num_items_in_batch=None: model.weight.sum() * 8
        trainer.accelerator = SimpleNamespace(
            sync_gradients=True,
            gradient_accumulation_steps=2,
            backward=lambda loss: (loss / 2).backward(),
        )

        logged_loss = trainer.training_step(model, {}, num_items_in_batch=1)

        torch.testing.assert_close(logged_loss, torch.tensor(expected_loss))
        torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, expected_loss))


def test_hyper_model_loader_uses_meta_then_parallelize_path():
    config = SimpleNamespace(model_type="test")
    model = torch.nn.Linear(2, 2, device="meta")
    model.config = config
    calls = []

    def _apply_v1_kernels(input_model, use_v1_kernels):
        assert input_model is model
        assert use_v1_kernels is True
        calls.append("v1")
        return input_model

    def _parallelize_model(input_model, *args, **kwargs):
        assert input_model is model
        calls.append("parallelize")
        return input_model

    class _AutoModel:
        @staticmethod
        def from_config(input_config, trust_remote_code=False, dtype=None):
            assert input_config is config
            assert trust_remote_code is False
            assert dtype is torch.bfloat16
            return model

    model_args = SimpleNamespace(
        model_name_or_path="checkpoint",
        train_from_scratch=False,
        trust_remote_code=False,
        use_kt=False,
        use_v1_kernels=True,
        print_param_status=False,
        compute_dtype=torch.bfloat16,
    )
    finetuning_args = SimpleNamespace(stage="sft")
    hp_args = SimpleNamespace(activation_mode="none", activation_swap_inputs=True)
    distributed_setup = object()

    with (
        patch("llamafactory.train.hyper_parallel.loader._get_init_kwargs", return_value={}),
        patch("llamafactory.train.hyper_parallel.loader.load_config", return_value=config),
        patch("llamafactory.train.hyper_parallel.loader.patch_config"),
        patch("llamafactory.train.hyper_parallel.loader.apply_liger_kernel"),
        patch("llamafactory.train.hyper_parallel.loader._get_model_class", return_value=_AutoModel),
        patch("llamafactory.train.hyper_parallel.loader._get_no_init_weights", return_value=nullcontext),
        patch("llamafactory.train.hyper_parallel.loader.init_empty_weights", return_value=nullcontext()),
        patch("llamafactory.train.hyper_parallel.loader.patch_model"),
        patch("llamafactory.train.hyper_parallel.loader.register_autoclass"),
        patch("llamafactory.train.hyper_parallel.loader.init_adapter", return_value=model),
        patch(
            "llamafactory.v1.plugins.model_plugins.kernels.interface.apply_v1_kernels",
            side_effect=_apply_v1_kernels,
        ) as apply_v1_kernels,
        patch(
            "llamafactory.train.hyper_parallel.loader.parallelize_model",
            side_effect=_parallelize_model,
        ) as parallelize,
        patch("llamafactory.train.hyper_parallel.loader._validate_conv3d_compatibility"),
    ):
        result = load_hyper_parallel_model(
            tokenizer=object(),
            model_args=model_args,
            finetuning_args=finetuning_args,
            distributed_setup=distributed_setup,
            hp_args=hp_args,
            is_trainable=True,
        )

    assert result is model
    assert calls == ["v1", "parallelize"]
    apply_v1_kernels.assert_called_once_with(model, use_v1_kernels=True)
    parallelize.assert_called_once_with(
        model,
        distributed_setup,
        pretrained_path="checkpoint",
        activation_checkpoint=None,
        swap_inputs=False,
    )


def _make_hyper_workflow_args(finetuning_type="full"):
    finetuning_args = SimpleNamespace(
        hyper_parallel_args=None,
        hyper_parallel_cp_size=1,
        hyper_parallel_ep_size=1,
        hyper_parallel_efsdp_size=1,
        hyper_parallel_token_dispatcher="all_to_all",
        finetuning_type=finetuning_type,
        use_asft_loss=False,
    )
    model_args = SimpleNamespace(
        disable_gradient_checkpointing=False,
        quantization_bit=None,
        use_unsloth=False,
        mixture_of_depths=None,
    )
    return finetuning_args, model_args


def test_prepare_hp_args_accepts_supported_full_tuning():
    finetuning_args, model_args = _make_hyper_workflow_args()
    training_args = SimpleNamespace(bf16=True, fp16=False)

    with patch("hyper_parallel.integration.llamafactory.utils.dist.get_world_size", return_value=1):
        hp_args = _prepare_hp_args(finetuning_args, model_args, training_args)

    assert hp_args.cp_size == 1
    assert hp_args.ep_size == 1
    assert hp_args.param_dtype == "bfloat16"


def test_prepare_hp_args_rejects_parameter_efficient_tuning():
    finetuning_args, model_args = _make_hyper_workflow_args(finetuning_type="lora")
    training_args = SimpleNamespace(bf16=True, fp16=False)

    try:
        _prepare_hp_args(finetuning_args, model_args, training_args)
    except ValueError as exc:
        assert "requires full fine-tuning" in str(exc)
    else:
        raise AssertionError("Expected HyperParallel to reject LoRA tuning.")


def test_qwen3_moe_parallel_plan_resolves_through_hyper_builder():
    from hyper_parallel.integration.llamafactory import HyperParallelArguments

    hp_args = HyperParallelArguments(
        cp_size=2,
        ep_size=4,
        device_type="npu",
        plan_overrides=[{"match": "*.mlp", "region_dispatch": True}],
    )
    apply_model_parallel_plan(hp_args, "qwen3_moe")

    with (
        patch("hyper_parallel.integration.llamafactory.utils.dist.get_world_size", return_value=8),
        patch("hyper_parallel.integration.llamafactory.utils.MeshContext.build_meshs"),
    ):
        setup = hp_args.build_distributed_setup()

    cp_spec = setup.plan_overrides["*.self_attn"]
    ep_spec = setup.plan_overrides["*.mlp"]
    assert cp_spec.inner_wrapper.callable.__name__ == "qwen3_moe_async_ulysses_cp_wrapper"
    assert ep_spec.local_compute_fn.callable.__name__ == "qwen3moe_ep_compute_fn"
    assert ep_spec.local_compute_fn.use_grouped_gemm is True
    assert ep_spec.region_dispatch is True
    assert not setup.module_replacements


def test_qwen3_vl_moe_parallel_plan_only_uses_its_model_adapter():
    hp_args = SimpleNamespace(plan_overrides=None)

    apply_model_parallel_plan(hp_args, "qwen3_vl_moe")

    targets = str(hp_args.plan_overrides)
    assert "hyper_parallel.models.qwen3_vl_moe.adapter" in targets
    assert "hyper_parallel.models.qwen3_moe.adapter" not in targets
    assert {entry["match"] for entry in hp_args.plan_overrides} == {
        "model.visual.blocks.*.attn",
        "model.visual.merger",
        "model.visual.deepstack_merger_list",
        "model.language_model",
        "model.language_model.layers.*.self_attn",
        "model.language_model.layers.*.mlp",
    }
    for entry in hp_args.plan_overrides:
        if entry["match"] in {"model.visual.merger", "model.visual.deepstack_merger_list"}:
            assert entry["params"] == {}
