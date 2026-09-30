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
from typing import TYPE_CHECKING, Any

from ...extras.logging import get_logger


if TYPE_CHECKING:
    from ...hparams import (
        DataArguments,
        FinetuningArguments,
        MegatronBridgeArguments,
        ModelArguments,
        TrainingArguments,
    )


logger = get_logger(__name__)

_LR_SCHEDULER_MAP = {
    "cosine": "cosine",
    "linear": "linear",
    "constant": "constant",
    "constant_with_warmup": "constant",
}


def _map_lr_scheduler_type(lr_scheduler_type: str) -> str:
    mapped = _LR_SCHEDULER_MAP.get(lr_scheduler_type)
    if mapped is None:
        logger.warning_rank0(
            f"lr_scheduler_type '{lr_scheduler_type}' is not supported by Megatron Bridge; using cosine."
        )
        return "cosine"
    return mapped


def _resolve_warmup_steps(training_args: "TrainingArguments", train_iters: int) -> int:
    r"""Resolve warmup steps with Hugging Face Trainer semantics.

    Absolute ``warmup_steps`` are kept as-is (even when larger than ``train_iters``)
    so short debugging runs with ``max_steps < warmup_steps`` match HF LR values.
    ``lr_decay_iters`` is expanded separately to avoid Megatron capping warmup.
    """
    warmup_steps = getattr(training_args, "warmup_steps", 0) or 0
    if warmup_steps > 0:
        return warmup_steps

    warmup_ratio = getattr(training_args, "warmup_ratio", 0.0) or 0.0
    if warmup_ratio > 0:
        return min(int(train_iters * warmup_ratio), train_iters)
    return 0


def _resolve_decay_iters(train_iters: int, warmup_steps: int) -> int:
    r"""Ensure decay span is longer than warmup so Megatron does not shrink warmup.

    Megatron Bridge caps ``lr_warmup_steps`` whenever it is ``>= lr_decay_steps``,
    so keep decay strictly larger than warmup on short comparison runs.
    """
    if warmup_steps <= 0:
        return train_iters
    return max(train_iters, warmup_steps + 1)


def _import_training_config():
    from megatron.bridge.training.config import (
        CheckpointConfig,
        ConfigContainer,
        FinetuningDatasetConfig,
        GPTDatasetConfig,
        LoggerConfig,
        RNGConfig,
        TrainingConfig,
    )

    try:
        from megatron.bridge.training.config import DistributedInitConfig
    except ImportError:
        DistributedInitConfig = None

    try:
        from megatron.bridge.training.config import ValidationConfig
    except ImportError:
        ValidationConfig = None

    try:
        from megatron.bridge.training.tokenizers.config import TokenizerConfig
    except ImportError:
        from megatron.bridge.training.config import TokenizerConfig

    return (
        CheckpointConfig,
        ConfigContainer,
        DistributedInitConfig,
        FinetuningDatasetConfig,
        GPTDatasetConfig,
        LoggerConfig,
        RNGConfig,
        TokenizerConfig,
        TrainingConfig,
        ValidationConfig,
    )


def _create_optimizer_scheduler(
    training_args: "TrainingArguments",
    warmup_steps: int,
    train_iters: int,
    finetuning_args: "FinetuningArguments",
    use_distributed_optimizer: bool,
):
    from megatron.bridge.training.config import OptimizerConfig, SchedulerConfig

    finetuning_type = finetuning_args.finetuning_type
    learning_rate = training_args.learning_rate
    if finetuning_type in ("lora", "full"):
        max_lr, min_lr, default_beta2 = learning_rate, 0.0, 0.999
    else:
        max_lr, min_lr, default_beta2 = learning_rate, learning_rate * 0.1, 0.95

    # Match Hugging Face Trainer defaults unless the user overrides them.
    adam_beta1 = getattr(training_args, "adam_beta1", 0.9)
    adam_beta2 = getattr(training_args, "adam_beta2", default_beta2)
    adam_eps = getattr(training_args, "adam_epsilon", 1e-8)
    weight_decay = getattr(training_args, "weight_decay", 0.0)
    max_grad_norm = getattr(training_args, "max_grad_norm", 1.0)
    decay_iters = _resolve_decay_iters(train_iters, warmup_steps)
    optimizer = OptimizerConfig(
        optimizer="adam",
        lr=max_lr,
        min_lr=min_lr,
        weight_decay=weight_decay,
        bf16=getattr(training_args, "bf16", True),
        fp16=getattr(training_args, "fp16", False),
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_eps=adam_eps,
        use_distributed_optimizer=use_distributed_optimizer,
        clip_grad=max_grad_norm,
    )
    scheduler = SchedulerConfig(
        start_weight_decay=weight_decay,
        end_weight_decay=weight_decay,
        weight_decay_incr_style="constant",
        lr_decay_style=_map_lr_scheduler_type(getattr(training_args, "lr_scheduler_type", "cosine")),
        lr_wsd_decay_style="minus_sqrt",
        lr_wsd_decay_iters=decay_iters,
        lr_warmup_iters=warmup_steps,
        lr_warmup_init=0.0,
        lr_decay_iters=decay_iters,
        override_opt_param_scheduler=True,
    )
    return optimizer, scheduler


def ensure_create_sft_dataset_applies_chat_template() -> None:
    r"""Apply ``dataset_kwargs['chat_template']`` before Megatron builds SFT datasets.

    Megatron Bridge only pops ``chat_template`` on the packed-sequence path. The
    non-packed finetuning path would otherwise forward it as an unexpected kwarg.
    Patch both the defining module and the builder import binding.
    """
    from megatron.bridge.data.builders import finetuning_dataset as finetuning_module
    from megatron.bridge.data.datasets import sft as sft_module

    if getattr(sft_module.create_sft_dataset, "_llamafactory_chat_template_patched", False):
        return

    original = sft_module.create_sft_dataset

    def create_sft_dataset(*args, **kwargs):
        chat_template = kwargs.pop("chat_template", None)
        tokenizer = kwargs.get("tokenizer")
        if tokenizer is None and len(args) >= 2:
            tokenizer = args[1]
        if chat_template is not None and tokenizer is not None:
            # Megatron `_chat_preprocess` may read either the wrapper or the
            # inner HuggingFace tokenizer depending on `legacy`.
            if hasattr(tokenizer, "chat_template"):
                tokenizer.chat_template = chat_template
            hf_tokenizer = getattr(tokenizer, "_tokenizer", None)
            if hf_tokenizer is not None and hasattr(hf_tokenizer, "chat_template"):
                hf_tokenizer.chat_template = chat_template
        return original(*args, **kwargs)

    create_sft_dataset._llamafactory_chat_template_patched = True  # type: ignore[attr-defined]
    sft_module.create_sft_dataset = create_sft_dataset
    finetuning_module.create_sft_dataset = create_sft_dataset
    logger.info_rank0("Patched Megatron create_sft_dataset to apply chat_template overrides.")


def _create_peft_config(finetuning_args: "FinetuningArguments"):
    if finetuning_args.finetuning_type != "lora":
        return None

    from megatron.bridge.peft.lora import LoRA

    default_targets = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    if list(finetuning_args.lora_target) != ["all"]:
        logger.warning_rank0(
            f"Custom lora_target {finetuning_args.lora_target} is not supported by Megatron Bridge. "
            f"Using default Megatron target modules: {default_targets}."
        )

    return LoRA(
        target_modules=default_targets,
        dim=finetuning_args.lora_rank,
        alpha=finetuning_args.lora_alpha,
    )


def _build_gpt_dataset_config(
    GPTDatasetConfig,
    dataset_path: str,
    seq_length: int,
    seed: int,
    num_workers: int,
):
    kwargs: dict[str, Any] = {
        "random_seed": seed,
        "reset_attention_mask": False,
        "reset_position_ids": False,
        "eod_mask_loss": False,
        "blend": ([dataset_path], 1.0),
        "split": "100,0,0",
        "num_workers": num_workers,
        "data_sharding": True,
        "dataloader_type": "single",
    }
    if "sequence_length" in GPTDatasetConfig.__dataclass_fields__:
        kwargs["sequence_length"] = seq_length
    else:
        kwargs["seq_length"] = seq_length
    if "num_dataset_builder_threads" in GPTDatasetConfig.__dataclass_fields__:
        kwargs["num_dataset_builder_threads"] = 1
    return GPTDatasetConfig(**kwargs)


def _build_finetuning_dataset_config(
    FinetuningDatasetConfig,
    dataset_root: str,
    seq_length: int,
    seed: int,
    num_workers: int,
    do_validation: bool,
    dataset_kwargs: dict[str, Any],
    packed_sequence_specs,
    disable_shuffling: bool = False,
):
    kwargs: dict[str, Any] = {
        "dataset_root": dataset_root,
        "seq_length": seq_length,
        "seed": seed,
        "num_workers": num_workers,
        "do_validation": do_validation,
        "do_test": False,
        "dataset_kwargs": dataset_kwargs,
        "packed_sequence_specs": packed_sequence_specs,
    }
    if "dataloader_type" in FinetuningDatasetConfig.__dataclass_fields__:
        from .dataset_export import get_finetuning_dataloader_type

        kwargs["dataloader_type"] = get_finetuning_dataloader_type(disable_shuffling=disable_shuffling)
    return FinetuningDatasetConfig(**kwargs)


def _has_megatron_checkpoint(output_dir: str) -> bool:
    r"""Return whether ``output_dir`` contains a resumable Megatron checkpoint."""
    return any(
        os.path.isfile(os.path.join(output_dir, name))
        for name in ("latest_checkpointed_iteration.txt", "latest_train_state.pt")
    )


def _should_resume_checkpoint(training_args: "TrainingArguments") -> bool:
    r"""Resume only when a tracker exists and ``overwrite_output_dir`` is false."""
    if getattr(training_args, "overwrite_output_dir", False):
        return False
    return _has_megatron_checkpoint(training_args.output_dir)


def _create_base_config(
    *,
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
    train_iters: int,
    micro_batch_size: int,
    global_batch_size: int,
    mb_args: "MegatronBridgeArguments",
    is_sft: bool,
):
    from megatron.core.distributed import DistributedDataParallelConfig

    (
        CheckpointConfig,
        ConfigContainer,
        DistributedInitConfig,
        _FinetuningDatasetConfig,
        _GPTDatasetConfig,
        LoggerConfig,
        RNGConfig,
        TokenizerConfig,
        TrainingConfig,
        ValidationConfig,
    ) = _import_training_config()

    warmup_steps = _resolve_warmup_steps(training_args, train_iters)
    opt_cfg, scheduler_cfg = _create_optimizer_scheduler(
        training_args=training_args,
        warmup_steps=warmup_steps,
        train_iters=train_iters,
        finetuning_args=finetuning_args,
        use_distributed_optimizer=mb_args.use_distributed_optimizer,
    )

    train_kwargs: dict[str, Any] = {
        "train_iters": train_iters,
        "global_batch_size": global_batch_size,
        "micro_batch_size": micro_batch_size,
    }
    eval_steps = training_args.eval_steps
    if ValidationConfig is not None:
        validation = ValidationConfig(eval_interval=eval_steps or 100, eval_iters=32)
    else:
        train_kwargs["eval_interval"] = eval_steps or 100
        train_kwargs["eval_iters"] = 32
        validation = None

    output_dir = training_args.output_dir
    resume_checkpoint = _should_resume_checkpoint(training_args)
    # mcore >= 0.14 removed ShardedTensor.flattened_range. The legacy default
    # sharding type ``fully_sharded_model_space`` still depends on it, so prefer
    # the fully_reshardable distributed-optimizer format when dist opt is on.
    dist_ckpt_optim_fully_reshardable = mb_args.use_distributed_optimizer
    dist_cfg = DistributedInitConfig() if DistributedInitConfig is not None else None
    container_kwargs: dict[str, Any] = {
        "model": None,
        "train": TrainingConfig(**train_kwargs),
        "optimizer": opt_cfg,
        "scheduler": scheduler_cfg,
        "ddp": DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=mb_args.overlap_grad_reduce,
            overlap_param_gather=mb_args.overlap_param_gather,
            use_distributed_optimizer=mb_args.use_distributed_optimizer,
        ),
        "dataset": None,
        "logger": LoggerConfig(
            log_interval=training_args.logging_steps,
            tensorboard_dir=os.path.join(output_dir, "tb_logs"),
        ),
        "tokenizer": TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=None,
        ),
        "checkpoint": CheckpointConfig(
            save_interval=training_args.save_steps,
            save=output_dir,
            load=output_dir if resume_checkpoint else None,
            # SFT from pretrained should not load optimizer/RNG from a partial ckpt.
            finetune=is_sft and not resume_checkpoint,
            ckpt_format="torch_dist",
            fully_parallel_save=False,
            use_persistent_ckpt_worker=False,
            save_optim=True,
            dist_ckpt_optim_fully_reshardable=dist_ckpt_optim_fully_reshardable,
        ),
        "rng": RNGConfig(seed=training_args.seed),
        "mixed_precision": mb_args.mixed_precision,
        "peft": _create_peft_config(finetuning_args) if is_sft and finetuning_args.finetuning_type == "lora" else None,
    }
    if validation is not None:
        container_kwargs["validation"] = validation
    if dist_cfg is not None:
        container_kwargs["dist"] = dist_cfg

    return ConfigContainer(**container_kwargs)


def _compute_train_schedule(
    training_args: "TrainingArguments",
    mb_args: "MegatronBridgeArguments",
    num_train_samples: int,
) -> tuple[int, int, int]:
    micro_batch_size = training_args.per_device_train_batch_size
    global_batch_size = micro_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    parallel_size = (
        mb_args.tensor_model_parallel_size
        * mb_args.pipeline_model_parallel_size
        * mb_args.context_parallel_size
        * mb_args.expert_model_parallel_size
    )
    global_batch_size //= parallel_size
    global_batch_size = max(global_batch_size, micro_batch_size)

    max_steps = getattr(training_args, "max_steps", -1)
    if max_steps is not None and max_steps > 0:
        train_iters = max_steps
    else:
        train_iters = max(1, math.ceil(num_train_samples / global_batch_size * training_args.num_train_epochs))
    return micro_batch_size, global_batch_size, train_iters


def _is_apex_grad_accum_fusion_available() -> bool:
    try:
        import fused_weight_gradient_mlp_cuda  # noqa: F401

        return True
    except ImportError:
        return False


def _apply_fusion_safety(model_provider) -> None:
    r"""Disable gradient_accumulation_fusion when the APEX CUDA extension is missing.

    Megatron Bridge enables this fusion when TransformerEngine is installed, but
    ColumnParallelLinear (e.g. the output layer) still requires the APEX
    fused_weight_gradient_mlp_cuda extension at model construction time.
    """
    if getattr(model_provider, "gradient_accumulation_fusion", False) and not _is_apex_grad_accum_fusion_available():
        logger.warning_rank0(
            "Disabling gradient_accumulation_fusion because the APEX CUDA extension "
            "fused_weight_gradient_mlp_cuda is not installed."
        )
        model_provider.gradient_accumulation_fusion = False


def _apply_optional_provider_attr(model_provider, name: str, value) -> None:
    if value is None or not hasattr(model_provider, name):
        return
    setattr(model_provider, name, value)


def _apply_model_parallelism(model_provider, mb_args: "MegatronBridgeArguments") -> None:
    model_provider.tensor_model_parallel_size = mb_args.tensor_model_parallel_size
    model_provider.pipeline_model_parallel_size = mb_args.pipeline_model_parallel_size
    if hasattr(model_provider, "expert_model_parallel_size"):
        model_provider.expert_model_parallel_size = mb_args.expert_model_parallel_size
    model_provider.context_parallel_size = mb_args.context_parallel_size
    model_provider.sequence_parallel = mb_args.sequence_parallel
    _apply_optional_provider_attr(
        model_provider, "virtual_pipeline_model_parallel_size", mb_args.virtual_pipeline_model_parallel_size
    )
    _apply_optional_provider_attr(model_provider, "recompute_granularity", mb_args.recompute_granularity)
    _apply_optional_provider_attr(model_provider, "recompute_method", mb_args.recompute_method)
    _apply_optional_provider_attr(model_provider, "recompute_num_layers", mb_args.recompute_num_layers)
    _apply_optional_provider_attr(
        model_provider, "account_for_embedding_in_pipeline_split", mb_args.account_for_embedding_in_pipeline_split
    )
    _apply_optional_provider_attr(
        model_provider, "account_for_loss_in_pipeline_split", mb_args.account_for_loss_in_pipeline_split
    )
    _apply_optional_provider_attr(model_provider, "bias_activation_fusion", mb_args.bias_activation_fusion)
    _apply_optional_provider_attr(model_provider, "apply_rope_fusion", mb_args.apply_rope_fusion)
    _apply_optional_provider_attr(model_provider, "masked_softmax_fusion", mb_args.masked_softmax_fusion)
    _apply_optional_provider_attr(model_provider, "cross_entropy_loss_fusion", mb_args.cross_entropy_loss_fusion)
    _apply_optional_provider_attr(model_provider, "moe_grouped_gemm", mb_args.moe_grouped_gemm)
    _apply_optional_provider_attr(model_provider, "moe_token_dispatcher_type", mb_args.moe_token_dispatcher_type)
    _apply_optional_provider_attr(model_provider, "calculate_per_token_loss", mb_args.calculate_per_token_loss)


def _apply_context_parallel_finetuning_requirements(cfg, mb_args: "MegatronBridgeArguments") -> None:
    r"""Apply Megatron Bridge SFT requirements when context parallelism is enabled."""
    if mb_args.context_parallel_size <= 1:
        return
    cfg.model.calculate_per_token_loss = True
    cfg.ddp.average_in_collective = False


def _apply_extra_overrides(cfg, extra: dict) -> None:
    for key, value in extra.items():
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)


def _reset_megatron_bridge_global_state_after_checkpoint_conversion() -> None:
    r"""Clear Megatron globals left over from HF-to-Megatron conversion.

    save_megatron_model() can implicitly initialize the rerun state machine while
    writing checkpoints. provide_distributed_model() also initializes model parallel
    groups. Training later calls initialize_megatron(), which expects a fresh global
    state and fails with "Rerun state machine is already initialized" or keeps stale
    parallel groups that mismatch the configured tensor/pipeline/context parallel sizes.
    """
    from megatron.core import parallel_state
    from megatron.core.rerun_state_machine import destroy_rerun_state_machine

    destroy_rerun_state_machine()
    parallel_state.destroy_model_parallel()


def ensure_megatron_pretrained_checkpoint(
    model_args: "ModelArguments",
    mb_args: "MegatronBridgeArguments",
    output_dir: str,
) -> str:
    r"""Convert Hugging Face weights to Megatron format when needed."""
    from megatron.bridge import AutoBridge

    if mb_args.megatron_pretrained_checkpoint and os.path.isdir(mb_args.megatron_pretrained_checkpoint):
        return mb_args.megatron_pretrained_checkpoint

    ckpt_dir = os.path.join(output_dir, "megatron_pretrained")
    if os.path.isdir(ckpt_dir) and os.listdir(ckpt_dir):
        logger.info_rank0(f"Reusing existing Megatron checkpoint at {ckpt_dir}.")
        return ckpt_dir

    os.makedirs(ckpt_dir, exist_ok=True)
    logger.info_rank0(f"Converting Hugging Face weights to Megatron format at {ckpt_dir}...")
    bridge = AutoBridge.from_hf_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    provider = bridge.to_megatron_provider()
    _apply_model_parallelism(provider, mb_args)
    _apply_fusion_safety(provider)
    if hasattr(provider, "finalize"):
        provider.finalize()
    # TP/PP/CP weight scatter uses NCCL, which cannot operate on CPU tensors.
    use_cpu_initialization = (
        mb_args.tensor_model_parallel_size == 1
        and mb_args.pipeline_model_parallel_size == 1
        and mb_args.context_parallel_size == 1
    )
    if not use_cpu_initialization:
        logger.info_rank0(
            "Using GPU initialization for Megatron checkpoint conversion because model parallelism "
            "requires NCCL scatter/gather on CUDA tensors."
        )
    try:
        megatron_model = provider.provide_distributed_model(
            wrap_with_ddp=False,
            use_cpu_initialization=use_cpu_initialization,
        )
        hf_tokenizer_kwargs = {"trust_remote_code": True} if model_args.trust_remote_code else None
        bridge.save_megatron_model(
            megatron_model,
            ckpt_dir,
            hf_tokenizer_path=model_args.model_name_or_path,
            hf_tokenizer_kwargs=hf_tokenizer_kwargs,
            low_memory_save=True,
        )
    finally:
        _reset_megatron_bridge_global_state_after_checkpoint_conversion()
    return ckpt_dir


def build_pretrain_config(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
    mb_args: "MegatronBridgeArguments",
    dataset_path: str,
    num_train_samples: int,
):
    from megatron.bridge import AutoBridge

    micro_batch_size, global_batch_size, train_iters = _compute_train_schedule(
        training_args, mb_args, num_train_samples
    )
    cfg = _create_base_config(
        training_args=training_args,
        finetuning_args=finetuning_args,
        train_iters=train_iters,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        mb_args=mb_args,
        is_sft=False,
    )

    (
        _CheckpointConfig,
        _ConfigContainer,
        _DistributedInitConfig,
        _FinetuningDatasetConfig,
        GPTDatasetConfig,
        _LoggerConfig,
        _RNGConfig,
        _TokenizerConfig,
        _TrainingConfig,
        _ValidationConfig,
    ) = _import_training_config()
    bridge = AutoBridge.from_hf_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    cfg.model = bridge.to_megatron_provider(load_weights=False)
    _apply_model_parallelism(cfg.model, mb_args)
    _apply_fusion_safety(cfg.model)
    if hasattr(cfg.model, "seq_length"):
        cfg.model.seq_length = data_args.cutoff_len

    cfg.tokenizer.tokenizer_model = model_args.model_name_or_path
    cfg.dataset = _build_gpt_dataset_config(
        GPTDatasetConfig,
        dataset_path=dataset_path,
        seq_length=data_args.cutoff_len,
        seed=training_args.seed,
        num_workers=data_args.preprocessing_num_workers,
    )

    _apply_extra_overrides(cfg, mb_args.load_extra_config())
    return cfg


def build_sft_config(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
    mb_args: "MegatronBridgeArguments",
    dataset_root: str,
    pretrained_checkpoint: str,
    num_train_samples: int,
):
    from megatron.bridge import AutoBridge
    from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs

    micro_batch_size, global_batch_size, train_iters = _compute_train_schedule(
        training_args, mb_args, num_train_samples
    )
    cfg = _create_base_config(
        training_args=training_args,
        finetuning_args=finetuning_args,
        train_iters=train_iters,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        mb_args=mb_args,
        is_sft=True,
    )

    (
        _CheckpointConfig,
        _ConfigContainer,
        _DistributedInitConfig,
        FinetuningDatasetConfig,
        _GPTDatasetConfig,
        _LoggerConfig,
        _RNGConfig,
        _TokenizerConfig,
        _TrainingConfig,
        _ValidationConfig,
    ) = _import_training_config()
    bridge = AutoBridge.from_hf_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    cfg.model = bridge.to_megatron_provider(load_weights=False)
    _apply_model_parallelism(cfg.model, mb_args)
    _apply_fusion_safety(cfg.model)
    if hasattr(cfg.model, "seq_length"):
        cfg.model.seq_length = data_args.cutoff_len

    cfg.tokenizer.tokenizer_model = model_args.model_name_or_path

    from .dataset_export import get_sft_dataset_kwargs

    ensure_create_sft_dataset_applies_chat_template()
    dataset_kwargs = get_sft_dataset_kwargs(
        tokenizer_path=model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        template_name=data_args.template,
    )
    packed_sequence_specs = None
    if mb_args.use_packed_sequences:
        pad_seq_to_mult = mb_args.context_parallel_size * 2 if mb_args.context_parallel_size > 1 else 1
        packed_sequence_specs = PackedSequenceSpecs(
            packed_sequence_size=data_args.cutoff_len,
            pad_seq_to_mult=pad_seq_to_mult,
        )
        dataset_kwargs["pad_to_max_length"] = True

    cfg.dataset = _build_finetuning_dataset_config(
        FinetuningDatasetConfig,
        dataset_root=dataset_root,
        seq_length=data_args.cutoff_len,
        seed=training_args.seed,
        num_workers=data_args.preprocessing_num_workers,
        do_validation=data_args.val_size > 0 or data_args.eval_dataset is not None,
        dataset_kwargs=dataset_kwargs,
        packed_sequence_specs=packed_sequence_specs,
        disable_shuffling=getattr(finetuning_args, "disable_shuffling", False),
    )
    cfg.checkpoint.pretrained_checkpoint = pretrained_checkpoint

    _apply_context_parallel_finetuning_requirements(cfg, mb_args)
    _apply_extra_overrides(cfg, mb_args.load_extra_config())
    return cfg
