# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import transformers
from omegaconf import OmegaConf
from transformers import HfArgumentParser
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import ParallelMode
from transformers.utils import is_torch_bf16_gpu_available, is_torch_npu_available

from ..extras import logging
from ..extras.constants import CHECKPOINT_NAMES, EngineName
from ..extras.misc import check_dependencies, check_version, get_current_device, is_env_enabled
from ..extras.packages import is_mcore_adapter_available, is_megatron_bridge_available
from .data_args import DataArguments
from .evaluation_args import EvaluationArguments
from .finetuning_args import FinetuningArguments
from .generating_args import GeneratingArguments
from .megatron_bridge_args import MegatronBridgeArguments
from .model_args import ModelArguments
from .training_args import RayArguments, TrainingArguments


logger = logging.get_logger(__name__)

check_dependencies()


@dataclass
class _KTransformersRuntimeArguments:
    kt_config: dict[str, Any] | None = field(
        default=None,
        metadata={"help": "Advanced KTransformers settings used during inference or evaluation."},
    )


_TRAIN_ARGS = [
    ModelArguments,
    DataArguments,
    TrainingArguments,
    FinetuningArguments,
    GeneratingArguments,
]
_TRAIN_CLS = tuple[ModelArguments, DataArguments, TrainingArguments, FinetuningArguments, GeneratingArguments]
_INFER_ARGS = [ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments, _KTransformersRuntimeArguments]
_INFER_CLS = tuple[ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments]
_EVAL_ARGS = [ModelArguments, DataArguments, EvaluationArguments, FinetuningArguments, _KTransformersRuntimeArguments]
_EVAL_CLS = tuple[ModelArguments, DataArguments, EvaluationArguments, FinetuningArguments]

if is_mcore_adapter_available() and is_env_enabled("USE_MCA"):
    from mcore_adapter import TrainingArguments as McaTrainingArguments

    _TRAIN_MCA_ARGS = [
        ModelArguments,
        DataArguments,
        McaTrainingArguments,
        FinetuningArguments,
        GeneratingArguments,
    ]
    _TRAIN_MCA_CLS = tuple[
        ModelArguments,
        DataArguments,
        McaTrainingArguments,
        FinetuningArguments,
        GeneratingArguments,
    ]
else:
    _TRAIN_MCA_ARGS = []
    _TRAIN_MCA_CLS = tuple()

_TRAIN_MBRIDGE_ARGS = [
    ModelArguments,
    DataArguments,
    TrainingArguments,
    FinetuningArguments,
    MegatronBridgeArguments,
    GeneratingArguments,
]
_TRAIN_MBRIDGE_CLS = tuple[
    ModelArguments,
    DataArguments,
    TrainingArguments,
    FinetuningArguments,
    MegatronBridgeArguments,
    GeneratingArguments,
]


def read_args(args: dict[str, Any] | list[str] | None = None) -> dict[str, Any] | list[str]:
    r"""Get arguments from the command line or a config file."""
    if args is not None:
        return args

    if len(sys.argv) > 1 and (sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml")):
        override_config = OmegaConf.from_cli(sys.argv[2:])
        dict_config = OmegaConf.load(Path(sys.argv[1]).absolute())
        return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
    elif len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        override_config = OmegaConf.from_cli(sys.argv[2:])
        dict_config = OmegaConf.create(json.load(Path(sys.argv[1]).absolute()))
        return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
    else:
        return sys.argv[1:]


def _get_kt_runtime_capacity(
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> int:
    r"""Return the largest local token batch submitted to a KT expert."""
    tokens_per_sample = data_args.cutoff_len
    if finetuning_args.stage == "sft" and data_args.packing:
        tokens_per_sample += 1
    if finetuning_args.stage == "sft" and training_args.do_train:
        tokens_per_sample = ((tokens_per_sample + 7) // 8) * 8

    local_batch_sizes = [1]
    if training_args.do_train:
        local_batch_sizes.append(training_args.per_device_train_batch_size)
    if training_args.do_eval or training_args.do_predict:
        local_batch_sizes.append(training_args.per_device_eval_batch_size)
    return tokens_per_sample * max(local_batch_sizes)


def _parse_args(
    parser: "HfArgumentParser", args: dict[str, Any] | list[str] | None = None, allow_extra_keys: bool = False
) -> tuple[Any]:
    args = read_args(args)
    if isinstance(args, dict):
        return parser.parse_dict(args, allow_extra_keys=allow_extra_keys)

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(args=args, return_remaining_strings=True)

    if unknown_args and not allow_extra_keys:
        print(parser.format_help())
        print(f"Got unknown args, potentially deprecated arguments: {unknown_args}")
        raise ValueError(f"Some specified arguments are not used by the HfArgumentParser: {unknown_args}")

    return tuple(parsed_args)


def _verify_trackio_args(training_args: "TrainingArguments") -> None:
    """Validates Trackio-specific arguments.

    Args:
        training_args: TrainingArguments instance (not a dictionary)
    """
    report_to = training_args.report_to
    if not report_to:
        return

    if isinstance(report_to, str):
        report_to = [report_to]

    if "trackio" not in report_to:
        return

    # --- Enforce project (required by Trackio) ---
    if not training_args.project:
        raise ValueError("`--project` must be specified when using Trackio.")

    # --- Validate trackio_space_id format ---
    space_id = training_args.trackio_space_id
    if space_id:
        if space_id != "trackio" and "/" not in space_id:
            logger.warning(
                f"trackio_space_id '{space_id}' should typically be in format "
                "'org/space' for Hugging Face Spaces deployment."
            )

    # --- Inform about default project usage ---
    if training_args.project == "huggingface":
        logger.info(
            "Using default project name 'huggingface'. "
            "Consider setting a custom project name with --project "
            "for better organization."
        )

    # --- Validate hub repo privacy flag ---
    if training_args.hub_private_repo:
        logger.info("Repository will be created as private on Hugging Face Hub.")

    # --- Recommend run_name for experiment clarity ---
    if not training_args.run_name:
        logger.warning("Consider setting --run_name for better experiment tracking clarity.")


def _set_transformers_logging() -> None:
    if os.getenv("LLAMAFACTORY_VERBOSITY", "INFO") in ["DEBUG", "INFO"]:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()


def _set_env_vars() -> None:
    if is_torch_npu_available():
        # avoid JIT compile on NPU devices, see https://zhuanlan.zhihu.com/p/660875458
        torch.npu.set_compile_mode(jit_compile=is_env_enabled("NPU_JIT_COMPILE"))
        # avoid use fork method on NPU devices, see https://github.com/hiyouga/LLaMA-Factory/issues/7447
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _verify_model_args(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    if model_args.adapter_name_or_path is not None and finetuning_args.finetuning_type != "lora":
        raise ValueError("Adapter is only valid for the LoRA method.")

    if model_args.quantization_bit is not None:
        if finetuning_args.finetuning_type not in ["lora", "oft"]:
            raise ValueError("Quantization is only compatible with the LoRA or OFT method.")

        if finetuning_args.pissa_init:
            raise ValueError("Please use scripts/pissa_init.py to initialize PiSSA for a quantized model.")

        if model_args.resize_vocab:
            raise ValueError("Cannot resize embedding layers of a quantized model.")

        if model_args.adapter_name_or_path is not None and finetuning_args.create_new_adapter:
            raise ValueError("Cannot create new adapter upon a quantized model.")

        if model_args.adapter_name_or_path is not None and len(model_args.adapter_name_or_path) != 1:
            raise ValueError("Quantized model only accepts a single adapter. Merge them first.")


def _check_extra_dependencies(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    training_args: Optional["TrainingArguments"] = None,
) -> None:
    if model_args.use_kt:
        check_version("kt-kernel", mandatory=True)
        check_version("transformers-kt", mandatory=True)
        check_version("accelerate-kt", mandatory=True)

    if model_args.use_unsloth:
        check_version("unsloth", mandatory=True)

    if model_args.enable_liger_kernel:
        check_version("liger-kernel", mandatory=True)

    if model_args.mixture_of_depths is not None:
        check_version("mixture-of-depth>=1.1.6", mandatory=True)

    if model_args.infer_backend == EngineName.VLLM:
        check_version("vllm>=0.4.3,<=0.11.0")
        check_version("vllm", mandatory=True)
    elif model_args.infer_backend == EngineName.SGLANG:
        check_version("sglang>=0.4.5")
        check_version("sglang", mandatory=True)

    if finetuning_args.use_galore:
        check_version("galore_torch", mandatory=True)

    if finetuning_args.use_apollo:
        check_version("apollo_torch", mandatory=True)

    if finetuning_args.use_badam:
        check_version("badam>=1.2.1", mandatory=True)

    if finetuning_args.use_adam_mini:
        check_version("adam-mini", mandatory=True)

    if finetuning_args.use_swanlab:
        check_version("swanlab", mandatory=True)

    if finetuning_args.plot_loss:
        check_version("matplotlib", mandatory=True)

    if finetuning_args.use_megatron_bridge:
        check_version("megatron-bridge", mandatory=True)

    if training_args is not None:
        if training_args.deepspeed:
            check_version("deepspeed", mandatory=True)

        if training_args.predict_with_generate:
            check_version("jieba", mandatory=True)
            check_version("nltk", mandatory=True)
            check_version("rouge_chinese", mandatory=True)


def _parse_train_args(args: dict[str, Any] | list[str] | None = None) -> _TRAIN_CLS:
    parser = HfArgumentParser(_TRAIN_ARGS)
    allow_extra_keys = is_env_enabled("ALLOW_EXTRA_ARGS")
    return _parse_args(parser, args, allow_extra_keys=allow_extra_keys)


def _parse_train_mca_args(args: dict[str, Any] | list[str] | None = None) -> _TRAIN_MCA_CLS:
    parser = HfArgumentParser(_TRAIN_MCA_ARGS)
    allow_extra_keys = is_env_enabled("ALLOW_EXTRA_ARGS")
    model_args, data_args, training_args, finetuning_args, generating_args = _parse_args(
        parser, args, allow_extra_keys=allow_extra_keys
    )

    _configure_mca_training_args(training_args, data_args, finetuning_args)

    return model_args, data_args, training_args, finetuning_args, generating_args


def _configure_mca_training_args(training_args, data_args, finetuning_args) -> None:
    """Patch training args to avoid args checking errors and sync MCA settings."""
    training_args.predict_with_generate = False
    training_args.generation_max_length = data_args.cutoff_len
    training_args.generation_num_beams = 1
    training_args.use_mca = True
    finetuning_args.use_mca = True


def _validate_megatron_bridge_parallel_args(mb_args: MegatronBridgeArguments, world_size: int) -> None:
    parallel_size = (
        mb_args.tensor_model_parallel_size
        * mb_args.pipeline_model_parallel_size
        * mb_args.context_parallel_size
        * mb_args.expert_model_parallel_size
    )
    if parallel_size > world_size:
        raise ValueError(f"Total Megatron Bridge parallel size ({parallel_size}) exceeds `world_size` ({world_size}).")
    if world_size % parallel_size != 0:
        raise ValueError(
            f"Total Megatron Bridge parallel size ({parallel_size}) must divide `world_size` ({world_size})."
        )


def _parse_train_mbridge_args(args: dict[str, Any] | list[str] | None = None) -> _TRAIN_MBRIDGE_CLS:
    parser = HfArgumentParser(_TRAIN_MBRIDGE_ARGS)
    allow_extra_keys = is_env_enabled("ALLOW_EXTRA_ARGS")
    model_args, data_args, training_args, finetuning_args, mb_args, generating_args = _parse_args(
        parser, args, allow_extra_keys=allow_extra_keys
    )
    _configure_mbridge_training_args(training_args, data_args, finetuning_args)
    return model_args, data_args, training_args, finetuning_args, mb_args, generating_args


def _configure_mbridge_training_args(training_args, data_args, finetuning_args) -> None:
    """Patch training args to avoid args checking errors and sync Megatron Bridge settings."""
    training_args.predict_with_generate = False
    training_args.generation_max_length = data_args.cutoff_len
    training_args.generation_num_beams = 1
    finetuning_args.use_megatron_bridge = True


def _parse_infer_args(args: dict[str, Any] | list[str] | None = None) -> _INFER_CLS:
    parser = HfArgumentParser(_INFER_ARGS)
    allow_extra_keys = is_env_enabled("ALLOW_EXTRA_ARGS")
    model_args, data_args, finetuning_args, generating_args, kt_args = _parse_args(
        parser, args, allow_extra_keys=allow_extra_keys
    )
    model_args._kt_inference_config = kt_args.kt_config
    return model_args, data_args, finetuning_args, generating_args


def _parse_eval_args(args: dict[str, Any] | list[str] | None = None) -> _EVAL_CLS:
    parser = HfArgumentParser(_EVAL_ARGS)
    allow_extra_keys = is_env_enabled("ALLOW_EXTRA_ARGS")
    model_args, data_args, eval_args, finetuning_args, kt_args = _parse_args(
        parser, args, allow_extra_keys=allow_extra_keys
    )
    model_args._kt_inference_config = kt_args.kt_config
    return model_args, data_args, eval_args, finetuning_args


def get_ray_args(args: dict[str, Any] | list[str] | None = None) -> RayArguments:
    parser = HfArgumentParser(RayArguments)
    (ray_args,) = _parse_args(parser, args, allow_extra_keys=True)
    return ray_args


def get_train_args(args: dict[str, Any] | list[str] | None = None) -> _TRAIN_CLS:
    mb_args = None
    if is_env_enabled("USE_MCA"):
        model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_mca_args(args)
    elif is_env_enabled("USE_MEGATRON_BRIDGE"):
        if not is_megatron_bridge_available():
            raise ImportError(
                "megatron-bridge is required when USE_MEGATRON_BRIDGE=1. "
                "Please install `megatron-bridge` and its dependencies."
            )
        model_args, data_args, training_args, finetuning_args, mb_args, generating_args = _parse_train_mbridge_args(
            args
        )
    else:
        model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_args(args)
        finetuning_args.use_mca = False
        finetuning_args.use_megatron_bridge = False

    # Setup logging
    if training_args.should_log:
        _set_transformers_logging()

    # Check arguments
    if finetuning_args.stage != "sft":
        if training_args.predict_with_generate:
            raise ValueError("`predict_with_generate` cannot be set as True except SFT.")

        if data_args.neat_packing:
            raise ValueError("`neat_packing` cannot be set as True except SFT.")

        if data_args.train_on_prompt or data_args.mask_history:
            raise ValueError("`train_on_prompt` or `mask_history` cannot be set as True except SFT.")

    if finetuning_args.stage == "sft" and training_args.do_predict and not training_args.predict_with_generate:
        raise ValueError("Please enable `predict_with_generate` to save model predictions.")

    if finetuning_args.use_megatron_bridge:
        if finetuning_args.use_mca or finetuning_args.use_hyper_parallel:
            raise ValueError("Megatron Bridge cannot be used together with MCA or HyperParallel.")
        if finetuning_args.stage not in ["pt", "sft"]:
            raise ValueError("Megatron Bridge only supports the `pt` and `sft` stages.")
        if finetuning_args.finetuning_type not in ["full", "lora"]:
            raise ValueError("Megatron Bridge only supports `full` and `lora` finetuning.")
        if model_args.quantization_bit is not None:
            raise ValueError("Quantized models are not supported with Megatron Bridge.")
        if training_args.deepspeed is not None:
            raise ValueError("Megatron Bridge is incompatible with DeepSpeed.")
        if mb_args is None:
            raise ValueError("Megatron Bridge arguments are missing. Please set USE_MEGATRON_BRIDGE=1.")
        _validate_megatron_bridge_parallel_args(mb_args, training_args.world_size)
        finetuning_args.megatron_bridge_args = mb_args

    if finetuning_args.stage in ["rm", "ppo"] and training_args.load_best_model_at_end:
        raise ValueError("RM and PPO stages do not support `load_best_model_at_end`.")

    if finetuning_args.stage == "ppo":
        if not training_args.do_train:
            raise ValueError("PPO training does not support evaluation, use the SFT stage to evaluate models.")

        if model_args.shift_attn:
            raise ValueError("PPO training is incompatible with S^2-Attn.")

        if finetuning_args.reward_model_type == "lora" and model_args.use_kt:
            raise ValueError("KTransformers does not support lora reward model.")

        if finetuning_args.reward_model_type == "lora" and model_args.use_unsloth:
            raise ValueError("Unsloth does not support lora reward model.")

        if training_args.report_to and any(
            logger not in ("wandb", "tensorboard", "trackio", "none") for logger in training_args.report_to
        ):
            raise ValueError("PPO only accepts wandb, tensorboard, or trackio logger.")

    if not model_args.use_kt and training_args.parallel_mode == ParallelMode.NOT_DISTRIBUTED:
        raise ValueError("Please launch distributed training with `llamafactory-cli` or `torchrun`.")

    if training_args.deepspeed and training_args.parallel_mode != ParallelMode.DISTRIBUTED:
        raise ValueError("Please use `FORCE_TORCHRUN=1` to launch DeepSpeed training.")

    if training_args.max_steps == -1 and data_args.streaming:
        raise ValueError("Please specify `max_steps` in streaming mode.")

    if training_args.do_train and data_args.dataset is None:
        raise ValueError("Please specify dataset for training.")

    if (training_args.do_eval or training_args.do_predict or training_args.predict_with_generate) and (
        data_args.eval_dataset is None and data_args.val_size < 1e-6
    ):
        raise ValueError("Please make sure eval_dataset be provided or val_size >1e-6")

    if training_args.predict_with_generate:
        if is_deepspeed_zero3_enabled():
            raise ValueError("`predict_with_generate` is incompatible with DeepSpeed ZeRO-3.")

        if finetuning_args.compute_accuracy:
            raise ValueError("Cannot use `predict_with_generate` and `compute_accuracy` together.")

    if training_args.do_train and model_args.quantization_device_map == "auto":
        raise ValueError("Cannot use device map for quantized models in training.")

    if finetuning_args.pissa_init and is_deepspeed_zero3_enabled():
        raise ValueError("Please use scripts/pissa_init.py to initialize PiSSA in DeepSpeed ZeRO-3.")

    if finetuning_args.pure_bf16:
        if not (is_torch_bf16_gpu_available() or (is_torch_npu_available() and torch.npu.is_bf16_supported())):
            raise ValueError("This device does not support `pure_bf16`.")

        if is_deepspeed_zero3_enabled():
            raise ValueError("`pure_bf16` is incompatible with DeepSpeed ZeRO-3.")

    if training_args.parallel_mode == ParallelMode.DISTRIBUTED:
        if finetuning_args.use_galore and finetuning_args.galore_layerwise:
            raise ValueError("Distributed training does not support layer-wise GaLore.")

        if finetuning_args.use_apollo and finetuning_args.apollo_layerwise:
            raise ValueError("Distributed training does not support layer-wise APOLLO.")

        if finetuning_args.use_badam:
            if finetuning_args.badam_mode == "ratio":
                raise ValueError("Radio-based BAdam does not yet support distributed training, use layer-wise BAdam.")
            elif not is_deepspeed_zero3_enabled():
                raise ValueError("Layer-wise BAdam only supports DeepSpeed ZeRO-3 training.")

    if training_args.deepspeed is not None and (finetuning_args.use_galore or finetuning_args.use_apollo):
        raise ValueError("GaLore and APOLLO are incompatible with DeepSpeed yet.")

    if (
        not finetuning_args.use_mca
        and not finetuning_args.use_megatron_bridge
        and training_args.fp8
        and model_args.quantization_bit is not None
    ):
        raise ValueError("FP8 training is not compatible with quantization. Please disable one of them.")

    if model_args.infer_backend != EngineName.HF:
        raise ValueError("vLLM/SGLang backend is only available for API, CLI and Web.")

    if model_args.use_unsloth and is_deepspeed_zero3_enabled():
        raise ValueError("Unsloth is incompatible with DeepSpeed ZeRO-3.")

    if model_args.use_kt and is_deepspeed_zero3_enabled():
        raise ValueError("KTransformers is incompatible with DeepSpeed ZeRO-3.")

    _set_env_vars()
    _verify_model_args(model_args, data_args, finetuning_args)
    _check_extra_dependencies(model_args, finetuning_args, training_args)
    _verify_trackio_args(training_args)

    if (
        not finetuning_args.use_mca
        and not finetuning_args.use_megatron_bridge
        and training_args.fp8_enable_fsdp_float8_all_gather
        and not training_args.fp8
    ):
        logger.warning_rank0("fp8_enable_fsdp_float8_all_gather requires fp8=True. Setting fp8=True.")
        model_args.fp8 = True

    if (
        training_args.do_train
        and finetuning_args.finetuning_type == "lora"
        and model_args.quantization_bit is None
        and model_args.resize_vocab
        and finetuning_args.additional_target is None
    ):
        logger.warning_rank0(
            "Remember to add embedding layers to `additional_target` to make the added tokens trainable."
        )

    if training_args.do_train and model_args.quantization_bit is not None and (not model_args.upcast_layernorm):
        logger.warning_rank0("We recommend enable `upcast_layernorm` in quantized training.")

    if training_args.do_train and (not training_args.fp16) and (not training_args.bf16):
        logger.warning_rank0("We recommend enable mixed precision training.")

    if (
        training_args.do_train
        and (finetuning_args.use_galore or finetuning_args.use_apollo)
        and not finetuning_args.pure_bf16
    ):
        logger.warning_rank0(
            "Using GaLore or APOLLO with mixed precision training may significantly increases GPU memory usage."
        )

    if (not training_args.do_train) and model_args.quantization_bit is not None:
        logger.warning_rank0("Evaluating model in 4/8-bit mode may cause lower scores.")

    if (not training_args.do_train) and finetuning_args.stage == "dpo" and finetuning_args.ref_model is None:
        logger.warning_rank0("Specify `ref_model` for computing rewards at evaluation.")

    # Post-process training arguments
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset

    if finetuning_args.finetuning_type == "lora":
        # https://github.com/huggingface/transformers/blob/v4.50.0/src/transformers/trainer.py#L782
        training_args.label_names = training_args.label_names or ["labels"]

    if "swanlab" in training_args.report_to and finetuning_args.use_swanlab:
        training_args.report_to.remove("swanlab")

    if (
        training_args.parallel_mode == ParallelMode.DISTRIBUTED
        and training_args.ddp_find_unused_parameters is None
        and finetuning_args.finetuning_type == "lora"
    ):
        logger.info_rank0("Set `ddp_find_unused_parameters` to False in DDP training since LoRA is enabled.")
        training_args.ddp_find_unused_parameters = False

    if finetuning_args.stage in ["rm", "ppo"] and finetuning_args.finetuning_type in ["full", "freeze"]:
        can_resume_from_checkpoint = False
        if training_args.resume_from_checkpoint is not None:
            logger.warning_rank0("Cannot resume from checkpoint in current stage.")
            training_args.resume_from_checkpoint = None
    else:
        can_resume_from_checkpoint = True

    if (
        training_args.resume_from_checkpoint is None
        and training_args.do_train
        and os.path.isdir(training_args.output_dir)
        and not getattr(training_args, "overwrite_output_dir", False)  # for mca training args and transformers >= 5.0
        and can_resume_from_checkpoint
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and any(
            os.path.isfile(os.path.join(training_args.output_dir, name)) for name in CHECKPOINT_NAMES
        ):
            raise ValueError("Output directory already exists and is not empty. Please set `overwrite_output_dir`.")

        if last_checkpoint is not None:
            training_args.resume_from_checkpoint = last_checkpoint
            logger.info_rank0(f"Resuming training from {training_args.resume_from_checkpoint}.")
            logger.info_rank0("Change `output_dir` or use `overwrite_output_dir` to avoid.")

    if (
        finetuning_args.stage in ["rm", "ppo"]
        and finetuning_args.finetuning_type == "lora"
        and training_args.resume_from_checkpoint is not None
    ):
        logger.warning_rank0(
            f"Add {training_args.resume_from_checkpoint} to `adapter_name_or_path` to resume training from checkpoint."
        )

    # Post-process model arguments
    if training_args.bf16 or finetuning_args.pure_bf16:
        model_args.compute_dtype = torch.bfloat16
    elif training_args.fp16:
        model_args.compute_dtype = torch.float16

    data_args.packing = data_args.packing if data_args.packing is not None else finetuning_args.stage == "pt"
    model_args.device_map = {"": get_current_device()}
    model_args.model_max_length = data_args.cutoff_len
    model_args.block_diag_attn = data_args.neat_packing

    # Log on each process the small summary
    logger.info(
        f"Process rank: {training_args.process_index}, "
        f"world size: {training_args.world_size}, device: {training_args.device}, "
        f"distributed training: {training_args.parallel_mode == ParallelMode.DISTRIBUTED}, "
        f"compute dtype: {str(model_args.compute_dtype)}"
    )
    transformers.set_seed(training_args.seed)

    if model_args.use_kt:
        model_args.apply_kt_config(
            finetuning_args,
            training_args,
            _get_kt_runtime_capacity(data_args, training_args, finetuning_args),
        )

    return model_args, data_args, training_args, finetuning_args, generating_args


def get_infer_args(args: dict[str, Any] | list[str] | None = None) -> _INFER_CLS:
    model_args, data_args, finetuning_args, generating_args = _parse_infer_args(args)

    # Setup logging
    _set_transformers_logging()

    # Check arguments
    if model_args.infer_backend == "vllm":
        if finetuning_args.stage != "sft":
            raise ValueError("vLLM engine only supports auto-regressive models.")

        if model_args.quantization_bit is not None:
            raise ValueError("vLLM engine does not support bnb quantization (GPTQ and AWQ are supported).")

        if model_args.rope_scaling is not None:
            raise ValueError("vLLM engine does not support RoPE scaling.")

        if model_args.adapter_name_or_path is not None and len(model_args.adapter_name_or_path) != 1:
            raise ValueError("vLLM only accepts a single adapter. Merge them first.")

    _set_env_vars()
    _verify_model_args(model_args, data_args, finetuning_args)
    _check_extra_dependencies(model_args, finetuning_args)

    # Post-process model arguments
    if model_args.export_dir is not None and model_args.export_device == "cpu":
        model_args.device_map = {"": torch.device("cpu")}
        if data_args.cutoff_len != DataArguments().cutoff_len:  # override cutoff_len if it is not default
            model_args.model_max_length = data_args.cutoff_len
    else:
        model_args.device_map = "auto"

    model_args.configure_kt_loading(finetuning_args, data_args.cutoff_len)

    return model_args, data_args, finetuning_args, generating_args


def get_eval_args(args: dict[str, Any] | list[str] | None = None) -> _EVAL_CLS:
    model_args, data_args, eval_args, finetuning_args = _parse_eval_args(args)

    # Setup logging
    _set_transformers_logging()

    # Check arguments
    if model_args.infer_backend != EngineName.HF:
        raise ValueError("vLLM/SGLang backend is only available for API, CLI and Web.")

    _set_env_vars()
    _verify_model_args(model_args, data_args, finetuning_args)
    _check_extra_dependencies(model_args, finetuning_args)

    model_args.device_map = "auto"
    model_args.configure_kt_loading(finetuning_args, data_args.cutoff_len)

    transformers.set_seed(eval_args.seed)

    return model_args, data_args, eval_args, finetuning_args
