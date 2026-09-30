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

import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

from transformers import AutoConfig as HfAutoConfig

from ...data.data_utils import split_dataset
from ...data.loader import _get_merged_dataset
from ...extras.constants import MEGATRON_BRIDGE_SUPPORTED_MODELS
from ...extras.logging import get_logger
from ...extras.packages import is_megatron_bridge_available
from .config_builder import (
    _apply_fusion_safety,
    build_pretrain_config,
    build_sft_config,
    ensure_megatron_pretrained_checkpoint,
)
from .dataset_export import export_dataset_for_megatron_bridge


if TYPE_CHECKING:
    from transformers import TrainerCallback

    from ...hparams import (
        DataArguments,
        FinetuningArguments,
        MegatronBridgeArguments,
        ModelArguments,
        TrainingArguments,
    )


logger = get_logger(__name__)


def _check_model_support(model_args: "ModelArguments") -> None:
    r"""Ensure the HF ``model_type`` is covered by the Megatron Bridge PT/SFT path."""
    config = HfAutoConfig.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    model_type = getattr(config, "model_type", None)
    if model_type not in MEGATRON_BRIDGE_SUPPORTED_MODELS:
        raise ValueError(
            f"Model type `{model_type}` is not supported by the Megatron Bridge PT/SFT path. "
            f"Supported model types: {sorted(MEGATRON_BRIDGE_SUPPORTED_MODELS)}. "
            "Multimodal / audio / omni models are not enabled in v0."
        )


def _run_on_main_process(training_args: "TrainingArguments", work: Callable[[], None], sync_dir: str) -> None:
    r"""Run ``work`` only on global rank 0, then synchronize other ranks.

    Prefer ``torch.distributed.barrier`` when the process group is already initialized;
    otherwise fall back to a file flag under ``sync_dir`` so non-main ranks wait for
    shared filesystem writes (e.g. dataset export) to finish.
    """
    done_file = os.path.join(sync_dir, ".main_process_done")
    is_main = getattr(training_args, "process_index", 0) == 0
    wait_start = time.time()

    import torch.distributed as dist

    dist_ready = dist.is_available() and dist.is_initialized()
    if is_main:
        os.makedirs(sync_dir, exist_ok=True)
        if os.path.isfile(done_file):
            os.remove(done_file)
        work()
        with open(done_file, "w", encoding="utf-8") as f:
            f.write("done")

    if dist_ready:
        dist.barrier()
    elif not is_main:
        while True:
            if os.path.isfile(done_file) and os.path.getmtime(done_file) >= wait_start - 1.0:
                break
            time.sleep(0.5)


def _check_backend_available() -> None:
    if not is_megatron_bridge_available():
        raise ImportError(
            "megatron-bridge is not installed. "
            "Please install it with `pip install --no-build-isolation megatron-bridge` "
            "or use the NeMo Framework container."
        )
    _patch_dataset_helper_compilation()
    _patch_dist_checkpoint_preload()


def _patch_dist_checkpoint_preload() -> None:
    r"""Use blocking GPU->CPU copies when saving distributed checkpoints.

    Megatron's default ``non_blocking=True`` preload can raise ``cudaErrorInvalidValue``
    on some GPUs (e.g. V100) when saving distributed optimizer shards, because pinned
    host memory allocation or async D2H transfer may fail under memory pressure.
    """
    from megatron.core.dist_checkpointing.strategies import filesystem_async

    if getattr(filesystem_async.FileSystemWriterAsync.preload_tensors, "_llamafactory_patched", False):
        return

    original_preload = filesystem_async.FileSystemWriterAsync.preload_tensors

    @staticmethod
    def preload_tensors(write_buckets, non_blocking=True):
        return original_preload(write_buckets, non_blocking=False)

    preload_tensors._llamafactory_patched = True
    filesystem_async.FileSystemWriterAsync.preload_tensors = preload_tensors
    logger.info_rank0("Patched Megatron dist checkpoint preload to use blocking GPU->CPU copies.")


def _patch_dataset_helper_compilation() -> None:
    r"""Skip make-based helper compilation when the pybind extension is prebuilt.

    Pip-installed megatron-core already ships helpers_cpp, but compile_helpers()
    still invokes make and fails when no Makefile is present.
    """
    from megatron.core.datasets import utils as dataset_utils

    if getattr(dataset_utils.compile_helpers, "_llamafactory_patched", False):
        return

    try:
        import megatron.core.datasets.helpers_cpp  # noqa: F401
    except ImportError:
        return

    def compile_helpers():
        import megatron.core.datasets.helpers_cpp  # noqa: F401

    compile_helpers._llamafactory_patched = True
    dataset_utils.compile_helpers = compile_helpers
    logger.info_rank0("Using prebuilt megatron.core.datasets.helpers_cpp; skipping make compilation.")


def _load_aligned_datasets(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    stage: str,
):
    dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)
    eval_dataset = _get_merged_dataset(
        data_args.eval_dataset,
        model_args,
        data_args,
        training_args,
        stage,
        return_dict=data_args.eval_on_each_dataset,
    )
    train_dict, eval_dict = split_dataset(dataset, eval_dataset, data_args, seed=training_args.seed)
    return train_dict.get("train"), eval_dict


def _latest_iter_checkpoint_dir(output_dir: str) -> Optional[str]:
    r"""Return the latest ``iter_*`` directory under ``output_dir``, if any.

    ``export_adapter_ckpt`` needs the iteration directory that holds the
    distributed checkpoint payload (``.distcp`` / ``run_config.yaml``), not the
    parent run directory.
    """
    if not os.path.isdir(output_dir):
        return None

    # Already pointing at an iteration directory.
    if os.path.isfile(os.path.join(output_dir, "run_config.yaml")) or os.path.exists(
        os.path.join(output_dir, ".metadata")
    ):
        return output_dir

    iter_dirs = [
        name
        for name in os.listdir(output_dir)
        if name.startswith("iter_") and os.path.isdir(os.path.join(output_dir, name))
    ]
    if not iter_dirs:
        return None

    def _iter_number(name: str) -> int:
        try:
            return int(name.replace("iter_", ""))
        except ValueError:
            return -1

    latest = max(iter_dirs, key=_iter_number)
    return os.path.join(output_dir, latest)


def _checkpoint_uses_peft(checkpoint_dir: str) -> bool:
    r"""Whether the Megatron checkpoint was saved with a PEFT (e.g. LoRA) config."""
    cfg_path = os.path.join(checkpoint_dir, "run_config.yaml")
    if not os.path.isfile(cfg_path):
        return False

    try:
        import yaml

        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return isinstance(cfg, dict) and bool(cfg.get("peft"))
    except Exception:
        return False


def _with_fusion_safe_provider(bridge):
    r"""Wrap ``to_megatron_provider`` so export paths disable missing APEX fusion.

    Training already applies ``_apply_fusion_safety``, but AutoBridge export helpers
    (e.g. ``export_adapter_ckpt``) rebuild a provider with the default
    ``gradient_accumulation_fusion=True`` whenever TransformerEngine is importable,
    even if ``fused_weight_gradient_mlp_cuda`` is absent.
    """
    original = bridge.to_megatron_provider

    def _to_megatron_provider(*args, **kwargs):
        provider = original(*args, **kwargs)
        _apply_fusion_safety(provider)
        return provider

    bridge.to_megatron_provider = _to_megatron_provider  # type: ignore[method-assign]
    return bridge


def _maybe_export_hf_checkpoint(
    model_args: "ModelArguments",
    mb_args: "MegatronBridgeArguments",
    output_dir: str,
) -> None:
    if not mb_args.export_hf_on_finish or not training_args_should_save(output_dir):
        return

    import torch.distributed as dist
    from megatron.bridge import AutoBridge

    checkpoint_dir = _latest_iter_checkpoint_dir(output_dir)
    if checkpoint_dir is None:
        logger.warning_rank0(f"No Megatron iteration checkpoint found under {output_dir}; skip HF export.")
        return

    export_dir = os.path.join(output_dir, "hf_export")
    bridge = _with_fusion_safe_provider(
        AutoBridge.from_hf_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
        )
    )

    # LoRA / PEFT checkpoints only store adapter weights. Loading them as a full
    # model raises KeyError for base tensors such as linear_proj.weight.
    if _checkpoint_uses_peft(checkpoint_dir):
        logger.info_rank0(f"Exporting LoRA adapter to Hugging Face PEFT format at {export_dir}...")
        bridge.export_adapter_ckpt(peft_checkpoint=checkpoint_dir, output_path=export_dir)
        return

    logger.info_rank0(f"Exporting Megatron checkpoint to Hugging Face format at {export_dir}...")
    if dist.is_initialized():
        # export_ckpt() always creates a fresh single-process gloo group, which fails
        # when torchrun has already initialized NCCL for training.
        megatron_model = bridge.load_megatron_model(output_dir)
        bridge.save_hf_pretrained(megatron_model, export_dir)
    else:
        bridge.export_ckpt(megatron_path=output_dir, hf_path=export_dir)


def training_args_should_save(output_dir: str) -> bool:
    return os.path.isdir(output_dir) and bool(os.listdir(output_dir))


def run_pt(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
    mb_args: "MegatronBridgeArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    if callbacks:
        logger.warning_rank0("Megatron Bridge does not support Trainer callbacks yet; ignoring provided callbacks.")
    _check_backend_available()
    _check_model_support(model_args)
    from megatron.bridge.training.gpt_step import forward_step
    from megatron.bridge.training.pretrain import pretrain

    train_dataset, eval_dict = _load_aligned_datasets(model_args, data_args, training_args, "pt")
    dataset_dir = os.path.join(training_args.output_dir, "mb_dataset")

    def _export_pt_dataset() -> None:
        export_dataset_for_megatron_bridge(
            train_dataset=train_dataset,
            output_dir=dataset_dir,
            eval_dataset=eval_dict.get("validation") if eval_dict else None,
            val_size=data_args.val_size,
            seed=training_args.seed,
            stage="pt",
        )

    _run_on_main_process(training_args, _export_pt_dataset, dataset_dir)

    cfg = build_pretrain_config(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        mb_args=mb_args,
        dataset_path=os.path.join(dataset_dir, "training.jsonl"),
        num_train_samples=len(train_dataset),
    )
    pretrain(cfg, forward_step)
    _maybe_export_hf_checkpoint(model_args, mb_args, training_args.output_dir)


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
    mb_args: "MegatronBridgeArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    if callbacks:
        logger.warning_rank0("Megatron Bridge does not support Trainer callbacks yet; ignoring provided callbacks.")
    _check_backend_available()
    _check_model_support(model_args)
    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.gpt_step import forward_step

    train_dataset, eval_dict = _load_aligned_datasets(model_args, data_args, training_args, "sft")
    dataset_dir = os.path.join(training_args.output_dir, "mb_dataset")

    def _export_sft_dataset() -> None:
        export_dataset_for_megatron_bridge(
            train_dataset=train_dataset,
            output_dir=dataset_dir,
            eval_dataset=eval_dict or None,
            val_size=data_args.val_size if not eval_dict else 0.0,
            seed=training_args.seed,
            stage="sft",
            model_name_or_path=model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            template_name=data_args.template,
        )

    _run_on_main_process(training_args, _export_sft_dataset, dataset_dir)

    pretrained_checkpoint = ensure_megatron_pretrained_checkpoint(
        model_args=model_args,
        mb_args=mb_args,
        output_dir=training_args.output_dir,
    )
    cfg = build_sft_config(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        mb_args=mb_args,
        dataset_root=dataset_dir,
        pretrained_checkpoint=pretrained_checkpoint,
        num_train_samples=len(train_dataset),
    )
    finetune(cfg, forward_step_func=forward_step)
    _maybe_export_hf_checkpoint(model_args, mb_args, training_args.output_dir)
