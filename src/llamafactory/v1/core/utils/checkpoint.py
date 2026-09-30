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

"""Checkpoint utilities: low-level helpers and full training save/resume orchestration."""

import glob
import json
import os
import random
import shutil
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from ...accelerator.helper import DeviceType, get_current_accelerator
from ...accelerator.interface import DistributedInterface
from ...utils import logging


logger = logging.get_logger(__name__)

CHECKPOINT_COMPLETE_MARKER = "CHECKPOINT_COMPLETE"


def _parse_checkpoint_step(path: str) -> int:
    """Extract the step number from a checkpoint directory name, or -1 if invalid."""
    try:
        return int(os.path.basename(path).split("-")[-1])
    except ValueError:
        return -1


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest valid checkpoint directory in output_dir."""
    pattern = os.path.join(output_dir, "checkpoint-*")
    ckpt_dirs = [d for d in glob.glob(pattern) if _parse_checkpoint_step(d) >= 0]
    ckpt_dirs.sort(key=_parse_checkpoint_step)
    for d in reversed(ckpt_dirs):
        if os.path.exists(os.path.join(d, CHECKPOINT_COMPLETE_MARKER)):
            return d
    return None


def rotate_checkpoints(output_dir: str, limit: int) -> None:
    """Keep only the latest `limit` complete checkpoints, delete older ones and incomplete leftovers."""
    pattern = os.path.join(output_dir, "checkpoint-*")
    all_dirs = [d for d in glob.glob(pattern) if _parse_checkpoint_step(d) >= 0]
    all_dirs.sort(key=_parse_checkpoint_step)

    complete_dirs = []
    for d in all_dirs:
        if os.path.exists(os.path.join(d, CHECKPOINT_COMPLETE_MARKER)):
            complete_dirs.append(d)
        else:
            shutil.rmtree(d)
            logger.info_rank0(f"Cleaned up incomplete checkpoint: {d}")

    while len(complete_dirs) > limit:
        oldest = complete_dirs.pop(0)
        shutil.rmtree(oldest)
        logger.info_rank0(f"Deleted old checkpoint: {oldest}")


def save_metadata(ckpt_dir: str, **kwargs) -> None:
    """Save training metadata as JSON (rank 0 only)."""
    with open(os.path.join(ckpt_dir, "metadata.json"), "w") as f:
        json.dump(kwargs, f, indent=2)


def load_metadata(ckpt_dir: str) -> dict:
    """Load training metadata from a checkpoint directory."""
    with open(os.path.join(ckpt_dir, "metadata.json")) as f:
        return json.load(f)


def _get_accelerator_rng_state():
    """Get RNG state for the current accelerator, device-agnostic."""
    device_type = get_current_accelerator().type
    if device_type == DeviceType.CUDA:
        return torch.cuda.get_rng_state_all()
    elif device_type == DeviceType.NPU:
        return torch.npu.get_rng_state_all()
    elif device_type == DeviceType.XPU:
        return torch.xpu.get_rng_state_all()
    return None


def _set_accelerator_rng_state(state) -> None:
    """Set RNG state for the current accelerator, device-agnostic."""
    if state is None:
        return

    device_type = get_current_accelerator().type
    if device_type == DeviceType.CUDA:
        torch.cuda.set_rng_state_all(state)
    elif device_type == DeviceType.NPU:
        torch.npu.set_rng_state_all(state)
    elif device_type == DeviceType.XPU:
        torch.xpu.set_rng_state_all(state)


def save_rng_state(ckpt_dir: str, rank: int) -> None:
    """Save per-rank RNG states for reproducibility."""
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "accelerator": _get_accelerator_rng_state(),
    }
    rng_dir = os.path.join(ckpt_dir, "rng_state")
    os.makedirs(rng_dir, exist_ok=True)
    torch.save(rng_state, os.path.join(rng_dir, f"rank_{rank}.pt"))


def load_rng_state(ckpt_dir: str, rank: int) -> None:
    """Restore per-rank RNG states from a checkpoint."""
    path = os.path.join(ckpt_dir, "rng_state", f"rank_{rank}.pt")

    if not os.path.exists(path):
        logger.warning_rank0(f"RNG state file not found at {path}. Skipping RNG state restoration.")
        return

    rng_state = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.random.set_rng_state(rng_state["torch"])
    _set_accelerator_rng_state(rng_state.get("accelerator"))


def mark_checkpoint_complete(ckpt_dir: str) -> None:
    """Write a marker file indicating the checkpoint is fully saved."""
    open(os.path.join(ckpt_dir, CHECKPOINT_COMPLETE_MARKER), "w").close()


def resolve_resume_checkpoint_path(ckpt_path: str, output_dir: str) -> str | None:
    """Resolve 'auto' to the latest valid checkpoint, or return the path as-is."""
    if ckpt_path == "auto":
        resolved = find_latest_checkpoint(output_dir)
        if resolved is None:
            logger.warning_rank0(
                "resume_from_checkpoint='auto' but no valid checkpoint found in "
                f"'{output_dir}'. Training from scratch."
            )
        else:
            logger.info_rank0(f"Auto-detected latest checkpoint: {resolved}")
        return resolved
    return ckpt_path


def _save_standard_training_states(
    ckpt_dir: str,
    model: Any,
    optimizer: torch.optim.Optimizer,
    processor: Any,
    save_ckpt_as_hf: bool,
) -> None:
    """Save model and optimizer for DDP / single-GPU via save_pretrained."""
    rank = DistributedInterface().get_rank()
    if rank == 0:
        model_to_save = model.module if hasattr(model, "module") else model
        model_dir = os.path.join(ckpt_dir, "model")
        model_to_save.save_pretrained(model_dir, state_dict=model_to_save.state_dict(), max_shard_size="4GB")
        processor.save_pretrained(model_dir)

        os.makedirs(os.path.join(ckpt_dir, "optimizer"), exist_ok=True)
        torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer", "state_dict.pt"))

        if save_ckpt_as_hf:
            logger.info("Standard saving already uses HF format. No additional 'hf_model' directory created.")


def _load_standard_training_states(
    ckpt_dir: str,
    model: Any,
    optimizer: torch.optim.Optimizer,
    map_location: torch.device,
) -> None:
    """Load model and optimizer for DDP / single-GPU."""
    model_dir = os.path.join(ckpt_dir, "model")
    model_to_load = model.module if hasattr(model, "module") else model

    is_adapter_ckpt = os.path.exists(os.path.join(model_dir, "adapter_config.json"))

    if is_adapter_ckpt:
        from peft import set_peft_model_state_dict

        adapter_file = os.path.join(model_dir, "adapter_model.safetensors")
        if not os.path.exists(adapter_file):
            adapter_file = os.path.join(model_dir, "adapter_model.bin")
            adapter_state = torch.load(adapter_file, map_location="cpu", weights_only=True)
        else:
            adapter_state = load_file(adapter_file, device="cpu")
        set_peft_model_state_dict(model_to_load, adapter_state)
    else:
        state_dict = {}
        for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            state_dict.update(load_file(f, device="cpu"))
        if not state_dict:
            for f in sorted(glob.glob(os.path.join(model_dir, "*.bin"))):
                state_dict.update(torch.load(f, map_location="cpu", weights_only=True))
        if state_dict:
            incompatible_keys = model_to_load.load_state_dict(state_dict, strict=False)
            if incompatible_keys.missing_keys:
                raise RuntimeError(
                    f"Unexpected missing keys when loading checkpoint model weights: {incompatible_keys.missing_keys}."
                )
        else:
            logger.warning_rank0(f"No model weights found in {model_dir}, skipping model state restore.")

    optim_path = os.path.join(ckpt_dir, "optimizer", "state_dict.pt")
    if os.path.exists(optim_path):
        optimizer.load_state_dict(torch.load(optim_path, map_location=map_location, weights_only=True))


class TrainingCheckpointCoordinator:
    """Coordinates full checkpoint save/resume for a trainer instance."""

    def __init__(self, trainer: Any) -> None:
        self._t = trainer

    @property
    def _dist_name(self) -> str | None:
        return self._t.args.dist_config.name if self._t.args.dist_config is not None else None

    def save(self, epoch: int) -> None:
        """Save a full training checkpoint at the current global step."""
        ckpt_dir = os.path.join(self._t.args.output_dir, f"checkpoint-{self._t.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        rank = DistributedInterface().get_rank()

        if rank == 0:
            save_metadata(
                ckpt_dir,
                global_step=self._t.global_step,
                epoch=epoch,
                num_training_steps=self._t.num_training_steps,
            )

        if self._dist_name in ("fsdp2", "fsdpturbo", "deepspeed"):
            from ...plugins.trainer_plugins.distributed.interface import DistributedPlugin

            DistributedPlugin(self._dist_name).save_checkpoint(
                self._t.model,
                self._t.optimizer,
                ckpt_dir,
                save_ckpt_as_hf=self._t.args.save_ckpt_as_hf,
                processor=self._t.renderer.processor,
            )
        else:
            _save_standard_training_states(
                ckpt_dir,
                self._t.model,
                self._t.optimizer,
                self._t.renderer.processor,
                self._t.args.save_ckpt_as_hf,
            )

        if self._dist_name != "deepspeed" and rank == 0:
            torch.save(self._t.lr_scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))

        dl_dir = os.path.join(ckpt_dir, "dataloader")
        os.makedirs(dl_dir, exist_ok=True)
        torch.save(
            self._t.train_batch_generator.state_dict(),
            os.path.join(dl_dir, f"rank_{rank}.pt"),
        )

        if self._dist_name != "deepspeed":
            save_rng_state(ckpt_dir, rank)

        DistributedInterface().sync()

        if rank == 0:
            mark_checkpoint_complete(ckpt_dir)
            if self._t.args.save_total_limit is not None:
                rotate_checkpoints(self._t.args.output_dir, self._t.args.save_total_limit)

        logger.info_rank0(f"Checkpoint saved to {ckpt_dir}")

    def resume(self, ckpt_path: str) -> None:
        """Restore full training state from a checkpoint directory."""
        ckpt_dir = resolve_resume_checkpoint_path(ckpt_path, self._t.args.output_dir)
        if ckpt_dir is None:
            return

        if not os.path.isdir(ckpt_dir):
            raise ValueError(f"Checkpoint directory does not exist: {ckpt_dir}")

        rank = DistributedInterface().get_rank()

        metadata = load_metadata(ckpt_dir)
        self._t.global_step = metadata["global_step"]
        self._t._resume_epoch = metadata["epoch"]

        if self._dist_name in ("fsdp2", "fsdpturbo", "deepspeed"):
            from ...plugins.trainer_plugins.distributed.interface import DistributedPlugin

            DistributedPlugin(self._dist_name).load_checkpoint(
                self._t.model,
                self._t.optimizer,
                ckpt_dir,
                processor=self._t.renderer.processor,
            )
        else:
            _load_standard_training_states(
                ckpt_dir,
                self._t.model,
                self._t.optimizer,
                self._t.device,
            )

        if self._dist_name != "deepspeed":
            sched_path = os.path.join(ckpt_dir, "scheduler.pt")
            if os.path.exists(sched_path):
                self._t.lr_scheduler.load_state_dict(torch.load(sched_path, map_location="cpu", weights_only=True))

        dl_path = os.path.join(ckpt_dir, "dataloader", f"rank_{rank}.pt")

        if os.path.exists(dl_path):
            self._t.train_batch_generator.load_state_dict(torch.load(dl_path, map_location="cpu", weights_only=False))
        else:
            logger.warning_rank0(
                f"Dataloader state file not found at {dl_path}. Skipping Dataloader state restoration."
            )

        if self._dist_name != "deepspeed":
            load_rng_state(ckpt_dir, rank)

        logger.info_rank0(f"Resumed from checkpoint: step={self._t.global_step}, epoch={self._t._resume_epoch}")
