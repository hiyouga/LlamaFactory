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

"""HyperParallel distributed trainer for LlamaFactory."""

import logging
import os
import types
from contextlib import nullcontext
from functools import partial, wraps
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from hyper_parallel.core.optimizer import ChainedOptimizer, get_hyper_optimizer
from hyper_parallel.data.parallel import shard_batch_for_cp
from hyper_parallel.integration.llamafactory import (
    HSDPModule,
    HyperParallelArguments,
    export_to_hf_format,
    hsdp_sync_stream,
    load_hsdp_model,
    load_hsdp_optimizer_and_scheduler,
    save_hsdp_checkpoint,
    wrap_optimizer_with_skip_dtensor_dispatch,
)
from hyper_parallel.integration.llamafactory import (
    clip_grad_norm_ as hp_clip_grad_norm_,
)
from torch import nn

from ..sft.trainer import CustomSeq2SeqTrainer


logger = logging.getLogger(__name__)


class _TrainerOptimizerAdapter(torch.optim.Optimizer):
    """Expose a Hyper optimizer chain through PyTorch's Trainer protocol."""

    def __init__(self, optimizer: ChainedOptimizer):
        self.optimizer = optimizer
        super().__init__(optimizer.param_groups, optimizer.defaults)
        optimizer.param_groups = self.param_groups
        self.optimizers_dict = optimizer.optimizers_dict

    def step(self, closure=None):
        return self.optimizer.step(closure=closure)

    def zero_grad(self, set_to_none: bool = True):
        return self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self.optimizer.load_state_dict(state_dict)


def _create_hyper_muon_optimizer(model: nn.Module, training_args) -> torch.optim.Optimizer:
    """Create HyperParallel's distributed Muon and AdamW optimizer chain."""
    muon_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim == 2 and "embed" not in name and "lm_head" not in name:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    optimizer = get_hyper_optimizer(
        model=model,
        muon_params=[{"params": muon_params}] if muon_params else [],
        adamw_params=[{"params": adamw_params}] if adamw_params else [],
        muon_kwargs={
            "muon_lr": training_args.learning_rate,
            "muon_weight_decay": training_args.weight_decay,
        },
        adamw_kwargs={
            "adamw_lr": training_args.learning_rate,
            "adamw_weight_decay": training_args.weight_decay,
            "adamw_betas": (training_args.adam_beta1, training_args.adam_beta2),
            "adamw_eps": training_args.adam_epsilon,
        },
    )
    logger.info(
        "Using HyperParallel Muon optimizer with %d Muon params and %d AdamW params.",
        len(muon_params),
        len(adamw_params),
    )
    return _TrainerOptimizerAdapter(optimizer)


def _shard_inputs_for_cp(inputs: dict[str, Any], cp_mesh) -> dict[str, Any]:
    """Adapt HuggingFace batches to Hyper's existing CP batch contract."""
    inputs = dict(inputs)
    seq_len = inputs["input_ids"].shape[1]
    if "position_ids" not in inputs:
        position_ids = torch.arange(
            seq_len,
            device=inputs["input_ids"].device,
            dtype=torch.long,
        ).unsqueeze(0)
        inputs["position_ids"] = position_ids.expand(inputs["input_ids"].shape[0], -1)

    global_attention_mask = inputs.get("attention_mask")
    global_labels = inputs.get("labels")
    preserve_attention_mask = (
        isinstance(global_attention_mask, torch.Tensor)
        and global_attention_mask.ndim == 2
        and global_attention_mask.shape[-1] == seq_len
    )
    sharded = shard_batch_for_cp(inputs, cp_mesh)
    if preserve_attention_mask:
        pad_len = (-seq_len) % (cp_mesh.size() * 2)
        if pad_len > 0:
            global_attention_mask = torch.nn.functional.pad(global_attention_mask, (0, pad_len), value=0)
        sharded["attention_mask"] = global_attention_mask
    if isinstance(global_labels, torch.Tensor) and global_labels.shape[-1] == seq_len:
        sharded["labels"] = global_labels
    return sharded


def _pad_inputs_for_model_cp(inputs: dict[str, Any], cp_mesh) -> dict[str, Any]:
    """Pad only token-aligned fields when a multimodal model shards internally."""
    inputs = dict(inputs)
    seq_len = inputs["input_ids"].shape[1]
    pad_len = (-seq_len) % (cp_mesh.size() * 2)
    if pad_len == 0:
        return inputs

    pad_values = {
        "input_ids": 0,
        "labels": -100,
        "attention_mask": 0,
        "mm_token_type_ids": 0,
    }
    for name in ("input_ids", "labels", "attention_mask", "mm_token_type_ids", "position_ids"):
        value = inputs.get(name)
        if not isinstance(value, torch.Tensor) or value.shape[-1] != seq_len:
            continue
        if name == "position_ids":
            increment = torch.arange(1, pad_len + 1, device=value.device, dtype=value.dtype)
            increment = increment.reshape(*([1] * (value.ndim - 1)), pad_len)
            pad = value[..., -1:] + increment
            pad = pad.expand(*value.shape[:-1], pad_len)
        else:
            pad = value.new_full((*value.shape[:-1], pad_len), pad_values[name])
        inputs[name] = torch.cat((value, pad), dim=-1)
    return inputs


def _build_cp_shift_labels(
    labels: torch.Tensor,
    local_seq_len: int,
    cp_rank: int,
    ignore_index: int,
) -> torch.Tensor:
    """Align full-sequence causal labels with one rank's local logits."""
    start = cp_rank * local_seq_len
    end = start + local_seq_len
    shift_labels = labels[..., start + 1 : min(end + 1, labels.shape[-1])]
    if shift_labels.shape[-1] < local_seq_len:
        pad_shape = (*shift_labels.shape[:-1], local_seq_len - shift_labels.shape[-1])
        shift_labels = torch.cat(
            (shift_labels, shift_labels.new_full(pad_shape, ignore_index)),
            dim=-1,
        )
    return shift_labels.contiguous()


def _enable_cp_causal_lm_loss(model: nn.Module, cp_mesh) -> None:
    """Use full labels to preserve causal boundaries and CP-global normalization."""
    cp_size = cp_mesh.size()
    cp_rank = cp_mesh.get_local_rank()
    cp_group = cp_mesh.get_group()
    original_loss_function = model.loss_function

    @wraps(original_loss_function)
    def cp_loss_function(
        logits,
        labels,
        vocab_size: int,
        num_items_in_batch=None,
        ignore_index: int = -100,
        shift_labels=None,
        **kwargs,
    ):
        if (
            shift_labels is not None
            or not isinstance(logits, torch.Tensor)
            or not isinstance(labels, torch.Tensor)
            or logits.ndim < 3
            or labels.ndim < 2
            or logits.shape[-2] * cp_size != labels.shape[-1]
        ):
            return original_loss_function(
                logits=logits,
                labels=labels,
                vocab_size=vocab_size,
                num_items_in_batch=num_items_in_batch,
                ignore_index=ignore_index,
                shift_labels=shift_labels,
                **kwargs,
            )

        local_shift_labels = _build_cp_shift_labels(
            labels,
            local_seq_len=logits.shape[-2],
            cp_rank=cp_rank,
            ignore_index=ignore_index,
        ).to(logits.device)
        local_logits = logits[..., :vocab_size].contiguous().float()
        local_loss_sum = F.cross_entropy(
            local_logits.view(-1, vocab_size),
            local_shift_labels.view(-1),
            ignore_index=ignore_index,
            reduction="sum",
        )
        if num_items_in_batch is not None:
            denominator = num_items_in_batch
            if isinstance(denominator, torch.Tensor):
                denominator = denominator.to(device=local_loss_sum.device, dtype=local_loss_sum.dtype)
            # Trainer counts the repeated full labels on every CP rank. Remove
            # that duplication here; Trainer applies its normal world-size
            # compensation after this loss function returns.
            return (
                local_loss_sum
                * cp_size
                / torch.clamp(
                    torch.as_tensor(denominator, device=local_loss_sum.device),
                    min=1,
                )
            )

        cp_num_items = local_shift_labels.ne(ignore_index).sum()
        dist.all_reduce(cp_num_items, group=cp_group)
        denominator = torch.clamp(cp_num_items.to(local_loss_sum.dtype), min=1)

        # Hyper FSDP averages gradients over the flattened DP x CP mesh.
        backward_loss = local_loss_sum * cp_size / denominator
        group_loss_sum = local_loss_sum.detach().clone()
        dist.all_reduce(group_loss_sum, group=cp_group)
        logging_loss = group_loss_sum / denominator
        return backward_loss + (logging_loss - backward_loss).detach()

    model.loss_function = cp_loss_function


class _CPBatchRepeatedBatchSampler(torch.utils.data.BatchSampler):
    """Repeat logical batches so Accelerate shards CP peers onto the same samples."""

    def __init__(self, sampler, batch_size: int, drop_last: bool, repeat_factor: int, logical_group_size: int):
        super().__init__(sampler, batch_size, drop_last)
        self.repeat_factor = repeat_factor
        self.logical_group_size = logical_group_size

    def __len__(self):
        logical_length = super().__len__()
        if not self.drop_last and logical_length > 0:
            logical_length = _ceil_div(logical_length, self.logical_group_size) * self.logical_group_size
        return logical_length * self.repeat_factor

    def __iter__(self):
        initial_data = []
        logical_count = 0
        pad_cursor = 0
        max_initial_data = self.batch_size * self.logical_group_size

        def collect_initial_data(batch):
            if len(initial_data) < max_initial_data:
                initial_data.extend(batch[: max_initial_data - len(initial_data)])

        def get_padding_item():
            nonlocal pad_cursor
            item = initial_data[pad_cursor % len(initial_data)]
            pad_cursor += 1
            return item

        def pad_batch(batch):
            batch = list(batch)
            if self.drop_last or len(batch) == self.batch_size:
                return batch

            while len(batch) < self.batch_size:
                batch.append(get_padding_item())
            return batch

        def make_padding_batch():
            return [get_padding_item() for _ in range(self.batch_size)]

        def repeat_batch(batch):
            for _ in range(self.repeat_factor):
                yield list(batch)

        for batch in super().__iter__():
            collect_initial_data(batch)
            batch = pad_batch(batch)
            logical_count += 1
            yield from repeat_batch(batch)

        if self.drop_last or logical_count == 0:
            return

        while logical_count % self.logical_group_size != 0:
            logical_count += 1
            yield from repeat_batch(make_padding_batch())


class _CPDataLoaderLengthProxy:
    """Keep baseline logical dataloader length while yielding CP-repeated batches."""

    def __init__(self, dataloader, logical_length: int):
        self._dataloader = dataloader
        self._logical_length = logical_length

    def __iter__(self):
        return iter(self._dataloader)

    def __len__(self):
        return self._logical_length

    def __getattr__(self, name):
        return getattr(self._dataloader, name)


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


class HyperParallelTrainer(CustomSeq2SeqTrainer):
    """Trainer that replaces Accelerate FSDP2 with HyperParallel fully_shard.

    Inherits CustomSeq2SeqTrainer for training algorithm logic (loss, metrics,
    prediction, sampler, etc.) and only overrides HSDP-specific behavior.
    """

    def __init__(
        self,
        hp_args: HyperParallelArguments,
        distributed_setup,
        finetuning_args=None,
        processor=None,
        **kwargs,
    ):
        # Keep LlamaFactory's Trainer lifecycle; Hyper owns only the already
        # constructed model's distributed layout and checkpoint state.
        super().__init__(
            finetuning_args=finetuning_args,
            processor=processor,
            ref_model=None,
            **kwargs,
        )

        if not getattr(self.accelerator, "is_fsdp2", False):
            raise ValueError("HyperParallel trainer requires Accelerate FSDP2 mode to be enabled.")

        self._cp_size = hp_args.cp_size
        self._cp_mesh = distributed_setup.mesh_context.cp_mesh
        if self._cp_size > 1 and self._cp_mesh is None:
            raise RuntimeError("HyperParallel CP requires a cp_mesh from DistributedSetup.")
        if self._cp_size > 1:
            _enable_cp_causal_lm_loss(self.model, self._cp_mesh)
        self._model_shards_cp_inputs = any(
            getattr(module, "_hp_cp_shards_inputs", False) for module in self.model.modules()
        )

        self._orig_accelerator_clip_grad_norm = self.accelerator.clip_grad_norm_
        self._orig_fsdp2_prepare_model = None
        self._orig_fsdp2_set_auto_wrap_policy = None
        self._accelerator_patches_active = False

    def _prepare_model_for_hyper_parallel(self, model: nn.Module) -> nn.Module:
        """Return a model already prepared by HyperParallel's unified model path."""
        if not isinstance(model, HSDPModule):
            raise RuntimeError(
                "HyperParallel models must be parallelized during meta initialization before Trainer preparation."
            )
        return model

    def _activate_accelerator_patches(self) -> None:
        """Keep Accelerate from rewrapping the prepared model and patch gradient clipping."""
        if self._accelerator_patches_active:
            return

        import accelerate.accelerator as acc_module  # pylint: disable=C0415

        self._orig_fsdp2_prepare_model = acc_module.fsdp2_prepare_model
        fsdp_plugin = self.accelerator.state.fsdp_plugin
        self._orig_fsdp2_set_auto_wrap_policy = fsdp_plugin.set_auto_wrap_policy

        def _hp_fsdp2_prepare_model(accelerator, model):
            return self._prepare_model_for_hyper_parallel(model)

        def _hp_set_auto_wrap_policy(plugin, model):
            del plugin
            self._prepare_model_for_hyper_parallel(model)

        acc_module.fsdp2_prepare_model = _hp_fsdp2_prepare_model
        fsdp_plugin.set_auto_wrap_policy = types.MethodType(_hp_set_auto_wrap_policy, fsdp_plugin)

        def _hp_clip_grad_norm(accelerator, parameters, max_norm, norm_type=2):
            if getattr(accelerator, "is_fsdp2", False):
                accelerator.unscale_gradients()
                parameter_list = list(parameters)
                parameter_ids = {id(param) for param in parameter_list}
                for model in accelerator._models:  # pylint: disable=protected-access
                    if not isinstance(model, HSDPModule):
                        continue
                    model_param_ids = {id(param) for param in model.parameters()}
                    if parameter_ids and parameter_ids.issubset(model_param_ids):
                        return hp_clip_grad_norm_(parameter_list, max_norm, norm_type=norm_type)
            return self._orig_accelerator_clip_grad_norm(parameters, max_norm, norm_type=norm_type)

        self.accelerator.clip_grad_norm_ = types.MethodType(_hp_clip_grad_norm, self.accelerator)
        self._accelerator_patches_active = True

    def _restore_accelerator_patches(self) -> None:
        """Restore original Accelerate methods."""
        if not self._accelerator_patches_active:
            return

        import accelerate.accelerator as acc_module  # pylint: disable=C0415

        if self._orig_fsdp2_prepare_model is not None:
            acc_module.fsdp2_prepare_model = self._orig_fsdp2_prepare_model
        if self._orig_fsdp2_set_auto_wrap_policy is not None:
            self.accelerator.state.fsdp_plugin.set_auto_wrap_policy = self._orig_fsdp2_set_auto_wrap_policy
        self.accelerator.clip_grad_norm_ = self._orig_accelerator_clip_grad_norm
        self._accelerator_patches_active = False

    def _wrap_model(self, model: nn.Module, training: bool = True, dataloader=None) -> nn.Module:
        """Keep Trainer from wrapping a model already prepared by HyperParallel."""
        del dataloader
        if isinstance(model, HSDPModule):
            return model
        return super()._wrap_model(model, training=training)

    def _get_train_sampler(self, train_dataset=None):
        """Match the no-CP baseline sampler semantics before CP repeats whole logical batches."""
        if train_dataset is None:
            train_dataset = self.train_dataset
        if getattr(self.finetuning_args, "disable_shuffling", False):
            return torch.utils.data.SequentialSampler(train_dataset)
        return super()._get_train_sampler(train_dataset)

    def _build_cp_batch_sampler(self, dataset, shuffle: bool, batch_size: int, drop_last: bool):
        """Repeat complete logical batches so CP groups consume the same baseline batch."""
        sampler = self._get_train_sampler(dataset) if shuffle else torch.utils.data.SequentialSampler(dataset)
        return _CPBatchRepeatedBatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            repeat_factor=self._cp_size,
            logical_group_size=max(1, dist.get_world_size() // self._cp_size),
        )

    def _get_cp_dataloader(self, dataset, batch_size: int, shuffle: bool):
        """Create a train dataloader whose logical batches are shared within each CP group."""
        if isinstance(dataset, torch.utils.data.IterableDataset):
            raise NotImplementedError(
                "HyperParallel CP training requires a map-style dataset because iterable datasets cannot "
                "repeat logical batches across CP ranks."
            )

        try:
            import datasets  # pylint: disable=C0415
        except ImportError:  # pragma: no cover
            datasets = None

        if datasets is not None and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description="Training")
            data_collator = self.data_collator
        else:
            data_collator = self._get_collator_with_removed_columns(self.data_collator, description="Training")

        batch_sampler = self._build_cp_batch_sampler(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=self.args.dataloader_drop_last,
        )
        logical_batches = len(batch_sampler) // self._cp_size
        dp_size = max(1, dist.get_world_size() // self._cp_size)
        logical_length = (
            logical_batches // dp_size if self.args.dataloader_drop_last else _ceil_div(logical_batches, dp_size)
        )

        dataloader_params = {
            "batch_sampler": batch_sampler,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers
            if self.args.dataloader_num_workers > 0
            else False,
        }
        if self.args.dataloader_num_workers > 0:
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        from transformers.trainer import seed_worker  # pylint: disable=C0415

        dataloader_params["worker_init_fn"] = partial(
            seed_worker,
            num_workers=self.args.dataloader_num_workers,
            rank=self.args.process_index,
        )

        dataloader = self.accelerator.prepare(torch.utils.data.DataLoader(dataset, **dataloader_params))
        return _CPDataLoaderLengthProxy(dataloader, logical_length)

    def get_train_dataloader(self):
        """Keep the no-CP logical batch stream, then repeat each whole batch across CP peers."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if self._cp_size <= 1:
            return super().get_train_dataloader()

        shuffle = not getattr(self.finetuning_args, "disable_shuffling", False)
        return self._get_cp_dataloader(
            dataset=self.train_dataset,
            batch_size=self._train_batch_size,
            shuffle=shuffle,
        )

    def _move_model_to_device(self, model: nn.Module, device: Optional[torch.device] = None):
        """Skip redundant device moves for HSDP-wrapped models."""
        if isinstance(model, HSDPModule):
            return model
        if device is None:
            return model
        return model.to(device)

    def train(self, *args, **kwargs):
        """Activate HP patches during training and restore afterwards."""
        self._activate_accelerator_patches()
        try:
            return super().train(*args, **kwargs)
        finally:
            self._restore_accelerator_patches()

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """Standard training step with HSDP sync plus optional CP input sharding."""
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self._cp_size > 1:
            if self._model_shards_cp_inputs:
                inputs = _pad_inputs_for_model_cp(inputs, self._cp_mesh)
            else:
                inputs = _shard_inputs_for_cp(inputs, self._cp_mesh)

        sync_gradients = getattr(self.accelerator, "sync_gradients", True)
        if isinstance(model, HSDPModule):
            model.set_is_last_backward(sync_gradients)
            model.set_requires_gradient_sync(sync_gradients)

        compute_loss_context_manager = getattr(self, "compute_loss_context_manager", nullcontext)
        with compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if not getattr(self, "model_accepts_loss_kwargs", False) and getattr(self, "compute_loss_func", None) is None:
            accumulation_steps = getattr(
                self,
                "current_gradient_accumulation_steps",
                self.args.gradient_accumulation_steps,
            )
            loss = loss / accumulation_steps

        # The loss already accounts for the accumulation window, either via
        # num_items_in_batch or the explicit division above. Compensate the
        # additional scaling inside Accelerator.backward() so it happens once.
        backward_loss = loss * self.accelerator.gradient_accumulation_steps
        self.accelerator.backward(backward_loss)

        if isinstance(model, HSDPModule) and sync_gradients:
            hsdp_sync_stream()

        return loss.detach()

    def create_optimizer(self, *args, **kwargs):
        """Create optimizer and wrap step with SkipDTensorDispatch."""
        if self.optimizer is None and getattr(self.finetuning_args, "use_muon", False):
            model = args[0] if args else kwargs.get("model", self.model)
            self.optimizer = _create_hyper_muon_optimizer(model, self.args)
            optimizer = self.optimizer
        else:
            optimizer = super().create_optimizer(*args, **kwargs)
        wrap_optimizer_with_skip_dtensor_dispatch(optimizer)
        return optimizer

    def _save_optimizer_and_scheduler(self, output_dir: str) -> None:
        """Save model/optimizer shards per-rank and scheduler."""
        save_hsdp_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            output_dir=output_dir,
            should_save_scheduler=self.args.should_save and self.lr_scheduler is not None,
        )

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model: Optional[nn.Module] = None) -> None:
        """Load model from HSDP sharded checkpoint."""
        target = model if model is not None else self.model
        loaded = load_hsdp_model(target, resume_from_checkpoint)
        if not loaded:
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        self._pending_hsdp_checkpoint = resume_from_checkpoint
        return None

    def _load_optimizer_and_scheduler(self, checkpoint: Optional[str] = None) -> None:
        """Load optimizer/scheduler from per-rank checkpoint files."""
        ckpt_dir = getattr(self, "_pending_hsdp_checkpoint", None) or checkpoint
        if ckpt_dir is None:
            return
        load_hsdp_optimizer_and_scheduler(self.optimizer, self.lr_scheduler, ckpt_dir)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save model weights in HuggingFace-compatible format."""
        save_dir = output_dir or self.args.output_dir
        os.makedirs(save_dir, exist_ok=True)
        export_to_hf_format(self.model, getattr(self, "processing_class", None), save_dir)
