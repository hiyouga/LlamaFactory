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

"""The definition of trainer.

Init Phase:

1. Init batch generator.
2. Init optimizer (deepspeed).
3. Shard model.
4. Init optimizer (fsdp).
5. Init lr scheduler.

Train Phase:
1. Train Loop

"""

from abc import abstractmethod

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from ..accelerator.helper import ReduceOp
from ..accelerator.interface import Dim, DistributedInterface
from ..config import BatchingStrategy, TrainingArguments
from ..utils import logging
from ..utils.callbacks import (
    CallbackHandler,
    LoggingCallback,
    TrainerCallback,
    TrainerState,
)
from ..utils.helper import compute_valid_tokens, is_tokenizer, model_uses_mrope
from ..utils.types import BatchInput, HFModel, ModelOutput, Tensor, TorchDataset
from .rendering import Renderer
from .utils.batching import BatchGenerator
from .utils.checkpoint import TrainingCheckpointCoordinator


logger = logging.get_logger(__name__)


class BaseTrainer:
    def __init__(
        self,
        args: TrainingArguments,
        model: HFModel,
        renderer: Renderer,
        train_dataset: TorchDataset,
        callbacks: list[TrainerCallback] | None = None,
    ) -> None:
        self.args = args
        self.model = model
        self.renderer = renderer
        self.train_dataset = train_dataset

        # info
        self.global_step = 0

        # cached variables
        self.device = DistributedInterface().current_device
        self.dp_size = DistributedInterface().get_world_size(Dim.DP)
        self.cp_size = DistributedInterface().get_world_size(Dim.CP)
        self.model_input_names = self.renderer.processor.model_input_names
        self._uses_mrope = model_uses_mrope(self.model.config)

        self._create_batch_generator()
        # Calculate num_training_steps: max_steps takes priority if set
        if self.args.max_steps is not None and self.args.max_steps > 0:
            self.num_training_steps = self.args.max_steps
        else:
            self.num_training_steps = self.args.num_train_epochs * len(self.train_batch_generator)

        if self.args.save_epochs is not None:
            steps_per_epoch = len(self.train_batch_generator)
            self.args.save_steps = max(1, int(steps_per_epoch * self.args.save_epochs))

        if self.args.enable_activation_checkpointing:
            self.model.gradient_checkpointing_enable({"use_reentrant": False})
            # Note: under FSDP2 bf16, encoder-tower nn.LayerNorms are made dtype-safe for the
            # checkpoint recompute inside the FSDP2 engine (see fsdp2.py prepare_model), so the
            # tower keeps activation checkpointing too.

        self._deepspeed_engine = None
        dist_name = self.args.dist_config.name if self.args.dist_config is not None else None

        if dist_name == "deepspeed":
            if self.args.cp_size > 1:
                raise ValueError("Context parallelism currently requires `dist_config.name: fsdp2`.")

            from ..plugins.trainer_plugins.distributed.interface import DistributedPlugin

            self._deepspeed_engine = DistributedPlugin("deepspeed").shard_model(
                self.model,
                self.args.dist_config,
                num_micro_batch=self.train_batch_generator.num_micro_batch,
                micro_batch_size=self.args.micro_batch_size,
            )
            self._init_optimizer()
            self._init_lr_scheduler()
            self.model, self.optimizer, self.lr_scheduler = self._deepspeed_engine.prepare(
                self.model, self.optimizer, self.lr_scheduler
            )
        else:
            # fsdp2 / DDP / no dist
            self._shard_model()
            self._init_optimizer()
            self._init_lr_scheduler()

        self._resume_epoch = 0
        self._checkpoint = TrainingCheckpointCoordinator(self)
        if self.args.resume_from_checkpoint:
            self._checkpoint.resume(self.args.resume_from_checkpoint)

        if self.args.save_ckpt_as_hf:
            logger.warning_rank0(
                "save_ckpt_as_hf is enabled. Intermediate checkpoints will be saved in Hugging Face format. "
                "Note that this will significantly increase memory consumption during saving."
            )

        # Callbacks
        self.callback_handler = CallbackHandler([LoggingCallback()], trainer=self)
        for cb in callbacks or []:
            self.callback_handler.add_callback(cb)

        # Callbacks: TrainerState tracks progress across the full run.
        self.state = TrainerState(
            num_training_steps=self.num_training_steps,
            global_step=self.global_step,
            epoch=self._resume_epoch,
        )
        # Keep callback state aligned with checkpoint-resumed trainer counters.
        self.state.global_step = self.global_step
        self.state.epoch = self._resume_epoch

        if self.args.cp_size > 1:
            from ..plugins.model_plugins.parallelization.sequence_parallel import SequenceParallelModelPlugin

            if model.config._attn_implementation != "flash_attention_2":
                raise ValueError(
                    "Sequence parallelism requires flash attention. Please set `flash_attn: flash_attention_2`."
                )

            SequenceParallelModelPlugin(self.args.cp_mode)(model, self.args.cp_size)

    def _create_batch_generator(self) -> None:
        if (
            self.args.batching_strategy == BatchingStrategy.PADDING_FREE
            and getattr(self.model.config, "_attn_implementation", None) != "flash_attention_2"
        ):
            raise ValueError("`padding_free` requires `flash_attn: flash_attention_2`.")

        self.train_batch_generator = BatchGenerator(
            dataset=self.train_dataset,
            renderer=self.renderer,
            micro_batch_size=self.args.micro_batch_size,
            global_batch_size=self.args.global_batch_size,
            cutoff_len=self.args.cutoff_len,
            batching_workers=self.args.batching_workers,
            batching_strategy=self.args.batching_strategy,
            seed=self.args.seed,
        )

    def _shard_model(self) -> None:
        if self.args.dist_config is None:
            if DistributedInterface().get_world_size(Dim.DP) > 1:
                from torch.nn.parallel import DistributedDataParallel as DDP

                logger.warning_rank0(
                    "dist_config is None but distributed training is enabled; falling back to DistributedDataParallel."
                )
                device_ids = None if self.device.type == "cpu" else [self.device.index]
                # Multimodal models invoke the vision tower only when a step carries media; a
                # globally media-less step leaves vision params unused, which trips DDP's default
                # all-params-used assertion. (FSDP tolerates a uniform skip; DDP does not.)
                find_unused = not is_tokenizer(self.renderer.processor)
                self.model = DDP(self.model, device_ids=device_ids, find_unused_parameters=find_unused)
        else:
            from ..plugins.trainer_plugins.distributed.interface import DistributedPlugin

            self.model = DistributedPlugin(self.args.dist_config.name).shard_model(
                self.model,
                self.args.dist_config,
                bf16=self.args.bf16,
            )

    def _init_optimizer(self) -> None:
        """Init optimizer."""
        if self.args.optim_config is None:
            _trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(_trainable_params, lr=self.args.learning_rate)
        else:
            from ..plugins.trainer_plugins.optimizers.optimizer import OptimizerPlugin

            self.optimizer = OptimizerPlugin(self.args.optim_config.name)(self.model, self.args.optim_config)

    def _init_lr_scheduler(self) -> None:
        """Init lr scheduler."""
        if self.args.lr_scheduler_config is None:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda x: 1.0)
        else:
            from ..plugins.trainer_plugins.lr_scheduler import LRSchedulerPlugin

            self.lr_scheduler = LRSchedulerPlugin(self.args.lr_scheduler_config.name)(
                self.optimizer, self.num_training_steps, self.args.lr_scheduler_config
            )

    def compute_log_probs(self, model: HFModel, batch: BatchInput) -> Tensor:
        """Compute log probs.

        log_probs: Tensor of shape (batch_size, seq_len - 1)
        """
        batch_size, _ = batch["labels"].shape
        model_inputs = {
            k: v.to(self.device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        # Let mRoPE models build their own multimodal 3D position ids (see _uses_mrope in __init__).
        if self._uses_mrope:
            model_inputs.pop("position_ids", None)
        labels = batch["labels"].to(self.device, non_blocking=True)
        outputs: ModelOutput = model(**model_inputs)
        logits = outputs.logits.float()
        shift_labels = labels[..., 1:].contiguous().view(-1)
        shift_logits = logits[..., :-1, :].contiguous().view(shift_labels.size(0), -1)
        return -F.cross_entropy(shift_logits, shift_labels, reduction="none").view(batch_size, -1)

    @abstractmethod
    def compute_loss(self, batch: BatchInput) -> Tensor:
        """Compute the scalar loss."""
        ...

    def fit(self) -> None:
        """Train the model."""
        self.model.train()
        self.callback_handler.on_train_begin(self.args, self.state)

        epoch = self._resume_epoch
        while self.global_step < self.num_training_steps:
            self.state.epoch = epoch
            self.train_batch_generator.set_epoch(epoch)
            self.callback_handler.on_epoch_begin(self.args, self.state)

            # BatchGenerator is an iterator; each loop step calls its __next__ to produce one optimizer step.
            for micro_batches in self.train_batch_generator:
                self.global_step += 1

                self.state.global_step = self.global_step
                self.callback_handler.on_step_begin(self.args, self.state)

                step_loss = 0
                step_valid_tokens = compute_valid_tokens(micro_batches)
                step_valid_tokens = DistributedInterface().all_reduce(step_valid_tokens, op=ReduceOp.SUM)
                num_micro = len(micro_batches)
                for i, micro_batch in enumerate(micro_batches):
                    if self.args.cp_size > 1:
                        from ..plugins.model_plugins.parallelization.sequence_parallel import (
                            SequenceParallelLossPlugin,
                        )

                        loss = SequenceParallelLossPlugin("sequence_parallel_loss")(self.model, micro_batch)
                    else:
                        loss = self.compute_loss(micro_batch)
                    mini_step_valid_tokens = compute_valid_tokens([micro_batch])
                    # fsdp uses mean reduction so we need to scale the loss by dp_size
                    loss = loss * mini_step_valid_tokens * self.dp_size / (step_valid_tokens + 1e-6)

                    if self._deepspeed_engine is not None:
                        # deepspeed: set sync_gradients so engine.step() only fires on last micro-batch
                        self._deepspeed_engine.accelerator.sync_gradients = i == num_micro - 1
                        self._deepspeed_engine.backward(loss)
                    else:
                        loss.backward()
                    step_loss += loss.item()

                if self._deepspeed_engine is not None:
                    # deepspeed: engine.step() already ran inside backward at the sync boundary
                    grad_norm = self._deepspeed_engine.get_grad_norm()
                else:
                    dist_name = self.args.dist_config.name if self.args.dist_config else None
                    if dist_name == "fsdpturbo":
                        from ..plugins.trainer_plugins.distributed.interface import DistributedPlugin

                        grad_norm = DistributedPlugin(dist_name).clip_grad_norm(self.model, self.args.max_grad_norm)
                    else:
                        # FSDP2 shards params/grads across the fsdp mesh, so clip_grad_norm_ returns a
                        # per-rank local shard norm. Materialize the true global norm before clipping.
                        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
                        total_norm = torch.nn.utils.get_total_norm(grads)
                        if isinstance(total_norm, DTensor):
                            # full_tensor all-reduces across the fsdp mesh (spans CP under default
                            # mp_shard=world); a separate CP reduce would over-count by sqrt(cp_size).
                            total_norm = total_norm.full_tensor()
                        torch.nn.utils.clip_grads_with_norm_(
                            self.model.parameters(), self.args.max_grad_norm, total_norm
                        )
                        grad_norm = total_norm.item()
                        # Do not retain a full generation of gradient tensors across optimizer
                        # steps. ``zero_grad(set_to_none=True)`` clears ``param.grad``, but this
                        # local list would otherwise keep every old gradient alive until the next
                        # assignment, doubling gradient memory during the following backward.
                        del grads

                    if not torch.isfinite(torch.tensor(grad_norm)):  # type: ignore # pyright: ignore [reportUnknownReturnType]
                        logger.warning_rank0(f"Gradient norm is not finite: {grad_norm}")
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                step_loss, grad_norm = DistributedInterface().all_reduce([step_loss, grad_norm])
                DistributedInterface().sync()

                # Update state with step metrics
                current_lr = (
                    self.lr_scheduler.get_last_lr()[0]
                    if hasattr(self.lr_scheduler, "get_last_lr")
                    else self.args.learning_rate
                )
                self.state.loss = step_loss
                self.state.grad_norm = grad_norm
                self.state.learning_rate = current_lr

                self.callback_handler.on_step_end(self.args, self.state)

                # Logging: trainer decides when to log
                if self.global_step % self.args.logging_steps == 0:
                    logs = {
                        "epoch": epoch,
                        "step": self.state.global_step,
                        "loss": step_loss,
                        "grad_norm": grad_norm,
                        "learning_rate": current_lr,
                    }
                    # Merge per-step trainer metrics (e.g. DPO rewards/logps/logits)
                    step_metrics = getattr(self, "_step_metrics", None)
                    if step_metrics:
                        logs.update(step_metrics)
                    self.callback_handler.on_log(self.args, self.state, logs)

                if self.args.save_steps and self.global_step % self.args.save_steps == 0:
                    self._checkpoint.save(epoch)

                # Check if max_steps is reached
                if self.global_step >= self.num_training_steps:
                    logger.info_rank0(f"Reached max_steps ({self.num_training_steps}), stopping training.")
                    self.callback_handler.on_epoch_end(self.args, self.state)
                    self.callback_handler.on_train_end(self.args, self.state)
                    return

            self.callback_handler.on_epoch_end(self.args, self.state)
            epoch += 1

        self.callback_handler.on_train_end(self.args, self.state)

    def save_model(self) -> None:
        """Save the model."""
        if self.args.dist_config is not None and self.args.dist_config.name in ("deepspeed", "fsdp2", "fsdpturbo"):
            from ..plugins.trainer_plugins.distributed.interface import DistributedPlugin

            DistributedPlugin(self.args.dist_config.name).save_model(
                self.model, self.args.output_dir, self.renderer.processor
            )
        else:
            model_to_save = self.model.module if hasattr(self.model, "module") else self.model
            model_to_save.save_pretrained(
                self.args.output_dir, state_dict=model_to_save.state_dict(), max_shard_size="4GB"
            )
            self.renderer.processor.save_pretrained(self.args.output_dir, max_shard_size="4GB")
            logger.info_rank0(f"Model saved to {self.args.output_dir}")

        self.callback_handler.on_save(self.args, self.state)
