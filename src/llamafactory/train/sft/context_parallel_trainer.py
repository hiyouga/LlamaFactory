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

from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...model.model_utils.context_parallel import configure_context_parallel_model
from .context_parallel import (
    ContextParallelSampler,
    context_parallel_cross_entropy,
    create_context_parallel_group,
    make_shift_labels,
    split_sequence_inputs,
)
from .trainer import CustomSeq2SeqTrainer


if TYPE_CHECKING:
    from torch.utils.data import Sampler


logger = logging.get_logger(__name__)


class ContextParallelSeq2SeqTrainer(CustomSeq2SeqTrainer):
    """SFT trainer that shards each text sequence across a Ulysses process group."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.context_parallel_size = self.args.ulysses_context_parallel_size
        if self.context_parallel_size <= 1:
            raise ValueError("ContextParallelSeq2SeqTrainer requires `ulysses_context_parallel_size > 1`.")

        self.context_parallel_group = create_context_parallel_group(self.context_parallel_size)
        self.context_parallel_rank = dist.get_rank(self.context_parallel_group)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        configure_context_parallel_model(
            unwrapped_model,
            self.context_parallel_group,
            self.context_parallel_size,
            self.finetuning_args,
        )

        # Transformers computes the full-world token count before this class
        # removes duplicate CP replicas in `_get_num_items_in_batch`.
        self.model_accepts_loss_kwargs = True
        logger.info_rank0(
            f"Enabled Ulysses context parallelism: cp_size={self.context_parallel_size}, "
            f"dp_size={dist.get_world_size() // self.context_parallel_size}."
        )

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["Sampler"]:
        sampler = super()._get_train_sampler(*args, **kwargs)
        if sampler is None:
            return None

        return ContextParallelSampler(sampler, self.context_parallel_size)

    @override
    def _get_num_items_in_batch(self, batch_samples, device=None):
        num_items = None
        for batch in batch_samples:
            try:
                labels = batch["labels"]
            except (KeyError, TypeError, IndexError):
                continue
            count = make_shift_labels(labels).ne(IGNORE_INDEX).sum()
            num_items = count if num_items is None else num_items + count

        if num_items is None:
            return None

        num_items = num_items.to(device if device is not None else self.args.device)
        cp_min_items = num_items.clone()
        cp_max_items = num_items.clone()
        dist.all_reduce(cp_min_items, op=dist.ReduceOp.MIN, group=self.context_parallel_group)
        dist.all_reduce(cp_max_items, op=dist.ReduceOp.MAX, group=self.context_parallel_group)
        if not torch.equal(cp_min_items, cp_max_items):
            raise RuntimeError("Ranks in a context-parallel group received different label counts.")

        dist.all_reduce(num_items, op=dist.ReduceOp.SUM)
        if num_items.remainder(self.context_parallel_size).item() != 0:
            raise RuntimeError("The full-world label count is not divisible by the context-parallel size.")

        return num_items // self.context_parallel_size

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        inputs = dict(inputs)
        if "position_ids" not in inputs:
            inputs["position_ids"] = (
                torch.arange(inputs["input_ids"].shape[1], device=inputs["input_ids"].device)
                .unsqueeze(0)
                .expand(inputs["input_ids"].shape[0], -1)
            )
        inputs["shift_labels"] = make_shift_labels(inputs["labels"])
        inputs = split_sequence_inputs(inputs, self.context_parallel_size, self.context_parallel_rank)
        shift_labels = inputs.pop("shift_labels")
        inputs.pop("labels")
        outputs = model(**inputs)
        num_items_in_batch = kwargs.get("num_items_in_batch")
        if num_items_in_batch is None:
            raise RuntimeError("Context parallel loss requires `num_items_in_batch`.")

        loss = context_parallel_cross_entropy(
            outputs.logits,
            shift_labels,
            num_items_in_batch=num_items_in_batch,
            world_size=dist.get_world_size(),
        )
        if kwargs.get("return_outputs", False):
            return loss, outputs

        return loss
