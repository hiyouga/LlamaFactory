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


import torch
import torch.nn.functional as F

from ..accelerator.interface import Dim, DistributedInterface
from ..config import InputArgument, TrainingArguments, get_args
from ..config.arg_utils import ModelClass
from ..core.base_trainer import BaseTrainer
from ..core.data_engine import DataEngine
from ..core.model_engine import ModelEngine
from ..utils import logging
from ..utils.types import BatchInput, HFModel, Tensor
from .dropout import disable_dropout_in_model


logger = logging.get_logger(__name__)


def _validate_rm_dataset_format(train_dataset: DataEngine, dataset_path: str) -> None:
    """Validate RM dataset format early for clearer error messages."""
    if len(train_dataset) == 0:
        raise ValueError(f"RM training dataset is empty: {dataset_path}")

    sample = train_dataset[0]
    if "chosen_messages" in sample and "rejected_messages" in sample:
        return

    dataset_name = sample.get("_dataset_name", "unknown")
    sample_keys = sorted(sample.keys())
    raise ValueError(
        "RM training requires pair-format samples containing chosen/rejected responses. "
        f"First sample from dataset '{dataset_name}' has keys: {sample_keys}. "
        "Please use pair data (e.g. a dataset with chosen_messages/rejected_messages, "
        "or set converter='pair' for raw chosen/rejected fields)."
    )


def _init_score_head(model: HFModel) -> None:
    """Initialize the score head for RM training with small Gaussian weights.

    Uses Gaussian initialization so that different parameters have distinct values,
    providing better gradient flow than zero initialization while keeping initial
    scores small enough that the starting loss is close to ln(2).
    """
    unwrapped = model.module if hasattr(model, "module") else model
    score = getattr(unwrapped, "score", None)
    if score is not None and hasattr(score, "weight"):
        hidden_size = score.weight.shape[-1]
        std = 1.0 / (hidden_size * 10)
        with torch.no_grad():
            score.weight.normal_(mean=0.0, std=std)
            if score.bias is not None:
                score.bias.zero_()
        logger.info_rank0(f"Initialized score head with Gaussian (std={std:.6f}): {score.weight.shape}")


class RMTrainer(BaseTrainer):
    def __init__(
        self,
        args: TrainingArguments,
        model: HFModel,
        renderer,
        train_dataset,
        callbacks=None,
        disable_dropout: bool = True,
    ) -> None:
        if args.cp_size > 1:
            raise NotImplementedError("RM trainer currently only supports cp_size == 1.")

        if disable_dropout:
            disable_dropout_in_model(model)

        super().__init__(args, model, renderer, train_dataset, callbacks)

    def _shard_model(self) -> None:
        if self.args.dist_config is None:
            if DistributedInterface().get_world_size(Dim.DP) > 1:
                from torch.nn.parallel import DistributedDataParallel as DDP

                device_ids = None if self.device.type == "cpu" else [self.device.index]
                self.model = DDP(self.model, device_ids=device_ids, find_unused_parameters=True)
        else:
            super()._shard_model()

    @property
    def _unwrapped_model(self):
        """Access the underlying model, unwrapping DDP/FSDP wrappers if present."""
        model = self.model
        if hasattr(model, "module"):
            model = model.module
        return model

    def compute_loss(self, batch: BatchInput) -> Tensor:
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)

        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is None:
            raise ValueError(
                "RM training requires pair data with token_type_ids. "
                "Ensure the dataset has chosen_messages/rejected_messages."
            )
        token_type_ids = token_type_ids.to(self.device, non_blocking=True)

        # Use token_type_ids as document-index attention mask (values: 1=chosen, 2=rejected, 0=padding).
        # Transformers v5 models natively support this format in _update_causal_mask,
        # constructing the correct block-diagonal causal mask internally for all attention backends.
        model_attention_mask = token_type_ids

        # Build position_ids that reset at each document boundary.
        batch_size, seq_len = token_type_ids.shape
        arange = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        chosen_mask = token_type_ids == 1
        rejected_mask = token_type_ids == 2
        chosen_lens = chosen_mask.sum(dim=1, keepdim=True)
        position_ids = torch.zeros_like(token_type_ids)
        position_ids[chosen_mask] = arange[chosen_mask]
        position_ids[rejected_mask] = (arange - chosen_lens)[rejected_mask]

        model_output = self.model(
            input_ids=input_ids,
            attention_mask=model_attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )

        rewards = model_output.logits.float().squeeze(-1)

        chosen_mask = token_type_ids == 1
        rejected_mask = token_type_ids == 2

        valid_pair_mask = chosen_mask.any(dim=-1) & rejected_mask.any(dim=-1)
        if not torch.any(valid_pair_mask):
            raise ValueError(
                "No valid RM pairs found in this micro-batch. "
                "This is usually caused by cutoff_len being too small and truncating chosen/rejected tokens."
            )

        rewards = rewards[valid_pair_mask]
        chosen_mask = chosen_mask[valid_pair_mask]
        rejected_mask = rejected_mask[valid_pair_mask]

        seq_len = rewards.size(-1)
        position_index = torch.arange(seq_len, device=self.device).unsqueeze(0)
        chosen_last_idx = (position_index * chosen_mask.long()).max(dim=-1).values
        rejected_last_idx = (position_index * rejected_mask.long()).max(dim=-1).values

        chosen_scores = rewards.gather(dim=1, index=chosen_last_idx.unsqueeze(-1)).squeeze(-1)
        rejected_scores = rewards.gather(dim=1, index=rejected_last_idx.unsqueeze(-1)).squeeze(-1)
        return -F.logsigmoid(chosen_scores - rejected_scores).mean()


def run_rm(args: InputArgument = None):
    model_args, data_args, training_args, _ = get_args(args)
    model_args.model_class = ModelClass.CLS
    DistributedInterface(training_args)
    train_dataset = DataEngine(data_args.train_dataset)
    _validate_rm_dataset_format(train_dataset, data_args.train_dataset)
    model_engine = ModelEngine(model_args, is_train=True)
    _init_score_head(model_engine.model)
    trainer = RMTrainer(
        args=training_args,
        model=model_engine.model,
        renderer=model_engine.renderer,
        train_dataset=train_dataset,
    )
    trainer.fit()
    trainer.save_model()
    DistributedInterface().destroy()


if __name__ == "__main__":
    run_rm()
