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


import copy
import os

import torch
import torch.nn.functional as F

from ..accelerator.interface import Dim, DistributedInterface
from ..config import InputArgument, TrainingArguments, get_args
from ..core.base_trainer import BaseTrainer
from ..core.data_engine import DataEngine
from ..core.model_engine import ModelEngine
from ..utils import logging
from ..utils.constants import IGNORE_INDEX
from ..utils.types import BatchInput, HFModel, Tensor


logger = logging.get_logger(__name__)


def compute_sigmoid_dpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    ref_chosen_logps: Tensor,
    ref_rejected_logps: Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tensor:
    r"""Standalone pure function for sigmoid DPO loss (Rafailov et al. 2023).

    .. math::
        \text{logits} = (\log\pi_\theta(y_c) - \log\pi_\text{ref}(y_c))
                      - (\log\pi_\theta(y_r) - \log\pi_\text{ref}(y_r))
        \mathcal{L} = -(1-\varepsilon)\log\sigma(\beta\cdot\text{logits})
                      - \varepsilon\log\sigma(-\beta\cdot\text{logits})

    Args:
        policy_chosen_logps: Log-probabilities from the policy model for chosen responses.
        policy_rejected_logps: Log-probabilities from the policy model for rejected responses.
        ref_chosen_logps: Log-probabilities from the reference model for chosen responses.
        ref_rejected_logps: Log-probabilities from the reference model for rejected responses.
        beta: Temperature / scaling factor for the DPO loss.
        label_smoothing: Label smoothing factor in [0, 1].

    Returns:
        Per-sample element-wise loss tensor.
    """
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    logits = chosen_logratios - rejected_logratios
    return -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing


def _validate_dpo_dataset_format(train_dataset: DataEngine, dataset_path: str) -> None:
    if train_dataset.streaming:
        return

    if len(train_dataset) == 0:
        raise ValueError(f"DPO training dataset is empty: {dataset_path}")

    sample = train_dataset[0]
    if "chosen_messages" in sample and "rejected_messages" in sample:
        return

    dataset_name = sample.get("_dataset_name", "unknown")
    sample_keys = sorted(sample.keys())
    raise ValueError(
        "DPO training requires pair-format samples containing chosen/rejected responses. "
        f"First sample from dataset '{dataset_name}' has keys: {sample_keys}. "
        "Please use pair data (e.g. a dataset with chosen_messages/rejected_messages)."
    )


class DPOTrainer(BaseTrainer):
    def __init__(
        self,
        args: TrainingArguments,
        model: HFModel,
        renderer,
        train_dataset,
        callbacks=None,
    ) -> None:
        if args.cp_size > 1:
            raise NotImplementedError("DPO trainer currently only supports cp_size == 1.")
        if args.chunk_loss_size is not None:
            raise NotImplementedError("Chunk Loss currently only supports SFT training.")

        self.pref_loss = args.pref_loss
        self.pref_beta = args.pref_beta
        self.pref_ftx = args.pref_ftx
        self.simpo_gamma = args.simpo_gamma
        self.ld_alpha = args.ld_alpha
        self.dpo_label_smoothing = args.dpo_label_smoothing

        # ref_model must be created AFTER super().__init__() because FSDP2 with
        # init_on_meta materialises the model during _shard_model().  We defer
        # creation to _init_ref_model() below.
        self.ref_model = None

        super().__init__(args, model, renderer, train_dataset, callbacks)

        if self.pref_loss == "sigmoid":
            self._init_ref_model()

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
        model = self.model
        if hasattr(model, "module"):
            model = model.module
        return model

    # ------------------------------------------------------------------
    # Reference model (frozen snapshot for sigmoid DPO)
    # ------------------------------------------------------------------

    @property
    def _use_lora_ref(self) -> bool:
        """Whether the policy model supports disable_adapter() for ref forward."""
        unwrapped = self._unwrapped_model
        return hasattr(unwrapped, "disable_adapter")

    def _init_ref_model(self) -> None:
        """Create a frozen copy of the initial model to serve as reference.

        For LoRA / PEFT models the base weights are already frozen, so we
        reuse the policy model with ``disable_adapter()`` instead of copying.
        For full fine-tuning a deep copy is required because the policy model's
        base weights change during training.

        Must be called AFTER super().__init__() so that FSDP2 / DDP sharding
        has materialised the model onto real devices.
        """
        if self._use_lora_ref:
            self.ref_model = None
            logger.info_rank0("LoRA detected — reference log-probs will reuse the base model via disable_adapter().")
            return

        unwrapped = self._unwrapped_model
        self.ref_model = copy.deepcopy(unwrapped)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad_(False)
        logger.info_rank0("Full fine-tuning — created independent reference model via deep copy.")

    # ------------------------------------------------------------------
    # Shared log-probability extraction from logits
    # ------------------------------------------------------------------

    def _extract_chosen_rejected_logps(
        self,
        logits: Tensor,
        labels: Tensor,
        token_type_ids: Tensor,
        use_ld: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Extract chosen / rejected log-probabilities (sum and average) from logits.

        Args:
            logits: (batch_size, seq_len, vocab_size)
            labels: (batch_size, seq_len)
            token_type_ids: (batch_size, seq_len) – 1=chosen, 2=rejected
            use_ld: Whether to apply LD-DPO length-dependent weighting. Should be
                ``False`` for the reference model to match the v0 behaviour where
                ``ld_alpha`` is only applied to the policy log-probs.

        Returns:
            chosen_logps:   (batch_size,) sum of per-token log-probs for chosen
            rejected_logps: (batch_size,) sum of per-token log-probs for rejected
            chosen_logps_avg:   (batch_size,) length-normalised chosen log-probs
            rejected_logps_avg: (batch_size,) length-normalised rejected log-probs
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_token_type_ids = token_type_ids[..., 1:]

        per_token_logps = -F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=IGNORE_INDEX,
        ).view(shift_labels.size(0), shift_labels.size(1))

        loss_mask = shift_labels != IGNORE_INDEX
        chosen_mask = (shift_token_type_ids == 1) & loss_mask
        rejected_mask = (shift_token_type_ids == 2) & loss_mask

        chosen_valid_len = chosen_mask.sum(dim=-1)
        rejected_valid_len = rejected_mask.sum(dim=-1)

        ld_alpha = self.ld_alpha if use_ld else None
        if ld_alpha is not None:
            min_lengths = torch.min(chosen_valid_len, rejected_valid_len)
            chosen_starts = torch.argmax(chosen_mask.int(), dim=1)
            rejected_starts = torch.argmax(rejected_mask.int(), dim=1)

            chosen_public_lengths = chosen_starts + min_lengths
            rejected_public_lengths = rejected_starts + min_lengths

            seq_len = shift_labels.size(1)
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

            chosen_ld_mask = position_ids < chosen_public_lengths.unsqueeze(1)
            rejected_ld_mask = position_ids < rejected_public_lengths.unsqueeze(1)

            chosen_front_mask = (chosen_ld_mask * chosen_mask).float()
            chosen_rear_mask = ((~chosen_ld_mask) * chosen_mask).float()
            rejected_front_mask = (rejected_ld_mask * rejected_mask).float()
            rejected_rear_mask = ((~rejected_ld_mask) * rejected_mask).float()

            chosen_logps = (per_token_logps * chosen_front_mask).sum(dim=-1) + ld_alpha * (
                per_token_logps * chosen_rear_mask
            ).sum(dim=-1)
            rejected_logps = (per_token_logps * rejected_front_mask).sum(dim=-1) + ld_alpha * (
                per_token_logps * rejected_rear_mask
            ).sum(dim=-1)
        else:
            chosen_logps = (per_token_logps * chosen_mask.float()).sum(dim=-1)
            rejected_logps = (per_token_logps * rejected_mask.float()).sum(dim=-1)

        chosen_logps_avg = chosen_logps / (chosen_valid_len + 1e-6)
        rejected_logps_avg = rejected_logps / (rejected_valid_len + 1e-6)

        return chosen_logps, rejected_logps, chosen_logps_avg, rejected_logps_avg

    # ------------------------------------------------------------------
    # Model inputs (block-diagonal attention + per-document position_ids)
    # ------------------------------------------------------------------

    def _prepare_model_inputs(self, input_ids: Tensor, token_type_ids: Tensor) -> dict[str, Tensor]:
        """Build model inputs with block-diagonal attention and per-document position IDs.

        In the v1 concatenated format each sample is::

            [chosen prompt | chosen response | rejected prompt | rejected response]

        with ``token_type_ids`` 1 / 2 marking the two documents.  A plain causal
        mask would let the rejected half attend to the chosen half and produce
        contiguous RoPE positions across the boundary, biasing the DPO objective.

        We instead:

        * pass ``token_type_ids`` as the attention mask so that Transformers v5
          builds a **block-diagonal** causal mask (each document only attends to
          itself — see :class:`RMTrainer` for the same pattern).
        * compute ``position_ids`` that **reset at each document boundary** so
          that every document gets its own RoPE positions starting from 0.
        """
        batch_size, seq_len = token_type_ids.shape
        arange = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)

        chosen_mask = token_type_ids == 1
        rejected_mask = token_type_ids == 2
        chosen_lens = chosen_mask.sum(dim=1, keepdim=True)

        position_ids = torch.zeros_like(token_type_ids)
        position_ids[chosen_mask] = arange[chosen_mask]
        position_ids[rejected_mask] = (arange - chosen_lens)[rejected_mask]

        return {
            "input_ids": input_ids,
            "attention_mask": token_type_ids,  # block-diagonal doc mask (v5)
            "position_ids": position_ids,
        }

    # ------------------------------------------------------------------
    # Reference log-probabilities (frozen model, no grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_ref_logps(self, batch: BatchInput) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward the frozen reference model and return chosen/rejected log-probs.

        For LoRA models the base weights are frozen, so we reuse the policy
        model with adapters disabled instead of maintaining a separate copy.
        """
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        token_type_ids = batch["token_type_ids"].to(self.device, non_blocking=True)

        model_inputs = self._prepare_model_inputs(input_ids, token_type_ids)

        if self._use_lora_ref:
            unwrapped = self._unwrapped_model
            with unwrapped.disable_adapter():
                ref_logits = unwrapped(**model_inputs, use_cache=False, return_dict=True).logits.float()
        else:
            ref_logits = self.ref_model(**model_inputs, use_cache=False, return_dict=True).logits.float()

        return self._extract_chosen_rejected_logps(ref_logits, labels, token_type_ids, use_ld=False)

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def _sigmoid_dpo_loss(
        self,
        policy_chosen_logps: Tensor,
        policy_rejected_logps: Tensor,
        ref_chosen_logps: Tensor,
        ref_rejected_logps: Tensor,
    ) -> Tensor:
        """Compute sigmoid DPO loss — delegates to :func:`compute_sigmoid_dpo_loss`."""
        return compute_sigmoid_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=self.pref_beta,
            label_smoothing=self.dpo_label_smoothing,
        )

    def _odds_ratio_loss(self, chosen_logps_avg: Tensor, rejected_logps_avg: Tensor) -> Tensor:
        log_odds = (chosen_logps_avg - rejected_logps_avg) - (
            torch.log1p(-torch.exp(chosen_logps_avg)) - torch.log1p(-torch.exp(rejected_logps_avg))
        )
        sft_loss = -chosen_logps_avg
        odds_ratio_loss = -F.logsigmoid(log_odds)
        return sft_loss + self.pref_beta * odds_ratio_loss

    def _simpo_loss(self, chosen_logps_avg: Tensor, rejected_logps_avg: Tensor) -> Tensor:
        pi_logratios = chosen_logps_avg - rejected_logps_avg
        gamma_logratios = self.simpo_gamma / self.pref_beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.pref_beta * logits)
        return simpo_loss

    # ------------------------------------------------------------------
    # Main compute_loss
    # ------------------------------------------------------------------

    def compute_loss(self, batch: BatchInput) -> Tensor:
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        token_type_ids = batch["token_type_ids"].to(self.device, non_blocking=True)

        # Block-diagonal attention (token_type_ids as doc mask) + per-document position_ids
        model_inputs = self._prepare_model_inputs(input_ids, token_type_ids)

        # --- Policy forward ---
        model_output = self.model(**model_inputs, use_cache=False, return_dict=True)
        logits = model_output.logits.float()

        # Split logits into chosen / rejected for metrics
        shift_logits = logits[..., :-1, :].contiguous()
        shift_token_type_ids = token_type_ids[..., 1:]
        chosen_logit_mask = (shift_token_type_ids == 1).float()
        rejected_logit_mask = (shift_token_type_ids == 2).float()

        policy_chosen_logps, policy_rejected_logps, chosen_logps_avg, rejected_logps_avg = (
            self._extract_chosen_rejected_logps(logits, labels, token_type_ids)
        )

        # Raw logits means (for logging)
        chosen_logits_mean = (shift_logits.mean(dim=-1) * chosen_logit_mask).sum() / (chosen_logit_mask.sum() + 1e-6)
        rejected_logits_mean = (shift_logits.mean(dim=-1) * rejected_logit_mask).sum() / (
            rejected_logit_mask.sum() + 1e-6
        )

        if self.pref_loss == "sigmoid":
            if not self._use_lora_ref and self.ref_model is None:
                raise RuntimeError(
                    "Reference model is required for sigmoid DPO loss but ref_model is None. "
                    "This should not happen; the ref model is created at __init__ for sigmoid loss."
                )

            ref_chosen_logps, ref_rejected_logps, _, _ = self._compute_ref_logps(batch)
            losses = self._sigmoid_dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
            )
            # DPO rewards: beta * (policy_logps - ref_logps)
            chosen_rewards = (self.pref_beta * (policy_chosen_logps - ref_chosen_logps)).detach()
            rejected_rewards = (self.pref_beta * (policy_rejected_logps - ref_rejected_logps)).detach()
        elif self.pref_loss == "orpo":
            losses = self._odds_ratio_loss(chosen_logps_avg, rejected_logps_avg)
            chosen_rewards = (self.pref_beta * chosen_logps_avg).detach()
            rejected_rewards = (self.pref_beta * rejected_logps_avg).detach()
        elif self.pref_loss == "simpo":
            losses = self._simpo_loss(chosen_logps_avg, rejected_logps_avg)
            chosen_rewards = (self.pref_beta * chosen_logps_avg).detach()
            rejected_rewards = (self.pref_beta * rejected_logps_avg).detach()
        else:
            raise ValueError(f"Unknown pref_loss: {self.pref_loss}")

        if self.pref_ftx > 1e-6:
            sft_loss = -chosen_logps_avg
            losses = losses + self.pref_ftx * sft_loss

        # --- Per-step DPO metrics (matches v0 logging) ---
        self._step_metrics = {
            "rewards/chosen": chosen_rewards.mean().item(),
            "rewards/rejected": rejected_rewards.mean().item(),
            "rewards/accuracies": (chosen_rewards > rejected_rewards).float().mean().item(),
            "rewards/margins": (chosen_rewards - rejected_rewards).mean().item(),
            "logps/chosen": policy_chosen_logps.mean().item(),
            "logps/rejected": policy_rejected_logps.mean().item(),
            "logits/chosen": chosen_logits_mean.item(),
            "logits/rejected": rejected_logits_mean.item(),
        }

        return losses.mean()


def run_dpo(args: InputArgument = None):
    model_args, data_args, training_args, _ = get_args(args)
    if getattr(training_args, "use_cpu", False):
        os.environ["FORCE_V1_CPU"] = "1"
    DistributedInterface(training_args)
    train_dataset = DataEngine(data_args.train_dataset)
    _validate_dpo_dataset_format(train_dataset, data_args.train_dataset)
    model_engine = ModelEngine(model_args, is_train=True)
    trainer = DPOTrainer(
        args=training_args,
        model=model_engine.model,
        renderer=model_engine.renderer,
        train_dataset=train_dataset,
    )
    trainer.fit()
    trainer.save_model()
    DistributedInterface().destroy()


if __name__ == "__main__":
    run_dpo()
