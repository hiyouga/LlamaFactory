# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
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
import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Literal, Optional, Union

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from trl.models.utils import prepare_deepspeed, prepare_fsdp
from trl.trainer import disable_dropout_in_model
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import (
    create_custom_optimizer,
    create_custom_scheduler,
    get_batch_logps,
    nested_detach,
    switch_ref_adapter_context,
)


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.f_divergence_type = "reverse_kl"
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.bco_gemma = finetuning_args.pref_bco_weight
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma
        self.ld_alpha = finetuning_args.ld_alpha

        Trainer.__init__(self, model=model, **kwargs)
        self.model_accepts_loss_kwargs = False  # overwrite trainer's default behavior
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                if self.accelerator.is_fsdp2:
                    from accelerate.utils.fsdp_utils import fsdp2_prepare_model

                    self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model)
                else:
                    self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if self.bco_gemma >= 1e-6:
            from trl.trainer import RunningMoments

            self.running = RunningMoments(self.accelerator)

    @override
    def create_optimizer(self, *args, **kwargs) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer(*args, **kwargs)

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def get_batch_samples(self, *args, **kwargs):
        r"""Replace the method of DPO Trainer with the one of the standard Trainer."""
        return Trainer.get_batch_samples(self, *args, **kwargs)

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""Compute ORPO's odds ratio (OR) loss for batched log probabilities of the policy model."""
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""Compute SimPO loss for batched log probabilities of the policy model."""
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.beta * logits)
        return simpo_loss

    def bco_loss(
        self,
        chosen_logps: "torch.Tensor",
        rejected_logps: "torch.Tensor",
        reference_chosen_logps: "torch.Tensor",
        reference_rejected_logps: "torch.Tensor",
    ) -> "torch.Tensor":
        chosen_logratios = chosen_logps - reference_chosen_logps
        rejected_logratios = rejected_logps - reference_rejected_logps
        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios
        rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
        self.running.update(rewards)  # update baseline
        delta = self.running.mean
        bco_loss = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
            -(self.beta * rejected_logratios - delta)
        )
        return bco_loss

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""Compute loss for preference learning."""
        if not self.finetuning_args.use_ref_model:
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
            else:
                raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")

            chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        else:
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            )

            if self.bco_gemma > 1e-6:
                bco_losses = self.bco_loss(
                    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
                )
                losses = (losses + bco_losses * self.bco_gemma) / (1.0 + self.bco_gemma)  # re-weight W_p and W_q

        return losses, chosen_rewards, rejected_rewards

    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"], is_ref_model: bool = False
    ) -> dict[str, "torch.Tensor"]:
        r"""Compute the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

        Otherwise the average log probabilities.
        """
        if self.finetuning_args.use_ref_model:
            batch = nested_detach(batch, clone=True)  # avoid error

        labels = batch.pop("labels")  # dpo do not need compute loss in forward
        all_logits: torch.Tensor = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        all_logps, valid_length = get_batch_logps(
            logits=all_logits, labels=labels, ld_alpha=(self.ld_alpha if not is_ref_model else None)
        )
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            chosen_logps_avg = chosen_logps
        else:
            chosen_logps_avg = chosen_logps / chosen_length

        return {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "chosen_logits": chosen_logits,
            "rejected_logits": rejected_logits,
            "chosen_logps_avg": chosen_logps_avg,
        }

    @override
    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"]
    ) -> tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Compute log probabilities of the reference model."""
        if not self.finetuning_args.use_ref_model:
            return None, None

        if self.ref_model is None:
            ref_model = model
            if self.finetuning_args.share_ref_base:
                ref_context = switch_ref_adapter_context(self.accelerator.unwrap_model(model))
            else:
                ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:
            ref_output = self.concatenated_forward(ref_model, batch, is_ref_model=True)
            reference_chosen_logps = ref_output["chosen_logps"]
            reference_rejected_logps = ref_output["rejected_logps"]

        return reference_chosen_logps, reference_rejected_logps

    @override
    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        r"""Only save the 'default' (policy) adapter when share_ref_base is enabled."""
        if self.finetuning_args.share_ref_base:
            from peft import PeftModel

            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            unwrapped = self.accelerator.unwrap_model(self.model)
            if isinstance(unwrapped, PeftModel):
                # Under FSDP, model.state_dict() is a collective op that hangs if called
                # only on rank 0. Use the pre-gathered state_dict from the trainer if available,
                # otherwise call state_dict() (safe when all ranks enter _save together under FSDP).
                if state_dict is None:
                    state_dict = self.model.state_dict()

                unwrapped.save_pretrained(
                    output_dir,
                    selected_adapters=["default"],
                    safe_serialization=getattr(self.args, "save_safetensors", True),
                    state_dict=state_dict,
                )
                if self.processing_class is not None:
                    self.processing_class.save_pretrained(output_dir)
                return

        super()._save(output_dir, state_dict)

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
        r"""Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        model_output = self.concatenated_forward(model, batch)
        policy_chosen_logps = model_output["chosen_logps"]
        policy_rejected_logps = model_output["rejected_logps"]
        policy_chosen_logits = model_output["chosen_logits"]
        policy_rejected_logits = model_output["rejected_logits"]
        policy_chosen_logps_avg = model_output["chosen_logps_avg"]

        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()
        if self.loss_type == "orpo":
            metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
            metrics[f"{prefix}odds_ratio_loss"] = ((losses - sft_loss) / self.beta).mean().item()

        return losses.mean(), metrics

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", tuple["torch.Tensor", list["torch.Tensor"]]]:
        r"""Subclass and override to accept extra kwargs."""
        return super().compute_loss(model, inputs, return_outputs)

    @override
    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        r"""Log `logs` on the various objects watching training, including stored metrics."""
        # logs either has "loss" or "eval_loss"
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        key_list, metric_list = [], []
        for key, metrics in self._stored_metrics[train_eval].items():
            key_list.append(key)
            metric_list.append(torch.tensor(metrics, dtype=torch.float).to(self.accelerator.device).mean().item())

        del self._stored_metrics[train_eval]
        if len(metric_list) < 10:  # pad to for all reduce
            for i in range(10 - len(metric_list)):
                key_list.append(f"dummy_{i}")
                metric_list.append(0.0)

        metric_list = torch.tensor(metric_list, dtype=torch.float).to(self.accelerator.device)
        metric_list = self.accelerator.reduce(metric_list, "mean").tolist()
        for key, metric in zip(key_list, metric_list):  # add remaining items
            if not key.startswith("dummy_"):
                logs[key] = metric

        return Trainer.log(self, logs, *args, **kwargs)
