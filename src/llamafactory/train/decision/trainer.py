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
from torch.distributed.fsdp import FSDPModule
from transformers.optimization import get_scheduler

from ..sft.trainer import CustomSeq2SeqTrainer


def decision_loss(logits, labels, weights, brier_weight=1.0):
    """Weighted distribution NLL plus expected multiclass Brier, in FP32."""
    log_probs = logits.float().log_softmax(-1)
    # Padded candidates have log(p)=-inf and target=0; avoid 0 * -inf.
    nll = -(labels * torch.where(labels > 0, log_probs, 0)).sum(-1)
    brier = (log_probs.exp() - labels).square().sum(-1) + 1 - labels.square().sum(-1)
    return ((nll + brier_weight * brier) * weights).mean()


class DecisionTrainer(CustomSeq2SeqTrainer):
    def __init__(self, **kwargs):
        kwargs["args"].remove_unused_columns = False
        kwargs["args"].label_names = ["labels"]
        super().__init__(**kwargs)
        # Our compute_loss consumes the global question count, not token counts.
        self.model_accepts_loss_kwargs = True

    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)
        accumulation = 1 if self.is_deepspeed_enabled else self.accelerator.gradient_accumulation_steps
        return loss / accumulation

    def _get_num_items_in_batch(self, batch_samples, device):
        count = sum(batch["question_weights"].count_nonzero() for batch in batch_samples)
        return self.accelerator.gather(torch.as_tensor(count, device=device)).sum().clamp_min(1)

    def create_optimizer(self, model=None):
        model = self.model if model is None else model
        if self.optimizer is None:
            groups = {}
            for name, parameter in model.named_parameters():
                name = name.replace("_fsdp_wrapped_module.", "")
                if not parameter.requires_grad:
                    continue
                router = ".attnres_routers." in name
                embedding = self.finetuning_args.decision_warmup and name == "language_model.embed_tokens.weight"
                ratio = 0.2 if embedding else 0.5 if router or name.startswith("language_model.") else 1.0
                decay = 0.0 if router or embedding else self.args.weight_decay
                group = groups.setdefault(
                    (ratio, decay), dict(params=[], lr=self.args.learning_rate * ratio, weight_decay=decay)
                )
                group["params"].append(parameter)
            optimizer_class, kwargs = self.get_optimizer_cls_and_kwargs(self.args, model)
            self.optimizer = optimizer_class(list(groups.values()), **kwargs)
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.args.lr_scheduler_type != "cosine_warmup_with_min_lr":
            return super().create_scheduler(num_training_steps, optimizer)
        if self.lr_scheduler is None:
            warmup = min(self.args.get_warmup_steps(num_training_steps), num_training_steps - 1)
            kwargs = dict(self.args.lr_scheduler_kwargs)
            if warmup == 1:
                kwargs["warmup_lr_rate"] = 1.0
            if not warmup:
                kwargs.pop("warmup_lr_rate", None)
            self.lr_scheduler = get_scheduler(
                "cosine_warmup_with_min_lr" if warmup else "cosine_with_min_lr",
                optimizer or self.optimizer,
                num_warmup_steps=warmup,
                num_training_steps=num_training_steps if warmup else max(1, num_training_steps - 1),
                scheduler_specific_kwargs=kwargs,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def _save(self, output_dir=None, state_dict=None):
        # Match the v1 FSDP2 exporter: remote-code auto_map must refer to GS1,
        # not PyTorch's dynamically generated FSDPGS1Model class.
        model_class = type(self.model)
        if isinstance(self.model, FSDPModule) and getattr(model_class, "_auto_class", None) is not None:
            self.model.__class__ = model_class.__bases__[1]
        try:
            super()._save(output_dir, state_dict=state_dict)
        finally:
            self.model.__class__ = model_class

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels, weights = inputs.pop("labels"), inputs.pop("question_weights")
        outputs = model(**inputs)  # Call the wrapped model to preserve DDP/Accelerate.
        loss = decision_loss(outputs.logits, labels, weights, self.finetuning_args.decision_brier_weight)
        if num_items_in_batch is not None:
            # Accelerate divides backward by accumulation steps; DeepSpeed's
            # Trainer path explicitly disables that scaling.
            accumulation = 1 if self.is_deepspeed_enabled else self.accelerator.gradient_accumulation_steps
            loss = loss * labels.shape[0] * self.args.world_size * accumulation / num_items_in_batch
        # Do not collect the full backbone hidden states during evaluation.
        return (loss, {"logits": outputs.logits}) if return_outputs else loss
