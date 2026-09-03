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

from ..accelerator.interface import DistributedInterface
from ..config import InputArgument, get_args
from ..core.base_trainer import BaseTrainer
from ..core.data_engine import DataEngine
from ..core.model_engine import ModelEngine
from ..utils.types import BatchInput, Tensor


class SFTTrainer(BaseTrainer):
    def compute_loss(self, batch: BatchInput) -> Tensor:
        if self._has_mtp():
            return self._compute_mtp_loss(batch)

        shift_loss_weights = batch["loss_weights"].to(self.device, non_blocking=True)[..., 1:]
        log_probs = self.compute_log_probs(self.model, batch)
        loss = (-log_probs * shift_loss_weights).sum() / (shift_loss_weights.sum() + 1e-6)
        return loss

    def _mtp_loss_scale(self) -> float:
        model = self.model.module if hasattr(self.model, "module") else self.model
        return float(getattr(model.config, "mtp_loss_scaling_factor", 0.3))

    def _compute_mtp_loss(self, batch: BatchInput) -> Tensor:
        """Main SFT loss plus the scaled MTP loss (non context-parallel path)."""
        from ..plugins.model_plugins.mtp import compute_mtp_loss

        batch_size, _ = batch["labels"].shape
        model_inputs = {
            k: v.to(self.device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        labels = batch["labels"].to(self.device, non_blocking=True)
        loss_weights = batch["loss_weights"].to(self.device, non_blocking=True)

        outputs = self.model(**model_inputs)

        # Main head: weighted cross-entropy, same as `compute_log_probs`.
        logits = outputs.logits.float()
        shift_labels = labels[..., 1:].contiguous().view(-1)
        shift_logits = logits[..., :-1, :].contiguous().view(shift_labels.size(0), -1)
        log_probs = -F.cross_entropy(shift_logits, shift_labels, reduction="none").view(batch_size, -1)
        shift_loss_weights = loss_weights[..., 1:]
        loss = (-log_probs * shift_loss_weights).sum() / (shift_loss_weights.sum() + 1e-6)

        # MTP heads: averaged per-head loss, scaled by `mtp_loss_scaling_factor`.
        mtp_logits = getattr(outputs, "mtp_logits", None)
        if mtp_logits:
            mtp_loss = compute_mtp_loss(mtp_logits, labels, loss_weights)
            loss = loss + mtp_loss * self._mtp_loss_scale()
            # Expose the unscaled per-head-mean MTP loss for logging (the main `loss` above
            # already includes the scaled contribution). Read back by BaseTrainer.fit.
            self.model._last_mtp_loss = float(mtp_loss.detach().item())

        return loss


def run_sft(args: InputArgument = None):
    model_args, data_args, training_args, _ = get_args(args)
    DistributedInterface(training_args)
    train_dataset = DataEngine(data_args.train_dataset)
    model_engine = ModelEngine(model_args, is_train=True)
    trainer = SFTTrainer(
        args=training_args,
        model=model_engine.model,
        renderer=model_engine.renderer,
        train_dataset=train_dataset,
    )
    trainer.fit()
    trainer.save_model()
    DistributedInterface().destroy()


if __name__ == "__main__":
    """
    python -m llamafactory.v1.trainers.sft_trainer --model Qwen/Qwen3-0.6B --train_dataset data/v1_sft_demo.yaml
    """
    run_sft()
