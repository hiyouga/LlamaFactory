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


from ..accelerator.interface import DistributedInterface
from ..config import InputArgument, TrainingArguments, get_args
from ..core.base_trainer import BaseTrainer
from ..core.data_engine import DataEngine
from ..core.model_engine import ModelEngine
from ..core.rendering import Renderer
from ..utils.callbacks import TrainerCallback
from ..utils.types import BatchInput, HFModel, Tensor, TorchDataset


class SFTTrainer(BaseTrainer):
    def __init__(
        self,
        args: TrainingArguments,
        model: HFModel,
        renderer: Renderer,
        train_dataset: TorchDataset,
        callbacks: list[TrainerCallback] | None = None,
    ) -> None:
        self._chunk_loss = None
        if args.chunk_loss_size is not None:
            from ..plugins.model_plugins.chunk_loss import LossPlugin

            self._chunk_loss = LossPlugin("chunk_loss")(model, args.chunk_loss_size)

        super().__init__(args, model, renderer, train_dataset, callbacks)

    def compute_loss(self, batch: BatchInput) -> Tensor:
        if self.cp_size > 1:
            from ..plugins.model_plugins.parallelization.sequence_parallel import SequenceParallelLossPlugin

            return SequenceParallelLossPlugin("sequence_parallel_loss")(
                self.model, batch, loss_fn=self._chunk_loss, uses_mrope=self._uses_mrope
            )
        if self._chunk_loss is not None:
            return self._chunk_loss.compute_loss(self.model, batch, device=self.device, uses_mrope=self._uses_mrope)

        shift_loss_weights = batch["loss_weights"].to(self.device, non_blocking=True)[..., 1:]
        log_probs = self.compute_log_probs(self.model, batch)
        loss = (-log_probs * shift_loss_weights).sum() / (shift_loss_weights.sum() + 1e-6)
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
