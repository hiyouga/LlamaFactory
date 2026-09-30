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

"""Chunked linear cross-entropy for SFT."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...utils.constants import IGNORE_INDEX
from ...utils.plugin import BasePlugin
from ...utils.types import BatchInput, HFModel, ModelOutput


class LossPlugin(BasePlugin):
    def __call__(self, model: HFModel, chunk_size: int) -> ChunkLoss:
        return super().__call__(model, chunk_size)


class _ChunkedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: Tensor,
        head_weight: Tensor,
        head_bias: Tensor | None,
        labels: Tensor,
        loss_weights: Tensor,
        chunk_size: int,
    ) -> Tensor:
        needs_hidden_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
        hidden_states_flat = hidden_states.reshape(-1, hidden_states.size(-1))
        labels_flat = labels.reshape(-1)
        loss_weights_flat = loss_weights.reshape(-1)

        loss = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
        grad_hidden = torch.empty_like(hidden_states_flat) if needs_hidden_grad else None
        # Avoid repeated BF16 rounding when summing head gradients across chunks.
        grad_weight = torch.zeros_like(head_weight, dtype=torch.float32) if needs_weight_grad else None
        grad_bias = (
            torch.zeros_like(head_bias, dtype=torch.float32) if head_bias is not None and needs_bias_grad else None
        )

        for start in range(0, hidden_states_flat.size(0), chunk_size):
            end = start + chunk_size
            with torch.enable_grad():
                hidden_arg = hidden_states_flat[start:end].detach().requires_grad_(needs_hidden_grad)
                weight_arg = head_weight.detach().requires_grad_(needs_weight_grad)
                bias_arg = head_bias.detach().requires_grad_(needs_bias_grad) if head_bias is not None else None
                logits = F.linear(hidden_arg, weight_arg, bias_arg).float()
                token_loss = F.cross_entropy(
                    logits,
                    labels_flat[start:end],
                    reduction="none",
                    ignore_index=IGNORE_INDEX,
                )
                chunk_loss = (token_loss * loss_weights_flat[start:end]).sum()
                grad_targets = [
                    tensor
                    for tensor, needed in (
                        (hidden_arg, needs_hidden_grad),
                        (weight_arg, needs_weight_grad),
                        (bias_arg, needs_bias_grad),
                    )
                    if tensor is not None and needed
                ]
                chunk_grads = torch.autograd.grad(chunk_loss, grad_targets) if grad_targets else ()

            loss.add_(chunk_loss.detach())
            grad_index = 0
            if grad_hidden is not None:
                grad_hidden[start:end].copy_(chunk_grads[grad_index])
                grad_index += 1
            if grad_weight is not None:
                grad_weight.add_(chunk_grads[grad_index])
                grad_index += 1
            if grad_bias is not None:
                grad_bias.add_(chunk_grads[grad_index])

        ctx.save_for_backward(
            grad_hidden.reshape_as(hidden_states) if grad_hidden is not None else None,
            grad_weight.to(head_weight.dtype) if grad_weight is not None else None,
            grad_bias.to(head_bias.dtype) if grad_bias is not None else None,
        )
        return loss

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_hidden, grad_weight, grad_bias = ctx.saved_tensors
        return (
            grad_hidden * grad_output if grad_hidden is not None else None,
            grad_weight * grad_output if grad_weight is not None else None,
            grad_bias * grad_output if grad_bias is not None else None,
            None,
            None,
            None,
        )


@dataclass
class _ChunkLossState:
    labels: Tensor
    loss_weights: Tensor
    loss: Tensor | None = None
    loss_version: int = 0


@LossPlugin("chunk_loss").register()
class ChunkLoss:
    """Install Chunk Loss before distributed wrapping.

    The model must call a plain Linear output head once and return its output
    directly as logits, without subsequent transformations. Other forwards keep
    their normal logits, including forwards of independently installed models.
    """

    def __init__(self, model: HFModel, chunk_size: int) -> None:
        self._output_head = model.get_output_embeddings()
        if type(self._output_head) is not nn.Linear:
            raise TypeError("Chunk Loss requires `get_output_embeddings()` to return a plain `torch.nn.Linear`.")

        self.chunk_size = chunk_size
        self._active_state: _ChunkLossState | None = None
        self._original_forward = self._output_head.forward
        self._output_head.forward = self._head_forward
        model.register_forward_hook(self._check_model_output)

    def __call__(
        self,
        model: HFModel,
        model_inputs: dict[str, Tensor],
        labels: Tensor,
        loss_weights: Tensor,
    ) -> Tensor:
        """Return a local weighted loss sum for already shifted targets."""
        state = _ChunkLossState(labels=labels, loss_weights=loss_weights)
        self._active_state = state
        try:
            outputs: ModelOutput = model(**model_inputs)
        finally:
            self._active_state = None

        if state.loss is None:
            raise RuntimeError("Chunk Loss did not reach the model output head.")
        # Use the outer output so distributed wrappers retain their backward hooks.
        return outputs.logits

    def compute_loss(
        self,
        model: HFModel,
        batch: BatchInput,
        *,
        device: torch.device,
        uses_mrope: bool,
    ) -> Tensor:
        """Prepare an unsharded SFT batch and compute its weighted mean Chunk Loss."""
        model_inputs = {
            key: value.to(device, non_blocking=True) for key, value in batch.items() if isinstance(value, torch.Tensor)
        }
        labels = model_inputs.pop("labels")
        loss_weights = model_inputs.pop("loss_weights")
        if uses_mrope:
            model_inputs.pop("position_ids", None)

        # Align each hidden state with its next-token target, as in the CP batch preparation.
        labels = F.pad(labels[..., 1:].contiguous(), (0, 1), value=IGNORE_INDEX)
        loss_weights = F.pad(loss_weights[..., 1:], (0, 1), value=0.0)
        numerator = self(model, model_inputs, labels, loss_weights)
        return numerator / (loss_weights.sum() + 1e-6)

    def _head_forward(self, hidden_states: Tensor) -> Tensor:
        state = self._active_state
        if state is None:
            return self._original_forward(hidden_states)
        if state.loss is not None:
            raise RuntimeError("Chunk Loss expects one output-head call per model forward.")
        if hidden_states.shape[:-1] != state.labels.shape or state.labels.shape != state.loss_weights.shape:
            raise ValueError(
                "Chunk Loss hidden states, labels, and loss weights must share the same token layout: "
                f"hidden_states={tuple(hidden_states.shape)}, labels={tuple(state.labels.shape)}, "
                f"loss_weights={tuple(state.loss_weights.shape)}."
            )

        state.loss = _ChunkedLinearCrossEntropy.apply(
            hidden_states,
            self._output_head.weight,
            self._output_head.bias,
            state.labels,
            state.loss_weights,
            self.chunk_size,
        )
        state.loss_version = state.loss._version
        return state.loss

    def _check_model_output(self, _model, _args, outputs: ModelOutput) -> None:
        state = self._active_state
        if state is None or state.loss is None:
            return
        # Validate before DDP and other wrappers can replace the output tensor.
        # The version counter also catches in-place logits transformations.
        if outputs.logits is not state.loss or state.loss._version != state.loss_version:
            raise NotImplementedError("Chunk Loss does not support transformations after the model output head.")
