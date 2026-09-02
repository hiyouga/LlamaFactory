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

"""Memory-efficient chunked loss computation for V1 SFT training."""

from __future__ import annotations

import types
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate

from ...accelerator.interface import Dim, DistributedInterface
from ...utils.constants import IGNORE_INDEX
from ...utils.plugin import BasePlugin
from ...utils.types import BatchInput, HFModel


_SEQUENCE_MODEL_INPUT_NAMES = frozenset(
    {
        "attention_mask",
        "input_ids",
        "mm_token_type_ids",
        "position_ids",
        "token_type_ids",
    }
)
_UNSUPPORTED_CP_INPUT_NAMES = frozenset({"input_features", "pixel_values", "pixel_values_videos"})


class LossPlugin(BasePlugin):
    """Route a loss plugin name to its callback factory."""


class _ChunkedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: Tensor,
        head_weight: Tensor,
        head_bias: Tensor | None,
        labels: Tensor,
        loss_weights: Tensor,
        denominator: Tensor,
        chunk_size: int,
    ) -> Tensor:
        needs_hidden_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
        accumulated_loss = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
        grad_hidden = torch.empty_like(hidden_states) if needs_hidden_grad else None
        grad_weight = torch.zeros_like(head_weight, dtype=torch.float32) if needs_weight_grad else None
        grad_bias = (
            torch.zeros_like(head_bias, dtype=torch.float32) if head_bias is not None and needs_bias_grad else None
        )

        hidden_chunks = torch.split(hidden_states, chunk_size, dim=1)
        label_chunks = torch.split(labels, chunk_size, dim=1)
        loss_weight_chunks = torch.split(loss_weights, chunk_size, dim=1)
        grad_hidden_chunks = torch.split(grad_hidden, chunk_size, dim=1) if grad_hidden is not None else None

        for index, (hidden_chunk, label_chunk, loss_weight_chunk) in enumerate(
            zip(hidden_chunks, label_chunks, loss_weight_chunks, strict=True)
        ):
            with torch.enable_grad():
                hidden_arg = hidden_chunk.detach().requires_grad_(needs_hidden_grad)
                weight_arg = head_weight.detach().requires_grad_(needs_weight_grad)
                bias_arg = None
                if head_bias is not None:
                    bias_arg = head_bias.detach().requires_grad_(needs_bias_grad)

                logits = F.linear(hidden_arg, weight_arg, bias_arg).float()
                token_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    label_chunk.reshape(-1),
                    reduction="none",
                    ignore_index=IGNORE_INDEX,
                )
                chunk_loss = (token_loss * loss_weight_chunk.reshape(-1)).sum() / denominator

                grad_targets = []
                if needs_hidden_grad:
                    grad_targets.append(hidden_arg)
                if needs_weight_grad:
                    grad_targets.append(weight_arg)
                if bias_arg is not None and needs_bias_grad:
                    grad_targets.append(bias_arg)
                chunk_grads = torch.autograd.grad(chunk_loss, grad_targets) if grad_targets else []

            accumulated_loss.add_(chunk_loss.detach())
            grad_index = 0
            if grad_hidden_chunks is not None:
                grad_hidden_chunks[index].copy_(chunk_grads[grad_index])
                grad_index += 1
            if grad_weight is not None:
                grad_weight.add_(chunk_grads[grad_index])
                grad_index += 1
            if grad_bias is not None:
                grad_bias.add_(chunk_grads[grad_index])

        if grad_weight is not None:
            grad_weight = grad_weight.to(head_weight.dtype)
        if grad_bias is not None and head_bias is not None:
            grad_bias = grad_bias.to(head_bias.dtype)

        empty = torch.empty(0, device=hidden_states.device)
        ctx.save_for_backward(
            grad_hidden if grad_hidden is not None else empty,
            grad_weight if grad_weight is not None else empty,
            grad_bias if grad_bias is not None else empty,
        )
        ctx.has_hidden_grad = grad_hidden is not None
        ctx.has_weight_grad = grad_weight is not None
        ctx.has_bias_grad = grad_bias is not None
        return accumulated_loss

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_hidden, grad_weight, grad_bias = ctx.saved_tensors
        return (
            grad_hidden * grad_output if ctx.has_hidden_grad else None,
            grad_weight * grad_output if ctx.has_weight_grad else None,
            grad_bias * grad_output if ctx.has_bias_grad else None,
            None,
            None,
            None,
            None,
        )


def _chunked_linear_cross_entropy(
    hidden_states: Tensor,
    head_weight: Tensor,
    head_bias: Tensor | None,
    labels: Tensor,
    loss_weights: Tensor,
    chunk_size: int,
    denominator: Tensor | None = None,
) -> Tensor:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if hidden_states.shape[:2] != labels.shape or labels.shape != loss_weights.shape:
        raise ValueError(
            "Chunk loss expects hidden_states, labels and loss_weights to have matching batch/sequence dimensions."
        )

    if denominator is None:
        denominator = loss_weights.float().sum() + 1e-6
    elif denominator.ndim != 0:
        raise ValueError("Chunk loss denominator must be a scalar tensor.")

    return _ChunkedLinearCrossEntropyFunction.apply(
        hidden_states,
        head_weight,
        head_bias,
        labels,
        loss_weights,
        denominator,
        chunk_size,
    )


@dataclass
class _ChunkLossCallState:
    labels: Tensor
    loss_weights: Tensor
    denominator: Tensor
    output_head_called: bool = False


_ACTIVE_CHUNK_LOSS_CALL: ContextVar[_ChunkLossCallState | None] = ContextVar(
    "active_chunk_loss_call",
    default=None,
)


@contextmanager
def _activate_chunk_loss_call(call_state: _ChunkLossCallState) -> Iterator[None]:
    if _ACTIVE_CHUNK_LOSS_CALL.get() is not None:
        raise RuntimeError("Nested Chunk Loss output-head contexts are not supported.")

    token = _ACTIVE_CHUNK_LOSS_CALL.set(call_state)
    try:
        yield
    finally:
        _ACTIVE_CHUNK_LOSS_CALL.reset(token)


def _to_local_tensor(tensor: Tensor | None, parameter_name: str) -> Tensor | None:
    if tensor is None:
        return None

    if not isinstance(tensor, DTensor):
        return tensor
    if not all(isinstance(placement, Replicate) for placement in tensor.placements):
        raise NotImplementedError(
            f"Chunk Loss requires an unsharded {parameter_name} DTensor during lm_head forward, "
            f"but found placements {tensor.placements}. Vocab/tensor-parallel output heads require "
            "a distributed softmax implementation."
        )

    local_tensor = tensor.to_local()
    if local_tensor.shape != tensor.shape:
        raise RuntimeError(
            f"The local {parameter_name} shape {tuple(local_tensor.shape)} does not match its global DTensor "
            f"shape {tuple(tensor.shape)}."
        )

    return local_tensor


def _get_local_output_head_parameters(output_head: nn.Linear) -> tuple[Tensor, Tensor | None]:
    weight = _to_local_tensor(output_head.weight, "lm_head.weight")
    bias = _to_local_tensor(output_head.bias, "lm_head.bias")
    assert weight is not None
    return weight, bias


def _install_chunk_loss_hook(model: HFModel, chunk_size: int) -> None:
    """Install the output-head interceptor before distributed model wrapping."""
    if chunk_size <= 0:
        raise ValueError("`chunk_loss_size` must be positive when chunk loss is enabled.")

    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    output_head = get_output_embeddings() if callable(get_output_embeddings) else None
    if not isinstance(output_head, nn.Linear):
        raise TypeError("Chunk Loss currently requires get_output_embeddings() to return torch.nn.Linear.")

    if getattr(output_head, "_llamafactory_chunk_loss_enabled", False):
        return

    output_head._original_forward = output_head.forward

    def intercepted_forward(self, hidden_states: Tensor, *args, **kwargs):
        call_state = _ACTIVE_CHUNK_LOSS_CALL.get()
        if call_state is None:
            return self._original_forward(hidden_states, *args, **kwargs)

        if call_state.output_head_called:
            raise RuntimeError("SFT Chunk Loss does not support more than one output head call per model forward.")
        if args or kwargs:
            raise TypeError("SFT Chunk Loss does not support extra lm_head forward arguments.")

        if hidden_states.ndim != 3 or hidden_states.shape[:2] != call_state.labels.shape:
            raise ValueError(
                "The lm_head hidden-state shape does not match the SFT targets. "
                "This model may slice the sequence before its output head and is not supported by SFT Chunk Loss."
            )

        head_weight, head_bias = _get_local_output_head_parameters(self)
        loss = _chunked_linear_cross_entropy(
            hidden_states=hidden_states,
            head_weight=head_weight,
            head_bias=head_bias,
            labels=call_state.labels,
            loss_weights=call_state.loss_weights,
            chunk_size=chunk_size,
            denominator=call_state.denominator,
        )
        call_state.output_head_called = True
        return loss

    output_head.forward = types.MethodType(intercepted_forward, output_head)
    output_head._llamafactory_chunk_loss_enabled = True


@dataclass(frozen=True)
class _ChunkLossBatch:
    """Model inputs and causal targets prepared for one Chunk Loss forward."""

    model_inputs: BatchInput
    labels: Tensor
    loss_weights: Tensor
    denominator: Tensor


def _prepare_chunk_loss_batch(
    batch: BatchInput,
    *,
    device: torch.device,
    uses_mrope: bool,
    shard_sequence: Callable[[Tensor, float | int], Tensor] | None = None,
) -> _ChunkLossBatch:
    """Move model inputs to the device and align causal targets for Chunk Loss."""
    model_inputs: BatchInput = {
        key: value.to(device, non_blocking=True) for key, value in batch.items() if isinstance(value, torch.Tensor)
    }
    labels = model_inputs.pop("labels")
    loss_weights = model_inputs.pop("loss_weights")
    sequence_length = labels.shape[-1]

    if uses_mrope and shard_sequence is None:
        model_inputs.pop("position_ids", None)

    labels = F.pad(labels[..., 1:].contiguous(), (0, 1), value=IGNORE_INDEX)
    loss_weights = F.pad(loss_weights[..., 1:], (0, 1), value=0.0)
    denominator = loss_weights.float().sum() + 1e-6

    if shard_sequence is not None:
        sharded_inputs: BatchInput = {}
        for key, value in model_inputs.items():
            if key in _SEQUENCE_MODEL_INPUT_NAMES:
                sharded_inputs[key] = shard_sequence(value, 0)
            else:
                if value.ndim > 1 and value.shape[-1] == sequence_length:
                    raise ValueError(
                        f"Model input {key!r} looks sequence-aligned but is not classified for context "
                        "parallelism. Add it to the explicit sequence input contract."
                    )
                sharded_inputs[key] = value

        model_inputs = sharded_inputs
        labels = shard_sequence(labels, IGNORE_INDEX)
        loss_weights = shard_sequence(loss_weights, 0.0)

    return _ChunkLossBatch(
        model_inputs=model_inputs,
        labels=labels,
        loss_weights=loss_weights,
        denominator=denominator,
    )


def _forward_chunk_loss(
    model: HFModel,
    prepared_batch: _ChunkLossBatch,
) -> Tensor:
    """Run the model once while the output head computes Chunk Loss."""
    call_state = _ChunkLossCallState(
        labels=prepared_batch.labels,
        loss_weights=prepared_batch.loss_weights,
        denominator=prepared_batch.denominator,
    )
    with _activate_chunk_loss_call(call_state):
        outputs = model(**prepared_batch.model_inputs)

    if not call_state.output_head_called:
        raise RuntimeError("SFT Chunk Loss did not observe a call to the patched output head.")

    loss = outputs.logits
    if loss.ndim != 0:
        raise RuntimeError(
            "SFT Chunk Loss expected the model to return the scalar output-head loss as outputs.logits, "
            f"but found shape {tuple(loss.shape)}. This model may consume or reshape logits after the output head "
            "and is not supported by the standard Chunk Loss implementation."
        )

    return loss


def _compute_chunk_loss(
    model: HFModel,
    batch: BatchInput,
    *,
    device: torch.device,
    uses_mrope: bool,
) -> Tensor:
    """Prepare and compute Chunk Loss, including its optional CP path."""
    distributed = DistributedInterface()
    cp_size = distributed.get_world_size(Dim.CP)
    shard_sequence = None
    cp_group = None

    if cp_size > 1:
        unsupported_inputs = sorted(name for name in _UNSUPPORTED_CP_INPUT_NAMES if batch.get(name) is not None)
        if unsupported_inputs:
            raise NotImplementedError(
                "SFT Chunk Loss with context parallelism does not currently support multimodal inputs "
                f"(found: {', '.join(unsupported_inputs)}). Disable context parallelism or Chunk Loss."
            )

        cp_rank = distributed.get_rank(Dim.CP)
        cp_group = distributed.get_group(Dim.CP)
        if cp_group is None:
            raise RuntimeError("Chunk Loss with context parallelism requires an initialized CP process group.")

        sequence_length = batch["labels"].shape[-1]
        local_length = torch.tensor(sequence_length, device=device, dtype=torch.int64)
        group_lengths = [torch.empty_like(local_length) for _ in range(cp_size)]
        dist.all_gather(group_lengths, local_length, group=cp_group)
        lengths = [int(length.item()) for length in group_lengths]
        if any(length != sequence_length for length in lengths):
            raise RuntimeError(
                f"Context-parallel ranks must receive the same batch sequence length, but found lengths {lengths}."
            )

        padded_length = sequence_length + (-sequence_length % cp_size)
        shard_length = padded_length // cp_size
        start = cp_rank * shard_length
        end = start + shard_length

        def shard_sequence_fn(tensor: Tensor, pad_value: float | int) -> Tensor:
            if tensor.ndim == 0 or tensor.shape[-1] != sequence_length:
                raise ValueError(
                    f"Sequence tensor must end in dimension {sequence_length}, but found shape {tuple(tensor.shape)}."
                )

            padded = F.pad(tensor, (0, padded_length - sequence_length), value=pad_value)
            return padded[..., start:end].contiguous()

        shard_sequence = shard_sequence_fn

    prepared_batch = _prepare_chunk_loss_batch(
        batch,
        device=device,
        uses_mrope=uses_mrope,
        shard_sequence=shard_sequence,
    )
    local_loss = _forward_chunk_loss(model, prepared_batch)
    if cp_size == 1:
        return local_loss

    gathered_losses = dist.nn.all_gather(local_loss.reshape(1), group=cp_group)
    return torch.cat(gathered_losses).sum()


@LossPlugin("chunk_loss").register()
def build_chunk_loss(
    model: HFModel,
    *,
    chunk_size: int,
    device: torch.device,
    uses_mrope: bool,
) -> Callable[[HFModel, BatchInput], Tensor]:
    """Install Chunk Loss and return its SFT loss callback."""
    _install_chunk_loss_hook(model, chunk_size)
    return partial(_compute_chunk_loss, device=device, uses_mrope=uses_mrope)
