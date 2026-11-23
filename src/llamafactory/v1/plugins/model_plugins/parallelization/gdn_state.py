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

"""GDN state-routing CP using Attention Gym's distributed delta-rule recipe."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ....accelerator.helper import DeviceType, get_current_accelerator
from ....utils import logging
from ....utils.packages import is_transformers_version_greater_than
from .gdn_attention import _get_gdn_module, is_gdn_layer


if TYPE_CHECKING:
    from attn_gym.linear.context_parallel import ContextParallelRouting
    from torch import Tensor
    from torch.distributed import ProcessGroup
    from transformers import Cache

    from ....utils.types import HFModel


logger = logging.get_logger(__name__)


def _contiguous_fragments(
    batch_size: int, sequence_length: int, cp_size: int
) -> tuple[list[int], list[list[tuple[int, int]]]]:
    """Describe the existing contiguous CP split without exchanging token metadata."""
    if min(batch_size, sequence_length, cp_size) < 1 or sequence_length % cp_size:
        raise ValueError("State-routing CP requires a positive sequence length divisible by cp_size.")

    local_length = sequence_length // cp_size
    boundaries = [row * sequence_length for row in range(batch_size + 1)]
    fragments = [
        [
            (row * sequence_length + rank * local_length, row * sequence_length + (rank + 1) * local_length)
            for row in range(batch_size)
        ]
        for rank in range(cp_size)
    ]
    return boundaries, fragments


def _sequence_layout(
    position_ids: Tensor, attention_mask: Tensor | None, cp_size: int
) -> tuple[list[int], list[list[tuple[int, int]]], int, bool]:
    """Build global document boundaries and fixed CP fragments locally.

    Right-padding tails are separate zero-input sequences. Internal packed
    documents increase local subsequences, but not the number of communicated
    fragment summaries: each rank still owns one fragment per batch row.
    """
    if position_ids.ndim != 2 or position_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("State-routing CP requires 2-D integer text position_ids.")
    batch_size, sequence_length = position_ids.shape
    _, fragments = _contiguous_fragments(batch_size, sequence_length, cp_size)
    if attention_mask is None:
        valid = torch.ones_like(position_ids, dtype=torch.bool)
    else:
        if attention_mask.shape != position_ids.shape or not bool((attention_mask.eq(0) | attention_mask.eq(1)).all()):
            raise ValueError("State-routing CP requires a binary 2-D attention mask matching position_ids.")
        valid = attention_mask.bool()
        if bool((valid[:, 1:] & ~valid[:, :-1]).any()):
            raise ValueError("State-routing CP currently supports right padding only.")

    previous = position_ids[:, :-1]
    following = position_ids[:, 1:]
    valid_pairs = valid[:, :-1] & valid[:, 1:]
    resets = following.le(previous) & (following.eq(0) | following.eq(1))
    if bool((valid_pairs & ~(following.sub(previous).eq(1) | resets)).any()):
        raise ValueError("State-routing CP requires contiguous text positions between document resets.")

    starts = torch.ones_like(position_ids, dtype=torch.bool)
    starts[:, 1:] = (valid_pairs & resets) | (valid[:, :-1] & ~valid[:, 1:])
    boundaries = starts.reshape(-1).nonzero(as_tuple=True)[0].tolist()
    boundaries.append(batch_size * sequence_length)
    max_subsequences = max(
        sum(bisect_left(boundaries, stop) - bisect_right(boundaries, start) + 1 for start, stop in owned)
        for owned in fragments
    )
    return boundaries, fragments, max_subsequences, bool((valid_pairs & resets).any())


def _make_state_routing(
    *,
    cu_seqlens_global: list[int],
    fragments: list[list[tuple[int, int]]],
    cp_rank: int,
    device: torch.device,
    slots: int,
    max_subsequences: int,
    conv_history: int,
) -> ContextParallelRouting:
    from attn_gym.linear.context_parallel import ContextParallelRouting

    return ContextParallelRouting.from_fragments(
        cu_seqlens_global=cu_seqlens_global,
        fragments=fragments,
        cp_rank=cp_rank,
        device=device,
        slots=slots,
        max_subsequences=max_subsequences,
        conv_history=conv_history,
    )


def _prepare_state_inputs(
    _module: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    cp_size: int,
    cp_rank: int,
    conv_history: int,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Attach one immutable routing object before the language-boundary split.

    Global positions and masks are already replicated on CP ranks, so this does
    not exchange per-token metadata. Forward kwargs retain the routing object
    for gradient-checkpoint recomputation.
    """
    inputs = kwargs.get("inputs_embeds")
    if inputs is None:
        inputs = kwargs.get("input_ids")
    if inputs is None:
        raise ValueError("State-routing CP requires input_ids or inputs_embeds at the language boundary.")

    batch_size, sequence_length = inputs.shape[:2]
    position_ids = kwargs.get("position_ids")
    if position_ids is None or position_ids.shape != (batch_size, sequence_length):
        raise ValueError("State-routing CP currently requires explicit 2-D text position_ids.")
    attention_mask = kwargs.get("attention_mask")
    boundaries, fragments, max_subsequences, packed = _sequence_layout(position_ids, attention_mask, cp_size)
    if packed:
        if batch_size != 1:
            raise ValueError("Packed state-routing CP requires a flattened batch for the full-attention backend.")
        if attention_mask is not None:
            if not bool(attention_mask.eq(1).all()):
                raise ValueError("Packed state-routing CP cannot combine document resets with a padding mask.")
            # HF prioritizes mask-based unpadding over packed position IDs. An all-ones
            # mask is redundant and must not hide document boundaries from full attention.
            kwargs["attention_mask"] = None
    kwargs["gdn_cp_routing"] = _make_state_routing(
        cu_seqlens_global=boundaries,
        fragments=fragments,
        cp_rank=cp_rank,
        device=inputs.device,
        slots=batch_size,
        max_subsequences=max_subsequences,
        conv_history=conv_history,
    )
    return args, kwargs


def gdn_forward_with_state_cp(
    self: nn.Module,
    hidden_states: Tensor,
    cache_params: Cache | None = None,
    attention_mask: Tensor | None = None,
    *,
    gdn_cp_routing: ContextParallelRouting | None = None,
    **kwargs: Any,
) -> Tensor:
    """Compute GDN on local tokens with differentiable state and convolution-halo exchange."""
    from attn_gym.linear.context_parallel import context_parallel_conv_history
    from attn_gym.linear.gdn.context_parallel import context_parallel_gdn
    from attn_gym.linear.kda.fwd.triton.l2norm_fwd import l2norm
    from attn_gym.linear.short_conv import causal_conv1d

    if cache_params is not None:
        raise ValueError("State-routing CP is a training path and does not support generation caches.")
    if gdn_cp_routing is None:
        raise RuntimeError("State-routing metadata did not reach GDN from the language-boundary hook.")
    batch_size, local_length, _ = hidden_states.shape
    if gdn_cp_routing.tokens != batch_size * local_length:
        raise ValueError("State-routing metadata does not match the local GDN token span.")

    if attention_mask is not None:
        hidden_states = (hidden_states * attention_mask.unsqueeze(-1)).to(hidden_states.dtype)
    mixed_qkv = self.in_proj_qkv(hidden_states).reshape(1, batch_size * local_length, -1)
    if mixed_qkv.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("State-routing GDN CP requires FP16 or BF16 computation; enable bf16 training.")
    history = context_parallel_conv_history(mixed_qkv, gdn_cp_routing, self._gdn_state_cp_group)
    mixed_qkv = causal_conv1d(
        mixed_qkv,
        self.conv1d.weight.squeeze(1).to(mixed_qkv.dtype).contiguous(),
        activation=self.activation,
        cu_seqlens=gdn_cp_routing.cu_seqlens,
        initial_state=history,
    )
    query, key, value = mixed_qkv.split((self.key_dim, self.key_dim, self.value_dim), dim=-1)
    query = query.reshape(1, batch_size * local_length, self.num_k_heads, self.head_k_dim)
    key = key.reshape_as(query)
    value = value.reshape(1, batch_size * local_length, self.num_v_heads, self.head_v_dim)
    gate = -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)
    beta = self.in_proj_b(hidden_states).sigmoid()

    kernel_options = {"autotune": False}
    query = l2norm(query, cu_seqlens=gdn_cp_routing.cu_seqlens, kernel_options=kernel_options)
    key = l2norm(key, cu_seqlens=gdn_cp_routing.cu_seqlens, kernel_options=kernel_options)
    output, _ = context_parallel_gdn(
        query,
        key,
        value,
        gate.reshape(1, batch_size * local_length, self.num_v_heads),
        beta.reshape(1, batch_size * local_length, self.num_v_heads).float(),
        routing=gdn_cp_routing,
        group=self._gdn_state_cp_group,
        scale=self.head_k_dim**-0.5,
        kernel_options=kernel_options,
    )
    z = self.in_proj_z(hidden_states).reshape(-1, self.head_v_dim)
    output = self.norm(output.reshape(-1, self.head_v_dim), z)
    return self.out_proj(output.reshape(batch_size, local_length, self.value_dim))


def apply_gdn_state_attention(
    model: HFModel, cp_size: int, group: ProcessGroup | None, language_boundary: nn.Module
) -> None:
    """Install state routing for split-projection Qwen3.5 GDN layers.

    Text batches may be right padded or flattened padding-free packed. Unsupported
    layouts are rejected at the global boundary before any model collective.
    """
    if not is_transformers_version_greater_than("5.9.0"):
        raise RuntimeError("State-routing CP requires Transformers >= 5.9 to forward routing metadata to GDN.")
    from torch.distributed import _functional_collectives

    if not all(callable(getattr(module, "all_gather_single", None)) for module in (dist, _functional_collectives)):
        raise RuntimeError(
            "State-routing GDN CP requires PyTorch >= 2.13 with distributed and functional all_gather_single APIs."
        )
    if get_current_accelerator().type != DeviceType.CUDA:
        raise RuntimeError("State-routing GDN CP currently requires CUDA.")
    try:
        import attn_gym.linear.context_parallel  # noqa: F401
        import attn_gym.linear.gdn.context_parallel  # noqa: F401
        from attn_gym.linear.short_conv import causal_conv1d  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "State-routing GDN CP requires Attention Gym; install requirements/attention-gym.txt."
        ) from exc

    group = dist.group.WORLD if group is None else group
    if dist.get_world_size(group) != cp_size:
        raise ValueError("cp_size must match the state-routing process group.")

    gdn_modules: dict[int, tuple[str, nn.Module]] = {}
    for name, module in model.named_modules():
        if is_gdn_layer(module):
            gdn_module = _get_gdn_module(module)
            gdn_name = name if gdn_module is module else f"{name}.linear_attn"
            gdn_modules.setdefault(id(gdn_module), (gdn_name, gdn_module))
    if not gdn_modules:
        raise ValueError("State-routing GDN CP requires a model with GDN layers.")

    conv_sizes: set[int] = set()
    for name, module in gdn_modules.values():
        if not all(hasattr(module, attr) for attr in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b")):
            raise ValueError(f"State-routing CP requires split Q/K/V, z, a and b projections in {name}.")
        if module.head_k_dim != 128 or module.head_v_dim != 128:
            raise ValueError("The supported Attention Gym GDN kernel requires key and value dimensions of 128.")
        if module.activation != "silu" or module.conv1d.stride != (1,) or module.conv1d.dilation != (1,):
            raise ValueError("State-routing CP requires a SiLU depthwise convolution with unit stride and dilation.")
        if module.conv1d.groups != module.conv_dim:
            raise ValueError("State-routing CP requires a depthwise GDN convolution.")
        if module.conv1d.bias is not None:
            raise ValueError("State-routing CP currently requires a bias-free GDN convolution.")
        conv_sizes.add(module.conv_kernel_size)
    if len(conv_sizes) != 1:
        raise ValueError("State-routing CP currently requires the same convolution width in every GDN layer.")

    prepare_hook = partial(
        _prepare_state_inputs,
        cp_size=cp_size,
        cp_rank=dist.get_rank(group),
        conv_history=next(iter(conv_sizes)) - 1,
    )
    language_boundary.register_forward_pre_hook(prepare_hook, with_kwargs=True, prepend=True)
    for name, module in gdn_modules.values():
        module._gdn_state_cp_group = group
        module.forward = MethodType(gdn_forward_with_state_cp, module)
        logger.info_rank0(f"Installed state-routing GDN context parallelism in {name}.")
