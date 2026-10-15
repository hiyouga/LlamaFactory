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

from functools import wraps
from types import MethodType

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather

from ...plugins.model_plugins.parallelization.seq_comm import SeqAllToAll4D
from ...utils import logging


logger = logging.get_logger(__name__)


def validate_deepseek_v4_sequence_parallel_inputs(
    position_ids: torch.Tensor, attention_mask: torch.Tensor | None
) -> None:
    """Reject unsupported segment layouts before the global sequence is split across CP ranks."""
    if position_ids.ndim != 2:
        raise ValueError("DeepSeek V4 context parallelism requires 2D position_ids.")
    if attention_mask is not None:
        if attention_mask.ndim != 2 or attention_mask.shape[-1] != position_ids.shape[-1]:
            raise ValueError(
                "DeepSeek V4 context parallelism requires a binary 2D padding mask; custom masks are unsupported."
            )
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("DeepSeek V4 context parallelism requires a binary 2D padding mask.")
        valid = attention_mask.bool()
    else:
        valid = torch.ones_like(position_ids, dtype=torch.bool)

    # A single sequence has one position offset, ignoring ordinary left/right padding.
    offsets = position_ids - torch.arange(position_ids.shape[-1], device=position_ids.device)
    offsets, valid = torch.broadcast_tensors(offsets, valid)
    first_valid = valid.int().argmax(dim=-1, keepdim=True)
    if torch.any(valid & (offsets != offsets.gather(-1, first_valid))):
        raise ValueError(
            "DeepSeek V4 context parallelism requires contiguous positions on unmasked tokens; "
            "packed sequences and position resets are unsupported."
        )


class _DeepseekV4AttentionModuleProxy:
    """Override HF eager metadata after LF Ulysses has expanded KV and sharded heads.

    Keep the original module untouched for recomputation and other CP calls;
    the narrowed sinks remain a differentiable view of the original parameter.
    """

    def __init__(self, module, sinks: torch.Tensor, num_key_value_groups: int):
        self._module = module
        self.sinks = sinks
        self.num_key_value_groups = num_key_value_groups

    def __getattr__(self, name):
        return getattr(self._module, name)


def _expand_deepseek_v4_attention_mask(module, query, key, attention_mask, group):
    """Rebuild the ordinary causal sliding-window mask after sequence all-to-all.

    Ulysses exchanges Q/K/V but the model produced a pre-exchange local mask.
    This reconstruction does not preserve arbitrary custom/packed mask semantics.
    """
    cp_size = dist.get_world_size(group)
    full_query_length = query.shape[2]
    full_key_length = key.shape[2]
    if full_query_length % cp_size != 0:
        raise ValueError(f"DeepSeek V4 query length {full_query_length} is not divisible by CP size {cp_size}.")
    if full_key_length < full_query_length:
        raise ValueError("DeepSeek V4 compressed KV length cannot be shorter than its query length.")

    if attention_mask is not None and attention_mask.shape[-2:] == (full_query_length, full_key_length):
        return attention_mask

    local_query_length = full_query_length // cp_size
    batch_size = query.shape[0]
    if attention_mask is None:
        local_key_valid = torch.ones((batch_size, 1, local_query_length), device=query.device, dtype=torch.int32)
    else:
        if attention_mask.ndim != 4 or attention_mask.shape[-2] != local_query_length:
            raise ValueError(
                "DeepSeek V4 Ulysses expects a local 4D attention mask before the sequence exchange, "
                f"got shape {tuple(attention_mask.shape)}."
            )
        local_sliding_mask = attention_mask[..., :local_query_length]
        local_allowed = local_sliding_mask if local_sliding_mask.dtype == torch.bool else local_sliding_mask == 0
        local_key_valid = torch.diagonal(local_allowed, dim1=-2, dim2=-1).to(torch.int32)

    gathered_key_valid = [torch.empty_like(local_key_valid) for _ in range(cp_size)]
    dist.all_gather(gathered_key_valid, local_key_valid.contiguous(), group=group)
    key_valid = torch.cat(gathered_key_valid, dim=-1).bool()

    positions = torch.arange(full_query_length, device=query.device)
    query_positions = positions.view(-1, 1)
    key_positions = positions.view(1, -1)
    sliding_window = int(getattr(module, "sliding_window", module.config.sliding_window))
    allowed = (key_positions <= query_positions) & (key_positions > query_positions - sliding_window)
    allowed = allowed.view(1, 1, full_query_length, full_query_length) & key_valid.unsqueeze(-2)

    full_mask = torch.zeros(
        (batch_size, 1, full_query_length, full_key_length), device=query.device, dtype=query.dtype
    )
    full_mask[..., :full_query_length].masked_fill_(~allowed, torch.finfo(query.dtype).min)
    return full_mask


def _get_deepseek_v4_eager_attention_fn(fn, group):
    """Adapt the official eager callable to LF's post-Ulysses tensors."""

    @wraps(fn)
    def wrapper(module, query, key, value, attention_mask=None, *args, **kwargs):
        cp_size = dist.get_world_size(group)
        cp_rank = dist.get_rank(group)
        local_heads = query.shape[1]
        sinks = module.sinks
        if sinks.shape[0] == local_heads:
            local_sinks = sinks
        elif sinks.shape[0] == local_heads * cp_size:
            local_sinks = sinks.narrow(0, cp_rank * local_heads, local_heads)
        else:
            raise ValueError(
                f"DeepSeek V4 has {sinks.shape[0]} attention sinks, expected {local_heads} or "
                f"{local_heads * cp_size} for CP size {cp_size}."
            )

        if query.shape[1] % key.shape[1] != 0:
            raise ValueError("DeepSeek V4 query heads must be divisible by the post-Ulysses KV heads.")
        module_proxy = _DeepseekV4AttentionModuleProxy(
            module,
            sinks=local_sinks,
            num_key_value_groups=query.shape[1] // key.shape[1],
        )
        attention_mask = _expand_deepseek_v4_attention_mask(module, query, key, attention_mask, group)
        kwargs["s_aux"] = local_sinks
        return fn(module_proxy, query, key, value, attention_mask, *args, **kwargs)

    return wrapper


def _deepseek_v4_attention_forward(
    module, hidden_states, position_embeddings, position_ids, attention_mask, past_key_values=None, **kwargs
):
    """Run native V4 projections/compression around LF-owned Ulysses exchanges."""
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb, eager_attention_forward

    if past_key_values is not None:
        raise ValueError("DeepSeek V4 context parallelism requires past_key_values=None during training.")
    group = module._lf_cp_group
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    cos, sin = (
        position_embeddings[module.rope_layer_type] if isinstance(position_embeddings, dict) else position_embeddings
    )
    q_residual = module.q_a_norm(module.q_a_proj(hidden_states))
    query = module.q_b_norm(module.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2))
    query = apply_rotary_pos_emb(query, cos, sin)
    key = module.kv_norm(module.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
    key = apply_rotary_pos_emb(key, cos, sin)

    compressed_kv, block_bias = None, None
    if module.compressor is not None:
        global_hidden = torch.cat(all_gather(hidden_states.contiguous(), group=group), dim=1)
        global_residual = torch.cat(all_gather(q_residual.contiguous(), group=group), dim=1)
        positions = [torch.empty_like(position_ids) for _ in range(dist.get_world_size(group))]
        dist.all_gather(positions, position_ids.contiguous(), group=group)
        compressed = module.compressor(
            global_hidden, global_residual, torch.cat(positions, dim=-1), None, module.layer_idx
        )
        # HF 5.8 returns KV; newer releases additionally return per-query block bias.
        if isinstance(compressed, tuple):
            compressed_kv, block_bias = compressed
        else:
            compressed_kv = compressed

    query = SeqAllToAll4D.apply(group, query, 1, 2)
    key = key.repeat_interleave(module.config.num_attention_heads // key.shape[1], dim=1)
    key = SeqAllToAll4D.apply(group, key, 1, 2)
    attention_mask = _expand_deepseek_v4_attention_mask(module, query, key, attention_mask, group)
    if compressed_kv is not None:
        compressed_kv = compressed_kv.repeat_interleave(key.shape[1] // compressed_kv.shape[1], dim=1)
        key = torch.cat([key, compressed_kv], dim=2)
        if block_bias is None:
            block_bias = attention_mask.new_zeros((*attention_mask.shape[:-1], compressed_kv.shape[2]))
        attention_mask = torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)

    attention = _get_deepseek_v4_eager_attention_fn(eager_attention_forward, group)
    output, weights = attention(
        module,
        query,
        key,
        key,
        attention_mask,
        dropout=0.0 if not module.training else module.attention_dropout,
        scaling=module.scaling,
        **kwargs,
    )
    output = SeqAllToAll4D.apply(group, output.transpose(1, 2), 2, 1).transpose(1, 2)
    output = apply_rotary_pos_emb(output.transpose(1, 2), cos, -sin).transpose(1, 2)
    grouped = output.reshape(*input_shape, module.config.o_groups, -1)
    return module.o_b_proj(module.o_a_proj(grouped).flatten(2)), weights


def apply_deepseek_v4_sequence_parallel(model: torch.nn.Module, group: dist.ProcessGroup) -> None:
    """Install V4 Ulysses on model instances without importing FSDPTurbo or patching HF globals."""
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Attention

    modules = [module for module in model.modules() if isinstance(module, DeepseekV4Attention)]
    if not modules:
        raise ValueError("No DeepSeek V4 attention modules found for context parallelism.")
    cp_size = dist.get_world_size(group)
    for module in modules:
        if module.config._attn_implementation != "eager":
            raise ValueError("DeepSeek V4 context parallelism requires eager attention.")
        if module.config.num_attention_heads % cp_size:
            raise ValueError("DeepSeek V4 attention heads must be divisible by CP size.")
        if hasattr(module, "_lf_cp_group") and module._lf_cp_group is not group:
            raise ValueError("DeepSeek V4 context parallelism is already installed with a different group.")
    for module in modules:
        if not hasattr(module, "_lf_cp_group"):
            module._lf_cp_group = group
            module.forward = MethodType(_deepseek_v4_attention_forward, module)
    logger.info_rank0("Enabled DeepSeek V4 eager attention with LlamaFactory Ulysses context parallelism.")
