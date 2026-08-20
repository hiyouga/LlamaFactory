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
import torch.distributed as dist
import torch.nn.functional as F

from .seq_comm import SeqAllToAll4D
from .ulysses import (
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
)


def is_gdn_layer(layer) -> bool:
    """Return True if the module is a GDN (linear attention) layer or a DecoderLayer containing one."""
    if hasattr(layer, "layer_type") and layer.layer_type == "linear_attention":
        return True
    if hasattr(layer, "block_type") and layer.block_type == "linear_attention":
        return True
    return False


def _get_gdn_module(module):
    """Return the actual GDN module from either a GDN layer or a DecoderLayer."""
    if hasattr(module, "in_proj_qkv"):
        return module
    if hasattr(module, "linear_attn"):
        return module.linear_attn
    raise AttributeError(
        f"Cannot find GDN module on {type(module).__name__}. "
        f"Expected either a GDN layer with in_proj_qkv or a DecoderLayer with linear_attn."
    )


def get_parameter_local_cp(param, dim, cp_group, split_sections=None):
    """Slice a parameter for the current CP rank.

    If split_sections is given, first split along dim into sub-groups,
    slice each sub-group independently for CP, then concatenate back.
    This ensures each CP rank gets a proportional slice of each sub-group.
    """
    cp_size = dist.get_world_size(group=cp_group)
    if cp_size == 1:
        return param
    cp_rank = dist.get_rank(group=cp_group)

    if split_sections is not None:
        inputs = torch.split(param, split_sections, dim=dim)
        outputs = []
        for p in inputs:
            p = get_parameter_local_cp(p, dim, cp_group)
            outputs.append(p)
        return torch.cat(outputs, dim=dim)

    slices = [slice(None)] * param.dim()
    dim_size = param.size(dim=dim)
    slices[dim] = slice(cp_rank * dim_size // cp_size, (cp_rank + 1) * dim_size // cp_size)
    return param[slices]


def gdn_forward_with_cp(self, hidden_states, attention_mask=None, **kwargs):
    """GDN forward with Context Parallel support.

    Uses SeqAllToAll4D (same as UlyssesAttention) for all all_to_all operations.
    Each component (Q/K/V/z/b/a) is independently reshaped to 4D and all_to_all'd
    with scatter heads / gather seq, avoiding the bug where uniform hidden-split on
    merged qkv gives rank-0 [Q+K] and rank-1 [V].

    Falls back to self.original_forward when cp_size <= 1.
    """
    cp_size = get_ulysses_sequence_parallel_world_size()
    if cp_size <= 1:
        return self.original_forward(hidden_states, attention_mask=attention_mask, **kwargs)

    cp_group = get_ulysses_sequence_parallel_group()

    if attention_mask is not None:
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

            hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        except ImportError:
            pass

    batch_size, seq_len, _ = hidden_states.shape
    full_seq_len = seq_len * cp_size

    # Extract position_ids and derive cu_seqlens for pack support
    position_ids = kwargs.get("position_ids", None)
    cu_seqlens = None
    if position_ids is not None and batch_size == 1:
        global_position_ids = [torch.empty_like(position_ids) for _ in range(cp_size)]
        dist.all_gather(global_position_ids, position_ids, group=cp_group)
        global_position_ids = torch.cat(global_position_ids, dim=-1).contiguous()
        try:
            from transformers.modeling_flash_attention_utils import prepare_fa_kwargs_from_position_ids
            cu_seqlens = prepare_fa_kwargs_from_position_ids(global_position_ids)[0][0]
        except ImportError:
            cu_seqlens = None

    # Input projections in CP layout: [B, seq/cp, hidden]
    qkv = self.in_proj_qkv(hidden_states)  # [B, seq/cp, key_dim*2 + value_dim]
    z = self.in_proj_z(hidden_states)  # [B, seq/cp, value_dim]
    b = self.in_proj_b(hidden_states)  # [B, seq/cp, num_v_heads]
    a = self.in_proj_a(hidden_states)  # [B, seq/cp, num_v_heads]

    # Split qkv into Q, K, V before all_to_all so each sub-group gets
    # proportional head distribution across ranks.
    q_proj, k_proj, v_proj = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

    # CP->HP all_to_all for each component: scatter heads (dim=2), gather seq (dim=1)
    # [B, S/cp, heads, head_dim] -> [B, S, heads/cp, head_dim]
    q_proj = q_proj.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
    q_proj = SeqAllToAll4D.apply(cp_group, q_proj, 2, 1)

    k_proj = k_proj.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
    k_proj = SeqAllToAll4D.apply(cp_group, k_proj, 2, 1)

    v_proj = v_proj.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
    v_proj = SeqAllToAll4D.apply(cp_group, v_proj, 2, 1)

    z = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
    z = SeqAllToAll4D.apply(cp_group, z, 2, 1)

    b = b.reshape(batch_size, seq_len, self.num_v_heads, 1)
    b = SeqAllToAll4D.apply(cp_group, b, 2, 1)

    a = a.reshape(batch_size, seq_len, self.num_v_heads, 1)
    a = SeqAllToAll4D.apply(cp_group, a, 2, 1)

    # Merge Q/K/V for conv1d (conv1d requires merged qkv)
    q_flat = q_proj.reshape(batch_size, full_seq_len, self.key_dim // cp_size)
    k_flat = k_proj.reshape(batch_size, full_seq_len, self.key_dim // cp_size)
    v_flat = v_proj.reshape(batch_size, full_seq_len, self.value_dim // cp_size)
    qkv = torch.cat([q_flat, k_flat, v_flat], dim=-1)  # [B, S, (key_dim*2+value_dim)/cp]

    # Conv1d in HP layout with CP-aware weight slicing
    mixed_qkv = qkv.transpose(1, 2).contiguous()  # [B, conv_dim/cp, S]
    conv1d_weight = get_parameter_local_cp(
        self.conv1d.weight,
        dim=0,
        cp_group=cp_group,
        split_sections=[self.key_dim, self.key_dim, self.value_dim],
    )
    conv1d_bias = None
    if self.conv1d.bias is not None:
        conv1d_bias = get_parameter_local_cp(
            self.conv1d.bias,
            dim=0,
            cp_group=cp_group,
            split_sections=[self.key_dim, self.key_dim, self.value_dim],
        )

    if self.causal_conv1d_fn is not None:
        mixed_qkv = self.causal_conv1d_fn(
            x=mixed_qkv,
            weight=conv1d_weight.squeeze(1),
            bias=conv1d_bias,
            activation=self.activation,
            seq_idx=None,
            **({"cu_seqlens": cu_seqlens} if cu_seqlens is not None else {}),
        )
    elif cu_seqlens is not None:
        raise RuntimeError(
            "cu_seqlens requires causal_conv1d_fn (FLA) but it is not available. "
            "Please install flash-linear-attention for pack support."
        )
    else:
        conv_out = F.conv1d(
            input=mixed_qkv,
            weight=conv1d_weight,
            bias=conv1d_bias,
            stride=self.conv1d.stride,
            padding=self.conv1d.padding,
            dilation=self.conv1d.dilation,
            groups=self.conv_dim // cp_size,
        )
        mixed_qkv = self.act(conv_out[..., :full_seq_len])
    mixed_qkv = mixed_qkv.transpose(1, 2).contiguous()  # [B, S, conv_dim/cp]

    query, key, value = torch.split(
        mixed_qkv,
        [self.key_dim // cp_size, self.key_dim // cp_size, self.value_dim // cp_size],
        dim=-1,
    )
    query = query.reshape(batch_size, full_seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, full_seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, full_seq_len, -1, self.head_v_dim)

    if self.num_v_heads // self.num_k_heads > 1:
        repeat_factor = self.num_v_heads // self.num_k_heads
        query = query.repeat_interleave(repeat_factor, dim=2)
        key = key.repeat_interleave(repeat_factor, dim=2)

    gate = z  # [B, S, num_v_heads/cp, head_v_dim]
    beta = b.squeeze(-1)  # [B, S, num_v_heads/cp]
    alpha = a.squeeze(-1)  # [B, S, num_v_heads/cp]

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    gate = gate.contiguous()
    beta = beta.contiguous()
    alpha = alpha.contiguous()

    A_log_local = get_parameter_local_cp(self.A_log, dim=0, cp_group=cp_group)
    dt_bias_local = get_parameter_local_cp(self.dt_bias, dim=0, cp_group=cp_group)
    g = -A_log_local.float().exp() * F.softplus(alpha.float() + dt_bias_local)
    beta_final = beta.sigmoid()

    # Gated delta rule in HP layout (needs full sequence)
    core_attn_out, _ = self.chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta_final,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        **({"cu_seqlens": cu_seqlens} if cu_seqlens is not None else {}),
    )

    z_shape_og = gate.shape
    core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
    z_flat = gate.reshape(-1, gate.shape[-1])
    core_attn_out = self.norm(core_attn_out, z_flat)
    core_attn_out = core_attn_out.reshape(z_shape_og)  # [B, S, num_v_heads/cp, head_v_dim]

    # HP->CP all_to_all: scatter seq (dim=1), gather heads (dim=2)
    # [B, S, num_v_heads/cp, head_v_dim] -> [B, S/cp, num_v_heads, head_v_dim]
    norm_out = SeqAllToAll4D.apply(cp_group, core_attn_out, 1, 2)
    norm_out = norm_out.reshape(batch_size, seq_len, -1)

    # Output projection in CP layout
    output = self.out_proj(norm_out)
    return output
