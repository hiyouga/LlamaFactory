# Copyright 2025 EleutherAI, HuggingFace Inc., Yukang Chen, and the LlamaFactory team.
#
# This code is based on the EleutherAI's GPT-NeoX and the HuggingFace's Transformers libraries.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llama/modeling_llama.py
# This code is also inspired by the original LongLoRA implementation.
# https://github.com/dvlab-research/LongLoRA/blob/main/llama_attn_replace.py
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

import math
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import transformers

from ...extras import logging
from ...extras.constants import SUPPORTED_CLASS_FOR_S2ATTN
from ...extras.misc import check_version
from ...extras.packages import is_transformers_version_greater_than


if not is_transformers_version_greater_than("4.48.0"):
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
    from transformers.models.llama.modeling_llama import (
        Cache,
        LlamaAttention,
        LlamaFlashAttention2,
        LlamaSdpaAttention,
        apply_rotary_pos_emb,
        repeat_kv,
    )


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


transformers_logger = transformers.utils.logging.get_logger(__name__)


# Modified from:
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llama/modeling_llama.py
def llama_attention_forward(
    self: "LlamaAttention",
    hidden_states: "torch.Tensor",
    attention_mask: Optional["torch.Tensor"] = None,
    position_ids: Optional["torch.LongTensor"] = None,
    past_key_value: Optional["Cache"] = None,
    output_attentions: bool = False,
    cache_position: Optional["torch.LongTensor"] = None,
    position_embeddings: Optional[tuple["torch.Tensor", "torch.Tensor"]] = None,
    **kwargs,
) -> tuple["torch.Tensor", Optional["torch.Tensor"], Optional[tuple["torch.Tensor"]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states: torch.Tensor = self.q_proj(hidden_states)
    key_states: torch.Tensor = self.k_proj(hidden_states)
    value_states: torch.Tensor = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if getattr(self.config, "group_size_ratio", None) and self.training:  # shift
        groupsz = int(q_len * getattr(self.config, "group_size_ratio"))
        assert q_len % groupsz == 0, f"q_len {q_len} should be divisible by group size {groupsz}."
        num_groups = q_len // groupsz

        def shift(state: "torch.Tensor") -> "torch.Tensor":
            state = state.transpose(1, 2)  # output: (bsz, seq_len, n_heads, head_dim)
            state = torch.cat(
                (state[:, :, : self.num_heads // 2], state[:, :, self.num_heads // 2 :].roll(-groupsz // 2, dims=1)),
                dim=2,
            )
            return state.reshape(bsz * num_groups, groupsz, self.num_heads, self.head_dim).transpose(1, 2)

        query_states, key_states, value_states = shift(query_states), shift(key_states), shift(value_states)
        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :groupsz, :groupsz].repeat(num_groups, 1, 1, 1)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)  # (bsz, :, seq_len, :) or (bsz * n_group, :, groupsz, :)
    attn_output = attn_output.transpose(1, 2).contiguous()

    if getattr(self.config, "group_size_ratio", None) and self.training:  # shift back
        attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)
        attn_output = torch.cat(
            (
                attn_output[:, :, : self.num_heads // 2],
                attn_output[:, :, self.num_heads // 2 :].roll(groupsz // 2, dims=1),
            ),
            dim=2,
        )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# Modified from:
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llama/modeling_llama.py
def llama_flash_attention_2_forward(
    self: "LlamaFlashAttention2",
    hidden_states: "torch.Tensor",
    attention_mask: Optional["torch.Tensor"] = None,
    position_ids: Optional["torch.LongTensor"] = None,
    past_key_value: Optional["Cache"] = None,
    output_attentions: bool = False,
    cache_position: Optional["torch.LongTensor"] = None,
    position_embeddings: Optional[tuple["torch.Tensor", "torch.Tensor"]] = None,
    **kwargs,
) -> tuple["torch.Tensor", Optional["torch.Tensor"], Optional[tuple["torch.Tensor"]]]:
    # LlamaFlashAttention2 attention does not support output_attentions
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states: torch.Tensor = self.q_proj(hidden_states)
    key_states: torch.Tensor = self.k_proj(hidden_states)
    value_states: torch.Tensor = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # FlashAttention requires the input to have the shape (bsz, seq_len, n_heads, head_dim)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        transformers_logger.warning_once("The input hidden states seems to be silently casted in float32.")
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    if getattr(self.config, "group_size_ratio", None) and self.training:  # shift
        groupsz = int(q_len * getattr(self.config, "group_size_ratio"))
        assert q_len % groupsz == 0, f"q_len {q_len} should be divisible by group size {groupsz}."
        num_groups = q_len // groupsz

        def shift(state: "torch.Tensor") -> "torch.Tensor":
            state = torch.cat(
                (state[:, :, : self.num_heads // 2], state[:, :, self.num_heads // 2 :].roll(-groupsz // 2, dims=1)),
                dim=2,
            )
            return state.reshape(bsz * num_groups, groupsz, self.num_heads, self.head_dim)

        query_states, key_states, value_states = shift(query_states), shift(key_states), shift(value_states)
        if attention_mask is not None:
            attention_mask = attention_mask[:, :groupsz].repeat(num_groups, 1)

        attn_output: torch.Tensor = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_states.size(1),
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

    if getattr(self.config, "group_size_ratio", None) and self.training:  # shift back
        attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)
        attn_output = torch.cat(
            (
                attn_output[:, :, : self.num_heads // 2],
                attn_output[:, :, self.num_heads // 2 :].roll(groupsz // 2, dims=1),
            ),
            dim=2,
        )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# Modified from:
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llama/modeling_llama.py
def llama_sdpa_attention_forward(
    self: "LlamaSdpaAttention",
    hidden_states: "torch.Tensor",
    attention_mask: Optional["torch.Tensor"] = None,
    position_ids: Optional["torch.LongTensor"] = None,
    past_key_value: Optional["Cache"] = None,
    output_attentions: bool = False,
    cache_position: Optional["torch.LongTensor"] = None,
    position_embeddings: Optional[tuple["torch.Tensor", "torch.Tensor"]] = None,
    **kwargs,
) -> tuple["torch.Tensor", Optional["torch.Tensor"], Optional[tuple["torch.Tensor"]]]:
    if output_attentions:
        transformers_logger.warning_once(
            "SDPA does not support `output_attentions=True`. Falling back to the vanilla attention"
        )
        return llama_attention_forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            cache_position=cache_position,
            **kwargs,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states: torch.Tensor = self.q_proj(hidden_states)
    key_states: torch.Tensor = self.k_proj(hidden_states)
    value_states: torch.Tensor = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if getattr(self.config, "group_size_ratio", None) and self.training:  # shift
        groupsz = int(q_len * getattr(self.config, "group_size_ratio"))
        assert q_len % groupsz == 0, f"q_len {q_len} should be divisible by group size {groupsz}."
        num_groups = q_len // groupsz

        def shift(state: "torch.Tensor") -> "torch.Tensor":
            state = state.transpose(1, 2)  # output: (bsz, seq_len, n_heads, head_dim)
            state = torch.cat(
                (state[:, :, : self.num_heads // 2], state[:, :, self.num_heads // 2 :].roll(-groupsz // 2, dims=1)),
                dim=2,
            )
            return state.reshape(bsz * num_groups, groupsz, self.num_heads, self.head_dim).transpose(1, 2)

        query_states, key_states, value_states = shift(query_states), shift(key_states), shift(value_states)
        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :groupsz, :groupsz].repeat(num_groups, 1, 1, 1)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    if query_states.device.type in ("cuda", "npu") and causal_mask is not None:  # avoid pytorch bug
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    is_causal = True if causal_mask is None and q_len > 1 else False
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    if getattr(self.config, "group_size_ratio", None) and self.training:  # shift back
        attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)
        attn_output = torch.cat(
            (
                attn_output[:, :, : self.num_heads // 2],
                attn_output[:, :, self.num_heads // 2 :].roll(groupsz // 2, dims=1),
            ),
            dim=2,
        )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def _apply_llama_patch() -> None:
    check_version("transformers>=4.45.0,<4.48.0", mandatory=True)
    LlamaAttention.forward = llama_attention_forward
    LlamaFlashAttention2.forward = llama_flash_attention_2_forward
    LlamaSdpaAttention.forward = llama_sdpa_attention_forward


def configure_longlora(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool) -> None:
    if not is_trainable or not model_args.shift_attn:
        return

    logger = logging.get_logger(__name__)

    if getattr(config, "model_type", None) in SUPPORTED_CLASS_FOR_S2ATTN:
        setattr(config, "group_size_ratio", 0.25)
        _apply_llama_patch()
        logger.info_rank0("Using shift short attention with group_size_ratio=1/4.")
    else:
        logger.warning_rank0("Current model does not support shift short attention.")
