# Copyright 2025 the LlamaFactory team.
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
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


from typing import Optional

import torch
import triton
import triton.language as tl

from .utils import get_autotune_config, get_npu_properties, prepare_chunk_indices, prepare_chunk_offsets


CUBE_CORE_NUM = get_npu_properties()["num_aicore"]


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=get_autotune_config(multibuffer_list=(False,)),
    key=["H", "K", "V", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    T_all = T
    NT_all = NT
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # Initialize hidden states
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    if IS_VARLEN:
        v = v + (i_h * T_all + bos) * V
        k = k + (i_h * T_all + bos) * K
        w = w + (i_h * T_all + bos) * K
        g = g + i_h * T_all + bos
        h = h + (i_h * NT_all + boh) * K * V
        if SAVE_NEW_VALUE:
            v_new_base = v_new + (i_h * T_all + bos) * V
    else:
        v = v + (i_n * H + i_h) * T * V
        k = k + (i_n * H + i_h) * T * K
        w = w + (i_n * H + i_h) * T * K
        g = g + (i_n * H + i_h) * T
        h = h + (i_n * H + i_h) * NT * K * V
        if SAVE_NEW_VALUE:
            v_new_base = v_new + (i_n * H + i_h) * T * V

    if USE_INITIAL_STATE:
        h0_ptr = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht_ptr = ht + i_nh * K * V

    # Load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # Main recurrence over chunks
    for i_t in range(NT):
        # Store current hidden state h_t
        p_h1 = tl.make_block_ptr(h + i_t * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(h + i_t * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(h + i_t * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(h + i_t * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        # Compute v_residual = v - w @ h
        p_w = tl.make_block_ptr(w, (T, K), (K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h4.to(b_w.dtype))

        p_v = tl.make_block_ptr(v, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v_new = tl.make_block_ptr(v_new_base, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1

        # Apply output gate g
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)).to(tl.float32) < T
            b_g_last = tl.load(g + last_idx)
            p_g = tl.make_block_ptr(g, (T,), (1,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v *= (m_t * tl.exp(b_g_last - b_g))[:, None]
            b_g_last_exp = tl.exp(b_g_last)
            b_h1 *= b_g_last_exp
            if K > 64:
                b_h2 *= b_g_last_exp
            if K > 128:
                b_h3 *= b_g_last_exp
            if K > 192:
                b_h4 *= b_g_last_exp

        # Apply key gate gk
        if USE_GK:
            o_k1 = tl.arange(0, 64).to(tl.float32)
            gk_base_ptr = gk + (i_n * H + i_h) * T * K
            b_gk_last1 = tl.load(gk_base_ptr + last_idx * K + o_k1, mask=(o_k1 < K), other=0.0)
            b_h1 *= tl.exp(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk_base_ptr + last_idx * K + o_k2, mask=(o_k2 < K), other=0.0)
                b_h2 *= tl.exp(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk_base_ptr + last_idx * K + o_k3, mask=(o_k3 < K), other=0.0)
                b_h3 *= tl.exp(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk_base_ptr + last_idx * K + o_k4, mask=(o_k4 < K), other=0.0)
                b_h4 *= tl.exp(b_gk_last4)[:, None]

        b_v = b_v.to(k.dtype.element_ty)

        # Update hidden state: h += k @ v
        p_k = tl.make_block_ptr(k, (K, T), (1, K), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if USE_GK:
            p_gk = tl.make_block_ptr(gk_base_ptr, (K, T), (1, K), (0, i_t * BT), (64, BT), (0, 1))
            b_k = (b_k * tl.exp(b_gk_last1[:, None] - tl.load(p_gk, boundary_check=(0, 1)))).to(b_k.dtype)
        b_h1 += tl.dot(b_k, b_v)

        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, K), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                p_gk = tl.make_block_ptr(gk_base_ptr, (K, T), (1, K), (64, i_t * BT), (64, BT), (0, 1))
                b_k = (b_k * tl.exp(b_gk_last2[:, None] - tl.load(p_gk, boundary_check=(0, 1)))).to(b_k.dtype)
            b_h2 += tl.dot(b_k, b_v)

        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, K), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                p_gk = tl.make_block_ptr(gk_base_ptr, (K, T), (1, K), (128, i_t * BT), (64, BT), (0, 1))
                b_k = (b_k * tl.exp(b_gk_last3[:, None] - tl.load(p_gk, boundary_check=(0, 1)))).to(b_k.dtype)
            b_h3 += tl.dot(b_k, b_v)

        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, K), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                p_gk = tl.make_block_ptr(gk_base_ptr, (K, T), (1, K), (192, i_t * BT), (64, BT), (0, 1))
                b_k = (b_k * tl.exp(b_gk_last4[:, None] - tl.load(p_gk, boundary_check=(0, 1)))).to(b_k.dtype)
            b_h4 += tl.dot(b_k, b_v)

    # Store final state
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,  # default:64
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, u.shape[-1]
    BT = chunk_size

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, K, V).permute(0, 2, 1, 3, 4).contiguous()
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    BV = 128

    v_new = torch.empty_like(u).permute(0, 2, 1, 3).contiguous() if save_new_value else None
    k = k.permute(0, 2, 1, 3).contiguous()
    w = w.permute(0, 2, 1, 3).contiguous()
    u = u.permute(0, 2, 1, 3).contiguous()
    g = g.permute(0, 2, 1).contiguous()
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[(triton.cdiv(V, BV), N * H)](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BV=BV,
        NT=NT,
    )
    h = h.permute(0, 2, 1, 3, 4).contiguous()
    v_new = v_new.permute(0, 2, 1, 3).contiguous()
    return h, v_new, final_state


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["dh0"] is not None,
        "USE_FINAL_STATE_GRADIENT": lambda args: args["dht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=get_autotune_config(multibuffer_list=(True, False)),
    key=["H", "K", "V", "BT", "BV", "USE_G", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64(
    q,
    k,
    w,
    g,
    gk,
    dht,
    dh0,
    do,
    dh,
    dv,
    dv2,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    T_all = T
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    do += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V
    dv2 += (bos * H + i_h) * V
    dh += (boh * H + i_h) * K * V
    if USE_GK:
        gk += (bos * H + i_h) * K

    if USE_INITIAL_STATE:
        dh0 += i_nh * K * V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K * V

    stride_v = H * V
    stride_h = H * K * V
    stride_k = H * K

    if USE_FINAL_STATE_GRADIENT:
        p_dht1 = tl.make_block_ptr(dht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_dh1 += tl.load(p_dht1, boundary_check=(0, 1))
        if K > 64:
            p_dht2 = tl.make_block_ptr(dht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_dh2 += tl.load(p_dht2, boundary_check=(0, 1))
        if K > 128:
            p_dht3 = tl.make_block_ptr(dht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_dh3 += tl.load(p_dht3, boundary_check=(0, 1))
        if K > 192:
            p_dht4 = tl.make_block_ptr(dht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_dh4 += tl.load(p_dht4, boundary_check=(0, 1))

    for i_t in range(NT - 1, -1, -1):
        p_dh1 = tl.make_block_ptr(dh + i_t * stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_dh2 = tl.make_block_ptr(dh + i_t * stride_h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_dh3 = tl.make_block_ptr(dh + i_t * stride_h, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_dh4 = tl.make_block_ptr(dh + i_t * stride_h, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            if IS_VARLEN:
                bos_g = i_h * T_all + bos
            else:
                bos_g = (i_n * H + i_h) * T_all
            bg_last = tl.load(g + bos_g + last_idx)
            bg_last_exp = tl.exp(bg_last)
            p_g = tl.make_block_ptr(
                base=g + bos_g, shape=(T,), strides=(1,), offsets=(i_t * BT,), block_shape=(BT,), order=(0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            b_g_exp = tl.exp(b_g)

        p_dv = tl.make_block_ptr(dv, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv2 = tl.make_block_ptr(dv2, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        b_do = tl.load(p_do, boundary_check=(0, 1))

        # Update dv
        p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + last_idx * H * K + o_k1, mask=(o_k1 < K), other=0.0)
        b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))

        if K > 64:
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + last_idx * H * K + o_k2, mask=(o_k2 < K), other=0.0)
            b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

        if K > 128:
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + last_idx * H * K + o_k3, mask=(o_k3 < K), other=0.0)
            b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

        if K > 192:
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + last_idx * H * K + o_k4, mask=(o_k4 < K), other=0.0)
            b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)).to(tl.float32) < T
            b_dv *= (m_t * tl.exp(bg_last - b_g))[:, None]
        b_dv += tl.load(p_dv, boundary_check=(0, 1))

        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
        # Update dh
        p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        if USE_G:
            b_dh1 *= bg_last_exp
            b_q = b_q * b_g_exp[None, :]
        if USE_GK:
            b_dh1 *= tl.exp(b_gk_last1[:, None])
        b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 64:
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh2 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                b_dh2 *= tl.exp(b_gk_last2[:, None])
            b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 128:
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh3 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                b_dh3 *= tl.exp(b_gk_last3[:, None])
            b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 192:
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh4 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                b_dh4 *= tl.exp(b_gk_last4[:, None])
            b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_dh1 = tl.make_block_ptr(dh0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_dh2 = tl.make_block_ptr(dh0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_dh3 = tl.make_block_ptr(dh0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *q.shape, do.shape[-1]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    BT = 64
    assert K <= 256, "current kernel does not support head dimension being larger than 256."

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    dh = q.new_empty(B, NT, H, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)

    BV = 128

    g = g.permute(0, 2, 1).contiguous()

    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64[(triton.cdiv(V, BV), N * H)](
        q=q,
        k=k,
        w=w,
        g=g,
        gk=gk,
        dht=dht,
        dh0=dh0,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BV=BV,
    )
    return dh, dh0, dv2
