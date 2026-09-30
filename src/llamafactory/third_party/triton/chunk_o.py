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

from .utils import exp, prepare_chunk_indices, prepare_chunk_offsets


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_G_GAMMA": lambda args: args["g_gamma"] is not None,
        "USE_DW": lambda args: args["dw"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_bwd_kernel_dqkwg(
    q,
    k,
    v,
    h,
    g,
    g_gamma,
    do,
    dh,
    dq,
    dk,
    dg,
    w,
    dv,
    dw,
    cu_seqlens,
    chunk_indices,
    scale,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_DW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    gdiff,
):
    i_t, i_b = tl.program_id(0), tl.program_id(1)
    T_max = T
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        total = B * T_max
        T = eos - bos
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T
        total = B * T_max

    NK = tl.cdiv(K, BK)
    for i_k in range(NK):
        if USE_G:
            dg_k = dg + i_k * total * H

        for i_h in range(H):
            v_h = v + (bos * H + i_h) * V
            do_h = do + (bos * H + i_h) * V
            h_h = h + (i_tg * H + i_h).to(tl.int64) * K * V
            dh_h = dh + (i_tg * H + i_h).to(tl.int64) * K * V
            q_h = q + (bos * H + i_h) * K
            k_h = k + (bos * H + i_h) * K
            dq_h = dq + (bos * H + i_h) * K
            dk_h = dk + (bos * H + i_h) * K

            if USE_DW:
                w_h = w + (bos * H + i_h) * K  # noqa: F841
                dw_h = dw + (bos * H + i_h) * K
                dv_h = dv + (bos * H + i_h) * V

            if USE_G:
                if IS_VARLEN:
                    dg_h = dg_k + i_h * T_max + bos
                    g_h = g + i_h * T_max + bos
                else:
                    dg_h = dg_k + (i_b * H + i_h) * T_max
                    g_h = g + (i_b * H + i_h) * T_max
                b_dg_last = tl.zeros(
                    [
                        1,
                    ],
                    dtype=tl.float32,
                )

            if USE_G_GAMMA:
                b_gamma = tl.load(g_gamma + i_h)
                b_g = b_gamma * (tl.arange(0, BT) + 1)
                b_g_last = b_gamma * min(BT, T - i_t * BT)

            b_dq = tl.zeros([BT, BK], dtype=tl.float32)
            b_dk = tl.zeros([BT, BK], dtype=tl.float32)
            b_ds = tl.zeros([BT, BT], dtype=tl.float32)
            b_dw = tl.zeros([BT, BK], dtype=tl.float32) if USE_DW else None

            for i_v in range(tl.cdiv(V, BV)):
                p_v = tl.make_block_ptr(v_h, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                p_do = tl.make_block_ptr(do_h, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                p_h = tl.make_block_ptr(h_h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                p_dh = tl.make_block_ptr(dh_h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))

                b_v = tl.load(p_v, boundary_check=(0, 1))
                b_do = tl.load(p_do, boundary_check=(0, 1))
                b_h = tl.load(p_h, boundary_check=(0, 1))
                b_dh = tl.load(p_dh, boundary_check=(0, 1))

                if USE_G:
                    b_dg_last += tl.sum(b_h * b_dh)

                b_ds += tl.dot(b_do, tl.trans(b_v))
                b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
                b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))

                if USE_DW:
                    p_dv = tl.make_block_ptr(dv_h, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                    b_dv = tl.load(p_dv, boundary_check=(0, 1))
                    b_dw += tl.dot(b_dv.to(b_v.dtype), b_h.to(b_v.dtype))

            if USE_DW:
                p_dw = tl.make_block_ptr(dw_h, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
                tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))

            tl.debug_barrier()

            p_q = tl.make_block_ptr(q_h, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(k_h, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))

            p_dq = tl.make_block_ptr(dq_h, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dk = tl.make_block_ptr(dk_h, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T
            m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)

            if USE_G:
                b_dg = tl.zeros(
                    [
                        BT,
                    ],
                    dtype=tl.float32,
                )
                p_g = tl.make_block_ptr(g_h, (T,), (1,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,))
                b_g_last = tl.load(g_h + (min(i_t * BT + BT, T) - 1) * 1)
                b_dg_last *= tl.exp(b_g_last)

                b_dq = b_dq * tl.exp(b_g)[:, None] * scale
                b_dg += tl.sum(b_dq * b_q, axis=1)

                b_dk = b_dk * tl.where(m_t, tl.exp(-b_g + b_g_last), 0)[:, None]
                b_dg -= tl.sum(b_k * b_dk, axis=1)
                b_dg_last += tl.sum(b_dk * b_k)

                if IS_VARLEN:
                    b_ds = tl.where(m_A, b_ds * exp(b_g[:, None] - b_g[None, :]), 0) * scale
                else:
                    p_gdiff = tl.make_block_ptr(
                        gdiff + i_b * H * NT * BT * BT + i_h * NT * BT * BT + i_t * BT * BT,
                        (BT, BT),
                        (BT, 1),
                        (0, 0),
                        (BT, BT),
                        (1, 0),
                    )
                    gdiff_ = tl.load(p_gdiff)
                    b_ds = b_ds * gdiff_ * scale

                b_ds2 = b_ds * tl.dot(b_q, tl.trans(b_k))
                b_dg += tl.sum(b_ds2, axis=1)
                b_dg -= tl.sum(b_ds2, axis=0)

                b_ds = b_ds.to(b_k.dtype)
                b_dq += tl.dot(b_ds, b_k)
                b_dk += tl.dot(tl.trans(b_ds), b_q)
                p_dg = tl.make_block_ptr(dg_h, (T,), (1,), (i_t * BT,), (BT,), (0,))

                last_index_local = min(BT, T - i_t * BT) - 1
                if last_index_local >= 0:
                    is_last_mask = tl.arange(0, BT) == last_index_local
                    b_dg = tl.where(is_last_mask, b_dg + b_dg_last, b_dg)
                else:
                    pass

                tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
                tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
                tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))

            elif USE_G_GAMMA:
                b_dq = b_dq * exp(b_g)[:, None] * scale
                b_dk = b_dk * tl.where(m_t, exp(-b_g + b_g_last), 0)[:, None]
                b_ds = tl.where(m_A, b_ds * exp(b_g[:, None] - b_g[None, :]), 0) * scale
                b_ds = b_ds.to(b_k.dtype)
                b_dq += tl.dot(b_ds, b_k)
                b_dk += tl.dot(tl.trans(b_ds), b_q)
                tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
                tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

            else:
                b_ds = tl.where(m_A, b_ds, 0)
                b_ds = b_ds.to(b_k.dtype)
                b_dq += tl.dot(b_ds, b_k)
                b_dk += tl.dot(tl.trans(b_ds), b_q) * scale
                b_dq *= scale
                tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
                tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_G_GAMMA": lambda args: args["g_gamma"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_bwd_kernel_dv_local(
    q,
    k,
    g,
    g_gamma,
    do,
    dv,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_b = tl.program_id(0), tl.program_id(1)
    T_max = T

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    for i_h in range(H):
        offset_kh = (bos * H + i_h) * K
        offset_vh = (bos * H + i_h) * V

        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_k = tl.make_block_ptr(k + offset_kh, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_q = tl.make_block_ptr(q + offset_kh, (K, T), (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_A += tl.dot(b_k, b_q)

        if USE_G:
            if IS_VARLEN:
                offset_g = i_h * T_max + bos
            else:
                offset_g = i_b * H * T_max + i_h * T_max

            p_g = tl.make_block_ptr(g + offset_g, (T,), (1,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))

        if USE_G_GAMMA:
            b_gamma = tl.load(g_gamma + i_h)
            b_g = b_gamma * (tl.arange(0, BT) + 1)

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)

        if USE_G:
            b_A = tl.where(m_A, b_A * tl.exp(b_g[None, :] - b_g[:, None]) * scale, 0).to(do.dtype.element_ty)
        else:
            b_A = tl.where(m_A, b_A * scale, 0).to(do.dtype.element_ty)

        for i_v in range(tl.cdiv(V, BV)):
            p_do = tl.make_block_ptr(do + offset_vh, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv = tl.make_block_ptr(dv + offset_vh, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
            tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_G_GAMMA": lambda args: args["g_gamma"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    g_gamma,
    o,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    N: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    T_max = T
    for i_v in range(tl.cdiv(V, BV)):
        for i_n in range(N):
            if IS_VARLEN:
                bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
                T = eos - bos
                NT = tl.cdiv(T, BT)
                boh = tl.load(chunk_offsets + i_n).to(tl.int64)
            else:
                bos, eos = i_n * T, i_n * T + T
                NT = tl.cdiv(T, BT)
                boh = i_n * NT

            core_id = tl.program_id(0)
            total_cores = tl.num_programs(0)
            base_chunks_per_pid = NT // total_cores
            remainder = NT % total_cores

            if core_id < remainder:
                chunks_this_pid = base_chunks_per_pid + 1
                start_idx = core_id * chunks_this_pid
            else:
                chunks_this_pid = base_chunks_per_pid
                start_idx = core_id * base_chunks_per_pid + remainder

            # offset calculation
            for i_h in range(0, H):
                q_offset = (bos * Hg + i_h // (H // Hg)) * K
                k_offset = (bos * Hg + i_h // (H // Hg)) * K
                v_offset = (bos * H + i_h) * V
                o_offset = (bos * H + i_h) * V

                for i_t in range(start_idx, start_idx + chunks_this_pid):
                    i_tg = boh + i_t
                    h_base = h + (i_tg * H + i_h).to(tl.int64) * K * V
                    b_o = tl.zeros([BT, BV], dtype=tl.float32)
                    b_A = tl.zeros([BT, BT], dtype=tl.float32)
                    for i_k in range(tl.cdiv(K, BK)):
                        p_q = tl.make_block_ptr(
                            q + q_offset, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
                        )
                        p_k = tl.make_block_ptr(
                            k + k_offset, (K, T), (1, Hg * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1)
                        )
                        p_h = tl.make_block_ptr(h_base, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
                        b_q = tl.load(p_q, boundary_check=(0, 1))
                        b_k = tl.load(p_k, boundary_check=(0, 1))
                        b_h = tl.load(p_h, boundary_check=(0, 1))

                        # [BT, BK] @ [BK, BV] -> [BT, BV]
                        b_o += tl.dot(b_q, b_h)
                        # [BT, BK] @ [BK, BT] -> [BT, BT]
                        b_A += tl.dot(b_q, b_k)

                    if USE_G:
                        if IS_VARLEN:
                            p_g = tl.make_block_ptr(g + bos + i_h * T_max, (T,), (1,), (i_t * BT,), (BT,), (0,))
                        else:
                            p_g = tl.make_block_ptr(g + bos * H + i_h * T_max, (T,), (1,), (i_t * BT,), (BT,), (0,))
                        b_g = tl.load(p_g, boundary_check=(0,))
                        b_o = b_o * exp(b_g)[:, None]
                        b_A = b_A * exp(b_g[:, None] - b_g[None, :])
                    if USE_G_GAMMA:
                        b_gamma = tl.load(g_gamma + i_h)
                        b_g = b_gamma * (tl.arange(0, BT) + 1)

                    o_i = tl.arange(0, BT)
                    m_A = o_i[:, None] >= o_i[None, :]
                    b_A = tl.where(m_A, b_A, 0)

                    p_v = tl.make_block_ptr(v + v_offset, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                    p_o = tl.make_block_ptr(o + o_offset, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                    b_v = tl.load(p_v, boundary_check=(0, 1))

                    # to fix mma -> mma layout conversion
                    # already solved by triton v3.2 or higher
                    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
                    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_bwd_dqkwg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    BK = 128 if cu_seqlens is None else 64
    BV = 64
    NK = triton.cdiv(K, BK)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    g = g.transpose(1, 2).contiguous()
    dg = torch.empty(NK, *g.shape, dtype=torch.float32, device=g.device) if g is not None else None
    dw = torch.empty_like(w) if w is not None else None
    grid = (NT, B)

    if cu_seqlens is None:
        if NT * BT == T:
            g_ = g.reshape(B, H, NT, BT)
            g_diff = g_[:, :, :, :, None] - g_[:, :, :, None, :]
            g_diff = g_diff.clamp(-60, 60).exp()
            g_diff[:, :, :] *= torch.tril(torch.ones(BT, BT), diagonal=0).to(g.device)
        else:
            diff = NT * BT - T
            g_ = torch.cat((g, torch.zeros(B, H, diff).to(g.device)), dim=-1).reshape(B, H, NT, BT)
            g_diff = g_[:, :, :, :, None] - g_[:, :, :, None, :]
            g_diff = g_diff.clamp(-60, 60).exp()
            g_diff[:, :, :] *= torch.tril(torch.ones(BT, BT), diagonal=0).to(g.device)
            bias = torch.arange(0, BT).to(g.device)
            o_t = (NT - 1) * BT + bias
            m_t = o_t < T
            m_A = m_t[:, None] & m_t
            g_diff[:, :, -1] *= m_A
    else:
        g_diff = None

    chunk_bwd_kernel_dqkwg[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        do=do,
        dh=dh,
        dv=dv,
        w=w,
        dw=dw,
        dq=dq,
        dk=dk,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        gdiff=g_diff,
    )

    if dg is not None:
        dg = dg.sum(0)
        dg = dg.transpose(1, 2).contiguous()
    return dq, dk, dw, dg


def chunk_bwd_dv_local(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    scale: float = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, H, K, V = *k.shape, do.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None

    BK = 128
    BV = 128
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    g = g.transpose(1, 2).contiguous()
    dv = torch.empty_like(do)
    grid = (NT, B)
    chunk_bwd_kernel_dv_local[grid](
        q=q,
        k=k,
        g=g,
        g_gamma=g_gamma,
        do=do,
        dv=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return dv


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)  # noqa: F841
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.empty_like(v)
    if cu_seqlens is None:
        N, chunk_offsets = B, None
    else:
        N, chunk_offsets = (
            len(cu_seqlens) - 1,
            prepare_chunk_offsets(cu_seqlens, BT),
        )

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    g = g.transpose(1, 2).contiguous()
    h = h.contiguous()
    CV_kernel_num = 24
    chunk_fwd_kernel_o[(CV_kernel_num,)](
        q,
        k,
        v,
        h,
        g,
        g_gamma,
        o,
        cu_seqlens,
        chunk_offsets,
        scale,
        T=T,
        H=H,
        N=N,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=128,
        BV=128,
    )
    return o


bwd_chunk_dqkwg = chunk_bwd_dqkwg
bwd_chunk_dv_local = chunk_bwd_dv_local
