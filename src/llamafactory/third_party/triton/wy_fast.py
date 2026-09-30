# Copyright 2025 the LlamaFactory team.
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.
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

from .utils import exp, prepare_chunk_indices


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def prepare_wy_repr_bwd_kernel(
    k,
    v,
    beta,
    g,
    A,
    dw,
    du,
    dk,
    dv,
    dbeta,
    dg,
    cu_seqlens,
    chunk_indices,
    T,
    B,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NT: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    core_id = tl.program_id(0)
    total_cores = tl.num_programs(0)
    T_max = T

    base_chunks_per_pid = NT // total_cores
    remainder_chunks = NT % total_cores

    if core_id < remainder_chunks:
        chunks_this_pid = base_chunks_per_pid + 1
        start_idx = core_id * chunks_this_pid
    else:
        chunks_this_pid = base_chunks_per_pid
        start_idx = core_id * chunks_this_pid + remainder_chunks

    for idx in range(start_idx, start_idx + chunks_this_pid):
        for i_b in range(B):
            if IS_VARLEN:
                i_n, i_t = (
                    tl.load(chunk_indices + idx * 2).to(tl.int32),
                    tl.load(chunk_indices + idx * 2 + 1).to(tl.int32),
                )
                bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
                T = eos - bos
            else:
                i_t = idx
                bos, eos = i_b * T, i_b * T + T

            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T
            m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
            for i_h in range(0, H):
                if IS_VARLEN:
                    offset = bos + i_h * T_max
                else:
                    offset = bos * H + i_h * T_max

                p_beta = tl.make_block_ptr(beta + offset, (T,), (1,), (i_t * BT,), (BT,), (0,))
                p_g = tl.make_block_ptr(g + offset, (T,), (1,), (i_t * BT,), (BT,), (0,))
                p_A = tl.make_block_ptr(
                    A + (bos * H + i_h) * BT, (BT, T), (1, H * BT), (0, i_t * BT), (BT, BT), (0, 1)
                )

                b_A = tl.load(p_A, boundary_check=(0, 1))
                b_beta = tl.load(p_beta, boundary_check=(0,))
                b_g = tl.load(p_g, boundary_check=(0,))
                b_g_exp = tl.exp(b_g)

                b_dbeta = tl.zeros([BT], dtype=tl.float32)
                b_dA = tl.zeros([BT, BT], dtype=tl.float32)
                b_dg = tl.zeros([BT], dtype=tl.float32)

                for i_k in range(tl.cdiv(K, BK)):
                    p_k = tl.make_block_ptr(
                        k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
                    )
                    p_dk = tl.make_block_ptr(
                        dk + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
                    )
                    p_dw = tl.make_block_ptr(
                        dw + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
                    )
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    b_k_beta_g = (b_k * b_beta[:, None] * b_g_exp[:, None]).to(b_k.dtype)
                    b_dw = tl.load(p_dw, boundary_check=(0, 1))
                    b_dA += tl.dot(b_dw, tl.trans(b_k_beta_g))
                    b_dk_beta_g = tl.dot(b_A, b_dw)
                    b_dk = b_dk_beta_g * b_beta[:, None] * b_g_exp[:, None]
                    b_dbeta += tl.sum(b_dk_beta_g * b_k * b_g_exp[:, None], 1)
                    b_dg += tl.sum(b_dk_beta_g * b_k * b_g_exp[:, None] * b_beta[:, None], 1)
                    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

                for i_v in range(tl.cdiv(V, BV)):
                    p_v = tl.make_block_ptr(
                        v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
                    )
                    p_dv = tl.make_block_ptr(
                        dv + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
                    )
                    p_du = tl.make_block_ptr(
                        du + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
                    )
                    b_v = tl.load(p_v, boundary_check=(0, 1))
                    b_v_beta = (b_v * b_beta[:, None]).to(b_v.dtype)
                    b_du = tl.load(p_du, boundary_check=(0, 1))
                    b_dA += tl.dot(b_du, tl.trans(b_v_beta))
                    b_dv_beta = tl.dot(b_A, b_du)
                    b_dv = b_dv_beta * b_beta[:, None]
                    b_dbeta += tl.sum(b_dv_beta * b_v, 1)
                    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

                b_dA = tl.where(m_A, b_dA, 0)
                b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
                b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
                b_dA = tl.where(m_A, -b_dA * exp(b_g[:, None] - b_g[None, :]), 0)
                b_dA = b_dA.to(k.dtype.element_ty)
                b_A = tl.zeros([BT, BT], dtype=tl.float32)

                for i_k in range(tl.cdiv(K, BK)):
                    p_k = tl.make_block_ptr(
                        k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
                    )
                    p_dk = tl.make_block_ptr(
                        dk + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
                    )
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    b_dk = tl.load(p_dk, boundary_check=(0, 1))
                    b_k_beta = (b_k * b_beta[:, None]).to(b_k.dtype)
                    b_A += tl.dot(b_k_beta, tl.trans(b_k))
                    b_dk_beta = tl.dot(b_dA, b_k)
                    b_dbeta += tl.sum(b_dk_beta * b_k, 1)
                    b_dk += tl.dot(tl.trans(b_dA), b_k_beta)
                    b_dk += b_dk_beta * b_beta[:, None]
                    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

                b_dA_A = b_dA * b_A
                b_dg += tl.sum(b_dA_A, axis=1) - tl.sum(b_dA_A, axis=0)
                p_dg = tl.make_block_ptr(dg + offset, (T,), (1,), (i_t * BT,), (BT,), (0,))
                p_dbeta = tl.make_block_ptr(dbeta + offset, (T,), (1,), (i_t * BT,), (BT,), (0,))
                tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))
                tl.store(p_dbeta, b_dbeta.to(p_dbeta.dtype.element_ty), boundary_check=(0,))


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    gk,
    cu_seqlens,
    chunk_indices,
    T_tmp,
    B,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NT: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    core_id = tl.program_id(0)
    total_cores = tl.num_programs(0)
    T_max = T_tmp

    base_chunks_per_pid = NT // total_cores
    remainder_chunks = NT % total_cores

    if core_id < remainder_chunks:
        chunks_this_pid = base_chunks_per_pid + 1
        start_idx = core_id * chunks_this_pid
    else:
        chunks_this_pid = base_chunks_per_pid
        start_idx = core_id * chunks_this_pid + remainder_chunks

    for idx in range(start_idx, start_idx + chunks_this_pid):
        for i_b in range(B):
            for i_h in range(0, H):
                if IS_VARLEN:
                    i_n, i_t = (
                        tl.load(chunk_indices + idx * 2).to(tl.int32),
                        tl.load(chunk_indices + idx * 2 + 1).to(tl.int32),
                    )
                    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
                    offset = bos + i_h * T_max
                    T = eos - bos
                else:
                    T = T_tmp
                    i_t = idx
                    bos, eos = i_b * T, i_b * T + T
                    offset = bos * H + i_h * T_max

                p_beta = tl.make_block_ptr(beta + offset, (T,), (1,), (i_t * BT,), (BT,), (0,))
                b_beta = tl.load(p_beta, boundary_check=(0,))

                p_A = tl.make_block_ptr(
                    A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
                )
                b_A = tl.load(p_A, boundary_check=(0, 1))

                for i_v in range(tl.cdiv(V, BV)):
                    p_v = tl.make_block_ptr(
                        v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
                    )
                    p_u = tl.make_block_ptr(
                        u + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
                    )
                    b_v = tl.load(p_v, boundary_check=(0, 1))
                    b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
                    b_u = tl.dot(b_A, b_vb, allow_tf32=False)
                    tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

                if USE_G:
                    p_g = tl.make_block_ptr(g + offset, (T,), (1,), (i_t * BT,), (BT,), (0,))
                    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))

                for i_k in range(tl.cdiv(K, BK)):
                    p_k = tl.make_block_ptr(
                        k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
                    )
                    p_w = tl.make_block_ptr(
                        w + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
                    )
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    b_kb = b_k * b_beta[:, None]
                    if USE_G:
                        b_kb *= b_g[:, None]
                    if USE_GK:
                        p_gk = tl.make_block_ptr(
                            gk + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
                        )
                        b_kb *= tl.exp(tl.load(p_gk, boundary_check=(0, 1)))
                    b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
                    tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    BK = 128
    BV = 128

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    g = g.transpose(1, 2).contiguous() if g is not None else None
    beta = beta.transpose(1, 2).contiguous()

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    cv_kernel_num = 24
    recompute_w_u_fwd_kernel[(cv_kernel_num,)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T_tmp=T,
        B=B,
        H=H,
        K=K,
        V=V,
        NT=NT,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return w, u


def prepare_wy_repr_bwd(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = 128
    BV = 128
    beta = beta.transpose(1, 2).contiguous()
    g = g.transpose(1, 2).contiguous()

    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dbeta = torch.empty_like(beta)
    dg = torch.empty_like(g)

    cv_kernel_num = 24
    prepare_wy_repr_bwd_kernel[(cv_kernel_num,)](
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=du,
        dk=dk,
        dv=dv,
        dbeta=dbeta,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        NT=NT,
        BT=BT,
        BK=BK,
        BV=BV,
    )

    dbeta = dbeta.transpose(1, 2).contiguous()
    dg = dg.transpose(1, 2).contiguous()

    return dk, dv, dbeta, dg


bwd_prepare_wy_repr = prepare_wy_repr_bwd

fwd_recompute_w_u = recompute_w_u_fwd
