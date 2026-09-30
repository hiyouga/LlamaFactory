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

from .utils import prepare_chunk_indices


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
    NT,
    B,
    TOTAL_TASKS,
):
    core_id = tl.program_id(0)
    num_blocks = tl.num_programs(0)
    T_max = T

    base_tasks_per_block = TOTAL_TASKS // num_blocks
    remainder_tasks = TOTAL_TASKS % num_blocks

    if core_id < remainder_tasks:
        tasks_this_core = base_tasks_per_block + 1
        start_idx = core_id * tasks_this_core
    else:
        tasks_this_core = base_tasks_per_block
        start_idx = core_id * base_tasks_per_block + remainder_tasks

    for idx in range(start_idx, start_idx + tasks_this_core):
        i_b = idx // NT
        local_idx = idx % NT

        if IS_VARLEN:
            i_n = tl.load(chunk_indices + local_idx * 2).to(tl.int32)
            i_t = tl.load(chunk_indices + local_idx * 2 + 1).to(tl.int32)
            bos = tl.load(cu_seqlens + i_n).to(tl.int32)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T_local = eos - bos
        else:
            bos, eos = 0, T
            i_t = local_idx
            T_local = T

        for i_h in range(H):
            k_batch_off = i_b * T_max * H * K
            beta_batch_off = i_b * H * T_max
            g_batch_off = i_b * H * T_max
            A_batch_off = i_b * T_max * H * BT

            p_beta = tl.make_block_ptr(
                beta + beta_batch_off + bos + i_h * T_max, (T_local,), (1,), (i_t * BT,), (BT,), (0,)
            )
            b_beta = tl.load(p_beta, boundary_check=(0,))

            b_A = tl.zeros([BT, BT], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                p_k = tl.make_block_ptr(
                    k + k_batch_off + (bos * H + i_h) * K,
                    (T_local, K),
                    (H * K, 1),
                    (i_t * BT, i_k * BK),
                    (BT, BK),
                    (1, 0),
                )
                b_k = tl.load(p_k, boundary_check=(0, 1))
                dot_product = tl.dot(b_k, tl.trans(b_k))

                o_t = i_t * BT + tl.arange(0, BT)
                o_t = o_t.to(tl.float32)
                T_mask = (o_t < T_local).to(tl.float32)

                row_indices = tl.arange(0, BT)[:, None]
                col_indices = tl.arange(0, BT)[None, :]
                tril_mask = (row_indices > col_indices).to(tl.float32)
                tril_mask = tril_mask * T_mask[:, None]
                masked_dot = dot_product * tril_mask
                b_A += masked_dot

            if USE_G:
                p_g = tl.make_block_ptr(
                    g + g_batch_off + bos + i_h * T_max, (T_local,), (1,), (i_t * BT,), (BT,), (0,)
                )
                b_g = tl.load(p_g, boundary_check=(0,))
                b_g_diff = b_g[:, None] - b_g[None, :]
                b_g_diff = tl.minimum(tl.maximum(b_g_diff, -50.0), 50.0)
                b_A *= tl.exp(b_g_diff)
            b_A *= b_beta[:, None]

            p_A = tl.make_block_ptr(
                A + A_batch_off + (bos * H + i_h) * BT, (T_local, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT), (1, 0)
            )
            tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(configs=[triton.Config({"BK": BK}) for BK in [32, 64]], key=["BC"])
@triton.jit(do_not_specialize=["T"])
def chunk_scaled_dot_kkt_fwd_kernel_intra_sub_inter(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_i, i_j = i_c // NC, i_c % NC

    for i_h in range(H):
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T_val = eos - bos
        else:
            bos, eos = i_b * T, i_b * T + T
            T_val = T

        should_compute = (i_t * BT + i_i * BC < T_val) and (i_i > i_j)

        if should_compute:
            k_ptr = k + (bos * H + i_h) * K
            g_ptr = g + (bos * H + i_h) * K
            A_ptr = A + (bos * H + i_h) * BT

            p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T_val,), (H,), (i_t * BT + i_i * BC,), (BC,), (0,))
            b_beta = tl.load(p_beta, boundary_check=(0,))

            b_A = tl.zeros([BC, BC], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                p_k = tl.make_block_ptr(
                    k_ptr, (T_val, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0)
                )
                p_g = tl.make_block_ptr(
                    g_ptr, (T_val, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0)
                )
                b_kt = tl.make_block_ptr(
                    k_ptr, (K, T_val), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1)
                )
                p_gk = tl.make_block_ptr(
                    g_ptr, (K, T_val), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1)
                )

                o_k = i_k * BK + tl.arange(0, BK)
                m_k = o_k < K
                b_gn = tl.load(g_ptr + (i_t * BT + i_i * BC) * H * K + o_k, mask=m_k, other=0)
                b_g = tl.load(p_g, boundary_check=(0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1)) * tl.exp(b_g - b_gn[None, :])
                b_gk = tl.load(p_gk, boundary_check=(0, 1))
                b_kt = tl.load(b_kt, boundary_check=(0, 1)) * tl.exp(b_gn[:, None] - b_gk)
                b_A += tl.dot(b_k, b_kt)
            b_A *= b_beta[:, None]

            p_A = tl.make_block_ptr(A_ptr, (T_val, BT), (H * BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
            tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def chunk_scaled_dot_kkt_fwd_kernel_intra_sub_intra(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_i, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    for i_h in range(H):
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T_val = eos - bos
        else:
            bos, eos = i_b * T, i_b * T + T
            T_val = T

        should_compute = i_t * BT + i_i * BC < T_val

        if should_compute:
            o_i = tl.arange(0, BC)
            o_k = tl.arange(0, BK)
            m_k = o_k < K
            m_A = (i_t * BT + i_i * BC + o_i) < T_val
            o_A = (bos + i_t * BT + i_i * BC + o_i) * H * BT + i_h * BT + i_i * BC

            p_k = tl.make_block_ptr(
                k + (bos * H + i_h) * K, (T_val, K), (H * K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0)
            )
            p_g = tl.make_block_ptr(
                g + (bos * H + i_h) * K, (T_val, K), (H * K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0)
            )
            p_beta = beta + (bos + i_t * BT + i_i * BC + o_i) * H + i_h

            b_k = tl.load(p_k, boundary_check=(0, 1)) * tl.load(p_beta, mask=m_A, other=0)[:, None]
            b_g = tl.load(p_g, boundary_check=(0, 1))

            p_kt = k + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
            p_gk = g + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k

            for j in range(0, min(BC, T_val - i_t * BT - i_i * BC)):
                b_kt = tl.load(p_kt, mask=m_k, other=0).to(tl.float32)
                b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
                b_A = tl.sum(b_k * b_kt[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
                # 转化成f32
                o_i_tmp = o_i.to(tl.float32)
                b_A = tl.where(o_i_tmp > j, b_A, 0.0)

                tl.store(A + o_A + j, b_A, mask=m_A)
                p_kt += H * K
                p_gk += H * K


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`. Default: `None`.
        gk (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H, K]` applied to the key tensor. Default: `None`.
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """
    B, T, H, K = k.shape
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    beta = beta.transpose(1, 2).contiguous()
    g = g.transpose(1, 2).contiguous()
    BK = 128
    kernel_num = 24

    if gk is None:
        A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
        chunk_scaled_dot_kkt_fwd_kernel[(kernel_num,)](
            k=k,
            g=g,
            beta=beta,
            A=A,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            K=K,
            BT=BT,
            BK=BK,
            NT=NT,
            B=B,
            TOTAL_TASKS=B * NT,
        )
        return A

    BC = min(16, BT)
    NC = triton.cdiv(BT, BC)
    BK = max(triton.next_power_of_2(K), 16)
    A = torch.zeros(B, T, H, BT, device=k.device, dtype=output_dtype)
    grid = (NT, NC * NC, B)
    chunk_scaled_dot_kkt_fwd_kernel_intra_sub_inter[grid](
        k=k,
        g=gk,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        NC=NC,
    )

    grid = (NT, NC, B)
    chunk_scaled_dot_kkt_fwd_kernel_intra_sub_intra[grid](
        k=k,
        g=gk,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
    )
    return A
