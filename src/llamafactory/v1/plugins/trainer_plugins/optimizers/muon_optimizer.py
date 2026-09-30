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
#
# This module is vendored into v1 (independent of v0 / `llamafactory.third_party.muon`)
# so that the v1 optimizer plugin does not depend on v0 code.
#
# Based on MoonshotAI's Moonlight library and Keller Jordan's Muon library:
#   https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py
#   https://github.com/KellerJordan/Muon/blob/master/muon.py
# (originally MIT-licensed; re-distributed here under Apache 2.0).

import math
import os

import torch
import torch.distributed as dist


def _dtensor_cls():
    """Return the DTensor class if available, else None."""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # pragma: no cover
        try:
            from torch.distributed._tensor import DTensor  # type: ignore[no-redef]
        except ImportError:
            return None
    return DTensor


def _is_dtensor(t) -> bool:
    """True if ``t`` is a DTensor (i.e. sharded by FSDP2)."""
    DT = _dtensor_cls()
    return DT is not None and isinstance(t, DT)


def _distribute(tensor, mesh, placements):
    """Scatter a full (replicated) tensor into a DTensor with the given mesh/placements."""
    try:
        from torch.distributed.tensor import distribute_tensor
    except ImportError:  # pragma: no cover
        from torch.distributed._tensor import distribute_tensor  # type: ignore[no-redef]
    return distribute_tensor(tensor, mesh, placements)


def _is_rank0() -> bool:
    """True on rank 0 (or when not distributed)."""
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def zeropower_via_newtonschulz5(G: "torch.Tensor", steps: int) -> "torch.Tensor":
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    We opt to use a quintic iteration whose coefficients are selected to maximize the slope at zero.
    For the purpose of minimizing steps, it turns out to be empirically effective to keep increasing
    the slope at zero even beyond the point where the iteration no longer converges all the way to
    one everywhere on the interval. This iteration therefore does not produce UV^T but rather something
    like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.

    Computation runs in ``bfloat16`` and the result is returned in ``bfloat16`` by design (NS is
    stable in bf16, matching upstream Keller Jordan / Moonlight). The caller's in-place ``add_``
    upcasts the operand to the parameter dtype, so no cast-back to ``G.dtype`` is needed.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz.

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        wd: The weight decay.
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

        self._diag_done = False

    def _v2_diag(self, p) -> None:
        """Print (once, rank0) the param/grad/data types needed to implement the DTensor-aware v2.

        Gate with env var LLAMAFACTORY_MUON_DIAG=1 so it is opt-in.
        """
        self._diag_done = True
        if os.environ.get("LLAMAFACTORY_MUON_DIAG") != "1":
            return
        if not _is_rank0():
            return
        DT = _dtensor_cls()
        g = p.grad
        is_dt = (DT is not None) and isinstance(p, DT)
        is_g_dt = (DT is not None) and isinstance(g, DT)
        lines = ["[Muon v2-diag] === info for writing the DTensor-aware v2 ==="]
        lines.append(f"  param: type={type(p).__name__} is_DT={is_dt} shape={tuple(p.shape)}")
        if is_dt:
            lines.append(f"    placements={p.placements} device_mesh={p.device_mesh}")
            try:
                lines.append(f"    p.to_local().shape={tuple(p.to_local().shape)}")
            except Exception as e:  # noqa: BLE001
                lines.append(f"    p.to_local() ERR={e!r}")
        lines.append(f"  grad:  type={type(g).__name__} is_DT={is_g_dt} shape={tuple(g.shape)}")
        lines.append(f"    grad.has_full_tensor={hasattr(g, 'full_tensor')}")
        if is_g_dt:
            lines.append(f"    grad.placements={g.placements} grad.device_mesh={g.device_mesh}")
        lines.append(f"  p.data: type={type(p.data).__name__} shape={tuple(p.data.shape)}")
        lines.append(
            f"  compare: p.shape==p.data.shape ? {tuple(p.shape) == tuple(p.data.shape)} ; "
            f"grad.shape==p.data.shape ? {tuple(g.shape) == tuple(p.data.shape)}"
        )
        print("\n".join(lines), flush=True)

    def adjust_lr_for_muon(self, lr: float, param_shape: list[int]) -> float:
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as described in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Muon loop
            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates in distributed fashion
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if not self._diag_done:
                    self._v2_diag(p)

                state = self.state[p]

                # v2: under FSDP2, p.grad is a sharded DTensor. Newton-Schulz must run on
                # the FULL 2D matrix (running it on the local shard computes a partial Gram
                # matrix and the NS iteration diverges -> NaN). Momentum accumulation is
                # elementwise, so the momentum buffer is kept sharded (mirroring g's
                # placements -> 1/N memory and FSDP2-checkpoint-native); we all-gather only
                # for the NS step, then scatter the update back to the local shard.
                sharded = _is_dtensor(g)
                if sharded:
                    p_mesh, p_placements = p.device_mesh, p.placements
                else:
                    p_mesh = p_placements = None

                # momentum buffer mirrors g's sharding (sharded DTensor under FSDP2, plain
                # tensor otherwise); elementwise accumulation is correct on the local shard.
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g_use = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                # all-gather ONLY here: NS needs the full 2D matrix (Gram matrix X @ X.T).
                g_full = g_use.full_tensor() if sharded else g_use
                if g_full.ndim > 2:
                    g_full = g_full.view(g_full.size(0), -1)
                u_full = zeropower_via_newtonschulz5(g_full, steps=group["ns_steps"])

                # scale update (p.shape is the DTensor global shape -> correct A, B)
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay (in-place on the local shard; elementwise -> correct)
                p.data.mul_(1 - lr * wd)

                # apply update; scatter the full update back to the local shard under FSDP2
                if sharded:
                    u_dt = _distribute(u_full, p_mesh, p_placements)
                    p.data.add_(u_dt, alpha=-adjusted_lr)
                else:
                    p.data.add_(u_full, alpha=-adjusted_lr)

            # Adam backup
            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss
