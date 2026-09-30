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

"""Precision tests for v1 sigmoid-based DPO loss."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.train.dpo.trainer import CustomDPOTrainer
from llamafactory.v1.trainers.dpo_trainer import DPOTrainer, compute_sigmoid_dpo_loss


# ==============================================================================
# Mock helpers
# ==============================================================================


def _make_mock_v1(
    pref_beta: float = 0.1,
    dpo_label_smoothing: float = 0.0,
    ld_alpha: float | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        pref_beta=pref_beta,
        dpo_label_smoothing=dpo_label_smoothing,
        ld_alpha=ld_alpha,
        device=torch.device("cpu"),
    )


def _make_mock_v0_dpo(beta: float = 0.1, label_smoothing: float = 0.0) -> SimpleNamespace:
    mock = SimpleNamespace()
    mock.beta = beta
    mock.label_smoothing = label_smoothing
    mock.reference_free = False
    mock.f_divergence_type = "reverse_kl"
    mock.f_divergence_params = None
    mock.accelerator = SimpleNamespace()
    mock.accelerator.device = torch.device("cpu")
    return mock


# ==============================================================================
# Fixed test inputs
# ==============================================================================

P_CHOSEN = torch.tensor([-3.0, -2.5, -4.0, -1.5])
P_REJECTED = torch.tensor([-5.0, -3.5, -6.0, -2.5])
R_CHOSEN = torch.tensor([-2.8, -2.3, -3.8, -1.4])
R_REJECTED = torch.tensor([-3.2, -2.7, -4.2, -1.8])


# ==============================================================================
# Test 1 — Core loss correctness (pure function ↔ v1 instance ↔ v0/TRL)
# ==============================================================================


def test_sigmoid_dpo_loss_correctness():
    """Comprehensive correctness check for compute_sigmoid_dpo_loss and its wrapper."""
    # ---- 1a: pure function matches instance method ----
    v1 = _make_mock_v1(pref_beta=0.1)
    actual = DPOTrainer._sigmoid_dpo_loss(v1, P_CHOSEN, P_REJECTED, R_CHOSEN, R_REJECTED)
    expected = compute_sigmoid_dpo_loss(P_CHOSEN, P_REJECTED, R_CHOSEN, R_REJECTED, beta=0.1)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    # ---- 1b: v1 matches v0 (TRL) on fixed inputs ----
    v0 = _make_mock_v0_dpo(beta=0.1)
    v0_losses, _, _ = CustomDPOTrainer.dpo_loss(
        v0,
        P_CHOSEN,
        P_REJECTED,
        R_CHOSEN,
        R_REJECTED,
        loss_type="sigmoid",
    )
    torch.testing.assert_close(actual, v0_losses, rtol=1e-6, atol=1e-6)

    # ---- 1c: multiple beta values (v1 ↔ v0) ----
    for beta in [0.01, 0.1, 0.5, 1.0]:
        v0b = _make_mock_v0_dpo(beta=beta)
        v1b = _make_mock_v1(pref_beta=beta)
        vl, _, _ = CustomDPOTrainer.dpo_loss(
            v0b,
            P_CHOSEN,
            P_REJECTED,
            R_CHOSEN,
            R_REJECTED,
            loss_type="sigmoid",
        )
        v1l = DPOTrainer._sigmoid_dpo_loss(v1b, P_CHOSEN, P_REJECTED, R_CHOSEN, R_REJECTED)
        torch.testing.assert_close(v1l, vl, rtol=1e-6, atol=1e-6)

    # ---- 1d: label_smoothing sweep (v1 ↔ v0) ----
    for ls in [0.0, 0.1, 0.2, 0.3]:
        v0s = _make_mock_v0_dpo(beta=0.1, label_smoothing=ls)
        v1s = _make_mock_v1(pref_beta=0.1, dpo_label_smoothing=ls)
        vl, _, _ = CustomDPOTrainer.dpo_loss(
            v0s,
            P_CHOSEN,
            P_REJECTED,
            R_CHOSEN,
            R_REJECTED,
            loss_type="sigmoid",
        )
        v1l = DPOTrainer._sigmoid_dpo_loss(v1s, P_CHOSEN, P_REJECTED, R_CHOSEN, R_REJECTED)
        torch.testing.assert_close(v1l, vl, rtol=1e-6, atol=1e-6)

    # ---- 1e: label_smoothing=0.5 symmetry (swap chosen↔rejected same loss) ----
    v1s = _make_mock_v1(pref_beta=0.1, dpo_label_smoothing=0.5)
    fwd = DPOTrainer._sigmoid_dpo_loss(v1s, P_CHOSEN, P_REJECTED, R_CHOSEN, R_REJECTED)
    swp = DPOTrainer._sigmoid_dpo_loss(v1s, P_REJECTED, P_CHOSEN, R_REJECTED, R_CHOSEN)
    torch.testing.assert_close(fwd, swp, rtol=1e-6, atol=1e-6)

    # ---- 1f: chosen better → lower loss ----
    v1c = _make_mock_v1(pref_beta=0.1)
    loss_good = DPOTrainer._sigmoid_dpo_loss(
        v1c,
        torch.tensor([-1.0]),
        torch.tensor([-10.0]),
        torch.tensor([-3.0]),
        torch.tensor([-3.0]),
    )
    loss_bad = DPOTrainer._sigmoid_dpo_loss(
        v1c,
        torch.tensor([-10.0]),
        torch.tensor([-1.0]),
        torch.tensor([-3.0]),
        torch.tensor([-3.0]),
    )
    assert loss_good.item() < loss_bad.item()

    # ---- 1g: policy == ref → loss = log(2) ≈ 0.693 ----
    logps = torch.tensor([-3.0, -2.0, -4.0])
    losses = DPOTrainer._sigmoid_dpo_loss(v1c, logps, logps, logps, logps)
    expected_log2 = torch.full_like(logps, -F.logsigmoid(torch.tensor(0.0)).item())
    torch.testing.assert_close(losses, expected_log2, rtol=1e-5, atol=1e-5)

    # ---- 1h: non-negative ----
    assert (actual >= 0).all()

    # ---- 1i: extreme logps stay finite ----
    v1x = _make_mock_v1(pref_beta=0.1)
    x = DPOTrainer._sigmoid_dpo_loss(
        v1x,
        torch.tensor([-0.1, -50.0, -0.5, -100.0]),
        torch.tensor([-0.2, -5.0, -30.0, -1.0]),
        torch.tensor([-0.15, -3.0, -0.6, -2.0]),
        torch.tensor([-0.25, -4.0, -5.0, -1.5]),
    )
    assert torch.isfinite(x).all()


# ==============================================================================
# Test 2 — Random cross-validation & reward equivalence
# ==============================================================================


def test_cross_validate_and_rewards():
    """Randomised v0↔v1 cross-validation (50 seeds) + reward-margin check."""
    torch.manual_seed(42)
    for _ in range(50):
        pc = -torch.rand(4) * 10 - 0.01
        pr = -torch.rand(4) * 15 - 0.01
        rc = -torch.rand(4) * 10 - 0.01
        rr = -torch.rand(4) * 12 - 0.01
        beta = 0.01 + torch.rand(1).item() * 0.5
        ls = torch.rand(1).item() * 0.3

        v0 = _make_mock_v0_dpo(beta=beta, label_smoothing=ls)
        v1 = _make_mock_v1(pref_beta=beta, dpo_label_smoothing=ls)

        v0_loss, _, _ = CustomDPOTrainer.dpo_loss(
            v0,
            pc,
            pr,
            rc,
            rr,
            loss_type="sigmoid",
        )
        v1_loss = DPOTrainer._sigmoid_dpo_loss(v1, pc, pr, rc, rr)
        torch.testing.assert_close(v1_loss, v0_loss, rtol=1e-5, atol=1e-5)

        # Reward margin = beta * (chosen_logratio - rejected_logratio)
        chosen_rewards = beta * (pc - rc)
        rejected_rewards = beta * (pr - rr)
        reward_margin = chosen_rewards - rejected_rewards
        logits = (pc - rc) - (pr - rr)
        torch.testing.assert_close(reward_margin, beta * logits, rtol=1e-6, atol=1e-6)

    # Fixed-input reward ordering
    cr = 0.1 * (P_CHOSEN - R_CHOSEN)
    rr = 0.1 * (P_REJECTED - R_REJECTED)
    assert (cr > rr).float().mean().item() == 1.0


# ==============================================================================
# Test 3 — End-to-end: log-prob extraction + synthetic batch + LD-DPO
# ==============================================================================


def _make_batch(num_pairs, seq_len, vocab_size, prompt_len=3, chosen_len=None, rejected_len=None):
    if chosen_len is None or rejected_len is None:
        rlen = (seq_len - prompt_len) // 2
        chosen_len = rlen
        rejected_len = rlen

    actual = prompt_len + chosen_len + rejected_len

    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (num_pairs, actual))
    labels = input_ids.clone()
    labels[:, :prompt_len] = IGNORE_INDEX

    token_type_ids = torch.zeros(num_pairs, actual, dtype=torch.long)
    token_type_ids[:, prompt_len : prompt_len + chosen_len] = 1
    token_type_ids[:, prompt_len + chosen_len :] = 2

    torch.manual_seed(99)
    logits = torch.randn(num_pairs, actual, vocab_size)
    return input_ids, labels, token_type_ids, logits


def test_logp_extraction_and_e2e_loss():
    """Log-prob extraction shapes + e2e sigmoid loss (equal & unequal lengths)."""
    # --- equal-length batch ---
    ids, labels, tt_ids, logits = _make_batch(2, 12, 64, prompt_len=2)
    v1 = _make_mock_v1(pref_beta=0.1)

    c_lp, r_lp, c_avg, r_avg = DPOTrainer._extract_chosen_rejected_logps(v1, logits, labels, tt_ids)
    assert c_lp.shape == r_lp.shape == c_avg.shape == r_avg.shape == (2,)
    assert (c_lp <= 1e-6).all() and (r_lp <= 1e-6).all()

    # Create "ref" logits with small noise
    torch.manual_seed(123)
    ref_logits = logits + 0.1 * torch.randn_like(logits)
    rc_lp, rr_lp, _, _ = DPOTrainer._extract_chosen_rejected_logps(v1, ref_logits, labels, tt_ids)

    losses = DPOTrainer._sigmoid_dpo_loss(v1, c_lp, r_lp, rc_lp, rr_lp)
    assert torch.isfinite(losses).all() and (losses >= 0).all()

    # --- unequal-length (LD-DPO) batch ---
    ids2, labels2, tt_ids2, logits2 = _make_batch(
        1,
        11,
        64,
        prompt_len=2,
        chosen_len=6,
        rejected_len=3,
    )
    v1_ld = _make_mock_v1(pref_beta=0.1, ld_alpha=0.5)

    c_lp2, r_lp2, _, _ = DPOTrainer._extract_chosen_rejected_logps(v1_ld, logits2, labels2, tt_ids2)

    torch.manual_seed(123)
    ref2 = logits2 + 0.1 * torch.randn_like(logits2)
    rc2, rr2, _, _ = DPOTrainer._extract_chosen_rejected_logps(v1_ld, ref2, labels2, tt_ids2)

    losses2 = DPOTrainer._sigmoid_dpo_loss(v1_ld, c_lp2, r_lp2, rc2, rr2)
    assert torch.isfinite(losses2).all()
