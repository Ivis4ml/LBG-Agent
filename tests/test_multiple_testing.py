"""Tests for lbg.verdict.multiple_testing.benjamini_hochberg."""

from __future__ import annotations

import numpy as np
import pytest

from lbg.verdict.multiple_testing import (
    DEFAULT_Q,
    BHResult,
    benjamini_hochberg,
)


def test_empty_input_returns_empty_result():
    res = benjamini_hochberg([])
    assert isinstance(res, BHResult)
    assert res.n_rejected == 0
    assert res.reject == ()
    assert res.p_values == ()


def test_q_must_be_in_open_unit_interval():
    with pytest.raises(ValueError, match="q must be in"):
        benjamini_hochberg([0.5], q=0.0)
    with pytest.raises(ValueError, match="q must be in"):
        benjamini_hochberg([0.5], q=1.0)
    with pytest.raises(ValueError, match="q must be in"):
        benjamini_hochberg([0.5], q=-0.1)


def test_out_of_range_p_value_raises():
    with pytest.raises(ValueError, match=r"p-values must lie in"):
        benjamini_hochberg([0.1, 1.5])
    with pytest.raises(ValueError, match=r"p-values must lie in"):
        benjamini_hochberg([0.1, -0.01])


def test_default_q_matches_proposal_lock():
    """PROPOSAL §20 item 32 locks BH at FDR = 0.10."""
    assert DEFAULT_Q == 0.10


def test_all_strong_rejects_all():
    """When every p is essentially zero, BH rejects every hypothesis."""
    res = benjamini_hochberg([1e-6, 1e-6, 1e-6, 1e-6, 1e-6], q=0.10)
    assert res.n_rejected == 5
    assert all(res.reject)


def test_all_null_rejects_none():
    """Uniform p-values near 1.0 should not be rejected."""
    res = benjamini_hochberg([0.9, 0.95, 0.99, 0.85, 0.92], q=0.10)
    assert res.n_rejected == 0
    assert not any(res.reject)


def test_known_case_two_of_five_rejected():
    """Hand-checked: p = [0.001, 0.008, 0.04, 0.20, 0.50], q = 0.10, m = 5.

    BH thresholds (sorted rank k/m * q): 0.02, 0.04, 0.06, 0.08, 0.10.
    Sorted p: 0.001 ≤ 0.02 ✓, 0.008 ≤ 0.04 ✓, 0.04 ≤ 0.06 ✓,
              0.20 ≤ 0.08 ✗, 0.50 ≤ 0.10 ✗.
    Largest k with p_(k) ≤ k*q/m is 3, so 3 rejected.
    """
    p = [0.001, 0.008, 0.04, 0.20, 0.50]
    res = benjamini_hochberg(p, q=0.10)
    assert res.n_rejected == 3
    assert res.reject == (True, True, True, False, False)


def test_input_order_preserved_under_shuffle():
    """The `reject` tuple is aligned with the *input* order, not sorted order."""
    p_sorted = [0.001, 0.008, 0.04, 0.20, 0.50]
    perm = [3, 1, 4, 0, 2]
    p_shuffled = [p_sorted[i] for i in perm]
    res_sorted = benjamini_hochberg(p_sorted, q=0.10)
    res_shuffled = benjamini_hochberg(p_shuffled, q=0.10)

    # n_rejected and the set of rejected p-values must match.
    assert res_sorted.n_rejected == res_shuffled.n_rejected
    rejected_p_sorted = {p_sorted[i] for i, r in enumerate(res_sorted.reject) if r}
    rejected_p_shuffled = {p_shuffled[i] for i, r in enumerate(res_shuffled.reject) if r}
    assert rejected_p_sorted == rejected_p_shuffled

    # And the per-input-position rank must follow the input order.
    assert res_shuffled.rank == tuple(
        sorted(range(len(p_shuffled)), key=lambda i: p_shuffled[i]).index(i) + 1
        for i in range(len(p_shuffled))
    )


def test_step_up_not_step_down():
    """Pathological case where a low-rank p exceeds its threshold but a
    higher-rank p clears its own threshold. BH is step-up: it should
    still reject everything at or below the largest passing rank.

    p = [0.01, 0.10, 0.02, 0.50], q = 0.20, m = 4.
    Sorted p: 0.01, 0.02, 0.10, 0.50.
    Thresholds: 0.05, 0.10, 0.15, 0.20.
    0.01 ≤ 0.05 ✓, 0.02 ≤ 0.10 ✓, 0.10 ≤ 0.15 ✓, 0.50 ≤ 0.20 ✗.
    Largest k = 3, so the first three (in sorted order) are rejected.
    Position-wise that means input positions 0, 1, 2 are True; 3 False.
    """
    res = benjamini_hochberg([0.01, 0.10, 0.02, 0.50], q=0.20)
    assert res.n_rejected == 3
    assert res.reject == (True, True, True, False)


def test_fdr_control_under_uniform_null_via_monte_carlo():
    """Empirical sanity check: under U(0,1) p-values, the expected
    fraction of false discoveries is ≤ q. This is the classical BH
    guarantee under independence; one Monte-Carlo experiment isn't a
    proof but a regression catches a procedure that drifted off.
    """
    rng = np.random.default_rng(20260527)
    n_trials = 500
    m = 50
    q = 0.10
    # FDR is defined as E[V / max(R, 1)] where V = false discoveries and
    # R = total rejections, averaged across replications. Under a full
    # null (every p is uniform), any rejection is false (V = R), so the
    # FDP per replication is either 0 (no rejections) or 1 (any
    # rejection). The expectation across many replications equals the
    # probability that BH rejects anything; under U(0, 1) this equals q
    # exactly (BH is conservative under independence, this is the tight
    # case).
    fdp_sum = 0.0
    for _ in range(n_trials):
        p = rng.uniform(0, 1, size=m).tolist()
        res = benjamini_hochberg(p, q=q)
        if res.n_rejected > 0:
            # Under full null, every rejection is false → FDP = 1 for
            # this replication.
            fdp_sum += 1.0

    fdr_empirical = fdp_sum / n_trials
    # Allow Monte-Carlo slack; standard error at n=500 ≈ sqrt(q(1-q)/500) ≈ 0.013.
    assert fdr_empirical <= q + 0.05, f"empirical FDR {fdr_empirical:.3f} exceeds {q} + slack"
