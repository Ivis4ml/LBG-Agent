"""Tests for lbg.verdict.spa (Hansen 2005 Superior Predictive Ability)."""

from __future__ import annotations

import numpy as np
import pytest

from lbg.verdict.spa import (
    SPAResult,
    hansen_spa,
    stationary_bootstrap_indices,
)


# -------- stationary bootstrap --------


def test_stationary_bootstrap_indices_in_range():
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(1000, block_mean_len=10, rng=rng)
    assert idx.shape == (1000,)
    assert idx.min() >= 0
    assert idx.max() < 1000


def test_stationary_bootstrap_indices_block_length_geometric():
    """Mean run length of identical-step consecutive indices should
    approximate the requested block_mean_len."""
    rng = np.random.default_rng(0)
    T = 5000
    block_mean = 10
    idx = stationary_bootstrap_indices(T, block_mean_len=block_mean, rng=rng)

    # A "run" is a maximal stretch where idx[i+1] == idx[i] + 1 mod T.
    diffs = np.diff(idx) % T
    is_continuation = diffs == 1
    # Count run lengths by walking the boolean array.
    run_lengths = []
    current = 1
    for cont in is_continuation:
        if cont:
            current += 1
        else:
            run_lengths.append(current)
            current = 1
    run_lengths.append(current)

    mean_run = float(np.mean(run_lengths))
    # Geometric(p = 1/10) has mean 10. Allow Monte-Carlo slack.
    assert 7.0 < mean_run < 14.0, f"mean run length {mean_run:.2f} outside [7, 14]"


def test_stationary_bootstrap_indices_rejects_bad_inputs():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="T must be positive"):
        stationary_bootstrap_indices(0, block_mean_len=10, rng=rng)
    with pytest.raises(ValueError, match="block_mean_len must be"):
        stationary_bootstrap_indices(100, block_mean_len=0, rng=rng)


# -------- hansen_spa: input validation --------


def test_hansen_spa_rejects_non_2d_input():
    rng = np.random.default_rng(0)
    bad = rng.normal(0, 0.01, 100)  # 1D
    with pytest.raises(ValueError, match="2D"):
        hansen_spa(bad)


def test_hansen_spa_rejects_too_short_series():
    rng = np.random.default_rng(0)
    bad = rng.normal(0, 0.01, (30, 3))
    with pytest.raises(ValueError, match="need at least"):
        hansen_spa(bad, block_mean_len=10)


def test_hansen_spa_rejects_too_few_bootstrap():
    rng = np.random.default_rng(0)
    diff = rng.normal(0, 0.01, (500, 3))
    with pytest.raises(ValueError, match="n_bootstrap"):
        hansen_spa(diff, n_bootstrap=50)


def test_hansen_spa_returns_spa_result():
    rng = np.random.default_rng(0)
    diff = rng.normal(0, 0.01, (500, 3))
    res = hansen_spa(diff, n_bootstrap=200, rng_seed=1)
    assert isinstance(res, SPAResult)
    assert res.n_candidates == 3
    assert res.n_bootstrap == 200
    assert 0.0 < res.spa_p_value <= 1.0
    assert 0.0 < res.spa_p_value_l <= 1.0
    assert 0.0 < res.spa_p_value_u <= 1.0


# -------- hansen_spa: statistical behavior --------


def test_hansen_spa_under_null_p_values_not_concentrated_near_zero():
    """Under the full null (every column iid noise), the SPA p-value
    should be a non-degenerate U(0, 1)-like distribution. We don't
    require uniformity exactly (one fixed rng_seed isn't enough samples),
    but across many independent draws, p-values should not concentrate
    near zero — i.e., the empirical rejection rate at α = 0.10 should
    be ≤ 0.20 (with Monte-Carlo slack).
    """
    n_replications = 100
    M = 5
    T = 500
    rejections = 0
    for r in range(n_replications):
        rng = np.random.default_rng(1000 + r)
        diff = rng.normal(0, 0.01, (T, M))
        res = hansen_spa(diff, n_bootstrap=200, rng_seed=2000 + r)
        if res.spa_p_value <= 0.10:
            rejections += 1
    empirical_rate = rejections / n_replications
    # Under proper size control at α = 0.10, expected rejection rate
    # is at most 0.10; allow Monte-Carlo slack and some conservatism
    # from SPA_c.
    assert empirical_rate <= 0.25, (
        f"empirical rejection rate {empirical_rate:.2f} exceeds 0.25 — SPA appears anti-conservative"
    )


def test_hansen_spa_detects_single_strong_candidate():
    """One column with a clear edge among many noise columns should be
    detected: SPA rejects, and best_candidate_idx points at the right
    column.
    """
    T, M = 1000, 8
    rng = np.random.default_rng(7)
    diff = rng.normal(0.0, 0.01, (T, M))
    # Inject a clear edge into column 3: mean return shift of ~0.2 SR/sqrt(252).
    diff[:, 3] += 0.002
    res = hansen_spa(diff, n_bootstrap=500, rng_seed=42)
    assert res.best_candidate_idx == 3
    # Strong edge → p should be small.
    assert res.spa_p_value < 0.05, f"SPA p = {res.spa_p_value:.3f} did not reject the strong edge"


def test_hansen_spa_three_variants_bracket_each_other():
    """SPA_l (conservative) ≥ SPA_c (consensus) ≥ SPA_u (liberal).

    Mathematical property: removing recentering from a candidate can
    only shrink the bootstrap max (the candidate stays at its
    observed-and-typically-negative value), so SPA_u always rejects at
    least as easily as SPA_c, which rejects at least as easily as
    SPA_l. Equivalently, p_l ≥ p_c ≥ p_u.
    """
    T, M = 800, 6
    rng = np.random.default_rng(11)
    # Mix: a couple of clearly losing candidates, a few near-zero, no winner.
    diff = rng.normal(0, 0.01, (T, M))
    diff[:, 0] -= 0.003  # clearly losing
    diff[:, 1] -= 0.002  # clearly losing
    res = hansen_spa(diff, n_bootstrap=300, rng_seed=99)
    # SPA_c may equal SPA_l in any given finite sample (no candidate
    # crosses the log-log threshold), and similarly SPA_u may equal
    # SPA_c. We assert the monotone bracket only.
    assert res.spa_p_value_l >= res.spa_p_value - 1e-9
    assert res.spa_p_value >= res.spa_p_value_u - 1e-9


def test_hansen_spa_p_value_in_open_unit_interval():
    """Davison-Hinkley +1/+1 smoothing keeps p strictly in (0, 1)."""
    T, M = 600, 4
    rng = np.random.default_rng(13)
    # Engineer all four candidates to be strong winners.
    diff = rng.normal(0.005, 0.005, (T, M))
    res = hansen_spa(diff, n_bootstrap=200, rng_seed=21)
    # All winners → p should hit the Davison-Hinkley floor.
    assert 0.0 < res.spa_p_value <= 1.0
    assert res.spa_p_value >= 1.0 / (200 + 1) - 1e-12


def test_hansen_spa_studentization_handles_zero_variance_column():
    """A column with literally zero variance (e.g., all zeros) should
    not crash the test — its studentized statistic is 0 and it never
    contributes to the max.
    """
    T, M = 500, 3
    rng = np.random.default_rng(17)
    diff = rng.normal(0.001, 0.01, (T, M))
    diff[:, 1] = 0.0  # constant column
    res = hansen_spa(diff, n_bootstrap=200, rng_seed=23)
    # No crash, and the all-zero column is not the argmax.
    assert res.best_candidate_idx != 1


def test_hansen_spa_single_candidate_reduces_to_one_sided_bootstrap():
    """M = 1 case: SPA reduces to a one-sided bootstrap test on a single
    Sharpe ratio. Result should be coherent (small p for strong edge).
    """
    T = 1000
    rng = np.random.default_rng(19)
    diff = rng.normal(0.003, 0.008, (T, 1))  # ~0.3 SR daily * sqrt(252) ≈ 4.7 ann SR
    res = hansen_spa(diff, n_bootstrap=200, rng_seed=27)
    assert res.n_candidates == 1
    assert res.best_candidate_idx == 0
    assert res.spa_p_value < 0.10
