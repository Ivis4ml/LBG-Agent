"""Tests for lbg.verdict.dsr (Bailey & López de Prado 2014)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lbg.verdict.dsr import (
    DSRResult,
    DEFAULT_DSR_THRESHOLD,
    EULER_MASCHERONI,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_returns,
    expected_max_sharpe_daily,
    norm_cdf,
    norm_ppf,
    probabilistic_sharpe_ratio,
)


# ---------- norm_cdf / norm_ppf ----------


def test_norm_cdf_known_values():
    # Standard reference values to ~6 decimal places.
    assert abs(norm_cdf(0.0) - 0.5) < 1e-12
    assert abs(norm_cdf(1.0) - 0.8413447461) < 1e-6
    assert abs(norm_cdf(-1.0) - 0.1586552539) < 1e-6
    assert abs(norm_cdf(1.96) - 0.9750021049) < 1e-6
    assert abs(norm_cdf(-1.96) - 0.0249978951) < 1e-6


def test_norm_ppf_known_values():
    assert abs(norm_ppf(0.5) - 0.0) < 1e-6
    assert abs(norm_ppf(0.975) - 1.959963985) < 1e-5
    assert abs(norm_ppf(0.025) - (-1.959963985)) < 1e-5
    assert abs(norm_ppf(0.95) - 1.644853627) < 1e-5


def test_norm_ppf_rejects_out_of_range():
    with pytest.raises(ValueError, match="p in"):
        norm_ppf(0.0)
    with pytest.raises(ValueError, match="p in"):
        norm_ppf(1.0)
    with pytest.raises(ValueError, match="p in"):
        norm_ppf(-0.1)


def test_norm_cdf_ppf_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(100):
        x = float(rng.uniform(-3, 3))
        p = norm_cdf(x)
        x_back = norm_ppf(p)
        assert abs(x - x_back) < 1e-5


# ---------- expected_max_sharpe ----------


def test_expected_max_sharpe_n_one_is_zero():
    """With N = 1 (no selection), the threshold is zero (no inflation)."""
    assert expected_max_sharpe_daily(n_trials=1, variance_of_trial_sharpes_daily=1.0) == 0.0


def test_expected_max_sharpe_zero_variance_is_zero():
    """If trial Sharpes don't vary, there's no selection bias to deflate."""
    assert expected_max_sharpe_daily(n_trials=100, variance_of_trial_sharpes_daily=0.0) == 0.0


def test_expected_max_sharpe_grows_with_n():
    """More trials → higher expected max-Sharpe under null."""
    v = 1.0
    thresholds = [
        expected_max_sharpe_daily(n_trials=n, variance_of_trial_sharpes_daily=v)
        for n in (2, 10, 100, 1000)
    ]
    assert thresholds[0] < thresholds[1] < thresholds[2] < thresholds[3]


def test_expected_max_sharpe_scales_sqrt_variance():
    """Doubling V should multiply threshold by √2."""
    t1 = expected_max_sharpe_daily(n_trials=100, variance_of_trial_sharpes_daily=1.0)
    t2 = expected_max_sharpe_daily(n_trials=100, variance_of_trial_sharpes_daily=2.0)
    assert abs(t2 - t1 * math.sqrt(2.0)) < 1e-9


def test_expected_max_sharpe_hand_value_n100_v1():
    """Hand calculation for N=100, V=1 daily.

    term1 = (1 - γ_em) * Φ^{-1}(1 - 1/100) = (1 - 0.5772) * Φ^{-1}(0.99) ≈ 0.4228 * 2.3263 ≈ 0.9836
    term2 = γ_em * Φ^{-1}(1 - 1/(100·e)) = 0.5772 * Φ^{-1}(0.996321) ≈ 0.5772 * 2.6824 ≈ 1.5484
    sum ≈ 2.5320, but the formula is a CONVEX combination, not a sum.
    Actually: term1 + term2 weighted by (1-γ) + γ = 1 → sr* ≈ 0.9836 + 1.5484... wait.

    Re-deriving with the actual additive form in the formula:
      SR* = √V * ((1-γ)·Φ^{-1}(1-1/N) + γ·Φ^{-1}(1-1/(N·e)))
            = 1 * ((1-0.5772)·2.3263 + 0.5772·2.6824)
            ≈ 0.4228·2.3263 + 0.5772·2.6824
            ≈ 0.9836 + 1.5484
            ≈ 2.5320

    So we expect SR* ≈ 2.53 daily. Wait that seems large — but for "max
    of 100 unit-variance Sharpe draws" expected max IS roughly 2.5, yes.
    """
    sr_star = expected_max_sharpe_daily(n_trials=100, variance_of_trial_sharpes_daily=1.0)
    # Hand calc → ~2.5320; allow 1% slack for the rational approximation.
    assert abs(sr_star - 2.5320) < 0.02


# ---------- probabilistic_sharpe_ratio ----------


def test_psr_normal_returns_threshold_zero():
    """With γ_3=0, γ_4=3, the variance term is 1 + ((3-1)/4)·SR² = 1 + SR²/2.

    PSR(SR*=0) = Φ(SR · √(T-1) / √(1 + SR²/2)).
    """
    sharpe_daily = 0.10
    n_obs = 1001
    psr = probabilistic_sharpe_ratio(sharpe_daily, 0.0, n_obs, skewness=0.0, kurtosis=3.0)
    variance_term = 1.0 + (sharpe_daily**2) / 2.0
    expected = norm_cdf(sharpe_daily * math.sqrt(n_obs - 1) / math.sqrt(variance_term))
    assert abs(psr - expected) < 1e-9


def test_psr_negative_skew_reduces_probability():
    """Negative skewness inflates Var(SR̂), so PSR < normal-case PSR."""
    psr_normal = probabilistic_sharpe_ratio(0.10, 0.0, 1001, skewness=0.0, kurtosis=3.0)
    psr_skew = probabilistic_sharpe_ratio(0.10, 0.0, 1001, skewness=-0.8, kurtosis=3.0)
    assert psr_skew < psr_normal


def test_psr_excess_kurtosis_reduces_probability():
    """Higher kurtosis (γ_4 > 3) inflates Var(SR̂), so PSR drops."""
    psr_normal = probabilistic_sharpe_ratio(0.10, 0.0, 1001, skewness=0.0, kurtosis=3.0)
    psr_fat = probabilistic_sharpe_ratio(0.10, 0.0, 1001, skewness=0.0, kurtosis=8.0)
    assert psr_fat < psr_normal


def test_psr_degenerate_returns_half():
    """T ≤ 1 or non-finite variance → return 0.5, don't crash."""
    assert probabilistic_sharpe_ratio(0.1, 0.0, 1, 0.0, 3.0) == 0.5
    # Pathological: γ_3 = 1000, SR̂ = 1 makes variance term negative.
    assert probabilistic_sharpe_ratio(1.0, 0.0, 100, skewness=1000.0, kurtosis=3.0) == 0.5


# ---------- deflated_sharpe_ratio ----------


def test_dsr_n_one_equals_psr_at_zero():
    """N = 1 → threshold = 0 → DSR equals PSR(SR* = 0)."""
    res = deflated_sharpe_ratio(
        0.10,
        n_obs=1001,
        skewness=0.0,
        kurtosis=3.0,
        n_trials=1,
        variance_of_trial_sharpes_daily=1.0,
    )
    assert abs(res.deflated_sharpe_ratio - res.psr_at_zero) < 1e-12
    assert res.sharpe_threshold_daily == 0.0


def test_dsr_larger_n_lowers_dsr():
    """More trials → higher SR* → DSR drops for the same observed Sharpe.

    Calibration: SR_daily = 0.20 and V = 0.01 keep all four threshold
    values in a regime where DSR is strictly between 0 and 1 (not
    floored at machine epsilon for the large-N case).
    """
    out = []
    for n in (1, 10, 100, 1000):
        res = deflated_sharpe_ratio(
            0.20,
            n_obs=1001,
            skewness=0.0,
            kurtosis=3.0,
            n_trials=n,
            variance_of_trial_sharpes_daily=0.01,
        )
        out.append(res.deflated_sharpe_ratio)
    assert out[0] > out[1] > out[2] > out[3]
    # And every value should be in the open unit interval (not floored).
    assert all(0.0 < x < 1.0 for x in out[1:])


def test_dsr_passes_threshold_for_strong_edge_low_n():
    """An observed SR clearly above any plausible null max → DSR > 0.95.

    With V_daily = 0.01 (modest cross-trial variance), the N = 2 null
    max-threshold is ≈ 0.052 daily; a daily Sharpe of 0.30 is well past
    it and DSR should pass.
    """
    res = deflated_sharpe_ratio(
        0.30,  # daily Sharpe ≈ 4.7 annualised
        n_obs=1001,
        skewness=0.0,
        kurtosis=3.0,
        n_trials=2,  # almost no selection adjustment
        variance_of_trial_sharpes_daily=0.01,
        pass_threshold=0.95,
    )
    assert res.passes
    assert res.deflated_sharpe_ratio > 0.95


def test_dsr_fails_threshold_for_marginal_edge_high_n():
    """A marginal SR drowned by selection across many trials → DSR < 0.95."""
    res = deflated_sharpe_ratio(
        0.03,  # daily SR ≈ 0.47 annualised, modest
        n_obs=1001,
        skewness=-0.3,
        kurtosis=5.0,
        n_trials=200,  # heavy selection
        variance_of_trial_sharpes_daily=1.0,
        pass_threshold=0.95,
    )
    assert not res.passes
    assert res.deflated_sharpe_ratio < 0.95


def test_dsr_result_summary_keys():
    res = deflated_sharpe_ratio(
        0.05,
        n_obs=500,
        skewness=0.0,
        kurtosis=3.0,
        n_trials=10,
        variance_of_trial_sharpes_daily=0.5,
    )
    s = res.as_summary()
    expected = {
        "sharpe_daily",
        "sharpe_threshold_daily",
        "n_obs",
        "skewness",
        "kurtosis",
        "n_trials",
        "variance_of_trial_sharpes_daily",
        "psr_at_zero",
        "deflated_sharpe_ratio",
        "passes",
        "pass_threshold",
    }
    assert set(s.keys()) == expected


def test_default_dsr_threshold_matches_paper():
    """Bailey & López de Prado 2014 use a 95% confidence threshold."""
    assert DEFAULT_DSR_THRESHOLD == 0.95


def test_euler_mascheroni_constant():
    """Sanity check the Euler-Mascheroni constant used in the formula."""
    # Reference value to 16 decimal places: 0.5772156649015329...
    assert abs(EULER_MASCHERONI - 0.5772156649015329) < 1e-15


# ---------- deflated_sharpe_ratio_from_returns ----------


def test_from_returns_converts_annualised_to_daily():
    """Annualised SR = 5.0 → daily SR = 5.0 / √252 ≈ 0.315."""
    rng = np.random.default_rng(0)
    diff = rng.normal(0.0, 0.01, 1000)
    res = deflated_sharpe_ratio_from_returns(
        diff,
        annualised_sharpe=5.0,
        n_trials=10,
        variance_of_trial_sharpes_annualised=252.0,  # = 1.0 daily
    )
    expected_daily_sharpe = 5.0 / math.sqrt(252)
    assert abs(res.sharpe_daily - expected_daily_sharpe) < 1e-9
    assert abs(res.variance_of_trial_sharpes_daily - 1.0) < 1e-9


def test_from_returns_estimates_moments_from_diff_series():
    """A diff series with engineered skew → DSR.skewness reflects that."""
    rng = np.random.default_rng(7)
    # Mix of normal + negative jumps to skew the distribution left.
    diff = rng.normal(0.0, 0.005, 1000)
    diff[::50] -= 0.03  # large negative shocks every 50 bars
    res = deflated_sharpe_ratio_from_returns(
        diff,
        annualised_sharpe=1.0,
        n_trials=10,
        variance_of_trial_sharpes_annualised=10.0,
    )
    # Engineered negative skewness.
    assert res.skewness < -0.5
    # Excess kurtosis from the jumps.
    assert res.kurtosis > 4.0


def test_from_returns_too_few_obs_returns_degenerate():
    res = deflated_sharpe_ratio_from_returns(
        np.array([0.0]),
        annualised_sharpe=2.0,
        n_trials=5,
        variance_of_trial_sharpes_annualised=1.0,
    )
    assert res.n_obs == 1
    assert res.deflated_sharpe_ratio == 0.5  # degenerate PSR


def test_from_returns_zero_variance_diff_handled():
    """If diff series is constant, std = 0 — module returns degenerate."""
    diff = np.zeros(500)
    res = deflated_sharpe_ratio_from_returns(
        diff,
        annualised_sharpe=0.0,
        n_trials=5,
        variance_of_trial_sharpes_annualised=1.0,
    )
    # Skewness and kurtosis defaults: 0 and 3 (normal-like) when std == 0.
    assert res.skewness == 0.0
    assert res.kurtosis == 3.0


def test_from_returns_returns_dsr_result_object():
    rng = np.random.default_rng(11)
    diff = rng.normal(0.001, 0.01, 600)
    res = deflated_sharpe_ratio_from_returns(
        diff,
        annualised_sharpe=1.5,
        n_trials=20,
        variance_of_trial_sharpes_annualised=5.0,
    )
    assert isinstance(res, DSRResult)
    assert 0.0 <= res.deflated_sharpe_ratio <= 1.0
    assert 0.0 <= res.psr_at_zero <= 1.0
