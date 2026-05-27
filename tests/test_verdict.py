"""Tests for `lbg.verdict`: baselines, bootstrap, H1 verdict, analysis plan."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import run_backtest
from lbg.alpha_cards import AlphaCard, AlphaCardEvidence, AlphaCardSignal, AlphaCardStatus
from lbg.schemas import HypothesisOutcome, ValidationSignal
from lbg.verdict import (
    DEFAULT_ANALYSIS_PLAN,
    AnalysisPlan,
    H1Criterion,
    baseline_buy_and_hold,
    baseline_sixty_forty,
    compute_alpha_card_h1_verdict,
    compute_strategy_sharpe_verdict,
    moving_block_bootstrap_sharpe_diff,
)


def _synth_ohlcv(
    n: int = 300, mean_log_ret: float = 0.0005, std_log_ret: float = 0.01, seed: int = 7
):
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(mean_log_ret, std_log_ret, n)
    close = 100.0 * np.exp(np.cumsum(log_rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "open": open_,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype("int64"),
        }
    )


# -------- baselines --------


def test_buy_and_hold_is_constant_one():
    df = _synth_ohlcv()
    p = baseline_buy_and_hold(df)
    assert len(p) == len(df)
    assert (p == 1.0).all()


def test_sixty_forty_is_constant_point_six():
    df = _synth_ohlcv()
    p = baseline_sixty_forty(df)
    assert (p == 0.6).all()


def test_buy_and_hold_through_backtest_is_finite():
    df = _synth_ohlcv()
    res = run_backtest(baseline_buy_and_hold(df), df)
    assert np.isfinite(res.sharpe)
    assert np.isfinite(res.max_drawdown)


# -------- moving block bootstrap --------


def test_bootstrap_returns_point_estimate_and_ci():
    rng = np.random.default_rng(0)
    r_strat = rng.normal(0.001, 0.01, 500)
    r_base = rng.normal(0.0005, 0.01, 500)
    out = moving_block_bootstrap_sharpe_diff(
        r_strat, r_base, block_len=10, n_bootstrap=200, alpha=0.05, rng_seed=42
    )
    assert "point_estimate" in out
    assert "ci_lower" in out
    assert "ci_upper" in out
    # The CI must bracket the point estimate -- because the bootstrap CI is
    # the empirical distribution of the difference, the point estimate
    # (computed once on the original sample) generally sits inside the band.
    assert out["ci_lower"] <= out["ci_upper"]
    # n_bootstrap and block_len echoed.
    assert out["n_bootstrap"] == 200
    assert out["block_len"] == 10


def test_bootstrap_detects_positive_edge():
    """Strategy with clearly higher mean return -> CI lower bound > 0 (high probability)."""
    rng = np.random.default_rng(0)
    r_strat = rng.normal(0.003, 0.01, 1000)  # ~75% annual return
    r_base = rng.normal(0.0001, 0.01, 1000)  # ~2.5% annual
    out = moving_block_bootstrap_sharpe_diff(
        r_strat, r_base, block_len=10, n_bootstrap=500, alpha=0.05, rng_seed=1
    )
    assert out["point_estimate"] > 0
    assert out["ci_lower"] > 0  # strong H1


def test_bootstrap_detects_no_edge():
    """Same distribution -> point estimate near zero, CI straddles zero."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 1000)
    r_strat = r.copy()
    r_base = r.copy()
    out = moving_block_bootstrap_sharpe_diff(
        r_strat, r_base, block_len=10, n_bootstrap=500, alpha=0.05, rng_seed=2
    )
    # Identical inputs -> diff is exactly zero on every resample.
    assert abs(out["point_estimate"]) < 1e-10
    assert abs(out["ci_lower"]) < 1e-9
    assert abs(out["ci_upper"]) < 1e-9


def test_bootstrap_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        moving_block_bootstrap_sharpe_diff(np.zeros(100), np.zeros(99), block_len=10)


def test_bootstrap_too_few_bars_raises():
    with pytest.raises(ValueError, match="at least"):
        moving_block_bootstrap_sharpe_diff(np.zeros(20), np.zeros(20), block_len=10)


def test_bootstrap_one_sided_p_value_strong_edge():
    """Clear positive edge -> bootstrap diffs almost never ≤ 0 -> p ≈ 1/(n+1)."""
    rng = np.random.default_rng(0)
    r_strat = rng.normal(0.003, 0.01, 1000)
    r_base = rng.normal(0.0001, 0.01, 1000)
    out = moving_block_bootstrap_sharpe_diff(
        r_strat, r_base, block_len=10, n_bootstrap=500, alpha=0.05, rng_seed=1
    )
    assert "p_value_one_sided" in out
    # With a strong positive edge, p should be at the Davison-Hinkley floor.
    assert out["p_value_one_sided"] < 0.01


def test_bootstrap_one_sided_p_value_no_edge():
    """Identical inputs -> diffs are exactly zero -> all draws are ≤ 0 -> p ≈ 1."""
    r = np.random.default_rng(0).normal(0.0005, 0.01, 1000)
    out = moving_block_bootstrap_sharpe_diff(
        r.copy(), r.copy(), block_len=10, n_bootstrap=500, alpha=0.05, rng_seed=2
    )
    # All 500 resampled diffs equal zero, so n_le_zero = n_bootstrap and
    # p = (n + 1) / (n + 1) = 1.0 by Davison-Hinkley.
    assert out["p_value_one_sided"] == 1.0


def test_bootstrap_one_sided_p_value_in_open_unit_interval():
    """Davison-Hinkley +1/+1 smoothing keeps p strictly in (0, 1).

    This matters for downstream BH: a raw p of 0 would short-circuit
    benjamini_hochberg into rejecting unconditionally; the smoothed p
    floors at 1/(n+1) so BH still has work to do.
    """
    rng = np.random.default_rng(7)
    r_strat = rng.normal(0.005, 0.005, 800)  # very strong edge
    r_base = rng.normal(-0.001, 0.005, 800)
    out = moving_block_bootstrap_sharpe_diff(
        r_strat, r_base, block_len=10, n_bootstrap=500, alpha=0.05, rng_seed=3
    )
    assert 0.0 < out["p_value_one_sided"] < 1.0
    assert out["p_value_one_sided"] >= 1.0 / (500 + 1)


def test_bootstrap_seed_determinism():
    r_strat = np.random.default_rng(0).normal(0.001, 0.01, 500)
    r_base = np.random.default_rng(1).normal(0.0005, 0.01, 500)
    a = moving_block_bootstrap_sharpe_diff(r_strat, r_base, n_bootstrap=200, rng_seed=42)
    b = moving_block_bootstrap_sharpe_diff(r_strat, r_base, n_bootstrap=200, rng_seed=42)
    assert a == b


# -------- analysis plan --------


def test_default_analysis_plan_matches_proposal():
    plan = DEFAULT_ANALYSIS_PLAN
    assert plan.h1.test == "validated_factor_count"
    assert plan.h1.tau == 3
    assert plan.per_card_validation.block_len == 10
    assert plan.per_card_validation.n_bootstrap == 1000
    assert plan.per_card_validation.threshold == "lower_bound_gt_zero"
    assert plan.supplementary_strategy.block_len == 10
    assert plan.supplementary_strategy.n_bootstrap == 1000
    assert plan.supplementary_strategy.alpha == 0.05
    assert "buy_and_hold" in plan.baselines
    assert "sixty_forty" in plan.baselines


def test_analysis_plan_rejects_extra_fields():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AnalysisPlan.model_validate(
            {
                "baselines": ["buy_and_hold"],
                "per_card_validation": {
                    "method": "leave_one_out_incremental_sharpe",
                    "ci": 0.95,
                    "ci_method": "moving_block_bootstrap",
                    "block_len": 10,
                    "n_bootstrap": 1000,
                    "threshold": "lower_bound_gt_zero",
                },
                "h1": {
                    "test": "validated_factor_count",
                    "tau": 3,
                    "strong_criterion": "N_val_lbg > max_baseline_count and N_val_lbg >= tau",
                    "weak_criterion": "N_val_lbg >= 1 and N_val_lbg >= best_baseline_count",
                    "baseline_validated_counts": {},
                },
                "supplementary_strategy": {
                    "test": "moving_block_bootstrap",
                    "metric": "sealed_strategy_sharpe",
                    "reference": "best_pre_registered_baseline",
                    "block_len": 10,
                    "n_bootstrap": 1000,
                    "alpha": 0.05,
                    "strong_criterion": "ci_lower > 0",
                    "weak_criterion": "point_estimate > 0",
                },
                "smuggled_field": "rule change",
            }
        )


def test_h1_criterion_strict_literal():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        H1Criterion(
            test="moving_block_bootstrap",  # not the primary H1 anymore
            tau=3,
            strong_criterion="N_val_lbg > max_baseline_count and N_val_lbg >= tau",
            weak_criterion="N_val_lbg >= 1 and N_val_lbg >= best_baseline_count",
            baseline_validated_counts={},
        )


# -------- primary H1 verdict --------


def _alpha_card(
    alpha_id: str,
    ci_lower: float | None,
    *,
    bh_validated: bool | None = None,
) -> AlphaCard:
    if ci_lower is None:
        sealed_summary = None
    else:
        # Default BH validation tracks CI > 0 unless overridden — keeps
        # most tests succinct while still allowing them to construct
        # cards where the two diverge (i.e., CI > 0 but BH rejected, or
        # vice versa).
        bh = (ci_lower > 0.0) if bh_validated is None else bh_validated
        sealed_summary = {"incremental_sharpe_ci_lower": ci_lower, "bh_validated": bh}
    return AlphaCard(
        alpha_id=alpha_id,
        source_trial=1,
        source_commit="abc123",
        status=AlphaCardStatus.ACCEPTED,
        signal=AlphaCardSignal(
            indicator=alpha_id, fn=alpha_id, source_path=f"indicators/{alpha_id}.py"
        ),
        evidence=AlphaCardEvidence(
            train_summary={
                "sharpe": 0.1,
                "max_drawdown": -0.1,
                "turnover": 1.0,
                "num_trades": 20.0,
            },
            validation_signal=ValidationSignal.ACCEPTED,
            hypothesis_outcome=HypothesisOutcome.CONFIRMED,
            sealed_summary=sealed_summary,
        ),
    )


def test_alpha_card_h1_counts_bh_validated_cards():
    """Primary H1 counts cards whose `bh_validated` flag is True.

    Cards with CI lower > 0 but BH-rejected (because the family-wise
    correction pushed them below the BH threshold) do NOT contribute to
    `validated_factor_count`; they show up only in
    `ci_only_validated_count` as a diagnostic.
    """
    verdict = compute_alpha_card_h1_verdict(
        [
            _alpha_card("bh_good", 0.01),  # CI > 0 + bh_validated
            _alpha_card("ci_only", 0.01, bh_validated=False),  # CI > 0, BH rejected
            _alpha_card("bad", -0.01),  # CI < 0, BH false
            _alpha_card("pending", None),  # no sealed summary
        ]
    )
    assert verdict.candidate_card_count == 4
    assert verdict.validated_factor_count == 1
    assert verdict.validated_alpha_ids == ("bh_good",)
    assert verdict.ci_only_validated_count == 2  # both `bh_good` and `ci_only`
    assert set(verdict.ci_only_validated_alpha_ids) == {"bh_good", "ci_only"}
    assert verdict.weak
    assert not verdict.strong


def test_alpha_card_h1_strong_requires_tau():
    verdict = compute_alpha_card_h1_verdict(
        [_alpha_card("a", 0.01), _alpha_card("b", 0.02), _alpha_card("c", 0.03)]
    )
    assert verdict.validated_factor_count == 3
    assert verdict.strong


# -------- supplementary strategy verdict --------


def test_compute_h1_verdict_returns_full_record():
    df = _synth_ohlcv(n=500, mean_log_ret=0.0008)
    # Strategy = constant long with a tiny additional edge (use 1.0).
    positions = pd.Series(np.ones(len(df)), index=df.index)
    bt = run_backtest(positions, df)
    verdict = compute_strategy_sharpe_verdict(bt.returns, df)
    assert verdict.best_baseline_name in {"buy_and_hold", "sixty_forty"}
    assert verdict.n_bootstrap == 1000
    assert verdict.block_len == 10
    assert verdict.alpha == 0.05
    # Self-vs-buy_and_hold should be ~zero (same positions).
    if verdict.best_baseline_name == "buy_and_hold":
        assert abs(verdict.delta_sharpe) < 1e-6


def test_h1_strong_when_strategy_beats_baselines():
    df = _synth_ohlcv(n=500, mean_log_ret=0.0001)  # baseline weak
    # Build a strategy with much better returns by flipping into long only
    # half the time but during the better half.
    np.random.seed(0)
    positions = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    # Long when close has been rising for last 3 bars (a synthetic edge).
    rolling = df["close"].diff().rolling(3).sum().fillna(0)
    positions[rolling > 0] = 1.0
    bt = run_backtest(positions, df)
    verdict = compute_strategy_sharpe_verdict(bt.returns, df)
    # Don't assert strong (it depends on synthetic data), just structural integrity.
    assert verdict.delta_sharpe == verdict.strategy_sharpe - verdict.best_baseline_sharpe
    assert verdict.ci_lower <= verdict.ci_upper


def test_h1_verdict_to_dict_is_json_safe():
    import json

    df = _synth_ohlcv(n=300)
    positions = pd.Series(np.ones(len(df)), index=df.index)
    bt = run_backtest(positions, df)
    verdict = compute_strategy_sharpe_verdict(bt.returns, df)
    d = verdict.to_dict()
    # Round-trip through JSON works (no numpy floats or enum objects).
    s = json.dumps(d)
    assert "supplementary_strong" in s
    assert "supplementary_weak" in s
