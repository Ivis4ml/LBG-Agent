"""Tests for `lbg.gate`: complexity, utility LCB, pareto, decide, redact."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import BacktestResult
from lbg.dsl import (
    CrossAboveRule,
    CrossBelowRule,
    FixedFractionSizing,
    IndicatorAboveFilter,
    IndicatorSpec,
    Strategy,
)
from lbg.gate import (
    GateConfig,
    GateDecision,
    TrialOutcome,
    complexity_score,
    decide,
    pareto_dominates,
    redact,
    utility_lcb_sharpe,
)
from lbg.gate.decision import (
    REASON_COMPLEXITY_INCREASE,
    REASON_DRAWDOWN_REGRESSION,
    REASON_NO_MEANINGFUL_IMPROVEMENT,
    REASON_OVERFIT_GAP,
    REASON_PARETO_IMPROVEMENT,
    REASON_TOO_FEW_TRADES,
    REASON_TURNOVER_TOO_HIGH,
    REASON_UTILITY_IMPROVEMENT,
)
from lbg.schemas import ValidationSignal

# -------- complexity --------


def _strategy_with(indicators: list[IndicatorSpec], filters: list = None) -> Strategy:
    return Strategy(
        name="x",
        indicators=indicators,
        entry=CrossAboveRule(rule="cross_above", fast=indicators[0].name, slow=indicators[-1].name),
        exit=CrossBelowRule(rule="cross_below", fast=indicators[0].name, slow=indicators[-1].name),
        filters=filters or [],
        sizing=FixedFractionSizing(mode="fixed_fraction", fraction=1.0, max_position=1.0),
    )


def test_complexity_two_indicator_baseline(tmp_path):
    """With no indicator files visible (empty indicators_dir), LOC/branch
    contributions are zero, so the count terms dominate."""
    s = _strategy_with(
        [
            IndicatorSpec(name="a", fn="sma", params={"period": 20}),
            IndicatorSpec(name="b", fn="sma", params={"period": 50}),
        ]
    )
    # 2 indicators * 1.0 + 0 filters + 2 params * 0.1 = 2.2
    assert complexity_score(s, indicators_dir=tmp_path) == pytest.approx(2.2)


def test_complexity_increases_with_filter(tmp_path):
    s_base = _strategy_with(
        [
            IndicatorSpec(name="a", fn="sma", params={"period": 20}),
            IndicatorSpec(name="b", fn="sma", params={"period": 50}),
        ]
    )
    s_filtered = _strategy_with(
        [
            IndicatorSpec(name="a", fn="sma", params={"period": 20}),
            IndicatorSpec(name="b", fn="sma", params={"period": 50}),
        ],
        filters=[IndicatorAboveFilter(rule="indicator_above", indicator="a", threshold=0.0)],
    )
    delta = complexity_score(s_filtered, indicators_dir=tmp_path) - complexity_score(
        s_base, indicators_dir=tmp_path
    )
    assert delta == pytest.approx(0.5)


def test_complexity_param_count_matters(tmp_path):
    s1 = _strategy_with([IndicatorSpec(name="a", fn="sma", params={"period": 20})])
    s3 = _strategy_with(
        [IndicatorSpec(name="a", fn="sma", params={"period": 20, "skip": 1, "weight": 0.5})]
    )
    delta = complexity_score(s3, indicators_dir=tmp_path) - complexity_score(
        s1, indicators_dir=tmp_path
    )
    assert delta == pytest.approx(0.2)


def test_complexity_loc_and_branches_contribute(tmp_path):
    """An indicator file with N effective LOC and B branch nodes adds
    λ_4*N + λ_5*B to the complexity score."""
    src = (
        "import pandas as pd\n"  # 1 LOC
        "\n"
        "# trailing comment-only line is not counted\n"
        "def my_ind(df, period=20):\n"  # 1 LOC
        "    if period < 1:\n"  # 1 LOC, 1 branch (If)
        "        raise ValueError('bad')\n"  # 1 LOC
        "    return df['close'].rolling(period).mean()\n"  # 1 LOC
    )
    # 5 effective LOC, 1 branch
    (tmp_path / "my_ind.py").write_text(src)

    s_one = _strategy_with([IndicatorSpec(name="a", fn="my_ind", params={"period": 20})])
    s_two = _strategy_with(
        [
            IndicatorSpec(name="a", fn="my_ind", params={"period": 20}),
            IndicatorSpec(name="b", fn="my_ind", params={"period": 50}),
        ]
    )
    # Same fn referenced twice does not double-count LOC/branches.
    score_one = complexity_score(s_one, indicators_dir=tmp_path)
    score_two = complexity_score(s_two, indicators_dir=tmp_path)
    # delta = +1 indicator + 1 param = 1.0 + 0.1 = 1.1 (no extra LOC/branches)
    assert score_two - score_one == pytest.approx(1.1)
    # The LOC + branch contribution to score_one is 5*0.005 + 1*0.05 = 0.075.
    # Count terms: 1*1.0 + 0*0.5 + 1*0.1 = 1.1. Total: 1.175.
    assert score_one == pytest.approx(1.175)


def test_complexity_branches_count_each_control_flow_node(tmp_path):
    src = (
        "import pandas as pd\n"
        "def x(df):\n"
        "    out = []\n"
        "    for i in range(len(df)):\n"  # branch 1: For
        "        if i % 2 == 0:\n"  # branch 2: If
        "            out.append(1)\n"
        "        else:\n"
        "            out.append(0)\n"
        "    return pd.Series(out)\n"
    )
    (tmp_path / "x.py").write_text(src)
    s = _strategy_with([IndicatorSpec(name="a", fn="x", params={})])
    # 9 LOC, 2 branches (For + If; else is part of the same If).
    score = complexity_score(s, indicators_dir=tmp_path)
    # 1 indicator * 1.0 + 0 filters + 0 params + 9 LOC * 0.005 + 2 branch * 0.05
    # = 1.0 + 0.045 + 0.10 = 1.145
    assert score == pytest.approx(1.145)


def test_complexity_missing_indicator_file_zero_loc(tmp_path):
    """If the .py file isn't found, LOC/branch contributions collapse to 0."""
    s = _strategy_with([IndicatorSpec(name="a", fn="absent", params={"period": 1})])
    score = complexity_score(s, indicators_dir=tmp_path)
    # 1 indicator + 1 param = 1.0 + 0.1 = 1.1; no LOC/branch terms.
    assert score == pytest.approx(1.1)


def test_complexity_repo_baseline_includes_real_sma_terms():
    """When pointed at the actual indicators/ dir, the score does pick up
    LOC + branches from indicators/sma.py."""
    s = _strategy_with(
        [
            IndicatorSpec(name="sma_fast", fn="sma", params={"period": 20}),
            IndicatorSpec(name="sma_slow", fn="sma", params={"period": 50}),
        ]
    )
    score_with_files = complexity_score(s)  # default `indicators/`
    # count terms = 2.2; total must be > 2.2 because sma.py has LOC + 1 branch.
    assert score_with_files > 2.2


# -------- utility LCB --------


def test_utility_lcb_short_series_is_zero():
    assert utility_lcb_sharpe(pd.Series([0.001] * 5)) == 0.0


def test_utility_lcb_zero_volatility_is_zero():
    assert utility_lcb_sharpe(pd.Series([0.001] * 100)) == 0.0


def test_utility_lcb_below_point_estimate():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 1000))
    lcb = utility_lcb_sharpe(r)
    point = float(r.mean() / r.std()) * np.sqrt(252)
    assert lcb < point  # one-sided lower bound is strictly less


def test_utility_lcb_unknown_alpha_requires_explicit_z():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 200))
    with pytest.raises(ValueError, match="explicit"):
        utility_lcb_sharpe(r, alpha=0.05)


# -------- gate decision: helpers --------


def _returns(n: int = 200, mean: float = 0.0005, std: float = 0.01, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, n))


def _bk(
    *,
    sharpe: float = 0.5,
    max_dd: float = -0.15,
    turnover: float = 3.0,
    num_trades: int = 30,
    returns: pd.Series | None = None,
) -> BacktestResult:
    r = returns if returns is not None else _returns()
    return BacktestResult(
        sharpe=sharpe,
        max_drawdown=max_dd,
        turnover=turnover,
        num_trades=num_trades,
        n_bars_used=len(r),
        cost_total=0.01,
        cagr=0.06,
        final_equity=1.5,
        returns=r,
        equity=pd.Series((1.0 + r).cumprod()),
    )


def _outcome(
    train_kw: dict | None = None,
    val_kw: dict | None = None,
    complexity: float = 2.2,
) -> TrialOutcome:
    return TrialOutcome(
        train=_bk(**(train_kw or {})),
        validation=_bk(**(val_kw or {})),
        complexity=complexity,
    )


# -------- gate decision: each reject path --------


def test_too_few_trades_rejected_first():
    incumbent = _outcome()
    candidate = _outcome(train_kw={"num_trades": 5})
    d = decide(incumbent, candidate)
    assert not d.accepted
    assert d.reason == REASON_TOO_FEW_TRADES


def test_drawdown_regression_rejected():
    incumbent = _outcome(val_kw={"max_dd": -0.10})
    # candidate val drawdown -0.20 < -0.10 * 1.15 = -0.115 -> rejected
    candidate = _outcome(val_kw={"max_dd": -0.20})
    d = decide(incumbent, candidate)
    assert not d.accepted
    assert d.reason == REASON_DRAWDOWN_REGRESSION


def test_turnover_too_high_rejected():
    incumbent = _outcome()
    candidate = _outcome(val_kw={"turnover": 12.0})
    cfg = GateConfig(max_turnover=10.0)
    d = decide(incumbent, candidate, cfg)
    assert not d.accepted
    assert d.reason == REASON_TURNOVER_TOO_HIGH


def test_complexity_increase_rejected():
    incumbent = _outcome(complexity=2.0)
    candidate = _outcome(complexity=6.0)
    cfg = GateConfig(max_complexity_delta=3.0)
    d = decide(incumbent, candidate, cfg)
    assert not d.accepted
    assert d.reason == REASON_COMPLEXITY_INCREASE
    assert d.complexity_delta == pytest.approx(4.0)


# -------- gate decision: each accept path --------


def test_utility_improvement_accepted():
    """Candidate has clearly better mean returns -> higher LCB -> accept."""
    incumbent_returns = _returns(mean=0.0001, std=0.01, seed=10)
    candidate_returns = _returns(mean=0.001, std=0.01, seed=11)
    incumbent = _outcome(val_kw={"returns": incumbent_returns})
    candidate = _outcome(val_kw={"returns": candidate_returns})
    d = decide(incumbent, candidate)
    assert d.accepted
    assert d.reason == REASON_UTILITY_IMPROVEMENT
    assert d.candidate_utility_lcb > d.incumbent_utility_lcb


def test_pareto_improvement_accepted_when_lower_complexity_same_sharpe():
    """Same Sharpe + lower complexity -> Pareto-accept (no utility improvement needed)."""
    # Use the same returns series so utility LCB matches and we fall through
    # to the Pareto branch.
    same_returns = _returns(seed=42)
    incumbent = _outcome(val_kw={"returns": same_returns}, complexity=4.0)
    candidate = _outcome(val_kw={"returns": same_returns}, complexity=2.5)
    d = decide(incumbent, candidate)
    assert d.accepted
    assert d.reason == REASON_PARETO_IMPROVEMENT


def test_pareto_check_requires_not_worse_sharpe():
    same_returns_a = _returns(mean=0.001, std=0.01, seed=20)
    same_returns_b = _returns(mean=0.0001, std=0.01, seed=21)
    incumbent = _outcome(val_kw={"returns": same_returns_a, "sharpe": 1.0}, complexity=4.0)
    candidate = _outcome(val_kw={"returns": same_returns_b, "sharpe": 0.2}, complexity=2.0)
    # Lower complexity but much lower Sharpe -> Pareto check fails.
    assert not pareto_dominates(candidate, incumbent, eps=0.005)


# -------- gate decision: overfit and "no meaningful" branches --------


def test_overfit_gap_rejected_when_train_sharpe_much_higher_than_val():
    same_returns = _returns(seed=30)
    incumbent = _outcome(val_kw={"returns": same_returns})
    # train.sharpe=1.5 - val.sharpe=0.3 = 1.2 > MAX_TRAIN_VAL_GAP=0.5
    candidate = _outcome(
        train_kw={"sharpe": 1.5},
        val_kw={"returns": same_returns, "sharpe": 0.3},
    )
    d = decide(incumbent, candidate)
    assert not d.accepted
    assert d.reason == REASON_OVERFIT_GAP


def test_no_meaningful_improvement_when_no_branch_matches():
    same_returns = _returns(seed=40)
    incumbent = _outcome(val_kw={"returns": same_returns, "sharpe": 0.5})
    candidate = _outcome(val_kw={"returns": same_returns, "sharpe": 0.5})
    d = decide(incumbent, candidate)
    assert not d.accepted
    assert d.reason == REASON_NO_MEANINGFUL_IMPROVEMENT


# -------- redact --------


def test_redact_covers_every_reason():
    decisions = [
        (REASON_UTILITY_IMPROVEMENT, ValidationSignal.ACCEPTED),
        (REASON_PARETO_IMPROVEMENT, ValidationSignal.ACCEPTED),
        (REASON_DRAWDOWN_REGRESSION, ValidationSignal.REJECTED_DRAWDOWN_REGRESSION),
        (REASON_TURNOVER_TOO_HIGH, ValidationSignal.REJECTED_TURNOVER),
        (REASON_COMPLEXITY_INCREASE, ValidationSignal.REJECTED_COMPLEXITY),
        (REASON_TOO_FEW_TRADES, ValidationSignal.REJECTED_NO_SIGNIFICANT_IMPROVEMENT),
        (REASON_OVERFIT_GAP, ValidationSignal.REJECTED_NO_SIGNIFICANT_IMPROVEMENT),
        (REASON_NO_MEANINGFUL_IMPROVEMENT, ValidationSignal.REJECTED_NO_SIGNIFICANT_IMPROVEMENT),
    ]
    for reason, expected in decisions:
        d = GateDecision(
            accepted=reason.startswith("accept_"),
            reason=reason,
            candidate_utility_lcb=0.0,
            incumbent_utility_lcb=0.0,
            complexity_delta=0.0,
        )
        assert redact(d) == expected, reason


def test_redact_rejects_unknown_reason():
    d = GateDecision(
        accepted=False,
        reason="reject_made_up",
        candidate_utility_lcb=0.0,
        incumbent_utility_lcb=0.0,
        complexity_delta=0.0,
    )
    with pytest.raises(ValueError, match="unknown gate reason"):
        redact(d)
