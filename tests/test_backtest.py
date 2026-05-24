"""Tests for `backtest`: metrics formulas and an end-to-end regression check."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest import (
    DEFAULT_COST_PER_SIDE,
    TRADING_DAYS_PER_YEAR,
    BacktestResult,
    _max_drawdown,
    _open_to_open_returns,
    _sharpe_annualized,
    run_backtest,
)
from lbg.data.loader import RAW_PARQUET_PATH, load_split
from lbg.dsl import load_strategy
from policy_interpreter import compute_positions

REPO = Path(__file__).resolve().parents[1]


# -------- pure formulas --------


def test_open_to_open_returns_definition():
    opens = pd.Series([100.0, 110.0, 121.0, 121.0], dtype=float)
    r = _open_to_open_returns(opens)
    # r[0] = 110/100 - 1; r[1] = 121/110 - 1; r[2] = 121/121 - 1; r[3] is NaN.
    assert pytest.approx(r.iloc[0], rel=1e-12) == 0.10
    assert pytest.approx(r.iloc[1], rel=1e-12) == 0.10
    assert pytest.approx(r.iloc[2], rel=1e-12) == 0.0
    assert pd.isna(r.iloc[3])


def test_sharpe_zero_volatility_returns_zero():
    r = pd.Series([0.001] * 100)
    assert _sharpe_annualized(r) == 0.0


def test_max_drawdown_monotonic_uptrend_is_zero():
    equity = pd.Series(np.linspace(1.0, 2.0, 100))
    assert _max_drawdown(equity) == 0.0


def test_max_drawdown_simple_case():
    equity = pd.Series([1.0, 1.2, 0.96, 1.5])  # drawdown 1.2 -> 0.96 = -20%
    assert pytest.approx(_max_drawdown(equity), rel=1e-12) == -0.20


# -------- backtest mechanics --------


def _flat_returns_frame(n: int = 50, daily_return: float = 0.001) -> pd.DataFrame:
    """Open series whose open-to-open returns are constant."""
    opens = (1.0 + daily_return) ** np.arange(n)
    return pd.DataFrame(
        {
            "open": opens,
            "high": opens * 1.001,
            "low": opens * 0.999,
            "close": opens,
            "volume": np.full(n, 1_000_000, dtype="int64"),
        }
    )


def test_zero_position_yields_zero_return_and_no_drawdown():
    df = _flat_returns_frame()
    p = pd.Series(np.zeros(len(df)), dtype=float)
    res = run_backtest(p, df, cost_per_side=0.0)
    assert res.sharpe == 0.0
    assert res.max_drawdown == 0.0
    assert res.num_trades == 0
    assert res.cost_total == 0.0
    assert pytest.approx(res.final_equity) == 1.0


def test_constant_long_position_no_cost_grows_at_daily_rate():
    df = _flat_returns_frame(n=253, daily_return=0.001)
    p = pd.Series(np.ones(len(df)), dtype=float)
    res = run_backtest(p, df, cost_per_side=0.0)
    # Even at zero cost, the *initial* establishment of the position counts as 1 unit
    # of turnover. Subsequent bars have delta = 0.
    assert res.num_trades == 1
    assert res.cost_total == 0.0
    # 252 daily compounds at 0.001 -> ~1.286
    assert pytest.approx(res.final_equity, rel=1e-6) == (1.001**252)


def test_cost_scales_linearly_with_position_changes():
    df = _flat_returns_frame(n=10, daily_return=0.0)
    # Position flips between 0 and 1 every bar: 9 deltas of 1 each.
    p = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=float)
    res_a = run_backtest(p, df, cost_per_side=0.001)
    res_b = run_backtest(p, df, cost_per_side=0.002)
    assert pytest.approx(res_b.cost_total, rel=1e-12) == 2 * res_a.cost_total


def test_mismatched_lengths_raise():
    df = _flat_returns_frame(n=20)
    with pytest.raises(ValueError, match="length"):
        run_backtest(pd.Series([0.0] * 19), df)


# -------- end-to-end regression --------


@pytest.mark.skipif(
    not RAW_PARQUET_PATH.exists(),
    reason="Parquet cache absent; run `uv run python -m lbg.data.loader fetch` first.",
)
def test_repo_strategy_on_split_a_regression():
    """Lock the SMA-cross baseline numbers so future refactors are caught.

    These values were captured on the first end-to-end run. They are not the
    proposal's target — they are the current implementation's behavior. Tighten
    or update when the policy/backtest changes deliberately.
    """
    strategy = load_strategy(REPO / "strategy.yaml")
    df = load_split("split_A")
    positions = compute_positions(strategy, df)
    res = run_backtest(positions, df)

    assert isinstance(res, BacktestResult)
    assert res.n_bars_used == 2263
    assert 40 <= res.num_trades <= 55, res.num_trades
    assert 0.5 <= res.sharpe <= 0.9, res.sharpe
    assert -0.30 <= res.max_drawdown <= -0.20, res.max_drawdown
    assert 4.0 <= res.turnover <= 6.5, res.turnover
    assert 1.6 <= res.final_equity <= 1.9, res.final_equity


def test_default_cost_matches_proposal_section_4_3():
    """Proposal §4.3: c = 5 bps base + 0.5 bp slippage proxy = 0.00055."""
    assert DEFAULT_COST_PER_SIDE == 0.00055


def test_trading_days_constant_matches_convention():
    assert TRADING_DAYS_PER_YEAR == 252
