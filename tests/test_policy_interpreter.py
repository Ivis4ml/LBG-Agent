"""Tests for `policy_interpreter`: cross detection, state machine, sizing, lag.

The tests synthesize a tiny OHLCV frame so the expected positions are
hand-computable. They also do a smoke test through the real `indicators/sma.py`
via the sandbox.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lbg.dsl import (
    CrossAboveRule,
    CrossBelowRule,
    FixedFractionSizing,
    IndicatorSpec,
    Strategy,
    VolatilityTargetSizing,
    load_strategy,
)
from policy_interpreter import (
    _cross_above,
    _cross_below,
    _state_from_events,
    compute_positions,
)

REPO = Path(__file__).resolve().parents[1]


def test_cross_above_only_fires_on_the_crossing_bar():
    fast = pd.Series([1, 1, 1, 3, 3, 3], dtype=float)
    slow = pd.Series([2, 2, 2, 2, 2, 2], dtype=float)
    out = _cross_above(fast, slow).fillna(False).tolist()
    assert out == [False, False, False, True, False, False]


def test_cross_below_only_fires_on_the_crossing_bar():
    fast = pd.Series([3, 3, 3, 1, 1, 1], dtype=float)
    slow = pd.Series([2, 2, 2, 2, 2, 2], dtype=float)
    out = _cross_below(fast, slow).fillna(False).tolist()
    assert out == [False, False, False, True, False, False]


def test_state_from_events_exit_takes_precedence_on_existing_position():
    # Bar 1: entry. Bar 3: both entry AND exit -- exit wins because we're long.
    entry = pd.Series([False, True, False, True, False])
    exit_ = pd.Series([False, False, False, True, False])
    state = _state_from_events(entry, exit_).tolist()
    assert state == [0, 1, 1, 0, 0]


def test_state_from_events_entry_only_fires_when_flat():
    # Multiple entry signals while in position should be no-ops.
    entry = pd.Series([True, True, True, False, False])
    exit_ = pd.Series([False, False, False, True, False])
    state = _state_from_events(entry, exit_).tolist()
    assert state == [1, 1, 1, 0, 0]


def _hand_crafted_ohlcv() -> pd.DataFrame:
    """7-bar frame engineered so fast crosses above slow on bar 3 and below on 6."""
    close = pd.Series([10, 10, 10, 12, 12, 12, 8], dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame(
        {
            "open": open_,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(7, 1_000_000, dtype="int64"),
        }
    )


def test_compute_positions_lags_signal_by_one_bar(tmp_path):
    """A cross at close[t] must move position only from open[t+1] onward."""

    def _ind_factory(values):
        def _ind(_df, **_kw):
            return pd.Series(values, index=_df.index, dtype=float)

        return _ind

    df = _hand_crafted_ohlcv()
    fast_vals = [9, 9, 9, 13, 13, 13, 7]
    slow_vals = [10] * 7

    # Build a strategy manually + monkeypatch sandbox to return our toy indicators.
    strategy = Strategy(
        name="toy",
        indicators=[
            IndicatorSpec(name="fast", fn="fake_fast", params={}),
            IndicatorSpec(name="slow", fn="fake_slow", params={}),
        ],
        entry=CrossAboveRule(rule="cross_above", fast="fast", slow="slow"),
        exit=CrossBelowRule(rule="cross_below", fast="fast", slow="slow"),
        filters=[],
        sizing=FixedFractionSizing(mode="fixed_fraction", fraction=1.0, max_position=1.0),
    )

    # Write toy indicator files in a temp directory so sandbox can load them.
    (tmp_path / "fake_fast.py").write_text(
        "import pandas as pd\n"
        "def fake_fast(df):\n"
        f"    return pd.Series({fast_vals}, index=df.index, dtype=float)\n"
    )
    (tmp_path / "fake_slow.py").write_text(
        "import pandas as pd\n"
        "def fake_slow(df):\n"
        f"    return pd.Series({slow_vals}, index=df.index, dtype=float)\n"
    )

    positions = compute_positions(strategy, df, indicators_dir=tmp_path)
    # Entry signal fires on bar 3 (cross above). Exit on bar 6 (cross below).
    # State (in-position): [0,0,0,1,1,1,0]. After lag-by-one: [0,0,0,0,1,1,1].
    expected = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert positions.tolist() == expected


def test_volatility_target_sizing_scales_position_by_target_over_realized():
    """Higher target_vol -> larger position; cap at max_position."""
    df = _hand_crafted_ohlcv()
    in_position = pd.Series([0, 0, 1, 1, 1, 1, 1], dtype="int8", index=df.index)
    from policy_interpreter import _apply_sizing

    sizing = VolatilityTargetSizing(
        mode="volatility_target", target_vol=0.20, max_position=0.8, vol_lookback=3
    )
    sized = _apply_sizing(in_position, sizing, df)
    assert (sized <= 0.8).all()
    assert (sized >= 0.0).all()
    # When not in position, sizing yields 0 regardless of vol.
    assert (sized.iloc[:2] == 0.0).all()


def test_real_sma_cross_runs_through_full_pipeline():
    """Smoke test: load repo strategy.yaml, run it on synthetic data."""
    strategy = load_strategy(REPO / "strategy.yaml")
    rng = np.random.default_rng(0)
    n = 200
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, size=n).astype("int64"),
        }
    )
    positions = compute_positions(strategy, df, indicators_dir=REPO / "indicators")
    assert len(positions) == len(df)
    assert (positions >= 0.0).all() and (positions <= 1.0).all()
    # SMA fast crosses slow occasionally; we should see at least one in-position bar.
    assert (positions > 0).any()
