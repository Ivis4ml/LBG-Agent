"""Deterministic policy interpreter (PROPOSAL.html §4.2, §6.5).

Reads a validated `Strategy` plus the OHLCV DataFrame, calls each indicator
through the sandbox, and produces a long-only position series in [0, 1].

This file is locked by the `no_python_edit` invariant: agents cannot edit
it during a Discovery run. The deterministic behavior here is part of the
experimental protocol — every change to this file invalidates prior trial
comparisons.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lbg.dsl.schema import (
    CrossAboveRule,
    CrossBelowRule,
    FixedFractionSizing,
    IndicatorAboveFilter,
    IndicatorBelowFilter,
    Strategy,
    VolatilityTargetSizing,
)
from lbg.sandbox import load_indicator, run_indicator

INDICATORS_DIR = Path("indicators")


def _compute_indicators(
    strategy: Strategy,
    df: pd.DataFrame,
    indicators_dir: Path,
    timeout_sec: float,
) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for spec in strategy.indicators:
        module_path = indicators_dir / f"{spec.fn}.py"
        fn = load_indicator(module_path)
        series = run_indicator(fn, df, params=spec.params, timeout_sec=timeout_sec)
        out[spec.name] = series.reindex(df.index)
    return out


def _cross_above(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """True at bar t when fast crossed above slow on bar t."""
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))


def _cross_below(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """True at bar t when fast crossed below slow on bar t."""
    return (fast < slow) & (fast.shift(1) >= slow.shift(1))


def _eval_cross_rule(
    rule: CrossAboveRule | CrossBelowRule,
    indicators: dict[str, pd.Series],
) -> pd.Series:
    fast = indicators[rule.fast]
    slow = indicators[rule.slow]
    if isinstance(rule, CrossAboveRule):
        return _cross_above(fast, slow).fillna(False).astype(bool)
    return _cross_below(fast, slow).fillna(False).astype(bool)


def _eval_filter(
    flt: IndicatorAboveFilter | IndicatorBelowFilter,
    indicators: dict[str, pd.Series],
) -> pd.Series:
    series = indicators[flt.indicator]
    if isinstance(flt, IndicatorAboveFilter):
        return (series >= flt.threshold).fillna(False).astype(bool)
    return (series <= flt.threshold).fillna(False).astype(bool)


def _state_from_events(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """Convert per-bar entry/exit events to a binary `in_position` series.

    At each bar, exit takes precedence on existing positions, and entry only
    fires when flat. This mirrors typical trend-following execution: ride a
    position until exit, then look for the next entry.
    """
    entry_arr = entry.to_numpy()
    exit_arr = exit_.to_numpy()
    n = len(entry_arr)
    state = np.zeros(n, dtype=np.int8)
    in_pos = False
    for t in range(n):
        if in_pos:
            if exit_arr[t]:
                in_pos = False
        else:
            if entry_arr[t]:
                in_pos = True
        state[t] = 1 if in_pos else 0
    return pd.Series(state, index=entry.index, dtype="int8")


def _apply_sizing(
    in_position: pd.Series,
    sizing: FixedFractionSizing | VolatilityTargetSizing,
    df: pd.DataFrame,
) -> pd.Series:
    if isinstance(sizing, FixedFractionSizing):
        base = in_position.astype(float) * sizing.fraction
    else:
        # volatility_target: target / realized vol over a trailing window.
        log_ret = np.log(df["close"] / df["close"].shift(1))
        realized = log_ret.rolling(sizing.vol_lookback).std() * np.sqrt(252)
        scaling = (sizing.target_vol / realized).replace([np.inf, -np.inf], np.nan)
        base = in_position.astype(float) * scaling.fillna(0.0)
    return base.clip(lower=0.0, upper=sizing.max_position)


def compute_positions(
    strategy: Strategy,
    df: pd.DataFrame,
    *,
    indicators_dir: Path = INDICATORS_DIR,
    timeout_sec: float = 10.0,
) -> pd.Series:
    """Run the strategy on `df` and return the position series p_{t+1} in [0, 1].

    The output is already lagged by one bar: the signal computed at the close
    of bar t determines the position held starting from the open of bar t+1
    (PROPOSAL.html §4.2). This is what makes close-on-close lookahead
    structurally impossible.
    """
    indicators = _compute_indicators(strategy, df, indicators_dir, timeout_sec)

    entry_signal = _eval_cross_rule(strategy.entry, indicators)
    exit_signal = _eval_cross_rule(strategy.exit, indicators)

    for flt in strategy.filters:
        entry_signal = entry_signal & _eval_filter(flt, indicators)

    in_position = _state_from_events(entry_signal, exit_signal)
    sized = _apply_sizing(in_position, strategy.sizing, df)
    return sized.shift(1).fillna(0.0).rename("position")
