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


def _state_from_events(
    entry: pd.Series,
    exit_: pd.Series,
    *,
    mute_signal: pd.Series | None = None,
    rearm_exit_signal: pd.Series | None = None,
) -> pd.Series:
    """Convert per-bar entry/exit events to a binary `in_position` series.

    Per-bar precedence:
      - if in position, an exit event flips us flat;
      - if flat AND not muted, an entry event flips us in-position;
      - additionally, if we are flat because of a rearm-capable exit
        filter (not because of the exit cross rule) and the mute lifts,
        we auto-resume in-position the same bar.

    `mute_signal[t]` blocks new entries on bar `t`. `rearm_exit_signal[t]`
    marks "the exit at bar t was caused purely by a rearm-capable exit
    filter, not by the exit cross rule" -- when the muting window
    eventually lifts, we re-enter even if the entry cross hasn't fired.
    This solves the cash-trap on edge-triggered baselines like buyhold,
    where the entry cross only fires once and would never re-arm on its
    own.

    Backward-compatible: with `rearm_exit_signal=None` (or all-False)
    behavior is exactly the prior wait-for-entry semantics; existing
    tests pass without changes.
    """
    entry_arr = entry.to_numpy()
    exit_arr = exit_.to_numpy()
    n = len(entry_arr)
    mute_arr = mute_signal.to_numpy() if mute_signal is not None else np.zeros(n, dtype=bool)
    rearm_arr = (
        rearm_exit_signal.to_numpy() if rearm_exit_signal is not None else np.zeros(n, dtype=bool)
    )

    state = np.zeros(n, dtype=np.int8)
    in_pos = False
    # `awaiting_rearm` = "we are out because a rearm-capable exit filter
    # forced us out; when the mute lifts we should auto-resume." Cleared
    # the moment we get back in-position (auto-resume or fresh entry).
    awaiting_rearm = False
    for t in range(n):
        if in_pos:
            if exit_arr[t]:
                in_pos = False
                awaiting_rearm = bool(rearm_arr[t])
        else:
            if not mute_arr[t]:
                if awaiting_rearm:
                    in_pos = True
                    awaiting_rearm = False
                elif entry_arr[t]:
                    in_pos = True
        state[t] = 1 if in_pos else 0
    return pd.Series(state, index=entry.index, dtype="int8")


def _mute_from_rearm_filter(
    flt: IndicatorAboveFilter | IndicatorBelowFilter,
    indicators: dict[str, pd.Series],
) -> pd.Series:
    """Per-bar `True` whenever this exit filter is in its post-fire pre-rearm window.

    Iterates the indicator series once, tracking a 2-state machine:
      - safe (mute = False): waiting for `threshold` to be breached
      - muted (mute = True): waiting for `rearm_threshold` to be crossed
                             back to the safe side; mute stays True for
                             every bar in this window so new entries are
                             blocked

    For `indicator_above` (threshold + rearm_threshold), trigger is
    `indicator >= threshold`, rearm is `indicator < rearm_threshold`.
    For `indicator_below`, trigger is `indicator <= threshold`, rearm is
    `indicator > rearm_threshold`. Filters with `rearm_threshold=None`
    don't get a mute window (the caller should not pass them here).
    """
    if flt.rearm_threshold is None:
        # Defensive: should not be reached.
        return pd.Series(False, index=indicators[flt.indicator].index)
    series = indicators[flt.indicator]
    vals = series.to_numpy(dtype=float, na_value=np.nan)
    n = len(vals)
    out = np.zeros(n, dtype=bool)
    muted = False
    is_above = isinstance(flt, IndicatorAboveFilter)
    threshold = float(flt.threshold)
    rearm = float(flt.rearm_threshold)
    for t in range(n):
        v = vals[t]
        if not np.isfinite(v):
            out[t] = muted
            continue
        if muted:
            # check rearm
            if is_above:
                if v < rearm:
                    muted = False
            else:
                if v > rearm:
                    muted = False
        else:
            # check trigger
            if is_above:
                if v >= threshold:
                    muted = True
            else:
                if v <= threshold:
                    muted = True
        out[t] = muted
    return pd.Series(out, index=series.index)


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
    cross_exit_signal = _eval_cross_rule(strategy.exit, indicators)

    # Entry filters AND-combine with the entry cross: every filter must
    # admit. Exit filters OR-combine with the exit cross: any True
    # exit_filter forces an exit on top of the cross_rule exit event.
    # Exit filters carrying a `rearm_threshold` additionally contribute
    # to `mute_signal` (block new entries while in post-fire pre-rearm
    # window) AND to `rearm_filter_exit_signal` (mark which bars' exits
    # were caused by such a filter -- the state machine uses this to
    # auto-resume position when the mute lifts, instead of waiting for
    # the entry cross to re-fire). Together this lets a one-shot entry
    # baseline (e.g. buyhold) survive a drawdown exit and recover.
    for flt in strategy.filters:
        entry_signal = entry_signal & _eval_filter(flt, indicators)
    filter_exit_signal = pd.Series(False, index=df.index)
    mute_signal = pd.Series(False, index=df.index)
    rearm_filter_exit_signal = pd.Series(False, index=df.index)
    for flt in strategy.exit_filters:
        triggered = _eval_filter(flt, indicators)
        filter_exit_signal = filter_exit_signal | triggered
        if flt.rearm_threshold is not None:
            mute_signal = mute_signal | _mute_from_rearm_filter(flt, indicators)
            rearm_filter_exit_signal = rearm_filter_exit_signal | triggered
    exit_signal = cross_exit_signal | filter_exit_signal
    # "auto-resume" only applies when a rearm-capable filter fired AND the
    # cross-exit didn't also fire on the same bar. If both fire we honor
    # the cross-exit's "stay out until next entry signal" semantic.
    rearm_exit_signal = rearm_filter_exit_signal & ~cross_exit_signal

    in_position = _state_from_events(
        entry_signal,
        exit_signal,
        mute_signal=mute_signal,
        rearm_exit_signal=rearm_exit_signal,
    )
    sized = _apply_sizing(in_position, strategy.sizing, df)
    return sized.shift(1).fillna(0.0).rename("position")
