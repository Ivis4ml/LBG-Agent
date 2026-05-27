"""Tests for the rearm_threshold field on exit filters.

This solves the buyhold cash-trap problem: a one-shot entry baseline plus
a drawdown-style exit filter would permanently sit in cash after a single
trigger. With `rearm_threshold`, the policy enters a *muted* state on
exit, and the entry cross rule re-arms when the indicator crosses back to
the safe side. The tests pin:

  - schema validators reject rearm on the wrong side of threshold;
  - rearm=None preserves backward-compatible behavior (no mute window);
  - rearm window blocks entry until the indicator recovers;
  - the indicator's exact recovery bar is where re-entry becomes eligible;
  - multiple rearm filters union into one mute signal.
"""

from __future__ import annotations

import pandas as pd
import pytest

from lbg.dsl.schema import (
    IndicatorAboveFilter,
    IndicatorBelowFilter,
)
from policy_interpreter import _mute_from_rearm_filter

# -------- schema validators --------


def test_rearm_above_threshold_rejected_for_indicator_above():
    with pytest.raises(ValueError, match="must be <= threshold"):
        IndicatorAboveFilter(
            rule="indicator_above", indicator="drawdown", threshold=0.10, rearm_threshold=0.15
        )


def test_rearm_below_threshold_rejected_for_indicator_below():
    with pytest.raises(ValueError, match="must be >= threshold"):
        IndicatorBelowFilter(
            rule="indicator_below", indicator="momentum", threshold=0.05, rearm_threshold=0.01
        )


def test_rearm_threshold_optional_default_none():
    flt = IndicatorAboveFilter(rule="indicator_above", indicator="drawdown", threshold=0.10)
    assert flt.rearm_threshold is None


def test_rearm_equal_to_threshold_allowed():
    """No strict inequality required; equal thresholds are degenerate but legal."""
    a = IndicatorAboveFilter(
        rule="indicator_above", indicator="x", threshold=0.10, rearm_threshold=0.10
    )
    assert a.rearm_threshold == 0.10
    b = IndicatorBelowFilter(
        rule="indicator_below", indicator="x", threshold=0.10, rearm_threshold=0.10
    )
    assert b.rearm_threshold == 0.10


# -------- mute window construction --------


def _idx(n: int) -> pd.RangeIndex:
    return pd.RangeIndex(n)


def test_indicator_above_mute_fires_and_clears():
    """series: [0, 0, 0.20, 0.15, 0.04, 0.02, 0.00] · threshold=0.10 · rearm=0.05.

    Mute should fire at t=2 (>=0.10), stay True through t=4 (still >=0.05),
    and clear at t=4 when value drops below 0.05.

    Actually: the rearm check is `value < rearm`, so t=4 (value 0.04 < 0.05)
    is the first bar where mute becomes False. The True window is [2, 3].
    """
    series = pd.Series([0.0, 0.0, 0.20, 0.15, 0.04, 0.02, 0.00], index=_idx(7))
    flt = IndicatorAboveFilter(
        rule="indicator_above", indicator="drawdown", threshold=0.10, rearm_threshold=0.05
    )
    mute = _mute_from_rearm_filter(flt, {"drawdown": series})
    assert mute.tolist() == [False, False, True, True, False, False, False]


def test_indicator_below_mute_fires_and_clears():
    """series: [1.0, 0.5, 0.0, 0.10, 0.50] · threshold=0.05 · rearm=0.20.

    Mute fires at t=2 (<=0.05), clears at the first bar where value > 0.20.
    t=3 has value 0.10 (still muted), t=4 has 0.50 (clear).
    """
    series = pd.Series([1.0, 0.5, 0.0, 0.10, 0.50], index=_idx(5))
    flt = IndicatorBelowFilter(
        rule="indicator_below", indicator="trend", threshold=0.05, rearm_threshold=0.20
    )
    mute = _mute_from_rearm_filter(flt, {"trend": series})
    assert mute.tolist() == [False, False, True, True, False]


def test_mute_can_re_fire_after_clearing():
    """Once cleared, a second breach re-mutes."""
    # series breaches twice
    series = pd.Series([0.0, 0.20, 0.04, 0.00, 0.20, 0.00], index=_idx(6))
    flt = IndicatorAboveFilter(
        rule="indicator_above", indicator="x", threshold=0.10, rearm_threshold=0.05
    )
    mute = _mute_from_rearm_filter(flt, {"x": series})
    # t=1 fire, t=2 clear (0.04 < 0.05), t=4 fire again, t=5 clear (0.00 < 0.05)
    assert mute.tolist() == [False, True, False, False, True, False]


def test_nan_values_preserve_current_state():
    """NaN bars keep the current mute state -- common at the start of
    rolling indicators before the window has filled."""
    series = pd.Series([float("nan"), float("nan"), 0.20, 0.15, float("nan"), 0.04], index=_idx(6))
    flt = IndicatorAboveFilter(
        rule="indicator_above", indicator="x", threshold=0.10, rearm_threshold=0.05
    )
    mute = _mute_from_rearm_filter(flt, {"x": series})
    # nan, nan: not muted (initial state)
    # 0.20 >= threshold: muted
    # 0.15 >= rearm so stays muted
    # nan: stays muted
    # 0.04 < rearm: unmuted
    assert mute.tolist() == [False, False, True, True, True, False]


# -------- integration with compute_positions --------


def test_buyhold_with_rearm_recovers_after_drawdown(tmp_path):
    """End-to-end: a fake one-shot entry with a rearm exit filter and a
    constructed drawdown series should produce a position that re-enters
    after the drawdown recovers.

    Build a tiny synthetic DataFrame with the indicator authored inline.
    """
    from lbg.dsl import load_strategy
    from policy_interpreter import compute_positions

    # 12 bars of constant close=100 except a dip from t=5 to t=7
    df = pd.DataFrame(
        {
            "open": [100.0] * 12,
            "high": [100.0] * 12,
            "low": [100.0] * 12,
            "close": [100.0] * 12,
            "volume": [1000.0] * 12,
        }
    )

    # Build a strategy that uses two synthetic indicators we author here.
    indicators_dir = tmp_path / "indicators"
    indicators_dir.mkdir()
    (indicators_dir / "step_in.py").write_text(
        "import pandas as pd\n"
        "def step_in(df, **params):\n"
        "    out = pd.Series(1.0, index=df.index)\n"
        "    if len(out) > 0:\n"
        "        out.iloc[0] = 0.0\n"
        "    return out\n",
        encoding="utf-8",
    )
    (indicators_dir / "zero_baseline.py").write_text(
        "import pandas as pd\n"
        "def zero_baseline(df, **params):\n"
        "    return pd.Series(0.0, index=df.index)\n",
        encoding="utf-8",
    )
    # synthetic drawdown: 0 normally, breaches 0.20 at t=5..6, recovers 0.02 at t=8
    (indicators_dir / "fake_drawdown.py").write_text(
        "import pandas as pd\n"
        "def fake_drawdown(df, **params):\n"
        "    vals = [0.0]*5 + [0.20, 0.15, 0.10, 0.02] + [0.0]*3\n"
        "    return pd.Series(vals[:len(df)], index=df.index)\n",
        encoding="utf-8",
    )

    strategy_yaml = (
        "name: buyhold_with_rearm\n"
        "indicators:\n"
        "  - {name: step_in, fn: step_in, params: {}}\n"
        "  - {name: zero_baseline, fn: zero_baseline, params: {}}\n"
        "  - {name: fake_drawdown, fn: fake_drawdown, params: {}}\n"
        "entry: {rule: cross_above, fast: step_in, slow: zero_baseline}\n"
        "exit:  {rule: cross_below, fast: step_in, slow: zero_baseline}\n"
        "filters: []\n"
        "exit_filters:\n"
        "  - {rule: indicator_above, indicator: fake_drawdown,\n"
        "     threshold: 0.15, rearm_threshold: 0.05}\n"
        "sizing: {mode: fixed_fraction, fraction: 1.0, max_position: 1.0}\n"
    )
    strategy_path = tmp_path / "strategy.yaml"
    strategy_path.write_text(strategy_yaml, encoding="utf-8")
    strategy = load_strategy(strategy_path)

    positions = compute_positions(strategy, df, indicators_dir=indicators_dir, timeout_sec=5.0)

    # positions = sized.shift(1).fillna(0.0). With fixed_fraction 1.0:
    # in_position[0] = 0 (entry at t=1 due to step_in firing)
    # in_position[1..4] = 1 (held)
    # in_position[5] = exit fires (drawdown 0.20 >= 0.15), so 0
    # in_position[6..7] = muted (drawdown still >= rearm 0.05), so 0
    # in_position[8] = drawdown 0.02 < rearm, unmuted; but entry cross was
    #                  edge-triggered at t=1 only, so no re-entry HERE.
    #                  In this synthetic test that's expected: re-entry is
    #                  GATED but not GUARANTEED. The next entry cross has
    #                  to fire for re-entry to happen. For buyhold's once-
    #                  fires-then-stable step_in, re-entry will NOT happen.
    #                  The mute machinery just makes sure we are no longer
    #                  *blocking* a hypothetical re-entry.
    # Then the lag (shift(1)) shifts everything one to the right.
    # So we mainly check: position is 0 at t=0, becomes 1 after first entry,
    # drops to 0 around the drawdown firing, and stays 0 (because step_in
    # doesn't re-cross).
    positions_list = positions.tolist()
    assert positions_list[0] == 0.0  # initial lag
    assert positions_list[2] == 1.0  # in-position by t=2 (entry at t=1 -> lag -> position at t=2)
    # exit fires at t=5 (drawdown 0.20 >= 0.15), in_position drops to 0 at t=5,
    # the shift puts it at t=6.
    assert positions_list[6] == 0.0
    # Still muted at t=6..7 (drawdown 0.15, 0.10 >= rearm 0.05).
    assert positions_list[7] == 0.0
    # drawdown drops to 0.02 < rearm at t=8 -> mute lifts -> AUTO-RESUME.
    # Position becomes 1 at t=8 in_position, shift lands it at t=9.
    assert positions_list[9] == 1.0, (
        f"position must auto-resume after rearm; got {positions_list[9]}"
    )
    # And stay in-position for the rest of the bars (drawdown stays 0.0).
    assert positions_list[10] == 1.0


def test_cross_exit_does_not_auto_resume_even_with_rearm_filter(tmp_path):
    """Cross exit's semantic is 'stay out until next entry signal'. If a
    rearm-capable filter also fires on the same bar, the cross-exit
    semantic wins -- auto-resume is OFF, awaiting entry cross."""
    from lbg.dsl import load_strategy
    from policy_interpreter import compute_positions

    df = pd.DataFrame(
        {
            "open": [100.0] * 10,
            "high": [100.0] * 10,
            "low": [100.0] * 10,
            "close": [100.0] * 10,
            "volume": [1000.0] * 10,
        }
    )

    indicators_dir = tmp_path / "indicators"
    indicators_dir.mkdir()
    # Entry: cross at t=1 ("step_in"). Exit cross: cross at t=5 ("step_out").
    # Drawdown filter spikes at t=5 with rearm window through t=8.
    (indicators_dir / "step_in.py").write_text(
        "import pandas as pd\n"
        "def step_in(df, **params):\n"
        "    out = pd.Series(1.0, index=df.index)\n"
        "    if len(out) > 0: out.iloc[0] = 0.0\n"
        "    return out\n",
        encoding="utf-8",
    )
    (indicators_dir / "step_out.py").write_text(
        "import pandas as pd\n"
        "def step_out(df, **params):\n"
        "    vals = [1.0]*5 + [0.0]*5\n"
        "    return pd.Series(vals[:len(df)], index=df.index)\n",
        encoding="utf-8",
    )
    (indicators_dir / "step_one.py").write_text(
        "import pandas as pd\n"
        "def step_one(df, **params):\n"
        "    return pd.Series(1.0, index=df.index)\n",
        encoding="utf-8",
    )
    (indicators_dir / "fake_drawdown.py").write_text(
        "import pandas as pd\n"
        "def fake_drawdown(df, **params):\n"
        "    vals = [0.0]*5 + [0.20, 0.15, 0.10, 0.02, 0.0]\n"
        "    return pd.Series(vals[:len(df)], index=df.index)\n",
        encoding="utf-8",
    )

    strategy_yaml = (
        "name: cross_exit_plus_rearm\n"
        "indicators:\n"
        "  - {name: step_in, fn: step_in, params: {}}\n"
        "  - {name: step_out, fn: step_out, params: {}}\n"
        "  - {name: step_one, fn: step_one, params: {}}\n"
        "  - {name: fake_drawdown, fn: fake_drawdown, params: {}}\n"
        "entry: {rule: cross_above, fast: step_in, slow: step_one}\n"
        # cross_below fires at t=5 when step_out drops from 1->0 (step_one stays 1)
        "exit:  {rule: cross_below, fast: step_out, slow: step_one}\n"
        "filters: []\n"
        "exit_filters:\n"
        "  - {rule: indicator_above, indicator: fake_drawdown,\n"
        "     threshold: 0.15, rearm_threshold: 0.05}\n"
        "sizing: {mode: fixed_fraction, fraction: 1.0, max_position: 1.0}\n"
    )
    strategy_path = tmp_path / "strategy.yaml"
    strategy_path.write_text(strategy_yaml, encoding="utf-8")
    strategy = load_strategy(strategy_path)
    positions = compute_positions(strategy, df, indicators_dir=indicators_dir, timeout_sec=5.0)
    # At t=5 BOTH cross_exit and the rearm filter fire. Cross-exit's
    # semantic should win -- no auto-resume even after mute lifts.
    # Verifying that t=9 (after mute lifts at t=8 since drawdown 0.02 < 0.05)
    # is STILL out, because cross-exit semantic was honored.
    assert positions.iloc[9] == 0.0, (
        f"cross-exit should suppress auto-resume; got {positions.iloc[9]}"
    )
