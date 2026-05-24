"""baselines/buyhold/ encodes a buy-and-hold strategy in the existing
CrossRule DSL via a step_in + zero_baseline indicator pair. Tests pin
the shape so future DSL/policy changes don't silently break it.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
BUYHOLD = REPO / "baselines" / "buyhold"


@pytest.fixture
def buyhold_workdir(tmp_path):
    shutil.copy(BUYHOLD / "strategy.yaml", tmp_path / "strategy.yaml")
    shutil.copytree(BUYHOLD / "indicators", tmp_path / "indicators")
    return tmp_path


def test_step_in_returns_zero_then_ones():
    sys.path.insert(0, str(BUYHOLD / "indicators"))
    try:
        from step_in import step_in
    finally:
        sys.path.pop(0)
    df = pd.DataFrame(
        {
            "open": [1.0] * 5,
            "high": [1.0] * 5,
            "low": [1.0] * 5,
            "close": [1.0] * 5,
            "volume": [1] * 5,
        }
    )
    out = step_in(df)
    assert out.tolist() == [0.0, 1.0, 1.0, 1.0, 1.0]


def test_zero_baseline_returns_zeros():
    sys.path.insert(0, str(BUYHOLD / "indicators"))
    try:
        from zero_baseline import zero_baseline
    finally:
        sys.path.pop(0)
    df = pd.DataFrame(
        {
            "open": [1.0] * 5,
            "high": [1.0] * 5,
            "low": [1.0] * 5,
            "close": [1.0] * 5,
            "volume": [1] * 5,
        }
    )
    out = zero_baseline(df)
    assert out.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_both_indicators_are_prefix_stable():
    """Output at index t depends only on t -- perturbing any future row
    cannot change the output. The dynamic prefix_stability check this
    asserts in production is satisfied trivially here."""
    sys.path.insert(0, str(BUYHOLD / "indicators"))
    try:
        from step_in import step_in
        from zero_baseline import zero_baseline
    finally:
        sys.path.pop(0)

    def df_of(n):
        return pd.DataFrame(
            {
                "open": [1.0] * n,
                "high": [1.0] * n,
                "low": [1.0] * n,
                "close": [1.0] * n,
                "volume": [1] * n,
            }
        )

    for n in (3, 10, 100):
        si = step_in(df_of(n))
        z = zero_baseline(df_of(n))
        assert si.iloc[0] == 0.0 and (si.iloc[1:] == 1.0).all()
        assert (z == 0.0).all()


def test_strategy_yaml_validates_against_dsl(buyhold_workdir):
    """The strategy file passes the Pydantic schema and references our
    two indicators correctly."""
    from lbg.dsl import load_strategy

    s = load_strategy(buyhold_workdir / "strategy.yaml")
    assert s.name == "buyhold_baseline"
    assert {i.name for i in s.indicators} == {"step_in", "zero_baseline"}
    assert s.entry.rule == "cross_above"
    assert s.entry.fast == "step_in"
    assert s.entry.slow == "zero_baseline"
    assert s.exit.rule == "cross_below"
    assert s.sizing.mode == "fixed_fraction"
    assert s.sizing.fraction == 1.0


def test_buyhold_strategy_produces_always_long_positions(buyhold_workdir):
    """The full policy pipeline: load strategy, compute positions on
    real SPY data, verify the output is 0 for one initial bar then 1
    thereafter. Positions are lagged by one bar (PROPOSAL §4.2) so the
    cross at bar 1 surfaces at bar 2."""
    from lbg.data.loader import load_split
    from lbg.dsl import load_strategy
    from policy_interpreter import compute_positions

    strategy = load_strategy(buyhold_workdir / "strategy.yaml")
    df = load_split("split_A")
    positions = compute_positions(
        strategy,
        df,
        indicators_dir=buyhold_workdir / "indicators",
        timeout_sec=10.0,
    )
    # Bars 0 and 1 are 0 (entry detected at bar 1, position takes effect
    # at bar 2 due to lag). Everything after is 1.0.
    assert positions.iloc[0] == 0.0
    assert positions.iloc[1] == 0.0
    assert (positions.iloc[2:] == 1.0).all()


def test_buyhold_sealed_beats_baseline_on_sealed_window(buyhold_workdir):
    """Sanity: the buy-and-hold baseline's sealed Sharpe is meaningfully
    above the SMA(20/50)'s 0.108. This is the whole reason for choosing
    this baseline -- it gives add_indicator a real Pareto path."""
    from backtest import run_backtest
    from lbg.data.loader import load_split
    from lbg.dsl import load_strategy
    from policy_interpreter import compute_positions

    strategy = load_strategy(buyhold_workdir / "strategy.yaml")
    df = load_split("split_C")
    positions = compute_positions(
        strategy,
        df,
        indicators_dir=buyhold_workdir / "indicators",
        timeout_sec=10.0,
    )
    res = run_backtest(positions, df)
    # SMA cross baseline sealed sharpe = 0.108; buy-and-hold-from-bar-2
    # should clear ~0.85 (close to the b&h analytical baseline of 0.86).
    assert res.sharpe > 0.5, f"expected buyhold sealed sharpe > 0.5, got {res.sharpe}"


def test_campaign_baseline_flag_resolves_buyhold():
    """The new --baseline flag pulls strategy.yaml + indicators from
    baselines/buyhold/."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("campaign", REPO / "scripts" / "campaign.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    s, i = mod._resolve_baseline("buyhold")
    assert s == REPO / "baselines" / "buyhold" / "strategy.yaml"
    assert i == REPO / "baselines" / "buyhold" / "indicators"


def test_campaign_baseline_flag_default_is_sma_cross():
    import importlib.util

    spec = importlib.util.spec_from_file_location("campaign", REPO / "scripts" / "campaign.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    s, i = mod._resolve_baseline("sma_cross")
    assert s == REPO / "strategy.yaml"
    assert i == REPO / "indicators"
