"""Tests for lbg.verdict.per_card_null (B3 building blocks)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from lbg.alpha_cards import AlphaCard
from lbg.dsl import load_strategy
from lbg.verdict.per_card_null import (
    bootstrap_dataframe,
    inject_card_into_strategy,
    stage_indicators_dir,
)

REPO = Path(__file__).resolve().parents[1]


def _load_card(alpha_id: str) -> AlphaCard:
    path = REPO / "alpha_cards_library" / "cards" / f"{alpha_id}.yaml"
    return AlphaCard.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


# ---------- inject_card_into_strategy ----------


def test_inject_adds_indicator_and_filter():
    baseline = load_strategy(REPO / "strategy.yaml")
    card = _load_card("vol_regime_zscore_trial_0001")  # exit filter, indicator_above
    n_ind_before = len(baseline.indicators)
    n_exit_before = len(baseline.exit_filters)

    with_strat = inject_card_into_strategy(baseline, card)

    assert len(with_strat.indicators) == n_ind_before + 1
    assert with_strat.indicators[-1].fn == "vol_regime_zscore"
    # attach_config target=exit → exit_filters grew, entry filters unchanged.
    assert len(with_strat.exit_filters) == n_exit_before + 1
    assert len(with_strat.filters) == len(baseline.filters)
    # Baseline object itself is not mutated (model_copy).
    assert len(baseline.indicators) == n_ind_before


def test_inject_raises_without_attach_config():
    baseline = load_strategy(REPO / "strategy.yaml")
    card = _load_card("vol_regime_zscore_trial_0001")
    stripped = card.model_copy(update={"attach_config": None})
    with pytest.raises(ValueError, match="no attach_config"):
        inject_card_into_strategy(baseline, stripped)


def test_inject_entry_vs_exit_routing():
    baseline = load_strategy(REPO / "strategy.yaml")
    card = _load_card("vol_regime_zscore_trial_0001")
    # Force an entry-target card and verify it lands in entry filters.
    attach = card.attach_config.model_copy(update={"target": "entry"})
    entry_card = card.model_copy(update={"attach_config": attach})
    with_strat = inject_card_into_strategy(baseline, entry_card)
    assert len(with_strat.filters) == len(baseline.filters) + 1
    assert len(with_strat.exit_filters) == len(baseline.exit_filters)


# ---------- bootstrap_dataframe ----------


def _synth_ohlcv(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.3, n),
            "high": close + np.abs(rng.normal(0, 0.3, n)),
            "low": close - np.abs(rng.normal(0, 0.3, n)),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype("int64"),
        }
    )


def test_bootstrap_dataframe_preserves_shape_and_columns():
    df = _synth_ohlcv(300)
    b = bootstrap_dataframe(df, rng_seed=1, mean_block_len=10)
    assert b.shape == df.shape
    assert list(b.columns) == list(df.columns)
    # Reset integer index.
    assert list(b.index) == list(range(len(df)))


def test_bootstrap_dataframe_deterministic_under_seed():
    df = _synth_ohlcv(300)
    b1 = bootstrap_dataframe(df, rng_seed=7, mean_block_len=10)
    b2 = bootstrap_dataframe(df, rng_seed=7, mean_block_len=10)
    pd.testing.assert_frame_equal(b1, b2)


def test_bootstrap_dataframe_different_seeds_differ():
    df = _synth_ohlcv(300)
    b1 = bootstrap_dataframe(df, rng_seed=1, mean_block_len=10)
    b2 = bootstrap_dataframe(df, rng_seed=2, mean_block_len=10)
    assert not b1["close"].equals(b2["close"])


def test_bootstrap_dataframe_rows_are_resampled_from_source():
    df = _synth_ohlcv(300)
    b = bootstrap_dataframe(df, rng_seed=3, mean_block_len=10)
    # Every resampled close must come from the source close vector.
    src = set(np.round(df["close"].to_numpy(), 9))
    got = set(np.round(b["close"].to_numpy(), 9))
    assert got.issubset(src)


# ---------- stage_indicators_dir ----------


def test_stage_indicators_dir_combines_baseline_and_card(tmp_path):
    dest = tmp_path / "staged"
    stage_indicators_dir(
        REPO / "indicators",
        REPO / "alpha_cards_library" / "indicators",
        "vol_regime_zscore",
        dest,
    )
    # Baseline sma.py present + the card's factor present.
    assert (dest / "sma.py").exists()
    assert (dest / "vol_regime_zscore.py").exists()


def test_stage_indicators_dir_missing_card_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="library indicator source missing"):
        stage_indicators_dir(
            REPO / "indicators",
            REPO / "alpha_cards_library" / "indicators",
            "does_not_exist_factor",
            tmp_path / "staged",
        )
