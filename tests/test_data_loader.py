"""Tests for lbg.data.loader.

Network-free tests (cross-check arithmetic) always run. Tests that exercise
the on-disk Parquet cache run only when the cache exists locally (created by
`uv run python -m lbg.data.loader fetch`).
"""

from __future__ import annotations

import pandas as pd
import pytest

from lbg.data.loader import (
    _SPLIT_BOUNDS,
    CROSS_CHECK_TOLERANCE,
    OHLCV_COLUMNS,
    RAW_PARQUET_PATH,
    _cross_check,
    load_split,
    split_size,
    summarize_split,
)

# -------- cross-check function (no network) --------


def _make_close_series(n: int = 300, start: float = 100.0, step: float = 0.5) -> pd.Series:
    idx = pd.date_range("2018-01-02", periods=n, freq="B")
    return pd.Series([start + i * step for i in range(n)], index=idx, name="close")


def test_cross_check_passes_when_sources_agree():
    yf_close = _make_close_series()
    tiingo = pd.DataFrame({"close": yf_close.values}, index=yf_close.index)
    summary = _cross_check(yf_close, tiingo, tol=CROSS_CHECK_TOLERANCE)
    assert summary["shared_bars"] == len(yf_close)
    assert summary["max_rel_diff"] == 0.0
    assert summary["fraction_within_tol"] == 1.0


def test_cross_check_fails_when_one_bar_disagrees_beyond_tol():
    yf_close = _make_close_series()
    tiingo = pd.DataFrame({"close": yf_close.values}, index=yf_close.index)
    # Inject a 5% disagreement on one bar (far above 0.5% tol).
    tiingo.iloc[50, 0] *= 1.05
    with pytest.raises(ValueError, match="disagree beyond tolerance"):
        _cross_check(yf_close, tiingo, tol=CROSS_CHECK_TOLERANCE)


def test_cross_check_fails_when_overlap_too_small():
    yf_close = _make_close_series(n=50)
    tiingo = pd.DataFrame({"close": yf_close.values}, index=yf_close.index)
    with pytest.raises(ValueError, match="Too few overlapping bars"):
        _cross_check(yf_close, tiingo, tol=CROSS_CHECK_TOLERANCE)


# -------- live split loading (needs Parquet cache) --------


pytestmark_needs_cache = pytest.mark.skipif(
    not RAW_PARQUET_PATH.exists(),
    reason="Parquet cache absent; run `uv run python -m lbg.data.loader fetch` first.",
)


@pytestmark_needs_cache
def test_split_a_row_count_matches_proposal_within_tolerance():
    """PROPOSAL.html §5: split_A ~ 2266 bars; allow +/- 5 for holiday-calendar drift."""
    n = split_size("split_A")
    assert 2261 <= n <= 2271, n


@pytestmark_needs_cache
def test_split_b_and_c_row_counts_match_proposal_within_tolerance():
    nb = split_size("split_B")
    nc = split_size("split_C")
    assert 500 <= nb <= 510, nb
    assert 1001 <= nc <= 1011, nc


@pytestmark_needs_cache
def test_load_split_returns_only_ohlcv_and_no_date_index():
    df = load_split("split_A")
    assert list(df.columns) == list(OHLCV_COLUMNS)
    assert isinstance(df.index, pd.RangeIndex), f"index leak risk: got {type(df.index).__name__}"
    # No column that even smells like a date.
    forbidden = {"date", "Date", "DATE", "timestamp", "year"}
    assert not (set(df.columns) & forbidden)


@pytestmark_needs_cache
def test_load_split_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown split"):
        load_split("split_Z")  # type: ignore[arg-type]


@pytestmark_needs_cache
def test_load_split_has_clean_ohlcv_values():
    """Production-data sanity check: no NaN, no non-positive close, no negative volume."""
    for split in _SPLIT_BOUNDS:
        df = load_split(split)
        s = summarize_split(df)
        assert s["nan_cells"] == 0, (split, s)
        assert s["non_positive_close"] == 0, (split, s)
        assert s["negative_volume"] == 0, (split, s)


@pytestmark_needs_cache
def test_split_a_high_low_close_consistency():
    """High >= max(open, close), Low <= min(open, close) on every bar."""
    df = load_split("split_A")
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all(), (
        "found bars where high < max(open, close)"
    )
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all(), (
        "found bars where low > min(open, close)"
    )


@pytestmark_needs_cache
def test_splits_are_disjoint_in_length_terms():
    """Total bars across splits equals fetched bars (no overlap, no gap)."""
    cached = pd.read_parquet(RAW_PARQUET_PATH)
    a = split_size("split_A")
    b = split_size("split_B")
    c = split_size("split_C")
    assert a + b + c == len(cached), (a, b, c, len(cached))
