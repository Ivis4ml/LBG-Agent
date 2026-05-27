"""Tests for lbg.verdict.synthetic_null."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lbg.verdict.synthetic_null import (
    bootstrap_spy_parquet,
    stationary_block_indices,
)


def _make_synth_parquet(path: Path, n: int = 200) -> Path:
    """Helper: build a tiny synthetic SPY-like parquet."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    close = 100.0 + np.cumsum(rng.normal(0.05, 1.0, n))
    df = pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.5, n),
            "high": close + np.abs(rng.normal(0, 0.5, n)),
            "low": close - np.abs(rng.normal(0, 0.5, n)),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype("int64"),
            "close_unadjusted": close,
        },
        index=idx,
    )
    df.index.name = "date"
    df.to_parquet(path)
    return path


# ---------- stationary_block_indices ----------


def test_stationary_block_indices_in_range_and_length():
    rng = np.random.default_rng(0)
    idx = stationary_block_indices(500, mean_block_len=10, rng=rng)
    assert idx.shape == (500,)
    assert idx.min() >= 0
    assert idx.max() < 500


def test_stationary_block_indices_block_length_geometric():
    rng = np.random.default_rng(0)
    idx = stationary_block_indices(5000, mean_block_len=10, rng=rng)
    # Mean run length of consecutive +1 steps should approach 10.
    diffs = np.diff(idx) % 5000
    is_continuation = diffs == 1
    run_lengths = []
    current = 1
    for cont in is_continuation:
        if cont:
            current += 1
        else:
            run_lengths.append(current)
            current = 1
    run_lengths.append(current)
    mean_run = float(np.mean(run_lengths))
    assert 7.0 < mean_run < 14.0


def test_stationary_block_indices_rejects_bad_inputs():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n must be positive"):
        stationary_block_indices(0, mean_block_len=10, rng=rng)
    with pytest.raises(ValueError, match="mean_block_len must be"):
        stationary_block_indices(100, mean_block_len=0, rng=rng)


# ---------- bootstrap_spy_parquet ----------


def test_bootstrap_preserves_schema_and_index(tmp_path):
    src = _make_synth_parquet(tmp_path / "src.parquet", n=200)
    dst = tmp_path / "boot.parquet"
    bootstrap_spy_parquet(src, dst, rng_seed=42, mean_block_len=10)
    src_df = pd.read_parquet(src)
    dst_df = pd.read_parquet(dst)
    # Identical schema and row count and date index.
    assert list(dst_df.columns) == list(src_df.columns)
    assert len(dst_df) == len(src_df)
    pd.testing.assert_index_equal(dst_df.index, src_df.index)


def test_bootstrap_resamples_rows(tmp_path):
    src = _make_synth_parquet(tmp_path / "src.parquet", n=400)
    dst = tmp_path / "boot.parquet"
    bootstrap_spy_parquet(src, dst, rng_seed=7, mean_block_len=10)
    src_df = pd.read_parquet(src)
    dst_df = pd.read_parquet(dst)
    # Almost certainly differs from source on most rows (probability of
    # an identical resample is < 1 / 400).
    diff = (src_df["close"].values != dst_df["close"].values).sum()
    assert diff > 100, f"only {diff} rows changed; bootstrap may be degenerate"
    # But each resampled close must still exist in the source close
    # vector (we're only permuting/repeating rows, not synthesizing
    # new values).
    src_set = set(np.round(src_df["close"].values, 9))
    dst_set = set(np.round(dst_df["close"].values, 9))
    assert dst_set.issubset(src_set)


def test_bootstrap_deterministic_under_seed(tmp_path):
    src = _make_synth_parquet(tmp_path / "src.parquet", n=200)
    dst1 = tmp_path / "boot1.parquet"
    dst2 = tmp_path / "boot2.parquet"
    bootstrap_spy_parquet(src, dst1, rng_seed=99, mean_block_len=10)
    bootstrap_spy_parquet(src, dst2, rng_seed=99, mean_block_len=10)
    pd.testing.assert_frame_equal(pd.read_parquet(dst1), pd.read_parquet(dst2))


def test_bootstrap_different_seeds_yield_different_samples(tmp_path):
    src = _make_synth_parquet(tmp_path / "src.parquet", n=200)
    dst1 = tmp_path / "boot1.parquet"
    dst2 = tmp_path / "boot2.parquet"
    bootstrap_spy_parquet(src, dst1, rng_seed=1, mean_block_len=10)
    bootstrap_spy_parquet(src, dst2, rng_seed=2, mean_block_len=10)
    df1 = pd.read_parquet(dst1)
    df2 = pd.read_parquet(dst2)
    assert not df1["close"].equals(df2["close"])


def test_bootstrap_within_row_invariants_preserved(tmp_path):
    """Stationary bootstrap of whole rows means OHLC stay coherent
    within each row even if day-to-day continuity is broken.
    """
    src = _make_synth_parquet(tmp_path / "src.parquet", n=200)
    dst = tmp_path / "boot.parquet"
    bootstrap_spy_parquet(src, dst, rng_seed=42, mean_block_len=10)
    dst_df = pd.read_parquet(dst)
    # Original synthetic data has high >= close and low <= close; the
    # bootstrap shouldn't violate that because each row is moved as
    # a unit.
    assert (dst_df["high"] >= dst_df["close"]).all()
    assert (dst_df["low"] <= dst_df["close"]).all()


def test_bootstrap_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="source parquet"):
        bootstrap_spy_parquet(tmp_path / "nope.parquet", tmp_path / "out.parquet", rng_seed=1)


# ---------- LBG_DATA_PATH override on load_split ----------


def test_load_split_honors_env_override(tmp_path, monkeypatch):
    """Setting LBG_DATA_PATH must redirect the loader."""
    # Build a tiny "real-shape" parquet whose date range covers split_C.
    n = 1100
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "open": np.linspace(100, 200, n) + rng.normal(0, 0.5, n),
            "high": np.linspace(100, 200, n) + 1.0,
            "low": np.linspace(100, 200, n) - 1.0,
            "close": np.linspace(100, 200, n),
            "volume": rng.integers(1_000_000, 5_000_000, n).astype("int64"),
            "close_unadjusted": np.linspace(100, 200, n),
        },
        index=idx,
    )
    df.index.name = "date"
    parq = tmp_path / "fake_spy.parquet"
    df.to_parquet(parq)

    monkeypatch.setenv("LBG_DATA_PATH", str(parq))
    from lbg.data.loader import load_split

    sealed = load_split("split_C")
    # Should pull bars in the 2021-01-01 to 2024-12-31 range from the
    # override file. Our fake parquet starts in 2021, so we expect a
    # non-empty slice with the override's column set.
    assert len(sealed) > 0
    assert set(sealed.columns) >= {"open", "high", "low", "close", "volume"}
