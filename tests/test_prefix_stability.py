"""Tests for `lbg.invariants.prefix_stability`.

The good `indicators/sma.py` must pass. The leaky fixtures
(`bad_whole_series_norm`, `bad_dynamic_lookahead`, `bad_future_shift`) must
fail. Tests use synthetic 200-bar OHLCV so they are fast and decoupled from
the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicators.sma import sma
from lbg.invariants import check_prefix_stability
from tests.fixtures.indicators.bad_dynamic_lookahead import (
    normalize_to_global_max,
)
from tests.fixtures.indicators.bad_future_shift import leak_via_shift
from tests.fixtures.indicators.bad_whole_series_norm import whole_series_zscore


def _synthetic_ohlcv(n: int = 200, seed: int = 7) -> pd.DataFrame:
    """Random-walk close, with derived O/H/L/V satisfying the usual ordering."""
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(loc=0.0005, scale=0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(log_rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    volume = rng.integers(1_000_000, 10_000_000, size=n).astype("int64")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


# -------- good indicator passes --------


def test_sma_passes_prefix_stability():
    df = _synthetic_ohlcv()
    violations = check_prefix_stability(
        sma, df, params={"period": 20}, n_samples=15, n_perturbations_per_t=2
    )
    assert violations == [], violations


def test_constant_function_passes():
    """Vacuously prefix-stable: output is independent of input."""
    df = _synthetic_ohlcv()

    def always_one(d: pd.DataFrame) -> pd.Series:
        return pd.Series(np.ones(len(d)), index=d.index)

    violations = check_prefix_stability(always_one, df, n_samples=10, n_perturbations_per_t=2)
    assert violations == []


def test_purely_past_lookback_passes():
    """`close - close.shift(1)` (yesterday's diff) is prefix-stable."""
    df = _synthetic_ohlcv()

    def lag1_return(d: pd.DataFrame) -> pd.Series:
        return d["close"] - d["close"].shift(1)

    violations = check_prefix_stability(lag1_return, df, n_samples=10, n_perturbations_per_t=2)
    assert violations == []


# -------- leaky indicators fail --------


def test_whole_series_zscore_fails():
    df = _synthetic_ohlcv()
    violations = check_prefix_stability(
        whole_series_zscore, df, n_samples=10, n_perturbations_per_t=2
    )
    assert len(violations) >= 1
    assert all(v.name == "prefix_stability" for v in violations)


def test_normalize_to_global_max_fails():
    df = _synthetic_ohlcv()
    violations = check_prefix_stability(
        normalize_to_global_max, df, n_samples=10, n_perturbations_per_t=2
    )
    assert len(violations) >= 1
    assert "first differing index" in violations[0].message


def test_future_shift_fails():
    df = _synthetic_ohlcv()
    violations = check_prefix_stability(leak_via_shift, df, n_samples=10, n_perturbations_per_t=2)
    assert len(violations) >= 1


# -------- contract checks --------


def test_short_df_raises():
    df = _synthetic_ohlcv(n=30)
    with pytest.raises(ValueError, match=">= 50 bars"):
        check_prefix_stability(sma, df, params={"period": 5})


def test_wrong_return_type_raises():
    df = _synthetic_ohlcv()

    def bad(d: pd.DataFrame):
        return d["close"].values  # ndarray, not Series

    with pytest.raises(TypeError, match="pandas.Series"):
        check_prefix_stability(bad, df, n_samples=2, n_perturbations_per_t=1)


def test_wrong_length_raises():
    df = _synthetic_ohlcv()

    def bad(d: pd.DataFrame) -> pd.Series:
        return d["close"].iloc[:50]

    with pytest.raises(ValueError, match="output length"):
        check_prefix_stability(bad, df, n_samples=2, n_perturbations_per_t=1)


# -------- determinism --------


def test_deterministic_with_same_seed():
    df = _synthetic_ohlcv()
    a = check_prefix_stability(
        whole_series_zscore,
        df,
        n_samples=5,
        n_perturbations_per_t=1,
        rng_seed=99,
    )
    b = check_prefix_stability(
        whole_series_zscore,
        df,
        n_samples=5,
        n_perturbations_per_t=1,
        rng_seed=99,
    )
    assert len(a) == len(b)
    assert [v.message for v in a] == [v.message for v in b]


# -------- perturbation preserves head exactly --------


def test_perturbation_leaves_head_byte_identical():
    """`_perturb_after` must not touch any row <= t."""
    from lbg.invariants.prefix_stability import _perturb_after

    df = _synthetic_ohlcv(n=100)
    rng = np.random.default_rng(0)
    t = 40
    perturbed = _perturb_after(df, t, rng)
    pd.testing.assert_frame_equal(
        df.iloc[: t + 1].reset_index(drop=True),
        perturbed.iloc[: t + 1].reset_index(drop=True),
    )
    # And rows after t must differ on at least one column
    assert not df.iloc[t + 1 :].equals(perturbed.iloc[t + 1 :])
