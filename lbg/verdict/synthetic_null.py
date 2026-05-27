"""Synthetic null calibration helpers (Significance Hardening step 4).

The goal is to measure the operating characteristic of the full
Discovery pipeline against a known-null version of the data: same SPY
daily structure, but with the day-to-day temporal information broken
by stationary block resampling. Acceptances on this data are by
construction false discoveries.

The caveat (advisor): the LLM still searches its training prior over
factor names (ADX, MACD, kama_ratio, ...) — bootstrap doesn't deflate
*that* prior, only the underlying data. So a 30% null acceptance rate
here is the operating characteristic of "LLM prior over factors against
randomized data", not the pure statistical FDR.

This module produces the bootstrapped SPY parquet; the orchestration
script in `scripts/synthetic_null_calibration.py` chains it into a
campaign run via the `LBG_DATA_PATH` override added to
`lbg.data.loader.load_split`.

Bootstrap design
----------------
We resample *whole rows* (a daily OHLCV bar plus the unadjusted-close
column) using Politis-Romano stationary block resampling on the row
index. The original DatetimeIndex is preserved so that
`_SPLIT_BOUNDS`'s date-based slicing still produces the same split
sizes (~2266 / ~505 / ~1006 bars).

This breaks day-to-day continuity (today's open ≠ yesterday's resampled
close), but preserves:

  · the marginal distribution of each column
  · within-row OHLC consistency (high ≥ close ≥ low etc.)
  · short-range autocorrelation up to the block length

For null-acceptance-rate calibration this is sufficient — any acceptance
on this data is, by construction, attributable to either gate over-
fitting or the LLM's prior over factor names. We are not claiming the
bootstrap respects every joint moment of the underlying process.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def stationary_block_indices(
    n: int,
    *,
    mean_block_len: int = 10,
    rng: np.random.Generator,
) -> np.ndarray:
    """Politis-Romano (1994) stationary bootstrap of indices into [0, n).

    Block lengths are i.i.d. Geometric(p = 1 / mean_block_len), giving
    an expected block length of `mean_block_len`. Block starts are
    uniform on [0, n). Blocks wrap modulo n when they run off the end.
    Returns an int array of length n.
    """
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")
    if mean_block_len < 1:
        raise ValueError(f"mean_block_len must be >= 1; got {mean_block_len}")
    p = 1.0 / mean_block_len
    out = np.empty(n, dtype=np.int64)
    i = 0
    while i < n:
        start = int(rng.integers(0, n))
        block_len = int(rng.geometric(p))
        end = min(i + block_len, n)
        for j in range(i, end):
            out[j] = (start + (j - i)) % n
        i = end
    return out


def bootstrap_spy_parquet(
    source_path: Path,
    dest_path: Path,
    *,
    rng_seed: int,
    mean_block_len: int = 10,
) -> Path:
    """Read SPY parquet, stationary-bootstrap its rows, write to `dest_path`.

    The destination parquet has identical schema and date index (and
    therefore identical split sizes) as the source; only the row
    contents have been resampled. Returns `dest_path` for convenience.
    """
    source_path = Path(source_path)
    dest_path = Path(dest_path)
    if not source_path.exists():
        raise FileNotFoundError(f"source parquet missing at {source_path}")

    raw = pd.read_parquet(source_path)
    n = len(raw)
    if n == 0:
        raise ValueError(f"source parquet is empty: {source_path}")

    rng = np.random.default_rng(rng_seed)
    idx = stationary_block_indices(n, mean_block_len=mean_block_len, rng=rng)
    resampled = raw.iloc[idx].reset_index(drop=True)
    # Glue the original date index back on so split slicing still works.
    resampled.index = raw.index

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    resampled.to_parquet(dest_path)
    return dest_path
