"""Ratio of yesterday's high-low range to today's, smoothed. > 1 =
range contracting (consolidation); < 1 = range expanding (breakout
phase). Prefix-stable: uses shift(1) only.
"""

import pandas as pd


def prev_day_range_ratio(df, period=10, **_):
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    rng = (df["high"] - df["low"]).replace(0.0, pd.NA)
    ratio = (rng.shift(1) / rng).astype(float)
    return ratio.rolling(period).mean().rename("prev_day_range_ratio")
