"""Ratio of bar body (|close - open|) to bar range (high - low).
Near 1 = decisive trend bar (full-body marubozu); near 0 = doji /
indecision. Rolling mean smooths it. Prefix-stable.
"""

import pandas as pd


def body_to_range(df, period=10, **_):
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0.0, pd.NA)
    ratio = (body / rng).astype(float).clip(lower=0.0, upper=1.0)
    return ratio.rolling(period).mean().rename("body_to_range")
