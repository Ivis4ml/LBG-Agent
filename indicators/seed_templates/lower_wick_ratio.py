"""Lower wick as fraction of bar range: (min(open, close) - low) /
(high - low). Large lower wick = buying support / hammer pattern.
Prefix-stable.
"""

import pandas as pd


def lower_wick_ratio(df, period=10, **_):
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    body_bottom = df[["open", "close"]].min(axis=1)
    lower = (body_bottom - df["low"]).clip(lower=0.0)
    rng = (df["high"] - df["low"]).replace(0.0, pd.NA)
    ratio = (lower / rng).astype(float).clip(lower=0.0, upper=1.0)
    return ratio.rolling(period).mean().rename("lower_wick_ratio")
