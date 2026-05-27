"""Upper wick as fraction of bar range: (high - max(open, close)) /
(high - low). Large upper wick = rejected upside; signal for trend
exhaustion / supply pressure. Prefix-stable.
"""

import pandas as pd


def upper_wick_ratio(df, period=10, **_):
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    body_top = df[["open", "close"]].max(axis=1)
    upper = (df["high"] - body_top).clip(lower=0.0)
    rng = (df["high"] - df["low"]).replace(0.0, pd.NA)
    ratio = (upper / rng).astype(float).clip(lower=0.0, upper=1.0)
    return ratio.rolling(period).mean().rename("upper_wick_ratio")
