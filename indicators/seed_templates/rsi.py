"""Relative Strength Index (Wilder 1978). Range [0, 100]. < 30 oversold,
> 70 overbought. Computed with Wilder's smoothing (EMA at alpha=1/period).
Prefix-stable because every rolling op is trailing.
"""

import pandas as pd


def rsi(df, period=14, **_):
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    delta = df["close"].diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    alpha = 1.0 / period
    avg_up = up.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_down = down.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    rs = avg_up / avg_down.replace(0.0, pd.NA)
    return (100.0 - (100.0 / (1.0 + rs))).astype(float).rename("rsi")
