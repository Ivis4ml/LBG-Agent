"""Average True Range as a fraction of close. Captures intra-bar
volatility on a unit-free scale. Wilder smoothing (EMA alpha=1/period).
> 2.5% typically signals an unusually wide bar. Prefix-stable.
"""

import pandas as pd


def atr_pct(df, period=14, **_):
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return (atr / close).rename("atr_pct")
