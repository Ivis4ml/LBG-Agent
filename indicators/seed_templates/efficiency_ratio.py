"""Kaufman's Efficiency Ratio: |price change over N bars| / sum of
absolute 1-bar changes. Ranges [0, 1]. Near 1 = clean trend; near 0
= choppy / no-trend. Cheap regime indicator separate from vol.
Prefix-stable.
"""

import pandas as pd


def efficiency_ratio(df, period=20, **_):
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    close = df["close"]
    change = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period).sum().replace(0.0, pd.NA)
    er = (change / volatility).astype(float).clip(lower=0.0, upper=1.0)
    return er.rename("efficiency_ratio")
