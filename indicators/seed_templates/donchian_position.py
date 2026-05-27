"""Where close sits inside the trailing N-bar Donchian channel,
normalised to [0, 1]. 0 = at the period low, 1 = at the period high,
0.5 = middle. Prefix-stable: uses trailing rolling max/min.
"""

import pandas as pd


def donchian_position(df, period=55, **_):
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    high_n = df["high"].rolling(period).max()
    low_n = df["low"].rolling(period).min()
    width = (high_n - low_n).replace(0.0, pd.NA)
    pos = (df["close"] - low_n) / width
    return pos.astype(float).clip(lower=0.0, upper=1.0).rename("donchian_position")
