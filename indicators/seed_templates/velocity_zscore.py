"""Z-score of price 1-bar log return against its trailing distribution.
Captures "unusual move today vs typical day". Symmetric: large
positive = unusual up move, large negative = unusual down move.
Different signal from momentum (cumulative) and from volatility
(magnitude only). Prefix-stable.
"""

import numpy as np


def velocity_zscore(df, lookback=60, **_):
    if lookback < 5:
        raise ValueError(f"lookback must be >= 5, got {lookback}")
    r = np.log(df["close"] / df["close"].shift(1))
    mean = r.rolling(lookback).mean()
    std = r.rolling(lookback).std()
    return ((r - mean) / std).rename("velocity_zscore")
