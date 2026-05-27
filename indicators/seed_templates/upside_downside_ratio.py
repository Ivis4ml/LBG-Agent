"""Ratio of upside vol to downside vol over a trailing window.
> 1 = upside dominates (bullish asymmetry); < 1 = downside dominates
(bearish asymmetry, risk-off regime). Mid-frequency signal.
Prefix-stable.
"""

import numpy as np


def upside_downside_ratio(df, lookback=20, **_):
    if lookback < 2:
        raise ValueError(f"lookback must be >= 2, got {lookback}")
    r = np.log(df["close"] / df["close"].shift(1))
    up = r.where(r > 0.0, 0.0)
    down = r.where(r < 0.0, 0.0)
    up_std = up.rolling(lookback).std()
    down_std = down.rolling(lookback).std().replace(0.0, np.nan)
    return (up_std / down_std).rename("upside_downside_ratio")
