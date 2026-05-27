"""Annualised standard deviation of NEGATIVE-only log returns over a
trailing window. Captures left-tail volatility specifically; pairs
with upside vol to form upside/downside vol ratio. Prefix-stable.
"""

import math

import numpy as np


def downside_volatility(df, lookback=20, **_):
    if lookback < 2:
        raise ValueError(f"lookback must be >= 2, got {lookback}")
    r = np.log(df["close"] / df["close"].shift(1))
    downside = r.where(r < 0.0, 0.0)
    std = downside.rolling(lookback).std()
    return (std * math.sqrt(252.0)).rename("downside_volatility")
