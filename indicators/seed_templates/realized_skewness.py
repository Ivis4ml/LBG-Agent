"""Realized skewness: rolling 3rd central moment of log returns,
normalised by rolling std^3. Negative skew = fat left tail (crash
risk). Positive skew = upward jumps. Classic factor for crash-risk
state identification. Prefix-stable.
"""

import numpy as np


def realized_skewness(df, lookback=60, **_):
    if lookback < 5:
        raise ValueError(f"lookback must be >= 5, got {lookback}")
    r = np.log(df["close"] / df["close"].shift(1))
    return r.rolling(lookback).skew().rename("realized_skewness")
