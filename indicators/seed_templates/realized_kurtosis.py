"""Realized excess kurtosis of log returns over a trailing window.
High kurtosis = fat tails / jump-heavy regime. Often pairs with
realized_skewness for "tail risk regime" detection. Prefix-stable.
"""

import numpy as np


def realized_kurtosis(df, lookback=60, **_):
    if lookback < 5:
        raise ValueError(f"lookback must be >= 5, got {lookback}")
    r = np.log(df["close"] / df["close"].shift(1))
    return r.rolling(lookback).kurt().rename("realized_kurtosis")
