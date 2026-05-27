"""Garman-Klass OHLC volatility estimator:
sigma^2 = 0.5 * ln(H/L)^2 - (2 ln 2 - 1) * ln(C/O)^2.
Uses all four prices; lower variance than Parkinson. Annualised.
Prefix-stable.
"""

import math

import numpy as np


def garman_klass_vol(df, period=20, **_):
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    hl_sq = np.log(df["high"] / df["low"]) ** 2
    co_sq = np.log(df["close"] / df["open"]) ** 2
    coeff = 2.0 * math.log(2.0) - 1.0
    daily_var = 0.5 * hl_sq - coeff * co_sq
    var = daily_var.rolling(period).mean().clip(lower=0.0)
    return (np.sqrt(var) * math.sqrt(252.0)).rename("garman_klass_vol")
