"""Parkinson high-low volatility estimator: sqrt( (1 / (4 ln 2)) *
mean( ln(H/L)^2 ) ). Tighter than close-to-close vol because it uses
the intra-bar range. Annualised by sqrt(252). Prefix-stable.
"""

import math

import numpy as np


def parkinson_vol(df, period=20, **_):
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    log_hl_sq = (np.log(df["high"] / df["low"]) ** 2).astype(float)
    constant = 1.0 / (4.0 * math.log(2.0))
    var = log_hl_sq.rolling(period).mean() * constant
    return (np.sqrt(var) * math.sqrt(252.0)).rename("parkinson_vol")
