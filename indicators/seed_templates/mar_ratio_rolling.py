"""Rolling MAR ratio: trailing-window CAGR / |max drawdown|. Higher is
better. Diagnostic of risk-adjusted regime; low values mark "bad
periods" where any strategy is structurally penalised. Prefix-stable.
"""

import numpy as np


def mar_ratio_rolling(df, lookback=252, **_):
    if lookback < 30:
        raise ValueError(f"lookback must be >= 30 for a meaningful CAGR, got {lookback}")
    close = df["close"]
    # CAGR over the window: (close / close.shift(L)) ^ (252/L) - 1
    growth = close / close.shift(lookback)
    cagr = growth.pow(252.0 / lookback) - 1.0
    # MaxDD over the same trailing window.
    peak = close.rolling(lookback, min_periods=1).max()
    drawdown = ((peak - close) / peak).clip(lower=0.0)
    mdd = drawdown.rolling(lookback, min_periods=1).max().replace(0.0, np.nan)
    return (cagr / mdd).rename("mar_ratio_rolling")
