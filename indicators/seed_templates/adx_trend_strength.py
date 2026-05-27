"""ADX (Wilder 1978) trend strength estimator. Direction-agnostic:
> 20-25 = trending market, < 20 = chop. Computed via Wilder's smoothing
(equivalent to an EMA with alpha = 1/period). Prefix-stable because
every EMA / rolling sum is trailing.
"""

import numpy as np
import pandas as pd


def adx_trend_strength(df, period=14, **_):
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr_components = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    tr = tr_components.max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move.clip(lower=0.0)
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move.clip(lower=0.0)

    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    return adx.rename("adx_trend_strength")
