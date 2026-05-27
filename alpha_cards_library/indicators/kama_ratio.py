"""Kaufman Adaptive Moving Average ratio: close / KAMA. KAMA varies
its smoothing constant from "fast" to "slow" based on the noise-to-
trend ratio (efficiency ratio). Returns the ratio of close to KAMA
so the indicator is unit-free. Prefix-stable: efficiency uses only
trailing diffs; EMA part is causal.
"""

import numpy as np
import pandas as pd


def kama_ratio(df, period=10, fast=2, slow=30, **_):
    if period < 2 or fast < 1 or slow < fast:
        raise ValueError(
            f"period >= 2 and 1 <= fast <= slow required; got {period}, {fast}, {slow}"
        )
    close = df["close"]
    change = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period).sum()
    er = (change / volatility.replace(0.0, np.nan)).clip(lower=0.0, upper=1.0).fillna(0.0)
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama = pd.Series(np.nan, index=close.index)
    prev = close.iloc[0]
    for i in range(len(close)):
        s = sc.iloc[i]
        if np.isnan(s):
            kama.iloc[i] = prev
            continue
        prev = prev + s * (close.iloc[i] - prev)
        kama.iloc[i] = prev
    return (close / kama).rename("kama_ratio")
