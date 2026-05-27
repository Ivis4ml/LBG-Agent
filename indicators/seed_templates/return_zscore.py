"""Rolling z-score of N-bar cumulative log return. Captures "how far
from neutral has price moved" in standardised units. Different from
velocity_zscore (1-bar) — this is N-bar accumulated. Useful for
mean-reversion entry timing. Prefix-stable.
"""

import numpy as np


def return_zscore(df, window=10, zscore_lookback=120, **_):
    if window < 1 or zscore_lookback < 5:
        raise ValueError(
            f"window >= 1 and zscore_lookback >= 5 required; got {window}, {zscore_lookback}"
        )
    r = np.log(df["close"] / df["close"].shift(window))
    mean = r.rolling(zscore_lookback).mean()
    std = r.rolling(zscore_lookback).std()
    return ((r - mean) / std).rename("return_zscore")
