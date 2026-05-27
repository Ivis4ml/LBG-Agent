"""Vol jump: ratio of short-window vol to long-window vol. > 1.5 ≈
volatility regime break (sudden expansion); < 0.7 ≈ regime collapse
(consolidation). Different timescale from realised-vol z-score
which uses rolling z. Prefix-stable.
"""

import math

import numpy as np


def vol_jump_indicator(df, short_period=10, long_period=60, **_):
    if short_period < 2 or long_period < short_period:
        raise ValueError(
            f"short_period >= 2 and long_period >= short_period required; "
            f"got {short_period}, {long_period}"
        )
    r = np.log(df["close"] / df["close"].shift(1))
    short_vol = r.rolling(short_period).std() * math.sqrt(252.0)
    long_vol = r.rolling(long_period).std() * math.sqrt(252.0)
    return (short_vol / long_vol).rename("vol_jump_indicator")
