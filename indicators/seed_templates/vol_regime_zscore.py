"""Z-score of realised close-to-close volatility vs its own ~1-year
distribution. > 1.5 = unusually high vol (regime shift); pair with
rearm_threshold around 0.5 to re-enter when vol normalises.
Prefix-stable.
"""

import math

import numpy as np


def vol_regime_zscore(df, vol_period=20, zscore_lookback=252, **_):
    if vol_period < 2 or zscore_lookback < 5:
        raise ValueError(
            f"vol_period >= 2 and zscore_lookback >= 5 required; "
            f"got {vol_period}, {zscore_lookback}"
        )
    log_ret = np.log(df["close"] / df["close"].shift(1))
    realised_vol = log_ret.rolling(vol_period).std() * math.sqrt(252.0)
    mean = realised_vol.rolling(zscore_lookback).mean()
    std = realised_vol.rolling(zscore_lookback).std()
    return ((realised_vol - mean) / std).rename("vol_regime_zscore")
