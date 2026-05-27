"""Rolling fraction of bars where realised vol is above its long-run
median. Captures "high-vol regime persistence" — high values mark
sustained turbulence; low values mark calm. Smoother than instantaneous
vol z-score, better for regime gating. Prefix-stable.
"""

import math

import numpy as np


def vol_regime_persistence(df, vol_period=20, lookback=120, baseline_lookback=252, **_):
    if vol_period < 2 or lookback < 5 or baseline_lookback < lookback:
        raise ValueError(
            f"vol_period >= 2, lookback >= 5, baseline_lookback >= lookback required; "
            f"got {vol_period}, {lookback}, {baseline_lookback}"
        )
    r = np.log(df["close"] / df["close"].shift(1))
    rv = r.rolling(vol_period).std() * math.sqrt(252.0)
    baseline_median = rv.rolling(baseline_lookback).median()
    above = (rv > baseline_median).astype(float)
    return above.rolling(lookback).mean().rename("vol_regime_persistence")
