"""Rolling Pearson correlation between close-to-close returns and
log-volume. Positive correlation = up moves on heavier volume (healthy
trend). Negative = volume rises on declines (distribution / stress).
Prefix-stable: rolling correlation of two trailing series.
"""

import numpy as np


def close_volume_corr(df, lookback=30, **_):
    if lookback < 5:
        raise ValueError(f"lookback must be >= 5, got {lookback}")
    ret = df["close"].pct_change()
    logv = np.log(df["volume"].replace(0.0, np.nan))
    return ret.rolling(lookback).corr(logv).rename("close_volume_corr")
