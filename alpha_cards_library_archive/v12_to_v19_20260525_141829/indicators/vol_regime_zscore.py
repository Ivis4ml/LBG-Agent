import pandas as pd

def vol_regime_zscore(df, vol_lookback=20, z_lookback=60, **_):
    if vol_lookback < 2:
        raise ValueError(f"vol_lookback must be >= 2, got {vol_lookback}")
    if z_lookback < 2:
        raise ValueError(f"z_lookback must be >= 2, got {z_lookback}")
    ret = df["close"].pct_change()
    vol = ret.rolling(vol_lookback).std()
    mean = vol.rolling(z_lookback).mean()
    std = vol.rolling(z_lookback).std()
    z = (vol - mean) / std.replace(0.0, pd.NA)
    return z.astype(float).rename("vol_regime_zscore")
