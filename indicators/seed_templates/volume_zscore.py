"""Volume z-score: (volume - rolling mean) / rolling std. Captures
unusually high or low participation vs the typical level. Prefix-stable.
"""



def volume_zscore(df, lookback=60, **_):
    if lookback < 5:
        raise ValueError(f"lookback must be >= 5, got {lookback}")
    v = df["volume"]
    mean = v.rolling(lookback).mean()
    std = v.rolling(lookback).std()
    return ((v - mean) / std).rename("volume_zscore")
