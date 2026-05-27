"""Ratio of current volume to its trailing SMA. > 1 = above-average
participation; classic confirmation signal for breakouts (volume
expansion). Prefix-stable.
"""



def volume_ma_ratio(df, period=20, **_):
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    v = df["volume"]
    sma = v.rolling(period).mean()
    return (v / sma).rename("volume_ma_ratio")
