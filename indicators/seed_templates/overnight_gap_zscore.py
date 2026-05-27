"""Overnight gap: open / prev_close - 1, z-scored against rolling
window. Large positive = bullish open; large negative = bearish gap
down. Useful as exit filter (e.g. exit after big gap down) or trend
confirmation. Prefix-stable.
"""



def overnight_gap_zscore(df, lookback=60, **_):
    if lookback < 5:
        raise ValueError(f"lookback must be >= 5, got {lookback}")
    gap = df["open"] / df["close"].shift(1) - 1.0
    mean = gap.rolling(lookback).mean()
    std = gap.rolling(lookback).std()
    return ((gap - mean) / std).rename("overnight_gap_zscore")
