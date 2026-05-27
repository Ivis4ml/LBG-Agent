"""(high - low) / close, z-scored against trailing window. Captures
unusually wide or narrow intraday range — high values mark expansion
days (volatility spikes), low values mark consolidation. Prefix-stable.
"""



def intraday_range_zscore(df, lookback=60, **_):
    if lookback < 5:
        raise ValueError(f"lookback must be >= 5, got {lookback}")
    rng_pct = (df["high"] - df["low"]) / df["close"]
    mean = rng_pct.rolling(lookback).mean()
    std = rng_pct.rolling(lookback).std()
    return ((rng_pct - mean) / std).rename("intraday_range_zscore")
