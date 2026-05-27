"""Bollinger Band z-score: (close - SMA) / std. Captures how many
standard deviations the price is from its rolling mean. Common mean-
reversion signal: < -1.5 = oversold, > +1.5 = overbought. Prefix-stable.
"""



def bb_zscore(df, period=20, n_std=2.0, **_):
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    if n_std <= 0:
        raise ValueError(f"n_std must be > 0, got {n_std}")
    close = df["close"]
    mean = close.rolling(period).mean()
    std = close.rolling(period).std()
    # n_std is part of the legacy BB parameterization; expose it so the
    # editor can scale the unit (e.g. report z relative to a 1.5-sigma band).
    z = (close - mean) / std
    return (z / n_std * 2.0).rename("bb_zscore")
