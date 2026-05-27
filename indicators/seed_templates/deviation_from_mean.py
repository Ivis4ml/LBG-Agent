"""(close - SMA(period)) -- raw price deviation from the rolling mean.
Unlike bb_zscore, no vol normalisation; preserves absolute spread.
Useful when an absolute price band matters more than vol scaling.
"""



def deviation_from_mean(df, period=20, **_):
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    sma = df["close"].rolling(period).mean()
    return (df["close"] - sma).rename("deviation_from_mean")
