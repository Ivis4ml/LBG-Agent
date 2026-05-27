"""close / SMA(period). Dimensionless trend / extension indicator.
> 1.0 = above moving average; > 1.10 often used as "stretched"; < 1.0
as bearish. Prefix-stable.
"""



def close_to_sma_ratio(df, period=50, **_):
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    sma = df["close"].rolling(period).mean()
    return (df["close"] / sma).rename("close_to_sma_ratio")
