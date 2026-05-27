"""close / SMA(period). The "stay-long-only-above-200-SMA" classic.
A very stable trend-regime filter with low false-positive rate vs
short-window momentum indicators. Prefix-stable.
"""



def trend_regime_sma_long(df, period=200, **_):
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    sma = df["close"].rolling(period).mean()
    return (df["close"] / sma).rename("trend_regime_sma_long")
