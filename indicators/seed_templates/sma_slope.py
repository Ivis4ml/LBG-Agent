"""Annualised slope of the trailing SMA: positive => trend up.

Computes a per-bar finite difference of the simple moving average,
scaled to annualised return units (252 trading days). Prefix-stable
because the SMA itself uses trailing rolling windows and the diff is
1-step backward (no future values).
"""



def sma_slope(df, period=50, **_):
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    sma = df["close"].rolling(period).mean()
    # 1-bar log-return-like rate on the SMA, then annualised to 252 bars.
    one_bar = (sma - sma.shift(1)) / sma.shift(1)
    return (one_bar * 252.0).rename("sma_slope")
