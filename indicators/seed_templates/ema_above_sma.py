"""EMA(fast) - SMA(slow). Positive when the fast EMA is above the
slow SMA -- a classic trend-up regime indicator. Prefix-stable: both
EMA and SMA use trailing windows; subtraction is per-bar.
"""



def ema_above_sma(df, fast_period=20, slow_period=100, **_):
    if fast_period < 2 or slow_period < 2:
        raise ValueError(f"periods must be >= 2, got fast={fast_period} slow={slow_period}")
    fast_ema = df["close"].ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
    slow_sma = df["close"].rolling(slow_period).mean()
    return (fast_ema - slow_sma).rename("ema_above_sma")
