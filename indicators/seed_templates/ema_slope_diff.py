"""Fast EMA slope minus slow SMA slope (both annualised). When the
fast slope is meaningfully above the slow slope, the trend is
accelerating; below, decelerating. Useful at entry timing.
Prefix-stable.
"""



def ema_slope_diff(df, fast_period=20, slow_period=60, **_):
    if fast_period < 2 or slow_period < 2:
        raise ValueError(f"periods >= 2 required; got {fast_period}, {slow_period}")
    close = df["close"]
    fast_ema = close.ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
    slow_sma = close.rolling(slow_period).mean()
    fast_slope = (fast_ema - fast_ema.shift(1)) / fast_ema.shift(1) * 252.0
    slow_slope = (slow_sma - slow_sma.shift(1)) / slow_sma.shift(1) * 252.0
    return (fast_slope - slow_slope).rename("ema_slope_diff")
