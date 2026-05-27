"""MACD histogram = (EMA_fast - EMA_slow) - signal_EMA. Crosses zero
when momentum accelerates / decelerates relative to its own moving
average. Prefix-stable: all three EMAs use adjust=False trailing form.
"""



def macd_histogram(df, fast=12, slow=26, signal=9, **_):
    if fast < 1 or slow < 1 or signal < 1:
        raise ValueError(f"all periods must be >= 1, got fast={fast} slow={slow} signal={signal}")
    close = df["close"]
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return (macd - signal_line).rename("macd_histogram")
