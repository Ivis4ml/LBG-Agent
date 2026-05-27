"""Detrended Price Oscillator: close minus a centred SMA, but with the
SMA shifted by `period/2 + 1` bars BACKWARD so the formula stays
prefix-stable (no future bars). Captures price cycles relative to the
trend without forward-looking centring. Prefix-stable.
"""



def detrended_price_oscillator(df, period=20, **_):
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    sma = df["close"].rolling(period).mean()
    shift_back = period // 2 + 1
    return (df["close"] - sma.shift(shift_back)).rename("detrended_price_oscillator")
