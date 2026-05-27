"""Aroon Up - Aroon Down (Chande). Aroon Up = (period - bars_since_high)
/ period * 100; Aroon Down similarly with bars_since_low. The
difference is in [-100, +100]: positive => recent high more recent than
recent low (up-trend). Prefix-stable: uses trailing argmax / argmin.
"""



def aroon_up_minus_down(df, period=25, **_):
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    high = df["high"]
    low = df["low"]
    # bars_since_high[t] = period - 1 - argmax over the last `period+1` bars
    # The +1 includes the current bar; period bars look-back means a
    # window of `period+1` values.
    window = period + 1

    def _bars_since_high(s):
        return float(window - 1 - s.values.argmax())

    def _bars_since_low(s):
        return float(window - 1 - s.values.argmin())

    bars_since_high = high.rolling(window).apply(_bars_since_high, raw=False)
    bars_since_low = low.rolling(window).apply(_bars_since_low, raw=False)
    aroon_up = (period - bars_since_high) / period * 100.0
    aroon_down = (period - bars_since_low) / period * 100.0
    return (aroon_up - aroon_down).rename("aroon_up_minus_down")
