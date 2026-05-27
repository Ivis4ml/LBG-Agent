"""Intraday return: close / open - 1, smoothed with a rolling mean.
Decomposes total return into the intraday (open-to-close) component.
Different signal vs overnight (which uses close-to-next-open).
Prefix-stable.
"""



def intraday_return(df, period=5, **_):
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    raw = df["close"] / df["open"] - 1.0
    return raw.rolling(period).mean().rename("intraday_return")
