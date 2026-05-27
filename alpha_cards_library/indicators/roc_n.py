"""Rate of change over N bars: (close - close.shift(N)) / close.shift(N).
Simple, fast, no smoothing. Sign is the only signal most use cases need.
Prefix-stable by construction (1-call shift).
"""



def roc_n(df, period=20, **_):
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    prev = df["close"].shift(period)
    return ((df["close"] - prev) / prev).rename("roc_n")
