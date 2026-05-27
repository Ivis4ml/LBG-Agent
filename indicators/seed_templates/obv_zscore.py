"""On-Balance Volume z-score. OBV is the running sum of signed volume
(positive on up bars, negative on down bars). Z-score against own
rolling window distinguishes "unusual accumulation" from baseline.
Prefix-stable: cumulative sum + trailing z-score window.
"""



def obv_zscore(df, lookback=60, **_):
    if lookback < 5:
        raise ValueError(f"lookback must be >= 5, got {lookback}")
    sign = df["close"].diff().apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    obv = (sign * df["volume"]).cumsum()
    mean = obv.rolling(lookback).mean()
    std = obv.rolling(lookback).std()
    return ((obv - mean) / std).rename("obv_zscore")
