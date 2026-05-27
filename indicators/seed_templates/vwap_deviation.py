"""Deviation of close from a rolling-window VWAP, normalised by the
window's close-price std. Positive = trading above fair value;
negative = below. Prefix-stable: all rolling windows are trailing.
"""



def vwap_deviation(df, period=20, **_):
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    rolling_pv = pv.rolling(period).sum()
    rolling_v = df["volume"].rolling(period).sum()
    vwap = rolling_pv / rolling_v
    std = df["close"].rolling(period).std()
    return ((df["close"] - vwap) / std).rename("vwap_deviation")
