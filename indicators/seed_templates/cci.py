"""Commodity Channel Index (Lambert). Measures the typical price's
deviation from its SMA, normalised by mean absolute deviation. Values
> +100 ≈ overbought; < -100 ≈ oversold. Different normalisation from
RSI / BB-zscore — uses absolute deviation, not std. Prefix-stable.
"""



def cci(df, period=20, constant=0.015, **_):
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    if constant <= 0:
        raise ValueError(f"constant must be > 0, got {constant}")
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    sma = typical.rolling(period).mean()
    mad = typical.rolling(period).apply(lambda x: (x - x.mean()).abs().mean(), raw=False)
    return ((typical - sma) / (constant * mad)).rename("cci")
