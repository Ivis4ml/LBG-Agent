"""True Strength Index: double-smoothed momentum oscillator.
TSI = 100 * EMA(EMA(price_diff, slow), fast) /
      EMA(EMA(|price_diff|, slow), fast)
Range roughly [-100, +100]. Less noisy than RSI. Prefix-stable: all
EMAs use adjust=False trailing form.
"""



def tsi(df, fast=13, slow=25, **_):
    if fast < 1 or slow < 1:
        raise ValueError(f"fast >= 1 and slow >= 1 required; got {fast}, {slow}")
    diff = df["close"].diff()
    abs_diff = diff.abs()
    e1 = diff.ewm(span=slow, adjust=False, min_periods=slow).mean()
    e2 = e1.ewm(span=fast, adjust=False, min_periods=fast).mean()
    e1a = abs_diff.ewm(span=slow, adjust=False, min_periods=slow).mean()
    e2a = e1a.ewm(span=fast, adjust=False, min_periods=fast).mean()
    return (100.0 * e2 / e2a).rename("tsi")
