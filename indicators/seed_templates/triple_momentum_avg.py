"""Average of momentum at three horizons (short / medium / long).
A composite robust to which horizon happens to fit best; smooths
the noisy single-horizon ROC signal. Prefix-stable.
"""



def triple_momentum_avg(df, short=20, medium=60, long=120, **_):
    if short < 1 or medium < short or long < medium:
        raise ValueError(f"1 <= short <= medium <= long required; got {short}, {medium}, {long}")
    close = df["close"]
    m1 = close / close.shift(short) - 1.0
    m2 = close / close.shift(medium) - 1.0
    m3 = close / close.shift(long) - 1.0
    return ((m1 + m2 + m3) / 3.0).rename("triple_momentum_avg")
