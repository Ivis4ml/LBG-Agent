"""Ulcer Index: root-mean-square of percentage drawdowns over a
trailing window. Penalises sustained drawdowns more than maxDD does
(it integrates the dip area). Higher = more chronic pain. Distinct
from pain_index which uses arithmetic mean. Prefix-stable.
"""




def ulcer_index(df, lookback=60, **_):
    if lookback < 2:
        raise ValueError(f"lookback must be >= 2, got {lookback}")
    close = df["close"]
    peak = close.rolling(lookback, min_periods=1).max()
    drawdown_pct = 100.0 * ((close - peak) / peak).clip(upper=0.0)
    sq = drawdown_pct.pow(2)
    return sq.rolling(lookback, min_periods=1).mean().pow(0.5).rename("ulcer_index")
