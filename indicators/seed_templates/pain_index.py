"""Average drawdown over a trailing window (the "pain index" by
Becker). Differs from max drawdown: instead of taking the worst dip,
averages every bar's drawdown vs its lookback peak. Smooth, more
predictive of chronic underperformance regimes than spike-driven
maxDD. Prefix-stable.
"""



def pain_index(df, lookback=60, **_):
    if lookback < 2:
        raise ValueError(f"lookback must be >= 2, got {lookback}")
    close = df["close"]
    peak = close.rolling(lookback, min_periods=1).max()
    drawdown = ((peak - close) / peak).clip(lower=0.0)
    return drawdown.rolling(lookback, min_periods=1).mean().rename("pain_index")
