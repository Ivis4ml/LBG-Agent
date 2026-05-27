"""Current drawdown from the trailing-window peak, expressed as a
positive percentage. 0.10 = 10% below the recent high. Use with the
exit_filter rearm_threshold mechanism so the strategy re-enters after
the drawdown recovers. Prefix-stable: uses trailing rolling.max().
"""



def rolling_drawdown_pct(df, lookback=60, **_):
    if lookback < 2:
        raise ValueError(f"lookback must be >= 2, got {lookback}")
    close = df["close"]
    peak = close.rolling(lookback, min_periods=1).max()
    drawdown = (peak - close) / peak
    return drawdown.clip(lower=0.0).rename("rolling_drawdown_pct")
