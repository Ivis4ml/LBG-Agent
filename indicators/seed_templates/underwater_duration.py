"""Number of bars since the trailing-window peak. 0 at a new high, grows
linearly until a new high is set. Persistent values (>30) indicate
prolonged drawdowns. Prefix-stable.
"""



def underwater_duration(df, lookback=252, **_):
    if lookback < 2:
        raise ValueError(f"lookback must be >= 2, got {lookback}")
    close = df["close"]

    def _bars_since_max(s):
        return float(len(s) - 1 - s.values.argmax())

    return (
        close.rolling(lookback, min_periods=1)
        .apply(_bars_since_max, raw=False)
        .rename("underwater_duration")
    )
