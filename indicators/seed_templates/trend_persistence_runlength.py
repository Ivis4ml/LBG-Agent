"""Run-length of consecutive bars where close > SMA(period). Resets to
0 on any bar that closes below SMA. Captures "how long has the
short-term trend held" — useful for trailing-stop logic. Prefix-stable.
"""

import numpy as np
import pandas as pd


def trend_persistence_runlength(df, period=20, **_):
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    sma = df["close"].rolling(period).mean()
    above = (df["close"] > sma).fillna(False).to_numpy()
    n = len(above)
    out = np.zeros(n, dtype=float)
    run = 0
    for i in range(n):
        if above[i]:
            run += 1
        else:
            run = 0
        out[i] = float(run)
    return pd.Series(out, index=df.index).rename("trend_persistence_runlength")
