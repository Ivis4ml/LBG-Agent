"""Where the close sits inside the bar's high-low range, in [0, 1].
0 = closed at the low (weak); 1 = closed at the high (strong).
Rolling mean smooths a multi-bar tendency signal. Prefix-stable.
"""

import pandas as pd


def close_position_in_range(df, period=10, **_):
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    rng = (df["high"] - df["low"]).replace(0.0, pd.NA)
    pos = (df["close"] - df["low"]) / rng
    pos = pos.astype(float).clip(lower=0.0, upper=1.0)
    return pos.rolling(period).mean().rename("close_position_in_range")
