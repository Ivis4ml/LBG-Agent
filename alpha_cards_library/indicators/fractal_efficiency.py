import pandas as pd

def fractal_efficiency(df, period=20, **_):
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    close = df["close"]
    net_move = (close - close.shift(period)).abs()
    path_length = close.diff().abs().rolling(period).sum().replace(0.0, pd.NA)
    efficiency = net_move / path_length
    return efficiency.astype(float).clip(lower=0.0, upper=1.0).rename("fractal_efficiency")
