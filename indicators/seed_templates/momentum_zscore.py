"""Z-score of N-bar momentum against its own long-history distribution.
Standardised, comparable across vol regimes. Prefix-stable: the z-score
window is itself trailing.
"""



def momentum_zscore(df, momentum_period=20, zscore_lookback=252, **_):
    if momentum_period < 1 or zscore_lookback < 5:
        raise ValueError(
            f"momentum_period >= 1 and zscore_lookback >= 5 required; "
            f"got {momentum_period}, {zscore_lookback}"
        )
    close = df["close"]
    momentum = (close - close.shift(momentum_period)) / close.shift(momentum_period)
    mean = momentum.rolling(zscore_lookback).mean()
    std = momentum.rolling(zscore_lookback).std()
    return ((momentum - mean) / std).rename("momentum_zscore")
