"""Z-score of normalised intra-bar dispersion ((high - low) / close)
against its own ~1-year distribution. High values mark confused / chop
regimes (wide bars, no clear direction). Use as exit filter to step
aside during dispersion spikes. Prefix-stable.
"""



def dispersion_regime(df, period=20, zscore_lookback=252, **_):
    if period < 2 or zscore_lookback < 5:
        raise ValueError(
            f"period >= 2 and zscore_lookback >= 5 required; got {period}, {zscore_lookback}"
        )
    bar_dispersion = (df["high"] - df["low"]) / df["close"]
    smoothed = bar_dispersion.rolling(period).mean()
    mean = smoothed.rolling(zscore_lookback).mean()
    std = smoothed.rolling(zscore_lookback).std()
    return ((smoothed - mean) / std).rename("dispersion_regime")
