"""sma: trailing simple moving average of close.

Pure-function indicator for the LBG-Trader catalog. Authored under the
constraints of PROPOSAL.html §4.5 and §6.

Constraints (enforced by invariants):
  - Pure function of the input DataFrame. No globals, no I/O, no side effects.
  - Only numpy, pandas, and math may be imported.
  - Prefix-stable: for every t, the value at t depends only on rows [0, t].
    `.rolling(N).mean()` is trailing by default; no forward-fill is applied.
  - Returns a pandas.Series aligned to the input index.

Parameters
----------
df : pandas.DataFrame
    OHLCV bars with columns: open, high, low, close, volume.
period : int
    Window length in bars. Must be >= 1.

Returns
-------
pandas.Series
    Trailing simple moving average of `close`, with the first `period - 1`
    rows NaN (no back-fill).

Notes
-----
Classical trend-following primitive. Crossings of price and SMA are a common
entry signal; combinations of two SMAs (short vs long) form the simplest
momentum filter.
"""

import pandas as pd


def sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    return df["close"].rolling(period).mean()
