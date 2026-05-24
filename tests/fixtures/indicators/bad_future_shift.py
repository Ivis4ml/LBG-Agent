"""Intentionally violates `no_future_shift`."""

import pandas as pd


def leak_via_shift(df: pd.DataFrame) -> pd.Series:
    # Pulls tomorrow's close into today's row.
    return df["close"].shift(-1)
