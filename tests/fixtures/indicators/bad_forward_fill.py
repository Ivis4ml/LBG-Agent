"""Intentionally violates `no_forward_fill_future` (two ways)."""

import pandas as pd


def leak_via_bfill(df: pd.DataFrame) -> pd.Series:
    # `.bfill()` carries future close values backward to fill NaNs.
    return df["close"].rolling(20).mean().bfill()


def leak_via_fillna_method(df: pd.DataFrame) -> pd.Series:
    return df["close"].rolling(20).mean().fillna(method="backfill")
