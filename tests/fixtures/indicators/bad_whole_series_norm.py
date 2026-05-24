"""Intentionally violates prefix_stability via whole-series normalization.

The indicator subtracts and divides by the mean/std computed over the entire
series, so changing any future bar shifts the mean and std, which changes
every past value. AST checks miss this because no shift/iloc/bfill appears.
"""

import pandas as pd


def whole_series_zscore(df: pd.DataFrame) -> pd.Series:
    s = df["close"]
    return (s - s.mean()) / s.std()
