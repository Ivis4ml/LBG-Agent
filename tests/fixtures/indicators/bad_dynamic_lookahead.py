"""Intentionally violates prefix_stability via future-aware percentile.

The indicator scales every row by `close / close.max()`, where `close.max()`
considers all rows including the future. AST checks miss this because no
shift/iloc/bfill/import appears; only `prefix_stability` catches it.
"""

import pandas as pd


def normalize_to_global_max(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].max()
