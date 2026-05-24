"""Violates two invariants at once (no_future_shift + whitelisted_imports_only)."""

import os  # noqa: F401

import pandas as pd


def double_leak(df: pd.DataFrame) -> pd.Series:
    return df["close"].shift(-3)
