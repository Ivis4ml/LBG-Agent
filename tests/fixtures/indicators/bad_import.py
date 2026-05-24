"""Intentionally violates `whitelisted_imports_only`."""

import os  # noqa: F401  -- the import itself is the violation

import pandas as pd


def calls_outside_world(df: pd.DataFrame) -> pd.Series:
    return df["close"]
