"""Two top-level public functions; sandbox must require `function_name=`."""

import pandas as pd


def fn_a(df: pd.DataFrame) -> pd.Series:
    return df["close"] * 2.0


def fn_b(df: pd.DataFrame) -> pd.Series:
    return df["close"] * 3.0
