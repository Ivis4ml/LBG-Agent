"""Intentionally violates `no_negative_indexing`."""

import pandas as pd


def leak_via_iloc(df: pd.DataFrame) -> pd.Series:
    out = []
    n = len(df)
    for i in range(n):
        # Look one bar ahead — forbidden.
        out.append(df["close"].iloc[i + 1] if i + 1 < n else float("nan"))
    return pd.Series(out, index=df.index)
