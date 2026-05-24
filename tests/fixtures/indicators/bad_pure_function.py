"""Intentionally violates `pure_function` in multiple ways."""

import pandas as pd

_state = 0  # module-level mutable state, mutated below


def impure_side_effects(df: pd.DataFrame) -> pd.Series:
    global _state
    _state += 1
    print(f"_state is now {_state}")
    df.to_csv("/tmp/leak.csv")
    return df["close"]
