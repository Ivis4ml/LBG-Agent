"""Pre-registered baselines for the H1 verdict (PROPOSAL.html §10.5)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def baseline_buy_and_hold(df: pd.DataFrame) -> pd.Series:
    """Position = 1.0 every bar.

    One entry trade at the first bar; cost amortized over the whole window
    (which is what `run_backtest` accounts via the initial position delta).
    """
    return pd.Series(np.ones(len(df), dtype=float), index=df.index, name="position")


def baseline_sixty_forty(df: pd.DataFrame) -> pd.Series:
    """Position = 0.6, with the remaining 40% in zero-return cash.

    Since SPY is the only risky asset in scope, the 40% allocation produces
    no return. This is the simplest deterministic 60/40 surrogate; documented
    in `analysis_plan.yaml` so it cannot be changed mid-experiment.
    """
    return pd.Series(np.full(len(df), 0.6, dtype=float), index=df.index, name="position")


BASELINE_REGISTRY: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "buy_and_hold": baseline_buy_and_hold,
    "sixty_forty": baseline_sixty_forty,
}
