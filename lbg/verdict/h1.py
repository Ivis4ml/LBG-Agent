"""Compute the H1 verdict on the sealed window (PROPOSAL.html §10.5).

Inputs:
  - strategy realized returns (BacktestResult.returns) on sealed
  - sealed-window DataFrame + the list of pre-registered baseline names
The function backtests each baseline on the same window, picks the
strongest by point Sharpe, then runs the block bootstrap against it.

Outputs `H1Verdict` with both the strong and weak criteria evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest import run_backtest
from lbg.verdict.analysis_plan import DEFAULT_ANALYSIS_PLAN, AnalysisPlan
from lbg.verdict.baselines import BASELINE_REGISTRY
from lbg.verdict.bootstrap import (
    TRADING_DAYS_PER_YEAR,
    moving_block_bootstrap_sharpe_diff,
)


@dataclass(frozen=True)
class H1Verdict:
    strategy_sharpe: float
    best_baseline_name: str
    best_baseline_sharpe: float
    delta_sharpe: float
    ci_lower: float
    ci_upper: float
    alpha: float
    n_bootstrap: int
    block_len: int
    strong: bool  # ci_lower > 0
    weak: bool  # delta_sharpe > 0

    def to_dict(self) -> dict:
        return {
            "strategy_sharpe": self.strategy_sharpe,
            "best_baseline_name": self.best_baseline_name,
            "best_baseline_sharpe": self.best_baseline_sharpe,
            "delta_sharpe": self.delta_sharpe,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "alpha": self.alpha,
            "n_bootstrap": self.n_bootstrap,
            "block_len": self.block_len,
            "h1_strong": self.strong,
            "h1_weak": self.weak,
        }


def compute_h1_verdict(
    strategy_returns: pd.Series,
    sealed_df: pd.DataFrame,
    *,
    plan: AnalysisPlan = DEFAULT_ANALYSIS_PLAN,
    rng_seed: int = 42,
) -> H1Verdict:
    """Backtest each baseline on `sealed_df`, pick the best, run bootstrap CI."""
    # Strategy Sharpe from its (already computed) realized returns.
    strat_arr = strategy_returns.to_numpy()

    # Sharpe + realized returns for each baseline.
    candidates: list[tuple[str, float, pd.Series]] = []
    for name in plan.baselines:
        if name not in BASELINE_REGISTRY:
            raise ValueError(f"unknown baseline {name!r}; plan must reference a registered one")
        positions = BASELINE_REGISTRY[name](sealed_df)
        bt = run_backtest(positions, sealed_df)
        candidates.append((name, bt.sharpe, bt.returns))
    if not candidates:
        raise ValueError("analysis plan has no baselines")

    # Pick the baseline with the highest point Sharpe.
    best_name, best_sharpe, best_returns = max(candidates, key=lambda c: c[1])

    # Align lengths: both series come from the same sealed_df and `run_backtest`
    # drops the last bar, so they're already length-aligned.
    base_arr = best_returns.to_numpy()
    if len(strat_arr) != len(base_arr):
        # Defensive trim in case future refactors change the alignment.
        n = min(len(strat_arr), len(base_arr))
        strat_arr = strat_arr[:n]
        base_arr = base_arr[:n]

    boot = moving_block_bootstrap_sharpe_diff(
        strat_arr,
        base_arr,
        block_len=plan.h1.block_len,
        n_bootstrap=plan.h1.n_bootstrap,
        alpha=plan.h1.alpha,
        rng_seed=rng_seed,
    )

    # ddof=1 matches `Series.std()` in `backtest._sharpe_annualized` so the
    # H1 strategy_sharpe lines up bit-for-bit with the backtest's sharpe.
    strat_std = strat_arr.std(ddof=1) if strat_arr.size > 1 else 0.0
    strat_sharpe = (
        float(strat_arr.mean() / strat_std * (TRADING_DAYS_PER_YEAR**0.5))
        if strat_std > 1e-12
        else 0.0
    )

    return H1Verdict(
        strategy_sharpe=strat_sharpe,
        best_baseline_name=best_name,
        best_baseline_sharpe=float(best_sharpe),
        delta_sharpe=float(boot["point_estimate"]),
        ci_lower=float(boot["ci_lower"]),
        ci_upper=float(boot["ci_upper"]),
        alpha=float(boot["alpha"]),
        n_bootstrap=int(boot["n_bootstrap"]),
        block_len=int(boot["block_len"]),
        strong=bool(boot["ci_lower"] > 0),
        weak=bool(boot["point_estimate"] > 0),
    )
