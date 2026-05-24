"""Stage 2: deterministic forward validation on post-sealed bars.

Loads a frozen `strategy.yaml` + `indicators/`, runs the policy on a
DataFrame of out-of-sealed bars (or any held-out window the caller
supplies), runs backtest, applies a small go/no-go gate.

This module never calls an LLM. The strategy is locked between
Discovery runs (PROPOSAL.html §3 / §6).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest import BacktestResult, run_backtest
from lbg.dsl import load_strategy
from policy_interpreter import compute_positions


@dataclass(frozen=True)
class Stage2Gate:
    """Go/no-go thresholds for advancing from Stage 2 to Stage 3.

    Looser than the per-trial ValidationGate because Stage 2 is a *single*
    deterministic check on the frozen artifact — we just want to confirm
    the strategy is not catastrophically broken on fresh bars before
    spending paper-trading wall-clock.
    """

    min_num_trades: int = 1
    max_drawdown_floor: float = -0.30  # MDD must be >= this (less negative)
    min_sharpe: float = 0.0


@dataclass(frozen=True)
class ForwardValidationResult:
    metrics: BacktestResult
    gate: Stage2Gate
    go: bool
    rejected_reasons: list[str]

    def to_dict(self) -> dict:
        return {
            "go": self.go,
            "rejected_reasons": list(self.rejected_reasons),
            "metrics": self.metrics.to_dict(),
            "gate": {
                "min_num_trades": self.gate.min_num_trades,
                "max_drawdown_floor": self.gate.max_drawdown_floor,
                "min_sharpe": self.gate.min_sharpe,
            },
        }


def _evaluate_gate(metrics: BacktestResult, gate: Stage2Gate) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if metrics.num_trades < gate.min_num_trades:
        reasons.append(f"num_trades={metrics.num_trades} < min={gate.min_num_trades}")
    if metrics.max_drawdown < gate.max_drawdown_floor:
        reasons.append(f"max_drawdown={metrics.max_drawdown:.3f} < floor={gate.max_drawdown_floor}")
    if metrics.sharpe < gate.min_sharpe:
        reasons.append(f"sharpe={metrics.sharpe:.3f} < min={gate.min_sharpe}")
    return (not reasons), reasons


def run_forward_validation(
    strategy_path: str | Path,
    forward_df: pd.DataFrame,
    *,
    indicators_dir: str | Path = "indicators",
    gate: Stage2Gate | None = None,
    cost_per_side: float = 0.00055,
) -> ForwardValidationResult:
    """Backtest the frozen strategy on `forward_df` and apply the Stage-2 gate.

    `forward_df` is OHLCV (same shape as `lbg.data.loader.load_split` output)
    representing the post-sealed bars. The caller is responsible for
    ensuring this window strictly post-dates the sealed window.
    """
    strategy = load_strategy(strategy_path)
    positions = compute_positions(strategy, forward_df, indicators_dir=Path(indicators_dir))
    metrics = run_backtest(positions, forward_df, cost_per_side=cost_per_side)
    g = gate or Stage2Gate()
    go, reasons = _evaluate_gate(metrics, g)
    return ForwardValidationResult(metrics=metrics, gate=g, go=go, rejected_reasons=reasons)
