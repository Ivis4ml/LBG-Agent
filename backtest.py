"""Vectorized SPY daily backtest (PROPOSAL.html §4.2, §4.3).

One bar is held from open[t+1] to open[t+2], so the realized PnL on bar t+1
uses the open-to-open return `R_{t+1} = open[t+2] / open[t+1] - 1`. Costs
are `c * |p_{t+1} - p_t|` per side, with `c` defaulting to 5 bps base + 0.5
bp slippage proxy = 0.00055 (proposal §4.3).

This file is locked by the `no_python_edit` invariant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_COST_PER_SIDE: float = 0.00055
TRADING_DAYS_PER_YEAR: int = 252


@dataclass(frozen=True)
class BacktestResult:
    sharpe: float
    max_drawdown: float
    turnover: float
    num_trades: int
    n_bars_used: int
    cost_total: float
    cagr: float
    final_equity: float
    # Full per-bar series for downstream consumers (gate, block-bootstrap).
    # Kept off the JSON view to keep trial records compact.
    returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float), repr=False)
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float), repr=False)

    def to_dict(self) -> dict:
        """Summary metrics only, JSON-safe (excludes the time series)."""
        return {
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "turnover": self.turnover,
            "num_trades": self.num_trades,
            "n_bars_used": self.n_bars_used,
            "cost_total": self.cost_total,
            "cagr": self.cagr,
            "final_equity": self.final_equity,
        }


def _open_to_open_returns(open_: pd.Series) -> pd.Series:
    """R_{t+1} = open[t+2] / open[t+1] - 1, with the final bar undefined."""
    return open_.shift(-1) / open_ - 1.0


def _sharpe_annualized(returns: pd.Series) -> float:
    std = returns.std()
    if not np.isfinite(std) or std < 1e-12:
        return 0.0
    return float(returns.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min()) if len(drawdown) else 0.0


def run_backtest(
    positions: pd.Series,
    df: pd.DataFrame,
    *,
    cost_per_side: float = DEFAULT_COST_PER_SIDE,
) -> BacktestResult:
    """Run the deterministic daily backtest.

    Parameters
    ----------
    positions : pd.Series
        Output of `policy_interpreter.compute_positions`. Already lagged so
        that `positions.iloc[t]` is the position held over bar t (open[t]
        to open[t+1]).
    df : pd.DataFrame
        OHLCV with `open` column required.
    cost_per_side : float
        Per-side transaction cost as a fraction of notional traded.

    Returns
    -------
    BacktestResult
        Annualized Sharpe, max drawdown, annualized turnover, trade count,
        and the bar count actually used in the PnL series.
    """
    if len(positions) != len(df):
        raise ValueError(f"positions length {len(positions)} != df length {len(df)}")

    R = _open_to_open_returns(df["open"])

    p = positions.astype(float)
    delta = p.diff().abs().fillna(p.iloc[0])

    cost = cost_per_side * delta
    raw_pnl = p * R
    realized = (raw_pnl - cost).iloc[:-1]  # drop the last bar (R undefined)

    equity = (1.0 + realized).cumprod()

    n_bars = int(len(realized))
    n_years = n_bars / TRADING_DAYS_PER_YEAR if n_bars else 1.0
    final_equity = float(equity.iloc[-1]) if n_bars else 1.0
    cagr = float(final_equity ** (1.0 / n_years) - 1.0) if n_years > 0 and final_equity > 0 else 0.0

    return BacktestResult(
        sharpe=_sharpe_annualized(realized),
        max_drawdown=_max_drawdown(equity),
        turnover=float(delta.iloc[:-1].sum() / n_years) if n_years > 0 else 0.0,
        num_trades=int((delta > 0).iloc[:-1].sum()),
        n_bars_used=n_bars,
        cost_total=float(cost.iloc[:-1].sum()),
        cagr=cagr,
        final_equity=final_equity,
        returns=realized,
        equity=equity,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    from lbg.data.loader import load_split
    from lbg.dsl import load_strategy
    from policy_interpreter import compute_positions

    parser = argparse.ArgumentParser(prog="backtest")
    parser.add_argument("--strategy", default="strategy.yaml")
    parser.add_argument("--split", default="split_A", choices=("split_A", "split_B"))
    parser.add_argument("--cost", type=float, default=DEFAULT_COST_PER_SIDE)
    args = parser.parse_args(argv)

    strategy = load_strategy(Path(args.strategy))
    df = load_split(args.split)
    positions = compute_positions(strategy, df)
    result = run_backtest(positions, df, cost_per_side=args.cost)

    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
