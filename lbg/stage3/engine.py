"""Bar-by-bar paper trading engine.

Holds a growing OHLCV buffer plus the frozen strategy + indicators. Each
new bar is appended; the policy interpreter is re-run on the full buffer
to derive the desired position; realized PnL on the *previous* bar is
computed on arrival of the *next* bar (open-to-open return convention,
same as `backtest.run_backtest`).

This module never calls an LLM. The strategy artifact is locked.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from lbg.dsl import load_strategy
from lbg.dsl.schema import Strategy
from policy_interpreter import compute_positions

DEFAULT_COST_PER_SIDE: float = 0.00055


@dataclass(frozen=True)
class PaperTradingLogEntry:
    bar_index: int  # 0-based position in the engine's history buffer
    iso_date: str  # ISO-8601 date string (developer-visible, not LLM)
    open: float
    close: float
    target_position: float
    held_position: float  # position carried over from previous bar
    realized_pnl: float  # PnL accrued from open[t] to open[t+1] using held_position
    cost_paid: float
    equity: float  # cumulative paper equity, baseline 1.0


class PaperTradingEngine:
    """Stateful bar-by-bar runner of the frozen strategy.

    Persistence: the engine keeps its history buffer and per-bar log in
    memory; callers may flush both to disk with `dump_log()` /
    `save_buffer()`. State is reconstructable by replaying the buffer.
    """

    def __init__(
        self,
        strategy_path: str | Path,
        *,
        indicators_dir: str | Path = "indicators",
        cost_per_side: float = DEFAULT_COST_PER_SIDE,
        initial_equity: float = 1.0,
    ) -> None:
        self.strategy_path = Path(strategy_path)
        self.indicators_dir = Path(indicators_dir)
        self.cost_per_side = cost_per_side
        self.strategy: Strategy = load_strategy(self.strategy_path)

        self._buffer: list[dict] = []  # each dict is one OHLCV bar + date
        self._log: list[PaperTradingLogEntry] = []
        self._held_position: float = 0.0
        self._equity: float = float(initial_equity)

    # ---- streaming API ----

    def step(
        self,
        iso_date: str,
        ohlcv: dict | pd.Series,
    ) -> PaperTradingLogEntry | None:
        """Append one bar; return a log entry for the *previous* bar's PnL.

        Returns None on the very first call (no prior bar to settle).
        """
        bar = dict(ohlcv)
        for key in ("open", "high", "low", "close", "volume"):
            if key not in bar:
                raise KeyError(f"bar missing required field {key!r}")
        bar = {k: bar[k] for k in ("open", "high", "low", "close", "volume")}
        self._buffer.append({"date": iso_date, **bar})

        if len(self._buffer) < 2:
            return None  # no settled bar yet

        # Recompute positions over the entire buffer (the policy is
        # prefix-stable so the historical positions don't shift).
        positions = self._positions_from_buffer()

        prev_idx = len(self._buffer) - 2
        prev_open = float(self._buffer[prev_idx]["open"])
        curr_open = float(self._buffer[prev_idx + 1]["open"])
        bar_return = curr_open / prev_open - 1.0 if prev_open > 0 else 0.0

        target_at_prev = float(positions.iloc[prev_idx])
        delta = abs(target_at_prev - self._held_position)
        cost = self.cost_per_side * delta
        realized = self._held_position * bar_return - cost
        self._equity *= 1.0 + realized
        # New held position takes effect from open[t+1] forward.
        self._held_position = target_at_prev

        entry = PaperTradingLogEntry(
            bar_index=prev_idx,
            iso_date=str(self._buffer[prev_idx]["date"]),
            open=prev_open,
            close=float(self._buffer[prev_idx]["close"]),
            target_position=target_at_prev,
            held_position=self._held_position,
            realized_pnl=realized,
            cost_paid=cost,
            equity=self._equity,
        )
        self._log.append(entry)
        return entry

    # ---- accessors ----

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def held_position(self) -> float:
        return self._held_position

    @property
    def n_bars(self) -> int:
        return len(self._buffer)

    @property
    def log(self) -> list[PaperTradingLogEntry]:
        return list(self._log)

    def buffer_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._buffer)

    # ---- persistence ----

    def dump_log(self, path: str | Path) -> Path:
        """Write the per-bar log to a JSONL file. Returns the path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for e in self._log:
                f.write(json.dumps(asdict(e)) + "\n")
        return p

    def save_buffer(self, path: str | Path) -> Path:
        """Serialize the OHLCV buffer to Parquet so a future session can resume."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df = self.buffer_dataframe()
        df.to_parquet(p)
        return p

    # ---- internals ----

    def _positions_from_buffer(self) -> pd.Series:
        df = pd.DataFrame(self._buffer).drop(columns=["date"])
        # `compute_positions` already lags by one bar so positions.iloc[t]
        # is the desired position to hold over bar t (open[t] -> open[t+1]).
        return compute_positions(self.strategy, df, indicators_dir=self.indicators_dir)
