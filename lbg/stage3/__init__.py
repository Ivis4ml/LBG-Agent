"""Stage 3: paper trading (PROPOSAL.html §3).

Runs the frozen strategy on a streamed (or recent-batch) data feed, records
positions + simulated PnL with a slippage model. No LLM involved.

The engine is bar-by-bar: each call to `PaperTradingEngine.step(date, ohlcv)`
appends one bar to the in-memory history buffer, recomputes the position
from the policy interpreter on the full buffer, and logs realized PnL when
the next bar arrives.
"""

from lbg.stage3.engine import (
    PaperTradingEngine,
    PaperTradingLogEntry,
)

__all__ = ["PaperTradingEngine", "PaperTradingLogEntry"]
