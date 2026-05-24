"""Stage 2: forward validation (PROPOSAL.html §3).

After the sealed test opens at the end of a Discovery run, the frozen
strategy artifact may be promoted to Stage 2 — a deterministic backtest
on post-sealed bars (newer-than-sealed-window data). No LLM is involved.
"""

from lbg.stage2.forward import (
    ForwardValidationResult,
    Stage2Gate,
    run_forward_validation,
)

__all__ = [
    "ForwardValidationResult",
    "Stage2Gate",
    "run_forward_validation",
]
