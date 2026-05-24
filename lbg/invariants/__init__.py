"""Invariant suite: AST static checks plus runtime checks.

PROPOSAL.html §6 prescribes that every candidate indicator must pass all
invariants before any backtest runs. AST checks live in `ast_checks`; the
prefix_stability dynamic check is added in a later step.
"""

from lbg.invariants.ast_checks import ALL_AST_CHECKS, run_ast_checks
from lbg.invariants.prefix_stability import check_prefix_stability
from lbg.invariants.violations import InvariantViolation

__all__ = [
    "ALL_AST_CHECKS",
    "InvariantViolation",
    "check_prefix_stability",
    "run_ast_checks",
]
