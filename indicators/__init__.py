"""Agent-authored indicators (PROPOSAL.html §4).

Every file in this directory must be a pure function of an OHLCV DataFrame:
deterministic, no globals, no I/O, prefix-stable. The AST and prefix_stability
invariants are enforced by `lbg.invariants` before any backtest runs.
"""
