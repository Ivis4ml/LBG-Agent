"""The multi-objective per-trial validation gate (PROPOSAL.html §9 line 1704)."""

from __future__ import annotations

from dataclasses import dataclass

from backtest import BacktestResult
from lbg.gate.utility import utility_lcb_sharpe
from lbg.schemas import ValidationSignal


@dataclass(frozen=True)
class TrialOutcome:
    """Inputs the gate needs about either the incumbent or the candidate."""

    train: BacktestResult
    validation: BacktestResult
    complexity: float


@dataclass(frozen=True)
class GateConfig:
    """Per-trial gate thresholds. Locked per PROPOSAL.html §20 once a Discovery
    run starts; do not edit during a run.

    Use the default GateConfig() for any H1 reporting / paper claim. The
    classmethod `permissive()` returns a looser preset for experiment-stage
    campaigns that need a non-zero accept rate to exercise downstream
    machinery (alpha_cards/, sealed verdicts on a non-baseline incumbent).
    Results from the permissive preset MUST NOT be cited as H1 evidence.
    """

    min_trades: int = 20
    drawdown_regression_factor: float = 1.15
    max_turnover: float = 10.0
    max_complexity_delta: float = 3.0
    max_train_val_sharpe_gap: float = 0.5
    utility_lcb_alpha: float = 0.20
    eps: float = 0.005

    @classmethod
    def permissive(cls) -> "GateConfig":
        """Relaxed thresholds for experimentation only. Concretely:

          * min_trades 20 → 10   (was the binding constraint on wider SMAs)
          * utility_lcb_alpha 0.20 → 0.40  (looser one-sided CI on Sharpe gain)
          * drawdown_regression_factor 1.15 → 1.30  (tolerate larger DD)

        Used by `scripts/campaign.py --gate permissive` to surface real
        alpha_cards from the Discovery loop. Not for any statistical claim.
        """
        return cls(
            min_trades=10,
            utility_lcb_alpha=0.40,
            drawdown_regression_factor=1.30,
        )


# Reason strings produced by `decide`. The first three accept reasons map to
# `ValidationSignal.ACCEPTED`; the rest map to the four reject categories.
REASON_UTILITY_IMPROVEMENT = "accept_utility_improvement"
REASON_PARETO_IMPROVEMENT = "accept_pareto_improvement"
REASON_TOO_FEW_TRADES = "reject_too_few_trades"
REASON_DRAWDOWN_REGRESSION = "reject_drawdown_regression"
REASON_TURNOVER_TOO_HIGH = "reject_turnover_too_high"
REASON_COMPLEXITY_INCREASE = "reject_complexity_increase"
REASON_OVERFIT_GAP = "reject_overfit_gap"
REASON_NO_MEANINGFUL_IMPROVEMENT = "reject_no_meaningful_improvement"


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reason: str
    # Diagnostic numbers, for the developer-facing trial record. The LLM
    # only ever sees the result of `redact(...)`.
    candidate_utility_lcb: float
    incumbent_utility_lcb: float
    complexity_delta: float


def pareto_dominates(
    candidate: TrialOutcome,
    incumbent: TrialOutcome,
    eps: float,
) -> bool:
    """Candidate Pareto-dominates incumbent on (sharpe, max_dd, turnover, complexity)?

    PROPOSAL.html §9: better on drawdown OR turnover OR complexity, without
    being worse on Sharpe. Direction conventions:
      - sharpe higher is better
      - max_drawdown higher (less negative) is better
      - turnover lower is better
      - complexity lower is better
    """
    cv, iv = candidate.validation, incumbent.validation
    not_worse_sharpe = cv.sharpe >= iv.sharpe - eps
    better_dd = cv.max_drawdown > iv.max_drawdown + eps
    better_turnover = cv.turnover < iv.turnover - eps
    better_complexity = candidate.complexity < incumbent.complexity - eps
    return not_worse_sharpe and (better_dd or better_turnover or better_complexity)


def decide(
    incumbent: TrialOutcome,
    candidate: TrialOutcome,
    config: GateConfig | None = None,
) -> GateDecision:
    """Run the full gate. Order of checks matches PROPOSAL.html §9 pseudocode."""
    cfg = config or GateConfig()

    complexity_delta = candidate.complexity - incumbent.complexity

    u_cand = utility_lcb_sharpe(candidate.validation.returns, alpha=cfg.utility_lcb_alpha)
    u_incumbent = utility_lcb_sharpe(incumbent.validation.returns, alpha=cfg.utility_lcb_alpha)

    def _make(accepted: bool, reason: str) -> GateDecision:
        return GateDecision(
            accepted=accepted,
            reason=reason,
            candidate_utility_lcb=u_cand,
            incumbent_utility_lcb=u_incumbent,
            complexity_delta=complexity_delta,
        )

    if candidate.train.num_trades < cfg.min_trades:
        return _make(False, REASON_TOO_FEW_TRADES)

    if (
        candidate.validation.max_drawdown
        < incumbent.validation.max_drawdown * cfg.drawdown_regression_factor
    ):
        return _make(False, REASON_DRAWDOWN_REGRESSION)

    if candidate.validation.turnover > cfg.max_turnover:
        return _make(False, REASON_TURNOVER_TOO_HIGH)

    if complexity_delta > cfg.max_complexity_delta:
        return _make(False, REASON_COMPLEXITY_INCREASE)

    if u_cand > u_incumbent + cfg.eps:
        return _make(True, REASON_UTILITY_IMPROVEMENT)

    if pareto_dominates(candidate, incumbent, cfg.eps):
        return _make(True, REASON_PARETO_IMPROVEMENT)

    if candidate.train.sharpe - candidate.validation.sharpe > cfg.max_train_val_sharpe_gap:
        return _make(False, REASON_OVERFIT_GAP)

    return _make(False, REASON_NO_MEANINGFUL_IMPROVEMENT)


_REDACT_MAP: dict[str, ValidationSignal] = {
    REASON_UTILITY_IMPROVEMENT: ValidationSignal.ACCEPTED,
    REASON_PARETO_IMPROVEMENT: ValidationSignal.ACCEPTED,
    REASON_DRAWDOWN_REGRESSION: ValidationSignal.REJECTED_DRAWDOWN_REGRESSION,
    REASON_TURNOVER_TOO_HIGH: ValidationSignal.REJECTED_TURNOVER,
    REASON_COMPLEXITY_INCREASE: ValidationSignal.REJECTED_COMPLEXITY,
    # too_few_trades, overfit_gap, and no_meaningful_improvement all map to
    # the "no significant improvement" bucket because the LLM should only
    # know that the candidate didn't actually advance the strategy.
    REASON_TOO_FEW_TRADES: ValidationSignal.REJECTED_NO_SIGNIFICANT_IMPROVEMENT,
    REASON_OVERFIT_GAP: ValidationSignal.REJECTED_NO_SIGNIFICANT_IMPROVEMENT,
    REASON_NO_MEANINGFUL_IMPROVEMENT: ValidationSignal.REJECTED_NO_SIGNIFICANT_IMPROVEMENT,
}


def redact(decision: GateDecision) -> ValidationSignal:
    """Map an internal gate decision to the LLM-facing categorical signal."""
    if decision.reason not in _REDACT_MAP:
        raise ValueError(f"unknown gate reason: {decision.reason!r}")
    return _REDACT_MAP[decision.reason]
