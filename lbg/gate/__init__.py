"""ValidationGate and HypothesisScorer (PROPOSAL.html §9, §6.5).

The gate is a multi-objective per-trial decision rule. It is deliberately
*not* the same as the final H1 verdict (a strict 95% CI block-bootstrap
test reserved for the sealed window). The per-trial gate uses a one-sided
utility LCB at α=0.20 plus Pareto-improvement acceptance, so the Discovery
loop can explore informatively without being starved by validation noise.

After the gate emits an internal `GateDecision`, `ValidationGate.redact`
maps it to one of five categorical `ValidationSignal` values; the LLM only
ever sees the redacted form. `HypothesisScorer.compare` mechanically
matches the Editor's a-priori predictions against the actual outcome, so
the Reflector cannot rationalize its own hypothesis.
"""

from lbg.gate.complexity import complexity_score
from lbg.gate.decision import (
    GateConfig,
    GateDecision,
    TrialOutcome,
    decide,
    pareto_dominates,
    redact,
)
from lbg.gate.scorer import score_hypothesis
from lbg.gate.utility import utility_lcb_sharpe

__all__ = [
    "GateConfig",
    "GateDecision",
    "TrialOutcome",
    "complexity_score",
    "decide",
    "pareto_dominates",
    "redact",
    "score_hypothesis",
    "utility_lcb_sharpe",
]
