"""HypothesisScorer: mechanical comparison of Editor predictions to actual outcome.

PROPOSAL.html §6.5 (line 1264) — the Reflector never makes this call. By
keeping the comparison mechanical, the project's defense against LLM
sycophancy stays load-bearing: the Editor cannot rewrite a disconfirmed
hypothesis into a confirmed one post-hoc.

Two predictions are compared:
  - `expected_validation_signal`  (accept / reject)  vs  the actual signal
  - `expected_train_signal`       (5-way category)   vs  the Sharpe delta
                                                          (candidate - incumbent)

Both match  → CONFIRMED
Both miss   → DISCONFIRMED
One matches → PARTIALLY_CONFIRMED
"""

from __future__ import annotations

from lbg.schemas import (
    ExpectedTrainSignal,
    ExpectedValidationSignal,
    HypothesisOutcome,
    ValidationSignal,
)

# Sharpe-delta thresholds for bucketing the actual train signal. The
# magnitudes are deliberately conservative; revisit once we have empirical
# distributions from real trials.
_STRONG_THRESHOLD: float = 0.30
_MILD_THRESHOLD: float = 0.05


def _bucketize_sharpe_delta(delta_sharpe: float) -> ExpectedTrainSignal:
    if delta_sharpe >= _STRONG_THRESHOLD:
        return ExpectedTrainSignal.STRONG_IMPROVEMENT
    if delta_sharpe >= _MILD_THRESHOLD:
        return ExpectedTrainSignal.MILD_IMPROVEMENT
    if delta_sharpe > -_MILD_THRESHOLD:
        return ExpectedTrainSignal.NEUTRAL
    if delta_sharpe > -_STRONG_THRESHOLD:
        return ExpectedTrainSignal.MILD_REGRESSION
    return ExpectedTrainSignal.STRONG_REGRESSION


def _validation_signal_matches(
    expected: ExpectedValidationSignal,
    actual: ValidationSignal,
) -> bool:
    actual_is_accept = actual == ValidationSignal.ACCEPTED
    if expected == ExpectedValidationSignal.ACCEPT:
        return actual_is_accept
    return not actual_is_accept


def score_hypothesis(
    *,
    expected_train: ExpectedTrainSignal,
    expected_validation: ExpectedValidationSignal,
    delta_sharpe_train: float,
    actual_validation: ValidationSignal,
) -> HypothesisOutcome:
    """Mechanical comparison of Editor predictions to actual outcomes."""
    actual_train_bucket = _bucketize_sharpe_delta(delta_sharpe_train)
    train_match = actual_train_bucket == expected_train
    val_match = _validation_signal_matches(expected_validation, actual_validation)

    if train_match and val_match:
        return HypothesisOutcome.CONFIRMED
    if not train_match and not val_match:
        return HypothesisOutcome.DISCONFIRMED
    return HypothesisOutcome.PARTIALLY_CONFIRMED
