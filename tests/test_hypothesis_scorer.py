"""Tests for `lbg.gate.score_hypothesis`."""

from __future__ import annotations

import pytest

from lbg.gate import score_hypothesis
from lbg.schemas import (
    ExpectedTrainSignal,
    ExpectedValidationSignal,
    HypothesisOutcome,
    ValidationSignal,
)

# -------- bucket boundaries on delta_sharpe --------


@pytest.mark.parametrize(
    ("delta", "expected_bucket"),
    [
        (0.50, ExpectedTrainSignal.STRONG_IMPROVEMENT),
        (0.30, ExpectedTrainSignal.STRONG_IMPROVEMENT),
        (0.20, ExpectedTrainSignal.MILD_IMPROVEMENT),
        (0.05, ExpectedTrainSignal.MILD_IMPROVEMENT),
        (0.00, ExpectedTrainSignal.NEUTRAL),
        (-0.04, ExpectedTrainSignal.NEUTRAL),
        (-0.05, ExpectedTrainSignal.MILD_REGRESSION),
        (-0.20, ExpectedTrainSignal.MILD_REGRESSION),
        (-0.30, ExpectedTrainSignal.STRONG_REGRESSION),
        (-0.60, ExpectedTrainSignal.STRONG_REGRESSION),
    ],
)
def test_score_recognizes_bucket(delta, expected_bucket):
    """Editor predicting exactly the actual bucket plus val signal match -> CONFIRMED."""
    outcome = score_hypothesis(
        expected_train=expected_bucket,
        expected_validation=ExpectedValidationSignal.ACCEPT,
        delta_sharpe_train=delta,
        actual_validation=ValidationSignal.ACCEPTED,
    )
    assert outcome == HypothesisOutcome.CONFIRMED


# -------- both match -> CONFIRMED --------


def test_both_predictions_correct():
    outcome = score_hypothesis(
        expected_train=ExpectedTrainSignal.MILD_IMPROVEMENT,
        expected_validation=ExpectedValidationSignal.ACCEPT,
        delta_sharpe_train=0.10,
        actual_validation=ValidationSignal.ACCEPTED,
    )
    assert outcome == HypothesisOutcome.CONFIRMED


def test_both_predictions_correct_when_expecting_rejection():
    outcome = score_hypothesis(
        expected_train=ExpectedTrainSignal.MILD_REGRESSION,
        expected_validation=ExpectedValidationSignal.REJECT,
        delta_sharpe_train=-0.10,
        actual_validation=ValidationSignal.REJECTED_DRAWDOWN_REGRESSION,
    )
    assert outcome == HypothesisOutcome.CONFIRMED


# -------- neither matches -> DISCONFIRMED --------


def test_both_predictions_wrong_in_opposite_direction():
    outcome = score_hypothesis(
        expected_train=ExpectedTrainSignal.STRONG_IMPROVEMENT,
        expected_validation=ExpectedValidationSignal.ACCEPT,
        delta_sharpe_train=-0.40,
        actual_validation=ValidationSignal.REJECTED_DRAWDOWN_REGRESSION,
    )
    assert outcome == HypothesisOutcome.DISCONFIRMED


# -------- one matches -> PARTIALLY_CONFIRMED --------


def test_train_right_validation_wrong():
    outcome = score_hypothesis(
        expected_train=ExpectedTrainSignal.MILD_IMPROVEMENT,
        expected_validation=ExpectedValidationSignal.ACCEPT,
        delta_sharpe_train=0.10,
        actual_validation=ValidationSignal.REJECTED_DRAWDOWN_REGRESSION,
    )
    assert outcome == HypothesisOutcome.PARTIALLY_CONFIRMED


def test_validation_right_train_wrong():
    outcome = score_hypothesis(
        expected_train=ExpectedTrainSignal.STRONG_IMPROVEMENT,
        expected_validation=ExpectedValidationSignal.ACCEPT,
        delta_sharpe_train=0.10,  # actually mild
        actual_validation=ValidationSignal.ACCEPTED,
    )
    assert outcome == HypothesisOutcome.PARTIALLY_CONFIRMED


# -------- rejection signal categories all collapse to "not accepted" for val match --------


@pytest.mark.parametrize(
    "rejected_signal",
    [
        ValidationSignal.REJECTED_DRAWDOWN_REGRESSION,
        ValidationSignal.REJECTED_TURNOVER,
        ValidationSignal.REJECTED_COMPLEXITY,
        ValidationSignal.REJECTED_NO_SIGNIFICANT_IMPROVEMENT,
    ],
)
def test_any_rejection_matches_expected_reject(rejected_signal):
    outcome = score_hypothesis(
        expected_train=ExpectedTrainSignal.NEUTRAL,
        expected_validation=ExpectedValidationSignal.REJECT,
        delta_sharpe_train=0.0,
        actual_validation=rejected_signal,
    )
    assert outcome == HypothesisOutcome.CONFIRMED
