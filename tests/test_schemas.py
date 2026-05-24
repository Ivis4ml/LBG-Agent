"""Tests for lbg.schemas.

Verifies that:
  - All enums match the values enumerated in PROPOSAL.html.
  - EditProposal and TrialRecord round-trip through JSON without loss.
  - Missing or invalid fields raise pydantic.ValidationError.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from lbg.schemas import (
    AgentCompute,
    Decision,
    EditProposal,
    EditSummary,
    EditType,
    ExpectedTrainSignal,
    ExpectedValidationSignal,
    HypothesisBlock,
    HypothesisOutcome,
    ProposedEdit,
    RoleOutputs,
    TrainMetrics,
    TrialRecord,
    ValidationSignal,
)

PROPOSAL_EDIT_TYPES = {
    "add_indicator",
    "parameter_change",
    "add_filter",
    "remove_filter",
    "change_sizing_mode",
    "change_exit_rule",
    "simplify",
    "revert_to_trial_N",
}

PROPOSAL_VALIDATION_SIGNALS = {
    "accepted",
    "rejected_drawdown_regression",
    "rejected_turnover",
    "rejected_complexity",
    "rejected_no_significant_improvement",
}


def test_edit_type_enum_matches_proposal():
    assert {e.value for e in EditType} == PROPOSAL_EDIT_TYPES


def test_validation_signal_enum_matches_proposal():
    assert {e.value for e in ValidationSignal} == PROPOSAL_VALIDATION_SIGNALS


def _sample_edit_proposal_dict() -> dict:
    """The change_sizing_mode example from PROPOSAL.html §6.2."""
    return {
        "trial_id": 23,
        "hypothesis": (
            "Recent rejected trials suggest fixed_fraction sizing is too "
            "aggressive during high-vol periods."
        ),
        "proposed_edit": {
            "type": "change_sizing_mode",
            "change": {
                "sizing": {
                    "mode": "volatility_target",
                    "target_vol": 0.12,
                    "max_position": 1.0,
                }
            },
        },
        "expected_train_signal": "mild_improvement",
        "expected_validation_signal": "accept",
        "fallback_if_rejected": "Try weaker target_vol (0.08).",
        "cited_factors": [],
    }


def test_edit_proposal_round_trip():
    raw = _sample_edit_proposal_dict()
    parsed = EditProposal.model_validate(raw)
    dumped = json.loads(parsed.model_dump_json())
    assert dumped == raw


def test_edit_proposal_rejects_unknown_edit_type():
    raw = _sample_edit_proposal_dict()
    raw["proposed_edit"]["type"] = "rewrite_everything"
    with pytest.raises(ValidationError):
        EditProposal.model_validate(raw)


def test_edit_proposal_rejects_missing_required_field():
    raw = _sample_edit_proposal_dict()
    del raw["hypothesis"]
    with pytest.raises(ValidationError):
        EditProposal.model_validate(raw)


def test_edit_proposal_rejects_extra_field():
    """`extra='forbid'` defends against silent context smuggling."""
    raw = _sample_edit_proposal_dict()
    raw["leak"] = "calendar_year=2019"
    with pytest.raises(ValidationError):
        EditProposal.model_validate(raw)


def _sample_trial_record_dict() -> dict:
    """The trial record example from PROPOSAL.html §10.2 (truncated to required fields)."""
    return {
        "trial_id": 23,
        "parent_commit": "abc123",
        "candidate_commit": "def456",
        "role_outputs": {
            "editor_output_path": "runs/0023/editor.yaml",
            "reflector_output_path": "runs/0023/reflector.yaml",
        },
        "edit": {
            "type": "change_sizing_mode",
            "target": "strategy.yaml",
            "summary": "fixed_fraction -> volatility_target",
        },
        "hypothesis": {
            "text": "...",
            "expected_train_signal": "mild_improvement",
            "expected_validation_signal": "accept",
        },
        "invariants": {
            "schema": "pass",
            "ast_static": "pass",
            "dynamic_lookahead": "pass",
            "cost_model": "pass",
            "no_data_snooping": "pass",
        },
        "train_metrics": {
            "sharpe": 0.91,
            "max_drawdown": -0.13,
            "turnover": 0.24,
            "num_trades": 38,
        },
        "validation_signal": "accepted",
        "hypothesis_outcome": "confirmed",
        "decision": "accept",
        "complexity_before": 4.0,
        "complexity_after": 5.0,
        "agent_compute": {
            "editor_input_tokens": 12000,
            "editor_output_tokens": 1300,
            "reflector_input_tokens": 4000,
            "reflector_output_tokens": 700,
            "model_editor": "claude-opus-4-7",
            "model_reflector": "claude-opus-4-7",
            "wall_clock_sec": 42.1,
        },
        "fallback_if_rejected": None,
        "cited_factors": [],
    }


def test_trial_record_round_trip():
    raw = _sample_trial_record_dict()
    parsed = TrialRecord.model_validate(raw)
    dumped = json.loads(parsed.model_dump_json())
    assert dumped == raw


def test_trial_record_rejects_negative_trial_id():
    raw = _sample_trial_record_dict()
    raw["trial_id"] = -1
    with pytest.raises(ValidationError):
        TrialRecord.model_validate(raw)


def test_trial_record_rejects_unknown_validation_signal():
    raw = _sample_trial_record_dict()
    raw["validation_signal"] = "rejected_for_no_reason"
    with pytest.raises(ValidationError):
        TrialRecord.model_validate(raw)


def test_enum_value_uniqueness():
    """No enum value should be duplicated; helps catch typo-induced collisions."""
    for cls in (
        EditType,
        ValidationSignal,
        ExpectedTrainSignal,
        ExpectedValidationSignal,
        HypothesisOutcome,
        Decision,
    ):
        values = [m.value for m in cls]
        assert len(values) == len(set(values)), f"duplicate values in {cls.__name__}"


def test_all_models_forbid_extra():
    """Defense in depth: every model rejects unknown fields."""
    models = (
        EditProposal,
        ProposedEdit,
        EditSummary,
        HypothesisBlock,
        RoleOutputs,
        TrainMetrics,
        AgentCompute,
        TrialRecord,
    )
    for model in models:
        assert model.model_config.get("extra") == "forbid", model.__name__
