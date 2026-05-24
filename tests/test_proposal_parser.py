"""Tests for `lbg.parser.parse_proposal`."""

from __future__ import annotations

import textwrap

import pytest

from lbg.parser import (
    AddFilterPayload,
    AddIndicatorPayload,
    ChangeExitRulePayload,
    ChangeSizingModePayload,
    ParameterChangePayload,
    ProposalParseError,
    RemoveFilterPayload,
    RevertToTrialPayload,
    SimplifyPayload,
    parse_proposal,
)
from lbg.schemas import EditProposal, EditType

VALID_CHANGE_SIZING_YAML = textwrap.dedent(
    """
    trial_id: 23
    hypothesis: |
      Switch from fixed_fraction to volatility_target during high-vol regimes.

    proposed_edit:
      type: change_sizing_mode
      change:
        sizing:
          mode: volatility_target
          target_vol: 0.12
          max_position: 1.0
          vol_lookback: 20

    expected_train_signal:     mild_improvement
    expected_validation_signal: accept

    fallback_if_rejected: Try weaker target_vol (0.08).
    """
)


# -------- happy path --------


def test_parse_change_sizing_yaml_returns_typed_payload():
    proposal, payload = parse_proposal(VALID_CHANGE_SIZING_YAML)
    assert isinstance(proposal, EditProposal)
    assert proposal.trial_id == 23
    assert proposal.proposed_edit.type == EditType.CHANGE_SIZING_MODE
    assert isinstance(payload, ChangeSizingModePayload)
    assert payload.sizing.mode == "volatility_target"
    assert payload.sizing.target_vol == 0.12


def test_parse_accepts_dict_input_too():
    import yaml as yaml_mod

    data = yaml_mod.safe_load(VALID_CHANGE_SIZING_YAML)
    proposal, payload = parse_proposal(data)
    assert proposal.trial_id == 23
    assert isinstance(payload, ChangeSizingModePayload)


# -------- one happy-path test per edit type --------


def _wrap_proposal(edit_type: str, change: dict) -> dict:
    return {
        "trial_id": 1,
        "hypothesis": "test hypothesis",
        "proposed_edit": {"type": edit_type, "change": change},
        "expected_train_signal": "neutral",
        "expected_validation_signal": "accept",
    }


def test_parse_parameter_change_payload():
    raw = _wrap_proposal(
        "parameter_change",
        {"path": "indicators[sma_fast].params.period", "value": 25},
    )
    proposal, payload = parse_proposal(raw)
    assert isinstance(payload, ParameterChangePayload)
    assert payload.path == "indicators[sma_fast].params.period"
    assert payload.value == 25


def test_parse_change_exit_rule_payload():
    raw = _wrap_proposal(
        "change_exit_rule",
        {"exit": {"rule": "cross_above", "fast": "x", "slow": "y"}},
    )
    proposal, payload = parse_proposal(raw)
    assert isinstance(payload, ChangeExitRulePayload)
    assert payload.exit.rule == "cross_above"


def test_parse_add_filter_payload():
    raw = _wrap_proposal(
        "add_filter",
        {"filter": {"rule": "indicator_above", "indicator": "sma_fast", "threshold": 100.0}},
    )
    proposal, payload = parse_proposal(raw)
    assert isinstance(payload, AddFilterPayload)
    assert payload.filter.rule == "indicator_above"
    assert payload.filter.threshold == 100.0


def test_parse_remove_filter_payload():
    raw = _wrap_proposal("remove_filter", {"index": 2})
    proposal, payload = parse_proposal(raw)
    assert isinstance(payload, RemoveFilterPayload)
    assert payload.index == 2


def test_parse_add_indicator_payload():
    raw = _wrap_proposal(
        "add_indicator",
        {
            "name": "rsi_14",
            "fn": "rsi",
            "source": "import pandas as pd\n\ndef rsi(df, period=14):\n    return df['close']\n",
            "params": {"period": 14},
        },
    )
    proposal, payload = parse_proposal(raw)
    assert isinstance(payload, AddIndicatorPayload)
    assert payload.name == "rsi_14"
    assert payload.fn == "rsi"
    assert "def rsi" in payload.source


def test_parse_simplify_payload():
    raw = _wrap_proposal("simplify", {"component": "filter", "target": "0"})
    proposal, payload = parse_proposal(raw)
    assert isinstance(payload, SimplifyPayload)
    assert payload.component == "filter"


def test_parse_revert_to_trial_payload():
    raw = _wrap_proposal("revert_to_trial_N", {"trial_id": 7})
    proposal, payload = parse_proposal(raw)
    assert isinstance(payload, RevertToTrialPayload)
    assert payload.trial_id == 7


# -------- failure paths --------


def test_parse_rejects_non_yaml_string():
    with pytest.raises(ProposalParseError, match="YAML parse error"):
        parse_proposal("just: a string\n  with: bad\n\tindent: here\n")


def test_parse_rejects_yaml_string_that_is_not_mapping():
    with pytest.raises(ProposalParseError, match="must be a mapping"):
        parse_proposal("- a\n- b\n")


def test_parse_rejects_unknown_edit_type():
    raw = _wrap_proposal("rewrite_everything", {})
    with pytest.raises(ProposalParseError, match="EditProposal validation failed"):
        parse_proposal(raw)


def test_parse_rejects_missing_outer_field():
    raw = _wrap_proposal("parameter_change", {"path": "x", "value": 1})
    del raw["hypothesis"]
    with pytest.raises(ProposalParseError, match="EditProposal validation"):
        parse_proposal(raw)


def test_parse_rejects_extra_field_in_proposal():
    """`extra='forbid'` on EditProposal: a smuggled field is rejected."""
    raw = _wrap_proposal("parameter_change", {"path": "x", "value": 1})
    raw["leak"] = "calendar_year=2019"
    with pytest.raises(ProposalParseError):
        parse_proposal(raw)


def test_parse_rejects_wrong_payload_shape():
    """parameter_change with the sizing-shaped change payload should fail."""
    raw = _wrap_proposal(
        "parameter_change",
        {"sizing": {"mode": "fixed_fraction", "fraction": 1.0, "max_position": 1.0}},
    )
    with pytest.raises(ProposalParseError, match="`change` payload invalid"):
        parse_proposal(raw)


def test_parse_rejects_missing_change_payload_fields():
    raw = _wrap_proposal("parameter_change", {"path": "x"})  # missing value
    with pytest.raises(ProposalParseError, match="`change` payload invalid"):
        parse_proposal(raw)


def test_parse_rejects_extra_field_in_payload():
    raw = _wrap_proposal(
        "parameter_change",
        {"path": "x", "value": 1, "secret": "leak"},
    )
    with pytest.raises(ProposalParseError, match="`change` payload invalid"):
        parse_proposal(raw)


def test_parse_rejects_negative_remove_index():
    raw = _wrap_proposal("remove_filter", {"index": -1})
    with pytest.raises(ProposalParseError, match="`change` payload invalid"):
        parse_proposal(raw)


def test_parse_handles_change_block_missing_entirely():
    """`change: null` (or omitted) collapses to an empty dict; downstream
    payload validation rejects the empty dict if it has required fields."""
    raw = _wrap_proposal("parameter_change", None)  # type: ignore[arg-type]
    raw["proposed_edit"]["change"] = None
    with pytest.raises(ProposalParseError, match="payload invalid"):
        parse_proposal(raw)
