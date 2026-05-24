"""Pydantic schemas for the Discovery loop.

Schemas trace to PROPOSAL.html as follows:
  - EditType                 §6.2 (Editor edit-type table)
  - ValidationSignal         §7   (redacted categorical signal)
  - EditProposal             §6.2 (Editor output schema)
  - TrialRecord              §10.2 (trial record schema)

PROPOSAL.html enumerates only single example values for
ExpectedTrainSignal, ExpectedValidationSignal, and HypothesisOutcome.
The extra values below are minimal extensions that preserve the
proposal's symmetry; revisit when the proposal pins these.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class EditType(StrEnum):
    """The eight discrete edit types the Editor may emit (PROPOSAL.html §6.2)."""

    ADD_INDICATOR = "add_indicator"
    PARAMETER_CHANGE = "parameter_change"
    ADD_FILTER = "add_filter"
    REMOVE_FILTER = "remove_filter"
    CHANGE_SIZING_MODE = "change_sizing_mode"
    CHANGE_EXIT_RULE = "change_exit_rule"
    SIMPLIFY = "simplify"
    REVERT_TO_TRIAL = "revert_to_trial_N"


class ValidationSignal(StrEnum):
    """Redacted categorical signal returned to the LLM (PROPOSAL.html §7)."""

    ACCEPTED = "accepted"
    REJECTED_DRAWDOWN_REGRESSION = "rejected_drawdown_regression"
    REJECTED_TURNOVER = "rejected_turnover"
    REJECTED_COMPLEXITY = "rejected_complexity"
    REJECTED_NO_SIGNIFICANT_IMPROVEMENT = "rejected_no_significant_improvement"


class ExpectedTrainSignal(StrEnum):
    STRONG_IMPROVEMENT = "strong_improvement"
    MILD_IMPROVEMENT = "mild_improvement"
    NEUTRAL = "neutral"
    MILD_REGRESSION = "mild_regression"
    STRONG_REGRESSION = "strong_regression"


class ExpectedValidationSignal(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"


class HypothesisOutcome(StrEnum):
    CONFIRMED = "confirmed"
    PARTIALLY_CONFIRMED = "partially_confirmed"
    DISCONFIRMED = "disconfirmed"


class Decision(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"


class ProposedEdit(BaseModel):
    """A single typed edit. The `change` payload is intentionally untyped here;
    a per-edit-type payload schema lives in `lbg.parser` (Step 9)."""

    model_config = ConfigDict(extra="forbid")

    type: EditType
    change: dict | None = None


class EditProposal(BaseModel):
    """Editor output schema (PROPOSAL.html §6.2)."""

    model_config = ConfigDict(extra="forbid")

    trial_id: int = Field(ge=0)
    hypothesis: str
    proposed_edit: ProposedEdit
    expected_train_signal: ExpectedTrainSignal
    expected_validation_signal: ExpectedValidationSignal
    fallback_if_rejected: str | None = None


class EditSummary(BaseModel):
    """Summary of an applied edit, stored on the trial record."""

    model_config = ConfigDict(extra="forbid")

    type: EditType
    target: str
    summary: str


class HypothesisBlock(BaseModel):
    """Hypothesis fields as written into the trial record."""

    model_config = ConfigDict(extra="forbid")

    text: str
    expected_train_signal: ExpectedTrainSignal
    expected_validation_signal: ExpectedValidationSignal


class RoleOutputs(BaseModel):
    """Paths to the raw LLM outputs for this trial."""

    model_config = ConfigDict(extra="forbid")

    editor_output_path: str
    reflector_output_path: str | None = None


class TrainMetrics(BaseModel):
    """Metrics reported on split_A (PROPOSAL.html §10.2)."""

    model_config = ConfigDict(extra="forbid")

    sharpe: float
    max_drawdown: float
    turnover: float
    num_trades: int = Field(ge=0)


class AgentCompute(BaseModel):
    """Per-trial token and wall-clock accounting."""

    model_config = ConfigDict(extra="forbid")

    editor_input_tokens: int = Field(ge=0)
    editor_output_tokens: int = Field(ge=0)
    reflector_input_tokens: int = Field(ge=0)
    reflector_output_tokens: int = Field(ge=0)
    model_editor: str
    model_reflector: str
    wall_clock_sec: float = Field(ge=0)


class TrialRecord(BaseModel):
    """One line in memory/trials.jsonl (PROPOSAL.html §10.2)."""

    model_config = ConfigDict(extra="forbid")

    trial_id: int = Field(ge=0)
    parent_commit: str
    candidate_commit: str
    role_outputs: RoleOutputs
    edit: EditSummary
    hypothesis: HypothesisBlock
    invariants: dict[str, str]
    train_metrics: TrainMetrics
    validation_signal: ValidationSignal
    hypothesis_outcome: HypothesisOutcome
    decision: Decision
    complexity_before: float = Field(ge=0)
    complexity_after: float = Field(ge=0)
    agent_compute: AgentCompute
