"""Pydantic schemas for the non-trial memory streams.

`TrialRecord` itself lives in `lbg.schemas` (defined in step 1 from the
proposal's example). The records here cover the three sibling streams
written by the Orchestrator.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from lbg.schemas import HypothesisOutcome


class ReflectorOutputPayload(BaseModel):
    """What the Reflector LLM writes -- the subset of `ReflectionRecord` whose
    fields the LLM owns. The Orchestrator injects `trial_id` and the
    mechanically computed `hypothesis_outcome` to produce a full
    `ReflectionRecord`."""

    model_config = ConfigDict(extra="forbid")

    explanation: str = Field(min_length=1)
    accepted_rules_updates: list[str] = Field(default_factory=list)
    failed_directions_updates: list[str] = Field(default_factory=list)
    open_questions_updates: list[str] = Field(default_factory=list)
    do_not_repeat_updates: list[str] = Field(default_factory=list)


class ReflectionRecord(BaseModel):
    """One line of `memory/reflections.jsonl`.

    The Reflector writes free-form prose explaining *why* the trial got the
    signal it did. The HypothesisOutcome here is computed mechanically by
    `HypothesisScorer` and copied in for cross-reference; the Reflector
    never sets it itself (PROPOSAL.html §6.5).
    """

    model_config = ConfigDict(extra="forbid")

    trial_id: int = Field(ge=0)
    explanation: str
    hypothesis_outcome: HypothesisOutcome
    accepted_rules_updates: list[str] = Field(default_factory=list)
    failed_directions_updates: list[str] = Field(default_factory=list)
    open_questions_updates: list[str] = Field(default_factory=list)
    do_not_repeat_updates: list[str] = Field(default_factory=list)


class InvariantFailureRecord(BaseModel):
    """One line of `memory/invariant_failures.jsonl`.

    Recorded *before* the backtest runs when an invariant rejects the
    candidate. The Orchestrator never proceeds to backtest if any invariant
    fails; the failure record stands in for the trial outcome.
    """

    model_config = ConfigDict(extra="forbid")

    trial_id: int = Field(ge=0)
    parent_commit: str
    invariant_name: str
    message: str
    file: str
    line: int = Field(ge=0)


class AgentComputeRecord(BaseModel):
    """One line of `memory/agent_compute.jsonl`.

    Per-role token + wall-clock accounting. Aggregated across trials this
    gives the cost-per-trial and cost-per-accepted-trial metrics in
    PROPOSAL.html §10.4.
    """

    model_config = ConfigDict(extra="forbid")

    trial_id: int = Field(ge=0)
    role: str
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    wall_clock_sec: float = Field(ge=0)
