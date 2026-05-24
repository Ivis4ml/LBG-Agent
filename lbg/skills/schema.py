"""Pydantic schema for `skills/<skill_id>.yaml` (PROPOSAL.html §6.2 line 1487)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lbg.schemas import EditType, ValidationSignal


class SkillStatus(StrEnum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    EXPLORATORY = "exploratory"


class SkillTrigger(BaseModel):
    """Conditions under which this skill applies."""

    model_config = ConfigDict(extra="forbid")

    validation_signal: ValidationSignal
    common_context: list[str] = Field(default_factory=list)


class SkillRecipe(BaseModel):
    """The edit shape the skill proposes when triggered.

    `change` is intentionally a free dict here -- it mirrors whatever the
    matching `ProposedEdit.change` payload looks like for the given
    `edit_type`. The actual payload is validated by `ProposalParser` when
    the Editor cites the skill.
    """

    model_config = ConfigDict(extra="forbid")

    edit_type: EditType
    change: dict[str, Any]


class SkillEvidence(BaseModel):
    """Trial IDs that motivated the skill or confirmed it on a future trial."""

    model_config = ConfigDict(extra="forbid")

    accepted_trials: list[int] = Field(default_factory=list)
    rejected_trials_that_motivated_it: list[int] = Field(default_factory=list)


class Skill(BaseModel):
    """One line of evidence in the `skills/` store."""

    model_config = ConfigDict(extra="forbid")

    skill_id: str = Field(min_length=1)
    status: SkillStatus
    trigger: SkillTrigger
    recipe: SkillRecipe
    evidence: SkillEvidence = Field(default_factory=SkillEvidence)
    known_failure_modes: list[str] = Field(default_factory=list)
    complexity_cost: float = Field(ge=0.0)
    last_curated_at_trial: int = Field(ge=0)

    @field_validator("skill_id")
    @classmethod
    def _no_path_separators(cls, v: str) -> str:
        if "/" in v or "\\" in v or ".." in v:
            raise ValueError(f"path separators not allowed in skill_id {v!r}")
        return v
