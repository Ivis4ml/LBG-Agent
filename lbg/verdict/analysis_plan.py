"""`analysis_plan.yaml`: pre-registered evaluation plan.

PROPOSAL.html §20 (no_data_snooping): the analysis plan must be committed
BEFORE the sealed test opens. Re-opening the plan after seeing sealed
results would be reward-hacking; the schema is `extra=forbid` to prevent
silently injecting new criteria mid-run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class H1Criterion(BaseModel):
    """The H1 strong-and-weak verdict definition (PROPOSAL.html §10.5)."""

    model_config = ConfigDict(extra="forbid")

    test: Literal["moving_block_bootstrap"]
    block_len: int = Field(ge=2)
    n_bootstrap: int = Field(ge=100)
    alpha: float = Field(gt=0.0, lt=0.5)
    strong_criterion: Literal["ci_lower > 0"]
    weak_criterion: Literal["point_estimate > 0"]


class AnalysisPlan(BaseModel):
    """The pre-registered evaluation plan for one Discovery run."""

    model_config = ConfigDict(extra="forbid")

    baselines: list[str] = Field(min_length=1)
    h1: H1Criterion


DEFAULT_ANALYSIS_PLAN: AnalysisPlan = AnalysisPlan(
    baselines=["buy_and_hold", "sixty_forty"],
    h1=H1Criterion(
        test="moving_block_bootstrap",
        block_len=10,
        n_bootstrap=1000,
        alpha=0.05,
        strong_criterion="ci_lower > 0",
        weak_criterion="point_estimate > 0",
    ),
)


def load_analysis_plan(path: str | Path) -> AnalysisPlan:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return AnalysisPlan.model_validate(raw)
