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


class PerCardValidationCriterion(BaseModel):
    """How one alpha card earns a validated-factor count."""

    model_config = ConfigDict(extra="forbid")

    # `pathwise_incremental_sharpe`: backtest the strategy at the parent
    # commit vs the trial commit on the sealed window, take paired daily
    # returns, and bootstrap the Sharpe difference. This is the only
    # operationalization that covers every edit type — strict LOO has no
    # clean definition for cards whose indicator sits in entry/exit
    # cross rules.
    method: Literal["pathwise_incremental_sharpe"]
    ci: float = Field(gt=0.0, lt=1.0)
    ci_method: Literal["moving_block_bootstrap"]
    block_len: int = Field(ge=2)
    n_bootstrap: int = Field(ge=100)
    threshold: Literal["lower_bound_gt_zero"]
    # PROPOSAL §20 item 32: Benjamini-Hochberg FDR control across the
    # family of accepted cards. The H1 validated-factor count counts
    # cards whose `bh_validated` flag is True. The raw CI lower bound
    # > 0 column is retained for backwards-compatible inspection but is
    # no longer the primary H1 signal.
    bh_q: float = Field(default=0.10, gt=0.0, lt=1.0)


class H1Criterion(BaseModel):
    """Primary H1: count sealed-validated alpha cards (PROPOSAL §10.5)."""

    model_config = ConfigDict(extra="forbid")

    test: Literal["validated_factor_count"]
    tau: int = Field(ge=1)
    strong_criterion: Literal["N_val_lbg > max_baseline_count and N_val_lbg >= tau"]
    weak_criterion: Literal["N_val_lbg >= 1 and N_val_lbg >= best_baseline_count"]
    baseline_validated_counts: dict[str, int] = Field(default_factory=dict)


class StrategySharpeCriterion(BaseModel):
    """Supplementary strategy-level comparison, not the primary H1."""

    model_config = ConfigDict(extra="forbid")

    test: Literal["moving_block_bootstrap"]
    metric: Literal["sealed_strategy_sharpe"]
    reference: Literal["best_pre_registered_baseline"]
    block_len: int = Field(ge=2)
    n_bootstrap: int = Field(ge=100)
    alpha: float = Field(gt=0.0, lt=0.5)
    strong_criterion: Literal["ci_lower > 0"]
    weak_criterion: Literal["point_estimate > 0"]


class AnalysisPlan(BaseModel):
    """The pre-registered evaluation plan for one Discovery run."""

    model_config = ConfigDict(extra="forbid")

    baselines: list[str] = Field(min_length=1)
    per_card_validation: PerCardValidationCriterion
    h1: H1Criterion
    supplementary_strategy: StrategySharpeCriterion


DEFAULT_ANALYSIS_PLAN: AnalysisPlan = AnalysisPlan(
    baselines=["buy_and_hold", "sixty_forty"],
    per_card_validation=PerCardValidationCriterion(
        method="pathwise_incremental_sharpe",
        ci=0.95,
        ci_method="moving_block_bootstrap",
        block_len=10,
        n_bootstrap=1000,
        threshold="lower_bound_gt_zero",
    ),
    h1=H1Criterion(
        test="validated_factor_count",
        tau=3,
        strong_criterion="N_val_lbg > max_baseline_count and N_val_lbg >= tau",
        weak_criterion="N_val_lbg >= 1 and N_val_lbg >= best_baseline_count",
        baseline_validated_counts={},
    ),
    supplementary_strategy=StrategySharpeCriterion(
        test="moving_block_bootstrap",
        metric="sealed_strategy_sharpe",
        reference="best_pre_registered_baseline",
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
