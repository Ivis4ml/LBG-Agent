"""Sealed verdict helpers (PROPOSAL.html §10.5).

Primary H1 is the count of sealed-validated alpha cards. The final strategy
Sharpe bootstrap remains available as a supplementary diagnostic.
"""

from lbg.verdict.analysis_plan import (
    DEFAULT_ANALYSIS_PLAN,
    AnalysisPlan,
    H1Criterion,
    PerCardValidationCriterion,
    StrategySharpeCriterion,
    load_analysis_plan,
)
from lbg.verdict.baselines import (
    BASELINE_REGISTRY,
    baseline_buy_and_hold,
    baseline_sixty_forty,
)
from lbg.verdict.bootstrap import moving_block_bootstrap_sharpe_diff
from lbg.verdict.h1 import (
    AlphaCardH1Verdict,
    H1Verdict,
    StrategySharpeVerdict,
    compute_alpha_card_h1_verdict,
    compute_h1_verdict,
    compute_strategy_sharpe_verdict,
)
from lbg.verdict.per_card import (
    PerCardValidationResult,
    compute_per_card_sealed_validation,
)

__all__ = [
    "AlphaCardH1Verdict",
    "BASELINE_REGISTRY",
    "DEFAULT_ANALYSIS_PLAN",
    "AnalysisPlan",
    "H1Criterion",
    "H1Verdict",
    "PerCardValidationCriterion",
    "PerCardValidationResult",
    "StrategySharpeCriterion",
    "StrategySharpeVerdict",
    "baseline_buy_and_hold",
    "baseline_sixty_forty",
    "compute_alpha_card_h1_verdict",
    "compute_h1_verdict",
    "compute_per_card_sealed_validation",
    "compute_strategy_sharpe_verdict",
    "load_analysis_plan",
    "moving_block_bootstrap_sharpe_diff",
]
