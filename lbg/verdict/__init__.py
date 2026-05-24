"""H1 verdict via moving-block bootstrap (PROPOSAL.html §10.5).

The per-trial validation gate is multi-objective and lenient. The final H1
verdict, opened exactly once on the sealed window, is strict:
moving-block-bootstrap 95% CI of paired Sharpe difference vs the strongest
pre-registered baseline. CI lower bound > 0 ⇒ H1 strong.

Per user direction the pre-registered baselines are:
  - `buy_and_hold` (position = 1.0)
  - `sixty_forty`  (position = 0.6, 40% in zero-return cash)
"""

from lbg.verdict.analysis_plan import (
    DEFAULT_ANALYSIS_PLAN,
    AnalysisPlan,
    H1Criterion,
    load_analysis_plan,
)
from lbg.verdict.baselines import (
    BASELINE_REGISTRY,
    baseline_buy_and_hold,
    baseline_sixty_forty,
)
from lbg.verdict.bootstrap import moving_block_bootstrap_sharpe_diff
from lbg.verdict.h1 import H1Verdict, compute_h1_verdict

__all__ = [
    "BASELINE_REGISTRY",
    "DEFAULT_ANALYSIS_PLAN",
    "AnalysisPlan",
    "H1Criterion",
    "H1Verdict",
    "baseline_buy_and_hold",
    "baseline_sixty_forty",
    "compute_h1_verdict",
    "load_analysis_plan",
    "moving_block_bootstrap_sharpe_diff",
]
