"""Strategy complexity (PROPOSAL.html §4.4 line 1032).

    Complexity(H) = λ_1 * |indicators|
                  + λ_2 * |filters|
                  + λ_3 * |non_default_params|
                  + λ_4 * LOC_total
                  + λ_5 * #branches_total

LOC and branches are summed across the agent-authored indicator files. A
"branch" is an `If`, `While`, `For`, `Try` AST node — any control-flow
fork. Comments and blank lines are not counted toward LOC.

Weights are frozen per PROPOSAL.html §20 ("locked per discovery run"). Do
not edit these without bumping the analysis plan.
"""

from __future__ import annotations

import ast
from pathlib import Path

from lbg.dsl.schema import Strategy

DEFAULT_INDICATORS_DIR = Path("indicators")

LAMBDA_INDICATOR: float = 1.0
LAMBDA_FILTER: float = 0.5
LAMBDA_PARAM: float = 0.1
LAMBDA_LOC: float = 0.005
LAMBDA_BRANCH: float = 0.05

_BRANCH_NODE_TYPES = (ast.If, ast.While, ast.For, ast.Try, ast.IfExp)


def _count_loc_and_branches(source: str) -> tuple[int, int]:
    """Return (effective_loc, branch_count) for one Python source string.

    Effective LOC = non-blank, non-comment-only lines. Branch count = the
    number of `If/While/For/Try/IfExp` AST nodes anywhere in the module.
    """
    lines = 0
    for raw in source.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        lines += 1

    branches = 0
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # An unparseable indicator would already have been rejected by the
        # invariant runner; assign zero branches and let the LOC dominate.
        return lines, 0
    for node in ast.walk(tree):
        if isinstance(node, _BRANCH_NODE_TYPES):
            branches += 1
    return lines, branches


def _indicator_source_totals(strategy: Strategy, indicators_dir: Path) -> tuple[int, int]:
    loc_total = 0
    branch_total = 0
    seen_fns: set[str] = set()
    for spec in strategy.indicators:
        if spec.fn in seen_fns:
            continue
        seen_fns.add(spec.fn)
        path = indicators_dir / f"{spec.fn}.py"
        if not path.exists():
            continue
        loc, br = _count_loc_and_branches(path.read_text(encoding="utf-8"))
        loc_total += loc
        branch_total += br
    return loc_total, branch_total


def complexity_score(
    strategy: Strategy, *, indicators_dir: Path | str = DEFAULT_INDICATORS_DIR
) -> float:
    """Return the locked complexity measure for `strategy`.

    When `indicators_dir` is missing or the indicator files cannot be
    opened, the LOC/branch terms collapse to zero — the indicator/filter/
    param-count terms still apply.
    """
    n_indicators = len(strategy.indicators)
    n_filters = len(strategy.filters)
    n_params = sum(len(i.params) for i in strategy.indicators)

    loc_total, branch_total = _indicator_source_totals(strategy, Path(indicators_dir))

    return (
        LAMBDA_INDICATOR * n_indicators
        + LAMBDA_FILTER * n_filters
        + LAMBDA_PARAM * n_params
        + LAMBDA_LOC * loc_total
        + LAMBDA_BRANCH * branch_total
    )
