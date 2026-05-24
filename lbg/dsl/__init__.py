"""DSL schemas and loader for `strategy.yaml`.

PROPOSAL.html §6.2 lists the eight edit types the Editor may emit. Five of
them (`add_indicator`, `parameter_change`, `add_filter`, `remove_filter`,
`change_sizing_mode`, `change_exit_rule`, `simplify`) modify `strategy.yaml`
in structured, discrete ways. The schemas here encode exactly the
structure those edits operate on, so the Editor cannot smuggle arbitrary
Python into the strategy.
"""

from lbg.dsl.loader import load_strategy
from lbg.dsl.schema import (
    CrossAboveRule,
    CrossBelowRule,
    FixedFractionSizing,
    IndicatorAboveFilter,
    IndicatorBelowFilter,
    IndicatorSpec,
    Strategy,
    VolatilityTargetSizing,
)

__all__ = [
    "CrossAboveRule",
    "CrossBelowRule",
    "FixedFractionSizing",
    "IndicatorAboveFilter",
    "IndicatorBelowFilter",
    "IndicatorSpec",
    "Strategy",
    "VolatilityTargetSizing",
    "load_strategy",
]
