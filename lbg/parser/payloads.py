"""Pydantic schemas for the `proposed_edit.change` block, per edit type.

Eight edit types from PROPOSAL.html §6.2. Each schema is `extra='forbid'` so
the Editor cannot smuggle calendar years or other free-form fields through
the structured edit channel.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from lbg.dsl.schema import CrossRule, Filter, IndicatorSpec, Sizing


class AddIndicatorPayload(BaseModel):
    """`proposed_edit.change` for `add_indicator`.

    Writes `indicators/<fn>.py` with `source`, and appends an `IndicatorSpec`
    to the strategy. The new indicator is not auto-wired into entry/exit;
    a follow-up `parameter_change` or `change_exit_rule` does that.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="logical name in strategy.yaml")
    fn: str = Field(min_length=1, description="indicators/<fn>.py basename")
    source: str = Field(min_length=1, description="Python source code")
    params: dict[str, Any] = Field(default_factory=dict)


class ParameterChangePayload(BaseModel):
    """`proposed_edit.change` for `parameter_change`.

    Supported `path` forms (the implementation list in `lbg.builder.apply`):
      - `sizing.<field>`              e.g. `sizing.fraction`, `sizing.target_vol`
      - `indicators[<name>].params.<key>`   e.g. `indicators[sma_fast].params.period`
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    value: int | float | bool | str


class AddFilterPayload(BaseModel):
    """`proposed_edit.change` for `add_filter`."""

    model_config = ConfigDict(extra="forbid")

    filter: Filter


class RemoveFilterPayload(BaseModel):
    """`proposed_edit.change` for `remove_filter`.

    Specifies the filter by index into the current `filters` list (0-based).
    """

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)


class ChangeSizingModePayload(BaseModel):
    """`proposed_edit.change` for `change_sizing_mode`. Replaces the whole sizing block."""

    model_config = ConfigDict(extra="forbid")

    sizing: Sizing


class ChangeExitRulePayload(BaseModel):
    """`proposed_edit.change` for `change_exit_rule`. Replaces the whole exit rule."""

    model_config = ConfigDict(extra="forbid")

    exit: CrossRule


class SimplifyPayload(BaseModel):
    """`proposed_edit.change` for `simplify`.

    `component='indicator'` -> remove the indicator with the given name
      (only if no entry/exit/filter still references it).
    `component='filter'`    -> remove the filter at the given index.
    """

    model_config = ConfigDict(extra="forbid")

    component: Literal["indicator", "filter"]
    target: str = Field(min_length=1, description="indicator name or filter index")


class RevertToTrialPayload(BaseModel):
    """`proposed_edit.change` for `revert_to_trial_N`. Roll back to a prior trial's commit."""

    model_config = ConfigDict(extra="forbid")

    trial_id: int = Field(ge=0)


# Optional re-export of the spec type so callers can construct AddFilterPayload
# inline without poking into lbg.dsl.schema.
__all__ = [
    "AddFilterPayload",
    "AddIndicatorPayload",
    "ChangeExitRulePayload",
    "ChangeSizingModePayload",
    "IndicatorSpec",
    "ParameterChangePayload",
    "RemoveFilterPayload",
    "RevertToTrialPayload",
    "SimplifyPayload",
]
