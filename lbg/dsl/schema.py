"""Pydantic schemas for `strategy.yaml`.

Discriminated unions on `rule` and `mode` mean an Editor proposal can only
swap one of a closed enumeration of block types — the structure validates
against arbitrary YAML noise.

Indicator function names refer to files in `indicators/` (loaded via
`lbg.sandbox.load_indicator`). The policy interpreter ties them together.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IndicatorSpec(BaseModel):
    """One indicator instance in the strategy."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    fn: str = Field(min_length=1, description="indicators/<fn>.py")
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "fn")
    @classmethod
    def _no_path_separators(cls, v: str) -> str:
        if "/" in v or "\\" in v or ".." in v:
            raise ValueError(f"path separators not allowed in {v!r}")
        return v


class CrossAboveRule(BaseModel):
    """Entry/exit on `fast` crossing above `slow`."""

    model_config = ConfigDict(extra="forbid")

    rule: Literal["cross_above"]
    fast: str
    slow: str


class CrossBelowRule(BaseModel):
    """Entry/exit on `fast` crossing below `slow`."""

    model_config = ConfigDict(extra="forbid")

    rule: Literal["cross_below"]
    fast: str
    slow: str


CrossRule = Annotated[
    CrossAboveRule | CrossBelowRule,
    Field(discriminator="rule"),
]


class IndicatorAboveFilter(BaseModel):
    """Entry allowed only when `indicator >= threshold`."""

    model_config = ConfigDict(extra="forbid")

    rule: Literal["indicator_above"]
    indicator: str
    threshold: float


class IndicatorBelowFilter(BaseModel):
    """Entry allowed only when `indicator <= threshold`."""

    model_config = ConfigDict(extra="forbid")

    rule: Literal["indicator_below"]
    indicator: str
    threshold: float


Filter = Annotated[
    IndicatorAboveFilter | IndicatorBelowFilter,
    Field(discriminator="rule"),
]


class FixedFractionSizing(BaseModel):
    """Constant position size when in-position."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["fixed_fraction"]
    fraction: float = Field(ge=0.0, le=1.0)
    max_position: float = Field(ge=0.0, le=1.0)


class VolatilityTargetSizing(BaseModel):
    """Position scaled by `target_vol / realized_vol` over a trailing window."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["volatility_target"]
    target_vol: float = Field(gt=0.0)
    max_position: float = Field(ge=0.0, le=1.0)
    vol_lookback: int = Field(ge=2, default=20)


Sizing = Annotated[
    FixedFractionSizing | VolatilityTargetSizing,
    Field(discriminator="mode"),
]


class Strategy(BaseModel):
    """Top-level `strategy.yaml` document."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    indicators: list[IndicatorSpec] = Field(min_length=1)
    entry: CrossRule
    exit: CrossRule
    filters: list[Filter] = Field(default_factory=list)
    # exit_filters: when in position, any True exit_filter forces an exit.
    # Combined with the exit CrossRule via OR (any path out fires). Allows
    # `add_indicator` on the buyhold baseline -- which has no entry events
    # to filter -- to produce a meaningful candidate by gating the EXIT
    # instead of the entry. (PROPOSAL §11 follow-on.)
    exit_filters: list[Filter] = Field(default_factory=list)
    sizing: Sizing

    @field_validator("indicators")
    @classmethod
    def _names_unique(cls, v: list[IndicatorSpec]) -> list[IndicatorSpec]:
        names = [i.name for i in v]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate indicator names: {names}")
        return v

    def referenced_indicator_names(self) -> set[str]:
        """Names referenced by entry, exit, or any filter (entry or exit).
        Indicators that compute a series but feed nothing in the policy
        are unwired."""
        names: set[str] = {
            self.entry.fast,
            self.entry.slow,
            self.exit.fast,
            self.exit.slow,
        }
        for f in self.filters:
            names.add(f.indicator)
        for f in self.exit_filters:
            names.add(f.indicator)
        return names

    def unwired_indicators(self) -> list[str]:
        """Indicator names declared but never read by entry/exit/filters.

        Adding an indicator without consuming it makes the candidate strategy
        bit-identical to the incumbent on every bar -- a noop. STAGE1_REPORT
        § 7 observed Editor wasting ~5/8 of one provider's trials on exactly
        this pattern, so CandidateBuilder rejects the candidate before any
        backtest happens.
        """
        referenced = self.referenced_indicator_names()
        return [i.name for i in self.indicators if i.name not in referenced]
