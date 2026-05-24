"""Step 3 · unwired-indicator detection and atomic-attach add_indicator.

PROPOSAL §6.5 + STAGE1_REPORT § 7 motivate the structural rule: every
indicator declared in `strategy.yaml` must be referenced by entry, exit,
or a filter. Otherwise the candidate produces positions bit-identical to
the incumbent and wastes the trial slot.

These tests cover three layers:
  * Strategy.unwired_indicators() returns the dead-code names.
  * CandidateBuilder rejects bare add_indicator (no attach).
  * CandidateBuilder accepts add_indicator + attach atomically.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest

from lbg.builder import CandidateBuildError, CandidateBuilder
from lbg.dsl import load_strategy
from lbg.dsl.schema import (
    CrossAboveRule,
    CrossBelowRule,
    FixedFractionSizing,
    IndicatorAboveFilter,
    IndicatorSpec,
    Strategy,
)
from lbg.parser import AddIndicatorPayload
from lbg.schemas import EditProposal, EditType, ExpectedTrainSignal, ExpectedValidationSignal

REPO = Path(__file__).resolve().parents[1]


def _baseline_with_extra(extra_name: str) -> Strategy:
    """Hand-build a strategy with one referenced + one unreferenced indicator."""
    return Strategy(
        name="t",
        indicators=[
            IndicatorSpec(name="sma_fast", fn="sma", params={"period": 20}),
            IndicatorSpec(name="sma_slow", fn="sma", params={"period": 50}),
            IndicatorSpec(name=extra_name, fn="rsi", params={"period": 14}),
        ],
        entry=CrossAboveRule(rule="cross_above", fast="sma_fast", slow="sma_slow"),
        exit=CrossBelowRule(rule="cross_below", fast="sma_fast", slow="sma_slow"),
        filters=[],
        sizing=FixedFractionSizing(mode="fixed_fraction", fraction=1.0, max_position=1.0),
    )


# -------- Strategy.unwired_indicators --------


def test_baseline_strategy_has_no_unwired_indicators():
    """The shipping baseline references sma_fast / sma_slow in entry+exit."""
    s = load_strategy(REPO / "strategy.yaml")
    assert s.unwired_indicators() == []


def test_unwired_indicator_detected_when_not_referenced():
    s = _baseline_with_extra("rsi_14")
    assert s.unwired_indicators() == ["rsi_14"]


def test_referenced_via_filter_is_not_unwired():
    """An indicator referenced only by a filter still counts as wired."""
    s = _baseline_with_extra("rsi_14")
    wired = s.model_copy(
        update={
            "filters": [
                IndicatorAboveFilter(rule="indicator_above", indicator="rsi_14", threshold=30.0)
            ]
        }
    )
    assert wired.unwired_indicators() == []


def test_referenced_indicator_names_returns_all_four_legs():
    s = load_strategy(REPO / "strategy.yaml")
    refs = s.referenced_indicator_names()
    # sma_fast appears as both entry.fast and exit.fast; the set dedupes.
    assert refs == {"sma_fast", "sma_slow"}


# -------- builder: bare add_indicator rejected --------


@pytest.fixture
def working_tree(tmp_path):
    shutil.copy(REPO / "strategy.yaml", tmp_path / "strategy.yaml")
    (tmp_path / "indicators").mkdir()
    shutil.copy(REPO / "indicators" / "sma.py", tmp_path / "indicators" / "sma.py")
    shutil.copy(REPO / "indicators" / "__init__.py", tmp_path / "indicators" / "__init__.py")
    return tmp_path


def _proposal(edit_type: EditType, change: dict) -> EditProposal:
    return EditProposal.model_validate(
        {
            "trial_id": 0,
            "hypothesis": "test",
            "proposed_edit": {"type": edit_type.value, "change": change},
            "expected_train_signal": ExpectedTrainSignal.NEUTRAL.value,
            "expected_validation_signal": ExpectedValidationSignal.REJECT.value,
            "fallback_if_rejected": "x",
        }
    )


_PURE_RSI = textwrap.dedent(
    """
    import pandas as pd

    def rsi(df, period=14):
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss
        return 100 - 100 / (1 + rs)
    """
).lstrip()


def test_bare_add_indicator_is_rejected(working_tree):
    """Without `attach`, the new indicator is not referenced anywhere and
    CandidateBuilder refuses to apply the edit."""
    parent = load_strategy(working_tree / "strategy.yaml")
    builder = CandidateBuilder(working_tree)
    payload = AddIndicatorPayload(
        name="rsi_14",
        fn="rsi",
        source=_PURE_RSI,
        params={"period": 14},
        attach=None,
    )
    with pytest.raises(CandidateBuildError, match="unwired"):
        builder.apply(parent, _proposal(EditType.ADD_INDICATOR, payload.model_dump()), payload)


def test_add_indicator_with_attach_succeeds(working_tree):
    """With `attach` referencing the new indicator, the candidate is wired."""
    parent = load_strategy(working_tree / "strategy.yaml")
    builder = CandidateBuilder(working_tree)
    payload = AddIndicatorPayload(
        name="rsi_14",
        fn="rsi",
        source=_PURE_RSI,
        params={"period": 14},
        attach=IndicatorAboveFilter(rule="indicator_above", indicator="rsi_14", threshold=30.0),
    )
    res = builder.apply(parent, _proposal(EditType.ADD_INDICATOR, payload.model_dump()), payload)
    assert res.new_strategy.unwired_indicators() == []
    assert res.new_strategy.filters[-1].indicator == "rsi_14"
    assert "rsi_14" in res.edit_summary.summary
    # File written.
    assert (working_tree / "indicators" / "rsi.py").exists()


def test_add_indicator_attach_must_reference_new_name(working_tree):
    """If attach.indicator names something other than the new indicator,
    the new one is still unwired -- the builder refuses with a clear msg."""
    parent = load_strategy(working_tree / "strategy.yaml")
    builder = CandidateBuilder(working_tree)
    payload = AddIndicatorPayload(
        name="rsi_14",
        fn="rsi",
        source=_PURE_RSI,
        params={"period": 14},
        attach=IndicatorAboveFilter(
            rule="indicator_above",
            indicator="sma_fast",  # not the new indicator
            threshold=0.0,
        ),
    )
    with pytest.raises(CandidateBuildError, match="attach.indicator must equal"):
        builder.apply(parent, _proposal(EditType.ADD_INDICATOR, payload.model_dump()), payload)
