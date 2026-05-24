"""(i) parameter_change must support `filters[<idx>].threshold`.

The live campaign (campaigns/iteration_002) caught the Editor trying to
tune a freshly-added filter's threshold via `parameter_change`. The DSL
rejected the path. This test pins the new accepted form and the error
messages around it.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest

from lbg.builder import CandidateBuildError, CandidateBuilder
from lbg.dsl import load_strategy
from lbg.dsl.schema import IndicatorAboveFilter
from lbg.parser import AddIndicatorPayload, ParameterChangePayload
from lbg.schemas import EditProposal, EditType, ExpectedTrainSignal, ExpectedValidationSignal

REPO = Path(__file__).resolve().parents[1]


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
            "hypothesis": "t",
            "proposed_edit": {"type": edit_type.value, "change": change},
            "expected_train_signal": ExpectedTrainSignal.NEUTRAL.value,
            "expected_validation_signal": ExpectedValidationSignal.REJECT.value,
            "fallback_if_rejected": "x",
        }
    )


_RSI = textwrap.dedent(
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


def _add_rsi_with_filter(builder: CandidateBuilder, parent, threshold: float):
    payload = AddIndicatorPayload(
        name="rsi_14",
        fn="rsi",
        source=_RSI,
        params={"period": 14},
        attach=IndicatorAboveFilter(
            rule="indicator_above", indicator="rsi_14", threshold=threshold
        ),
    )
    return builder.apply(parent, _proposal(EditType.ADD_INDICATOR, payload.model_dump()), payload)


def test_parameter_change_tunes_filter_threshold(working_tree):
    parent = load_strategy(working_tree / "strategy.yaml")
    builder = CandidateBuilder(working_tree)
    res = _add_rsi_with_filter(builder, parent, threshold=30.0)
    # Now retune the filter via parameter_change.
    after_add = res.new_strategy
    pc = ParameterChangePayload(path="filters[0].threshold", value=50.0)
    res2 = builder.apply(after_add, _proposal(EditType.PARAMETER_CHANGE, pc.model_dump()), pc)
    assert res2.new_strategy.filters[0].threshold == pytest.approx(50.0)
    assert "filters[0].threshold" in res2.edit_summary.summary


def test_filter_threshold_out_of_range_raises(working_tree):
    parent = load_strategy(working_tree / "strategy.yaml")
    builder = CandidateBuilder(working_tree)
    res = _add_rsi_with_filter(builder, parent, threshold=30.0)
    pc = ParameterChangePayload(path="filters[1].threshold", value=50.0)
    with pytest.raises(CandidateBuildError, match="out of range"):
        builder.apply(res.new_strategy, _proposal(EditType.PARAMETER_CHANGE, pc.model_dump()), pc)


def test_filter_threshold_requires_numeric_value(working_tree):
    parent = load_strategy(working_tree / "strategy.yaml")
    builder = CandidateBuilder(working_tree)
    res = _add_rsi_with_filter(builder, parent, threshold=30.0)
    pc = ParameterChangePayload(path="filters[0].threshold", value="high")
    with pytest.raises(CandidateBuildError, match="must be a number"):
        builder.apply(res.new_strategy, _proposal(EditType.PARAMETER_CHANGE, pc.model_dump()), pc)


def test_error_message_lists_all_three_supported_path_forms(working_tree):
    parent = load_strategy(working_tree / "strategy.yaml")
    builder = CandidateBuilder(working_tree)
    pc = ParameterChangePayload(path="entry.rule", value="cross_below")
    with pytest.raises(CandidateBuildError, match="filters\\[<idx>\\].threshold"):
        builder.apply(parent, _proposal(EditType.PARAMETER_CHANGE, pc.model_dump()), pc)
