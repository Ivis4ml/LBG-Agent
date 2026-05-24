"""exit_filters: when the strategy is in position and any exit_filter
fires, force an exit. Motivated by § 12: buyhold baseline has only one
entry event, so entry filters cut it to zero trades; exit filters gate
the EXIT decision and are the structural path to non-trivial
`add_indicator` accepts on that baseline.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from lbg.builder import CandidateBuildError, CandidateBuilder
from lbg.dsl import load_strategy
from lbg.dsl.schema import (
    CrossAboveRule,
    CrossBelowRule,
    FixedFractionSizing,
    IndicatorAboveFilter,
    IndicatorBelowFilter,
    IndicatorSpec,
    Strategy,
)
from lbg.parser import AddFilterPayload, AddIndicatorPayload, ParameterChangePayload
from lbg.schemas import EditProposal, EditType, ExpectedTrainSignal, ExpectedValidationSignal

REPO = Path(__file__).resolve().parents[1]


# -------- schema --------


def test_strategy_exit_filters_defaults_empty():
    s = load_strategy(REPO / "strategy.yaml")
    assert s.exit_filters == []


def test_referenced_indicator_names_includes_exit_filters():
    """An indicator wired only into exit_filters still counts as
    referenced (must not trigger the unwired check)."""
    s = Strategy(
        name="t",
        indicators=[
            IndicatorSpec(name="sma_fast", fn="sma", params={"period": 20}),
            IndicatorSpec(name="sma_slow", fn="sma", params={"period": 50}),
            IndicatorSpec(name="rsi_14", fn="rsi", params={"period": 14}),
        ],
        entry=CrossAboveRule(rule="cross_above", fast="sma_fast", slow="sma_slow"),
        exit=CrossBelowRule(rule="cross_below", fast="sma_fast", slow="sma_slow"),
        filters=[],
        exit_filters=[
            IndicatorAboveFilter(rule="indicator_above", indicator="rsi_14", threshold=70.0)
        ],
        sizing=FixedFractionSizing(mode="fixed_fraction", fraction=1.0, max_position=1.0),
    )
    refs = s.referenced_indicator_names()
    assert "rsi_14" in refs
    assert s.unwired_indicators() == []


# -------- policy_interpreter --------


def test_exit_filter_forces_position_exit_when_triggered():
    """Construct a tiny strategy where exit_filter > threshold forces
    exit even when the cross_below exit rule never fires. Verify the
    binary position state drops to 0 at the bar where the filter fires."""
    from policy_interpreter import _eval_filter, _state_from_events

    n = 10
    df = pd.DataFrame(
        {
            "open": [1.0] * n,
            "high": [1.0] * n,
            "low": [1.0] * n,
            "close": [1.0] * n,
            "volume": [1] * n,
        }
    )
    # Entry fires at bar 2; exit cross rule never fires; exit_filter
    # fires at bar 5 (a spike series above threshold).
    entry = pd.Series([False, False, True, False, False, False, False, False, False, False])
    exit_cross = pd.Series([False] * n)
    # Synthetic indicator series: 0 everywhere except bar 5 = 1.0; threshold 0.5.
    spike = pd.Series([0.0] * n)
    spike.iloc[5] = 1.0
    indicators = {"spike": spike}
    exit_filter_signal = _eval_filter(
        IndicatorAboveFilter(rule="indicator_above", indicator="spike", threshold=0.5),
        indicators,
    )
    exit_combined = exit_cross | exit_filter_signal
    state = _state_from_events(entry, exit_combined)
    # In position from bar 2 inclusive, until bar 5 forces exit.
    assert state.tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]


# -------- CandidateBuilder · add_filter target=exit --------


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


def test_add_filter_target_exit_appends_to_exit_filters(working_tree):
    parent = load_strategy(working_tree / "strategy.yaml")
    builder = CandidateBuilder(working_tree)
    payload = AddFilterPayload(
        filter=IndicatorAboveFilter(rule="indicator_above", indicator="sma_fast", threshold=100.0),
        target="exit",
    )
    res = builder.apply(parent, _proposal(EditType.ADD_FILTER, payload.model_dump()), payload)
    assert len(res.new_strategy.filters) == 0
    assert len(res.new_strategy.exit_filters) == 1
    assert res.new_strategy.exit_filters[0].indicator == "sma_fast"
    assert "exit_" in res.edit_summary.summary


def test_add_filter_target_entry_default_still_appends_to_filters(working_tree):
    parent = load_strategy(working_tree / "strategy.yaml")
    builder = CandidateBuilder(working_tree)
    payload = AddFilterPayload(
        filter=IndicatorAboveFilter(rule="indicator_above", indicator="sma_fast", threshold=100.0)
        # target defaults to "entry"
    )
    res = builder.apply(parent, _proposal(EditType.ADD_FILTER, payload.model_dump()), payload)
    assert len(res.new_strategy.filters) == 1
    assert len(res.new_strategy.exit_filters) == 0


# -------- CandidateBuilder · add_indicator attach_target=exit --------


_PURE_DRAWDOWN = textwrap.dedent(
    """
    import pandas as pd
    def drawdown(df, lookback=60, **params):
        close = df['close']
        peak = close.rolling(int(lookback), min_periods=1).max()
        return (peak - close) / peak
    """
).lstrip()


def test_add_indicator_attach_target_exit_wires_into_exit_filters(working_tree):
    parent = load_strategy(working_tree / "strategy.yaml")
    builder = CandidateBuilder(working_tree)
    payload = AddIndicatorPayload(
        name="drawdown_60",
        fn="drawdown",
        source=_PURE_DRAWDOWN,
        params={"lookback": 60},
        attach=IndicatorAboveFilter(
            rule="indicator_above", indicator="drawdown_60", threshold=0.10
        ),
        attach_target="exit",
    )
    res = builder.apply(parent, _proposal(EditType.ADD_INDICATOR, payload.model_dump()), payload)
    # Indicator added.
    assert any(i.name == "drawdown_60" for i in res.new_strategy.indicators)
    # Wired as exit_filter, not entry filter.
    assert len(res.new_strategy.filters) == 0
    assert len(res.new_strategy.exit_filters) == 1
    assert res.new_strategy.exit_filters[0].indicator == "drawdown_60"
    # Not unwired.
    assert res.new_strategy.unwired_indicators() == []
    # Summary reflects the exit path.
    assert "exit_" in res.edit_summary.summary


def test_add_indicator_default_attach_target_remains_entry(working_tree):
    """Backward compat: payloads without attach_target default to entry."""
    parent = load_strategy(working_tree / "strategy.yaml")
    builder = CandidateBuilder(working_tree)
    payload = AddIndicatorPayload(
        name="drawdown_60",
        fn="drawdown",
        source=_PURE_DRAWDOWN,
        params={"lookback": 60},
        attach=IndicatorAboveFilter(
            rule="indicator_above", indicator="drawdown_60", threshold=0.10
        ),
        # attach_target defaults to "entry"
    )
    res = builder.apply(parent, _proposal(EditType.ADD_INDICATOR, payload.model_dump()), payload)
    assert len(res.new_strategy.filters) == 1
    assert len(res.new_strategy.exit_filters) == 0


# -------- parameter_change exit_filters[N].threshold --------


def test_parameter_change_tunes_exit_filter_threshold(working_tree):
    parent = load_strategy(working_tree / "strategy.yaml")
    builder = CandidateBuilder(working_tree)
    # First add an exit_filter via add_indicator + attach_target=exit.
    add = AddIndicatorPayload(
        name="drawdown_60",
        fn="drawdown",
        source=_PURE_DRAWDOWN,
        params={"lookback": 60},
        attach=IndicatorAboveFilter(
            rule="indicator_above", indicator="drawdown_60", threshold=0.10
        ),
        attach_target="exit",
    )
    res1 = builder.apply(parent, _proposal(EditType.ADD_INDICATOR, add.model_dump()), add)
    # Tune the threshold via parameter_change.
    pc = ParameterChangePayload(path="exit_filters[0].threshold", value=0.20)
    res2 = builder.apply(
        res1.new_strategy, _proposal(EditType.PARAMETER_CHANGE, pc.model_dump()), pc
    )
    assert res2.new_strategy.exit_filters[0].threshold == pytest.approx(0.20)
    # The original entry filter list is unchanged (empty).
    assert res2.new_strategy.filters == []


def test_parameter_change_filters_index_still_works(working_tree):
    """Backward compat: filters[<idx>].threshold (entry list) still resolves."""
    parent = load_strategy(working_tree / "strategy.yaml")
    builder = CandidateBuilder(working_tree)
    add = AddIndicatorPayload(
        name="rsi_14",
        fn="rsi",
        source=textwrap.dedent(
            """
            import pandas as pd
            def rsi(df, period=14, **params):
                delta = df['close'].diff()
                gain = delta.clip(lower=0).rolling(period).mean()
                loss = (-delta.clip(upper=0)).rolling(period).mean()
                rs = gain / loss
                return 100 - 100 / (1 + rs)
            """
        ).lstrip(),
        params={"period": 14},
        attach=IndicatorAboveFilter(rule="indicator_above", indicator="rsi_14", threshold=30.0),
        attach_target="entry",
    )
    res1 = builder.apply(parent, _proposal(EditType.ADD_INDICATOR, add.model_dump()), add)
    pc = ParameterChangePayload(path="filters[0].threshold", value=40.0)
    res2 = builder.apply(
        res1.new_strategy, _proposal(EditType.PARAMETER_CHANGE, pc.model_dump()), pc
    )
    assert res2.new_strategy.filters[0].threshold == pytest.approx(40.0)


# -------- end-to-end on buyhold baseline --------


def test_buyhold_with_drawdown_exit_filter_reduces_drawdown(tmp_path):
    """The whole point: on buyhold, an exit_filter that fires during
    drawdowns should produce a candidate with strictly better train MDD
    than baseline at minimal Sharpe cost. This is the structural path
    to a non-trivial add_indicator accept."""
    # Copy buyhold baseline into a fresh working tree.
    buyhold = REPO / "baselines" / "buyhold"
    shutil.copy(buyhold / "strategy.yaml", tmp_path / "strategy.yaml")
    shutil.copytree(buyhold / "indicators", tmp_path / "indicators")

    # Add a drawdown exit_filter via add_indicator.
    parent = load_strategy(tmp_path / "strategy.yaml")
    builder = CandidateBuilder(tmp_path)
    add = AddIndicatorPayload(
        name="drawdown_60",
        fn="drawdown",
        source=_PURE_DRAWDOWN,
        params={"lookback": 60},
        attach=IndicatorAboveFilter(
            rule="indicator_above", indicator="drawdown_60", threshold=0.10
        ),
        attach_target="exit",
    )
    res = builder.apply(parent, _proposal(EditType.ADD_INDICATOR, add.model_dump()), add)
    assert len(res.new_strategy.exit_filters) == 1

    # Backtest both.
    from backtest import run_backtest
    from lbg.data.loader import load_split
    from policy_interpreter import compute_positions

    df = load_split("split_A")
    positions_base = compute_positions(
        parent, df, indicators_dir=tmp_path / "indicators", timeout_sec=10.0
    )
    positions_cand = compute_positions(
        res.new_strategy, df, indicators_dir=tmp_path / "indicators", timeout_sec=10.0
    )
    base = run_backtest(positions_base, df)
    cand = run_backtest(positions_cand, df)
    # The filter should reduce |MaxDD| (positions go to 0 during drawdowns).
    assert cand.max_drawdown > base.max_drawdown, (
        f"expected exit_filter to reduce |MDD|, baseline mdd={base.max_drawdown:.4f}, "
        f"candidate mdd={cand.max_drawdown:.4f}"
    )
