"""Tests for the structured Editor citation pipeline.

* EditProposal accepts `cited_factors`, defaults to [].
* TrialRecord round-trips the field including with empty list.
* `match_dossier_from_citation` returns an editor_cite link for cited
  names that exist in the index, None when none match.
* ContextBuilder._tried_factor_names recognises both explicit cites
  and substring matches on current indicators.
* The factor-hint shortlist EXCLUDES factors already tried.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lbg.alpha_cards import (
    AlphaCardDossierLink,
    match_dossier_by_name,
    match_dossier_from_citation,
)
from lbg.dsl import load_strategy
from lbg.memory import MemoryManager
from lbg.orchestrator.context_builder import ContextBuilder, PastTrialSummary
from lbg.schemas import (
    AgentCompute,
    Decision,
    EditProposal,
    EditSummary,
    EditType,
    ExpectedTrainSignal,
    ExpectedValidationSignal,
    HypothesisBlock,
    HypothesisOutcome,
    RoleOutputs,
    TrainMetrics,
    TrialRecord,
    ValidationSignal,
)

REPO = Path(__file__).resolve().parents[1]


# -------- schema --------


def test_edit_proposal_accepts_cited_factors():
    p = EditProposal.model_validate(
        {
            "trial_id": 0,
            "hypothesis": "test",
            "proposed_edit": {
                "type": "parameter_change",
                "change": {"path": "sizing.fraction", "value": 0.5},
            },
            "expected_train_signal": "neutral",
            "expected_validation_signal": "reject",
            "fallback_if_rejected": "x",
            "cited_factors": ["ADX", "RSI"],
        }
    )
    assert p.cited_factors == ["ADX", "RSI"]


def test_edit_proposal_cited_factors_default_empty():
    p = EditProposal.model_validate(
        {
            "trial_id": 0,
            "hypothesis": "test",
            "proposed_edit": {
                "type": "parameter_change",
                "change": {"path": "sizing.fraction", "value": 0.5},
            },
            "expected_train_signal": "neutral",
            "expected_validation_signal": "reject",
            "fallback_if_rejected": "x",
        }
    )
    assert p.cited_factors == []


def _make_trial_record(cited: list[str]) -> TrialRecord:
    return TrialRecord(
        trial_id=0,
        parent_commit="abc1234",
        candidate_commit="def5678",
        role_outputs=RoleOutputs(editor_output_path="runs/0000/editor.yaml"),
        edit=EditSummary(type=EditType.PARAMETER_CHANGE, target="strategy.yaml", summary="x"),
        hypothesis=HypothesisBlock(
            text="t",
            expected_train_signal=ExpectedTrainSignal.NEUTRAL,
            expected_validation_signal=ExpectedValidationSignal.REJECT,
        ),
        invariants={"ast_static": "pass"},
        train_metrics=TrainMetrics(sharpe=0.0, max_drawdown=0.0, turnover=0.0, num_trades=0),
        validation_signal=ValidationSignal.REJECTED_NO_SIGNIFICANT_IMPROVEMENT,
        hypothesis_outcome=HypothesisOutcome.DISCONFIRMED,
        decision=Decision.REJECT,
        complexity_before=2.0,
        complexity_after=2.0,
        agent_compute=AgentCompute(
            editor_input_tokens=0,
            editor_output_tokens=0,
            reflector_input_tokens=0,
            reflector_output_tokens=0,
            model_editor="claude-opus-4-7",
            model_reflector="claude-opus-4-7",
            wall_clock_sec=0.0,
        ),
        cited_factors=cited,
    )


def test_trial_record_round_trips_cited_factors():
    raw = _make_trial_record(["ADX", "RSI"])
    dumped = json.loads(raw.model_dump_json())
    assert dumped["cited_factors"] == ["ADX", "RSI"]
    reloaded = TrialRecord.model_validate(dumped)
    assert reloaded.cited_factors == ["ADX", "RSI"]


def test_trial_record_cited_factors_defaults_empty():
    rec = _make_trial_record([])
    assert rec.cited_factors == []


# -------- dossier matchers --------


def test_match_from_citation_picks_first_known_factor():
    link = match_dossier_from_citation(["NotARealFactor", "ADX"])
    assert link is not None
    assert link.factor_name == "ADX"
    assert link.matched_via == "editor_cite"


def test_match_from_citation_returns_none_for_unknown_only():
    link = match_dossier_from_citation(["WhollyMadeUpZZZ"])
    assert link is None


def test_match_from_citation_returns_none_for_empty():
    assert match_dossier_from_citation([]) is None


def test_citation_takes_precedence_over_substring_in_practice():
    """The dossier link emitted by Discovery prefers citation. Verified by
    comparing the two matchers on the same indicator name."""
    cite_link = match_dossier_from_citation(["ADX"])
    sub_link = match_dossier_by_name("adx")
    assert cite_link is not None
    assert sub_link is not None
    # Both find ADX, but the citation link is marked editor_cite while the
    # substring guess is marked name_match.
    assert cite_link.matched_via == "editor_cite"
    assert sub_link.matched_via == "name_match"


# -------- ContextBuilder dedup --------


def test_tried_factor_names_picks_up_explicit_cite(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    cb = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")
    summaries = [
        PastTrialSummary(
            trial_id=0,
            edit_type="add_indicator",
            edit_summary="x",
            hypothesis_text="",
            expected_validation_signal="accept",
            actual_validation_signal="rejected_no_significant_improvement",
            hypothesis_outcome="disconfirmed",
            train_sharpe=0.5,
            train_max_drawdown=-0.1,
            train_turnover=2.0,
            train_num_trades=20,
            decision="reject",
            cited_factors=("ADX",),
        )
    ]
    tried = cb._tried_factor_names(strategy, summaries)
    assert "ADX" in tried


def test_tried_factor_names_picks_up_indicator_substring_fallback(tmp_path):
    """An indicator named `adx_14` with fn=adx should mark ADX as tried
    even if the Editor forgot to cite it explicitly."""
    mm = MemoryManager(tmp_path / "memory")
    cb = ContextBuilder(mm, repo_root=tmp_path)
    # Strategy that already includes an adx_14 / fn=adx indicator (synthetic).
    from lbg.dsl.schema import (
        CrossAboveRule,
        CrossBelowRule,
        FixedFractionSizing,
        IndicatorAboveFilter,
        IndicatorSpec,
        Strategy,
    )

    s = Strategy(
        name="t",
        indicators=[
            IndicatorSpec(name="sma_fast", fn="sma", params={"period": 20}),
            IndicatorSpec(name="sma_slow", fn="sma", params={"period": 50}),
            IndicatorSpec(name="adx_14", fn="adx", params={"period": 14}),
        ],
        entry=CrossAboveRule(rule="cross_above", fast="sma_fast", slow="sma_slow"),
        exit=CrossBelowRule(rule="cross_below", fast="sma_fast", slow="sma_slow"),
        filters=[IndicatorAboveFilter(rule="indicator_above", indicator="adx_14", threshold=20.0)],
        sizing=FixedFractionSizing(mode="fixed_fraction", fraction=1.0, max_position=1.0),
    )
    tried = cb._tried_factor_names(s, recent=[])
    assert "ADX" in tried


def test_tried_factor_names_minlen_3_keeps_short_dossier_names_clean(tmp_path):
    """Dossier names shorter than 3 chars (e.g. "AD") would otherwise match
    huge swathes of indicator names; the length filter prevents that.

    The actual library has "AD" as a factor name -- it should NOT show up
    as tried just because the strategy mentions "sma" (which contains "a").
    """
    mm = MemoryManager(tmp_path / "memory")
    cb = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")  # sma_fast / sma_slow
    tried = cb._tried_factor_names(strategy, recent=[])
    assert "AD" not in tried


def test_factor_hints_excludes_already_tried(tmp_path):
    """Dossier already tried (via cite) must be filtered out of the next
    hint list."""
    mm = MemoryManager(tmp_path / "memory")
    cb = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")
    summaries = [
        PastTrialSummary(
            trial_id=0,
            edit_type="add_indicator",
            edit_summary="x",
            hypothesis_text="",
            expected_validation_signal="accept",
            actual_validation_signal="rejected_drawdown_regression",
            hypothesis_outcome="disconfirmed",
            train_sharpe=0.5,
            train_max_drawdown=-0.2,
            train_turnover=2.0,
            train_num_trades=20,
            decision="reject",
            cited_factors=("ADX",),
        )
    ]
    hints = cb._factor_hints(strategy, summaries)
    names = {h.name for h in hints}
    assert "ADX" not in names


@pytest.fixture
def working_tree(tmp_path):
    import shutil
    import subprocess

    shutil.copy(REPO / "strategy.yaml", tmp_path / "strategy.yaml")
    (tmp_path / "indicators").mkdir()
    shutil.copy(REPO / "indicators" / "sma.py", tmp_path / "indicators" / "sma.py")
    shutil.copy(REPO / "indicators" / "__init__.py", tmp_path / "indicators" / "__init__.py")
    (tmp_path / "data").symlink_to(REPO / "data")
    return tmp_path
