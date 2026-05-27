"""Tests for the factor-knowledge-base hint pipeline.

ContextBuilder pulls a shortlist of dossier entries from `knowledge/factors/`
and presents them to the Editor under `EditorContext.factor_hints`. These
tests pin three properties:
  1. Search terms are derived from the current strategy + recent rejection
     signals (with the rejection-signal map giving zh keywords).
  2. The shortlist is deduplicated and respects `factor_hints_total_cap`.
  3. Year tokens that would trip the redaction guard are scrubbed in place.
"""

from __future__ import annotations

from pathlib import Path

from lbg.dsl import load_strategy
from lbg.knowledge.factors import search as factor_search
from lbg.memory import MemoryManager
from lbg.orchestrator.context_builder import (
    ContextBuilder,
    PastTrialSummary,
    _scrub_years,
    _truncate,
)
from lbg.orchestrator.redaction import scan_for_leaks
from lbg.schemas import (
    AgentCompute,
    Decision,
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


def _trial(trial_id: int, signal: ValidationSignal) -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        parent_commit="aaaaaaa",
        candidate_commit="bbbbbbb",
        role_outputs=RoleOutputs(
            editor_output_path=f"runs/{trial_id:04d}/editor.yaml",
            reflector_output_path=None,
        ),
        edit=EditSummary(
            type=EditType.PARAMETER_CHANGE,
            target="strategy.yaml",
            summary=f"x{trial_id}",
        ),
        hypothesis=HypothesisBlock(
            text="placeholder",
            expected_train_signal=ExpectedTrainSignal.MILD_IMPROVEMENT,
            expected_validation_signal=ExpectedValidationSignal.ACCEPT,
        ),
        invariants={"ast_static": "pass"},
        train_metrics=TrainMetrics(sharpe=0.5, max_drawdown=-0.2, turnover=2.0, num_trades=20),
        validation_signal=signal,
        hypothesis_outcome=HypothesisOutcome.DISCONFIRMED,
        decision=(Decision.ACCEPT if signal == ValidationSignal.ACCEPTED else Decision.REJECT),
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
    )


# -------- search-term derivation --------


def test_derive_search_terms_includes_indicator_fn_and_name(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    cb = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")  # sma_fast, sma_slow
    terms = cb._derive_search_terms(strategy, recent=[])
    # fn 'sma' once (it appears twice in the strategy but is deduped); both
    # specific names also appear since they differ from the fn.
    assert "sma" in terms
    assert "sma_fast" in terms
    assert "sma_slow" in terms
    # No rejection history -> no Chinese rejection terms.
    assert "波动率" not in terms


def test_derive_search_terms_adds_zh_keywords_per_rejection_signal(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    cb = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")

    recent_summaries = [
        PastTrialSummary(
            trial_id=0,
            edit_type="parameter_change",
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
        ),
        PastTrialSummary(
            trial_id=1,
            edit_type="parameter_change",
            edit_summary="x",
            hypothesis_text="",
            expected_validation_signal="accept",
            actual_validation_signal="rejected_turnover",
            hypothesis_outcome="disconfirmed",
            train_sharpe=0.5,
            train_max_drawdown=-0.2,
            train_turnover=4.0,
            train_num_trades=20,
            decision="reject",
        ),
    ]
    terms = cb._derive_search_terms(strategy, recent_summaries)

    # Drawdown signal -> 波动率/回撤; turnover signal -> 趋势强度/状态识别.
    assert "波动率" in terms
    assert "回撤" in terms
    assert "趋势强度" in terms
    assert "状态识别" in terms


def test_derive_search_terms_skips_complexity_rejection(tmp_path):
    """REJECTED_COMPLEXITY should NOT add new search terms: the system is
    already too big and we don't want to surface more factors to add."""
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
            actual_validation_signal="rejected_complexity",
            hypothesis_outcome="disconfirmed",
            train_sharpe=0.5,
            train_max_drawdown=-0.2,
            train_turnover=2.0,
            train_num_trades=20,
            decision="reject",
        )
    ]
    terms = cb._derive_search_terms(strategy, summaries)
    # Only strategy-derived terms; no rejection-derived zh terms.
    rejection_zh = {"波动率", "回撤", "趋势强度", "状态识别", "动量", "均值回归"}
    assert not (set(terms) & rejection_zh)


def test_buyhold_cold_start_gets_risk_off_search_terms(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    cb = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "baselines" / "buyhold" / "strategy.yaml")
    terms = cb._derive_search_terms(strategy, recent=[])
    assert terms[:4] == ["回撤", "波动率", "趋势强度", "状态识别"]


# -------- shortlist properties --------


def test_factor_hints_capped_and_deduplicated(tmp_path):
    """The total-cap arg is respected; FactorHint.name appears at most once."""
    mm = MemoryManager(tmp_path / "memory")
    cb = ContextBuilder(
        mm,
        repo_root=tmp_path,
        factor_hints_per_term=3,
        factor_hints_total_cap=5,
    )
    strategy = load_strategy(REPO / "strategy.yaml")
    summaries = [
        PastTrialSummary(
            trial_id=i,
            edit_type="parameter_change",
            edit_summary="x",
            hypothesis_text="",
            expected_validation_signal="accept",
            actual_validation_signal=sig,
            hypothesis_outcome="disconfirmed",
            train_sharpe=0.5,
            train_max_drawdown=-0.2,
            train_turnover=2.0,
            train_num_trades=20,
            decision="reject",
        )
        for i, sig in enumerate(
            [
                "rejected_drawdown_regression",
                "rejected_no_significant_improvement",
                "rejected_turnover",
            ]
        )
    ]
    hints = cb._factor_hints(strategy, summaries)
    assert len(hints) <= 5
    names = [h.name for h in hints]
    assert len(set(names)) == len(names)  # deduped


def test_factor_search_prioritizes_exact_name_over_substring_noise():
    names = [hit["factor_name"] for hit in factor_search("sma", top_k=3)]
    assert names[0] == "SMA"


def test_factor_hints_returned_via_editor_view(tmp_path):
    """The hints surface on EditorContext.factor_hints, not as a side channel."""
    mm = MemoryManager(tmp_path / "memory")
    mm.append_trial(_trial(0, ValidationSignal.REJECTED_DRAWDOWN_REGRESSION))
    cb = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")
    view = cb.editor_view(strategy)
    assert view.factor_hints, "expected at least one hint with rejection history"
    # The drawdown rejection should pull in volatility / drawdown family
    # factors; ADX, ATR, and Calmar are all reasonable hits.
    names = {h.name for h in view.factor_hints}
    assert names & {"ADX", "ATR", "CalmarRatio", "MaxDrawdownDuration"}


def test_factor_hints_one_line_truncated_below_cap(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    cb = ContextBuilder(
        mm,
        repo_root=tmp_path,
        factor_one_line_max_chars=80,
    )
    strategy = load_strategy(REPO / "strategy.yaml")
    hints = cb._factor_hints(strategy, recent=[])
    for h in hints:
        assert len(h.one_line) <= 80


# -------- year scrubbing --------


def test_scrub_years_replaces_in_window():
    out = _scrub_years("Aroon was introduced in 1995 by Tushar Chande")
    assert "1995" not in out
    assert "<YEAR>" in out


def test_scrub_years_leaves_pre_1990_alone():
    """1978 is outside the redaction window so it never trips the guard;
    keeping the original is informative for the agent."""
    out = _scrub_years("Welles Wilder, 1978")
    assert "1978" in out


def test_scrubbed_dossier_text_passes_redaction_guard(tmp_path):
    """End-to-end: a dossier mentioning 1995/2002 must not produce leaks
    once routed through ContextBuilder."""
    mm = MemoryManager(tmp_path / "memory")
    mm.append_trial(_trial(0, ValidationSignal.REJECTED_DRAWDOWN_REGRESSION))
    cb = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")
    hints = cb._factor_hints(strategy, recent=cb._recent_trial_summaries())
    for h in hints:
        leaks = scan_for_leaks(h.one_line) + scan_for_leaks(h.category)
        assert not leaks, f"hint {h.name} leaked: {leaks}"


def test_truncate_under_cap_is_identity():
    assert _truncate("hello", 10) == "hello"


def test_truncate_over_cap_adds_ellipsis():
    out = _truncate("x" * 50, 10)
    assert len(out) == 10
    assert out.endswith("…")
