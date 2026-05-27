"""Tests for `lbg.orchestrator.ContextBuilder.editor_view`."""

from __future__ import annotations

from pathlib import Path

from lbg.dsl import load_strategy
from lbg.memory import MemoryManager
from lbg.orchestrator.context_builder import (
    SEMANTIC_MEMORY_FILES,
    ContextBuilder,
    PastTrialSummary,
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


def _trial(trial_id: int, signal: ValidationSignal, outcome: HypothesisOutcome) -> TrialRecord:
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
            summary=f"period change in trial {trial_id}",
        ),
        hypothesis=HypothesisBlock(
            text="testing whether a different lookback helps",
            expected_train_signal=ExpectedTrainSignal.MILD_IMPROVEMENT,
            expected_validation_signal=ExpectedValidationSignal.ACCEPT,
        ),
        invariants={"ast_static": "pass"},
        train_metrics=TrainMetrics(
            sharpe=0.7,
            max_drawdown=-0.15,
            turnover=3.0,
            num_trades=30,
        ),
        validation_signal=signal,
        hypothesis_outcome=outcome,
        decision=Decision.ACCEPT if signal == ValidationSignal.ACCEPTED else Decision.REJECT,
        complexity_before=2.0,
        complexity_after=2.2,
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


# -------- shape --------


def test_editor_view_returns_strategy_yaml(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    builder = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")
    ctx = builder.editor_view(strategy)
    # Round-tripped strategy contains its name and indicators.
    assert "sma_cross_baseline" in ctx.strategy_yaml
    assert "sma_fast" in ctx.strategy_yaml
    assert "sma_slow" in ctx.strategy_yaml


def test_editor_view_includes_current_indicator_source(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    builder = ContextBuilder(mm, repo_root=REPO)
    strategy = load_strategy(REPO / "strategy.yaml")
    ctx = builder.editor_view(strategy)
    assert ctx.indicator_code
    assert ctx.indicator_code[0].fn == "sma"
    assert "def sma" in ctx.indicator_code[0].source_excerpt
    assert scan_for_leaks(ctx.indicator_code[0].source_excerpt) == []


def test_editor_view_pulls_recent_trials(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    mm.append_trial(_trial(0, ValidationSignal.ACCEPTED, HypothesisOutcome.CONFIRMED))
    mm.append_trial(
        _trial(1, ValidationSignal.REJECTED_DRAWDOWN_REGRESSION, HypothesisOutcome.DISCONFIRMED)
    )
    builder = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")
    ctx = builder.editor_view(strategy)
    assert len(ctx.recent_trials) == 2
    assert ctx.recent_trials[0].trial_id == 0
    assert ctx.recent_trials[1].trial_id == 1
    assert ctx.recent_trials[1].actual_validation_signal == "rejected_drawdown_regression"


def test_recent_trials_limit_respected(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    for i in range(15):
        mm.append_trial(_trial(i, ValidationSignal.ACCEPTED, HypothesisOutcome.CONFIRMED))
    builder = ContextBuilder(mm, repo_root=tmp_path, recent_trials_limit=5)
    strategy = load_strategy(REPO / "strategy.yaml")
    ctx = builder.editor_view(strategy)
    assert len(ctx.recent_trials) == 5
    assert [t.trial_id for t in ctx.recent_trials] == [10, 11, 12, 13, 14]


def test_semantic_memory_loaded_from_disk(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    (mm.memory_dir / "accepted_rules.md").write_text("- volatility_target works in high vol\n")
    (mm.memory_dir / "do_not_repeat.md").write_text("- never widen target_vol past 0.20\n")
    builder = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")
    ctx = builder.editor_view(strategy)
    assert "accepted_rules.md" in ctx.semantic_memory
    assert "do_not_repeat.md" in ctx.semantic_memory
    assert "volatility_target" in ctx.semantic_memory["accepted_rules.md"]


def test_skills_placeholder_is_empty(tmp_path):
    """Skills are wired by SkillManager in step 12; for now the list is empty."""
    mm = MemoryManager(tmp_path / "memory")
    builder = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")
    ctx = builder.editor_view(strategy)
    assert ctx.skills == []


def test_past_trial_summary_omits_validation_metrics():
    """The summary type does not even *have* a `val_sharpe` field -- structural defense."""
    fields = {f for f in PastTrialSummary.__dataclass_fields__}
    forbidden = {"val_sharpe", "val_max_drawdown", "val_turnover", "validation_sharpe"}
    assert not (fields & forbidden)
    # And only categorical signal is exposed.
    assert "actual_validation_signal" in fields
    assert "expected_validation_signal" in fields


def test_semantic_memory_files_match_proposal():
    """PROPOSAL.html §4 lists exactly these four documents."""
    assert set(SEMANTIC_MEMORY_FILES) == {
        "accepted_rules.md",
        "failed_directions.md",
        "open_questions.md",
        "do_not_repeat.md",
    }


# -------- structural redaction: rendered context excludes calendar tokens --------


def test_rendered_strategy_yaml_passes_redaction(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    builder = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")
    ctx = builder.editor_view(strategy)
    # The strategy.yaml does not contain any year-like or event tokens, so a
    # full leak scan on the assembled view should return empty.
    findings = scan_for_leaks(ctx.strategy_yaml)
    assert findings == [], findings


def test_recent_trials_summary_text_passes_redaction(tmp_path):
    """Round-tripped trial summaries (after we render them to prompt text)
    must not introduce leaks even when train metrics are included."""
    mm = MemoryManager(tmp_path / "memory")
    mm.append_trial(_trial(0, ValidationSignal.ACCEPTED, HypothesisOutcome.CONFIRMED))
    builder = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")
    ctx = builder.editor_view(strategy)

    from lbg.orchestrator.role_runner import _format_recent_trials

    text = _format_recent_trials(ctx.recent_trials)
    assert scan_for_leaks(text) == [], text
