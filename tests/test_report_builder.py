"""Tests for `lbg.orchestrator.report_builder.ReportBuilder`."""

from __future__ import annotations

from lbg.memory import MemoryManager
from lbg.memory.records import AgentComputeRecord, ReflectionRecord
from lbg.orchestrator.report_builder import ReportBuilder
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
from lbg.sealed_vault import SealedVault


def _trial(trial_id: int, signal=ValidationSignal.ACCEPTED) -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        parent_commit="aaaaaaa",
        candidate_commit="bbbbbbb",
        role_outputs=RoleOutputs(
            editor_output_path=f"runs/{trial_id:04d}/editor.yaml",
            reflector_output_path=f"runs/{trial_id:04d}/reflector.yaml",
        ),
        edit=EditSummary(
            type=EditType.PARAMETER_CHANGE,
            target="strategy.yaml",
            summary=f"trial {trial_id} edit",
        ),
        hypothesis=HypothesisBlock(
            text="test",
            expected_train_signal=ExpectedTrainSignal.MILD_IMPROVEMENT,
            expected_validation_signal=ExpectedValidationSignal.ACCEPT,
        ),
        invariants={"ast_static": "pass"},
        train_metrics=TrainMetrics(sharpe=0.72, max_drawdown=-0.18, turnover=3.5, num_trades=40),
        validation_signal=signal,
        hypothesis_outcome=HypothesisOutcome.CONFIRMED,
        decision=Decision.ACCEPT if signal == ValidationSignal.ACCEPTED else Decision.REJECT,
        complexity_before=2.0,
        complexity_after=2.2,
        agent_compute=AgentCompute(
            editor_input_tokens=1000,
            editor_output_tokens=500,
            reflector_input_tokens=600,
            reflector_output_tokens=200,
            model_editor="claude-opus-4-7",
            model_reflector="claude-opus-4-7",
            wall_clock_sec=12.0,
        ),
    )


def test_report_renders_empty_memory(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    report_path = tmp_path / "report.html"
    out = ReportBuilder(mm, vault=None, output_path=report_path).build()
    assert out == report_path
    text = report_path.read_text()
    assert "LBG-Trader" in text
    assert "Total trials:</strong> 0" in text
    assert "No trials recorded" in text


def test_report_includes_per_trial_rows(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    mm.append_trial(_trial(0, ValidationSignal.ACCEPTED))
    mm.append_trial(_trial(1, ValidationSignal.REJECTED_DRAWDOWN_REGRESSION))
    mm.append_reflection(
        ReflectionRecord(
            trial_id=0,
            explanation="trial 0 worked because of trend filter",
            hypothesis_outcome=HypothesisOutcome.CONFIRMED,
        )
    )
    report_path = tmp_path / "report.html"
    ReportBuilder(mm, output_path=report_path).build()
    text = report_path.read_text()
    assert "trial 0 edit" in text
    assert "trial 1 edit" in text
    assert "rejected_drawdown_regression" in text
    assert "trend filter" in text


def test_report_includes_sealed_payload_when_present(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    vault = SealedVault(tmp_path / "sealed.json")
    vault.open_and_write({"sharpe": 0.91, "max_drawdown": -0.13})
    report_path = tmp_path / "report.html"
    ReportBuilder(mm, vault=vault, output_path=report_path).build()
    text = report_path.read_text()
    assert "sealed_at_utc" in text
    assert "0.91" in text


def test_report_says_not_sealed_when_vault_empty(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    vault = SealedVault(tmp_path / "sealed.json")
    report_path = tmp_path / "report.html"
    ReportBuilder(mm, vault=vault, output_path=report_path).build()
    text = report_path.read_text()
    assert "not yet sealed" in text


def test_report_counts_input_output_tokens(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    mm.append_agent_compute(
        AgentComputeRecord(
            trial_id=0,
            role="editor",
            model="claude-opus-4-7",
            input_tokens=1000,
            output_tokens=300,
            wall_clock_sec=5.0,
        )
    )
    mm.append_agent_compute(
        AgentComputeRecord(
            trial_id=0,
            role="reflector",
            model="claude-opus-4-7",
            input_tokens=600,
            output_tokens=200,
            wall_clock_sec=4.0,
        )
    )
    report_path = tmp_path / "report.html"
    ReportBuilder(mm, output_path=report_path).build()
    text = report_path.read_text()
    assert "Input tokens:</strong> 1600" in text
    assert "Output tokens:</strong> 500" in text


def test_memory_section_lists_all_four_files(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    mm.append_to_md("accepted_rules.md", ["rule X"])
    report_path = tmp_path / "report.html"
    ReportBuilder(mm, output_path=report_path).build()
    text = report_path.read_text()
    for name in (
        "accepted_rules.md",
        "failed_directions.md",
        "open_questions.md",
        "do_not_repeat.md",
    ):
        assert name in text
    assert "rule X" in text
