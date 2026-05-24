"""Tests for `lbg.memory.MemoryManager`.

Cover append + read round-trip for all four streams, next-trial-id math,
and the append-only contract (existing lines must survive new writes).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from lbg.memory import (
    AgentComputeRecord,
    InvariantFailureRecord,
    MemoryManager,
    ReflectionRecord,
)
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


def _trial_record(trial_id: int = 0) -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        parent_commit="abc123",
        candidate_commit="def456",
        role_outputs=RoleOutputs(
            editor_output_path=f"runs/{trial_id:04d}/editor.yaml",
            reflector_output_path=f"runs/{trial_id:04d}/reflector.yaml",
        ),
        edit=EditSummary(
            type=EditType.CHANGE_SIZING_MODE,
            target="strategy.yaml",
            summary="fixed_fraction -> volatility_target",
        ),
        hypothesis=HypothesisBlock(
            text="reduce vol exposure during high-vol regimes",
            expected_train_signal=ExpectedTrainSignal.MILD_IMPROVEMENT,
            expected_validation_signal=ExpectedValidationSignal.ACCEPT,
        ),
        invariants={"ast_static": "pass", "dynamic_lookahead": "pass"},
        train_metrics=TrainMetrics(sharpe=0.91, max_drawdown=-0.13, turnover=0.24, num_trades=38),
        validation_signal=ValidationSignal.ACCEPTED,
        hypothesis_outcome=HypothesisOutcome.CONFIRMED,
        decision=Decision.ACCEPT,
        complexity_before=4.0,
        complexity_after=5.0,
        agent_compute=AgentCompute(
            editor_input_tokens=12000,
            editor_output_tokens=1300,
            reflector_input_tokens=4000,
            reflector_output_tokens=700,
            model_editor="claude-opus-4-7",
            model_reflector="claude-opus-4-7",
            wall_clock_sec=42.1,
        ),
    )


# -------- trials.jsonl --------


def test_append_and_read_single_trial(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    mm.append_trial(_trial_record(trial_id=0))
    out = mm.read_trials()
    assert len(out) == 1
    assert out[0].trial_id == 0


def test_read_empty_when_no_file(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    assert mm.read_trials() == []
    assert mm.read_reflections() == []
    assert mm.read_invariant_failures() == []
    assert mm.read_agent_compute() == []


def test_multiple_trials_preserve_order(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    for i in range(5):
        mm.append_trial(_trial_record(trial_id=i))
    out = mm.read_trials()
    assert [t.trial_id for t in out] == [0, 1, 2, 3, 4]


def test_next_trial_id_starts_at_zero(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    assert mm.next_trial_id() == 0


def test_next_trial_id_increments_past_max(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    mm.append_trial(_trial_record(trial_id=0))
    mm.append_trial(_trial_record(trial_id=1))
    mm.append_trial(_trial_record(trial_id=2))
    assert mm.next_trial_id() == 3


def test_next_trial_id_uses_max_not_count(tmp_path):
    """If trial_id 5 exists alone, next is 6 -- not 1."""
    mm = MemoryManager(tmp_path / "memory")
    mm.append_trial(_trial_record(trial_id=5))
    assert mm.next_trial_id() == 6


def test_append_does_not_truncate_existing(tmp_path):
    """Append-only contract: existing records must survive new writes."""
    mm = MemoryManager(tmp_path / "memory")
    mm.append_trial(_trial_record(trial_id=0))
    mm.append_trial(_trial_record(trial_id=1))
    raw = mm.trials_path.read_text()
    assert raw.count("\n") == 2
    assert "abc123" in raw  # parent_commit on both records


def test_jsonl_lines_are_singleline(tmp_path):
    """Every line in trials.jsonl is one valid JSON document (no multiline)."""
    import json

    mm = MemoryManager(tmp_path / "memory")
    mm.append_trial(_trial_record(trial_id=0))
    for line in mm.trials_path.read_text().splitlines():
        parsed = json.loads(line)
        assert parsed["trial_id"] == 0


def test_malformed_jsonl_line_raises_validation_error(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    mm.append_trial(_trial_record(trial_id=0))
    with mm.trials_path.open("a") as f:
        f.write('{"trial_id": "not_an_int"}\n')
    with pytest.raises(ValidationError):
        mm.read_trials()


# -------- reflections.jsonl --------


def test_reflection_round_trip(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    rec = ReflectionRecord(
        trial_id=7,
        explanation="The edit shifted sizing during the high-vol stretch; drawdown improved.",
        hypothesis_outcome=HypothesisOutcome.PARTIALLY_CONFIRMED,
        accepted_rules_updates=["volatility_target works better in high-vol periods"],
    )
    mm.append_reflection(rec)
    out = mm.read_reflections()
    assert len(out) == 1
    assert out[0].trial_id == 7
    assert out[0].hypothesis_outcome == HypothesisOutcome.PARTIALLY_CONFIRMED
    assert len(out[0].accepted_rules_updates) == 1


# -------- invariant_failures.jsonl --------


def test_invariant_failure_round_trip(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    rec = InvariantFailureRecord(
        trial_id=11,
        parent_commit="abc123",
        invariant_name="no_future_shift",
        message=".shift(-1) called",
        file="indicators/leaky.py",
        line=14,
    )
    mm.append_invariant_failure(rec)
    out = mm.read_invariant_failures()
    assert len(out) == 1
    assert out[0].invariant_name == "no_future_shift"


# -------- agent_compute.jsonl --------


def test_agent_compute_round_trip(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    rec = AgentComputeRecord(
        trial_id=3,
        role="editor",
        model="claude-opus-4-7",
        input_tokens=11500,
        output_tokens=1200,
        wall_clock_sec=8.3,
    )
    mm.append_agent_compute(rec)
    out = mm.read_agent_compute()
    assert len(out) == 1
    assert out[0].role == "editor"
    assert out[0].input_tokens == 11500


# -------- directory creation --------


def test_constructor_creates_memory_dir(tmp_path):
    target = tmp_path / "nested" / "memory"
    assert not target.parent.exists()
    target.parent.mkdir()
    MemoryManager(target)
    assert target.exists() and target.is_dir()
    # Subsequent constructor calls must not fail on an existing dir.
    MemoryManager(target)


def test_paths_are_named_per_proposal(tmp_path):
    mm = MemoryManager(tmp_path)
    assert mm.trials_path.name == "trials.jsonl"
    assert mm.reflections_path.name == "reflections.jsonl"
    assert mm.invariant_failures_path.name == "invariant_failures.jsonl"
    assert mm.agent_compute_path.name == "agent_compute.jsonl"


def test_can_read_back_files_not_written_through_manager(tmp_path):
    """A pre-existing memory/ directory with valid lines should be readable."""
    mem = tmp_path / "memory"
    mem.mkdir()
    rec = _trial_record(trial_id=0)
    (mem / "trials.jsonl").write_text(rec.model_dump_json() + "\n", encoding="utf-8")
    mm = MemoryManager(mem)
    assert len(mm.read_trials()) == 1


# -------- absolute path support --------


def test_manager_accepts_path_object_or_string(tmp_path):
    mm1 = MemoryManager(tmp_path / "mem1")
    mm2 = MemoryManager(str(tmp_path / "mem2"))
    assert mm1.trials_path.parent.exists()
    assert mm2.trials_path.parent.exists()
    assert isinstance(mm2.memory_dir, Path)
