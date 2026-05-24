"""Tests for `RoleRunner.reflector` + `MemoryManager.append_to_md`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from lbg.dsl import load_strategy
from lbg.memory import MemoryManager
from lbg.orchestrator import (
    ContextBuilder,
    RedactionError,
    RoleRunner,
    RoleRunnerError,
)
from lbg.schemas import (
    EditProposal,
    HypothesisOutcome,
    TrainMetrics,
    ValidationSignal,
)

REPO = Path(__file__).resolve().parents[1]


# -------- fakes --------


@dataclass
class _FakeUsage:
    input_tokens: int = 80
    output_tokens: int = 60


@dataclass
class _FakeBlock:
    text: str


@dataclass
class _FakeMessage:
    content: list[_FakeBlock]
    usage: _FakeUsage


class _FakeClient:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.call_count = 0
        self.last_user = None
        self.messages = self

    def create(self, *, model, max_tokens, system, messages):
        self.call_count += 1
        self.last_user = messages[0]["content"]
        return _FakeMessage(
            content=[_FakeBlock(text=self.response_text)],
            usage=_FakeUsage(),
        )


def _proposal() -> EditProposal:
    return EditProposal.model_validate(
        {
            "trial_id": 1,
            "hypothesis": "Switching to volatility_target should suppress drawdowns",
            "proposed_edit": {
                "type": "change_sizing_mode",
                "change": {
                    "sizing": {
                        "mode": "volatility_target",
                        "target_vol": 0.12,
                        "max_position": 1.0,
                        "vol_lookback": 20,
                    }
                },
            },
            "expected_train_signal": "mild_improvement",
            "expected_validation_signal": "accept",
        }
    )


def _build_context(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    builder = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")
    return mm, builder.editor_view(strategy)


def _train_metrics() -> TrainMetrics:
    return TrainMetrics(sharpe=0.72, max_drawdown=-0.18, turnover=4.2, num_trades=44)


# -------- happy path --------


_VALID_REFLECTION = """\
```yaml
explanation: |
  The volatility_target sizing reduced exposure during higher-vol stretches,
  which kept the candidate from breaching the drawdown_regression branch of
  the gate. Pareto improvement on drawdown carried the acceptance.
accepted_rules_updates:
  - "volatility_target sizing helps drawdown when realized vol elevates"
failed_directions_updates: []
open_questions_updates:
  - "would a smaller vol_lookback exaggerate the effect?"
do_not_repeat_updates: []
```
"""


def test_reflector_parses_canonical_response(tmp_path):
    mm, ctx = _build_context(tmp_path)
    client = _FakeClient(_VALID_REFLECTION)
    runner = RoleRunner(client=client)

    result = runner.reflector(
        trial_id=1,
        editor_context=ctx,
        proposal=_proposal(),
        hypothesis_outcome=HypothesisOutcome.CONFIRMED,
        actual_validation_signal=ValidationSignal.ACCEPTED,
        actual_train_metrics=_train_metrics(),
    )

    assert client.call_count == 1
    assert result.record.trial_id == 1
    # Reflector cannot overwrite the mechanically computed outcome.
    assert result.record.hypothesis_outcome == HypothesisOutcome.CONFIRMED
    assert len(result.record.accepted_rules_updates) == 1
    assert "volatility_target" in result.record.accepted_rules_updates[0]
    assert result.record.failed_directions_updates == []
    assert result.compute.role == "reflector"


def test_reflector_user_prompt_contains_mechanical_outcome(tmp_path):
    mm, ctx = _build_context(tmp_path)
    client = _FakeClient(_VALID_REFLECTION)
    runner = RoleRunner(client=client)
    runner.reflector(
        trial_id=1,
        editor_context=ctx,
        proposal=_proposal(),
        hypothesis_outcome=HypothesisOutcome.DISCONFIRMED,
        actual_validation_signal=ValidationSignal.REJECTED_DRAWDOWN_REGRESSION,
        actual_train_metrics=_train_metrics(),
    )
    assert "disconfirmed" in client.last_user
    assert "rejected_drawdown_regression" in client.last_user
    assert "do not re-judge" in client.last_user


def test_reflector_injects_machine_outcome_even_if_llm_omits_it(tmp_path):
    """The LLM's payload schema doesn't have hypothesis_outcome; we inject
    the machine-computed one. Disconfirmed input -> disconfirmed record."""
    mm, ctx = _build_context(tmp_path)
    client = _FakeClient(_VALID_REFLECTION)
    runner = RoleRunner(client=client)
    result = runner.reflector(
        trial_id=2,
        editor_context=ctx,
        proposal=_proposal(),
        hypothesis_outcome=HypothesisOutcome.DISCONFIRMED,
        actual_validation_signal=ValidationSignal.REJECTED_DRAWDOWN_REGRESSION,
        actual_train_metrics=_train_metrics(),
    )
    assert result.record.hypothesis_outcome == HypothesisOutcome.DISCONFIRMED


# -------- redaction --------


def test_reflector_rejects_year_in_response(tmp_path):
    bad_response = """\
```yaml
explanation: |
  After the 2020 drawdown the strategy underperformed.
accepted_rules_updates: []
failed_directions_updates: []
open_questions_updates: []
do_not_repeat_updates: []
```
"""
    mm, ctx = _build_context(tmp_path)
    client = _FakeClient(bad_response)
    runner = RoleRunner(client=client)
    with pytest.raises(RedactionError):
        runner.reflector(
            trial_id=1,
            editor_context=ctx,
            proposal=_proposal(),
            hypothesis_outcome=HypothesisOutcome.CONFIRMED,
            actual_validation_signal=ValidationSignal.ACCEPTED,
            actual_train_metrics=_train_metrics(),
        )


# -------- parse failure --------


def test_reflector_rejects_missing_explanation(tmp_path):
    """`explanation` is required by ReflectorOutputPayload."""
    bad_response = """\
```yaml
accepted_rules_updates: []
failed_directions_updates: []
open_questions_updates: []
do_not_repeat_updates: []
```
"""
    mm, ctx = _build_context(tmp_path)
    client = _FakeClient(bad_response)
    runner = RoleRunner(client=client)
    with pytest.raises(RoleRunnerError, match="did not parse"):
        runner.reflector(
            trial_id=1,
            editor_context=ctx,
            proposal=_proposal(),
            hypothesis_outcome=HypothesisOutcome.CONFIRMED,
            actual_validation_signal=ValidationSignal.ACCEPTED,
            actual_train_metrics=_train_metrics(),
        )


def test_reflector_rejects_extra_field(tmp_path):
    """`extra='forbid'` on ReflectorOutputPayload."""
    bad_response = """\
```yaml
explanation: ok
accepted_rules_updates: []
failed_directions_updates: []
open_questions_updates: []
do_not_repeat_updates: []
secret_field: leaked_value
```
"""
    mm, ctx = _build_context(tmp_path)
    client = _FakeClient(bad_response)
    runner = RoleRunner(client=client)
    with pytest.raises(RoleRunnerError, match="did not parse"):
        runner.reflector(
            trial_id=1,
            editor_context=ctx,
            proposal=_proposal(),
            hypothesis_outcome=HypothesisOutcome.CONFIRMED,
            actual_validation_signal=ValidationSignal.ACCEPTED,
            actual_train_metrics=_train_metrics(),
        )


# -------- semantic memory append --------


def test_memory_manager_append_to_md_writes_bullets(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    written = mm.append_to_md("accepted_rules.md", ["rule A", "rule B"])
    assert written == 2
    content = mm.read_md("accepted_rules.md")
    assert content == "- rule A\n- rule B\n"


def test_memory_manager_append_to_md_is_idempotent_append(tmp_path):
    """Two appends accumulate; the file is never overwritten."""
    mm = MemoryManager(tmp_path / "memory")
    mm.append_to_md("open_questions.md", ["q1"])
    mm.append_to_md("open_questions.md", ["q2"])
    assert mm.read_md("open_questions.md") == "- q1\n- q2\n"


def test_memory_manager_append_to_md_skips_empty_items(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    written = mm.append_to_md("failed_directions.md", ["", "  ", "real item"])
    assert written == 1
    assert mm.read_md("failed_directions.md") == "- real item\n"


def test_memory_manager_append_to_md_rejects_unknown_file(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    with pytest.raises(ValueError, match="unknown semantic memory file"):
        mm.append_to_md("notes.md", ["x"])


def test_memory_manager_read_md_returns_empty_for_missing(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    assert mm.read_md("accepted_rules.md") == ""


# -------- end-to-end: reflector record -> md updates --------


def test_reflection_updates_flow_through_to_memory(tmp_path):
    mm, ctx = _build_context(tmp_path)
    client = _FakeClient(_VALID_REFLECTION)
    runner = RoleRunner(client=client)
    result = runner.reflector(
        trial_id=1,
        editor_context=ctx,
        proposal=_proposal(),
        hypothesis_outcome=HypothesisOutcome.CONFIRMED,
        actual_validation_signal=ValidationSignal.ACCEPTED,
        actual_train_metrics=_train_metrics(),
    )

    # Persist the record + apply the bullets.
    mm.append_reflection(result.record)
    mm.append_to_md("accepted_rules.md", result.record.accepted_rules_updates)
    mm.append_to_md("failed_directions.md", result.record.failed_directions_updates)
    mm.append_to_md("open_questions.md", result.record.open_questions_updates)
    mm.append_to_md("do_not_repeat.md", result.record.do_not_repeat_updates)

    reflections = mm.read_reflections()
    assert len(reflections) == 1
    assert "volatility_target" in mm.read_md("accepted_rules.md")
    assert "vol_lookback" in mm.read_md("open_questions.md")
    assert mm.read_md("failed_directions.md") == ""
    assert mm.read_md("do_not_repeat.md") == ""


# -------- live --------


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; skipping live Reflector call",
)
def test_reflector_live_one_call(tmp_path):
    """One real Reflector call. Confirms the system prompt + parsing fit
    together with the live model, on a synthetic but coherent trial."""
    mm, ctx = _build_context(tmp_path)
    runner = RoleRunner()
    result = runner.reflector(
        trial_id=1,
        editor_context=ctx,
        proposal=_proposal(),
        hypothesis_outcome=HypothesisOutcome.PARTIALLY_CONFIRMED,
        actual_validation_signal=ValidationSignal.REJECTED_DRAWDOWN_REGRESSION,
        actual_train_metrics=_train_metrics(),
    )
    assert result.record.trial_id == 1
    assert result.record.hypothesis_outcome == HypothesisOutcome.PARTIALLY_CONFIRMED
    assert len(result.record.explanation) > 20  # non-trivial explanation
    assert result.compute.role == "reflector"
    assert result.compute.input_tokens > 0
    assert result.compute.output_tokens > 0
