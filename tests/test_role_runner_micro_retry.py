"""Tests for Editor micro-retry inside `RoleRunner.editor`.

STAGE1_REPORT § 7 observed that ~12 of 20 MIMO trials died on missing-YAML
parser aborts. The retry loop lets the model self-correct in-call (multi
-turn) before we burn the trial slot. These tests pin the protocol:

  * Happy path: a recoverable failure on attempt 1 succeeds on attempt 2,
    with retry_attempts==2 and retry_reasons recording the kind.
  * Exhaustion: 3 consecutive bad responses raise RoleRunnerError with the
    attempt count surfaced.
  * Redaction-leak retries never replay the leaked assistant turn -- the
    corrective message must keep the leak out of the model's own context.
  * Each retry reason maps to its own canonical label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from lbg.dsl import load_strategy
from lbg.memory import MemoryManager
from lbg.orchestrator import ContextBuilder, RoleRunner, RoleRunnerError

REPO = Path(__file__).resolve().parents[1]


# ---- fake client that returns a *sequence* of responses ----


@dataclass
class _FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class _FakeBlock:
    text: str


@dataclass
class _FakeMessage:
    content: list[_FakeBlock]
    usage: _FakeUsage


class _SequenceFakeClient:
    """Returns response_texts[0], then response_texts[1], etc. Records the
    full messages list passed to each .create() so tests can pin the
    multi-turn correction structure."""

    def __init__(self, response_texts: list[str]):
        self.response_texts = list(response_texts)
        self.call_count = 0
        self.messages_log: list[list[dict[str, str]]] = []
        self.messages = self

    def create(self, *, model, max_tokens, system, messages):  # noqa: ARG002
        idx = min(self.call_count, len(self.response_texts) - 1)
        self.messages_log.append(list(messages))
        self.call_count += 1
        text = self.response_texts[idx]
        return _FakeMessage(
            content=[_FakeBlock(text=text)],
            usage=_FakeUsage(input_tokens=10, output_tokens=20),
        )


def _build_context(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    builder = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")
    return builder.editor_view(strategy)


_VALID = """\
```yaml
trial_id: 1
hypothesis: |
  Tighten sizing to defend against the next drawdown burst.
proposed_edit:
  type: parameter_change
  change:
    path: sizing.fraction
    value: 0.5
expected_train_signal: mild_improvement
expected_validation_signal: accept
fallback_if_rejected: |
  Try a sizing mode change instead.
```
"""


# ---- happy path: recoverable single failure then success ----


def test_missing_yaml_block_recovers_on_retry(tmp_path):
    no_yaml = "I cannot use YAML formatting today, here's prose only."
    client = _SequenceFakeClient([no_yaml, _VALID])
    runner = RoleRunner(client=client)
    ctx = _build_context(tmp_path)

    result = runner.editor(ctx, trial_id=1)

    assert client.call_count == 2
    assert result.compute.retry_attempts == 2
    assert result.compute.retry_reasons == ["missing_yaml"]
    # Tokens accumulated across both attempts -- accounting must not under-
    # report the true cost of the trial.
    assert result.compute.input_tokens == 20
    assert result.compute.output_tokens == 40


def test_parse_error_recovers_on_retry(tmp_path):
    """Schema-violating YAML on attempt 1, valid on attempt 2."""
    bad_schema = """\
```yaml
trial_id: 1
hypothesis: test
proposed_edit:
  type: parameter_change
  change:
    # wrong key: parser_expects `path` and `value`
    target_path: sizing.fraction
    target_value: 0.5
expected_train_signal: mild_improvement
expected_validation_signal: accept
fallback_if_rejected: nothing
```
"""
    client = _SequenceFakeClient([bad_schema, _VALID])
    runner = RoleRunner(client=client)
    ctx = _build_context(tmp_path)

    result = runner.editor(ctx, trial_id=1)
    assert result.compute.retry_attempts == 2
    assert result.compute.retry_reasons == ["parse_error"]


def test_redaction_leak_recovers_on_retry(tmp_path):
    leaky = """\
```yaml
trial_id: 1
hypothesis: |
  The 2020 covid drawdown suggests we need volatility protection.
proposed_edit:
  type: parameter_change
  change:
    path: sizing.fraction
    value: 0.5
expected_train_signal: neutral
expected_validation_signal: accept
fallback_if_rejected: try something else
```
"""
    client = _SequenceFakeClient([leaky, _VALID])
    runner = RoleRunner(client=client)
    ctx = _build_context(tmp_path)

    result = runner.editor(ctx, trial_id=1)
    assert result.compute.retry_attempts == 2
    assert result.compute.retry_reasons == ["redaction"]


def test_redaction_retry_does_not_replay_leaked_assistant_turn(tmp_path):
    """The leaked response must not appear as an assistant message in the
    follow-up call -- replaying it would re-introduce '2020 covid' into the
    model's own context window."""
    leaky = """\
```yaml
trial_id: 1
hypothesis: |
  The 2020 covid drawdown suggests volatility protection.
proposed_edit:
  type: parameter_change
  change: { path: sizing.fraction, value: 0.5 }
expected_train_signal: neutral
expected_validation_signal: accept
fallback_if_rejected: x
```
"""
    client = _SequenceFakeClient([leaky, _VALID])
    runner = RoleRunner(client=client)
    ctx = _build_context(tmp_path)
    runner.editor(ctx, trial_id=1)

    # Inspect the second call's messages: must be a single-turn user message
    # (no assistant role replayed); must not contain "2020" or "covid".
    second_call_messages = client.messages_log[1]
    assert all(m["role"] == "user" for m in second_call_messages), (
        f"redaction retry leaked assistant turn back: {second_call_messages}"
    )
    joined = "\n".join(m["content"] for m in second_call_messages)
    assert "2020" not in joined
    assert "covid" not in joined.lower()


def test_parse_error_retry_replays_assistant_turn(tmp_path):
    """Unlike redaction, a parse error is safe to replay -- the model
    benefits from seeing what it produced and the parser's complaint."""
    bad_schema = """\
```yaml
trial_id: 1
hypothesis: test
proposed_edit:
  type: parameter_change
  change:
    target_path: sizing.fraction
    target_value: 0.5
expected_train_signal: mild_improvement
expected_validation_signal: accept
fallback_if_rejected: x
```
"""
    client = _SequenceFakeClient([bad_schema, _VALID])
    runner = RoleRunner(client=client)
    ctx = _build_context(tmp_path)
    runner.editor(ctx, trial_id=1)

    second_call_messages = client.messages_log[1]
    roles = [m["role"] for m in second_call_messages]
    assert roles == ["user", "assistant", "user"], roles
    assert "target_path" in second_call_messages[1]["content"]


# ---- exhaustion path ----


def test_three_bad_responses_exhaust_and_raise(tmp_path):
    no_yaml = "still no YAML, sorry"
    client = _SequenceFakeClient([no_yaml, no_yaml, no_yaml])
    runner = RoleRunner(client=client)
    ctx = _build_context(tmp_path)

    with pytest.raises(RoleRunnerError, match="YAML"):
        runner.editor(ctx, trial_id=1)

    # MAX_EDITOR_ATTEMPTS = 3, so the client was hit three times.
    assert client.call_count == 3


def test_redaction_exhaustion_wraps_in_role_runner_error(tmp_path):
    leaky = """\
```yaml
trial_id: 1
hypothesis: |
  The 2020 covid drawdown is real.
proposed_edit:
  type: parameter_change
  change: { path: sizing.fraction, value: 0.5 }
expected_train_signal: neutral
expected_validation_signal: accept
fallback_if_rejected: x
```
"""
    client = _SequenceFakeClient([leaky, leaky, leaky])
    runner = RoleRunner(client=client)
    ctx = _build_context(tmp_path)

    with pytest.raises(RoleRunnerError, match="leaked forbidden token"):
        runner.editor(ctx, trial_id=1)
    assert client.call_count == 3


def test_first_attempt_success_records_no_retries(tmp_path):
    """When attempt 1 already passes everything, retry_attempts==1 and
    retry_reasons is the empty list."""
    client = _SequenceFakeClient([_VALID])
    runner = RoleRunner(client=client)
    ctx = _build_context(tmp_path)
    result = runner.editor(ctx, trial_id=1)
    assert client.call_count == 1
    assert result.compute.retry_attempts == 1
    assert result.compute.retry_reasons == []
