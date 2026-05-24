"""Tests for `lbg.orchestrator.RoleRunner.editor`.

The real anthropic SDK is mocked. A separate test (skipped when
ANTHROPIC_API_KEY is not set) makes one live call to validate the full
loop end-to-end.
"""

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
from lbg.orchestrator.role_runner import (
    _extract_yaml_block,
    _render_editor_user_prompt,
)
from lbg.schemas import EditType

REPO = Path(__file__).resolve().parents[1]


# -------- helpers --------


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


class _FakeClient:
    """Stand-in for `anthropic.Anthropic` that returns canned responses."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.call_count = 0
        self.last_system = None
        self.last_user = None
        self.messages = self  # mimic `client.messages.create`

    def create(self, *, model, max_tokens, system, messages):
        self.call_count += 1
        self.last_system = system
        self.last_user = messages[0]["content"]
        return _FakeMessage(
            content=[_FakeBlock(text=self.response_text)],
            usage=_FakeUsage(
                input_tokens=len(system) // 4, output_tokens=len(self.response_text) // 4
            ),
        )


def _build_context(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    builder = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")
    return builder.editor_view(strategy)


# -------- happy path --------


_VALID_LLM_RESPONSE = """\
Here is my proposal:

```yaml
trial_id: 7
hypothesis: |
  Increase the slow SMA period to reduce whipsaw and tighten drawdowns.
proposed_edit:
  type: parameter_change
  change:
    path: indicators[sma_slow].params.period
    value: 75
expected_train_signal: mild_improvement
expected_validation_signal: accept
fallback_if_rejected: |
  Try a tighter exit_rule next.
```

That's my edit for this trial.
"""


def test_editor_parses_canonical_response(tmp_path):
    client = _FakeClient(_VALID_LLM_RESPONSE)
    runner = RoleRunner(client=client)
    ctx = _build_context(tmp_path)

    result = runner.editor(ctx, trial_id=7)

    assert client.call_count == 1
    assert result.proposal.trial_id == 7
    assert result.proposal.proposed_edit.type == EditType.PARAMETER_CHANGE
    assert result.compute.role == "editor"
    assert result.compute.model == "claude-opus-4-7"
    assert result.compute.input_tokens > 0
    assert result.compute.output_tokens > 0


def test_editor_sends_system_prompt_and_user_prompt(tmp_path):
    client = _FakeClient(_VALID_LLM_RESPONSE)
    runner = RoleRunner(client=client)
    ctx = _build_context(tmp_path)

    runner.editor(ctx, trial_id=7)

    assert "Editor agent" in client.last_system
    assert "sma_cross_baseline" in client.last_user
    assert "trial_id: 7" in client.last_user


# -------- redaction guards --------


def test_outgoing_prompt_with_year_aborts_before_llm_call(tmp_path):
    """If the assembled prompt contains a leak, the LLM is never called."""
    client = _FakeClient(_VALID_LLM_RESPONSE)
    runner = RoleRunner(client=client)
    ctx = _build_context(tmp_path)
    # Inject a year token into the strategy_yaml; this should crash the runner.
    poisoned = ctx.__class__(
        strategy_yaml=ctx.strategy_yaml + "\n# baseline since 2010\n",
        recent_trials=ctx.recent_trials,
        semantic_memory=ctx.semantic_memory,
        skills=ctx.skills,
    )
    with pytest.raises(RedactionError):
        runner.editor(poisoned, trial_id=7)
    assert client.call_count == 0


def test_incoming_response_with_year_is_rejected(tmp_path):
    """If the LLM hallucinates a year into its reply, the runner rejects."""
    bad_response = """\
```yaml
trial_id: 1
hypothesis: |
  Following the 2020 covid drawdown, I propose volatility_target.
proposed_edit:
  type: parameter_change
  change:
    path: sizing.fraction
    value: 0.5
expected_train_signal: neutral
expected_validation_signal: accept
fallback_if_rejected: |
  Try a different size.
```
"""
    client = _FakeClient(bad_response)
    runner = RoleRunner(client=client)
    ctx = _build_context(tmp_path)
    with pytest.raises(RedactionError):
        runner.editor(ctx, trial_id=1)


# -------- parse failures --------


def test_unparseable_yaml_raises_role_runner_error(tmp_path):
    client = _FakeClient("```yaml\nnot: a: valid: proposal\n```\n")
    runner = RoleRunner(client=client)
    ctx = _build_context(tmp_path)
    with pytest.raises(RoleRunnerError, match="did not parse"):
        runner.editor(ctx, trial_id=1)


def test_missing_yaml_block_raises(tmp_path):
    client = _FakeClient("I propose nothing because I forgot the code fence.\n")
    runner = RoleRunner(client=client)
    ctx = _build_context(tmp_path)
    with pytest.raises(RoleRunnerError, match="did not contain a YAML code block"):
        runner.editor(ctx, trial_id=1)


# -------- API key wiring --------


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    runner = RoleRunner(provider="anthropic", api_key=None)
    with pytest.raises(RoleRunnerError, match="ANTHROPIC_API_KEY"):
        runner._client()


# -------- provider switching --------


def test_default_provider_is_anthropic():
    """No explicit provider + no LBG_PROVIDER env -> anthropic."""
    runner = RoleRunner(api_key="dummy")
    assert runner.provider == "anthropic"
    assert runner.model == "claude-opus-4-7"


def test_provider_arg_switches_defaults():
    runner = RoleRunner(provider="mimo", api_key="dummy")
    assert runner.provider == "mimo"
    assert runner.model == "mimo-v2.5-pro"


def test_provider_env_var_switches_default(monkeypatch):
    monkeypatch.setenv("LBG_PROVIDER", "mimo")
    runner = RoleRunner(api_key="dummy")
    assert runner.provider == "mimo"


def test_explicit_model_overrides_provider_default():
    runner = RoleRunner(provider="mimo", api_key="dummy", model="some-other-mimo-model")
    assert runner.provider == "mimo"
    assert runner.model == "some-other-mimo-model"


def test_unknown_provider_raises():
    with pytest.raises(RoleRunnerError, match="unknown LLM provider"):
        RoleRunner(provider="nope")


def test_provider_missing_api_key_message_names_correct_env_var(monkeypatch):
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    runner = RoleRunner(provider="mimo", api_key=None)
    with pytest.raises(RoleRunnerError, match="MIMO_API_KEY"):
        runner._client()


def test_provider_registered_entries():
    """PROVIDERS registry exposes both anthropic and mimo with expected shapes."""
    from lbg.orchestrator import PROVIDERS

    assert {"anthropic", "mimo"} <= set(PROVIDERS)
    assert PROVIDERS["anthropic"].base_url is None
    assert PROVIDERS["mimo"].base_url == "https://api.xiaomimimo.com/anthropic"
    assert PROVIDERS["anthropic"].api_key_env == "ANTHROPIC_API_KEY"
    assert PROVIDERS["mimo"].api_key_env == "MIMO_API_KEY"


# -------- helpers --------


def test_extract_yaml_block_handles_no_fence():
    """When the model omits the code fence, accept text that starts with trial_id."""
    text = "trial_id: 1\nhypothesis: x\nproposed_edit:\n  type: parameter_change\n"
    assert _extract_yaml_block(text).startswith("trial_id: 1")


def test_render_user_prompt_includes_history(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    builder = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")
    ctx = builder.editor_view(strategy)
    text = _render_editor_user_prompt(ctx, trial_id=42)
    assert "trial_id: 42" in text
    assert "Recent trial history" in text
    assert "(no prior trials)" in text


# -------- live API call (opt-in) --------


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; skipping live Editor call",
)
def test_editor_live_one_call(tmp_path):
    """One real call to Claude with the canonical context. Validates that
    the system prompt + redaction + parsing all fit together end-to-end."""
    mm = MemoryManager(tmp_path / "memory")
    builder = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")
    ctx = builder.editor_view(strategy)

    runner = RoleRunner(provider="anthropic")
    result = runner.editor(ctx, trial_id=0)

    # The Editor's trial_id is advisory (Orchestrator assigns the real one);
    # only require that it's parseable as a non-negative int.
    assert result.proposal.trial_id >= 0
    assert result.proposal.proposed_edit.type in set(EditType)
    assert result.compute.input_tokens > 0
    assert result.compute.output_tokens > 0
    assert result.compute.wall_clock_sec >= 0
    # The compute record carries the Orchestrator-assigned trial_id.
    assert result.compute.trial_id == 0


@pytest.mark.skipif(
    not os.environ.get("MIMO_API_KEY"),
    reason="MIMO_API_KEY not set; skipping live MIMO Editor call",
)
def test_editor_live_via_mimo_provider(tmp_path):
    """Same Editor flow but routed through MIMO's anthropic-compatible endpoint.

    Validates that switching providers needs only `provider='mimo'` — the
    redaction layer, prompt rendering, and YAML parsing are provider-agnostic.
    """
    mm = MemoryManager(tmp_path / "memory")
    builder = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")
    ctx = builder.editor_view(strategy)

    runner = RoleRunner(provider="mimo")
    assert runner.provider == "mimo"
    assert runner.model == "mimo-v2.5-pro"
    result = runner.editor(ctx, trial_id=0)

    assert result.proposal.trial_id >= 0
    assert result.proposal.proposed_edit.type in set(EditType)
    assert result.compute.input_tokens > 0
    assert result.compute.output_tokens > 0
    assert result.compute.model == "mimo-v2.5-pro"
