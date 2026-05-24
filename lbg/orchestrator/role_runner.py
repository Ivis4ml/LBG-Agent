"""Invoke the Editor (and later Reflector/Curator) LLM with redaction guards.

The contract:
  1. Render the assembled context into a single user-prompt string.
  2. Run the prompt through `assert_redacted` -- crash hard if any leak.
  3. Call the LLM (anthropic SDK, model `claude-opus-4-7`).
  4. Run the raw response text through `assert_redacted` again -- guards
     against the LLM hallucinating a calendar year into its proposal.
  5. Extract the YAML code block, hand it to `parse_proposal`.
  6. Return `(EditProposal, payload, AgentComputeRecord)` to the caller.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from lbg.memory.records import (
    AgentComputeRecord,
    ReflectionRecord,
    ReflectorOutputPayload,
)
from lbg.orchestrator.context_builder import EditorContext, PastTrialSummary
from lbg.orchestrator.redaction import assert_redacted
from lbg.parser import ProposalParseError, parse_proposal
from lbg.schemas import (
    EditProposal,
    HypothesisOutcome,
    TrainMetrics,
    ValidationSignal,
)

load_dotenv()

EDITOR_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "editor_system.md"
REFLECTOR_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "reflector_system.md"
DEFAULT_MAX_TOKENS = 4096

_YAML_BLOCK_RE = re.compile(r"```(?:yaml)?\s*\n(?P<body>.*?)```", re.DOTALL)


@dataclass(frozen=True)
class ProviderConfig:
    """One LLM provider's connection details + default model.

    All registered providers must expose an Anthropic-compatible
    `messages.create(...)` surface so the rest of the Orchestrator stays
    provider-agnostic.
    """

    name: str
    api_key_env: str
    base_url: str | None  # None = use SDK default (anthropic.com)
    default_model: str


PROVIDERS: dict[str, ProviderConfig] = {
    "anthropic": ProviderConfig(
        name="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        base_url=None,
        default_model="claude-opus-4-7",
    ),
    "mimo": ProviderConfig(
        name="mimo",
        api_key_env="MIMO_API_KEY",
        base_url="https://api.xiaomimimo.com/anthropic",
        default_model="mimo-v2.5-pro",
    ),
}

# Back-compat constant used by older tests + scripts.
DEFAULT_MODEL = PROVIDERS["anthropic"].default_model


class RoleRunnerError(RuntimeError):
    """Raised on any LLM call failure: missing API key, unparseable output, ..."""


@dataclass(frozen=True)
class EditorRunResult:
    proposal: EditProposal
    payload: BaseModel
    raw_text: str
    compute: AgentComputeRecord


@dataclass(frozen=True)
class ReflectorRunResult:
    record: ReflectionRecord
    raw_text: str
    compute: AgentComputeRecord


def _resolve_provider(name: str | None) -> ProviderConfig:
    chosen = name or os.environ.get("LBG_PROVIDER", "anthropic")
    if chosen not in PROVIDERS:
        raise RoleRunnerError(f"unknown LLM provider {chosen!r}; registered: {sorted(PROVIDERS)}")
    return PROVIDERS[chosen]


class RoleRunner:
    """Wraps the LLM client + redaction + parsing for each LLM role.

    Provider selection:
      - explicit `provider=` constructor arg, or
      - `LBG_PROVIDER` environment variable, or
      - falls back to `"anthropic"`.

    The actual API key is read from the provider's registered env var
    (`ANTHROPIC_API_KEY` or `MIMO_API_KEY`). Pass `api_key=...` to override.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client=None,
    ) -> None:
        # `client` argument lets tests inject a mock without going through the
        # anthropic SDK at all.
        self._explicit_client = client
        self._provider_cfg = _resolve_provider(provider)
        self._api_key = api_key or os.environ.get(self._provider_cfg.api_key_env)
        self.model = model or self._provider_cfg.default_model
        self.max_tokens = max_tokens

    @property
    def provider(self) -> str:
        return self._provider_cfg.name

    def _client(self):
        if self._explicit_client is not None:
            return self._explicit_client
        if not self._api_key:
            raise RoleRunnerError(
                f"{self._provider_cfg.api_key_env} is not set; "
                "put it in .env or export it before running"
            )
        from anthropic import Anthropic

        if self._provider_cfg.base_url is None:
            return Anthropic(api_key=self._api_key)
        return Anthropic(api_key=self._api_key, base_url=self._provider_cfg.base_url)

    def editor(
        self,
        context: EditorContext,
        *,
        trial_id: int,
    ) -> EditorRunResult:
        system_prompt = EDITOR_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        user_prompt = _render_editor_user_prompt(context, trial_id=trial_id)

        # Defense layer 1: outgoing prompt cannot contain forbidden tokens.
        assert_redacted(system_prompt)
        assert_redacted(user_prompt)

        client = self._client()
        started = time.monotonic()
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        elapsed = time.monotonic() - started

        raw_text = _extract_text(response)

        # Defense layer 2: incoming response cannot contain forbidden tokens.
        assert_redacted(raw_text)

        yaml_block = _extract_yaml_block(raw_text)
        try:
            proposal, payload = parse_proposal(yaml_block)
        except ProposalParseError as e:
            raise RoleRunnerError(
                f"Editor output did not parse: {e}\n--- raw response ---\n{raw_text}"
            ) from e

        compute = AgentComputeRecord(
            trial_id=trial_id,
            role="editor",
            model=self.model,
            input_tokens=getattr(response.usage, "input_tokens", 0),
            output_tokens=getattr(response.usage, "output_tokens", 0),
            wall_clock_sec=elapsed,
        )
        return EditorRunResult(
            proposal=proposal, payload=payload, raw_text=raw_text, compute=compute
        )

    def reflector(
        self,
        *,
        trial_id: int,
        editor_context: EditorContext,
        proposal: EditProposal,
        hypothesis_outcome: HypothesisOutcome,
        actual_validation_signal: ValidationSignal,
        actual_train_metrics: TrainMetrics,
    ) -> ReflectorRunResult:
        """Invoke the Reflector LLM. The mechanically computed
        `hypothesis_outcome` is shown to the Reflector but never overwritten
        by it -- the Orchestrator copies it through to the final
        `ReflectionRecord`."""
        system_prompt = REFLECTOR_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        user_prompt = _render_reflector_user_prompt(
            editor_context=editor_context,
            trial_id=trial_id,
            proposal=proposal,
            hypothesis_outcome=hypothesis_outcome,
            actual_validation_signal=actual_validation_signal,
            actual_train_metrics=actual_train_metrics,
        )

        assert_redacted(system_prompt)
        assert_redacted(user_prompt)

        client = self._client()
        started = time.monotonic()
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        elapsed = time.monotonic() - started

        raw_text = _extract_text(response)
        assert_redacted(raw_text)

        yaml_block = _extract_yaml_block(raw_text, role="Reflector")
        import yaml as _yaml

        try:
            data = _yaml.safe_load(yaml_block)
        except _yaml.YAMLError as e:
            raise RoleRunnerError(
                f"Reflector output not valid YAML: {e}\n--- raw ---\n{raw_text}"
            ) from e
        if not isinstance(data, dict):
            raise RoleRunnerError(f"Reflector output is not a mapping; got {type(data).__name__}")
        try:
            payload = ReflectorOutputPayload.model_validate(data)
        except ValidationError as e:
            raise RoleRunnerError(
                f"Reflector output did not parse: {e}\n--- raw ---\n{raw_text}"
            ) from e

        record = ReflectionRecord(
            trial_id=trial_id,
            explanation=payload.explanation,
            hypothesis_outcome=hypothesis_outcome,
            accepted_rules_updates=payload.accepted_rules_updates,
            failed_directions_updates=payload.failed_directions_updates,
            open_questions_updates=payload.open_questions_updates,
            do_not_repeat_updates=payload.do_not_repeat_updates,
        )
        compute = AgentComputeRecord(
            trial_id=trial_id,
            role="reflector",
            model=self.model,
            input_tokens=getattr(response.usage, "input_tokens", 0),
            output_tokens=getattr(response.usage, "output_tokens", 0),
            wall_clock_sec=elapsed,
        )
        return ReflectorRunResult(record=record, raw_text=raw_text, compute=compute)


def _extract_text(response) -> str:
    """Extract the assistant's text from an anthropic `Message` response."""
    chunks: list[str] = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text is not None:
            chunks.append(text)
    return "\n".join(chunks)


def _extract_yaml_block(text: str, *, role: str = "Editor") -> str:
    """Pull the first ```yaml ... ``` block out of the LLM response.

    Falls back to interpreting the whole response as YAML when no code fence
    is present, *only* if the response starts with a YAML-ish key:value line.
    Otherwise raises.
    """
    m = _YAML_BLOCK_RE.search(text)
    if m:
        return m.group("body").strip()
    stripped = text.strip()
    first_line = stripped.splitlines()[0] if stripped else ""
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*:", first_line):
        return stripped
    raise RoleRunnerError(f"{role} response did not contain a YAML code block:\n{text[:500]}")


def _render_editor_user_prompt(context: EditorContext, *, trial_id: int) -> str:
    parts: list[str] = []
    parts.append(f"## Current trial id\n\ntrial_id: {trial_id}\n")
    parts.append("## Current strategy\n\n```yaml\n" + context.strategy_yaml + "```\n")
    parts.append("## Recent trial history\n\n" + _format_recent_trials(context.recent_trials))
    if context.semantic_memory:
        parts.append("## Semantic memory\n\n" + _format_semantic_memory(context.semantic_memory))
    if context.skills:
        parts.append("## Active skills\n\n" + _format_skills(context.skills))
    parts.append(
        "## Your turn\n\n"
        "Propose exactly one edit. Output a single YAML code block matching "
        "the schema in your system prompt -- no prose outside the block."
    )
    return "\n".join(parts)


def _render_reflector_user_prompt(
    *,
    editor_context: EditorContext,
    trial_id: int,
    proposal: EditProposal,
    hypothesis_outcome: HypothesisOutcome,
    actual_validation_signal: ValidationSignal,
    actual_train_metrics: TrainMetrics,
) -> str:
    parts: list[str] = []
    parts.append(f"## Trial just finished\n\ntrial_id: {trial_id}\n")
    parts.append(
        "## What the Editor proposed\n\n"
        f"- edit_type: `{proposal.proposed_edit.type.value}`\n"
        f"- hypothesis: {proposal.hypothesis.strip()!r}\n"
        f"- expected_train_signal: `{proposal.expected_train_signal.value}`\n"
        f"- expected_validation_signal: `{proposal.expected_validation_signal.value}`\n"
    )
    parts.append(
        "## Mechanically computed outcome\n\n"
        f"- hypothesis_outcome: `{hypothesis_outcome.value}` "
        "(set by HypothesisScorer; do not re-judge)\n"
        f"- actual_validation_signal: `{actual_validation_signal.value}`\n"
    )
    parts.append(
        "## Training-window metrics\n\n"
        f"- sharpe: {actual_train_metrics.sharpe:.3f}\n"
        f"- max_drawdown: {actual_train_metrics.max_drawdown:.3f}\n"
        f"- turnover: {actual_train_metrics.turnover:.2f}\n"
        f"- num_trades: {actual_train_metrics.num_trades}\n"
    )
    parts.append("## Current strategy\n\n```yaml\n" + editor_context.strategy_yaml + "```\n")
    parts.append(
        "## Recent trial history\n\n" + _format_recent_trials(editor_context.recent_trials)
    )
    if editor_context.semantic_memory:
        parts.append(
            "## Existing semantic memory (avoid duplicating these bullets)\n\n"
            + _format_semantic_memory(editor_context.semantic_memory)
        )
    parts.append(
        "## Your turn\n\n"
        "Output a single YAML code block matching the schema in your system "
        "prompt. Explain mechanically; do not re-judge the hypothesis."
    )
    return "\n".join(parts)


def _format_recent_trials(trials: list[PastTrialSummary]) -> str:
    if not trials:
        return "(no prior trials)\n"
    lines: list[str] = []
    for t in trials:
        lines.append(
            f"- trial {t.trial_id}: edit_type={t.edit_type}, "
            f"summary={t.edit_summary!r}, hypothesis={t.hypothesis_text!r}, "
            f"expected={t.expected_validation_signal}, "
            f"actual={t.actual_validation_signal}, "
            f"hypothesis_outcome={t.hypothesis_outcome}, "
            f"train_metrics=(sharpe={t.train_sharpe:.3f}, "
            f"max_dd={t.train_max_drawdown:.3f}, "
            f"turnover={t.train_turnover:.2f}, "
            f"trades={t.train_num_trades}), "
            f"decision={t.decision}"
        )
    return "\n".join(lines) + "\n"


def _format_semantic_memory(docs: dict[str, str]) -> str:
    parts: list[str] = []
    for name, content in docs.items():
        parts.append(f"### {name}\n\n{content.strip()}\n")
    return "\n".join(parts)


def _format_skills(skills) -> str:
    """One block per Skill: trigger + recipe + evidence summary."""
    lines: list[str] = []
    for s in skills:
        lines.append(f"### {s.skill_id}  (status={s.status.value})")
        lines.append(f"  trigger: validation_signal={s.trigger.validation_signal.value}")
        if s.trigger.common_context:
            for ctx in s.trigger.common_context:
                lines.append(f"    - {ctx}")
        lines.append(f"  recipe: edit_type={s.recipe.edit_type.value}")
        if s.evidence.accepted_trials:
            lines.append(f"  evidence: accepted_trials={s.evidence.accepted_trials}")
        if s.known_failure_modes:
            lines.append("  known_failure_modes:")
            for m in s.known_failure_modes:
                lines.append(f"    - {m}")
        lines.append("")
    return "\n".join(lines)
