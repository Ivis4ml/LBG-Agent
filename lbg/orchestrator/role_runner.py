"""Invoke the Editor (and later Reflector/Curator) LLM with redaction guards.

The contract:
  1. Render the assembled context into a single user-prompt string.
  2. Run the prompt through `assert_redacted` -- crash hard if any leak.
  3. Call the LLM (anthropic SDK, model `claude-sonnet-4-6` by default).
  4. Run the raw response text through `assert_redacted` again -- guards
     against the LLM hallucinating a calendar year into its proposal.
  5. Extract the YAML code block, hand it to `parse_proposal`.
  6. Return `(EditProposal, payload, AgentComputeRecord)` to the caller.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from lbg.translator import TranslatorDossier, TranslatorInput

from lbg.memory.records import (
    AgentComputeRecord,
    ReflectionRecord,
    ReflectorOutputPayload,
)
from lbg.orchestrator.context_builder import (
    EditorContext,
    FactorHint,
    IndicatorCodeSummary,
    PastTrialSummary,
)
from lbg.orchestrator.redaction import RedactionError, assert_redacted
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
TRANSLATOR_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "translator_system.md"
DEFAULT_MAX_TOKENS = 4096

# Editor micro-retry: STAGE1_REPORT § 7 showed that ~12 of 20 MIMO trials
# died on missing-YAML-block parser aborts. Letting the LLM correct itself
# in-call (multi-turn) is far cheaper than burning trial slots. Cap at K=2
# extra attempts -- empirically the second attempt almost always fixes a
# format issue; further retries waste tokens.
MAX_EDITOR_ATTEMPTS = 3

_YAML_BLOCK_RE = re.compile(r"```(?:yaml)?\s*\n(?P<body>.*?)```", re.DOTALL)


_RETRY_CORRECTION = {
    "redaction": (
        "Your previous response contained a forbidden token: a 4-digit year "
        "in the modern range, an ISO date of the form YYYY-MM-DD, or a "
        "named historical event such as a pandemic or financial crisis. "
        "The protocol forbids any calendar or event reference in your "
        "output. Re-issue exactly the same proposal but stripped of those "
        "tokens. Index-only references like `window=20` are fine; calendar "
        "references like 'the recent crash' or naming a specific year are not."
    ),
    "missing_yaml": (
        "Your previous response did not contain a fenced YAML code block. "
        "The parser cannot extract a proposal without ```yaml ... ``` "
        "markers. Re-issue your proposal wrapped in a single fenced YAML "
        "block, with no prose before or after the fence."
    ),
    "parse_error": (
        "Your previous YAML block did not match the proposal schema. "
        "Re-issue the proposal using only the field names listed in the "
        "'Allowed sub-fields per edit type' table in your system prompt. "
        "Extra or renamed keys are rejected. The parser reported: {detail}"
    ),
}


@dataclass(frozen=True)
class ProviderConfig:
    """One LLM provider's connection details + default model.

    All registered providers must expose an Anthropic-compatible
    `messages.create(...)` surface so the rest of the Orchestrator stays
    provider-agnostic. `kind` picks the transport:

      * `"anthropic_sdk"` -- HTTP via the `anthropic` Python SDK. Reads
        `api_key_env` and optional `base_url`.
      * `"claude_cli"`    -- subprocess wrapper around `claude -p`. Uses
        the user's Claude Code OAuth (Max plan) instead of API billing;
        `api_key_env` and `base_url` are ignored.
    """

    name: str
    api_key_env: str | None
    base_url: str | None  # None = use SDK default (anthropic.com)
    default_model: str
    kind: str = "anthropic_sdk"


PROVIDERS: dict[str, ProviderConfig] = {
    "anthropic": ProviderConfig(
        name="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        base_url=None,
        # Default switched from opus-4-7 to sonnet-4-6 (2026-05-24) to keep
        # campaign costs sustainable. Opus stays usable via `model=` arg.
        default_model="claude-sonnet-4-6",
    ),
    "mimo": ProviderConfig(
        name="mimo",
        api_key_env="MIMO_API_KEY",
        base_url="https://api.xiaomimimo.com/anthropic",
        default_model="mimo-v2.5-pro",
    ),
    # MIMO Token-Plan SGP endpoint -- a separate quota/billing plane from
    # the standard MIMO key, anthropic-compatible. Mirrors the `mimo`
    # provider shape with a distinct base_url and env var so the two keys
    # can be exercised independently without cross-billing.
    "mimo_tp": ProviderConfig(
        name="mimo_tp",
        api_key_env="MIMO_TP_API_KEY",
        base_url="https://token-plan-sgp.xiaomimimo.com/anthropic",
        default_model="mimo-v2.5-pro",
    ),
    # Claude Code CLI -- no API cost on Max-plan subscriptions. See
    # lbg/orchestrator/cli_provider.py for the subprocess wrapper. The
    # `claude` binary must be on PATH; pass `--model` per call so the
    # provider can route opus vs sonnet without re-launching the CLI
    # against a different default.
    "claude_cli": ProviderConfig(
        name="claude_cli",
        api_key_env=None,
        base_url=None,
        default_model="claude-sonnet-4-6",
        kind="claude_cli",
    ),
    # Codex CLI -- OpenAI's subscription-CLI counterpart to claude_cli.
    # Auth via `codex login` (ChatGPT Plus/Pro/Team). Default model is
    # gpt-5.5 (verified available in codex-cli 0.133.0); pass other
    # ChatGPT-accessible models via `RoleRunner(model=...)`.
    "codex_cli": ProviderConfig(
        name="codex_cli",
        api_key_env=None,
        base_url=None,
        default_model="gpt-5.5",
        kind="codex_cli",
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


@dataclass(frozen=True)
class TranslatorRunResult:
    """Output of the Translator role: a dossier object plus accounting."""

    dossier: TranslatorDossier
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
    (`ANTHROPIC_API_KEY`, `MIMO_API_KEY`, or `MIMO_TP_API_KEY`). Pass
    `api_key=...` to override.
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
        # CLI providers don't take an API key -- they authenticate via the
        # respective tool's subscription OAuth (claude_cli = Claude Code,
        # codex_cli = ChatGPT). Keep the field as None for them.
        if self._provider_cfg.kind in ("claude_cli", "codex_cli"):
            self._api_key = None
        else:
            self._api_key = api_key or (
                os.environ.get(self._provider_cfg.api_key_env)
                if self._provider_cfg.api_key_env
                else None
            )
        self.model = model or self._provider_cfg.default_model
        self.max_tokens = max_tokens

    @property
    def provider(self) -> str:
        return self._provider_cfg.name

    def _client(self):
        if self._explicit_client is not None:
            return self._explicit_client
        if self._provider_cfg.kind == "claude_cli":
            from lbg.orchestrator.cli_provider import ClaudeCliClient

            return ClaudeCliClient()
        if self._provider_cfg.kind == "codex_cli":
            from lbg.orchestrator.cli_provider import CodexCliClient

            return CodexCliClient()
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

        # Multi-turn message history. On a successful first call this is
        # exactly the single user-message slot we had before. On a retry we
        # append the failed assistant turn plus a corrective user turn so the
        # model sees both what it produced and what went wrong with it.
        messages: list[dict[str, str]] = [{"role": "user", "content": user_prompt}]

        total_in_tokens = 0
        total_out_tokens = 0
        total_elapsed = 0.0
        retry_reasons: list[str] = []
        last_error: Exception | None = None

        for attempt in range(1, MAX_EDITOR_ATTEMPTS + 1):
            started = time.monotonic()
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=messages,
            )
            total_elapsed += time.monotonic() - started
            total_in_tokens += getattr(response.usage, "input_tokens", 0)
            total_out_tokens += getattr(response.usage, "output_tokens", 0)

            raw_text = _extract_text(response)

            # Try to validate the response. Catch each failure mode and queue
            # a structured correction for the next attempt -- never quote the
            # raw response in the correction (it might contain the very leak
            # we're trying to scrub).
            try:
                assert_redacted(raw_text)
            except RedactionError as e:
                last_error = e
                if attempt < MAX_EDITOR_ATTEMPTS:
                    messages = _append_correction(messages, raw_text, reason="redaction")
                    retry_reasons.append("redaction")
                    continue
                raise RoleRunnerError(
                    f"Editor output leaked forbidden token after {attempt} attempts: {e}"
                ) from e

            try:
                yaml_block = _extract_yaml_block(raw_text)
            except RoleRunnerError as e:
                last_error = e
                if attempt < MAX_EDITOR_ATTEMPTS:
                    messages = _append_correction(messages, raw_text, reason="missing_yaml")
                    retry_reasons.append("missing_yaml")
                    continue
                raise

            try:
                proposal, payload = parse_proposal(yaml_block)
            except ProposalParseError as e:
                last_error = e
                if attempt < MAX_EDITOR_ATTEMPTS:
                    messages = _append_correction(
                        messages, raw_text, reason="parse_error", detail=str(e)
                    )
                    retry_reasons.append("parse_error")
                    continue
                raise RoleRunnerError(
                    f"Editor output did not parse after {attempt} attempts: {e}"
                ) from e

            compute = AgentComputeRecord(
                trial_id=trial_id,
                role="editor",
                model=self.model,
                input_tokens=total_in_tokens,
                output_tokens=total_out_tokens,
                wall_clock_sec=total_elapsed,
                retry_attempts=attempt,
                retry_reasons=retry_reasons,
            )
            return EditorRunResult(
                proposal=proposal, payload=payload, raw_text=raw_text, compute=compute
            )

        # Loop exits only via `return` or `raise`. This is defensive.
        raise RoleRunnerError(  # pragma: no cover
            f"Editor exhausted {MAX_EDITOR_ATTEMPTS} attempts: {last_error}"
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

    def translator(
        self,
        translator_input: TranslatorInput,
    ) -> TranslatorRunResult:
        """Invoke the Translator LLM. The Translator turns an accepted
        Python factor function plus its alpha-card evidence into a
        declarative dossier matching the seed-library schema.

        Failure modes mirror reflector(): one shot, no retry. A bad
        dossier doesn't block the Discovery loop -- it just means no
        dossier got emitted for this trial (the alpha_card is still on
        disk). Callers should catch RoleRunnerError and continue.
        """
        from lbg.translator import TranslatorDossier

        system_prompt = TRANSLATOR_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        user_prompt = _render_translator_user_prompt(translator_input)

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

        json_block = _extract_json_block(raw_text, role="Translator")
        try:
            data = json.loads(json_block)
        except json.JSONDecodeError as e:
            raise RoleRunnerError(
                f"Translator output not valid JSON: {e}\n--- raw ---\n{raw_text[:500]}"
            ) from e
        if not isinstance(data, dict):
            raise RoleRunnerError(f"Translator output is not a mapping; got {type(data).__name__}")
        try:
            dossier = TranslatorDossier.model_validate(data)
        except ValidationError as e:
            raise RoleRunnerError(
                f"Translator output missing required keys: {e}\n--- raw ---\n{raw_text[:500]}"
            ) from e

        compute = AgentComputeRecord(
            trial_id=translator_input.trial_id,
            role="translator",
            model=self.model,
            input_tokens=getattr(response.usage, "input_tokens", 0),
            output_tokens=getattr(response.usage, "output_tokens", 0),
            wall_clock_sec=elapsed,
        )
        return TranslatorRunResult(dossier=dossier, raw_text=raw_text, compute=compute)


def _append_correction(
    messages: list[dict[str, str]],
    raw_text: str,
    *,
    reason: str,
    detail: str = "",
) -> list[dict[str, str]]:
    """Build the next attempt's message history with a structured correction.

    For `redaction` failures the leaked assistant turn is dropped on the
    floor: replaying it would re-introduce the forbidden token into the
    model's own context window. Instead we prepend the correction to the
    original user prompt and reissue as a single-turn message.

    For `missing_yaml` / `parse_error` the assistant turn is safe to
    replay (it just had wrong shape, not a redaction violation), so we
    follow the usual multi-turn correction pattern: keep the failed turn
    visible, append a new user turn that explains the fix.
    """
    template = _RETRY_CORRECTION[reason]
    correction = template.format(detail=detail) if "{detail}" in template else template
    if reason == "redaction":
        original = messages[0]["content"]
        return [{"role": "user", "content": correction + "\n\n---\n\n" + original}]
    return messages + [
        {"role": "assistant", "content": raw_text},
        {"role": "user", "content": correction},
    ]


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(?P<body>.*?)```", re.DOTALL)


def _extract_json_block(text: str, *, role: str = "Translator") -> str:
    m = _JSON_BLOCK_RE.search(text)
    if m:
        return m.group("body").strip()
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    raise RoleRunnerError(f"{role} response did not contain a JSON code block:\n{text[:500]}")


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
    if context.indicator_code:
        parts.append(
            "## Current indicator source excerpts\n\n"
            + _format_indicator_code(context.indicator_code)
        )
    if context.factor_hints:
        parts.append(
            "## Candidate factors from knowledge base\n\n"
            + _format_factor_hints(context.factor_hints)
        )
    if context.seed_templates:
        parts.append(
            "## Executable seed templates (you can copy these verbatim)\n\n"
            + _format_seed_templates(context.seed_templates)
        )
    if context.library_cards:
        parts.append(
            "## Alpha cards already in the library\n\n"
            + _format_library_cards(context.library_cards)
        )
    parts.append("## Recent trial history\n\n" + _format_recent_trials(context.recent_trials))
    if context.banned_indicator_fns:
        parts.append(
            "## Indicator names you must NOT repeat\n\n"
            + _format_banned_indicator_fns(context.banned_indicator_fns)
        )
    # B · forced fallback. If the *most recent* trial was rejected AND had a
    # non-empty fallback note, surface it as a dedicated reminder. The
    # Editor doesn't have to follow it, but ignoring it should require a
    # reason -- which is exactly the multi-step planning we want to force.
    pending = _pending_fallback(context.recent_trials)
    if pending is not None:
        parts.append(
            "## Your previously promised fallback\n\n"
            f'On trial {pending.trial_id} you wrote: "{pending.fallback_if_rejected}". '
            "That trial was rejected. Either execute that fallback now, or "
            "state in your hypothesis why you are choosing a different "
            "direction (one sentence is enough). Repeated bare drift from "
            "your own promises is the single most-cited Editor failure mode."
        )
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


def _render_translator_user_prompt(ti: TranslatorInput) -> str:
    """Pack all evidence the Translator needs into a single user message.

    Order: source code first so the model anchors on the executable
    artifact; then metadata; then the Editor's hypothesis; then the
    observed train metrics. cited_factors at the end -- they go into
    the dossier's `lbg_provenance.cited_factors` and influence the
    "combination patterns" section.
    """
    cited = "\n".join(f"- {c}" for c in ti.cited_factors) or "(none cited by Editor)"
    return (
        f"## Source code · indicators/{ti.indicator_fn}.py\n\n"
        "```python\n"
        f"{ti.indicator_source.rstrip()}\n"
        "```\n\n"
        f"## Alpha card metadata\n\n"
        f"- factor / indicator name: `{ti.indicator_name}`\n"
        f"- function name: `{ti.indicator_fn}`\n"
        f"- params: `{json.dumps(ti.indicator_params, ensure_ascii=False)}`\n"
        f"- source_commit: `{ti.source_commit}`\n"
        f"- trial_id: {ti.trial_id}\n\n"
        f"## Editor's original hypothesis\n\n{ti.hypothesis_text.strip()}\n\n"
        f"## Observed training-window metrics (split_A)\n\n"
        f"- sharpe: {ti.train_metrics.sharpe:.4f}\n"
        f"- max_drawdown: {ti.train_metrics.max_drawdown:.4f}\n"
        f"- turnover: {ti.train_metrics.turnover:.3f}\n"
        f"- num_trades: {ti.train_metrics.num_trades}\n"
        f"- validation_signal (categorical): `{ti.validation_signal.value}`\n\n"
        f"## Editor-cited seed-library factors\n\n{cited}\n\n"
        "## Your turn\n\n"
        "Emit one fenced JSON code block whose object matches the schema "
        "in your system prompt. Ground every numeric claim in the metrics "
        "above; mark every other quantitative statement as inference. No "
        "prose outside the code block."
    )


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


def _pending_fallback(trials: list[PastTrialSummary]) -> PastTrialSummary | None:
    """Return the most recent rejected trial that wrote a fallback note.

    Empty fallback or an accepted trial both yield None (no pending promise
    to call out). The Editor's "your previously promised fallback" prompt
    section is conditioned on this returning non-None.
    """
    for t in reversed(trials):
        if t.decision == "reject" and t.fallback_if_rejected:
            return t
        if t.decision == "accept":
            return None
    return None


def _format_banned_indicator_fns(banned: tuple[str, ...]) -> str:
    """Render the hard-ban list. Each fn name on its own bulleted line so the
    Editor can't miss any of them. The trailing reminder is what gives the
    list teeth: invented variants on a banned name (e.g. `chandelier_long`
    after `chandelier_stop_22` was banned) still violate this rule."""
    lines = [f"- `{fn}`" for fn in banned]
    lines.append("")
    lines.append(
        "These indicator names were already proposed and rejected in earlier "
        "trials of this campaign. **Do NOT propose any of them again, nor a "
        "minor renaming of one of them (e.g. adding `_tight`, `_v2`, a new "
        "lookback suffix).** If you want to revisit the underlying idea, "
        "change its structural form, not just the name."
    )
    return "\n".join(lines) + "\n"


def _format_factor_hints(hints: list[FactorHint]) -> str:
    """Render the read-only factor library shortlist for the Editor.

    Each hint is a bullet with name, short category, and the (already
    year-scrubbed and truncated) one-line summary.
    """
    lines: list[str] = []
    for h in hints:
        category = h.category.strip() or "(no category)"
        lines.append(f"- **{h.name}** [{category}]")
        if h.one_line.strip():
            lines.append(f"  - {h.one_line.strip()}")
    lines.append(
        "\nThese are hints from the read-only seed library, not authority. "
        "If you author an `add_indicator`, citing one of these names in the "
        "hypothesis is encouraged but not required. Invariants run regardless."
    )
    return "\n".join(lines) + "\n"


def _format_seed_templates(templates: list) -> str:
    """Render the executable seed-template shortlist.

    Each template carries: fn, category/role tags, suggested threshold
    and rearm_threshold (when role=exit), one-line description, and the
    Python source itself. The Editor is encouraged to copy the source
    verbatim into `add_indicator.change.source` rather than authoring
    from scratch.
    """
    parts: list[str] = []
    for t in templates:
        thresh_note = ""
        if t.suggested_threshold is not None:
            thresh_note = f" · suggested_threshold={t.suggested_threshold}"
        rearm_note = ""
        if t.suggested_rearm_threshold is not None:
            rearm_note = f" · suggested_rearm_threshold={t.suggested_rearm_threshold}"
        inputs_note = f" · inputs={list(t.inputs)}" if t.inputs else ""
        default_params_note = (
            f" · default_params={dict(t.default_params)}" if t.default_params else ""
        )
        header = (
            f"### {t.fn}  [category={t.category} · role={t.role}"
            f"{inputs_note}{default_params_note}{thresh_note}{rearm_note}]\n"
            f"{t.one_line.strip()}\n"
        )
        body = "```python\n" + t.source.rstrip() + "\n```"
        parts.append(header + "\n" + body)
    tail = (
        "\nThese are runnable templates. When you propose `add_indicator`, "
        "prefer to copy one of these verbatim as `change.source`, set "
        "`change.params` to the suggested defaults (or your own tuned values), "
        "and `change.attach` using the suggested_threshold "
        "(plus suggested_rearm_threshold when role=exit) to wire it in. "
        "Invariants and the validation gate still apply -- no template is "
        "exempt from prefix-stability or any other check.\n"
    )
    return "\n\n".join(parts) + "\n" + tail


def _format_library_cards(cards: list) -> str:
    """Render the persistent alpha-card library state for the Editor.

    The Editor should treat these as "factors already discovered, do
    NOT re-propose them; instead look for complementary directions".
    Each entry shows fn name + dossier link + the recorded sealed
    incremental Sharpe point estimate (when present), so the Editor
    has both identifier and signal-strength context.
    """
    if not cards:
        return "(library is empty)\n"
    lines: list[str] = []
    for c in cards:
        point_note = ""
        if c.incremental_sharpe_point is not None:
            point_note = f" · incremental_sharpe_point={c.incremental_sharpe_point:+.3f}"
        dossier_note = f" · dossier={c.dossier_factor}" if c.dossier_factor else ""
        lines.append(f"- `{c.fn}` (indicator `{c.indicator_name}`){point_note}{dossier_note}")
    lines.append(
        "\n**Do not re-propose any of the above `fn` names**. The library is "
        "accumulating; identical-fn re-proposals don't grow it. Aim for a "
        "**complementary** signal -- a different category (trend / momentum / "
        "mean-reversion / volatility / drawdown / regime), a different input "
        "column, or a structurally different formula. If the strategy already "
        "wires one of the above factors, your edit should *combine* with it "
        "(e.g. exit filter on a different regime indicator + entry filter "
        "from a momentum factor) rather than replace it."
    )
    return "\n".join(lines) + "\n"


def _format_indicator_code(items: list[IndicatorCodeSummary]) -> str:
    parts: list[str] = []
    for item in items:
        parts.append(
            f"### {item.name} -> indicators/{item.fn}.py\n\n"
            "```python\n"
            f"{item.source_excerpt.rstrip()}\n"
            "```"
        )
    return "\n\n".join(parts) + "\n"


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
