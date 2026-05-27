"""Subscription-CLI providers · run LBG-Trader without per-call API billing.

Two providers live here, both implementing the same Anthropic-SDK-shaped
surface that RoleRunner expects:

  * `ClaudeCliClient` -- spawns `claude -p ... --output-format json` per
    call. Auth via Claude Code OAuth (Max plan subscription).
  * `CodexCliClient`  -- spawns `codex exec ... -o <tmpfile>` per call.
    Auth via ChatGPT Plus/Pro/Team subscription (run `codex login` once).
    Default model is `gpt-5.5`; pass any other model the user has access
    to via `RoleRunner(model=...)`.

Both expose `client.messages.create(model=..., max_tokens=..., system=...,
messages=[...])` returning an object with `.content[0].text` plus
`.usage.input_tokens` / `.usage.output_tokens` -- so the rest of RoleRunner
needs no branching by provider.

Limitations of CLI-backed providers vs the in-process SDK path:
  - Each call is a fresh subprocess. Cold-start adds ~1-3s of overhead.
  - Both binaries accept one prompt. Anthropic-style multi-turn message
    arrays get flattened into a single labeled transcript. Codex has no
    separate `--system-prompt` slot, so for `codex_cli` system and user
    are merged into one body (system first, separator, then user turns).
  - Rate limits and quotas come from the subscription tier; both
    wrappers retry with exponential backoff when their respective
    "credit / rate limit" error texts appear.

Pick via `RoleRunner(provider="claude_cli" | "codex_cli")` or
`LBG_PROVIDER=...`. See `lbg/orchestrator/role_runner.py::PROVIDERS`.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


class ClaudeCliError(RuntimeError):
    """Raised when `claude -p` exits non-zero or returns malformed output."""


@dataclass(frozen=True)
class _CliBlock:
    """Single content block matching the anthropic SDK's surface."""

    text: str


@dataclass(frozen=True)
class _CliUsage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class _CliMessage:
    """Mimics `anthropic.types.Message` enough for RoleRunner's call sites.

    `content` is a list of blocks each with a `.text` attribute, matching
    what `_extract_text(response)` in role_runner.py iterates over.
    `usage` carries the two token fields that get summed into
    `AgentComputeRecord`.
    """

    content: list[_CliBlock]
    usage: _CliUsage


class _Messages:
    def __init__(self, client: ClaudeCliClient) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        max_tokens: int,  # noqa: ARG002 (claude -p has no equivalent flag; ignored)
        system: str,
        messages: list[dict[str, str]],
        **_: object,
    ) -> _CliMessage:
        return self._client._invoke(model=model, system=system, messages=messages)


class ClaudeCliClient:
    """Subprocess wrapper that exposes the Anthropic SDK's `messages.create` shape.

    Construct with a working directory if you want to control which
    CLAUDE.md the CLI auto-discovers (default: a fresh tmp dir, so the
    project's own CLAUDE.md is not loaded as the agent's system context).
    """

    DEFAULT_FLAGS: tuple[str, ...] = (
        "--tools",
        "",
        "--no-session-persistence",
        "--disable-slash-commands",
        "--permission-mode",
        "bypassPermissions",
        "--exclude-dynamic-system-prompt-sections",
    )

    def __init__(
        self,
        *,
        binary: str = "claude",
        working_dir: str | Path | None = None,
        extra_flags: tuple[str, ...] = (),
        timeout_sec: float = 600.0,
    ) -> None:
        self.binary = binary
        self.working_dir = Path(working_dir) if working_dir else None
        self.extra_flags = tuple(extra_flags)
        self.timeout_sec = timeout_sec
        self.messages = _Messages(self)

    # Errors whose text we should back off and retry on. These are transient
    # rate-limit / subscription-window phenomena, not bugs in our prompts.
    _TRANSIENT_ERROR_SUBSTRINGS: tuple[str, ...] = (
        "Credit balance is too low",
        "rate limit",
        "Rate limit",
        "overloaded",
        "Overloaded",
    )

    _MAX_RETRIES: int = 3
    _BACKOFF_BASE_SECONDS: float = 30.0

    def _invoke(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, str]],
    ) -> _CliMessage:
        user_text = _flatten_messages(messages)
        cmd = [
            self.binary,
            "-p",
            "--output-format",
            "json",
            "--model",
            model,
            "--system-prompt",
            system,
            *self.DEFAULT_FLAGS,
            *self.extra_flags,
            user_text,
        ]

        cwd = self._resolve_cwd()
        if cwd is not None:
            cwd.mkdir(parents=True, exist_ok=True)

        last_error: str | None = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(cwd) if cwd is not None else None,
                    check=False,
                    timeout=self.timeout_sec,
                )
            except subprocess.TimeoutExpired as e:
                raise ClaudeCliError(f"claude -p timed out after {self.timeout_sec:.0f}s") from e
            finally:
                self._cleanup_cwd(cwd)

            # Parse stdout first if present (claude -p often returns rate-limit
            # errors as is_error:true JSON with exit code 1 -- handle both).
            payload: dict | None = None
            if result.stdout:
                try:
                    payload = json.loads(result.stdout)
                except json.JSONDecodeError:
                    payload = None

            err_text = ""
            if payload is not None and payload.get("is_error"):
                err_text = str(payload.get("result") or payload.get("api_error_status") or "")
            elif result.returncode != 0:
                err_text = (
                    f"stderr={(result.stderr or '')[:300]!r} stdout={(result.stdout or '')[:300]!r}"
                )

            if err_text and any(s in err_text for s in self._TRANSIENT_ERROR_SUBSTRINGS):
                last_error = err_text
                if attempt < self._MAX_RETRIES:
                    import time as _time

                    delay = self._BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                    _time.sleep(delay)
                    continue
                raise ClaudeCliError(
                    f"claude -p rate-limited after {attempt} attempts: {err_text[:300]}"
                )

            if result.returncode != 0 or (payload is not None and payload.get("is_error")):
                raise ClaudeCliError(
                    f"claude -p exited {result.returncode}: "
                    f"err={err_text[:300] if err_text else '(empty)'}"
                )
            break
        else:  # pragma: no cover
            raise ClaudeCliError(f"claude -p exhausted retries: {last_error}")

        if payload is None:
            raise ClaudeCliError(f"claude -p output was not JSON; head={result.stdout[:300]!r}")

        text = payload.get("result")
        if not isinstance(text, str):
            raise ClaudeCliError(
                f"claude -p missing string `result` field; payload keys={sorted(payload)}"
            )

        usage = payload.get("usage") or {}
        return _CliMessage(
            content=[_CliBlock(text=text)],
            usage=_CliUsage(
                input_tokens=int(usage.get("input_tokens", 0) or 0),
                output_tokens=int(usage.get("output_tokens", 0) or 0),
            ),
        )

    def _resolve_cwd(self) -> Path | None:
        """Return the working directory to spawn `claude -p` in.

        Defaults to a fresh tmp dir so the project's CLAUDE.md (which is
        about Claude Code behavior, not about being-the-LBG-Editor) isn't
        loaded as the agent's auto-discovered context. The caller can
        override by passing `working_dir=` to the constructor.
        """
        if self.working_dir is not None:
            return self.working_dir
        # Single shared session-tmp dir works fine -- we don't write
        # anything in it and claude -p reads CLAUDE.md lazily.
        env_dir = os.environ.get("LBG_CLAUDE_CLI_CWD")
        if env_dir:
            return Path(env_dir)
        return Path(tempfile.gettempdir()) / "lbg_claude_cli_neutral"

    def _cleanup_cwd(self, cwd: Path | None) -> None:
        # We currently keep the neutral tmpdir alive across calls (cheap,
        # nothing in it). Override hook left here for future cleanup.
        return None


def _flatten_messages(messages: list[dict[str, str]]) -> str:
    """Flatten Anthropic-style multi-turn messages into one user prompt.

    `claude -p` accepts a single positional prompt argument. For the common
    case (one user message), this is identity. For micro-retry context
    (user -> assistant -> user), the transcript is labeled so the model
    still sees its own prior attempt and the correction.
    """
    if not messages:
        return ""
    if len(messages) == 1 and messages[0].get("role") == "user":
        return str(messages[0].get("content", ""))
    chunks: list[str] = []
    for m in messages:
        role = (m.get("role") or "").upper() or "USER"
        content = str(m.get("content", ""))
        chunks.append(f"=== {role} ===\n{content}")
    return "\n\n".join(chunks)


# ============================== codex_cli ==============================


class CodexCliError(RuntimeError):
    """Raised when `codex exec` exits non-zero or returns malformed output."""


class _CodexMessages:
    def __init__(self, client: CodexCliClient) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        max_tokens: int,  # noqa: ARG002 (no codex equivalent)
        system: str,
        messages: list[dict[str, str]],
        **_: object,
    ) -> _CliMessage:
        return self._client._invoke(model=model, system=system, messages=messages)


class CodexCliClient:
    """Subprocess wrapper around `codex exec`. Mirrors ClaudeCliClient's
    surface so the rest of the Orchestrator stays provider-agnostic.

    Differences from claude_cli (handled internally):
      - Codex has no `--system-prompt` flag. We prepend the system body
        to the user prompt with a labeled separator (see `_compose_prompt`).
      - Codex emits the final assistant message to a file via
        `-o / --output-last-message`; stdout carries metadata + a
        `tokens used\\nN` line that we parse for accounting.
      - The default subscription model is `gpt-5.5`. Pass another via
        `RoleRunner(model=...)`. Codex routes model selection through
        ChatGPT subscription so the user must have access to the
        requested model in their plan.
    """

    DEFAULT_FLAGS: tuple[str, ...] = (
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--ignore-user-config",
        "--ignore-rules",
        "--color",
        "never",
    )

    _TRANSIENT_ERROR_SUBSTRINGS: tuple[str, ...] = (
        "rate limit",
        "Rate limit",
        "rate_limit",
        "quota",
        "Quota",
        "overloaded",
        "Overloaded",
        "Credit balance",
    )

    _MAX_RETRIES: int = 3
    _BACKOFF_BASE_SECONDS: float = 30.0

    # Recognised codex reasoning-effort levels (passed via
    # `-c model_reasoning_effort="..."`). `xhigh` and `max` use the model's
    # extended reasoning budget; defaults to `medium` when unset (codex's
    # own default), which is light on cache cost.
    _ALLOWED_EFFORTS: frozenset[str] = frozenset({"low", "medium", "high", "xhigh", "max"})

    def __init__(
        self,
        *,
        binary: str = "codex",
        working_dir: str | Path | None = None,
        extra_flags: tuple[str, ...] = (),
        timeout_sec: float = 900.0,
        effort: str | None = None,
    ) -> None:
        self.binary = binary
        self.working_dir = Path(working_dir) if working_dir else None
        # Pick up effort from constructor arg, then env, then leave it
        # unset (= codex's own default). Empty string in the env means
        # "explicitly disable", so we honour that as None.
        chosen_effort = effort
        if chosen_effort is None:
            env_effort = os.environ.get("LBG_CODEX_EFFORT", "").strip()
            chosen_effort = env_effort or None
        if chosen_effort is not None and chosen_effort not in self._ALLOWED_EFFORTS:
            raise CodexCliError(
                f"unknown codex effort {chosen_effort!r}; "
                f"expected one of {sorted(self._ALLOWED_EFFORTS)}"
            )
        self.effort = chosen_effort
        # Compose extra flags: caller-supplied ones first, then the
        # effort override if set. This way callers can still pass
        # arbitrary `-c <k>=<v>` overrides and the effort is always
        # appended at the end where codex picks it up cleanly.
        base = list(extra_flags)
        if self.effort is not None:
            base.extend(["-c", f'model_reasoning_effort="{self.effort}"'])
        self.extra_flags = tuple(base)
        self.timeout_sec = timeout_sec
        self.messages = _CodexMessages(self)

    def _invoke(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, str]],
    ) -> _CliMessage:
        body = _compose_prompt(system=system, messages=messages)

        cwd = self._resolve_cwd()
        if cwd is not None:
            cwd.mkdir(parents=True, exist_ok=True)

        last_error: str | None = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            with tempfile.NamedTemporaryFile(
                mode="w+",
                suffix=".txt",
                delete=False,
                dir=str(cwd) if cwd is not None else None,
            ) as last_msg_file:
                last_msg_path = Path(last_msg_file.name)
            try:
                cmd = [
                    self.binary,
                    "exec",
                    *self.DEFAULT_FLAGS,
                    "-m",
                    model,
                    "-o",
                    str(last_msg_path),
                    *self.extra_flags,
                    "-",  # read prompt from stdin
                ]
                try:
                    result = subprocess.run(
                        cmd,
                        input=body,
                        capture_output=True,
                        text=True,
                        cwd=str(cwd) if cwd is not None else None,
                        check=False,
                        timeout=self.timeout_sec,
                    )
                except subprocess.TimeoutExpired as e:
                    raise CodexCliError(
                        f"codex exec timed out after {self.timeout_sec:.0f}s"
                    ) from e

                stderr = result.stderr or ""
                stdout = result.stdout or ""
                err_text = ""
                if result.returncode != 0:
                    err_text = f"stderr={stderr[:400]!r} stdout={stdout[:400]!r}"
                elif not last_msg_path.exists() or last_msg_path.stat().st_size == 0:
                    err_text = "codex exec produced empty final message"

                if err_text and any(s in err_text for s in self._TRANSIENT_ERROR_SUBSTRINGS):
                    last_error = err_text
                    if attempt < self._MAX_RETRIES:
                        import time as _time

                        delay = self._BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                        _time.sleep(delay)
                        continue
                    raise CodexCliError(
                        f"codex exec rate-limited after {attempt} attempts: {err_text[:300]}"
                    )

                if result.returncode != 0:
                    raise CodexCliError(f"codex exec exited {result.returncode}: {err_text[:300]}")

                text = last_msg_path.read_text(encoding="utf-8")
                if not text.strip():
                    raise CodexCliError(
                        f"codex exec returned empty message; stdout head={stdout[:200]!r}"
                    )

                in_tok, out_tok = _parse_codex_token_usage(stdout)
                return _CliMessage(
                    content=[_CliBlock(text=text)],
                    usage=_CliUsage(input_tokens=in_tok, output_tokens=out_tok),
                )
            finally:
                try:
                    last_msg_path.unlink(missing_ok=True)
                except OSError:
                    pass
        raise CodexCliError(  # pragma: no cover
            f"codex exec exhausted retries: {last_error}"
        )

    def _resolve_cwd(self) -> Path | None:
        if self.working_dir is not None:
            return self.working_dir
        env_dir = os.environ.get("LBG_CODEX_CLI_CWD")
        if env_dir:
            return Path(env_dir)
        return Path(tempfile.gettempdir()) / "lbg_codex_cli_neutral"


def _compose_prompt(*, system: str, messages: list[dict[str, str]]) -> str:
    """Merge Anthropic-style (system, messages[]) into one Codex prompt body.

    Codex `exec` has no separate system-prompt slot, so the system text is
    prepended with an obvious separator. Multi-turn message arrays are
    rendered with role labels, matching the claude_cli flattening style
    so the model can still see its own prior failed attempt during a
    micro-retry.
    """
    chunks: list[str] = []
    sys_text = (system or "").strip()
    if sys_text:
        chunks.append("=== SYSTEM ===\n" + sys_text)

    if len(messages) == 1 and messages[0].get("role") == "user":
        chunks.append("=== USER ===\n" + str(messages[0].get("content", "")))
    else:
        for m in messages:
            role = (m.get("role") or "").upper() or "USER"
            content = str(m.get("content", ""))
            chunks.append(f"=== {role} ===\n{content}")
    return "\n\n".join(chunks)


def _parse_codex_token_usage(stdout: str) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from `codex exec` stdout.

    Codex prints a `tokens used\\nN` block; the integer is the total
    (input + output) and we have no easy split. Report it as input_tokens
    with 0 output_tokens so AgentComputeRecord still sums sensibly. The
    cost-accounting downstream uses (input+output) which is correct.
    """
    total = 0
    lines = stdout.splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("tokens used") and i + 1 < len(lines):
            try:
                total = int(lines[i + 1].replace(",", "").strip())
                break
            except ValueError:
                continue
    return (total, 0)


__all__ = [
    "ClaudeCliClient",
    "ClaudeCliError",
    "CodexCliClient",
    "CodexCliError",
]
