"""Tests for the codex_cli provider.

Subprocess is mocked so CI never burns the user's ChatGPT subscription
quota. A separate live test (skipped unless LBG_CODEX_LIVE=1) covers the
real binary. Pinned behaviour:

  - Command shape: `codex exec ... -m <model> -o <tmp> -` reading prompt
    from stdin.
  - System + user messages are merged into a single prompt body (Codex
    has no separate system slot).
  - Final assistant message comes from the -o tmp file, NOT stdout.
  - Token count is parsed from the "tokens used\\nN" block in stdout.
  - Transient errors (rate limit / quota / overloaded) trigger retries.
  - Non-zero exit code with non-transient error raises immediately.
  - RoleRunner constructs cleanly without OPENAI_API_KEY.
"""

from __future__ import annotations

import os
import subprocess
from unittest.mock import patch

import pytest

from lbg.orchestrator.cli_provider import (
    CodexCliClient,
    CodexCliError,
    _compose_prompt,
    _parse_codex_token_usage,
)


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["codex"], returncode=returncode, stdout=stdout, stderr=stderr
    )


# -------- prompt composition --------


def test_compose_prompt_single_user_turn_includes_system_block():
    body = _compose_prompt(
        system="You are terse.",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert "=== SYSTEM ===" in body
    assert "You are terse." in body
    assert "=== USER ===" in body
    assert "Hi" in body


def test_compose_prompt_multi_turn_labels_each_message():
    body = _compose_prompt(
        system="sys",
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "wrong"},
            {"role": "user", "content": "try again"},
        ],
    )
    assert body.count("=== USER ===") == 2
    assert body.count("=== ASSISTANT ===") == 1
    assert "first" in body and "wrong" in body and "try again" in body


def test_compose_prompt_empty_system_omits_block():
    body = _compose_prompt(system="", messages=[{"role": "user", "content": "x"}])
    assert "=== SYSTEM ===" not in body


# -------- token-usage parser --------


def test_parse_token_usage_extracts_count():
    stdout = "user\nfoo\n\ncodex\nbar\ntokens used\n1234\nbar\n"
    in_tok, out_tok = _parse_codex_token_usage(stdout)
    assert in_tok == 1234
    assert out_tok == 0


def test_parse_token_usage_handles_comma_grouping():
    stdout = "tokens used\n12,345\n"
    in_tok, _ = _parse_codex_token_usage(stdout)
    assert in_tok == 12345


def test_parse_token_usage_missing_block_returns_zero():
    assert _parse_codex_token_usage("no token info here") == (0, 0)


# -------- subprocess call shape --------


def test_invocation_command_uses_codex_exec_with_stdin_prompt(tmp_path):
    client = CodexCliClient(working_dir=tmp_path)
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        # Locate the -o argument so we can fake writing to it.
        oi = cmd.index("-o")
        out_path = cmd[oi + 1]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("hello world")
        return _completed(0, stdout="codex\nhello world\ntokens used\n42\nhello world\n")

    with patch("lbg.orchestrator.cli_provider.subprocess.run", side_effect=fake_run):
        resp = client.messages.create(
            model="gpt-5.5",
            max_tokens=1024,
            system="sys",
            messages=[{"role": "user", "content": "say hi"}],
        )

    assert resp.content[0].text == "hello world"
    assert resp.usage.input_tokens == 42
    assert resp.usage.output_tokens == 0

    cmd = captured["cmd"]
    assert cmd[0] == "codex"
    assert cmd[1] == "exec"
    assert "-m" in cmd and cmd[cmd.index("-m") + 1] == "gpt-5.5"
    assert "-o" in cmd
    assert cmd[-1] == "-", "prompt must come from stdin (positional '-')"
    # Safety / quiet flags pinned.
    assert "--ephemeral" in cmd
    assert "--sandbox" in cmd and cmd[cmd.index("--sandbox") + 1] == "read-only"
    assert "--ignore-user-config" in cmd
    # Stdin contains the merged body.
    assert "sys" in captured["kwargs"]["input"]
    assert "say hi" in captured["kwargs"]["input"]


# -------- error paths --------


def test_nonzero_exit_raises(tmp_path):
    client = CodexCliClient(working_dir=tmp_path)
    with (
        patch(
            "lbg.orchestrator.cli_provider.subprocess.run",
            return_value=_completed(2, stderr="boom"),
        ),
        pytest.raises(CodexCliError, match="exited 2"),
    ):
        client.messages.create(
            model="gpt-5.5",
            max_tokens=10,
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
        )


def test_rate_limit_retries_then_fails(tmp_path):
    client = CodexCliClient(working_dir=tmp_path)
    client._BACKOFF_BASE_SECONDS = 0.0  # avoid 30s sleep in tests

    proc = _completed(1, stderr="rate limit exceeded for gpt-5.5")
    with (
        patch("lbg.orchestrator.cli_provider.subprocess.run", return_value=proc),
        pytest.raises(CodexCliError, match="rate-limited after 3"),
    ):
        client.messages.create(
            model="gpt-5.5",
            max_tokens=10,
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
        )


def test_empty_last_message_file_raises(tmp_path):
    client = CodexCliClient(working_dir=tmp_path)

    def fake_run(cmd, **kwargs):
        # Don't write to the -o file. Codex returns success but we should
        # detect the missing/empty final message and raise.
        return _completed(0, stdout="tokens used\n10\n")

    with (
        patch("lbg.orchestrator.cli_provider.subprocess.run", side_effect=fake_run),
        pytest.raises(CodexCliError, match="empty"),
    ):
        client.messages.create(
            model="gpt-5.5",
            max_tokens=10,
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
        )


def test_timeout_raises(tmp_path):
    client = CodexCliClient(working_dir=tmp_path, timeout_sec=0.1)
    with (
        patch(
            "lbg.orchestrator.cli_provider.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["codex"], timeout=0.1),
        ),
        pytest.raises(CodexCliError, match="timed out"),
    ):
        client.messages.create(
            model="gpt-5.5",
            max_tokens=10,
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
        )


# -------- effort --------


def test_effort_constructor_arg_appends_config_override(tmp_path, monkeypatch):
    monkeypatch.delenv("LBG_CODEX_EFFORT", raising=False)
    client = CodexCliClient(working_dir=tmp_path, effort="xhigh")
    assert "-c" in client.extra_flags
    idx = client.extra_flags.index("-c")
    assert client.extra_flags[idx + 1] == 'model_reasoning_effort="xhigh"'


def test_effort_picked_up_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LBG_CODEX_EFFORT", "high")
    client = CodexCliClient(working_dir=tmp_path)
    assert client.effort == "high"
    assert 'model_reasoning_effort="high"' in client.extra_flags


def test_effort_invalid_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("LBG_CODEX_EFFORT", raising=False)
    with pytest.raises(CodexCliError, match="unknown codex effort"):
        CodexCliClient(working_dir=tmp_path, effort="ultra")


def test_effort_default_is_unset_no_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("LBG_CODEX_EFFORT", raising=False)
    client = CodexCliClient(working_dir=tmp_path)
    assert client.effort is None
    # No `-c model_reasoning_effort=...` entry when effort is unset.
    assert not any("model_reasoning_effort" in f for f in client.extra_flags)


# -------- integration with RoleRunner --------


def test_role_runner_with_codex_cli_does_not_require_api_key(monkeypatch):
    from lbg.orchestrator.role_runner import RoleRunner

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LBG_PROVIDER", raising=False)
    runner = RoleRunner(provider="codex_cli")
    assert runner.provider == "codex_cli"
    assert runner.model == "gpt-5.5"
    client = runner._client()
    assert isinstance(client, CodexCliClient)


def test_lbg_provider_env_picks_codex_cli(monkeypatch):
    from lbg.orchestrator.role_runner import RoleRunner

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LBG_PROVIDER", "codex_cli")
    runner = RoleRunner()
    assert runner.provider == "codex_cli"


# -------- live opt-in --------


@pytest.mark.skipif(
    os.environ.get("LBG_CODEX_LIVE") != "1",
    reason="set LBG_CODEX_LIVE=1 to run against the real codex binary",
)
def test_live_codex_cli_returns_real_response():
    """Live sanity check. Costs one short ChatGPT-subscription call. Off by default."""
    client = CodexCliClient()
    resp = client.messages.create(
        model="gpt-5.5",
        max_tokens=64,
        system="You are terse. Reply in one word.",
        messages=[{"role": "user", "content": "What color is the sky? Reply only the color."}],
    )
    assert isinstance(resp.content[0].text, str)
    assert len(resp.content[0].text.strip()) > 0
