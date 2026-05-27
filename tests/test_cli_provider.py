"""Tests for the claude_cli provider.

These tests mock `subprocess.run` so they do NOT invoke the real CLI. A
separate live test (skipped when `claude` is missing) covers the actual
binary, gated behind `LBG_CLI_LIVE=1` so it only fires on explicit opt-in.
The mocked tests pin:

  - The exact flag set passed to `claude -p` (model, system, output-format,
    safety flags).
  - Single-turn vs multi-turn prompt flattening.
  - JSON result -> `_CliMessage` shape (text + token counts).
  - Error paths: non-zero exit, malformed JSON, `is_error: true`.
"""

from __future__ import annotations

import json
import os
import subprocess
from unittest.mock import patch

import pytest

from lbg.orchestrator.cli_provider import (
    ClaudeCliClient,
    ClaudeCliError,
    _flatten_messages,
)


def _ok_payload(text: str = "hello", in_tok: int = 12, out_tok: int = 4) -> dict:
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": text,
        "usage": {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        },
    }


def _completed_process(
    payload: dict, returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["claude"],
        returncode=returncode,
        stdout=json.dumps(payload),
        stderr=stderr,
    )


# -------- happy path --------


def test_single_user_turn_calls_subprocess_with_expected_flags():
    client = ClaudeCliClient()
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _completed_process(_ok_payload("ok"))

    with patch("lbg.orchestrator.cli_provider.subprocess.run", side_effect=fake_run):
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system="You are terse.",
            messages=[{"role": "user", "content": "hi"}],
        )

    assert resp.content[0].text == "ok"
    assert resp.usage.input_tokens == 12
    assert resp.usage.output_tokens == 4

    cmd = captured["cmd"]
    assert cmd[0] == "claude"
    assert "-p" in cmd
    assert "--output-format" in cmd and "json" in cmd
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "claude-sonnet-4-6"
    assert "--system-prompt" in cmd
    assert cmd[cmd.index("--system-prompt") + 1] == "You are terse."
    # The user prompt is always the last positional argument.
    assert cmd[-1] == "hi"
    # Safety flags so the CLI does not write session files / enable tools.
    assert "--no-session-persistence" in cmd
    assert "--tools" in cmd
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "--disable-slash-commands" in cmd


def test_multi_turn_messages_get_flattened_into_labeled_transcript():
    client = ClaudeCliClient()
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _completed_process(_ok_payload("retry-ok"))

    with patch("lbg.orchestrator.cli_provider.subprocess.run", side_effect=fake_run):
        client.messages.create(
            model="claude-opus-4-7",
            max_tokens=8000,
            system="sys",
            messages=[
                {"role": "user", "content": "original"},
                {"role": "assistant", "content": "bad reply"},
                {"role": "user", "content": "please try again"},
            ],
        )

    last = captured["cmd"][-1]
    assert "=== USER ===" in last
    assert "=== ASSISTANT ===" in last
    assert "original" in last and "bad reply" in last and "please try again" in last


def test_flatten_single_turn_is_identity():
    assert _flatten_messages([{"role": "user", "content": "x"}]) == "x"


def test_flatten_empty_returns_empty_string():
    assert _flatten_messages([]) == ""


# -------- error paths --------


def test_nonzero_exit_raises_claude_cli_error():
    client = ClaudeCliClient()
    with (
        patch(
            "lbg.orchestrator.cli_provider.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["claude"], returncode=2, stdout="", stderr="oops"
            ),
        ),
        pytest.raises(ClaudeCliError, match="exited 2"),
    ):
        client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
        )


def test_is_error_payload_raises():
    client = ClaudeCliClient()
    payload = {
        "type": "result",
        "is_error": True,
        "result": "Not logged in",
        "usage": {},
    }
    with (
        patch(
            "lbg.orchestrator.cli_provider.subprocess.run",
            return_value=_completed_process(payload),
        ),
        pytest.raises(ClaudeCliError, match="Not logged in"),
    ):
        client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
        )


def test_malformed_json_raises():
    client = ClaudeCliClient()
    proc = subprocess.CompletedProcess(args=["claude"], returncode=0, stdout="not-json", stderr="")
    with (
        patch("lbg.orchestrator.cli_provider.subprocess.run", return_value=proc),
        pytest.raises(ClaudeCliError, match="not JSON"),
    ):
        client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
        )


def test_missing_result_field_raises():
    client = ClaudeCliClient()
    payload = {"type": "result", "is_error": False, "usage": {}}
    with (
        patch(
            "lbg.orchestrator.cli_provider.subprocess.run",
            return_value=_completed_process(payload),
        ),
        pytest.raises(ClaudeCliError, match="missing string .result"),
    ):
        client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
        )


def test_timeout_raises():
    client = ClaudeCliClient(timeout_sec=0.1)
    with (
        patch(
            "lbg.orchestrator.cli_provider.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["claude"], timeout=0.1),
        ),
        pytest.raises(ClaudeCliError, match="timed out"),
    ):
        client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
        )


# -------- integration with RoleRunner --------


def test_role_runner_with_claude_cli_provider_does_not_require_api_key(monkeypatch):
    """A claude_cli RoleRunner instance should construct without ANTHROPIC_API_KEY."""
    from lbg.orchestrator.role_runner import RoleRunner

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LBG_PROVIDER", raising=False)
    runner = RoleRunner(provider="claude_cli")
    assert runner.provider == "claude_cli"
    # Building the client should not raise even with no API key set.
    client = runner._client()
    assert isinstance(client, ClaudeCliClient)


def test_lbg_provider_env_picks_claude_cli(monkeypatch):
    from lbg.orchestrator.role_runner import RoleRunner

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("LBG_PROVIDER", "claude_cli")
    runner = RoleRunner()
    assert runner.provider == "claude_cli"


# -------- live opt-in --------


@pytest.mark.skipif(
    os.environ.get("LBG_CLI_LIVE") != "1",
    reason="set LBG_CLI_LIVE=1 to run against the real claude binary",
)
def test_live_claude_cli_returns_real_response():
    """Live sanity check. Costs one short Max-plan call. Off by default."""
    client = ClaudeCliClient()
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=64,
        system="You are terse. Reply in one word.",
        messages=[{"role": "user", "content": "What color is the sky?"}],
    )
    assert isinstance(resp.content[0].text, str)
    assert len(resp.content[0].text) > 0
