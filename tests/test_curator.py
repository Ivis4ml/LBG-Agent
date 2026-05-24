"""Tests for `lbg.orchestrator.curator.Curator` (shadow mode)."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

import pytest

from lbg.git_manager import GitManager
from lbg.memory import MemoryManager
from lbg.orchestrator import RedactionError, RoleRunner
from lbg.orchestrator.curator import Curator, CuratorOutputPayload

# -------- fakes --------


@dataclass
class _FakeUsage:
    input_tokens: int = 50
    output_tokens: int = 80


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
        return _FakeMessage(content=[_FakeBlock(text=self.response_text)], usage=_FakeUsage())


_VALID_CURATOR_RESPONSE = """\
```yaml
accepted_rules_md: |
  - longer SMA reduces whipsaw at the cost of late entries
  - vol-target sizing helps drawdown when realized vol elevates
failed_directions_md: |
  - removing exit_rule entirely (no exit) is unsafe
open_questions_md: |
  - would a regime filter improve Sharpe further?
do_not_repeat_md: ""
note: |
  Merged three duplicate bullets about vol-target into one.
```
"""


# -------- triggers --------


def test_should_run_only_on_multiples(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    cur = Curator(runner=RoleRunner(client=_FakeClient("")), memory=mm, cycle_interval=10)
    assert cur.should_run(0) is False
    assert cur.should_run(5) is False
    assert cur.should_run(10) is True
    assert cur.should_run(15) is False
    assert cur.should_run(20) is True


def test_should_run_respects_custom_interval(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    cur = Curator(runner=RoleRunner(client=_FakeClient("")), memory=mm, cycle_interval=3)
    assert cur.should_run(3) is True
    assert cur.should_run(6) is True
    assert cur.should_run(4) is False


# -------- compression overwrites memory --------


def test_curator_overwrites_memory_files(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    # Seed with redundant pre-compression content.
    mm.append_to_md("accepted_rules.md", ["x", "x duplicated", "x essentially"])
    mm.append_to_md("failed_directions.md", ["no-exit was bad"])

    runner = RoleRunner(client=_FakeClient(_VALID_CURATOR_RESPONSE))
    cur = Curator(runner=runner, memory=mm)

    result = cur.run(cycle=1, accepted_count=10)

    assert isinstance(result.payload, CuratorOutputPayload)
    # File contents replaced.
    accepted = mm.read_md("accepted_rules.md")
    assert "longer SMA" in accepted
    assert "duplicated" not in accepted
    assert mm.read_md("do_not_repeat.md") == ""  # empty string preserved
    assert result.compute.role == "curator"
    assert result.commit_sha is None  # no git provided


def test_curator_prompt_includes_current_memory(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    mm.append_to_md("open_questions.md", ["does ATR help?"])
    client = _FakeClient(_VALID_CURATOR_RESPONSE)
    runner = RoleRunner(client=client)
    cur = Curator(runner=runner, memory=mm)
    cur.run(cycle=1, accepted_count=10)
    assert "does ATR help?" in client.last_user
    assert "accepted_count=10" in client.last_user


# -------- redaction --------


def test_curator_rejects_year_in_response(tmp_path):
    bad = """\
```yaml
accepted_rules_md: |
  - the 2020 drawdown demonstrated regime fragility
failed_directions_md: ""
open_questions_md: ""
do_not_repeat_md: ""
note: ok
```
"""
    mm = MemoryManager(tmp_path / "memory")
    runner = RoleRunner(client=_FakeClient(bad))
    cur = Curator(runner=runner, memory=mm)
    with pytest.raises(RedactionError):
        cur.run(cycle=1, accepted_count=10)


# -------- git integration --------


def test_curator_commits_on_curator_branch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "--initial-branch=main"], cwd=repo, check=True, capture_output=True
    )
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=repo, check=True)
    (repo / "README.md").write_text("init\n")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

    mm = MemoryManager(repo / "memory")
    git = GitManager(repo)
    runner = RoleRunner(client=_FakeClient(_VALID_CURATOR_RESPONSE))
    cur = Curator(runner=runner, memory=mm, git=git)

    result = cur.run(cycle=2, accepted_count=20)

    assert result.commit_sha is not None
    assert git.current_branch() == "curator/cycle-002"
    log = git.list_recent_commits(limit=2)
    assert log[0][1].startswith("curator cycle 002:")


# -------- live --------


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; skipping live Curator call",
)
def test_curator_live_one_call(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    mm.append_to_md(
        "accepted_rules.md",
        [
            "Long-period SMA reduces whipsaw at the cost of late entries",
            "Long-period moving averages reduce whipsaw but enter late",  # near-duplicate
            "Volatility-target sizing helps drawdown",
        ],
    )
    runner = RoleRunner()
    cur = Curator(runner=runner, memory=mm)
    result = cur.run(cycle=1, accepted_count=10)
    assert result.compute.input_tokens > 0
    assert result.compute.output_tokens > 0
    # After compression, the rules file should be non-empty (LLM kept at least
    # one bullet) and we should see fewer bullets than before.
    after = mm.read_md("accepted_rules.md")
    assert len(after.strip()) > 0
    n_bullets_after = after.count("\n- ") + (1 if after.startswith("- ") else 0)
    assert n_bullets_after <= 3  # at most original count; typically fewer
