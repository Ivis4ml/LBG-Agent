"""Regression test: Discovery must give every budget step a unique trial_id,
even when the Editor keeps aborting on the same iteration.

Bug it covers: `next_trial_id()` reads max(trial_id) from trials.jsonl. Aborted
trials never write to trials.jsonl, so without the fix the loop would pin all
aborted attempts to the same trial_id and waste budget retrying.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from lbg.orchestrator import RoleRunner, RoleRunnerError
from lbg.orchestrator.discovery import Discovery
from lbg.sealed_vault import SealedVault

REPO = Path(__file__).resolve().parents[1]


class _AlwaysAbortingRunner(RoleRunner):
    """A RoleRunner whose editor() always raises -- forces every trial to abort."""

    def __init__(self):
        super().__init__(api_key="dummy", client=object())  # client never used
        self.calls: list[int] = []

    def editor(self, context, *, trial_id):  # noqa: ARG002
        self.calls.append(trial_id)
        raise RoleRunnerError("simulated editor abort")


@pytest.fixture
def working_tree(tmp_path):
    """Repo skeleton + data symlink + git init -- enough to run Discovery."""
    shutil.copy(REPO / "strategy.yaml", tmp_path / "strategy.yaml")
    (tmp_path / "indicators").mkdir()
    shutil.copy(REPO / "indicators" / "sma.py", tmp_path / "indicators" / "sma.py")
    shutil.copy(REPO / "indicators" / "__init__.py", tmp_path / "indicators" / "__init__.py")
    (tmp_path / "data").symlink_to(REPO / "data")
    (tmp_path / "memory").mkdir()
    (tmp_path / "skills").mkdir()

    subprocess.run(
        ["git", "init", "--initial-branch=main", "-q"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("test\n")
    subprocess.run(["git", "add", "README.md", "strategy.yaml"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "baseline"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    return tmp_path


def test_aborted_trials_get_unique_sequential_ids(working_tree):
    runner = _AlwaysAbortingRunner()
    disc = Discovery(repo_root=working_tree, runner=runner)
    vault = SealedVault(working_tree / "vault.json")
    res = disc.run(
        budget=4,
        seal_at_end=True,
        vault=vault,
        report_path=None,
    )
    assert res.n_trials_attempted == 4
    assert res.n_aborted == 4
    # The key invariant: editor() was called with 4 distinct trial_ids.
    assert runner.calls == sorted(set(runner.calls))
    assert len(set(runner.calls)) == 4


def test_aborted_trials_dont_share_id_even_when_no_writes_to_trials_jsonl(working_tree):
    """trials.jsonl stays empty (all aborts) yet trial_id still increments."""
    runner = _AlwaysAbortingRunner()
    disc = Discovery(repo_root=working_tree, runner=runner)
    vault = SealedVault(working_tree / "vault.json")
    disc.run(budget=3, seal_at_end=True, vault=vault, report_path=None)
    # trials.jsonl should not exist or be empty.
    p = working_tree / "memory" / "trials.jsonl"
    if p.exists():
        assert p.read_text() == ""
    # All three calls happened with distinct ids.
    assert len(runner.calls) == 3
    assert len(set(runner.calls)) == 3
