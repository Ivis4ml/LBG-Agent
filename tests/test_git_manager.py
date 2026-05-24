"""Tests for `lbg.git_manager.GitManager`.

The fixture builds a throw-away repo in `tmp_path` with `git init` + an
initial commit. We never touch the real repository.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from lbg.git_manager import GitCommandError, GitManager


@pytest.fixture
def fresh_repo(tmp_path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "--initial-branch=main"], cwd=repo, check=True, capture_output=True
    )
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=repo, check=True)
    (repo / "README.md").write_text("test repo\n")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True, capture_output=True)
    return repo


# -------- read APIs --------


def test_head_sha_is_a_40_char_hex(fresh_repo):
    gm = GitManager(fresh_repo)
    sha = gm.head_sha()
    assert len(sha) == 40
    int(sha, 16)  # raises if not hex


def test_current_branch_after_init(fresh_repo):
    gm = GitManager(fresh_repo)
    assert gm.current_branch() == "main"


def test_is_clean_after_init(fresh_repo):
    gm = GitManager(fresh_repo)
    assert gm.is_clean() is True


def test_is_dirty_when_uncommitted_change(fresh_repo):
    (fresh_repo / "scratch.txt").write_text("x\n")
    gm = GitManager(fresh_repo)
    assert gm.is_clean() is False


# -------- stage + commit --------


def test_stage_and_commit_creates_new_head(fresh_repo):
    gm = GitManager(fresh_repo)
    before = gm.head_sha()
    (fresh_repo / "a.txt").write_text("alpha\n")
    gm.stage(["a.txt"])
    new_sha = gm.commit("add a.txt")
    after = gm.head_sha()
    assert new_sha == after
    assert new_sha != before


def test_stage_without_paths_raises_value_error(fresh_repo):
    gm = GitManager(fresh_repo)
    with pytest.raises(ValueError, match="no paths"):
        gm.stage([])


def test_commit_trial_uses_canonical_message(fresh_repo):
    gm = GitManager(fresh_repo)
    (fresh_repo / "b.txt").write_text("beta\n")
    gm.stage(["b.txt"])
    sha = gm.commit_trial(trial_id=7, summary="add b.txt")
    log = gm.list_recent_commits(limit=1)
    assert sha.startswith(log[0][0])  # short SHA prefix matches
    assert log[0][1] == "trial 0007: add b.txt"


def test_commit_zero_pads_trial_id_to_four_digits(fresh_repo):
    gm = GitManager(fresh_repo)
    (fresh_repo / "c.txt").write_text("c\n")
    gm.stage(["c.txt"])
    gm.commit_trial(trial_id=42, summary="any")
    assert gm.list_recent_commits(limit=1)[0][1].startswith("trial 0042:")


def test_commit_with_nothing_staged_raises(fresh_repo):
    gm = GitManager(fresh_repo)
    with pytest.raises(GitCommandError):
        gm.commit("empty commit -- expect to fail")


def test_allow_empty_commit_succeeds(fresh_repo):
    gm = GitManager(fresh_repo)
    before = gm.head_sha()
    sha = gm.commit("empty commit", allow_empty=True)
    assert sha != before


# -------- list_recent_commits --------


def test_list_recent_commits_returns_most_recent_first(fresh_repo):
    gm = GitManager(fresh_repo)
    (fresh_repo / "a.txt").write_text("a\n")
    gm.stage(["a.txt"])
    gm.commit_trial(0, "first edit")
    (fresh_repo / "b.txt").write_text("b\n")
    gm.stage(["b.txt"])
    gm.commit_trial(1, "second edit")
    log = gm.list_recent_commits(limit=5)
    assert log[0][1] == "trial 0001: second edit"
    assert log[1][1] == "trial 0000: first edit"
    assert log[2][1] == "initial"


# -------- branches --------


def test_create_branch_and_checkout(fresh_repo):
    gm = GitManager(fresh_repo)
    gm.create_branch("feature/x")
    assert gm.current_branch() == "feature/x"


def test_curator_branch_name_convention():
    gm = GitManager(".")  # path doesn't matter for this helper
    assert gm.curator_branch_name(0) == "curator/cycle-000"
    assert gm.curator_branch_name(12) == "curator/cycle-012"


def test_create_branch_without_checkout(fresh_repo):
    gm = GitManager(fresh_repo)
    gm.create_branch("ghost", checkout=False)
    assert gm.current_branch() == "main"
    # Branch exists.
    branches = subprocess.run(
        ["git", "branch", "--list"], cwd=fresh_repo, check=True, capture_output=True, text=True
    ).stdout
    assert "ghost" in branches


def test_checkout_existing_branch(fresh_repo):
    gm = GitManager(fresh_repo)
    gm.create_branch("alt")
    gm.checkout("main")
    assert gm.current_branch() == "main"
    gm.checkout("alt")
    assert gm.current_branch() == "alt"


# -------- error semantics --------


def test_invalid_git_command_raises_git_command_error(fresh_repo):
    gm = GitManager(fresh_repo)
    with pytest.raises(GitCommandError, match="exit"):
        gm.checkout("does-not-exist")


# -------- historical read --------


def test_read_file_at_ref_returns_committed_content(fresh_repo):
    gm = GitManager(fresh_repo)
    initial_sha = gm.head_sha()
    # mutate then commit
    (fresh_repo / "README.md").write_text("v2\n")
    gm.stage(["README.md"])
    gm.commit("v2 commit")
    # current HEAD has v2; the initial ref still has the original content.
    assert gm.read_file_at_ref(initial_sha, "README.md") == "test repo\n"
    assert gm.read_file_at_ref("HEAD", "README.md") == "v2\n"


def test_read_file_at_ref_missing_path_raises(fresh_repo):
    gm = GitManager(fresh_repo)
    with pytest.raises(GitCommandError):
        gm.read_file_at_ref("HEAD", "no-such-file.txt")


def test_list_tree_at_ref(fresh_repo):
    gm = GitManager(fresh_repo)
    (fresh_repo / "indicators").mkdir()
    (fresh_repo / "indicators" / "a.py").write_text("x\n")
    (fresh_repo / "indicators" / "b.py").write_text("y\n")
    gm.stage(["indicators/a.py", "indicators/b.py"])
    gm.commit("add a, b")
    tree = gm.list_tree_at_ref("HEAD", "indicators")
    assert "indicators/a.py" in tree
    assert "indicators/b.py" in tree
    assert "README.md" not in tree  # subdir scoping works
