"""GitManager: one commit per trial, one branch per Curator cycle.

PROPOSAL.html §10: full lineage must remain auditable; no rebase, no
force-push. This module wraps the small subset of `git` we actually need
and exposes a typed interface so the Orchestrator never builds shell
strings by hand.

All commands run via `subprocess.run(["git", ...])` with `check=True`. Any
non-zero exit raises `GitCommandError`, which keeps the trial loop from
silently continuing on a partial write.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path


class GitCommandError(RuntimeError):
    """Raised when a `git` subprocess returns non-zero."""


class GitManager:
    def __init__(self, repo_root: str | Path = ".") -> None:
        self.repo_root = Path(repo_root).resolve()

    # ---- read ----

    def head_sha(self) -> str:
        return self._git("rev-parse", "HEAD").strip()

    def current_branch(self) -> str:
        return self._git("branch", "--show-current").strip()

    def list_recent_commits(self, limit: int = 20) -> list[tuple[str, str]]:
        """Return `[(sha_short, subject), ...]` for the last `limit` commits."""
        raw = self._git("log", f"-{limit}", "--pretty=format:%h\t%s")
        out: list[tuple[str, str]] = []
        for line in raw.splitlines():
            if "\t" in line:
                sha, subject = line.split("\t", 1)
                out.append((sha.strip(), subject.strip()))
        return out

    def is_clean(self) -> bool:
        return self._git("status", "--porcelain").strip() == ""

    # ---- write ----

    def stage(self, paths: Iterable[str | Path]) -> None:
        args = ["add", "--"]
        for p in paths:
            args.append(str(p))
        if len(args) == 2:
            raise ValueError("stage() called with no paths")
        self._git(*args)

    def commit(self, message: str, *, allow_empty: bool = False) -> str:
        """Commit currently staged changes; returns the new HEAD SHA."""
        args = ["commit", "-m", message]
        if allow_empty:
            args.append("--allow-empty")
        self._git(*args)
        return self.head_sha()

    def commit_trial(
        self,
        trial_id: int,
        summary: str,
        *,
        allow_empty: bool = False,
    ) -> str:
        """Commit the staged trial artifacts with a canonical subject.

        Subject format: ``trial NNNN: <summary>`` so `git log` reveals the
        full trial sequence at a glance.
        """
        subject = f"trial {trial_id:04d}: {summary}"
        return self.commit(subject, allow_empty=allow_empty)

    def create_branch(self, name: str, *, checkout: bool = True) -> None:
        if checkout:
            self._git("checkout", "-b", name)
        else:
            self._git("branch", name)

    def checkout(self, ref: str) -> None:
        self._git("checkout", ref)

    def curator_branch_name(self, cycle: int) -> str:
        """The canonical branch name for the n-th Curator cycle."""
        return f"curator/cycle-{cycle:03d}"

    # ---- historical read ----

    def read_file_at_ref(self, ref: str, relpath: str) -> str:
        """Return the contents of `relpath` as of commit `ref`."""
        return self._git("show", f"{ref}:{relpath}")

    def list_tree_at_ref(self, ref: str, subdir: str = "") -> list[str]:
        """List the files tracked under `subdir` at commit `ref`.

        Returns repo-relative paths. `subdir=""` lists the whole tree.
        """
        args = ["ls-tree", "-r", "--name-only", ref]
        if subdir:
            args.append(subdir)
        raw = self._git(*args)
        return [line.strip() for line in raw.splitlines() if line.strip()]

    # ---- internals ----

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise GitCommandError(
                f"git {' '.join(args)} -> exit {result.returncode}: {result.stderr.strip()}"
            )
        return result.stdout
