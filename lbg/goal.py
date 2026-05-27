"""Project-level goal tracker.

Goal as set 2026-05-25 by the user:
"持续产生多个不同的因子以及dossiers" -- continuously produce multiple
distinct factors and dossiers.

Operationalisation:
  - target_distinct_fns : the number of unique indicator `fn` names in
    `alpha_cards_library/` we want to reach (default 5).
  - current_distinct_fns: counted live by `read_state(library_dir)`.
  - At target met, the goal flips to `met=True` -- campaigns can keep
    running but the runtime stops nagging.
  - Each campaign appends a `goal_progress.jsonl` line after every
    iteration sync, recording (ts, library_total, distinct_fns,
    target, met, delta_since_last). The dashboard / report consume
    this jsonl for the live progress widget.

Persistence: a single `goal_state.json` in the repo root. Read-only
for consumers; the only writer is the campaign loop after each sync.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_GOAL_STATE_FILE = "goal_state.json"
DEFAULT_GOAL_PROGRESS_FILE = "goal_progress.jsonl"
DEFAULT_TARGET_DISTINCT_FNS = 5


@dataclass(frozen=True)
class GoalState:
    """Current goal target + observed status."""

    target_distinct_fns: int
    current_distinct_fns: int
    distinct_fns: tuple[str, ...]
    library_total: int
    met: bool
    ts_utc: str

    def to_dict(self) -> dict:
        return {
            "target_distinct_fns": self.target_distinct_fns,
            "current_distinct_fns": self.current_distinct_fns,
            "distinct_fns": list(self.distinct_fns),
            "library_total": self.library_total,
            "met": self.met,
            "ts_utc": self.ts_utc,
        }


def _count_distinct_fns_in_library(library_dir: Path) -> tuple[int, tuple[str, ...], int]:
    """Read `library_dir/index.jsonl`, return (distinct_fns_count,
    distinct_fns_tuple_in_first_seen_order, total_card_count)."""
    index_path = library_dir / "index.jsonl"
    if not index_path.exists():
        return (0, (), 0)
    seen: list[str] = []
    seen_set: set[str] = set()
    total = 0
    for line in index_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        total += 1
        fn = entry.get("fn")
        if fn and fn not in seen_set:
            seen.append(fn)
            seen_set.add(fn)
    return (len(seen), tuple(seen), total)


def read_state(
    repo_root: str | Path,
    *,
    library_dir: str | Path | None = None,
    target_distinct_fns: int | None = None,
) -> GoalState:
    """Compute the current goal state from disk.

    `target_distinct_fns` is taken from `goal_state.json` if not provided
    (so the campaign loop doesn't have to re-pass it every iteration);
    a fresh repo without that file defaults to DEFAULT_TARGET_DISTINCT_FNS.
    """
    repo_root = Path(repo_root)
    state_path = repo_root / DEFAULT_GOAL_STATE_FILE
    persisted_target: int | None = None
    if state_path.exists():
        try:
            persisted = json.loads(state_path.read_text(encoding="utf-8"))
            persisted_target = int(
                persisted.get("target_distinct_fns", DEFAULT_TARGET_DISTINCT_FNS)
            )
        except (json.JSONDecodeError, ValueError):
            persisted_target = None

    target = target_distinct_fns or persisted_target or DEFAULT_TARGET_DISTINCT_FNS

    if library_dir is None:
        library_dir = repo_root / "alpha_cards_library"
    library_dir = Path(library_dir)
    distinct_count, distinct_fns, library_total = _count_distinct_fns_in_library(library_dir)

    return GoalState(
        target_distinct_fns=target,
        current_distinct_fns=distinct_count,
        distinct_fns=distinct_fns,
        library_total=library_total,
        met=distinct_count >= target,
        ts_utc=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def write_state(repo_root: str | Path, state: GoalState) -> Path:
    """Overwrite `goal_state.json` with the current state."""
    repo_root = Path(repo_root)
    path = repo_root / DEFAULT_GOAL_STATE_FILE
    path.write_text(
        json.dumps(state.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def append_progress(
    repo_root: str | Path,
    state: GoalState,
    *,
    run_root: str | Path | None = None,
    iteration_id: int | None = None,
) -> Path:
    """Append a JSONL line recording the state observed after a sync."""
    repo_root = Path(repo_root)
    path = repo_root / DEFAULT_GOAL_PROGRESS_FILE
    entry = {
        **state.to_dict(),
        "run_root": str(run_root) if run_root is not None else None,
        "iteration_id": iteration_id,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


def update_after_sync(
    repo_root: str | Path,
    *,
    library_dir: str | Path | None = None,
    target_distinct_fns: int | None = None,
    run_root: str | Path | None = None,
    iteration_id: int | None = None,
) -> GoalState:
    """One-shot helper: compute state, write goal_state.json, append progress.

    Returns the GoalState so callers (e.g. CampaignRunner) can log it.
    """
    state = read_state(repo_root, library_dir=library_dir, target_distinct_fns=target_distinct_fns)
    write_state(repo_root, state)
    append_progress(repo_root, state, run_root=run_root, iteration_id=iteration_id)
    return state


__all__ = [
    "DEFAULT_GOAL_PROGRESS_FILE",
    "DEFAULT_GOAL_STATE_FILE",
    "DEFAULT_TARGET_DISTINCT_FNS",
    "GoalState",
    "append_progress",
    "read_state",
    "update_after_sync",
    "write_state",
]
