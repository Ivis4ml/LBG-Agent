"""Tests for the project-level /goal tracker.

The goal is set 2026-05-25: produce N distinct factors in the alpha-card
library, each with a dossier. The tracker counts distinct `fn` names in
`alpha_cards_library/index.jsonl`, persists state to `goal_state.json`,
and appends progress to `goal_progress.jsonl` after each campaign sync.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lbg.goal import (
    DEFAULT_GOAL_PROGRESS_FILE,
    DEFAULT_GOAL_STATE_FILE,
    DEFAULT_TARGET_DISTINCT_FNS,
    GoalState,
    append_progress,
    read_state,
    update_after_sync,
    write_state,
)


def _seed_library_index(library_dir: Path, fns: list[str]) -> None:
    """Drop one index.jsonl line per fn into the library."""
    library_dir.mkdir(parents=True, exist_ok=True)
    with (library_dir / "index.jsonl").open("a", encoding="utf-8") as f:
        for i, fn in enumerate(fns):
            f.write(
                json.dumps(
                    {
                        "alpha_id": f"{fn}_trial_{i:04d}",
                        "source_trial": i,
                        "source_commit": "a" * 40,
                        "indicator": fn,
                        "fn": fn,
                        "status": "accepted",
                        "dossier_factor": None,
                        "card_path": f"cards/{fn}_trial_{i:04d}.yaml",
                    }
                )
                + "\n"
            )


@pytest.fixture
def repo(tmp_path) -> Path:
    return tmp_path


# -------- empty / default state --------


def test_read_state_with_no_library_uses_default_target(repo):
    state = read_state(repo)
    assert state.target_distinct_fns == DEFAULT_TARGET_DISTINCT_FNS
    assert state.current_distinct_fns == 0
    assert state.distinct_fns == ()
    assert state.library_total == 0
    assert state.met is False


def test_read_state_with_empty_library_uses_default(repo):
    (repo / "alpha_cards_library").mkdir()
    state = read_state(repo)
    assert state.current_distinct_fns == 0
    assert state.library_total == 0


# -------- counting --------


def test_distinct_fns_counted_correctly(repo):
    library_dir = repo / "alpha_cards_library"
    _seed_library_index(library_dir, ["vol_regime_zscore", "rsi", "vol_regime_zscore", "atr_pct"])
    state = read_state(repo, library_dir=library_dir)
    assert state.library_total == 4
    assert state.current_distinct_fns == 3
    assert state.distinct_fns == ("vol_regime_zscore", "rsi", "atr_pct")


def test_goal_met_when_distinct_count_hits_target(repo):
    library_dir = repo / "alpha_cards_library"
    _seed_library_index(library_dir, ["a", "b", "c", "d", "e"])
    state = read_state(repo, library_dir=library_dir, target_distinct_fns=5)
    assert state.met is True


def test_goal_not_met_below_target(repo):
    library_dir = repo / "alpha_cards_library"
    _seed_library_index(library_dir, ["a", "b", "c"])
    state = read_state(repo, library_dir=library_dir, target_distinct_fns=5)
    assert state.met is False
    assert state.current_distinct_fns == 3


# -------- persistence --------


def test_write_state_then_read_returns_same_target(repo):
    state = GoalState(
        target_distinct_fns=7,
        current_distinct_fns=2,
        distinct_fns=("a", "b"),
        library_total=2,
        met=False,
        ts_utc="2026-05-25T00:00:00+00:00",
    )
    write_state(repo, state)
    assert (repo / DEFAULT_GOAL_STATE_FILE).exists()

    # Now read with no library and no explicit target -- should pick up 7 from disk.
    state2 = read_state(repo)
    assert state2.target_distinct_fns == 7
    # current count is recomputed from the (empty) library, so 0.
    assert state2.current_distinct_fns == 0


def test_append_progress_writes_jsonl_line(repo):
    library_dir = repo / "alpha_cards_library"
    _seed_library_index(library_dir, ["a"])
    state = read_state(repo, library_dir=library_dir, target_distinct_fns=5)
    append_progress(repo, state, run_root="/tmp/whatever", iteration_id=2)
    path = repo / DEFAULT_GOAL_PROGRESS_FILE
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["target_distinct_fns"] == 5
    assert entry["current_distinct_fns"] == 1
    assert entry["iteration_id"] == 2
    assert entry["run_root"] == "/tmp/whatever"


def test_update_after_sync_writes_both_files(repo):
    library_dir = repo / "alpha_cards_library"
    _seed_library_index(library_dir, ["a", "b"])
    state = update_after_sync(repo, library_dir=library_dir, target_distinct_fns=3)
    assert state.current_distinct_fns == 2
    assert state.met is False
    assert (repo / DEFAULT_GOAL_STATE_FILE).exists()
    assert (repo / DEFAULT_GOAL_PROGRESS_FILE).exists()


def test_explicit_target_overrides_persisted(repo):
    """Passing target_distinct_fns wins over what's in goal_state.json."""
    write_state(
        repo,
        GoalState(
            target_distinct_fns=3,
            current_distinct_fns=0,
            distinct_fns=(),
            library_total=0,
            met=False,
            ts_utc="2026-05-25T00:00:00+00:00",
        ),
    )
    library_dir = repo / "alpha_cards_library"
    _seed_library_index(library_dir, ["a", "b"])
    state = read_state(repo, library_dir=library_dir, target_distinct_fns=10)
    assert state.target_distinct_fns == 10


def test_progress_accumulates_across_calls(repo):
    """Repeated update_after_sync appends jsonl lines without truncating."""
    library_dir = repo / "alpha_cards_library"
    _seed_library_index(library_dir, ["a"])
    update_after_sync(repo, library_dir=library_dir, target_distinct_fns=3)
    _seed_library_index(library_dir, ["b"])
    update_after_sync(repo, library_dir=library_dir, target_distinct_fns=3)
    lines = (repo / DEFAULT_GOAL_PROGRESS_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["current_distinct_fns"] == 1
    assert json.loads(lines[1])["current_distinct_fns"] == 2
