"""Seed templates · executable factor library for the Editor.

`indicators/seed_templates/` holds ~20 small, prefix-stable Python
factor functions plus an `_index.yaml` describing each one. The Editor
sees a filtered shortlist on each trial and can copy a template's
source into an `add_indicator` proposal, then attach it via
entry / exit filter wiring.

The seed templates are NOT the same thing as `knowledge/factors/`:
  * `knowledge/factors/` is the read-only seed library of ~564 dossier
    text files. Hints, not code.
  * `indicators/seed_templates/` is the executable counterpart: small
    runnable .py files that the Editor can lift wholesale. Closes the
    "free-form code authoring" gap that made 8 prior campaigns produce
    zero accepted `add_indicator` trials.

This module:
  - parses `_index.yaml` once per process (LRU cache);
  - exposes `list_templates()` and `search()` for ContextBuilder;
  - exposes `load_template_source(fn)` so the Editor / consumers can
    read the .py source verbatim.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SEED_TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "indicators" / "seed_templates"
INDEX_PATH = SEED_TEMPLATES_DIR / "_index.yaml"


@dataclass(frozen=True)
class SeedTemplate:
    """One seed-template entry from `_index.yaml`. Mirrors the YAML
    schema; tiny dataclass keeps the call sites typed."""

    fn: str
    category: str  # trend | momentum | mean_reversion | volatility | drawdown | regime
    role: str  # entry | exit | sizing
    inputs: tuple[str, ...]
    default_params: dict[str, Any]
    suggested_threshold: float | None
    suggested_rearm_threshold: float | None
    one_line: str

    @property
    def source_path(self) -> Path:
        return SEED_TEMPLATES_DIR / f"{self.fn}.py"

    def load_source(self) -> str:
        return self.source_path.read_text(encoding="utf-8")


@functools.lru_cache(maxsize=1)
def load_index() -> list[SeedTemplate]:
    """Read the index once. Validates that every fn has a matching .py."""
    if not INDEX_PATH.exists():
        return []
    raw = yaml.safe_load(INDEX_PATH.read_text(encoding="utf-8")) or {}
    templates_raw = raw.get("templates", [])
    out: list[SeedTemplate] = []
    for entry in templates_raw:
        fn = entry["fn"]
        py_path = SEED_TEMPLATES_DIR / f"{fn}.py"
        if not py_path.exists():
            # Don't crash a Discovery run on a broken seed index -- skip
            # the bad entry with a stable warning. Tests check that index
            # and .py files match in lockstep.
            continue
        out.append(
            SeedTemplate(
                fn=fn,
                category=entry["category"],
                role=entry["role"],
                inputs=tuple(entry.get("inputs", [])),
                default_params=dict(entry.get("default_params", {})),
                suggested_threshold=entry.get("suggested_threshold"),
                suggested_rearm_threshold=entry.get("suggested_rearm_threshold"),
                one_line=str(entry.get("one_line", "")).strip(),
            )
        )
    return out


def list_templates() -> list[SeedTemplate]:
    return list(load_index())


def list_by_category(category: str) -> list[SeedTemplate]:
    return [t for t in load_index() if t.category == category]


def list_by_role(role: str) -> list[SeedTemplate]:
    return [t for t in load_index() if t.role == role]


def get_template(fn: str) -> SeedTemplate | None:
    for t in load_index():
        if t.fn == fn:
            return t
    return None


def search(
    *,
    categories: list[str] | None = None,
    roles: list[str] | None = None,
    top_k: int = 6,
) -> list[SeedTemplate]:
    """Filtered shortlist for ContextBuilder.

    `categories` / `roles` are inclusive filters (OR within each list,
    AND across lists). When both are None, return the first `top_k`
    templates in index order. Ordering within a category is not load-
    bearing; the editor uses the metadata to make its own choice.
    """
    out: list[SeedTemplate] = []
    for t in load_index():
        if categories and t.category not in categories:
            continue
        if roles and t.role not in roles:
            continue
        out.append(t)
        if len(out) >= top_k:
            break
    return out


__all__ = [
    "INDEX_PATH",
    "SEED_TEMPLATES_DIR",
    "SeedTemplate",
    "get_template",
    "list_by_category",
    "list_by_role",
    "list_templates",
    "load_index",
    "search",
]
