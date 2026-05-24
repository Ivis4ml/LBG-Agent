"""Retrieval over knowledge/factors/.

Reads the append-only index.jsonl produced by scripts/migrate_factors.py.
ContextBuilder uses search() to surface a small set of relevant dossiers
to the Editor; nothing here is allowed to mutate the knowledge base.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

FACTORS_ROOT = Path(__file__).resolve().parents[2] / "knowledge" / "factors"
INDEX_PATH = FACTORS_ROOT / "index.jsonl"


@functools.lru_cache(maxsize=1)
def load_index() -> list[dict[str, Any]]:
    """Return the index as a list of dicts. Cached for the process lifetime.

    The cache is intentional: the knowledge base is read-only at runtime
    (it changes only when scripts/migrate_factors.py reruns offline).
    """
    if not INDEX_PATH.exists():
        return []
    with INDEX_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Substring match over factor_name + category + one_line, lowercased.

    This is a deliberately dumb implementation. Embeddings would be over-
    engineering at 564 entries; if the corpus ever exceeds a few thousand,
    swap this for a real retrieval API.
    """
    q = query.lower()
    if not q:
        return []
    out: list[dict[str, Any]] = []
    for entry in load_index():
        haystack = " ".join(
            str(entry.get(k, "")) for k in ("factor_name", "category", "one_line")
        ).lower()
        if q in haystack:
            out.append(entry)
            if len(out) >= top_k:
                break
    return out


def extract_factor_names_from_string(text: str, *, min_len: int = 3) -> list[str]:
    """Return dossier factor names whose lower-cased form appears in `text`.

    Used by Discovery to auto-cite when the Editor writes an indicator
    `fn` like 'adx' or 'lsma_dev' but forgets to populate `cited_factors`.
    Case-insensitive substring match; the `min_len` filter prevents short
    codes like "AD" from matching every "sma_*" indicator name.

    Returns dossier names in their canonical case (e.g. "ADX"), sorted by
    descending name length so the longer / more specific match comes first.
    """
    if not text:
        return []
    low = text.lower()
    out: list[str] = []
    for entry in load_index():
        name = (entry.get("factor_name") or "").strip()
        if len(name) < min_len:
            continue
        if name.lower() in low:
            out.append(name)
    out.sort(key=len, reverse=True)
    # Dedupe preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for n in out:
        if n not in seen:
            seen.add(n)
            deduped.append(n)
    return deduped


def get_dossier(factor_name: str) -> dict[str, Any] | None:
    """Return the parsed dossier for a factor, or a raw-text fallback dict.

    If the dossier JSON is malformed (parse_status='raw' in the index),
    the returned dict has the shape {'_raw': <full text>, '_parse_failed': True}
    so the Editor can still get the text without the framework lying about
    structure that isn't there.
    """
    for entry in load_index():
        if entry.get("factor_name") == factor_name:
            path = FACTORS_ROOT / entry["path"]
            text = path.read_text(encoding="utf-8")
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                return {"_raw": text, "_parse_failed": True, "_index": entry}
            return obj
    return None
