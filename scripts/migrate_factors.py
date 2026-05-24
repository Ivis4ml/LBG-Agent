"""Migrate libs/{Name}_{ts}/response*.txt → knowledge/factors/dossiers/{Name}_{ts}.txt
and generate knowledge/factors/index.jsonl.

Run once after dropping new dossiers into libs/. Idempotent: existing dossiers
with the same {Name}_{ts}.txt are overwritten; index.jsonl is fully rebuilt.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import sys

SRC = "libs"
DST_DIR = "knowledge/factors"
DOSSIERS = f"{DST_DIR}/dossiers"
INDEX = f"{DST_DIR}/index.jsonl"

VALID_ESCAPES = set('"\\/bfnrtu')


def safe_parse(text: str) -> tuple[dict | None, str]:
    """Three-tier parse: strict → escape-fixed strict → give up."""
    try:
        return json.loads(text), "strict"
    except json.JSONDecodeError:
        pass
    fixed = re.sub(
        r"\\(.)",
        lambda m: m.group(0) if m.group(1) in VALID_ESCAPES else "\\\\" + m.group(1),
        text,
    )
    try:
        return json.loads(fixed), "fixed"
    except json.JSONDecodeError:
        return None, "raw"


def first_present(d: dict, *keys, default=None):
    for k in keys:
        v = d.get(k)
        if v:
            return v
    return default


def extract_meta(obj: dict | None, raw: str, folder: str) -> tuple[str, str, str, str]:
    """Return (factor_name, ts, category, one_line)."""
    # folder is like "Aroon_20260427_023945"
    m = re.match(r"^(.+)_(\d{8})_(\d{6})$", folder)
    if m:
        fallback_name, date, time = m.groups()
        ts = f"{date}_{time}"
    else:
        fallback_name, ts = folder, "unknown"

    if obj is None or not isinstance(obj, dict):
        factor_name = fallback_name
        category = "unknown"
        summary = ""
    else:
        factor_name = obj.get("factor_name") or fallback_name
        # Try several known sub-block names (schema is not standardized).
        ca_candidates = [
            obj.get("core_analysis"),
            obj.get("analysis"),
            obj.get("dimensions"),
            obj.get("supplemental_dimensions"),
            obj.get("base_query_response"),
        ]
        ca = next((c for c in ca_candidates if isinstance(c, dict)), {})

        category_keys = (
            "factor_category",
            "factor_classification",
            "category",
            "因子分类",
            "因子逻辑分类体系",
            "factor_category_system",
        )
        category = first_present(ca, *category_keys) or first_present(
            obj, *category_keys, default="unknown"
        )
        if isinstance(category, dict):
            # Some dossiers nest category under a dict with a "primary" or "main" key
            category = (
                category.get("primary")
                or category.get("main")
                or category.get("type")
                or str(list(category.values())[0])[:60]
                if category
                else "unknown"
            )

        summary_keys = (
            "strict_definition",
            "rigorous_definition",
            "comprehensive_definition",
            "definition",
            "factor_definition",
            "financial_meaning",
            "financial_interpretation",
            "严格定义",
            "严密定义",
            "金融含义",
        )
        summary = first_present(ca, *summary_keys) or first_present(obj, *summary_keys, default="")

    # If structured extraction failed, fall back to regex on raw text
    if not summary:
        m = re.search(
            r'(?:strict_definition|financial_meaning|严格定义|金融含义)["\']?\s*:\s*["\']([^"\']{20,400})',
            raw,
        )
        summary = m.group(1) if m else ""

    # Last-ditch: take cleaned head of raw text
    if not summary:
        clean = re.sub(r"[\s{}]+", " ", raw).strip()
        summary = clean[:240]

    # Coerce any non-string structured value (dict / list) to its JSON repr
    if not isinstance(summary, str):
        summary = json.dumps(summary, ensure_ascii=False)
    summary = summary.strip().replace("\n", " ")[:240]
    return factor_name, ts, str(category), summary


def safe_basename(name: str) -> str:
    """Slug-ify factor_name for use as a filename component."""
    return re.sub(r"[^\w\-]", "_", name)


def main() -> int:
    if not os.path.isdir(SRC):
        print(f"source {SRC}/ not found", file=sys.stderr)
        return 1

    os.makedirs(DOSSIERS, exist_ok=True)
    entries = []
    skipped = 0

    for folder in sorted(os.listdir(SRC)):
        src_sub = os.path.join(SRC, folder)
        if not os.path.isdir(src_sub):
            continue
        txts = sorted(glob.glob(f"{src_sub}/*.txt"))
        if not txts:
            skipped += 1
            continue
        src_file = txts[0]
        with open(src_file, encoding="utf-8") as f:
            raw = f.read()
        obj, status = safe_parse(raw)
        factor_name, ts, category, one_line = extract_meta(obj, raw, folder)

        dst_name = f"{safe_basename(factor_name)}_{ts}.txt"
        dst_path = os.path.join(DOSSIERS, dst_name)
        shutil.copy2(src_file, dst_path)

        entries.append(
            {
                "factor_name": factor_name,
                "ts": ts,
                "version": "v1",
                "path": f"dossiers/{dst_name}",
                "category": category,
                "parse_status": status,
                "one_line": one_line,
            }
        )

    with open(INDEX, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    by_status = {"strict": 0, "fixed": 0, "raw": 0}
    for e in entries:
        by_status[e["parse_status"]] += 1

    print(f"migrated {len(entries)} dossiers → {DOSSIERS}/")
    print(f"index written → {INDEX} ({len(entries)} lines)")
    print(f"  strict parse: {by_status['strict']}")
    print(f"  fixed parse:  {by_status['fixed']}")
    print(f"  raw fallback: {by_status['raw']}")
    print(f"  skipped (no .txt): {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
