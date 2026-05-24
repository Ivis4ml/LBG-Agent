"""Load and validate `strategy.yaml`."""

from __future__ import annotations

from pathlib import Path

import yaml

from lbg.dsl.schema import Strategy


def load_strategy(path: str | Path) -> Strategy:
    """Read a strategy YAML file and validate against the schema.

    Cross-references (entry.fast, entry.slow, exit.fast, exit.slow,
    filter.indicator) must each name an entry in `indicators[]`.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    strategy = Strategy.model_validate(raw)
    _validate_cross_references(strategy)
    return strategy


def _validate_cross_references(strategy: Strategy) -> None:
    declared = {i.name for i in strategy.indicators}
    refs: list[tuple[str, str]] = []
    refs += [("entry.fast", strategy.entry.fast), ("entry.slow", strategy.entry.slow)]
    refs += [("exit.fast", strategy.exit.fast), ("exit.slow", strategy.exit.slow)]
    for i, flt in enumerate(strategy.filters):
        refs.append((f"filters[{i}].indicator", flt.indicator))
    missing = [(field, name) for field, name in refs if name not in declared]
    if missing:
        lines = ", ".join(f"{f}={n!r}" for f, n in missing)
        raise ValueError(f"unknown indicator reference(s): {lines}; declared: {sorted(declared)}")
