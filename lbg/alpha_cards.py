"""Alpha cards (PROPOSAL.html §6.6).

An alpha card is the metadata view binding an agent-authored Python factor
function (`indicators/<fn>.py`) to its declarative dossier form in
`knowledge/factors/`. Every accepted trial whose edit adds an indicator
produces exactly one card under `alpha_cards/trial_NNNN.yaml`, plus a line
in the append-only `alpha_cards/index.jsonl`.

This file is the H1 counting layer: H1 (per PROPOSAL §4) is operationalized
as "count cards whose sealed_summary shows individual incremental Sharpe
with CI lower bound > 0". The sealed_summary is populated later by the
sealed-evaluation pipeline (initially None on emission).

Schema is part of the PROPOSAL lock list -- changing field names mid-run
breaks H1 counting and audit. Pydantic `extra="forbid"` enforces that.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from lbg.dsl.schema import Strategy
from lbg.schemas import HypothesisOutcome, TrainMetrics, ValidationSignal


class AlphaCardStatus(StrEnum):
    """The lifecycle position of an alpha card.

    `accepted` is the initial state on emission; `deprecated` is set by a
    later Curator cycle when newer evidence overrides this card.
    """

    ACCEPTED = "accepted"
    DEPRECATED = "deprecated"
    EXPLORATORY = "exploratory"


class AlphaCardSignal(BaseModel):
    """The factor function being preserved. `source_path` is repo-relative."""

    model_config = ConfigDict(extra="forbid")

    indicator: str = Field(min_length=1, description="logical name in strategy.yaml")
    fn: str = Field(min_length=1, description="indicators/<fn>.py basename")
    params: dict[str, Any] = Field(default_factory=dict)
    source_path: str = Field(min_length=1, description="path to .py file, repo-relative")


class AlphaCardEvidence(BaseModel):
    """Train metrics + validation signal + sealed summary (initially None).

    Validation metrics are categorical only (per PROPOSAL §7). The sealed
    summary is populated by the H1 verdict pipeline, not at card emission.
    """

    model_config = ConfigDict(extra="forbid")

    train_summary: dict[str, float]  # sharpe / max_drawdown / turnover / num_trades
    validation_signal: ValidationSignal
    hypothesis_outcome: HypothesisOutcome
    # `dict[str, float | bool]` because Benjamini-Hochberg adds a
    # `bh_validated: bool` flag alongside the numeric per-card metrics.
    # bool subclasses int in Python; without explicitly allowing bool
    # here pydantic coerces True/False to 1.0/0.0, which silently breaks
    # the downstream H1 reader (it does `summary.get("bh_validated", False)`
    # and would never see True after a round-trip).
    sealed_summary: dict[str, float | bool] | None = None


class AlphaCardDossierLink(BaseModel):
    """Round-trip link to `knowledge/factors/`.

    `matched_via=editor_cite` means the Editor explicitly named this dossier
    in its hypothesis text (future step). `matched_via=name_match` means
    ContextBuilder's auto-match found the dossier by indicator-name overlap
    (used in this milestone since `cited_factors` is not yet in the schema).
    """

    model_config = ConfigDict(extra="forbid")

    factor_name: str = Field(min_length=1)
    dossier_path: str = Field(min_length=1, description="repo-relative")
    matched_via: str = Field(default="name_match")


class AlphaCardAttach(BaseModel):
    """The attach configuration the Editor used when the indicator was
    accepted. Captures *how* the factor was wired into the strategy so
    a future campaign can auto-inject it as the starting strategy state
    (PROPOSAL §20 lock #30 "validated library factors enter seed pool").

    `target`:
      - "entry" : AND-combine with the entry cross rule (filter in
                  `strategy.filters`)
      - "exit"  : OR-combine into the exit signal (filter in
                  `strategy.exit_filters`); paired with `rearm_threshold`
                  it controls the auto-resume-after-mute behaviour
    """

    model_config = ConfigDict(extra="forbid")

    rule: str  # indicator_above | indicator_below
    threshold: float
    target: str  # entry | exit
    rearm_threshold: float | None = None


class AlphaCard(BaseModel):
    """One alpha card. Schema is locked per PROPOSAL §20."""

    model_config = ConfigDict(extra="forbid")

    alpha_id: str = Field(min_length=1)
    source_trial: int = Field(ge=0)
    source_commit: str = Field(min_length=1)
    status: AlphaCardStatus = AlphaCardStatus.ACCEPTED
    signal: AlphaCardSignal
    evidence: AlphaCardEvidence
    dossier_link: AlphaCardDossierLink | None = None
    # Attach configuration captured at acceptance time -- the wiring needed
    # to re-instantiate this factor on a future campaign's baseline strategy.
    # Optional for backward compatibility with cards written before this
    # field existed; new emissions always include it.
    attach_config: AlphaCardAttach | None = None


class AlphaCardWriter:
    """Emits alpha cards under `<root>/alpha_cards/`.

    Two artifacts per accepted add_indicator trial:
      - `alpha_cards/trial_NNNN.yaml` -- the full card
      - `alpha_cards/index.jsonl`      -- append-only index, one line per card
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.cards_dir = self.root / "alpha_cards"
        self.index_path = self.cards_dir / "index.jsonl"

    def write(self, card: AlphaCard) -> Path:
        """Persist a card and append the index line. Returns the card path."""
        self.cards_dir.mkdir(parents=True, exist_ok=True)
        card_path = self.cards_dir / f"trial_{card.source_trial:04d}.yaml"
        card_path.write_text(
            yaml.safe_dump(
                card.model_dump(mode="json"),
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        index_entry = {
            "alpha_id": card.alpha_id,
            "source_trial": card.source_trial,
            "source_commit": card.source_commit,
            "indicator": card.signal.indicator,
            "fn": card.signal.fn,
            "status": card.status.value,
            "dossier_factor": (card.dossier_link.factor_name if card.dossier_link else None),
            "card_path": str(card_path.relative_to(self.root)),
        }
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(index_entry, ensure_ascii=False) + "\n")
        return card_path

    def write_for_added_indicator(
        self,
        *,
        trial_id: int,
        source_commit: str,
        added_indicator_name: str,
        strategy: Strategy,
        train_metrics: TrainMetrics,
        validation_signal: ValidationSignal,
        hypothesis_outcome: HypothesisOutcome,
        dossier_link: AlphaCardDossierLink | None = None,
        attach_config: AlphaCardAttach | None = None,
    ) -> Path:
        """Build + write the alpha card for an accepted add_indicator trial.

        Looks up the newly-added IndicatorSpec by `added_indicator_name` in
        the post-edit strategy and pins its fn / params at the moment of
        acceptance. Errors out clearly if the named indicator is missing --
        that is a Discovery wiring bug, not an LLM mistake.
        """
        spec = next((i for i in strategy.indicators if i.name == added_indicator_name), None)
        if spec is None:
            raise ValueError(
                f"indicator {added_indicator_name!r} not in strategy after "
                f"apply; strategy has {[i.name for i in strategy.indicators]}"
            )
        card = AlphaCard(
            alpha_id=f"{added_indicator_name}_trial_{trial_id:04d}",
            source_trial=trial_id,
            source_commit=source_commit,
            status=AlphaCardStatus.ACCEPTED,
            signal=AlphaCardSignal(
                indicator=spec.name,
                fn=spec.fn,
                params=dict(spec.params),
                source_path=f"indicators/{spec.fn}.py",
            ),
            evidence=AlphaCardEvidence(
                train_summary={
                    "sharpe": float(train_metrics.sharpe),
                    "max_drawdown": float(train_metrics.max_drawdown),
                    "turnover": float(train_metrics.turnover),
                    "num_trades": float(train_metrics.num_trades),
                },
                validation_signal=validation_signal,
                hypothesis_outcome=hypothesis_outcome,
                sealed_summary=None,
            ),
            dossier_link=dossier_link,
            attach_config=attach_config,
        )
        return self.write(card)


def match_dossier_by_name(
    indicator_fn: str,
    *,
    knowledge_root: str | Path = "knowledge/factors",
) -> AlphaCardDossierLink | None:
    """Best-effort link from an indicator fn name to a dossier.

    Uses `lbg.knowledge.factors.search()` -- substring match on factor_name
    is the highest-precision signal we have without an explicit cite from
    the Editor. Returns None when no dossier matches; callers must handle
    the None case. Prefer `match_dossier_from_citation()` whenever the
    Editor provided `cited_factors` -- the cite is explicit, the substring
    match is a guess.
    """
    from lbg.knowledge.factors import search

    hits = search(indicator_fn, top_k=1)
    if not hits:
        return None
    hit = hits[0]
    name = (hit.get("factor_name") or "").strip()
    path = hit.get("path") or ""
    if not name or not path:
        return None
    return AlphaCardDossierLink(
        factor_name=name,
        dossier_path=str(Path(knowledge_root) / path),
        matched_via="name_match",
    )


def match_dossier_from_citation(
    cited_factors: list[str],
    *,
    knowledge_root: str | Path = "knowledge/factors",
) -> AlphaCardDossierLink | None:
    """Explicit Editor citation → dossier link. Picks the first cited
    factor whose name resolves to a dossier path; falls back to None
    when none of the citations match the index (e.g. Editor cited a name
    not actually in the seed library)."""
    from lbg.knowledge.factors import load_index

    if not cited_factors:
        return None
    index = {e["factor_name"]: e for e in load_index() if e.get("factor_name")}
    for cite in cited_factors:
        entry = index.get(cite)
        if entry is None:
            continue
        path = entry.get("path") or ""
        if not path:
            continue
        return AlphaCardDossierLink(
            factor_name=cite,
            dossier_path=str(Path(knowledge_root) / path),
            matched_via="editor_cite",
        )
    return None


__all__ = [
    "AlphaCard",
    "AlphaCardAttach",
    "AlphaCardDossierLink",
    "AlphaCardEvidence",
    "AlphaCardSignal",
    "AlphaCardStatus",
    "AlphaCardWriter",
    "match_dossier_by_name",
    "match_dossier_from_citation",
]
