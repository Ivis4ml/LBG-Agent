"""Translator agent: Python factor function → declarative dossier.

The 4th LLM role (Editor, Reflector, Curator, Translator). Closes the
round-trip loop in PROPOSAL §6.7: every accepted add_indicator trial gets
its source code translated back into a JSON dossier matching the seed
library schema (`knowledge/factors/dossiers/*.txt`), written to
`alpha_cards/dossiers/<factor_name>_trial_NNNN.txt`.

The dossier is **evidence-grounded**: train metrics that were actually
observed, hypothesis text the Editor actually wrote, citations the Editor
explicitly named. Speculative fields are tagged inference and explained.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lbg.schemas import TrainMetrics, ValidationSignal


class TranslatorInput(BaseModel):
    """All evidence the Translator sees about one accepted trial."""

    model_config = ConfigDict(extra="forbid")

    trial_id: int = Field(ge=0)
    source_commit: str
    indicator_name: str  # alpha_card.signal.indicator
    indicator_fn: str  # alpha_card.signal.fn
    indicator_params: dict[str, Any]
    indicator_source: str  # Python source code of indicators/<fn>.py
    hypothesis_text: str  # Editor's original hypothesis
    train_metrics: TrainMetrics
    validation_signal: ValidationSignal
    cited_factors: list[str] = Field(default_factory=list)


class TranslatorDossier(BaseModel):
    """Minimal-validation wrapper. The schema is intentionally loose: the
    seed library uses ~28 free-form Chinese fields and we want the same
    shape for emitted dossiers. We enforce only the two structural bits
    that downstream tooling depends on: factor_name and lbg_provenance.
    Everything else is preserved verbatim from the LLM."""

    model_config = ConfigDict(extra="allow")

    factor_name: str
    lbg_provenance: dict[str, Any]


class DossierWriter:
    """Writes translated dossiers under alpha_cards/dossiers/.

    Path convention `<root>/alpha_cards/dossiers/<name>_trial_NNNN.txt`
    intentionally mirrors the seed library's
    `knowledge/factors/dossiers/<Name>_<timestamp>.txt` so future
    retrieval can union the two corpora without schema drift.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.dossiers_dir = self.root / "alpha_cards" / "dossiers"

    def write(
        self,
        dossier: TranslatorDossier,
        *,
        trial_id: int,
    ) -> Path:
        self.dossiers_dir.mkdir(parents=True, exist_ok=True)
        # File name: <factor_name>_trial_NNNN.txt to keep one dossier per
        # accepted-add_indicator trial (a factor that re-accepts later gets
        # a new file, the older one stays in place as a historical artifact).
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in dossier.factor_name)
        path = self.dossiers_dir / f"{safe_name}_trial_{trial_id:04d}.txt"
        path.write_text(
            json.dumps(dossier.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path


__all__ = [
    "DossierWriter",
    "TranslatorDossier",
    "TranslatorInput",
]
