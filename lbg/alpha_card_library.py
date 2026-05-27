"""Persistent alpha-card library shared across Discovery runs.

`AlphaCardWriter` (in `lbg.alpha_cards`) writes one card per accepted
`add_indicator` trial into the run's working directory:
`<run_root>/alpha_cards/`. That directory lives wherever `--out` points
-- typically `/tmp/lbg_xxx`, which is cleared on reboot.

This module adds the durable layer the proposal §20 lock #30 requires:

  > The validated-factor library is append-only and shared across
  > re-discoveries. A surviving function (and its card) from run N
  > enters the seed pool for run N+1; it is re-validated against the
  > new sealed window and either retained, marked `deprecated`, or
  > merged. Functions are never silently dropped.

`AlphaCardLibrary` is that shared pool. Each campaign syncs newly-emitted
cards (and their `.py` indicator source, plus any Translator-generated
dossier) into a stable repo path that git tracks. The library is
append-only; sync is idempotent on `alpha_id` so re-running the same
campaign does not duplicate entries.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from lbg.alpha_cards import AlphaCard

logger = logging.getLogger(__name__)


DEFAULT_LIBRARY_DIR = Path("alpha_cards_library")


@dataclass(frozen=True)
class SyncResult:
    """Outcome of one library sync. Plays no part in the H1 computation;
    purely for CLI / log reporting."""

    new_cards: int
    duplicate_cards: int
    new_dossiers: int
    library_total: int

    def to_dict(self) -> dict[str, int]:
        return {
            "new_cards": self.new_cards,
            "duplicate_cards": self.duplicate_cards,
            "new_dossiers": self.new_dossiers,
            "library_total": self.library_total,
        }


class AlphaCardLibrary:
    """Append-only, idempotent index of validated alpha cards.

    Disk layout:

      <library_dir>/
        index.jsonl                                    # one line per card, dedup key = alpha_id
        cards/<alpha_id>.yaml                          # full AlphaCard
        indicators/<fn>.py                             # the Python source frozen at the trial
        dossiers/<alpha_id>.txt                        # Translator output, if any
        sync_log.jsonl                                 # one line per sync (audit)

    The library never overwrites an existing card. If a new run produces
    a card with an `alpha_id` already present, the new card is logged as
    a duplicate and skipped. Updates flow through the Curator's
    `deprecate / merge` workflow, which appends new lines rather than
    overwriting old ones.
    """

    def __init__(self, library_dir: str | Path) -> None:
        self.library_dir = Path(library_dir)
        self.cards_dir = self.library_dir / "cards"
        self.indicators_dir = self.library_dir / "indicators"
        self.dossiers_dir = self.library_dir / "dossiers"
        self.index_path = self.library_dir / "index.jsonl"
        self.sync_log_path = self.library_dir / "sync_log.jsonl"

    # ---- read ----

    def existing_alpha_ids(self) -> set[str]:
        """Set of alpha_ids already in the library. O(N) on first call."""
        if not self.index_path.exists():
            return set()
        out: set[str] = set()
        with self.index_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                alpha_id = entry.get("alpha_id")
                if alpha_id:
                    out.add(alpha_id)
        return out

    def load_cards(self) -> list[AlphaCard]:
        """All cards currently in the library, in index order."""
        if not self.index_path.exists():
            return []
        out: list[AlphaCard] = []
        with self.index_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                card_path = self.library_dir / entry["card_path"]
                if not card_path.exists():
                    logger.warning("library index points at missing card %s; skipping", card_path)
                    continue
                raw = yaml.safe_load(card_path.read_text(encoding="utf-8"))
                out.append(AlphaCard.model_validate(raw))
        return out

    # ---- write ----

    def sync_from_run(
        self,
        run_root: str | Path,
        *,
        iteration_id: int | None = None,
    ) -> SyncResult:
        """Copy newly-emitted cards from `<run_root>/alpha_cards/` into the library.

        Idempotent: if the same campaign is re-run with the same alpha_ids,
        no card gets duplicated. The associated indicator source (.py) and
        Translator dossier (.txt) are copied beside the card. A sync_log
        entry is appended for audit.
        """
        run_root = Path(run_root)
        run_cards_dir = run_root / "alpha_cards"
        if not run_cards_dir.exists():
            # Nothing to sync, but still log the no-op so the audit trail is honest.
            result = SyncResult(0, 0, 0, len(self.existing_alpha_ids()))
            self._append_sync_log(run_root, iteration_id, result)
            return result

        self._ensure_dirs()
        existing = self.existing_alpha_ids()

        new_count = 0
        dup_count = 0
        new_dossiers = 0

        for card_yaml_path in sorted(run_cards_dir.glob("trial_*.yaml")):
            raw = yaml.safe_load(card_yaml_path.read_text(encoding="utf-8"))
            try:
                card = AlphaCard.model_validate(raw)
            except Exception as e:  # noqa: BLE001
                logger.warning("library sync: %s did not parse as AlphaCard: %s", card_yaml_path, e)
                continue

            if card.alpha_id in existing:
                dup_count += 1
                continue

            # Copy the card body itself.
            dest_card_path = self.cards_dir / f"{card.alpha_id}.yaml"
            dest_card_path.write_text(
                yaml.safe_dump(
                    card.model_dump(mode="json"),
                    sort_keys=False,
                    default_flow_style=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )

            # Copy the indicator source if it still exists in the run tree.
            indicator_src = run_root / "indicators" / f"{card.signal.fn}.py"
            if indicator_src.exists():
                dest_indicator = self.indicators_dir / f"{card.signal.fn}.py"
                # Skip overwrite if the destination already exists with the
                # same content -- two cards may share an fn under future
                # merges, and accidental clobber would lose history.
                if not dest_indicator.exists() or dest_indicator.read_text(
                    encoding="utf-8"
                ) != indicator_src.read_text(encoding="utf-8"):
                    shutil.copy2(indicator_src, dest_indicator)

            # Copy the Translator dossier if any.
            # Translator's DossierWriter writes to alpha_cards/dossiers/<name>_trial_NNNN.txt
            dossier_src = self._find_dossier_for_card(run_root, card)
            if dossier_src is not None and dossier_src.exists():
                dest_dossier = self.dossiers_dir / f"{card.alpha_id}.txt"
                shutil.copy2(dossier_src, dest_dossier)
                new_dossiers += 1

            # Append to the library index.
            index_entry = {
                "alpha_id": card.alpha_id,
                "source_trial": card.source_trial,
                "source_commit": card.source_commit,
                "indicator": card.signal.indicator,
                "fn": card.signal.fn,
                "status": card.status.value,
                "dossier_factor": (card.dossier_link.factor_name if card.dossier_link else None),
                "card_path": str(dest_card_path.relative_to(self.library_dir)),
                "indicator_path": (
                    str(
                        (self.indicators_dir / f"{card.signal.fn}.py").relative_to(self.library_dir)
                    )
                    if (self.indicators_dir / f"{card.signal.fn}.py").exists()
                    else None
                ),
                "dossier_path": (
                    str((self.dossiers_dir / f"{card.alpha_id}.txt").relative_to(self.library_dir))
                    if dossier_src is not None and dossier_src.exists()
                    else None
                ),
                "ts_synced_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            with self.index_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(index_entry, ensure_ascii=False) + "\n")

            existing.add(card.alpha_id)
            new_count += 1

        result = SyncResult(
            new_cards=new_count,
            duplicate_cards=dup_count,
            new_dossiers=new_dossiers,
            library_total=len(existing),
        )
        self._append_sync_log(run_root, iteration_id, result)
        return result

    # ---- internals ----

    def _ensure_dirs(self) -> None:
        for d in (self.library_dir, self.cards_dir, self.indicators_dir, self.dossiers_dir):
            d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _find_dossier_for_card(run_root: Path, card: AlphaCard) -> Path | None:
        """Locate the Translator's dossier for `card`, if it wrote one.

        DossierWriter writes to alpha_cards/dossiers/<indicator_name>_trial_NNNN.txt.
        We don't know the exact filename ahead of time (the trial id is
        zero-padded), so glob over candidates.
        """
        candidate_dir = run_root / "alpha_cards" / "dossiers"
        if not candidate_dir.exists():
            return None
        pattern = f"{card.signal.indicator}_trial_{card.source_trial:04d}.txt"
        target = candidate_dir / pattern
        if target.exists():
            return target
        # Fallback: any file containing both the indicator name and the
        # zero-padded trial id. Catches small naming drifts in DossierWriter.
        for p in candidate_dir.glob(f"{card.signal.indicator}*trial_{card.source_trial:04d}*.txt"):
            return p
        return None

    def _append_sync_log(
        self,
        run_root: Path,
        iteration_id: int | None,
        result: SyncResult,
    ) -> None:
        entry: dict[str, Any] = {
            "ts_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "run_root": str(run_root),
            "iteration_id": iteration_id,
            **result.to_dict(),
        }
        self.library_dir.mkdir(parents=True, exist_ok=True)
        with self.sync_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


__all__ = [
    "DEFAULT_LIBRARY_DIR",
    "AlphaCardLibrary",
    "SyncResult",
]
