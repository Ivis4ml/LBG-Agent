"""Tests for the persistent alpha-card library.

The library lives at `<repo>/alpha_cards_library/` by default and is the
durable layer that catches per-run `alpha_cards/` before the `/tmp` run
directory gets cleared. Tests pin:

  - First sync copies cards + indicator source into the library.
  - Re-syncing the same run is idempotent (no duplicates).
  - Translator dossiers come along when present.
  - Missing run dir is a clean no-op (with audit entry).
  - The index is append-only JSONL with one line per card.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lbg.alpha_card_library import AlphaCardLibrary
from lbg.alpha_cards import (
    AlphaCard,
    AlphaCardEvidence,
    AlphaCardSignal,
    AlphaCardStatus,
)
from lbg.schemas import HypothesisOutcome, ValidationSignal


def _make_card(alpha_id: str, fn: str, trial: int) -> AlphaCard:
    return AlphaCard(
        alpha_id=alpha_id,
        source_trial=trial,
        source_commit="a" * 40,
        status=AlphaCardStatus.ACCEPTED,
        signal=AlphaCardSignal(
            indicator=alpha_id.split("_trial")[0],
            fn=fn,
            params={"period": 14},
            source_path=f"indicators/{fn}.py",
        ),
        evidence=AlphaCardEvidence(
            train_summary={
                "sharpe": 0.5,
                "max_drawdown": -0.1,
                "turnover": 1.0,
                "num_trades": 50.0,
            },
            validation_signal=ValidationSignal.ACCEPTED,
            hypothesis_outcome=HypothesisOutcome.CONFIRMED,
            sealed_summary=None,
        ),
    )


@pytest.fixture
def run_with_one_card(tmp_path) -> Path:
    """A minimal run tree with one accepted alpha card + its indicator source."""
    run = tmp_path / "run"
    (run / "alpha_cards").mkdir(parents=True)
    (run / "indicators").mkdir()

    card = _make_card("rsi_14_trial_0003", fn="rsi", trial=3)
    card_path = run / "alpha_cards" / "trial_0003.yaml"
    card_path.write_text(
        yaml.safe_dump(card.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (run / "indicators" / "rsi.py").write_text(
        "import pandas as pd\n\ndef rsi(df, period=14):\n    return df['close']\n",
        encoding="utf-8",
    )
    return run


# -------- happy path --------


def test_first_sync_copies_card_and_indicator_into_library(run_with_one_card, tmp_path):
    library = AlphaCardLibrary(tmp_path / "library")
    result = library.sync_from_run(run_with_one_card)

    assert result.new_cards == 1
    assert result.duplicate_cards == 0
    assert result.library_total == 1
    assert (library.cards_dir / "rsi_14_trial_0003.yaml").exists()
    assert (library.indicators_dir / "rsi.py").exists()


def test_index_jsonl_has_one_line_per_card(run_with_one_card, tmp_path):
    library = AlphaCardLibrary(tmp_path / "library")
    library.sync_from_run(run_with_one_card)
    lines = library.index_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["alpha_id"] == "rsi_14_trial_0003"
    assert entry["fn"] == "rsi"
    assert entry["card_path"] == "cards/rsi_14_trial_0003.yaml"
    assert entry["indicator_path"] == "indicators/rsi.py"


def test_resync_is_idempotent(run_with_one_card, tmp_path):
    library = AlphaCardLibrary(tmp_path / "library")
    library.sync_from_run(run_with_one_card)
    second = library.sync_from_run(run_with_one_card)
    assert second.new_cards == 0
    assert second.duplicate_cards == 1
    # The index still has one line (not two).
    lines = library.index_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_load_cards_returns_persisted_cards(run_with_one_card, tmp_path):
    library = AlphaCardLibrary(tmp_path / "library")
    library.sync_from_run(run_with_one_card)
    cards = library.load_cards()
    assert len(cards) == 1
    assert cards[0].alpha_id == "rsi_14_trial_0003"


def test_dossier_is_copied_when_present(run_with_one_card, tmp_path):
    dossier_dir = run_with_one_card / "alpha_cards" / "dossiers"
    dossier_dir.mkdir()
    dossier_dir.joinpath("rsi_14_trial_0003.txt").write_text(
        '{"factor_name": "RSI", "lbg_provenance": {}}', encoding="utf-8"
    )
    library = AlphaCardLibrary(tmp_path / "library")
    result = library.sync_from_run(run_with_one_card)
    assert result.new_dossiers == 1
    assert (library.dossiers_dir / "rsi_14_trial_0003.txt").exists()


# -------- multiple runs feeding into one library --------


def test_two_different_runs_accumulate_in_library(run_with_one_card, tmp_path):
    library = AlphaCardLibrary(tmp_path / "library")
    library.sync_from_run(run_with_one_card)

    # Second independent run with a different card.
    run2 = tmp_path / "run2"
    (run2 / "alpha_cards").mkdir(parents=True)
    (run2 / "indicators").mkdir()
    card = _make_card("adx_14_trial_0001", fn="adx", trial=1)
    (run2 / "alpha_cards" / "trial_0001.yaml").write_text(
        yaml.safe_dump(card.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (run2 / "indicators" / "adx.py").write_text(
        "import pandas as pd\n\ndef adx(df, period=14):\n    return df['close']\n",
        encoding="utf-8",
    )

    result = library.sync_from_run(run2)
    assert result.new_cards == 1
    assert result.library_total == 2
    assert library.existing_alpha_ids() == {"rsi_14_trial_0003", "adx_14_trial_0001"}


# -------- empty / missing --------


def test_missing_run_alpha_cards_dir_is_noop(tmp_path):
    """A campaign that emitted no cards must not crash the sync."""
    library = AlphaCardLibrary(tmp_path / "library")
    result = library.sync_from_run(tmp_path / "empty_run")
    assert result.new_cards == 0
    assert result.duplicate_cards == 0
    # Sync log was still written (audit trail).
    assert library.sync_log_path.exists()
    log_lines = library.sync_log_path.read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 1
    entry = json.loads(log_lines[0])
    assert entry["new_cards"] == 0


def test_existing_alpha_ids_handles_missing_index(tmp_path):
    library = AlphaCardLibrary(tmp_path / "library")
    assert library.existing_alpha_ids() == set()


def test_load_cards_handles_missing_index(tmp_path):
    library = AlphaCardLibrary(tmp_path / "library")
    assert library.load_cards() == []


def test_sync_log_records_one_line_per_sync(run_with_one_card, tmp_path):
    library = AlphaCardLibrary(tmp_path / "library")
    library.sync_from_run(run_with_one_card)
    library.sync_from_run(run_with_one_card)
    lines = library.sync_log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    second = json.loads(lines[1])
    assert second["new_cards"] == 0
    assert second["duplicate_cards"] == 1
