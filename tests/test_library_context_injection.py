"""Tests for the library-state injection into Editor context.

When ContextBuilder is constructed with `library_dir=<path>`, the
EditorContext exposes a `library_cards` list summarising whatever cards
are in `<path>/index.jsonl`. The Editor user-prompt renderer surfaces
this as an "Alpha cards already in the library" section so the model
knows what already exists and is told to look for complementary
factors.

When `library_dir` is None (the default for single-Discovery runs that
aren't driven by CampaignRunner), the list is empty and no section is
rendered -- backward-compatible behaviour.
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
from lbg.dsl import load_strategy
from lbg.memory import MemoryManager
from lbg.orchestrator.context_builder import ContextBuilder
from lbg.orchestrator.role_runner import _format_library_cards
from lbg.schemas import HypothesisOutcome, ValidationSignal

REPO = Path(__file__).resolve().parents[1]


def _seed_library(library_dir: Path, alpha_id: str, fn: str, point: float | None) -> None:
    """Drop a single card into a library directory the way AlphaCardLibrary would."""
    library_dir.mkdir(parents=True, exist_ok=True)
    cards_dir = library_dir / "cards"
    indicators_dir = library_dir / "indicators"
    cards_dir.mkdir(exist_ok=True)
    indicators_dir.mkdir(exist_ok=True)
    sealed_summary = None
    if point is not None:
        sealed_summary = {
            "incremental_sharpe_point": point,
            "incremental_sharpe_ci_lower": -0.1,
            "incremental_sharpe_ci_upper": 0.6,
            "n_bootstrap": 1000.0,
            "block_len": 10.0,
            "alpha": 0.05,
        }
    card = AlphaCard(
        alpha_id=alpha_id,
        source_trial=3,
        source_commit="a" * 40,
        status=AlphaCardStatus.ACCEPTED,
        signal=AlphaCardSignal(
            indicator=fn,
            fn=fn,
            params={"period": 20},
            source_path=f"indicators/{fn}.py",
        ),
        evidence=AlphaCardEvidence(
            train_summary={
                "sharpe": 0.5,
                "max_drawdown": -0.1,
                "turnover": 1.0,
                "num_trades": 30.0,
            },
            validation_signal=ValidationSignal.ACCEPTED,
            hypothesis_outcome=HypothesisOutcome.CONFIRMED,
            sealed_summary=sealed_summary,
        ),
    )
    (cards_dir / f"{alpha_id}.yaml").write_text(
        yaml.safe_dump(card.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (indicators_dir / f"{fn}.py").write_text(
        f"import pandas as pd\n\ndef {fn}(df, **params):\n    return df['close']\n",
        encoding="utf-8",
    )
    index_entry = {
        "alpha_id": alpha_id,
        "source_trial": 3,
        "source_commit": "a" * 40,
        "indicator": fn,
        "fn": fn,
        "status": "accepted",
        "dossier_factor": None,
        "card_path": f"cards/{alpha_id}.yaml",
        "indicator_path": f"indicators/{fn}.py",
        "dossier_path": None,
    }
    with (library_dir / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(index_entry) + "\n")


@pytest.fixture
def repo_skeleton(tmp_path) -> Path:
    """Tmp dir with strategy.yaml + indicators/ + memory/."""
    (tmp_path / "indicators").mkdir()
    (tmp_path / "indicators" / "sma.py").write_text(
        "import pandas as pd\n\ndef sma(df, period=20):\n    return df['close'].rolling(period).mean()\n",
        encoding="utf-8",
    )
    (tmp_path / "strategy.yaml").write_text(
        (REPO / "strategy.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "memory").mkdir()
    return tmp_path


# -------- context builder injection --------


def test_no_library_dir_means_empty_library_cards(repo_skeleton):
    """Backward compat: ContextBuilder constructed without library_dir
    must produce an empty library_cards list (and the render returns
    an empty-state string, but the section is omitted entirely from
    the user prompt)."""
    mm = MemoryManager(repo_skeleton / "memory")
    cb = ContextBuilder(mm, repo_root=repo_skeleton)
    strategy = load_strategy(repo_skeleton / "strategy.yaml")
    ctx = cb.editor_view(strategy)
    assert ctx.library_cards == []


def test_library_cards_loaded_from_index(repo_skeleton, tmp_path):
    library = tmp_path / "library"
    _seed_library(library, "vol_regime_zscore_trial_0003", "vol_regime_zscore", point=0.271)
    _seed_library(library, "adx_trend_strength_trial_0001", "adx_trend_strength", point=0.12)

    mm = MemoryManager(repo_skeleton / "memory")
    cb = ContextBuilder(mm, repo_root=repo_skeleton, library_dir=library)
    strategy = load_strategy(repo_skeleton / "strategy.yaml")
    ctx = cb.editor_view(strategy)

    assert len(ctx.library_cards) == 2
    fns = {c.fn for c in ctx.library_cards}
    assert fns == {"vol_regime_zscore", "adx_trend_strength"}
    # Sealed point estimates pulled off the card YAMLs.
    by_fn = {c.fn: c for c in ctx.library_cards}
    assert by_fn["vol_regime_zscore"].incremental_sharpe_point == pytest.approx(0.271)
    assert by_fn["adx_trend_strength"].incremental_sharpe_point == pytest.approx(0.12)


def test_library_missing_index_is_safe(repo_skeleton, tmp_path):
    """library_dir pointing at a path with no index.jsonl is the case
    for a *fresh* library (no campaign has run yet). Should produce
    empty list, not crash."""
    library = tmp_path / "library"
    library.mkdir()
    mm = MemoryManager(repo_skeleton / "memory")
    cb = ContextBuilder(mm, repo_root=repo_skeleton, library_dir=library)
    strategy = load_strategy(repo_skeleton / "strategy.yaml")
    assert cb.editor_view(strategy).library_cards == []


def test_library_cards_dedupe_by_alpha_id(repo_skeleton, tmp_path):
    """Re-syncing the same card into the index would append duplicate
    lines; the loader must dedupe on alpha_id so the Editor doesn't
    see double-counted factors."""
    library = tmp_path / "library"
    _seed_library(library, "x_trial_0001", "x_fn", point=0.1)
    # Append the SAME alpha_id row a second time -- simulates a re-sync.
    with (library / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "alpha_id": "x_trial_0001",
                    "source_trial": 1,
                    "source_commit": "a" * 40,
                    "indicator": "x_fn",
                    "fn": "x_fn",
                    "status": "accepted",
                    "dossier_factor": None,
                    "card_path": "cards/x_trial_0001.yaml",
                }
            )
            + "\n"
        )
    mm = MemoryManager(repo_skeleton / "memory")
    cb = ContextBuilder(mm, repo_root=repo_skeleton, library_dir=library)
    strategy = load_strategy(repo_skeleton / "strategy.yaml")
    cards = cb.editor_view(strategy).library_cards
    assert len(cards) == 1
    assert cards[0].alpha_id == "x_trial_0001"


# -------- rendering --------


def test_format_library_cards_empty_message():
    assert "library is empty" in _format_library_cards([])


def test_format_library_cards_lists_fn_and_dossier():
    from lbg.orchestrator.context_builder import LibraryCardSummary

    cards = [
        LibraryCardSummary(
            alpha_id="vol_regime_zscore_trial_0003",
            fn="vol_regime_zscore",
            indicator_name="vol_regime_zscore",
            dossier_factor="VolRegime",
            incremental_sharpe_point=0.271,
        ),
        LibraryCardSummary(
            alpha_id="adx_trial_0001",
            fn="adx",
            indicator_name="adx_14",
            dossier_factor=None,
            incremental_sharpe_point=None,
        ),
    ]
    text = _format_library_cards(cards)
    assert "vol_regime_zscore" in text
    assert "adx" in text
    assert "VolRegime" in text  # dossier link surfaced when present
    assert "+0.271" in text  # point estimate rendered
    assert "complementary" in text or "complementary signal" in text


def test_editor_system_prompt_documents_library_rule():
    """The system prompt must spell out the new library-state rule next
    to the banned-name rule, so the Editor knows the library section
    in its user prompt isn't decoration."""
    sys_prompt = (REPO / "lbg" / "orchestrator" / "prompts" / "editor_system.md").read_text(
        encoding="utf-8"
    )
    assert "Alpha card library state" in sys_prompt
    assert "complementary" in sys_prompt.lower()
