"""Tests for CampaignRunner._inject_library_into_baseline.

PROPOSAL §20 lock #30 promises that validated library factors enter the
seed pool of subsequent campaigns. This module realises that promise:
at iter 1 trial 0, the baseline strategy is augmented with each
accepted library card's indicator + its recorded attach config.

Pinned behaviour:
  - empty library = noop
  - one card with attach_target=exit + rearm_threshold = exit_filter
    added with rearm
  - one card with attach_target=entry = entry filter added
  - already-present fn (same name in baseline) = skipped (no duplicate)
  - card.attach_config = None = skipped (legacy schema)
  - indicator source .py copied from library/indicators into the run
  - subsequent _reset_for_iteration restores the AUGMENTED baseline
    (i.e. the library state is now part of the campaign's start)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lbg.alpha_card_library import AlphaCardLibrary
from lbg.alpha_cards import (
    AlphaCard,
    AlphaCardAttach,
    AlphaCardEvidence,
    AlphaCardSignal,
    AlphaCardStatus,
)
from lbg.dsl import load_strategy
from lbg.dsl.schema import (
    IndicatorAboveFilter,
)
from lbg.orchestrator.campaign import CampaignRunner
from lbg.schemas import HypothesisOutcome, ValidationSignal

REPO = Path(__file__).resolve().parents[1]


def _seed_card(
    library: AlphaCardLibrary,
    *,
    alpha_id: str,
    fn: str,
    indicator_name: str,
    rule: str = "indicator_above",
    threshold: float = 1.5,
    rearm_threshold: float | None = 0.5,
    target: str = "exit",
    indicator_source: str = "import pandas as pd\ndef X(df, **params):\n    return df['close']\n",
) -> None:
    """Drop a fully-formed card + indicator source into the library."""
    library._ensure_dirs()
    card = AlphaCard(
        alpha_id=alpha_id,
        source_trial=3,
        source_commit="a" * 40,
        status=AlphaCardStatus.ACCEPTED,
        signal=AlphaCardSignal(
            indicator=indicator_name,
            fn=fn,
            params={"period": 20},
            source_path=f"indicators/{fn}.py",
        ),
        evidence=AlphaCardEvidence(
            train_summary={
                "sharpe": 0.9,
                "max_drawdown": -0.1,
                "turnover": 1.0,
                "num_trades": 40.0,
            },
            validation_signal=ValidationSignal.ACCEPTED,
            hypothesis_outcome=HypothesisOutcome.CONFIRMED,
            sealed_summary=None,
        ),
        attach_config=AlphaCardAttach(
            rule=rule,
            threshold=threshold,
            rearm_threshold=rearm_threshold,
            target=target,
        ),
    )
    (library.cards_dir / f"{alpha_id}.yaml").write_text(
        yaml.safe_dump(card.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (library.indicators_dir / f"{fn}.py").write_text(
        indicator_source.replace("X", fn), encoding="utf-8"
    )
    with library.index_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "alpha_id": alpha_id,
                    "source_trial": 3,
                    "source_commit": "a" * 40,
                    "indicator": indicator_name,
                    "fn": fn,
                    "status": "accepted",
                    "dossier_factor": None,
                    "card_path": f"cards/{alpha_id}.yaml",
                    "indicator_path": f"indicators/{fn}.py",
                }
            )
            + "\n"
        )


@pytest.fixture
def repo_skeleton(tmp_path) -> Path:
    """Minimal repo skeleton with sma_cross baseline, git-initialised.

    git init + an initial commit is required so the auto-inject's own
    commit (added in Bug (a) fix) has a parent to extend. The fixture
    used to skip git, but every real campaign runs against an
    initialised repo, and the missing-git path was masking the bug.
    """
    import subprocess

    (tmp_path / "indicators").mkdir()
    (tmp_path / "indicators" / "sma.py").write_text(
        (REPO / "indicators" / "sma.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "strategy.yaml").write_text(
        (REPO / "strategy.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    subprocess.run(
        ["git", "init", "--initial-branch=main", "-q"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "config", "user.email", "t@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "indicators/", "strategy.yaml"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "baseline"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    return tmp_path


# -------- empty library --------


def test_inject_empty_library_is_noop(repo_skeleton, tmp_path):
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    cr = CampaignRunner(repo_skeleton, library_dir=library_dir)
    n = cr._inject_library_into_baseline()
    assert n == 0
    # strategy.yaml unchanged
    s = load_strategy(repo_skeleton / "strategy.yaml")
    assert len(s.indicators) == 2  # sma_fast + sma_slow
    assert s.exit_filters == []


def test_inject_no_library_dir_is_noop(repo_skeleton):
    cr = CampaignRunner(repo_skeleton, library_dir=None)
    assert cr._inject_library_into_baseline() == 0


def test_inject_commits_indicator_files_to_git(repo_skeleton, tmp_path):
    """Bug (a) regression: auto-inject must commit indicator .py + strategy.yaml
    so subsequent trial commits have a parent that knows about the
    library factors. Without this, `git show <parent>:indicators/<fn>.py`
    in per_card sealed validation fails with exit 128.
    """
    import subprocess

    library_dir = tmp_path / "library"
    library = AlphaCardLibrary(library_dir)
    _seed_card(
        library,
        alpha_id="vol_regime_zscore_trial_0003",
        fn="vol_regime_zscore",
        indicator_name="vol_regime_zscore",
        target="exit",
    )

    head_before = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_skeleton,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    cr = CampaignRunner(repo_skeleton, library_dir=library_dir)
    n = cr._inject_library_into_baseline()
    assert n == 1

    head_after = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_skeleton,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert head_after != head_before, "auto-inject must produce a new commit"

    # The committed tree must include the injected indicator .py.
    show = subprocess.run(
        ["git", "show", f"{head_after}:indicators/vol_regime_zscore.py"],
        cwd=repo_skeleton,
        check=False,
        capture_output=True,
        text=True,
    )
    assert show.returncode == 0, f"file missing from commit tree: {show.stderr}"
    assert "vol_regime_zscore" in show.stdout

    # And the strategy.yaml in the commit must reflect the inject too.
    show_strat = subprocess.run(
        ["git", "show", f"{head_after}:strategy.yaml"],
        cwd=repo_skeleton,
        check=False,
        capture_output=True,
        text=True,
    )
    assert show_strat.returncode == 0
    assert "vol_regime_zscore" in show_strat.stdout


# -------- single-card injection --------


def test_inject_one_exit_card_with_rearm(repo_skeleton, tmp_path):
    library_dir = tmp_path / "library"
    library = AlphaCardLibrary(library_dir)
    _seed_card(
        library,
        alpha_id="vol_regime_zscore_trial_0003",
        fn="vol_regime_zscore",
        indicator_name="vol_regime_zscore",
        rule="indicator_above",
        threshold=1.5,
        rearm_threshold=0.5,
        target="exit",
    )

    cr = CampaignRunner(repo_skeleton, library_dir=library_dir)
    n = cr._inject_library_into_baseline()

    assert n == 1
    # strategy.yaml augmented
    s = load_strategy(repo_skeleton / "strategy.yaml")
    fns = {spec.fn for spec in s.indicators}
    assert "vol_regime_zscore" in fns
    # exit_filter wired with rearm
    assert len(s.exit_filters) == 1
    flt = s.exit_filters[0]
    assert isinstance(flt, IndicatorAboveFilter)
    assert flt.indicator == "vol_regime_zscore"
    assert flt.threshold == 1.5
    assert flt.rearm_threshold == 0.5
    # Indicator source copied into run's indicators/
    assert (repo_skeleton / "indicators" / "vol_regime_zscore.py").exists()


def test_inject_entry_card(repo_skeleton, tmp_path):
    library_dir = tmp_path / "library"
    library = AlphaCardLibrary(library_dir)
    _seed_card(
        library,
        alpha_id="ema_above_sma_trial_0001",
        fn="ema_above_sma",
        indicator_name="ema_above_sma",
        rule="indicator_above",
        threshold=0.0,
        rearm_threshold=None,
        target="entry",
    )

    cr = CampaignRunner(repo_skeleton, library_dir=library_dir)
    n = cr._inject_library_into_baseline()

    assert n == 1
    s = load_strategy(repo_skeleton / "strategy.yaml")
    assert len(s.filters) == 1
    assert s.exit_filters == []
    assert s.filters[0].indicator == "ema_above_sma"


# -------- skip cases --------


def test_inject_skips_existing_fn(repo_skeleton, tmp_path):
    """A card whose fn duplicates one in the baseline is not re-injected."""
    library_dir = tmp_path / "library"
    library = AlphaCardLibrary(library_dir)
    # `sma` already in baseline (sma_fast + sma_slow both use fn=sma).
    _seed_card(library, alpha_id="sma_trial_0001", fn="sma", indicator_name="sma_lib")
    cr = CampaignRunner(repo_skeleton, library_dir=library_dir)
    assert cr._inject_library_into_baseline() == 0


def test_inject_skips_card_without_attach_config(repo_skeleton, tmp_path):
    """Legacy cards (pre-attach_config schema) are skipped; we can't
    guess the wiring."""
    library_dir = tmp_path / "library"
    library = AlphaCardLibrary(library_dir)
    library._ensure_dirs()
    card = AlphaCard(
        alpha_id="legacy_trial_0001",
        source_trial=1,
        source_commit="b" * 40,
        status=AlphaCardStatus.ACCEPTED,
        signal=AlphaCardSignal(
            indicator="legacy",
            fn="legacy",
            params={},
            source_path="indicators/legacy.py",
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
            sealed_summary=None,
        ),
        # attach_config explicitly omitted
    )
    (library.cards_dir / "legacy_trial_0001.yaml").write_text(
        yaml.safe_dump(card.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    (library.indicators_dir / "legacy.py").write_text("def legacy(df, **p): return df['close']\n")
    with library.index_path.open("a") as f:
        f.write(
            json.dumps(
                {
                    "alpha_id": "legacy_trial_0001",
                    "source_trial": 1,
                    "source_commit": "b" * 40,
                    "indicator": "legacy",
                    "fn": "legacy",
                    "status": "accepted",
                    "card_path": "cards/legacy_trial_0001.yaml",
                }
            )
            + "\n"
        )
    cr = CampaignRunner(repo_skeleton, library_dir=library_dir)
    assert cr._inject_library_into_baseline() == 0


# -------- snapshot side effect --------


def test_inject_updates_baseline_snapshot(repo_skeleton, tmp_path):
    """After injection, subsequent _reset_for_iteration must restore the
    AUGMENTED baseline (not the pre-inject one). Otherwise carry_strategy
    semantics get confusing in iter 2+."""
    library_dir = tmp_path / "library"
    library = AlphaCardLibrary(library_dir)
    _seed_card(
        library,
        alpha_id="vol_regime_zscore_trial_0003",
        fn="vol_regime_zscore",
        indicator_name="vol_regime_zscore",
        target="exit",
    )
    cr = CampaignRunner(repo_skeleton, library_dir=library_dir)
    cr._inject_library_into_baseline()

    # baseline snapshot now contains the injected indicator
    assert "vol_regime_zscore" in cr.baseline_strategy_yaml
    assert "vol_regime_zscore.py" in cr.baseline_indicator_files

    # Mutate strategy + remove file, then reset, then verify restoration.
    (repo_skeleton / "strategy.yaml").write_text("name: garbage\n", encoding="utf-8")
    (repo_skeleton / "indicators" / "vol_regime_zscore.py").unlink()
    cr._reset_for_iteration()
    s = load_strategy(repo_skeleton / "strategy.yaml")
    assert any(spec.fn == "vol_regime_zscore" for spec in s.indicators)
    assert (repo_skeleton / "indicators" / "vol_regime_zscore.py").exists()
