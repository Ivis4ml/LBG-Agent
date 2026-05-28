"""Smoke test for compute_per_card_sealed_validation.

Builds an isolated git repo with a baseline strategy commit and a trial
commit that adds one indicator + an exit_filter to wire it in. Writes a
hand-crafted alpha card pointing at the parent commit and runs the
producer. Asserts:

  - The card's sealed_summary becomes a populated dict with
    `incremental_sharpe_ci_lower` (the key the H1 reader expects).
  - compute_alpha_card_h1_verdict now counts the card when the CI lower
    bound > 0, validating that the producer/consumer pair is wired
    correctly. Whether the lower bound is positive on real SPY bars is
    not asserted -- that depends on data and is not the smoke test's job.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from lbg.alpha_cards import (
    AlphaCard,
    AlphaCardEvidence,
    AlphaCardSignal,
    AlphaCardStatus,
)
from lbg.data.loader import load_split
from lbg.git_manager import GitManager
from lbg.schemas import HypothesisOutcome, ValidationSignal
from lbg.verdict import (
    compute_alpha_card_h1_verdict,
    compute_per_card_sealed_validation,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def card_repo(tmp_path) -> Path:
    """A git repo containing a baseline strategy commit and one trial commit.

    Layout copied minimally from the live repo: strategy.yaml + an
    indicator. The trial commit replaces strategy.yaml with a version
    that adds a second indicator wired through `exit_filters`.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "--initial-branch=main"], cwd=repo, check=True, capture_output=True
    )
    subprocess.run(["git", "config", "user.email", "t@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=repo, check=True)

    # ---- baseline (parent) commit: sma_cross_baseline ----
    (repo / "indicators").mkdir()
    shutil.copy(REPO / "indicators" / "sma.py", repo / "indicators" / "sma.py")
    baseline_yaml = (
        "name: sma_cross_baseline\n"
        "indicators:\n"
        "  - name: sma_fast\n    fn: sma\n    params: {period: 20}\n"
        "  - name: sma_slow\n    fn: sma\n    params: {period: 50}\n"
        "entry: {rule: cross_above, fast: sma_fast, slow: sma_slow}\n"
        "exit: {rule: cross_below, fast: sma_fast, slow: sma_slow}\n"
        "filters: []\n"
        "exit_filters: []\n"
        "sizing: {mode: fixed_fraction, fraction: 1.0, max_position: 1.0}\n"
    )
    (repo / "strategy.yaml").write_text(baseline_yaml)
    subprocess.run(
        ["git", "add", "strategy.yaml", "indicators/sma.py"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "baseline: sma_cross_baseline"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    # ---- trial 0001 commit: add a long-window SMA as an exit_filter ----
    # Use sma with period=200 as a "trend regime" indicator; require it
    # to stay above its own value (always true) so we don't actually
    # filter, but the strategy text + indicator wiring differ.
    trial_yaml = (
        "name: sma_cross_plus_trend\n"
        "indicators:\n"
        "  - name: sma_fast\n    fn: sma\n    params: {period: 20}\n"
        "  - name: sma_slow\n    fn: sma\n    params: {period: 50}\n"
        "  - name: sma_trend\n    fn: sma\n    params: {period: 200}\n"
        "entry: {rule: cross_above, fast: sma_fast, slow: sma_slow}\n"
        "exit: {rule: cross_below, fast: sma_fast, slow: sma_slow}\n"
        "filters: []\n"
        "exit_filters:\n"
        "  - {rule: indicator_below, indicator: sma_trend, threshold: 0.0}\n"
        "sizing: {mode: fixed_fraction, fraction: 1.0, max_position: 1.0}\n"
    )
    (repo / "strategy.yaml").write_text(trial_yaml)
    subprocess.run(["git", "add", "strategy.yaml"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "trial 0001: add_indicator sma_trend"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return repo


def _write_card(repo: Path, parent_sha: str) -> Path:
    cards_dir = repo / "alpha_cards"
    cards_dir.mkdir()
    card = AlphaCard(
        alpha_id="sma_trend_trial_0001",
        source_trial=1,
        source_commit=parent_sha,
        status=AlphaCardStatus.ACCEPTED,
        signal=AlphaCardSignal(
            indicator="sma_trend",
            fn="sma",
            params={"period": 200},
            source_path="indicators/sma.py",
        ),
        evidence=AlphaCardEvidence(
            train_summary={
                "sharpe": 0.1,
                "max_drawdown": -0.1,
                "turnover": 1.0,
                "num_trades": 10.0,
            },
            validation_signal=ValidationSignal.ACCEPTED,
            hypothesis_outcome=HypothesisOutcome.CONFIRMED,
            sealed_summary=None,
        ),
    )
    path = cards_dir / f"trial_{card.source_trial:04d}.yaml"
    path.write_text(
        yaml.safe_dump(card.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def test_producer_populates_sealed_summary_keys(card_repo):
    """The producer writes the exact keys the H1 reader expects."""
    git = GitManager(card_repo)
    parent_sha = git.list_recent_commits(limit=10)[1][0]  # second-most-recent = baseline
    # Resolve to full SHA so source_commit matches `git show <ref>:`.
    parent_full = git._git("rev-parse", parent_sha).strip()
    card_path = _write_card(card_repo, parent_full)

    sealed_df = load_split("split_C")  # any non-trivial window will do
    results = compute_per_card_sealed_validation(card_repo, sealed_df, git=git)

    assert len(results) == 1, "exactly one accepted card"
    r = results[0]
    assert r.error is None, f"validation should succeed, got: {r.error}"

    # Outcome wrapper carries a SPA verdict on the family of valid cards.
    assert results.spa is not None
    assert results.spa.n_candidates == 1
    assert 0.0 < results.spa.spa_p_value <= 1.0
    assert 0.0 < results.spa.spa_p_value_l <= 1.0
    assert 0.0 < results.spa.spa_p_value_u <= 1.0
    # The three Hansen variants must bracket: SPA_l ≥ SPA_c ≥ SPA_u.
    assert results.spa.spa_p_value_l >= results.spa.spa_p_value - 1e-9
    assert results.spa.spa_p_value >= results.spa.spa_p_value_u - 1e-9

    # Persisted on disk?
    raw = yaml.safe_load(card_path.read_text(encoding="utf-8"))
    summary = raw["evidence"]["sealed_summary"]
    assert summary is not None, "sealed_summary must be populated"
    assert "incremental_sharpe_ci_lower" in summary
    assert "incremental_sharpe_ci_upper" in summary
    assert "incremental_sharpe_point" in summary
    # CI bounds must bracket the point estimate.
    assert summary["incremental_sharpe_ci_lower"] <= summary["incremental_sharpe_point"]
    assert summary["incremental_sharpe_ci_upper"] >= summary["incremental_sharpe_point"]

    # BH FDR correction fields (PROPOSAL §20 item 32).
    assert "p_value_one_sided" in summary
    assert "bh_validated" in summary
    assert "bh_q" in summary
    assert summary["bh_q"] == pytest.approx(0.10)
    # p-value must be in the Davison-Hinkley open unit interval.
    assert 0.0 < summary["p_value_one_sided"] <= 1.0
    # bh_validated is a bool, not a truthy float.
    assert isinstance(summary["bh_validated"], bool)
    # With only one card in the family, BH reduces to "p ≤ q".
    expected_bh = summary["p_value_one_sided"] <= summary["bh_q"]
    assert summary["bh_validated"] == expected_bh


def test_h1_consumer_reads_producer_output(card_repo):
    """End-to-end: producer writes -> consumer counts. Closes the round trip."""
    git = GitManager(card_repo)
    parent_full = git._git("rev-parse", "HEAD~1").strip()
    _write_card(card_repo, parent_full)

    sealed_df = load_split("split_C")
    compute_per_card_sealed_validation(card_repo, sealed_df, git=git)

    verdict = compute_alpha_card_h1_verdict(card_repo)
    assert verdict.candidate_card_count == 1
    # The card is either validated (ci_lower > 0) or not -- both honest.
    # Asserting the bool is fine; we just confirm the pipeline ran.
    assert verdict.validated_factor_count in (0, 1)


def test_dsr_attached_when_trial_context_supplied(card_repo):
    """When n_trials + trial_sharpes are passed explicitly, DSR
    populates the sealed_summary and the result dataclass."""
    git = GitManager(card_repo)
    parent_full = git._git("rev-parse", "HEAD~1").strip()
    card_path = _write_card(card_repo, parent_full)

    sealed_df = load_split("split_C")
    # 50 attempted trials with modest cross-section of annualised Sharpes.
    trial_sharpes = [0.5, 0.3, 0.7, 0.2, 0.9] * 10
    outcome = compute_per_card_sealed_validation(
        card_repo,
        sealed_df,
        git=git,
        n_trials_attempted=len(trial_sharpes),
        trial_sharpes_annualised=trial_sharpes,
    )

    assert len(outcome) == 1
    r = outcome[0]
    assert r.error is None
    # DSR populated.
    assert 0.0 <= r.deflated_sharpe_ratio <= 1.0
    assert isinstance(r.dsr_passes, bool)
    assert r.dsr_threshold == pytest.approx(0.95)

    # Persisted to YAML too.
    raw = yaml.safe_load(card_path.read_text(encoding="utf-8"))
    summary = raw["evidence"]["sealed_summary"]
    assert "deflated_sharpe_ratio" in summary
    assert "dsr_passes" in summary
    assert "dsr_threshold" in summary
    assert isinstance(summary["dsr_passes"], bool)
    assert 0.0 <= summary["deflated_sharpe_ratio"] <= 1.0


def test_missing_trial_commit_records_error(card_repo, tmp_path):
    """A card pointing at a trial that has no `trial NNNN:` commit yields
    a validation_error rather than crashing the seal step."""
    git = GitManager(card_repo)
    parent_full = git._git("rev-parse", "HEAD~1").strip()
    cards_dir = card_repo / "alpha_cards"
    cards_dir.mkdir(exist_ok=True)
    # Hand-write a card whose source_trial does not match any commit subject.
    card = AlphaCard(
        alpha_id="orphan",
        source_trial=999,
        source_commit=parent_full,
        status=AlphaCardStatus.ACCEPTED,
        signal=AlphaCardSignal(
            indicator="sma_fast",
            fn="sma",
            source_path="indicators/sma.py",
        ),
        evidence=AlphaCardEvidence(
            train_summary={
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "turnover": 0.0,
                "num_trades": 0.0,
            },
            validation_signal=ValidationSignal.ACCEPTED,
            hypothesis_outcome=HypothesisOutcome.CONFIRMED,
            sealed_summary=None,
        ),
    )
    (cards_dir / "trial_0999.yaml").write_text(
        yaml.safe_dump(card.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )

    sealed_df = load_split("split_C")
    results = compute_per_card_sealed_validation(card_repo, sealed_df, git=git)
    orphan = next(r for r in results if r.alpha_id == "orphan")
    assert orphan.error is not None
    assert "trial commit" in orphan.error

    # The failure reason is persisted to the card's own audit trail, not
    # dropped to a bare None (codex review, Issue 4).
    raw = yaml.safe_load((cards_dir / "trial_0999.yaml").read_text(encoding="utf-8"))
    summary = raw["evidence"]["sealed_summary"]
    assert summary is not None
    assert "trial commit" in summary["validation_error"]
