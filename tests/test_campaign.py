"""C · CampaignRunner: tests for the multi-iteration loop.

The key invariants we pin:
  1. Each iteration begins with the baseline strategy.yaml + indicators/.
  2. Event memory (trials.jsonl etc.) is wiped between iterations.
  3. Semantic memory (.md files) PERSISTS -- that's the whole point.
  4. The per-iteration SealedVault writes under campaigns/iteration_NNN/.
  5. The campaign summary is dumped to campaigns/campaign_summary.json.

The Discovery itself is exercised with a runner that always aborts so the
tests stay deterministic and offline (no LLM calls).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from lbg.orchestrator import RoleRunner, RoleRunnerError
from lbg.orchestrator.campaign import CampaignRunner

REPO = Path(__file__).resolve().parents[1]


class _AlwaysAbortingRunner(RoleRunner):
    """Forces every trial to abort. CampaignRunner still needs to run all
    iterations and reset state between them; aborting trials are the easiest
    deterministic stub."""

    def __init__(self):
        super().__init__(api_key="dummy", client=object())
        self.calls: list[int] = []

    def editor(self, context, *, trial_id):  # noqa: ARG002
        self.calls.append(trial_id)
        raise RoleRunnerError("simulated editor abort")


@pytest.fixture
def working_tree(tmp_path):
    """Minimal repo with strategy.yaml, baseline indicators, a data symlink,
    and a one-commit git history."""
    shutil.copy(REPO / "strategy.yaml", tmp_path / "strategy.yaml")
    (tmp_path / "indicators").mkdir()
    shutil.copy(REPO / "indicators" / "sma.py", tmp_path / "indicators" / "sma.py")
    shutil.copy(REPO / "indicators" / "__init__.py", tmp_path / "indicators" / "__init__.py")
    (tmp_path / "data").symlink_to(REPO / "data")
    (tmp_path / "memory").mkdir()
    (tmp_path / "skills").mkdir()

    subprocess.run(
        ["git", "init", "--initial-branch=main", "-q"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("test\n")
    subprocess.run(["git", "add", "README.md", "strategy.yaml"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "baseline"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    return tmp_path


def test_campaign_runs_n_iterations(working_tree):
    runner = _AlwaysAbortingRunner()
    cr = CampaignRunner(working_tree, runner=runner)
    res = cr.run(n_iterations=3, budget_per_iteration=2)
    assert res.n_iterations == 3
    assert len(res.iteration_results) == 3
    assert all(r.discovery_result.n_trials_attempted == 2 for r in res.iteration_results)


def test_campaign_writes_summary_json(working_tree):
    runner = _AlwaysAbortingRunner()
    cr = CampaignRunner(working_tree, runner=runner)
    cr.run(n_iterations=2, budget_per_iteration=1)
    summary_path = working_tree / "campaigns" / "campaign_summary.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text())
    assert data["n_iterations"] == 2
    assert len(data["iterations"]) == 2


def test_event_memory_is_wiped_between_iterations(working_tree):
    """Pre-seed an event-memory file with a junk line; after reset_for_iteration
    runs as part of the campaign, the file should be gone."""
    junk = working_tree / "memory" / "trials.jsonl"
    junk.write_text('{"trial_id": 999}\n')
    runner = _AlwaysAbortingRunner()
    cr = CampaignRunner(working_tree, runner=runner)
    cr.run(n_iterations=1, budget_per_iteration=1)
    # After iteration 1 starts, the seeded junk was wiped. The aborted trials
    # never wrote anything new, so trials.jsonl is either absent or empty.
    if junk.exists():
        assert junk.read_text() == ""


def test_semantic_memory_persists_across_iterations(working_tree):
    """accepted_rules.md placed before the campaign starts must still exist
    after all iterations run -- that's the carry-over the campaign exists for."""
    rules = working_tree / "memory" / "accepted_rules.md"
    rules.write_text("- vol_target works in high-vol regimes\n")
    runner = _AlwaysAbortingRunner()
    cr = CampaignRunner(working_tree, runner=runner)
    cr.run(n_iterations=2, budget_per_iteration=1)
    assert rules.exists()
    assert "vol_target works" in rules.read_text()


def test_baseline_strategy_yaml_restored_each_iteration(working_tree):
    """Mutate strategy.yaml mid-campaign-stage; the next iteration's reset
    must restore the snapshot from __init__ time."""
    runner = _AlwaysAbortingRunner()
    cr = CampaignRunner(working_tree, runner=runner)
    # Corrupt the strategy after the runner has snapshot-captured it.
    (working_tree / "strategy.yaml").write_text("name: corrupted\nbroken: yes\n")
    cr.run(n_iterations=1, budget_per_iteration=1)
    # After the iteration's reset, strategy.yaml is back to the baseline.
    text = (working_tree / "strategy.yaml").read_text()
    assert "sma_cross_baseline" in text


def test_non_baseline_indicators_pruned_each_iteration(working_tree):
    """Drop a fake extra indicator file into indicators/; the iteration's
    reset must delete it."""
    fake = working_tree / "indicators" / "extra_fake.py"
    runner = _AlwaysAbortingRunner()
    cr = CampaignRunner(working_tree, runner=runner)
    fake.write_text("import pandas as pd\ndef extra_fake(df):\n    return df['close']\n")
    cr.run(n_iterations=1, budget_per_iteration=1)
    assert not fake.exists()
    assert (working_tree / "indicators" / "sma.py").exists()


def test_alpha_cards_directory_preserved_between_iterations(working_tree):
    """An alpha card placed before iteration 2 must still exist after."""
    (working_tree / "alpha_cards").mkdir(parents=True)
    seed = working_tree / "alpha_cards" / "trial_0000.yaml"
    seed.write_text("alpha_id: pre_existing\n")
    runner = _AlwaysAbortingRunner()
    cr = CampaignRunner(working_tree, runner=runner)
    cr.run(n_iterations=2, budget_per_iteration=1)
    assert seed.exists()


def test_each_iteration_has_its_own_sealed_vault_path(working_tree):
    """Per-iteration sealed file lands under campaigns/iteration_NNN/."""
    runner = _AlwaysAbortingRunner()
    cr = CampaignRunner(working_tree, runner=runner)
    cr.run(n_iterations=2, budget_per_iteration=1)
    for i in range(2):
        vault = working_tree / "campaigns" / f"iteration_{i:03d}" / "sealed_test_final.json"
        # Even with all-aborting runner, _seal still runs and writes the file
        # (sealed evaluation runs on the incumbent regardless of trial outcomes).
        assert vault.exists(), f"iteration {i} should have written its sealed vault"


def test_total_alpha_cards_aggregates_emitted_per_iteration(working_tree):
    """No trials accept, so 0 cards are emitted across iterations."""
    runner = _AlwaysAbortingRunner()
    cr = CampaignRunner(working_tree, runner=runner)
    res = cr.run(n_iterations=2, budget_per_iteration=2)
    assert res.total_alpha_cards == 0
    assert res.total_accepted == 0
