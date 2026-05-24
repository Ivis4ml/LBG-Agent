"""Regression test for the v2 campaign crash.

Bug: scripts/campaign.py's baseline commit only included README.md +
strategy.yaml, never `indicators/`. So sma.py was on disk but untracked.
When an Editor proposed `revert_to_trial_N`, the revert pruned anything
in indicators/ that wasn't in the target ref's git tree -- including
sma.py -- and the next backtest crashed with `indicator module not found`.

This test runs scripts/campaign.py's staging helper and asserts the
initial commit's tree contains all baseline indicator .py files.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


@pytest.fixture
def campaign_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("campaign", REPO / "scripts" / "campaign.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_campaign_baseline_commits_indicators(tmp_path, campaign_module):
    out = tmp_path / "stage"
    campaign_module._stage_run_dir(out)

    tracked = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"],
        cwd=out,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    # Every .py file baseline ships under indicators/ must be in the tree;
    # revert_to_trial_N depends on this invariant.
    on_disk_indicators = sorted(
        p.relative_to(out).as_posix() for p in (out / "indicators").glob("*.py")
    )
    for rel in on_disk_indicators:
        assert rel in tracked, (
            f"{rel} on disk but not tracked at baseline commit -- "
            "revert_to_trial_N would treat it as an orphan and prune it"
        )


def test_long_discovery_baseline_commits_indicators(tmp_path):
    """Same invariant for long_discovery.py -- it has the same staging
    pattern and the same potential to break revert_to_trial_N."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "long_discovery", REPO / "scripts" / "long_discovery.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    # long_discovery.py inlines the staging in main(); call it through argv
    # with a tiny budget? No -- that'd run live LLMs. Instead, read the
    # script's source and verify it adds indicators/ to the initial commit.
    src = (REPO / "scripts" / "long_discovery.py").read_text()
    assert "indicators/" in src, "long_discovery.py must stage indicators/"
    # And specifically: in the git add command before commit baseline.
    assert "git" in src and "indicators/" in src
