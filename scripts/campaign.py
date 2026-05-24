"""Run a Discovery campaign: K iterations, persisted skills + alpha cards.

Usage:
    uv run python scripts/campaign.py --iterations 3 --budget 5 --out /tmp/lbg_camp

Each iteration starts from the baseline strategy + indicator snapshot, with
event memory wiped but semantic memory (.md files) and alpha_cards/ kept.
The Atari-style "keep playing on failure" lives here.

Output:
    <out>/campaigns/iteration_NNN/sealed_test_final.json   (one per iteration)
    <out>/campaigns/campaign_summary.json                  (aggregated)
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from lbg.orchestrator.campaign import CampaignRunner
from lbg.orchestrator.role_runner import RoleRunner

REPO = Path(__file__).resolve().parent.parent


def _stage_run_dir(out: Path) -> Path:
    """Copy the baseline working tree into a fresh out/ dir so we don't
    mutate the repo's strategy.yaml / indicators/ when running campaigns.

    Mirrors scripts/long_discovery.py's staging pattern.
    """
    out.mkdir(parents=True, exist_ok=True)
    for name in ("strategy.yaml",):
        shutil.copy(REPO / name, out / name)
    indicators_src = REPO / "indicators"
    indicators_dst = out / "indicators"
    indicators_dst.mkdir(exist_ok=True)
    for p in indicators_src.glob("*.py"):
        shutil.copy(p, indicators_dst / p.name)
    # data is read-only; symlink rather than copy.
    data_dst = out / "data"
    if not data_dst.exists():
        data_dst.symlink_to(REPO / "data")
    # knowledge/factors is the read-only seed library; symlink too.
    knowledge_dst = out / "knowledge"
    if not knowledge_dst.exists():
        knowledge_dst.symlink_to(REPO / "knowledge")
    (out / "memory").mkdir(exist_ok=True)
    (out / "skills").mkdir(exist_ok=True)

    # Minimal git for GitManager.
    import subprocess

    if not (out / ".git").exists():
        subprocess.run(
            ["git", "init", "--initial-branch=main", "-q"],
            cwd=out,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "config", "user.email", "campaign@lbg"], cwd=out, check=True)
        subprocess.run(["git", "config", "user.name", "campaign"], cwd=out, check=True)
        subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=out, check=True)
        subprocess.run(["git", "add", "."], cwd=out, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "campaign baseline"],
            cwd=out,
            check=True,
            capture_output=True,
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--budget", type=int, default=5, help="trial budget per iteration")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provider", type=str, default=None, help="anthropic | mimo")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s · %(message)s",
    )

    run_dir = _stage_run_dir(args.out.resolve())
    runner = RoleRunner(provider=args.provider) if args.provider else RoleRunner()
    cr = CampaignRunner(run_dir, runner=runner)
    result = cr.run(
        n_iterations=args.iterations,
        budget_per_iteration=args.budget,
    )

    print(
        f"campaign complete · iterations={result.n_iterations} "
        f"accepted={result.total_accepted} alpha_cards={result.total_alpha_cards}"
    )
    print(f"summary: {run_dir / 'campaigns' / 'campaign_summary.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
