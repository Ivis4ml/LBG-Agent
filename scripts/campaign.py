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
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _stage_run_dir(out: Path) -> Path:
    """Copy baseline + the live lbg/ package into a fresh out/ dir so we
    don't mutate the repo's strategy.yaml / indicators/ when running
    campaigns. Mirrors scripts/long_discovery.py's staging pattern.
    """
    out.mkdir(parents=True, exist_ok=True)
    for name in (
        "strategy.yaml",
        "policy_interpreter.py",
        "backtest.py",
        "pyproject.toml",
        ".env",
    ):
        src = REPO / name
        if src.exists():
            shutil.copy(src, out / name)

    indicators_dst = out / "indicators"
    if indicators_dst.exists():
        shutil.rmtree(indicators_dst)
    shutil.copytree(REPO / "indicators", indicators_dst)

    lbg_dst = out / "lbg"
    if lbg_dst.exists():
        shutil.rmtree(lbg_dst)
    shutil.copytree(REPO / "lbg", lbg_dst)

    data_dst = out / "data"
    if not data_dst.exists():
        data_dst.symlink_to(REPO / "data")
    knowledge_dst = out / "knowledge"
    if not knowledge_dst.exists():
        knowledge_dst.symlink_to(REPO / "knowledge")

    (out / "memory").mkdir(exist_ok=True)
    (out / "skills").mkdir(exist_ok=True)

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
        (out / "README.md").write_text("campaign baseline\n")
        # Include indicators/ in the baseline commit so revert_to_trial_N can
        # walk back to any earlier point without sma.py becoming an orphan
        # the git-driven restorer would delete. Bug found in v2 campaign:
        # the Editor proposed revert_to_trial_N → revert restored only the
        # committed .py files → sma.py (never committed) got pruned →
        # sealing crashed with "indicator module not found".
        subprocess.run(
            ["git", "add", "README.md", "strategy.yaml", "indicators/"],
            cwd=out,
            check=True,
        )
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
    parser.add_argument(
        "--gate",
        choices=["strict", "permissive"],
        default="strict",
        help="strict = PROPOSAL §7 thresholds (default, use for H1 claims); "
        "permissive = looser min_trades / utility_lcb / drawdown thresholds "
        "to exercise alpha_cards in experimentation",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    run_dir = _stage_run_dir(args.out.resolve())

    log_path = run_dir / "campaign.log"
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s · %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )

    sys.path.insert(0, str(run_dir))
    os.chdir(run_dir)

    from lbg.gate import GateConfig
    from lbg.orchestrator.campaign import CampaignRunner
    from lbg.orchestrator.role_runner import RoleRunner

    runner = RoleRunner(provider=args.provider) if args.provider else RoleRunner()
    gate_config = GateConfig.permissive() if args.gate == "permissive" else GateConfig()
    cr = CampaignRunner(run_dir, runner=runner, gate_config=gate_config)

    t0 = time.monotonic()
    result = cr.run(
        n_iterations=args.iterations,
        budget_per_iteration=args.budget,
    )
    elapsed = time.monotonic() - t0

    print(
        f"\ncampaign complete · iterations={result.n_iterations} "
        f"accepted={result.total_accepted} alpha_cards={result.total_alpha_cards} "
        f"elapsed={elapsed:.0f}s"
    )
    print(f"summary: {run_dir / 'campaigns' / 'campaign_summary.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
