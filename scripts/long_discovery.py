"""Long Discovery run — runs in a tmp dir to keep the real repo clean.

Usage:
    uv run python scripts/long_discovery.py --budget 20 --out /tmp/lbg_long_run

Produces under `--out`:
    - strategy.yaml          (final incumbent)
    - memory/                (jsonl + md)
    - artifacts/sealed/      (sealed_test_final.json)
    - artifacts/reports/     (discovery_report.html)
    - runs/NNNN/             (per-trial editor + reflector yaml)
    - summary.json           (high-level result)
    - run.log                (info-level logging)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="long_discovery")
    p.add_argument("--budget", type=int, default=20)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path("/Users/xinyu/Code/LBG-Agent"))
    p.add_argument(
        "--provider",
        type=str,
        default=None,
        help="anthropic | mimo | mimo_tp | claude_cli | codex_cli. Defaults to "
        "the LBG_PROVIDER env or anthropic. claude_cli routes through the local "
        "`claude -p` binary; mimo_tp uses the Token-Plan SGP endpoint.",
    )
    p.add_argument("--model", type=str, default=None, help="model name (e.g. claude-opus-4-7)")
    p.add_argument(
        "--library-dir",
        type=Path,
        default=None,
        help="durable alpha-card library to sync into (default: <repo>/alpha_cards_library/). "
        "Pass --library-dir '' to disable persistence.",
    )
    args = p.parse_args(argv)

    out: Path = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    logger = logging.getLogger("long_discovery")

    repo: Path = args.repo.resolve()

    # Copy repo skeleton into out/.
    for name in ("strategy.yaml", "policy_interpreter.py", "backtest.py", "pyproject.toml", ".env"):
        src = repo / name
        if src.exists():
            shutil.copy(src, out / name)
    if (out / "indicators").exists():
        shutil.rmtree(out / "indicators")
    shutil.copytree(repo / "indicators", out / "indicators")
    if (out / "lbg").exists():
        shutil.rmtree(out / "lbg")
    shutil.copytree(repo / "lbg", out / "lbg")
    data_link = out / "data"
    if not data_link.exists():
        data_link.symlink_to(repo / "data")
    (out / "memory").mkdir(exist_ok=True)
    (out / "skills").mkdir(exist_ok=True)

    # Init git repo for one-commit-per-trial.
    if not (out / ".git").exists():
        subprocess.run(["git", "init", "--initial-branch=main", "-q"], cwd=out, check=True)
        subprocess.run(["git", "config", "user.email", "discovery@long"], cwd=out, check=True)
        subprocess.run(["git", "config", "user.name", "discovery"], cwd=out, check=True)
        subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=out, check=True)
        (out / "README.md").write_text("long discovery run\n")
        subprocess.run(
            ["git", "add", "README.md", "strategy.yaml", "indicators/"],
            cwd=out,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-q", "-m", "baseline"],
            cwd=out,
            check=True,
            capture_output=True,
        )

    sys.path.insert(0, str(out))
    os.chdir(out)

    from lbg.orchestrator.discovery import Discovery
    from lbg.orchestrator.role_runner import RoleRunner
    from lbg.sealed_vault import SealedVault

    logger.info("starting long Discovery run, budget=%d", args.budget)
    t0 = time.monotonic()
    runner_kwargs = {}
    if args.provider:
        runner_kwargs["provider"] = args.provider
    if args.model:
        runner_kwargs["model"] = args.model
    runner = RoleRunner(**runner_kwargs) if runner_kwargs else None
    disc = Discovery(repo_root=out, runner=runner) if runner else Discovery(repo_root=out)
    vault = SealedVault(out / "artifacts/sealed/sealed_test_final.json")
    result = disc.run(
        budget=args.budget,
        seal_at_end=True,
        vault=vault,
        report_path=out / "artifacts/reports/discovery_report.html",
    )
    elapsed = time.monotonic() - t0
    logger.info("elapsed: %.1f s", elapsed)

    # Sync newly-emitted alpha cards into the durable library so they
    # survive /tmp cleanup. Same default as scripts/campaign.py: the repo's
    # own alpha_cards_library/. Pass --library-dir '' to opt out.
    if args.library_dir is None:
        library_dir: Path | None = args.repo.resolve() / "alpha_cards_library"
    elif str(args.library_dir).strip() == "":
        library_dir = None
    else:
        library_dir = args.library_dir.resolve()
    if library_dir is not None:
        from lbg.alpha_card_library import AlphaCardLibrary

        sync = AlphaCardLibrary(library_dir).sync_from_run(out, iteration_id=None)
        logger.info(
            "library sync: +%d new cards (%d dup), +%d dossiers, total %d",
            sync.new_cards,
            sync.duplicate_cards,
            sync.new_dossiers,
            sync.library_total,
        )

    summary = {
        "budget": args.budget,
        "elapsed_sec": round(elapsed, 1),
        "trials_attempted": result.n_trials_attempted,
        "accepted": result.n_accepted,
        "rejected": result.n_rejected,
        "invariant_failures": result.n_invariant_failures,
        "aborted": result.n_aborted,
        "accepted_trial_ids": result.accepted_trial_ids,
        "final_train_sharpe": result.final_incumbent_train_sharpe,
        "final_val_sharpe": result.final_incumbent_val_sharpe,
        "sealed_metrics": result.sealed_metrics,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Print log lines for quick review.
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
