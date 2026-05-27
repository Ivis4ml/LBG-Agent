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


def _resolve_baseline(name: str) -> tuple[Path, Path]:
    """Resolve a baseline name to its (strategy_yaml, indicators_dir) sources.

    The default `sma_cross` reads the repo-root files (strategy.yaml +
    indicators/sma.py + __init__.py). `buyhold` and any future named
    baseline live under `baselines/<name>/`.
    """
    if name == "sma_cross":
        return REPO / "strategy.yaml", REPO / "indicators"
    candidate = REPO / "baselines" / name
    if not candidate.is_dir():
        raise SystemExit(f"unknown baseline {name!r}; have: sma_cross, {list_baselines()}")
    return candidate / "strategy.yaml", candidate / "indicators"


def list_baselines() -> list[str]:
    root = REPO / "baselines"
    if not root.is_dir():
        return ["sma_cross"]
    return ["sma_cross"] + sorted(p.name for p in root.iterdir() if p.is_dir())


def _stage_run_dir(out: Path, *, baseline: str = "sma_cross") -> Path:
    """Copy baseline + the live lbg/ package into a fresh out/ dir so we
    don't mutate the repo's strategy.yaml / indicators/ when running
    campaigns. Mirrors scripts/long_discovery.py's staging pattern.
    """
    strategy_src, indicators_src = _resolve_baseline(baseline)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(strategy_src, out / "strategy.yaml")
    for name in (
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
    shutil.copytree(indicators_src, indicators_dst)

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
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="anthropic | mimo | mimo_tp | claude_cli | codex_cli. claude_cli "
        "spawns the local `claude -p` binary (no per-call API cost on Max "
        "subscriptions); anthropic uses ANTHROPIC_API_KEY; mimo uses "
        "MIMO_API_KEY; mimo_tp uses MIMO_TP_API_KEY (Token-Plan SGP endpoint).",
    )
    parser.add_argument(
        "--gate",
        choices=["strict", "permissive"],
        default="strict",
        help="strict = PROPOSAL §7 thresholds (default, use for H1 claims); "
        "permissive = looser min_trades / utility_lcb / drawdown thresholds "
        "to exercise alpha_cards in experimentation",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="sma_cross",
        help="starting strategy: sma_cross (repo default) or buyhold "
        "(baselines/buyhold/); see scripts/campaign.py for list",
    )
    parser.add_argument(
        "--seal-only-last",
        action="store_true",
        help="publication mode: open the sealed window only on the final "
        "iteration. Default (dev mode) seals every iteration so the live "
        "dashboard can show H1 progress, but those numbers are NOT "
        "independent evidence (effective n=1 sealed window).",
    )
    parser.add_argument(
        "--carry-strategy",
        action="store_true",
        help="iter 2+ keeps the prior iter's strategy.yaml + indicators/ "
        "instead of resetting to baseline. Enables multi-factor "
        "accumulation (Editor adds ON TOP of previous accepts) -- the "
        "PROPOSAL §17 re-discovery spirit. Event memory still wipes "
        "between iters either way.",
    )
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=None,
        help="durable alpha-card library to sync into (default: <repo>/alpha_cards_library/). "
        "Disable with --library-dir '' to skip persistence entirely.",
    )
    parser.add_argument(
        "--goal",
        type=int,
        default=None,
        help="target number of DISTINCT factors (by fn name) in "
        "alpha_cards_library/. Persisted to goal_state.json in the repo "
        "root. The campaign loop updates goal_state.json + appends to "
        "goal_progress.jsonl after every iteration's library sync. "
        "Default: 5 (or whatever's already in goal_state.json).",
    )
    parser.add_argument(
        "--codex-effort",
        type=str,
        default=None,
        choices=[None, "low", "medium", "high", "xhigh", "max"],
        help="reasoning effort for codex_cli provider. Maps to "
        '`-c model_reasoning_effort="<level>"`. Only meaningful when '
        "--provider codex_cli; ignored otherwise. Higher levels burn more "
        "tokens but give the model more reasoning budget.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    run_dir = _stage_run_dir(args.out.resolve(), baseline=args.baseline)

    log_path = run_dir / "campaign.log"
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s · %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )

    sys.path.insert(0, str(run_dir))
    os.chdir(run_dir)

    # Effort plumbing: CodexCliClient picks LBG_CODEX_EFFORT off the env at
    # construction time. Setting it here keeps the wrapper interface stable
    # while letting the CLI control the level per-run.
    if args.codex_effort is not None:
        os.environ["LBG_CODEX_EFFORT"] = args.codex_effort

    # If --goal was passed, write it into goal_state.json before the
    # campaign so the in-loop _goal_update reads the new target.
    if args.goal is not None:
        from lbg.goal import GoalState, read_state, write_state

        existing = read_state(REPO)
        write_state(
            REPO,
            GoalState(
                target_distinct_fns=int(args.goal),
                current_distinct_fns=existing.current_distinct_fns,
                distinct_fns=existing.distinct_fns,
                library_total=existing.library_total,
                met=existing.current_distinct_fns >= int(args.goal),
                ts_utc=existing.ts_utc,
            ),
        )

    import dataclasses as _dc

    from lbg.gate import GateConfig
    from lbg.goal import read_state as _read_goal_state
    from lbg.orchestrator.campaign import CampaignRunner
    from lbg.orchestrator.role_runner import RoleRunner

    runner = RoleRunner(provider=args.provider) if args.provider else RoleRunner()
    gate_config = GateConfig.permissive() if args.gate == "permissive" else GateConfig()
    # Resolve the active library-diversity goal: explicit --goal wins,
    # otherwise read the persisted goal_state.json (defaults to 5). Pass
    # this into the gate config so the diversity discount + growth-phase
    # Pareto rules treat the user's chosen target as authoritative,
    # rather than the hardcoded 5 in GateConfig.permissive(). Strict gate
    # leaves library_diversity_goal=0 (discount off).
    if args.gate == "permissive":
        if args.goal is not None:
            resolved_goal = int(args.goal)
        else:
            resolved_goal = _read_goal_state(REPO).target_distinct_fns
        gate_config = _dc.replace(gate_config, library_diversity_goal=resolved_goal)
    # Resolve library destination. Default to <REPO>/alpha_cards_library/ so
    # the durable pool lives in the source repo (git-tracked). User passes
    # --library-dir '' to opt out entirely.
    if args.library_dir is None:
        library_dir: Path | None = REPO / "alpha_cards_library"
    elif str(args.library_dir).strip() == "":
        library_dir = None
    else:
        library_dir = args.library_dir.resolve()
    cr = CampaignRunner(run_dir, runner=runner, gate_config=gate_config, library_dir=library_dir)

    t0 = time.monotonic()
    result = cr.run(
        n_iterations=args.iterations,
        budget_per_iteration=args.budget,
        seal_only_last_iteration=args.seal_only_last,
        carry_strategy=args.carry_strategy,
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
