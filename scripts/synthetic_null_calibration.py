"""Synthetic-null calibration runner (Significance Hardening step 4).

For each replication r:
  1. Stationary-bootstrap the SPY parquet (preserves marginal
     distribution and short-range autocorrelation; destroys real signal).
  2. Run a small Discovery campaign on that bootstrap with the
     specified LLM provider, with `LBG_DATA_PATH` pointing at the
     bootstrap parquet so neither the gate nor the agents see real data.
  3. Record acceptance / CI-pass / BH-pass / SPA-p / DSR-pass counts.

After N replications per provider, the aggregate produces the
operating characteristic plot: distribution of acceptance metrics under
a known-null hypothesis (caveat: the LLM's training prior over factor
names is NOT bootstrapped; see docs/SIGNIFICANCE_HARDENING.md §4).

Usage
-----

    uv run python scripts/synthetic_null_calibration.py \\
        --provider codex_cli \\
        --n-replications 20 \\
        --budget 6 --iterations 1 \\
        --base-out /tmp/null_codex

    # then for the second provider:
    uv run python scripts/synthetic_null_calibration.py \\
        --provider mimo_tp \\
        --n-replications 20 \\
        --budget 6 --iterations 1 \\
        --base-out /tmp/null_mimo_tp

    # compare:
    uv run python scripts/synthetic_null_calibration.py \\
        --compare /tmp/null_codex /tmp/null_mimo_tp

The `--compare` mode prints a side-by-side summary of acceptance rate,
BH-pass count, SPA p-value, DSR-pass count across the two providers'
replication populations.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lbg.verdict.synthetic_null import bootstrap_spy_parquet  # noqa: E402

logger = logging.getLogger("synthetic_null")

DEFAULT_SOURCE_PARQUET = REPO_ROOT / "data" / "spy_daily.parquet"


def run_one_replication(
    rep_id: int,
    *,
    provider: str,
    base_out: Path,
    budget: int,
    iterations: int,
    source_parquet: Path,
    source_library_dir: Path | None,
    bootstrap_seed: int,
) -> dict:
    """One bootstrap + campaign + parse cycle. Returns the metrics dict."""
    rep_dir = base_out / f"rep_{rep_id:03d}"
    rep_dir.mkdir(parents=True, exist_ok=True)

    # Step A: stationary-bootstrap SPY parquet into the replication dir.
    boot_parquet = rep_dir / "spy_daily_boot.parquet"
    bootstrap_spy_parquet(source_parquet, boot_parquet, rng_seed=bootstrap_seed, mean_block_len=10)

    # Step B: copy the source library into a per-rep scratch dir so
    # campaign's `sync_from_run` writes accepted (bootstrap-derived!)
    # cards into the scratch copy, not the real `alpha_cards_library/`.
    # Bug (b): without this, null-data acceptances (e.g., dispersion_regime
    # with -99.99% drawdown) silently entered the production library.
    if source_library_dir is not None and source_library_dir.exists():
        rep_library_dir: Path | None = rep_dir / "alpha_cards_library"
        # shutil.copytree refuses an existing destination; clean first.
        if rep_library_dir.exists():
            import shutil as _shutil

            _shutil.rmtree(rep_library_dir)
        import shutil as _shutil

        _shutil.copytree(source_library_dir, rep_library_dir)
    else:
        rep_library_dir = None

    # Step C: run a tiny campaign on the bootstrap, with LBG_DATA_PATH
    # pointing at our bootstrap parquet. campaign.py writes its
    # artefacts to --out. Permissive gate matches the v26 real-data
    # setting (per STAGE2_REPORT lines 952/961) so the comparison is
    # apples-to-apples.
    campaign_out = rep_dir / "campaign"
    env = os.environ.copy()
    env["LBG_DATA_PATH"] = str(boot_parquet)
    env["LBG_PROVIDER"] = provider

    campaign_argv = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "campaign.py"),
        "--provider",
        provider,
        "--iterations",
        str(iterations),
        "--budget",
        str(budget),
        "--gate",
        "permissive",
        "--out",
        str(campaign_out),
    ]
    if rep_library_dir is not None:
        campaign_argv += ["--library-dir", str(rep_library_dir)]

    t0 = time.time()
    proc = subprocess.run(
        campaign_argv,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        logger.warning(
            "replication %d campaign exited %d after %.1fs; stderr tail:\n%s",
            rep_id,
            proc.returncode,
            elapsed,
            "\n".join(proc.stderr.splitlines()[-30:]),
        )

    # Step C: parse metrics. Tolerant of partial outputs from a failed
    # replication — every field gets a None when missing.
    metrics: dict = {
        "rep_id": rep_id,
        "provider": provider,
        "bootstrap_seed": bootstrap_seed,
        "elapsed_sec": round(elapsed, 1),
        "campaign_exit_code": proc.returncode,
    }

    summary_path = campaign_out / "campaigns" / "campaign_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics["n_iterations"] = summary.get("n_iterations")
        metrics["total_accepted"] = summary.get("total_accepted")
        metrics["total_alpha_cards"] = summary.get("total_alpha_cards")
        # Sealed verdict on the LAST iteration is what the H1 reader uses.
        last_iter = (summary.get("iterations") or [])[-1] if summary.get("iterations") else None
        sealed = (last_iter or {}).get("sealed_summary") or {}
        h1 = sealed.get("h1_verdict") or {}
        metrics["validated_factor_count"] = h1.get("validated_factor_count")
        metrics["ci_only_validated_count"] = h1.get("ci_only_validated_count")
        # SPA + DSR pass rate from per-card validation.
        per_card = sealed.get("per_card_validation") or []
        metrics["n_per_card"] = len(per_card)
        metrics["n_ci_pass"] = sum(
            1
            for c in per_card
            if isinstance(c.get("incremental_sharpe_ci_lower"), (int, float))
            and c["incremental_sharpe_ci_lower"] > 0.0
        )
        metrics["n_bh_pass"] = sum(1 for c in per_card if c.get("bh_validated") is True)
        metrics["n_dsr_pass"] = sum(1 for c in per_card if c.get("dsr_passes") is True)
        # SPA verdict on the family.
        spa = sealed.get("spa_verdict") or {}
        metrics["spa_p_value"] = spa.get("spa_p_value")
        metrics["spa_p_value_l"] = spa.get("spa_p_value_l")
        metrics["spa_p_value_u"] = spa.get("spa_p_value_u")
        # Strategy-level supplementary verdict (less interesting but log it).
        metrics["sealed_strategy_sharpe"] = sealed.get("sharpe")
    else:
        metrics["error"] = "no_campaign_summary"

    return metrics


def cmd_run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    base_out = Path(args.base_out).resolve()
    base_out.mkdir(parents=True, exist_ok=True)
    source = Path(args.source_parquet).resolve()
    if not source.exists():
        logger.error(
            "source parquet missing at %s. Run `uv run python -m lbg.data.loader fetch` first.",
            source,
        )
        return 2

    # Source library: by default the production library. Each replication
    # gets a fresh COPY in its scratch dir to isolate it from real campaigns
    # (Bug (b) fix). Empty string disables library auto-inject entirely.
    if args.source_library_dir is None:
        source_library_dir: Path | None = REPO_ROOT / "alpha_cards_library"
    elif str(args.source_library_dir).strip() == "":
        source_library_dir = None
    else:
        source_library_dir = Path(args.source_library_dir).resolve()

    log_path = base_out / "replications.jsonl"
    summary_path = base_out / "summary.json"
    rng = np.random.default_rng(args.rng_seed)

    records: list[dict] = []
    if log_path.exists() and args.resume:
        # Read previous progress; skip already-done rep_ids.
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        done_ids = {r["rep_id"] for r in records}
        logger.info("resume: %d replications already done", len(done_ids))
    else:
        done_ids = set()

    for r in range(args.n_replications):
        if r in done_ids:
            continue
        bootstrap_seed = int(rng.integers(0, 2**31 - 1))
        logger.info(
            "rep %d/%d · provider=%s · bootstrap_seed=%d",
            r,
            args.n_replications,
            args.provider,
            bootstrap_seed,
        )
        rec = run_one_replication(
            r,
            provider=args.provider,
            base_out=base_out,
            budget=args.budget,
            iterations=args.iterations,
            source_parquet=source,
            source_library_dir=source_library_dir,
            bootstrap_seed=bootstrap_seed,
        )
        records.append(rec)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        logger.info(
            "rep %d done · accepted=%s · ci_pass=%s · bh_pass=%s · dsr_pass=%s · spa_p=%.3f",
            r,
            rec.get("total_accepted"),
            rec.get("n_ci_pass"),
            rec.get("n_bh_pass"),
            rec.get("n_dsr_pass"),
            rec.get("spa_p_value") if rec.get("spa_p_value") is not None else float("nan"),
        )

    # Aggregate.
    summary = _summarize(records, provider=args.provider)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("done. summary written to %s", summary_path)
    _print_summary(summary)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rows: list[tuple[str, dict]] = []
    for d in args.compare:
        path = Path(d) / "summary.json"
        if not path.exists():
            logger.error("missing summary.json under %s", d)
            return 2
        rows.append((d, json.loads(path.read_text(encoding="utf-8"))))

    # Side-by-side print.
    print("\nsynthetic null calibration · side-by-side")
    print("=" * 78)
    keys = [
        ("n_replications", "n_replications"),
        ("provider", "provider"),
        ("acceptance_rate_mean", "mean(accepted / budget)"),
        ("acceptance_rate_std", "std(accepted / budget)"),
        ("ci_pass_mean", "mean(ci_pass cards / rep)"),
        ("bh_pass_mean", "mean(bh_pass cards / rep)"),
        ("dsr_pass_mean", "mean(dsr_pass cards / rep)"),
        ("spa_p_mean", "mean SPA p-value"),
        ("spa_p_rejection_rate_10", "P(SPA p ≤ 0.10)"),
    ]
    header = ["metric"] + [r[1]["provider"] or Path(r[0]).name for r in rows]
    print(" | ".join(f"{c:<28}" for c in header))
    print("-" * 78)
    for k, label in keys:
        line = [label]
        for _, s in rows:
            v = s.get(k, "—")
            if isinstance(v, float):
                line.append(f"{v:.4f}")
            else:
                line.append(str(v))
        print(" | ".join(f"{c:<28}" for c in line))
    print("=" * 78 + "\n")
    return 0


def _summarize(records: list[dict], *, provider: str) -> dict:
    if not records:
        return {"provider": provider, "n_replications": 0}
    budgets = [r.get("total_accepted") for r in records]
    accepted = np.array(
        [r["total_accepted"] for r in records if r.get("total_accepted") is not None], dtype=float
    )
    # The budget per iteration is configurable; approximate via the
    # MAX of total_accepted as a denominator if we can't see it.
    # We pass `budget` through env upstream; for now use total trials = 6 (default).
    # The real per-rep budget is in records as `budget` field if we add it.
    bh_pass = np.array(
        [r["n_bh_pass"] for r in records if r.get("n_bh_pass") is not None], dtype=float
    )
    ci_pass = np.array(
        [r["n_ci_pass"] for r in records if r.get("n_ci_pass") is not None], dtype=float
    )
    dsr_pass = np.array(
        [r["n_dsr_pass"] for r in records if r.get("n_dsr_pass") is not None], dtype=float
    )
    spa_p = np.array(
        [r["spa_p_value"] for r in records if r.get("spa_p_value") is not None], dtype=float
    )

    def mean(a: np.ndarray) -> float:
        return float(a.mean()) if a.size else float("nan")

    def std(a: np.ndarray) -> float:
        return float(a.std(ddof=1)) if a.size > 1 else 0.0

    return {
        "provider": provider,
        "n_replications": len(records),
        "n_replications_with_summary": sum(
            1 for r in records if r.get("total_accepted") is not None
        ),
        "acceptance_count_mean": mean(accepted),
        "acceptance_count_std": std(accepted),
        "acceptance_rate_mean": mean(accepted / np.maximum(1.0, np.full_like(accepted, 6.0))),
        "acceptance_rate_std": std(accepted / np.maximum(1.0, np.full_like(accepted, 6.0))),
        "ci_pass_mean": mean(ci_pass),
        "bh_pass_mean": mean(bh_pass),
        "dsr_pass_mean": mean(dsr_pass),
        "spa_p_mean": mean(spa_p),
        "spa_p_rejection_rate_10": (float((spa_p <= 0.10).mean()) if spa_p.size else float("nan")),
        "elapsed_sec_total": sum(r.get("elapsed_sec", 0.0) for r in records),
    }


def _print_summary(summary: dict) -> None:
    print()
    print(f"provider                            : {summary.get('provider')}")
    print(f"replications (with parsed summary)  : {summary.get('n_replications_with_summary')}")
    print(
        f"acceptance count   mean ± std       : {summary.get('acceptance_count_mean'):.3f} ± {summary.get('acceptance_count_std'):.3f}"
    )
    print(f"acceptance rate    mean             : {summary.get('acceptance_rate_mean'):.4f}")
    print(f"ci_pass / rep      mean             : {summary.get('ci_pass_mean'):.3f}")
    print(f"bh_pass / rep      mean             : {summary.get('bh_pass_mean'):.3f}")
    print(f"dsr_pass / rep     mean             : {summary.get('dsr_pass_mean'):.3f}")
    spa_p = summary.get("spa_p_mean")
    spa_rej = summary.get("spa_p_rejection_rate_10")
    print(
        f"SPA p-value        mean             : {spa_p:.4f}"
        if spa_p == spa_p
        else "SPA p-value mean : nan"
    )
    print(
        f"SPA reject rate at α=0.10           : {spa_rej:.4f}"
        if spa_rej == spa_rej
        else "SPA reject rate  : nan"
    )
    print(f"total elapsed                       : {summary.get('elapsed_sec_total'):.1f} s")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run / aggregate synthetic-null calibration replications."
    )
    sub = parser.add_subparsers(dest="cmd", required=False)
    parser.set_defaults(cmd=None)

    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="LLM provider for replications (codex_cli | mimo_tp | claude_cli | ...)",
    )
    parser.add_argument("--n-replications", type=int, default=20)
    parser.add_argument("--budget", type=int, default=6, help="trial budget per replication")
    parser.add_argument(
        "--iterations", type=int, default=1, help="campaign iterations per replication"
    )
    parser.add_argument(
        "--source-parquet",
        type=str,
        default=str(DEFAULT_SOURCE_PARQUET),
        help="path to the source SPY parquet to bootstrap from",
    )
    parser.add_argument(
        "--source-library-dir",
        type=str,
        default=None,
        help=(
            "library to copy into each replication's scratch dir as the "
            "auto-inject source. Defaults to alpha_cards_library/ at the "
            "repo root. Pass '' to disable library auto-inject entirely."
        ),
    )
    parser.add_argument("--base-out", type=str, default=None)
    parser.add_argument(
        "--rng-seed", type=int, default=2026, help="seed for the meta-bootstrap RNG"
    )
    parser.add_argument(
        "--resume", action="store_true", help="skip rep_ids already present in replications.jsonl"
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs="+",
        help="instead of running, side-by-side print summary.json from these dirs",
    )

    args = parser.parse_args(argv)
    if args.compare:
        return cmd_compare(args)
    if not args.provider or not args.base_out:
        parser.error("either --compare DIR... or both --provider and --base-out are required")
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
