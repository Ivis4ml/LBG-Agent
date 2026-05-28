"""B3 · per-card empirical null distribution (Significance Hardening).

For each library card, compute its single-factor-vs-baseline sealed
pathwise CI on:
  - the real sealed split (observed), and
  - N stationary-bootstrap draws of that split (null).

Then place each card's observed CI lower bound on the empirical null
distribution (per-card percentile) and run a Westfall-Young min-p
family-wise correction across all cards (the independent unit is the
bootstrap draw, within which the cards are correlated because they
share the same null data).

No LLM is involved — this is pure backtest + bootstrap.

Usage
-----

    uv run python scripts/per_card_null_distribution.py \\
        --n-null 100 --split split_C --out /tmp/b3_null

Output
------
  <out>/per_card_null.json   full observed + null arrays per card
  stdout                     per-card percentile table + family-wise min-p
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lbg.alpha_cards import AlphaCard  # noqa: E402
from lbg.data.loader import load_split  # noqa: E402
from lbg.dsl import load_strategy  # noqa: E402
from lbg.verdict.per_card_null import (  # noqa: E402
    bootstrap_dataframe,
    card_pathwise_ci,
    inject_card_into_strategy,
    stage_indicators_dir,
)

logger = logging.getLogger("per_card_null")


def _load_cards(library_dir: Path) -> list[AlphaCard]:
    cards_dir = library_dir / "cards"
    out: list[AlphaCard] = []
    for p in sorted(cards_dir.glob("*.yaml")):
        card = AlphaCard.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
        if card.attach_config is not None:
            out.append(card)
        else:
            logger.warning("skip %s: no attach_config", card.alpha_id)
    return out


def _ci_dict(boot: dict) -> dict:
    return {
        "point": float(boot["point_estimate"]),
        "ci_lower": float(boot["ci_lower"]),
        "ci_upper": float(boot["ci_upper"]),
        "p_value_one_sided": float(boot["p_value_one_sided"]),
    }


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    library_dir = Path(args.library_dir).resolve()

    baseline = load_strategy(REPO_ROOT / "strategy.yaml")
    cards = _load_cards(library_dir)
    if not cards:
        logger.error("no cards with attach_config under %s", library_dir / "cards")
        return 2
    logger.info("loaded %d cards with attach_config", len(cards))

    sealed_real = load_split(args.split)  # type: ignore[arg-type]
    logger.info("real %s: %d bars", args.split, len(sealed_real))

    # Stage one indicators dir per card (baseline + that card's .py). The
    # "without" arm is always the bare baseline indicators dir.
    tmp = Path(tempfile.mkdtemp(prefix="b3_"))
    with_dirs: dict[str, Path] = {}
    with_strats: dict[str, object] = {}
    for card in cards:
        with_strats[card.alpha_id] = inject_card_into_strategy(baseline, card)
        with_dirs[card.alpha_id] = stage_indicators_dir(
            REPO_ROOT / "indicators",
            library_dir / "indicators",
            card.signal.fn,
            tmp / card.alpha_id,
        )

    def one_pass(df, *, rng_seed: int) -> dict[str, dict]:
        """Compute every card's CI on `df`. Same df for all cards so the
        family shares the (null or real) data."""
        results: dict[str, dict] = {}
        for card in cards:
            try:
                boot = card_pathwise_ci(
                    with_strats[card.alpha_id],
                    baseline,
                    with_indicators_dir=with_dirs[card.alpha_id],
                    without_indicators_dir=REPO_ROOT / "indicators",
                    df=df,
                    block_len=args.block_len,
                    n_bootstrap=args.n_bootstrap,
                    alpha=args.alpha,
                    rng_seed=rng_seed,
                )
                results[card.alpha_id] = _ci_dict(boot)
            except (ValueError, OSError) as e:
                results[card.alpha_id] = {"error": str(e)}
        return results

    t0 = time.time()
    # Observed pass on real data.
    logger.info("observed pass on real %s ...", args.split)
    observed = one_pass(sealed_real, rng_seed=args.rng_seed)

    # Null passes on bootstrap draws. The meta-RNG picks each draw's
    # bootstrap seed; all cards within a draw see the SAME bootstrap df.
    meta = np.random.default_rng(args.rng_seed)
    null_by_card: dict[str, list[dict]] = {c.alpha_id: [] for c in cards}
    for k in range(args.n_null):
        draw_seed = int(meta.integers(0, 2**31 - 1))
        df_boot = bootstrap_dataframe(
            sealed_real, rng_seed=draw_seed, mean_block_len=args.block_len
        )
        res = one_pass(df_boot, rng_seed=args.rng_seed)
        for aid, r in res.items():
            r["draw_seed"] = draw_seed
            null_by_card[aid].append(r)
        if (k + 1) % 10 == 0:
            logger.info(
                "null draw %d/%d done (%.0fs elapsed)", k + 1, args.n_null, time.time() - t0
            )

    payload = {
        "split": args.split,
        "n_null": args.n_null,
        "n_bootstrap": args.n_bootstrap,
        "block_len": args.block_len,
        "alpha": args.alpha,
        "definition": "single_factor_vs_baseline",
        "cards": {
            c.alpha_id: {"observed": observed[c.alpha_id], "null": null_by_card[c.alpha_id]}
            for c in cards
        },
    }
    out_path = out_dir / "per_card_null.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s (%.0fs total)", out_path, time.time() - t0)

    _analyze(payload)
    return 0


def _analyze(payload: dict) -> None:
    """Per-card empirical percentile + Westfall-Young min-p family test."""
    cards = payload["cards"]
    print("\nB3 · per-card empirical null (single-factor-vs-baseline)")
    print(f"split={payload['split']}  n_null={payload['n_null']}  def={payload['definition']}")
    print("=" * 96)
    print(
        f"{'card':<34} {'obs CI_lower':>12} {'obs p':>8} "
        f"{'null<=obs':>10} {'emp pct':>8} {'emp p(point)':>13}"
    )
    print("-" * 96)

    # Collect per-draw arrays for the two Westfall-Young family variants.
    n_null = payload["n_null"]
    null_p_by_draw: list[list[float]] = [[] for _ in range(n_null)]  # bootstrap p per card
    null_pt_by_draw: list[list[float]] = [[] for _ in range(n_null)]  # point per card
    observed_ps: list[float] = []
    observed_points: list[float] = []

    for aid, blk in cards.items():
        obs = blk["observed"]
        nulls = blk["null"]
        if "error" in obs:
            print(f"{aid:<34} {'ERROR':>12}")
            continue
        obs_cl = obs["ci_lower"]
        obs_pt = obs["point"]
        obs_p = obs["p_value_one_sided"]
        observed_ps.append(obs_p)
        observed_points.append(obs_pt)

        valid_null = [n for n in nulls if "error" not in n]
        null_cl = np.array([n["ci_lower"] for n in valid_null], dtype=float)
        null_pt = np.array([n["point"] for n in valid_null], dtype=float)
        # Per-card percentile of observed CI_lower among null CI_lowers.
        pct = float((null_cl <= obs_cl).mean()) if null_cl.size else float("nan")
        # Empirical one-sided p: P(null point >= observed point).
        emp_p_point = float((null_pt >= obs_pt).mean()) if null_pt.size else float("nan")

        for k, n in enumerate(valid_null):
            if k < n_null:
                null_p_by_draw[k].append(n["p_value_one_sided"])
                null_pt_by_draw[k].append(n["point"])

        print(
            f"{aid:<34} {obs_cl:>12.4f} {obs_p:>8.4f} "
            f"{int((null_cl <= obs_cl).sum()):>4}/{null_cl.size:<5} "
            f"{pct:>8.2%} {emp_p_point:>13.4f}"
        )

    # Westfall-Young family-wise correction, two variants. Both treat the
    # bootstrap DRAW as the independent unit (cards within a draw share
    # null data and are therefore correlated — the resampling preserves
    # that dependence automatically).
    print("-" * 96)
    if observed_ps:
        # minP variant: per-card statistic = bootstrap one-sided p; the
        # family stat is the smallest p across cards. Sensitive to the
        # card with the tightest null (here fractal_efficiency).
        obs_min_p = min(observed_ps)
        null_min_p = np.array([min(ps) for ps in null_p_by_draw if ps], dtype=float)
        if null_min_p.size:
            fwer_minp = float((null_min_p <= obs_min_p).mean())
            print(
                f"Westfall-Young minP : observed min-p={obs_min_p:.4f}, "
                f"P(null min-p <= obs) = {fwer_minp:.4f}  (n_draws={null_min_p.size})"
            )
        # maxT variant: per-card statistic = point ΔSharpe; the family
        # stat is the largest point across cards. Dominated by the
        # highest-variance card (here vol_regime_zscore).
        obs_max_pt = max(observed_points)
        null_max_pt = np.array([max(pts) for pts in null_pt_by_draw if pts], dtype=float)
        if null_max_pt.size:
            fwer_maxt = float((null_max_pt >= obs_max_pt).mean())
            print(
                f"Westfall-Young maxT : observed max point={obs_max_pt:.4f}, "
                f"P(null max point >= obs) = {fwer_maxt:.4f}  (n_draws={null_max_pt.size})"
            )
        print(
            "  → family-wise adjusted p for the strongest card under each "
            "statistic. Neither significant unless small."
        )
    print("=" * 96 + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="B3 per-card empirical null distribution")
    p.add_argument("--n-null", type=int, default=100, help="number of bootstrap null draws")
    p.add_argument("--split", type=str, default="split_C", help="which split is the sealed window")
    p.add_argument("--library-dir", type=str, default=str(REPO_ROOT / "alpha_cards_library"))
    p.add_argument("--out", type=str, default="/tmp/b3_null")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--block-len", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument(
        "--analyze-only",
        type=str,
        default=None,
        help="skip computation; re-analyze an existing per_card_null.json",
    )
    args = p.parse_args(argv)

    if args.analyze_only:
        payload = json.loads(Path(args.analyze_only).read_text(encoding="utf-8"))
        _analyze(payload)
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
