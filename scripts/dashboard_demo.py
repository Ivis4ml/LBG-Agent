"""Emit synthetic dashboard events so the new per-card CI chart can be
inspected without running a real campaign.

Use this to visually verify the dashboard's "alpha card 增量 Sharpe · 95%
区间" chart, the H1 KPI, and the event-log handling of
`alpha_card_summary`. No LLM, no sealed-data access — pure UI demo.

Usage::

    uv run python scripts/dashboard_demo.py --out /tmp/lbg_demo
    open /tmp/lbg_demo/artifacts/live/dashboard.html

The demo emits one iter_start, three trials (mix of accept/reject), one
iter_end, and one alpha_card_summary with three cards (one validated,
one not, one with a `validation_error`). Mirrors what a real campaign
would produce so the visual layout is faithful.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lbg.orchestrator.live_status import LiveStatusWriter  # noqa: E402
from lbg.schemas import TrainMetrics  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    w = LiveStatusWriter(args.out)
    w.iter_start(0, budget=3)
    w.trial(
        iter_id=0,
        trial_id=0,
        edit_summary="+indicator adx_14 attached as indicator_above:adx_14:20.0",
        decision="reject",
        gate_reason="reject_no_meaningful_improvement",
        train_metrics=TrainMetrics(sharpe=0.42, max_drawdown=-0.24, turnover=1.9, num_trades=32),
        cited_factors=("ADX",),
    )
    w.trial(
        iter_id=0,
        trial_id=1,
        edit_summary="+indicator drawdown_trail attached as exit_filter:indicator_below:-0.10",
        decision="accept",
        gate_reason=None,
        train_metrics=TrainMetrics(sharpe=0.71, max_drawdown=-0.15, turnover=1.1, num_trades=18),
        cited_factors=("MaxDrawdownDuration",),
    )
    w.trial(
        iter_id=0,
        trial_id=2,
        edit_summary="+indicator vol_filter attached as exit_filter:indicator_above:0.30",
        decision="accept",
        gate_reason=None,
        train_metrics=TrainMetrics(sharpe=0.65, max_drawdown=-0.18, turnover=1.0, num_trades=20),
        cited_factors=("HistoricalVolatility",),
    )
    w.iter_end(
        0,
        n_accepted=2,
        n_rejected=1,
        n_invariant=0,
        sealed_sharpe=0.91,
        sealed_max_drawdown=-0.13,
    )

    # Synthetic primary-H1 result: 3 candidate cards, 1 validated.
    h1 = {
        "candidate_card_count": 3,
        "validated_factor_count": 1,
        "best_baseline_count": 0,
        "tau": 3,
        "h1_strong": False,
        "h1_weak": True,
    }
    per_card = [
        {
            "alpha_id": "drawdown_trail_trial_0001",
            "source_trial": 1,
            "incremental_sharpe_point": 0.18,
            "incremental_sharpe_ci_lower": 0.04,
            "incremental_sharpe_ci_upper": 0.33,
        },
        {
            "alpha_id": "vol_filter_trial_0002",
            "source_trial": 2,
            "incremental_sharpe_point": -0.05,
            "incremental_sharpe_ci_lower": -0.21,
            "incremental_sharpe_ci_upper": 0.11,
        },
        {
            "alpha_id": "orphan_trial_0042",
            "source_trial": 42,
            "validation_error": "trial commit for trial_id=42 not found in git log",
        },
    ]
    w.alpha_card_summary(iter_id=0, h1_verdict=h1, per_card_results=per_card)
    w.campaign_end(total_accepted=2, total_alpha_cards=2, elapsed_sec=12.5)

    dashboard = args.out / "artifacts" / "live" / "dashboard.html"
    jsonl = args.out / "artifacts" / "live" / "run_status.jsonl"
    print(f"dashboard demo emitted · {jsonl}")
    print(f"open: {dashboard}")
    print("the H1 KPI should read '1' of 3 candidate cards · H1 weak;")
    print("the alpha-card chart should show two CI bars (green/red) and one faded errored row.")
    # Give the OS a tick so file timestamps land in order in tail-following tools.
    time.sleep(0.1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
