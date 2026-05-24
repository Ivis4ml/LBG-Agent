"""Live run-status writer · feed the browser-based dashboard.

A campaign or long Discovery run is opaque from the outside: log lines
scroll by, but it's hard to see Sharpe / accept-rate / factor exploration
in real time. This module fixes that.

`LiveStatusWriter` writes one JSON line per event to
`<repo>/artifacts/live/run_status.jsonl`. The companion static HTML at
`<repo>/artifacts/live/dashboard.html` polls that file from the browser
(no server) and re-renders Chart.js plots every 2 seconds. The user opens
the HTML once before launching the run; the page updates in place.

Event shape:

  {"kind": "trial",
   "ts": "...",
   "iter": 1,
   "trial_id": 0,
   "edit": "+indicator adx_14 attached as indicator_above:adx_14",
   "decision": "reject" | "accept" | "invariant",
   "gate_reason": "reject_no_meaningful_improvement",
   "train": {"sharpe": 0.42, "max_drawdown": -0.24, "turnover": 1.9,
             "num_trades": 32},
   "cited_factors": ["ADX"]}

  {"kind": "iter_start", "ts": "...", "iter": 2}
  {"kind": "iter_end",   "ts": "...", "iter": 1, "n_accepted": 0,
   "sealed_sharpe": 0.108}
  {"kind": "campaign_end", "ts": "...", "total_accepted": 4,
   "total_alpha_cards": 0}

Append-only, JSON lines for trivial fetch + parse in the browser.
"""

from __future__ import annotations

import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DASHBOARD_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "live_dashboard.html"


class LiveStatusWriter:
    """Append-only JSONL writer + dashboard.html bootstrap.

    Thread-safe (the writer is called from a single thread today, but
    Discovery's reflector + alpha card emission share a per-trial section;
    cheap insurance to lock).

    The dashboard HTML is copied once at __init__ time so the user can
    open it before the first trial finishes.
    """

    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root)
        self.live_dir = self.repo_root / "artifacts" / "live"
        self.jsonl_path = self.live_dir / "run_status.jsonl"
        self.dashboard_path = self.live_dir / "dashboard.html"
        self._lock = threading.Lock()
        self.live_dir.mkdir(parents=True, exist_ok=True)
        # Fresh run: truncate the jsonl and reseat the HTML.
        self.jsonl_path.write_text("", encoding="utf-8")
        if DASHBOARD_TEMPLATE_PATH.exists():
            shutil.copy(DASHBOARD_TEMPLATE_PATH, self.dashboard_path)

    def _write(self, event: dict[str, Any]) -> None:
        event = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event}
        line = json.dumps(event, ensure_ascii=False)
        with self._lock, self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ---- public events ----

    def iter_start(self, iter_id: int, *, budget: int) -> None:
        self._write({"kind": "iter_start", "iter": iter_id, "budget": budget})

    def trial(
        self,
        *,
        iter_id: int,
        trial_id: int,
        edit_summary: str,
        decision: str,
        gate_reason: str | None,
        train_metrics: Any | None,
        cited_factors: list[str] | tuple[str, ...] = (),
    ) -> None:
        payload: dict[str, Any] = {
            "kind": "trial",
            "iter": iter_id,
            "trial_id": trial_id,
            "edit": edit_summary,
            "decision": decision,
            "gate_reason": gate_reason,
            "cited_factors": list(cited_factors),
        }
        if train_metrics is not None:
            payload["train"] = {
                "sharpe": float(train_metrics.sharpe),
                "max_drawdown": float(train_metrics.max_drawdown),
                "turnover": float(train_metrics.turnover),
                "num_trades": int(train_metrics.num_trades),
            }
        self._write(payload)

    def invariant_failure(
        self,
        *,
        iter_id: int,
        trial_id: int,
        invariant_name: str,
        message: str,
    ) -> None:
        self._write(
            {
                "kind": "invariant",
                "iter": iter_id,
                "trial_id": trial_id,
                "invariant_name": invariant_name,
                "message": message[:200],
            }
        )

    def iter_end(
        self,
        iter_id: int,
        *,
        n_accepted: int,
        n_rejected: int,
        n_invariant: int,
        sealed_sharpe: float | None,
        sealed_max_drawdown: float | None,
    ) -> None:
        self._write(
            {
                "kind": "iter_end",
                "iter": iter_id,
                "n_accepted": n_accepted,
                "n_rejected": n_rejected,
                "n_invariant": n_invariant,
                "sealed_sharpe": sealed_sharpe,
                "sealed_max_drawdown": sealed_max_drawdown,
            }
        )

    def campaign_end(
        self, *, total_accepted: int, total_alpha_cards: int, elapsed_sec: float
    ) -> None:
        self._write(
            {
                "kind": "campaign_end",
                "total_accepted": total_accepted,
                "total_alpha_cards": total_alpha_cards,
                "elapsed_sec": round(elapsed_sec, 1),
            }
        )


__all__ = ["LiveStatusWriter"]
