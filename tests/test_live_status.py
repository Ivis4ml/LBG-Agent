"""Live run-status writer · append-only jsonl + dashboard bootstrap.

Tests pin the event protocol consumed by the browser dashboard:
  * jsonl truncated on writer __init__ (fresh run, fresh dashboard).
  * dashboard.html copied to artifacts/live/ at __init__.
  * one event per call; each event has a `kind` and a `ts`.
  * trial events include train metrics when provided.
  * invariant events do NOT carry train metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

from lbg.orchestrator.live_status import LiveStatusWriter
from lbg.schemas import TrainMetrics

REPO = Path(__file__).resolve().parents[1]


def _events(jsonl: Path) -> list[dict]:
    return [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]


def test_init_truncates_jsonl(tmp_path):
    """A previous run's events must not leak into the new dashboard."""
    live_dir = tmp_path / "artifacts" / "live"
    live_dir.mkdir(parents=True)
    junk = live_dir / "run_status.jsonl"
    junk.write_text('{"stale": true}\n')
    LiveStatusWriter(tmp_path)
    assert junk.read_text() == ""


def test_init_copies_dashboard_template(tmp_path):
    LiveStatusWriter(tmp_path)
    html = tmp_path / "artifacts" / "live" / "dashboard.html"
    assert html.exists()
    body = html.read_text()
    assert "LBG-Trader" in body
    assert "run_status.jsonl" in body  # the html polls this file


def test_iter_start_event_shape(tmp_path):
    w = LiveStatusWriter(tmp_path)
    w.iter_start(0, budget=5)
    evs = _events(w.jsonl_path)
    assert len(evs) == 1
    assert evs[0]["kind"] == "iter_start"
    assert evs[0]["iter"] == 0
    assert evs[0]["budget"] == 5
    assert "ts" in evs[0]


def test_trial_event_includes_train_metrics(tmp_path):
    w = LiveStatusWriter(tmp_path)
    tm = TrainMetrics(sharpe=0.71, max_drawdown=-0.12, turnover=0.35, num_trades=42)
    w.trial(
        iter_id=0,
        trial_id=2,
        edit_summary="+indicator rsi_14",
        decision="accept",
        gate_reason=None,
        train_metrics=tm,
        cited_factors=["RSI"],
    )
    ev = _events(w.jsonl_path)[0]
    assert ev["kind"] == "trial"
    assert ev["decision"] == "accept"
    assert ev["train"]["sharpe"] == 0.71
    assert ev["train"]["num_trades"] == 42
    assert ev["cited_factors"] == ["RSI"]
    assert ev["gate_reason"] is None


def test_trial_event_without_train_metrics(tmp_path):
    """An aborted-before-backtest trial has no train metrics. The writer
    must still emit a usable event, just without the `train` key."""
    w = LiveStatusWriter(tmp_path)
    w.trial(
        iter_id=0,
        trial_id=3,
        edit_summary="(aborted)",
        decision="reject",
        gate_reason=None,
        train_metrics=None,
        cited_factors=[],
    )
    ev = _events(w.jsonl_path)[0]
    assert "train" not in ev
    assert ev["cited_factors"] == []


def test_invariant_event_shape(tmp_path):
    w = LiveStatusWriter(tmp_path)
    w.invariant_failure(
        iter_id=0,
        trial_id=4,
        invariant_name="unwired_indicator",
        message="cannot leave RSI unbound to any filter",
    )
    ev = _events(w.jsonl_path)[0]
    assert ev["kind"] == "invariant"
    assert ev["invariant_name"] == "unwired_indicator"
    assert "RSI" in ev["message"]


def test_invariant_message_truncated(tmp_path):
    """A pathological 500-char invariant message gets clipped to 200."""
    w = LiveStatusWriter(tmp_path)
    w.invariant_failure(iter_id=0, trial_id=4, invariant_name="x", message="x" * 500)
    ev = _events(w.jsonl_path)[0]
    assert len(ev["message"]) == 200


def test_iter_end_event(tmp_path):
    w = LiveStatusWriter(tmp_path)
    w.iter_end(
        0,
        n_accepted=2,
        n_rejected=2,
        n_invariant=1,
        sealed_sharpe=0.108,
        sealed_max_drawdown=-0.28,
    )
    ev = _events(w.jsonl_path)[0]
    assert ev["kind"] == "iter_end"
    assert ev["n_accepted"] == 2
    assert ev["sealed_sharpe"] == 0.108


def test_iter_end_handles_no_sealed(tmp_path):
    w = LiveStatusWriter(tmp_path)
    w.iter_end(
        0,
        n_accepted=0,
        n_rejected=5,
        n_invariant=0,
        sealed_sharpe=None,
        sealed_max_drawdown=None,
    )
    ev = _events(w.jsonl_path)[0]
    assert ev["sealed_sharpe"] is None


def test_campaign_end_event(tmp_path):
    w = LiveStatusWriter(tmp_path)
    w.campaign_end(total_accepted=4, total_alpha_cards=0, elapsed_sec=294.7)
    ev = _events(w.jsonl_path)[0]
    assert ev["kind"] == "campaign_end"
    assert ev["total_accepted"] == 4
    assert ev["elapsed_sec"] == 294.7


def test_event_ordering_preserved(tmp_path):
    """A campaign's full sequence must keep the temporal order."""
    w = LiveStatusWriter(tmp_path)
    w.iter_start(0, budget=2)
    w.trial(
        iter_id=0,
        trial_id=0,
        edit_summary="x",
        decision="reject",
        gate_reason="reject_no_meaningful_improvement",
        train_metrics=TrainMetrics(sharpe=0.5, max_drawdown=-0.2, turnover=2.0, num_trades=20),
    )
    w.trial(
        iter_id=0,
        trial_id=1,
        edit_summary="y",
        decision="accept",
        gate_reason=None,
        train_metrics=TrainMetrics(sharpe=0.7, max_drawdown=-0.1, turnover=1.0, num_trades=30),
    )
    w.iter_end(
        0, n_accepted=1, n_rejected=1, n_invariant=0, sealed_sharpe=0.5, sealed_max_drawdown=-0.1
    )
    w.campaign_end(total_accepted=1, total_alpha_cards=0, elapsed_sec=10.0)
    kinds = [e["kind"] for e in _events(w.jsonl_path)]
    assert kinds == ["iter_start", "trial", "trial", "iter_end", "campaign_end"]
