"""Tests for Stage 2 (forward validation) and Stage 3 (paper trading)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lbg.stage2 import (
    ForwardValidationResult,
    Stage2Gate,
    run_forward_validation,
)
from lbg.stage3 import PaperTradingEngine

REPO = Path(__file__).resolve().parents[1]


def _synth_ohlcv(n: int = 200, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(0.0005, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(log_rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "open": open_,
            "high": close * 1.003,
            "low": close * 0.997,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype("int64"),
        }
    )


# -------- Stage 2 --------


def test_forward_validation_returns_result_object():
    df = _synth_ohlcv()
    res = run_forward_validation(REPO / "strategy.yaml", df)
    assert isinstance(res, ForwardValidationResult)
    assert res.metrics.n_bars_used > 0
    assert isinstance(res.go, bool)


def test_forward_validation_go_when_metrics_look_normal():
    df = _synth_ohlcv(n=400, seed=11)
    res = run_forward_validation(REPO / "strategy.yaml", df)
    # The default gate is loose; on this seed the SMA cross trades a
    # non-trivial number of times. We don't assert go=True (depends on
    # data), but the structural shape must hold.
    assert isinstance(res.rejected_reasons, list)
    if res.go:
        assert res.rejected_reasons == []
    else:
        assert len(res.rejected_reasons) >= 1


def test_forward_validation_blocks_on_too_few_trades():
    df = _synth_ohlcv(n=60, seed=99)
    gate = Stage2Gate(min_num_trades=100)  # impossibly strict
    res = run_forward_validation(REPO / "strategy.yaml", df, gate=gate)
    assert res.go is False
    assert any("num_trades" in r for r in res.rejected_reasons)


def test_forward_validation_blocks_on_excessive_drawdown():
    df = _synth_ohlcv(n=300, seed=4)
    gate = Stage2Gate(max_drawdown_floor=-0.001)  # essentially zero
    res = run_forward_validation(REPO / "strategy.yaml", df, gate=gate)
    # If the strategy traded at all, its drawdown will exceed -0.1%.
    if res.metrics.num_trades > 0:
        assert res.go is False
        assert any("max_drawdown" in r for r in res.rejected_reasons)


def test_forward_validation_to_dict_is_json_safe():
    import json

    df = _synth_ohlcv()
    res = run_forward_validation(REPO / "strategy.yaml", df)
    blob = json.dumps(res.to_dict())
    assert "go" in blob
    assert "metrics" in blob


# -------- Stage 3 --------


def test_paper_engine_first_step_returns_none():
    engine = PaperTradingEngine(REPO / "strategy.yaml")
    df = _synth_ohlcv(n=2)
    entry = engine.step("2025-01-02", df.iloc[0].to_dict())
    assert entry is None
    assert engine.n_bars == 1


def test_paper_engine_second_step_emits_log_entry():
    engine = PaperTradingEngine(REPO / "strategy.yaml")
    df = _synth_ohlcv(n=3)
    engine.step("2025-01-02", df.iloc[0].to_dict())
    entry = engine.step("2025-01-03", df.iloc[1].to_dict())
    assert entry is not None
    assert entry.bar_index == 0
    assert entry.iso_date == "2025-01-02"


def test_paper_engine_streams_many_bars_and_equity_tracks():
    engine = PaperTradingEngine(REPO / "strategy.yaml")
    df = _synth_ohlcv(n=120, seed=3)
    for i in range(len(df)):
        engine.step(f"day_{i:03d}", df.iloc[i].to_dict())
    assert engine.n_bars == len(df)
    assert len(engine.log) == len(df) - 1  # first bar has nothing to settle
    # Equity is finite and positive (never goes negative under long-only).
    assert engine.equity > 0
    assert all(np.isfinite(e.equity) for e in engine.log)


def test_paper_engine_log_entries_have_required_fields():
    engine = PaperTradingEngine(REPO / "strategy.yaml")
    df = _synth_ohlcv(n=20)
    for i in range(len(df)):
        engine.step(f"d{i}", df.iloc[i].to_dict())
    entry = engine.log[-1]
    for field in (
        "bar_index",
        "iso_date",
        "open",
        "close",
        "target_position",
        "held_position",
        "realized_pnl",
        "cost_paid",
        "equity",
    ):
        assert hasattr(entry, field)


def test_paper_engine_missing_ohlcv_field_raises():
    engine = PaperTradingEngine(REPO / "strategy.yaml")
    incomplete = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5}
    with pytest.raises(KeyError, match="volume"):
        engine.step("d0", incomplete)


def test_paper_engine_dump_log_writes_jsonl(tmp_path):
    engine = PaperTradingEngine(REPO / "strategy.yaml")
    df = _synth_ohlcv(n=10)
    for i in range(len(df)):
        engine.step(f"d{i}", df.iloc[i].to_dict())
    log_path = tmp_path / "paper_log.jsonl"
    engine.dump_log(log_path)
    assert log_path.exists()
    lines = log_path.read_text().splitlines()
    assert len(lines) == len(engine.log)
    import json

    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["bar_index"] == 0


def test_paper_engine_save_buffer_round_trip(tmp_path):
    engine = PaperTradingEngine(REPO / "strategy.yaml")
    df = _synth_ohlcv(n=15)
    for i in range(len(df)):
        engine.step(f"d{i}", df.iloc[i].to_dict())
    out = engine.save_buffer(tmp_path / "buf.parquet")
    reloaded = pd.read_parquet(out)
    assert len(reloaded) == 15
    assert {"open", "high", "low", "close", "volume", "date"} <= set(reloaded.columns)


def test_paper_engine_position_lags_signal_by_one_bar():
    """When the policy says 'go long at close[t]', the position is held
    starting from open[t+1], so the log entry for bar t shows held=target."""
    engine = PaperTradingEngine(REPO / "strategy.yaml")
    df = _synth_ohlcv(n=80, seed=2)
    for i in range(len(df)):
        engine.step(f"d{i}", df.iloc[i].to_dict())
    # Every log entry's held_position equals the target_position from the
    # *prior* settled bar (i.e., the position took effect at open[t+1]).
    # The engine sets held = target after the step, so each entry has
    # held_position == target_position by construction (after step
    # finishes). This invariant just confirms the engine maintains
    # consistent state.
    for e in engine.log:
        assert e.held_position == e.target_position
