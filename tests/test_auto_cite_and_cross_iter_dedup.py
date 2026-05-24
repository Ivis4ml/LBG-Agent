"""Two dedup fixes after campaign v3 diagnostics:

Fix 1: auto-extract `cited_factors` from indicator fn / name when the
  Editor forgot the field. Campaign v3 had only 4/11 trials with
  structured citations; the rest were factors the Editor mentioned in
  hypothesis text but didn't list explicitly.

Fix 2: cross-iteration dedup. The Editor used to re-propose ADX in
  iter 2 even though iter 0 had already cited it -- event memory wipes
  between iterations were carrying the dedup miss. Now a separate
  `memory/tried_factors.jsonl` persists across resets and feeds the
  ContextBuilder hint pipeline.
"""

from __future__ import annotations

from pathlib import Path

from lbg.dsl import load_strategy
from lbg.knowledge.factors import extract_factor_names_from_string
from lbg.memory import MemoryManager, TriedFactorRecord
from lbg.orchestrator.context_builder import ContextBuilder

REPO = Path(__file__).resolve().parents[1]


# -------- fix 1: auto-extract --------


def test_extract_picks_up_adx_from_indicator_fn():
    """`adx_14` should resolve to dossier name 'ADX'."""
    names = extract_factor_names_from_string("adx_14 adx")
    assert "ADX" in names


def test_extract_returns_longest_match_first():
    """If multiple dossiers match (e.g. RSI vs RSIDivergence), the
    longer / more specific name should appear first."""
    names = extract_factor_names_from_string("aroonoscillator and aroon")
    # AroonOscillator is longer than Aroon; both match the haystack.
    if "AroonOscillator" in names and "Aroon" in names:
        assert names.index("AroonOscillator") < names.index("Aroon")


def test_extract_minlen_filter_rejects_two_char_dossiers():
    """The 'AD' dossier should never match every 'add_*' / 'amihud_*'
    indicator name; the 3-char floor catches that."""
    names = extract_factor_names_from_string("sma_fast sma_slow")
    assert "AD" not in names


def test_extract_dedupes_repeated_names():
    """A haystack mentioning 'adx adx adx' should only return ADX once."""
    names = extract_factor_names_from_string("adx adx adx_14")
    assert names.count("ADX") == 1


def test_extract_empty_haystack_returns_empty():
    assert extract_factor_names_from_string("") == []
    assert extract_factor_names_from_string("   ") == []


# -------- fix 2: cross-iteration persistence --------


def test_tried_factors_jsonl_round_trip(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    mm.append_tried_factors(
        TriedFactorRecord(trial_id=0, factors=["ADX"], decision="reject", source="editor_cite")
    )
    mm.append_tried_factors(
        TriedFactorRecord(
            trial_id=1, factors=["LSMA", "ADX"], decision="reject", source="auto_extract"
        )
    )
    records = mm.read_tried_factors()
    assert len(records) == 2
    assert records[0].factors == ["ADX"]
    assert records[1].source == "auto_extract"


def test_context_builder_reads_persistent_tried_factors(tmp_path):
    """A jsonl entry from a prior iteration must mark the factor tried
    even when `recent_trials` is empty (the dedup gap we hit in v3)."""
    mm = MemoryManager(tmp_path / "memory")
    mm.append_tried_factors(
        TriedFactorRecord(trial_id=5, factors=["ADX"], decision="reject", source="editor_cite")
    )
    cb = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")
    tried = cb._tried_factor_names(strategy, recent=[])
    assert "ADX" in tried


def test_context_builder_unions_recent_and_persistent(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    # iter 0 logged in the jsonl
    mm.append_tried_factors(
        TriedFactorRecord(trial_id=0, factors=["ADX"], decision="reject", source="editor_cite")
    )
    cb = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")
    # current iter still has its in-memory recent_trials; both signals should
    # contribute to the tried set.
    from lbg.orchestrator.context_builder import PastTrialSummary

    recent = [
        PastTrialSummary(
            trial_id=10,
            edit_type="add_indicator",
            edit_summary="rsi_14 (rsi) attached as indicator_above:rsi_14",
            hypothesis_text="",
            expected_validation_signal="accept",
            actual_validation_signal="rejected_no_significant_improvement",
            hypothesis_outcome="disconfirmed",
            train_sharpe=0.5,
            train_max_drawdown=-0.2,
            train_turnover=2.0,
            train_num_trades=20,
            decision="reject",
            cited_factors=("RSI",),
        )
    ]
    tried = cb._tried_factor_names(strategy, recent)
    assert "ADX" in tried  # from jsonl
    assert "RSI" in tried  # from recent


def test_hints_exclude_persistent_tried(tmp_path):
    """The full _factor_hints pipeline filters jsonl-tried factors out."""
    mm = MemoryManager(tmp_path / "memory")
    mm.append_tried_factors(
        TriedFactorRecord(trial_id=0, factors=["ADX"], decision="reject", source="editor_cite")
    )
    cb = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")
    hints = cb._factor_hints(strategy, recent=[])
    names = {h.name for h in hints}
    assert "ADX" not in names


def test_tried_factors_survives_event_memory_wipe(tmp_path):
    """Sanity: the jsonl file is in memory_dir but NOT covered by the
    campaign's `_EVENT_MEMORY_FILENAMES` wipe list."""
    from lbg.orchestrator.campaign import _EVENT_MEMORY_FILENAMES

    assert "tried_factors.jsonl" not in _EVENT_MEMORY_FILENAMES
