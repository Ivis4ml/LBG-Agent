"""B · forced fallback. When the most recent trial was rejected with a
non-empty `fallback_if_rejected`, the next Editor prompt must surface that
promise so the Editor either executes it or explicitly explains the
divergence."""

from __future__ import annotations

from lbg.orchestrator.context_builder import EditorContext, PastTrialSummary
from lbg.orchestrator.role_runner import _pending_fallback, _render_editor_user_prompt


def _rejected(trial_id: int, fallback: str) -> PastTrialSummary:
    return PastTrialSummary(
        trial_id=trial_id,
        edit_type="parameter_change",
        edit_summary="x",
        hypothesis_text="h",
        expected_validation_signal="accept",
        actual_validation_signal="rejected_drawdown_regression",
        hypothesis_outcome="disconfirmed",
        train_sharpe=0.5,
        train_max_drawdown=-0.2,
        train_turnover=2.0,
        train_num_trades=20,
        decision="reject",
        fallback_if_rejected=fallback,
    )


def _accepted(trial_id: int, fallback: str = "ignored") -> PastTrialSummary:
    return PastTrialSummary(
        trial_id=trial_id,
        edit_type="parameter_change",
        edit_summary="x",
        hypothesis_text="h",
        expected_validation_signal="accept",
        actual_validation_signal="accepted",
        hypothesis_outcome="confirmed",
        train_sharpe=0.7,
        train_max_drawdown=-0.1,
        train_turnover=1.0,
        train_num_trades=30,
        decision="accept",
        fallback_if_rejected=fallback,
    )


def _ctx(trials: list[PastTrialSummary]) -> EditorContext:
    return EditorContext(
        strategy_yaml="name: t\n",
        recent_trials=trials,
        semantic_memory={},
        skills=[],
        factor_hints=[],
    )


# -------- _pending_fallback selector --------


def test_pending_fallback_picks_most_recent_rejected(tmp_path):
    trials = [_rejected(0, "try X"), _rejected(1, "try Y")]
    p = _pending_fallback(trials)
    assert p is not None
    assert p.trial_id == 1
    assert p.fallback_if_rejected == "try Y"


def test_pending_fallback_none_when_last_accepted():
    """An accept clears any earlier pending promise -- the accepted trial
    discharged the search, so previous fallbacks no longer apply."""
    trials = [_rejected(0, "try X"), _accepted(1)]
    assert _pending_fallback(trials) is None


def test_pending_fallback_none_when_no_rejected_trials():
    assert _pending_fallback([_accepted(0)]) is None


def test_pending_fallback_skips_rejected_trial_with_empty_fallback():
    """A trial that didn't write a fallback note can't promise anything."""
    trials = [_rejected(0, "try X"), _rejected(1, "")]
    p = _pending_fallback(trials)
    assert p is not None
    assert p.trial_id == 0  # skip the empty fallback, fall back to trial 0


def test_pending_fallback_none_on_empty_history():
    assert _pending_fallback([]) is None


# -------- prompt surface --------


def test_prompt_includes_pending_fallback_section(tmp_path):
    prompt = _render_editor_user_prompt(
        _ctx([_rejected(3, "switch to volatility_target sizing")]),
        trial_id=4,
    )
    assert "previously promised fallback" in prompt
    assert "switch to volatility_target sizing" in prompt
    assert "trial 3" in prompt


def test_prompt_omits_section_when_no_pending_fallback(tmp_path):
    prompt = _render_editor_user_prompt(_ctx([_accepted(3)]), trial_id=4)
    assert "previously promised fallback" not in prompt


def test_prompt_omits_section_on_empty_history(tmp_path):
    prompt = _render_editor_user_prompt(_ctx([]), trial_id=0)
    assert "previously promised fallback" not in prompt
