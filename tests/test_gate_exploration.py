"""Tests for the anti-sizing-exploitation gate rules.

Two rules are tested:

  1. `strict_sharpe_for_sizing` -- sizing edits cannot pass via the
     Pareto-on-MDD branch; they must show strict Sharpe improvement.
  2. `max_consecutive_sizing_edits` -- after N consecutive accepted
     sizing edits, the next one is rejected with `reject_exploration_required`.

The strict-gate path (current PROPOSAL §9 default) is unaffected:
backward-compatibility coverage in test_gate.py / test_gate_permissive.py
still passes without changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import BacktestResult
from lbg.gate.decision import (
    REASON_EXPLORATION_REQUIRED,
    REASON_NO_MEANINGFUL_IMPROVEMENT,
    REASON_PARETO_IMPROVEMENT,
    REASON_UTILITY_IMPROVEMENT,
    GateConfig,
    TrialOutcome,
    decide,
    edit_category,
    redact,
)
from lbg.schemas import ValidationSignal


def _bt(sharpe: float, mdd: float, turnover: float, n_trades: int) -> BacktestResult:
    """Helper backtest result. Returns a flat returns series so the
    utility_lcb computation has something to chew on but the test is
    really about the deterministic gate logic."""
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0001, 0.01, 250))
    return BacktestResult(
        sharpe=sharpe,
        max_drawdown=mdd,
        turnover=turnover,
        num_trades=n_trades,
        n_bars_used=250,
        cost_total=0.0,
        cagr=0.05,
        final_equity=1.1,
        returns=returns,
        equity=(1.0 + returns).cumprod(),
    )


def _outcome(sharpe: float, mdd: float, complexity: float = 2.5) -> TrialOutcome:
    bt = _bt(sharpe, mdd, turnover=1.0, n_trades=50)
    return TrialOutcome(train=bt, validation=bt, complexity=complexity)


# -------- edit_category mapping --------


def test_category_change_sizing_mode_is_sizing():
    assert edit_category("change_sizing_mode") == "sizing"


def test_category_parameter_change_sizing_path_is_sizing():
    assert edit_category("parameter_change", {"path": "sizing.fraction", "value": 0.8}) == "sizing"
    assert (
        edit_category("parameter_change", {"path": "sizing.target_vol", "value": 0.12}) == "sizing"
    )


def test_category_parameter_change_non_sizing_returns_none():
    assert (
        edit_category("parameter_change", {"path": "indicators[sma].params.period", "value": 30})
        is None
    )
    assert edit_category("parameter_change", {"path": "filters[0].threshold", "value": 25}) is None


def test_category_other_edit_types_return_none():
    assert edit_category("add_indicator") is None
    assert edit_category("add_filter") is None
    assert edit_category("simplify") is None


def test_category_via_summary_summary_form():
    """Historical-record call site uses the summary string."""
    assert edit_category("parameter_change", summary="sizing.fraction: 1.0 -> 0.8") == "sizing"
    assert edit_category("parameter_change", summary="filters[0].threshold: 20 -> 30") is None


# -------- rule 1: strict Sharpe for sizing --------


def test_sizing_edit_passes_when_sharpe_strictly_better():
    """Strict-sharpe rule: sizing edit with Sharpe up still passes via the
    utility_lcb branch (rule 1 doesn't block legitimate improvements)."""
    cfg = GateConfig.permissive()
    inc = _outcome(sharpe=0.5, mdd=-0.20)
    cand = _outcome(sharpe=0.7, mdd=-0.15)  # better sharpe AND better mdd
    d = decide(inc, cand, cfg, edit_category="sizing", recent_accepted_sizing_streak=0)
    # Either utility_lcb or pareto -- both are legal accepts here.
    assert d.accepted is True


def test_sizing_edit_blocked_when_only_mdd_improves():
    """The v11 pathology: same Sharpe, smaller MDD. Strict-sharpe rule
    blocks the Pareto-on-MDD path so this gets rejected."""
    cfg = GateConfig.permissive()
    inc = _outcome(sharpe=0.5, mdd=-0.20)
    cand = _outcome(sharpe=0.5, mdd=-0.10)  # same sharpe, smaller mdd
    d = decide(inc, cand, cfg, edit_category="sizing", recent_accepted_sizing_streak=0)
    assert d.accepted is False
    assert d.reason == REASON_NO_MEANINGFUL_IMPROVEMENT


def test_non_sizing_edit_unaffected_by_strict_sharpe_rule():
    """A filter / indicator edit with same Sharpe + smaller MDD still
    passes via Pareto-on-MDD when not categorised as sizing."""
    cfg = GateConfig.permissive()
    inc = _outcome(sharpe=0.5, mdd=-0.20)
    cand = _outcome(sharpe=0.5, mdd=-0.10)
    d = decide(inc, cand, cfg, edit_category=None, recent_accepted_sizing_streak=0)
    assert d.accepted is True
    assert d.reason == REASON_PARETO_IMPROVEMENT


# -------- rule 2: max consecutive sizing edits --------


def test_third_consecutive_sizing_edit_rejected_with_exploration_required():
    cfg = GateConfig.permissive()
    # Even a perfectly good sizing edit gets rejected once the streak cap hits.
    inc = _outcome(sharpe=0.5, mdd=-0.20)
    cand = _outcome(sharpe=0.9, mdd=-0.10)
    d = decide(inc, cand, cfg, edit_category="sizing", recent_accepted_sizing_streak=2)
    assert d.accepted is False
    assert d.reason == REASON_EXPLORATION_REQUIRED


def test_streak_of_one_does_not_block():
    cfg = GateConfig.permissive()
    inc = _outcome(sharpe=0.5, mdd=-0.20)
    cand = _outcome(sharpe=0.9, mdd=-0.10)
    d = decide(inc, cand, cfg, edit_category="sizing", recent_accepted_sizing_streak=1)
    assert d.accepted is True


def test_non_sizing_edit_does_not_count_against_streak_block():
    """Streak cap only applies to sizing edits; a filter edit goes through
    even when the streak counter is high."""
    cfg = GateConfig.permissive()
    inc = _outcome(sharpe=0.5, mdd=-0.20)
    cand = _outcome(sharpe=0.5, mdd=-0.10)
    d = decide(inc, cand, cfg, edit_category=None, recent_accepted_sizing_streak=5)
    assert d.accepted is True


def test_strict_gate_unaffected_by_anti_exploitation_rules():
    """The strict GateConfig() default keeps both knobs off, so behaviour
    must match the locked PROPOSAL §9 path even when sizing streak is
    high."""
    cfg = GateConfig()  # defaults: strict_sharpe_for_sizing=False, max_consecutive_sizing_edits=0
    inc = _outcome(sharpe=0.5, mdd=-0.20)
    cand = _outcome(sharpe=0.5, mdd=-0.10)
    d = decide(inc, cand, cfg, edit_category="sizing", recent_accepted_sizing_streak=10)
    # Sizing-only edit with same Sharpe + smaller MDD passes via Pareto
    # in strict-gate mode (unchanged behaviour).
    assert d.accepted is True
    assert d.reason == REASON_PARETO_IMPROVEMENT


# -------- redact --------


def test_exploration_required_redacts_to_no_significant_improvement():
    """The procedural rule hides behind the standard 'no improvement'
    category-LLM only sees its proposed edit didn't help. The dedicated
    Editor-prompt section is what explains the streak rule."""
    from lbg.gate.decision import GateDecision

    d = GateDecision(
        accepted=False,
        reason=REASON_EXPLORATION_REQUIRED,
        candidate_utility_lcb=0.0,
        incumbent_utility_lcb=0.0,
        complexity_delta=0.0,
    )
    assert redact(d) == ValidationSignal.REJECTED_NO_SIGNIFICANT_IMPROVEMENT


# -------- defensive --------


def test_decide_default_behaviour_unchanged():
    """Calling decide() without the new kwargs (the way existing tests do)
    must produce identical results."""
    cfg = GateConfig.permissive()
    inc = _outcome(sharpe=0.5, mdd=-0.20)
    cand = _outcome(sharpe=0.5, mdd=-0.10)
    d_old_call = decide(inc, cand, cfg)
    # Without edit_category, neither rule fires -- same path as a non-sizing edit.
    assert d_old_call.accepted is True
    assert d_old_call.reason == REASON_PARETO_IMPROVEMENT


def test_filter_tweak_cap_blocks_second_rearm_change():
    """v17 pathology: Editor keeps tuning exit_filters[0].rearm_threshold
    once per iter. Permissive cap=1 blocks the second tweak with
    REASON_EXPLORATION_REQUIRED."""
    cfg = GateConfig.permissive()
    inc = _outcome(sharpe=0.5, mdd=-0.20)
    cand = _outcome(sharpe=0.9, mdd=-0.10)  # would otherwise pass

    d = decide(
        inc,
        cand,
        cfg,
        edit_category=None,
        edit_type="parameter_change",
        edit_change_path="exit_filters[0].rearm_threshold",
        recent_accepted_filter_tweak_count=1,  # one tweak already accepted
    )
    assert d.accepted is False
    assert d.reason == REASON_EXPLORATION_REQUIRED


def test_first_filter_tweak_under_cap_passes():
    """First tweak of a filter still goes through; only the second+ blocks."""
    cfg = GateConfig.permissive()
    inc = _outcome(sharpe=0.5, mdd=-0.20)
    cand = _outcome(sharpe=0.9, mdd=-0.10)
    d = decide(
        inc,
        cand,
        cfg,
        edit_category=None,
        edit_type="parameter_change",
        edit_change_path="exit_filters[0].rearm_threshold",
        recent_accepted_filter_tweak_count=0,
    )
    assert d.accepted is True


def test_val_widening_gap_blocks_when_candidate_widens_meaningfully():
    """v17 rearm-tweak shape: incumbent val_train_gap=0.6, candidate
    val_train_gap=0.85 (widened by 0.25 > 0.20 cap). Should reject."""
    from lbg.gate.decision import REASON_OVERFIT_GAP

    cfg = GateConfig.permissive()
    # incumbent: train 0.5, val 1.1 → gap = 0.6
    inc = TrialOutcome(
        train=_bt(0.5, -0.20, 1.0, 30),
        validation=_bt(1.1, -0.10, 1.0, 30),
        complexity=2.4,
    )
    # candidate: train 0.45, val 1.3 → gap = 0.85 (widened by 0.25)
    cand = TrialOutcome(
        train=_bt(0.45, -0.18, 1.0, 30),
        validation=_bt(1.3, -0.08, 1.0, 30),
        complexity=2.4,
    )
    d = decide(inc, cand, cfg, edit_type="parameter_change")
    assert d.accepted is False
    assert d.reason == REASON_OVERFIT_GAP


def test_val_widening_naturally_wide_baseline_passes_unchanged_gap():
    """If incumbent already has gap=0.94 (v18 baseline) and candidate
    keeps the same gap, the delta-based check does NOT reject. (The
    absolute-gap design from earlier would have wrongly rejected.)"""
    cfg = GateConfig.permissive()
    inc = TrialOutcome(
        train=_bt(0.72, -0.20, 1.0, 30),
        validation=_bt(1.65, -0.10, 1.0, 30),
        complexity=4.14,
    )
    # candidate: identical gap (1.65 - 0.72 = 0.93)
    cand = TrialOutcome(
        train=_bt(0.74, -0.18, 1.0, 30),
        validation=_bt(1.66, -0.08, 1.0, 30),  # gap 0.92 ~ unchanged
        complexity=4.27,
    )
    d = decide(inc, cand, cfg, edit_type="add_indicator")
    # NOT rejected on overfit_gap. May pass or fail elsewhere; key is
    # the rule didn't fire just because the baseline has a wide gap.
    from lbg.gate.decision import REASON_OVERFIT_GAP

    assert d.reason != REASON_OVERFIT_GAP


def test_val_widening_off_in_strict_gate():
    """The widening check is permissive-only; strict gate (locked) must
    not fire even when gap widens dramatically."""
    cfg = GateConfig()  # strict
    inc = TrialOutcome(
        train=_bt(0.5, -0.20, 1.0, 30),
        validation=_bt(0.5, -0.20, 1.0, 30),
        complexity=2.4,
    )
    cand = TrialOutcome(
        train=_bt(0.3, -0.18, 1.0, 30),
        validation=_bt(1.5, -0.08, 1.0, 30),  # gap = 1.2 widening of 1.2
        complexity=2.4,
    )
    from lbg.gate.decision import REASON_OVERFIT_GAP

    d = decide(inc, cand, cfg, edit_type="parameter_change")
    assert d.reason != REASON_OVERFIT_GAP


def test_library_diversity_discount_accepts_marginal_add_indicator():
    """v21/v22 saturation: with library at 2/5, a candidate that drops
    val Sharpe slightly should still accept under diversity discount
    even without Pareto-better-on-MDD. add_indicator mechanically
    raises complexity so that branch was systematically blocking
    library growth; growth-phase rule drops it."""
    cfg = GateConfig.permissive()
    # incumbent val 1.83 (the v22 augmented baseline level)
    inc = TrialOutcome(
        train=_bt(0.78, -0.20, 1.0, 50),
        validation=_bt(1.83, -0.20, 1.0, 50),
        complexity=5.83,
    )
    # candidate: val drops to 1.78 (-0.05 worse), complexity bumps up
    # (typical add_indicator pattern). Without growth-phase rule this
    # would fail (no other-dim Pareto improvement). With 3 missing
    # factors × 0.05 = 0.15 slack, val drop of 0.05 is OK and we
    # don't require strict Pareto-better-on-other.
    cand = TrialOutcome(
        train=_bt(0.78, -0.18, 1.0, 50),
        validation=_bt(1.78, -0.20, 1.0, 50),  # same MDD
        complexity=6.83,  # complexity +1
    )
    d = decide(
        inc,
        cand,
        cfg,
        edit_type="add_indicator",
        current_library_distinct_fns=2,
    )
    assert d.accepted is True
    assert d.reason == REASON_PARETO_IMPROVEMENT


def test_library_diversity_discount_off_when_library_at_goal():
    """At library=5/5 the discount goes to 0; same candidate as above
    no longer passes."""
    cfg = GateConfig.permissive()
    inc = TrialOutcome(
        train=_bt(0.78, -0.20, 1.0, 50),
        validation=_bt(1.83, -0.20, 1.0, 50),
        complexity=5.83,
    )
    cand = TrialOutcome(
        train=_bt(0.78, -0.18, 1.0, 50),
        validation=_bt(1.78, -0.10, 1.0, 50),
        complexity=5.83,
    )
    d = decide(
        inc,
        cand,
        cfg,
        edit_type="add_indicator",
        current_library_distinct_fns=5,
    )
    assert d.accepted is False


def test_library_diversity_discount_does_not_apply_to_parameter_change():
    """Knob-tuning edits still face the strict rule even when library
    < goal. The discount is for structural growth only."""
    cfg = GateConfig.permissive()
    inc = TrialOutcome(
        train=_bt(0.78, -0.20, 1.0, 50),
        validation=_bt(1.83, -0.20, 1.0, 50),
        complexity=5.83,
    )
    cand = TrialOutcome(
        train=_bt(0.78, -0.18, 1.0, 50),
        validation=_bt(1.78, -0.10, 1.0, 50),
        complexity=5.83,
    )
    d = decide(
        inc,
        cand,
        cfg,
        edit_type="parameter_change",
        current_library_distinct_fns=2,
    )
    assert d.accepted is False


def test_diversity_goal_configurable_via_dataclasses_replace():
    """The scripts/campaign.py path uses dataclasses.replace() to swap
    library_diversity_goal at runtime. Verify the gate respects the
    overridden value (e.g. --goal 10 should keep the discount active
    at library=6/10 even though the hardcoded permissive() default is 5)."""
    import dataclasses as _dc

    cfg = _dc.replace(GateConfig.permissive(), library_diversity_goal=10)
    assert cfg.library_diversity_goal == 10
    # At library=6/10, 4 missing × 0.05 = 0.20 sharpe slack.
    inc = TrialOutcome(
        train=_bt(0.78, -0.20, 1.0, 50),
        validation=_bt(1.83, -0.20, 1.0, 50),
        complexity=5.83,
    )
    cand = TrialOutcome(
        train=_bt(0.78, -0.18, 1.0, 50),
        validation=_bt(1.66, -0.20, 1.0, 50),  # val drops 0.17, within 0.20 slack
        complexity=6.83,
    )
    d = decide(inc, cand, cfg, edit_type="add_indicator", current_library_distinct_fns=6)
    assert d.accepted is True
    # Without override (goal=5), library=6 exceeds goal so discount=0;
    # same candidate should reject under default permissive.
    cfg_default = GateConfig.permissive()
    d_default = decide(
        inc, cand, cfg_default, edit_type="add_indicator", current_library_distinct_fns=6
    )
    assert d_default.accepted is False


def test_library_diversity_discount_off_in_strict_gate():
    """Strict gate has discount=0 by default; library < goal doesn't
    relax anything."""
    cfg = GateConfig()  # strict
    inc = TrialOutcome(
        train=_bt(0.78, -0.20, 1.0, 50),
        validation=_bt(1.83, -0.20, 1.0, 50),
        complexity=5.83,
    )
    cand = TrialOutcome(
        train=_bt(0.78, -0.18, 1.0, 50),
        validation=_bt(1.78, -0.10, 1.0, 50),
        complexity=5.83,
    )
    d = decide(
        inc,
        cand,
        cfg,
        edit_type="add_indicator",
        current_library_distinct_fns=0,
    )
    assert d.accepted is False


def test_parameter_change_non_sizing_also_blocked_by_strict_rule():
    """v16 pathology: rearm_threshold tweaks (parameter_change but path
    is not sizing.*) were exploiting Pareto-on-MDD. The generalised
    strict_sharpe_for_parameter_change rule blocks this too."""
    cfg = GateConfig.permissive()
    inc = _outcome(sharpe=0.5, mdd=-0.20)
    cand = _outcome(sharpe=0.5, mdd=-0.10)  # same sharpe, smaller mdd
    d = decide(
        inc,
        cand,
        cfg,
        edit_category=None,  # not a sizing edit
        edit_type="parameter_change",  # but a parameter_change of, say, rearm_threshold
        recent_accepted_sizing_streak=0,
    )
    assert d.accepted is False
    assert d.reason == REASON_NO_MEANINGFUL_IMPROVEMENT


def test_add_indicator_still_uses_pareto_path():
    """Structural edits keep the Pareto-on-MDD path; only knob-tuning
    edits are tightened."""
    cfg = GateConfig.permissive()
    inc = _outcome(sharpe=0.5, mdd=-0.20)
    cand = _outcome(sharpe=0.5, mdd=-0.10)
    d = decide(
        inc,
        cand,
        cfg,
        edit_category=None,
        edit_type="add_indicator",
        recent_accepted_sizing_streak=0,
    )
    assert d.accepted is True
    assert d.reason == REASON_PARETO_IMPROVEMENT


def test_utility_lcb_accept_works_for_sizing_too():
    """The utility_lcb_sharpe accept path is unaffected by the strict
    rule -- only the Pareto-on-MDD branch is gated."""
    cfg = GateConfig.permissive()
    # Construct outcomes whose utility_lcb difference is dominated by sharpe.
    inc = _outcome(sharpe=0.2, mdd=-0.20)
    cand = _outcome(sharpe=0.8, mdd=-0.15)
    d = decide(inc, cand, cfg, edit_category="sizing", recent_accepted_sizing_streak=0)
    assert d.accepted is True
    assert d.reason in (REASON_UTILITY_IMPROVEMENT, REASON_PARETO_IMPROVEMENT)
