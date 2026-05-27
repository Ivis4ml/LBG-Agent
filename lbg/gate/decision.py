"""The multi-objective per-trial validation gate (PROPOSAL.html §9 line 1704)."""

from __future__ import annotations

from dataclasses import dataclass

from backtest import BacktestResult
from lbg.gate.utility import utility_lcb_sharpe
from lbg.schemas import ValidationSignal


@dataclass(frozen=True)
class TrialOutcome:
    """Inputs the gate needs about either the incumbent or the candidate."""

    train: BacktestResult
    validation: BacktestResult
    complexity: float


@dataclass(frozen=True)
class GateConfig:
    """Per-trial gate thresholds. Locked per PROPOSAL.html §20 once a Discovery
    run starts; do not edit during a run.

    Use the default GateConfig() for any H1 reporting / paper claim. The
    classmethod `permissive()` returns a looser preset for experiment-stage
    campaigns that need a non-zero accept rate to exercise downstream
    machinery (alpha_cards/, sealed verdicts on a non-baseline incumbent).
    Results from the permissive preset MUST NOT be cited as H1 evidence.
    """

    min_trades: int = 20
    drawdown_regression_factor: float = 1.15
    max_turnover: float = 10.0
    max_complexity_delta: float = 3.0
    # train >> val (train Sharpe higher than val by this much) = the
    # classical train-overfit signal.
    max_train_val_sharpe_gap: float = 0.5
    # Cap on how much the candidate is allowed to WIDEN the val-train
    # Sharpe gap vs the incumbent. The v17 pathology was rearm tweaks
    # that lifted val while degrading train -- each tweak widened the
    # gap by ~0.15. Setting this to 0.2 catches the pattern. An
    # absolute-gap check (the prior design) doesn't work because
    # naturally favorable val windows can give legitimate baselines a
    # wide gap; the right semantic is "candidate must not make val-
    # overfit worse than incumbent already had it". Default 999.0 =
    # off on strict gate (preserves PROPOSAL §9). permissive() sets it.
    max_val_train_gap_widening: float = 999.0
    utility_lcb_alpha: float = 0.20
    eps: float = 0.005
    # Anti-knob-exploitation knobs (see permissive() docstring). On
    # strict gate they're off so behaviour is unchanged for the locked
    # H1 verdict path.
    strict_sharpe_for_sizing: bool = False
    # Same idea applied to ALL parameter_change edits (filter thresholds,
    # rearm_threshold, indicator params, sizing scalars). Catches the
    # v16 pathology where Editor exploited Pareto-on-MDD via rearm
    # threshold tweaks instead of sizing.fraction.
    strict_sharpe_for_parameter_change: bool = False
    max_consecutive_sizing_edits: int = 0
    # Hard cap on how many parameter_change edits of filter thresholds
    # (entry or exit, threshold or rearm_threshold) the campaign accepts
    # in total. Once the cap is hit, additional tweaks are rejected
    # with REASON_EXPLORATION_REQUIRED so the Editor is forced toward
    # add_indicator. 0 = unlimited (strict gate default).
    max_filter_tweaks_per_campaign: int = 0
    # Library-diversity discount on the Pareto branch (add_indicator only).
    # When the campaign's library has fewer than `library_diversity_goal`
    # distinct factor fn names, an add_indicator is allowed to *slightly
    # degrade* val Sharpe (up to `(goal - current) * per_missing` of
    # slack) as long as it Pareto-dominates on some other dimension
    # (MDD / turnover / complexity). This buys coverage of new factor
    # categories at the cost of strict val improvement. Once the library
    # hits the goal, the discount goes to 0 and Pareto returns to strict.
    library_diversity_discount_per_missing_factor: float = 0.0
    library_diversity_goal: int = 0

    @classmethod
    def permissive(cls) -> GateConfig:
        """Relaxed thresholds for experimentation only. Concretely:

          * min_trades 20 → 5    (let rearm-driven re-entries through to
                                  Pareto / utility evaluation -- previously
                                  10 was rejecting cases where an exit
                                  filter with rearm produced 5-9 trades)
          * utility_lcb_alpha 0.20 → 0.40  (looser one-sided CI on Sharpe gain)
          * drawdown_regression_factor 1.15 → 1.30  (tolerate larger DD)
          * strict_sharpe_for_sizing = True : for sizing-only edits (mode
            switch or sizing.* parameter change), require strict Sharpe
            improvement -- skip the Pareto-on-MDD shortcut that lets a
            chain of fraction cuts pass trivially. Sealed-Sharpe-killing
            "just keep scaling down" runaways are the v11 pathology this
            blocks.
          * strict_sharpe_for_parameter_change = True : same rule applied
            to ALL `parameter_change` edits (filter thresholds, rearm
            thresholds, indicator params, sizing scalars). Catches v16
            where the Editor exploited the Pareto-on-MDD loophole via
            rearm_threshold tweaks instead of sizing knobs. Structural
            edits (`add_indicator`, `add_filter`, `change_exit_rule`,
            `simplify`) still get the Pareto branch.
          * max_consecutive_sizing_edits = 2 : after this many sizing
            edits in a row have been accepted, the next sizing edit is
            rejected with `reject_exploration_required`. Forces Editor
            to diversify into add_indicator / filter changes before
            another sizing tweak.

        Used by `scripts/campaign.py --gate permissive` to surface real
        alpha_cards from the Discovery loop. Not for any statistical claim.
        """
        return cls(
            min_trades=5,
            utility_lcb_alpha=0.40,
            drawdown_regression_factor=1.30,
            strict_sharpe_for_sizing=True,
            strict_sharpe_for_parameter_change=True,
            max_consecutive_sizing_edits=2,
            max_filter_tweaks_per_campaign=1,
            # Block trials that WIDEN the val-train gap by >0.20 vs
            # the incumbent. Catches the v17 rearm-tweak pattern that
            # lifted val ~0.1-0.2 per accept while train degraded.
            max_val_train_gap_widening=0.20,
            # Library-diversity discount: 0.05 per missing factor up
            # to goal=5. At 2/5 → 0.15 val_sharpe slack on add_indicator
            # Pareto branch. At 5/5 → 0 (back to strict Pareto). Trades
            # marginal val accuracy for breadth of factor coverage.
            library_diversity_discount_per_missing_factor=0.05,
            library_diversity_goal=5,
        )


# Reason strings produced by `decide`. The first three accept reasons map to
# `ValidationSignal.ACCEPTED`; the rest map to the four reject categories.
REASON_UTILITY_IMPROVEMENT = "accept_utility_improvement"
REASON_PARETO_IMPROVEMENT = "accept_pareto_improvement"
REASON_TOO_FEW_TRADES = "reject_too_few_trades"
REASON_DRAWDOWN_REGRESSION = "reject_drawdown_regression"
REASON_TURNOVER_TOO_HIGH = "reject_turnover_too_high"
REASON_COMPLEXITY_INCREASE = "reject_complexity_increase"
REASON_OVERFIT_GAP = "reject_overfit_gap"
REASON_NO_MEANINGFUL_IMPROVEMENT = "reject_no_meaningful_improvement"
REASON_EXPLORATION_REQUIRED = "reject_exploration_required"


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reason: str
    # Diagnostic numbers, for the developer-facing trial record. The LLM
    # only ever sees the result of `redact(...)`.
    candidate_utility_lcb: float
    incumbent_utility_lcb: float
    complexity_delta: float


def pareto_dominates(
    candidate: TrialOutcome,
    incumbent: TrialOutcome,
    eps: float,
) -> bool:
    """Candidate Pareto-dominates incumbent on (sharpe, max_dd, turnover, complexity)?

    PROPOSAL.html §9: better on drawdown OR turnover OR complexity, without
    being worse on Sharpe. Direction conventions:
      - sharpe higher is better
      - max_drawdown higher (less negative) is better
      - turnover lower is better
      - complexity lower is better
    """
    cv, iv = candidate.validation, incumbent.validation
    not_worse_sharpe = cv.sharpe >= iv.sharpe - eps
    better_dd = cv.max_drawdown > iv.max_drawdown + eps
    better_turnover = cv.turnover < iv.turnover - eps
    better_complexity = candidate.complexity < incumbent.complexity - eps
    return not_worse_sharpe and (better_dd or better_turnover or better_complexity)


def decide(
    incumbent: TrialOutcome,
    candidate: TrialOutcome,
    config: GateConfig | None = None,
    *,
    edit_category: str | None = None,
    edit_type: str | None = None,
    edit_change_path: str | None = None,
    recent_accepted_sizing_streak: int = 0,
    recent_accepted_filter_tweak_count: int = 0,
    current_library_distinct_fns: int = 0,
) -> GateDecision:
    """Run the full gate. Order of checks matches PROPOSAL.html §9 pseudocode.

    `edit_category` / `edit_type` / `recent_accepted_sizing_streak` drive
    the anti-knob-exploitation rules (permissive gate only). With defaults
    the behaviour is unchanged from the locked PROPOSAL §9 path.
    """
    cfg = config or GateConfig()

    complexity_delta = candidate.complexity - incumbent.complexity

    u_cand = utility_lcb_sharpe(candidate.validation.returns, alpha=cfg.utility_lcb_alpha)
    u_incumbent = utility_lcb_sharpe(incumbent.validation.returns, alpha=cfg.utility_lcb_alpha)

    def _make(accepted: bool, reason: str) -> GateDecision:
        return GateDecision(
            accepted=accepted,
            reason=reason,
            candidate_utility_lcb=u_cand,
            incumbent_utility_lcb=u_incumbent,
            complexity_delta=complexity_delta,
        )

    is_sizing_edit = edit_category == "sizing"

    # Anti-sizing-exploitation rule #2: hard block after the run-length
    # cap is exceeded. Enforced before any other gate logic so the
    # rejection reason is unambiguous.
    if (
        is_sizing_edit
        and cfg.max_consecutive_sizing_edits > 0
        and recent_accepted_sizing_streak >= cfg.max_consecutive_sizing_edits
    ):
        return _make(False, REASON_EXPLORATION_REQUIRED)

    # Filter-tweak cap: total accepted parameter_change of filter
    # threshold or rearm_threshold edits across the campaign. Once the
    # cap is hit, additional tweaks of the SAME family are rejected to
    # force Editor onto add_indicator / structural paths.
    is_filter_tweak = (
        edit_type == "parameter_change"
        and edit_change_path is not None
        and (".threshold" in edit_change_path or ".rearm_threshold" in edit_change_path)
    )
    if (
        is_filter_tweak
        and cfg.max_filter_tweaks_per_campaign > 0
        and recent_accepted_filter_tweak_count >= cfg.max_filter_tweaks_per_campaign
    ):
        return _make(False, REASON_EXPLORATION_REQUIRED)

    if candidate.train.num_trades < cfg.min_trades:
        return _make(False, REASON_TOO_FEW_TRADES)

    # Val-overfit widening check. Catches the v17 pattern where each
    # rearm tweak widened the val-train gap by ~0.15 while train
    # Sharpe degraded. Measured as a delta vs the incumbent's gap so
    # legitimately val-favourable windows don't reject every trial.
    # Default 999.0 = off on strict gate.
    cand_gap = candidate.validation.sharpe - candidate.train.sharpe
    inc_gap = incumbent.validation.sharpe - incumbent.train.sharpe
    if cand_gap - inc_gap > cfg.max_val_train_gap_widening:
        return _make(False, REASON_OVERFIT_GAP)

    if (
        candidate.validation.max_drawdown
        < incumbent.validation.max_drawdown * cfg.drawdown_regression_factor
    ):
        return _make(False, REASON_DRAWDOWN_REGRESSION)

    if candidate.validation.turnover > cfg.max_turnover:
        return _make(False, REASON_TURNOVER_TOO_HIGH)

    if complexity_delta > cfg.max_complexity_delta:
        return _make(False, REASON_COMPLEXITY_INCREASE)

    if u_cand > u_incumbent + cfg.eps:
        return _make(True, REASON_UTILITY_IMPROVEMENT)

    # Pareto-on-MDD path. Default `pareto_dominates` only requires Sharpe
    # to be "not worse"; for knob-tuning edits under the permissive gate
    # that lets a chain of parameter tweaks pass trivially (v11 sizing
    # pathology, v16 rearm_threshold pathology). When the strict-Sharpe
    # rules are on, require strict Sharpe improvement before the Pareto
    # branch accepts these edits. Structural edits (add_indicator /
    # add_filter / change_exit_rule / simplify / revert) still get the
    # regular Pareto branch.
    #
    # For `add_indicator` specifically, the library-diversity discount
    # adds slack: when the library has fewer than goal distinct factor
    # fns, val_sharpe is allowed to drop by up to
    # `(goal - current) * per_missing`, provided the candidate Pareto-
    # dominates on another dimension. Catches the v21/v22 saturation
    # where an already-strong augmented baseline blocks any further
    # add_indicator on strict Pareto.
    is_param_change = edit_type == "parameter_change"
    is_add_indicator = edit_type == "add_indicator"

    # Compute the diversity discount: only widens the Sharpe tolerance,
    # not the "strictly better on MDD/turnover/complexity" eps.
    diversity_slack = 0.0
    if is_add_indicator and cfg.library_diversity_goal > 0:
        missing = max(0, cfg.library_diversity_goal - current_library_distinct_fns)
        diversity_slack = missing * cfg.library_diversity_discount_per_missing_factor

    if diversity_slack > 0.0:
        # Growth-phase Pareto: when the campaign library hasn't yet hit
        # its diversity goal, accept add_indicator on Sharpe-not-worse
        # alone (with the diversity slack). The "Pareto-better-on-
        # MDD-or-turnover-or-complexity" sub-requirement is dropped --
        # `add_indicator` mechanically increases complexity, so that
        # branch was systematically blocking library growth. The hard
        # `max_complexity_delta` check earlier in the gate still caps
        # how much the strategy can bloat per trial.
        cv = candidate.validation
        iv = incumbent.validation
        not_worse_sharpe = cv.sharpe >= iv.sharpe - cfg.eps - diversity_slack
        pareto_pass = not_worse_sharpe
    else:
        pareto_pass = pareto_dominates(candidate, incumbent, cfg.eps)

    require_strict_sharpe = (is_sizing_edit and cfg.strict_sharpe_for_sizing) or (
        is_param_change and cfg.strict_sharpe_for_parameter_change
    )
    if pareto_pass and require_strict_sharpe:
        sharpe_strictly_better = candidate.validation.sharpe > incumbent.validation.sharpe + cfg.eps
        if not sharpe_strictly_better:
            pareto_pass = False
    if pareto_pass:
        return _make(True, REASON_PARETO_IMPROVEMENT)

    if candidate.train.sharpe - candidate.validation.sharpe > cfg.max_train_val_sharpe_gap:
        return _make(False, REASON_OVERFIT_GAP)

    return _make(False, REASON_NO_MEANINGFUL_IMPROVEMENT)


def edit_category(
    edit_type: str,
    change: dict | None = None,
    summary: str | None = None,
) -> str | None:
    """Map an (edit_type, change-payload) pair to a category used by the
    anti-sizing-exploitation rules. Currently only the `sizing` category
    is recognised; everything else returns None.

    Two equivalent input forms supported:
      - Live call (Discovery._run_one_trial): pass `change` from the
        `ProposedEdit.change` dict. Looks at `change["path"]` for
        `parameter_change` edits.
      - Historical-record call (Discovery._recent_accepted_sizing_streak):
        pass `summary` (the `EditSummary.summary` string). The summary
        for sizing-related parameter_change starts with `"sizing."`.

    `sizing` covers both `change_sizing_mode` (always) and
    `parameter_change` whose path starts with `sizing.` (e.g.
    `sizing.fraction`, `sizing.target_vol`). The Editor cannot tune
    sizing through any other edit_type, so this captures the full set
    of sizing-mutating actions.
    """
    if edit_type == "change_sizing_mode":
        return "sizing"
    if edit_type == "parameter_change":
        if isinstance(change, dict):
            path = str(change.get("path", ""))
            if path.startswith("sizing."):
                return "sizing"
        if summary and summary.startswith("sizing."):
            return "sizing"
    return None


_REDACT_MAP: dict[str, ValidationSignal] = {
    REASON_UTILITY_IMPROVEMENT: ValidationSignal.ACCEPTED,
    REASON_PARETO_IMPROVEMENT: ValidationSignal.ACCEPTED,
    REASON_DRAWDOWN_REGRESSION: ValidationSignal.REJECTED_DRAWDOWN_REGRESSION,
    REASON_TURNOVER_TOO_HIGH: ValidationSignal.REJECTED_TURNOVER,
    REASON_COMPLEXITY_INCREASE: ValidationSignal.REJECTED_COMPLEXITY,
    # too_few_trades, overfit_gap, no_meaningful_improvement, and
    # exploration_required all map to the "no significant improvement"
    # bucket because the LLM should only know that the candidate didn't
    # actually advance the strategy. `exploration_required` is a
    # procedural meta-rule (too many consecutive sizing edits) and is
    # surfaced to the Editor via a dedicated prompt section rather than
    # the categorical signal.
    REASON_TOO_FEW_TRADES: ValidationSignal.REJECTED_NO_SIGNIFICANT_IMPROVEMENT,
    REASON_OVERFIT_GAP: ValidationSignal.REJECTED_NO_SIGNIFICANT_IMPROVEMENT,
    REASON_NO_MEANINGFUL_IMPROVEMENT: ValidationSignal.REJECTED_NO_SIGNIFICANT_IMPROVEMENT,
    REASON_EXPLORATION_REQUIRED: ValidationSignal.REJECTED_NO_SIGNIFICANT_IMPROVEMENT,
}


def redact(decision: GateDecision) -> ValidationSignal:
    """Map an internal gate decision to the LLM-facing categorical signal."""
    if decision.reason not in _REDACT_MAP:
        raise ValueError(f"unknown gate reason: {decision.reason!r}")
    return _REDACT_MAP[decision.reason]
