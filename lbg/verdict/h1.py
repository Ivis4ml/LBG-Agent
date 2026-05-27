"""Compute sealed verdicts (PROPOSAL.html §10.5).

The primary H1 in the current proposal is not "did the final strategy beat
buy-and-hold on one sealed split?" It is the count of alpha cards whose own
sealed leave-one-out incremental Sharpe CI lower bound is positive. The old
strategy-level moving-block bootstrap is still useful, but only as a
supplementary diagnostic.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from backtest import run_backtest
from lbg.alpha_cards import AlphaCard, AlphaCardStatus
from lbg.verdict.analysis_plan import DEFAULT_ANALYSIS_PLAN, AnalysisPlan
from lbg.verdict.baselines import BASELINE_REGISTRY
from lbg.verdict.bootstrap import (
    TRADING_DAYS_PER_YEAR,
    moving_block_bootstrap_sharpe_diff,
)


@dataclass(frozen=True)
class AlphaCardH1Verdict:
    candidate_card_count: int
    validated_factor_count: int
    validated_alpha_ids: tuple[str, ...]
    baseline_validated_counts: dict[str, int]
    best_baseline_count: int
    tau: int
    strong: bool
    weak: bool
    # Auxiliary column: cards whose raw CI lower bound > 0, no
    # multiple-testing correction. Retained so the report can show how
    # much the BH correction tightens the count, but H1 is decided on
    # `validated_factor_count` (= BH-validated count).
    ci_only_validated_count: int = 0
    ci_only_validated_alpha_ids: tuple[str, ...] = ()
    bh_q: float = 0.10

    def to_dict(self) -> dict:
        return {
            "candidate_card_count": self.candidate_card_count,
            "validated_factor_count": self.validated_factor_count,
            "validated_alpha_ids": list(self.validated_alpha_ids),
            "baseline_validated_counts": dict(self.baseline_validated_counts),
            "best_baseline_count": self.best_baseline_count,
            "tau": self.tau,
            "h1_strong": self.strong,
            "h1_weak": self.weak,
            "ci_only_validated_count": self.ci_only_validated_count,
            "ci_only_validated_alpha_ids": list(self.ci_only_validated_alpha_ids),
            "bh_q": self.bh_q,
        }


@dataclass(frozen=True)
class StrategySharpeVerdict:
    strategy_sharpe: float
    best_baseline_name: str
    best_baseline_sharpe: float
    delta_sharpe: float
    ci_lower: float
    ci_upper: float
    alpha: float
    n_bootstrap: int
    block_len: int
    strong: bool  # ci_lower > 0
    weak: bool  # delta_sharpe > 0

    def to_dict(self) -> dict:
        return {
            "comparison": "supplementary_strategy_sharpe",
            "strategy_sharpe": self.strategy_sharpe,
            "best_baseline_name": self.best_baseline_name,
            "best_baseline_sharpe": self.best_baseline_sharpe,
            "delta_sharpe": self.delta_sharpe,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "alpha": self.alpha,
            "n_bootstrap": self.n_bootstrap,
            "block_len": self.block_len,
            "supplementary_strong": self.strong,
            "supplementary_weak": self.weak,
        }


def compute_alpha_card_h1_verdict(
    cards: str | Path | Iterable[AlphaCard],
    *,
    plan: AnalysisPlan = DEFAULT_ANALYSIS_PLAN,
) -> AlphaCardH1Verdict:
    """Count alpha cards with sealed CI lower bound > 0.

    `cards` may be a repository root, an `alpha_cards/` directory, or an
    iterable of already-validated `AlphaCard` objects. Cards without
    `evidence.sealed_summary` are honest non-validations, not failures.
    """
    loaded = _load_alpha_cards(cards)
    # H1 (primary): cards with `bh_validated = True` in the sealed
    # summary. compute_per_card_sealed_validation runs BH across the
    # whole family of accepted cards and writes the flag back.
    bh_validated_ids: list[str] = []
    ci_only_ids: list[str] = []
    for card in loaded:
        if card.status != AlphaCardStatus.ACCEPTED:
            continue
        summary = card.evidence.sealed_summary or {}
        if summary.get("bh_validated", False):
            bh_validated_ids.append(card.alpha_id)
        ci_lower = _sealed_incremental_ci_lower(card)
        if ci_lower is not None and ci_lower > 0.0:
            ci_only_ids.append(card.alpha_id)

    baseline_counts = dict(plan.h1.baseline_validated_counts)
    best_baseline_count = max(baseline_counts.values(), default=0)
    n_val = len(bh_validated_ids)
    strong = n_val > best_baseline_count and n_val >= plan.h1.tau
    weak = n_val >= 1 and n_val >= best_baseline_count
    return AlphaCardH1Verdict(
        candidate_card_count=len(loaded),
        validated_factor_count=n_val,
        validated_alpha_ids=tuple(bh_validated_ids),
        baseline_validated_counts=baseline_counts,
        best_baseline_count=best_baseline_count,
        tau=plan.h1.tau,
        strong=strong,
        weak=weak,
        ci_only_validated_count=len(ci_only_ids),
        ci_only_validated_alpha_ids=tuple(ci_only_ids),
        bh_q=plan.per_card_validation.bh_q,
    )


def compute_strategy_sharpe_verdict(
    strategy_returns: pd.Series,
    sealed_df: pd.DataFrame,
    *,
    plan: AnalysisPlan = DEFAULT_ANALYSIS_PLAN,
    rng_seed: int = 42,
) -> StrategySharpeVerdict:
    """Backtest each baseline on `sealed_df`, pick the best, run bootstrap CI."""
    # Strategy Sharpe from its (already computed) realized returns.
    strat_arr = strategy_returns.to_numpy()

    # Sharpe + realized returns for each baseline.
    candidates: list[tuple[str, float, pd.Series]] = []
    for name in plan.baselines:
        if name not in BASELINE_REGISTRY:
            raise ValueError(f"unknown baseline {name!r}; plan must reference a registered one")
        positions = BASELINE_REGISTRY[name](sealed_df)
        bt = run_backtest(positions, sealed_df)
        candidates.append((name, bt.sharpe, bt.returns))
    if not candidates:
        raise ValueError("analysis plan has no baselines")

    # Pick the baseline with the highest point Sharpe.
    best_name, best_sharpe, best_returns = max(candidates, key=lambda c: c[1])

    # Align lengths: both series come from the same sealed_df and `run_backtest`
    # drops the last bar, so they're already length-aligned.
    base_arr = best_returns.to_numpy()
    if len(strat_arr) != len(base_arr):
        # Defensive trim in case future refactors change the alignment.
        n = min(len(strat_arr), len(base_arr))
        strat_arr = strat_arr[:n]
        base_arr = base_arr[:n]

    boot = moving_block_bootstrap_sharpe_diff(
        strat_arr,
        base_arr,
        block_len=plan.supplementary_strategy.block_len,
        n_bootstrap=plan.supplementary_strategy.n_bootstrap,
        alpha=plan.supplementary_strategy.alpha,
        rng_seed=rng_seed,
    )

    # ddof=1 matches `Series.std()` in `backtest._sharpe_annualized` so the
    # H1 strategy_sharpe lines up bit-for-bit with the backtest's sharpe.
    strat_std = strat_arr.std(ddof=1) if strat_arr.size > 1 else 0.0
    strat_sharpe = (
        float(strat_arr.mean() / strat_std * (TRADING_DAYS_PER_YEAR**0.5))
        if strat_std > 1e-12
        else 0.0
    )

    return StrategySharpeVerdict(
        strategy_sharpe=strat_sharpe,
        best_baseline_name=best_name,
        best_baseline_sharpe=float(best_sharpe),
        delta_sharpe=float(boot["point_estimate"]),
        ci_lower=float(boot["ci_lower"]),
        ci_upper=float(boot["ci_upper"]),
        alpha=float(boot["alpha"]),
        n_bootstrap=int(boot["n_bootstrap"]),
        block_len=int(boot["block_len"]),
        strong=bool(boot["ci_lower"] > 0),
        weak=bool(boot["point_estimate"] > 0),
    )


def compute_h1_verdict(
    strategy_returns: pd.Series,
    sealed_df: pd.DataFrame,
    *,
    plan: AnalysisPlan = DEFAULT_ANALYSIS_PLAN,
    rng_seed: int = 42,
) -> StrategySharpeVerdict:
    """Backward-compatible alias for the supplementary Sharpe verdict.

    New sealed payloads should call `compute_alpha_card_h1_verdict()` for
    primary H1 and store this result under `supplementary_strategy_verdict`.
    """
    return compute_strategy_sharpe_verdict(
        strategy_returns,
        sealed_df,
        plan=plan,
        rng_seed=rng_seed,
    )


def _load_alpha_cards(cards: str | Path | Iterable[AlphaCard]) -> list[AlphaCard]:
    if not isinstance(cards, (str, Path)):
        return list(cards)
    path = Path(cards)
    cards_dir = path / "alpha_cards" if (path / "alpha_cards").is_dir() else path
    if not cards_dir.exists():
        return []
    out: list[AlphaCard] = []
    for p in sorted(cards_dir.glob("trial_*.yaml")):
        out.append(AlphaCard.model_validate(yaml.safe_load(p.read_text(encoding="utf-8"))))
    return out


def _sealed_incremental_ci_lower(card: AlphaCard) -> float | None:
    summary = card.evidence.sealed_summary or {}
    for key in ("incremental_sharpe_ci_lower", "delta_sharpe_ci_lower", "ci_lower"):
        if key in summary:
            return float(summary[key])
    return None


# Back-compat name for older imports. The class now denotes the supplementary
# strategy-level verdict; primary H1 uses AlphaCardH1Verdict.
H1Verdict = StrategySharpeVerdict
