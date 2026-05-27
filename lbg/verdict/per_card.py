"""Per-card sealed validation (PROPOSAL.html §10.5, analysis_plan.yaml).

This is the producer side of the primary H1. `compute_alpha_card_h1_verdict`
in `h1.py` counts cards whose `evidence.sealed_summary` shows the
incremental Sharpe CI lower bound > 0. Without a producer that populates
that field, `validated_factor_count` would be structurally zero. This
module is that producer.

Operational definition (pathwise incremental Sharpe):
  - parent commit  = `card.source_commit`, the strategy *before* the trial
  - trial commit   = the first commit whose subject starts with
                     `trial {source_trial:04d}:`
  - backtest both on the sealed window
  - paired daily returns -> moving-block bootstrap of `Sharpe(trial) - Sharpe(parent)`
  - persist `{incremental_sharpe_point, incremental_sharpe_ci_lower,
    incremental_sharpe_ci_upper, n_bootstrap, block_len, alpha}` into
    `card.evidence.sealed_summary`, then rewrite the card YAML.

If the trial commit cannot be located, or the indicator source cannot be
read at that ref, the card is annotated with `sealed_summary = {"validation_error": ...}`.
That is an honest non-validation; it does not crash the seal step.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from backtest import run_backtest
from lbg.alpha_cards import AlphaCard, AlphaCardStatus
from lbg.dsl import load_strategy
from lbg.git_manager import GitCommandError, GitManager
from lbg.verdict.analysis_plan import DEFAULT_ANALYSIS_PLAN, AnalysisPlan
from lbg.verdict.bootstrap import moving_block_bootstrap_sharpe_diff
from lbg.verdict.dsr import (
    DEFAULT_DSR_THRESHOLD,
    deflated_sharpe_ratio_from_returns,
)
from lbg.verdict.multiple_testing import DEFAULT_Q, benjamini_hochberg
from lbg.verdict.spa import SPAResult, hansen_spa
from policy_interpreter import compute_positions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerCardValidationOutcome:
    """Per-card list + family-wise SPA verdict for one sealed pass.

    `per_card` carries the standard list of one result per accepted
    card. `spa` is the Hansen 2005 SPA p-value bundle computed across
    the family of cards whose CI computation succeeded; it is `None`
    when the family is empty (no accepted cards or all errored).
    """

    per_card: list["PerCardValidationResult"]
    spa: SPAResult | None

    def __iter__(self):
        """Backward-compat: callers that used to receive a plain list
        still see the per-card iteration. Anything new should use
        `.per_card` and `.spa` explicitly."""
        return iter(self.per_card)

    def __len__(self) -> int:
        return len(self.per_card)

    def __getitem__(self, idx):
        return self.per_card[idx]


@dataclass(frozen=True)
class PerCardValidationResult:
    alpha_id: str
    source_trial: int
    incremental_sharpe_point: float
    incremental_sharpe_ci_lower: float
    incremental_sharpe_ci_upper: float
    n_bootstrap: int
    block_len: int
    alpha: float
    p_value_one_sided: float = 1.0
    bh_validated: bool = False
    bh_q: float = DEFAULT_Q
    # Bailey-LdP Deflated Sharpe Ratio. -1.0 means "not computed"
    # (insufficient trial-search context); a real DSR is in [0, 1].
    deflated_sharpe_ratio: float = -1.0
    dsr_passes: bool = False
    dsr_threshold: float = 0.95
    error: str | None = None

    def to_summary(self) -> dict[str, float | str | int | bool]:
        if self.error is not None:
            return {"validation_error": self.error}
        out: dict[str, float | str | int | bool] = {
            "incremental_sharpe_point": self.incremental_sharpe_point,
            "incremental_sharpe_ci_lower": self.incremental_sharpe_ci_lower,
            "incremental_sharpe_ci_upper": self.incremental_sharpe_ci_upper,
            "n_bootstrap": self.n_bootstrap,
            "block_len": self.block_len,
            "alpha": self.alpha,
            "p_value_one_sided": self.p_value_one_sided,
            "bh_validated": self.bh_validated,
            "bh_q": self.bh_q,
        }
        if self.deflated_sharpe_ratio >= 0.0:
            out["deflated_sharpe_ratio"] = self.deflated_sharpe_ratio
            out["dsr_passes"] = self.dsr_passes
            out["dsr_threshold"] = self.dsr_threshold
        return out


def compute_per_card_sealed_validation(
    repo_root: str | Path,
    sealed_df: pd.DataFrame,
    *,
    plan: AnalysisPlan = DEFAULT_ANALYSIS_PLAN,
    timeout_sec: float = 30.0,
    rng_seed: int = 42,
    git: GitManager | None = None,
    n_trials_attempted: int | None = None,
    trial_sharpes_annualised: list[float] | None = None,
    dsr_threshold: float = DEFAULT_DSR_THRESHOLD,
) -> PerCardValidationOutcome:
    """Populate `evidence.sealed_summary` on every accepted card in `alpha_cards/`.

    Mutates the per-card YAML files in place. Returns
    `PerCardValidationOutcome(per_card, spa)`: a list of one result per
    accepted card (in card-path order) plus the family-wise Hansen 2005
    SPA verdict. Cards whose per-card evaluation fails are tagged with
    a `validation_error` in `sealed_summary` rather than dropped — they
    are excluded from both the BH family and the SPA family but the
    alpha card lineage stays intact for audit.

    For backward compatibility, the returned outcome is iterable and
    supports `len()` / indexing, so existing callers that treated the
    return value as `list[PerCardValidationResult]` continue to work.
    """
    repo_root = Path(repo_root).resolve()
    cards_dir = repo_root / "alpha_cards"
    if not cards_dir.exists():
        return PerCardValidationOutcome(per_card=[], spa=None)
    git = git or GitManager(repo_root)

    # Pass 1: compute per-card incremental Sharpe + one-sided p-value, and
    # collect the daily diff series for the family-wise SPA test. Do not
    # write to disk yet — we need the family of p-values first to apply
    # Benjamini-Hochberg correctly.
    bh_q = plan.per_card_validation.bh_q
    card_paths: list[Path] = []
    cards: list[AlphaCard] = []
    results: list[PerCardValidationResult] = []
    diff_series_list: list[np.ndarray | None] = []
    for card_path in sorted(cards_dir.glob("trial_*.yaml")):
        raw = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        card = AlphaCard.model_validate(raw)
        if card.status != AlphaCardStatus.ACCEPTED:
            continue
        result, diff_series = _validate_one_card(
            card,
            sealed_df,
            git=git,
            plan=plan,
            timeout_sec=timeout_sec,
            rng_seed=rng_seed,
        )
        card_paths.append(card_path)
        cards.append(card)
        results.append(result)
        diff_series_list.append(diff_series)

    # Pass 2: apply Benjamini-Hochberg across the family of cards whose
    # CI computation succeeded. Errored cards are excluded from the BH
    # family (they have no valid p-value to contribute); they will
    # carry `bh_validated = False` and sealed_summary = None as before.
    valid_indices = [i for i, r in enumerate(results) if r.error is None]
    bh_reject_for: dict[int, bool] = {}
    if valid_indices:
        family_p = [results[i].p_value_one_sided for i in valid_indices]
        bh = benjamini_hochberg(family_p, q=bh_q)
        for k, idx in enumerate(valid_indices):
            bh_reject_for[idx] = bool(bh.reject[k])

    # Pass 2b: family-wise Hansen 2005 SPA on the same family of valid
    # cards. Requires identical-length diff series to stack into a matrix;
    # we trim every column to the shortest valid series so the stack is
    # rectangular. Skipped when the family is empty.
    spa_result: SPAResult | None = None
    valid_series = [diff_series_list[i] for i in valid_indices if diff_series_list[i] is not None]
    if valid_series:
        min_len = min(len(s) for s in valid_series)
        if min_len >= plan.per_card_validation.block_len * 5:
            diff_matrix = np.column_stack([s[:min_len] for s in valid_series])
            try:
                spa_result = hansen_spa(
                    diff_matrix,
                    block_mean_len=plan.per_card_validation.block_len,
                    n_bootstrap=plan.per_card_validation.n_bootstrap,
                    rng_seed=rng_seed,
                )
            except ValueError as e:
                logger.warning("SPA computation skipped: %s", e)
                spa_result = None

    # Pass 2c: Bailey-LdP Deflated Sharpe Ratio per card. Needs the
    # trial-search context (total attempted trials + cross-sectional
    # variance of trial Sharpes). When not passed explicitly, default
    # to reading `memory/trials.jsonl` from the repo. If neither path
    # yields useful context, DSR is skipped — per_card YAML simply
    # omits the DSR fields.
    n_trials_for_dsr, var_trials_for_dsr = _resolve_dsr_context(
        repo_root,
        n_trials_attempted=n_trials_attempted,
        trial_sharpes_annualised=trial_sharpes_annualised,
    )
    dsr_for: dict[int, "DSRResult | None"] = {}
    if n_trials_for_dsr is not None and var_trials_for_dsr is not None:
        for i in valid_indices:
            series = diff_series_list[i]
            if series is None or len(series) < 2:
                dsr_for[i] = None
                continue
            try:
                dsr_for[i] = deflated_sharpe_ratio_from_returns(
                    series,
                    annualised_sharpe=results[i].incremental_sharpe_point,
                    n_trials=n_trials_for_dsr,
                    variance_of_trial_sharpes_annualised=var_trials_for_dsr,
                    pass_threshold=dsr_threshold,
                )
            except (ValueError, ZeroDivisionError) as e:
                logger.warning("DSR computation skipped for card %d: %s", i, e)
                dsr_for[i] = None

    # Pass 3: rewrite the results with bh_validated / DSR set, then
    # persist the per-card YAML files. We rebuild the dataclasses
    # rather than mutate because they are frozen.
    final_results: list[PerCardValidationResult] = []
    for i, (card_path, card, result) in enumerate(zip(card_paths, cards, results, strict=True)):
        if result.error is None:
            dsr_obj = dsr_for.get(i)
            patched = PerCardValidationResult(
                alpha_id=result.alpha_id,
                source_trial=result.source_trial,
                incremental_sharpe_point=result.incremental_sharpe_point,
                incremental_sharpe_ci_lower=result.incremental_sharpe_ci_lower,
                incremental_sharpe_ci_upper=result.incremental_sharpe_ci_upper,
                n_bootstrap=result.n_bootstrap,
                block_len=result.block_len,
                alpha=result.alpha,
                p_value_one_sided=result.p_value_one_sided,
                bh_validated=bh_reject_for.get(i, False),
                bh_q=bh_q,
                deflated_sharpe_ratio=(
                    dsr_obj.deflated_sharpe_ratio if dsr_obj is not None else -1.0
                ),
                dsr_passes=(dsr_obj.passes if dsr_obj is not None else False),
                dsr_threshold=(dsr_obj.pass_threshold if dsr_obj is not None else dsr_threshold),
            )
            summary: dict[str, float | bool] = {
                "incremental_sharpe_point": patched.incremental_sharpe_point,
                "incremental_sharpe_ci_lower": patched.incremental_sharpe_ci_lower,
                "incremental_sharpe_ci_upper": patched.incremental_sharpe_ci_upper,
                "n_bootstrap": float(patched.n_bootstrap),
                "block_len": float(patched.block_len),
                "alpha": patched.alpha,
                "p_value_one_sided": patched.p_value_one_sided,
                "bh_validated": patched.bh_validated,
                "bh_q": patched.bh_q,
            }
            if patched.deflated_sharpe_ratio >= 0.0:
                summary["deflated_sharpe_ratio"] = patched.deflated_sharpe_ratio
                summary["dsr_passes"] = patched.dsr_passes
                summary["dsr_threshold"] = patched.dsr_threshold
            card.evidence.sealed_summary = summary
        else:
            patched = result
            card.evidence.sealed_summary = None
        card_path.write_text(
            yaml.safe_dump(
                card.model_dump(mode="json"),
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        final_results.append(patched)
    return PerCardValidationOutcome(per_card=final_results, spa=spa_result)


def _validate_one_card(
    card: AlphaCard,
    sealed_df: pd.DataFrame,
    *,
    git: GitManager,
    plan: AnalysisPlan,
    timeout_sec: float,
    rng_seed: int,
) -> tuple[PerCardValidationResult, "np.ndarray | None"]:
    """Compute the per-card sealed metrics and return the daily diff series.

    Returns `(result, diff_series)` where `diff_series` is the daily
    `with - without` paired return series used by the family-wise SPA
    test downstream. `diff_series` is `None` for errored cards so the
    SPA family-builder can skip them cleanly.
    """
    boot_cfg = plan.per_card_validation
    try:
        trial_sha = _find_trial_commit(git, card.source_trial)
    except ValueError as e:
        return _error_result(card, plan, str(e)), None

    try:
        with tempfile.TemporaryDirectory(prefix=f"lbg_card_{card.source_trial}_") as td:
            tmpdir = Path(td)
            parent_dir = tmpdir / "parent"
            trial_dir = tmpdir / "trial"
            parent_strategy = _materialize_at_ref(git, card.source_commit, parent_dir)
            trial_strategy = _materialize_at_ref(git, trial_sha, trial_dir)

            r_with = _sealed_returns(
                trial_strategy, sealed_df, trial_dir / "indicators", timeout_sec
            )
            r_without = _sealed_returns(
                parent_strategy, sealed_df, parent_dir / "indicators", timeout_sec
            )
    except (FileNotFoundError, GitCommandError, ValueError, OSError) as e:
        return _error_result(card, plan, f"materialize_or_backtest_failed: {e}"), None

    n = min(len(r_with), len(r_without))
    if n < boot_cfg.block_len * 5:
        return (
            _error_result(
                card,
                plan,
                f"insufficient_sealed_bars: have {n}, need {boot_cfg.block_len * 5}",
            ),
            None,
        )

    r_with_arr = r_with.to_numpy()[:n]
    r_without_arr = r_without.to_numpy()[:n]
    diff_series = r_with_arr - r_without_arr

    boot = moving_block_bootstrap_sharpe_diff(
        r_with_arr,
        r_without_arr,
        block_len=boot_cfg.block_len,
        n_bootstrap=boot_cfg.n_bootstrap,
        alpha=1.0 - boot_cfg.ci,
        rng_seed=rng_seed,
    )
    result = PerCardValidationResult(
        alpha_id=card.alpha_id,
        source_trial=card.source_trial,
        incremental_sharpe_point=float(boot["point_estimate"]),
        incremental_sharpe_ci_lower=float(boot["ci_lower"]),
        incremental_sharpe_ci_upper=float(boot["ci_upper"]),
        n_bootstrap=int(boot["n_bootstrap"]),
        block_len=int(boot["block_len"]),
        alpha=float(boot["alpha"]),
        p_value_one_sided=float(boot["p_value_one_sided"]),
        # `bh_validated` is filled in by the family-level pass in
        # `compute_per_card_sealed_validation` once every card's p-value
        # is known. The per-card path here cannot know the family.
        bh_validated=False,
    )
    return result, diff_series


def _error_result(
    card: AlphaCard,
    plan: AnalysisPlan,
    message: str,
) -> PerCardValidationResult:
    boot_cfg = plan.per_card_validation
    return PerCardValidationResult(
        alpha_id=card.alpha_id,
        source_trial=card.source_trial,
        incremental_sharpe_point=0.0,
        incremental_sharpe_ci_lower=0.0,
        incremental_sharpe_ci_upper=0.0,
        n_bootstrap=int(boot_cfg.n_bootstrap),
        block_len=int(boot_cfg.block_len),
        alpha=float(1.0 - boot_cfg.ci),
        error=message,
    )


def _resolve_dsr_context(
    repo_root: Path,
    *,
    n_trials_attempted: int | None,
    trial_sharpes_annualised: list[float] | None,
) -> tuple[int | None, float | None]:
    """Decide (N, V_annualised) for Bailey-LdP DSR.

    Order of preference:
      1. Caller-supplied kwargs (both must be present and consistent).
      2. `memory/trials.jsonl` in `repo_root`: count its non-empty
         lines for N, and compute sample variance of
         `train_metrics.sharpe` for V.

    Returns `(None, None)` when neither path yields ≥ 2 trial Sharpes
    (DSR is undefined). Callers must skip DSR on that signal.
    """
    if n_trials_attempted is not None and trial_sharpes_annualised is not None:
        if len(trial_sharpes_annualised) >= 2:
            var = float(np.var(trial_sharpes_annualised, ddof=1))
            return int(n_trials_attempted), var
        return None, None

    trials_path = repo_root / "memory" / "trials.jsonl"
    if not trials_path.exists():
        return None, None

    sharpes: list[float] = []
    n = 0
    import json as _json  # local import: only needed on this path

    with trials_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            n += 1
            tm = obj.get("train_metrics") or {}
            s = tm.get("sharpe")
            if isinstance(s, (int, float)) and not _isnan(s):
                sharpes.append(float(s))
    if len(sharpes) < 2:
        return None, None
    return n, float(np.var(sharpes, ddof=1))


def _isnan(x: float) -> bool:
    return x != x  # cheap NaN check without importing math


def _find_trial_commit(git: GitManager, trial_id: int) -> str:
    """Locate the trial commit by `trial NNNN:` subject prefix.

    Uses `--all` so commits on inactive curator branches still resolve.
    Returns the most recent matching SHA when multiple campaigns share a
    trial id; matching by the trial id and the alpha-card path on disk is
    what binds the result to one specific card.
    """
    prefix = f"trial {trial_id:04d}:"
    raw = git._git("log", "--all", "--pretty=format:%H\t%s")
    for line in raw.splitlines():
        if "\t" not in line:
            continue
        sha, subject = line.split("\t", 1)
        if subject.lstrip().startswith(prefix):
            return sha.strip()
    raise ValueError(f"trial commit for trial_id={trial_id} not found in git log")


def _materialize_at_ref(git: GitManager, ref: str, dest: Path):
    """Write strategy.yaml + every referenced indicator into `dest` as of `ref`.

    Returns the parsed `Strategy`. Only files actually referenced by the
    strategy are materialized — keeps the tmpdir minimal and avoids
    pulling unrelated indicator code that may not exist at `ref`.
    """
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "indicators").mkdir(parents=True, exist_ok=True)
    strategy_text = git.read_file_at_ref(ref, "strategy.yaml")
    strategy_path = dest / "strategy.yaml"
    strategy_path.write_text(strategy_text, encoding="utf-8")
    strategy = load_strategy(strategy_path)
    seen: set[str] = set()
    for spec in strategy.indicators:
        if spec.fn in seen:
            continue
        seen.add(spec.fn)
        source = git.read_file_at_ref(ref, f"indicators/{spec.fn}.py")
        (dest / "indicators" / f"{spec.fn}.py").write_text(source, encoding="utf-8")
    return strategy


def _sealed_returns(
    strategy,
    sealed_df: pd.DataFrame,
    indicators_dir: Path,
    timeout_sec: float,
) -> pd.Series:
    positions = compute_positions(
        strategy, sealed_df, indicators_dir=indicators_dir, timeout_sec=timeout_sec
    )
    return run_backtest(positions, sealed_df).returns


def iter_accepted_cards(cards_dir: Path) -> Iterable[AlphaCard]:
    """Helper for ad-hoc tooling. Reads cards under `cards_dir` in path order."""
    for p in sorted(cards_dir.glob("trial_*.yaml")):
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        card = AlphaCard.model_validate(raw)
        if card.status == AlphaCardStatus.ACCEPTED:
            yield card


__all__ = [
    "PerCardValidationResult",
    "compute_per_card_sealed_validation",
    "iter_accepted_cards",
]
