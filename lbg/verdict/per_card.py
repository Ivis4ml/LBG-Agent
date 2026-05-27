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

import pandas as pd
import yaml

from backtest import run_backtest
from lbg.alpha_cards import AlphaCard, AlphaCardStatus
from lbg.dsl import load_strategy
from lbg.git_manager import GitCommandError, GitManager
from lbg.verdict.analysis_plan import DEFAULT_ANALYSIS_PLAN, AnalysisPlan
from lbg.verdict.bootstrap import moving_block_bootstrap_sharpe_diff
from policy_interpreter import compute_positions

logger = logging.getLogger(__name__)


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
    error: str | None = None

    def to_summary(self) -> dict[str, float | str | int]:
        if self.error is not None:
            return {"validation_error": self.error}
        return {
            "incremental_sharpe_point": self.incremental_sharpe_point,
            "incremental_sharpe_ci_lower": self.incremental_sharpe_ci_lower,
            "incremental_sharpe_ci_upper": self.incremental_sharpe_ci_upper,
            "n_bootstrap": self.n_bootstrap,
            "block_len": self.block_len,
            "alpha": self.alpha,
        }


def compute_per_card_sealed_validation(
    repo_root: str | Path,
    sealed_df: pd.DataFrame,
    *,
    plan: AnalysisPlan = DEFAULT_ANALYSIS_PLAN,
    timeout_sec: float = 30.0,
    rng_seed: int = 42,
    git: GitManager | None = None,
) -> list[PerCardValidationResult]:
    """Populate `evidence.sealed_summary` on every accepted card in `alpha_cards/`.

    Mutates the per-card YAML files in place. Returns one
    `PerCardValidationResult` per accepted card found, in card-path order.
    Cards whose evaluation cannot be done (missing trial commit, missing
    indicator source at ref, bootstrap size error) are tagged with a
    `validation_error` in `sealed_summary` rather than dropped — the alpha
    card lineage stays intact for audit.
    """
    repo_root = Path(repo_root).resolve()
    cards_dir = repo_root / "alpha_cards"
    if not cards_dir.exists():
        return []
    git = git or GitManager(repo_root)

    results: list[PerCardValidationResult] = []
    for card_path in sorted(cards_dir.glob("trial_*.yaml")):
        raw = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        card = AlphaCard.model_validate(raw)
        if card.status != AlphaCardStatus.ACCEPTED:
            continue
        result = _validate_one_card(
            card,
            sealed_df,
            git=git,
            plan=plan,
            timeout_sec=timeout_sec,
            rng_seed=rng_seed,
        )
        # Keep sealed_summary schema = `dict[str, float] | None`. Errors are
        # an honest non-validation -- leave the field None so the H1 reader
        # treats this card as not-yet-validated, and route the error string
        # through the discovery payload via `result.to_summary()`.
        if result.error is None:
            card.evidence.sealed_summary = {
                "incremental_sharpe_point": result.incremental_sharpe_point,
                "incremental_sharpe_ci_lower": result.incremental_sharpe_ci_lower,
                "incremental_sharpe_ci_upper": result.incremental_sharpe_ci_upper,
                "n_bootstrap": float(result.n_bootstrap),
                "block_len": float(result.block_len),
                "alpha": result.alpha,
            }
        else:
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
        results.append(result)
    return results


def _validate_one_card(
    card: AlphaCard,
    sealed_df: pd.DataFrame,
    *,
    git: GitManager,
    plan: AnalysisPlan,
    timeout_sec: float,
    rng_seed: int,
) -> PerCardValidationResult:
    boot_cfg = plan.per_card_validation
    try:
        trial_sha = _find_trial_commit(git, card.source_trial)
    except ValueError as e:
        return _error_result(card, plan, str(e))

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
        return _error_result(card, plan, f"materialize_or_backtest_failed: {e}")

    n = min(len(r_with), len(r_without))
    if n < boot_cfg.block_len * 5:
        return _error_result(
            card,
            plan,
            f"insufficient_sealed_bars: have {n}, need {boot_cfg.block_len * 5}",
        )

    boot = moving_block_bootstrap_sharpe_diff(
        r_with.to_numpy()[:n],
        r_without.to_numpy()[:n],
        block_len=boot_cfg.block_len,
        n_bootstrap=boot_cfg.n_bootstrap,
        alpha=1.0 - boot_cfg.ci,
        rng_seed=rng_seed,
    )
    return PerCardValidationResult(
        alpha_id=card.alpha_id,
        source_trial=card.source_trial,
        incremental_sharpe_point=float(boot["point_estimate"]),
        incremental_sharpe_ci_lower=float(boot["ci_lower"]),
        incremental_sharpe_ci_upper=float(boot["ci_upper"]),
        n_bootstrap=int(boot["n_bootstrap"]),
        block_len=int(boot["block_len"]),
        alpha=float(boot["alpha"]),
    )


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
