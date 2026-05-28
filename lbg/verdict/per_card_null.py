"""B3 · per-card empirical null distribution (Significance Hardening).

Step 4's LLM-driven null campaign answers a *pipeline-level* question
(how often does the full Editor → gate → seal loop produce a false
positive on null data) but yields too few `add_indicator` candidates to
build a usable per-card null distribution — the Editor rarely proposes
new factors on bootstrapped noise.

B3 sidesteps the LLM entirely. For each library card it asks the
narrower, purely-computational question:

    "If SPY were null (stationary-bootstrapped), what is the distribution
     of THIS card's sealed pathwise ΔSharpe CI lower bound?"

against which the card's observed real-data CI lower bound can be placed
by empirical percentile.

Definitional note
------------------
B3 uses a **single-factor-vs-baseline** increment, not the
pathwise-along-trajectory increment used in the live v26 campaign:

  · "without" = the bare baseline strategy (sma_cross)
  · "with"    = baseline + this one card's indicator wired in per its
                attach_config

The v26 trial commits (which defined the original pathwise parent) lived
in throwaway /tmp campaign repos and are gone, so the trajectory
definition cannot be replayed. The single-factor definition is recomputed
identically for the observed and null arms, so they stay comparable. It
answers "this factor's marginal contribution to the bare baseline" — a
cleaner, more interpretable quantity than "its contribution given whatever
else was accepted before it".

What B3 does NOT do (state in any writeup):
  · It does not measure pipeline-level FDR.
  · It does not validate SPA / DSR as standalone procedures.
  · It does not address the Stage-2 paper-trading OOS question.
  · With only 11 cards the family-wise correction has little power.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import run_backtest
from lbg.alpha_cards import AlphaCard
from lbg.dsl.schema import (
    IndicatorAboveFilter,
    IndicatorBelowFilter,
    IndicatorSpec,
    Strategy,
)
from lbg.verdict.bootstrap import moving_block_bootstrap_sharpe_diff
from lbg.verdict.synthetic_null import stationary_block_indices
from policy_interpreter import compute_positions


@dataclass(frozen=True)
class CardCIResult:
    """One (card, dataset) sealed pathwise CI under the B3 definition."""

    alpha_id: str
    point: float
    ci_lower: float
    ci_upper: float
    p_value_one_sided: float
    error: str | None = None


def inject_card_into_strategy(baseline: Strategy, card: AlphaCard) -> Strategy:
    """Return a copy of `baseline` with `card`'s indicator + filter added.

    Single-card analogue of CampaignRunner._inject_library_into_baseline.
    Raises ValueError if the card has no attach_config or an unknown rule
    — B3 cannot guess the wiring.
    """
    attach = card.attach_config
    if attach is None:
        raise ValueError(f"card {card.alpha_id} has no attach_config; cannot inject")

    indicator = IndicatorSpec(
        name=card.signal.indicator,
        fn=card.signal.fn,
        params=dict(card.signal.params),
    )
    filter_kwargs = {
        "rule": attach.rule,
        "indicator": card.signal.indicator,
        "threshold": attach.threshold,
    }
    if attach.rearm_threshold is not None:
        filter_kwargs["rearm_threshold"] = attach.rearm_threshold
    if attach.rule == "indicator_above":
        flt = IndicatorAboveFilter(**filter_kwargs)
    elif attach.rule == "indicator_below":
        flt = IndicatorBelowFilter(**filter_kwargs)
    else:
        raise ValueError(f"card {card.alpha_id} has unknown attach rule {attach.rule!r}")

    entry_filters = list(baseline.filters)
    exit_filters = list(baseline.exit_filters)
    if attach.target == "exit":
        exit_filters.append(flt)
    else:
        entry_filters.append(flt)

    return baseline.model_copy(
        update={
            "indicators": list(baseline.indicators) + [indicator],
            "filters": entry_filters,
            "exit_filters": exit_filters,
        }
    )


def stage_indicators_dir(
    baseline_indicators_dir: Path,
    library_indicators_dir: Path,
    card_fn: str,
    dest: Path,
) -> Path:
    """Stage a dir with the baseline indicators + the one card's .py.

    Returns `dest`. Idempotent: clears `dest` first.
    """
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(baseline_indicators_dir, dest)
    src = library_indicators_dir / f"{card_fn}.py"
    if not src.exists():
        raise FileNotFoundError(f"library indicator source missing: {src}")
    (dest / f"{card_fn}.py").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def bootstrap_dataframe(
    df: pd.DataFrame,
    *,
    rng_seed: int,
    mean_block_len: int = 10,
) -> pd.DataFrame:
    """Stationary-bootstrap the rows of `df` (in memory, no parquet write).

    Resamples whole rows so within-row OHLC coherence is preserved; the
    integer index is reset. Same mechanic as
    `synthetic_null.bootstrap_spy_parquet` but for an already-loaded
    split DataFrame.
    """
    n = len(df)
    rng = np.random.default_rng(rng_seed)
    idx = stationary_block_indices(n, mean_block_len=mean_block_len, rng=rng)
    return df.iloc[idx].reset_index(drop=True)


def card_pathwise_ci(
    with_strategy: Strategy,
    without_strategy: Strategy,
    *,
    with_indicators_dir: Path,
    without_indicators_dir: Path,
    df: pd.DataFrame,
    block_len: int = 10,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng_seed: int = 42,
    timeout_sec: float = 30.0,
) -> dict:
    """Single-factor-vs-baseline sealed pathwise CI on `df`.

    Backtests both strategies on `df`, takes the paired daily-return
    difference, and bootstraps the Sharpe difference. Returns the dict
    from `moving_block_bootstrap_sharpe_diff` (point/ci/p). Raises on
    backtest failure — the caller decides whether to record an error.
    """
    pos_with = compute_positions(
        with_strategy, df, indicators_dir=with_indicators_dir, timeout_sec=timeout_sec
    )
    pos_without = compute_positions(
        without_strategy, df, indicators_dir=without_indicators_dir, timeout_sec=timeout_sec
    )
    r_with = run_backtest(pos_with, df).returns.to_numpy()
    r_without = run_backtest(pos_without, df).returns.to_numpy()
    n = min(len(r_with), len(r_without))
    if n < block_len * 5:
        raise ValueError(f"insufficient bars: have {n}, need {block_len * 5}")
    return moving_block_bootstrap_sharpe_diff(
        r_with[:n],
        r_without[:n],
        block_len=block_len,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        rng_seed=rng_seed,
    )
