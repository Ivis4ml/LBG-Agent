"""Moving-block bootstrap for paired Sharpe difference (PROPOSAL.html §10.5).

The block bootstrap resamples contiguous blocks of length `block_len` from
the paired daily-returns series with replacement, recomputes annualized
Sharpe for each resample, and reports percentile CI of the difference.
Block length 10 is the proposal's default.
"""

from __future__ import annotations

import numpy as np

TRADING_DAYS_PER_YEAR: int = 252


def _annualized_sharpe(returns: np.ndarray) -> float:
    # ddof=1 matches pandas' `Series.std()` default so this aligns with
    # `backtest._sharpe_annualized` bit-for-bit when called on the same
    # underlying realized-returns series.
    std = returns.std(ddof=1) if returns.size > 1 else 0.0
    if not np.isfinite(std) or std < 1e-12:
        return 0.0
    return float(returns.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def moving_block_bootstrap_sharpe_diff(
    r_strategy: np.ndarray | list[float],
    r_baseline: np.ndarray | list[float],
    *,
    block_len: int = 10,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng_seed: int = 42,
) -> dict[str, float | int]:
    """Return point estimate + (1-alpha) CI for `Sharpe(strategy) - Sharpe(baseline)`.

    Both inputs must be the same length: the paired daily realized returns
    aligned bar-by-bar. Block resampling preserves short-horizon
    autocorrelation that an iid bootstrap would erase.
    """
    rs = np.asarray(r_strategy, dtype=float)
    rb = np.asarray(r_baseline, dtype=float)
    if rs.shape != rb.shape:
        raise ValueError(f"shape mismatch: strategy {rs.shape} vs baseline {rb.shape}")
    n = rs.size
    if n < block_len * 5:
        raise ValueError(f"need at least {block_len * 5} bars for block_len={block_len}, got {n}")

    rng = np.random.default_rng(rng_seed)
    n_blocks = (n + block_len - 1) // block_len
    max_start = n - block_len  # inclusive

    diffs = np.empty(n_bootstrap, dtype=float)
    for k in range(n_bootstrap):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
        sharpe_s = _annualized_sharpe(rs[idx])
        sharpe_b = _annualized_sharpe(rb[idx])
        diffs[k] = sharpe_s - sharpe_b

    point_estimate = _annualized_sharpe(rs) - _annualized_sharpe(rb)
    ci_lower = float(np.percentile(diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))

    return {
        "point_estimate": float(point_estimate),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "alpha": float(alpha),
        "n_bootstrap": int(n_bootstrap),
        "block_len": int(block_len),
    }
