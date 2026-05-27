"""Hansen 2005 Superior Predictive Ability (SPA) test.

PROPOSAL §10.5 step 1 measures whether individual cards survive sealed
testing (CI lower bound > 0, BH-corrected). SPA answers the
*family-wise* question instead: across the entire set of candidate
cards, is at least one of them genuinely better than the no-card
baseline, accounting for the dependence structure between candidates?

The test statistic is the maximum studentized Sharpe difference across
the family:

    V_obs = max_m  √T * SR̂_m / ω̂_m

where SR̂_m is the annualised pathwise ΔSharpe of card m vs the no-card
baseline on the sealed window, and ω̂_m is its long-run standard error
(estimated via the stationary bootstrap).

The null distribution V* is built by stationary-bootstrap resampling
the paired ΔSharpe daily series with Hansen's recentering — under the
least favourable null, every card has expected ΔSharpe = 0, but
candidates whose observed t-stat is below a (2 log log T) threshold
are treated as "definitely losing" and not recentered, so they do not
artificially inflate the bootstrap max.

This module computes all three Hansen variants — `l` (lower / White
2000, most conservative), `c` (consensus, the standard recommendation),
`u` (upper, most liberal) — in one pass. The H1 verdict reports SPA_c.

References
----------
Hansen, P. R. (2005). "A Test for Superior Predictive Ability."
  Journal of Business & Economic Statistics, 23(4), 365-380.
Politis, D. N., & Romano, J. P. (1994). "The Stationary Bootstrap."
  Journal of the American Statistical Association, 89(428), 1303-1313.
White, H. (2000). "A Reality Check for Data Snooping."
  Econometrica, 68(5), 1097-1126.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class SPAResult:
    """Hansen 2005 SPA test outcome with all three recentering variants.

    `spa_p_value` is the SPA_c (consensus) p-value, the recommended
    default. `spa_p_value_l` (conservative) and `spa_p_value_u`
    (liberal) bracket it and are reported alongside for transparency.

    `best_candidate_idx` is the *observed* argmax of √T * SR̂_m / ω̂_m,
    not a statistical claim. A SPA-rejected test means the maximum is
    significantly > 0; SPA does not certify which candidate produced it.
    """

    spa_p_value: float
    spa_p_value_l: float
    spa_p_value_u: float
    observed_statistic: float
    best_candidate_idx: int
    n_candidates: int
    n_bootstrap: int
    block_mean_len: int
    sr_observed: tuple[float, ...]
    se_observed: tuple[float, ...]
    studentized_observed: tuple[float, ...]

    def as_summary(self) -> dict:
        return {
            "spa_p_value": self.spa_p_value,
            "spa_p_value_l": self.spa_p_value_l,
            "spa_p_value_u": self.spa_p_value_u,
            "observed_statistic": self.observed_statistic,
            "best_candidate_idx": self.best_candidate_idx,
            "n_candidates": self.n_candidates,
            "n_bootstrap": self.n_bootstrap,
            "block_mean_len": self.block_mean_len,
            "sr_observed": list(self.sr_observed),
            "se_observed": list(self.se_observed),
            "studentized_observed": list(self.studentized_observed),
        }


def stationary_bootstrap_indices(
    T: int,
    *,
    block_mean_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Politis-Romano (1994) stationary bootstrap of indices into [0, T).

    Block lengths are i.i.d. Geometric(p = 1 / block_mean_len), giving an
    expected block length of `block_mean_len`. Block starts are uniform
    on [0, T). When a block runs past T-1 it wraps modulo T, which keeps
    the resampled series stationary in the bootstrap world.

    Returns an int array of length T.
    """
    if T <= 0:
        raise ValueError(f"T must be positive; got {T}")
    if block_mean_len < 1:
        raise ValueError(f"block_mean_len must be >= 1; got {block_mean_len}")
    p = 1.0 / block_mean_len
    out = np.empty(T, dtype=np.int64)
    i = 0
    while i < T:
        start = int(rng.integers(0, T))
        # geometric block length, at least 1
        block_len = int(rng.geometric(p))
        end = min(i + block_len, T)
        for j in range(i, end):
            out[j] = (start + (j - i)) % T
        i = end
    return out


def _annualised_sharpe(returns: np.ndarray) -> float:
    if returns.size <= 1:
        return 0.0
    std = float(returns.std(ddof=1))
    if std < 1e-15:
        return 0.0
    return float(returns.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def hansen_spa(
    diff_matrix: np.ndarray,
    *,
    block_mean_len: int = 10,
    n_bootstrap: int = 1000,
    rng_seed: int = 42,
) -> SPAResult:
    """Run Hansen 2005 SPA on a (T, M) matrix of paired daily ΔSharpe series.

    Each column `m` of `diff_matrix` is the daily return differential of
    card `m` (with-card minus without-card pathwise series) on the sealed
    window. The null tested is

        H_0: max_m  E[annualised Sharpe of column m]  ≤  0,

    i.e. no card has a genuinely positive pathwise contribution.

    Parameters
    ----------
    diff_matrix:
        Shape (T, M). T is the number of sealed bars; M the number of
        cards. NaN or all-zero columns are valid — they will show
        SR̂ = 0 and contribute nothing to the max.
    block_mean_len:
        Expected stationary-bootstrap block length (Politis-Romano
        geometric). Default 10 mirrors the per-card moving-block CI.
    n_bootstrap:
        Number of bootstrap resamples. Default 1000.
    rng_seed:
        Seed for reproducibility.
    """
    diff_matrix = np.asarray(diff_matrix, dtype=float)
    if diff_matrix.ndim != 2:
        raise ValueError(f"diff_matrix must be 2D (T, M); got shape {diff_matrix.shape}")
    T, M = diff_matrix.shape
    if T < block_mean_len * 5:
        raise ValueError(
            f"need at least {block_mean_len * 5} bars for block_mean_len={block_mean_len}, got {T}"
        )
    if M < 1:
        raise ValueError(f"need at least one candidate column; got M={M}")
    if n_bootstrap < 100:
        raise ValueError(f"n_bootstrap must be >= 100 for a credible p-value; got {n_bootstrap}")

    rng = np.random.default_rng(rng_seed)

    # Observed annualised Sharpe per card.
    sr_obs = np.array([_annualised_sharpe(diff_matrix[:, m]) for m in range(M)])

    # First bootstrap pass: build the bootstrap distribution of per-card
    # Sharpe, used both for ω̂_m (standard error) and for the SPA test
    # statistic itself. Storing all M*B Sharpe values is the standard
    # implementation; with M = 11 and B = 1000 that's 11k floats, fine.
    sr_boot = np.empty((n_bootstrap, M), dtype=float)
    for b in range(n_bootstrap):
        idx = stationary_bootstrap_indices(T, block_mean_len=block_mean_len, rng=rng)
        resampled = diff_matrix[idx]
        for m in range(M):
            sr_boot[b, m] = _annualised_sharpe(resampled[:, m])

    # ω̂_m = sample standard deviation of the bootstrap distribution of
    # SR̂_m. This is the long-run SE of the Sharpe statistic that the
    # stationary bootstrap implicitly estimates.
    se_obs = sr_boot.std(axis=0, ddof=1)
    # Floor to a tiny positive number so the studentization is finite
    # for all-zero columns (which legitimately have SE = 0); in that
    # case studentized = 0 too and the column contributes nothing.
    se_floor = np.maximum(se_obs, 1e-12)

    sqrt_T = np.sqrt(T)
    studentized_obs = sqrt_T * sr_obs / se_floor

    # SPA test statistic: max over candidates of the positive part.
    v_obs = float(max(np.max(studentized_obs), 0.0))

    # Hansen 2005 recentering. The three variants share the same
    # bootstrap distribution; they only differ in which candidates get
    # their observed Sharpe subtracted to enforce the null. Let
    # g_m = recentering term subtracted from sr_boot[b, m] to convert
    # the bootstrap draw into a null-world draw.
    #
    # SPA_l (lower / White 2000): g_m = sr_obs[m] for every m. Forces
    #   every candidate's bootstrap distribution to have mean 0.
    # SPA_c (consensus, default): g_m = sr_obs[m] only when the
    #   candidate is "relevant", i.e. its observed Sharpe is not too far
    #   below 0. The threshold is the law-of-iterated-logarithm cutoff
    #   sr_obs[m] >= -se_obs[m] * sqrt(2 log log T / T) * sqrt(T)
    #   = -se_obs[m] * sqrt(2 log log T). Candidates failing this stay
    #   at their observed (negative) Sharpe in the bootstrap and never
    #   reach the max — they are excluded from the family-wise correction.
    # SPA_u (upper / most liberal): g_m = sr_obs[m] only when
    #   sr_obs[m] > 0. Cards with non-positive observed Sharpe are kept
    #   at their observed value in the bootstrap.
    if T < 3:
        # log(log(T)) is undefined for T < 3; treat as no exclusion
        # (degenerates SPA_c into SPA_l).
        log_log_threshold = 0.0
    else:
        log_log_threshold = float(np.sqrt(2.0 * np.log(np.log(T))))

    g_l = sr_obs.copy()
    g_c = np.where(sr_obs >= -se_obs * log_log_threshold, sr_obs, 0.0)
    g_u = np.where(sr_obs > 0.0, sr_obs, 0.0)

    p_l = _spa_p_value(sr_boot, g_l, se_floor, sqrt_T, v_obs)
    p_c = _spa_p_value(sr_boot, g_c, se_floor, sqrt_T, v_obs)
    p_u = _spa_p_value(sr_boot, g_u, se_floor, sqrt_T, v_obs)

    return SPAResult(
        spa_p_value=p_c,
        spa_p_value_l=p_l,
        spa_p_value_u=p_u,
        observed_statistic=v_obs,
        best_candidate_idx=int(np.argmax(sr_obs)),
        n_candidates=M,
        n_bootstrap=n_bootstrap,
        block_mean_len=block_mean_len,
        sr_observed=tuple(float(x) for x in sr_obs),
        se_observed=tuple(float(x) for x in se_obs),
        studentized_observed=tuple(float(x) for x in studentized_obs),
    )


def _spa_p_value(
    sr_boot: np.ndarray,
    g: np.ndarray,
    se: np.ndarray,
    sqrt_T: float,
    v_obs: float,
) -> float:
    """Davison-Hinkley one-sided p-value with the supplied recentering."""
    centred = sr_boot - g[None, :]
    studentized = sqrt_T * centred / se[None, :]
    v_star = np.maximum(np.max(studentized, axis=1), 0.0)
    n = v_star.size
    return float((np.sum(v_star >= v_obs) + 1) / (n + 1))
