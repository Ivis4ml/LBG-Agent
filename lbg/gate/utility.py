"""One-sided lower confidence bound on annualized Sharpe (PROPOSAL.html §9).

Per-trial gate uses α=0.20 — looser than the 95% CI reserved for the final
H1 verdict on the sealed window. The standard error follows Lo (2002)'s
asymptotic form assuming iid daily returns:

    SE(Ŝ) ≈ sqrt((1 + 0.5 * Ŝ²) / n) * sqrt(periods_per_year)

The z-quantile is hard-coded for α=0.20 to avoid a scipy dependency just for
this one number. Other α's are supported by passing `z_quantile` explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR: int = 252

# norm.ppf(0.80) -- the one-sided 80% z-quantile used at α = 0.20.
Z_QUANTILE_80: float = 0.8416212335729143

# Pre-computed norm.ppf(1 - α) for the α values the orchestrator uses.
# Keeps `utility_lcb_sharpe` scipy-free for the common cases; callers can
# still pass an explicit `z_quantile` for arbitrary α.
_Z_QUANTILES: dict[float, float] = {
    0.40: 0.2533471031357997,  # GateConfig.permissive
    0.30: 0.5244005127080407,
    0.20: Z_QUANTILE_80,  # GateConfig default (PROPOSAL §9)
    0.10: 1.2815515655446004,
    0.05: 1.6448536269514722,
}


def utility_lcb_sharpe(
    returns: pd.Series,
    *,
    alpha: float = 0.20,
    z_quantile: float | None = None,
    periods_per_year: int = PERIODS_PER_YEAR,
    min_obs: int = 30,
) -> float:
    """Return the one-sided lower confidence bound on annualized Sharpe.

    The lookup table covers α ∈ {0.05, 0.10, 0.20, 0.30, 0.40}. For other
    α's pass the corresponding `norm.ppf(1 - alpha)` explicitly.
    """
    n = int(len(returns))
    if n < min_obs:
        return 0.0

    std = returns.std()
    if not np.isfinite(std) or std < 1e-12:
        return 0.0

    sharpe = float(returns.mean() / std) * np.sqrt(periods_per_year)
    se = float(np.sqrt((1.0 + 0.5 * sharpe**2) / n) * np.sqrt(periods_per_year))

    if z_quantile is None:
        for known_alpha, q in _Z_QUANTILES.items():
            if np.isclose(alpha, known_alpha):
                z_quantile = q
                break
        if z_quantile is None:
            raise ValueError(
                f"alpha={alpha} not in pre-computed table {sorted(_Z_QUANTILES)}; "
                "pass an explicit `z_quantile=norm.ppf(1-alpha)`"
            )

    return sharpe - z_quantile * se
