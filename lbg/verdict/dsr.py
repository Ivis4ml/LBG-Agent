r"""Deflated Sharpe Ratio (Bailey & López de Prado 2014).

The pathwise CI in PROPOSAL §10.5 step 2 + the BH FDR correction in
step 1 together control per-card false-discovery rate, but neither
accounts for *backtest overfitting*: when a candidate is the best of
\(N\) trials, the highest observed Sharpe will be biased upward simply
because of the selection. The Deflated Sharpe Ratio compresses this
bias into a single probability.

Given an observed Sharpe \(\widehat{SR}\) over \(T\) periods with sample
skewness \(\gamma_3\) and excess kurtosis \(\gamma_4\), and given the
number of trials \(N\) plus the variance of Sharpe estimates across
those trials \(V\), DSR returns

  \(\mathrm{DSR} \;=\; \Phi\!\Bigl(\frac{(\widehat{SR} - SR^{*})\,\sqrt{T-1}}{\sqrt{1 - \gamma_3\,\widehat{SR} + \tfrac{\gamma_4 - 1}{4}\,\widehat{SR}^2}}\Bigr)\)

where the deflated threshold \(SR^{*}\) is Bailey-LdP's analytic
approximation of \(\mathbb{E}[\max\{\widehat{SR}_n\}]\) under the null
that every trial has \(\mathrm{SR} = 0\):

  \(SR^{*} \;=\; \sqrt{V}\,\bigl((1 - \gamma_{em})\,\Phi^{-1}(1 - 1/N) \,+\, \gamma_{em}\,\Phi^{-1}(1 - 1/(N e))\bigr)\)

with \(\gamma_{em} \approx 0.5772\) (Euler-Mascheroni) and \(\Phi^{-1}\)
the standard-normal inverse CDF.

All inputs are in *daily* (per-bar) units so that the variance
estimator \(1 - \gamma_3\,\widehat{SR} + \tfrac{\gamma_4 - 1}{4}\,\widehat{SR}^2\)
is dimensionally consistent. The public helper
`deflated_sharpe_ratio_from_returns` accepts annualised inputs (matching
the rest of the verdict layer) and converts internally.

References
----------
Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
  Journal of Portfolio Management, 40(5), 94-107.
Mertens, E. (2002). "Comments on Variance of the IID Estimator in Lo (2002)."
Lo, A. W. (2002). "The Statistics of Sharpe Ratios."
  Financial Analysts Journal, 58(4), 36-52.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TRADING_DAYS_PER_YEAR = 252
EULER_MASCHERONI = 0.5772156649015329
DEFAULT_DSR_THRESHOLD = 0.95


@dataclass(frozen=True)
class DSRResult:
    """Per-card Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    `deflated_sharpe_ratio` is the headline DSR — a probability in
    [0, 1] that the observed Sharpe genuinely exceeds the
    selection-adjusted threshold. `passes` is `True` when DSR exceeds
    `pass_threshold` (default 0.95, matching the paper).

    `psr_at_zero` is the Probabilistic Sharpe Ratio without selection
    adjustment (SR* = 0); reported so the reader can see how much
    Bailey-LdP's deflation actually moves the verdict for this card.
    """

    sharpe_daily: float
    sharpe_threshold_daily: float
    n_obs: int
    skewness: float
    kurtosis: float
    n_trials: int
    variance_of_trial_sharpes_daily: float
    psr_at_zero: float
    deflated_sharpe_ratio: float
    passes: bool
    pass_threshold: float

    def as_summary(self) -> dict:
        return {
            "sharpe_daily": self.sharpe_daily,
            "sharpe_threshold_daily": self.sharpe_threshold_daily,
            "n_obs": self.n_obs,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "n_trials": self.n_trials,
            "variance_of_trial_sharpes_daily": self.variance_of_trial_sharpes_daily,
            "psr_at_zero": self.psr_at_zero,
            "deflated_sharpe_ratio": self.deflated_sharpe_ratio,
            "passes": self.passes,
            "pass_threshold": self.pass_threshold,
        }


# ---------- standard-normal helpers (no scipy dep) ----------


def norm_cdf(x: float) -> float:
    """Standard-normal CDF via stdlib math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Standard-normal inverse CDF (Acklam 2003 rational approximation).

    Accurate to within ~1.15e-9 relative error for 1e-10 < p < 1 - 1e-10.
    Outside that range we clamp to ±8.21, which is the inverse of
    the float64-machine-precision tail probability.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf requires p in (0, 1); got {p}")

    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        return num / den
    if p <= p_high:
        q = p - 0.5
        r = q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        return num / den
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    return -num / den


# ---------- DSR core ----------


def expected_max_sharpe_daily(
    n_trials: int,
    variance_of_trial_sharpes_daily: float,
) -> float:
    """Bailey-LdP analytic approximation of E[max(SR̂_n)] under H_0: SR = 0.

    All inputs and outputs are in daily (per-bar) units.
    """
    if n_trials <= 1:
        return 0.0
    if variance_of_trial_sharpes_daily <= 0.0:
        return 0.0
    term1 = (1.0 - EULER_MASCHERONI) * norm_ppf(1.0 - 1.0 / n_trials)
    term2 = EULER_MASCHERONI * norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(variance_of_trial_sharpes_daily) * (term1 + term2)


def probabilistic_sharpe_ratio(
    sharpe_daily: float,
    sharpe_threshold_daily: float,
    n_obs: int,
    skewness: float,
    kurtosis: float,
) -> float:
    """Bailey-LdP PSR formula: P(true SR > threshold | observed sample).

    Sharpe-ratio variance under non-normal moments (Mertens 2002 / Lo 2002):

      Var(SR̂) ≈ (1 - γ_3·SR̂ + ((γ_4 - 1)/4)·SR̂²) / (T - 1)

    Returns 0.5 when the variance estimator is degenerate (T ≤ 1 or
    variance term ≤ 0) — the "no information" answer is a Φ(0) = 0.5
    that doesn't crash downstream code.
    """
    if n_obs <= 1:
        return 0.5
    variance_term = 1.0 - skewness * sharpe_daily + ((kurtosis - 1.0) / 4.0) * sharpe_daily**2
    if variance_term <= 0.0:
        return 0.5
    se_sr = math.sqrt(variance_term / (n_obs - 1))
    return norm_cdf((sharpe_daily - sharpe_threshold_daily) / se_sr)


def deflated_sharpe_ratio(
    sharpe_daily: float,
    *,
    n_obs: int,
    skewness: float,
    kurtosis: float,
    n_trials: int,
    variance_of_trial_sharpes_daily: float,
    pass_threshold: float = DEFAULT_DSR_THRESHOLD,
) -> DSRResult:
    """Compute DSR for one observed daily Sharpe."""
    sr_threshold = expected_max_sharpe_daily(n_trials, variance_of_trial_sharpes_daily)
    psr_zero = probabilistic_sharpe_ratio(sharpe_daily, 0.0, n_obs, skewness, kurtosis)
    dsr = probabilistic_sharpe_ratio(sharpe_daily, sr_threshold, n_obs, skewness, kurtosis)
    return DSRResult(
        sharpe_daily=float(sharpe_daily),
        sharpe_threshold_daily=float(sr_threshold),
        n_obs=int(n_obs),
        skewness=float(skewness),
        kurtosis=float(kurtosis),
        n_trials=int(n_trials),
        variance_of_trial_sharpes_daily=float(variance_of_trial_sharpes_daily),
        psr_at_zero=float(psr_zero),
        deflated_sharpe_ratio=float(dsr),
        passes=bool(dsr >= pass_threshold),
        pass_threshold=float(pass_threshold),
    )


def deflated_sharpe_ratio_from_returns(
    diff_series: np.ndarray,
    *,
    annualised_sharpe: float,
    n_trials: int,
    variance_of_trial_sharpes_annualised: float,
    pass_threshold: float = DEFAULT_DSR_THRESHOLD,
) -> DSRResult:
    """Friendlier entry point: accepts annualised inputs and converts.

    Parameters
    ----------
    diff_series:
        Daily pathwise ΔSharpe daily return series (one column). Used to
        estimate skewness and kurtosis of the differential.
    annualised_sharpe:
        The annualised pathwise Sharpe (i.e., `incremental_sharpe_point`
        from PerCardValidationResult). Converted to daily by dividing
        by √252.
    n_trials:
        Total number of attempted Discovery trials this campaign.
    variance_of_trial_sharpes_annualised:
        Variance of attempted-trial annualised Sharpes (e.g., over
        `trials.jsonl`'s `train_metrics.sharpe`). Converted to daily
        variance by dividing by 252.
    pass_threshold:
        DSR threshold above which the card "passes" (default 0.95).
    """
    diff_arr = np.asarray(diff_series, dtype=float)
    n_obs = diff_arr.size
    if n_obs < 2:
        # Degenerate: cannot estimate moments; return a Φ(0) = 0.5 verdict.
        return deflated_sharpe_ratio(
            0.0,
            n_obs=n_obs,
            skewness=0.0,
            kurtosis=3.0,
            n_trials=n_trials,
            variance_of_trial_sharpes_daily=0.0,
            pass_threshold=pass_threshold,
        )

    # Sample skewness and (non-excess) kurtosis on the daily diff series.
    mean = float(diff_arr.mean())
    centred = diff_arr - mean
    var = float((centred**2).mean())
    if var <= 0.0:
        skewness = 0.0
        kurtosis = 3.0
    else:
        std = math.sqrt(var)
        skewness = float((centred**3).mean() / (std**3))
        kurtosis = float((centred**4).mean() / (std**4))  # γ_4, NOT excess

    sharpe_daily = annualised_sharpe / math.sqrt(TRADING_DAYS_PER_YEAR)
    variance_of_trial_sharpes_daily = variance_of_trial_sharpes_annualised / TRADING_DAYS_PER_YEAR

    return deflated_sharpe_ratio(
        sharpe_daily,
        n_obs=n_obs,
        skewness=skewness,
        kurtosis=kurtosis,
        n_trials=n_trials,
        variance_of_trial_sharpes_daily=variance_of_trial_sharpes_daily,
        pass_threshold=pass_threshold,
    )
