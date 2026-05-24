"""Dynamic prefix-stability invariant (PROPOSAL.html §4.5, §6).

Formal statement: for any indicator f, every t, and any extension of the
series beyond t, we require

    f(D_{1:T})[t] == f(D_{1:t})[t]

Operationalization: for randomly sampled t, replace `df.iloc[t+1:]` with a
randomized OHLCV block (scale rows by `U(0.5, 1.5)`) and re-evaluate f.
If the head `f.iloc[: t+1]` ever changes, the indicator leaks information
from the future and is not safe.

This check is strictly stronger than the AST checks: it catches
whole-series normalization (`(x - x.mean()) / x.std()`), forward-looking
percentiles, rolling windows with leakage, and any other channel through
which information from D_{>t} can enter f(D)[t].
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from lbg.invariants.violations import InvariantViolation

IndicatorFn = Callable[..., pd.Series]


def _perturb_after(df: pd.DataFrame, t: int, rng: np.random.Generator) -> pd.DataFrame:
    """Return a copy of `df` whose rows after index `t` are randomly scaled.

    Rows `[0, t]` are byte-for-byte identical to `df`. OHLC columns are scaled
    by a per-row factor in `U(0.5, 1.5)` (the same factor across O/H/L/C so
    intra-bar ordering high >= close >= low remains intact). Volume is scaled
    independently and cast back to int64.
    """
    n = len(df)
    if t + 1 >= n:
        return df.copy()
    n_rest = n - (t + 1)
    price_factor = rng.uniform(0.5, 1.5, size=n_rest)
    volume_factor = rng.uniform(0.5, 1.5, size=n_rest)

    head = df.iloc[: t + 1]
    tail = df.iloc[t + 1 :].copy()
    for col in ("open", "high", "low", "close"):
        tail[col] = tail[col].values * price_factor
    tail["volume"] = (tail["volume"].values * volume_factor).astype("int64")
    out = pd.concat([head, tail], axis=0)
    return out.reset_index(drop=True)


def _values_disagree(
    baseline: pd.Series,
    perturbed: pd.Series,
    *,
    rtol: float,
    atol: float,
) -> tuple[bool, int | None]:
    """Return (disagrees, first_diff_idx). NaNs only match other NaNs."""
    if len(baseline) != len(perturbed):
        return True, 0
    b_nan = baseline.isna().to_numpy()
    p_nan = perturbed.isna().to_numpy()
    nan_pattern_changed = bool((b_nan != p_nan).any())
    if nan_pattern_changed:
        diff_idx = int(np.where(b_nan != p_nan)[0][0])
        return True, diff_idx
    valid_mask = ~b_nan
    if not valid_mask.any():
        return False, None
    b_vals = baseline.to_numpy()[valid_mask]
    p_vals = perturbed.to_numpy()[valid_mask]
    close = np.isclose(b_vals, p_vals, rtol=rtol, atol=atol)
    if not close.all():
        first_local = int(np.where(~close)[0][0])
        first_global = int(np.where(valid_mask)[0][first_local])
        return True, first_global
    return False, None


def check_prefix_stability(
    indicator_fn: IndicatorFn,
    df: pd.DataFrame,
    *,
    params: dict[str, Any] | None = None,
    n_samples: int = 30,
    n_perturbations_per_t: int = 3,
    rng_seed: int = 42,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    source_path: str = "<indicator>",
) -> list[InvariantViolation]:
    """Verify `indicator_fn` is prefix-stable on `df`.

    Returns a list of violations (empty list means stable). At most one
    violation is reported per sampled t to keep failures readable.
    """
    if params is None:
        params = {}
    n = len(df)
    if n < 50:
        raise ValueError(f"prefix-stability needs >= 50 bars, got {n}")

    rng = np.random.default_rng(rng_seed)

    baseline = indicator_fn(df, **params)
    if not isinstance(baseline, pd.Series):
        raise TypeError(f"indicator must return pandas.Series, got {type(baseline).__name__}")
    if len(baseline) != n:
        raise ValueError(f"indicator output length {len(baseline)} != input length {n}")

    # Avoid the leading NaN region of trailing-window indicators (period - 1
    # rows). 19 is a generous lower bound for typical defaults.
    earliest_t = 20
    latest_t = n - 2  # need at least one row after t to perturb
    if latest_t <= earliest_t:
        raise ValueError(f"not enough room to sample t in [{earliest_t}, {latest_t}]")
    candidates = np.arange(earliest_t, latest_t + 1)
    k = min(n_samples, len(candidates))
    sampled_ts = rng.choice(candidates, size=k, replace=False)

    violations: list[InvariantViolation] = []
    for t in sampled_ts:
        t_int = int(t)
        for _ in range(n_perturbations_per_t):
            perturbed_df = _perturb_after(df, t_int, rng)
            try:
                perturbed_out = indicator_fn(perturbed_df, **params)
            except Exception as e:  # noqa: BLE001
                violations.append(
                    InvariantViolation(
                        name="prefix_stability",
                        message=(
                            f"indicator raised {type(e).__name__} under "
                            f"perturbation of D_>{t_int}: {e!s}"
                        ),
                        file=source_path,
                        line=0,
                    )
                )
                break
            head_baseline = baseline.iloc[: t_int + 1]
            head_perturbed = perturbed_out.iloc[: t_int + 1]
            disagrees, diff_idx = _values_disagree(
                head_baseline, head_perturbed, rtol=rtol, atol=atol
            )
            if disagrees:
                violations.append(
                    InvariantViolation(
                        name="prefix_stability",
                        message=(
                            f"head [0..{t_int}] changed after perturbing D_>{t_int} "
                            f"(first differing index: {diff_idx})"
                        ),
                        file=source_path,
                        line=0,
                    )
                )
                break
    return violations
