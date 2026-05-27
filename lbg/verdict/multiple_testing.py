"""Multiple-testing corrections for per-card sealed validation.

PROPOSAL §20 item 32 locks Benjamini-Hochberg at FDR = 0.10 as the per-card
multiple-testing correction. This module implements the procedure and a
small typed result object so the verdict layer can apply it without
pulling in a third-party statistics dependency.

The BH procedure (Benjamini & Hochberg 1995):

  1. Sort the m p-values ascending: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
  2. Find the largest k such that p_(k) ≤ k * q / m
  3. Reject H_(1), ..., H_(k); accept the rest

The classical guarantee is that FDR ≤ q under independence or positive
regression dependence (PRDS). For dependent test statistics in finance
the Benjamini-Yekutieli variant (an extra log(m) factor in the threshold)
is more conservative; PROPOSAL §20 locks the classical BH form, so this
module follows that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_Q = 0.10
"""PROPOSAL §20 item 32: FDR control level for per-card validation."""


@dataclass(frozen=True)
class BHResult:
    """Per-test outcome of Benjamini-Hochberg correction.

    `reject` carries the rejection decision aligned with the input
    p-value order. `adjusted_threshold` is the p-value cutoff that the
    test would have had to clear given its rank; it is reported so that
    downstream consumers can show a single "BH cutoff" column instead of
    re-deriving it from `(rank, q, m)`.
    """

    p_values: tuple[float, ...]
    q: float
    reject: tuple[bool, ...]
    rank: tuple[int, ...]
    adjusted_threshold: tuple[float, ...]
    n_rejected: int

    def as_summary(self) -> dict:
        return {
            "q": self.q,
            "n_tests": len(self.p_values),
            "n_rejected": self.n_rejected,
            "p_values": list(self.p_values),
            "reject": list(self.reject),
            "rank": list(self.rank),
            "adjusted_threshold": list(self.adjusted_threshold),
        }


def benjamini_hochberg(
    p_values: list[float] | tuple[float, ...], *, q: float = DEFAULT_Q
) -> BHResult:
    """Apply the Benjamini-Hochberg step-up procedure to one-sided p-values.

    Parameters
    ----------
    p_values:
        One-sided p-values, one per test, in the *input* order — the
        caller's card / candidate order is preserved in the returned
        `reject` and `rank` tuples.
    q:
        FDR control level. Defaults to PROPOSAL §20 item 32 = 0.10.

    Notes
    -----
    Empty input returns an empty BHResult. Inputs outside [0, 1] are
    rejected with a `ValueError` rather than silently clamped: an
    out-of-range p-value usually means the caller miscomputed it
    (e.g. forgot the Davison-Hinkley +1/+1 smoothing on a percentile
    bootstrap), and silently clamping would mask the bug.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1); got {q!r}")

    p_arr = np.asarray(p_values, dtype=float)
    if p_arr.size == 0:
        return BHResult(
            p_values=(),
            q=q,
            reject=(),
            rank=(),
            adjusted_threshold=(),
            n_rejected=0,
        )
    if not np.all((p_arr >= 0.0) & (p_arr <= 1.0)):
        bad = p_arr[(p_arr < 0.0) | (p_arr > 1.0)]
        raise ValueError(f"p-values must lie in [0, 1]; got {bad.tolist()!r}")

    m = p_arr.size
    # argsort returns indices that would sort the array; rank[i] is the
    # 1-based position of p_arr[i] in the sorted order.
    order = np.argsort(p_arr, kind="stable")
    rank_zero_based = np.empty(m, dtype=int)
    rank_zero_based[order] = np.arange(m)
    rank_one_based = rank_zero_based + 1

    # BH threshold per test under its rank.
    threshold = rank_one_based * q / m

    # Step-up: find the largest k such that p_(k) ≤ k * q / m, then
    # reject all tests with rank ≤ k.
    sorted_p = p_arr[order]
    sorted_thresh = np.arange(1, m + 1) * q / m
    below = sorted_p <= sorted_thresh
    if not np.any(below):
        cutoff_rank = 0
    else:
        cutoff_rank = int(np.max(np.where(below)[0]) + 1)  # 1-based

    reject = rank_one_based <= cutoff_rank

    return BHResult(
        p_values=tuple(float(p) for p in p_arr),
        q=float(q),
        reject=tuple(bool(r) for r in reject),
        rank=tuple(int(r) for r in rank_one_based),
        adjusted_threshold=tuple(float(t) for t in threshold),
        n_rejected=int(reject.sum()),
    )
