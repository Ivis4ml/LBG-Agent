"""step_in: 0 at bar 0, 1.0 thereafter.

Paired with `zero_baseline` to encode a buy-and-hold strategy under the
existing CrossRule-based entry/exit DSL:

  entry: cross_above(step_in, zero_baseline)
    bar 0: step_in=0, zero_baseline=0 (equal, no cross)
    bar 1: step_in=1 > 0 AND step_in.shift(1)=0 <= 0 → cross_above fires
    bar 2+: step_in=1 > 0 but step_in.shift(1)=1 > 0 → no cross, just stays

  exit: cross_below(step_in, zero_baseline)
    step_in is never < zero_baseline after bar 0 → never fires

Net effect: enter at bar 1, never exit → 100% long the whole window
(modulo the sizing block). Used by the `buyhold` baseline so Discovery
campaigns can start from an always-long position rather than the SMA
cross. From this starting point, `add_indicator` filters have a real
Pareto path: any filter that drops out of risk-off periods reduces
drawdown without necessarily hurting Sharpe.

Constraints (PROPOSAL §4.5 / §6):
  * Pure function. No globals, I/O, side effects.
  * Imports: pandas only.
  * Prefix-stable trivially: the output at index `t` is a function of
    `t` alone, not of any data values.
"""

import pandas as pd


def step_in(df, **params):
    """Return a Series of [0.0, 1.0, 1.0, ..., 1.0] aligned to df.index.

    The step at index 1 is the one-time "enter the position" event when
    paired with `zero_baseline` under a cross_above entry rule.
    """
    out = pd.Series(1.0, index=df.index)
    if len(out) > 0:
        out.iloc[0] = 0.0
    return out
