"""zero_baseline: constant 0.0.

Paired with `step_in` to materialize the one-time entry event of a
buy-and-hold strategy under the existing CrossRule DSL (see
`step_in.py` docstring for the full mechanic). On its own this is just
a constant series used as the "slow" leg of the entry cross.

Constraints (PROPOSAL §4.5 / §6):
  * Pure function. No globals, I/O, side effects.
  * Imports: pandas only.
  * Prefix-stable trivially: returns the constant 0.0 at every index.
"""

import pandas as pd


def zero_baseline(df, **params):
    """Return a Series of zeros aligned to df.index."""
    return pd.Series(0.0, index=df.index)
