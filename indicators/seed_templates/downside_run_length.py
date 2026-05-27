"""Current run length of consecutive down bars (close < close.shift(1)).
Resets to 0 on any up bar. Captures "we've been bleeding for N days in
a row" as a regime signal. Prefix-stable: stateful but uses only past.
"""

import numpy as np
import pandas as pd


def downside_run_length(df, **_):
    diff = df["close"].diff()
    is_down = (diff < 0.0).astype(int)
    vals = is_down.to_numpy()
    n = len(vals)
    out = np.zeros(n, dtype=float)
    run = 0
    for i in range(n):
        if vals[i] == 1:
            run += 1
        else:
            run = 0
        out[i] = float(run)
    return pd.Series(out, index=df.index).rename("downside_run_length")
