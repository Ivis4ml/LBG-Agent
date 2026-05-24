---
name: new-indicator
description: Scaffold a new pure-function indicator under indicators/ following the LBG-Trader proposal's required template (PROPOSAL.html §4.5). Use when adding a new indicator the Editor agent (or a developer) wants in the indicator catalog. Pass the indicator name as $ARGUMENTS (snake_case, no .py).
disable-model-invocation: true
---

# New indicator scaffold

You are scaffolding a new indicator file for the LBG-Agent project. The indicator must satisfy the proposal's constraints exactly — these are enforced at runtime by `prefix_stability` and the AST invariants, so violations will be rejected by the Orchestrator.

## Inputs

- `$ARGUMENTS` — the indicator name in `snake_case` (no extension, no path). Example: `ema_slope_20`, `donchian_breakout`.

If `$ARGUMENTS` is empty, ask the user for an indicator name before doing anything else.

## What to do

1. Validate the name is `snake_case`, ASCII, and not a Python keyword. If it isn't, ask the user to rename.
2. Check that `indicators/` exists at the repo root. If not, create it.
3. Check that `indicators/<name>.py` does not already exist. If it does, report and stop.
4. Write `indicators/<name>.py` using the template below, substituting `<name>` consistently.
5. Print a short confirmation including: file path, the function signature, and a one-line reminder that the user must implement the body and that prefix-stability will be checked at runtime.

Do **not** also register the indicator in `strategy.yaml` — wiring is done by the Editor agent through an `add_indicator` edit, not by this skill.

## Template

```python
"""<name>: <one-line purpose>.

Pure-function indicator for the LBG-Trader catalog. Authored under the
constraints of PROPOSAL.html §4.5 and §6.

Constraints (enforced by invariants):
  - Pure function of the input DataFrame. No globals, no I/O, no side effects.
  - Only numpy, pandas, and math may be imported.
  - Prefix-stable: for every t, the value at t must depend only on rows
    [0, t]. No .shift(-N), no negative-step iloc, no forward-fill of
    future values, no whole-series normalization.
  - Returns a pandas.Series aligned to the input index.

Parameters
----------
df : pandas.DataFrame
    OHLCV bars with columns: open, high, low, close, volume.
    Index is a monotonically increasing trading-bar index.

Returns
-------
pandas.Series
    Indicator value per bar, indexed identically to df.

Notes
-----
Add a brief description of the economic intuition or signal the
indicator is meant to capture. The Reflector reads this docstring
when explaining outcomes; keep it factual.
"""

import numpy as np
import pandas as pd


def <name>(df: pd.DataFrame) -> pd.Series:
    # TODO: implement. Remember: prefix-stable, no future leakage.
    raise NotImplementedError("<name> is not implemented yet")
```

## Reminders for after scaffolding

- The body must be implemented before any backtest will accept the indicator.
- Verify prefix stability informally with: pick a t, mutate `df.iloc[t+1:]` arbitrarily, recompute, and assert values at `[0, t]` are unchanged. The Orchestrator's `InvariantRunner` will do this rigorously.
- Rolling windows are fine (`df['close'].rolling(N).mean()`), but `min_periods` should be set so early rows are NaN, not back-filled from future data.
- If a normalization step is needed, normalize using an **expanding** window (`expanding().mean()`), never a whole-series mean.
