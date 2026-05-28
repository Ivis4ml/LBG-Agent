# B3 · Per-card empirical null distribution · 2026-05-27

The experiment that answers the user's recurring question — *"FDR 怎么算？
基于 null distribution？"* — for the 11 v26 library cards, without the LLM.

## Method

For each library card:
- **without** = bare baseline strategy (`sma_cross`)
- **with** = baseline + that card's indicator wired in per its `attach_config`
- compute the **single-factor-vs-baseline** sealed pathwise ΔSharpe CI
  on the real sealed split (observed) and on N=200 stationary-bootstrap
  draws of that split (null).

The independent unit is the bootstrap draw; within a draw all 11 cards
see the same null data, so they are correlated — the resampling
preserves that dependence automatically (this matters for the
family-wise correction).

Definitional note: this is **not** the pathwise-along-trajectory
increment used in the live v26 campaign (the v26 trial commits lived in
throwaway /tmp repos and are gone). The single-factor definition is
recomputed identically for observed and null arms, so they stay
comparable. For `vol_regime_zscore` — the first factor accepted in v26 —
the two definitions coincide exactly (its pathwise parent *was* the bare
baseline), which is why its B3 observed CI reproduces the v26 number
bit-for-bit: point +0.2486, CI [-0.0913, +0.6852]. That coincidence is
the cross-check that the B3 mechanic is correct.

## Headline result (N=200)

| card | obs point | obs CI_lo | empirical p(point) |
|------|-----------|-----------|--------------------|
| vol_regime_zscore (v26 headliner) | +0.249 | -0.091 | **0.240** |
| fractal_efficiency | +0.213 | 0.000 | 0.010 |
| ema_above_sma | +0.193 | -0.389 | 0.045 |
| donchian / macd / roc / aroon / body_to_range / upper_wick | 0.000 | 0.000 | inert on bare baseline |
| kama_ratio | -0.181 | -0.506 | 0.825 |
| velocity_zscore | -0.243 | -0.922 | 0.915 |

`empirical p(point)` = fraction of the 200 null draws whose point ΔSharpe
≥ the card's observed point.

Family-wise (Westfall-Young, draw as the unit):
- **minP** (per-card stat = bootstrap one-sided p; family stat = min):
  observed min-p 0.0619 → adjusted p **0.27**
- **maxT** (per-card stat = point ΔSharpe; family stat = max):
  observed max point +0.249 → adjusted p **0.295**

## Interpretation

1. **vol_regime_zscore**, the v26 headliner, has empirical p = 0.24 — not
   significant. Its point estimate is the largest, but so is its
   variance: 24% of null draws produce a larger point ΔSharpe by chance.
   This is the same lesson as the B2 finding that null can produce point
   estimates up to +0.56.
2. **fractal_efficiency** has the lowest standalone empirical p (0.010),
   but: (a) its observed CI lower is exactly 0.0 (the factor barely
   moves the bare-baseline positions — a fragile signal); (b) 0.010 is a
   *selected* p — fractal_efficiency was chosen from the whole v26 search,
   not pre-registered; (c) it does not survive the family-wise correction.
3. **No card survives family-wise correction** (best adjusted p ≈ 0.27).
   At the per-card level, **H1 = False on this window under this
   definition** — now with a defensible empirical-null p-value behind the
   statement, not just "the CI happened to include zero".
4. Several v26 factors (aroon, body_to_range, donchian, macd, roc,
   upper_wick) are **inert on the bare baseline**: their filters never
   fire without the other factors present. They only contributed in
   combination during v26. The single-factor definition makes this
   explicit (CI = 0, point = 0).

## What B3 does not establish

- Not pipeline-level FDR (that needs the LLM in the loop; Step 4 / B2).
- Not SPA / DSR standalone calibration.
- Not the Stage-2 paper-trading OOS question (the only fully clean OOS).
- With only 11 cards the family-wise correction has little power; a
  larger surviving library would tighten it.

## Files

| file | what |
|------|------|
| `per_card_null_n200.json` | full observed + 200 null CI records per card |
| `analysis_n200.txt` | the printed per-card table + both family-wise tests |

## Reproduce

```bash
uv run python scripts/per_card_null_distribution.py --n-null 200 --out /tmp/b3_null
# or re-analyze the archived JSON without recomputing:
uv run python scripts/per_card_null_distribution.py \
    --analyze-only docs/synthetic_null_results/b3_per_card_null/per_card_null_n200.json
```
