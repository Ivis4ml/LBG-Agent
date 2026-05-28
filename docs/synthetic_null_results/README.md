# Synthetic-Null Calibration Results · 2026-05-27

## Step 4 · cross-provider gate FPR (budget 6)

Raw artefacts from the Significance Hardening Step 4 ablation. Two providers
(`codex_cli` running gpt-5.5, `mimo_tp` running mimo-v2.5-pro) each ran
15 replications of `scripts/synthetic_null_calibration.py` with
`--budget 6 --iterations 1 --gate permissive`.

Files: `{codex_cli,mimo_tp}_summary.json`, `{codex_cli,mimo_tp}_replications.jsonl`.

## B2 · larger-budget rerun after the bug fixes (budget 15)

After fixing the auto-inject git commit (bug a) and the library-pollution
(bug b), a single-provider rerun with `codex_cli --n-replications 30
--budget 15` produced the first usable sealed-validation null data.

Files: `codex_cli_b2_budget15_summary.json`,
`codex_cli_b2_budget15_replications.jsonl` (records carry per-card
`per_card_null_stats` and a `budget` field).

Headline (27/30 reps OK, 3 transient codex mid-rep failures):

| metric | value |
|--------|-------|
| gate FPR per trial (budget 15) | 15.1% (mean 2.26 / 15 accepts per rep) |
| add_indicator null cards reaching sealed | **6** (vs 1 in Step 4, all errored) |
| of which errored | **0** (bug (a) fix confirmed) |
| null CI_lower distribution | [-1.548, -1.390, -0.853, -0.620, -0.395, 0.000] |
| null point ΔSharpe distribution | [-0.731, -0.636, 0.000, 0.065, 0.114, +0.559] |
| null cards with CI_lower > 0 | **0 / 6** |
| null cards BH-validated | 0 / 6 |
| SPA p-values (reps with ≥1 sealable card) | {0.046, 0.271, 0.507, 1.0, 1.0}, n=5 |

What B2 establishes:
- The sealed CI specificity claim now has *some* empirical support: across
  6 add_indicator null cards, none cleared CI_lower > 0. The null point
  ΔSharpe is centred near zero — but note the max null point estimate is
  **+0.559**, demonstrating that bootstrapped null data can produce a
  point ΔSharpe larger than v26's best real card (+0.249) by chance. This
  is exactly why point estimates without CIs mislead.

Positioning v26 `vol_regime_zscore` (observed point +0.249, CI_lower -0.091)
against the B2 null pool: 83rd percentile, empirical one-sided p ≈ 0.167.
A near-miss — stronger than ~83% of null cards but nowhere near
significant. Consistent with the §10.5 resolution floor (ΔSR ≈ 0.30).

**n = 6 is too small.** The empirical p floors at 1/6 ≈ 0.167; we cannot
resolve below that. This motivates **B3** (a separate experiment): compute
each v26 card's sealed CI directly over N≈100 bootstrap draws, bypassing
the LLM (which rarely proposes add_indicator on null data, throttling
yield). B3 uses a single-factor-vs-baseline definition rather than the
pathwise-along-trajectory definition (the v26 trial commits are gone), so
it recomputes each card's observed CI under the matching definition.

### Provider-quota note (B2 run, 2026-05-27 evening)

All three subscription providers hit limits during this work:
codex_cli (ChatGPT usage cap), claude_cli (Anthropic credit on large
prompts), mimo_tp (cluster 429). The B2 dataset is pure codex_cli after
the ChatGPT cap reset; the 4 budget-15 codex reps from the first attempt
plus 23 more after reset.

---

## Step 4 detail (budget 6)

Each replication stationary-bootstraps SPY daily OHLCV (Politis-Romano,
mean block length 10), preserving the date index but resampling row
contents. Both providers use the same `rng_seed=2026`, so they see
identical bootstrap parquets — observed differences are LLM behaviour
only.

## Files

| file                              | what                                                                  |
|-----------------------------------|-----------------------------------------------------------------------|
| `codex_cli_summary.json`          | aggregated metrics for codex_cli                                      |
| `codex_cli_replications.jsonl`    | per-replication record (one line per rep)                             |
| `mimo_tp_summary.json`            | aggregated metrics for mimo_tp                                        |
| `mimo_tp_replications.jsonl`      | per-replication record (one line per rep)                             |

## Headline result

| metric                          | codex_cli         | mimo_tp           |
|---------------------------------|-------------------|-------------------|
| gate acceptance rate per trial  | 8.9% ± 10.7%      | 7.8% ± 8.6%       |
| trials accepted (total)         | 8 / 90            | 7 / 90            |
| of which `add_indicator`        | **1**             | **0**             |
| sealed `CI > 0` validated       | 0 / 1 (1 errored) | n/a               |
| BH-validated                    | 0                 | 0                 |
| DSR-validated                   | 0                 | 0                 |
| wall time                       | ~30 min           | ~114 min          |

## What this *does* and *does not* show

**Does**: the gate's per-trial FPR on bootstrapped SPY data is roughly
8% for both providers, with no detectable provider difference at n=15.
LLMs in this configuration naturally propose mostly `remove_filter` /
`change_sizing_mode` / `simplify` edits when faced with null data — a
secondary finding worth noting: bootstrap removes the signal that
auto-injected library factors latch onto, so the Editor's most
"productive" move on null is to strip those filters back off.

**Does NOT**: this run does *not* meaningfully test sealed validation's
specificity. Only 1 `add_indicator` candidate was produced across the
30 replications, and that single candidate errored out during sealed
validation due to an orthogonal git-materialize bug. Any claim of the
form "sealed validation correctly filtered the null false positives"
is not supported by this data.

## Known issues (filed for future fix)

**(a) Sealed validation `git materialize` failure on auto-injected
factors.**

`lbg/verdict/per_card.py::_validate_one_card` calls
`git show <SHA>:indicators/<fn>.py` to recover the parent strategy at
the trial's parent commit. When library factors are auto-injected at
iteration 1 but their `.py` files end up on disk without being part of
the relevant commit lineage (this happens in the null calibration
subprocess setup but not in the live repo's main campaign), this
command fails with exit 128. Real v26 campaigns avoid this because the
auto-inject step commits the indicator files explicitly; the null
calibration subprocess setup does not. Filed as a follow-up.

**(b) Null campaign pollutes the persistent factor library.**

The campaign subprocess uses `library_dir` defaulting to the source
repo's `alpha_cards_library/`. Even under `LBG_DATA_PATH` (bootstrapped
data), `sync_from_run` at end of each replication wrote the accepted
`dispersion_regime` card from codex rep_007 into the source library
(observable by its absurd -99.99% drawdown). The pollution was reverted
manually before this commit, but the bug remains: the null calibration
script should set `--library-dir` to a per-replication scratch directory
so library state stays isolated from real runs.

Both bugs need fixing before running B2 (larger-budget null
calibration) to avoid (a) under-counting sealed validation and (b)
corrupting the production library.

## Reproducing

```bash
uv run python scripts/synthetic_null_calibration.py \
    --provider codex_cli --n-replications 15 --budget 6 --iterations 1 \
    --base-out /tmp/null_codex

uv run python scripts/synthetic_null_calibration.py \
    --provider mimo_tp --n-replications 15 --budget 6 --iterations 1 \
    --base-out /tmp/null_mimo_tp

uv run python scripts/synthetic_null_calibration.py \
    --compare /tmp/null_codex /tmp/null_mimo_tp
```
