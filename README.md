# LBG-Agent

> A heuristic learning agent for trading, inspired by Jiayi Weng's [Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/).

LBG-Agent uses an LLM as a *strategy editor* under a strict sealed-evaluation
protocol. It proposes incremental edits to a SPY daily strategy through a
deterministic Python control plane (the **Orchestrator**), which decides
each trial's fate by running invariants, backtests, and a multi-objective
gate. No LLM ever touches the sealed test window.

**Documentation**
- [`PROPOSAL.html`](PROPOSAL.html) — source-of-truth specification for
  architecture, invariants, and verdict criteria
- [`docs/STAGE1_REPORT.html`](docs/STAGE1_REPORT.html) — Stage 1 实施回顾
  （中文图文）
- [`CLAUDE.md`](CLAUDE.md) — guidance for Claude Code sessions

## What is this

The framework runs a multi-trial Discovery loop. Each trial:

1. The **Editor** (LLM) proposes one structured edit to `strategy.yaml` or
   `indicators/`, with an explicit hypothesis and an expected outcome.
2. The Orchestrator validates the proposal against a five-layer invariant
   stack (Pydantic schema, AST static checks, restricted-namespace sandbox,
   `prefix_stability` dynamic check, gate-side runtime checks).
3. Two backtests run (train + validation). The **ValidationGate** decides
   accept or reject from a small set of categorical signals.
4. The **Reflector** (LLM) explains the outcome and updates four semantic
   memory documents. It can *never* override the mechanically computed
   hypothesis outcome — that's the project's main defense against
   sycophancy.
5. Every ten accepted trials, the **Curator** (LLM, shadow mode) compresses
   the four memory documents.
6. After the configured budget, the sealed test window opens *exactly once*
   and the H1 verdict is computed via a moving block bootstrap.

## The three LLM agents

| Agent | When | Input | Output |
|-------|------|-------|--------|
| **Editor** | Start of every trial | current `strategy.yaml`, recent trial summaries (categorical val signal only), four semantic memory `.md` files, active skills | one YAML proposal: edit type + hypothesis + expected train/val signals + fallback |
| **Reflector** | After the gate decides | proposal, mechanical `hypothesis_outcome`, actual `validation_signal`, train metrics | mechanical explanation + incremental bullet updates to the four `.md` files |
| **Curator** | Every 10 accepted trials (shadow mode) | the four `.md` files in full | compressed, deduplicated rewrites of the four `.md` files |

## Repository layout

```
backtest.py               # vectorized SPY daily backtest (~150 LOC)
policy_interpreter.py     # deterministic policy engine
strategy.yaml             # initial DSL config (sma_cross_baseline)

indicators/               # agent-authored pure-function indicators
lbg/
  builder/                # CandidateBuilder + 8 edit type appliers
  data/                   # yfinance + Tiingo cross-check, split_A/B/C aliasing
  dsl/                    # Pydantic schemas for strategy.yaml
  gate/                   # ValidationGate, HypothesisScorer, complexity
  git_manager.py          # one commit per trial, one branch per Curator cycle
  invariants/             # AST checks + prefix_stability
  knowledge/              # external factor knowledge base (read-only)
  memory/                 # MemoryManager (jsonl + md)
  orchestrator/           # ContextBuilder, RoleRunner, Curator, Discovery, prompts/
  parser/                 # ProposalParser, per-edit-type payloads
  sandbox/                # restricted-namespace exec + SIGALRM timeout
  schemas.py              # EditProposal, TrialRecord, 6 StrEnums
  sealed_vault.py         # write-once container for sealed_test_final.json
  skills/                 # SkillManager (skill_id.yaml store)
  stage2/                 # forward validation engine
  stage3/                 # PaperTradingEngine (bar-by-bar streaming)
  verdict/                # H1 moving block bootstrap + analysis_plan

scripts/
  long_discovery.py       # CLI entry for a multi-trial Discovery run
  migrate_factors.py      # one-shot import of the factor knowledge base
  reshuffle_proposal.py

tests/                    # 330+ tests, pytest-driven
docs/                     # human-facing reports
artifacts/                # runtime: sealed/, reports/  (gitignored)
memory/                   # runtime: per-run jsonl + md  (auditable)
runs/                     # runtime: per-trial editor.yaml + reflector.yaml
```

## Quick start

The project uses `uv` for dependencies and `ruff` for lint + format.

```bash
# 1. install
uv sync

# 2. configure secrets (.env is gitignored)
cat > .env <<'EOF'
TIINGO_TOKEN=<your tiingo token>          # for data cross-check
ANTHROPIC_API_KEY=<your anthropic key>    # default LLM provider
MIMO_API_KEY=<your mimo key>              # optional: MIMO provider
EOF

# 3. fetch SPY data (one-time; persists to data/spy_daily.parquet)
uv run python -m lbg.data.loader fetch

# 4. run a short Discovery (writes to /tmp/lbg_run/, doesn't touch this repo)
uv run python scripts/long_discovery.py --budget 5 --out /tmp/lbg_run

# 5. inspect the artifacts
open /tmp/lbg_run/artifacts/reports/discovery_report.html
cat /tmp/lbg_run/artifacts/sealed/sealed_test_final.json
```

## Switching LLM providers

Two providers are registered out of the box: Anthropic (default) and
MIMO (Anthropic-compatible REST endpoint).

```bash
LBG_PROVIDER=mimo uv run python scripts/long_discovery.py --budget 5 --out /tmp/lbg_mimo
```

Or in code:

```python
from lbg.orchestrator import RoleRunner
runner = RoleRunner(provider="mimo")          # or provider="anthropic"
```

The Orchestrator is provider-agnostic. Both providers produce bit-identical
sealed verdicts on the same baseline strategy, by design.

## Testing

```bash
uv run pytest -q                # 330+ tests; skips live LLM tests if keys absent
uv run ruff check               # lint
uv run ruff format --check      # format
```

Live LLM tests (`test_editor_live_*`, `test_reflector_live_one_call`,
`test_curator_live_one_call`, `test_editor_live_via_mimo_provider`) are
skipped automatically when `ANTHROPIC_API_KEY` / `MIMO_API_KEY` are not
set. They make real API calls and cost a small amount per run.

## Stage 2 + Stage 3

After a frozen strategy artifact is produced by Stage 1, the same backtest
engine drives two downstream stages, neither of which calls an LLM:

- **Stage 2 — forward validation**: `lbg.stage2.run_forward_validation`
  runs a deterministic backtest on post-sealed bars with a loose go/no-go
  gate (min trades, max drawdown floor, min Sharpe).
- **Stage 3 — paper trading**: `lbg.stage3.PaperTradingEngine` streams
  daily bars one at a time, recomputes positions on the growing history
  buffer, and logs realized PnL with the same cost model as the backtest.

Stage 4 (live capital) is intentionally out of scope.

## Project status

| Stage | Status |
|-------|--------|
| Stage 1 — Discovery loop (LLM-driven) | done, 13 Orchestrator submodules, all 8 edit types |
| Stage 2 — forward validation (no LLM) | done |
| Stage 3 — paper trading (no LLM) | done, simulation only |
| Stage 4 — live capital broker adapter | out of scope |
| Active Curator mode | not enabled (shadow only) |

Experimental result on the SMA(20/50) baseline (both providers, budget=20):
`H1 strong = False`, `H1 weak = False`. The strategy underperforms
buy-and-hold on the sealed window by 0.75 Sharpe units; the framework
reported this cleanly. See `docs/STAGE1_REPORT.html` § 7 – § 8.

## License

MIT — see [LICENSE](LICENSE).
