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
6. After the configured budget, the sealed test window opens *exactly once*.
   Primary H1 is the count of sealed-validated alpha cards; the final
   strategy Sharpe bootstrap is reported separately as a supplementary
   diagnostic.

## The four LLM agents

| Agent | When | Input | Output |
|-------|------|-------|--------|
| **Editor** | Start of every trial | current `strategy.yaml`, recent trial summaries (categorical val signal only), four semantic memory `.md` files, active skills, dossier hint shortlist from `knowledge/factors/`, banned-indicator-name list | one YAML proposal: edit type + hypothesis + expected train/val signals + fallback + `cited_factors` |
| **Reflector** | After the gate decides | proposal, mechanical `hypothesis_outcome`, actual `validation_signal`, train metrics | mechanical explanation + incremental bullet updates to the four `.md` files |
| **Curator** | Every 10 accepted trials (shadow mode) | the four `.md` files in full | compressed, deduplicated rewrites of the four `.md` files |
| **Translator** | After each accepted `add_indicator` | the agent-authored `.py` source + alpha card metadata + observed train metrics + cited factors | JSON dossier matching the seed library schema, written to `alpha_cards/dossiers/<name>_trial_NNNN.txt` |

## Repository layout

```
backtest.py               # vectorized SPY daily backtest (~150 LOC)
policy_interpreter.py     # deterministic policy engine
strategy.yaml             # initial DSL config (sma_cross_baseline)

indicators/               # agent-authored pure-function indicators
lbg/
  alpha_cards.py          # AlphaCardWriter + dossier-link matcher (PROPOSAL §6.6)
  builder/                # CandidateBuilder + 8 edit type appliers (incl. add_indicator attach)
  data/                   # yfinance + Tiingo cross-check, split_A/B/C aliasing
  dsl/                    # Pydantic schemas for strategy.yaml
  gate/                   # ValidationGate, HypothesisScorer, complexity (strict / permissive presets)
  git_manager.py          # one commit per trial, one branch per Curator cycle
  invariants/             # AST checks + prefix_stability
  knowledge/              # factor-library retrieval (search / get_dossier / load_index)
  memory/                 # MemoryManager (jsonl + md + tried_factors.jsonl cross-iter log)
  orchestrator/           # ContextBuilder, RoleRunner (4 roles), Curator, Discovery, Campaign, live_status, prompts/, templates/
  parser/                 # ProposalParser, per-edit-type payloads
  sandbox/                # restricted-namespace exec + SIGALRM timeout
  schemas.py              # EditProposal, TrialRecord, 6 StrEnums
  sealed_vault.py         # write-once container for sealed_test_final.json
  skills/                 # SkillManager (skill_id.yaml store)
  stage2/                 # forward validation engine
  stage3/                 # PaperTradingEngine (bar-by-bar streaming)
  translator.py           # 4th LLM role · .py → JSON dossier (round-trip bridge)
  verdict/                # alpha-card H1 + supplementary Sharpe verdicts

knowledge/factors/        # 564 read-only seed dossiers + index.jsonl
scripts/
  campaign.py             # multi-iteration Discovery with persisted skills + alpha cards
  long_discovery.py       # single Discovery run (legacy CLI)
  migrate_factors.py      # one-shot import of the factor knowledge base

tests/                    # 440 tests, pytest-driven
docs/                     # human-facing reports (STAGE1_REPORT.html · 11 sections)
artifacts/                # runtime: sealed/, reports/, live/  (live/ dashboard.html for browser)
alpha_cards/              # runtime: per-accepted-trial card + dossier JSON (auditable)
memory/                   # runtime: per-run jsonl + md + tried_factors.jsonl  (auditable)
campaigns/                # runtime: per-iter sealed vault + campaign_summary.json
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

# 4a. single Discovery (writes to /tmp/lbg_run/, doesn't touch this repo)
uv run python scripts/long_discovery.py --budget 5 --out /tmp/lbg_run

# 4b. or a multi-iteration campaign with cross-iter persisted memory
uv run python scripts/campaign.py --iterations 3 --budget 5 \
    --out /tmp/lbg_camp --gate permissive

# 5a. inspect Discovery artifacts
open /tmp/lbg_run/artifacts/reports/discovery_report.html
cat /tmp/lbg_run/artifacts/sealed/sealed_test_final.json

# 5b. open the live browser dashboard during a campaign (2s auto-refresh)
open /tmp/lbg_camp/artifacts/live/dashboard.html
```

## Switching LLM providers

Four providers are registered out of the box:

| Provider     | Auth                                | Default model      | When to use                                      |
|--------------|-------------------------------------|--------------------|--------------------------------------------------|
| `anthropic`  | `ANTHROPIC_API_KEY` (per-call)      | claude-sonnet-4-6  | production / publication runs                    |
| `mimo`       | `MIMO_API_KEY` (per-call)           | mimo-v2.5-pro      | Anthropic-compatible REST alternative            |
| `claude_cli` | Claude Code OAuth (subscription)    | claude-sonnet-4-6  | dev / testing using a Claude Code Max plan       |
| `codex_cli`  | ChatGPT OAuth (subscription)        | gpt-5.5            | dev / testing using a ChatGPT Plus/Pro plan      |

```bash
# direct API
LBG_PROVIDER=mimo uv run python scripts/long_discovery.py --budget 5 --out /tmp/lbg_mimo

# Claude Code CLI -- spawns `claude -p` per call, no per-call API cost
LBG_PROVIDER=claude_cli uv run python scripts/long_discovery.py --budget 5 --out /tmp/lbg_cli

# Codex CLI -- spawns `codex exec` per call, charged to ChatGPT subscription
LBG_PROVIDER=codex_cli uv run python scripts/long_discovery.py --budget 5 --out /tmp/lbg_codex

# also exposed as --provider on both scripts
uv run python scripts/campaign.py --provider claude_cli --iterations 2 --budget 3 --out /tmp/lbg_cli
uv run python scripts/campaign.py --provider codex_cli  --iterations 2 --budget 3 --out /tmp/lbg_codex
```

In code:

```python
from lbg.orchestrator import RoleRunner
runner = RoleRunner(provider="claude_cli")    # or "anthropic" | "mimo" | "codex_cli"
```

Both subscription-CLI providers spawn a fresh subprocess per call. They
require the respective binary on PATH and an active subscription login:

- `claude_cli` — run `claude` interactively once. Invokes
  `claude -p --output-format json --model <model> --system-prompt <sys>`
  with tools / sessions / slash commands disabled.
- `codex_cli` — run `codex login` once. Invokes
  `codex exec --ephemeral --sandbox read-only --ignore-user-config -m <model>
  -o <tmp> -` with the system + user prompts merged on stdin (Codex has
  no separate `--system-prompt` slot).

Cold-start overhead is ~1-3s per call; in exchange there is no per-token
API billing on subscription plans. Both wrappers retry with exponential
backoff (30s, 60s, 120s) on rate-limit / quota errors. The Orchestrator
stays provider-agnostic.

## Testing

```bash
uv run pytest -q                # 479 tests; skips live LLM tests if keys absent
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
| Translator agent (4th LLM role) | wired but untriggered (no accepted `add_indicator` yet) |
| Campaign multi-iteration loop | done, 4 live runs (v2-v5b) under Anthropic Opus 4.7 |
| Live browser dashboard | done, `artifacts/live/dashboard.html` |
| Active Curator mode | not enabled (shadow only) |

Experimental result on the SMA(20/50) baseline across six historical
campaigns (90 trials total, strict + permissive gate variants): primary
H1 remains false because 0 `add_indicator` accepts produced 0 alpha cards.
Earlier reports that discuss `H1 weak` from strategy-level ΔSharpe should
be read as supplementary strategy diagnostics, not as the current proposal's
primary H1. The default Anthropic model is now `claude-sonnet-4-6` to keep
new discovery runs cheaper; Opus remains available through `RoleRunner(model=...)`.

The primary H1 metric is now computable end-to-end. `lbg.verdict.per_card.compute_per_card_sealed_validation`
runs the pathwise incremental Sharpe + moving-block bootstrap for each
accepted alpha card during `Discovery._seal` and writes
`incremental_sharpe_ci_lower` into `card.evidence.sealed_summary`;
`compute_alpha_card_h1_verdict` then counts cards with CI lower bound > 0.
The live dashboard surfaces both the count and per-card CI bounds.
For honest single-shot H1 evidence, pass `--seal-only-last` to
`scripts/campaign.py`; the default still seals every iteration for
dashboard visibility (development mode, effective n=1).

## License

MIT — see [LICENSE](LICENSE).
