# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LBG-Agent is a research implementation of **LBG-Trader**, a sealed-evaluation, multi-agent LLM strategy-discovery system for SPY daily trading. The repo is currently greenfield — only `PROPOSAL.html` and `README.md` exist. `PROPOSAL.html` is the **source of truth** for architecture, protocol, invariants, and verdict criteria; read the relevant section before any non-trivial design decision. Section numbers below refer to `PROPOSAL.html`.

## Behavioral guidelines

Before writing or reviewing code, invoke the `karpathy-guidelines` skill (from the bundled `andrej-karpathy-skills` plugin). It keeps changes surgical, surfaces assumptions, and forces verifiable success criteria — all of which matter for a sealed-evaluation experiment where one careless edit invalidates the run.

## Architecture (§3–§4)

Two engines share one frozen artifact:

- **Discovery engine** (Stage 1, LLM-driven) runs the LBG loop: Editor → Orchestrator builds candidate → invariants → train+validation backtest → ValidationGate → Reflector → MemoryManager. Curator runs every 10 accepted trials. The sealed test is opened exactly once at the end of the run.
- **Execution engine** (Stages 2–4, no LLM) runs deterministic backtest, paper trading, and live capital. The strategy is frozen between re-discoveries.

The **Orchestrator** is a deterministic Python control plane of 13 single-purpose submodules: `ContextBuilder`, `RoleRunner`, `ProposalParser`, `CandidateBuilder`, `SandboxExecutor`, `InvariantRunner`, `BacktestRunner`, `ValidationGate`, `HypothesisScorer`, `MemoryManager`, `SkillManager`, `GitManager`, `SealedVault`, `ReportBuilder`. **No LLM calls anywhere in the Orchestrator** — its determinism is the credibility of the experiment.

Four LLM roles (`claude-sonnet-4-6` by default since 2026-05-24; opus-4-7 still usable via the `model=` arg on `RoleRunner`):

- **Editor** — proposes exactly one of eight discrete edit types per trial as structured YAML: `add_indicator`, `parameter_change`, `add_filter`, `remove_filter`, `change_sizing_mode`, `change_exit_rule`, `simplify`, `revert_to_trial_N`.
- **Reflector** — explains outcomes mechanically; never judges the hypothesis (the Orchestrator's `HypothesisScorer` does that).
- **Curator** — runs every 10 accepted trials; starts in shadow mode (memory compression only).

## Prescribed layout (§4)

```text
indicators/                    pure-function indicators (agent-authored)
policy_interpreter.py          fixed deterministic policy engine
strategy.yaml                  DSL config (filters, sizing mode, exit rules)
invariants/                    static invariant rule files
memory/
  trials.jsonl                 append-only trial events
  reflections.jsonl
  invariant_failures.jsonl
  agent_compute.jsonl
  accepted_rules.md            compressed semantic memory
  failed_directions.md
  open_questions.md
  do_not_repeat.md
artifacts/accepted/trial_NNNN/ frozen per-accepted-trial snapshots
artifacts/sealed/              write-once sealed-test artifact
artifacts/reports/             final HTML reports
runs/NNNN/                     per-trial LLM outputs
skills/                        in-system edit recipes (curated by Curator)
```

The project's `skills/` directory holds in-system edit recipes — a proposal concept. Claude Code skills live under `.claude/skills/`. Different things; do not conflate.

## Sealed evaluation (§5 — highest priority)

Splits are aliased so the LLM cannot infer the calendar:

- `split_A` (train, ~2266 bars) — LLM sees full summary and metrics.
- `split_B` (validation, ~505 bars) — LLM sees **only** categorical signals: `accepted`, `rejected_drawdown_regression`, `rejected_turnover`, `rejected_complexity`, `rejected_no_significant_improvement`. Never raw numbers.
- `split_C` (sealed, ~1006 bars) — opened once by `SealedVault` at end of run; refused on any second write. No LLM ever sees this data.

Calendar years and named events are redacted from all agent-facing context (enforced by the `no_data_snooping` invariant). When generating agent prompts or context, never include raw dates or event names.

## Invariants (§6)

All invariants run before any backtest. The checks Claude would otherwise miss:

- **`prefix_stability`** (dynamic, most critical) — perturb `D_{>t}` and assert `f(D)[t]` unchanged for every `t`. Catches whole-series normalization, look-ahead in rolling windows, and forward-fills that leak future values. Stricter than AST checks.
- AST-static: `no_negative_indexing`, `no_future_shift`, `no_forward_fill_future`, `whitelisted_imports_only` (numpy / pandas / math), `pure_function` (no globals, no side effects, no I/O).
- Configuration: `no_python_edit` (agents cannot modify `backtest.py` or `policy_interpreter.py`), `no_forbidden_file_access`, `no_data_snooping`.

## Validation gate (§7)

Per-trial, multi-objective — **not** a strict CI > 0. Combines: minimum trade count, drawdown-regression cap (`val.max_drawdown < current.max_drawdown * 1.15`), turnover cap, complexity cap (`Δcomplexity < MAX_DELTA`), one-sided utility LCB at α=0.20, Pareto improvement, and overfit-gap cap. Final H1 verdict on `split_C`: moving-block bootstrap (block_len=10, n_bootstrap=1000) of paired ΔSharpe vs best baseline. H1-strong requires 95% CI lower bound > 0.

## Do-not list

- Do not let agents see raw OHLCV, raw validation numbers, sealed-window data, calendar years, or named events.
- Do not edit `backtest.py`, `policy_interpreter.py`, the invariant suite, gate thresholds, or the model-selection config during a discovery run (locked per §20).
- Do not embed LLM calls in any Orchestrator submodule. The control plane stays deterministic.
- Do not reuse sealed windows across re-discoveries.
- Do not skip pre-registration — `analysis_plan.yaml` must be committed before the sealed test opens. Post-hoc comparisons are exploratory, not evidence for H1.
- Do not edit strategy in Stages 2–4; re-discovery is the only way to change a frozen artifact.

## Coding conventions

- Indicators are **pure functions** of a single OHLCV DataFrame: deterministic, no globals, no I/O, prefix-stable. Use the docstring template in §4.5; invoke `/new-indicator <name>` to scaffold.
- DSL edits are the eight discrete types only; free-form code editing is forbidden outside `add_indicator`.
- `snake_case` for functions, YAML for config, `.jsonl` for append-only logs, `.md` for semantic memory.

## Git discipline (§10)

One commit per trial; one branch per Curator cycle. Full lineage must remain auditable — do not rebase or force-push trial commits.

## Tooling

- Package manager: `uv` (`uv add`, `uv run`, `uv sync`).
- Lint and format: `ruff` (single tool for both).
- LLM: `anthropic` Python SDK by default. Default provider is Anthropic, default model `claude-sonnet-4-6` (switched from opus-4-7 on 2026-05-24 to keep campaign costs sustainable; pass `model="claude-opus-4-7"` to RoleRunner for opus). Four providers registered: `anthropic` (API key), `mimo` (Xiaomi, anthropic-compatible API key), `claude_cli` (Claude Code OAuth subscription, no per-call cost), `codex_cli` (ChatGPT OAuth subscription via `codex exec`, default model `gpt-5.5`). Switch with `RoleRunner(provider="...")` or `LBG_PROVIDER=...`. Provider table is in `lbg/orchestrator/role_runner.py` `PROVIDERS`; both subscription-CLI subprocess wrappers live in `lbg/orchestrator/cli_provider.py` (`ClaudeCliClient`, `CodexCliClient`). Auth per provider: `anthropic` reads `ANTHROPIC_API_KEY`, `mimo` reads `MIMO_API_KEY`, `claude_cli` uses existing `claude` OAuth (run `claude` once interactively), `codex_cli` uses existing `codex` OAuth (run `codex login` once).
- Data: SPY daily OHLCV via `yfinance`, cross-checked against Tiingo (`TIINGO_TOKEN`), versioned as Parquet.
