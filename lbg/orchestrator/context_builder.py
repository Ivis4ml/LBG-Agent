"""ContextBuilder.editor_view -- assemble the Editor's context with redaction.

The Editor sees:
  - The current strategy DSL (`strategy.yaml` content) and source excerpts
    for currently wired indicators.
  - A summary of recent trials: trial_id, edit type + summary, hypothesis,
    expected vs actual validation signal (categorical), and hypothesis outcome.
    Train metrics are numeric (Editor reads its own training feedback) but
    validation metrics are NEVER passed -- only the categorical signal.
  - The four semantic-memory documents: accepted_rules.md, failed_directions.md,
    open_questions.md, do_not_repeat.md (whatever the Reflector / Curator
    has written so far).

It never sees:
  - Calendar years, dates, named events (enforced by `assert_redacted`).
  - Raw OHLCV.
  - Validation metric numbers (sharpe / mdd / turnover on split_B).
  - Sealed data of any kind.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from lbg.dsl.schema import Strategy
from lbg.knowledge.factors import search as factor_search
from lbg.memory.manager import SEMANTIC_MEMORY_FILES, MemoryManager
from lbg.schemas import (
    HypothesisOutcome,
    TrialRecord,
    ValidationSignal,
)
from lbg.skills import Skill, SkillManager

# Map each rejection-style validation signal to a short list of Chinese
# search terms that recover relevant dossiers from knowledge/factors/. The
# Chinese terms are required because dossier `category` and `one_line` are
# authored in zh; English queries would mostly miss.
_REJECTION_QUERY_TERMS: dict[ValidationSignal, tuple[str, ...]] = {
    ValidationSignal.REJECTED_DRAWDOWN_REGRESSION: (
        "CalmarRatio",
        "MaxDrawdownDuration",
        "波动率",
        "回撤",
    ),
    ValidationSignal.REJECTED_TURNOVER: ("趋势强度", "状态识别"),
    ValidationSignal.REJECTED_NO_SIGNIFICANT_IMPROVEMENT: ("动量", "均值回归"),
    # complexity rejection: do NOT add factors -- the system is already too
    # large. Empty tuple means no rejection-derived hints from this signal.
    ValidationSignal.REJECTED_COMPLEXITY: (),
    ValidationSignal.ACCEPTED: (),
}

_BUYHOLD_RISK_QUERY_TERMS: tuple[str, ...] = (
    "回撤",
    "波动率",
    "趋势强度",
    "状态识别",
)

# Same range as redaction's _YEAR_RE: 1990-2049 plus 2050. The Editor's
# outgoing-prompt redaction guard refuses these tokens. Dossiers can mention
# e.g. "1995 由 Tushar Chande 提出"; we replace the year with <YEAR> before
# injection rather than censor the surrounding sentence.
_REDACT_YEAR_RE = re.compile(r"\b(?:19[9]\d|20[0-4]\d|2050)\b")


@dataclass(frozen=True)
class PastTrialSummary:
    """One line in the Editor's view of trial history.

    Numeric train metrics are included (Editor's own training feedback).
    Validation metrics are categorical only.
    """

    trial_id: int
    edit_type: str
    edit_summary: str
    hypothesis_text: str
    expected_validation_signal: str
    actual_validation_signal: str  # one of 5 categorical values
    hypothesis_outcome: str
    train_sharpe: float
    train_max_drawdown: float
    train_turnover: float
    train_num_trades: int
    decision: str  # "accept" | "reject"
    # B · forced fallback: what the Editor said it would try next if this
    # trial was rejected. Empty string when the trial accepted (no fallback
    # is owed) or when the field is missing on a historical trial.
    fallback_if_rejected: str = ""
    # Dossier factor names the Editor cited for this trial; used by the
    # hint pipeline to skip factors that were already tried.
    cited_factors: tuple[str, ...] = ()


@dataclass(frozen=True)
class FactorHint:
    """One entry the Editor sees from the read-only factor knowledge base.

    Produced by ContextBuilder from `lbg.knowledge.factors.search()`. The
    Editor treats the list as hints, never as authority: invariants run
    regardless of whether an indicator was inspired by a dossier.
    """

    name: str  # dossier factor_name, e.g. "ADX"
    category: str  # short Chinese category line from the dossier index
    one_line: str  # one-paragraph definition; years already scrubbed


@dataclass(frozen=True)
class IndicatorCodeSummary:
    """Source excerpt for one currently wired indicator function."""

    name: str
    fn: str
    source_path: str
    source_excerpt: str


@dataclass(frozen=True)
class SeedTemplateHint:
    """One executable seed template surfaced to the Editor.

    Unlike `FactorHint` (text-only dossier from knowledge/factors/), a
    seed template carries the Python source itself. The Editor can copy
    the source verbatim into `add_indicator.change.source` and use the
    suggested threshold / rearm_threshold to attach it. This is the path
    designed to close the 0-accept-rate problem on `add_indicator`.
    """

    fn: str
    category: str  # trend | momentum | mean_reversion | volatility | drawdown | regime
    role: str  # entry | exit | sizing
    inputs: tuple[str, ...]
    default_params: dict[str, object]
    suggested_threshold: float | None
    suggested_rearm_threshold: float | None
    one_line: str
    source: str


@dataclass(frozen=True)
class LibraryCardSummary:
    """One previously-validated alpha card surfaced to the Editor.

    The Editor should treat these as "already-discovered factors" and
    look for COMPLEMENTARY directions, not re-propose them. Both the
    fn name and the dossier factor link are shown so the Editor can
    reason about coverage of the factor space.
    """

    alpha_id: str
    fn: str
    indicator_name: str
    dossier_factor: str | None
    incremental_sharpe_point: float | None


@dataclass(frozen=True)
class EditorContext:
    strategy_yaml: str  # rendered DSL document (already cross-checked for redaction)
    recent_trials: list[PastTrialSummary] = field(default_factory=list)
    semantic_memory: dict[str, str] = field(default_factory=dict)  # filename -> content
    skills: list[Skill] = field(default_factory=list)  # active skills from SkillManager
    factor_hints: list[FactorHint] = field(default_factory=list)
    indicator_code: list[IndicatorCodeSummary] = field(default_factory=list)
    seed_templates: list[SeedTemplateHint] = field(default_factory=list)
    library_cards: list[LibraryCardSummary] = field(default_factory=list)
    # Indicator fn names that earlier add_indicator trials proposed --
    # accepted OR rejected. Editor MUST NOT re-propose any of these by
    # the same fn name. The campaign library accumulates; same-fn
    # re-proposals don't add coverage and clog the loop with churn.
    banned_indicator_fns: tuple[str, ...] = ()


class ContextBuilder:
    def __init__(
        self,
        memory: MemoryManager,
        *,
        repo_root: str | Path = ".",
        recent_trials_limit: int = 10,
        skills: SkillManager | None = None,
        factor_hints_per_term: int = 3,
        factor_hints_total_cap: int = 8,
        factor_one_line_max_chars: int = 240,
        seed_template_cap: int = 6,
        library_dir: str | Path | None = None,
    ) -> None:
        self.memory = memory
        self.repo_root = Path(repo_root)
        self.recent_trials_limit = recent_trials_limit
        self.skills = skills or SkillManager(self.repo_root / "skills")
        self.factor_hints_per_term = factor_hints_per_term
        self.factor_hints_total_cap = factor_hints_total_cap
        self.factor_one_line_max_chars = factor_one_line_max_chars
        self.seed_template_cap = seed_template_cap
        # Durable alpha-card library to inject into the Editor view as
        # "factors already discovered, please build on top". None = no
        # injection (back-compat for non-campaign Discovery runs).
        self.library_dir = Path(library_dir) if library_dir is not None else None

    def editor_view(self, current_strategy: Strategy) -> EditorContext:
        """Build the Editor's view of the system. Redaction is enforced by the
        caller (RoleRunner) on the final rendered prompt -- not here, so we
        don't reject legitimate strategy content that happens to include
        digits like `period: 2020` (legitimate but ambiguous; the renderer
        handles spacing to disambiguate)."""

        strategy_yaml = yaml.safe_dump(
            current_strategy.model_dump(mode="python"),
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )

        recent = self._recent_trial_summaries()
        semantic = self._read_semantic_memory()
        active_skills = self.skills.list_active()
        factor_hints = self._factor_hints(current_strategy, recent)
        indicator_code = self._indicator_code_summaries(current_strategy)
        seed_templates = self._seed_template_hints(current_strategy, recent)
        library_cards = self._library_summaries()
        banned_fns = self._banned_indicator_fns(current_strategy)

        return EditorContext(
            strategy_yaml=strategy_yaml,
            recent_trials=recent,
            semantic_memory=semantic,
            skills=active_skills,
            factor_hints=factor_hints,
            indicator_code=indicator_code,
            seed_templates=seed_templates,
            library_cards=library_cards,
            banned_indicator_fns=banned_fns,
        )

    # ---- internals ----

    def _recent_trial_summaries(self) -> list[PastTrialSummary]:
        trials = self.memory.read_trials()
        recent = trials[-self.recent_trials_limit :]
        return [_summarize(t) for t in recent]

    def _read_semantic_memory(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for fname in SEMANTIC_MEMORY_FILES:
            p = self.memory.memory_dir / fname
            if p.exists():
                out[fname] = p.read_text(encoding="utf-8")
        return out

    def _indicator_code_summaries(self, strategy: Strategy) -> list[IndicatorCodeSummary]:
        seen: set[str] = set()
        out: list[IndicatorCodeSummary] = []
        for spec in strategy.indicators:
            if spec.fn in seen:
                continue
            seen.add(spec.fn)
            path = self.repo_root / "indicators" / f"{spec.fn}.py"
            if not path.exists():
                continue
            source = path.read_text(encoding="utf-8")
            out.append(
                IndicatorCodeSummary(
                    name=spec.name,
                    fn=spec.fn,
                    source_path=str(path.relative_to(self.repo_root)),
                    source_excerpt=_source_excerpt(source),
                )
            )
        return out

    def _derive_search_terms(
        self,
        strategy: Strategy,
        recent: list[PastTrialSummary],
    ) -> list[str]:
        """Build an ordered, deduplicated list of factor-search queries.

        Strategy is to combine (a) what the current strategy already uses --
        indicator names and their backing function names -- with (b) the
        rejection modes seen in recent trials. The order matters because
        ContextBuilder caps the total hint count: items earlier in this list
        contribute hits first.
        """
        terms: list[str] = []
        if _looks_like_buyhold(strategy):
            terms.extend(_BUYHOLD_RISK_QUERY_TERMS)
        for spec in strategy.indicators:
            if spec.fn:
                terms.append(spec.fn)
            if spec.name and spec.name != spec.fn:
                terms.append(spec.name)
        for t in recent:
            try:
                signal = ValidationSignal(t.actual_validation_signal)
            except ValueError:
                continue
            for q in _REJECTION_QUERY_TERMS.get(signal, ()):
                terms.append(q)
        # dict.fromkeys preserves insertion order while deduplicating.
        return list(dict.fromkeys(terms))

    def _factor_hints(
        self,
        strategy: Strategy,
        recent: list[PastTrialSummary],
    ) -> list[FactorHint]:
        """Retrieve a deduplicated, length-capped list of factor hints.

        Skips factors already tried in `recent` (via `cited_factors` or
        indicator-name substring on the strategy's current indicators) --
        the campaign should explore the dossier breadth-first, not show
        the same top-5 every trial.

        Defensive against a missing knowledge base: factor_search returns
        an empty list when knowledge/factors/index.jsonl is absent, so the
        method simply returns []. This keeps ContextBuilder usable in
        unit-test repos that ship without the factor library.
        """
        terms = self._derive_search_terms(strategy, recent)
        if not terms:
            return []
        tried = self._tried_factor_names(strategy, recent)
        seen: set[str] = set()
        out: list[FactorHint] = []
        for term in terms:
            for hit in factor_search(term, top_k=self.factor_hints_per_term):
                name = (hit.get("factor_name") or "").strip()
                if not name or name in seen:
                    continue
                if name in tried:
                    continue
                seen.add(name)
                one_line = _truncate(
                    _scrub_years(hit.get("one_line") or ""), self.factor_one_line_max_chars
                )
                out.append(
                    FactorHint(
                        name=name,
                        category=_scrub_years(hit.get("category") or ""),
                        one_line=one_line,
                    )
                )
                if len(out) >= self.factor_hints_total_cap:
                    return out
        return out

    def _seed_template_hints(
        self,
        strategy: Strategy,
        recent: list[PastTrialSummary],
    ) -> list[SeedTemplateHint]:
        """Surface a short list of executable seed templates to the Editor.

        Filtering heuristic:
          - on a buyhold-like baseline (one-shot entry), bias toward
            role=exit templates so add_indicator has a non-cash-trap
            attach path (drawdown / vol-regime + rearm_threshold);
          - if recent trials hit `reject_drawdown_regression`, prefer
            drawdown + volatility templates;
          - if recent trials hit `reject_no_significant_improvement`,
            broaden to trend + momentum so the search doesn't get stuck;
          - otherwise pick a balanced default across categories.

        Templates whose fn already appears in the current strategy or in
        the cross-iteration `tried_factors.jsonl` are filtered out so the
        Editor sees fresh material every trial.
        """
        from lbg.seed_templates import list_templates

        all_templates = list_templates()
        if not all_templates:
            return []

        in_use_fns = {s.fn for s in strategy.indicators}
        # Cross-iter dedup: previously-tried indicator fns should not be
        # re-surfaced. Same mechanism the FactorHint pipeline uses.
        tried_fns: set[str] = set()
        for rec in self.memory.read_tried_factors():
            if rec.indicator_fn:
                tried_fns.add(rec.indicator_fn)

        # Decide category preference.
        prefer_exit = _looks_like_buyhold(strategy)
        recent_signals = {t.actual_validation_signal for t in recent}
        prefer_drawdown_vol = (
            "rejected_drawdown_regression" in recent_signals
            or "rejected_turnover" in recent_signals
        )
        prefer_broad = "rejected_no_significant_improvement" in recent_signals

        # Build prioritized category order. Earlier entries get picked first.
        category_order: list[str]
        if prefer_drawdown_vol:
            category_order = ["drawdown", "volatility", "regime", "trend", "momentum"]
        elif prefer_exit:
            category_order = ["drawdown", "regime", "volatility", "trend", "momentum"]
        elif prefer_broad:
            category_order = ["trend", "momentum", "regime", "mean_reversion", "volatility"]
        else:
            category_order = [
                "trend",
                "momentum",
                "mean_reversion",
                "volatility",
                "drawdown",
                "regime",
            ]

        # Walk categories in priority order, take at most ~2 per category
        # so the shortlist stays diverse.
        per_cat_cap = 2
        out: list[SeedTemplateHint] = []
        seen_fns: set[str] = set()
        for cat in category_order:
            cat_count = 0
            for t in all_templates:
                if t.category != cat:
                    continue
                if t.fn in in_use_fns or t.fn in tried_fns or t.fn in seen_fns:
                    continue
                source = t.load_source()
                out.append(
                    SeedTemplateHint(
                        fn=t.fn,
                        category=t.category,
                        role=t.role,
                        inputs=tuple(t.inputs),
                        default_params=dict(t.default_params),
                        suggested_threshold=t.suggested_threshold,
                        suggested_rearm_threshold=t.suggested_rearm_threshold,
                        one_line=_truncate(_scrub_years(t.one_line), 200),
                        source=source,
                    )
                )
                seen_fns.add(t.fn)
                cat_count += 1
                if cat_count >= per_cat_cap or len(out) >= self.seed_template_cap:
                    break
            if len(out) >= self.seed_template_cap:
                break
        return out

    def _library_summaries(self) -> list[LibraryCardSummary]:
        """Read `alpha_cards_library/index.jsonl` and emit a compact list
        of factors already validated by prior campaign runs.

        The Editor sees this section and is asked to propose *complementary*
        factors, not duplicates. When `library_dir` is unset (e.g.
        single-Discovery runs not driven by CampaignRunner), returns an
        empty list -- no injection, behaviour unchanged.
        """
        if self.library_dir is None:
            return []
        index_path = self.library_dir / "index.jsonl"
        if not index_path.exists():
            return []
        import json as _json

        out: list[LibraryCardSummary] = []
        seen_alpha_ids: set[str] = set()
        for line in index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            alpha_id = entry.get("alpha_id")
            if not alpha_id or alpha_id in seen_alpha_ids:
                continue
            seen_alpha_ids.add(alpha_id)

            # Best-effort: pull incremental_sharpe_point off the card YAML.
            point = None
            card_rel = entry.get("card_path")
            if card_rel:
                card_path = self.library_dir / card_rel
                if card_path.exists():
                    try:
                        card_raw = yaml.safe_load(card_path.read_text(encoding="utf-8"))
                        sealed = (card_raw or {}).get("evidence", {}).get("sealed_summary") or {}
                        if "incremental_sharpe_point" in sealed:
                            point = float(sealed["incremental_sharpe_point"])
                    except (yaml.YAMLError, OSError, ValueError):
                        point = None
            out.append(
                LibraryCardSummary(
                    alpha_id=alpha_id,
                    fn=str(entry.get("fn", "")),
                    indicator_name=str(entry.get("indicator", "")),
                    dossier_factor=entry.get("dossier_factor"),
                    incremental_sharpe_point=point,
                )
            )
        return out

    def _banned_indicator_fns(self, strategy: Strategy) -> tuple[str, ...]:
        """Indicator fn names that must NOT be proposed again via add_indicator.

        Union of three sources:
          1. fns currently wired in `strategy.indicators` -- can't add what's
             already there; Editor must tune via parameter_change.
          2. fns from `memory/tried_factors.jsonl` (accepted OR rejected).
             Once tried in this campaign, the same fn doesn't add coverage.
          3. (implicitly, via #1) fns auto-injected from the alpha-card
             library at campaign start -- they're already in the strategy.

        Pushing the Editor toward fresh fn names is what makes the library
        *grow* rather than converge on one factor.
        """
        banned: list[str] = []
        seen: set[str] = set()
        # Source 1: current strategy.
        for spec in strategy.indicators:
            if spec.fn and spec.fn not in seen:
                banned.append(spec.fn)
                seen.add(spec.fn)
        # Source 2: cross-iter tried_factors.jsonl.
        for rec in self.memory.read_tried_factors():
            fn = rec.indicator_fn
            if not fn or fn in seen:
                continue
            banned.append(fn)
            seen.add(fn)
        return tuple(banned)

    def _tried_factor_names(
        self,
        strategy: Strategy,
        recent: list[PastTrialSummary],
    ) -> set[str]:
        """Factor names already explored. A dossier name counts as tried if:

          * any past trial's `cited_factors` lists it exactly (within iter), or
          * any entry in `memory/tried_factors.jsonl` lists it
            (cross-iteration -- survives CampaignRunner._reset_for_iteration), or
          * any current strategy indicator's `name` or `fn` contains the
            dossier name as a case-insensitive substring (e.g. `adx_14`
            counts as ADX tried; `sma_fast` does NOT count -- min length 3).

        The three signals are unioned; the jsonl is the dedup safety net
        across iterations because the in-process `recent` list is event-
        memory-bound and gets wiped on iter reset. Returns dossier
        `factor_name` values case-preserved so the FactorHint output
        matches the index.
        """
        from lbg.knowledge.factors import load_index

        index = load_index()
        if not index:
            return set()
        dossier_names = {(e.get("factor_name") or "").strip(): None for e in index}
        dossier_names = {k: None for k in dossier_names if k and len(k) >= 3}

        tried: set[str] = set()
        # Authoritative source #1: explicit citations on past trials in this iter.
        for t in recent:
            for cite in t.cited_factors:
                if cite in dossier_names:
                    tried.add(cite)
        # Authoritative source #2: cross-iteration jsonl (persists across resets).
        for rec in self.memory.read_tried_factors():
            for f in rec.factors:
                if f in dossier_names:
                    tried.add(f)
        # Safety net: indicator names / fns that contain a dossier name.
        indicator_strings = [s.name.lower() for s in strategy.indicators] + [
            s.fn.lower() for s in strategy.indicators
        ]
        for real_name in dossier_names:
            low_name = real_name.lower()
            for h in indicator_strings:
                if low_name in h:
                    tried.add(real_name)
                    break
        return tried


def _looks_like_buyhold(strategy: Strategy) -> bool:
    name = strategy.name.lower()
    if "buyhold" in name or "buy_and_hold" in name:
        return True
    fns = {s.fn for s in strategy.indicators}
    return {"step_in", "zero_baseline"} <= fns


def _source_excerpt(source: str, max_chars: int = 1600) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _truncate(_scrub_years(source), max_chars)

    lines = source.splitlines()
    chunks: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.Import | ast.ImportFrom | ast.FunctionDef):
            continue
        end = getattr(node, "end_lineno", node.lineno)
        chunks.extend(lines[node.lineno - 1 : end])
        chunks.append("")
    excerpt = "\n".join(chunks).strip() or source
    return _truncate(_scrub_years(excerpt), max_chars)


def _scrub_years(text: str) -> str:
    """Replace [1990, 2050] year tokens with `<YEAR>` so dossier text can pass
    the Editor's outgoing-prompt redaction guard. Years outside that range
    (e.g. 1978 in the ADX dossier) are left intact, which matches what the
    redaction guard would have allowed anyway."""
    return _REDACT_YEAR_RE.sub("<YEAR>", text)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _summarize(record: TrialRecord) -> PastTrialSummary:
    return PastTrialSummary(
        trial_id=record.trial_id,
        edit_type=record.edit.type.value,
        edit_summary=record.edit.summary,
        hypothesis_text=record.hypothesis.text,
        expected_validation_signal=record.hypothesis.expected_validation_signal.value,
        actual_validation_signal=record.validation_signal.value,
        hypothesis_outcome=record.hypothesis_outcome.value,
        train_sharpe=record.train_metrics.sharpe,
        train_max_drawdown=record.train_metrics.max_drawdown,
        train_turnover=record.train_metrics.turnover,
        train_num_trades=record.train_metrics.num_trades,
        decision=record.decision.value,
        fallback_if_rejected=(record.fallback_if_rejected or "").strip(),
        cited_factors=tuple(record.cited_factors or ()),
    )


__all__ = [
    "ContextBuilder",
    "EditorContext",
    "FactorHint",
    "IndicatorCodeSummary",
    "HypothesisOutcome",
    "LibraryCardSummary",
    "PastTrialSummary",
    "SEMANTIC_MEMORY_FILES",
    "SeedTemplateHint",
    "ValidationSignal",
]
