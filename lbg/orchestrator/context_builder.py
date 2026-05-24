"""ContextBuilder.editor_view -- assemble the Editor's context with redaction.

The Editor sees:
  - The current strategy DSL (`strategy.yaml` content).
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
    ValidationSignal.REJECTED_DRAWDOWN_REGRESSION: ("波动率", "回撤"),
    ValidationSignal.REJECTED_TURNOVER: ("趋势强度", "状态识别"),
    ValidationSignal.REJECTED_NO_SIGNIFICANT_IMPROVEMENT: ("动量", "均值回归"),
    # complexity rejection: do NOT add factors -- the system is already too
    # large. Empty tuple means no rejection-derived hints from this signal.
    ValidationSignal.REJECTED_COMPLEXITY: (),
    ValidationSignal.ACCEPTED: (),
}

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
class EditorContext:
    strategy_yaml: str  # rendered DSL document (already cross-checked for redaction)
    recent_trials: list[PastTrialSummary] = field(default_factory=list)
    semantic_memory: dict[str, str] = field(default_factory=dict)  # filename -> content
    skills: list[Skill] = field(default_factory=list)  # active skills from SkillManager
    factor_hints: list[FactorHint] = field(default_factory=list)


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
    ) -> None:
        self.memory = memory
        self.repo_root = Path(repo_root)
        self.recent_trials_limit = recent_trials_limit
        self.skills = skills or SkillManager(self.repo_root / "skills")
        self.factor_hints_per_term = factor_hints_per_term
        self.factor_hints_total_cap = factor_hints_total_cap
        self.factor_one_line_max_chars = factor_one_line_max_chars

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

        return EditorContext(
            strategy_yaml=strategy_yaml,
            recent_trials=recent,
            semantic_memory=semantic,
            skills=active_skills,
            factor_hints=factor_hints,
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

    def _tried_factor_names(
        self,
        strategy: Strategy,
        recent: list[PastTrialSummary],
    ) -> set[str]:
        """Factor names already explored. A dossier name counts as tried if:

          * any past trial's `cited_factors` lists it exactly, or
          * any current strategy indicator's `name` or `fn` contains the
            dossier name as a case-insensitive substring (e.g. `adx_14`
            counts as ADX tried; `sma_fast` does NOT count -- min length 3).

        Past trial citations are the authoritative signal; the indicator-
        name substring rule is a safety net for sessions where the Editor
        forgot to cite explicitly. Returns dossier `factor_name` values
        case-preserved so the FactorHint output matches the index.
        """
        from lbg.knowledge.factors import load_index

        index = load_index()
        if not index:
            return set()
        dossier_names = {(e.get("factor_name") or "").strip(): None for e in index}
        dossier_names = {k: None for k in dossier_names if k and len(k) >= 3}

        tried: set[str] = set()
        # Authoritative source: explicit citations on past trials.
        for t in recent:
            for cite in t.cited_factors:
                if cite in dossier_names:
                    tried.add(cite)
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
    "HypothesisOutcome",
    "PastTrialSummary",
    "SEMANTIC_MEMORY_FILES",
    "ValidationSignal",
]
