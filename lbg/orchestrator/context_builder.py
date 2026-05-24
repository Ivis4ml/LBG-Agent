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

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from lbg.dsl.schema import Strategy
from lbg.memory.manager import SEMANTIC_MEMORY_FILES, MemoryManager
from lbg.schemas import (
    HypothesisOutcome,
    TrialRecord,
    ValidationSignal,
)
from lbg.skills import Skill, SkillManager


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


@dataclass(frozen=True)
class EditorContext:
    strategy_yaml: str  # rendered DSL document (already cross-checked for redaction)
    recent_trials: list[PastTrialSummary] = field(default_factory=list)
    semantic_memory: dict[str, str] = field(default_factory=dict)  # filename -> content
    skills: list[Skill] = field(default_factory=list)  # active skills from SkillManager


class ContextBuilder:
    def __init__(
        self,
        memory: MemoryManager,
        *,
        repo_root: str | Path = ".",
        recent_trials_limit: int = 10,
        skills: SkillManager | None = None,
    ) -> None:
        self.memory = memory
        self.repo_root = Path(repo_root)
        self.recent_trials_limit = recent_trials_limit
        self.skills = skills or SkillManager(self.repo_root / "skills")

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

        return EditorContext(
            strategy_yaml=strategy_yaml,
            recent_trials=recent,
            semantic_memory=semantic,
            skills=active_skills,
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
    )


__all__ = [
    "ContextBuilder",
    "EditorContext",
    "HypothesisOutcome",
    "PastTrialSummary",
    "SEMANTIC_MEMORY_FILES",
    "ValidationSignal",
]
