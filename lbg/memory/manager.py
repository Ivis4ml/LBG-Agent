"""Append-only writer + reader for the four JSONL memory streams."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from lbg.memory.records import (
    AgentComputeRecord,
    InvariantFailureRecord,
    ReflectionRecord,
)
from lbg.schemas import TrialRecord

T = TypeVar("T", bound=BaseModel)


TRIALS_FILE = "trials.jsonl"
REFLECTIONS_FILE = "reflections.jsonl"
INVARIANT_FAILURES_FILE = "invariant_failures.jsonl"
AGENT_COMPUTE_FILE = "agent_compute.jsonl"

# The four semantic-memory documents (PROPOSAL.html §4 / §8). The Reflector
# writes incremental bullets to these; the Curator compresses them every 10
# accepted trials.
SEMANTIC_MEMORY_FILES: tuple[str, ...] = (
    "accepted_rules.md",
    "failed_directions.md",
    "open_questions.md",
    "do_not_repeat.md",
)


class MemoryManager:
    """Owns the four append-only JSONL streams under `memory/`.

    Append-only contract:
      - Existing lines must never be modified or deleted.
      - Writes are atomic per record (single `f.write(...)` of a one-line JSON
        document followed by `"\\n"`).
      - Trial records are written *after* all per-trial computation finishes,
        so a crash mid-trial leaves no half-written line. Invariant failures
        are recorded the moment they're detected.
    """

    def __init__(self, memory_dir: str | Path = "memory") -> None:
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)

    # ---- paths ----

    @property
    def trials_path(self) -> Path:
        return self.memory_dir / TRIALS_FILE

    @property
    def reflections_path(self) -> Path:
        return self.memory_dir / REFLECTIONS_FILE

    @property
    def invariant_failures_path(self) -> Path:
        return self.memory_dir / INVARIANT_FAILURES_FILE

    @property
    def agent_compute_path(self) -> Path:
        return self.memory_dir / AGENT_COMPUTE_FILE

    # ---- generic append + read ----

    @staticmethod
    def _append_one(path: Path, record: BaseModel) -> None:
        line = record.model_dump_json()
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")

    @staticmethod
    def _read_all(path: Path, model: type[T]) -> list[T]:
        if not path.exists():
            return []
        out: list[T] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    out.append(model.model_validate_json(stripped))
        return out

    # ---- trials ----

    def append_trial(self, record: TrialRecord) -> None:
        self._append_one(self.trials_path, record)

    def read_trials(self) -> list[TrialRecord]:
        return self._read_all(self.trials_path, TrialRecord)

    def next_trial_id(self) -> int:
        records = self.read_trials()
        if not records:
            return 0
        return max(r.trial_id for r in records) + 1

    # ---- reflections ----

    def append_reflection(self, record: ReflectionRecord) -> None:
        self._append_one(self.reflections_path, record)

    def read_reflections(self) -> list[ReflectionRecord]:
        return self._read_all(self.reflections_path, ReflectionRecord)

    # ---- invariant failures ----

    def append_invariant_failure(self, record: InvariantFailureRecord) -> None:
        self._append_one(self.invariant_failures_path, record)

    def read_invariant_failures(self) -> list[InvariantFailureRecord]:
        return self._read_all(self.invariant_failures_path, InvariantFailureRecord)

    # ---- agent compute ----

    def append_agent_compute(self, record: AgentComputeRecord) -> None:
        self._append_one(self.agent_compute_path, record)

    def read_agent_compute(self) -> list[AgentComputeRecord]:
        return self._read_all(self.agent_compute_path, AgentComputeRecord)

    # ---- semantic memory (markdown) ----

    def semantic_memory_path(self, filename: str) -> Path:
        if filename not in SEMANTIC_MEMORY_FILES:
            raise ValueError(
                f"unknown semantic memory file: {filename!r}; "
                f"allowed: {list(SEMANTIC_MEMORY_FILES)}"
            )
        return self.memory_dir / filename

    def append_to_md(self, filename: str, items: list[str]) -> int:
        """Append each `item` as a markdown bullet to the named memory file.

        Empty/whitespace-only items are skipped. Returns the number actually
        written. Each bullet is prefixed with `- ` and given a trailing newline.
        """
        path = self.semantic_memory_path(filename)
        cleaned = [s.strip() for s in items if s and s.strip()]
        if not cleaned:
            return 0
        with path.open("a", encoding="utf-8") as f:
            for item in cleaned:
                f.write(f"- {item}\n")
        return len(cleaned)

    def read_md(self, filename: str) -> str:
        """Read the named semantic memory file; returns empty string if absent."""
        path = self.semantic_memory_path(filename)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def write_md(self, filename: str, content: str) -> None:
        """Replace the entire contents of a semantic memory file.

        Only the Curator should call this -- it owns memory compression in
        shadow mode (PROPOSAL.html §6.5). The Reflector uses `append_to_md`.
        """
        path = self.semantic_memory_path(filename)
        if not content.endswith("\n") and content:
            content = content + "\n"
        path.write_text(content, encoding="utf-8")
