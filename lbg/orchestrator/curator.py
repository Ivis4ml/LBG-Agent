"""Shadow-mode Curator (PROPOSAL.html §6.5 line 1232).

Runs every `cycle_interval` accepted trials. In shadow mode it can only
compress the four semantic-memory documents -- never `strategy.yaml`. The
LLM returns replacement contents for each document; the Curator overwrites
them via `MemoryManager.write_md`.

Each Curator cycle is committed on its own branch (`curator/cycle-NNN`,
PROPOSAL.html §10) so the audit trail records exactly which trials drove
each compression pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from lbg.git_manager import GitManager
from lbg.memory import MemoryManager
from lbg.memory.records import AgentComputeRecord

CURATOR_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "curator_system.md"


class CuratorOutputPayload(BaseModel):
    """What the Curator LLM returns -- replacement contents for each md."""

    model_config = ConfigDict(extra="forbid")

    accepted_rules_md: str = Field(default="")
    failed_directions_md: str = Field(default="")
    open_questions_md: str = Field(default="")
    do_not_repeat_md: str = Field(default="")
    note: str = Field(default="")


@dataclass(frozen=True)
class CuratorRunResult:
    cycle: int
    payload: CuratorOutputPayload
    raw_text: str
    compute: AgentComputeRecord
    commit_sha: str | None  # None when GitManager not provided


class Curator:
    """Wraps the LLM call + memory-write + git-commit for one cycle."""

    def __init__(
        self,
        runner,  # RoleRunner (avoid circular import)
        memory: MemoryManager,
        *,
        git: GitManager | None = None,
        cycle_interval: int = 10,
    ) -> None:
        self.runner = runner
        self.memory = memory
        self.git = git
        self.cycle_interval = cycle_interval

    def should_run(self, accepted_count: int) -> bool:
        """Trigger every `cycle_interval` accepted trials (after the first)."""
        return accepted_count > 0 and accepted_count % self.cycle_interval == 0

    def run(self, *, cycle: int, accepted_count: int) -> CuratorRunResult:
        """Run one Curator pass. Overwrites the four md files in place."""
        import time

        from lbg.orchestrator.role_runner import _extract_text, _extract_yaml_block

        # Read current memory state.
        memory_snapshot = {
            "accepted_rules.md": self.memory.read_md("accepted_rules.md"),
            "failed_directions.md": self.memory.read_md("failed_directions.md"),
            "open_questions.md": self.memory.read_md("open_questions.md"),
            "do_not_repeat.md": self.memory.read_md("do_not_repeat.md"),
        }

        system_prompt = CURATOR_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        user_prompt = _render_curator_user_prompt(memory_snapshot, accepted_count=accepted_count)

        from lbg.orchestrator.redaction import assert_redacted

        assert_redacted(system_prompt)
        assert_redacted(user_prompt)

        client = self.runner._client()
        started = time.monotonic()
        response = client.messages.create(
            model=self.runner.model,
            max_tokens=self.runner.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        elapsed = time.monotonic() - started

        raw_text = _extract_text(response)
        assert_redacted(raw_text)

        yaml_block = _extract_yaml_block(raw_text, role="Curator")
        try:
            data = yaml.safe_load(yaml_block)
        except yaml.YAMLError as e:
            from lbg.orchestrator.role_runner import RoleRunnerError

            raise RoleRunnerError(
                f"Curator output not valid YAML: {e}\n--- raw ---\n{raw_text}"
            ) from e
        if not isinstance(data, dict):
            from lbg.orchestrator.role_runner import RoleRunnerError

            raise RoleRunnerError(f"Curator output is not a mapping; got {type(data).__name__}")
        try:
            payload = CuratorOutputPayload.model_validate(data)
        except ValidationError as e:
            from lbg.orchestrator.role_runner import RoleRunnerError

            raise RoleRunnerError(
                f"Curator output did not parse: {e}\n--- raw ---\n{raw_text}"
            ) from e

        # Overwrite the 4 md files with the compressed versions.
        self.memory.write_md("accepted_rules.md", payload.accepted_rules_md)
        self.memory.write_md("failed_directions.md", payload.failed_directions_md)
        self.memory.write_md("open_questions.md", payload.open_questions_md)
        self.memory.write_md("do_not_repeat.md", payload.do_not_repeat_md)

        commit_sha: str | None = None
        if self.git is not None:
            branch = self.git.curator_branch_name(cycle)
            # Create + switch to the branch only if it doesn't already exist.
            try:
                self.git.create_branch(branch)
            except Exception:
                # Branch may already exist from a prior run; just check out.
                self.git.checkout(branch)
            self.git.stage(
                [
                    self.memory.semantic_memory_path("accepted_rules.md"),
                    self.memory.semantic_memory_path("failed_directions.md"),
                    self.memory.semantic_memory_path("open_questions.md"),
                    self.memory.semantic_memory_path("do_not_repeat.md"),
                ]
            )
            commit_sha = self.git.commit(
                f"curator cycle {cycle:03d}: compress semantic memory",
                allow_empty=True,
            )

        compute = AgentComputeRecord(
            trial_id=accepted_count,  # tag with the accepted-count milestone
            role="curator",
            model=self.runner.model,
            input_tokens=getattr(response.usage, "input_tokens", 0),
            output_tokens=getattr(response.usage, "output_tokens", 0),
            wall_clock_sec=elapsed,
        )
        return CuratorRunResult(
            cycle=cycle, payload=payload, raw_text=raw_text, compute=compute, commit_sha=commit_sha
        )


def _render_curator_user_prompt(memory_snapshot: dict[str, str], *, accepted_count: int) -> str:
    parts: list[str] = [
        f"## Curator cycle at accepted_count={accepted_count}\n",
        "## Current memory contents\n",
    ]
    for name, content in memory_snapshot.items():
        body = content.strip() or "(empty)"
        parts.append(f"### {name}\n\n{body}\n")
    parts.append(
        "## Your turn\n\n"
        "Return one YAML block with replacement contents for all four "
        "documents, plus a short `note` describing the largest compression."
    )
    return "\n".join(parts)
