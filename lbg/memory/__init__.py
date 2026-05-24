"""MemoryManager: append-only event memory for the Discovery loop.

PROPOSAL.html §8 and §10.2 describe four event streams (`trials.jsonl`,
`reflections.jsonl`, `invariant_failures.jsonl`, `agent_compute.jsonl`) and
four semantic-memory documents (`accepted_rules.md`, `failed_directions.md`,
`open_questions.md`, `do_not_repeat.md`). This module owns the event streams.

Semantic memory documents are updated by the Reflector and Curator and are
implemented in their respective steps.
"""

from lbg.memory.manager import SEMANTIC_MEMORY_FILES, MemoryManager
from lbg.memory.records import (
    AgentComputeRecord,
    InvariantFailureRecord,
    ReflectionRecord,
    ReflectorOutputPayload,
    TriedFactorRecord,
)

__all__ = [
    "SEMANTIC_MEMORY_FILES",
    "AgentComputeRecord",
    "InvariantFailureRecord",
    "MemoryManager",
    "ReflectionRecord",
    "ReflectorOutputPayload",
    "TriedFactorRecord",
]
