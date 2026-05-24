"""Common violation record for all invariant checks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InvariantViolation:
    """One violation produced by an invariant check.

    `name` is the invariant identifier from PROPOSAL.html §6 (e.g.
    `no_future_shift`). `file` and `line` locate the offending source
    location; `message` is a short human-readable explanation.
    """

    name: str
    message: str
    file: str
    line: int
