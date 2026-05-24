"""Orchestrator subpackage: ContextBuilder + RoleRunner + Editor wiring.

PROPOSAL.html §6.5 lists three LLM-facing pieces of the Orchestrator:
`ContextBuilder` (assembles role-specific context, enforces redaction),
`RoleRunner` (invokes the LLM with typed input + demands typed output), and
the three roles themselves. This step wires up Editor; Reflector and
Curator follow in step 11/12.
"""

from lbg.orchestrator.context_builder import (
    ContextBuilder,
    EditorContext,
    PastTrialSummary,
)
from lbg.orchestrator.redaction import (
    RedactionError,
    assert_redacted,
    scan_for_leaks,
)
from lbg.orchestrator.role_runner import (
    PROVIDERS,
    EditorRunResult,
    ProviderConfig,
    ReflectorRunResult,
    RoleRunner,
    RoleRunnerError,
)

__all__ = [
    "PROVIDERS",
    "ContextBuilder",
    "EditorContext",
    "EditorRunResult",
    "PastTrialSummary",
    "ProviderConfig",
    "RedactionError",
    "ReflectorRunResult",
    "RoleRunner",
    "RoleRunnerError",
    "assert_redacted",
    "scan_for_leaks",
]
