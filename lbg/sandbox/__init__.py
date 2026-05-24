"""SandboxExecutor: restricted-namespace execution of indicator modules.

Corresponds to PROPOSAL.html §6.5 (`SandboxExecutor`) and the invariant
`sandbox_execution`. Two responsibilities:

  1. `load_indicator`  — exec the indicator file in a hardened global namespace
     (whitelisted imports only, no `open`/`exec`/`eval`/file I/O builtins).
  2. `run_indicator`   — invoke the resulting callable with a wall-clock
     timeout enforced via `signal.SIGALRM`.

The sandbox is the second line of defense behind the AST `whitelisted_imports_only`
and `pure_function` checks: anything that slipped past static analysis (or
that doesn't show up as a literal in the AST) still has nowhere to go because
the import system and builtins are stripped.
"""

from lbg.sandbox.executor import (
    SandboxError,
    SandboxForbiddenImport,
    SandboxTimeout,
    load_indicator,
    run_indicator,
)

__all__ = [
    "SandboxError",
    "SandboxForbiddenImport",
    "SandboxTimeout",
    "load_indicator",
    "run_indicator",
]
