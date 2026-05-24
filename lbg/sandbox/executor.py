"""Restricted-namespace loader and timeout-bounded runner for indicators."""

from __future__ import annotations

import builtins as _real_builtins
import inspect
import signal
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

WHITELISTED_IMPORTS: frozenset[str] = frozenset({"numpy", "pandas", "math"})


class SandboxError(Exception):
    """Base class for sandbox failures."""


class SandboxForbiddenImport(SandboxError):
    """Raised when the sandboxed module tries to import outside the whitelist."""


class SandboxTimeout(SandboxError):
    """Raised when the indicator exceeds its wall-clock budget."""


class SandboxLoadError(SandboxError):
    """Raised when the indicator module fails to load (syntax, missing fn, ...)."""


def _restricted_import(
    name: str,
    globals_: dict | None = None,
    locals_: dict | None = None,
    fromlist: tuple = (),
    level: int = 0,
):
    top = name.split(".")[0] if name else ""
    if top not in WHITELISTED_IMPORTS:
        raise SandboxForbiddenImport(
            f"import of {name!r} forbidden by sandbox; "
            f"allowed top-level modules: {sorted(WHITELISTED_IMPORTS)}"
        )
    return _real_builtins.__import__(name, globals_, locals_, fromlist, level)


# Builtins judged safe for pure-function indicator code. Notably excluded:
# open, input, exec, eval, compile, globals, locals, vars, breakpoint, help,
# exit, quit, dir, getattr, setattr, delattr, hasattr, __import__, memoryview.
_SAFE_BUILTIN_NAMES: tuple[str, ...] = (
    "abs",
    "all",
    "any",
    "bool",
    "bytes",
    "callable",
    "chr",
    "complex",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "ord",
    "pow",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "type",
    "zip",
    # Exceptions the indicator might need to raise or catch.
    "Exception",
    "ArithmeticError",
    "AssertionError",
    "AttributeError",
    "IndexError",
    "KeyError",
    "NotImplementedError",
    "OverflowError",
    "RuntimeError",
    "StopIteration",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
)


def _make_safe_builtins() -> dict[str, Any]:
    safe: dict[str, Any] = {name: getattr(_real_builtins, name) for name in _SAFE_BUILTIN_NAMES}
    safe.update(
        {
            "None": None,
            "True": True,
            "False": False,
            "NotImplemented": NotImplemented,
            "__import__": _restricted_import,
            # `__build_class__` is needed if the module defines a class; harmless
            # to expose. Without it, even decorator-style classes would fail.
            "__build_class__": _real_builtins.__build_class__,
            # `__name__` lookup inside the module needs `__name__` itself in
            # the module globals (we set it below); no builtin needed.
        }
    )
    return safe


def load_indicator(
    module_path: str | Path,
    *,
    function_name: str | None = None,
) -> Callable[..., pd.Series]:
    """Exec `module_path` in a hardened namespace and return the indicator fn.

    Discovery: if `function_name` is given, that exact name must be defined
    at module level. Otherwise the module must define exactly one public
    function; ambiguity raises `SandboxLoadError`.
    """
    path = Path(module_path)
    if not path.exists():
        raise SandboxLoadError(f"indicator module not found: {path}")

    try:
        source = path.read_text(encoding="utf-8")
    except OSError as e:
        raise SandboxLoadError(f"cannot read {path}: {e}") from e

    try:
        code = compile(source, str(path), "exec")
    except SyntaxError as e:
        raise SandboxLoadError(f"syntax error in {path}: {e}") from e

    sandbox_module_name = f"lbg_sandbox.{path.stem}"
    module_globals: dict[str, Any] = {
        "__builtins__": _make_safe_builtins(),
        "__name__": sandbox_module_name,
        "__file__": str(path),
    }

    try:
        exec(code, module_globals)
    except SandboxForbiddenImport:
        raise
    except Exception as e:  # noqa: BLE001
        raise SandboxLoadError(f"executing {path} raised {type(e).__name__}: {e}") from e

    candidates: dict[str, Callable[..., Any]] = {
        name: obj
        for name, obj in module_globals.items()
        if inspect.isfunction(obj)
        and not name.startswith("_")
        and getattr(obj, "__module__", None) == sandbox_module_name
    }

    if function_name is not None:
        if function_name not in candidates:
            raise SandboxLoadError(
                f"function {function_name!r} not found in {path}; "
                f"public functions: {sorted(candidates)}"
            )
        return candidates[function_name]

    if len(candidates) == 1:
        return next(iter(candidates.values()))
    if not candidates:
        raise SandboxLoadError(f"{path} defines no public function; nothing to call")
    raise SandboxLoadError(
        f"{path} defines multiple public functions {sorted(candidates)}; "
        "pass `function_name=` explicitly"
    )


def run_indicator(
    fn: Callable[..., pd.Series],
    df: pd.DataFrame,
    *,
    params: dict[str, Any] | None = None,
    timeout_sec: float = 10.0,
) -> pd.Series:
    """Call `fn(df, **params)` with a wall-clock timeout.

    The timeout is enforced via `signal.SIGALRM` and `signal.setitimer`, which
    is fine for our single-threaded Orchestrator (PROPOSAL.html §6.5). Callers
    in worker threads need a different mechanism; we'd cross that bridge then.
    """
    if params is None:
        params = {}
    if timeout_sec <= 0:
        raise ValueError(f"timeout_sec must be > 0, got {timeout_sec}")

    def _handler(signum, frame):
        raise SandboxTimeout(f"indicator exceeded {timeout_sec}s wall-clock budget")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_sec)
    try:
        return fn(df, **params)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
