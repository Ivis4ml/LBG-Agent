"""AST-level static invariants for agent-authored indicators (PROPOSAL.html §6).

Five checks, each a pure function over a parsed `ast.Module`:

  - `no_future_shift`          forbids `.shift(<negative literal>)`
  - `no_negative_indexing`     forbids `.iloc[<X> + <positive int literal>]`
  - `no_forward_fill_future`   forbids `.bfill()`, `.backfill()`,
                               `.fillna(method="bfill"|"backfill")`,
                               and `.interpolate(limit_direction="backward"|"both")`
  - `whitelisted_imports_only` only numpy, pandas, math (top-level module)
  - `pure_function`            no `global`/`nonlocal`; no builtin I/O calls;
                               no DataFrame/Series file-write methods

Limitations
-----------
These are *static* checks. They catch named patterns but cannot follow
aliasing or dynamic dispatch. `prefix_stability` (a runtime check) is the
load-bearing defense against subtle leakage and is added in a later step.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

from lbg.invariants.violations import InvariantViolation

WHITELISTED_TOP_LEVEL_MODULES: frozenset[str] = frozenset({"numpy", "pandas", "math"})

FORBIDDEN_BUILTINS: frozenset[str] = frozenset(
    {
        "open",
        "print",
        "input",
        "exec",
        "eval",
        "compile",
        "__import__",
        "breakpoint",
        "globals",
        "locals",
        "vars",
    }
)

# DataFrame / Series methods that touch disk or external state.
FORBIDDEN_IO_METHODS: frozenset[str] = frozenset(
    {
        "to_csv",
        "to_parquet",
        "to_pickle",
        "to_hdf",
        "to_feather",
        "to_json",
        "to_excel",
        "to_sql",
        "to_clipboard",
        "read_csv",
        "read_parquet",
        "read_pickle",
        "read_hdf",
        "read_json",
        "read_excel",
        "read_sql",
    }
)

CheckFn = Callable[[ast.AST, str], list[InvariantViolation]]


def _is_negative_literal(node: ast.AST) -> bool:
    """True if `node` is `-N` or `<negative int/float literal>`."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = node.operand
        return (
            isinstance(operand, ast.Constant)
            and isinstance(operand.value, (int, float))
            and operand.value > 0
        )
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value < 0
    return False


def _shift_periods_arg(call: ast.Call) -> ast.AST | None:
    """Return the AST node passed as `periods` to a `.shift(...)` call."""
    if call.args:
        return call.args[0]
    for kw in call.keywords:
        if kw.arg in ("periods", "n"):
            return kw.value
    return None


def check_no_future_shift(tree: ast.AST, path: str) -> list[InvariantViolation]:
    out: list[InvariantViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "shift":
            continue
        periods = _shift_periods_arg(node)
        if periods is not None and _is_negative_literal(periods):
            out.append(
                InvariantViolation(
                    name="no_future_shift",
                    message=".shift(...) called with a negative literal (pulls future to present)",
                    file=path,
                    line=node.lineno,
                )
            )
    return out


def _binop_has_positive_int_addend(node: ast.AST) -> bool:
    """True if `node` is `<X> + <positive int literal>` or its symmetric form."""
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
        return False
    for side in (node.left, node.right):
        if (
            isinstance(side, ast.Constant)
            and isinstance(side.value, int)
            and not isinstance(side.value, bool)
            and side.value > 0
        ):
            return True
    return False


def _iloc_subscript_indices(slice_node: ast.AST) -> list[ast.AST]:
    """Flatten the slice expression into individual index components.

    Handles `df.iloc[i + 1]` and `df.iloc[i + 1, j]`.
    """
    if isinstance(slice_node, ast.Tuple):
        return list(slice_node.elts)
    return [slice_node]


def check_no_negative_indexing(tree: ast.AST, path: str) -> list[InvariantViolation]:
    """Forbid `df.iloc[i + k]` where `k` is a positive int literal.

    The check name follows PROPOSAL.html §6 verbatim; the intent (per the
    proposal's own gloss) is to forbid relative access to future bars.
    """
    out: list[InvariantViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if not isinstance(node.value, ast.Attribute) or node.value.attr != "iloc":
            continue
        for idx_expr in _iloc_subscript_indices(node.slice):
            if _binop_has_positive_int_addend(idx_expr):
                out.append(
                    InvariantViolation(
                        name="no_negative_indexing",
                        message=".iloc[X + positive_int] accesses a future bar",
                        file=path,
                        line=node.lineno,
                    )
                )
                break
    return out


_BACKWARD_FILL_METHODS = {"bfill", "backfill"}


def check_no_forward_fill_future(tree: ast.AST, path: str) -> list[InvariantViolation]:
    out: list[InvariantViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        attr = node.func.attr
        if attr in _BACKWARD_FILL_METHODS:
            out.append(
                InvariantViolation(
                    name="no_forward_fill_future",
                    message=f".{attr}() carries future values backward into the present",
                    file=path,
                    line=node.lineno,
                )
            )
            continue
        if attr == "fillna":
            for kw in node.keywords:
                if (
                    kw.arg == "method"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value in _BACKWARD_FILL_METHODS
                ):
                    out.append(
                        InvariantViolation(
                            name="no_forward_fill_future",
                            message=(
                                f".fillna(method={kw.value.value!r}) carries future values backward"
                            ),
                            file=path,
                            line=node.lineno,
                        )
                    )
        elif attr == "interpolate":
            for kw in node.keywords:
                if (
                    kw.arg == "limit_direction"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value in ("backward", "both")
                ):
                    out.append(
                        InvariantViolation(
                            name="no_forward_fill_future",
                            message=(
                                f".interpolate(limit_direction={kw.value.value!r}) leaks future"
                            ),
                            file=path,
                            line=node.lineno,
                        )
                    )
    return out


def check_whitelisted_imports_only(tree: ast.AST, path: str) -> list[InvariantViolation]:
    out: list[InvariantViolation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in WHITELISTED_TOP_LEVEL_MODULES:
                    out.append(
                        InvariantViolation(
                            name="whitelisted_imports_only",
                            message=(
                                f"import {alias.name!r} not in whitelist "
                                f"{sorted(WHITELISTED_TOP_LEVEL_MODULES)}"
                            ),
                            file=path,
                            line=node.lineno,
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top not in WHITELISTED_TOP_LEVEL_MODULES:
                out.append(
                    InvariantViolation(
                        name="whitelisted_imports_only",
                        message=(
                            f"from {node.module!r} import ... not in whitelist "
                            f"{sorted(WHITELISTED_TOP_LEVEL_MODULES)}"
                        ),
                        file=path,
                        line=node.lineno,
                    )
                )
    return out


def check_pure_function(tree: ast.AST, path: str) -> list[InvariantViolation]:
    out: list[InvariantViolation] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            kind = type(node).__name__.lower()
            out.append(
                InvariantViolation(
                    name="pure_function",
                    message=f"`{kind}` declaration disallowed (breaks purity)",
                    file=path,
                    line=node.lineno,
                )
            )
            continue
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_BUILTINS:
            out.append(
                InvariantViolation(
                    name="pure_function",
                    message=f"forbidden builtin call `{node.func.id}(...)` (I/O or side effect)",
                    file=path,
                    line=node.lineno,
                )
            )
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_IO_METHODS:
            out.append(
                InvariantViolation(
                    name="pure_function",
                    message=f"forbidden method `.{node.func.attr}(...)` (file or DB I/O)",
                    file=path,
                    line=node.lineno,
                )
            )
    return out


ALL_AST_CHECKS: tuple[CheckFn, ...] = (
    check_no_future_shift,
    check_no_negative_indexing,
    check_no_forward_fill_future,
    check_whitelisted_imports_only,
    check_pure_function,
)


def run_ast_checks(path: str | Path, source: str | None = None) -> list[InvariantViolation]:
    """Run every static check on `path` and return all violations found."""
    path = Path(path)
    if source is None:
        source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    violations: list[InvariantViolation] = []
    for check in ALL_AST_CHECKS:
        violations.extend(check(tree, str(path)))
    return violations
