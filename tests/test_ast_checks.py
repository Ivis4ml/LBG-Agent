"""Tests for `lbg.invariants.ast_checks`.

Each invariant has a paired fixture that violates only it (or it plus one
neighbor for the multi-violation test). The good `indicators/sma.py` should
pass every check.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lbg.invariants import (
    ALL_AST_CHECKS,
    InvariantViolation,
    run_ast_checks,
)
from lbg.invariants.ast_checks import (
    check_no_forward_fill_future,
    check_no_future_shift,
    check_no_negative_indexing,
    check_pure_function,
    check_whitelisted_imports_only,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "indicators"
GOOD_SMA = REPO_ROOT / "indicators" / "sma.py"


def _violation_names(vs: list[InvariantViolation]) -> list[str]:
    return [v.name for v in vs]


# -------- good indicator passes everything --------


def test_good_sma_passes_all_ast_checks():
    violations = run_ast_checks(GOOD_SMA)
    assert violations == [], violations


def test_all_checks_are_registered():
    """ALL_AST_CHECKS must cover the five names listed in PROPOSAL.html §6."""
    expected = {
        check_no_future_shift,
        check_no_negative_indexing,
        check_no_forward_fill_future,
        check_whitelisted_imports_only,
        check_pure_function,
    }
    assert set(ALL_AST_CHECKS) == expected


# -------- one violation per fixture --------


def test_catches_no_future_shift():
    violations = run_ast_checks(FIXTURES / "bad_future_shift.py")
    assert "no_future_shift" in _violation_names(violations)


def test_catches_no_negative_indexing():
    violations = run_ast_checks(FIXTURES / "bad_negative_indexing.py")
    assert "no_negative_indexing" in _violation_names(violations)


def test_catches_no_forward_fill_future_bfill_and_fillna_method():
    violations = run_ast_checks(FIXTURES / "bad_forward_fill.py")
    names = _violation_names(violations)
    assert names.count("no_forward_fill_future") == 2, violations


def test_catches_whitelisted_imports_only():
    violations = run_ast_checks(FIXTURES / "bad_import.py")
    assert "whitelisted_imports_only" in _violation_names(violations)


def test_catches_pure_function_globals_print_and_to_csv():
    violations = run_ast_checks(FIXTURES / "bad_pure_function.py")
    names = _violation_names(violations)
    # We expect three distinct triggers: global declaration, print, to_csv.
    assert names.count("pure_function") >= 3, violations


# -------- a single file with two violations --------


def test_reports_multiple_violations_independently():
    violations = run_ast_checks(FIXTURES / "bad_multi_violation.py")
    names = set(_violation_names(violations))
    assert "no_future_shift" in names
    assert "whitelisted_imports_only" in names


# -------- false-positive guards on the good SMA --------


@pytest.mark.parametrize("check", ALL_AST_CHECKS)
def test_each_check_yields_no_violation_on_good_sma(check):
    import ast

    tree = ast.parse(GOOD_SMA.read_text())
    assert check(tree, str(GOOD_SMA)) == []


# -------- precise location reporting --------


def test_violation_records_carry_file_and_line():
    violations = run_ast_checks(FIXTURES / "bad_future_shift.py")
    v = next(v for v in violations if v.name == "no_future_shift")
    assert v.file.endswith("bad_future_shift.py")
    assert v.line >= 1
    assert v.message  # non-empty


# -------- shift(+N) is allowed; only negative literals are forbidden --------


def test_positive_shift_does_not_trigger_no_future_shift():
    source = (
        "import pandas as pd\n"
        "def f(df):\n"
        "    return df['close'].shift(1)  # shifts past into present, allowed\n"
    )
    import ast

    tree = ast.parse(source)
    assert check_no_future_shift(tree, "<inline>") == []


# -------- `iloc[i - 1]` is allowed (looks back); only `+ positive_int` is forbidden --------


def test_iloc_minus_one_does_not_trigger_no_negative_indexing():
    source = (
        "def f(df):\n"
        "    n = len(df)\n"
        "    return [df.iloc[i - 1] if i > 0 else None for i in range(n)]\n"
    )
    import ast

    tree = ast.parse(source)
    assert check_no_negative_indexing(tree, "<inline>") == []
