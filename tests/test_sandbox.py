"""Tests for `lbg.sandbox`.

Covers: loading a real indicator, discovery of a unique public function,
ambiguity errors, blocked imports, runtime NameError for stripped builtins,
and SIGALRM-based timeout enforcement.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lbg.sandbox import (
    SandboxForbiddenImport,
    SandboxTimeout,
    load_indicator,
    run_indicator,
)
from lbg.sandbox.executor import SandboxLoadError

REPO = Path(__file__).resolve().parents[1]
GOOD_SMA = REPO / "indicators" / "sma.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "indicators"


def _synthetic_ohlcv(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, size=n).astype("int64"),
        }
    )


# -------- loading the good indicator --------


def test_loads_sma_and_runs():
    fn = load_indicator(GOOD_SMA)
    assert callable(fn) and fn.__name__ == "sma"
    df = _synthetic_ohlcv(n=60)
    out = run_indicator(fn, df, params={"period": 10}, timeout_sec=2.0)
    assert isinstance(out, pd.Series)
    assert len(out) == len(df)
    # First 9 entries should be NaN; the 10th onward, finite.
    assert out.iloc[:9].isna().all()
    assert out.iloc[10:].notna().all()


# -------- import whitelist enforcement --------


def test_module_with_forbidden_import_is_rejected_at_load():
    with pytest.raises(SandboxForbiddenImport, match="forbidden"):
        load_indicator(FIXTURES / "bad_import.py")


def test_runtime_import_inside_function_is_also_blocked():
    """`__import__` is hooked, so even runtime `import` inside an indicator fails."""
    src = (
        "import pandas as pd\n"
        "\n"
        "def tries_runtime_import(df):\n"
        "    import os  # raises at call time\n"
        "    return df['close']\n"
    )
    path = FIXTURES / "_tmp_runtime_import.py"
    path.write_text(src)
    try:
        fn = load_indicator(path)
        with pytest.raises(SandboxForbiddenImport):
            run_indicator(fn, _synthetic_ohlcv())
    finally:
        path.unlink(missing_ok=True)


# -------- stripped builtins (defense in depth on top of pure_function) --------


def test_runtime_print_raises_name_error():
    fn = load_indicator(FIXTURES / "bad_pure_function.py")
    with pytest.raises(NameError, match="print"):
        run_indicator(fn, _synthetic_ohlcv(), timeout_sec=2.0)


def test_runtime_eval_raises_name_error():
    src = "import pandas as pd\n\ndef calls_eval(df):\n    eval('1+1')\n    return df['close']\n"
    path = FIXTURES / "_tmp_eval.py"
    path.write_text(src)
    try:
        fn = load_indicator(path)
        with pytest.raises(NameError, match="eval"):
            run_indicator(fn, _synthetic_ohlcv())
    finally:
        path.unlink(missing_ok=True)


# -------- discovery --------


def test_multi_function_module_requires_explicit_name():
    with pytest.raises(SandboxLoadError, match="multiple public functions"):
        load_indicator(FIXTURES / "multi_fn.py")


def test_multi_function_module_with_explicit_name_works():
    fn = load_indicator(FIXTURES / "multi_fn.py", function_name="fn_b")
    df = _synthetic_ohlcv(n=10)
    out = run_indicator(fn, df, timeout_sec=2.0)
    assert (out == df["close"] * 3.0).all()


def test_function_name_not_found_raises():
    with pytest.raises(SandboxLoadError, match="not found"):
        load_indicator(FIXTURES / "multi_fn.py", function_name="nope")


def test_missing_file_raises():
    with pytest.raises(SandboxLoadError, match="not found"):
        load_indicator(REPO / "indicators" / "does_not_exist.py")


def test_function_module_local_helpers_are_ignored():
    """A private `_helper` does not count as the indicator; the public fn wins."""
    src = (
        "import pandas as pd\n"
        "\n"
        "def _helper(x):\n"
        "    return x + 1\n"
        "\n"
        "def public_one(df):\n"
        "    return df['close']\n"
    )
    path = FIXTURES / "_tmp_private.py"
    path.write_text(src)
    try:
        fn = load_indicator(path)
        assert fn.__name__ == "public_one"
    finally:
        path.unlink(missing_ok=True)


# -------- timeout --------


def test_run_indicator_times_out_on_slow_fn():
    df = _synthetic_ohlcv()

    def slow(d: pd.DataFrame) -> pd.Series:
        time.sleep(1.5)
        return d["close"]

    t0 = time.monotonic()
    with pytest.raises(SandboxTimeout, match="exceeded"):
        run_indicator(slow, df, timeout_sec=0.3)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0, f"timeout did not fire promptly (took {elapsed:.2f}s)"


def test_run_indicator_rejects_non_positive_timeout():
    df = _synthetic_ohlcv()
    with pytest.raises(ValueError, match="timeout_sec must be > 0"):
        run_indicator(lambda d: d["close"], df, timeout_sec=0)
    with pytest.raises(ValueError, match="timeout_sec must be > 0"):
        run_indicator(lambda d: d["close"], df, timeout_sec=-1.0)


def test_run_indicator_clears_signal_handler_after_success():
    """Subsequent code should not see a stale SIGALRM raising into it."""
    import signal as _signal

    df = _synthetic_ohlcv()
    run_indicator(lambda d: d["close"], df, timeout_sec=1.0)
    # If the handler was left installed and the itimer not cleared, the
    # process would error here. A quick sanity check: getitimer reads zero.
    seconds, _ = _signal.getitimer(_signal.ITIMER_REAL)
    assert seconds == 0.0
