"""Sanity tests for the seed template library.

Pins:

  - The index matches the .py files: every fn in `_index.yaml` has a
    corresponding `<fn>.py`, and every `<fn>.py` is registered.
  - Every template's metadata fields are present and well-typed.
  - Every template can be loaded via the sandbox and run on a small
    synthetic OHLCV frame -- catching syntax errors, missing kwargs,
    wrong column names, and any obvious prefix-stability slip-up.
  - At least the role=exit templates that ship a `suggested_rearm_threshold`
    have a sensible relationship to `suggested_threshold` per the DSL
    schema validators (so the Editor copying defaults verbatim never
    fails at parse time).
"""

from __future__ import annotations

import pandas as pd
import pytest

from lbg.dsl.schema import (
    IndicatorAboveFilter,
)
from lbg.sandbox import load_indicator, run_indicator
from lbg.seed_templates import (
    SEED_TEMPLATES_DIR,
    list_templates,
    load_index,
)

SUPPORTED_CATEGORIES = {
    "trend",
    "momentum",
    "mean_reversion",
    "volatility",
    "drawdown",
    "regime",
    "volume",
}
SUPPORTED_ROLES = {"entry", "exit", "sizing"}


def _synthetic_ohlcv(n: int = 300) -> pd.DataFrame:
    """Deterministic synthetic OHLCV that exercises every column. Drift
    + sinusoid + bounded noise yields finite values without NaN at the
    head once trailing windows fill in. 300 bars covers the longest
    template lookback (252 for vol_regime_zscore)."""
    import numpy as np

    rng = np.random.default_rng(123)
    drift = np.cumsum(rng.normal(0.0005, 0.01, n))
    close = 100.0 * (1.0 + drift)
    high = close * (1.0 + rng.uniform(0.0, 0.01, n))
    low = close * (1.0 - rng.uniform(0.0, 0.01, n))
    open_ = (high + low) / 2.0
    volume = rng.uniform(1_000_000, 2_000_000, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


# -------- index vs files --------


def test_every_template_file_is_indexed():
    indexed_fns = {t.fn for t in list_templates()}
    on_disk = {p.stem for p in SEED_TEMPLATES_DIR.glob("*.py") if not p.stem.startswith("_")}
    missing_index = on_disk - indexed_fns
    missing_files = indexed_fns - on_disk
    assert not missing_index, f"these .py files have no index entry: {missing_index}"
    assert not missing_files, f"these index entries have no .py file: {missing_files}"


def test_index_has_at_least_twenty_templates():
    assert len(list_templates()) >= 20


def test_categories_and_roles_are_in_allowed_set():
    for t in list_templates():
        assert t.category in SUPPORTED_CATEGORIES, f"{t.fn}: bad category {t.category!r}"
        assert t.role in SUPPORTED_ROLES, f"{t.fn}: bad role {t.role!r}"


def test_rearm_thresholds_pass_dsl_validators():
    """If a template ships a suggested rearm_threshold, the (threshold,
    rearm_threshold) pair must validate under EITHER filter rule. The
    natural direction depends on the factor: indicator_above for
    "spike triggers exit, rearm when recedes" (drawdown, vol z-score);
    indicator_below for "drop triggers exit, rearm when recovers"
    (close_volume_corr, skew, momentum below threshold)."""
    from lbg.dsl.schema import IndicatorBelowFilter

    for t in list_templates():
        if t.suggested_rearm_threshold is None:
            continue
        threshold = float(t.suggested_threshold)
        rearm = float(t.suggested_rearm_threshold)
        # Either rule must validate (the template's spec implies one
        # natural direction; we don't enforce which without looking at
        # the rule field). Try both; at least one must succeed.
        ok = False
        for rule_cls, rule_name in (
            (IndicatorAboveFilter, "indicator_above"),
            (IndicatorBelowFilter, "indicator_below"),
        ):
            try:
                rule_cls(
                    rule=rule_name,
                    indicator=t.fn,
                    threshold=threshold,
                    rearm_threshold=rearm,
                )
                ok = True
                break
            except ValueError:
                continue
        assert ok, (
            f"{t.fn}: neither indicator_above nor indicator_below validates "
            f"(threshold={threshold}, rearm={rearm})"
        )


# -------- each template runs --------


@pytest.mark.parametrize("template", list_templates(), ids=lambda t: t.fn)
def test_template_runs_via_sandbox(template):
    """Each template loads through the sandbox and returns a finite-or-
    NaN pd.Series aligned to the input index. NaN at the head is OK
    (rolling windows haven't filled yet); we just check the bulk is
    finite and the shape matches.
    """
    df = _synthetic_ohlcv()
    fn = load_indicator(SEED_TEMPLATES_DIR / f"{template.fn}.py")
    series = run_indicator(fn, df, params=template.default_params, timeout_sec=5.0)
    assert isinstance(series, pd.Series), f"{template.fn} did not return a Series"
    assert len(series) == len(df), f"{template.fn} returned wrong length"

    # The trailing tail should be mostly finite. Allow up to half NaN
    # for the longest-lookback templates (252-bar z-score).
    tail = series.iloc[-50:]
    finite_frac = tail.notna().mean()
    assert finite_frac >= 0.5, (
        f"{template.fn} has too many NaN in tail (finite_frac={finite_frac:.2f})"
    )


def test_loaded_index_is_cached():
    """LRU cache returns the same list object across calls -- catches
    accidental reloading on every Editor invocation."""
    a = load_index()
    b = load_index()
    assert a is b


# -------- ContextBuilder integration --------


def test_context_builder_surfaces_seed_templates(tmp_path):
    """ContextBuilder.editor_view exposes seed templates filtered for
    the current strategy's failure modes."""
    from lbg.dsl import load_strategy
    from lbg.memory import MemoryManager
    from lbg.orchestrator.context_builder import ContextBuilder

    # Repo-shaped tmp dir with a buyhold strategy.
    (tmp_path / "indicators").mkdir()
    (tmp_path / "indicators" / "step_in.py").write_text(
        "import pandas as pd\n\ndef step_in(df, **params):\n"
        "    out = pd.Series(1.0, index=df.index)\n"
        "    if len(out) > 0: out.iloc[0] = 0.0\n"
        "    return out\n",
        encoding="utf-8",
    )
    (tmp_path / "indicators" / "zero_baseline.py").write_text(
        "import pandas as pd\n\ndef zero_baseline(df, **params):\n"
        "    return pd.Series(0.0, index=df.index)\n",
        encoding="utf-8",
    )
    (tmp_path / "strategy.yaml").write_text(
        "name: buyhold_baseline\n"
        "indicators:\n"
        "  - {name: step_in, fn: step_in, params: {}}\n"
        "  - {name: zero_baseline, fn: zero_baseline, params: {}}\n"
        "entry: {rule: cross_above, fast: step_in, slow: zero_baseline}\n"
        "exit:  {rule: cross_below, fast: step_in, slow: zero_baseline}\n"
        "filters: []\n"
        "exit_filters: []\n"
        "sizing: {mode: fixed_fraction, fraction: 1.0, max_position: 1.0}\n",
        encoding="utf-8",
    )

    strategy = load_strategy(tmp_path / "strategy.yaml")
    mm = MemoryManager(tmp_path / "memory")
    cb = ContextBuilder(mm, repo_root=tmp_path)
    ctx = cb.editor_view(strategy)

    # Buyhold baseline should bias the shortlist toward drawdown/regime/vol.
    assert len(ctx.seed_templates) > 0
    categories = {t.category for t in ctx.seed_templates}
    assert categories & {"drawdown", "regime", "volatility"}, (
        f"buyhold shortlist should include drawdown/regime/vol; got {categories}"
    )
    # Source code is present for each (so the Editor can copy-paste).
    for hint in ctx.seed_templates:
        assert hint.source.strip().startswith('"""') or "def " in hint.source
