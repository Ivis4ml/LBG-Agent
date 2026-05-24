"""Tests for `lbg.dsl.schema` and `lbg.dsl.loader`."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from lbg.dsl import (
    CrossAboveRule,
    FixedFractionSizing,
    IndicatorSpec,
    Strategy,
    load_strategy,
)

REPO = Path(__file__).resolve().parents[1]


def _minimal_strategy_dict() -> dict:
    return {
        "name": "sma_cross_baseline",
        "indicators": [
            {"name": "sma_fast", "fn": "sma", "params": {"period": 20}},
            {"name": "sma_slow", "fn": "sma", "params": {"period": 50}},
        ],
        "entry": {"rule": "cross_above", "fast": "sma_fast", "slow": "sma_slow"},
        "exit": {"rule": "cross_below", "fast": "sma_fast", "slow": "sma_slow"},
        "filters": [],
        "sizing": {"mode": "fixed_fraction", "fraction": 1.0, "max_position": 1.0},
    }


# -------- schema validation --------


def test_minimal_strategy_validates():
    s = Strategy.model_validate(_minimal_strategy_dict())
    assert s.name == "sma_cross_baseline"
    assert len(s.indicators) == 2
    assert isinstance(s.entry, CrossAboveRule)
    assert isinstance(s.sizing, FixedFractionSizing)


def test_duplicate_indicator_names_rejected():
    d = _minimal_strategy_dict()
    d["indicators"][1]["name"] = "sma_fast"
    with pytest.raises(ValidationError, match="duplicate"):
        Strategy.model_validate(d)


def test_unknown_entry_rule_rejected():
    d = _minimal_strategy_dict()
    d["entry"]["rule"] = "make_up_a_signal"
    with pytest.raises(ValidationError):
        Strategy.model_validate(d)


def test_unknown_sizing_mode_rejected():
    d = _minimal_strategy_dict()
    d["sizing"] = {"mode": "kelly", "fraction": 0.5}
    with pytest.raises(ValidationError):
        Strategy.model_validate(d)


def test_extra_top_level_field_rejected():
    d = _minimal_strategy_dict()
    d["leak"] = "calendar_year=2019"
    with pytest.raises(ValidationError):
        Strategy.model_validate(d)


def test_path_separator_in_indicator_name_rejected():
    with pytest.raises(ValidationError, match="path separators"):
        IndicatorSpec.model_validate({"name": "../leak", "fn": "sma"})


def test_fraction_out_of_range_rejected():
    d = _minimal_strategy_dict()
    d["sizing"]["fraction"] = 1.5
    with pytest.raises(ValidationError):
        Strategy.model_validate(d)


# -------- cross-reference validation --------


def _write_yaml(tmp_path: Path, content: dict | str) -> Path:
    import yaml

    p = tmp_path / "strategy.yaml"
    if isinstance(content, dict):
        p.write_text(yaml.safe_dump(content))
    else:
        p.write_text(content)
    return p


def test_load_strategy_accepts_repo_default():
    s = load_strategy(REPO / "strategy.yaml")
    assert s.name == "sma_cross_baseline"


def test_load_strategy_rejects_unknown_indicator_reference(tmp_path):
    d = _minimal_strategy_dict()
    d["entry"]["slow"] = "does_not_exist"
    p = _write_yaml(tmp_path, d)
    with pytest.raises(ValueError, match="unknown indicator reference"):
        load_strategy(p)


def test_load_strategy_rejects_unknown_filter_indicator(tmp_path):
    d = _minimal_strategy_dict()
    d["filters"] = [{"rule": "indicator_above", "indicator": "ghost", "threshold": 0.0}]
    p = _write_yaml(tmp_path, d)
    with pytest.raises(ValueError, match="unknown indicator reference"):
        load_strategy(p)


def test_volatility_target_sizing_validates(tmp_path):
    d = _minimal_strategy_dict()
    d["sizing"] = {
        "mode": "volatility_target",
        "target_vol": 0.12,
        "max_position": 1.0,
        "vol_lookback": 20,
    }
    p = _write_yaml(tmp_path, d)
    s = load_strategy(p)
    assert s.sizing.mode == "volatility_target"
    assert s.sizing.target_vol == 0.12


def test_yaml_typo_in_rule_caught_by_discriminator(tmp_path):
    text = textwrap.dedent(
        """
        name: typo
        indicators:
          - {name: a, fn: sma, params: {}}
          - {name: b, fn: sma, params: {}}
        entry: {rule: crossaove, fast: a, slow: b}
        exit:  {rule: cross_below, fast: a, slow: b}
        filters: []
        sizing: {mode: fixed_fraction, fraction: 1.0, max_position: 1.0}
        """
    )
    p = _write_yaml(tmp_path, text)
    with pytest.raises(ValidationError):
        load_strategy(p)
