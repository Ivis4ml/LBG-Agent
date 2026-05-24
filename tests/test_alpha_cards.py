"""Tests for `lbg.alpha_cards`.

Schema-level: extra=forbid catches typos; round-trip is stable.
Writer-level: a card lands at trial_NNNN.yaml and is indexed in
alpha_cards/index.jsonl. Dossier matcher returns None when no factor name
substring-matches the indicator fn.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from lbg.alpha_cards import (
    AlphaCard,
    AlphaCardDossierLink,
    AlphaCardEvidence,
    AlphaCardSignal,
    AlphaCardStatus,
    AlphaCardWriter,
    match_dossier_by_name,
)
from lbg.dsl import load_strategy
from lbg.dsl.schema import IndicatorSpec, Strategy
from lbg.schemas import HypothesisOutcome, TrainMetrics, ValidationSignal

REPO = Path(__file__).resolve().parents[1]


def _train_metrics() -> TrainMetrics:
    return TrainMetrics(sharpe=0.81, max_drawdown=-0.12, turnover=0.4, num_trades=33)


def _build_card() -> AlphaCard:
    return AlphaCard(
        alpha_id="my_factor_trial_0007",
        source_trial=7,
        source_commit="abc1234",
        status=AlphaCardStatus.ACCEPTED,
        signal=AlphaCardSignal(
            indicator="my_factor",
            fn="my_factor",
            params={"period": 20},
            source_path="indicators/my_factor.py",
        ),
        evidence=AlphaCardEvidence(
            train_summary={
                "sharpe": 0.81,
                "max_drawdown": -0.12,
                "turnover": 0.4,
                "num_trades": 33,
            },
            validation_signal=ValidationSignal.ACCEPTED,
            hypothesis_outcome=HypothesisOutcome.CONFIRMED,
        ),
    )


# -------- schema --------


def test_extra_field_forbidden_on_card():
    with pytest.raises(ValidationError):
        AlphaCard(
            alpha_id="x",
            source_trial=0,
            source_commit="abc",
            signal=AlphaCardSignal(indicator="x", fn="x", source_path="indicators/x.py"),
            evidence=AlphaCardEvidence(
                train_summary={"sharpe": 0.0},
                validation_signal=ValidationSignal.ACCEPTED,
                hypothesis_outcome=HypothesisOutcome.CONFIRMED,
            ),
            something_extra="boom",  # type: ignore[call-arg]
        )


def test_extra_field_forbidden_on_signal():
    with pytest.raises(ValidationError):
        AlphaCardSignal(
            indicator="x",
            fn="x",
            source_path="indicators/x.py",
            extra="boom",  # type: ignore[call-arg]
        )


def test_sealed_summary_default_is_none():
    card = _build_card()
    assert card.evidence.sealed_summary is None


def test_status_default_is_accepted():
    card = _build_card()
    assert card.status == AlphaCardStatus.ACCEPTED


def test_round_trip_through_yaml(tmp_path):
    card = _build_card()
    dump = yaml.safe_dump(card.model_dump(mode="json"))
    reloaded = AlphaCard.model_validate(yaml.safe_load(dump))
    assert reloaded == card


# -------- writer --------


def test_writer_creates_card_and_index(tmp_path):
    writer = AlphaCardWriter(tmp_path)
    card = _build_card()
    path = writer.write(card)
    assert path == tmp_path / "alpha_cards" / "trial_0007.yaml"
    assert path.exists()
    # Card content is valid YAML and re-validates.
    reloaded = AlphaCard.model_validate(yaml.safe_load(path.read_text()))
    assert reloaded.alpha_id == "my_factor_trial_0007"
    # Index has one line whose card_path points at the file.
    idx_path = tmp_path / "alpha_cards" / "index.jsonl"
    lines = idx_path.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["alpha_id"] == "my_factor_trial_0007"
    assert entry["card_path"] == "alpha_cards/trial_0007.yaml"


def test_writer_appends_index_across_multiple_cards(tmp_path):
    writer = AlphaCardWriter(tmp_path)
    a = _build_card()
    b = _build_card()
    b = b.model_copy(update={"alpha_id": "b", "source_trial": 8})
    writer.write(a)
    writer.write(b)
    lines = (tmp_path / "alpha_cards" / "index.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["source_trial"] == 8


# -------- write_for_added_indicator --------


def _strategy_with_extra(name: str = "vol_adj_mom") -> Strategy:
    """Take the baseline strategy and append a fake indicator spec."""
    base = load_strategy(REPO / "strategy.yaml")
    extra = IndicatorSpec(name=name, fn=name, params={"lookback": 60})
    return base.model_copy(update={"indicators": [*base.indicators, extra]})


def test_write_for_added_indicator_pulls_spec_from_strategy(tmp_path):
    writer = AlphaCardWriter(tmp_path)
    strategy = _strategy_with_extra("vol_adj_mom")
    path = writer.write_for_added_indicator(
        trial_id=12,
        source_commit="deadbee",
        added_indicator_name="vol_adj_mom",
        strategy=strategy,
        train_metrics=_train_metrics(),
        validation_signal=ValidationSignal.ACCEPTED,
        hypothesis_outcome=HypothesisOutcome.CONFIRMED,
    )
    card = AlphaCard.model_validate(yaml.safe_load(path.read_text()))
    assert card.alpha_id == "vol_adj_mom_trial_0012"
    assert card.signal.params == {"lookback": 60}
    assert card.signal.source_path == "indicators/vol_adj_mom.py"
    assert card.evidence.train_summary["sharpe"] == pytest.approx(0.81)
    assert card.evidence.validation_signal == ValidationSignal.ACCEPTED


def test_write_for_added_indicator_errors_if_name_missing(tmp_path):
    writer = AlphaCardWriter(tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")  # no vol_adj_mom in it
    with pytest.raises(ValueError, match="not in strategy"):
        writer.write_for_added_indicator(
            trial_id=12,
            source_commit="deadbee",
            added_indicator_name="vol_adj_mom",
            strategy=strategy,
            train_metrics=_train_metrics(),
            validation_signal=ValidationSignal.ACCEPTED,
            hypothesis_outcome=HypothesisOutcome.CONFIRMED,
        )


# -------- dossier matcher --------


def test_match_dossier_returns_link_when_factor_exists():
    """The real factor library ships ADX; a search for the substring 'adx'
    should find it. We're calling the live retrieval helper -- knowledge/
    is read-only and must remain identical across the test run."""
    link = match_dossier_by_name("ADX")
    assert link is not None
    assert link.factor_name == "ADX"
    assert link.dossier_path.startswith("knowledge/factors/dossiers/")
    assert link.matched_via == "name_match"


def test_match_dossier_returns_none_when_no_factor_matches():
    """A nonsense indicator name produces no link rather than fabricating one."""
    link = match_dossier_by_name("zzzqqq_no_such_factor_anywhere")
    assert link is None
