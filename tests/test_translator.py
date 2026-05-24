"""Translator agent: source code + alpha card evidence → JSON dossier.

The fake client returns a canned dossier JSON; tests pin that:
  * RoleRunner.translator() parses + validates the JSON.
  * Missing required keys (factor_name / lbg_provenance) raise.
  * DossierWriter writes to alpha_cards/dossiers/<name>_trial_NNNN.txt.
  * The dossier round-trips through json.loads cleanly.
  * Loose schema preserves arbitrary Chinese keys (extra="allow").
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from lbg.orchestrator import RoleRunner, RoleRunnerError
from lbg.schemas import TrainMetrics, ValidationSignal
from lbg.translator import DossierWriter, TranslatorDossier, TranslatorInput

REPO = Path(__file__).resolve().parents[1]


# ---- fake client ----


@dataclass
class _FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class _FakeBlock:
    text: str


@dataclass
class _FakeMessage:
    content: list[_FakeBlock]
    usage: _FakeUsage


class _FakeClient:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.call_count = 0
        self.last_system = None
        self.last_messages = None
        self.messages = self

    def create(self, *, model, max_tokens, system, messages):
        self.call_count += 1
        self.last_system = system
        self.last_messages = messages
        return _FakeMessage(
            content=[_FakeBlock(text=self.response_text)],
            usage=_FakeUsage(),
        )


def _ti() -> TranslatorInput:
    return TranslatorInput(
        trial_id=7,
        source_commit="abc1234",
        indicator_name="rsi_14",
        indicator_fn="rsi",
        indicator_params={"period": 14},
        indicator_source="import pandas as pd\ndef rsi(df, period=14):\n    return df['close']\n",
        hypothesis_text="RSI under 30 should mark oversold reversal candidates.",
        train_metrics=TrainMetrics(sharpe=0.71, max_drawdown=-0.12, turnover=0.35, num_trades=42),
        validation_signal=ValidationSignal.ACCEPTED,
        cited_factors=["RSI"],
    )


_VALID_DOSSIER = json.dumps(
    {
        "factor_name": "rsi_14",
        "lbg_provenance": {
            "kind": "lbg_emitted",
            "trial_id": 7,
            "source_commit": "abc1234",
            "source_path": "indicators/rsi.py",
            "cited_factors": ["RSI"],
        },
        "严密的因子定义": "...",
        "实测训练表现": {"sharpe": 0.71, "max_drawdown": -0.12},
    },
    ensure_ascii=False,
)


# ---- happy path ----


def test_translator_parses_canned_dossier(tmp_path):
    client = _FakeClient(f"```json\n{_VALID_DOSSIER}\n```\n")
    runner = RoleRunner(client=client)
    res = runner.translator(_ti())
    assert client.call_count == 1
    assert res.dossier.factor_name == "rsi_14"
    assert res.dossier.lbg_provenance["trial_id"] == 7
    assert res.compute.role == "translator"
    assert res.compute.trial_id == 7


def test_translator_preserves_arbitrary_chinese_keys(tmp_path):
    """extra='allow' means unknown keys round-trip verbatim. Critical for
    matching the seed library's 28+ free-form fields."""
    client = _FakeClient(f"```json\n{_VALID_DOSSIER}\n```\n")
    runner = RoleRunner(client=client)
    res = runner.translator(_ti())
    dumped = res.dossier.model_dump()
    assert "严密的因子定义" in dumped
    assert "实测训练表现" in dumped


def test_translator_writes_to_alpha_cards_dossiers(tmp_path):
    """End-to-end: runner returns dossier, DossierWriter persists it."""
    client = _FakeClient(f"```json\n{_VALID_DOSSIER}\n```\n")
    runner = RoleRunner(client=client)
    res = runner.translator(_ti())
    writer = DossierWriter(tmp_path)
    path = writer.write(res.dossier, trial_id=7)
    assert path == tmp_path / "alpha_cards" / "dossiers" / "rsi_14_trial_0007.txt"
    assert path.exists()
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["factor_name"] == "rsi_14"
    assert reloaded["lbg_provenance"]["trial_id"] == 7


# ---- failure path ----


def test_translator_missing_factor_name_raises():
    bad = json.dumps({"lbg_provenance": {"kind": "lbg_emitted"}})
    client = _FakeClient(f"```json\n{bad}\n```\n")
    runner = RoleRunner(client=client)
    with pytest.raises(RoleRunnerError, match="missing required keys"):
        runner.translator(_ti())


def test_translator_missing_provenance_raises():
    bad = json.dumps({"factor_name": "rsi_14"})
    client = _FakeClient(f"```json\n{bad}\n```\n")
    runner = RoleRunner(client=client)
    with pytest.raises(RoleRunnerError, match="missing required keys"):
        runner.translator(_ti())


def test_translator_non_json_response_raises():
    client = _FakeClient("just prose, no code block")
    runner = RoleRunner(client=client)
    with pytest.raises(RoleRunnerError, match="JSON code block"):
        runner.translator(_ti())


def test_translator_malformed_json_raises():
    client = _FakeClient("```json\n{not valid json,,,\n```\n")
    runner = RoleRunner(client=client)
    with pytest.raises(RoleRunnerError, match="not valid JSON"):
        runner.translator(_ti())


def test_translator_yaml_block_works_as_fallback():
    """A bare JSON object (no fence) is also acceptable."""
    client = _FakeClient(_VALID_DOSSIER)
    runner = RoleRunner(client=client)
    res = runner.translator(_ti())
    assert res.dossier.factor_name == "rsi_14"


# ---- user prompt content ----


def test_translator_user_prompt_includes_source_and_metrics(tmp_path):
    client = _FakeClient(f"```json\n{_VALID_DOSSIER}\n```\n")
    runner = RoleRunner(client=client)
    runner.translator(_ti())
    user_msg = client.last_messages[0]["content"]
    # Source code anchored.
    assert "def rsi" in user_msg
    # Hypothesis text echoed.
    assert "oversold reversal" in user_msg
    # Observed metrics surface as exact strings the prompt formats.
    assert "sharpe: 0.7100" in user_msg
    assert "num_trades: 42" in user_msg
    # Cited factors echoed.
    assert "RSI" in user_msg


# ---- DossierWriter standalone ----


def test_dossier_writer_handles_unsafe_factor_names(tmp_path):
    """A bad name (slashes, dots) is sanitized to underscores."""
    writer = DossierWriter(tmp_path)
    d = TranslatorDossier(
        factor_name="weird/name..with.dots",
        lbg_provenance={"kind": "lbg_emitted"},
    )
    path = writer.write(d, trial_id=3)
    assert "/" not in path.name
    assert ".." not in path.name
