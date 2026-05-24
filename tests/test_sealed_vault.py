"""Tests for `lbg.sealed_vault.SealedVault`."""

from __future__ import annotations

import json

import pytest

from lbg.sealed_vault import SealedVault, SealedVaultError


def test_write_creates_file_with_timestamp(tmp_path):
    vault = SealedVault(tmp_path / "sealed.json")
    assert not vault.is_sealed()
    vault.open_and_write({"sharpe": 0.5, "max_drawdown": -0.2})
    assert vault.is_sealed()
    data = json.loads((tmp_path / "sealed.json").read_text())
    assert data["sharpe"] == 0.5
    assert "sealed_at_utc" in data


def test_second_write_raises(tmp_path):
    vault = SealedVault(tmp_path / "sealed.json")
    vault.open_and_write({"x": 1})
    with pytest.raises(SealedVaultError, match="already sealed"):
        vault.open_and_write({"x": 2})


def test_read_before_seal_raises(tmp_path):
    vault = SealedVault(tmp_path / "sealed.json")
    with pytest.raises(SealedVaultError, match="not been sealed"):
        vault.read()


def test_read_returns_payload(tmp_path):
    vault = SealedVault(tmp_path / "sealed.json")
    vault.open_and_write({"name": "x"})
    data = vault.read()
    assert data["name"] == "x"


def test_file_is_read_only_after_seal(tmp_path):
    vault = SealedVault(tmp_path / "sealed.json")
    vault.open_and_write({"x": 1})
    mode = (tmp_path / "sealed.json").stat().st_mode
    # Owner write bit must be off.
    assert mode & 0o200 == 0


def test_parent_dirs_created(tmp_path):
    p = tmp_path / "artifacts" / "sealed" / "verdict.json"
    vault = SealedVault(p)
    vault.open_and_write({"x": 1})
    assert p.exists()
