"""Tests for `lbg.orchestrator.redaction`."""

from __future__ import annotations

import pytest

from lbg.orchestrator.redaction import (
    RedactionError,
    assert_redacted,
    scan_for_leaks,
)

# -------- safe text --------


def test_clean_text_passes():
    text = "current sharpe is 0.69; we propose change_sizing_mode to volatility_target."
    assert scan_for_leaks(text) == []
    assert_redacted(text)


def test_period_20_does_not_trigger_year_match():
    """`period: 20` and `period: 200` must not match the year pattern."""
    assert scan_for_leaks("period: 20\nperiod: 200\nperiod: 1500\nperiod: 999\n") == []


def test_normal_integers_in_metric_range_do_not_trigger():
    assert scan_for_leaks("num_trades: 38\nsharpe: 0.69\nmax_dd: -0.13\n") == []


# -------- years --------


def test_year_token_is_caught():
    findings = scan_for_leaks("the 2019 high was 200")
    kinds = {f.kind for f in findings}
    assert "year" in kinds


def test_each_decade_of_recent_history_caught():
    for year in (1995, 2008, 2010, 2019, 2020, 2024, 2045):
        findings = scan_for_leaks(f"value at {year} bar")
        assert any(f.token == str(year) for f in findings), year


def test_2099_is_outside_range():
    """`_YEAR_RE` covers 1990-2050; later years are not flagged here."""
    findings = scan_for_leaks("year 2099")
    assert findings == []


# -------- iso dates --------


def test_iso_date_caught():
    findings = scan_for_leaks("start=2019-01-04")
    kinds = {f.kind for f in findings}
    assert "date" in kinds


def test_year_overlapping_with_date_not_double_reported():
    findings = scan_for_leaks("split bound: 2018-12-31")
    # Exactly one date finding, no separate year for the same span.
    dates = [f for f in findings if f.kind == "date"]
    years = [f for f in findings if f.kind == "year"]
    assert len(dates) == 1
    assert years == []


# -------- event names --------


def test_covid_caught():
    findings = scan_for_leaks("during covid the strategy underperformed")
    kinds = {f.kind for f in findings}
    assert "event_name" in kinds


def test_event_name_case_insensitive():
    findings = scan_for_leaks("After GFC the model improved")
    assert any(f.token == "gfc" for f in findings)


def test_word_boundary_avoids_false_positive_inside_word():
    """`tarif` should not match `tariff war`; substring inside another word must not fire."""
    # 'discovery' contains 'cover' but no event token, ensures we don't have false positives.
    assert scan_for_leaks("the discovery loop runs many trials") == []


def test_partial_substring_does_not_match():
    """'pandemicial' (made-up word containing 'pandemic') -- word boundary should still match
    because boundary is on alnum; here 'pandemic' is followed by 'i', so no match."""
    assert scan_for_leaks("pandemicial behavior") == []


# -------- assert_redacted --------


def test_assert_redacted_passes_on_clean_text():
    assert_redacted("no leaks here")


def test_assert_redacted_raises_on_year():
    with pytest.raises(RedactionError, match="leak"):
        assert_redacted("the 2020 bear market")


def test_assert_redacted_raises_on_event_name():
    with pytest.raises(RedactionError, match="leak"):
        assert_redacted("the covid crash")


def test_redaction_error_carries_findings():
    try:
        assert_redacted("during covid in 2020")
    except RedactionError as e:
        kinds = {f.kind for f in e.findings}
        assert "event_name" in kinds
        assert "year" in kinds
    else:
        pytest.fail("expected RedactionError")
