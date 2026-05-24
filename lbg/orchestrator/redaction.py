"""Operationalize the `no_data_snooping` invariant (PROPOSAL.html §6).

Any string about to be sent to or received from an agent must be free of:
  - 4-digit year tokens in [1990, 2050] (the realistic SPY-bar range plus a
    margin) -- this catches "2010", "2020", "the 2019 high", etc.
  - Named market events that the LLM might cross-reference against its
    pre-training priors and use to infer the calendar (per §0 of the spec).

The intent is *textual* leakage defense. The proposal explicitly notes
(§0 / line 832) that this mechanism cannot defend against the LLM's own
pre-training knowledge of specific historical periods -- only protocol
discipline can.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Named events whose mention is, by itself, enough to anchor the calendar
# year. Lower-cased; matching is case-insensitive.
_EVENT_NAMES: tuple[str, ...] = (
    "covid",
    "covid-19",
    "covid19",
    "coronavirus",
    "pandemic",
    "gfc",
    "great financial crisis",
    "financial crisis",
    "global financial crisis",
    "9/11",
    "september 11",
    "lehman",
    "lehman brothers",
    "subprime",
    "dot-com",
    "dot com",
    "dotcom",
    "y2k",
    "brexit",
    "tariff war",
    "trade war",
    "trump",
    "biden",
    "obama",
    "bush",
    "yellen",
    "powell",
    "bernanke",
    "qe1",
    "qe2",
    "qe3",
    "taper tantrum",
    "flash crash",
    "volmageddon",
    "fukushima",
    "ukraine war",
)

# 4-digit year in the SPY range plus margin. We exclude obviously
# non-year integers (e.g. 1234 in test data) by requiring word boundaries
# AND that the year sits in [1990, 2050].
_YEAR_RE = re.compile(r"\b(?:19[9]\d|20[0-4]\d|2050)\b")

# Common ISO date forms that don't get caught by _YEAR_RE alone (e.g.
# "2019-01-04T00:00:00Z" -- the year is matched, but flag the whole token).
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


@dataclass(frozen=True)
class RedactionFinding:
    kind: str  # "year", "date", "event_name"
    token: str
    position: int


class RedactionError(RuntimeError):
    """Raised when text destined for the agent (or coming back) contains a leak."""

    def __init__(self, findings: list[RedactionFinding]):
        self.findings = findings
        msg = "; ".join(f"{f.kind}={f.token!r}" for f in findings[:5])
        super().__init__(f"redaction leak ({len(findings)}): {msg}")


def scan_for_leaks(text: str) -> list[RedactionFinding]:
    """Return every leak found in `text`. Empty list means safe."""
    findings: list[RedactionFinding] = []

    for m in _DATE_RE.finditer(text):
        findings.append(RedactionFinding(kind="date", token=m.group(), position=m.start()))

    # Skip year matches that overlap a date match (already reported).
    date_spans = [(m.start(), m.end()) for m in _DATE_RE.finditer(text)]
    for m in _YEAR_RE.finditer(text):
        if any(s <= m.start() < e for s, e in date_spans):
            continue
        findings.append(RedactionFinding(kind="year", token=m.group(), position=m.start()))

    lower = text.lower()
    for name in _EVENT_NAMES:
        start = 0
        while True:
            idx = lower.find(name, start)
            if idx == -1:
                break
            # Word-boundary check on both sides.
            left_ok = idx == 0 or not lower[idx - 1].isalnum()
            right_idx = idx + len(name)
            right_ok = right_idx == len(lower) or not lower[right_idx].isalnum()
            if left_ok and right_ok:
                findings.append(RedactionFinding(kind="event_name", token=name, position=idx))
            start = idx + 1

    return findings


def assert_redacted(text: str, *, context: str = "agent text") -> None:
    """Raise `RedactionError` if `text` contains any leak."""
    findings = scan_for_leaks(text)
    if findings:
        raise RedactionError(findings)
