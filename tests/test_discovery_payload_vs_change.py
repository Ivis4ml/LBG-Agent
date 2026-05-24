"""Regression: Discovery must read indicator fn/name/params from the typed
editor_result.payload, NOT from editor_result.proposal.proposed_edit.change.

The latter is a raw `dict | None` per the EditProposal schema; getattr on a
dict returns the fallback, silently producing None for `name` and `fn`.
That silent failure dropped alpha_cards / Translator / tried_factors-fn
on the floor for every accepted add_indicator in campaigns v1-v5.
"""

from __future__ import annotations

from lbg.parser import AddIndicatorPayload, parse_proposal


def test_proposal_change_is_dict_not_payload():
    """The schema bug: `proposed_edit.change` is `dict | None`, so attribute
    access (`change.name`, `change.fn`) silently returns None."""
    raw = {
        "trial_id": 0,
        "hypothesis": "t",
        "proposed_edit": {
            "type": "add_indicator",
            "change": {
                "name": "rsi_14",
                "fn": "rsi",
                "source": "import pandas as pd\ndef rsi(df):\n    return df['close']\n",
                "params": {"period": 14},
                "attach": {
                    "rule": "indicator_above",
                    "indicator": "rsi_14",
                    "threshold": 30.0,
                },
            },
        },
        "expected_train_signal": "neutral",
        "expected_validation_signal": "reject",
        "fallback_if_rejected": "x",
    }
    proposal, payload = parse_proposal(raw)
    change = proposal.proposed_edit.change
    # change is a dict; getattr returns the default.
    assert isinstance(change, dict)
    assert getattr(change, "name", None) is None
    assert getattr(change, "fn", None) is None
    # The typed payload is what carries the real values.
    assert isinstance(payload, AddIndicatorPayload)
    assert payload.name == "rsi_14"
    assert payload.fn == "rsi"
    assert payload.params == {"period": 14}
