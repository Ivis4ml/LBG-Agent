"""The Editor user prompt must surface FactorHint entries verbatim, and
must omit the section entirely when the hint list is empty -- prompt noise
on a useless section would waste tokens for no signal.
"""

from __future__ import annotations

from lbg.orchestrator.context_builder import EditorContext, FactorHint
from lbg.orchestrator.role_runner import _format_factor_hints, _render_editor_user_prompt


def _ctx(hints: list[FactorHint]) -> EditorContext:
    return EditorContext(
        strategy_yaml="name: x\n",
        recent_trials=[],
        semantic_memory={},
        skills=[],
        factor_hints=hints,
    )


def test_factor_hints_section_omitted_when_empty():
    prompt = _render_editor_user_prompt(_ctx([]), trial_id=0)
    assert "Candidate factors from knowledge base" not in prompt


def test_factor_hints_section_lists_each_name_and_category():
    hints = [
        FactorHint(name="ADX", category="trend strength", one_line="quantifies trend"),
        FactorHint(name="AmihudIlliq", category="liquidity", one_line="illiquidity proxy"),
    ]
    prompt = _render_editor_user_prompt(_ctx(hints), trial_id=0)
    assert "Candidate factors from knowledge base" in prompt
    assert "ADX" in prompt
    assert "trend strength" in prompt
    assert "quantifies trend" in prompt
    assert "AmihudIlliq" in prompt
    assert "liquidity" in prompt


def test_factor_hints_section_disclaims_authority():
    hints = [FactorHint(name="X", category="c", one_line="o")]
    out = _format_factor_hints(hints)
    # The section's footer reminds the Editor invariants run regardless.
    assert "hints" in out.lower()
    assert "invariants" in out.lower()


def test_factor_hints_section_appears_after_strategy_before_recent_trials():
    """Order matters: strategy first so the Editor sees what's there, then
    factor library so retrieval is contextualised, then trial history."""
    hints = [FactorHint(name="ADX", category="trend", one_line="trend strength")]
    prompt = _render_editor_user_prompt(_ctx(hints), trial_id=0)
    strat_idx = prompt.index("Current strategy")
    hints_idx = prompt.index("Candidate factors from knowledge base")
    trials_idx = prompt.index("Recent trial history")
    assert strat_idx < hints_idx < trials_idx


def test_blank_category_renders_placeholder():
    """A factor with no category in the index should still render cleanly."""
    hints = [FactorHint(name="X", category="", one_line="something")]
    out = _format_factor_hints(hints)
    assert "(no category)" in out
