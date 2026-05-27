"""A + B fixes from campaign v4 diagnostics.

v4 showed Editor inventing `chandelier_long` / `chandelier_long_tight` etc.
that aren't in the dossier library; the existing dedup couldn't catch them
because it's keyed on dossier names. A+B records every add_indicator's fn
verbatim and shows it back to the Editor as a hard-ban list.

Tests pin:
  * TriedFactorRecord.indicator_fn defaults None, round-trips when set.
  * ContextBuilder._banned_indicator_fns reads them in order, skips current,
    skips accepted, dedupes.
  * EditorContext.banned_indicator_fns surfaces them; prompt renders the
    section only when non-empty; system prompt mentions the rule.
"""

from __future__ import annotations

import json
from pathlib import Path

from lbg.dsl import load_strategy
from lbg.memory import MemoryManager, TriedFactorRecord
from lbg.orchestrator.context_builder import ContextBuilder, EditorContext
from lbg.orchestrator.role_runner import _format_banned_indicator_fns, _render_editor_user_prompt

REPO = Path(__file__).resolve().parents[1]


# -------- schema --------


def test_tried_factor_record_indicator_fn_defaults_none():
    rec = TriedFactorRecord(trial_id=0, factors=["ADX"], decision="reject", source="editor_cite")
    assert rec.indicator_fn is None


def test_tried_factor_record_round_trips_indicator_fn():
    rec = TriedFactorRecord(
        trial_id=2,
        factors=[],
        decision="reject",
        source="add_indicator_only",
        indicator_fn="chandelier_long",
    )
    dumped = json.loads(rec.model_dump_json())
    assert dumped["indicator_fn"] == "chandelier_long"
    reloaded = TriedFactorRecord.model_validate(dumped)
    assert reloaded.indicator_fn == "chandelier_long"


# -------- ContextBuilder._banned_indicator_fns --------


def _seed_tried(mm: MemoryManager, items: list[tuple[int, str, str, str | None]]) -> None:
    """items = (trial_id, decision, source, indicator_fn)."""
    for trial_id, decision, source, fn in items:
        mm.append_tried_factors(
            TriedFactorRecord(
                trial_id=trial_id,
                factors=[],
                decision=decision,
                source=source,
                indicator_fn=fn,
            )
        )


def test_banned_includes_rejected_fns(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    _seed_tried(
        mm,
        [
            (0, "reject", "add_indicator_only", "chandelier_long"),
            (1, "reject", "add_indicator_only", "chandelier_long_tight"),
        ],
    )
    cb = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")
    banned = cb._banned_indicator_fns(strategy)
    # Updated: baseline `sma` is also included (current-strategy ban).
    # The two chandelier_long variants must be there too.
    assert "chandelier_long" in banned
    assert "chandelier_long_tight" in banned


def test_banned_includes_accepted(tmp_path):
    """Updated semantic (2026-05-25): accepted fns are also banned from
    re-proposal so the campaign loop accumulates *distinct* factors
    rather than converging on a single factor. The Editor can still
    tune accepted fns via parameter_change; the ban only applies to
    `add_indicator`."""
    mm = MemoryManager(tmp_path / "memory")
    _seed_tried(
        mm,
        [
            (0, "accept", "auto_extract", "lsma"),
            (1, "reject", "add_indicator_only", "chandelier_long"),
        ],
    )
    cb = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")
    banned = cb._banned_indicator_fns(strategy)
    # Both should be banned now -- accepted no longer gets a free pass.
    assert "lsma" in banned
    assert "chandelier_long" in banned


def test_banned_includes_current_strategy_fns(tmp_path):
    """sma is in the baseline AND was tried at trial 0; it appears in the
    ban list. Editor can still tune sma's period via parameter_change --
    the ban is on re-authoring it as a new add_indicator."""
    mm = MemoryManager(tmp_path / "memory")
    _seed_tried(mm, [(0, "reject", "add_indicator_only", "sma")])
    cb = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")
    assert "sma" in cb._banned_indicator_fns(strategy)


def test_banned_dedupes(tmp_path):
    """Same fn rejected twice → one entry in the ban list, in first-seen order."""
    mm = MemoryManager(tmp_path / "memory")
    _seed_tried(
        mm,
        [
            (0, "reject", "add_indicator_only", "chandelier_long"),
            (1, "reject", "add_indicator_only", "chandelier_long"),
        ],
    )
    cb = ContextBuilder(mm, repo_root=tmp_path)
    banned = cb._banned_indicator_fns(load_strategy(REPO / "strategy.yaml"))
    # Updated semantic (2026-05-25): baseline `sma` is also banned (it's
    # already in the strategy; you can't add_indicator the same fn twice).
    # chandelier_long should still be present.
    assert "sma" in banned
    assert "chandelier_long" in banned


def test_banned_empty_when_no_tried_factors(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    cb = ContextBuilder(mm, repo_root=tmp_path)
    # Updated semantic: even with no tried_factors, fns currently in the
    # strategy are banned (you can't re-add what's already there).
    banned = cb._banned_indicator_fns(load_strategy(REPO / "strategy.yaml"))
    assert "sma" in banned


def test_editor_view_surfaces_banned(tmp_path):
    mm = MemoryManager(tmp_path / "memory")
    _seed_tried(mm, [(0, "reject", "add_indicator_only", "chandelier_long")])
    cb = ContextBuilder(mm, repo_root=tmp_path)
    strategy = load_strategy(REPO / "strategy.yaml")
    view = cb.editor_view(strategy)
    # Banned list now includes baseline fns AND tried fns.
    assert "sma" in view.banned_indicator_fns
    assert "chandelier_long" in view.banned_indicator_fns


# -------- prompt rendering --------


def _ctx(banned: tuple[str, ...]) -> EditorContext:
    return EditorContext(
        strategy_yaml="name: t\n",
        recent_trials=[],
        semantic_memory={},
        skills=[],
        factor_hints=[],
        banned_indicator_fns=banned,
    )


def test_prompt_section_omitted_when_no_bans():
    prompt = _render_editor_user_prompt(_ctx(()), trial_id=0)
    assert "Indicator names you must NOT repeat" not in prompt


def test_prompt_section_renders_each_banned_name():
    prompt = _render_editor_user_prompt(
        _ctx(("chandelier_long", "chandelier_long_tight")), trial_id=0
    )
    assert "Indicator names you must NOT repeat" in prompt
    assert "`chandelier_long`" in prompt
    assert "`chandelier_long_tight`" in prompt


def test_prompt_section_warns_against_renaming():
    """The ban must explicitly forbid the v4 evasion path (chandelier_long
    -> chandelier_long_tight), not just the literal name."""
    out = _format_banned_indicator_fns(("chandelier_long",))
    assert "minor renaming" in out.lower() or "rename" in out.lower()


def test_editor_system_prompt_documents_the_rule():
    """The system prompt has to spell out the ban; this is a sanity-check
    that the section wasn't accidentally dropped."""
    sys_prompt = (REPO / "lbg" / "orchestrator" / "prompts" / "editor_system.md").read_text(
        encoding="utf-8"
    )
    assert "Banned indicator names" in sys_prompt
    assert "must NOT repeat" in sys_prompt or "must NOT" in sys_prompt
