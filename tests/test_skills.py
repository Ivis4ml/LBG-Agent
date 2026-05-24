"""Tests for `lbg.skills.SkillManager` + ContextBuilder integration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from lbg.dsl import load_strategy
from lbg.memory import MemoryManager
from lbg.orchestrator.context_builder import ContextBuilder
from lbg.schemas import EditType, ValidationSignal
from lbg.skills import (
    Skill,
    SkillEvidence,
    SkillManager,
    SkillRecipe,
    SkillStatus,
    SkillTrigger,
)

REPO = Path(__file__).resolve().parents[1]


def _sample_skill(skill_id: str = "vol_target_after_drawdown", status=SkillStatus.ACTIVE) -> Skill:
    return Skill(
        skill_id=skill_id,
        status=status,
        trigger=SkillTrigger(
            validation_signal=ValidationSignal.REJECTED_DRAWDOWN_REGRESSION,
            common_context=[
                "fixed_fraction sizing currently active",
                "drawdown elevated",
            ],
        ),
        recipe=SkillRecipe(
            edit_type=EditType.CHANGE_SIZING_MODE,
            change={
                "sizing": {
                    "mode": "volatility_target",
                    "target_vol": 0.12,
                    "max_position": 1.0,
                    "vol_lookback": 20,
                }
            },
        ),
        evidence=SkillEvidence(
            accepted_trials=[12, 18, 27],
            rejected_trials_that_motivated_it=[9, 11],
        ),
        known_failure_modes=[
            "target_vol below 0.08 often undertrades",
            "target_vol above 0.15 can fail drawdown gate",
        ],
        complexity_cost=1.0,
        last_curated_at_trial=30,
    )


# -------- schema --------


def test_skill_round_trip_through_yaml(tmp_path):
    s = _sample_skill()
    raw = yaml.safe_dump(s.model_dump(mode="json"), sort_keys=False)
    s2 = Skill.model_validate(yaml.safe_load(raw))
    assert s == s2


def test_skill_status_enum_values():
    assert {x.value for x in SkillStatus} == {"active", "deprecated", "exploratory"}


def test_skill_id_rejects_path_separators():
    with pytest.raises(ValidationError, match="path separators"):
        _sample_skill(skill_id="../escape")


def test_skill_extra_field_rejected():
    """extra='forbid' on Skill -- agent can't smuggle fields through skill artifacts."""
    raw = _sample_skill().model_dump(mode="python")
    raw["leak"] = "calendar_year=2019"
    with pytest.raises(ValidationError):
        Skill.model_validate(raw)


# -------- read/write --------


def test_write_then_read_round_trip(tmp_path):
    mgr = SkillManager(tmp_path / "skills")
    s = _sample_skill()
    path = mgr.write(s)
    assert path.exists()
    s2 = mgr.read("vol_target_after_drawdown")
    assert s2 == s


def test_read_missing_skill_raises(tmp_path):
    mgr = SkillManager(tmp_path / "skills")
    with pytest.raises(FileNotFoundError, match="no skill"):
        mgr.read("does_not_exist")


def test_list_all_returns_sorted_filenames(tmp_path):
    mgr = SkillManager(tmp_path / "skills")
    mgr.write(_sample_skill("b_skill"))
    mgr.write(_sample_skill("a_skill"))
    mgr.write(_sample_skill("c_skill"))
    all_ids = [s.skill_id for s in mgr.list_all()]
    assert all_ids == ["a_skill", "b_skill", "c_skill"]


def test_list_active_filters_out_deprecated(tmp_path):
    mgr = SkillManager(tmp_path / "skills")
    mgr.write(_sample_skill("active_one", status=SkillStatus.ACTIVE))
    mgr.write(_sample_skill("deprecated_one", status=SkillStatus.DEPRECATED))
    mgr.write(_sample_skill("exploratory_one", status=SkillStatus.EXPLORATORY))
    active = mgr.list_active()
    assert [s.skill_id for s in active] == ["active_one"]


def test_list_all_skips_malformed_files(tmp_path, caplog):
    mgr = SkillManager(tmp_path / "skills")
    mgr.write(_sample_skill("good"))
    (mgr.skills_dir / "bad.yaml").write_text("this is not a valid skill\n")
    with caplog.at_level("WARNING"):
        out = mgr.list_all()
    assert [s.skill_id for s in out] == ["good"]
    assert "bad.yaml" in caplog.text


def test_deprecate_flips_status(tmp_path):
    mgr = SkillManager(tmp_path / "skills")
    mgr.write(_sample_skill())
    mgr.deprecate("vol_target_after_drawdown")
    s = mgr.read("vol_target_after_drawdown")
    assert s.status == SkillStatus.DEPRECATED


def test_deprecate_is_idempotent(tmp_path):
    mgr = SkillManager(tmp_path / "skills")
    mgr.write(_sample_skill(status=SkillStatus.DEPRECATED))
    # Should not raise or change anything.
    mgr.deprecate("vol_target_after_drawdown")
    s = mgr.read("vol_target_after_drawdown")
    assert s.status == SkillStatus.DEPRECATED


def test_delete_removes_file(tmp_path):
    mgr = SkillManager(tmp_path / "skills")
    mgr.write(_sample_skill())
    assert mgr.path_for("vol_target_after_drawdown").exists()
    mgr.delete("vol_target_after_drawdown")
    assert not mgr.path_for("vol_target_after_drawdown").exists()


def test_delete_missing_is_noop(tmp_path):
    mgr = SkillManager(tmp_path / "skills")
    # Should not raise.
    mgr.delete("does_not_exist")


# -------- ContextBuilder integration --------


def test_editor_view_loads_active_skills(tmp_path):
    mem = MemoryManager(tmp_path / "memory")
    skills = SkillManager(tmp_path / "skills")
    skills.write(_sample_skill("s_active", status=SkillStatus.ACTIVE))
    skills.write(_sample_skill("s_deprecated", status=SkillStatus.DEPRECATED))
    cb = ContextBuilder(mem, repo_root=tmp_path, skills=skills)
    strategy = load_strategy(REPO / "strategy.yaml")
    ctx = cb.editor_view(strategy)
    assert [s.skill_id for s in ctx.skills] == ["s_active"]


def test_editor_view_with_no_skills_is_empty(tmp_path):
    mem = MemoryManager(tmp_path / "memory")
    cb = ContextBuilder(mem, repo_root=tmp_path)  # default skills dir auto-created
    strategy = load_strategy(REPO / "strategy.yaml")
    ctx = cb.editor_view(strategy)
    assert ctx.skills == []


def test_skill_renders_in_editor_user_prompt(tmp_path):
    """The renderer puts skills into the prompt under '## Active skills'."""
    from lbg.orchestrator.role_runner import _render_editor_user_prompt

    mem = MemoryManager(tmp_path / "memory")
    skills = SkillManager(tmp_path / "skills")
    skills.write(_sample_skill())
    cb = ContextBuilder(mem, repo_root=tmp_path, skills=skills)
    strategy = load_strategy(REPO / "strategy.yaml")
    ctx = cb.editor_view(strategy)
    text = _render_editor_user_prompt(ctx, trial_id=1)
    assert "Active skills" in text
    assert "vol_target_after_drawdown" in text
    assert "rejected_drawdown_regression" in text
    assert "change_sizing_mode" in text
