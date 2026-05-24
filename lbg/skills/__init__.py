"""SkillManager: typed `skills/*.yaml` storage (PROPOSAL.html §6.2 line 1487).

A skill is a *compressed reusable edit recipe with evidence and known
failure modes* — NOT a memory summary. The Curator extracts skills from
patterns of accepted trials and writes them here. The Editor reads active
skills (via `ContextBuilder`) to bias future proposals.

Per user direction: SkillManager only writes skill artifacts; it does
**not** modify `strategy.yaml`. Strategy edits remain the Editor's job.
"""

from lbg.skills.manager import SkillManager
from lbg.skills.schema import (
    Skill,
    SkillEvidence,
    SkillRecipe,
    SkillStatus,
    SkillTrigger,
)

__all__ = [
    "Skill",
    "SkillEvidence",
    "SkillManager",
    "SkillRecipe",
    "SkillStatus",
    "SkillTrigger",
]
