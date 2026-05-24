"""Read/write the `skills/` directory."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from lbg.skills.schema import Skill, SkillStatus

logger = logging.getLogger(__name__)


class SkillManager:
    def __init__(self, skills_dir: str | Path = "skills") -> None:
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(exist_ok=True)

    def path_for(self, skill_id: str) -> Path:
        return self.skills_dir / f"{skill_id}.yaml"

    def write(self, skill: Skill) -> Path:
        """Persist `skill` as a single YAML document. Returns the file path."""
        path = self.path_for(skill.skill_id)
        text = yaml.safe_dump(
            skill.model_dump(mode="json"),
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
        path.write_text(text, encoding="utf-8")
        return path

    def read(self, skill_id: str) -> Skill:
        path = self.path_for(skill_id)
        if not path.exists():
            raise FileNotFoundError(f"no skill {skill_id!r} at {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return Skill.model_validate(data)

    def list_all(self) -> list[Skill]:
        """Return every Skill on disk. Skips files that fail to validate."""
        out: list[Skill] = []
        for path in sorted(self.skills_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                out.append(Skill.model_validate(data))
            except (ValidationError, yaml.YAMLError) as e:
                logger.warning("skipping malformed skill at %s: %s", path, e)
        return out

    def list_active(self) -> list[Skill]:
        return [s for s in self.list_all() if s.status == SkillStatus.ACTIVE]

    def deprecate(self, skill_id: str) -> None:
        """Flip a skill's status to deprecated. Idempotent."""
        s = self.read(skill_id)
        if s.status == SkillStatus.DEPRECATED:
            return
        self.write(s.model_copy(update={"status": SkillStatus.DEPRECATED}))

    def delete(self, skill_id: str) -> None:
        path = self.path_for(skill_id)
        if path.exists():
            path.unlink()
