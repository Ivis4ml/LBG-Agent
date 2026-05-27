"""Apply a parsed `(EditProposal, payload)` to the candidate working tree.

Eight edit types from PROPOSAL.html §6.2. Six are implemented as MVP:
`parameter_change`, `change_sizing_mode`, `change_exit_rule`, `add_filter`,
`remove_filter`, `add_indicator`. The remaining two (`simplify`,
`revert_to_trial_N`) raise `NotImplementedError` -- they need the SkillManager
and trial history they'll get in step 12.

The builder writes files synchronously and returns a typed result. It never
runs git itself; the Orchestrator stages and commits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel

from lbg.dsl.schema import IndicatorSpec, Strategy
from lbg.parser.payloads import (
    AddFilterPayload,
    AddIndicatorPayload,
    ChangeExitRulePayload,
    ChangeSizingModePayload,
    ParameterChangePayload,
    RemoveFilterPayload,
)
from lbg.schemas import EditProposal, EditSummary, EditType


class CandidateBuildError(RuntimeError):
    """Raised when an edit cannot be applied (missing reference, bad path, ...)."""


@dataclass(frozen=True)
class CandidateApplyResult:
    new_strategy: Strategy
    touched_paths: list[Path]
    edit_summary: EditSummary


_INDICATOR_PATH_RE = re.compile(
    r"^indicators\[(?P<name>[A-Za-z_][A-Za-z0-9_]*)\]\.params\.(?P<key>[A-Za-z_][A-Za-z0-9_]*)$"
)
_FILTER_PATH_RE = re.compile(
    r"^(?P<list>filters|exit_filters)\[(?P<idx>\d+)\]\.(?P<field>threshold|rearm_threshold)$"
)


class CandidateBuilder:
    def __init__(self, repo_root: str | Path = ".") -> None:
        self.repo_root = Path(repo_root).resolve()
        self.strategy_path = self.repo_root / "strategy.yaml"
        self.indicators_dir = self.repo_root / "indicators"

    def apply(
        self,
        parent_strategy: Strategy,
        proposal: EditProposal,
        payload: BaseModel,
    ) -> CandidateApplyResult:
        edit_type = proposal.proposed_edit.type
        if edit_type == EditType.PARAMETER_CHANGE:
            return self._apply_parameter_change(parent_strategy, payload)  # type: ignore[arg-type]
        if edit_type == EditType.CHANGE_SIZING_MODE:
            return self._apply_change_sizing_mode(parent_strategy, payload)  # type: ignore[arg-type]
        if edit_type == EditType.CHANGE_EXIT_RULE:
            return self._apply_change_exit_rule(parent_strategy, payload)  # type: ignore[arg-type]
        if edit_type == EditType.ADD_FILTER:
            return self._apply_add_filter(parent_strategy, payload)  # type: ignore[arg-type]
        if edit_type == EditType.REMOVE_FILTER:
            return self._apply_remove_filter(parent_strategy, payload)  # type: ignore[arg-type]
        if edit_type == EditType.ADD_INDICATOR:
            return self._apply_add_indicator(parent_strategy, payload)  # type: ignore[arg-type]
        if edit_type == EditType.SIMPLIFY:
            return self._apply_simplify(parent_strategy, payload)  # type: ignore[arg-type]
        if edit_type == EditType.REVERT_TO_TRIAL:
            return self._apply_revert_to_trial(parent_strategy, payload)  # type: ignore[arg-type]
        raise CandidateBuildError(f"unsupported edit type: {edit_type!r}")

    # ------------------------------------------------------------------ parameter_change

    def _apply_parameter_change(
        self,
        parent: Strategy,
        payload: ParameterChangePayload,
    ) -> CandidateApplyResult:
        data = parent.model_dump()
        before, after = _set_path(data, payload.path, payload.value)
        new_strategy = Strategy.model_validate(data)
        self._write_strategy(new_strategy)
        return CandidateApplyResult(
            new_strategy=new_strategy,
            touched_paths=[self.strategy_path],
            edit_summary=EditSummary(
                type=EditType.PARAMETER_CHANGE,
                target="strategy.yaml",
                summary=f"{payload.path}: {before!r} -> {after!r}",
            ),
        )

    # ------------------------------------------------------------------ change_sizing_mode

    def _apply_change_sizing_mode(
        self,
        parent: Strategy,
        payload: ChangeSizingModePayload,
    ) -> CandidateApplyResult:
        new_strategy = parent.model_copy(update={"sizing": payload.sizing})
        self._write_strategy(new_strategy)
        return CandidateApplyResult(
            new_strategy=new_strategy,
            touched_paths=[self.strategy_path],
            edit_summary=EditSummary(
                type=EditType.CHANGE_SIZING_MODE,
                target="strategy.yaml",
                summary=f"{parent.sizing.mode} -> {payload.sizing.mode}",
            ),
        )

    # ------------------------------------------------------------------ change_exit_rule

    def _apply_change_exit_rule(
        self,
        parent: Strategy,
        payload: ChangeExitRulePayload,
    ) -> CandidateApplyResult:
        new_strategy = parent.model_copy(update={"exit": payload.exit})
        self._write_strategy(new_strategy)
        return CandidateApplyResult(
            new_strategy=new_strategy,
            touched_paths=[self.strategy_path],
            edit_summary=EditSummary(
                type=EditType.CHANGE_EXIT_RULE,
                target="strategy.yaml",
                summary=f"{parent.exit.rule} -> {payload.exit.rule}",
            ),
        )

    # ------------------------------------------------------------------ add_filter / remove_filter

    def _apply_add_filter(
        self,
        parent: Strategy,
        payload: AddFilterPayload,
    ) -> CandidateApplyResult:
        # Cross-check: filter must reference an existing indicator name.
        declared = {i.name for i in parent.indicators}
        if payload.filter.indicator not in declared:
            raise CandidateBuildError(
                f"filter references unknown indicator {payload.filter.indicator!r}; "
                f"declared: {sorted(declared)}"
            )
        list_field = "exit_filters" if payload.target == "exit" else "filters"
        current = getattr(parent, list_field)
        new_filters = [*current, payload.filter]
        new_strategy = parent.model_copy(update={list_field: new_filters})
        self._write_strategy(new_strategy)
        return CandidateApplyResult(
            new_strategy=new_strategy,
            touched_paths=[self.strategy_path],
            edit_summary=EditSummary(
                type=EditType.ADD_FILTER,
                target="strategy.yaml",
                summary=(
                    f"+{payload.target}_{payload.filter.rule}("
                    f"{payload.filter.indicator} vs {payload.filter.threshold})"
                ),
            ),
        )

    def _apply_remove_filter(
        self,
        parent: Strategy,
        payload: RemoveFilterPayload,
    ) -> CandidateApplyResult:
        list_field = "exit_filters" if payload.target == "exit" else "filters"
        current = getattr(parent, list_field)
        if payload.index >= len(current):
            raise CandidateBuildError(
                f"{list_field} index {payload.index} out of range; have {len(current)} filter(s)"
            )
        removed = current[payload.index]
        new_filters = list(current)
        del new_filters[payload.index]
        new_strategy = parent.model_copy(update={list_field: new_filters})
        self._write_strategy(new_strategy)
        return CandidateApplyResult(
            new_strategy=new_strategy,
            touched_paths=[self.strategy_path],
            edit_summary=EditSummary(
                type=EditType.REMOVE_FILTER,
                target="strategy.yaml",
                summary=f"-{payload.target}_{removed.rule}({removed.indicator})",
            ),
        )

    # ------------------------------------------------------------------ add_indicator

    def _apply_add_indicator(
        self,
        parent: Strategy,
        payload: AddIndicatorPayload,
    ) -> CandidateApplyResult:
        existing_names = {i.name for i in parent.indicators}
        if payload.name in existing_names:
            raise CandidateBuildError(f"indicator name {payload.name!r} already exists in strategy")

        # Path-traversal defense: payload.fn cannot contain separators (the
        # IndicatorSpec validator below also enforces this; doing it here for
        # defence in depth).
        if "/" in payload.fn or "\\" in payload.fn or ".." in payload.fn:
            raise CandidateBuildError(f"illegal characters in fn {payload.fn!r}")

        indicator_path = self.indicators_dir / f"{payload.fn}.py"
        if indicator_path.exists():
            raise CandidateBuildError(
                f"indicator file already exists at {indicator_path}; "
                "use parameter_change or simplify instead"
            )

        new_spec = IndicatorSpec(name=payload.name, fn=payload.fn, params=payload.params)
        new_indicators = [*parent.indicators, new_spec]
        # Bundle the attach filter atomically so the new indicator can be
        # wired in a single trial; STAGE1_REPORT § 7 showed bare add_indicator
        # wastes ~5/8 of one provider's trials. Without attach the indicator
        # is dead code and the unwired check below rejects the candidate.
        update: dict[str, object] = {"indicators": new_indicators}
        attach_summary = ""
        if payload.attach is not None:
            if payload.attach.indicator != payload.name:
                raise CandidateBuildError(
                    f"add_indicator.attach.indicator must equal the new "
                    f"indicator name; got attach references "
                    f"{payload.attach.indicator!r} but the new indicator is "
                    f"{payload.name!r}"
                )
            list_field = "exit_filters" if payload.attach_target == "exit" else "filters"
            current = getattr(parent, list_field)
            update[list_field] = [*current, payload.attach]
            attach_summary = (
                f" attached as {payload.attach_target}_{payload.attach.rule}:"
                f"{payload.attach.indicator}"
            )
        new_strategy = parent.model_copy(update=update)

        if new_strategy.unwired_indicators():
            raise CandidateBuildError(
                f"add_indicator would leave {new_strategy.unwired_indicators()!r} "
                "unwired: every indicator must be referenced by entry/exit "
                "or a filter. Provide `attach: <filter>` in the payload to "
                "bind the new indicator in the same trial."
            )

        # Write the indicator file first; if it fails we haven't touched
        # strategy.yaml. Source ends with newline for clean diffs.
        self.indicators_dir.mkdir(exist_ok=True)
        source = payload.source if payload.source.endswith("\n") else payload.source + "\n"
        indicator_path.write_text(source, encoding="utf-8")
        self._write_strategy(new_strategy)

        return CandidateApplyResult(
            new_strategy=new_strategy,
            touched_paths=[indicator_path, self.strategy_path],
            edit_summary=EditSummary(
                type=EditType.ADD_INDICATOR,
                target=f"indicators/{payload.fn}.py",
                summary=f"+indicator {payload.name} ({payload.fn}){attach_summary}",
            ),
        )

    # ------------------------------------------------------------------ simplify

    def _apply_simplify(
        self,
        parent: Strategy,
        payload,  # SimplifyPayload (avoid the import-cycle annotation)
    ) -> CandidateApplyResult:
        if payload.component == "indicator":
            return self._simplify_indicator(parent, payload.target)
        if payload.component == "filter":
            return self._simplify_filter(parent, payload.target)
        raise CandidateBuildError(f"unknown simplify component: {payload.component!r}")

    def _simplify_indicator(self, parent: Strategy, name: str) -> CandidateApplyResult:
        match = next((i for i in parent.indicators if i.name == name), None)
        if match is None:
            raise CandidateBuildError(
                f"indicator {name!r} not in strategy; declared: "
                f"{[i.name for i in parent.indicators]}"
            )
        # Refuse to remove if entry/exit/any filter still references it.
        referenced_by: list[str] = []
        if parent.entry.fast == name or parent.entry.slow == name:
            referenced_by.append("entry")
        if parent.exit.fast == name or parent.exit.slow == name:
            referenced_by.append("exit")
        for i, flt in enumerate(parent.filters):
            if flt.indicator == name:
                referenced_by.append(f"filters[{i}]")
        for i, flt in enumerate(parent.exit_filters):
            if flt.indicator == name:
                referenced_by.append(f"exit_filters[{i}]")
        if referenced_by:
            raise CandidateBuildError(
                f"cannot simplify indicator {name!r}: still referenced by "
                f"{referenced_by}; remove those references first"
            )
        # Refuse if it would leave fewer than 1 indicator (entry/exit need at
        # least 2 distinct ones in the current cross-rule DSL).
        new_indicators = [i for i in parent.indicators if i.name != name]
        if len(new_indicators) < 2:
            raise CandidateBuildError(
                f"removing {name!r} would leave only {len(new_indicators)} indicator(s); "
                "cross rules require at least 2"
            )
        new_strategy = parent.model_copy(update={"indicators": new_indicators})
        # If the indicator's fn file is no longer referenced by any spec,
        # delete it. (Multiple specs can share an fn, e.g. sma_fast and sma_slow.)
        deleted_path = None
        still_used = {spec.fn for spec in new_indicators}
        if match.fn not in still_used:
            candidate_path = self.indicators_dir / f"{match.fn}.py"
            if candidate_path.exists():
                candidate_path.unlink()
                deleted_path = candidate_path
        self._write_strategy(new_strategy)
        touched = [self.strategy_path]
        if deleted_path is not None:
            touched.append(deleted_path)
        return CandidateApplyResult(
            new_strategy=new_strategy,
            touched_paths=touched,
            edit_summary=EditSummary(
                type=EditType.SIMPLIFY,
                target="strategy.yaml",
                summary=f"-indicator {name}",
            ),
        )

    def _simplify_filter(self, parent: Strategy, target_index_str: str) -> CandidateApplyResult:
        try:
            idx = int(target_index_str)
        except ValueError as e:
            raise CandidateBuildError(
                f"filter target must be an int index, got {target_index_str!r}"
            ) from e
        if idx < 0 or idx >= len(parent.filters):
            raise CandidateBuildError(
                f"filter index {idx} out of range; have {len(parent.filters)} filter(s)"
            )
        removed = parent.filters[idx]
        new_filters = list(parent.filters)
        del new_filters[idx]
        new_strategy = parent.model_copy(update={"filters": new_filters})
        self._write_strategy(new_strategy)
        return CandidateApplyResult(
            new_strategy=new_strategy,
            touched_paths=[self.strategy_path],
            edit_summary=EditSummary(
                type=EditType.SIMPLIFY,
                target="strategy.yaml",
                summary=f"-filter[{idx}] ({removed.rule}({removed.indicator}))",
            ),
        )

    # ------------------------------------------------------------------ revert_to_trial_N

    def _apply_revert_to_trial(
        self,
        parent: Strategy,
        payload,  # RevertToTrialPayload
    ) -> CandidateApplyResult:
        from lbg.dsl.loader import load_strategy
        from lbg.git_manager import GitCommandError, GitManager

        target_trial = payload.trial_id
        git = GitManager(self.repo_root)
        # Find the commit by subject prefix `trial NNNN:`. GitManager exposes
        # list_recent_commits which gives short SHA + subject -- search there.
        target_subject_prefix = f"trial {target_trial:04d}:"
        log = git.list_recent_commits(limit=500)
        match = next(
            (sha for sha, subject in log if subject.startswith(target_subject_prefix)), None
        )
        if match is None:
            raise CandidateBuildError(
                f"no commit found with subject starting {target_subject_prefix!r}; "
                "trial may not have been committed yet"
            )

        # Restore strategy.yaml + indicators/ from that commit.
        try:
            strategy_at_ref = git.read_file_at_ref(match, "strategy.yaml")
        except GitCommandError as e:
            raise CandidateBuildError(f"could not read strategy.yaml at {match}: {e}") from e

        # Wipe current indicators/ and restore each tracked indicator file.
        # We only restore files that were tracked at the target ref.
        try:
            indicator_files_at_ref = git.list_tree_at_ref(match, "indicators")
        except GitCommandError as e:
            raise CandidateBuildError(f"could not list indicators/ at {match}: {e}") from e

        # Apply: write strategy.yaml, replace indicators directory contents.
        self.strategy_path.write_text(strategy_at_ref, encoding="utf-8")
        # Remove .py files not in the target set (preserve __init__.py).
        target_indicator_files = {
            p for p in indicator_files_at_ref if p.endswith(".py") and p != "indicators/__init__.py"
        }
        for existing in list(self.indicators_dir.glob("*.py")):
            rel = f"indicators/{existing.name}"
            if rel not in target_indicator_files and existing.name != "__init__.py":
                existing.unlink()
        # Write back the target files.
        for rel in target_indicator_files:
            content = git.read_file_at_ref(match, rel)
            (self.repo_root / rel).write_text(content, encoding="utf-8")

        new_strategy = load_strategy(self.strategy_path)
        return CandidateApplyResult(
            new_strategy=new_strategy,
            touched_paths=[self.strategy_path]
            + [self.repo_root / rel for rel in target_indicator_files],
            edit_summary=EditSummary(
                type=EditType.REVERT_TO_TRIAL,
                target="strategy.yaml",
                summary=f"revert to trial {target_trial:04d} ({match})",
            ),
        )

    # ------------------------------------------------------------------ serialization

    def _write_strategy(self, strategy: Strategy) -> None:
        data = strategy.model_dump(mode="python")
        text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True)
        self.strategy_path.write_text(text, encoding="utf-8")


def _set_path(data: dict, path: str, value):
    """Apply a `ParameterChange.path` mutation to `data`. Returns (before, after).

    Supported forms:
      - `sizing.<field>`
      - `indicators[<name>].params.<key>`
      - `filters[<idx>].threshold`   (added (i) -- the live campaign showed
        Editor naturally wants to tune filter thresholds in a follow-up trial)
    """
    if path.startswith("sizing."):
        field = path[len("sizing.") :]
        if "." in field:
            raise CandidateBuildError(f"unsupported nested sizing path: {path!r}")
        if field not in data["sizing"]:
            raise CandidateBuildError(
                f"sizing has no field {field!r}; available: {sorted(data['sizing'])}"
            )
        before = data["sizing"][field]
        data["sizing"][field] = value
        return before, value

    fm = _FILTER_PATH_RE.match(path)
    if fm:
        list_name = fm.group("list")
        idx = int(fm.group("idx"))
        field = fm.group("field")
        filters = data.get(list_name) or []
        if idx < 0 or idx >= len(filters):
            raise CandidateBuildError(
                f"{list_name} index {idx} out of range; strategy has {len(filters)} {list_name}"
            )
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise CandidateBuildError(
                f"filter {field} must be a number; got {type(value).__name__}"
            )
        before = filters[idx].get(field)
        filters[idx][field] = float(value)
        return before, float(value)

    m = _INDICATOR_PATH_RE.match(path)
    if not m:
        raise CandidateBuildError(
            f"unsupported parameter_change path: {path!r}; "
            "expected `sizing.<field>`, `indicators[<name>].params.<key>`, "
            "`filters[<idx>].threshold`, `exit_filters[<idx>].threshold`, "
            "or `exit_filters[<idx>].rearm_threshold`"
        )
    name, key = m.group("name"), m.group("key")
    for spec in data["indicators"]:
        if spec["name"] == name:
            if key not in spec.get("params", {}):
                raise CandidateBuildError(
                    f"indicator {name!r} has no param {key!r}; "
                    f"available: {sorted(spec.get('params', {}))}"
                )
            before = spec["params"][key]
            spec["params"][key] = value
            return before, value
    raise CandidateBuildError(
        f"indicator {name!r} not found in strategy; "
        f"declared: {[s['name'] for s in data['indicators']]}"
    )
