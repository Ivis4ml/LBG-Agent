"""Regression test: alpha-card `source_commit` must point at the true
incumbent commit, never at a rejected trial's commit.

Bug it covers (codex review, Issue 1): every trial -- accepted OR rejected --
commits at the end of `_run_one_trial`, advancing HEAD. The old code read
`parent_commit = self.git.head_sha()` at record time, so after a rejected
trial the *next* trial recorded the rejected commit as its parent. That
parent is what `source_commit` materializes as the "without card" baseline
in sealed per-card validation, so a `reject -> accept add_indicator`
sequence measured the card's increment against `incumbent + rejected_edit`
instead of the pure incumbent.

The fix tracks `incumbent_commit_sha` in `run()`, refreshed only after a
successful accept commit, and uses it for `parent_commit`. These tests pin
the recorded `parent_commit` in trials.jsonl, which is exactly the value
fed to alpha-card `source_commit` (discovery.py: `source_commit=parent_commit`).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from lbg.gate.decision import (
    REASON_NO_MEANINGFUL_IMPROVEMENT,
    REASON_UTILITY_IMPROVEMENT,
    GateDecision,
)
from lbg.memory.records import AgentComputeRecord, ReflectionRecord
from lbg.orchestrator import RoleRunner
from lbg.orchestrator.discovery import Discovery
from lbg.orchestrator.role_runner import EditorRunResult, ReflectorRunResult
from lbg.parser.payloads import ParameterChangePayload
from lbg.schemas import (
    EditProposal,
    EditType,
    ExpectedTrainSignal,
    ExpectedValidationSignal,
    ProposedEdit,
)
from lbg.sealed_vault import SealedVault

REPO = Path(__file__).resolve().parents[1]

_PARAM_PATH = "indicators[sma_fast].params.period"


class _ParamChangeRunner(RoleRunner):
    """Emits the same parameter_change every trial (no .py, so AST/prefix
    checks are skipped) and a trivial reflection. The gate decision is
    forced by monkeypatching `decide`, so the value need only be valid."""

    def __init__(self, value: int = 7):
        super().__init__(api_key="dummy", client=object())  # client never used
        self.value = value
        self.editor_calls: list[int] = []

    def editor(self, context, *, trial_id):  # noqa: ARG002
        self.editor_calls.append(trial_id)
        proposal = EditProposal(
            trial_id=trial_id,
            hypothesis="nudge sma_fast period",
            proposed_edit=ProposedEdit(
                type=EditType.PARAMETER_CHANGE,
                change={"path": _PARAM_PATH, "value": self.value},
            ),
            expected_train_signal=ExpectedTrainSignal.NEUTRAL,
            expected_validation_signal=ExpectedValidationSignal.REJECT,
        )
        payload = ParameterChangePayload(path=_PARAM_PATH, value=self.value)
        compute = AgentComputeRecord(
            trial_id=trial_id,
            role="editor",
            model="stub",
            input_tokens=1,
            output_tokens=1,
            wall_clock_sec=0.0,
        )
        return EditorRunResult(proposal=proposal, payload=payload, raw_text="stub", compute=compute)

    def reflector(
        self,
        *,
        trial_id,
        editor_context,  # noqa: ARG002
        proposal,  # noqa: ARG002
        hypothesis_outcome,
        actual_validation_signal,  # noqa: ARG002
        actual_train_metrics,  # noqa: ARG002
    ):
        record = ReflectionRecord(
            trial_id=trial_id,
            explanation="stub",
            hypothesis_outcome=hypothesis_outcome,
        )
        compute = AgentComputeRecord(
            trial_id=trial_id,
            role="reflector",
            model="stub",
            input_tokens=1,
            output_tokens=1,
            wall_clock_sec=0.0,
        )
        return ReflectorRunResult(record=record, raw_text="stub", compute=compute)


def _gate_factory(*, accepted: bool):
    reason = REASON_UTILITY_IMPROVEMENT if accepted else REASON_NO_MEANINGFUL_IMPROVEMENT

    def _decide(*_args, **_kwargs):
        return GateDecision(
            accepted=accepted,
            reason=reason,
            candidate_utility_lcb=0.0,
            incumbent_utility_lcb=0.0,
            complexity_delta=0.0,
        )

    return _decide


@pytest.fixture
def working_tree(tmp_path):
    shutil.copy(REPO / "strategy.yaml", tmp_path / "strategy.yaml")
    (tmp_path / "indicators").mkdir()
    shutil.copy(REPO / "indicators" / "sma.py", tmp_path / "indicators" / "sma.py")
    shutil.copy(REPO / "indicators" / "__init__.py", tmp_path / "indicators" / "__init__.py")
    (tmp_path / "data").symlink_to(REPO / "data")
    (tmp_path / "memory").mkdir()
    (tmp_path / "skills").mkdir()

    subprocess.run(
        ["git", "init", "--initial-branch=main", "-q"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("test\n")
    subprocess.run(
        ["git", "add", "README.md", "strategy.yaml", "indicators/"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "baseline"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    return tmp_path


def _head_sha(cwd: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parent_commits(working_tree: Path) -> dict[int, str]:
    lines = (working_tree / "memory" / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    out: dict[int, str] = {}
    for line in lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["trial_id"]] = rec["parent_commit"]
    return out


def test_rejected_trials_keep_incumbent_parent(working_tree, monkeypatch):
    """Two consecutive rejected trials: both must record the run-start SHA
    as parent_commit. With the bug, trial 2's parent is trial 1's (rejected)
    commit because HEAD advanced."""
    monkeypatch.setattr("lbg.orchestrator.discovery.decide", _gate_factory(accepted=False))
    start_sha = _head_sha(working_tree)

    disc = Discovery(repo_root=working_tree, runner=_ParamChangeRunner())
    vault = SealedVault(working_tree / "vault.json")
    res = disc.run(budget=2, seal_at_end=False, vault=vault, report_path=None)

    assert res.n_rejected == 2
    parents = _parent_commits(working_tree)
    assert len(parents) == 2
    # HEAD advanced (each reject committed) ...
    assert _head_sha(working_tree) != start_sha
    # ... yet every rejected trial's recorded parent is the unchanged incumbent.
    assert all(sha == start_sha for sha in parents.values()), parents


def test_accept_advances_incumbent_parent(working_tree, monkeypatch):
    """First trial accepts; the second trial's parent must be the first
    trial's commit (the refreshed incumbent), not the run-start SHA."""
    monkeypatch.setattr("lbg.orchestrator.discovery.decide", _gate_factory(accepted=True))
    start_sha = _head_sha(working_tree)

    disc = Discovery(repo_root=working_tree, runner=_ParamChangeRunner())
    vault = SealedVault(working_tree / "vault.json")
    res = disc.run(budget=2, seal_at_end=False, vault=vault, report_path=None)

    assert res.n_accepted == 2
    parents = _parent_commits(working_tree)
    # Trial 1 branched off the run-start incumbent.
    trial_ids = sorted(parents)
    first, second = trial_ids[0], trial_ids[1]
    assert parents[first] == start_sha
    # Trial 2 branched off trial 1's accept commit -- incumbent refreshed.
    assert parents[second] != start_sha
    assert parents[second] != parents[first]
