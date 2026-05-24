"""CandidateBuilder: apply a parsed edit to a candidate working tree.

The builder writes files but does **not** commit. The Orchestrator decides
whether to commit (via GitManager.commit_trial) based on the gate verdict.
"""

from lbg.builder.candidate_builder import (
    CandidateApplyResult,
    CandidateBuilder,
    CandidateBuildError,
)

__all__ = [
    "CandidateApplyResult",
    "CandidateBuildError",
    "CandidateBuilder",
]
