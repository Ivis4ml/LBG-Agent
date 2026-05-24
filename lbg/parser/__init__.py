"""ProposalParser: validate Editor YAML output against per-edit-type payloads.

Two-layer validation:
  1. `EditProposal` (lbg.schemas) checks the *outer* structure.
  2. The per-edit-type payload schemas defined here check the *inner* `change`
     block. An ambiguous or malformed payload is rejected before backtest
     (PROPOSAL.html §6.5 -- ProposalParser).
"""

from lbg.parser.payloads import (
    AddFilterPayload,
    AddIndicatorPayload,
    ChangeExitRulePayload,
    ChangeSizingModePayload,
    ParameterChangePayload,
    RemoveFilterPayload,
    RevertToTrialPayload,
    SimplifyPayload,
)
from lbg.parser.proposal_parser import (
    ProposalParseError,
    parse_proposal,
)

__all__ = [
    "AddFilterPayload",
    "AddIndicatorPayload",
    "ChangeExitRulePayload",
    "ChangeSizingModePayload",
    "ParameterChangePayload",
    "ProposalParseError",
    "RemoveFilterPayload",
    "RevertToTrialPayload",
    "SimplifyPayload",
    "parse_proposal",
]
