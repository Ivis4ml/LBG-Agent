"""Two-layer parser for Editor YAML proposals."""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

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
from lbg.schemas import EditProposal, EditType


class ProposalParseError(ValueError):
    """Raised when the Editor's YAML output cannot be parsed/validated."""


_PAYLOAD_SCHEMAS: dict[EditType, type[BaseModel]] = {
    EditType.ADD_INDICATOR: AddIndicatorPayload,
    EditType.PARAMETER_CHANGE: ParameterChangePayload,
    EditType.ADD_FILTER: AddFilterPayload,
    EditType.REMOVE_FILTER: RemoveFilterPayload,
    EditType.CHANGE_SIZING_MODE: ChangeSizingModePayload,
    EditType.CHANGE_EXIT_RULE: ChangeExitRulePayload,
    EditType.SIMPLIFY: SimplifyPayload,
    EditType.REVERT_TO_TRIAL: RevertToTrialPayload,
}


def parse_proposal(source: str | dict[str, Any]) -> tuple[EditProposal, BaseModel]:
    """Parse the Editor's output and return `(proposal, payload)`.

    `source` may be the raw YAML text or an already-parsed dict.

    Raises
    ------
    ProposalParseError
        On any failure: malformed YAML, wrong outer structure, unknown edit
        type, or an inner `change` payload that doesn't match the schema for
        the declared edit type.
    """
    if isinstance(source, str):
        try:
            data = yaml.safe_load(source)
        except yaml.YAMLError as e:
            raise ProposalParseError(f"YAML parse error: {e}") from e
    else:
        data = source

    if not isinstance(data, dict):
        raise ProposalParseError(f"proposal must be a mapping, got {type(data).__name__}")

    try:
        proposal = EditProposal.model_validate(data)
    except ValidationError as e:
        raise ProposalParseError(f"EditProposal validation failed: {e}") from e

    payload_cls = _PAYLOAD_SCHEMAS.get(proposal.proposed_edit.type)
    if payload_cls is None:
        raise ProposalParseError(
            f"no payload schema registered for edit type {proposal.proposed_edit.type!r}"
        )

    change_block = proposal.proposed_edit.change or {}
    if not isinstance(change_block, dict):
        raise ProposalParseError(
            f"`proposed_edit.change` must be a mapping, got {type(change_block).__name__}"
        )

    try:
        payload = payload_cls.model_validate(change_block)
    except ValidationError as e:
        raise ProposalParseError(
            f"`change` payload invalid for {proposal.proposed_edit.type.value}: {e}"
        ) from e

    return proposal, payload
