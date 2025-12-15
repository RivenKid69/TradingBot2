# -*- coding: utf-8 -*-
"""
Change Class Enforcer - Ensures TRADING_IMPACTING changes require approval.

Phase 3 (3.5): No silent upgrades for trading-impacting changes.

Design Doc:
- TRADING_IMPACTING changes ALWAYS require local approval
- Cloud CANNOT bypass approval requirement
- Agent enforces fail-closed: no approval = no execution

CLOUD ZONE ONLY (this module) + AGENT ZONE (separate enforcement).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Final, FrozenSet, List, Optional, Set
from uuid import UUID


class ChangeClass(str, Enum):
    """Change classification per Design Doc."""
    TRADING_IMPACTING = "trading_impacting"
    NON_TRADING_IMPACTING = "non_trading_impacting"
    OPERATIONAL = "operational"
    DOCUMENTATION_ONLY = "documentation_only"


class CommandType(str, Enum):
    """Command types with their default change classes."""
    REQUEST_START_RUN = "REQUEST_START_RUN"
    REQUEST_STOP_RUN = "REQUEST_STOP_RUN"
    REQUEST_PAUSE_RUN = "REQUEST_PAUSE_RUN"
    REQUEST_UPGRADE_ARTIFACT = "REQUEST_UPGRADE_ARTIFACT"
    REQUEST_UPDATE_CONFIG = "REQUEST_UPDATE_CONFIG"
    REQUEST_ROTATE_AGENT_SESSION = "REQUEST_ROTATE_AGENT_SESSION"
    REQUEST_EXPORT_LOGS = "REQUEST_EXPORT_LOGS"


# Commands that are ALWAYS trading-impacting
ALWAYS_TRADING_IMPACTING: Final[FrozenSet[str]] = frozenset([
    "REQUEST_START_RUN",
    "REQUEST_UPGRADE_ARTIFACT",
])

# Commands that MAY be trading-impacting (depends on payload)
POTENTIALLY_TRADING_IMPACTING: Final[FrozenSet[str]] = frozenset([
    "REQUEST_UPDATE_CONFIG",
])

# Commands that are safety/operational (can apply without approval)
# NOTE: These still CANNOT send orders/signals - they only stop/pause
SAFETY_OPERATIONAL: Final[FrozenSet[str]] = frozenset([
    "REQUEST_STOP_RUN",
    "REQUEST_PAUSE_RUN",
])

# Commands that are purely administrative
ADMINISTRATIVE: Final[FrozenSet[str]] = frozenset([
    "REQUEST_ROTATE_AGENT_SESSION",
    "REQUEST_EXPORT_LOGS",
])


@dataclass
class ChangeClassDetermination:
    """Result of change class determination."""
    change_class: ChangeClass
    requires_approval: bool
    reason: str
    command_type: str
    overridden: bool = False  # True if Cloud tried to downgrade
    original_class: Optional[ChangeClass] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "change_class": self.change_class.value,
            "requires_approval": self.requires_approval,
            "reason": self.reason,
            "command_type": self.command_type,
            "overridden": self.overridden,
            "original_class": self.original_class.value if self.original_class else None,
        }


@dataclass
class ApprovalRequirement:
    """Approval requirement for a command."""
    required: bool
    reason: str
    change_class: ChangeClass
    can_be_auto_approved: bool = False  # For Agent-side auto-approve policies
    minimum_approval_level: str = "operator"  # operator, admin, etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "required": self.required,
            "reason": self.reason,
            "change_class": self.change_class.value,
            "can_be_auto_approved": self.can_be_auto_approved,
            "minimum_approval_level": self.minimum_approval_level,
        }


class ChangeClassEnforcer:
    """
    Enforces change class rules on Cloud side.

    Phase 3 (3.5): No silent upgrades.

    Key invariants:
    1. TRADING_IMPACTING ALWAYS requires approval - Cloud CANNOT bypass
    2. Cloud cannot downgrade TRADING_IMPACTING to bypass approval
    3. STOP/PAUSE are safety commands - can execute without approval
    4. Agent has final say - this is Cloud-side pre-enforcement

    Usage:
        enforcer = ChangeClassEnforcer()

        # Determine change class for command
        determination = enforcer.determine_change_class(
            command_type="REQUEST_UPDATE_CONFIG",
            requested_class=ChangeClass.NON_TRADING_IMPACTING,  # Cloud's request
            config_diff_result=diff_result,
        )

        # Enforce approval requirement
        requirement = enforcer.get_approval_requirement(determination)

        if requirement.required:
            # Must route through approval workflow
            command.requires_approval = True
            command.status = "PENDING_APPROVAL"
    """

    def __init__(self, strict_mode: bool = True):
        """
        Initialize enforcer.

        Args:
            strict_mode: If True, always require approval for TRADING_IMPACTING
        """
        self._strict_mode = strict_mode

    def determine_change_class(
        self,
        command_type: str,
        requested_class: Optional[ChangeClass] = None,
        config_has_trading_impact: bool = False,
        artifact_change_class: Optional[str] = None,
    ) -> ChangeClassDetermination:
        """
        Determine the effective change class for a command.

        Cloud can REQUEST a change class, but certain commands
        CANNOT be downgraded from TRADING_IMPACTING.

        Args:
            command_type: Type of command
            requested_class: Cloud's requested change class
            config_has_trading_impact: True if config diff shows trading impact
            artifact_change_class: Change class from artifact manifest

        Returns:
            ChangeClassDetermination
        """
        cmd_type = command_type.upper()

        # Start with the requested class or default
        effective_class = requested_class or ChangeClass.OPERATIONAL
        overridden = False
        original_class = None

        # Rule 1: Always trading-impacting commands
        if cmd_type in ALWAYS_TRADING_IMPACTING:
            if effective_class != ChangeClass.TRADING_IMPACTING:
                overridden = True
                original_class = effective_class
            effective_class = ChangeClass.TRADING_IMPACTING
            reason = f"{cmd_type} is always TRADING_IMPACTING per Design Doc"

        # Rule 2: Config updates - check actual impact
        elif cmd_type in POTENTIALLY_TRADING_IMPACTING:
            if config_has_trading_impact:
                if effective_class != ChangeClass.TRADING_IMPACTING:
                    overridden = True
                    original_class = effective_class
                effective_class = ChangeClass.TRADING_IMPACTING
                reason = "Config changes include trading-impacting fields"
            else:
                reason = "Config changes are non-trading-impacting"

        # Rule 3: Safety/operational commands
        elif cmd_type in SAFETY_OPERATIONAL:
            # These reduce risk - can be operational
            effective_class = ChangeClass.OPERATIONAL
            reason = f"{cmd_type} is a safety command (reduces risk)"

        # Rule 4: Administrative commands
        elif cmd_type in ADMINISTRATIVE:
            effective_class = ChangeClass.OPERATIONAL
            reason = f"{cmd_type} is administrative"

        else:
            # Unknown command - default to trading-impacting for safety
            effective_class = ChangeClass.TRADING_IMPACTING
            reason = f"Unknown command type {cmd_type} - defaulting to TRADING_IMPACTING"

        # Consider artifact manifest change_class
        if artifact_change_class == "trading_impacting":
            if effective_class != ChangeClass.TRADING_IMPACTING:
                overridden = True
                original_class = effective_class
            effective_class = ChangeClass.TRADING_IMPACTING
            reason += " (artifact manifest declares trading_impacting)"

        return ChangeClassDetermination(
            change_class=effective_class,
            requires_approval=effective_class == ChangeClass.TRADING_IMPACTING,
            reason=reason,
            command_type=cmd_type,
            overridden=overridden,
            original_class=original_class,
        )

    def get_approval_requirement(
        self,
        determination: ChangeClassDetermination,
    ) -> ApprovalRequirement:
        """
        Get approval requirement based on change class.

        Args:
            determination: Change class determination

        Returns:
            ApprovalRequirement
        """
        if determination.change_class == ChangeClass.TRADING_IMPACTING:
            return ApprovalRequirement(
                required=True,
                reason=determination.reason,
                change_class=determination.change_class,
                can_be_auto_approved=False,  # Never for trading-impacting
                minimum_approval_level="operator",
            )

        elif determination.change_class == ChangeClass.OPERATIONAL:
            # Safety commands - no approval needed
            return ApprovalRequirement(
                required=False,
                reason=determination.reason,
                change_class=determination.change_class,
                can_be_auto_approved=True,
            )

        else:
            # Non-trading-impacting - may be auto-approved
            return ApprovalRequirement(
                required=False,
                reason=determination.reason,
                change_class=determination.change_class,
                can_be_auto_approved=True,
            )

    def validate_command_approval_state(
        self,
        command_type: str,
        change_class: ChangeClass,
        requires_approval: bool,
    ) -> tuple[bool, str]:
        """
        Validate that command approval state is consistent.

        Cloud CANNOT create TRADING_IMPACTING commands without requires_approval=True.

        Args:
            command_type: Command type
            change_class: Declared change class
            requires_approval: Whether approval is required

        Returns:
            Tuple of (is_valid, error_message)
        """
        if change_class == ChangeClass.TRADING_IMPACTING and not requires_approval:
            return False, (
                f"Command {command_type} with change_class=TRADING_IMPACTING "
                "MUST have requires_approval=True. "
                "Cloud CANNOT bypass approval for trading-impacting changes. "
                "(Phase 3, 3.5: No silent upgrades)"
            )

        return True, ""


class AgentChangeClassEnforcer:
    """
    Agent-side enforcement of change class rules.

    This runs on the Agent and is the FINAL authority.
    Even if Cloud sends a command, Agent enforces approval requirements.

    Key invariants:
    1. TRADING_IMPACTING without approval attestation = REJECT
    2. Unknown change_class = treat as TRADING_IMPACTING
    3. Missing approval evidence for required approval = REJECT
    """

    def __init__(self, fail_closed: bool = True):
        """
        Initialize agent enforcer.

        Args:
            fail_closed: If True, reject uncertain cases
        """
        self._fail_closed = fail_closed

    def validate_command(
        self,
        command_type: str,
        change_class: str,
        requires_approval: bool,
        has_approval: bool,
        approval_evidence_hash: Optional[str] = None,
    ) -> tuple[bool, str]:
        """
        Validate command on Agent side.

        Args:
            command_type: Command type
            change_class: Declared change class
            requires_approval: Whether approval is required
            has_approval: Whether command has been approved
            approval_evidence_hash: Hash of approval evidence

        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        # Normalize change class
        try:
            cc = ChangeClass(change_class.lower())
        except ValueError:
            if self._fail_closed:
                return False, f"Unknown change_class '{change_class}' - treating as TRADING_IMPACTING requires approval"
            cc = ChangeClass.TRADING_IMPACTING

        # Rule 1: TRADING_IMPACTING ALWAYS requires approval
        if cc == ChangeClass.TRADING_IMPACTING:
            if not requires_approval:
                return False, (
                    f"INVARIANT VIOLATION: {command_type} is TRADING_IMPACTING "
                    "but requires_approval=False. Agent rejects."
                )
            if not has_approval:
                return False, (
                    f"{command_type} is TRADING_IMPACTING and requires local approval. "
                    "Awaiting operator approval."
                )
            if not approval_evidence_hash:
                return False, (
                    f"{command_type} approved but missing evidence hash. "
                    "Cannot proceed without audit trail."
                )

        # Rule 2: Commands that should require approval but don't
        cmd_upper = command_type.upper()
        if cmd_upper in ALWAYS_TRADING_IMPACTING and cc != ChangeClass.TRADING_IMPACTING:
            if self._fail_closed:
                return False, (
                    f"{command_type} MUST be TRADING_IMPACTING but received {change_class}. "
                    "Potential bypass attempt. Agent rejects."
                )

        return True, ""


def enforce_trading_impacting_approval(
    command_type: str,
    change_class: str,
    requires_approval: bool,
) -> tuple[bool, bool, str]:
    """
    Convenience function to enforce TRADING_IMPACTING approval.

    Returns corrected requires_approval and any warnings.

    Args:
        command_type: Command type
        change_class: Declared change class
        requires_approval: Requested approval requirement

    Returns:
        Tuple of (corrected_requires_approval, was_corrected, message)
    """
    enforcer = ChangeClassEnforcer()

    determination = enforcer.determine_change_class(
        command_type=command_type,
        requested_class=ChangeClass(change_class) if change_class else None,
    )

    if determination.change_class == ChangeClass.TRADING_IMPACTING:
        if not requires_approval:
            return True, True, (
                f"Corrected: {command_type} is TRADING_IMPACTING, "
                "requires_approval set to True (Phase 3, 3.5)"
            )
        return True, False, ""

    return requires_approval, False, ""
