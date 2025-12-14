# -*- coding: utf-8 -*-
"""
CCEA Shared Contract Enums.

Single source of truth for all enum values and state machines.
Schema (protocol_messages.schema.json) is the authoritative source.

This module defines:
1. Protocol-side enums (UPPERCASE, used in messages)
2. DB-side enums (lowercase, stored in PostgreSQL)
3. Mappings between protocol and DB representations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Final, FrozenSet, List, Optional, Set

# ============================================================================
# Command Types (Schema is authoritative)
# ============================================================================

# Allowed command types from protocol_messages.schema.json
COMMAND_TYPES: Final[FrozenSet[str]] = frozenset({
    "REQUEST_START_RUN",
    "REQUEST_STOP_RUN",
    "REQUEST_PAUSE_RUN",
    "REQUEST_UPGRADE_ARTIFACT",
    "REQUEST_UPDATE_CONFIG",
    "REQUEST_ROTATE_AGENT_SESSION",
    "REQUEST_EXPORT_LOGS",
})


@dataclass
class CommandTypeContract:
    """Command type contract with metadata."""
    name: str
    requires_approval: bool
    change_class: str  # "TRADING_IMPACTING" | "NON_TRADING_IMPACTING" | "OPERATIONAL"
    is_safety_operation: bool = False

    @classmethod
    def get_all(cls) -> List["CommandTypeContract"]:
        """Get all command type contracts."""
        return [
            cls("REQUEST_START_RUN", True, "TRADING_IMPACTING"),
            cls("REQUEST_STOP_RUN", False, "OPERATIONAL", is_safety_operation=True),
            cls("REQUEST_PAUSE_RUN", False, "OPERATIONAL", is_safety_operation=True),
            cls("REQUEST_UPGRADE_ARTIFACT", True, "TRADING_IMPACTING"),
            cls("REQUEST_UPDATE_CONFIG", True, "TRADING_IMPACTING"),  # if trading-impacting
            cls("REQUEST_ROTATE_AGENT_SESSION", True, "OPERATIONAL"),
            cls("REQUEST_EXPORT_LOGS", True, "DATA_SENSITIVE"),
        ]


# ============================================================================
# Message Types (Schema is authoritative)
# ============================================================================

MESSAGE_TYPES: Final[FrozenSet[str]] = frozenset({
    "HEARTBEAT",
    "POLL_COMMANDS",
    "COMMAND_BATCH",
    "COMMAND_ACK",
    "COMMAND_APPROVAL",
    "COMMAND_RESULT",
    "TELEMETRY",
})


# ============================================================================
# Command Status
# ============================================================================

# Protocol-side (UPPERCASE) - from protocol_messages.schema.json
COMMAND_STATUS_PROTOCOL: Final[FrozenSet[str]] = frozenset({
    "PENDING",
    "RECEIVED",
    "AWAITING_APPROVAL",
    "APPROVED",
    "REJECTED",
    "EXECUTING",
    "COMPLETED",
    "FAILED",
})

# DB-side (lowercase) - from packages/cloud/control_plane/models.py
COMMAND_STATUS_DB: Final[FrozenSet[str]] = frozenset({
    "pending",
    "sent",
    "acknowledged",
    "pending_approval",
    "approved",
    "rejected",
    "executed",
    "failed",
    "expired",
})


class CommandStatusMapping:
    """
    Mapping between protocol and DB command status values.

    Protocol uses agent-perspective states (RECEIVED, EXECUTING, COMPLETED).
    DB uses cloud-perspective states (sent, acknowledged, executed).
    """

    # Protocol -> DB mapping (agent reports to cloud)
    PROTOCOL_TO_DB: Final[Dict[str, str]] = {
        "PENDING": "pending",
        "RECEIVED": "acknowledged",
        "AWAITING_APPROVAL": "pending_approval",
        "APPROVED": "approved",
        "REJECTED": "rejected",
        "EXECUTING": "acknowledged",  # Still in "acknowledged" from cloud view
        "COMPLETED": "executed",
        "FAILED": "failed",
    }

    # DB -> Protocol mapping (cloud status to agent view)
    DB_TO_PROTOCOL: Final[Dict[str, str]] = {
        "pending": "PENDING",
        "sent": "PENDING",  # Agent hasn't received yet
        "acknowledged": "RECEIVED",
        "pending_approval": "AWAITING_APPROVAL",
        "approved": "APPROVED",
        "rejected": "REJECTED",
        "executed": "COMPLETED",
        "failed": "FAILED",
        "expired": "FAILED",  # Expired treated as failed
    }

    @classmethod
    def to_db(cls, protocol_status: str) -> str:
        """Convert protocol status to DB status."""
        return cls.PROTOCOL_TO_DB.get(protocol_status, "pending")

    @classmethod
    def to_protocol(cls, db_status: str) -> str:
        """Convert DB status to protocol status."""
        return cls.DB_TO_PROTOCOL.get(db_status, "PENDING")


# ============================================================================
# Deployment State
# ============================================================================

# Protocol-side (from protocol_messages.schema.json definitions.deployment_state)
DEPLOYMENT_STATE_PROTOCOL: Final[FrozenSet[str]] = frozenset({
    "CREATED",
    "PENDING",
    "ENROLLED",
    "RUNNING",
    "PAUSED",
    "STOPPED",
    "REVOKED",
    "TERMINATED",
})

# DB-side (from packages/cloud/control_plane/models.py DeploymentState)
DEPLOYMENT_STATE_DB: Final[FrozenSet[str]] = frozenset({
    "created",
    "pending_approval",
    "approved",
    "deploying",
    "deployed",
    "suspended",
    "failed",
    "retired",
})


class DeploymentStateMapping:
    """
    Mapping between protocol and DB deployment states.

    Protocol uses agent-side states (ENROLLED, RUNNING, STOPPED).
    DB uses cloud-side states (deploying, deployed, retired).
    """

    PROTOCOL_TO_DB: Final[Dict[str, str]] = {
        "CREATED": "created",
        "PENDING": "pending_approval",
        "ENROLLED": "deployed",
        "RUNNING": "deployed",  # Running is a run state, not deployment
        "PAUSED": "deployed",  # Still deployed, just paused
        "STOPPED": "deployed",  # Still deployed, just stopped
        "REVOKED": "retired",
        "TERMINATED": "retired",
    }

    DB_TO_PROTOCOL: Final[Dict[str, str]] = {
        "created": "CREATED",
        "pending_approval": "PENDING",
        "approved": "PENDING",
        "deploying": "PENDING",
        "deployed": "ENROLLED",
        "suspended": "PAUSED",
        "failed": "TERMINATED",
        "retired": "TERMINATED",
    }

    @classmethod
    def to_db(cls, protocol_state: str) -> str:
        """Convert protocol state to DB state."""
        return cls.PROTOCOL_TO_DB.get(protocol_state, "created")

    @classmethod
    def to_protocol(cls, db_state: str) -> str:
        """Convert DB state to protocol state."""
        return cls.DB_TO_PROTOCOL.get(db_state, "CREATED")


# ============================================================================
# Run State
# ============================================================================

# Protocol-side (from protocol_messages.schema.json definitions.run_state)
RUN_STATE_PROTOCOL: Final[FrozenSet[str]] = frozenset({
    "INITIALIZING",
    "RUNNING",
    "PAUSED",
    "HALTED",
    "STOPPED",
})

# DB-side (from packages/cloud/control_plane/models.py RunState)
RUN_STATE_DB: Final[FrozenSet[str]] = frozenset({
    "created",
    "pending_approval",
    "approved",
    "starting",
    "running",
    "paused",
    "stopping",
    "stopped",
    "failed",
    "completed",
})


class RunStateMapping:
    """
    Mapping between protocol and DB run states.

    Protocol uses simplified agent states (INITIALIZING, HALTED).
    DB uses detailed cloud tracking states (pending_approval, stopping).
    """

    PROTOCOL_TO_DB: Final[Dict[str, str]] = {
        "INITIALIZING": "starting",
        "RUNNING": "running",
        "PAUSED": "paused",
        "HALTED": "stopped",  # HALTED is emergency stop
        "STOPPED": "stopped",
    }

    DB_TO_PROTOCOL: Final[Dict[str, str]] = {
        "created": "INITIALIZING",
        "pending_approval": "INITIALIZING",
        "approved": "INITIALIZING",
        "starting": "INITIALIZING",
        "running": "RUNNING",
        "paused": "PAUSED",
        "stopping": "RUNNING",  # Still running while stopping
        "stopped": "STOPPED",
        "failed": "HALTED",
        "completed": "STOPPED",
    }

    @classmethod
    def to_db(cls, protocol_state: str) -> str:
        """Convert protocol state to DB state."""
        return cls.PROTOCOL_TO_DB.get(protocol_state, "created")

    @classmethod
    def to_protocol(cls, db_state: str) -> str:
        """Convert DB state to protocol state."""
        return cls.DB_TO_PROTOCOL.get(db_state, "INITIALIZING")


# ============================================================================
# Approval Status
# ============================================================================

APPROVAL_STATUS: Final[FrozenSet[str]] = frozenset({
    "APPROVED",
    "REJECTED",
    "TIMEOUT",
})


# ============================================================================
# Change Class
# ============================================================================

CHANGE_CLASS: Final[FrozenSet[str]] = frozenset({
    "TRADING_IMPACTING",
    "NON_TRADING_IMPACTING",
})

# DB uses lowercase with additional categories
CHANGE_CLASS_DB: Final[FrozenSet[str]] = frozenset({
    "operational",
    "trading_impacting",
    "security_sensitive",
    "data_sensitive",
})


# ============================================================================
# Telemetry Level
# ============================================================================

TELEMETRY_LEVEL_PROTOCOL: Final[FrozenSet[str]] = frozenset({
    "AGGREGATED",
    "DETAILED_NON_SENSITIVE",
    "RAW_ORDER_EVENTS",
})

TELEMETRY_LEVEL_DB: Final[FrozenSet[str]] = frozenset({
    "aggregated",
    "detailed_non_sensitive",
    "raw_order_events",
})


# ============================================================================
# Signature Algorithms
# ============================================================================

SIGNATURE_ALGORITHMS: Final[FrozenSet[str]] = frozenset({
    "ed25519",
    "ecdsa-p256",
    "rsa-pss",
})


# ============================================================================
# Helper Functions
# ============================================================================

def get_all_protocol_enums() -> Dict[str, FrozenSet[str]]:
    """Get all protocol-side enum sets."""
    return {
        "command_types": COMMAND_TYPES,
        "message_types": MESSAGE_TYPES,
        "command_status": COMMAND_STATUS_PROTOCOL,
        "deployment_state": DEPLOYMENT_STATE_PROTOCOL,
        "run_state": RUN_STATE_PROTOCOL,
        "approval_status": APPROVAL_STATUS,
        "change_class": CHANGE_CLASS,
        "telemetry_level": TELEMETRY_LEVEL_PROTOCOL,
        "signature_algorithms": SIGNATURE_ALGORITHMS,
    }


def get_all_db_enums() -> Dict[str, FrozenSet[str]]:
    """Get all DB-side enum sets."""
    return {
        "command_status": COMMAND_STATUS_DB,
        "deployment_state": DEPLOYMENT_STATE_DB,
        "run_state": RUN_STATE_DB,
        "change_class": CHANGE_CLASS_DB,
        "telemetry_level": TELEMETRY_LEVEL_DB,
    }
