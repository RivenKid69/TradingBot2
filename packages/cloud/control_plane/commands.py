# -*- coding: utf-8 -*-
"""
Command Dispatcher - DEPRECATED, use services.command_service instead.

CLOUD ZONE ONLY.

Phase 7 (WI-CLOUD-03): This in-memory dispatcher is DEPRECATED.
Use packages.cloud.control_plane.services.command_service.CommandService
for production use. The DB-backed service provides:
- Persistent storage
- Tenant isolation via RLS
- Long-poll support for agents
- Approval workflow integration

This file is kept for backwards compatibility and reference.
The types (CommandType, CommandStatus, etc.) are re-exported from here
for existing code that imports them.

Commands are LIFECYCLE requests only:
- Start/Stop/Pause runs
- Upgrade artifacts
- Update configuration
- Rotate sessions
- Export logs

PROHIBITED:
- Order-like payloads (side, qty, price, target_position)
- Direct trading instructions
- Signal/intent transmission
"""
from __future__ import annotations

import warnings

# Deprecation warning - module level (only when explicitly imported)
def _warn_deprecated():
    warnings.warn(
        "commands.CommandDispatcher is deprecated. "
        "Use services.command_service.CommandService instead.",
        DeprecationWarning,
        stacklevel=3,
    )

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Final, FrozenSet, List, Optional, Set
from uuid import UUID, uuid4


class CommandType(str, Enum):
    """
    Allowed command types.

    These are the ONLY commands Cloud can send.
    Adding new command types requires security review.

    NOTE: REQUEST_RESUME_RUN was removed per Design Doc - not in allowed command list.
          Use REQUEST_START_RUN to restart a stopped run.
    """

    REQUEST_START_RUN = "REQUEST_START_RUN"
    REQUEST_STOP_RUN = "REQUEST_STOP_RUN"
    REQUEST_PAUSE_RUN = "REQUEST_PAUSE_RUN"
    REQUEST_UPGRADE_ARTIFACT = "REQUEST_UPGRADE_ARTIFACT"
    REQUEST_UPDATE_CONFIG = "REQUEST_UPDATE_CONFIG"
    REQUEST_ROTATE_AGENT_SESSION = "REQUEST_ROTATE_AGENT_SESSION"
    REQUEST_EXPORT_LOGS = "REQUEST_EXPORT_LOGS"


class CommandStatus(str, Enum):
    """Command status."""

    PENDING = "pending"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    APPROVED = "approved"  # Locally approved
    EXECUTED = "executed"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


# Allowed command types (frozen for security)
ALLOWED_COMMAND_TYPES: Final[FrozenSet[str]] = frozenset(
    [ct.value for ct in CommandType]
)

# Fields that are PROHIBITED in command payloads
PROHIBITED_PAYLOAD_FIELDS: Final[FrozenSet[str]] = frozenset(
    [
        "side",
        "quantity",
        "qty",
        "price",
        "limit_price",
        "stop_price",
        "order_type",
        "target_position",
        "execute_order",
        "place_order",
        "submit_order",
        "intent",
        "signal",
        "trade",
        "order",
    ]
)


class CommandValidationError(Exception):
    """Command validation failed."""

    pass


@dataclass
class Command:
    """
    Lifecycle command to send to agent.
    """

    command_id: UUID = field(default_factory=uuid4)
    command_type: CommandType = CommandType.REQUEST_STOP_RUN
    idempotency_key: str = ""

    # Target
    agent_id: str = ""
    deployment_id: Optional[str] = None
    run_id: Optional[str] = None

    # Payload (must not contain order-like fields)
    payload: Dict[str, Any] = field(default_factory=dict)
    payload_digest: str = ""  # SHA256 of payload

    # Approval
    requires_approval: bool = True
    change_class: str = "operational"

    # Status
    status: CommandStatus = CommandStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    sent_at: Optional[datetime] = None
    acked_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None

    # Result
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        """Validate command after creation."""
        self._validate()

    def _validate(self) -> None:
        """Validate command doesn't contain prohibited fields."""
        # Check command type
        if self.command_type.value not in ALLOWED_COMMAND_TYPES:
            raise CommandValidationError(
                f"Command type not allowed: {self.command_type}"
            )

        # Check payload for prohibited fields
        violations = self._find_prohibited_fields(self.payload)
        if violations:
            raise CommandValidationError(
                f"Prohibited fields in payload: {violations}"
            )

    def _find_prohibited_fields(
        self,
        obj: Any,
        path: str = "",
    ) -> List[str]:
        """Recursively find prohibited fields."""
        violations = []

        if isinstance(obj, dict):
            for key, value in obj.items():
                key_lower = key.lower()
                if key_lower in PROHIBITED_PAYLOAD_FIELDS:
                    violations.append(f"{path}.{key}" if path else key)
                # Recurse
                violations.extend(
                    self._find_prohibited_fields(value, f"{path}.{key}" if path else key)
                )
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                violations.extend(
                    self._find_prohibited_fields(item, f"{path}[{i}]")
                )

        return violations

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "command_id": str(self.command_id),
            "command_type": self.command_type.value,
            "idempotency_key": self.idempotency_key,
            "agent_id": self.agent_id,
            "deployment_id": self.deployment_id,
            "run_id": self.run_id,
            "payload": self.payload,
            "payload_digest": self.payload_digest,
            "requires_approval": self.requires_approval,
            "change_class": self.change_class,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "acked_at": self.acked_at.isoformat() if self.acked_at else None,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "result": self.result,
            "error_message": self.error_message,
        }


@dataclass
class CommandResult:
    """Result of command execution."""

    command_id: UUID
    success: bool
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "command_id": str(self.command_id),
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


class CommandDispatcher:
    """
    Dispatches lifecycle commands to agents.

    IMPORTANT: This dispatcher ONLY sends lifecycle commands.
    It validates that no order-like payloads are sent.

    Usage:
        dispatcher = CommandDispatcher()

        # Create command
        cmd = dispatcher.create_command(
            command_type=CommandType.REQUEST_START_RUN,
            agent_id="agent-123",
            payload={"config_digest": "sha256:abc..."},
        )

        # Dispatch
        result = dispatcher.dispatch(cmd)
    """

    def __init__(self):
        """Initialize dispatcher."""
        self._commands: Dict[UUID, Command] = {}
        self._by_idempotency_key: Dict[str, Command] = {}

    def create_command(
        self,
        command_type: CommandType,
        agent_id: str,
        payload: Optional[Dict[str, Any]] = None,
        deployment_id: Optional[str] = None,
        run_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Command:
        """
        Create a new command.

        Args:
            command_type: Type of command
            agent_id: Target agent ID
            payload: Command payload (validated for prohibited fields)
            deployment_id: Target deployment
            run_id: Target run
            idempotency_key: Key for deduplication

        Returns:
            Created command

        Raises:
            CommandValidationError: If payload contains prohibited fields
        """
        # Generate idempotency key if not provided
        if not idempotency_key:
            idempotency_key = f"{agent_id}:{command_type.value}:{uuid4()}"

        # Check for duplicate by idempotency key
        if idempotency_key in self._by_idempotency_key:
            return self._by_idempotency_key[idempotency_key]

        # Compute payload digest
        from packages.shared.utils.hashing import compute_content_hash

        payload = payload or {}
        payload_digest = compute_content_hash(payload, with_prefix=True)

        # Determine if approval required
        requires_approval = command_type in (
            CommandType.REQUEST_START_RUN,
            CommandType.REQUEST_UPGRADE_ARTIFACT,
            CommandType.REQUEST_UPDATE_CONFIG,
        )
        change_class = "trading_impacting" if requires_approval else "operational"

        # Create command (validation happens in __post_init__)
        cmd = Command(
            command_type=command_type,
            agent_id=agent_id,
            payload=payload,
            payload_digest=payload_digest,
            deployment_id=deployment_id,
            run_id=run_id,
            idempotency_key=idempotency_key,
            requires_approval=requires_approval,
            change_class=change_class,
        )

        # Store
        self._commands[cmd.command_id] = cmd
        self._by_idempotency_key[idempotency_key] = cmd

        return cmd

    def dispatch(self, command: Command) -> bool:
        """
        Dispatch command to agent.

        RELIABILITY: Tech Debt Tracking CCEA-OPS-001
        Status: CONTROLLED - follows CCEA polling architecture

        Per CCEA Design Doc Section 10.1, agents use outbound polling
        (HTTPS long-poll or WebSocket) to fetch commands. Cloud stores
        commands with PENDING status; agents poll via POLL_COMMANDS.

        Current implementation:
        - Marks command as SENT (ready for agent pickup)
        - Agent polls /api/v1/commands endpoint to retrieve
        - Actual delivery confirmed via agent ACK

        Production metrics required:
        - command_dispatch_latency: time from SENT to ACKED
        - command_delivery_rate: % commands successfully delivered
        - command_expiry_rate: % commands that expired undelivered

        Args:
            command: Command to dispatch

        Returns:
            True if marked for dispatch successfully
        """
        # Re-validate before dispatch
        command._validate()

        # Update status - command now available for agent polling
        command.status = CommandStatus.SENT
        command.sent_at = datetime.utcnow()

        # Note: Actual delivery occurs when agent polls and ACKs
        # This follows CCEA outbound-connection model (no inbound to agent)

        return True

    def update_status(
        self,
        command_id: UUID,
        status: CommandStatus,
        result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Update command status (from agent callback).

        Args:
            command_id: Command ID
            status: New status
            result: Execution result
            error_message: Error message if failed

        Returns:
            True if updated
        """
        cmd = self._commands.get(command_id)
        if not cmd:
            return False

        cmd.status = status

        if status == CommandStatus.ACKNOWLEDGED:
            cmd.acked_at = datetime.utcnow()
        elif status == CommandStatus.EXECUTED:
            cmd.executed_at = datetime.utcnow()

        if result:
            cmd.result = result
        if error_message:
            cmd.error_message = error_message

        return True

    def get_command(self, command_id: UUID) -> Optional[Command]:
        """Get command by ID."""
        return self._commands.get(command_id)

    def get_pending_commands(self, agent_id: str) -> List[Command]:
        """Get pending commands for agent."""
        return [
            cmd
            for cmd in self._commands.values()
            if cmd.agent_id == agent_id
            and cmd.status in (CommandStatus.PENDING, CommandStatus.SENT)
        ]
