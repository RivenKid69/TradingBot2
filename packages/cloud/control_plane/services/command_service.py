# -*- coding: utf-8 -*-
"""
Unified Command Service - DB-backed command dispatch and lifecycle.

Phase 7 (WI-CLOUD-03): Replaces in-memory CommandDispatcher with DB-backed service.

CLOUD ZONE ONLY.

This service provides:
- Command creation with idempotency
- Command polling for agents (long-poll ready)
- Command acknowledgement and result submission
- Command state machine enforcement
- Boundary validation integration

PROHIBITED:
- Order-like payloads (side, qty, price, target_position)
- Direct trading instructions
- Signal/intent transmission
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Final, FrozenSet, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models import (
    Agent,
    ApprovalRecord,
    ChangeClass,
    Command,
    CommandStatus,
    Deployment,
    Run,
    TrustState,
)


# ============================================================================
# Constants (from commands.py, now centralized)
# ============================================================================

class CommandType(str, Enum):
    """
    Allowed command types per Design Doc.

    These are the ONLY commands Cloud can send.
    Adding new command types requires security review.
    """
    REQUEST_START_RUN = "REQUEST_START_RUN"
    REQUEST_STOP_RUN = "REQUEST_STOP_RUN"
    REQUEST_PAUSE_RUN = "REQUEST_PAUSE_RUN"
    REQUEST_UPGRADE_ARTIFACT = "REQUEST_UPGRADE_ARTIFACT"
    REQUEST_UPDATE_CONFIG = "REQUEST_UPDATE_CONFIG"
    REQUEST_ROTATE_AGENT_SESSION = "REQUEST_ROTATE_AGENT_SESSION"
    REQUEST_EXPORT_LOGS = "REQUEST_EXPORT_LOGS"


ALLOWED_COMMAND_TYPES: Final[FrozenSet[str]] = frozenset(
    [ct.value for ct in CommandType]
)

# Fields prohibited in command payloads
PROHIBITED_PAYLOAD_FIELDS: Final[FrozenSet[str]] = frozenset([
    "side", "quantity", "qty", "price", "limit_price", "stop_price",
    "order_type", "target_position", "execute_order", "place_order",
    "submit_order", "intent", "signal", "trade", "order",
])

# Valid state transitions
COMMAND_STATE_TRANSITIONS: Dict[CommandStatus, List[CommandStatus]] = {
    CommandStatus.PENDING: [
        CommandStatus.PENDING_APPROVAL,
        CommandStatus.SENT,
        CommandStatus.EXPIRED,
    ],
    CommandStatus.PENDING_APPROVAL: [
        CommandStatus.APPROVED,
        CommandStatus.REJECTED,
        CommandStatus.EXPIRED,
    ],
    CommandStatus.APPROVED: [
        CommandStatus.SENT,
        CommandStatus.EXPIRED,
    ],
    CommandStatus.SENT: [
        CommandStatus.ACKNOWLEDGED,
        CommandStatus.FAILED,
        CommandStatus.EXPIRED,
    ],
    CommandStatus.ACKNOWLEDGED: [
        CommandStatus.EXECUTED,
        CommandStatus.FAILED,
        CommandStatus.EXPIRED,
    ],
    # Terminal states
    CommandStatus.REJECTED: [],
    CommandStatus.EXECUTED: [],
    CommandStatus.FAILED: [],
    CommandStatus.EXPIRED: [],
}


# ============================================================================
# Exceptions
# ============================================================================

class CommandServiceError(Exception):
    """Base exception for command service errors."""
    pass


class CommandNotFoundError(CommandServiceError):
    """Command not found."""
    pass


class CommandStateError(CommandServiceError):
    """Invalid state transition."""
    pass


class DuplicateCommandError(CommandServiceError):
    """Duplicate idempotency key."""
    pass


class CommandValidationError(CommandServiceError):
    """Command validation failed."""
    pass


# ============================================================================
# DTOs
# ============================================================================

@dataclass
class CommandCreateRequest:
    """Request to create a new command."""
    workspace_id: UUID
    agent_id: UUID
    command_type: str
    payload_ref: str
    deployment_id: Optional[UUID] = None
    run_id: Optional[UUID] = None
    change_class: ChangeClass = ChangeClass.OPERATIONAL
    requires_approval: bool = False
    expires_at: Optional[datetime] = None
    idempotency_key: Optional[str] = None
    # Phase 3 (3.1A): Components for deterministic idempotency key
    artifact_digest: Optional[str] = None
    config_digest: Optional[str] = None


@dataclass
class CommandPollResult:
    """Result of polling for pending commands."""
    commands: List[Command]
    has_more: bool
    poll_again_after_sec: int


@dataclass
class CommandAckRequest:
    """Request to acknowledge a command."""
    command_id: UUID
    agent_id: UUID


@dataclass
class CommandResultRequest:
    """Request to submit command execution result."""
    command_id: UUID
    agent_id: UUID
    success: bool
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


# ============================================================================
# Command Service
# ============================================================================

class CommandService:
    """
    Unified command service - DB-backed command lifecycle management.

    Phase 7 (WI-CLOUD-03): Replaces in-memory CommandDispatcher.

    Usage:
        service = CommandService(session)

        # Create command
        cmd = await service.create_command(request)

        # Agent polls for commands
        result = await service.poll_commands(agent_id, workspace_id)

        # Agent acknowledges command
        cmd = await service.acknowledge_command(ack_request)

        # Agent submits result
        cmd = await service.submit_result(result_request)
    """

    def __init__(self, session: AsyncSession):
        """Initialize service with database session."""
        self.session = session

    # ========================================================================
    # Command Creation
    # ========================================================================

    async def create_command(
        self,
        request: CommandCreateRequest,
    ) -> Command:
        """
        Create a new command.

        Args:
            request: Command creation request

        Returns:
            Created command

        Raises:
            CommandValidationError: If command type or payload invalid
            DuplicateCommandError: If idempotency key already exists
        """
        # Validate command type
        if request.command_type.upper() not in ALLOWED_COMMAND_TYPES:
            raise CommandValidationError(
                f"Invalid command type: {request.command_type}. "
                f"Allowed: {sorted(ALLOWED_COMMAND_TYPES)}"
            )

        # Phase 3 (3.1A): Generate DETERMINISTIC idempotency key based on command content
        # This ensures "same desired state" doesn't create duplicate commands
        if request.idempotency_key:
            idempotency_key = request.idempotency_key
        else:
            idempotency_key = self._generate_deterministic_idempotency_key(request)

        # Check for duplicate
        existing = await self.session.execute(
            select(Command).where(
                Command.workspace_id == request.workspace_id,
                Command.idempotency_key == idempotency_key,
            )
        )
        if existing.scalar_one_or_none():
            raise DuplicateCommandError(
                f"Command with idempotency key '{idempotency_key}' already exists"
            )

        # Verify agent exists and is enrolled
        agent = await self._get_enrolled_agent(request.agent_id, request.workspace_id)
        if not agent:
            raise CommandValidationError(
                f"Agent {request.agent_id} not found or not enrolled"
            )

        # Enforce invariant: TRADING_IMPACTING always requires local approval (Cloud cannot bypass).
        requires_approval = bool(request.requires_approval)
        if request.change_class == ChangeClass.TRADING_IMPACTING:
            requires_approval = True

        initial_status = (
            CommandStatus.PENDING_APPROVAL.value if requires_approval else CommandStatus.PENDING.value
        )

        # Create command
        command = Command(
            workspace_id=request.workspace_id,
            idempotency_key=idempotency_key,
            agent_id=request.agent_id,
            deployment_id=request.deployment_id,
            run_id=request.run_id,
            command_type=request.command_type.upper(),
            payload_ref=request.payload_ref,
            change_class=request.change_class.value,
            requires_approval=requires_approval,
            status=initial_status,
            expires_at=request.expires_at,
        )

        self.session.add(command)
        await self.session.flush()
        await self.session.refresh(command)

        return command

    # ========================================================================
    # Command Polling (for Agents)
    # ========================================================================

    async def poll_commands(
        self,
        agent_id: UUID,
        workspace_id: UUID,
        limit: int = 10,
        timeout_sec: int = 30,
    ) -> CommandPollResult:
        """
        Poll for pending commands for an agent.

        This is the main endpoint for agent command retrieval.
        Supports long-poll pattern (returns immediately if commands available).

        Args:
            agent_id: Agent ID
            workspace_id: Workspace ID
            limit: Maximum commands to return
            timeout_sec: Long-poll timeout (future enhancement)

        Returns:
            Poll result with commands and pagination info
        """
        now = datetime.now(timezone.utc)

        # Get pending commands for this agent.
        # States:
        # - PENDING (no approval needed)
        # - PENDING_APPROVAL (awaiting local approval by agent)
        # - APPROVED (approved by local)
        # - SENT (re-delivery until ack)
        query = (
            select(Command)
            .where(
                Command.agent_id == agent_id,
                Command.workspace_id == workspace_id,
                Command.status.in_([
                    CommandStatus.PENDING.value,
                    CommandStatus.PENDING_APPROVAL.value,
                    CommandStatus.APPROVED.value,
                    # SENT commands can be re-polled if not yet acknowledged
                    CommandStatus.SENT.value,
                ]),
                or_(
                    Command.expires_at.is_(None),
                    Command.expires_at > now,
                ),
            )
            .order_by(Command.created_at.asc())
            .limit(limit + 1)  # +1 to check for more
        )

        result = await self.session.execute(query)
        commands = list(result.scalars().all())

        has_more = len(commands) > limit
        if has_more:
            commands = commands[:limit]

        return CommandPollResult(
            commands=commands,
            has_more=has_more,
            poll_again_after_sec=5 if not commands else 60,
        )

    async def get_pending_count(
        self,
        agent_id: UUID,
        workspace_id: UUID,
    ) -> int:
        """
        Get count of pending commands for an agent.

        Used in heartbeat response.

        Args:
            agent_id: Agent ID
            workspace_id: Workspace ID

        Returns:
            Count of pending commands
        """
        now = datetime.now(timezone.utc)

        query = select(func.count(Command.id)).where(
            Command.agent_id == agent_id,
            Command.workspace_id == workspace_id,
            Command.status.in_([
                CommandStatus.PENDING.value,
                CommandStatus.PENDING_APPROVAL.value,
                CommandStatus.APPROVED.value,
                CommandStatus.SENT.value,
            ]),
            or_(
                Command.expires_at.is_(None),
                Command.expires_at > now,
            ),
        )

        result = await self.session.execute(query)
        return result.scalar() or 0

    # ========================================================================
    # Command Acknowledgement
    # ========================================================================

    async def acknowledge_command(
        self,
        request: CommandAckRequest,
    ) -> Command:
        """
        Acknowledge receipt of a command by agent.

        Transitions command from SENT to ACKNOWLEDGED.

        Args:
            request: Acknowledgement request

        Returns:
            Updated command

        Raises:
            CommandNotFoundError: If command not found
            CommandStateError: If invalid state transition
        """
        command = await self._get_command_for_agent(
            request.command_id,
            request.agent_id,
        )

        # Mark as sent if still pending/approved
        if command.status in [
            CommandStatus.PENDING.value,
            CommandStatus.APPROVED.value,
        ]:
            command.status = CommandStatus.SENT.value
            command.sent_at = datetime.now(timezone.utc)
            await self.session.flush()

        # Validate transition
        if command.status != CommandStatus.SENT.value:
            raise CommandStateError(
                f"Cannot acknowledge command in state {command.status}. "
                f"Expected: {CommandStatus.SENT.value}"
            )

        # Update to acknowledged
        command.status = CommandStatus.ACKNOWLEDGED.value
        command.acknowledged_at = datetime.now(timezone.utc)

        await self.session.flush()
        await self.session.refresh(command)

        return command

    # ========================================================================
    # Command Result Submission
    # ========================================================================

    async def submit_result(
        self,
        request: CommandResultRequest,
    ) -> Command:
        """
        Submit execution result for a command.

        Transitions command to EXECUTED or FAILED.

        Args:
            request: Result request

        Returns:
            Updated command

        Raises:
            CommandNotFoundError: If command not found
            CommandStateError: If invalid state transition
        """
        command = await self._get_command_for_agent(
            request.command_id,
            request.agent_id,
        )

        # Must be in ACKNOWLEDGED state
        if command.status != CommandStatus.ACKNOWLEDGED.value:
            raise CommandStateError(
                f"Cannot submit result for command in state {command.status}. "
                f"Expected: {CommandStatus.ACKNOWLEDGED.value}"
            )

        # Update status based on success
        now = datetime.now(timezone.utc)
        if request.success:
            command.status = CommandStatus.EXECUTED.value
            command.executed_at = now
        else:
            command.status = CommandStatus.FAILED.value

        # Set result/error
        if request.result is not None:
            command.result = request.result
        if request.error_message:
            command.error_message = request.error_message

        await self.session.flush()
        await self.session.refresh(command)

        return command

    # ========================================================================
    # Command Dispatch (for sending to agent)
    # ========================================================================

    async def dispatch_command(
        self,
        command_id: UUID,
        workspace_id: UUID,
    ) -> Command:
        """
        Mark command as sent/dispatched.

        Called when command is sent to agent (e.g., via poll response).

        Args:
            command_id: Command ID
            workspace_id: Workspace ID

        Returns:
            Updated command
        """
        command = await self._get_command(command_id, workspace_id)

        # Check if already sent
        if command.status == CommandStatus.SENT.value:
            return command

        # Validate transition
        allowed = [CommandStatus.PENDING.value, CommandStatus.APPROVED.value]
        if command.status not in allowed:
            raise CommandStateError(
                f"Cannot dispatch command in state {command.status}. "
                f"Expected one of: {allowed}"
            )

        # Check approval requirement
        if command.requires_approval and command.status != CommandStatus.APPROVED.value:
            raise CommandStateError(
                "Command requires approval before dispatch"
            )

        command.status = CommandStatus.SENT.value
        command.sent_at = datetime.now(timezone.utc)

        await self.session.flush()
        await self.session.refresh(command)

        return command

    # ========================================================================
    # Command Expiration
    # ========================================================================

    async def expire_stale_commands(
        self,
        workspace_id: UUID,
    ) -> int:
        """
        Expire all stale commands in workspace.

        Args:
            workspace_id: Workspace ID

        Returns:
            Number of expired commands
        """
        now = datetime.now(timezone.utc)
        non_terminal = [
            CommandStatus.PENDING.value,
            CommandStatus.PENDING_APPROVAL.value,
            CommandStatus.APPROVED.value,
            CommandStatus.SENT.value,
            CommandStatus.ACKNOWLEDGED.value,
        ]

        result = await self.session.execute(
            update(Command)
            .where(
                Command.workspace_id == workspace_id,
                Command.status.in_(non_terminal),
                Command.expires_at.isnot(None),
                Command.expires_at < now,
            )
            .values(
                status=CommandStatus.EXPIRED.value,
                updated_at=now,
            )
        )

        return result.rowcount

    # ========================================================================
    # Helper Methods
    # ========================================================================

    async def _get_command(
        self,
        command_id: UUID,
        workspace_id: UUID,
    ) -> Command:
        """Get command by ID with workspace validation."""
        result = await self.session.execute(
            select(Command)
            .options(selectinload(Command.approvals))
            .where(
                Command.id == command_id,
                Command.workspace_id == workspace_id,
            )
        )
        command = result.scalar_one_or_none()

        if command is None:
            raise CommandNotFoundError(f"Command {command_id} not found")

        return command

    async def _get_command_for_agent(
        self,
        command_id: UUID,
        agent_id: UUID,
    ) -> Command:
        """Get command by ID with agent validation."""
        result = await self.session.execute(
            select(Command)
            .where(
                Command.id == command_id,
                Command.agent_id == agent_id,
            )
        )
        command = result.scalar_one_or_none()

        if command is None:
            raise CommandNotFoundError(
                f"Command {command_id} not found for agent {agent_id}"
            )

        return command

    async def _get_enrolled_agent(
        self,
        agent_id: UUID,
        workspace_id: UUID,
    ) -> Optional[Agent]:
        """Get agent if enrolled."""
        result = await self.session.execute(
            select(Agent).where(
                Agent.id == agent_id,
                Agent.workspace_id == workspace_id,
                Agent.trust_state == TrustState.ENROLLED.value,
                Agent.deleted_at.is_(None),
            )
        )
        return result.scalar_one_or_none()

    def _generate_idempotency_key(self) -> str:
        """Generate a unique idempotency key (legacy, random)."""
        return secrets.token_urlsafe(24)

    def _generate_deterministic_idempotency_key(
        self,
        request: CommandCreateRequest,
    ) -> str:
        """
        Generate a DETERMINISTIC idempotency key based on command content.

        Phase 3 (3.1A): Key is computed from semantic content of the command,
        so "same desired state" produces the same key.

        This ensures:
        - Duplicate requests don't create multiple commands
        - Safe retry behavior in Cloud→Agent communication
        - Agent can safely re-poll without duplicate execution

        Args:
            request: Command creation request

        Returns:
            Deterministic idempotency key
        """
        from .idempotency import generate_deterministic_idempotency_key

        return generate_deterministic_idempotency_key(
            workspace_id=request.workspace_id,
            agent_id=request.agent_id,
            command_type=request.command_type,
            deployment_id=request.deployment_id,
            run_id=request.run_id,
            artifact_digest=request.artifact_digest,
            config_digest=request.config_digest,
            payload_ref=request.payload_ref,
        )

    @staticmethod
    def validate_command_type(command_type: str) -> Tuple[bool, str]:
        """
        Validate command type against allowlist.

        Args:
            command_type: Command type to validate

        Returns:
            Tuple of (is_valid, normalized_type_or_error)
        """
        normalized = command_type.upper().strip()
        if normalized in ALLOWED_COMMAND_TYPES:
            return True, normalized
        return False, f"Invalid command type: {command_type}"
