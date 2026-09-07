# -*- coding: utf-8 -*-
"""
CCEA Command Service.

Handles command dispatch to agents:
- Command queue per agent (long-poll)
- Idempotency enforcement
- Status tracking
- Approval integration

Security:
- Only allowlisted command types
- No order-like payloads
- Commands authenticated with signature
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from queue import Queue, Empty
import threading

from ccea.models.protocol import (
    Command,
    CommandType,
    CommandStatus,
    ChangeClass,
    RequestStartRunCommand,
    RequestStopRunCommand,
    RequestPauseRunCommand,
    RequestUpgradeArtifactCommand,
    RequestUpdateConfigCommand,
    RequestRotateAgentSessionCommand,
    RequestExportLogsCommand,
)
from ccea.crypto.tokens import generate_command_id, generate_idempotency_key


logger = logging.getLogger(__name__)


# ============================================================================
# Command Types Allowlist (Design Doc E2)
# ============================================================================

ALLOWED_COMMAND_TYPES: Set[str] = {
    CommandType.REQUEST_START_RUN.value,
    CommandType.REQUEST_STOP_RUN.value,
    CommandType.REQUEST_PAUSE_RUN.value,
    CommandType.REQUEST_UPGRADE_ARTIFACT.value,
    CommandType.REQUEST_UPDATE_CONFIG.value,
    CommandType.REQUEST_ROTATE_AGENT_SESSION.value,
    CommandType.REQUEST_EXPORT_LOGS.value,
}

# Commands that require local approval
APPROVAL_REQUIRED: Set[str] = {
    CommandType.REQUEST_START_RUN.value,
    CommandType.REQUEST_UPGRADE_ARTIFACT.value,
    CommandType.REQUEST_UPDATE_CONFIG.value,  # if TRADING_IMPACTING
    CommandType.REQUEST_ROTATE_AGENT_SESSION.value,
    CommandType.REQUEST_EXPORT_LOGS.value,
}

# Safety commands (no approval needed)
SAFETY_COMMANDS: Set[str] = {
    CommandType.REQUEST_STOP_RUN.value,
    CommandType.REQUEST_PAUSE_RUN.value,
}


# ============================================================================
# Command Record
# ============================================================================


@dataclass
class CommandRecord:
    """
    Record of a dispatched command.
    """

    command_id: str
    idempotency_key: str
    agent_id: str
    command_type: str
    payload: Dict[str, Any]
    status: CommandStatus = CommandStatus.PENDING
    requires_approval: bool = False
    change_class: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    received_at: Optional[datetime] = None
    approved_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    approval_status: Optional[str] = None
    evidence_hash: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, str]] = None

    def is_complete(self) -> bool:
        """Check if command is in terminal state."""
        return self.status in {
            CommandStatus.COMPLETED,
            CommandStatus.FAILED,
            CommandStatus.REJECTED,
        }


# ============================================================================
# Command Store Interface
# ============================================================================


class CommandStore(ABC):
    """Abstract command store interface."""

    @abstractmethod
    def save(self, record: CommandRecord) -> None:
        """Save a command record."""
        pass

    @abstractmethod
    def get(self, command_id: str) -> Optional[CommandRecord]:
        """Get command by ID."""
        pass

    @abstractmethod
    def get_by_idempotency_key(self, agent_id: str, key: str) -> Optional[CommandRecord]:
        """Get command by idempotency key."""
        pass

    @abstractmethod
    def update(self, record: CommandRecord) -> None:
        """Update command record."""
        pass

    @abstractmethod
    def list_pending(self, agent_id: str) -> List[CommandRecord]:
        """List pending commands for agent."""
        pass


class InMemoryCommandStore(CommandStore):
    """In-memory command store for development/testing."""

    def __init__(self):
        self._commands: Dict[str, CommandRecord] = {}
        self._by_idem_key: Dict[str, str] = {}
        self._lock = threading.Lock()

    def save(self, record: CommandRecord) -> None:
        with self._lock:
            self._commands[record.command_id] = record
            idem_key = f"{record.agent_id}:{record.idempotency_key}"
            self._by_idem_key[idem_key] = record.command_id

    def get(self, command_id: str) -> Optional[CommandRecord]:
        with self._lock:
            return self._commands.get(command_id)

    def get_by_idempotency_key(self, agent_id: str, key: str) -> Optional[CommandRecord]:
        with self._lock:
            idem_key = f"{agent_id}:{key}"
            command_id = self._by_idem_key.get(idem_key)
            if command_id:
                return self._commands.get(command_id)
            return None

    def update(self, record: CommandRecord) -> None:
        with self._lock:
            if record.command_id in self._commands:
                self._commands[record.command_id] = record

    def list_pending(self, agent_id: str) -> List[CommandRecord]:
        with self._lock:
            return [
                cmd
                for cmd in self._commands.values()
                if cmd.agent_id == agent_id and not cmd.is_complete()
            ]


# ============================================================================
# Command Queue
# ============================================================================


class CommandQueue:
    """
    Per-agent command queue for long-polling.
    """

    def __init__(self, poll_timeout: float = 30.0):
        """
        Initialize command queue.

        Args:
            poll_timeout: Long-poll timeout in seconds
        """
        self.poll_timeout = poll_timeout
        self._queues: Dict[str, Queue] = defaultdict(Queue)
        self._lock = threading.Lock()

    def enqueue(self, agent_id: str, command: CommandRecord) -> None:
        """Add command to agent's queue."""
        with self._lock:
            self._queues[agent_id].put(command)

    def poll(
        self,
        agent_id: str,
        timeout: Optional[float] = None,
    ) -> List[CommandRecord]:
        """
        Poll for commands (long-poll).

        Args:
            agent_id: Agent ID
            timeout: Poll timeout (default: self.poll_timeout)

        Returns:
            List of pending commands (may be empty if timeout)
        """
        timeout = timeout or self.poll_timeout
        commands = []

        with self._lock:
            queue = self._queues[agent_id]

        # Get first command with wait
        try:
            cmd = queue.get(timeout=timeout)
            commands.append(cmd)

            # Drain any additional commands (non-blocking)
            while True:
                try:
                    cmd = queue.get_nowait()
                    commands.append(cmd)
                except Empty:
                    break
        except Empty:
            pass  # Timeout, return empty

        return commands

    def clear(self, agent_id: str) -> None:
        """Clear agent's queue."""
        with self._lock:
            if agent_id in self._queues:
                while not self._queues[agent_id].empty():
                    try:
                        self._queues[agent_id].get_nowait()
                    except Empty:
                        break


# ============================================================================
# Command Service
# ============================================================================


class CommandService:
    """
    Command management service.

    Handles:
    - Command creation with validation
    - Idempotency enforcement
    - Status tracking
    - Queue management
    """

    def __init__(
        self,
        command_store: Optional[CommandStore] = None,
        command_queue: Optional[CommandQueue] = None,
    ):
        """
        Initialize command service.

        Args:
            command_store: Command storage backend
            command_queue: Command queue for polling
        """
        self.store = command_store or InMemoryCommandStore()
        self.queue = command_queue or CommandQueue()

    def create_command(
        self,
        agent_id: str,
        command: Command,
        idempotency_key: Optional[str] = None,
    ) -> CommandRecord:
        """
        Create and dispatch a command.

        Args:
            agent_id: Target agent
            command: Command to dispatch
            idempotency_key: Optional idempotency key

        Returns:
            CommandRecord

        Raises:
            ValueError: If command type not allowed
        """
        # Validate command type
        cmd_type = command.command_type.value
        if cmd_type not in ALLOWED_COMMAND_TYPES:
            raise ValueError(f"Command type not allowed: {cmd_type}")

        # Use provided or generate idempotency key
        idem_key = idempotency_key or generate_idempotency_key()

        # Check for existing command with same idempotency key
        existing = self.store.get_by_idempotency_key(agent_id, idem_key)
        if existing:
            logger.info(
                "Returning existing command for idempotency key",
                extra={
                    "command_id": existing.command_id,
                    "idempotency_key": idem_key[:16] + "...",
                },
            )
            return existing

        # Determine approval requirements
        requires_approval = cmd_type in APPROVAL_REQUIRED
        change_class = None

        if hasattr(command, "change_class"):
            change_class = command.change_class.value if command.change_class else None
            # REQUEST_UPDATE_CONFIG only requires approval if TRADING_IMPACTING
            if cmd_type == CommandType.REQUEST_UPDATE_CONFIG.value:
                requires_approval = change_class == ChangeClass.TRADING_IMPACTING.value

        # Create record
        command_id = generate_command_id()
        record = CommandRecord(
            command_id=command_id,
            idempotency_key=idem_key,
            agent_id=agent_id,
            command_type=cmd_type,
            payload=command.model_dump(mode="json"),
            requires_approval=requires_approval,
            change_class=change_class,
        )

        # Save and enqueue
        self.store.save(record)
        self.queue.enqueue(agent_id, record)

        logger.info(
            "Command created",
            extra={
                "command_id": command_id,
                "command_type": cmd_type,
                "agent_id": agent_id,
                "requires_approval": requires_approval,
            },
        )

        return record

    def acknowledge(
        self,
        command_id: str,
        status: CommandStatus = CommandStatus.RECEIVED,
    ) -> bool:
        """
        Acknowledge command receipt.

        Args:
            command_id: Command ID
            status: New status

        Returns:
            True if updated, False if not found
        """
        record = self.store.get(command_id)
        if record is None:
            return False

        record.status = status
        record.received_at = datetime.utcnow()
        self.store.update(record)

        logger.debug(
            "Command acknowledged", extra={"command_id": command_id, "status": status.value}
        )

        return True

    def set_approval(
        self,
        command_id: str,
        approved: bool,
        evidence_hash: Optional[str] = None,
    ) -> bool:
        """
        Set command approval status.

        Args:
            command_id: Command ID
            approved: Whether approved
            evidence_hash: Hash of approval evidence

        Returns:
            True if updated, False if not found
        """
        record = self.store.get(command_id)
        if record is None:
            return False

        if approved:
            record.status = CommandStatus.APPROVED
            record.approval_status = "APPROVED"
        else:
            record.status = CommandStatus.REJECTED
            record.approval_status = "REJECTED"

        record.approved_at = datetime.utcnow()
        record.evidence_hash = evidence_hash
        self.store.update(record)

        logger.info(
            "Command approval set",
            extra={
                "command_id": command_id,
                "approved": approved,
                "evidence_hash": evidence_hash[:16] + "..." if evidence_hash else None,
            },
        )

        return True

    def complete(
        self,
        command_id: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Mark command as complete.

        Args:
            command_id: Command ID
            result: Success result data
            error: Error data if failed

        Returns:
            True if updated, False if not found
        """
        record = self.store.get(command_id)
        if record is None:
            return False

        if error:
            record.status = CommandStatus.FAILED
            record.error = error
        else:
            record.status = CommandStatus.COMPLETED
            record.result = result

        record.completed_at = datetime.utcnow()
        self.store.update(record)

        logger.info(
            "Command completed",
            extra={
                "command_id": command_id,
                "status": record.status.value,
            },
        )

        return True

    def get(self, command_id: str) -> Optional[CommandRecord]:
        """Get command by ID."""
        return self.store.get(command_id)

    def poll_commands(
        self,
        agent_id: str,
        timeout: Optional[float] = None,
    ) -> List[CommandRecord]:
        """
        Poll for pending commands (long-poll).

        Args:
            agent_id: Agent ID
            timeout: Poll timeout

        Returns:
            List of pending commands
        """
        return self.queue.poll(agent_id, timeout)


# ============================================================================
# Command Dispatcher
# ============================================================================


class CommandDispatcher:
    """
    High-level command dispatcher.

    Provides typed methods for creating specific commands.
    """

    def __init__(self, service: CommandService):
        """Initialize dispatcher."""
        self.service = service

    def request_start_run(
        self,
        agent_id: str,
        deployment_id: str,
        artifact_digest: str,
        config_digest: Optional[str] = None,
    ) -> CommandRecord:
        """Request agent to start a run."""
        command = RequestStartRunCommand(
            deployment_id=deployment_id,
            artifact_digest=artifact_digest,
            config_digest=config_digest,
            idempotency_key=generate_idempotency_key(),
        )
        return self.service.create_command(agent_id, command)

    def request_stop_run(
        self,
        agent_id: str,
        deployment_id: str,
        reason: Optional[str] = None,
    ) -> CommandRecord:
        """Request agent to stop a run (safety - no approval)."""
        command = RequestStopRunCommand(
            deployment_id=deployment_id,
            reason=reason,
            idempotency_key=generate_idempotency_key(),
        )
        return self.service.create_command(agent_id, command)

    def request_pause_run(
        self,
        agent_id: str,
        deployment_id: str,
        reason: Optional[str] = None,
    ) -> CommandRecord:
        """Request agent to pause a run (safety - no approval)."""
        command = RequestPauseRunCommand(
            deployment_id=deployment_id,
            reason=reason,
            idempotency_key=generate_idempotency_key(),
        )
        return self.service.create_command(agent_id, command)

    def request_upgrade_artifact(
        self,
        agent_id: str,
        deployment_id: str,
        new_artifact_digest: str,
        old_artifact_digest: Optional[str] = None,
    ) -> CommandRecord:
        """Request artifact upgrade."""
        command = RequestUpgradeArtifactCommand(
            deployment_id=deployment_id,
            new_artifact_digest=new_artifact_digest,
            old_artifact_digest=old_artifact_digest,
            idempotency_key=generate_idempotency_key(),
        )
        return self.service.create_command(agent_id, command)

    def request_update_config(
        self,
        agent_id: str,
        deployment_id: str,
        new_config_digest: str,
        old_config_digest: Optional[str] = None,
        trading_impacting: bool = True,
    ) -> CommandRecord:
        """Request config update."""
        command = RequestUpdateConfigCommand(
            deployment_id=deployment_id,
            new_config_digest=new_config_digest,
            old_config_digest=old_config_digest,
            change_class=(
                ChangeClass.TRADING_IMPACTING
                if trading_impacting
                else ChangeClass.NON_TRADING_IMPACTING
            ),
            requires_approval=trading_impacting,
            idempotency_key=generate_idempotency_key(),
        )
        return self.service.create_command(agent_id, command)
