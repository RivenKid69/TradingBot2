# -*- coding: utf-8 -*-
"""
CCEA Agent Command Handler.

Handles incoming commands from cloud:
- Command filtering (allowlist only)
- Idempotency enforcement
- Approval workflow integration
- Command execution

Security:
- Only allowlisted commands accepted
- No order-like commands possible
- Outbound-only polling (never push)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import threading

from ccea.models.protocol import (
    CommandType,
    CommandStatus,
    ChangeClass,
    Command,
    RequestStartRunCommand,
    RequestStopRunCommand,
    RequestPauseRunCommand,
    RequestUpgradeArtifactCommand,
    RequestUpdateConfigCommand,
    RequestRotateAgentSessionCommand,
    RequestExportLogsCommand,
)


logger = logging.getLogger(__name__)


# ============================================================================
# Command Allowlist (Design Doc E2)
# ============================================================================

ALLOWED_COMMANDS: Set[str] = {
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

# Safety commands (execute without approval)
SAFETY_COMMANDS: Set[str] = {
    CommandType.REQUEST_STOP_RUN.value,
    CommandType.REQUEST_PAUSE_RUN.value,
}


# ============================================================================
# Command Record
# ============================================================================

class LocalCommandStatus(str, Enum):
    """Local command processing status."""
    RECEIVED = "RECEIVED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    FILTERED = "FILTERED"


@dataclass
class LocalCommandRecord:
    """Local record of a received command."""
    command_id: str
    idempotency_key: str
    command_type: str
    payload: Dict[str, Any]
    status: LocalCommandStatus = LocalCommandStatus.RECEIVED
    requires_approval: bool = False
    received_at: datetime = field(default_factory=datetime.utcnow)
    approved_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    evidence_hash: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    filter_reason: Optional[str] = None


# ============================================================================
# Command Filter
# ============================================================================

class CommandFilter:
    """
    Command allowlist filter.

    Ensures only allowed commands are processed.
    CRITICAL: Prevents any order-like commands from being accepted.
    """

    def __init__(
        self,
        allowed_commands: Optional[Set[str]] = None,
        custom_filters: Optional[List[Callable[[Dict[str, Any]], bool]]] = None,
    ):
        """
        Initialize filter.

        Args:
            allowed_commands: Set of allowed command types
            custom_filters: Additional filter functions
        """
        self.allowed_commands = allowed_commands or ALLOWED_COMMANDS
        self.custom_filters = custom_filters or []

    def is_allowed(self, command: Dict[str, Any]) -> tuple[bool, str]:
        """
        Check if command is allowed.

        Args:
            command: Command payload

        Returns:
            Tuple of (is_allowed, reason)
        """
        command_type = command.get("command_type")

        # Check command type allowlist
        if command_type not in self.allowed_commands:
            return False, f"Command type not in allowlist: {command_type}"

        # Check for prohibited fields (order-like)
        prohibited = self._check_prohibited_fields(command)
        if prohibited:
            return False, f"Prohibited field detected: {prohibited}"

        # Run custom filters
        for filter_fn in self.custom_filters:
            try:
                if not filter_fn(command):
                    return False, "Rejected by custom filter"
            except Exception as e:
                return False, f"Filter error: {e}"

        return True, ""

    def _check_prohibited_fields(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Check for prohibited order-like fields.

        CRITICAL: Prevents injection of order payloads.
        """
        prohibited_fields = {
            "side", "quantity", "qty", "price", "order_type",
            "target_position", "execute_order", "place_order",
            "submit_order", "intent", "signal"
        }

        def check_recursive(obj: Any, path: str = "") -> Optional[str]:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in prohibited_fields:
                        return f"{path}.{key}" if path else key
                    result = check_recursive(value, f"{path}.{key}" if path else key)
                    if result:
                        return result
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    result = check_recursive(item, f"{path}[{i}]")
                    if result:
                        return result
            return None

        return check_recursive(data)


# ============================================================================
# Command Handler
# ============================================================================

CommandExecutor = Callable[[LocalCommandRecord], Dict[str, Any]]


class CommandHandler:
    """
    Agent command handler.

    Handles:
    - Receiving commands from poll
    - Filtering commands
    - Managing approval workflow
    - Executing commands
    - Tracking idempotency
    """

    def __init__(
        self,
        command_filter: Optional[CommandFilter] = None,
        approval_timeout: int = 300,  # 5 minutes
    ):
        """
        Initialize command handler.

        Args:
            command_filter: Command filter instance
            approval_timeout: Approval timeout in seconds
        """
        self.filter = command_filter or CommandFilter()
        self.approval_timeout = approval_timeout

        self._commands: Dict[str, LocalCommandRecord] = {}
        self._idempotency_keys: Dict[str, str] = {}
        self._executors: Dict[str, CommandExecutor] = {}
        self._lock = threading.Lock()

        # Register default executors
        self._register_default_executors()

    def register_executor(
        self,
        command_type: str,
        executor: CommandExecutor,
    ) -> None:
        """Register command executor."""
        self._executors[command_type] = executor

    def handle_command(self, command: Dict[str, Any]) -> LocalCommandRecord:
        """
        Handle incoming command.

        Args:
            command: Command payload from cloud

        Returns:
            LocalCommandRecord with processing status
        """
        command_id = command.get("command_id", command.get("idempotency_key", "unknown"))
        idempotency_key = command.get("idempotency_key", "")
        command_type = command.get("command_type", "")

        with self._lock:
            # Check idempotency
            if idempotency_key and idempotency_key in self._idempotency_keys:
                existing_id = self._idempotency_keys[idempotency_key]
                if existing_id in self._commands:
                    logger.info(f"Returning existing command for idempotency key: {idempotency_key[:16]}...")
                    return self._commands[existing_id]

            # Create record
            record = LocalCommandRecord(
                command_id=command_id,
                idempotency_key=idempotency_key,
                command_type=command_type,
                payload=command,
            )

            # Filter command
            allowed, reason = self.filter.is_allowed(command)
            if not allowed:
                record.status = LocalCommandStatus.FILTERED
                record.filter_reason = reason
                logger.warning(f"Command filtered: {reason}", extra={"command_type": command_type})
                self._commands[command_id] = record
                return record

            # Check if approval required
            record.requires_approval = self._requires_approval(command)

            if record.requires_approval:
                record.status = LocalCommandStatus.AWAITING_APPROVAL
                logger.info(
                    f"Command awaiting approval",
                    extra={"command_id": command_id, "command_type": command_type}
                )
            else:
                # Safety commands execute immediately
                record.status = LocalCommandStatus.APPROVED

            # Store
            self._commands[command_id] = record
            if idempotency_key:
                self._idempotency_keys[idempotency_key] = command_id

        return record

    def approve_command(
        self,
        command_id: str,
        evidence_hash: Optional[str] = None,
    ) -> bool:
        """
        Approve a command.

        Args:
            command_id: Command to approve
            evidence_hash: Hash of approval evidence

        Returns:
            True if approved, False if not found/already processed
        """
        with self._lock:
            record = self._commands.get(command_id)
            if record is None:
                return False

            if record.status != LocalCommandStatus.AWAITING_APPROVAL:
                return False

            record.status = LocalCommandStatus.APPROVED
            record.approved_at = datetime.utcnow()
            record.evidence_hash = evidence_hash

        logger.info(f"Command approved: {command_id}")
        return True

    def reject_command(
        self,
        command_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Reject a command.

        Args:
            command_id: Command to reject
            reason: Rejection reason

        Returns:
            True if rejected, False if not found/already processed
        """
        with self._lock:
            record = self._commands.get(command_id)
            if record is None:
                return False

            if record.status != LocalCommandStatus.AWAITING_APPROVAL:
                return False

            record.status = LocalCommandStatus.REJECTED
            record.error = reason or "Rejected by user"

        logger.info(f"Command rejected: {command_id}")
        return True

    def execute_command(self, command_id: str) -> Optional[Dict[str, Any]]:
        """
        Execute an approved command.

        Args:
            command_id: Command to execute

        Returns:
            Execution result or None if not executable
        """
        with self._lock:
            record = self._commands.get(command_id)
            if record is None:
                return None

            if record.status != LocalCommandStatus.APPROVED:
                return None

            record.status = LocalCommandStatus.EXECUTING

        try:
            executor = self._executors.get(record.command_type)
            if executor is None:
                raise ValueError(f"No executor for command type: {record.command_type}")

            result = executor(record)

            with self._lock:
                record.status = LocalCommandStatus.COMPLETED
                record.completed_at = datetime.utcnow()
                record.result = result

            logger.info(f"Command executed: {command_id}")
            return result

        except Exception as e:
            with self._lock:
                record.status = LocalCommandStatus.FAILED
                record.error = str(e)

            logger.exception(f"Command execution failed: {command_id}")
            return {"error": str(e)}

    def get_pending_approvals(self) -> List[LocalCommandRecord]:
        """Get commands awaiting approval."""
        with self._lock:
            return [
                r for r in self._commands.values()
                if r.status == LocalCommandStatus.AWAITING_APPROVAL
            ]

    def get_command(self, command_id: str) -> Optional[LocalCommandRecord]:
        """Get command by ID."""
        with self._lock:
            return self._commands.get(command_id)

    def _requires_approval(self, command: Dict[str, Any]) -> bool:
        """Check if command requires local approval."""
        command_type = command.get("command_type", "")

        # Safety commands don't require approval
        if command_type in SAFETY_COMMANDS:
            return False

        # Check explicit requires_approval field
        if command.get("requires_approval") is False:
            return False

        # Check change class for config updates
        if command_type == CommandType.REQUEST_UPDATE_CONFIG.value:
            change_class = command.get("change_class")
            if change_class == ChangeClass.NON_TRADING_IMPACTING.value:
                return False

        return command_type in APPROVAL_REQUIRED

    def _register_default_executors(self) -> None:
        """Register default command executors."""
        # These are placeholders - actual implementations in daemon
        self._executors[CommandType.REQUEST_STOP_RUN.value] = self._execute_stop
        self._executors[CommandType.REQUEST_PAUSE_RUN.value] = self._execute_pause

    def _execute_stop(self, record: LocalCommandRecord) -> Dict[str, Any]:
        """Execute stop command."""
        deployment_id = record.payload.get("deployment_id")
        return {"action": "stopped", "deployment_id": deployment_id}

    def _execute_pause(self, record: LocalCommandRecord) -> Dict[str, Any]:
        """Execute pause command."""
        deployment_id = record.payload.get("deployment_id")
        return {"action": "paused", "deployment_id": deployment_id}
