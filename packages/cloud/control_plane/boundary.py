# -*- coding: utf-8 -*-
"""
Cloud API Boundary Enforcement - CLOUD ZONE ONLY.

Enforces the critical security boundary between Cloud and Agent:
- Cloud can only send lifecycle commands
- Cloud CANNOT send intents, signals, or order-like payloads
- All outgoing messages are validated before transmission

This module provides:
1. CloudBoundaryValidator - validates all outgoing Cloud messages
2. CloudCommandFilter - filters commands to ensure only lifecycle ops
3. RuntimeBoundaryEnforcer - runtime validation with fail-closed behavior

SECURITY CRITICAL:
    Any bypass of this boundary is a CRITICAL security violation.
    The Cloud MUST NOT be able to inject trading intents into the Agent.

Design Doc Reference:
    - Phase 3: "Cloud отправляет только lifecycle requests"
    - Protocol: "Order-like payloads are forbidden per Design Doc 10.5/19.2"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Final, FrozenSet, List, Optional, TypeVar

from ccea.guardrails.intent_prohibition import (
    PROHIBITED_INTENT_FIELDS,
    PROHIBITED_COMMAND_TYPES,
    ALLOWED_CLOUD_TO_AGENT_TYPES,
    check_dict_for_prohibited_fields,
    check_command_type,
    IntentProhibitionViolation,
    ViolationSeverity,
)


# ============================================================================
# Constants
# ============================================================================

# Lifecycle commands - the ONLY commands Cloud can send
LIFECYCLE_COMMANDS: Final[FrozenSet[str]] = frozenset({
    "REQUEST_START_RUN",
    "REQUEST_STOP_RUN",
    "REQUEST_PAUSE_RUN",
    "REQUEST_UPGRADE_ARTIFACT",
    "REQUEST_UPDATE_CONFIG",
    "REQUEST_ROTATE_AGENT_SESSION",
    "REQUEST_EXPORT_LOGS",
})

# Fields allowed in lifecycle commands (whitelist approach)
ALLOWED_LIFECYCLE_FIELDS: Final[FrozenSet[str]] = frozenset({
    # Identifiers
    "command_type",
    "message_type",
    "idempotency_key",
    "command_id",
    "deployment_id",
    "agent_id",
    "run_id",

    # Artifacts
    "artifact_digest",
    "new_artifact_digest",
    "old_artifact_digest",
    "config_digest",
    "new_config_digest",
    "old_config_digest",

    # Metadata
    "timestamp",
    "signature",
    "change_class",
    "requires_approval",
    "reason",

    # Export
    "export_config",
    "break_glass_reason",

    # Batch
    "commands",
})


# ============================================================================
# Data Classes
# ============================================================================

class BoundaryViolationType(str, Enum):
    """Types of boundary violations."""

    PROHIBITED_COMMAND = "prohibited_command"
    PROHIBITED_FIELD = "prohibited_field"
    INTENT_INJECTION = "intent_injection"
    UNKNOWN_FIELD = "unknown_field"


@dataclass
class BoundaryViolation:
    """Represents a Cloud boundary violation."""

    violation_type: BoundaryViolationType
    severity: str  # critical, high, medium
    message: str
    field_path: str = ""
    blocked: bool = True
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "violation_type": self.violation_type.value,
            "severity": self.severity,
            "message": self.message,
            "field_path": self.field_path,
            "blocked": self.blocked,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BoundaryValidationResult:
    """Result of boundary validation."""

    valid: bool = True
    violations: List[BoundaryViolation] = field(default_factory=list)
    blocked: bool = False
    checked_at: datetime = field(default_factory=datetime.utcnow)

    def add_violation(self, violation: BoundaryViolation) -> None:
        """Add a violation."""
        self.violations.append(violation)
        if violation.blocked:
            self.valid = False
            self.blocked = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "blocked": self.blocked,
            "violation_count": len(self.violations),
            "violations": [v.to_dict() for v in self.violations],
            "checked_at": self.checked_at.isoformat(),
        }


# ============================================================================
# Validators
# ============================================================================

class CloudBoundaryValidator:
    """
    Validates Cloud -> Agent messages at the boundary.

    Ensures that ONLY lifecycle commands are sent, with NO trading intents.

    Usage:
        validator = CloudBoundaryValidator()

        # Validate before sending
        result = validator.validate_outgoing_message(message)
        if not result.valid:
            raise SecurityError(result.violations)

        # Send message...
    """

    def __init__(self, strict_mode: bool = True):
        """
        Initialize validator.

        Args:
            strict_mode: If True, unknown fields are violations
        """
        self._strict_mode = strict_mode
        self._violations_log: List[BoundaryViolation] = []

    def validate_outgoing_message(
        self,
        message: Dict[str, Any],
        context: str = "",
    ) -> BoundaryValidationResult:
        """
        Validate an outgoing Cloud -> Agent message.

        Args:
            message: Message to validate
            context: Context for logging

        Returns:
            BoundaryValidationResult
        """
        result = BoundaryValidationResult()

        # 1. Check command type is lifecycle
        if "command_type" in message:
            cmd_type = str(message["command_type"])
            if cmd_type not in LIFECYCLE_COMMANDS:
                result.add_violation(BoundaryViolation(
                    violation_type=BoundaryViolationType.PROHIBITED_COMMAND,
                    severity="critical",
                    message=f"Non-lifecycle command type: '{cmd_type}'",
                    field_path="command_type",
                ))

            # Also check against prohibited
            if cmd_type.upper() in PROHIBITED_COMMAND_TYPES:
                result.add_violation(BoundaryViolation(
                    violation_type=BoundaryViolationType.INTENT_INJECTION,
                    severity="critical",
                    message=f"BLOCKED: Intent injection attempt via '{cmd_type}'",
                    field_path="command_type",
                ))

        # 2. Check for prohibited intent fields (recursive)
        self._check_prohibited_fields_recursive(message, "", result)

        # 3. Check for unknown fields in strict mode
        if self._strict_mode:
            self._check_unknown_fields(message, "", result)

        # 4. Check commands array if present
        if "commands" in message and isinstance(message["commands"], list):
            for i, cmd in enumerate(message["commands"]):
                if isinstance(cmd, dict):
                    cmd_result = self.validate_outgoing_message(
                        cmd,
                        f"{context}.commands[{i}]",
                    )
                    for v in cmd_result.violations:
                        result.add_violation(v)

        # Log violations
        for v in result.violations:
            self._violations_log.append(v)

        return result

    def validate_command(
        self,
        command: Dict[str, Any],
    ) -> BoundaryValidationResult:
        """
        Validate a single command.

        Args:
            command: Command dictionary

        Returns:
            BoundaryValidationResult
        """
        return self.validate_outgoing_message(command, "command")

    def validate_command_batch(
        self,
        batch: Dict[str, Any],
    ) -> BoundaryValidationResult:
        """
        Validate a command batch.

        Args:
            batch: Command batch with 'commands' list

        Returns:
            BoundaryValidationResult
        """
        return self.validate_outgoing_message(batch, "command_batch")

    def _check_prohibited_fields_recursive(
        self,
        obj: Any,
        path: str,
        result: BoundaryValidationResult,
    ) -> None:
        """Check object recursively for prohibited fields."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                key_lower = key.lower()

                # Check against prohibited fields
                if key_lower in PROHIBITED_INTENT_FIELDS:
                    result.add_violation(BoundaryViolation(
                        violation_type=BoundaryViolationType.PROHIBITED_FIELD,
                        severity="critical",
                        message=f"Prohibited field '{key}' - potential intent injection",
                        field_path=current_path,
                    ))

                # Recurse
                self._check_prohibited_fields_recursive(value, current_path, result)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._check_prohibited_fields_recursive(item, f"{path}[{i}]", result)

    def _check_unknown_fields(
        self,
        obj: Any,
        path: str,
        result: BoundaryValidationResult,
    ) -> None:
        """Check for unknown fields (not in allowed list)."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key

                # Skip nested objects (they have their own checks)
                if key not in ALLOWED_LIFECYCLE_FIELDS:
                    # Check if it's a subfield of an allowed field
                    parent = path.split(".")[-1] if path else ""
                    if parent not in ("export_config", "signature"):
                        result.add_violation(BoundaryViolation(
                            violation_type=BoundaryViolationType.UNKNOWN_FIELD,
                            severity="medium",
                            message=f"Unknown field '{key}' not in allowed list",
                            field_path=current_path,
                            blocked=False,  # Warning only in strict mode
                        ))

    def get_violations_log(self) -> List[BoundaryViolation]:
        """Get log of all violations."""
        return list(self._violations_log)


class CloudCommandFilter:
    """
    Filters and sanitizes commands before sending.

    Removes any prohibited fields and validates command structure.
    Use this as a pre-processing step before validation.

    Usage:
        filter = CloudCommandFilter()
        sanitized = filter.sanitize_command(command)
    """

    @staticmethod
    def sanitize_command(command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize a command by removing prohibited fields.

        Args:
            command: Command to sanitize

        Returns:
            Sanitized command (new dict)
        """
        def _sanitize_recursive(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {
                    k: _sanitize_recursive(v)
                    for k, v in obj.items()
                    if k.lower() not in PROHIBITED_INTENT_FIELDS
                }
            elif isinstance(obj, list):
                return [_sanitize_recursive(item) for item in obj]
            else:
                return obj

        return _sanitize_recursive(command)

    @staticmethod
    def is_lifecycle_command(command_type: str) -> bool:
        """Check if command type is a valid lifecycle command."""
        return command_type.upper() in LIFECYCLE_COMMANDS


class RuntimeBoundaryEnforcer:
    """
    Runtime enforcer for Cloud boundary - FAIL-CLOSED.

    Wraps Cloud control plane methods to enforce boundary at runtime.
    If validation fails, the operation is BLOCKED.

    Usage:
        enforcer = RuntimeBoundaryEnforcer()

        @enforcer.enforce
        def send_command(self, command: dict):
            # This will be blocked if command contains prohibited fields
            ...
    """

    def __init__(self, validator: Optional[CloudBoundaryValidator] = None):
        """
        Initialize enforcer.

        Args:
            validator: Validator to use (creates default if None)
        """
        self._validator = validator or CloudBoundaryValidator(strict_mode=True)
        self._blocked_count = 0
        self._passed_count = 0

    def enforce(self, func: Callable) -> Callable:
        """
        Decorator to enforce boundary on function.

        The decorated function's dict arguments are validated.
        If validation fails, a BoundaryViolationError is raised.

        Args:
            func: Function to wrap

        Returns:
            Wrapped function
        """
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate all dict arguments
            for i, arg in enumerate(args):
                if isinstance(arg, dict):
                    result = self._validator.validate_outgoing_message(
                        arg, f"arg[{i}]"
                    )
                    if not result.valid:
                        self._blocked_count += 1
                        raise BoundaryViolationError(
                            f"Boundary violation in {func.__name__}: "
                            f"{result.violations[0].message}"
                        )

            for key, value in kwargs.items():
                if isinstance(value, dict):
                    result = self._validator.validate_outgoing_message(
                        value, f"kwarg[{key}]"
                    )
                    if not result.valid:
                        self._blocked_count += 1
                        raise BoundaryViolationError(
                            f"Boundary violation in {func.__name__}: "
                            f"{result.violations[0].message}"
                        )

            self._passed_count += 1
            return func(*args, **kwargs)

        return wrapper

    def validate_and_send(
        self,
        message: Dict[str, Any],
        send_func: Callable[[Dict[str, Any]], Any],
    ) -> Any:
        """
        Validate message and send if valid.

        Args:
            message: Message to send
            send_func: Function to call if valid

        Returns:
            Result of send_func

        Raises:
            BoundaryViolationError: If validation fails
        """
        result = self._validator.validate_outgoing_message(message)

        if not result.valid:
            self._blocked_count += 1
            raise BoundaryViolationError(
                f"Message blocked: {result.violations[0].message}"
            )

        self._passed_count += 1
        return send_func(message)

    @property
    def stats(self) -> Dict[str, int]:
        """Get enforcement statistics."""
        return {
            "blocked": self._blocked_count,
            "passed": self._passed_count,
            "total": self._blocked_count + self._passed_count,
        }


class BoundaryViolationError(Exception):
    """Exception raised when Cloud boundary is violated."""

    def __init__(self, message: str, violations: Optional[List[BoundaryViolation]] = None):
        super().__init__(message)
        self.violations = violations or []


# ============================================================================
# Factory Functions
# ============================================================================

def create_boundary_validator(strict: bool = True) -> CloudBoundaryValidator:
    """Create a configured boundary validator."""
    return CloudBoundaryValidator(strict_mode=strict)


def create_boundary_enforcer() -> RuntimeBoundaryEnforcer:
    """Create a configured boundary enforcer."""
    return RuntimeBoundaryEnforcer()


# ============================================================================
# Convenience Functions
# ============================================================================

def validate_cloud_message(message: Dict[str, Any]) -> bool:
    """
    Quick validation of a Cloud message.

    Args:
        message: Message to validate

    Returns:
        True if valid, False if blocked
    """
    validator = CloudBoundaryValidator()
    result = validator.validate_outgoing_message(message)
    return result.valid


def assert_no_intent_fields(data: Dict[str, Any]) -> None:
    """
    Assert that data contains no intent fields.

    Args:
        data: Data to check

    Raises:
        BoundaryViolationError: If intent fields found
    """
    violations = check_dict_for_prohibited_fields(data, "data")
    if violations:
        raise BoundaryViolationError(
            f"Intent field detected: {violations[0].message}",
            [BoundaryViolation(
                violation_type=BoundaryViolationType.PROHIBITED_FIELD,
                severity="critical",
                message=v.message,
                field_path=v.location,
            ) for v in violations]
        )
