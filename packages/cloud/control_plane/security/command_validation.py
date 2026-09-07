# -*- coding: utf-8 -*-
"""
Command Type Validation - WI-CLOUD-01.

CLOUD ZONE ONLY.

Enforces fail-closed command type validation using enum allowlist.
Only lifecycle commands from the schema are allowed.

Design Doc Reference:
    - "Cloud отправляет только lifecycle requests"
    - CCEA Boundary: no order-like payloads
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Final, FrozenSet, List, Optional

from ccea.contracts.enums import COMMAND_TYPES


# ============================================================================
# Constants - Single Source of Truth from Schema
# ============================================================================

# Allowed command types (from protocol_messages.schema.json via contracts)
ALLOWED_COMMAND_TYPES: Final[FrozenSet[str]] = COMMAND_TYPES

# Command type metadata for validation and approval workflow
COMMAND_TYPE_METADATA: Final[Dict[str, Dict[str, Any]]] = {
    "REQUEST_START_RUN": {
        "requires_approval_default": True,
        "change_class": "trading_impacting",
        "description": "Start a deployment run",
    },
    "REQUEST_STOP_RUN": {
        "requires_approval_default": False,
        "change_class": "operational",
        "is_safety_operation": True,
        "description": "Stop a running deployment",
    },
    "REQUEST_PAUSE_RUN": {
        "requires_approval_default": False,
        "change_class": "operational",
        "is_safety_operation": True,
        "description": "Pause a running deployment",
    },
    "REQUEST_UPGRADE_ARTIFACT": {
        "requires_approval_default": True,
        "change_class": "trading_impacting",
        "description": "Upgrade deployed artifact to new version",
    },
    "REQUEST_UPDATE_CONFIG": {
        "requires_approval_default": True,
        "change_class": "trading_impacting",
        "description": "Update deployment configuration",
    },
    "REQUEST_ROTATE_AGENT_SESSION": {
        "requires_approval_default": True,
        "change_class": "security_sensitive",
        "description": "Rotate agent session credentials",
    },
    "REQUEST_EXPORT_LOGS": {
        "requires_approval_default": True,
        "change_class": "data_sensitive",
        "description": "Export agent logs (break-glass operation)",
    },
}


# ============================================================================
# Validation Results
# ============================================================================


class ValidationSeverity(str, Enum):
    """Severity of validation error."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class CommandValidationError:
    """Error from command type validation."""

    message: str
    severity: ValidationSeverity
    command_type: str
    blocked: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message,
            "severity": self.severity.value,
            "command_type": self.command_type,
            "blocked": self.blocked,
        }


@dataclass
class CommandValidationResult:
    """Result of command type validation."""

    valid: bool
    command_type: str
    errors: List[CommandValidationError]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "command_type": self.command_type,
            "errors": [e.to_dict() for e in self.errors],
            "metadata": self.metadata,
        }


# ============================================================================
# Validator
# ============================================================================


class CommandTypeValidator:
    """
    Validates command types against the schema allowlist.

    FAIL-CLOSED: Unknown or invalid command types are rejected.

    Usage:
        validator = CommandTypeValidator()
        result = validator.validate("REQUEST_START_RUN")
        if not result.valid:
            raise ValidationError(result.errors[0].message)
    """

    def __init__(self, allowed_types: Optional[FrozenSet[str]] = None):
        """
        Initialize validator.

        Args:
            allowed_types: Override allowed types (for testing)
        """
        self._allowed_types = allowed_types or ALLOWED_COMMAND_TYPES
        self._metadata = COMMAND_TYPE_METADATA

    def validate(self, command_type: str) -> CommandValidationResult:
        """
        Validate a command type.

        Args:
            command_type: Command type string to validate

        Returns:
            CommandValidationResult with validation status
        """
        errors: List[CommandValidationError] = []

        # Normalize to uppercase for comparison
        normalized = command_type.upper().strip()

        # Check if empty
        if not normalized:
            errors.append(
                CommandValidationError(
                    message="Command type cannot be empty",
                    severity=ValidationSeverity.CRITICAL,
                    command_type=command_type,
                )
            )
            return CommandValidationResult(
                valid=False,
                command_type=command_type,
                errors=errors,
            )

        # Check against allowlist (FAIL-CLOSED)
        if normalized not in self._allowed_types:
            errors.append(
                CommandValidationError(
                    message=f"Unknown command type '{command_type}'. "
                    f"Allowed types: {sorted(self._allowed_types)}",
                    severity=ValidationSeverity.CRITICAL,
                    command_type=command_type,
                )
            )
            return CommandValidationResult(
                valid=False,
                command_type=command_type,
                errors=errors,
            )

        # Get metadata for valid command
        metadata = self._metadata.get(normalized)

        return CommandValidationResult(
            valid=True,
            command_type=normalized,
            errors=[],
            metadata=metadata,
        )

    def is_valid(self, command_type: str) -> bool:
        """Quick check if command type is valid."""
        return self.validate(command_type).valid

    def get_allowed_types(self) -> List[str]:
        """Get list of allowed command types."""
        return sorted(self._allowed_types)

    def get_metadata(self, command_type: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a command type."""
        normalized = command_type.upper().strip()
        return self._metadata.get(normalized)

    def requires_approval(self, command_type: str) -> bool:
        """Check if command type requires approval by default."""
        metadata = self.get_metadata(command_type)
        if metadata is None:
            return True  # Unknown commands require approval (fail-safe)
        return metadata.get("requires_approval_default", True)

    def is_safety_operation(self, command_type: str) -> bool:
        """Check if command type is a safety operation (stop/pause)."""
        metadata = self.get_metadata(command_type)
        if metadata is None:
            return False
        return metadata.get("is_safety_operation", False)


# ============================================================================
# Convenience Functions
# ============================================================================

# Singleton validator instance
_default_validator: Optional[CommandTypeValidator] = None


def get_validator() -> CommandTypeValidator:
    """Get the default validator instance."""
    global _default_validator
    if _default_validator is None:
        _default_validator = CommandTypeValidator()
    return _default_validator


def validate_command_type(command_type: str) -> CommandValidationResult:
    """
    Validate a command type using the default validator.

    Args:
        command_type: Command type string to validate

    Returns:
        CommandValidationResult
    """
    return get_validator().validate(command_type)


def is_valid_command_type(command_type: str) -> bool:
    """Quick check if command type is valid."""
    return get_validator().is_valid(command_type)


def assert_valid_command_type(command_type: str) -> str:
    """
    Assert that command type is valid.

    Args:
        command_type: Command type to validate

    Returns:
        Normalized command type

    Raises:
        ValueError: If command type is invalid
    """
    result = validate_command_type(command_type)
    if not result.valid:
        raise ValueError(result.errors[0].message)
    return result.command_type
