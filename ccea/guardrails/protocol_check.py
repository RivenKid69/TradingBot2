# -*- coding: utf-8 -*-
"""
CCEA Protocol Validation.

Validates protocol changes and enforces command type allowlist.
Any new command type requires security review.

Usage:
    python -m ccea.guardrails.protocol_check --base base.json --new new.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Final, FrozenSet, List, Optional, Set

# ============================================================================
# Constants
# ============================================================================

# Allowed command types (Cloud -> Agent)
ALLOWED_COMMAND_TYPES: Final[FrozenSet[str]] = frozenset({
    "REQUEST_START_RUN",
    "REQUEST_STOP_RUN",
    "REQUEST_PAUSE_RUN",
    "REQUEST_UPGRADE_ARTIFACT",
    "REQUEST_UPDATE_CONFIG",
    "REQUEST_ROTATE_AGENT_SESSION",
    "REQUEST_EXPORT_LOGS",
})

# Allowed message types (bidirectional)
ALLOWED_MESSAGE_TYPES: Final[FrozenSet[str]] = frozenset({
    "HEARTBEAT",
    "POLL_COMMANDS",
    "COMMAND_BATCH",
    "COMMAND_ACK",
    "COMMAND_APPROVAL",
    "COMMAND_RESULT",
    "TELEMETRY",
})

# Commands that require local approval
APPROVAL_REQUIRED_COMMANDS: Final[FrozenSet[str]] = frozenset({
    "REQUEST_START_RUN",
    "REQUEST_UPGRADE_ARTIFACT",
    "REQUEST_UPDATE_CONFIG",  # if trading_impacting
    "REQUEST_ROTATE_AGENT_SESSION",
    "REQUEST_EXPORT_LOGS",
})

# Commands that are safety operations (no approval needed)
SAFETY_COMMANDS: Final[FrozenSet[str]] = frozenset({
    "REQUEST_STOP_RUN",
    "REQUEST_PAUSE_RUN",
})


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ProtocolChange:
    """Represents a protocol change."""
    change_type: str  # 'added', 'removed', 'modified'
    element_type: str  # 'command', 'message', 'field'
    element_name: str
    details: str = ""
    requires_security_review: bool = False

    def __str__(self) -> str:
        review = " [SECURITY REVIEW REQUIRED]" if self.requires_security_review else ""
        return f"{self.change_type.upper()}: {self.element_type} '{self.element_name}'{review}"


@dataclass
class ProtocolCheckResult:
    """Result of protocol validation."""
    changes: List[ProtocolChange] = field(default_factory=list)
    requires_security_review: bool = False
    new_commands: Set[str] = field(default_factory=set)
    new_messages: Set[str] = field(default_factory=set)
    removed_commands: Set[str] = field(default_factory=set)
    passed: bool = True

    def add_change(self, change: ProtocolChange) -> None:
        self.changes.append(change)
        if change.requires_security_review:
            self.requires_security_review = True


# ============================================================================
# Validation Functions
# ============================================================================

def validate_command_type(command_type: str) -> tuple[bool, str]:
    """
    Validate that a command type is in the allowlist.

    Args:
        command_type: Command type to validate

    Returns:
        Tuple of (is_valid, reason)
    """
    if command_type in ALLOWED_COMMAND_TYPES:
        return True, ""

    # Check if it looks like a command but isn't allowed
    if command_type.startswith("REQUEST_"):
        return False, f"Unknown command type '{command_type}' - requires security review"

    # Check for prohibited command patterns
    prohibited_patterns = [
        "PLACE_ORDER",
        "SUBMIT_ORDER",
        "EXECUTE",
        "SEND_ORDER",
        "CREATE_ORDER",
        "SET_TARGET",
        "SET_POSITION",
    ]
    for pattern in prohibited_patterns:
        if pattern in command_type.upper():
            return False, f"PROHIBITED: '{command_type}' appears to be an order command"

    return False, f"Unknown command type '{command_type}'"


def validate_message_type(message_type: str) -> tuple[bool, str]:
    """
    Validate that a message type is in the allowlist.

    Args:
        message_type: Message type to validate

    Returns:
        Tuple of (is_valid, reason)
    """
    if message_type in ALLOWED_MESSAGE_TYPES:
        return True, ""

    # Check against the command allowlist too
    if message_type in ALLOWED_COMMAND_TYPES:
        return True, ""

    return False, f"Unknown message type '{message_type}'"


def extract_command_types(schema: Dict[str, Any]) -> Set[str]:
    """
    Extract all command types from a protocol schema.

    Args:
        schema: Protocol schema dictionary

    Returns:
        Set of command type names
    """
    commands = set()

    # Look in definitions.messages
    if "definitions" in schema:
        defs = schema["definitions"]
        if "messages" in defs and isinstance(defs["messages"], dict):
            for msg_name, msg_spec in defs["messages"].items():
                if msg_name.startswith("request_"):
                    # Convert to uppercase
                    cmd = msg_name.upper()
                    commands.add(cmd)

                # Also check for command_type const
                if isinstance(msg_spec, dict):
                    # Check in allOf
                    if "allOf" in msg_spec:
                        for item in msg_spec["allOf"]:
                            if isinstance(item, dict) and "properties" in item:
                                props = item["properties"]
                                if "command_type" in props:
                                    cmd_spec = props["command_type"]
                                    if "const" in cmd_spec:
                                        commands.add(cmd_spec["const"])

    return commands


def extract_message_types(schema: Dict[str, Any]) -> Set[str]:
    """
    Extract all message types from a protocol schema.

    Args:
        schema: Protocol schema dictionary

    Returns:
        Set of message type names
    """
    messages = set()

    # Look in definitions.messages
    if "definitions" in schema:
        defs = schema["definitions"]
        if "messages" in defs and isinstance(defs["messages"], dict):
            for msg_name, msg_spec in defs["messages"].items():
                # Check for message_type const
                if isinstance(msg_spec, dict):
                    if "properties" in msg_spec:
                        props = msg_spec["properties"]
                        if "message_type" in props:
                            mt_spec = props["message_type"]
                            if "const" in mt_spec:
                                messages.add(mt_spec["const"])

    return messages


def check_protocol_changes(
    base_schema: Dict[str, Any],
    new_schema: Dict[str, Any],
) -> ProtocolCheckResult:
    """
    Check for changes between two protocol schemas.

    Args:
        base_schema: Previous version of schema
        new_schema: New version of schema

    Returns:
        ProtocolCheckResult with detected changes
    """
    result = ProtocolCheckResult()

    # Extract command types
    base_commands = extract_command_types(base_schema)
    new_commands = extract_command_types(new_schema)

    # Check for new commands
    added_commands = new_commands - base_commands
    for cmd in added_commands:
        result.new_commands.add(cmd)
        is_valid, reason = validate_command_type(cmd)

        if not is_valid:
            result.add_change(ProtocolChange(
                change_type="added",
                element_type="command",
                element_name=cmd,
                details=reason,
                requires_security_review=True,
            ))
            result.passed = False
        else:
            result.add_change(ProtocolChange(
                change_type="added",
                element_type="command",
                element_name=cmd,
                details="New command in allowlist",
                requires_security_review=True,  # All new commands need review
            ))

    # Check for removed commands
    removed_commands = base_commands - new_commands
    for cmd in removed_commands:
        result.removed_commands.add(cmd)
        result.add_change(ProtocolChange(
            change_type="removed",
            element_type="command",
            element_name=cmd,
            requires_security_review=False,
        ))

    # Extract message types
    base_messages = extract_message_types(base_schema)
    new_messages = extract_message_types(new_schema)

    # Check for new messages
    added_messages = new_messages - base_messages
    for msg in added_messages:
        result.new_messages.add(msg)
        is_valid, reason = validate_message_type(msg)

        if not is_valid:
            result.add_change(ProtocolChange(
                change_type="added",
                element_type="message",
                element_name=msg,
                details=reason,
                requires_security_review=True,
            ))
            result.passed = False
        else:
            result.add_change(ProtocolChange(
                change_type="added",
                element_type="message",
                element_name=msg,
                requires_security_review=True,
            ))

    return result


def validate_protocol_allowlist(schema: Dict[str, Any]) -> ProtocolCheckResult:
    """
    Validate that all command/message types in schema are in allowlist.

    Args:
        schema: Protocol schema to validate

    Returns:
        ProtocolCheckResult with any violations
    """
    result = ProtocolCheckResult()

    commands = extract_command_types(schema)
    messages = extract_message_types(schema)

    for cmd in commands:
        is_valid, reason = validate_command_type(cmd)
        if not is_valid:
            result.add_change(ProtocolChange(
                change_type="invalid",
                element_type="command",
                element_name=cmd,
                details=reason,
                requires_security_review=True,
            ))
            result.passed = False

    for msg in messages:
        is_valid, reason = validate_message_type(msg)
        if not is_valid:
            result.add_change(ProtocolChange(
                change_type="invalid",
                element_type="message",
                element_name=msg,
                details=reason,
                requires_security_review=True,
            ))
            result.passed = False

    return result


# ============================================================================
# CLI Interface
# ============================================================================

def main() -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CCEA Protocol Validation"
    )
    parser.add_argument(
        "--schema",
        type=Path,
        help="Protocol schema file to validate",
    )
    parser.add_argument(
        "--base",
        type=Path,
        help="Base schema for comparison",
    )
    parser.add_argument(
        "--new",
        type=Path,
        help="New schema for comparison",
    )

    args = parser.parse_args()

    if args.schema:
        # Validate single schema
        try:
            with open(args.schema, "r", encoding="utf-8") as f:
                schema = json.load(f)
        except Exception as e:
            print(f"Error loading schema: {e}")
            return 1

        result = validate_protocol_allowlist(schema)

        print(f"Protocol validation for {args.schema}")
        print(f"  Commands found: {len(extract_command_types(schema))}")
        print(f"  Messages found: {len(extract_message_types(schema))}")

        if result.changes:
            print("\nIssues:")
            for change in result.changes:
                print(f"  {change}")

        if result.passed:
            print("\nProtocol validation PASSED")
            return 0
        else:
            print("\nProtocol validation FAILED")
            return 1

    elif args.base and args.new:
        # Compare schemas
        try:
            with open(args.base, "r", encoding="utf-8") as f:
                base_schema = json.load(f)
            with open(args.new, "r", encoding="utf-8") as f:
                new_schema = json.load(f)
        except Exception as e:
            print(f"Error loading schemas: {e}")
            return 1

        result = check_protocol_changes(base_schema, new_schema)

        print("Protocol change detection:")
        print(f"  New commands: {result.new_commands}")
        print(f"  New messages: {result.new_messages}")
        print(f"  Removed commands: {result.removed_commands}")

        if result.changes:
            print("\nChanges:")
            for change in result.changes:
                print(f"  {change}")

        if result.requires_security_review:
            print("\nSECURITY REVIEW REQUIRED")
            return 1

        return 0

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
