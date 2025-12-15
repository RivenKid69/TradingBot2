# -*- coding: utf-8 -*-
"""
CCEA Intent Prohibition Guardrails.

Enforces CRITICAL security boundary: Cloud CANNOT send intents/signals to Agent.

Design Doc Phase 3 Requirement:
    "В cloud API/протоколе отсутствуют сообщения, несущие live‑intent/targets.
     Cloud отправляет только lifecycle requests."

This guardrail ensures:
1. No intent/signal fields in protocol messages
2. No order-like payloads in any Cloud API
3. No target positions or trading parameters
4. Lifecycle-only commands from Cloud to Agent

SECURITY: Violations of this guardrail are CRITICAL security issues.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Final, FrozenSet, List, Optional, Set


# ============================================================================
# Prohibited Field Definitions
# ============================================================================

# Fields that indicate trading intent - STRICTLY PROHIBITED
PROHIBITED_INTENT_FIELDS: Final[FrozenSet[str]] = frozenset({
    # Order fields
    "side",
    "quantity",
    "qty",
    "price",
    "order_type",
    "order_size",
    "fill_price",
    "limit_price",
    "stop_price",

    # Intent fields
    "intent",
    "intent_type",
    "order_intent",
    "trading_intent",
    "signal",
    "signal_value",
    "trade_signal",

    # Target fields
    "target_position",
    "target_quantity",
    "target_notional",
    "target_qty",
    "desired_position",
    "target_size",

    # Action fields
    "execute_order",
    "place_order",
    "submit_order",
    "create_order",
    "send_order",

    # Position change fields
    "position_change",
    "delta_position",
    "adjustment",
})

# Patterns that indicate intent/signal - checked via regex
PROHIBITED_PATTERNS: Final[List[str]] = [
    r"side\s*[=:]",
    r"quantity\s*[=:]",
    r"price\s*[=:]",
    r"intent\s*[=:]",
    r"signal\s*[=:]",
    r"target.*position",
    r"place.*order",
    r"execute.*order",
    r"submit.*order",
    r"order.*intent",
    r"trading.*signal",
]

# Message types that are allowed in Cloud -> Agent direction
ALLOWED_CLOUD_TO_AGENT_TYPES: Final[FrozenSet[str]] = frozenset({
    "COMMAND_BATCH",
    "REQUEST_START_RUN",
    "REQUEST_STOP_RUN",
    "REQUEST_PAUSE_RUN",
    "REQUEST_UPGRADE_ARTIFACT",
    "REQUEST_UPDATE_CONFIG",
    "REQUEST_ROTATE_AGENT_SESSION",
    "REQUEST_EXPORT_LOGS",
})

# Command types that would indicate intent injection
PROHIBITED_COMMAND_TYPES: Final[FrozenSet[str]] = frozenset({
    "PLACE_ORDER",
    "SUBMIT_ORDER",
    "EXECUTE_ORDER",
    "SEND_ORDER",
    "CREATE_ORDER",
    "SET_TARGET",
    "SET_POSITION",
    "PUSH_INTENT",
    "PUSH_SIGNAL",
    "SEND_INTENT",
    "SEND_SIGNAL",
    "EXECUTE_INTENT",
    "SET_TARGET_POSITION",
    "ADJUST_POSITION",
})


# ============================================================================
# Violation Classes
# ============================================================================

class ViolationSeverity(str, Enum):
    """Severity of intent prohibition violation."""

    CRITICAL = "critical"  # Direct order-like payload
    HIGH = "high"  # Intent/signal field detected
    MEDIUM = "medium"  # Suspicious pattern
    LOW = "low"  # Potential issue


@dataclass
class IntentProhibitionViolation:
    """Represents an intent prohibition violation."""

    severity: ViolationSeverity
    message: str
    location: str = ""  # file:line or message field
    field_name: Optional[str] = None
    detected_value: Optional[str] = None

    def __str__(self) -> str:
        loc = f" at {self.location}" if self.location else ""
        return f"[{self.severity.value.upper()}]{loc}: {self.message}"


@dataclass
class IntentProhibitionResult:
    """Result of intent prohibition check."""

    passed: bool = True
    violations: List[IntentProhibitionViolation] = field(default_factory=list)
    checked_items: int = 0

    def add_violation(self, violation: IntentProhibitionViolation) -> None:
        """Add a violation and update passed status."""
        self.violations.append(violation)
        if violation.severity in (ViolationSeverity.CRITICAL, ViolationSeverity.HIGH):
            self.passed = False

    @property
    def critical_count(self) -> int:
        """Count critical violations."""
        return sum(1 for v in self.violations if v.severity == ViolationSeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        """Count high severity violations."""
        return sum(1 for v in self.violations if v.severity == ViolationSeverity.HIGH)


# ============================================================================
# Check Functions
# ============================================================================

def check_dict_for_prohibited_fields(
    data: Dict[str, Any],
    location: str = "",
) -> List[IntentProhibitionViolation]:
    """
    Check dictionary for prohibited intent/signal fields.

    Args:
        data: Dictionary to check
        location: Location for error messages

    Returns:
        List of violations found
    """
    violations = []

    def _check_recursive(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key

                # Check key against prohibited fields
                key_lower = key.lower()
                if key_lower in PROHIBITED_INTENT_FIELDS:
                    violations.append(IntentProhibitionViolation(
                        severity=ViolationSeverity.CRITICAL if key_lower in {"side", "quantity", "price"} else ViolationSeverity.HIGH,
                        message=f"Prohibited field '{key}' found in protocol message",
                        location=f"{location}:{current_path}" if location else current_path,
                        field_name=key,
                        detected_value=str(value)[:100],
                    ))

                # Recurse
                _check_recursive(value, current_path)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _check_recursive(item, f"{path}[{i}]")

    _check_recursive(data, "")
    return violations


def check_json_string_for_patterns(
    json_string: str,
    location: str = "",
) -> List[IntentProhibitionViolation]:
    """
    Check JSON string for prohibited patterns.

    Args:
        json_string: JSON string to check
        location: Location for error messages

    Returns:
        List of violations found
    """
    violations = []

    for pattern in PROHIBITED_PATTERNS:
        matches = re.findall(pattern, json_string, re.IGNORECASE)
        if matches:
            violations.append(IntentProhibitionViolation(
                severity=ViolationSeverity.MEDIUM,
                message=f"Suspicious pattern found: '{pattern}'",
                location=location,
                detected_value=matches[0] if matches else None,
            ))

    return violations


def check_command_type(
    command_type: str,
    location: str = "",
) -> Optional[IntentProhibitionViolation]:
    """
    Check if command type is prohibited.

    Args:
        command_type: Command type to check
        location: Location for error messages

    Returns:
        Violation if prohibited, None if OK
    """
    command_upper = command_type.upper()

    if command_upper in PROHIBITED_COMMAND_TYPES:
        return IntentProhibitionViolation(
            severity=ViolationSeverity.CRITICAL,
            message=f"PROHIBITED command type: '{command_type}' - Intent injection attempt",
            location=location,
            field_name="command_type",
            detected_value=command_type,
        )

    # Check for order-like patterns
    order_patterns = ["ORDER", "TRADE", "EXECUTE", "TARGET", "SIGNAL", "INTENT"]
    for pattern in order_patterns:
        if pattern in command_upper and command_upper not in ALLOWED_CLOUD_TO_AGENT_TYPES:
            return IntentProhibitionViolation(
                severity=ViolationSeverity.HIGH,
                message=f"Suspicious command type: '{command_type}' - May indicate intent injection",
                location=location,
                field_name="command_type",
                detected_value=command_type,
            )

    return None


def check_protocol_message(
    message: Dict[str, Any],
    message_name: str = "",
) -> IntentProhibitionResult:
    """
    Check a protocol message for intent prohibition violations.

    Args:
        message: Message dictionary
        message_name: Name of message type

    Returns:
        IntentProhibitionResult with all violations
    """
    result = IntentProhibitionResult()
    result.checked_items = 1

    # Check command type if present
    if "command_type" in message:
        violation = check_command_type(
            str(message["command_type"]),
            f"{message_name}.command_type",
        )
        if violation:
            result.add_violation(violation)

    # Check for prohibited fields
    violations = check_dict_for_prohibited_fields(message, message_name)
    for v in violations:
        result.add_violation(v)

    return result


def check_python_source_for_intent_injection(
    source_code: str,
    file_path: str = "",
) -> IntentProhibitionResult:
    """
    Check Python source code for intent injection vulnerabilities.

    Args:
        source_code: Python source code
        file_path: Path to file

    Returns:
        IntentProhibitionResult with violations
    """
    result = IntentProhibitionResult()

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return result

    class IntentInjectionVisitor(ast.NodeVisitor):
        def visit_Dict(self, node: ast.Dict) -> None:
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    if key.value.lower() in PROHIBITED_INTENT_FIELDS:
                        result.add_violation(IntentProhibitionViolation(
                            severity=ViolationSeverity.HIGH,
                            message=f"Prohibited field '{key.value}' in dict literal",
                            location=f"{file_path}:{key.lineno}",
                            field_name=key.value,
                        ))
            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript) -> None:
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                if node.slice.value.lower() in PROHIBITED_INTENT_FIELDS:
                    result.add_violation(IntentProhibitionViolation(
                        severity=ViolationSeverity.MEDIUM,
                        message=f"Accessing prohibited field '{node.slice.value}'",
                        location=f"{file_path}:{node.lineno}",
                        field_name=node.slice.value,
                    ))
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            # Check for dict() calls with prohibited keys
            for keyword in node.keywords:
                if keyword.arg and keyword.arg.lower() in PROHIBITED_INTENT_FIELDS:
                    result.add_violation(IntentProhibitionViolation(
                        severity=ViolationSeverity.HIGH,
                        message=f"Prohibited keyword argument '{keyword.arg}'",
                        location=f"{file_path}:{node.lineno}",
                        field_name=keyword.arg,
                    ))
            self.generic_visit(node)

    visitor = IntentInjectionVisitor()
    visitor.visit(tree)
    result.checked_items = 1

    return result


def check_cloud_package_for_intents(
    cloud_package_path: Path,
    exclude_tests: bool = True,
) -> IntentProhibitionResult:
    """
    Check entire Cloud package for intent prohibition violations.

    Args:
        cloud_package_path: Path to packages/cloud
        exclude_tests: If True, exclude test files (they need prohibited
                       fields to test the guardrails themselves)

    Returns:
        IntentProhibitionResult with all violations
    """
    result = IntentProhibitionResult()

    if not cloud_package_path.exists():
        return result

    # Find all Python files
    python_files = list(cloud_package_path.rglob("*.py"))

    for py_file in python_files:
        # Skip test files - they legitimately need prohibited fields
        # to test that guardrails detect them correctly
        if exclude_tests:
            # Check if file is in a tests directory (relative to cloud_package_path)
            # This avoids matching pytest temp directories that contain "test_"
            try:
                relative_path = py_file.relative_to(cloud_package_path)
                relative_str = str(relative_path)
                if (
                    relative_str.startswith("tests/")
                    or "/tests/" in relative_str
                    or py_file.name.startswith("test_")
                    or py_file.name.endswith("_test.py")
                ):
                    continue
            except ValueError:
                pass  # Not relative, check anyway

        try:
            source = py_file.read_text(encoding="utf-8")
            file_result = check_python_source_for_intent_injection(
                source,
                str(py_file),
            )
            result.checked_items += file_result.checked_items
            for v in file_result.violations:
                result.add_violation(v)
        except Exception:
            pass

    return result


def validate_cloud_api_endpoint(
    request_body: Dict[str, Any],
    endpoint: str = "",
) -> IntentProhibitionResult:
    """
    Validate a Cloud API request body for intent prohibition.

    This should be called on every Cloud -> Agent communication.

    Args:
        request_body: Request body dictionary
        endpoint: API endpoint for logging

    Returns:
        IntentProhibitionResult
    """
    result = IntentProhibitionResult()
    result.checked_items = 1

    # Check for prohibited fields
    violations = check_dict_for_prohibited_fields(request_body, endpoint)
    for v in violations:
        result.add_violation(v)

    # Check command type if this is a command
    if "command_type" in request_body:
        cmd_violation = check_command_type(
            str(request_body["command_type"]),
            endpoint,
        )
        if cmd_violation:
            result.add_violation(cmd_violation)

    # Check commands array
    if "commands" in request_body and isinstance(request_body["commands"], list):
        for i, cmd in enumerate(request_body["commands"]):
            if isinstance(cmd, dict):
                cmd_violations = check_dict_for_prohibited_fields(
                    cmd,
                    f"{endpoint}.commands[{i}]",
                )
                for v in cmd_violations:
                    result.add_violation(v)

                if "command_type" in cmd:
                    cmd_violation = check_command_type(
                        str(cmd["command_type"]),
                        f"{endpoint}.commands[{i}].command_type",
                    )
                    if cmd_violation:
                        result.add_violation(cmd_violation)

    return result


# ============================================================================
# CI Integration
# ============================================================================

def run_ci_check(cloud_package_path: Path) -> int:
    """
    Run CI check for intent prohibition.

    Args:
        cloud_package_path: Path to packages/cloud

    Returns:
        Exit code (0 = pass, 1 = fail)
    """
    print("=" * 60)
    print("CCEA Intent Prohibition Check")
    print("=" * 60)

    result = check_cloud_package_for_intents(cloud_package_path)

    print(f"Checked {result.checked_items} files")
    print(f"Critical violations: {result.critical_count}")
    print(f"High severity violations: {result.high_count}")
    print(f"Total violations: {len(result.violations)}")

    if result.violations:
        print("\nViolations:")
        for v in result.violations:
            print(f"  {v}")

    if result.passed:
        print("\n[PASS] Intent prohibition check PASSED")
        return 0
    else:
        print("\n[FAIL] Intent prohibition check FAILED")
        return 1


# ============================================================================
# Runtime Validation Decorator
# ============================================================================

def prohibit_intent_fields(func):
    """
    Decorator to validate function arguments don't contain prohibited fields.

    Usage:
        @prohibit_intent_fields
        def send_command(self, command_data: dict):
            ...
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check kwargs
        for key, value in kwargs.items():
            if isinstance(value, dict):
                violations = check_dict_for_prohibited_fields(value, f"kwarg:{key}")
                if violations:
                    raise ValueError(
                        f"Intent prohibition violation in {func.__name__}: "
                        f"{violations[0].message}"
                    )

        # Check args that are dicts
        for i, arg in enumerate(args):
            if isinstance(arg, dict):
                violations = check_dict_for_prohibited_fields(arg, f"arg:{i}")
                if violations:
                    raise ValueError(
                        f"Intent prohibition violation in {func.__name__}: "
                        f"{violations[0].message}"
                    )

        return func(*args, **kwargs)

    return wrapper


# ============================================================================
# CLI Entry Point
# ============================================================================

def main() -> int:
    """CLI entry point."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="CCEA Intent Prohibition Check"
    )
    parser.add_argument(
        "--cloud-path",
        type=Path,
        default=Path("packages/cloud"),
        help="Path to Cloud package",
    )
    parser.add_argument(
        "--check-file",
        type=Path,
        help="Check a specific Python file",
    )

    args = parser.parse_args()

    if args.check_file:
        if not args.check_file.exists():
            print(f"File not found: {args.check_file}")
            return 1

        source = args.check_file.read_text(encoding="utf-8")
        result = check_python_source_for_intent_injection(source, str(args.check_file))

        print(f"Checked: {args.check_file}")
        if result.violations:
            print("Violations:")
            for v in result.violations:
                print(f"  {v}")
            return 1
        else:
            print("No violations found")
            return 0

    return run_ci_check(args.cloud_path)


if __name__ == "__main__":
    import sys
    sys.exit(main())
