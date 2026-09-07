# -*- coding: utf-8 -*-
"""
CCEA Schema Validation.

Validates JSON schemas to ensure they don't contain order-like payloads.
This is a critical security guardrail for CCEA architecture.

Usage:
    python -m ccea.guardrails.schema_check docs/schemas/
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Set

# ============================================================================
# Constants
# ============================================================================

# Fields that indicate order-like payload (PROHIBITED in protocol)
PROHIBITED_FIELDS: Final[Set[str]] = frozenset(
    {
        "side",
        "quantity",
        "qty",
        "price",
        "order_type",
        "target_position",
        "execute_order",
        "place_order",
        "submit_order",
        "intent",
        "signal",
    }
)

# Values that indicate order-like context
PROHIBITED_VALUES: Final[Dict[str, List[str]]] = {
    "side": ["BUY", "SELL", "buy", "sell"],
    "order_type": ["MARKET", "LIMIT", "market", "limit"],
}

# Fields that are allowed in certain contexts
ALLOWED_CONTEXTS: Final[Dict[str, Set[str]]] = {
    # In artifact_manifest, risk_profile_suggested is allowed to have these
    "risk_profile_suggested": set(),
    # In telemetry, we allow aggregated metrics
    "telemetry": {"trade_count", "order_count"},
}


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class SchemaViolation:
    """Represents a schema violation."""

    file_path: str
    json_path: str
    field_name: str
    reason: str
    severity: str = "ERROR"

    def __str__(self) -> str:
        return (
            f"[{self.severity}] {self.file_path}: "
            f"{self.json_path}.{self.field_name} - {self.reason}"
        )


@dataclass
class SchemaCheckResult:
    """Result of schema validation."""

    violations: List[SchemaViolation] = field(default_factory=list)
    warnings: List[SchemaViolation] = field(default_factory=list)
    files_checked: int = 0
    passed: bool = True

    def add_violation(self, violation: SchemaViolation) -> None:
        if violation.severity == "WARNING":
            self.warnings.append(violation)
        else:
            self.violations.append(violation)
            self.passed = False


# ============================================================================
# Schema Validation
# ============================================================================


def check_prohibited_fields(
    schema: Dict[str, Any],
    path: str = "",
    file_path: str = "",
    context: Optional[str] = None,
) -> List[SchemaViolation]:
    """
    Recursively check schema for prohibited fields.

    Args:
        schema: JSON schema dictionary
        path: Current JSON path
        file_path: Source file path
        context: Current context (e.g., 'telemetry')

    Returns:
        List of violations found
    """
    violations = []

    if not isinstance(schema, dict):
        return violations

    # Check properties
    if "properties" in schema:
        for field_name, field_spec in schema["properties"].items():
            current_path = f"{path}.properties.{field_name}" if path else f"properties.{field_name}"

            # Check if field is prohibited
            if field_name in PROHIBITED_FIELDS:
                # Check if allowed in current context
                if context and field_name in ALLOWED_CONTEXTS.get(context, set()):
                    continue

                violations.append(
                    SchemaViolation(
                        file_path=file_path,
                        json_path=path,
                        field_name=field_name,
                        reason=f"order-like field '{field_name}' prohibited in protocol",
                    )
                )

            # Check for prohibited enum values
            if "enum" in field_spec:
                if field_name in PROHIBITED_VALUES:
                    for val in field_spec["enum"]:
                        if val in PROHIBITED_VALUES[field_name]:
                            violations.append(
                                SchemaViolation(
                                    file_path=file_path,
                                    json_path=current_path,
                                    field_name=field_name,
                                    reason=f"order-like value '{val}' prohibited",
                                )
                            )

            # Recurse into nested schemas
            nested_context = field_name if field_name in ALLOWED_CONTEXTS else context
            violations.extend(
                check_prohibited_fields(field_spec, current_path, file_path, nested_context)
            )

    # Check items (for arrays)
    if "items" in schema:
        items_path = f"{path}.items" if path else "items"
        violations.extend(check_prohibited_fields(schema["items"], items_path, file_path, context))

    # Check allOf, anyOf, oneOf
    for combinator in ("allOf", "anyOf", "oneOf"):
        if combinator in schema:
            for i, sub_schema in enumerate(schema[combinator]):
                comb_path = f"{path}.{combinator}[{i}]" if path else f"{combinator}[{i}]"
                violations.extend(
                    check_prohibited_fields(sub_schema, comb_path, file_path, context)
                )

    # Check definitions
    if "definitions" in schema:
        for def_name, def_spec in schema["definitions"].items():
            # Skip the prohibited_fields definition itself
            if "prohibited" in def_name.lower():
                continue
            def_path = f"{path}.definitions.{def_name}" if path else f"definitions.{def_name}"
            violations.extend(check_prohibited_fields(def_spec, def_path, file_path, context))

    # Check $defs (JSON Schema draft-07+)
    if "$defs" in schema:
        for def_name, def_spec in schema["$defs"].items():
            if "prohibited" in def_name.lower():
                continue
            def_path = f"{path}.$defs.{def_name}" if path else f"$defs.{def_name}"
            violations.extend(check_prohibited_fields(def_spec, def_path, file_path, context))

    return violations


def validate_manifest_schema(schema_path: Path) -> SchemaCheckResult:
    """
    Validate artifact manifest schema.

    Args:
        schema_path: Path to schema file

    Returns:
        SchemaCheckResult with any violations
    """
    result = SchemaCheckResult(files_checked=1)

    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        result.add_violation(
            SchemaViolation(
                file_path=str(schema_path),
                json_path="",
                field_name="",
                reason=f"Failed to load schema: {e}",
            )
        )
        return result

    # Check for prohibited fields
    violations = check_prohibited_fields(
        schema,
        path="",
        file_path=str(schema_path),
        context="manifest",
    )

    for v in violations:
        result.add_violation(v)

    # Verify required fields
    required_fields = {
        "schema_version",
        "artifact_id",
        "artifact_type",
        "entrypoint",
        "runtime",
        "deps_lock_digest",
        "created_at",
    }

    if "required" in schema:
        schema_required = set(schema["required"])
        missing = required_fields - schema_required
        if missing:
            result.add_violation(
                SchemaViolation(
                    file_path=str(schema_path),
                    json_path="required",
                    field_name="",
                    reason=f"Missing required fields: {missing}",
                    severity="WARNING",
                )
            )

    return result


def validate_protocol_schema(schema_path: Path) -> SchemaCheckResult:
    """
    Validate protocol messages schema.

    Args:
        schema_path: Path to schema file

    Returns:
        SchemaCheckResult with any violations
    """
    result = SchemaCheckResult(files_checked=1)

    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        result.add_violation(
            SchemaViolation(
                file_path=str(schema_path),
                json_path="",
                field_name="",
                reason=f"Failed to load schema: {e}",
            )
        )
        return result

    # Check for prohibited fields in message definitions
    # Skip the explicit "prohibited_order_fields" definition
    if "definitions" in schema:
        for def_name, def_spec in schema["definitions"].items():
            if "prohibited" in def_name.lower():
                continue
            if def_name == "messages":
                # Check each message type
                if isinstance(def_spec, dict):
                    for msg_name, msg_spec in def_spec.items():
                        violations = check_prohibited_fields(
                            msg_spec,
                            path=f"definitions.messages.{msg_name}",
                            file_path=str(schema_path),
                            context="protocol",
                        )
                        for v in violations:
                            result.add_violation(v)

    return result


def validate_schema_file(schema_path: Path) -> SchemaCheckResult:
    """
    Validate any schema file.

    Args:
        schema_path: Path to schema file

    Returns:
        SchemaCheckResult with any violations
    """
    if "manifest" in schema_path.name.lower():
        return validate_manifest_schema(schema_path)
    elif "protocol" in schema_path.name.lower():
        return validate_protocol_schema(schema_path)
    else:
        # Generic validation
        result = SchemaCheckResult(files_checked=1)
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
            violations = check_prohibited_fields(
                schema,
                path="",
                file_path=str(schema_path),
            )
            for v in violations:
                result.add_violation(v)
        except Exception as e:
            result.add_violation(
                SchemaViolation(
                    file_path=str(schema_path),
                    json_path="",
                    field_name="",
                    reason=f"Failed to validate: {e}",
                )
            )
        return result


def validate_schemas_directory(directory: Path) -> SchemaCheckResult:
    """
    Validate all schema files in a directory.

    Args:
        directory: Directory containing schema files

    Returns:
        Combined SchemaCheckResult
    """
    combined_result = SchemaCheckResult()

    schema_files = list(directory.glob("*.schema.json")) + list(directory.glob("*.json"))

    for schema_file in schema_files:
        result = validate_schema_file(schema_file)
        combined_result.files_checked += result.files_checked
        combined_result.violations.extend(result.violations)
        combined_result.warnings.extend(result.warnings)
        if not result.passed:
            combined_result.passed = False

    return combined_result


# ============================================================================
# CLI Interface
# ============================================================================


def main() -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="CCEA Schema Validation")
    parser.add_argument(
        "path",
        type=Path,
        help="Schema file or directory to validate",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Treat warnings as errors",
    )

    args = parser.parse_args()

    path = args.path

    if path.is_dir():
        result = validate_schemas_directory(path)
    elif path.is_file():
        result = validate_schema_file(path)
    else:
        print(f"Error: {path} not found")
        return 1

    print(f"Schema validation: {result.files_checked} files checked")

    if result.violations:
        print(f"\nErrors ({len(result.violations)}):")
        for v in result.violations:
            print(f"  {v}")

    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for w in result.warnings:
            print(f"  {w}")

    if result.passed:
        print("\nSchema validation PASSED")
        return 0
    else:
        print("\nSchema validation FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
