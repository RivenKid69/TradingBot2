# -*- coding: utf-8 -*-
"""
CCEA Contract Validation.

Validates consistency between schema, protocol models, and DB models.
Runs as part of CI to prevent enum/state-machine drift.

Usage:
    python -m ccea.contracts.validation

Exit codes:
    0: All contracts consistent
    1: Contract drift detected
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Final, FrozenSet, List, Optional, Set

from ccea.contracts.enums import (
    COMMAND_TYPES,
    MESSAGE_TYPES,
    COMMAND_STATUS_PROTOCOL,
    DEPLOYMENT_STATE_PROTOCOL,
    RUN_STATE_PROTOCOL,
    APPROVAL_STATUS,
    TELEMETRY_LEVEL_PROTOCOL,
    SIGNATURE_ALGORITHMS,
)


# Default schema path
DEFAULT_SCHEMA_PATH: Final[str] = "docs/schemas/protocol_messages.schema.json"


@dataclass
class EnumDrift:
    """Represents drift between two enum sets."""
    enum_name: str
    source: str
    target: str
    missing_in_target: Set[str] = field(default_factory=set)
    extra_in_target: Set[str] = field(default_factory=set)

    @property
    def has_drift(self) -> bool:
        return bool(self.missing_in_target or self.extra_in_target)

    def __str__(self) -> str:
        parts = [f"Enum '{self.enum_name}' ({self.source} vs {self.target}):"]
        if self.missing_in_target:
            parts.append(f"  Missing in {self.target}: {self.missing_in_target}")
        if self.extra_in_target:
            parts.append(f"  Extra in {self.target}: {self.extra_in_target}")
        return "\n".join(parts)


@dataclass
class ContractValidationResult:
    """Result of contract validation."""
    passed: bool = True
    drifts: List[EnumDrift] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    checked_enums: int = 0

    def add_drift(self, drift: EnumDrift) -> None:
        if drift.has_drift:
            self.drifts.append(drift)
            self.passed = False

    def add_error(self, error: str) -> None:
        self.errors.append(error)
        self.passed = False


def extract_schema_enums(schema_path: Path) -> Dict[str, Set[str]]:
    """
    Extract all enum definitions from protocol schema.

    Args:
        schema_path: Path to protocol_messages.schema.json

    Returns:
        Dict mapping enum name to set of values
    """
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    enums: Dict[str, Set[str]] = {}

    # Extract from definitions
    definitions = schema.get("definitions", {})

    # command_status
    if "command_status" in definitions:
        cs = definitions["command_status"]
        if "enum" in cs:
            enums["command_status"] = set(cs["enum"])

    # approval_status
    if "approval_status" in definitions:
        app = definitions["approval_status"]
        if "enum" in app:
            enums["approval_status"] = set(app["enum"])

    # run_state
    if "run_state" in definitions:
        rs = definitions["run_state"]
        if "enum" in rs:
            enums["run_state"] = set(rs["enum"])

    # deployment_state
    if "deployment_state" in definitions:
        ds = definitions["deployment_state"]
        if "enum" in ds:
            enums["deployment_state"] = set(ds["enum"])

    # signature algorithms
    if "signature" in definitions:
        sig = definitions["signature"]
        if "properties" in sig and "algorithm" in sig["properties"]:
            alg = sig["properties"]["algorithm"]
            if "enum" in alg:
                enums["signature_algorithms"] = set(alg["enum"])

    # telemetry level - in telemetry message
    messages = definitions.get("messages", {})
    if "telemetry" in messages:
        tel = messages["telemetry"]
        if "properties" in tel and "level" in tel["properties"]:
            lvl = tel["properties"]["level"]
            if "enum" in lvl:
                enums["telemetry_level"] = set(lvl["enum"])

    # Extract command types from message definitions
    command_types = set()
    for msg_name, msg_spec in messages.items():
        if isinstance(msg_spec, dict):
            # Check in allOf
            if "allOf" in msg_spec:
                for item in msg_spec["allOf"]:
                    if isinstance(item, dict) and "properties" in item:
                        props = item["properties"]
                        if "command_type" in props:
                            cmd_spec = props["command_type"]
                            if "const" in cmd_spec:
                                command_types.add(cmd_spec["const"])
    if command_types:
        enums["command_types"] = command_types

    # Extract message types
    message_types = set()
    for msg_name, msg_spec in messages.items():
        if isinstance(msg_spec, dict):
            if "properties" in msg_spec and "message_type" in msg_spec["properties"]:
                mt = msg_spec["properties"]["message_type"]
                if "const" in mt:
                    message_types.add(mt["const"])
    if message_types:
        enums["message_types"] = message_types

    return enums


def extract_protocol_model_enums() -> Dict[str, Set[str]]:
    """
    Extract enum values from ccea/models/protocol.py.

    Returns:
        Dict mapping enum name to set of values
    """
    from ccea.models.protocol import (
        CommandType,
        CommandStatus,
        ApprovalStatus,
        RunState,
        DeploymentState,
        TelemetryLevel,
        SignatureAlgorithm,
        MessageType,
    )

    return {
        "command_types": set(e.value for e in CommandType),
        "command_status": set(e.value for e in CommandStatus),
        "approval_status": set(e.value for e in ApprovalStatus),
        "run_state": set(e.value for e in RunState),
        "deployment_state": set(e.value for e in DeploymentState),
        "telemetry_level": set(e.value for e in TelemetryLevel),
        "signature_algorithms": set(e.value for e in SignatureAlgorithm),
        "message_types": set(e.value for e in MessageType),
    }


def extract_cloud_db_command_types() -> Set[str]:
    """
    Extract command types from packages/cloud/control_plane/commands.py.

    Returns:
        Set of command type values
    """
    from packages.cloud.control_plane.commands import CommandType

    return set(e.value for e in CommandType)


def extract_boundary_command_types() -> Set[str]:
    """
    Extract lifecycle commands from packages/cloud/control_plane/boundary.py.

    Returns:
        Set of lifecycle command types
    """
    from packages.cloud.control_plane.boundary import LIFECYCLE_COMMANDS

    return set(LIFECYCLE_COMMANDS)


def compare_enum_sets(
    source_name: str,
    target_name: str,
    source: Set[str],
    target: Set[str],
) -> EnumDrift:
    """
    Compare two enum sets and return drift.

    Args:
        source_name: Name of source (e.g., "schema")
        target_name: Name of target (e.g., "protocol_models")
        source: Source enum values
        target: Target enum values

    Returns:
        EnumDrift with any differences
    """
    return EnumDrift(
        enum_name="",  # Set by caller
        source=source_name,
        target=target_name,
        missing_in_target=source - target,
        extra_in_target=target - source,
    )


def validate_contract_consistency(
    schema_path: Optional[Path] = None,
    project_root: Optional[Path] = None,
) -> ContractValidationResult:
    """
    Validate consistency between schema, models, and DB enums.

    Args:
        schema_path: Path to schema file
        project_root: Project root directory

    Returns:
        ContractValidationResult
    """
    result = ContractValidationResult()

    # Determine paths
    if project_root is None:
        cwd = Path.cwd()
        if (cwd / "pyproject.toml").exists():
            project_root = cwd
        elif (cwd.parent / "pyproject.toml").exists():
            project_root = cwd.parent
        else:
            project_root = cwd

    if schema_path is None:
        schema_path = project_root / DEFAULT_SCHEMA_PATH

    # Load schema enums
    try:
        schema_enums = extract_schema_enums(schema_path)
    except FileNotFoundError:
        result.add_error(f"Schema file not found: {schema_path}")
        return result
    except Exception as e:
        result.add_error(f"Error loading schema: {e}")
        return result

    # Load protocol model enums
    try:
        model_enums = extract_protocol_model_enums()
    except ImportError as e:
        result.add_error(f"Cannot import protocol models: {e}")
        return result

    # Load contracts enums (our source of truth)
    contracts_enums = {
        "command_types": COMMAND_TYPES,
        "command_status": COMMAND_STATUS_PROTOCOL,
        "approval_status": APPROVAL_STATUS,
        "run_state": RUN_STATE_PROTOCOL,
        "deployment_state": DEPLOYMENT_STATE_PROTOCOL,
        "telemetry_level": TELEMETRY_LEVEL_PROTOCOL,
        "signature_algorithms": SIGNATURE_ALGORITHMS,
        "message_types": MESSAGE_TYPES,
    }

    # Compare schema vs contracts
    for enum_name, contracts_values in contracts_enums.items():
        result.checked_enums += 1

        if enum_name in schema_enums:
            drift = compare_enum_sets(
                "schema",
                "contracts",
                schema_enums[enum_name],
                contracts_values,
            )
            drift.enum_name = enum_name
            result.add_drift(drift)

    # Compare protocol models vs contracts
    for enum_name, contracts_values in contracts_enums.items():
        result.checked_enums += 1

        if enum_name in model_enums:
            drift = compare_enum_sets(
                "contracts",
                "protocol_models",
                contracts_values,
                model_enums[enum_name],
            )
            drift.enum_name = f"{enum_name} (models)"
            result.add_drift(drift)

    # Special check: cloud commands.py vs contracts
    try:
        cloud_commands = extract_cloud_db_command_types()
        drift = compare_enum_sets(
            "contracts",
            "cloud_commands.py",
            COMMAND_TYPES,
            cloud_commands,
        )
        drift.enum_name = "command_types (cloud/commands.py)"
        result.add_drift(drift)
        result.checked_enums += 1
    except ImportError:
        pass  # Skip if not importable

    # Special check: boundary.py vs contracts
    try:
        boundary_commands = extract_boundary_command_types()
        drift = compare_enum_sets(
            "contracts",
            "boundary.py",
            COMMAND_TYPES,
            boundary_commands,
        )
        drift.enum_name = "command_types (boundary.py)"
        result.add_drift(drift)
        result.checked_enums += 1
    except ImportError:
        pass  # Skip if not importable

    return result


def main() -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CCEA Contract Validation"
    )
    parser.add_argument(
        "--schema",
        type=Path,
        help="Path to protocol schema file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    print("CCEA Contract Consistency Validation")
    print("=" * 40)

    result = validate_contract_consistency(schema_path=args.schema)

    print(f"\nChecked {result.checked_enums} enum sets")

    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  [X] {error}")

    if result.drifts:
        print("\nDrift detected:")
        for drift in result.drifts:
            print(f"\n  {drift}")

    if result.passed:
        print("\n[OK] Contract validation PASSED")
        return 0
    else:
        print("\n[FAIL] Contract validation FAILED - drift detected")
        return 1


if __name__ == "__main__":
    sys.exit(main())
