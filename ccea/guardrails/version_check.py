# -*- coding: utf-8 -*-
"""
CCEA Version Consistency Check.

Ensures that version constants in ccea/__init__.py match the schema metadata
in protocol_messages.schema.json.

This is a critical CI guardrail to prevent version drift.

Usage:
    python -m ccea.guardrails.version_check

Exit codes:
    0: All versions match
    1: Version mismatch detected
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final, List, Optional

# Default paths (relative to project root)
DEFAULT_SCHEMA_PATH: Final[str] = "docs/schemas/protocol_messages.schema.json"
DEFAULT_INIT_MODULE: Final[str] = "ccea"


@dataclass
class VersionCheckResult:
    """Result of version consistency check."""
    passed: bool = True
    schema_version: Optional[str] = None
    schema_min_version: Optional[str] = None
    schema_max_version: Optional[str] = None
    code_version: Optional[str] = None
    code_min_version: Optional[str] = None
    code_max_version: Optional[str] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def load_schema_versions(schema_path: Path) -> Dict[str, str]:
    """
    Load version metadata from protocol schema.

    Args:
        schema_path: Path to protocol_messages.schema.json

    Returns:
        Dict with version metadata
    """
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
    except FileNotFoundError:
        # When installed as a package (outside the repo), fall back to packaged schemas.
        from ccea.schemas import load_schema_json

        schema = load_schema_json(schema_path.name)

    return {
        "schema_version": schema.get("x-schema-version", ""),
        "min_supported": schema.get("x-min-supported-version", ""),
        "max_supported": schema.get("x-max-supported-version", ""),
    }


def load_code_versions() -> Dict[str, str]:
    """
    Load version constants from ccea module.

    Returns:
        Dict with version constants
    """
    import ccea

    return {
        "schema_version": ccea.SCHEMA_VERSION,
        "min_supported": ccea.MIN_SUPPORTED_SCHEMA_VERSION,
        "max_supported": ccea.MAX_SUPPORTED_SCHEMA_VERSION,
    }


def check_version_consistency(
    schema_path: Optional[Path] = None,
    project_root: Optional[Path] = None,
) -> VersionCheckResult:
    """
    Check that version constants match between code and schema.

    Args:
        schema_path: Path to schema file (uses default if None)
        project_root: Project root directory (uses cwd if None)

    Returns:
        VersionCheckResult with comparison results
    """
    result = VersionCheckResult()

    # Determine paths
    if project_root is None:
        # Try to find project root by looking for pyproject.toml
        cwd = Path.cwd()
        if (cwd / "pyproject.toml").exists():
            project_root = cwd
        elif (cwd.parent / "pyproject.toml").exists():
            project_root = cwd.parent
        else:
            project_root = cwd

    if schema_path is None:
        schema_path = project_root / DEFAULT_SCHEMA_PATH

    # Load schema versions
    try:
        schema_versions = load_schema_versions(schema_path)
        result.schema_version = schema_versions["schema_version"]
        result.schema_min_version = schema_versions["min_supported"]
        result.schema_max_version = schema_versions["max_supported"]
    except FileNotFoundError:
        result.passed = False
        result.errors.append(f"Schema file not found: {schema_path}")
        return result
    except json.JSONDecodeError as e:
        result.passed = False
        result.errors.append(f"Invalid JSON in schema: {e}")
        return result
    except Exception as e:
        result.passed = False
        result.errors.append(f"Error loading schema: {e}")
        return result

    # Load code versions
    try:
        code_versions = load_code_versions()
        result.code_version = code_versions["schema_version"]
        result.code_min_version = code_versions["min_supported"]
        result.code_max_version = code_versions["max_supported"]
    except ImportError as e:
        result.passed = False
        result.errors.append(f"Cannot import ccea module: {e}")
        return result
    except AttributeError as e:
        result.passed = False
        result.errors.append(f"Missing version constant in ccea: {e}")
        return result

    # Compare versions
    if result.schema_version != result.code_version:
        result.passed = False
        result.errors.append(
            f"SCHEMA_VERSION mismatch: "
            f"schema={result.schema_version}, code={result.code_version}"
        )

    if result.schema_min_version != result.code_min_version:
        result.passed = False
        result.errors.append(
            f"MIN_SUPPORTED_SCHEMA_VERSION mismatch: "
            f"schema={result.schema_min_version}, code={result.code_min_version}"
        )

    if result.schema_max_version != result.code_max_version:
        result.passed = False
        result.errors.append(
            f"MAX_SUPPORTED_SCHEMA_VERSION mismatch: "
            f"schema={result.schema_max_version}, code={result.code_max_version}"
        )

    return result


def check_schema_versioning_module() -> VersionCheckResult:
    """
    Check that schema_versioning.py constants match.

    Returns:
        VersionCheckResult
    """
    result = VersionCheckResult()

    try:
        from ccea.protocol.schema_versioning import (
            CURRENT_SCHEMA_VERSION,
            MIN_SUPPORTED_VERSION,
            MAX_SUPPORTED_VERSION,
        )
        import ccea

        if CURRENT_SCHEMA_VERSION != ccea.SCHEMA_VERSION:
            result.passed = False
            result.errors.append(
                f"schema_versioning.CURRENT_SCHEMA_VERSION ({CURRENT_SCHEMA_VERSION}) "
                f"!= ccea.SCHEMA_VERSION ({ccea.SCHEMA_VERSION})"
            )

        if MIN_SUPPORTED_VERSION != ccea.MIN_SUPPORTED_SCHEMA_VERSION:
            result.passed = False
            result.errors.append(
                f"schema_versioning.MIN_SUPPORTED_VERSION ({MIN_SUPPORTED_VERSION}) "
                f"!= ccea.MIN_SUPPORTED_SCHEMA_VERSION ({ccea.MIN_SUPPORTED_SCHEMA_VERSION})"
            )

        if MAX_SUPPORTED_VERSION != ccea.MAX_SUPPORTED_SCHEMA_VERSION:
            result.passed = False
            result.errors.append(
                f"schema_versioning.MAX_SUPPORTED_VERSION ({MAX_SUPPORTED_VERSION}) "
                f"!= ccea.MAX_SUPPORTED_SCHEMA_VERSION ({ccea.MAX_SUPPORTED_SCHEMA_VERSION})"
            )

    except ImportError as e:
        result.passed = False
        result.errors.append(f"Import error: {e}")

    return result


def main() -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CCEA Version Consistency Check"
    )
    parser.add_argument(
        "--schema",
        type=Path,
        help="Path to protocol schema file",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Project root directory",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    print("CCEA Version Consistency Check")
    print("=" * 40)

    # Check main version consistency
    result = check_version_consistency(
        schema_path=args.schema,
        project_root=args.project_root,
    )

    if args.verbose or not result.passed:
        print("\nSchema versions:")
        print(f"  x-schema-version: {result.schema_version}")
        print(f"  x-min-supported-version: {result.schema_min_version}")
        print(f"  x-max-supported-version: {result.schema_max_version}")

        print("\nCode versions (ccea/__init__.py):")
        print(f"  SCHEMA_VERSION: {result.code_version}")
        print(f"  MIN_SUPPORTED_SCHEMA_VERSION: {result.code_min_version}")
        print(f"  MAX_SUPPORTED_SCHEMA_VERSION: {result.code_max_version}")

    # Check schema_versioning module
    sv_result = check_schema_versioning_module()
    result.errors.extend(sv_result.errors)
    if not sv_result.passed:
        result.passed = False

    # Report results
    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  [X] {error}")

    if result.passed:
        print("\n[OK] Version consistency check PASSED")
        return 0
    else:
        print("\n[FAIL] Version consistency check FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
