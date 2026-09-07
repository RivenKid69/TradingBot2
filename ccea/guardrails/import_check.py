# -*- coding: utf-8 -*-
"""
CCEA Import Boundary Check.

Enforces architectural boundaries between Cloud/Agent/Shared zones.
Cloud build must not contain order execution or trading libraries.

Usage:
    python -m ccea.guardrails.import_check --target cloud
    python -m ccea.guardrails.import_check --target agent
"""

from __future__ import annotations

import ast
import fnmatch
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Final, List, Optional, Set

# ============================================================================
# Constants
# ============================================================================


class ZoneType(str, Enum):
    """Module zone classification."""

    SHARED = "shared"
    AGENT = "agent"
    CLOUD = "cloud"
    UNKNOWN = "unknown"


# Modules prohibited in Cloud build
PROHIBITED_IN_CLOUD: Final[List[str]] = [
    "adapters.*.order_execution",
    "adapters.*.options_execution",
    "adapters.alpaca.order_execution",
    "adapters.alpaca.options_execution",
    "adapters.oanda.order_execution",
    "adapters.ib.order_execution",
    "adapters.ib.options_combo",
    # Live execution modules
    "execution_providers",
    "execution_providers_l3",
    "execution_providers_futures_l3",
    "execution_providers_cme_l3",
    # Live runtime
    "service_signal_runner",
]

# Third-party packages prohibited in Cloud
PROHIBITED_PACKAGES: Final[List[str]] = [
    # When used for order submission
    # Note: ccxt can be used for market data, so this is context-dependent
]

# Modules that belong to AGENT zone only
AGENT_ONLY_MODULES: Final[List[str]] = [
    "adapters.*.order_execution",
    "adapters.*.options_execution",
    "execution_providers*",
    "service_signal_runner",
    "risk_guard",
    "impl_circuit_breaker",
]

# Modules that belong to SHARED zone
SHARED_MODULES: Final[List[str]] = [
    "core_*",
    "impl_*",
    "adapters.*.market_data",
    "adapters.*.fees",
    "adapters.*.trading_hours",
    "adapters.*.exchange_info",
    "features_pipeline",
    "transformers",
    "data_loader_*",
    "execution_sim",
    "distributional_ppo",
    "train_model_multi_patch",
    "lob.*",
]

# Modules that belong to CLOUD zone
CLOUD_MODULES: Final[List[str]] = [
    "app",
    "service_backtest",
    "service_train",
    "service_eval",
]


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class ImportViolation:
    """Represents an import boundary violation."""

    file_path: str
    line_number: int
    module_imported: str
    reason: str
    zone_from: ZoneType
    zone_to: ZoneType

    def __str__(self) -> str:
        return (
            f"{self.file_path}:{self.line_number}: "
            f"prohibited import '{self.module_imported}' "
            f"({self.zone_from.value} -> {self.zone_to.value}): {self.reason}"
        )


@dataclass
class CheckResult:
    """Result of import boundary check."""

    violations: List[ImportViolation] = field(default_factory=list)
    files_checked: int = 0
    passed: bool = True

    def add_violation(self, violation: ImportViolation) -> None:
        self.violations.append(violation)
        self.passed = False


# ============================================================================
# Zone Classification
# ============================================================================


def matches_pattern(module: str, patterns: List[str]) -> bool:
    """Check if module matches any of the patterns."""
    for pattern in patterns:
        if fnmatch.fnmatch(module, pattern):
            return True
        # Also check with dots replaced
        if fnmatch.fnmatch(module.replace(".", "/"), pattern.replace(".", "/")):
            return True
    return False


def get_zone_for_module(module_name: str) -> ZoneType:
    """
    Determine which zone a module belongs to.

    Args:
        module_name: Full module name (e.g., 'adapters.alpaca.order_execution')

    Returns:
        ZoneType indicating the module's zone
    """
    if matches_pattern(module_name, AGENT_ONLY_MODULES):
        return ZoneType.AGENT
    if matches_pattern(module_name, CLOUD_MODULES):
        return ZoneType.CLOUD
    if matches_pattern(module_name, SHARED_MODULES):
        return ZoneType.SHARED
    return ZoneType.UNKNOWN


# ============================================================================
# Import Extraction
# ============================================================================


def extract_imports_from_file(file_path: Path) -> List[tuple[str, int]]:
    """
    Extract all imports from a Python file.

    Args:
        file_path: Path to Python file

    Returns:
        List of (module_name, line_number) tuples
    """
    imports = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError):
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append((node.module, node.lineno))
                # Also add full imports like 'from x import y'
                for alias in node.names:
                    if alias.name != "*":
                        imports.append((f"{node.module}.{alias.name}", node.lineno))

    return imports


# ============================================================================
# Cloud Import Check
# ============================================================================


def check_cloud_imports(
    directory: Path,
    prohibited_modules: Optional[List[str]] = None,
    prohibited_packages: Optional[List[str]] = None,
) -> CheckResult:
    """
    Check that Cloud build doesn't contain prohibited imports.

    Args:
        directory: Directory to scan
        prohibited_modules: List of prohibited module patterns
        prohibited_packages: List of prohibited package names

    Returns:
        CheckResult with any violations found
    """
    if prohibited_modules is None:
        prohibited_modules = PROHIBITED_IN_CLOUD
    if prohibited_packages is None:
        prohibited_packages = PROHIBITED_PACKAGES

    result = CheckResult()

    # Find all Python files
    py_files = list(directory.rglob("*.py"))
    result.files_checked = len(py_files)

    for py_file in py_files:
        # Skip test files
        if "test" in py_file.name.lower() or "/tests/" in str(py_file):
            continue
        # Skip agent-only modules
        if any(p in str(py_file) for p in ["packages/agent", "ccea/agent"]):
            continue

        imports = extract_imports_from_file(py_file)

        for module, line in imports:
            # Check against prohibited modules
            if matches_pattern(module, prohibited_modules):
                result.add_violation(
                    ImportViolation(
                        file_path=str(py_file),
                        line_number=line,
                        module_imported=module,
                        reason="trading/execution module prohibited in Cloud",
                        zone_from=ZoneType.CLOUD,
                        zone_to=ZoneType.AGENT,
                    )
                )

            # Check against prohibited packages
            for pkg in prohibited_packages:
                if module == pkg or module.startswith(f"{pkg}."):
                    result.add_violation(
                        ImportViolation(
                            file_path=str(py_file),
                            line_number=line,
                            module_imported=module,
                            reason=f"package '{pkg}' prohibited in Cloud",
                            zone_from=ZoneType.CLOUD,
                            zone_to=ZoneType.AGENT,
                        )
                    )

    return result


def check_agent_imports(directory: Path) -> CheckResult:
    """
    Check Agent imports (ensure no cloud internals imported).

    Args:
        directory: Directory to scan

    Returns:
        CheckResult with any violations found
    """
    result = CheckResult()

    # Agent should not import cloud internals
    prohibited_in_agent = ["packages.cloud.internal"]

    py_files = list(directory.rglob("*.py"))
    result.files_checked = len(py_files)

    for py_file in py_files:
        if "test" in py_file.name.lower() or "/tests/" in str(py_file):
            continue

        imports = extract_imports_from_file(py_file)

        for module, line in imports:
            if matches_pattern(module, prohibited_in_agent):
                result.add_violation(
                    ImportViolation(
                        file_path=str(py_file),
                        line_number=line,
                        module_imported=module,
                        reason="Cloud internal module prohibited in Agent",
                        zone_from=ZoneType.AGENT,
                        zone_to=ZoneType.CLOUD,
                    )
                )

    return result


# ============================================================================
# CLI Interface
# ============================================================================


def main() -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="CCEA Import Boundary Check")
    parser.add_argument(
        "--target",
        choices=["cloud", "agent", "all"],
        default="all",
        help="Target zone to check",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path.cwd(),
        help="Directory to scan",
    )
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        default=True,
        help="Exit with error code on violations",
    )

    args = parser.parse_args()

    results: List[CheckResult] = []

    if args.target in ("cloud", "all"):
        print(f"Checking Cloud imports in {args.directory}...")
        result = check_cloud_imports(args.directory)
        results.append(result)
        print(f"  Files checked: {result.files_checked}")
        print(f"  Violations: {len(result.violations)}")
        for v in result.violations:
            print(f"    {v}")

    if args.target in ("agent", "all"):
        print(f"Checking Agent imports in {args.directory}...")
        result = check_agent_imports(args.directory)
        results.append(result)
        print(f"  Files checked: {result.files_checked}")
        print(f"  Violations: {len(result.violations)}")
        for v in result.violations:
            print(f"    {v}")

    all_passed = all(r.passed for r in results)

    if all_passed:
        print("\nAll import boundary checks PASSED")
        return 0
    else:
        print("\nImport boundary checks FAILED")
        if args.fail_on_violation:
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
