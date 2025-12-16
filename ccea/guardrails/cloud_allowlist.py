# -*- coding: utf-8 -*-
"""
CCEA Cloud Dependency Allowlist.

Enforces that Cloud builds only use approved dependencies.
Includes transitive dependency checking to prevent indirect imports
of trading/execution libraries.

Usage:
    python -m ccea.guardrails.cloud_allowlist --check
    python -m ccea.guardrails.cloud_allowlist --validate-manifest cloud-build.json
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Final, FrozenSet, List, Optional, Set, Tuple

# ============================================================================
# Cloud Dependency Allowlist
# ============================================================================

# Python standard library modules (safe for all zones)
STDLIB_MODULES: Final[FrozenSet[str]] = frozenset([
    "abc", "asyncio", "base64", "collections", "contextlib",
    "copy", "csv", "dataclasses", "datetime", "decimal", "enum",
    "functools", "gzip", "hashlib", "io", "itertools", "json",
    "logging", "math", "os", "pathlib", "pickle", "random", "re",
    "secrets", "shutil", "sqlite3", "statistics", "struct", "sys",
    "tempfile", "threading", "time", "traceback", "typing", "unittest",
    "urllib", "uuid", "warnings", "weakref", "zipfile", "zlib",
    # Typing extensions
    "typing_extensions",
    # Common submodules
    "collections.abc", "concurrent.futures", "email.mime",
])

# Third-party packages allowed in Cloud
ALLOWED_THIRD_PARTY: Final[FrozenSet[str]] = frozenset([
    # Data processing
    "numpy", "pandas", "pyarrow", "scipy",
    # Configuration
    "pydantic", "pydantic_core", "yaml", "pyyaml",
    # ML/AI (training/research only)
    "torch", "stable_baselines3", "sb3_contrib", "gymnasium",
    "optuna", "joblib", "cloudpickle",
    # HTTP (for public market data only)
    "aiohttp", "httpx", "requests", "websockets",
    # Visualization (for research)
    "matplotlib", "seaborn", "plotly",
    # Utilities
    "sortedcontainers", "arch",
    # Web API (cloud API server)
    "fastapi", "uvicorn", "starlette",
])

# Packages PROHIBITED in Cloud (trading/execution)
PROHIBITED_PACKAGES: Final[FrozenSet[str]] = frozenset([
    # Exchange SDKs that include order execution
    "alpaca_trade_api",  # Old Alpaca SDK
    # Note: alpaca-py is allowed for market data, but trading endpoints are blocked
    "ib_insync",  # Interactive Brokers (has order submission)
    # We may need to reconsider ccxt if used only for market data
])

# Internal modules PROHIBITED in Cloud
PROHIBITED_INTERNAL: Final[FrozenSet[str]] = frozenset([
    # Order execution modules
    "adapters.alpaca.order_execution",
    "adapters.alpaca.options_execution",
    "adapters.binance.futures_order_execution",
    "adapters.binance_spot_private",
    "adapters.oanda.order_execution",
    "adapters.ib.order_execution",
    "adapters.ib.options_combo",
    # Execution providers (live)
    "execution_providers",
    "execution_providers_l3",
    "execution_providers_futures",
    "execution_providers_futures_l3",
    "execution_providers_cme",
    "execution_providers_cme_l3",
    # Live runtime
    "service_signal_runner",
    "script_live",
    "script_futures_live",
    # Agent zone packages
    "packages.agent",
    "packages.agent.vault",
    "packages.agent.execution",
    "packages.agent.policy",
    "packages.agent.reconciliation",
    "packages.agent.approval",
])

# Patterns that indicate trading/execution code
PROHIBITED_PATTERNS: Final[FrozenSet[str]] = frozenset([
    "order_execution",
    "options_execution",
    "_private.",  # Private trading endpoints (e.g. binance_spot_private.foo)
    ".private.",  # Private modules
    "spot_private",  # Binance spot private API
    "live_loop",
    "broker_connector",
])


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DependencyViolation:
    """Represents a dependency violation."""
    module: str
    reason: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    is_transitive: bool = False
    chain: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        location = ""
        if self.file_path:
            location = f"{self.file_path}"
            if self.line_number:
                location += f":{self.line_number}"
            location += ": "

        chain_info = ""
        if self.is_transitive and self.chain:
            chain_info = f" (via: {' -> '.join(self.chain)})"

        return f"{location}prohibited import '{self.module}'{chain_info}: {self.reason}"


@dataclass
class AllowlistCheckResult:
    """Result of allowlist check."""
    violations: List[DependencyViolation] = field(default_factory=list)
    files_checked: int = 0
    modules_checked: int = 0
    passed: bool = True

    def add_violation(self, violation: DependencyViolation) -> None:
        self.violations.append(violation)
        self.passed = False


# ============================================================================
# Dependency Analysis
# ============================================================================

def extract_imports_from_file(file_path: Path) -> List[Tuple[str, int]]:
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
                for alias in node.names:
                    if alias.name != "*":
                        full_name = f"{node.module}.{alias.name}"
                        imports.append((full_name, node.lineno))

    return imports


def get_base_package(module: str) -> str:
    """Get the base package name from a module path."""
    return module.split(".")[0]


def is_stdlib(module: str) -> bool:
    """Check if module is from Python standard library."""
    base = get_base_package(module)
    return base in STDLIB_MODULES or module in STDLIB_MODULES


def is_allowed_third_party(module: str) -> bool:
    """Check if module is an allowed third-party package."""
    base = get_base_package(module)
    return base in ALLOWED_THIRD_PARTY


def is_prohibited_package(module: str) -> bool:
    """Check if module is a prohibited package."""
    base = get_base_package(module)
    return base in PROHIBITED_PACKAGES or module in PROHIBITED_PACKAGES


def is_prohibited_internal(module: str) -> bool:
    """Check if module is a prohibited internal module."""
    # Check exact match
    if module in PROHIBITED_INTERNAL:
        return True

    # Check patterns
    for pattern in PROHIBITED_PATTERNS:
        if pattern in module:
            return True

    # Check if starts with prohibited prefix
    for prohibited in PROHIBITED_INTERNAL:
        if module.startswith(f"{prohibited}."):
            return True

    return False


def is_cloud_allowed(module: str) -> bool:
    """
    Check if module is allowed in Cloud zone.

    Args:
        module: Module name to check

    Returns:
        True if module is allowed in Cloud
    """
    # Standard library is always allowed
    if is_stdlib(module):
        return True

    # Check prohibited packages first
    if is_prohibited_package(module):
        return False

    # Check prohibited internal modules
    if is_prohibited_internal(module):
        return False

    # Allowed third-party
    if is_allowed_third_party(module):
        return True

    # Internal project modules (need further validation)
    # Allow shared zone modules
    if module.startswith("packages.shared"):
        return True
    if module.startswith("packages.cloud"):
        return True

    # Allow core modules (shared infrastructure)
    if any(module.startswith(prefix) for prefix in [
        "core_", "impl_", "adapters.base", "adapters.models",
        "adapters.registry", "adapters.config", "adapters.websocket_base",
    ]):
        return True

    # Allow CCEA shared infrastructure modules (crypto, models, guardrails)
    # These are needed by Cloud for command signing and validation
    if any(module.startswith(prefix) for prefix in [
        "ccea.crypto",  # Cryptographic utilities for signing/verification
        "ccea.models",  # Shared data models and protocols
        "ccea.guardrails",  # CI/CD guardrails
    ]):
        return True

    # Allow data-only adapter modules
    if any(pattern in module for pattern in [
        "market_data", "fees", "trading_hours", "exchange_info",
    ]):
        # But not if it also has execution patterns
        if not any(p in module for p in PROHIBITED_PATTERNS):
            return True

    # Default: allow (unknown modules are checked separately)
    return True


# ============================================================================
# Transitive Dependency Checking
# ============================================================================

class TransitiveDependencyChecker:
    """
    Checks for transitive imports of prohibited modules.

    Builds an import graph and checks if any path leads
    to prohibited modules.
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.import_graph: Dict[str, Set[str]] = {}
        self._built = False

    def build_graph(self) -> None:
        """Build the import graph for the project."""
        if self._built:
            return

        py_files = list(self.project_root.rglob("*.py"))

        for py_file in py_files:
            # Skip test files
            if "test" in py_file.name.lower() or "/tests/" in str(py_file):
                continue

            # Get module name from file path
            rel_path = py_file.relative_to(self.project_root)
            module_parts = list(rel_path.parts[:-1]) + [rel_path.stem]
            if module_parts[-1] == "__init__":
                module_parts = module_parts[:-1]
            module_name = ".".join(module_parts) if module_parts else ""

            if not module_name:
                continue

            imports = extract_imports_from_file(py_file)
            self.import_graph[module_name] = {imp[0] for imp in imports}

        self._built = True

    def find_transitive_violations(
        self,
        start_module: str,
        max_depth: int = 5,
    ) -> List[DependencyViolation]:
        """
        Find transitive imports of prohibited modules.

        Args:
            start_module: Module to start checking from
            max_depth: Maximum depth to search

        Returns:
            List of violations with import chains
        """
        self.build_graph()

        violations = []
        visited = set()

        def dfs(module: str, chain: List[str], depth: int) -> None:
            if depth > max_depth:
                return
            if module in visited:
                return
            visited.add(module)

            # Check if this module is prohibited
            if is_prohibited_internal(module) and len(chain) > 1:
                violations.append(DependencyViolation(
                    module=module,
                    reason="transitive import of trading/execution module",
                    is_transitive=True,
                    chain=list(chain),
                ))
                return

            # Get imports of this module
            imports = self.import_graph.get(module, set())
            for imp in imports:
                dfs(imp, chain + [imp], depth + 1)

        dfs(start_module, [start_module], 0)
        return violations

    def check_cloud_zone(self) -> AllowlistCheckResult:
        """
        Check all Cloud zone modules for transitive violations.

        Returns:
            AllowlistCheckResult with any violations
        """
        self.build_graph()
        result = AllowlistCheckResult()

        # Find all Cloud zone modules
        cloud_modules = [
            m for m in self.import_graph.keys()
            if m.startswith("packages.cloud") or m.startswith("packages/cloud")
        ]

        result.modules_checked = len(cloud_modules)

        for module in cloud_modules:
            violations = self.find_transitive_violations(module)
            for v in violations:
                result.add_violation(v)

        return result


# ============================================================================
# Cloud Build Validation
# ============================================================================

def validate_cloud_build(
    cloud_directory: Path,
    check_transitive: bool = True,
) -> AllowlistCheckResult:
    """
    Validate that Cloud build only uses allowed dependencies.

    Args:
        cloud_directory: Directory containing Cloud code
        check_transitive: Also check transitive dependencies

    Returns:
        AllowlistCheckResult with any violations
    """
    result = AllowlistCheckResult()

    # Find all Python files in cloud directory
    py_files = list(cloud_directory.rglob("*.py"))
    result.files_checked = len(py_files)

    modules_analyzed = 0
    for py_file in py_files:
        # Skip test files for violation checking (but still count as checked)
        is_test_file = "test" in py_file.name.lower() or "/tests/" in str(py_file)
        if is_test_file:
            continue

        imports = extract_imports_from_file(py_file)
        modules_analyzed += 1

        for module, line in imports:
            # Check prohibited packages
            if is_prohibited_package(module):
                result.add_violation(DependencyViolation(
                    module=module,
                    reason="prohibited package in Cloud zone",
                    file_path=str(py_file),
                    line_number=line,
                ))

            # Check prohibited internal modules
            if is_prohibited_internal(module):
                result.add_violation(DependencyViolation(
                    module=module,
                    reason="trading/execution module prohibited in Cloud",
                    file_path=str(py_file),
                    line_number=line,
                ))

    # Set modules_checked to count of non-test Python files analyzed
    result.modules_checked = modules_analyzed

    # Check transitive dependencies
    if check_transitive:
        project_root = cloud_directory.parent.parent  # Assuming packages/cloud
        if project_root.exists():
            checker = TransitiveDependencyChecker(project_root)
            trans_result = checker.check_cloud_zone()
            for v in trans_result.violations:
                result.add_violation(v)
            # Add transitive modules checked
            result.modules_checked += trans_result.modules_checked

    return result


def validate_cloud_manifest(manifest_path: Path) -> AllowlistCheckResult:
    """
    Validate a Cloud build manifest for allowed dependencies.

    Args:
        manifest_path: Path to build manifest JSON

    Returns:
        AllowlistCheckResult with any violations
    """
    import json

    result = AllowlistCheckResult()

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        result.add_violation(DependencyViolation(
            module="manifest",
            reason=f"Failed to read manifest: {e}",
        ))
        return result

    # Check dependencies in manifest
    dependencies = manifest.get("dependencies", [])
    for dep in dependencies:
        module_name = dep if isinstance(dep, str) else dep.get("name", "")

        if is_prohibited_package(module_name):
            result.add_violation(DependencyViolation(
                module=module_name,
                reason="prohibited package in Cloud manifest",
            ))

        if is_prohibited_internal(module_name):
            result.add_violation(DependencyViolation(
                module=module_name,
                reason="trading/execution module in Cloud manifest",
            ))

    return result


# ============================================================================
# CLI Interface
# ============================================================================

def main() -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CCEA Cloud Dependency Allowlist Check"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check Cloud directory for violations",
    )
    parser.add_argument(
        "--validate-manifest",
        type=Path,
        help="Validate a build manifest file",
    )
    parser.add_argument(
        "--cloud-dir",
        type=Path,
        default=Path.cwd() / "packages" / "cloud",
        help="Cloud directory to check",
    )
    parser.add_argument(
        "--no-transitive",
        action="store_true",
        help="Skip transitive dependency check",
    )
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        default=True,
        help="Exit with error code on violations",
    )

    args = parser.parse_args()

    if args.validate_manifest:
        print(f"Validating manifest: {args.validate_manifest}")
        result = validate_cloud_manifest(args.validate_manifest)
    elif args.check or True:  # Default to check
        print(f"Checking Cloud directory: {args.cloud_dir}")
        result = validate_cloud_build(
            args.cloud_dir,
            check_transitive=not args.no_transitive,
        )

    print(f"Files checked: {result.files_checked}")
    print(f"Modules checked: {result.modules_checked}")
    print(f"Violations: {len(result.violations)}")

    for v in result.violations:
        print(f"  {v}")

    # Fail-closed: if files were found but no modules were analyzed, something is wrong
    if result.files_checked > 0 and result.modules_checked == 0:
        print("\n[FAIL] Cloud allowlist check FAILED")
        print("  ERROR: files_checked > 0 but modules_checked == 0 - guardrail ineffective")
        return 1

    if result.passed:
        print("\n[PASS] Cloud allowlist check PASSED")
        return 0
    else:
        print("\n[FAIL] Cloud allowlist check FAILED")
        if args.fail_on_violation:
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
