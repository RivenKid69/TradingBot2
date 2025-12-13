# -*- coding: utf-8 -*-
"""
CCEA Build Artifact Check.

Verifies that Cloud build artifacts do not contain:
- Live execution modules
- Order execution code
- Private trading clients
- Broker credentials handling

This is a post-build verification step that scans the actual
compiled/packaged artifacts before deployment.

Usage:
    python -m ccea.guardrails.build_artifact_check --artifact dist/cloud-1.0.0.whl
    python -m ccea.guardrails.build_artifact_check --directory build/cloud
"""

from __future__ import annotations

import ast
import io
import re
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, FrozenSet, List, Optional, Set, Tuple

# ============================================================================
# Prohibited Patterns in Cloud Build
# ============================================================================

# Module names that must NOT appear in Cloud build
PROHIBITED_MODULES: Final[FrozenSet[str]] = frozenset([
    # Order execution modules
    "order_execution",
    "options_execution",
    "futures_order_execution",
    # Live execution providers
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
    # Private trading
    "binance_spot_private",
    # Agent-only packages
    "packages/agent/vault",
    "packages/agent/execution",
    "packages/agent/policy",
])

# File patterns that indicate trading/execution code
PROHIBITED_FILE_PATTERNS: Final[FrozenSet[str]] = frozenset([
    "*order_execution*",
    "*options_execution*",
    "*_private*",
    "*live_loop*",
    "*broker_connector*",
    "*vault/local*",
    "*vault/credential*",
])

# Code patterns that must not appear in Cloud
PROHIBITED_CODE_PATTERNS: Final[List[Tuple[str, str]]] = [
    # Order submission patterns
    (r"\.submit_order\s*\(", "order submission call"),
    (r"\.place_order\s*\(", "order placement call"),
    (r"\.create_order\s*\(", "order creation call"),
    (r"\.execute_order\s*\(", "order execution call"),
    (r"\.send_order\s*\(", "order sending call"),
    # Credential patterns
    (r"api_key\s*=\s*['\"][^'\"]+['\"]", "hardcoded API key"),
    (r"api_secret\s*=\s*['\"][^'\"]+['\"]", "hardcoded API secret"),
    (r"\.decrypt\s*\(", "credential decryption"),
    # Live trading patterns
    (r"LiveExecutionEngine", "live execution engine"),
    (r"BrokerConnector", "broker connector"),
    (r"OrderRouter", "order router"),
]

# Imports that indicate trading capability
PROHIBITED_IMPORTS: Final[FrozenSet[str]] = frozenset([
    "packages.agent.vault",
    "packages.agent.execution",
    "packages.agent.policy.firewall",
    "adapters.alpaca.order_execution",
    "adapters.binance.futures_order_execution",
    "adapters.oanda.order_execution",
    "adapters.ib.order_execution",
    "execution_providers",
    "service_signal_runner",
])


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ArtifactViolation:
    """Represents a violation found in build artifact."""
    file_path: str
    violation_type: str
    description: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None

    def __str__(self) -> str:
        location = self.file_path
        if self.line_number:
            location += f":{self.line_number}"

        result = f"{location}: [{self.violation_type}] {self.description}"
        if self.code_snippet:
            result += f"\n    >>> {self.code_snippet[:100]}"
        return result


@dataclass
class ArtifactCheckResult:
    """Result of artifact check."""
    violations: List[ArtifactViolation] = field(default_factory=list)
    files_checked: int = 0
    passed: bool = True
    artifact_path: Optional[str] = None

    def add_violation(self, violation: ArtifactViolation) -> None:
        self.violations.append(violation)
        self.passed = False


# ============================================================================
# Artifact Scanning
# ============================================================================

def check_file_name(file_path: str) -> List[ArtifactViolation]:
    """
    Check if file name indicates prohibited module.

    Args:
        file_path: Path to check

    Returns:
        List of violations
    """
    violations = []
    file_name = Path(file_path).name

    for prohibited in PROHIBITED_MODULES:
        if prohibited in file_path:
            violations.append(ArtifactViolation(
                file_path=file_path,
                violation_type="prohibited_module",
                description=f"file path contains prohibited module '{prohibited}'",
            ))
            break

    return violations


def check_file_content(file_path: str, content: str) -> List[ArtifactViolation]:
    """
    Check file content for prohibited patterns.

    Args:
        file_path: Path to file
        content: File content

    Returns:
        List of violations
    """
    violations = []
    lines = content.split("\n")

    # Check code patterns
    for pattern, description in PROHIBITED_CODE_PATTERNS:
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line):
                violations.append(ArtifactViolation(
                    file_path=file_path,
                    violation_type="prohibited_code",
                    description=description,
                    line_number=i,
                    code_snippet=line.strip(),
                ))

    return violations


def check_imports(file_path: str, content: str) -> List[ArtifactViolation]:
    """
    Check imports in Python file.

    Args:
        file_path: Path to file
        content: File content

    Returns:
        List of violations
    """
    violations = []

    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError:
        return violations

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in PROHIBITED_IMPORTS:
                    violations.append(ArtifactViolation(
                        file_path=file_path,
                        violation_type="prohibited_import",
                        description=f"imports prohibited module '{alias.name}'",
                        line_number=node.lineno,
                    ))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                full_module = node.module
                if full_module in PROHIBITED_IMPORTS:
                    violations.append(ArtifactViolation(
                        file_path=file_path,
                        violation_type="prohibited_import",
                        description=f"imports from prohibited module '{full_module}'",
                        line_number=node.lineno,
                    ))
                # Check specific imports
                for alias in node.names:
                    full_import = f"{full_module}.{alias.name}"
                    if full_import in PROHIBITED_IMPORTS:
                        violations.append(ArtifactViolation(
                            file_path=file_path,
                            violation_type="prohibited_import",
                            description=f"imports '{alias.name}' from prohibited module",
                            line_number=node.lineno,
                        ))

    return violations


def scan_wheel_artifact(wheel_path: Path) -> ArtifactCheckResult:
    """
    Scan a wheel file for prohibited content.

    Args:
        wheel_path: Path to .whl file

    Returns:
        ArtifactCheckResult with violations
    """
    result = ArtifactCheckResult(artifact_path=str(wheel_path))

    if not wheel_path.exists():
        result.add_violation(ArtifactViolation(
            file_path=str(wheel_path),
            violation_type="not_found",
            description="Artifact file not found",
        ))
        return result

    try:
        with zipfile.ZipFile(wheel_path, "r") as zf:
            for name in zf.namelist():
                # Check file name
                violations = check_file_name(name)
                for v in violations:
                    result.add_violation(v)

                # Check Python files content
                if name.endswith(".py"):
                    result.files_checked += 1
                    try:
                        content = zf.read(name).decode("utf-8")

                        # Check content patterns
                        violations = check_file_content(name, content)
                        for v in violations:
                            result.add_violation(v)

                        # Check imports
                        violations = check_imports(name, content)
                        for v in violations:
                            result.add_violation(v)

                    except UnicodeDecodeError:
                        pass  # Skip binary files

    except zipfile.BadZipFile:
        result.add_violation(ArtifactViolation(
            file_path=str(wheel_path),
            violation_type="invalid_artifact",
            description="Invalid wheel file format",
        ))

    return result


def scan_directory(directory: Path) -> ArtifactCheckResult:
    """
    Scan a directory for prohibited content.

    Args:
        directory: Directory to scan

    Returns:
        ArtifactCheckResult with violations
    """
    result = ArtifactCheckResult(artifact_path=str(directory))

    if not directory.exists():
        result.add_violation(ArtifactViolation(
            file_path=str(directory),
            violation_type="not_found",
            description="Directory not found",
        ))
        return result

    # Find all Python files
    py_files = list(directory.rglob("*.py"))
    result.files_checked = len(py_files)

    for py_file in py_files:
        rel_path = py_file.relative_to(directory)

        # Check file name
        violations = check_file_name(str(rel_path))
        for v in violations:
            result.add_violation(v)

        # Check content
        try:
            content = py_file.read_text(encoding="utf-8")

            # Check content patterns
            violations = check_file_content(str(rel_path), content)
            for v in violations:
                result.add_violation(v)

            # Check imports
            violations = check_imports(str(rel_path), content)
            for v in violations:
                result.add_violation(v)

        except UnicodeDecodeError:
            pass

    return result


def verify_cloud_artifact(
    artifact_path: Path,
    strict: bool = True,
) -> ArtifactCheckResult:
    """
    Verify that a Cloud build artifact is clean.

    Args:
        artifact_path: Path to artifact (wheel or directory)
        strict: If True, any violation fails the check

    Returns:
        ArtifactCheckResult with verification status
    """
    if artifact_path.suffix == ".whl":
        result = scan_wheel_artifact(artifact_path)
    elif artifact_path.is_dir():
        result = scan_directory(artifact_path)
    else:
        result = ArtifactCheckResult(artifact_path=str(artifact_path))
        result.add_violation(ArtifactViolation(
            file_path=str(artifact_path),
            violation_type="unsupported",
            description="Unsupported artifact type (must be .whl or directory)",
        ))

    return result


# ============================================================================
# Pre-Build Verification
# ============================================================================

def verify_cloud_source(
    source_directory: Path,
    cloud_packages: Optional[List[str]] = None,
) -> ArtifactCheckResult:
    """
    Verify Cloud source before building.

    This is a pre-build check that ensures only allowed
    modules are included in the Cloud package.

    Args:
        source_directory: Project root directory
        cloud_packages: List of packages to include in Cloud

    Returns:
        ArtifactCheckResult with verification status
    """
    result = ArtifactCheckResult(artifact_path=str(source_directory))

    if cloud_packages is None:
        cloud_packages = ["packages/cloud", "packages/shared"]

    for pkg in cloud_packages:
        pkg_path = source_directory / pkg
        if pkg_path.exists():
            pkg_result = scan_directory(pkg_path)
            result.files_checked += pkg_result.files_checked
            for v in pkg_result.violations:
                result.add_violation(v)

    return result


# ============================================================================
# Manifest Verification
# ============================================================================

def verify_cloud_manifest(manifest: dict) -> ArtifactCheckResult:
    """
    Verify a Cloud build manifest.

    Checks that the manifest doesn't include prohibited modules
    in dependencies or entry points.

    Args:
        manifest: Build manifest dictionary

    Returns:
        ArtifactCheckResult with verification status
    """
    result = ArtifactCheckResult()

    # Check modules
    modules = manifest.get("modules", [])
    for module in modules:
        for prohibited in PROHIBITED_MODULES:
            if prohibited in module:
                result.add_violation(ArtifactViolation(
                    file_path="manifest",
                    violation_type="prohibited_module",
                    description=f"manifest includes prohibited module '{module}'",
                ))

    # Check entry points
    entry_points = manifest.get("entry_points", {})
    for group, points in entry_points.items():
        for name, target in points.items():
            for prohibited in PROHIBITED_IMPORTS:
                if prohibited in target:
                    result.add_violation(ArtifactViolation(
                        file_path="manifest",
                        violation_type="prohibited_entry_point",
                        description=f"entry point '{name}' references prohibited module",
                    ))

    # Check dependencies
    dependencies = manifest.get("dependencies", [])
    prohibited_deps = {"ib_insync", "alpaca_trade_api"}
    for dep in dependencies:
        dep_name = dep.split(">=")[0].split("==")[0].split("<")[0].strip()
        if dep_name in prohibited_deps:
            result.add_violation(ArtifactViolation(
                file_path="manifest",
                violation_type="prohibited_dependency",
                description=f"manifest includes prohibited dependency '{dep_name}'",
            ))

    return result


# ============================================================================
# CLI Interface
# ============================================================================

def main() -> int:
    """CLI entry point."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="CCEA Build Artifact Check"
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        help="Path to artifact to verify (wheel file)",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        help="Directory to scan",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to build manifest JSON",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Source directory for pre-build check",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        default=True,
        help="Exit with error code on violations",
    )

    args = parser.parse_args()

    results: List[ArtifactCheckResult] = []

    if args.artifact:
        print(f"Checking artifact: {args.artifact}")
        result = verify_cloud_artifact(args.artifact)
        results.append(result)

    if args.directory:
        print(f"Checking directory: {args.directory}")
        result = scan_directory(args.directory)
        results.append(result)

    if args.source:
        print(f"Checking source: {args.source}")
        result = verify_cloud_source(args.source)
        results.append(result)

    if args.manifest:
        print(f"Checking manifest: {args.manifest}")
        try:
            with open(args.manifest, "r") as f:
                manifest = json.load(f)
            result = verify_cloud_manifest(manifest)
            results.append(result)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Failed to read manifest: {e}")
            return 1

    if not results:
        print("No artifact specified. Use --artifact, --directory, --source, or --manifest")
        return 1

    # Output results
    total_violations = sum(len(r.violations) for r in results)
    total_files = sum(r.files_checked for r in results)

    if args.json:
        output = {
            "passed": all(r.passed for r in results),
            "total_files_checked": total_files,
            "total_violations": total_violations,
            "results": [
                {
                    "artifact": r.artifact_path,
                    "passed": r.passed,
                    "files_checked": r.files_checked,
                    "violations": [
                        {
                            "file": v.file_path,
                            "type": v.violation_type,
                            "description": v.description,
                            "line": v.line_number,
                        }
                        for v in r.violations
                    ],
                }
                for r in results
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\nFiles checked: {total_files}")
        print(f"Violations: {total_violations}")

        for r in results:
            if r.violations:
                print(f"\nViolations in {r.artifact_path}:")
                for v in r.violations:
                    print(f"  {v}")

    all_passed = all(r.passed for r in results)

    if all_passed:
        print("\nCloud artifact check PASSED")
        return 0
    else:
        print("\nCloud artifact check FAILED")
        if args.fail_on_violation:
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
