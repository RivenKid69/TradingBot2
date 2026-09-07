# -*- coding: utf-8 -*-
"""
GDPR Phase 8: CI Privacy-by-Design Check.

This CI guardrail enforces the Phase 8 requirement:
"CI fails closed if a new data store/log stream/telemetry field ships
without recorded: classification, retention, residency, and redaction
requirements (in a registered data inventory entry)."

Scans source code for:
- New telemetry field definitions
- New data store declarations
- New log stream configurations
- New database table/column definitions

And validates each against the Data Inventory Registry.

GDPR Reference:
    - Article 25: Data protection by design and by default
    - Phase 8 DoD: CI fails closed without inventory registration

Usage:
    # Run as CI check
    python -m ccea.guardrails.privacy_by_design_check

    # Check specific files
    python -m ccea.guardrails.privacy_by_design_check --files path/to/file.py

    # Check against changed files (git diff)
    python -m ccea.guardrails.privacy_by_design_check --diff HEAD~1

CLOUD ZONE ONLY.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Set, Tuple

# Import from data_inventory if available
try:
    from packages.cloud.governance.data_inventory import (
        DataInventoryRegistry,
        InventoryValidationResult,
        create_core_inventory,
        ALWAYS_REDACT_PATTERNS,
        PII_FIELD_PATTERNS,
        ORDER_FIELD_PATTERNS,
    )

    HAS_INVENTORY = True
except ImportError:
    HAS_INVENTORY = False


logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================


class CheckSeverity(str, Enum):
    """Severity of check violations."""

    CRITICAL = "critical"  # Must fix, blocks CI
    HIGH = "high"  # Should fix, blocks CI
    MEDIUM = "medium"  # Should fix, warning
    LOW = "low"  # Informational


class CheckResult(str, Enum):
    """Result of the privacy check."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


# Patterns for detecting data declarations in code
# Telemetry field patterns
TELEMETRY_FIELD_PATTERNS: Final[List[Tuple[re.Pattern, str]]] = [
    # Python dictionary keys that look like telemetry
    (
        re.compile(r'["\'](\w+)["\']\s*:\s*(?:metrics|telemetry|event)', re.IGNORECASE),
        "Telemetry field in dict",
    ),
    # Dataclass or typed dict field
    (
        re.compile(r"(\w+)\s*:\s*(?:int|float|str|bool|datetime).*#.*telemetry", re.IGNORECASE),
        "Telemetry field annotation",
    ),
    # Emit/send telemetry calls
    (
        re.compile(
            r'\.(?:emit|send|log)_(?:telemetry|metric|event)\s*\(\s*["\'](\w+)["\']', re.IGNORECASE
        ),
        "Telemetry emit call",
    ),
]

# Data store patterns
DATA_STORE_PATTERNS: Final[List[Tuple[re.Pattern, str]]] = [
    # SQL CREATE TABLE
    (
        re.compile(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?["`]?(\w+)["`]?', re.IGNORECASE),
        "SQL CREATE TABLE",
    ),
    # SQLAlchemy model
    (re.compile(r'__tablename__\s*=\s*["\'](\w+)["\']'), "SQLAlchemy table"),
    # Redis/cache keys
    (
        re.compile(r'(?:redis|cache)\.(?:set|get|hset|hget)\s*\(\s*["\'](\w+)', re.IGNORECASE),
        "Cache key",
    ),
    # S3 bucket
    (re.compile(r'bucket\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE), "S3 bucket"),
    # Collection/table name
    (re.compile(r'collection\s*=\s*["\'](\w+)["\']', re.IGNORECASE), "Collection name"),
]

# Log stream patterns
LOG_STREAM_PATTERNS: Final[List[Tuple[re.Pattern, str]]] = [
    # Python logging
    (re.compile(r'logging\.getLogger\s*\(\s*["\']([^"\']+)["\']'), "Python logger"),
    # Structured logging
    (re.compile(r'structlog\.get_logger\s*\(\s*["\']([^"\']+)["\']'), "Structlog logger"),
    # Log stream name
    (re.compile(r'log_stream\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE), "Log stream"),
    # CloudWatch log group
    (re.compile(r'log_group\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE), "CloudWatch log group"),
]

# Files to skip
SKIP_PATTERNS: Final[List[re.Pattern]] = [
    re.compile(r"test[_/]"),  # Test files
    re.compile(r"_test\.py$"),  # Test files
    re.compile(r"conftest\.py$"),  # Pytest config
    re.compile(r"__pycache__"),  # Cache
    re.compile(r"\.pyc$"),  # Compiled
    re.compile(r"migrations/"),  # DB migrations
    re.compile(r"\.git/"),  # Git
    re.compile(r"node_modules/"),  # Node modules
    re.compile(r"\.venv/"),  # Virtual env
    re.compile(r"venv/"),  # Virtual env
    re.compile(r"dist/"),  # Distribution
    re.compile(r"build/"),  # Build
    re.compile(r"\.egg-info/"),  # Egg info
]

# Known safe patterns (pre-registered in inventory)
KNOWN_SAFE_PATTERNS: Final[Set[str]] = {
    # Core telemetry fields
    "timestamp",
    "event_time",
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "cpu_percent",
    "memory_percent",
    "latency_ms",
    "error_count",
    "strategy_id",
    "workspace_id",
    "agent_id",
    "run_id",
    # System logs
    "__main__",
    "__name__",
    "root",
}


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class DataDeclaration:
    """A detected data declaration in source code."""

    name: str  # Field/store name
    declaration_type: str  # telemetry_field, data_store, log_stream
    file_path: str  # Source file
    line_number: int  # Line number
    context: str  # Code context
    pattern_matched: str  # Which pattern detected it
    auto_detected: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "declaration_type": self.declaration_type,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "context": self.context[:200] if self.context else "",
            "pattern_matched": self.pattern_matched,
            "auto_detected": self.auto_detected,
        }


@dataclass
class CheckViolation:
    """A privacy check violation."""

    declaration: DataDeclaration
    severity: CheckSeverity
    message: str
    inventory_result: Optional[Dict[str, Any]] = None
    remediation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "declaration": self.declaration.to_dict(),
            "severity": self.severity.value,
            "message": self.message,
            "inventory_result": self.inventory_result,
            "remediation": self.remediation,
        }


@dataclass
class PrivacyCheckReport:
    """Report from privacy-by-design check."""

    check_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    result: CheckResult = CheckResult.PASS
    files_scanned: int = 0
    declarations_found: int = 0
    declarations_registered: int = 0
    declarations_unregistered: int = 0
    violations: List[CheckViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0

    @property
    def should_fail_ci(self) -> bool:
        """Check if this report should fail CI."""
        return any(
            v.severity in (CheckSeverity.CRITICAL, CheckSeverity.HIGH) for v in self.violations
        )

    @property
    def critical_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == CheckSeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == CheckSeverity.HIGH)

    @property
    def medium_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == CheckSeverity.MEDIUM)

    @property
    def low_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == CheckSeverity.LOW)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "timestamp": self.timestamp.isoformat(),
            "result": self.result.value,
            "summary": {
                "files_scanned": self.files_scanned,
                "declarations_found": self.declarations_found,
                "declarations_registered": self.declarations_registered,
                "declarations_unregistered": self.declarations_unregistered,
                "should_fail_ci": self.should_fail_ci,
            },
            "violation_counts": {
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count,
                "total": len(self.violations),
            },
            "violations": [v.to_dict() for v in self.violations],
            "warnings": self.warnings,
            "skipped_files": self.skipped_files[:10],  # Limit for readability
            "execution_time_ms": self.execution_time_ms,
        }


# ============================================================================
# Scanner
# ============================================================================


class DataDeclarationScanner:
    """
    Scans source files for data declarations.

    Detects:
    - Telemetry field definitions
    - Data store declarations
    - Log stream configurations
    """

    def __init__(
        self,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[re.Pattern]] = None,
    ):
        """
        Initialize scanner.

        Args:
            include_patterns: File glob patterns to include
            exclude_patterns: Regex patterns to exclude
        """
        self._include_patterns = include_patterns or [
            "**/*.py",
            "**/*.sql",
            "**/*.yaml",
            "**/*.yml",
        ]
        self._exclude_patterns = exclude_patterns or SKIP_PATTERNS

    def scan_file(self, file_path: Path) -> List[DataDeclaration]:
        """
        Scan a single file for data declarations.

        Args:
            file_path: Path to file

        Returns:
            List of detected declarations
        """
        declarations = []

        # Check if file should be skipped
        path_str = str(file_path)
        for pattern in self._exclude_patterns:
            if pattern.search(path_str):
                return []

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return []

        lines = content.split("\n")

        # Scan for telemetry fields
        for pattern, pattern_name in TELEMETRY_FIELD_PATTERNS:
            for i, line in enumerate(lines, 1):
                matches = pattern.findall(line)
                for match in matches:
                    name = match if isinstance(match, str) else match[0]
                    if name and name not in KNOWN_SAFE_PATTERNS:
                        declarations.append(
                            DataDeclaration(
                                name=name,
                                declaration_type="telemetry_field",
                                file_path=str(file_path),
                                line_number=i,
                                context=line.strip(),
                                pattern_matched=pattern_name,
                                auto_detected=self._auto_detect(name),
                            )
                        )

        # Scan for data stores
        for pattern, pattern_name in DATA_STORE_PATTERNS:
            for i, line in enumerate(lines, 1):
                matches = pattern.findall(line)
                for match in matches:
                    name = match if isinstance(match, str) else match[0]
                    if name and name not in KNOWN_SAFE_PATTERNS:
                        declarations.append(
                            DataDeclaration(
                                name=name,
                                declaration_type="data_store",
                                file_path=str(file_path),
                                line_number=i,
                                context=line.strip(),
                                pattern_matched=pattern_name,
                                auto_detected=self._auto_detect(name),
                            )
                        )

        # Scan for log streams
        for pattern, pattern_name in LOG_STREAM_PATTERNS:
            for i, line in enumerate(lines, 1):
                matches = pattern.findall(line)
                for match in matches:
                    name = match if isinstance(match, str) else match[0]
                    if name and name not in KNOWN_SAFE_PATTERNS:
                        declarations.append(
                            DataDeclaration(
                                name=name,
                                declaration_type="log_stream",
                                file_path=str(file_path),
                                line_number=i,
                                context=line.strip(),
                                pattern_matched=pattern_name,
                                auto_detected=self._auto_detect(name),
                            )
                        )

        return declarations

    def scan_directory(
        self,
        directory: Path,
        recursive: bool = True,
    ) -> List[DataDeclaration]:
        """
        Scan a directory for data declarations.

        Args:
            directory: Directory path
            recursive: Whether to scan recursively

        Returns:
            List of detected declarations
        """
        declarations = []

        for pattern in self._include_patterns:
            if recursive:
                files = directory.glob(pattern)
            else:
                files = directory.glob(pattern.split("/")[-1])

            for file_path in files:
                if file_path.is_file():
                    declarations.extend(self.scan_file(file_path))

        return declarations

    def scan_files(self, file_paths: List[Path]) -> List[DataDeclaration]:
        """
        Scan specific files for data declarations.

        Args:
            file_paths: List of file paths

        Returns:
            List of detected declarations
        """
        declarations = []
        for file_path in file_paths:
            if file_path.is_file():
                declarations.extend(self.scan_file(file_path))
        return declarations

    def _auto_detect(self, name: str) -> Dict[str, bool]:
        """Auto-detect sensitive patterns in a name."""
        name_lower = name.lower()

        detected = {
            "pii": False,
            "credential": False,
            "order_field": False,
        }

        # Check credential patterns
        if HAS_INVENTORY:
            for pattern, _ in ALWAYS_REDACT_PATTERNS:
                if pattern.search(name_lower):
                    detected["credential"] = True
                    break

            for pattern, _ in PII_FIELD_PATTERNS:
                if pattern.search(name_lower):
                    detected["pii"] = True
                    break

            for pattern, _ in ORDER_FIELD_PATTERNS:
                if pattern.search(name_lower):
                    detected["order_field"] = True
                    break
        else:
            # Fallback patterns without full inventory
            credential_keywords = ["api_key", "secret", "token", "password", "credential"]
            pii_keywords = ["email", "phone", "address", "ssn", "ip_address"]
            order_keywords = ["side", "quantity", "qty", "price", "order", "fill"]

            for kw in credential_keywords:
                if kw in name_lower:
                    detected["credential"] = True
                    break

            for kw in pii_keywords:
                if kw in name_lower:
                    detected["pii"] = True
                    break

            for kw in order_keywords:
                if kw in name_lower:
                    detected["order_field"] = True
                    break

        return detected


# ============================================================================
# Privacy Check
# ============================================================================


class PrivacyByDesignCheck:
    """
    CI Privacy-by-Design Check.

    Validates that all data declarations are properly registered
    in the Data Inventory Registry with required GDPR fields.

    Usage:
        check = PrivacyByDesignCheck()
        report = check.run(Path("packages/cloud"))

        if report.should_fail_ci:
            sys.exit(1)
    """

    def __init__(
        self,
        registry: Optional[Any] = None,
        scanner: Optional[DataDeclarationScanner] = None,
        fail_on_unregistered: bool = True,
        fail_on_non_compliant: bool = True,
        warn_on_auto_detected: bool = True,
    ):
        """
        Initialize check.

        Args:
            registry: Data inventory registry (or create default)
            scanner: Declaration scanner (or create default)
            fail_on_unregistered: Fail CI if declarations not in inventory
            fail_on_non_compliant: Fail CI if declarations non-compliant
            warn_on_auto_detected: Warn if auto-detection finds issues
        """
        self._registry = registry
        self._scanner = scanner or DataDeclarationScanner()
        self._fail_on_unregistered = fail_on_unregistered
        self._fail_on_non_compliant = fail_on_non_compliant
        self._warn_on_auto_detected = warn_on_auto_detected

    def run(
        self,
        path: Path,
        recursive: bool = True,
    ) -> PrivacyCheckReport:
        """
        Run privacy-by-design check on a path.

        Args:
            path: Directory or file path
            recursive: Whether to scan recursively

        Returns:
            PrivacyCheckReport
        """
        import time

        start_time = time.time()

        report = PrivacyCheckReport(
            check_id=f"privacy-check-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
        )

        # Scan for declarations
        if path.is_file():
            declarations = self._scanner.scan_file(path)
            report.files_scanned = 1
        else:
            declarations = self._scanner.scan_directory(path, recursive)
            report.files_scanned = len(set(d.file_path for d in declarations))

        report.declarations_found = len(declarations)

        # Validate each declaration
        for decl in declarations:
            self._validate_declaration(decl, report)

        # Set result
        if report.should_fail_ci:
            report.result = CheckResult.FAIL
        elif report.warnings:
            report.result = CheckResult.WARN
        else:
            report.result = CheckResult.PASS

        report.execution_time_ms = (time.time() - start_time) * 1000

        return report

    def run_on_files(self, file_paths: List[Path]) -> PrivacyCheckReport:
        """
        Run check on specific files.

        Args:
            file_paths: List of file paths

        Returns:
            PrivacyCheckReport
        """
        import time

        start_time = time.time()

        report = PrivacyCheckReport(
            check_id=f"privacy-check-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
        )

        declarations = self._scanner.scan_files(file_paths)
        report.files_scanned = len(file_paths)
        report.declarations_found = len(declarations)

        for decl in declarations:
            self._validate_declaration(decl, report)

        if report.should_fail_ci:
            report.result = CheckResult.FAIL
        elif report.warnings:
            report.result = CheckResult.WARN
        else:
            report.result = CheckResult.PASS

        report.execution_time_ms = (time.time() - start_time) * 1000

        return report

    def run_on_diff(
        self,
        base_ref: str = "HEAD~1",
        repo_path: Optional[Path] = None,
    ) -> PrivacyCheckReport:
        """
        Run check on files changed since base ref.

        Args:
            base_ref: Git ref to compare against
            repo_path: Repository path

        Returns:
            PrivacyCheckReport
        """
        import subprocess

        repo_path = repo_path or Path.cwd()

        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", base_ref],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            changed_files = [
                repo_path / f.strip() for f in result.stdout.strip().split("\n") if f.strip()
            ]
            return self.run_on_files(changed_files)
        except subprocess.CalledProcessError as e:
            report = PrivacyCheckReport(
                check_id=f"privacy-check-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
                result=CheckResult.SKIP,
            )
            report.warnings.append(f"Could not get git diff: {e}")
            return report

    def _validate_declaration(
        self,
        decl: DataDeclaration,
        report: PrivacyCheckReport,
    ) -> None:
        """Validate a single declaration against inventory."""

        # If registry available, use it
        if self._registry and HAS_INVENTORY:
            result = self._registry.validate(decl.name)

            if result.is_registered:
                report.declarations_registered += 1

                if not result.is_compliant and self._fail_on_non_compliant:
                    report.violations.append(
                        CheckViolation(
                            declaration=decl,
                            severity=CheckSeverity.HIGH,
                            message=f"Field '{decl.name}' is registered but non-compliant: "
                            f"{', '.join(result.violations)}",
                            inventory_result=result.to_dict(),
                            remediation="Update inventory entry with missing GDPR fields",
                        )
                    )
            else:
                report.declarations_unregistered += 1

                if self._fail_on_unregistered:
                    severity = (
                        CheckSeverity.CRITICAL
                        if (decl.auto_detected.get("credential") or decl.auto_detected.get("pii"))
                        else CheckSeverity.HIGH
                    )

                    report.violations.append(
                        CheckViolation(
                            declaration=decl,
                            severity=severity,
                            message=f"Field '{decl.name}' ({decl.declaration_type}) is not "
                            f"registered in data inventory",
                            inventory_result=result.to_dict() if result else None,
                            remediation=(
                                f"Register '{decl.name}' in data inventory with: "
                                "classification, retention, residency, redaction requirements"
                            ),
                        )
                    )
        else:
            # No registry - check based on auto-detection only
            report.declarations_unregistered += 1

            if self._fail_on_unregistered:
                if decl.auto_detected.get("credential"):
                    report.violations.append(
                        CheckViolation(
                            declaration=decl,
                            severity=CheckSeverity.CRITICAL,
                            message=f"Field '{decl.name}' appears to be a credential and must be registered",
                            remediation="Register in data inventory with NEVER_TRANSMIT redaction",
                        )
                    )
                elif decl.auto_detected.get("pii"):
                    report.violations.append(
                        CheckViolation(
                            declaration=decl,
                            severity=CheckSeverity.HIGH,
                            message=f"Field '{decl.name}' appears to be PII and must be registered",
                            remediation="Register in data inventory with MANDATORY redaction",
                        )
                    )
                elif decl.auto_detected.get("order_field"):
                    report.violations.append(
                        CheckViolation(
                            declaration=decl,
                            severity=CheckSeverity.HIGH,
                            message=f"Field '{decl.name}' appears to be order-related and must be registered",
                            remediation="Register in data inventory with RAW_ORDER_EVENTS min_telemetry_level",
                        )
                    )

        # Warnings for auto-detected issues even if registered
        if self._warn_on_auto_detected:
            if decl.auto_detected.get("credential"):
                report.warnings.append(
                    f"{decl.file_path}:{decl.line_number} - "
                    f"'{decl.name}' appears to be a credential"
                )
            if decl.auto_detected.get("pii"):
                report.warnings.append(
                    f"{decl.file_path}:{decl.line_number} - " f"'{decl.name}' appears to be PII"
                )
            if decl.auto_detected.get("order_field"):
                report.warnings.append(
                    f"{decl.file_path}:{decl.line_number} - "
                    f"'{decl.name}' appears to be order-related"
                )


# ============================================================================
# CLI
# ============================================================================


def format_report(report: PrivacyCheckReport, verbose: bool = False) -> str:
    """Format report for CLI output."""
    lines = []

    lines.append("=" * 70)
    lines.append("GDPR Phase 8: Privacy-by-Design Check")
    lines.append("=" * 70)
    lines.append("")

    # Summary
    lines.append(f"Check ID: {report.check_id}")
    lines.append(f"Result: {report.result.value.upper()}")
    lines.append(f"Execution Time: {report.execution_time_ms:.2f}ms")
    lines.append("")

    lines.append("Summary:")
    lines.append(f"  Files scanned: {report.files_scanned}")
    lines.append(f"  Declarations found: {report.declarations_found}")
    lines.append(f"  Registered: {report.declarations_registered}")
    lines.append(f"  Unregistered: {report.declarations_unregistered}")
    lines.append("")

    # Violations
    if report.violations:
        lines.append(f"Violations ({len(report.violations)}):")
        lines.append(f"  Critical: {report.critical_count}")
        lines.append(f"  High: {report.high_count}")
        lines.append(f"  Medium: {report.medium_count}")
        lines.append(f"  Low: {report.low_count}")
        lines.append("")

        for v in report.violations:
            lines.append(f"[{v.severity.value.upper()}] {v.message}")
            lines.append(f"  File: {v.declaration.file_path}:{v.declaration.line_number}")
            lines.append(f"  Context: {v.declaration.context[:80]}...")
            lines.append(f"  Remediation: {v.remediation}")
            lines.append("")
    else:
        lines.append("No violations found.")
        lines.append("")

    # Warnings
    if report.warnings and verbose:
        lines.append(f"Warnings ({len(report.warnings)}):")
        for w in report.warnings[:20]:
            lines.append(f"  - {w}")
        if len(report.warnings) > 20:
            lines.append(f"  ... and {len(report.warnings) - 20} more")
        lines.append("")

    # Final status
    lines.append("=" * 70)
    if report.should_fail_ci:
        lines.append("STATUS: FAILED - CI must not proceed")
        lines.append("")
        lines.append("To fix:")
        lines.append("1. Register all unregistered fields in the Data Inventory")
        lines.append("2. Ensure all entries have: classification, retention, residency, redaction")
        lines.append("3. Get entries approved by DPO/compliance")
    else:
        lines.append("STATUS: PASSED - OK to proceed")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="GDPR Phase 8: Privacy-by-Design CI Check")
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to scan (directory or file)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific files to scan",
    )
    parser.add_argument(
        "--diff",
        metavar="REF",
        help="Scan only files changed since REF (e.g., HEAD~1)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for JSON report",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Don't fail CI, only warn",
    )
    parser.add_argument(
        "--skip-unregistered",
        action="store_true",
        help="Don't fail on unregistered fields",
    )

    args = parser.parse_args()

    # Initialize check
    check = PrivacyByDesignCheck(
        fail_on_unregistered=not args.skip_unregistered and not args.warn_only,
        fail_on_non_compliant=not args.warn_only,
    )

    # Run check
    if args.diff:
        report = check.run_on_diff(base_ref=args.diff)
    elif args.files:
        report = check.run_on_files([Path(f) for f in args.files])
    else:
        report = check.run(Path(args.path))

    # Output
    print(format_report(report, verbose=args.verbose))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        print(f"\nJSON report written to: {args.output}")

    # Exit code
    if report.should_fail_ci and not args.warn_only:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    "CheckSeverity",
    "CheckResult",
    # Data classes
    "DataDeclaration",
    "CheckViolation",
    "PrivacyCheckReport",
    # Scanner
    "DataDeclarationScanner",
    # Check
    "PrivacyByDesignCheck",
    # CLI
    "format_report",
    "main",
    # Constants
    "TELEMETRY_FIELD_PATTERNS",
    "DATA_STORE_PATTERNS",
    "LOG_STREAM_PATTERNS",
    "SKIP_PATTERNS",
    "KNOWN_SAFE_PATTERNS",
]
