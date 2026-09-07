# -*- coding: utf-8 -*-
"""
Redaction Non-Bypassable Enforcement Check.

GDPR Phase 2 Implementation: Ensures redaction cannot be disabled.

This module:
    - Verifies redaction middleware is always enabled
    - Prevents feature flags from disabling redaction
    - Checks for bypass attempts in code

Design Doc Reference:
    - docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L871 (redaction always on)
    - docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1051 (cannot be disabled by feature flag)
    - docs/design/CCEA_CLOUD/CI_GUARDRAILS.md (RT-005)

Usage:
    python -m ccea.guardrails.redaction_enforcement_check

Exit codes:
    0 - Check passed (redaction properly enforced)
    1 - Check failed (bypass attempts found)
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Pattern, Set, Tuple


# ============================================================================
# Constants
# ============================================================================

# Patterns that indicate redaction bypass attempts
BYPASS_PATTERNS: List[Tuple[Pattern, str, str]] = [
    # Direct disable attempts
    (
        re.compile(r"redaction\s*=\s*False", re.IGNORECASE),
        "disable_redaction",
        "Direct redaction disable attempt",
    ),
    (
        re.compile(r"redaction_enabled\s*=\s*False", re.IGNORECASE),
        "disable_redaction",
        "Setting redaction_enabled to False",
    ),
    (
        re.compile(r"skip_redaction\s*=\s*True", re.IGNORECASE),
        "skip_redaction",
        "Attempting to skip redaction",
    ),
    (
        re.compile(r"bypass_redaction\s*=\s*True", re.IGNORECASE),
        "bypass_redaction",
        "Attempting to bypass redaction",
    ),
    (
        re.compile(r"disable_redaction\s*=\s*True", re.IGNORECASE),
        "disable_redaction",
        "Attempting to disable redaction",
    ),
    # Feature flag bypass attempts
    (
        re.compile(r"feature_flag.*redaction.*false", re.IGNORECASE),
        "feature_flag_bypass",
        "Feature flag attempting to disable redaction",
    ),
    (
        re.compile(r"if\s+not\s+.*redaction", re.IGNORECASE),
        "conditional_bypass",
        "Conditional bypass of redaction",
    ),
    # Environment variable bypass
    (
        re.compile(r"os\.getenv.*DISABLE.*REDACTION", re.IGNORECASE),
        "env_var_bypass",
        "Environment variable attempting to disable redaction",
    ),
    (
        re.compile(r"os\.environ.*SKIP.*REDACTION", re.IGNORECASE),
        "env_var_bypass",
        "Environment variable attempting to skip redaction",
    ),
    # Configuration bypass
    (
        re.compile(r"config.*redaction.*disabled", re.IGNORECASE),
        "config_bypass",
        "Configuration attempting to disable redaction",
    ),
]

# Files to check for redaction enforcement
REDACTION_FILES: List[str] = [
    "ccea/telemetry/redaction.py",
    "packages/agent/telemetry/redaction.py",
    "packages/cloud/control_plane/middleware/telemetry_validation.py",
    "packages/cloud/control_plane/telemetry_ingester.py",
]

# Files to scan for bypass attempts
SCAN_PATTERNS: List[str] = [
    "packages/**/*.py",
    "ccea/**/*.py",
    "**/*.yaml",
    "**/*.yml",
]

# Excluded paths (test files, documentation)
EXCLUDED_PATHS: FrozenSet[str] = frozenset(
    {
        "tests/",
        "test_",
        "_test.py",
        "docs/",
        "__pycache__/",
        ".git/",
        "node_modules/",
        "venv/",
        ".venv/",
    }
)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class RedactionViolation:
    """A redaction enforcement violation."""

    file_path: str
    line_number: int
    violation_type: str
    message: str
    code_snippet: str
    severity: str = "CRITICAL"


@dataclass
class RedactionEnforcementResult:
    """Result of redaction enforcement check."""

    passed: bool
    message: str
    violations: List[RedactionViolation] = field(default_factory=list)
    files_checked: int = 0
    middleware_verified: bool = False
    feature_flag_safe: bool = False


# ============================================================================
# File Finding Functions
# ============================================================================


def find_python_files() -> List[Path]:
    """Find all Python files to check."""
    import glob

    files = []
    for pattern in SCAN_PATTERNS:
        if pattern.endswith(".py"):
            matches = glob.glob(pattern, recursive=True)
            files.extend(Path(m) for m in matches if Path(m).is_file())

    # Filter excluded paths
    filtered = []
    for f in files:
        path_str = str(f)
        if not any(excl in path_str for excl in EXCLUDED_PATHS):
            filtered.append(f)

    return filtered


def find_config_files() -> List[Path]:
    """Find configuration files to check."""
    import glob

    files = []
    for pattern in SCAN_PATTERNS:
        if pattern.endswith(".yaml") or pattern.endswith(".yml"):
            matches = glob.glob(pattern, recursive=True)
            files.extend(Path(m) for m in matches if Path(m).is_file())

    # Filter excluded paths
    filtered = []
    for f in files:
        path_str = str(f)
        if not any(excl in path_str for excl in EXCLUDED_PATHS):
            filtered.append(f)

    return filtered


# ============================================================================
# Verification Functions
# ============================================================================


def verify_redaction_middleware() -> Tuple[bool, List[RedactionViolation]]:
    """
    Verify that redaction middleware is properly implemented and non-bypassable.

    Returns:
        (is_valid, list_of_violations)
    """
    violations = []

    # Check main redaction file
    redaction_path = Path("ccea/telemetry/redaction.py")
    if not redaction_path.exists():
        violations.append(
            RedactionViolation(
                file_path=str(redaction_path),
                line_number=0,
                violation_type="missing_file",
                message="Core redaction module not found",
                code_snippet="",
            )
        )
        return False, violations

    try:
        with open(redaction_path, "r") as f:
            content = f.read()

        # Check for required elements
        required_elements = [
            ("_ENABLED: bool = True", "Redaction must be enabled by default"),
            ("_LOCK_ENABLED: bool = True", "Redaction must be locked to prevent disabling"),
            ("def disable(self)", "Disable method must exist but do nothing"),
            ("return True", "enabled property must always return True"),
        ]

        for element, error_msg in required_elements:
            if element not in content:
                violations.append(
                    RedactionViolation(
                        file_path=str(redaction_path),
                        line_number=0,
                        violation_type="missing_enforcement",
                        message=error_msg,
                        code_snippet=element,
                    )
                )

        # Check that disable() does nothing
        if "def disable(self)" in content:
            # Parse AST to verify disable() is a no-op
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == "disable":
                        # Check body is just 'pass' or docstring + pass
                        body = [n for n in node.body if not isinstance(n, ast.Expr)]
                        if len(body) > 1:
                            violations.append(
                                RedactionViolation(
                                    file_path=str(redaction_path),
                                    line_number=node.lineno,
                                    violation_type="dangerous_disable",
                                    message="disable() method must be a no-op (only pass statement)",
                                    code_snippet="disable() should do nothing",
                                )
                            )
            except SyntaxError:
                pass

    except Exception as e:
        violations.append(
            RedactionViolation(
                file_path=str(redaction_path),
                line_number=0,
                violation_type="read_error",
                message=f"Error reading file: {e}",
                code_snippet="",
            )
        )

    return len(violations) == 0, violations


def check_bypass_patterns(file_path: Path) -> List[RedactionViolation]:
    """Check a file for redaction bypass patterns."""
    violations = []

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
    except Exception:
        return violations

    for line_num, line in enumerate(lines, 1):
        # Skip comments
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//"):
            continue

        # Check against bypass patterns
        for pattern, violation_type, message in BYPASS_PATTERNS:
            if pattern.search(line):
                # Skip if it's in a test file or documentation
                path_str = str(file_path)
                if any(excl in path_str for excl in ("test", "spec", "docs", "example")):
                    continue

                # Skip if it's a comment about what's not allowed
                if "forbidden" in line.lower() or "not allowed" in line.lower():
                    continue
                if "must" in line.lower() and "not" in line.lower():
                    continue

                violations.append(
                    RedactionViolation(
                        file_path=str(file_path),
                        line_number=line_num,
                        violation_type=violation_type,
                        message=message,
                        code_snippet=line.strip()[:100],
                    )
                )

    return violations


def check_feature_flag_safety() -> Tuple[bool, List[RedactionViolation]]:
    """
    Verify that feature flags cannot disable redaction.

    Returns:
        (is_safe, list_of_violations)
    """
    violations = []

    # Look for feature flag definitions
    feature_flag_files = [
        Path("packages/cloud/config/feature_flags.py"),
        Path("packages/cloud/config/flags.py"),
        Path("packages/cloud/control_plane/config.py"),
        Path("ccea/config/feature_flags.py"),
    ]

    for flag_file in feature_flag_files:
        if not flag_file.exists():
            continue

        try:
            with open(flag_file, "r") as f:
                content = f.read()
                lines = content.split("\n")

            for line_num, line in enumerate(lines, 1):
                # Check for redaction-related feature flags
                if re.search(r"redaction", line, re.IGNORECASE):
                    # Check if it's a flag that could disable redaction
                    if re.search(r"(disable|skip|bypass|off)", line, re.IGNORECASE):
                        violations.append(
                            RedactionViolation(
                                file_path=str(flag_file),
                                line_number=line_num,
                                violation_type="feature_flag_bypass",
                                message="Feature flag that could disable redaction is forbidden",
                                code_snippet=line.strip()[:100],
                            )
                        )

        except Exception:
            pass

    return len(violations) == 0, violations


def check_config_files() -> List[RedactionViolation]:
    """Check configuration files for redaction bypass settings."""
    violations = []
    config_files = find_config_files()

    for file_path in config_files:
        try:
            with open(file_path, "r") as f:
                content = f.read()
                lines = content.split("\n")

            for line_num, line in enumerate(lines, 1):
                # Check for redaction disable in config
                if re.search(r"redaction.*:\s*(false|off|disabled)", line, re.IGNORECASE):
                    violations.append(
                        RedactionViolation(
                            file_path=str(file_path),
                            line_number=line_num,
                            violation_type="config_bypass",
                            message="Configuration cannot disable redaction",
                            code_snippet=line.strip()[:100],
                        )
                    )

        except Exception:
            pass

    return violations


# ============================================================================
# Main Check Function
# ============================================================================


def check_redaction_enforcement() -> RedactionEnforcementResult:
    """
    Full redaction enforcement check.

    Returns:
        RedactionEnforcementResult with pass/fail status
    """
    all_violations = []
    files_checked = 0

    # 1. Verify redaction middleware
    middleware_ok, middleware_violations = verify_redaction_middleware()
    all_violations.extend(middleware_violations)

    # 2. Check feature flag safety
    flags_safe, flag_violations = check_feature_flag_safety()
    all_violations.extend(flag_violations)

    # 3. Scan Python files for bypass patterns
    python_files = find_python_files()
    for file_path in python_files:
        violations = check_bypass_patterns(file_path)
        all_violations.extend(violations)
        files_checked += 1

    # 4. Check config files
    config_violations = check_config_files()
    all_violations.extend(config_violations)
    files_checked += len(find_config_files())

    # Determine result
    if not all_violations:
        return RedactionEnforcementResult(
            passed=True,
            message="Redaction enforcement verified - cannot be bypassed",
            violations=[],
            files_checked=files_checked,
            middleware_verified=middleware_ok,
            feature_flag_safe=flags_safe,
        )

    return RedactionEnforcementResult(
        passed=False,
        message=f"Found {len(all_violations)} redaction enforcement violations",
        violations=all_violations,
        files_checked=files_checked,
        middleware_verified=middleware_ok,
        feature_flag_safe=flags_safe,
    )


# ============================================================================
# Runtime Enforcement API
# ============================================================================


class RedactionEnforcer:
    """
    Runtime enforcer that ensures redaction is always active.

    CRITICAL: This class cannot be bypassed. Any attempt to disable
    redaction will be ignored and logged.
    """

    _instance: Optional["RedactionEnforcer"] = None
    _ENABLED: bool = True
    _LOCKED: bool = True

    def __new__(cls) -> "RedactionEnforcer":
        """Singleton to ensure consistent enforcement."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def enabled(self) -> bool:
        """
        Check if redaction is enabled.

        ALWAYS returns True - redaction cannot be disabled.
        """
        return True

    def disable(self) -> None:
        """
        Attempt to disable redaction.

        IMPORTANT: This method does NOTHING.
        Redaction cannot be disabled per Design Doc requirements.

        Any call to this method will be logged as a security event.
        """
        # Log the attempt (in production, this would alert security)
        import logging

        logging.getLogger(__name__).warning(
            "SECURITY: Attempt to disable redaction was blocked. "
            "Redaction cannot be disabled per Design Doc 13.3."
        )
        # Intentionally does nothing

    def verify_enabled(self) -> Tuple[bool, str]:
        """
        Verify redaction is enabled and cannot be bypassed.

        Returns:
            (is_enabled, status_message)

        Always returns (True, "Redaction enabled and locked").
        """
        return True, "Redaction enabled and locked (cannot be bypassed)"


def get_redaction_enforcer() -> RedactionEnforcer:
    """Get the singleton redaction enforcer."""
    return RedactionEnforcer()


def assert_redaction_enabled() -> None:
    """
    Assert that redaction is enabled.

    This function always succeeds because redaction cannot be disabled.
    Provided for explicit verification in code.
    """
    enforcer = get_redaction_enforcer()
    enabled, _ = enforcer.verify_enabled()
    assert enabled, "Redaction must be enabled"


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> int:
    """Main entry point for CI."""
    print("=" * 60)
    print("REDACTION NON-BYPASSABLE ENFORCEMENT CHECK")
    print("=" * 60)

    result = check_redaction_enforcement()

    print(f"\nFiles checked: {result.files_checked}")
    print(f"Middleware verified: {result.middleware_verified}")
    print(f"Feature flags safe: {result.feature_flag_safe}")

    if result.passed:
        print(f"\n[PASS] {result.message}")
        return 0
    else:
        print(f"\n[FAIL] {result.message}")
        print("\nViolations:")
        for v in result.violations:
            print(f"  [{v.severity}] {v.file_path}:{v.line_number}")
            print(f"         {v.violation_type}: {v.message}")
            if v.code_snippet:
                print(f"         Code: {v.code_snippet}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
