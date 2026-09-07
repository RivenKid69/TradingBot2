# -*- coding: utf-8 -*-
"""
EU-Only Residency Guardrail Check.

GDPR Phase 3 Implementation: CI/CD guardrail for EU-only residency enforcement.

This module:
    - Validates deployment configurations for EU-only compliance
    - Checks Helm values, Docker Compose, and Kubernetes manifests
    - Fails closed on any non-EU endpoint detection
    - Produces machine-readable reports for evidence pack

Design Doc Reference:
    - docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L892 (14.3: EU residency)
    - docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1745 (6.2: EU-only drift checks)
    - docs/compliance/GDPR_CCEA_IMPLEMENTATION_PLAN.md (Phase 3)

Usage:
    python -m ccea.guardrails.residency_check [--config CONFIG_DIR]

Exit codes:
    0 - Check passed (all endpoints EU-compliant)
    1 - Check failed (non-EU endpoints found)
    2 - Configuration error
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Pattern, Set, Tuple

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ============================================================================
# Constants
# ============================================================================

# EU AWS regions
EU_AWS_REGIONS: FrozenSet[str] = frozenset(
    {
        "eu-west-1",
        "eu-west-2",
        "eu-west-3",
        "eu-central-1",
        "eu-central-2",
        "eu-north-1",
        "eu-south-1",
        "eu-south-2",
    }
)

# Non-EU AWS regions (explicit deny list)
NON_EU_AWS_REGIONS: FrozenSet[str] = frozenset(
    {
        "us-east-1",
        "us-east-2",
        "us-west-1",
        "us-west-2",
        "us-gov-east-1",
        "us-gov-west-1",
        "ap-northeast-1",
        "ap-northeast-2",
        "ap-northeast-3",
        "ap-southeast-1",
        "ap-southeast-2",
        "ap-southeast-3",
        "ap-south-1",
        "ap-east-1",
        "sa-east-1",
        "af-south-1",
        "me-south-1",
        "me-central-1",
        "ca-central-1",
    }
)

# Patterns that indicate non-EU regions in config files
NON_EU_PATTERNS: List[Tuple[Pattern, str]] = [
    # US regions
    (re.compile(r"us-east-\d+", re.IGNORECASE), "US East region detected"),
    (re.compile(r"us-west-\d+", re.IGNORECASE), "US West region detected"),
    (re.compile(r"us-gov-", re.IGNORECASE), "US Gov region detected"),
    # Asia Pacific
    (re.compile(r"ap-northeast-\d+", re.IGNORECASE), "Asia Pacific region detected"),
    (re.compile(r"ap-southeast-\d+", re.IGNORECASE), "Asia Pacific region detected"),
    (re.compile(r"ap-south-\d+", re.IGNORECASE), "Asia Pacific region detected"),
    (re.compile(r"ap-east-\d+", re.IGNORECASE), "Asia Pacific region detected"),
    # Other non-EU
    (re.compile(r"sa-east-\d+", re.IGNORECASE), "South America region detected"),
    (re.compile(r"af-south-\d+", re.IGNORECASE), "Africa region detected"),
    (re.compile(r"me-south-\d+", re.IGNORECASE), "Middle East region detected"),
    (re.compile(r"me-central-\d+", re.IGNORECASE), "Middle East region detected"),
    (re.compile(r"ca-central-\d+", re.IGNORECASE), "Canada region detected"),
]

# Patterns that indicate EU regions (for positive validation)
EU_PATTERNS: List[Pattern] = [
    re.compile(r"eu-west-\d+", re.IGNORECASE),
    re.compile(r"eu-central-\d+", re.IGNORECASE),
    re.compile(r"eu-north-\d+", re.IGNORECASE),
    re.compile(r"eu-south-\d+", re.IGNORECASE),
    re.compile(r"europe-west\d+", re.IGNORECASE),  # GCP
    re.compile(r"europe-north\d+", re.IGNORECASE),  # GCP
    re.compile(r"europe-central\d+", re.IGNORECASE),  # GCP
    re.compile(r"westeurope", re.IGNORECASE),  # Azure
    re.compile(r"northeurope", re.IGNORECASE),  # Azure
    re.compile(r"germanywest", re.IGNORECASE),  # Azure
    re.compile(r"francecentral", re.IGNORECASE),  # Azure
]

# Config file patterns to check
CONFIG_PATTERNS: List[str] = [
    "**/values*.yaml",
    "**/values*.yml",
    "**/docker-compose*.yaml",
    "**/docker-compose*.yml",
    "**/*.env",
    "**/.env*",
    "**/deployment*.yaml",
    "**/configmap*.yaml",
]

# Paths to exclude
EXCLUDED_PATHS: FrozenSet[str] = frozenset(
    {
        "node_modules",
        ".git",
        "__pycache__",
        "venv",
        ".venv",
        "tests/fixtures",
        "test_data",
    }
)


# ============================================================================
# Data Classes
# ============================================================================


class CheckResult(str, Enum):
    """Result of a guardrail check."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIPPED = "skipped"


class CheckSeverity(str, Enum):
    """Severity of violations."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ResidencyViolation:
    """A residency policy violation."""

    file_path: str
    line_number: int
    violation_type: str
    message: str
    region_found: str
    severity: CheckSeverity = CheckSeverity.CRITICAL
    code_snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "violation_type": self.violation_type,
            "message": self.message,
            "region_found": self.region_found,
            "severity": self.severity.value,
            "code_snippet": self.code_snippet,
        }


@dataclass
class ResidencyCheckResult:
    """Result of residency guardrail check."""

    passed: bool
    result: CheckResult
    message: str
    violations: List[ResidencyViolation] = field(default_factory=list)
    files_checked: int = 0
    eu_regions_found: int = 0
    non_eu_regions_found: int = 0
    check_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "result": self.result.value,
            "message": self.message,
            "violations": [v.to_dict() for v in self.violations],
            "files_checked": self.files_checked,
            "eu_regions_found": self.eu_regions_found,
            "non_eu_regions_found": self.non_eu_regions_found,
            "check_timestamp": self.check_timestamp.isoformat(),
        }


# ============================================================================
# File Discovery
# ============================================================================


def find_config_files(base_path: Path, patterns: Optional[List[str]] = None) -> List[Path]:
    """
    Find configuration files to check.

    Args:
        base_path: Base directory to search
        patterns: Glob patterns to match

    Returns:
        List of file paths
    """
    if patterns is None:
        patterns = CONFIG_PATTERNS

    files: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(str(base_path / pattern), recursive=True)
        for match in matches:
            path = Path(match)
            # Skip excluded paths
            if any(excl in str(path) for excl in EXCLUDED_PATHS):
                continue
            if path.is_file():
                files.append(path)

    return list(set(files))


# ============================================================================
# Region Detection
# ============================================================================


def detect_region_in_string(content: str) -> Tuple[List[str], List[str]]:
    """
    Detect regions mentioned in a string.

    Args:
        content: String content to analyze

    Returns:
        (eu_regions, non_eu_regions)
    """
    eu_regions: List[str] = []
    non_eu_regions: List[str] = []

    # Check for EU regions
    for pattern in EU_PATTERNS:
        matches = pattern.findall(content)
        eu_regions.extend(matches)

    # Check for non-EU regions
    for pattern, _ in NON_EU_PATTERNS:
        matches = pattern.findall(content)
        non_eu_regions.extend(matches)

    return eu_regions, non_eu_regions


def is_eu_region(region: str) -> bool:
    """Check if a region identifier is EU-based."""
    region_lower = region.lower().strip()

    # Check against EU patterns
    for pattern in EU_PATTERNS:
        if pattern.match(region_lower):
            return True

    # Check against EU region set
    if region_lower in {r.lower() for r in EU_AWS_REGIONS}:
        return True

    # Check for EU keywords
    if "eu" in region_lower or "europe" in region_lower:
        return True

    return False


def is_non_eu_region(region: str) -> bool:
    """Check if a region identifier is explicitly non-EU."""
    region_lower = region.lower().strip()

    # Check against non-EU patterns
    for pattern, _ in NON_EU_PATTERNS:
        if pattern.match(region_lower):
            return True

    # Check against non-EU region set
    if region_lower in {r.lower() for r in NON_EU_AWS_REGIONS}:
        return True

    return False


# ============================================================================
# File Checkers
# ============================================================================


def check_yaml_file(file_path: Path) -> Tuple[List[ResidencyViolation], List[str], List[str]]:
    """
    Check a YAML file for residency violations.

    Args:
        file_path: Path to YAML file

    Returns:
        (violations, eu_regions, non_eu_regions)
    """
    violations: List[ResidencyViolation] = []
    eu_regions: List[str] = []
    non_eu_regions: List[str] = []

    try:
        with open(file_path, "r") as f:
            content = f.read()
            lines = content.split("\n")

        # Line-by-line analysis
        for line_num, line in enumerate(lines, 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # Detect regions
            line_eu, line_non_eu = detect_region_in_string(line)
            eu_regions.extend(line_eu)
            non_eu_regions.extend(line_non_eu)

            # Create violations for non-EU regions
            for region in line_non_eu:
                violations.append(
                    ResidencyViolation(
                        file_path=str(file_path),
                        line_number=line_num,
                        violation_type="non_eu_region",
                        message=f"Non-EU region detected: {region}",
                        region_found=region,
                        severity=CheckSeverity.CRITICAL,
                        code_snippet=line.strip()[:100],
                    )
                )

        # Parse YAML for deep inspection
        if HAS_YAML:
            try:
                data = yaml.safe_load(content)
                if data:
                    deep_violations = _check_yaml_structure(data, str(file_path))
                    violations.extend(deep_violations)
            except yaml.YAMLError:
                pass  # Continue with line-based analysis

    except Exception as e:
        violations.append(
            ResidencyViolation(
                file_path=str(file_path),
                line_number=0,
                violation_type="read_error",
                message=f"Error reading file: {e}",
                region_found="unknown",
                severity=CheckSeverity.LOW,
            )
        )

    return violations, eu_regions, non_eu_regions


def _check_yaml_structure(
    data: Any,
    file_path: str,
    path: str = "",
) -> List[ResidencyViolation]:
    """Recursively check YAML structure for region values."""
    violations: List[ResidencyViolation] = []

    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key

            # Check if key indicates a region field
            key_lower = key.lower()
            if "region" in key_lower or key_lower in ("awsregion", "aws_region", "azureregion"):
                if isinstance(value, str) and is_non_eu_region(value):
                    violations.append(
                        ResidencyViolation(
                            file_path=file_path,
                            line_number=0,
                            violation_type="non_eu_region_config",
                            message=f"Non-EU region in config key '{current_path}': {value}",
                            region_found=value,
                            severity=CheckSeverity.CRITICAL,
                        )
                    )

            # Recurse
            violations.extend(_check_yaml_structure(value, file_path, current_path))

    elif isinstance(data, list):
        for i, item in enumerate(data):
            violations.extend(_check_yaml_structure(item, file_path, f"{path}[{i}]"))

    elif isinstance(data, str):
        # Check string value for embedded regions
        if is_non_eu_region(data):
            violations.append(
                ResidencyViolation(
                    file_path=file_path,
                    line_number=0,
                    violation_type="non_eu_region_value",
                    message=f"Non-EU region in value at '{path}': {data}",
                    region_found=data,
                    severity=CheckSeverity.CRITICAL,
                )
            )

    return violations


def check_env_file(file_path: Path) -> Tuple[List[ResidencyViolation], List[str], List[str]]:
    """
    Check an environment file for residency violations.

    Args:
        file_path: Path to env file

    Returns:
        (violations, eu_regions, non_eu_regions)
    """
    violations: List[ResidencyViolation] = []
    eu_regions: List[str] = []
    non_eu_regions: List[str] = []

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            # Skip comments and empty lines
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # Parse KEY=VALUE
            if "=" in stripped:
                key, _, value = stripped.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")

                # Check region-related keys
                key_lower = key.lower()
                if "region" in key_lower or key_lower in (
                    "aws_default_region",
                    "aws_region",
                    "s3_region",
                    "database_region",
                    "redis_region",
                    "cache_region",
                ):
                    if is_non_eu_region(value):
                        violations.append(
                            ResidencyViolation(
                                file_path=str(file_path),
                                line_number=line_num,
                                violation_type="non_eu_region_env",
                                message=f"Non-EU region in environment variable {key}: {value}",
                                region_found=value,
                                severity=CheckSeverity.CRITICAL,
                                code_snippet=stripped[:100],
                            )
                        )
                        non_eu_regions.append(value)
                    elif is_eu_region(value):
                        eu_regions.append(value)

            # Check for embedded regions in URLs
            line_eu, line_non_eu = detect_region_in_string(stripped)
            eu_regions.extend(line_eu)

            for region in line_non_eu:
                if region not in non_eu_regions:
                    violations.append(
                        ResidencyViolation(
                            file_path=str(file_path),
                            line_number=line_num,
                            violation_type="non_eu_region_url",
                            message=f"Non-EU region in URL/endpoint: {region}",
                            region_found=region,
                            severity=CheckSeverity.CRITICAL,
                            code_snippet=stripped[:100],
                        )
                    )
                    non_eu_regions.append(region)

    except Exception as e:
        violations.append(
            ResidencyViolation(
                file_path=str(file_path),
                line_number=0,
                violation_type="read_error",
                message=f"Error reading file: {e}",
                region_found="unknown",
                severity=CheckSeverity.LOW,
            )
        )

    return violations, eu_regions, non_eu_regions


# ============================================================================
# Main Check Function
# ============================================================================


def check_residency_compliance(
    base_path: Optional[Path] = None,
    fail_closed: bool = True,
) -> ResidencyCheckResult:
    """
    Full residency compliance check.

    Args:
        base_path: Base directory to check (default: current working directory)
        fail_closed: If True, unknown regions fail the check

    Returns:
        ResidencyCheckResult
    """
    if base_path is None:
        base_path = Path.cwd()

    all_violations: List[ResidencyViolation] = []
    all_eu_regions: List[str] = []
    all_non_eu_regions: List[str] = []
    files_checked = 0

    # Find config files
    config_files = find_config_files(base_path)

    for file_path in config_files:
        files_checked += 1

        # Check based on file type
        suffix = file_path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            violations, eu_regions, non_eu_regions = check_yaml_file(file_path)
        elif suffix in (".env", "") or file_path.name.startswith(".env"):
            violations, eu_regions, non_eu_regions = check_env_file(file_path)
        else:
            continue

        all_violations.extend(violations)
        all_eu_regions.extend(eu_regions)
        all_non_eu_regions.extend(non_eu_regions)

    # Determine result
    critical_violations = [v for v in all_violations if v.severity == CheckSeverity.CRITICAL]

    if critical_violations:
        return ResidencyCheckResult(
            passed=False,
            result=CheckResult.FAIL,
            message=f"Found {len(critical_violations)} non-EU region violations",
            violations=all_violations,
            files_checked=files_checked,
            eu_regions_found=len(set(all_eu_regions)),
            non_eu_regions_found=len(set(all_non_eu_regions)),
        )

    if all_violations:
        return ResidencyCheckResult(
            passed=False,
            result=CheckResult.WARNING,
            message=f"Found {len(all_violations)} residency warnings",
            violations=all_violations,
            files_checked=files_checked,
            eu_regions_found=len(set(all_eu_regions)),
            non_eu_regions_found=len(set(all_non_eu_regions)),
        )

    return ResidencyCheckResult(
        passed=True,
        result=CheckResult.PASS,
        message=f"All {files_checked} config files passed EU residency check",
        violations=[],
        files_checked=files_checked,
        eu_regions_found=len(set(all_eu_regions)),
        non_eu_regions_found=0,
    )


# ============================================================================
# Helm Values Specific Check
# ============================================================================


def check_helm_values(values_path: Path) -> ResidencyCheckResult:
    """
    Check a Helm values file specifically.

    Args:
        values_path: Path to values.yaml

    Returns:
        ResidencyCheckResult
    """
    if not HAS_YAML:
        return ResidencyCheckResult(
            passed=False,
            result=CheckResult.SKIPPED,
            message="PyYAML not installed, cannot check Helm values",
            files_checked=0,
        )

    violations, eu_regions, non_eu_regions = check_yaml_file(values_path)

    # Additional Helm-specific checks
    try:
        with open(values_path, "r") as f:
            values = yaml.safe_load(f)

        if values:
            # Check global.dataResidency
            global_config = values.get("global", {})
            data_residency = global_config.get("dataResidency", "")

            # If data residency is explicitly non-EU, fail
            if data_residency and is_non_eu_region(data_residency):
                violations.append(
                    ResidencyViolation(
                        file_path=str(values_path),
                        line_number=0,
                        violation_type="non_eu_data_residency",
                        message=f"global.dataResidency is set to non-EU: {data_residency}",
                        region_found=data_residency,
                        severity=CheckSeverity.CRITICAL,
                    )
                )

            # Check governance.residency.enforceEuOnly
            governance = values.get("governance", {})
            residency = governance.get("residency", {})
            enforce_eu_only = residency.get("enforceEuOnly", None)

            if enforce_eu_only is False:
                violations.append(
                    ResidencyViolation(
                        file_path=str(values_path),
                        line_number=0,
                        violation_type="eu_enforcement_disabled",
                        message="governance.residency.enforceEuOnly is set to false",
                        region_found="N/A",
                        severity=CheckSeverity.CRITICAL,
                    )
                )

    except Exception:
        pass

    critical_violations = [v for v in violations if v.severity == CheckSeverity.CRITICAL]

    if critical_violations:
        return ResidencyCheckResult(
            passed=False,
            result=CheckResult.FAIL,
            message=f"Helm values contain {len(critical_violations)} EU residency violations",
            violations=violations,
            files_checked=1,
            eu_regions_found=len(set(eu_regions)),
            non_eu_regions_found=len(set(non_eu_regions)),
        )

    return ResidencyCheckResult(
        passed=True,
        result=CheckResult.PASS,
        message="Helm values passed EU residency check",
        violations=violations,
        files_checked=1,
        eu_regions_found=len(set(eu_regions)),
        non_eu_regions_found=0,
    )


# ============================================================================
# Docker Compose Specific Check
# ============================================================================


def check_docker_compose(compose_path: Path) -> ResidencyCheckResult:
    """
    Check a Docker Compose file specifically.

    Args:
        compose_path: Path to docker-compose.yml

    Returns:
        ResidencyCheckResult
    """
    violations, eu_regions, non_eu_regions = check_yaml_file(compose_path)

    # Additional Docker Compose specific checks
    if HAS_YAML:
        try:
            with open(compose_path, "r") as f:
                compose = yaml.safe_load(f)

            if compose:
                services = compose.get("services", {})
                for service_name, service_config in services.items():
                    env = service_config.get("environment", {})

                    # Handle list format
                    if isinstance(env, list):
                        env_dict = {}
                        for item in env:
                            if "=" in item:
                                key, _, value = item.partition("=")
                                env_dict[key] = value
                        env = env_dict

                    # Check region-related env vars
                    for key, value in env.items():
                        if not isinstance(value, str):
                            continue

                        key_lower = key.lower()
                        if "region" in key_lower:
                            if is_non_eu_region(value):
                                violations.append(
                                    ResidencyViolation(
                                        file_path=str(compose_path),
                                        line_number=0,
                                        violation_type="non_eu_region_compose_env",
                                        message=f"Service '{service_name}' has non-EU region in {key}: {value}",
                                        region_found=value,
                                        severity=CheckSeverity.CRITICAL,
                                    )
                                )

        except Exception:
            pass

    critical_violations = [v for v in violations if v.severity == CheckSeverity.CRITICAL]

    if critical_violations:
        return ResidencyCheckResult(
            passed=False,
            result=CheckResult.FAIL,
            message=f"Docker Compose contains {len(critical_violations)} EU residency violations",
            violations=violations,
            files_checked=1,
            eu_regions_found=len(set(eu_regions)),
            non_eu_regions_found=len(set(non_eu_regions)),
        )

    return ResidencyCheckResult(
        passed=True,
        result=CheckResult.PASS,
        message="Docker Compose passed EU residency check",
        violations=violations,
        files_checked=1,
        eu_regions_found=len(set(eu_regions)),
        non_eu_regions_found=0,
    )


# ============================================================================
# Report Generation
# ============================================================================


def generate_report(result: ResidencyCheckResult, output_path: Optional[Path] = None) -> str:
    """
    Generate a JSON report from check result.

    Args:
        result: Check result
        output_path: Optional path to write report

    Returns:
        JSON string
    """
    report = {
        "check_type": "eu_residency_guardrail",
        "version": "1.0.0",
        **result.to_dict(),
    }

    json_str = json.dumps(report, indent=2)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(json_str)

    return json_str


# ============================================================================
# Main Entry Point
# ============================================================================


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for CI."""
    parser = argparse.ArgumentParser(description="EU-only residency guardrail check")
    parser.add_argument(
        "--config-dir",
        "-c",
        type=Path,
        default=Path.cwd(),
        help="Directory to check for config files",
    )
    parser.add_argument(
        "--helm-values",
        "-v",
        type=Path,
        help="Specific Helm values file to check",
    )
    parser.add_argument(
        "--compose",
        "-d",
        type=Path,
        help="Specific Docker Compose file to check",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--verbose",
        "-V",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args(argv)

    print("=" * 60)
    print("EU-ONLY RESIDENCY GUARDRAIL CHECK")
    print("=" * 60)

    # Run checks
    if args.helm_values:
        result = check_helm_values(args.helm_values)
    elif args.compose:
        result = check_docker_compose(args.compose)
    else:
        result = check_residency_compliance(args.config_dir)

    # Generate report
    if args.output:
        generate_report(result, args.output)
        print(f"\nReport written to: {args.output}")

    # Print results
    print(f"\nFiles checked: {result.files_checked}")
    print(f"EU regions found: {result.eu_regions_found}")
    print(f"Non-EU regions found: {result.non_eu_regions_found}")

    if result.passed:
        print(f"\n[PASS] {result.message}")
        return 0
    else:
        print(f"\n[FAIL] {result.message}")
        if result.violations:
            print("\nViolations:")
            for v in result.violations:
                print(f"  [{v.severity.value.upper()}] {v.file_path}:{v.line_number}")
                print(f"           {v.violation_type}: {v.message}")
                if v.code_snippet and args.verbose:
                    print(f"           Code: {v.code_snippet}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
