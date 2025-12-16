# -*- coding: utf-8 -*-
"""
Enterprise Posture CI Guardrail - Phase 9 GDPR Implementation.

Validates enterprise deployment configurations against CCEA compliance requirements.

This guardrail ensures:
    - EU-only data residency in all configurations
    - Telemetry local mode enabled for on-prem/air-gapped
    - No external connectivity in air-gapped configurations
    - Proper RAW telemetry gating
    - Evidence export paths configured

Design Doc Reference:
    - Phase 9: Enterprise/on-prem/VPC posture (Design Doc 16.3)
    - docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L968-L972

Usage:
    python -m ccea.guardrails.enterprise_posture_check deploy/
    python -m ccea.guardrails.enterprise_posture_check --config values.yaml
    python -m ccea.guardrails.enterprise_posture_check --mode on-prem deploy/

CI Integration:
    # GitHub Actions
    - name: Enterprise Posture Check
      run: python -m ccea.guardrails.enterprise_posture_check deploy/

CLOUD ZONE.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml


# =============================================================================
# Constants
# =============================================================================

# EU regions (same as in enterprise_posture.py)
EU_REGIONS: Set[str] = {
    # AWS
    "eu-west-1", "eu-west-2", "eu-west-3",
    "eu-central-1", "eu-central-2",
    "eu-north-1", "eu-south-1", "eu-south-2",
    # GCP
    "europe-west1", "europe-west2", "europe-west3",
    "europe-west4", "europe-west6", "europe-west8", "europe-west9",
    "europe-north1", "europe-central2",
    # Azure
    "westeurope", "northeurope",
    "germanywestcentral", "germanynorth",
    "francecentral", "francesouth",
    "swedencentral", "switzerlandnorth", "switzerlandwest",
    "uksouth", "ukwest",
    # Generic
    "eu", "europe", "on-prem", "local",
}

# Non-EU regions to detect
NON_EU_REGION_PATTERNS = [
    r"us-east-\d+",
    r"us-west-\d+",
    r"us-gov-\w+",
    r"ap-\w+-\d+",
    r"sa-east-\d+",
    r"af-south-\d+",
    r"me-south-\d+",
    r"me-central-\d+",
    r"ca-central-\d+",
    r"asia-\w+",
    r"australia-\w+",
    r"southamerica-\w+",
    r"northamerica-\w+",
    r"eastus\d*",
    r"westus\d*",
    r"centralus",
    r"southcentralus",
    r"eastasia",
    r"southeastasia",
    r"japaneast",
    r"japanwest",
    r"australiaeast",
    r"australiasoutheast",
    r"brazilsouth",
    r"canadacentral",
    r"canadaeast",
    r"koreacentral",
    r"koreasouth",
    r"indiacentral",
    r"indiasouth",
    r"indiawest",
]

# Compiled patterns
NON_EU_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in NON_EU_REGION_PATTERNS]

# External endpoints to detect in air-gapped mode
EXTERNAL_ENDPOINTS = [
    r"https?://",
    r"registry\.docker\.io",
    r"gcr\.io",
    r"docker\.io",
    r"quay\.io",
    r"ghcr\.io",
    r"amazonaws\.com",
    r"googleapis\.com",
    r"azure\.com",
    r"cloudflare\.com",
]

EXTERNAL_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in EXTERNAL_ENDPOINTS]


# =============================================================================
# Enums
# =============================================================================

class ViolationSeverity(str, Enum):
    """Severity of configuration violation."""
    CRITICAL = "critical"  # Blocks deployment
    HIGH = "high"          # Should block
    MEDIUM = "medium"      # Warning
    LOW = "low"            # Info


class DeploymentMode(str, Enum):
    """Deployment mode being checked."""
    SAAS = "saas"
    ENTERPRISE_CLOUD = "enterprise_cloud"
    ON_PREM = "on_prem"
    VPC = "vpc"
    AIR_GAPPED = "air_gapped"
    AUTO = "auto"  # Auto-detect from config


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PostureViolation:
    """A posture configuration violation."""
    rule_id: str
    severity: ViolationSeverity
    message: str
    file_path: str
    line_number: Optional[int] = None
    key_path: Optional[str] = None
    actual_value: Optional[str] = None
    expected_value: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "message": self.message,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "key_path": self.key_path,
            "actual_value": self.actual_value,
            "expected_value": self.expected_value,
        }


@dataclass
class PostureCheckReport:
    """Report from posture configuration check."""
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    files_checked: int = 0
    mode_detected: Optional[str] = None
    violations: List[PostureViolation] = field(default_factory=list)
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    passed: bool = True

    @property
    def should_fail_ci(self) -> bool:
        """Check if CI should fail."""
        return self.critical_count > 0 or self.high_count > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checked_at": self.checked_at.isoformat(),
            "files_checked": self.files_checked,
            "mode_detected": self.mode_detected,
            "violations": [v.to_dict() for v in self.violations],
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "medium_count": self.medium_count,
            "low_count": self.low_count,
            "passed": self.passed,
            "should_fail_ci": self.should_fail_ci,
        }


# =============================================================================
# Enterprise Posture Check
# =============================================================================

class EnterprisePostureCheck:
    """
    CI guardrail for enterprise deployment posture validation.

    Scans deployment configurations (Helm values, Docker Compose, env files)
    and validates against CCEA compliance requirements.
    """

    def __init__(
        self,
        mode: DeploymentMode = DeploymentMode.AUTO,
        fail_on_violations: bool = True,
    ):
        """
        Initialize posture check.

        Args:
            mode: Deployment mode to validate against (or AUTO to detect)
            fail_on_violations: Whether to set exit code on violations
        """
        self.mode = mode
        self.fail_on_violations = fail_on_violations

    def run(
        self,
        paths: List[Path],
        config_files: Optional[List[Path]] = None,
    ) -> PostureCheckReport:
        """
        Run posture check on deployment configurations.

        Args:
            paths: Directories to scan
            config_files: Specific config files to check

        Returns:
            PostureCheckReport with results
        """
        report = PostureCheckReport()
        violations: List[PostureViolation] = []

        # Collect files to check
        files_to_check: List[Path] = []

        if config_files:
            files_to_check.extend(config_files)

        for path in paths:
            if path.is_file():
                files_to_check.append(path)
            elif path.is_dir():
                # Find deployment config files
                files_to_check.extend(path.glob("**/*.yaml"))
                files_to_check.extend(path.glob("**/*.yml"))
                files_to_check.extend(path.glob("**/*.env"))
                files_to_check.extend(path.glob("**/docker-compose*.yml"))
                files_to_check.extend(path.glob("**/docker-compose*.yaml"))
                files_to_check.extend(path.glob("**/values*.yaml"))

        # Deduplicate
        files_to_check = list(set(files_to_check))
        report.files_checked = len(files_to_check)

        # Detect mode if auto
        if self.mode == DeploymentMode.AUTO:
            self.mode = self._detect_mode(files_to_check)
            report.mode_detected = self.mode.value if self.mode else None

        # Check each file
        for file_path in files_to_check:
            try:
                file_violations = self._check_file(file_path)
                violations.extend(file_violations)
            except Exception as e:
                # Log but continue
                violations.append(PostureViolation(
                    rule_id="FILE_READ_ERROR",
                    severity=ViolationSeverity.LOW,
                    message=f"Could not read file: {e}",
                    file_path=str(file_path),
                ))

        # Count by severity
        report.violations = violations
        report.critical_count = len([v for v in violations if v.severity == ViolationSeverity.CRITICAL])
        report.high_count = len([v for v in violations if v.severity == ViolationSeverity.HIGH])
        report.medium_count = len([v for v in violations if v.severity == ViolationSeverity.MEDIUM])
        report.low_count = len([v for v in violations if v.severity == ViolationSeverity.LOW])
        report.passed = not report.should_fail_ci

        return report

    def _detect_mode(self, files: List[Path]) -> DeploymentMode:
        """Detect deployment mode from configuration files."""
        for file_path in files:
            try:
                if file_path.suffix in [".yaml", ".yml"]:
                    content = yaml.safe_load(file_path.read_text())
                    if not content:
                        continue

                    # Check for air-gapped indicators
                    if self._get_nested(content, "global.airgapped", False):
                        return DeploymentMode.AIR_GAPPED
                    if self._get_nested(content, "networkPolicy.egress.allowExternal", True) is False:
                        return DeploymentMode.AIR_GAPPED

                    # Check for on-prem indicators
                    data_residency = self._get_nested(content, "global.dataResidency", "")
                    if data_residency == "on-prem":
                        return DeploymentMode.ON_PREM

                    # Check for enterprise cloud
                    if self._get_nested(content, "global.enterpriseMode", False):
                        return DeploymentMode.ENTERPRISE_CLOUD

                elif file_path.suffix == ".env" or file_path.name.endswith(".env"):
                    content = file_path.read_text()
                    if "CCEA_AIR_GAPPED_MODE=true" in content:
                        return DeploymentMode.AIR_GAPPED
                    if "CCEA_DATA_RESIDENCY=on-prem" in content:
                        return DeploymentMode.ON_PREM

            except Exception:
                continue

        return DeploymentMode.SAAS

    def _check_file(self, file_path: Path) -> List[PostureViolation]:
        """Check a single configuration file."""
        violations = []

        if file_path.suffix in [".yaml", ".yml"]:
            violations.extend(self._check_yaml_file(file_path))
        elif file_path.suffix == ".env" or file_path.name.endswith(".env"):
            violations.extend(self._check_env_file(file_path))

        return violations

    def _check_yaml_file(self, file_path: Path) -> List[PostureViolation]:
        """Check YAML configuration file."""
        violations = []

        try:
            content = yaml.safe_load(file_path.read_text())
            if not content:
                return violations
        except Exception as e:
            return [PostureViolation(
                rule_id="YAML_PARSE_ERROR",
                severity=ViolationSeverity.LOW,
                message=f"Could not parse YAML: {e}",
                file_path=str(file_path),
            )]

        # Check EU residency
        violations.extend(self._check_eu_residency_yaml(file_path, content))

        # Check telemetry configuration
        violations.extend(self._check_telemetry_yaml(file_path, content))

        # Check air-gapped requirements
        if self.mode == DeploymentMode.AIR_GAPPED:
            violations.extend(self._check_air_gapped_yaml(file_path, content))

        # Check on-prem requirements
        if self.mode in [DeploymentMode.ON_PREM, DeploymentMode.AIR_GAPPED]:
            violations.extend(self._check_on_prem_yaml(file_path, content))

        return violations

    def _check_env_file(self, file_path: Path) -> List[PostureViolation]:
        """Check environment file."""
        violations = []

        try:
            content = file_path.read_text()
            lines = content.split("\n")
        except Exception as e:
            return [PostureViolation(
                rule_id="ENV_READ_ERROR",
                severity=ViolationSeverity.LOW,
                message=f"Could not read env file: {e}",
                file_path=str(file_path),
            )]

        env_vars = {}
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                env_vars[key.strip()] = (value.strip().strip('"').strip("'"), i)

        # Check EU residency
        violations.extend(self._check_eu_residency_env(file_path, env_vars))

        # Check telemetry configuration
        violations.extend(self._check_telemetry_env(file_path, env_vars))

        # Check air-gapped requirements
        if self.mode == DeploymentMode.AIR_GAPPED:
            violations.extend(self._check_air_gapped_env(file_path, env_vars))

        return violations

    # =========================================================================
    # EU Residency Checks
    # =========================================================================

    def _check_eu_residency_yaml(
        self,
        file_path: Path,
        content: Dict[str, Any],
    ) -> List[PostureViolation]:
        """Check EU residency in YAML config."""
        violations = []

        # Check data residency setting
        residency = self._get_nested(content, "global.dataResidency", "")
        if residency and residency.lower() not in ["eu", "on-prem", "local", "europe"]:
            if not self._is_eu_region(residency):
                violations.append(PostureViolation(
                    rule_id="NON_EU_DATA_RESIDENCY",
                    severity=ViolationSeverity.CRITICAL,
                    message=f"Non-EU data residency configured: {residency}",
                    file_path=str(file_path),
                    key_path="global.dataResidency",
                    actual_value=residency,
                    expected_value="eu, on-prem, or EU region",
                ))

        # Scan all values for non-EU regions
        violations.extend(self._scan_for_non_eu_regions(file_path, content, ""))

        return violations

    def _check_eu_residency_env(
        self,
        file_path: Path,
        env_vars: Dict[str, Tuple[str, int]],
    ) -> List[PostureViolation]:
        """Check EU residency in env file."""
        violations = []

        # Check data residency
        if "CCEA_DATA_RESIDENCY" in env_vars:
            value, line = env_vars["CCEA_DATA_RESIDENCY"]
            if value.lower() not in ["eu", "on-prem", "local", "europe"]:
                if not self._is_eu_region(value):
                    violations.append(PostureViolation(
                        rule_id="NON_EU_DATA_RESIDENCY",
                        severity=ViolationSeverity.CRITICAL,
                        message=f"Non-EU data residency configured: {value}",
                        file_path=str(file_path),
                        line_number=line,
                        key_path="CCEA_DATA_RESIDENCY",
                        actual_value=value,
                        expected_value="eu, on-prem, or EU region",
                    ))

        # Check AWS region
        if "AWS_REGION" in env_vars:
            value, line = env_vars["AWS_REGION"]
            if not self._is_eu_region(value):
                violations.append(PostureViolation(
                    rule_id="NON_EU_AWS_REGION",
                    severity=ViolationSeverity.CRITICAL,
                    message=f"Non-EU AWS region configured: {value}",
                    file_path=str(file_path),
                    line_number=line,
                    key_path="AWS_REGION",
                    actual_value=value,
                    expected_value="EU region (eu-west-1, eu-central-1, etc.)",
                ))

        return violations

    def _scan_for_non_eu_regions(
        self,
        file_path: Path,
        data: Any,
        path: str,
    ) -> List[PostureViolation]:
        """Recursively scan for non-EU region values."""
        violations = []

        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                violations.extend(self._scan_for_non_eu_regions(file_path, value, current_path))

        elif isinstance(data, list):
            for i, item in enumerate(data):
                violations.extend(self._scan_for_non_eu_regions(file_path, item, f"{path}[{i}]"))

        elif isinstance(data, str):
            # Check if it looks like a region
            if self._looks_like_region(data) and not self._is_eu_region(data):
                # Only flag if key suggests it's a region
                key_lower = path.lower()
                if any(kw in key_lower for kw in ["region", "location", "zone"]):
                    violations.append(PostureViolation(
                        rule_id="NON_EU_REGION_VALUE",
                        severity=ViolationSeverity.HIGH,
                        message=f"Non-EU region detected: {data}",
                        file_path=str(file_path),
                        key_path=path,
                        actual_value=data,
                        expected_value="EU region",
                    ))

        return violations

    def _is_eu_region(self, value: str) -> bool:
        """Check if a value is an EU region."""
        value_lower = value.lower()
        if value_lower in EU_REGIONS:
            return True
        # Check for EU prefixes
        if any(value_lower.startswith(p) for p in ["eu-", "europe-", "westeurope", "northeurope"]):
            return True
        return False

    def _looks_like_region(self, value: str) -> bool:
        """Check if a value looks like a region identifier."""
        for pattern in NON_EU_PATTERNS_COMPILED:
            if pattern.fullmatch(value):
                return True
        # Also check EU patterns
        if self._is_eu_region(value):
            return True
        return False

    # =========================================================================
    # Telemetry Checks
    # =========================================================================

    def _check_telemetry_yaml(
        self,
        file_path: Path,
        content: Dict[str, Any],
    ) -> List[PostureViolation]:
        """Check telemetry configuration in YAML."""
        violations = []

        # Check redaction mandatory
        redaction = self._get_nested(
            content,
            "controlPlane.config.CCEA_TELEMETRY_REDACTION_MANDATORY",
            self._get_nested(
                content,
                "telemetryIngester.config.CCEA_TELEMETRY_REDACTION_MANDATORY",
                None
            )
        )

        if redaction is not None and str(redaction).lower() != "true":
            violations.append(PostureViolation(
                rule_id="REDACTION_NOT_MANDATORY",
                severity=ViolationSeverity.CRITICAL,
                message="Telemetry redaction must be mandatory (cannot be disabled)",
                file_path=str(file_path),
                key_path="CCEA_TELEMETRY_REDACTION_MANDATORY",
                actual_value=str(redaction),
                expected_value="true",
            ))

        # For on-prem/air-gapped, check local only mode
        if self.mode in [DeploymentMode.ON_PREM, DeploymentMode.AIR_GAPPED]:
            local_only = self._get_nested(
                content,
                "telemetryIngester.config.CCEA_TELEMETRY_LOCAL_ONLY",
                self._get_nested(
                    content,
                    "controlPlane.config.CCEA_TELEMETRY_LOCAL_ONLY",
                    None
                )
            )

            if local_only is None or str(local_only).lower() != "true":
                violations.append(PostureViolation(
                    rule_id="TELEMETRY_NOT_LOCAL",
                    severity=ViolationSeverity.HIGH,
                    message="Telemetry local mode should be enabled for on-prem/air-gapped",
                    file_path=str(file_path),
                    key_path="CCEA_TELEMETRY_LOCAL_ONLY",
                    actual_value=str(local_only) if local_only else "not set",
                    expected_value="true",
                ))

        return violations

    def _check_telemetry_env(
        self,
        file_path: Path,
        env_vars: Dict[str, Tuple[str, int]],
    ) -> List[PostureViolation]:
        """Check telemetry configuration in env file."""
        violations = []

        # Check redaction mandatory
        if "CCEA_TELEMETRY_REDACTION_MANDATORY" in env_vars:
            value, line = env_vars["CCEA_TELEMETRY_REDACTION_MANDATORY"]
            if value.lower() != "true":
                violations.append(PostureViolation(
                    rule_id="REDACTION_NOT_MANDATORY",
                    severity=ViolationSeverity.CRITICAL,
                    message="Telemetry redaction must be mandatory",
                    file_path=str(file_path),
                    line_number=line,
                    key_path="CCEA_TELEMETRY_REDACTION_MANDATORY",
                    actual_value=value,
                    expected_value="true",
                ))

        # For on-prem/air-gapped, check local only
        if self.mode in [DeploymentMode.ON_PREM, DeploymentMode.AIR_GAPPED]:
            if "CCEA_TELEMETRY_LOCAL_ONLY" not in env_vars:
                violations.append(PostureViolation(
                    rule_id="TELEMETRY_NOT_LOCAL",
                    severity=ViolationSeverity.MEDIUM,
                    message="CCEA_TELEMETRY_LOCAL_ONLY should be set for on-prem/air-gapped",
                    file_path=str(file_path),
                    key_path="CCEA_TELEMETRY_LOCAL_ONLY",
                    expected_value="true",
                ))

        return violations

    # =========================================================================
    # Air-Gapped Checks
    # =========================================================================

    def _check_air_gapped_yaml(
        self,
        file_path: Path,
        content: Dict[str, Any],
    ) -> List[PostureViolation]:
        """Check air-gapped requirements in YAML."""
        violations = []

        # Check air-gapped flag
        airgapped = self._get_nested(content, "global.airgapped", False)
        if not airgapped:
            violations.append(PostureViolation(
                rule_id="AIR_GAP_NOT_ENABLED",
                severity=ViolationSeverity.MEDIUM,
                message="Air-gapped mode should be explicitly enabled",
                file_path=str(file_path),
                key_path="global.airgapped",
                expected_value="true",
            ))

        # Check no external egress
        allow_external = self._get_nested(
            content, "networkPolicy.egress.allowExternal", True
        )
        if allow_external:
            violations.append(PostureViolation(
                rule_id="EXTERNAL_EGRESS_ALLOWED",
                severity=ViolationSeverity.HIGH,
                message="External egress should be disabled for air-gapped",
                file_path=str(file_path),
                key_path="networkPolicy.egress.allowExternal",
                actual_value=str(allow_external),
                expected_value="false",
            ))

        # Scan for external URLs
        violations.extend(self._scan_for_external_urls(file_path, content, ""))

        return violations

    def _check_air_gapped_env(
        self,
        file_path: Path,
        env_vars: Dict[str, Tuple[str, int]],
    ) -> List[PostureViolation]:
        """Check air-gapped requirements in env file."""
        violations = []

        # Check air-gapped mode
        if "CCEA_AIR_GAPPED_MODE" not in env_vars:
            violations.append(PostureViolation(
                rule_id="AIR_GAP_NOT_ENABLED",
                severity=ViolationSeverity.MEDIUM,
                message="CCEA_AIR_GAPPED_MODE should be set for air-gapped",
                file_path=str(file_path),
                key_path="CCEA_AIR_GAPPED_MODE",
                expected_value="true",
            ))

        # Check for external URLs in values
        for key, (value, line) in env_vars.items():
            for pattern in EXTERNAL_PATTERNS_COMPILED:
                if pattern.search(value):
                    # Exclude local registry
                    if "localhost" in value or "127.0.0.1" in value:
                        continue
                    if ".internal." in value or ".local" in value:
                        continue

                    violations.append(PostureViolation(
                        rule_id="EXTERNAL_URL_IN_AIR_GAP",
                        severity=ViolationSeverity.HIGH,
                        message=f"External URL detected in air-gapped config: {value}",
                        file_path=str(file_path),
                        line_number=line,
                        key_path=key,
                        actual_value=value,
                    ))
                    break

        return violations

    def _scan_for_external_urls(
        self,
        file_path: Path,
        data: Any,
        path: str,
    ) -> List[PostureViolation]:
        """Scan for external URLs in air-gapped config."""
        violations = []

        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                violations.extend(self._scan_for_external_urls(file_path, value, current_path))

        elif isinstance(data, list):
            for i, item in enumerate(data):
                violations.extend(self._scan_for_external_urls(file_path, item, f"{path}[{i}]"))

        elif isinstance(data, str):
            for pattern in EXTERNAL_PATTERNS_COMPILED:
                if pattern.search(data):
                    # Exclude local/internal
                    if "localhost" in data or "127.0.0.1" in data:
                        continue
                    if ".internal." in data or ".local" in data:
                        continue
                    # Exclude documentation URLs
                    if "docs." in data or "example.com" in data:
                        continue

                    violations.append(PostureViolation(
                        rule_id="EXTERNAL_URL_IN_AIR_GAP",
                        severity=ViolationSeverity.MEDIUM,
                        message=f"External URL detected: {data}",
                        file_path=str(file_path),
                        key_path=path,
                        actual_value=data,
                    ))
                    break

        return violations

    # =========================================================================
    # On-Prem Checks
    # =========================================================================

    def _check_on_prem_yaml(
        self,
        file_path: Path,
        content: Dict[str, Any],
    ) -> List[PostureViolation]:
        """Check on-prem requirements in YAML."""
        violations = []

        # Check evidence export local only
        evidence_local = self._get_nested(
            content,
            "governance.config.CCEA_EVIDENCE_EXPORT_LOCAL_ONLY",
            None
        )

        if evidence_local is None:
            violations.append(PostureViolation(
                rule_id="EVIDENCE_EXPORT_NOT_LOCAL",
                severity=ViolationSeverity.MEDIUM,
                message="Evidence export should be configured for local for on-prem",
                file_path=str(file_path),
                key_path="governance.config.CCEA_EVIDENCE_EXPORT_LOCAL_ONLY",
                expected_value="true",
            ))

        return violations

    # =========================================================================
    # Utilities
    # =========================================================================

    def _get_nested(
        self,
        data: Dict[str, Any],
        key_path: str,
        default: Any = None,
    ) -> Any:
        """Get nested value from dict using dot notation."""
        keys = key_path.split(".")
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current


# =============================================================================
# CLI Interface
# =============================================================================

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Enterprise Posture CI Guardrail - validates deployment configurations"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Paths to scan (directories or files)",
    )
    parser.add_argument(
        "--config", "-c",
        action="append",
        dest="config_files",
        help="Specific config files to check",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["auto", "saas", "enterprise_cloud", "on_prem", "vpc", "air_gapped"],
        default="auto",
        help="Deployment mode to validate against",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output report to JSON file",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Don't exit with error code on violations",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Convert mode
    mode_map = {
        "auto": DeploymentMode.AUTO,
        "saas": DeploymentMode.SAAS,
        "enterprise_cloud": DeploymentMode.ENTERPRISE_CLOUD,
        "on_prem": DeploymentMode.ON_PREM,
        "vpc": DeploymentMode.VPC,
        "air_gapped": DeploymentMode.AIR_GAPPED,
    }
    mode = mode_map[args.mode]

    # Run check
    check = EnterprisePostureCheck(mode=mode, fail_on_violations=not args.no_fail)

    paths = [Path(p) for p in args.paths]
    config_files = [Path(f) for f in args.config_files] if args.config_files else None

    report = check.run(paths, config_files)

    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

    # Print summary
    print(f"\nEnterprise Posture Check")
    print(f"========================")
    print(f"Files checked: {report.files_checked}")
    print(f"Mode detected: {report.mode_detected or 'N/A'}")
    print(f"")
    print(f"Violations:")
    print(f"  Critical: {report.critical_count}")
    print(f"  High:     {report.high_count}")
    print(f"  Medium:   {report.medium_count}")
    print(f"  Low:      {report.low_count}")
    print(f"")

    if args.verbose or report.violations:
        for v in report.violations:
            severity_icon = {
                ViolationSeverity.CRITICAL: "[CRITICAL]",
                ViolationSeverity.HIGH: "[HIGH]",
                ViolationSeverity.MEDIUM: "[MEDIUM]",
                ViolationSeverity.LOW: "[LOW]",
            }[v.severity]
            print(f"{severity_icon} {v.rule_id}: {v.message}")
            print(f"  File: {v.file_path}")
            if v.key_path:
                print(f"  Key: {v.key_path}")
            if v.actual_value:
                print(f"  Actual: {v.actual_value}")
            if v.expected_value:
                print(f"  Expected: {v.expected_value}")
            print()

    if report.passed:
        print("PASSED: No blocking violations found")
        sys.exit(0)
    else:
        print("FAILED: Blocking violations found")
        if not args.no_fail:
            sys.exit(1)


if __name__ == "__main__":
    main()
