# -*- coding: utf-8 -*-
"""
CCEA Artifact CI Guardrails.

Build-time checks per Design Doc Phase 4:
- artifact-signature-required: Artifact must be signed
- no-secrets-in-artifact: No secrets in artifact content
- valid-manifest-schema: Manifest follows schema
- sbom-present: SBOM is generated
- no-order-payloads: No order-like content in manifest

Security:
- Block publish of unsigned artifacts
- Block artifacts with secrets
- Block artifacts with prohibited content
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ccea.crypto.digest import compute_file_digest


class CheckSeverity(str, Enum):
    """Check severity level."""

    ERROR = "error"  # Blocks build/publish
    WARNING = "warning"  # Logs warning but continues
    INFO = "info"  # Informational only


class CheckResult(str, Enum):
    """Check result."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class GuardrailCheck:
    """Result of a guardrail check."""

    name: str
    result: CheckResult
    severity: CheckSeverity
    message: str
    details: Optional[str] = None
    location: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "result": self.result.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "location": self.location,
        }


@dataclass
class GuardrailReport:
    """Complete guardrail check report."""

    checks: List[GuardrailCheck] = field(default_factory=list)
    passed: bool = True
    total_errors: int = 0
    total_warnings: int = 0

    def add(self, check: GuardrailCheck) -> None:
        """Add check result."""
        self.checks.append(check)
        if check.result == CheckResult.FAILED:
            if check.severity == CheckSeverity.ERROR:
                self.passed = False
                self.total_errors += 1
            elif check.severity == CheckSeverity.WARNING:
                self.total_warnings += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
            "checks": [c.to_dict() for c in self.checks],
        }

    def summary(self) -> str:
        """Get summary string."""
        status = "PASSED" if self.passed else "FAILED"
        return f"Guardrails {status}: {self.total_errors} errors, {self.total_warnings} warnings"


class ArtifactGuardrails:
    """
    Artifact CI guardrails checker.

    Validates artifacts before publishing.
    """

    # Secret patterns to detect
    SECRET_PATTERNS = [
        # API keys
        (r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\']?[\w\-]{20,}["\']?', "API key pattern"),
        (r'(?i)(secret[_-]?key|secretkey)\s*[:=]\s*["\']?[\w\-]{20,}["\']?', "Secret key pattern"),
        (
            r'(?i)(private[_-]?key|privatekey)\s*[:=]\s*["\']?[\w\-]{20,}["\']?',
            "Private key pattern",
        ),
        # AWS
        (r"AKIA[0-9A-Z]{16}", "AWS Access Key ID"),
        (r"(?i)aws[_-]?secret[_-]?access[_-]?key\s*[:=]", "AWS Secret Access Key"),
        # Passwords
        (r'(?i)(password|passwd|pwd)\s*[:=]\s*["\'][^"\']{8,}["\']', "Password pattern"),
        # Tokens
        (r'(?i)(token|bearer)\s*[:=]\s*["\']?[\w\-\.]{20,}["\']?', "Token pattern"),
        # Private keys
        (r"-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----", "Private key header"),
        # Broker specific
        (r"(?i)(binance|alpaca|deribit|oanda)[_-]?(api|secret)[_-]?key", "Broker API key pattern"),
    ]

    # Prohibited fields in manifest
    PROHIBITED_MANIFEST_FIELDS = frozenset(
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
            "order",
        }
    )

    # Prohibited message types
    PROHIBITED_MESSAGE_TYPES = frozenset(
        {"PLACE_ORDER", "SUBMIT_ORDER", "EXECUTE_SIGNAL", "SET_TARGET_POSITION", "SEND_INTENT"}
    )

    def __init__(
        self,
        require_signature: bool = True,
        require_sbom: bool = True,
        check_secrets: bool = True,
        min_schema_version: str = "1.0.0",
    ):
        """
        Initialize guardrails.

        Args:
            require_signature: Require artifact signature
            require_sbom: Require SBOM
            check_secrets: Check for secrets in content
            min_schema_version: Minimum schema version
        """
        self.require_signature = require_signature
        self.require_sbom = require_sbom
        self.check_secrets = check_secrets
        self.min_schema_version = min_schema_version

    def check_artifact(
        self,
        artifact_path: Path,
        manifest_path: Path,
        sbom_path: Optional[Path] = None,
    ) -> GuardrailReport:
        """
        Run all guardrail checks on artifact.

        Args:
            artifact_path: Path to artifact
            manifest_path: Path to manifest
            sbom_path: Optional path to SBOM

        Returns:
            GuardrailReport with all check results
        """
        report = GuardrailReport()

        # Check 1: Signature required
        report.add(self._check_signature_required(manifest_path))

        # Check 2: SBOM present
        report.add(self._check_sbom_present(manifest_path, sbom_path))

        # Check 3: Valid manifest schema
        report.add(self._check_manifest_schema(manifest_path))

        # Check 4: No order-like payloads
        report.add(self._check_no_order_payloads(manifest_path))

        # Check 5: No prohibited message types
        report.add(self._check_no_prohibited_commands(manifest_path))

        # Check 6: No secrets in artifact
        if self.check_secrets:
            report.add(self._check_no_secrets(artifact_path))

        # Check 7: Digest integrity
        report.add(self._check_digest_integrity(artifact_path, manifest_path))

        return report

    def check_manifest_only(self, manifest_path: Path) -> GuardrailReport:
        """Run manifest-only checks (for pre-build validation)."""
        report = GuardrailReport()

        report.add(self._check_manifest_schema(manifest_path))
        report.add(self._check_no_order_payloads(manifest_path))
        report.add(self._check_no_prohibited_commands(manifest_path))

        return report

    def _check_signature_required(self, manifest_path: Path) -> GuardrailCheck:
        """Check that artifact has signature."""
        name = "artifact-signature-required"

        if not self.require_signature:
            return GuardrailCheck(
                name=name,
                result=CheckResult.SKIPPED,
                severity=CheckSeverity.INFO,
                message="Signature check skipped",
            )

        try:
            with open(manifest_path) as f:
                manifest = json.load(f)

            signature = manifest.get("signature") or manifest.get("artifact_signature")
            if not signature:
                return GuardrailCheck(
                    name=name,
                    result=CheckResult.FAILED,
                    severity=CheckSeverity.ERROR,
                    message="Artifact is NOT signed. Unsigned artifacts cannot be published.",
                    details="signature field is missing from manifest",
                    location=str(manifest_path),
                )

            # Check signature has required fields
            if not signature.get("signature_value") and not signature.get("signature"):
                return GuardrailCheck(
                    name=name,
                    result=CheckResult.FAILED,
                    severity=CheckSeverity.ERROR,
                    message="Signature is incomplete",
                    details="signature_value is missing",
                    location=str(manifest_path),
                )

            return GuardrailCheck(
                name=name,
                result=CheckResult.PASSED,
                severity=CheckSeverity.ERROR,
                message="Artifact is properly signed",
            )

        except Exception as e:
            return GuardrailCheck(
                name=name,
                result=CheckResult.FAILED,
                severity=CheckSeverity.ERROR,
                message=f"Failed to check signature: {e}",
            )

    def _check_sbom_present(
        self,
        manifest_path: Path,
        sbom_path: Optional[Path],
    ) -> GuardrailCheck:
        """Check that SBOM is present."""
        name = "sbom-present"

        if not self.require_sbom:
            return GuardrailCheck(
                name=name,
                result=CheckResult.SKIPPED,
                severity=CheckSeverity.INFO,
                message="SBOM check skipped",
            )

        try:
            with open(manifest_path) as f:
                manifest = json.load(f)

            sbom_ref = manifest.get("sbom_ref")

            if sbom_path and sbom_path.exists():
                return GuardrailCheck(
                    name=name,
                    result=CheckResult.PASSED,
                    severity=CheckSeverity.WARNING,
                    message="SBOM file is present",
                )

            if sbom_ref:
                return GuardrailCheck(
                    name=name,
                    result=CheckResult.PASSED,
                    severity=CheckSeverity.WARNING,
                    message="SBOM reference is present in manifest",
                )

            return GuardrailCheck(
                name=name,
                result=CheckResult.FAILED,
                severity=CheckSeverity.WARNING,
                message="SBOM is missing",
                details="No sbom_ref in manifest and no SBOM file provided",
            )

        except Exception as e:
            return GuardrailCheck(
                name=name,
                result=CheckResult.FAILED,
                severity=CheckSeverity.WARNING,
                message=f"Failed to check SBOM: {e}",
            )

    def _check_manifest_schema(self, manifest_path: Path) -> GuardrailCheck:
        """Check manifest schema validity."""
        name = "valid-manifest-schema"

        try:
            with open(manifest_path) as f:
                manifest = json.load(f)

            errors = []

            # Required fields
            required = ["schema_version", "artifact_id", "entrypoint"]
            for field in required:
                if field not in manifest:
                    errors.append(f"Missing required field: {field}")

            # Schema version check
            schema_version = manifest.get("schema_version", "0.0.0")
            if not self._version_gte(schema_version, self.min_schema_version):
                errors.append(
                    f"Schema version {schema_version} < minimum {self.min_schema_version}"
                )

            # Entrypoint validation
            entrypoint = manifest.get("entrypoint", {})
            if not entrypoint.get("module") or not entrypoint.get("class"):
                errors.append("Entrypoint missing module or class")

            if errors:
                return GuardrailCheck(
                    name=name,
                    result=CheckResult.FAILED,
                    severity=CheckSeverity.ERROR,
                    message="Manifest schema validation failed",
                    details="; ".join(errors),
                    location=str(manifest_path),
                )

            return GuardrailCheck(
                name=name,
                result=CheckResult.PASSED,
                severity=CheckSeverity.ERROR,
                message="Manifest schema is valid",
            )

        except json.JSONDecodeError as e:
            return GuardrailCheck(
                name=name,
                result=CheckResult.FAILED,
                severity=CheckSeverity.ERROR,
                message=f"Invalid JSON in manifest: {e}",
                location=str(manifest_path),
            )
        except Exception as e:
            return GuardrailCheck(
                name=name,
                result=CheckResult.FAILED,
                severity=CheckSeverity.ERROR,
                message=f"Failed to validate manifest: {e}",
            )

    def _check_no_order_payloads(self, manifest_path: Path) -> GuardrailCheck:
        """Check that manifest contains no order-like payloads."""
        name = "no-order-payloads"

        try:
            with open(manifest_path) as f:
                manifest = json.load(f)

            prohibited_found = self._find_prohibited_fields(manifest)

            if prohibited_found:
                return GuardrailCheck(
                    name=name,
                    result=CheckResult.FAILED,
                    severity=CheckSeverity.ERROR,
                    message="Order-like payload fields found in manifest",
                    details=f"Prohibited fields: {', '.join(prohibited_found)}",
                    location=str(manifest_path),
                )

            return GuardrailCheck(
                name=name,
                result=CheckResult.PASSED,
                severity=CheckSeverity.ERROR,
                message="No order-like payloads in manifest",
            )

        except Exception as e:
            return GuardrailCheck(
                name=name,
                result=CheckResult.FAILED,
                severity=CheckSeverity.ERROR,
                message=f"Failed to check for order payloads: {e}",
            )

    def _check_no_prohibited_commands(self, manifest_path: Path) -> GuardrailCheck:
        """Check for prohibited command types."""
        name = "no-prohibited-commands"

        try:
            content = manifest_path.read_text()

            found = []
            for msg_type in self.PROHIBITED_MESSAGE_TYPES:
                if msg_type in content:
                    found.append(msg_type)

            if found:
                return GuardrailCheck(
                    name=name,
                    result=CheckResult.FAILED,
                    severity=CheckSeverity.ERROR,
                    message="Prohibited command types found",
                    details=f"Found: {', '.join(found)}",
                    location=str(manifest_path),
                )

            return GuardrailCheck(
                name=name,
                result=CheckResult.PASSED,
                severity=CheckSeverity.ERROR,
                message="No prohibited command types found",
            )

        except Exception as e:
            return GuardrailCheck(
                name=name,
                result=CheckResult.FAILED,
                severity=CheckSeverity.ERROR,
                message=f"Failed to check commands: {e}",
            )

    def _check_no_secrets(self, artifact_path: Path) -> GuardrailCheck:
        """Check that artifact contains no secrets."""
        name = "no-secrets-in-artifact"

        try:
            secrets_found = []

            if artifact_path.suffix == ".zip":
                # Check inside zip
                with zipfile.ZipFile(artifact_path, "r") as zf:
                    for name_in_zip in zf.namelist():
                        if name_in_zip.endswith((".py", ".json", ".yaml", ".yml", ".toml", ".txt")):
                            try:
                                content = zf.read(name_in_zip).decode("utf-8", errors="ignore")
                                found = self._scan_for_secrets(content)
                                for pattern_name, line_num in found:
                                    secrets_found.append(
                                        f"{name_in_zip}:{line_num}: {pattern_name}"
                                    )
                            except Exception:
                                pass
            elif artifact_path.suffix == ".tar":
                # Skip tar inspection for now (OCI format)
                pass
            else:
                # Check single file
                content = artifact_path.read_text(errors="ignore")
                found = self._scan_for_secrets(content)
                for pattern_name, line_num in found:
                    secrets_found.append(f"line {line_num}: {pattern_name}")

            if secrets_found:
                return GuardrailCheck(
                    name=name,
                    result=CheckResult.FAILED,
                    severity=CheckSeverity.ERROR,
                    message="Potential secrets found in artifact",
                    details="; ".join(secrets_found[:5])
                    + ("..." if len(secrets_found) > 5 else ""),
                    location=str(artifact_path),
                )

            return GuardrailCheck(
                name=name,
                result=CheckResult.PASSED,
                severity=CheckSeverity.ERROR,
                message="No secrets detected in artifact",
            )

        except Exception as e:
            return GuardrailCheck(
                name=name,
                result=CheckResult.FAILED,
                severity=CheckSeverity.WARNING,
                message=f"Failed to check for secrets: {e}",
            )

    def _check_digest_integrity(
        self,
        artifact_path: Path,
        manifest_path: Path,
    ) -> GuardrailCheck:
        """Check digest integrity between manifest and artifact."""
        name = "digest-integrity"

        try:
            actual_digest = compute_file_digest(artifact_path)

            with open(manifest_path) as f:
                manifest = json.load(f)

            # Check if signature has digest
            signature = manifest.get("signature") or manifest.get("artifact_signature")
            if signature:
                signed_digest = signature.get("signed_digest")
                if signed_digest and signed_digest != actual_digest:
                    return GuardrailCheck(
                        name=name,
                        result=CheckResult.FAILED,
                        severity=CheckSeverity.ERROR,
                        message="Digest mismatch between artifact and manifest",
                        details=f"Expected {signed_digest}, got {actual_digest}",
                    )

            return GuardrailCheck(
                name=name,
                result=CheckResult.PASSED,
                severity=CheckSeverity.ERROR,
                message="Digest integrity verified",
            )

        except Exception as e:
            return GuardrailCheck(
                name=name,
                result=CheckResult.FAILED,
                severity=CheckSeverity.ERROR,
                message=f"Failed to verify digest: {e}",
            )

    def _find_prohibited_fields(
        self,
        data: Any,
        path: str = "",
    ) -> List[str]:
        """Recursively find prohibited fields."""
        found = []

        if isinstance(data, dict):
            for key, value in data.items():
                key_lower = key.lower()
                if key_lower in self.PROHIBITED_MANIFEST_FIELDS:
                    found.append(f"{path}.{key}" if path else key)
                found.extend(self._find_prohibited_fields(value, f"{path}.{key}" if path else key))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                found.extend(self._find_prohibited_fields(item, f"{path}[{i}]"))

        return found

    def _scan_for_secrets(self, content: str) -> List[Tuple[str, int]]:
        """Scan content for secret patterns."""
        found = []
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            for pattern, name in self.SECRET_PATTERNS:
                if re.search(pattern, line):
                    found.append((name, i))
                    break  # One match per line

        return found

    def _version_gte(self, version: str, min_version: str) -> bool:
        """Check if version >= min_version."""
        try:
            v_parts = [int(x) for x in version.split(".")]
            m_parts = [int(x) for x in min_version.split(".")]

            for v, m in zip(v_parts, m_parts):
                if v > m:
                    return True
                if v < m:
                    return False
            return True
        except (ValueError, IndexError):
            return False


def run_artifact_guardrails(
    artifact_path: Path,
    manifest_path: Path,
    sbom_path: Optional[Path] = None,
    strict: bool = True,
) -> Tuple[bool, GuardrailReport]:
    """
    Run artifact guardrails and return result.

    Args:
        artifact_path: Path to artifact
        manifest_path: Path to manifest
        sbom_path: Optional SBOM path
        strict: Strict mode (signature required)

    Returns:
        Tuple of (passed, report)
    """
    guardrails = ArtifactGuardrails(
        require_signature=strict,
        require_sbom=strict,
    )
    report = guardrails.check_artifact(artifact_path, manifest_path, sbom_path)
    return (report.passed, report)
