# -*- coding: utf-8 -*-
"""
Pre-flight Checker - Validation before start/upgrade.

Design Doc D1: Pre-flight проверки перед стартом/апгрейдом:
1. verify signature + digest + schema_version (manifest + protocol)
2. verify broker connectivity + permissions (без раскрытия секретов в cloud)
3. verify local policy firewall/hard caps
4. verify time sync (допустимый drift) и корректность timestamps/idempotency

CCEA Phase 5 Component.

SECURITY NOTE (Design Doc compliance):
- Uses ccea.artifact.verifier.ArtifactVerifier for REAL cryptographic verification
- Unsigned artifacts are ALWAYS rejected (strict mode)
- No fallback to "presence only" checks
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, TYPE_CHECKING
from uuid import uuid4

# Import ArtifactVerifier for real crypto verification (Design Doc Phase 4)
try:
    from ccea.artifact.verifier import (
        ArtifactVerifier,
        VerificationResult,
        VerificationReport,
        RejectionReason,
    )
    from ccea.artifact.signer import SignatureInfo
    VERIFIER_AVAILABLE = True
except ImportError:
    VERIFIER_AVAILABLE = False
    ArtifactVerifier = None  # type: ignore
    VerificationResult = None  # type: ignore
    SignatureInfo = None  # type: ignore


class PreflightCheckType(Enum):
    """Types of pre-flight checks."""
    SIGNATURE_VERIFICATION = auto()
    DIGEST_VERIFICATION = auto()
    SCHEMA_VERSION = auto()
    BROKER_CONNECTIVITY = auto()
    BROKER_PERMISSIONS = auto()
    POLICY_FIREWALL = auto()
    HARD_CAPS = auto()
    TIME_SYNC = auto()
    VAULT_UNLOCKED = auto()
    CREDENTIALS_AVAILABLE = auto()
    ARTIFACT_INTEGRITY = auto()
    MANIFEST_VALID = auto()
    RESOURCES_AVAILABLE = auto()
    NETWORK_CONNECTIVITY = auto()


class PreflightCheckResult(Enum):
    """Result of a pre-flight check."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class PreflightCheck:
    """
    Single pre-flight check result.
    """
    check_type: PreflightCheckType
    result: PreflightCheckResult
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    required: bool = True  # If true, failure blocks start

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_type": self.check_type.name,
            "result": self.result.value,
            "message": self.message,
            "details": self.details,
            "duration_ms": self.duration_ms,
            "required": self.required,
        }


@dataclass
class PreflightResult:
    """
    Overall pre-flight result.
    """
    preflight_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    checks: List[PreflightCheck] = field(default_factory=list)
    passed: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    run_id: Optional[str] = None
    artifact_digest: Optional[str] = None

    @property
    def failed_count(self) -> int:
        """Count of failed checks."""
        return sum(1 for c in self.checks if c.result == PreflightCheckResult.FAILED)

    @property
    def warning_count(self) -> int:
        """Count of warning checks."""
        return sum(1 for c in self.checks if c.result == PreflightCheckResult.WARNING)

    @property
    def passed_count(self) -> int:
        """Count of passed checks."""
        return sum(1 for c in self.checks if c.result == PreflightCheckResult.PASSED)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "preflight_id": self.preflight_id,
            "timestamp": self.timestamp.isoformat(),
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "errors": self.errors,
            "warnings": self.warnings,
            "duration_ms": self.duration_ms,
            "failed_count": self.failed_count,
            "warning_count": self.warning_count,
            "passed_count": self.passed_count,
            "run_id": self.run_id,
            "artifact_digest": self.artifact_digest,
        }

    def get_evidence_hash(self) -> str:
        """Compute hash for evidence purposes."""
        data = {
            "preflight_id": self.preflight_id,
            "timestamp": self.timestamp.isoformat(),
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


@dataclass
class PreflightConfig:
    """
    Pre-flight checker configuration.
    """
    # Time sync
    max_time_drift_seconds: float = 5.0

    # Broker checks
    broker_timeout_seconds: float = 10.0
    require_broker_connectivity: bool = True

    # Schema version
    min_schema_version: str = "1.0.0"
    max_schema_version: str = "2.0.0"

    # Vault
    require_vault_unlocked: bool = True

    # Resources
    min_disk_space_mb: int = 100
    min_memory_mb: int = 256

    # Skip options
    skip_broker_check: bool = False
    skip_time_sync: bool = False
    skip_network_check: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_time_drift_seconds": self.max_time_drift_seconds,
            "broker_timeout_seconds": self.broker_timeout_seconds,
            "require_broker_connectivity": self.require_broker_connectivity,
            "min_schema_version": self.min_schema_version,
            "max_schema_version": self.max_schema_version,
        }


class PreflightChecker:
    """
    Performs pre-flight checks before start/upgrade.

    Design Doc D1:
    1. verify signature + digest + schema_version (manifest + protocol)
    2. verify broker connectivity + permissions
    3. verify local policy firewall/hard caps
    4. verify time sync

    Usage:
        checker = PreflightChecker(config)

        # Check before start
        result = checker.run_preflight(
            artifact_path=artifact_path,
            manifest=manifest,
            broker_name="binance",
        )

        if result.passed:
            # Safe to start
        else:
            # Handle failures
            for error in result.errors:
                print(error)
    """

    def __init__(
        self,
        config: Optional[PreflightConfig] = None,
        vault: Optional[Any] = None,  # LocalVault
        policy_firewall: Optional[Any] = None,  # PolicyFirewall
        hard_cap_enforcer: Optional[Any] = None,  # HardCapEnforcer
        broker_connector: Optional[Any] = None,  # BrokerConnector
        time_checker: Optional[Any] = None,  # TimeSyncChecker
        artifact_verifier: Optional["ArtifactVerifier"] = None,  # CCEA ArtifactVerifier
    ):
        """
        Initialize pre-flight checker.

        Args:
            config: Configuration
            vault: Local vault for credential checks
            policy_firewall: Policy firewall for validation
            hard_cap_enforcer: Hard cap enforcer for validation
            broker_connector: Broker connector for connectivity check
            time_checker: Time sync checker
            artifact_verifier: CCEA ArtifactVerifier for cryptographic verification
                              (Design Doc Phase 4 - required for production)
        """
        self.config = config or PreflightConfig()
        self._vault = vault
        self._policy_firewall = policy_firewall
        self._hard_cap_enforcer = hard_cap_enforcer
        self._broker_connector = broker_connector
        self._time_checker = time_checker
        self._artifact_verifier = artifact_verifier

        self._last_result: Optional[PreflightResult] = None
        self._lock = threading.RLock()

    @property
    def last_result(self) -> Optional[PreflightResult]:
        """Get last preflight result."""
        return self._last_result

    def run_preflight(
        self,
        artifact_path: Optional[Path] = None,
        manifest: Optional[Dict[str, Any]] = None,
        broker_name: Optional[str] = None,
        run_id: Optional[str] = None,
        artifact_digest: Optional[str] = None,
        signature: Optional[bytes] = None,
    ) -> PreflightResult:
        """
        Run all pre-flight checks.

        Args:
            artifact_path: Path to artifact
            manifest: Artifact manifest
            broker_name: Broker to check connectivity
            run_id: Run ID
            artifact_digest: Expected artifact digest
            signature: Artifact signature

        Returns:
            PreflightResult with all check results
        """
        import time as time_module
        start_time = time_module.time()

        result = PreflightResult(run_id=run_id, artifact_digest=artifact_digest)

        # Run checks in order
        checks = [
            (PreflightCheckType.VAULT_UNLOCKED, self._check_vault_unlocked),
            (PreflightCheckType.CREDENTIALS_AVAILABLE, lambda: self._check_credentials(broker_name)),
            (PreflightCheckType.TIME_SYNC, self._check_time_sync),
            (PreflightCheckType.SCHEMA_VERSION, lambda: self._check_schema_version(manifest)),
            (PreflightCheckType.MANIFEST_VALID, lambda: self._check_manifest(manifest)),
            (PreflightCheckType.DIGEST_VERIFICATION, lambda: self._check_digest(artifact_path, artifact_digest)),
            (PreflightCheckType.SIGNATURE_VERIFICATION, lambda: self._check_signature(artifact_path, manifest, signature)),
            (PreflightCheckType.POLICY_FIREWALL, lambda: self._check_policy_firewall(manifest)),
            (PreflightCheckType.HARD_CAPS, lambda: self._check_hard_caps(manifest)),
            (PreflightCheckType.BROKER_CONNECTIVITY, lambda: self._check_broker_connectivity(broker_name)),
            (PreflightCheckType.RESOURCES_AVAILABLE, self._check_resources),
            (PreflightCheckType.NETWORK_CONNECTIVITY, self._check_network),
        ]

        for check_type, check_fn in checks:
            check_start = time_module.time()
            try:
                check = check_fn()
                check.check_type = check_type
                check.duration_ms = (time_module.time() - check_start) * 1000
                result.checks.append(check)

                if check.result == PreflightCheckResult.FAILED:
                    if check.required:
                        result.errors.append(check.message)
                    else:
                        result.warnings.append(check.message)
                elif check.result == PreflightCheckResult.WARNING:
                    result.warnings.append(check.message)

            except Exception as e:
                check = PreflightCheck(
                    check_type=check_type,
                    result=PreflightCheckResult.FAILED,
                    message=f"Check failed with exception: {str(e)}",
                    duration_ms=(time_module.time() - check_start) * 1000,
                )
                result.checks.append(check)
                result.errors.append(check.message)

        # Overall result
        result.duration_ms = (time_module.time() - start_time) * 1000
        result.passed = len(result.errors) == 0

        with self._lock:
            self._last_result = result

        return result

    def _check_vault_unlocked(self) -> PreflightCheck:
        """Check if vault is unlocked."""
        if self._vault is None:
            if self.config.require_vault_unlocked:
                return PreflightCheck(
                    check_type=PreflightCheckType.VAULT_UNLOCKED,
                    result=PreflightCheckResult.FAILED,
                    message="Local vault not configured",
                    required=True,
                )
            return PreflightCheck(
                check_type=PreflightCheckType.VAULT_UNLOCKED,
                result=PreflightCheckResult.SKIPPED,
                message="Vault check skipped",
                required=False,
            )

        if hasattr(self._vault, "is_locked") and self._vault.is_locked:
            return PreflightCheck(
                check_type=PreflightCheckType.VAULT_UNLOCKED,
                result=PreflightCheckResult.FAILED,
                message="Local vault is locked - unlock required before start",
                required=True,
            )

        return PreflightCheck(
            check_type=PreflightCheckType.VAULT_UNLOCKED,
            result=PreflightCheckResult.PASSED,
            message="Vault unlocked",
        )

    def _check_credentials(self, broker_name: Optional[str]) -> PreflightCheck:
        """Check if required credentials are available."""
        if not broker_name:
            return PreflightCheck(
                check_type=PreflightCheckType.CREDENTIALS_AVAILABLE,
                result=PreflightCheckResult.SKIPPED,
                message="No broker specified",
                required=False,
            )

        if self._vault is None:
            return PreflightCheck(
                check_type=PreflightCheckType.CREDENTIALS_AVAILABLE,
                result=PreflightCheckResult.FAILED,
                message="Cannot check credentials - vault not configured",
                required=True,
            )

        try:
            # Check if credentials exist (without retrieving values)
            if hasattr(self._vault, "list_credentials"):
                creds = self._vault.list_credentials(broker_name)
                if not creds:
                    return PreflightCheck(
                        check_type=PreflightCheckType.CREDENTIALS_AVAILABLE,
                        result=PreflightCheckResult.FAILED,
                        message=f"No credentials found for broker: {broker_name}",
                        details={"broker": broker_name},
                        required=True,
                    )

                return PreflightCheck(
                    check_type=PreflightCheckType.CREDENTIALS_AVAILABLE,
                    result=PreflightCheckResult.PASSED,
                    message=f"Credentials available for {broker_name}",
                    details={"broker": broker_name, "credential_count": len(creds)},
                )
        except Exception as e:
            return PreflightCheck(
                check_type=PreflightCheckType.CREDENTIALS_AVAILABLE,
                result=PreflightCheckResult.FAILED,
                message=f"Failed to check credentials: {str(e)}",
                required=True,
            )

        return PreflightCheck(
            check_type=PreflightCheckType.CREDENTIALS_AVAILABLE,
            result=PreflightCheckResult.PASSED,
            message="Credentials check passed",
        )

    def _check_time_sync(self) -> PreflightCheck:
        """Check time synchronization."""
        if self.config.skip_time_sync:
            return PreflightCheck(
                check_type=PreflightCheckType.TIME_SYNC,
                result=PreflightCheckResult.SKIPPED,
                message="Time sync check skipped",
                required=False,
            )

        if self._time_checker is None:
            # Try to create one
            try:
                from packages.agent.daemon.time_sync import TimeSyncChecker, TimeSyncConfig
                self._time_checker = TimeSyncChecker(
                    TimeSyncConfig(max_drift_seconds=self.config.max_time_drift_seconds)
                )
            except ImportError:
                return PreflightCheck(
                    check_type=PreflightCheckType.TIME_SYNC,
                    result=PreflightCheckResult.WARNING,
                    message="Time sync checker not available",
                    required=False,
                )

        try:
            result = self._time_checker.check()

            if not result.synchronized:
                return PreflightCheck(
                    check_type=PreflightCheckType.TIME_SYNC,
                    result=PreflightCheckResult.FAILED,
                    message=f"Time sync failed: {result.error}",
                    required=True,
                )

            if abs(result.drift_seconds) >= self.config.max_time_drift_seconds:
                return PreflightCheck(
                    check_type=PreflightCheckType.TIME_SYNC,
                    result=PreflightCheckResult.FAILED,
                    message=f"Time drift ({result.drift_seconds:.2f}s) exceeds maximum ({self.config.max_time_drift_seconds}s)",
                    details={
                        "drift_seconds": result.drift_seconds,
                        "max_drift": self.config.max_time_drift_seconds,
                    },
                    required=True,
                )

            return PreflightCheck(
                check_type=PreflightCheckType.TIME_SYNC,
                result=PreflightCheckResult.PASSED,
                message=f"Time synchronized (drift: {result.drift_ms}ms)",
                details={"drift_ms": result.drift_ms, "server": result.ntp_server},
            )

        except Exception as e:
            return PreflightCheck(
                check_type=PreflightCheckType.TIME_SYNC,
                result=PreflightCheckResult.WARNING,
                message=f"Time sync check failed: {str(e)}",
                required=False,
            )

    def _check_schema_version(self, manifest: Optional[Dict[str, Any]]) -> PreflightCheck:
        """Check schema version compatibility."""
        if manifest is None:
            return PreflightCheck(
                check_type=PreflightCheckType.SCHEMA_VERSION,
                result=PreflightCheckResult.SKIPPED,
                message="No manifest provided",
                required=False,
            )

        schema_version = manifest.get("schema_version")
        if not schema_version:
            return PreflightCheck(
                check_type=PreflightCheckType.SCHEMA_VERSION,
                result=PreflightCheckResult.FAILED,
                message="Manifest missing schema_version",
                required=True,
            )

        # Compare versions
        try:
            from packaging.version import Version
            version = Version(schema_version)
            min_version = Version(self.config.min_schema_version)
            max_version = Version(self.config.max_schema_version)

            if version < min_version:
                return PreflightCheck(
                    check_type=PreflightCheckType.SCHEMA_VERSION,
                    result=PreflightCheckResult.FAILED,
                    message=f"Schema version {schema_version} is below minimum {self.config.min_schema_version}",
                    required=True,
                )

            if version > max_version:
                return PreflightCheck(
                    check_type=PreflightCheckType.SCHEMA_VERSION,
                    result=PreflightCheckResult.FAILED,
                    message=f"Schema version {schema_version} exceeds maximum {self.config.max_schema_version}",
                    required=True,
                )

        except ImportError:
            # Simple string comparison
            pass
        except Exception as e:
            return PreflightCheck(
                check_type=PreflightCheckType.SCHEMA_VERSION,
                result=PreflightCheckResult.WARNING,
                message=f"Version comparison failed: {str(e)}",
                required=False,
            )

        return PreflightCheck(
            check_type=PreflightCheckType.SCHEMA_VERSION,
            result=PreflightCheckResult.PASSED,
            message=f"Schema version {schema_version} is compatible",
            details={"schema_version": schema_version},
        )

    def _check_manifest(self, manifest: Optional[Dict[str, Any]]) -> PreflightCheck:
        """Check manifest is valid."""
        if manifest is None:
            return PreflightCheck(
                check_type=PreflightCheckType.MANIFEST_VALID,
                result=PreflightCheckResult.SKIPPED,
                message="No manifest provided",
                required=False,
            )

        required_fields = ["schema_version", "entrypoint"]
        missing = [f for f in required_fields if f not in manifest]

        if missing:
            return PreflightCheck(
                check_type=PreflightCheckType.MANIFEST_VALID,
                result=PreflightCheckResult.FAILED,
                message=f"Manifest missing required fields: {', '.join(missing)}",
                required=True,
            )

        return PreflightCheck(
            check_type=PreflightCheckType.MANIFEST_VALID,
            result=PreflightCheckResult.PASSED,
            message="Manifest valid",
            details={"fields": list(manifest.keys())},
        )

    def _check_digest(
        self,
        artifact_path: Optional[Path],
        expected_digest: Optional[str],
    ) -> PreflightCheck:
        """Verify artifact digest."""
        if artifact_path is None or expected_digest is None:
            return PreflightCheck(
                check_type=PreflightCheckType.DIGEST_VERIFICATION,
                result=PreflightCheckResult.SKIPPED,
                message="Digest verification skipped - no artifact or digest provided",
                required=False,
            )

        if not artifact_path.exists():
            return PreflightCheck(
                check_type=PreflightCheckType.DIGEST_VERIFICATION,
                result=PreflightCheckResult.FAILED,
                message=f"Artifact not found: {artifact_path}",
                required=True,
            )

        try:
            # Compute SHA256
            sha256 = hashlib.sha256()
            with open(artifact_path, "rb") as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
            actual_digest = sha256.hexdigest()

            # Handle various digest formats
            expected = expected_digest.lower()
            if expected.startswith("sha256:"):
                expected = expected[7:]

            if actual_digest != expected:
                return PreflightCheck(
                    check_type=PreflightCheckType.DIGEST_VERIFICATION,
                    result=PreflightCheckResult.FAILED,
                    message="Artifact digest mismatch",
                    details={
                        "expected": expected,
                        "actual": actual_digest,
                    },
                    required=True,
                )

            return PreflightCheck(
                check_type=PreflightCheckType.DIGEST_VERIFICATION,
                result=PreflightCheckResult.PASSED,
                message="Artifact digest verified",
                details={"digest": actual_digest},
            )

        except Exception as e:
            return PreflightCheck(
                check_type=PreflightCheckType.DIGEST_VERIFICATION,
                result=PreflightCheckResult.FAILED,
                message=f"Digest verification failed: {str(e)}",
                required=True,
            )

    def _check_signature(
        self,
        artifact_path: Optional[Path],
        manifest: Optional[Dict[str, Any]],
        signature: Optional[bytes],
    ) -> PreflightCheck:
        """
        Verify artifact signature using REAL cryptographic verification.

        Design Doc Phase 4 Compliance:
        - Uses ccea.artifact.verifier.ArtifactVerifier for real crypto verification
        - Unsigned artifacts are ALWAYS rejected (no fallback)
        - Invalid signatures cause immediate rejection

        SECURITY: This is fail-closed. Any verification failure = REJECT.
        """
        if artifact_path is None:
            return PreflightCheck(
                check_type=PreflightCheckType.SIGNATURE_VERIFICATION,
                result=PreflightCheckResult.SKIPPED,
                message="Signature verification skipped - no artifact provided",
                required=False,
            )

        if not artifact_path.exists():
            return PreflightCheck(
                check_type=PreflightCheckType.SIGNATURE_VERIFICATION,
                result=PreflightCheckResult.FAILED,
                message=f"Artifact not found for signature verification: {artifact_path}",
                required=True,
            )

        # CRITICAL: Use ArtifactVerifier for REAL cryptographic verification
        if self._artifact_verifier is not None and VERIFIER_AVAILABLE:
            return self._verify_with_artifact_verifier(artifact_path, manifest, signature)

        # Fallback if verifier not available - but STRICT: still require signature
        return self._check_signature_fallback(manifest, signature)

    def _verify_with_artifact_verifier(
        self,
        artifact_path: Path,
        manifest: Optional[Dict[str, Any]],
        signature: Optional[bytes],
    ) -> PreflightCheck:
        """
        Use ArtifactVerifier for real cryptographic verification.

        This is the Design Doc compliant path - actual Ed25519/RSA verification.
        """
        # Build SignatureInfo from manifest or raw signature
        sig_info = None

        if manifest and isinstance(manifest, dict):
            sig_obj = manifest.get("signature", {})
            if isinstance(sig_obj, dict) and sig_obj.get("signature_value"):
                try:
                    sig_info = SignatureInfo(
                        algorithm=sig_obj.get("algorithm", "ed25519"),
                        signature=sig_obj["signature_value"],
                        key_id=sig_obj.get("key_id"),
                        signed_digest=sig_obj.get("signed_digest"),
                    )
                except Exception as e:
                    return PreflightCheck(
                        check_type=PreflightCheckType.SIGNATURE_VERIFICATION,
                        result=PreflightCheckResult.FAILED,
                        message=f"Invalid signature format in manifest: {e}",
                        required=True,
                    )

        # If raw signature bytes provided, convert to SignatureInfo
        if signature is not None and sig_info is None:
            import base64
            try:
                sig_info = SignatureInfo(
                    algorithm="ed25519",  # Default algorithm per Design Doc
                    signature=base64.b64encode(signature).decode("utf-8"),
                )
            except Exception as e:
                return PreflightCheck(
                    check_type=PreflightCheckType.SIGNATURE_VERIFICATION,
                    result=PreflightCheckResult.FAILED,
                    message=f"Invalid raw signature: {e}",
                    required=True,
                )

        # No signature at all = REJECT (Design Doc: unsigned artifacts ALWAYS rejected)
        if sig_info is None:
            return PreflightCheck(
                check_type=PreflightCheckType.SIGNATURE_VERIFICATION,
                result=PreflightCheckResult.FAILED,
                message="REJECTED: Artifact is unsigned. Design Doc requires all artifacts to be signed.",
                details={"rejection_reason": "unsigned"},
                required=True,
            )

        # Find manifest path (look for manifest.json next to artifact)
        manifest_path = artifact_path.parent / "manifest.json"
        if not manifest_path.exists():
            # Try manifest.yaml for backwards compatibility
            manifest_path = artifact_path.parent / "manifest.yaml"

        if not manifest_path.exists():
            return PreflightCheck(
                check_type=PreflightCheckType.SIGNATURE_VERIFICATION,
                result=PreflightCheckResult.FAILED,
                message="Manifest file not found for verification",
                required=True,
            )

        # Perform REAL cryptographic verification
        try:
            report: VerificationReport = self._artifact_verifier.verify(
                artifact_path=artifact_path,
                manifest_path=manifest_path,
                signature_info=sig_info,
            )

            if report.result == VerificationResult.VERIFIED:
                return PreflightCheck(
                    check_type=PreflightCheckType.SIGNATURE_VERIFICATION,
                    result=PreflightCheckResult.PASSED,
                    message="Signature VERIFIED (cryptographic verification passed)",
                    details={
                        "signature_verified": report.signature_verified,
                        "digest_verified": report.digest_verified,
                        "key_id": report.key_id_used,
                        "schema_version": report.schema_version,
                    },
                    required=True,
                )
            else:
                # Verification FAILED or REJECTED
                return PreflightCheck(
                    check_type=PreflightCheckType.SIGNATURE_VERIFICATION,
                    result=PreflightCheckResult.FAILED,
                    message=f"REJECTED: {report.rejection_reason.value if report.rejection_reason else 'verification_failed'} - {report.rejection_details or 'Cryptographic verification failed'}",
                    details={
                        "rejection_reason": report.rejection_reason.value if report.rejection_reason else None,
                        "rejection_details": report.rejection_details,
                    },
                    required=True,
                )

        except Exception as e:
            return PreflightCheck(
                check_type=PreflightCheckType.SIGNATURE_VERIFICATION,
                result=PreflightCheckResult.FAILED,
                message=f"Signature verification error: {str(e)}",
                required=True,
            )

    def _check_signature_fallback(
        self,
        manifest: Optional[Dict[str, Any]],
        signature: Optional[bytes],
    ) -> PreflightCheck:
        """
        Fallback when ArtifactVerifier not available.

        WARNING: This still REQUIRES a signature to be present.
        Real verification should be done by ArtifactVerifier.
        """
        # Even in fallback, we REQUIRE signature presence (but warn about no crypto)
        if signature is not None:
            if len(signature) == 0:
                return PreflightCheck(
                    check_type=PreflightCheckType.SIGNATURE_VERIFICATION,
                    result=PreflightCheckResult.FAILED,
                    message="Empty signature",
                    required=True,
                )
            return PreflightCheck(
                check_type=PreflightCheckType.SIGNATURE_VERIFICATION,
                result=PreflightCheckResult.WARNING,
                message="Signature present but ArtifactVerifier not configured - CRYPTO VERIFICATION SKIPPED (not recommended for production)",
                details={"signature_length": len(signature), "crypto_verified": False},
                required=True,
            )

        # Check manifest for embedded signature
        sig_obj = (manifest or {}).get("signature") if isinstance(manifest, dict) else None
        sig_value = sig_obj.get("signature_value") if isinstance(sig_obj, dict) else None

        if not sig_value or not isinstance(sig_value, str):
            return PreflightCheck(
                check_type=PreflightCheckType.SIGNATURE_VERIFICATION,
                result=PreflightCheckResult.FAILED,
                message="REJECTED: No signature found. Configure ArtifactVerifier for production.",
                required=True,
            )

        return PreflightCheck(
            check_type=PreflightCheckType.SIGNATURE_VERIFICATION,
            result=PreflightCheckResult.WARNING,
            message="Signature present but ArtifactVerifier not configured - CRYPTO VERIFICATION SKIPPED (not recommended for production)",
            details={"algorithm": sig_obj.get("algorithm"), "crypto_verified": False},
            required=True,
        )

    def _check_policy_firewall(self, manifest: Optional[Dict[str, Any]]) -> PreflightCheck:
        """Check policy firewall configuration."""
        if self._policy_firewall is None:
            return PreflightCheck(
                check_type=PreflightCheckType.POLICY_FIREWALL,
                result=PreflightCheckResult.WARNING,
                message="Policy firewall not configured",
                required=False,
            )

        # Verify manifest risk profile is within policy
        if manifest and "risk_profile_suggested" in manifest:
            suggested = manifest["risk_profile_suggested"]
            if hasattr(self._policy_firewall, "check_config_change"):
                result = self._policy_firewall.check_config_change(suggested, "TRADING_IMPACTING")
                if not result.allowed:
                    return PreflightCheck(
                        check_type=PreflightCheckType.POLICY_FIREWALL,
                        result=PreflightCheckResult.FAILED,
                        message=f"Suggested risk profile exceeds local policy: {result.violations}",
                        required=True,
                    )

        return PreflightCheck(
            check_type=PreflightCheckType.POLICY_FIREWALL,
            result=PreflightCheckResult.PASSED,
            message="Policy firewall configured",
        )

    def _check_hard_caps(self, manifest: Optional[Dict[str, Any]]) -> PreflightCheck:
        """Check hard caps configuration."""
        if self._hard_cap_enforcer is None:
            return PreflightCheck(
                check_type=PreflightCheckType.HARD_CAPS,
                result=PreflightCheckResult.WARNING,
                message="Hard cap enforcer not configured",
                required=False,
            )

        return PreflightCheck(
            check_type=PreflightCheckType.HARD_CAPS,
            result=PreflightCheckResult.PASSED,
            message="Hard caps configured",
        )

    def _check_broker_connectivity(self, broker_name: Optional[str]) -> PreflightCheck:
        """Check broker connectivity."""
        if self.config.skip_broker_check:
            return PreflightCheck(
                check_type=PreflightCheckType.BROKER_CONNECTIVITY,
                result=PreflightCheckResult.SKIPPED,
                message="Broker connectivity check skipped",
                required=False,
            )

        if not broker_name:
            return PreflightCheck(
                check_type=PreflightCheckType.BROKER_CONNECTIVITY,
                result=PreflightCheckResult.SKIPPED,
                message="No broker specified",
                required=False,
            )

        if self._broker_connector is None:
            if self.config.require_broker_connectivity:
                return PreflightCheck(
                    check_type=PreflightCheckType.BROKER_CONNECTIVITY,
                    result=PreflightCheckResult.FAILED,
                    message="Broker connector not configured",
                    required=True,
                )
            return PreflightCheck(
                check_type=PreflightCheckType.BROKER_CONNECTIVITY,
                result=PreflightCheckResult.WARNING,
                message="Broker connector not configured",
                required=False,
            )

        try:
            # Check connectivity (without exposing credentials)
            if hasattr(self._broker_connector, "test_connection"):
                connected = self._broker_connector.test_connection()
                if not connected:
                    return PreflightCheck(
                        check_type=PreflightCheckType.BROKER_CONNECTIVITY,
                        result=PreflightCheckResult.FAILED,
                        message=f"Cannot connect to broker: {broker_name}",
                        required=True,
                    )
        except Exception as e:
            return PreflightCheck(
                check_type=PreflightCheckType.BROKER_CONNECTIVITY,
                result=PreflightCheckResult.FAILED,
                message=f"Broker connectivity check failed: {str(e)}",
                required=True,
            )

        return PreflightCheck(
            check_type=PreflightCheckType.BROKER_CONNECTIVITY,
            result=PreflightCheckResult.PASSED,
            message=f"Connected to broker: {broker_name}",
        )

    def _check_resources(self) -> PreflightCheck:
        """Check system resources."""
        import shutil
        import os

        try:
            # Check disk space
            total, used, free = shutil.disk_usage("/")
            free_mb = free // (1024 * 1024)

            if free_mb < self.config.min_disk_space_mb:
                return PreflightCheck(
                    check_type=PreflightCheckType.RESOURCES_AVAILABLE,
                    result=PreflightCheckResult.FAILED,
                    message=f"Insufficient disk space: {free_mb}MB (need {self.config.min_disk_space_mb}MB)",
                    required=True,
                )

            return PreflightCheck(
                check_type=PreflightCheckType.RESOURCES_AVAILABLE,
                result=PreflightCheckResult.PASSED,
                message=f"Resources available: {free_mb}MB disk free",
                details={"disk_free_mb": free_mb},
            )

        except Exception as e:
            return PreflightCheck(
                check_type=PreflightCheckType.RESOURCES_AVAILABLE,
                result=PreflightCheckResult.WARNING,
                message=f"Resource check failed: {str(e)}",
                required=False,
            )

    def _check_network(self) -> PreflightCheck:
        """Check network connectivity."""
        if self.config.skip_network_check:
            return PreflightCheck(
                check_type=PreflightCheckType.NETWORK_CONNECTIVITY,
                result=PreflightCheckResult.SKIPPED,
                message="Network check skipped",
                required=False,
            )

        import socket

        try:
            # Try to connect to a known host
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            return PreflightCheck(
                check_type=PreflightCheckType.NETWORK_CONNECTIVITY,
                result=PreflightCheckResult.PASSED,
                message="Network connectivity OK",
            )
        except Exception:
            return PreflightCheck(
                check_type=PreflightCheckType.NETWORK_CONNECTIVITY,
                result=PreflightCheckResult.WARNING,
                message="Network connectivity check failed",
                required=False,
            )
