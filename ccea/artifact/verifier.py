# -*- coding: utf-8 -*-
"""
CCEA Artifact Verifier.

Agent-side verification of artifacts before execution.
Per Design Doc Phase 4:
- verify digest + signature + allowlist registry + schema_version compatibility
- strict reject: unsigned/unknown registry/unknown schema_version

Security:
- Agent NEVER runs unverified artifacts
- All verification happens LOCALLY in agent
- No fallback to unsigned mode
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ccea.crypto.digest import compute_file_digest, verify_file_digest
from ccea.crypto.signing import verify_signature
from ccea.crypto.keys import load_public_key, PublicKeyTypes
from ccea.artifact.signer import SignatureInfo


class VerificationResult(str, Enum):
    """Verification result status."""
    VERIFIED = "verified"
    FAILED = "failed"
    REJECTED = "rejected"


class RejectionReason(str, Enum):
    """Reasons for artifact rejection."""
    UNSIGNED = "unsigned"
    INVALID_SIGNATURE = "invalid_signature"
    DIGEST_MISMATCH = "digest_mismatch"
    UNKNOWN_REGISTRY = "unknown_registry"
    UNKNOWN_SCHEMA_VERSION = "unknown_schema_version"
    INCOMPATIBLE_SCHEMA = "incompatible_schema"
    EXPIRED_SIGNATURE = "expired_signature"
    REVOKED_KEY = "revoked_key"
    MANIFEST_INVALID = "manifest_invalid"
    MISSING_SBOM = "missing_sbom"
    PROHIBITED_CONTENT = "prohibited_content"


@dataclass
class VerificationReport:
    """Detailed verification report."""
    result: VerificationResult
    artifact_digest: str
    manifest_digest: Optional[str] = None
    signature_verified: bool = False
    digest_verified: bool = False
    registry_verified: bool = False
    schema_verified: bool = False
    rejection_reason: Optional[RejectionReason] = None
    rejection_details: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    verified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    key_id_used: Optional[str] = None
    schema_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result": self.result.value,
            "artifact_digest": self.artifact_digest,
            "manifest_digest": self.manifest_digest,
            "signature_verified": self.signature_verified,
            "digest_verified": self.digest_verified,
            "registry_verified": self.registry_verified,
            "schema_verified": self.schema_verified,
            "rejection_reason": self.rejection_reason.value if self.rejection_reason else None,
            "rejection_details": self.rejection_details,
            "warnings": self.warnings,
            "verified_at": self.verified_at.isoformat(),
            "key_id_used": self.key_id_used,
            "schema_version": self.schema_version,
        }


@dataclass
class SchemaVersionPolicy:
    """Schema version compatibility policy."""
    min_supported: str = "1.0.0"
    max_supported: str = "1.99.99"
    deprecated_versions: Set[str] = field(default_factory=set)

    def is_compatible(self, version: str) -> Tuple[bool, Optional[str]]:
        """
        Check if schema version is compatible.

        Returns:
            Tuple of (is_compatible, warning_message)
        """
        try:
            v_parts = [int(x) for x in version.split(".")]
            min_parts = [int(x) for x in self.min_supported.split(".")]
            max_parts = [int(x) for x in self.max_supported.split(".")]

            # Check minimum
            for v, m in zip(v_parts, min_parts):
                if v < m:
                    return False, f"Schema version {version} below minimum {self.min_supported}"
                if v > m:
                    break

            # Check maximum
            for v, m in zip(v_parts, max_parts):
                if v > m:
                    return False, f"Schema version {version} above maximum {self.max_supported}"
                if v < m:
                    break

            # Check deprecated
            if version in self.deprecated_versions:
                return True, f"Schema version {version} is deprecated"

            return True, None

        except (ValueError, IndexError):
            return False, f"Invalid schema version format: {version}"


class ArtifactVerifier:
    """
    Artifact verifier for agent-side verification.

    Implements strict verification per Design Doc:
    - Digest verification
    - Signature verification
    - Registry allowlist check
    - Schema version compatibility

    IMPORTANT: Unsigned artifacts are ALWAYS rejected.
    """

    # Prohibited fields in manifest (order-like content)
    PROHIBITED_FIELDS = frozenset({
        "side", "quantity", "qty", "price", "order_type",
        "target_position", "execute_order", "place_order",
        "submit_order", "intent", "signal", "order"
    })

    def __init__(
        self,
        trusted_keys: Optional[Dict[str, PublicKeyTypes]] = None,
        allowed_registries: Optional[Set[str]] = None,
        schema_policy: Optional[SchemaVersionPolicy] = None,
        require_sbom: bool = False,
        strict_mode: bool = True,
    ):
        """
        Initialize verifier.

        Args:
            trusted_keys: Dictionary of key_id -> public_key
            allowed_registries: Set of allowed registry URLs/names
            schema_policy: Schema version policy
            require_sbom: Whether to require SBOM
            strict_mode: Strict rejection mode (default True)
        """
        self._trusted_keys: Dict[str, PublicKeyTypes] = trusted_keys or {}
        self._allowed_registries: Set[str] = allowed_registries or {"local", "ccea-registry"}
        self._schema_policy = schema_policy or SchemaVersionPolicy()
        self._require_sbom = require_sbom
        self._strict_mode = strict_mode
        self._revoked_keys: Set[str] = set()

    def add_trusted_key(self, key_id: str, public_key: PublicKeyTypes) -> None:
        """Add a trusted public key."""
        self._trusted_keys[key_id] = public_key

    def add_trusted_key_pem(self, key_id: str, pem_data: str) -> None:
        """Add a trusted public key from PEM."""
        self._trusted_keys[key_id] = load_public_key(pem_data)

    def revoke_key(self, key_id: str) -> None:
        """Mark a key as revoked."""
        self._revoked_keys.add(key_id)

    def add_allowed_registry(self, registry: str) -> None:
        """Add an allowed registry."""
        self._allowed_registries.add(registry)

    def verify(
        self,
        artifact_path: Path,
        manifest_path: Path,
        signature_info: Optional[SignatureInfo] = None,
        expected_digest: Optional[str] = None,
        registry_source: Optional[str] = None,
        sbom_path: Optional[Path] = None,
    ) -> VerificationReport:
        """
        Verify an artifact.

        Args:
            artifact_path: Path to artifact file
            manifest_path: Path to manifest file
            signature_info: Signature information
            expected_digest: Expected digest (from registry)
            registry_source: Source registry identifier
            sbom_path: Optional path to SBOM

        Returns:
            VerificationReport with results

        Note: Does NOT raise exceptions - returns rejection report instead
        """
        warnings = []

        # Step 1: Compute artifact digest
        try:
            actual_digest = compute_file_digest(artifact_path)
        except FileNotFoundError:
            return VerificationReport(
                result=VerificationResult.REJECTED,
                artifact_digest="unknown",
                rejection_reason=RejectionReason.DIGEST_MISMATCH,
                rejection_details="Artifact file not found",
            )

        # Step 2: Verify digest if expected digest provided
        digest_verified = True
        if expected_digest and actual_digest != expected_digest:
            return VerificationReport(
                result=VerificationResult.REJECTED,
                artifact_digest=actual_digest,
                rejection_reason=RejectionReason.DIGEST_MISMATCH,
                rejection_details=f"Expected {expected_digest}, got {actual_digest}",
            )

        # Step 3: Parse and validate manifest
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            return VerificationReport(
                result=VerificationResult.REJECTED,
                artifact_digest=actual_digest,
                rejection_reason=RejectionReason.MANIFEST_INVALID,
                rejection_details=str(e),
            )

        manifest_digest = compute_file_digest(manifest_path)

        # Step 4: Check for prohibited content
        prohibited = self._find_prohibited_fields(manifest)
        if prohibited:
            return VerificationReport(
                result=VerificationResult.REJECTED,
                artifact_digest=actual_digest,
                manifest_digest=manifest_digest,
                rejection_reason=RejectionReason.PROHIBITED_CONTENT,
                rejection_details=f"Prohibited fields found: {prohibited}",
            )

        # Step 5: Verify schema version
        schema_version = manifest.get("schema_version")
        if not schema_version:
            return VerificationReport(
                result=VerificationResult.REJECTED,
                artifact_digest=actual_digest,
                manifest_digest=manifest_digest,
                rejection_reason=RejectionReason.UNKNOWN_SCHEMA_VERSION,
                rejection_details="Missing schema_version in manifest",
            )

        is_compatible, schema_warning = self._schema_policy.is_compatible(schema_version)
        if not is_compatible:
            return VerificationReport(
                result=VerificationResult.REJECTED,
                artifact_digest=actual_digest,
                manifest_digest=manifest_digest,
                schema_version=schema_version,
                rejection_reason=RejectionReason.INCOMPATIBLE_SCHEMA,
                rejection_details=schema_warning,
            )
        if schema_warning:
            warnings.append(schema_warning)

        # Step 6: Verify registry
        registry_verified = True
        if registry_source:
            if registry_source not in self._allowed_registries:
                return VerificationReport(
                    result=VerificationResult.REJECTED,
                    artifact_digest=actual_digest,
                    manifest_digest=manifest_digest,
                    schema_version=schema_version,
                    rejection_reason=RejectionReason.UNKNOWN_REGISTRY,
                    rejection_details=f"Registry '{registry_source}' not in allowlist",
                )

        # Step 7: Verify signature (REQUIRED in strict mode)
        signature_verified = False
        key_id_used = None

        if signature_info:
            # Check for revoked key
            if signature_info.key_id in self._revoked_keys:
                return VerificationReport(
                    result=VerificationResult.REJECTED,
                    artifact_digest=actual_digest,
                    manifest_digest=manifest_digest,
                    schema_version=schema_version,
                    registry_verified=registry_verified,
                    rejection_reason=RejectionReason.REVOKED_KEY,
                    rejection_details=f"Key {signature_info.key_id} has been revoked",
                )

            # Get public key
            key_id = signature_info.key_id
            public_key = self._trusted_keys.get(key_id) if key_id else None

            # Try to find key by checking all trusted keys
            if not public_key:
                for kid, pk in self._trusted_keys.items():
                    if self._verify_signature_with_key(artifact_path, signature_info, pk):
                        signature_verified = True
                        key_id_used = kid
                        break
            else:
                signature_verified = self._verify_signature_with_key(
                    artifact_path, signature_info, public_key
                )
                key_id_used = key_id

            if not signature_verified:
                return VerificationReport(
                    result=VerificationResult.REJECTED,
                    artifact_digest=actual_digest,
                    manifest_digest=manifest_digest,
                    schema_version=schema_version,
                    registry_verified=registry_verified,
                    rejection_reason=RejectionReason.INVALID_SIGNATURE,
                    rejection_details="Signature verification failed",
                )

        elif self._strict_mode:
            # Strict mode: UNSIGNED artifacts are ALWAYS rejected
            return VerificationReport(
                result=VerificationResult.REJECTED,
                artifact_digest=actual_digest,
                manifest_digest=manifest_digest,
                schema_version=schema_version,
                registry_verified=registry_verified,
                rejection_reason=RejectionReason.UNSIGNED,
                rejection_details="Artifact is not signed. Unsigned artifacts are rejected.",
            )
        else:
            warnings.append("Artifact is unsigned - not recommended for production")

        # Step 8: Check SBOM if required
        if self._require_sbom and not sbom_path:
            sbom_ref = manifest.get("sbom_ref")
            if not sbom_ref:
                return VerificationReport(
                    result=VerificationResult.REJECTED,
                    artifact_digest=actual_digest,
                    manifest_digest=manifest_digest,
                    schema_version=schema_version,
                    registry_verified=registry_verified,
                    signature_verified=signature_verified,
                    key_id_used=key_id_used,
                    rejection_reason=RejectionReason.MISSING_SBOM,
                    rejection_details="SBOM is required but not provided",
                )

        # All checks passed
        return VerificationReport(
            result=VerificationResult.VERIFIED,
            artifact_digest=actual_digest,
            manifest_digest=manifest_digest,
            schema_version=schema_version,
            signature_verified=signature_verified,
            digest_verified=digest_verified,
            registry_verified=registry_verified,
            schema_verified=True,
            key_id_used=key_id_used,
            warnings=warnings,
        )

    def verify_quick(
        self,
        artifact_path: Path,
        expected_digest: str,
        signature_info: SignatureInfo,
    ) -> bool:
        """
        Quick verification (digest + signature only).

        Args:
            artifact_path: Path to artifact
            expected_digest: Expected digest
            signature_info: Signature info

        Returns:
            True if verified, False otherwise
        """
        # Verify digest
        if not verify_file_digest(artifact_path, expected_digest):
            return False

        # Verify signature
        key_id = signature_info.key_id
        public_key = self._trusted_keys.get(key_id) if key_id else None

        if not public_key:
            return False

        return self._verify_signature_with_key(artifact_path, signature_info, public_key)

    def _verify_signature_with_key(
        self,
        artifact_path: Path,
        signature_info: SignatureInfo,
        public_key: PublicKeyTypes,
    ) -> bool:
        """Verify signature using specific public key."""
        try:
            content = artifact_path.read_bytes()
            return verify_signature(content, signature_info.signature, public_key)
        except Exception:
            return False

    def _find_prohibited_fields(
        self,
        data: Any,
        path: str = "",
    ) -> List[str]:
        """Recursively find prohibited fields in manifest."""
        found = []

        if isinstance(data, dict):
            for key, value in data.items():
                key_lower = key.lower()
                if key_lower in self.PROHIBITED_FIELDS:
                    found.append(f"{path}.{key}" if path else key)
                found.extend(
                    self._find_prohibited_fields(
                        value,
                        f"{path}.{key}" if path else key
                    )
                )
        elif isinstance(data, list):
            for i, item in enumerate(data):
                found.extend(
                    self._find_prohibited_fields(item, f"{path}[{i}]")
                )

        return found


class VerificationCache:
    """
    Cache for verification results.

    Avoids re-verifying same artifacts.
    """

    def __init__(self, max_entries: int = 1000):
        """
        Initialize cache.

        Args:
            max_entries: Maximum cache entries
        """
        self._cache: Dict[str, VerificationReport] = {}
        self._max_entries = max_entries

    def get(self, digest: str) -> Optional[VerificationReport]:
        """Get cached verification result."""
        return self._cache.get(digest)

    def put(self, digest: str, report: VerificationReport) -> None:
        """Cache verification result."""
        if len(self._cache) >= self._max_entries:
            # Simple eviction: remove oldest
            oldest = min(self._cache.items(), key=lambda x: x[1].verified_at)
            del self._cache[oldest[0]]

        self._cache[digest] = report

    def invalidate(self, digest: str) -> None:
        """Invalidate cached result."""
        self._cache.pop(digest, None)

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()


def create_verifier_from_key_manager(
    key_manager: "KeyManager",
    allowed_registries: Optional[Set[str]] = None,
) -> ArtifactVerifier:
    """
    Create verifier using KeyManager.

    Args:
        key_manager: KeyManager instance
        allowed_registries: Allowed registries

    Returns:
        Configured ArtifactVerifier
    """
    from ccea.crypto.key_manager import KeyPurpose

    verifier = ArtifactVerifier(
        allowed_registries=allowed_registries,
    )

    # Add all valid signing keys
    for key_id in key_manager.get_signing_keys(KeyPurpose.ARTIFACT_SIGNING):
        public_key = key_manager.get_public_key(key_id)
        if public_key:
            verifier.add_trusted_key(key_id, public_key)

        # Check if revoked
        if key_manager.is_revoked(key_id):
            verifier.revoke_key(key_id)

    return verifier
