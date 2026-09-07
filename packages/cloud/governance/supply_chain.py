"""
Supply Chain Security Service - Phase 7 GDPR Implementation

Implements Design Doc Section 15.1 supply chain security controls:
- Signed artifacts
- Digest pinning (no "latest" tags)
- Registry allowlist
- SBOM management and storage

Reference: docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L909-L915
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4


# =============================================================================
# Enums and Constants
# =============================================================================


class ArtifactType(str, Enum):
    """Types of artifacts in the supply chain."""

    CONTAINER_IMAGE = "container_image"
    CONFIG_BLOB = "config_blob"
    BINARY = "binary"
    LIBRARY = "library"
    HELM_CHART = "helm_chart"
    AGENT_PACKAGE = "agent_package"
    RESEARCH_ARTIFACT = "research_artifact"
    MODEL_WEIGHTS = "model_weights"


class SigningAlgorithm(str, Enum):
    """Supported signing algorithms."""

    RSA_PSS_SHA256 = "rsa_pss_sha256"
    RSA_PKCS1_SHA256 = "rsa_pkcs1_sha256"
    ED25519 = "ed25519"
    ECDSA_P256 = "ecdsa_p256"
    ECDSA_P384 = "ecdsa_p384"


class SignatureStatus(str, Enum):
    """Signature verification status."""

    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    UNKNOWN_SIGNER = "unknown_signer"
    NOT_SIGNED = "not_signed"


class SBOMFormat(str, Enum):
    """SBOM format standards."""

    SPDX_2_3 = "spdx_2.3"
    SPDX_3_0 = "spdx_3.0"
    CYCLONEDX_1_4 = "cyclonedx_1.4"
    CYCLONEDX_1_5 = "cyclonedx_1.5"


class VulnerabilitySeverity(str, Enum):
    """Vulnerability severity levels (CVSS-based)."""

    CRITICAL = "critical"  # CVSS 9.0-10.0
    HIGH = "high"  # CVSS 7.0-8.9
    MEDIUM = "medium"  # CVSS 4.0-6.9
    LOW = "low"  # CVSS 0.1-3.9
    NONE = "none"  # CVSS 0.0


class SupplyChainEventType(str, Enum):
    """Types of supply chain events."""

    ARTIFACT_REGISTERED = "artifact_registered"
    ARTIFACT_SIGNED = "artifact_signed"
    SIGNATURE_VERIFIED = "signature_verified"
    SIGNATURE_FAILED = "signature_failed"
    DIGEST_PINNED = "digest_pinned"
    DIGEST_UNPINNED = "digest_unpinned"
    REGISTRY_ADDED = "registry_added"
    REGISTRY_REMOVED = "registry_removed"
    SBOM_UPLOADED = "sbom_uploaded"
    SBOM_ACCESSED = "sbom_accessed"
    VULNERABILITY_DETECTED = "vulnerability_detected"
    VULNERABILITY_RESOLVED = "vulnerability_resolved"
    POLICY_VIOLATION = "policy_violation"
    ARTIFACT_REJECTED = "artifact_rejected"


# Constants
DIGEST_PATTERN = re.compile(r"^sha256:[a-f0-9]{64}$")
TAG_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
DEFAULT_PIN_EXPIRY_DAYS = 90
MAX_PIN_EXPIRY_DAYS = 365
TRUSTED_SIGNERS_CACHE_TTL_SECONDS = 300


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SignedArtifact:
    """A signed artifact in the supply chain."""

    artifact_id: str = field(default_factory=lambda: str(uuid4()))
    artifact_type: ArtifactType = ArtifactType.CONTAINER_IMAGE
    name: str = ""
    digest: str = ""  # sha256:hexdigest
    size_bytes: int = 0
    signature: str = ""  # Base64-encoded signature
    signer_id: str = ""
    signer_name: str = ""
    signing_algorithm: SigningAlgorithm = SigningAlgorithm.ED25519
    signing_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    certificate_chain: List[str] = field(default_factory=list)
    registry: str = ""
    repository: str = ""
    tags: List[str] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""
    sbom_digest: Optional[str] = None  # Link to SBOM

    def __post_init__(self) -> None:
        if self.digest and not DIGEST_PATTERN.match(self.digest):
            raise ValueError(f"Invalid digest format: {self.digest}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type.value,
            "name": self.name,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
            "signature": self.signature,
            "signer_id": self.signer_id,
            "signer_name": self.signer_name,
            "signing_algorithm": self.signing_algorithm.value,
            "signing_timestamp": self.signing_timestamp.isoformat(),
            "certificate_chain": self.certificate_chain,
            "registry": self.registry,
            "repository": self.repository,
            "tags": self.tags,
            "labels": self.labels,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "sbom_digest": self.sbom_digest,
        }


@dataclass
class SignatureVerification:
    """Result of signature verification."""

    artifact_id: str = ""
    digest: str = ""
    status: SignatureStatus = SignatureStatus.NOT_SIGNED
    verified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    signer_id: Optional[str] = None
    signer_name: Optional[str] = None
    algorithm: Optional[SigningAlgorithm] = None
    certificate_valid: bool = False
    certificate_expiry: Optional[datetime] = None
    error: Optional[str] = None
    verification_time_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "digest": self.digest,
            "status": self.status.value,
            "verified_at": self.verified_at.isoformat(),
            "signer_id": self.signer_id,
            "signer_name": self.signer_name,
            "algorithm": self.algorithm.value if self.algorithm else None,
            "certificate_valid": self.certificate_valid,
            "certificate_expiry": (
                self.certificate_expiry.isoformat() if self.certificate_expiry else None
            ),
            "error": self.error,
            "verification_time_ms": self.verification_time_ms,
        }


@dataclass
class DigestPin:
    """A pinned digest for an artifact."""

    pin_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    artifact_type: ArtifactType = ArtifactType.CONTAINER_IMAGE
    artifact_name: str = ""
    pinned_digest: str = ""
    pinned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    pinned_by: str = ""
    expires_at: Optional[datetime] = None
    reason: str = ""
    active: bool = True

    def __post_init__(self) -> None:
        if self.pinned_digest and not DIGEST_PATTERN.match(self.pinned_digest):
            raise ValueError(f"Invalid digest format: {self.pinned_digest}")

    def is_expired(self) -> bool:
        """Check if the pin has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) >= self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pin_id": self.pin_id,
            "workspace_id": self.workspace_id,
            "artifact_type": self.artifact_type.value,
            "artifact_name": self.artifact_name,
            "pinned_digest": self.pinned_digest,
            "pinned_at": self.pinned_at.isoformat(),
            "pinned_by": self.pinned_by,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "reason": self.reason,
            "active": self.active,
        }


@dataclass
class RegistryConfig:
    """Configuration for an allowed registry."""

    registry_id: str = field(default_factory=lambda: str(uuid4()))
    hostname: str = ""  # e.g., "gcr.io/project", "ecr.eu-west-1.amazonaws.com"
    require_signature: bool = True
    require_sbom: bool = True
    allowed_namespaces: Set[str] = field(default_factory=set)  # Empty = all allowed
    denied_namespaces: Set[str] = field(default_factory=set)
    trusted_signers: Set[str] = field(default_factory=set)  # Signer IDs
    eu_only: bool = True  # Must be EU region
    added_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    added_by: str = ""
    description: str = ""

    def matches_artifact(self, registry: str, repository: str) -> bool:
        """Check if an artifact matches this registry configuration."""
        if not registry.startswith(self.hostname):
            return False

        namespace = repository.split("/")[0] if "/" in repository else repository

        if self.denied_namespaces and namespace in self.denied_namespaces:
            return False

        if self.allowed_namespaces and namespace not in self.allowed_namespaces:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "registry_id": self.registry_id,
            "hostname": self.hostname,
            "require_signature": self.require_signature,
            "require_sbom": self.require_sbom,
            "allowed_namespaces": list(self.allowed_namespaces),
            "denied_namespaces": list(self.denied_namespaces),
            "trusted_signers": list(self.trusted_signers),
            "eu_only": self.eu_only,
            "added_at": self.added_at.isoformat(),
            "added_by": self.added_by,
            "description": self.description,
        }


@dataclass
class RegistryAllowlist:
    """Allowlist of registries."""

    allowlist_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    registries: List[RegistryConfig] = field(default_factory=list)
    deny_all_by_default: bool = True
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_by: str = ""

    def is_registry_allowed(
        self, registry: str, repository: str
    ) -> Tuple[bool, Optional[RegistryConfig]]:
        """Check if a registry/repository is allowed."""
        for config in self.registries:
            if config.matches_artifact(registry, repository):
                return True, config

        if self.deny_all_by_default:
            return False, None

        return True, None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowlist_id": self.allowlist_id,
            "workspace_id": self.workspace_id,
            "registries": [r.to_dict() for r in self.registries],
            "deny_all_by_default": self.deny_all_by_default,
            "updated_at": self.updated_at.isoformat(),
            "updated_by": self.updated_by,
        }


@dataclass
class SBOMComponent:
    """A component in an SBOM."""

    name: str = ""
    version: str = ""
    purl: str = ""  # Package URL
    license: str = ""
    licenses: List[str] = field(default_factory=list)
    supplier: str = ""
    checksum: str = ""
    component_type: str = ""  # library, framework, application, etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "purl": self.purl,
            "license": self.license,
            "licenses": self.licenses,
            "supplier": self.supplier,
            "checksum": self.checksum,
            "component_type": self.component_type,
        }


@dataclass
class Vulnerability:
    """A vulnerability associated with an SBOM component."""

    vuln_id: str = ""  # CVE ID
    severity: VulnerabilitySeverity = VulnerabilitySeverity.MEDIUM
    cvss_score: float = 0.0
    cvss_vector: str = ""
    affected_component: str = ""
    affected_versions: str = ""
    fixed_version: Optional[str] = None
    description: str = ""
    references: List[str] = field(default_factory=list)
    published_at: Optional[datetime] = None
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vuln_id": self.vuln_id,
            "severity": self.severity.value,
            "cvss_score": self.cvss_score,
            "cvss_vector": self.cvss_vector,
            "affected_component": self.affected_component,
            "affected_versions": self.affected_versions,
            "fixed_version": self.fixed_version,
            "description": self.description,
            "references": self.references,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "detected_at": self.detected_at.isoformat(),
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


@dataclass
class SBOM:
    """Software Bill of Materials."""

    sbom_id: str = field(default_factory=lambda: str(uuid4()))
    artifact_digest: str = ""  # Links to signed artifact
    format: SBOMFormat = SBOMFormat.CYCLONEDX_1_5
    format_version: str = "1.5"
    spec_version: str = "1.5"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""
    tool_name: str = ""
    tool_version: str = ""
    components: List[SBOMComponent] = field(default_factory=list)
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    integrity_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = json.dumps(
            {
                "sbom_id": self.sbom_id,
                "artifact_digest": self.artifact_digest,
                "format": self.format.value,
                "created_at": self.created_at.isoformat(),
                "components_count": len(self.components),
                "vulnerabilities_count": len(self.vulnerabilities),
            },
            sort_keys=True,
        )
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def get_vulnerability_summary(self) -> Dict[str, int]:
        """Get vulnerability count by severity."""
        summary: Dict[str, int] = {s.value: 0 for s in VulnerabilitySeverity}
        for vuln in self.vulnerabilities:
            if not vuln.resolved:
                summary[vuln.severity.value] += 1
        return summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sbom_id": self.sbom_id,
            "artifact_digest": self.artifact_digest,
            "format": self.format.value,
            "format_version": self.format_version,
            "spec_version": self.spec_version,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "components": [c.to_dict() for c in self.components],
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "integrity_hash": self.integrity_hash,
            "metadata": self.metadata,
        }


@dataclass
class TrustedSigner:
    """A trusted signer for artifact verification."""

    signer_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    public_key: str = ""  # PEM-encoded public key
    algorithm: SigningAlgorithm = SigningAlgorithm.ED25519
    fingerprint: str = ""  # Key fingerprint
    valid_from: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_until: Optional[datetime] = None
    revoked: bool = False
    revoked_at: Optional[datetime] = None
    revocation_reason: Optional[str] = None
    created_by: str = ""
    description: str = ""

    def is_valid(self) -> bool:
        """Check if the signer is currently valid."""
        if self.revoked:
            return False
        now = datetime.now(timezone.utc)
        if now < self.valid_from:
            return False
        if self.valid_until and now > self.valid_until:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signer_id": self.signer_id,
            "name": self.name,
            "public_key": (
                self.public_key[:50] + "..." if len(self.public_key) > 50 else self.public_key
            ),
            "algorithm": self.algorithm.value,
            "fingerprint": self.fingerprint,
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "revoked": self.revoked,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            "revocation_reason": self.revocation_reason,
            "created_by": self.created_by,
            "description": self.description,
        }


@dataclass
class SupplyChainEvent:
    """Event in the supply chain audit log."""

    event_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    event_type: SupplyChainEventType = SupplyChainEventType.ARTIFACT_REGISTERED
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    actor_id: str = ""
    artifact_id: Optional[str] = None
    artifact_digest: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    result: str = "success"
    integrity_hash: str = ""

    def __post_init__(self) -> None:
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = json.dumps(
            {
                "event_id": self.event_id,
                "workspace_id": self.workspace_id,
                "event_type": self.event_type.value,
                "timestamp": self.timestamp.isoformat(),
                "actor_id": self.actor_id,
                "artifact_digest": self.artifact_digest,
                "result": self.result,
            },
            sort_keys=True,
        )
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "workspace_id": self.workspace_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "actor_id": self.actor_id,
            "artifact_id": self.artifact_id,
            "artifact_digest": self.artifact_digest,
            "details": self.details,
            "result": self.result,
            "integrity_hash": self.integrity_hash,
        }


@dataclass
class SupplyChainStats:
    """Statistics for supply chain security."""

    workspace_id: str = ""
    total_artifacts: int = 0
    signed_artifacts: int = 0
    unsigned_artifacts: int = 0
    pinned_digests: int = 0
    expired_pins: int = 0
    total_sboms: int = 0
    total_components: int = 0
    vulnerabilities_critical: int = 0
    vulnerabilities_high: int = 0
    vulnerabilities_medium: int = 0
    vulnerabilities_low: int = 0
    allowed_registries: int = 0
    trusted_signers: int = 0
    policy_violations_30d: int = 0
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "total_artifacts": self.total_artifacts,
            "signed_artifacts": self.signed_artifacts,
            "unsigned_artifacts": self.unsigned_artifacts,
            "pinned_digests": self.pinned_digests,
            "expired_pins": self.expired_pins,
            "total_sboms": self.total_sboms,
            "total_components": self.total_components,
            "vulnerabilities_critical": self.vulnerabilities_critical,
            "vulnerabilities_high": self.vulnerabilities_high,
            "vulnerabilities_medium": self.vulnerabilities_medium,
            "vulnerabilities_low": self.vulnerabilities_low,
            "allowed_registries": self.allowed_registries,
            "trusted_signers": self.trusted_signers,
            "policy_violations_30d": self.policy_violations_30d,
            "evaluated_at": self.evaluated_at.isoformat(),
        }


# =============================================================================
# Supply Chain Service
# =============================================================================


class SupplyChainService:
    """
    Supply Chain Security Service.

    Manages:
    - Signed artifact registration and verification
    - Digest pinning (no "latest" tags)
    - Registry allowlist
    - SBOM storage and retrieval
    - Vulnerability tracking

    Reference: Design Doc Section 15.1
    """

    def __init__(
        self,
        require_signatures: bool = True,
        require_sbom: bool = True,
        deny_latest_tag: bool = True,
        on_event: Optional[Callable[[SupplyChainEvent], None]] = None,
        on_policy_violation: Optional[Callable[[SupplyChainEvent], None]] = None,
    ):
        """
        Initialize Supply Chain Service.

        Args:
            require_signatures: Require all artifacts to be signed
            require_sbom: Require SBOM for all artifacts
            deny_latest_tag: Reject artifacts with "latest" tag
            on_event: Callback for supply chain events
            on_policy_violation: Callback for policy violations
        """
        self._lock = threading.RLock()  # Re-entrant lock for nested calls

        # Configuration
        self._require_signatures = require_signatures
        self._require_sbom = require_sbom
        self._deny_latest_tag = deny_latest_tag

        # Callbacks
        self._on_event = on_event
        self._on_policy_violation = on_policy_violation

        # Storage
        self._artifacts: Dict[str, SignedArtifact] = {}  # artifact_id -> artifact
        self._artifacts_by_digest: Dict[str, str] = {}  # digest -> artifact_id
        self._pins: Dict[str, DigestPin] = {}  # pin_id -> pin
        self._allowlists: Dict[str, RegistryAllowlist] = {}  # workspace_id -> allowlist
        self._sboms: Dict[str, SBOM] = {}  # sbom_id -> sbom
        self._sboms_by_digest: Dict[str, str] = {}  # artifact_digest -> sbom_id
        self._signers: Dict[str, TrustedSigner] = {}  # signer_id -> signer
        self._events: List[SupplyChainEvent] = []

    # =========================================================================
    # Artifact Management
    # =========================================================================

    def register_artifact(
        self,
        workspace_id: str,
        artifact_type: ArtifactType,
        name: str,
        digest: str,
        registry: str,
        repository: str,
        created_by: str,
        size_bytes: int = 0,
        tags: Optional[List[str]] = None,
        labels: Optional[Dict[str, str]] = None,
        signature: Optional[str] = None,
        signer_id: Optional[str] = None,
        signing_algorithm: Optional[SigningAlgorithm] = None,
    ) -> SignedArtifact:
        """Register a new artifact."""
        # Validate digest format
        if not DIGEST_PATTERN.match(digest):
            raise ValueError(f"Invalid digest format: {digest}. Must be sha256:hexdigest")

        # Check for "latest" tag
        if self._deny_latest_tag and tags and "latest" in tags:
            event = SupplyChainEvent(
                workspace_id=workspace_id,
                event_type=SupplyChainEventType.POLICY_VIOLATION,
                actor_id=created_by,
                artifact_digest=digest,
                details={"violation": "latest_tag_denied", "tags": tags},
                result="failure",
            )
            self._log_event(event)
            if self._on_policy_violation:
                try:
                    self._on_policy_violation(event)
                except Exception:
                    pass
            raise ValueError("'latest' tag is not allowed. Use digest pinning instead.")

        # Check registry allowlist
        allowlist = self._allowlists.get(workspace_id)
        if allowlist:
            allowed, config = allowlist.is_registry_allowed(registry, repository)
            if not allowed:
                event = SupplyChainEvent(
                    workspace_id=workspace_id,
                    event_type=SupplyChainEventType.POLICY_VIOLATION,
                    actor_id=created_by,
                    artifact_digest=digest,
                    details={"violation": "registry_not_allowed", "registry": registry},
                    result="failure",
                )
                self._log_event(event)
                if self._on_policy_violation:
                    try:
                        self._on_policy_violation(event)
                    except Exception:
                        pass
                raise ValueError(f"Registry {registry} is not in allowlist")

        # Check signature requirement
        if self._require_signatures and not signature:
            event = SupplyChainEvent(
                workspace_id=workspace_id,
                event_type=SupplyChainEventType.POLICY_VIOLATION,
                actor_id=created_by,
                artifact_digest=digest,
                details={"violation": "signature_required"},
                result="failure",
            )
            self._log_event(event)
            if self._on_policy_violation:
                try:
                    self._on_policy_violation(event)
                except Exception:
                    pass
            raise ValueError("Artifact signature is required")

        artifact = SignedArtifact(
            artifact_type=artifact_type,
            name=name,
            digest=digest,
            size_bytes=size_bytes,
            signature=signature or "",
            signer_id=signer_id or "",
            signing_algorithm=signing_algorithm or SigningAlgorithm.ED25519,
            registry=registry,
            repository=repository,
            tags=tags or [],
            labels=labels or {},
            created_by=created_by,
        )

        with self._lock:
            self._artifacts[artifact.artifact_id] = artifact
            self._artifacts_by_digest[digest] = artifact.artifact_id

        # Log event
        event = SupplyChainEvent(
            workspace_id=workspace_id,
            event_type=SupplyChainEventType.ARTIFACT_REGISTERED,
            actor_id=created_by,
            artifact_id=artifact.artifact_id,
            artifact_digest=digest,
            details={
                "name": name,
                "type": artifact_type.value,
                "registry": registry,
                "signed": bool(signature),
            },
        )
        self._log_event(event)

        return artifact

    def get_artifact(self, artifact_id: str) -> Optional[SignedArtifact]:
        """Get artifact by ID."""
        with self._lock:
            return self._artifacts.get(artifact_id)

    def get_artifact_by_digest(self, digest: str) -> Optional[SignedArtifact]:
        """Get artifact by digest."""
        with self._lock:
            artifact_id = self._artifacts_by_digest.get(digest)
            if artifact_id:
                return self._artifacts.get(artifact_id)
        return None

    def list_artifacts(
        self,
        workspace_id: Optional[str] = None,
        artifact_type: Optional[ArtifactType] = None,
        registry: Optional[str] = None,
        signed_only: bool = False,
    ) -> List[SignedArtifact]:
        """List artifacts with filtering."""
        with self._lock:
            artifacts = list(self._artifacts.values())

        if artifact_type:
            artifacts = [a for a in artifacts if a.artifact_type == artifact_type]
        if registry:
            artifacts = [a for a in artifacts if a.registry == registry]
        if signed_only:
            artifacts = [a for a in artifacts if a.signature]

        return artifacts

    def verify_signature(
        self,
        artifact_id: str,
        workspace_id: str,
    ) -> SignatureVerification:
        """Verify artifact signature."""
        artifact = self.get_artifact(artifact_id)
        if not artifact:
            return SignatureVerification(
                artifact_id=artifact_id,
                status=SignatureStatus.NOT_SIGNED,
                error="Artifact not found",
            )

        if not artifact.signature:
            result = SignatureVerification(
                artifact_id=artifact_id,
                digest=artifact.digest,
                status=SignatureStatus.NOT_SIGNED,
                error="Artifact is not signed",
            )

            event = SupplyChainEvent(
                workspace_id=workspace_id,
                event_type=SupplyChainEventType.SIGNATURE_FAILED,
                actor_id="system",
                artifact_id=artifact_id,
                artifact_digest=artifact.digest,
                details={"reason": "not_signed"},
                result="failure",
            )
            self._log_event(event)

            return result

        # Check signer
        signer = self._signers.get(artifact.signer_id)
        if not signer:
            result = SignatureVerification(
                artifact_id=artifact_id,
                digest=artifact.digest,
                status=SignatureStatus.UNKNOWN_SIGNER,
                signer_id=artifact.signer_id,
                error="Signer not found in trusted signers",
            )

            event = SupplyChainEvent(
                workspace_id=workspace_id,
                event_type=SupplyChainEventType.SIGNATURE_FAILED,
                actor_id="system",
                artifact_id=artifact_id,
                artifact_digest=artifact.digest,
                details={"reason": "unknown_signer", "signer_id": artifact.signer_id},
                result="failure",
            )
            self._log_event(event)

            return result

        if signer.revoked:
            result = SignatureVerification(
                artifact_id=artifact_id,
                digest=artifact.digest,
                status=SignatureStatus.REVOKED,
                signer_id=signer.signer_id,
                signer_name=signer.name,
                error="Signer key has been revoked",
            )

            event = SupplyChainEvent(
                workspace_id=workspace_id,
                event_type=SupplyChainEventType.SIGNATURE_FAILED,
                actor_id="system",
                artifact_id=artifact_id,
                artifact_digest=artifact.digest,
                details={"reason": "signer_revoked", "signer_id": signer.signer_id},
                result="failure",
            )
            self._log_event(event)

            return result

        if not signer.is_valid():
            result = SignatureVerification(
                artifact_id=artifact_id,
                digest=artifact.digest,
                status=SignatureStatus.EXPIRED,
                signer_id=signer.signer_id,
                signer_name=signer.name,
                certificate_expiry=signer.valid_until,
                error="Signer certificate is expired or not yet valid",
            )

            event = SupplyChainEvent(
                workspace_id=workspace_id,
                event_type=SupplyChainEventType.SIGNATURE_FAILED,
                actor_id="system",
                artifact_id=artifact_id,
                artifact_digest=artifact.digest,
                details={"reason": "signer_expired", "signer_id": signer.signer_id},
                result="failure",
            )
            self._log_event(event)

            return result

        # In production: actual cryptographic verification here
        # For now, we assume signature is valid if signer is trusted and valid
        result = SignatureVerification(
            artifact_id=artifact_id,
            digest=artifact.digest,
            status=SignatureStatus.VALID,
            signer_id=signer.signer_id,
            signer_name=signer.name,
            algorithm=signer.algorithm,
            certificate_valid=True,
            certificate_expiry=signer.valid_until,
        )

        event = SupplyChainEvent(
            workspace_id=workspace_id,
            event_type=SupplyChainEventType.SIGNATURE_VERIFIED,
            actor_id="system",
            artifact_id=artifact_id,
            artifact_digest=artifact.digest,
            details={"signer_id": signer.signer_id, "signer_name": signer.name},
        )
        self._log_event(event)

        return result

    def validate_artifact_for_deployment(
        self,
        digest: str,
        workspace_id: str,
    ) -> Tuple[bool, List[str]]:
        """
        Validate an artifact is ready for deployment.

        Checks:
        - Artifact exists
        - Signature is valid
        - SBOM exists (if required)
        - Digest is pinned (if required)
        - No critical vulnerabilities

        Returns: (valid, list of errors)
        """
        errors = []

        artifact = self.get_artifact_by_digest(digest)
        if not artifact:
            errors.append(f"Artifact with digest {digest} not found")
            return False, errors

        # Check signature
        if self._require_signatures:
            verification = self.verify_signature(artifact.artifact_id, workspace_id)
            if verification.status != SignatureStatus.VALID:
                errors.append(f"Signature verification failed: {verification.status.value}")

        # Check SBOM
        if self._require_sbom:
            sbom = self.get_sbom_by_artifact_digest(digest)
            if not sbom:
                errors.append("SBOM not found for artifact")
            else:
                # Check for critical vulnerabilities
                summary = sbom.get_vulnerability_summary()
                if summary.get("critical", 0) > 0:
                    errors.append(f"Artifact has {summary['critical']} critical vulnerabilities")

        return len(errors) == 0, errors

    # =========================================================================
    # Digest Pinning
    # =========================================================================

    def pin_digest(
        self,
        workspace_id: str,
        artifact_type: ArtifactType,
        artifact_name: str,
        digest: str,
        pinned_by: str,
        reason: str = "",
        expires_in_days: Optional[int] = None,
    ) -> DigestPin:
        """Pin a digest for an artifact."""
        if not DIGEST_PATTERN.match(digest):
            raise ValueError(f"Invalid digest format: {digest}")

        expires_at = None
        if expires_in_days:
            if expires_in_days > MAX_PIN_EXPIRY_DAYS:
                raise ValueError(f"Pin expiry cannot exceed {MAX_PIN_EXPIRY_DAYS} days")
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

        pin = DigestPin(
            workspace_id=workspace_id,
            artifact_type=artifact_type,
            artifact_name=artifact_name,
            pinned_digest=digest,
            pinned_by=pinned_by,
            expires_at=expires_at,
            reason=reason,
        )

        with self._lock:
            self._pins[pin.pin_id] = pin

        # Log event
        event = SupplyChainEvent(
            workspace_id=workspace_id,
            event_type=SupplyChainEventType.DIGEST_PINNED,
            actor_id=pinned_by,
            artifact_digest=digest,
            details={
                "artifact_name": artifact_name,
                "artifact_type": artifact_type.value,
                "expires_at": expires_at.isoformat() if expires_at else None,
            },
        )
        self._log_event(event)

        return pin

    def unpin_digest(
        self,
        pin_id: str,
        unpinned_by: str,
        reason: str = "",
    ) -> Optional[DigestPin]:
        """Unpin a digest."""
        with self._lock:
            pin = self._pins.get(pin_id)
            if not pin:
                return None

            pin.active = False

        # Log event
        event = SupplyChainEvent(
            workspace_id=pin.workspace_id,
            event_type=SupplyChainEventType.DIGEST_UNPINNED,
            actor_id=unpinned_by,
            artifact_digest=pin.pinned_digest,
            details={
                "artifact_name": pin.artifact_name,
                "reason": reason,
            },
        )
        self._log_event(event)

        return pin

    def get_pinned_digest(
        self,
        workspace_id: str,
        artifact_name: str,
    ) -> Optional[DigestPin]:
        """Get the active pinned digest for an artifact."""
        with self._lock:
            for pin in self._pins.values():
                if (
                    pin.workspace_id == workspace_id
                    and pin.artifact_name == artifact_name
                    and pin.active
                    and not pin.is_expired()
                ):
                    return pin
        return None

    def list_pins(
        self,
        workspace_id: Optional[str] = None,
        active_only: bool = True,
        include_expired: bool = False,
    ) -> List[DigestPin]:
        """List digest pins."""
        with self._lock:
            pins = list(self._pins.values())

        if workspace_id:
            pins = [p for p in pins if p.workspace_id == workspace_id]
        if active_only:
            pins = [p for p in pins if p.active]
        if not include_expired:
            pins = [p for p in pins if not p.is_expired()]

        return pins

    def get_expired_pins(self, workspace_id: Optional[str] = None) -> List[DigestPin]:
        """Get expired pins that need review."""
        with self._lock:
            pins = list(self._pins.values())

        if workspace_id:
            pins = [p for p in pins if p.workspace_id == workspace_id]

        return [p for p in pins if p.active and p.is_expired()]

    def validate_digest_reference(self, reference: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that a reference is a proper digest, not a tag like "latest".

        Returns: (is_valid, error_message)
        """
        if reference == "latest":
            return False, "'latest' tag is not allowed. Use digest pinning."

        if reference.startswith("sha256:"):
            if DIGEST_PATTERN.match(reference):
                return True, None
            return False, "Invalid sha256 digest format"

        # Check if it looks like a tag
        if TAG_PATTERN.match(reference):
            return False, f"Tag reference '{reference}' is not allowed. Use sha256 digest."

        return False, f"Invalid reference format: {reference}"

    # =========================================================================
    # Registry Allowlist
    # =========================================================================

    def set_registry_allowlist(
        self,
        workspace_id: str,
        registries: List[RegistryConfig],
        updated_by: str,
        deny_all_by_default: bool = True,
    ) -> RegistryAllowlist:
        """Set the registry allowlist for a workspace."""
        allowlist = RegistryAllowlist(
            workspace_id=workspace_id,
            registries=registries,
            deny_all_by_default=deny_all_by_default,
            updated_by=updated_by,
        )

        with self._lock:
            self._allowlists[workspace_id] = allowlist

        # Log event for each registry
        for registry in registries:
            event = SupplyChainEvent(
                workspace_id=workspace_id,
                event_type=SupplyChainEventType.REGISTRY_ADDED,
                actor_id=updated_by,
                details={
                    "hostname": registry.hostname,
                    "require_signature": registry.require_signature,
                    "require_sbom": registry.require_sbom,
                    "eu_only": registry.eu_only,
                },
            )
            self._log_event(event)

        return allowlist

    def add_registry_to_allowlist(
        self,
        workspace_id: str,
        registry: RegistryConfig,
        added_by: str,
    ) -> RegistryAllowlist:
        """Add a registry to the workspace allowlist."""
        with self._lock:
            allowlist = self._allowlists.get(workspace_id)
            if not allowlist:
                allowlist = RegistryAllowlist(
                    workspace_id=workspace_id,
                    updated_by=added_by,
                )
                self._allowlists[workspace_id] = allowlist

            registry.added_by = added_by
            registry.added_at = datetime.now(timezone.utc)
            allowlist.registries.append(registry)
            allowlist.updated_at = datetime.now(timezone.utc)
            allowlist.updated_by = added_by

        # Log event
        event = SupplyChainEvent(
            workspace_id=workspace_id,
            event_type=SupplyChainEventType.REGISTRY_ADDED,
            actor_id=added_by,
            details={
                "hostname": registry.hostname,
                "require_signature": registry.require_signature,
            },
        )
        self._log_event(event)

        return allowlist

    def remove_registry_from_allowlist(
        self,
        workspace_id: str,
        registry_id: str,
        removed_by: str,
    ) -> Optional[RegistryAllowlist]:
        """Remove a registry from the allowlist."""
        with self._lock:
            allowlist = self._allowlists.get(workspace_id)
            if not allowlist:
                return None

            original_count = len(allowlist.registries)
            allowlist.registries = [r for r in allowlist.registries if r.registry_id != registry_id]

            if len(allowlist.registries) < original_count:
                allowlist.updated_at = datetime.now(timezone.utc)
                allowlist.updated_by = removed_by

                # Log event
                event = SupplyChainEvent(
                    workspace_id=workspace_id,
                    event_type=SupplyChainEventType.REGISTRY_REMOVED,
                    actor_id=removed_by,
                    details={"registry_id": registry_id},
                )
                self._log_event(event)

            return allowlist

    def get_registry_allowlist(self, workspace_id: str) -> Optional[RegistryAllowlist]:
        """Get the registry allowlist for a workspace."""
        with self._lock:
            return self._allowlists.get(workspace_id)

    def is_registry_allowed(
        self,
        workspace_id: str,
        registry: str,
        repository: str,
    ) -> Tuple[bool, Optional[RegistryConfig]]:
        """Check if a registry is allowed for a workspace."""
        allowlist = self.get_registry_allowlist(workspace_id)
        if not allowlist:
            # No allowlist = deny all by default
            return False, None

        return allowlist.is_registry_allowed(registry, repository)

    # =========================================================================
    # Trusted Signers
    # =========================================================================

    def add_trusted_signer(
        self,
        name: str,
        public_key: str,
        algorithm: SigningAlgorithm,
        created_by: str,
        valid_days: Optional[int] = None,
        description: str = "",
    ) -> TrustedSigner:
        """Add a trusted signer."""
        fingerprint = f"sha256:{hashlib.sha256(public_key.encode()).hexdigest()[:40]}"

        valid_until = None
        if valid_days:
            valid_until = datetime.now(timezone.utc) + timedelta(days=valid_days)

        signer = TrustedSigner(
            name=name,
            public_key=public_key,
            algorithm=algorithm,
            fingerprint=fingerprint,
            valid_until=valid_until,
            created_by=created_by,
            description=description,
        )

        with self._lock:
            self._signers[signer.signer_id] = signer

        return signer

    def revoke_signer(
        self,
        signer_id: str,
        revoked_by: str,
        reason: str,
    ) -> Optional[TrustedSigner]:
        """Revoke a trusted signer."""
        with self._lock:
            signer = self._signers.get(signer_id)
            if not signer:
                return None

            signer.revoked = True
            signer.revoked_at = datetime.now(timezone.utc)
            signer.revocation_reason = reason

        return signer

    def get_signer(self, signer_id: str) -> Optional[TrustedSigner]:
        """Get a trusted signer."""
        with self._lock:
            return self._signers.get(signer_id)

    def list_signers(self, valid_only: bool = False) -> List[TrustedSigner]:
        """List trusted signers."""
        with self._lock:
            signers = list(self._signers.values())

        if valid_only:
            signers = [s for s in signers if s.is_valid()]

        return signers

    # =========================================================================
    # SBOM Management
    # =========================================================================

    def upload_sbom(
        self,
        workspace_id: str,
        artifact_digest: str,
        sbom_format: SBOMFormat,
        components: List[SBOMComponent],
        created_by: str,
        tool_name: str = "",
        tool_version: str = "",
        vulnerabilities: Optional[List[Vulnerability]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SBOM:
        """Upload an SBOM for an artifact."""
        if not DIGEST_PATTERN.match(artifact_digest):
            raise ValueError(f"Invalid artifact digest format: {artifact_digest}")

        sbom = SBOM(
            artifact_digest=artifact_digest,
            format=sbom_format,
            created_by=created_by,
            tool_name=tool_name,
            tool_version=tool_version,
            components=components,
            vulnerabilities=vulnerabilities or [],
            metadata=metadata or {},
        )

        with self._lock:
            self._sboms[sbom.sbom_id] = sbom
            self._sboms_by_digest[artifact_digest] = sbom.sbom_id

            # Update artifact with SBOM reference
            artifact_id = self._artifacts_by_digest.get(artifact_digest)
            if artifact_id:
                artifact = self._artifacts.get(artifact_id)
                if artifact:
                    artifact.sbom_digest = sbom.sbom_id

        # Log event
        event = SupplyChainEvent(
            workspace_id=workspace_id,
            event_type=SupplyChainEventType.SBOM_UPLOADED,
            actor_id=created_by,
            artifact_digest=artifact_digest,
            details={
                "format": sbom_format.value,
                "components_count": len(components),
                "vulnerabilities_count": len(vulnerabilities) if vulnerabilities else 0,
            },
        )
        self._log_event(event)

        # Log vulnerabilities if any
        if vulnerabilities:
            for vuln in vulnerabilities:
                if vuln.severity in (VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH):
                    vuln_event = SupplyChainEvent(
                        workspace_id=workspace_id,
                        event_type=SupplyChainEventType.VULNERABILITY_DETECTED,
                        actor_id="system",
                        artifact_digest=artifact_digest,
                        details={
                            "vuln_id": vuln.vuln_id,
                            "severity": vuln.severity.value,
                            "cvss_score": vuln.cvss_score,
                            "affected_component": vuln.affected_component,
                        },
                    )
                    self._log_event(vuln_event)

        return sbom

    def get_sbom(self, sbom_id: str) -> Optional[SBOM]:
        """Get SBOM by ID."""
        with self._lock:
            return self._sboms.get(sbom_id)

    def get_sbom_by_artifact_digest(self, artifact_digest: str) -> Optional[SBOM]:
        """Get SBOM by artifact digest."""
        with self._lock:
            sbom_id = self._sboms_by_digest.get(artifact_digest)
            if sbom_id:
                return self._sboms.get(sbom_id)
        return None

    def list_sboms(
        self,
        workspace_id: Optional[str] = None,
        sbom_format: Optional[SBOMFormat] = None,
    ) -> List[SBOM]:
        """List SBOMs."""
        with self._lock:
            sboms = list(self._sboms.values())

        if sbom_format:
            sboms = [s for s in sboms if s.format == sbom_format]

        return sboms

    def add_vulnerability_to_sbom(
        self,
        sbom_id: str,
        vulnerability: Vulnerability,
        workspace_id: str,
    ) -> Optional[SBOM]:
        """Add a vulnerability to an SBOM."""
        with self._lock:
            sbom = self._sboms.get(sbom_id)
            if not sbom:
                return None

            sbom.vulnerabilities.append(vulnerability)
            sbom.integrity_hash = sbom._compute_hash()

        # Log event
        event = SupplyChainEvent(
            workspace_id=workspace_id,
            event_type=SupplyChainEventType.VULNERABILITY_DETECTED,
            actor_id="system",
            artifact_digest=sbom.artifact_digest,
            details={
                "vuln_id": vulnerability.vuln_id,
                "severity": vulnerability.severity.value,
                "cvss_score": vulnerability.cvss_score,
            },
        )
        self._log_event(event)

        return sbom

    def resolve_vulnerability(
        self,
        sbom_id: str,
        vuln_id: str,
        workspace_id: str,
        resolved_by: str,
    ) -> Optional[SBOM]:
        """Mark a vulnerability as resolved."""
        with self._lock:
            sbom = self._sboms.get(sbom_id)
            if not sbom:
                return None

            for vuln in sbom.vulnerabilities:
                if vuln.vuln_id == vuln_id:
                    vuln.resolved = True
                    vuln.resolved_at = datetime.now(timezone.utc)
                    break

            sbom.integrity_hash = sbom._compute_hash()

        # Log event
        event = SupplyChainEvent(
            workspace_id=workspace_id,
            event_type=SupplyChainEventType.VULNERABILITY_RESOLVED,
            actor_id=resolved_by,
            artifact_digest=sbom.artifact_digest,
            details={"vuln_id": vuln_id},
        )
        self._log_event(event)

        return sbom

    # =========================================================================
    # Statistics and Export
    # =========================================================================

    def get_stats(self, workspace_id: str) -> SupplyChainStats:
        """Get supply chain statistics for a workspace."""
        artifacts = self.list_artifacts()
        pins = self.list_pins(workspace_id=workspace_id)
        expired_pins = self.get_expired_pins(workspace_id=workspace_id)
        sboms = self.list_sboms()
        signers = self.list_signers(valid_only=True)
        allowlist = self.get_registry_allowlist(workspace_id)

        # Calculate vulnerability counts
        vuln_counts = {s.value: 0 for s in VulnerabilitySeverity}
        total_components = 0
        for sbom in sboms:
            total_components += len(sbom.components)
            summary = sbom.get_vulnerability_summary()
            for sev, count in summary.items():
                vuln_counts[sev] += count

        # Count policy violations in last 30 days
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        with self._lock:
            violations = [
                e
                for e in self._events
                if e.workspace_id == workspace_id
                and e.event_type == SupplyChainEventType.POLICY_VIOLATION
                and e.timestamp >= cutoff
            ]

        return SupplyChainStats(
            workspace_id=workspace_id,
            total_artifacts=len(artifacts),
            signed_artifacts=len([a for a in artifacts if a.signature]),
            unsigned_artifacts=len([a for a in artifacts if not a.signature]),
            pinned_digests=len(pins),
            expired_pins=len(expired_pins),
            total_sboms=len(sboms),
            total_components=total_components,
            vulnerabilities_critical=vuln_counts.get("critical", 0),
            vulnerabilities_high=vuln_counts.get("high", 0),
            vulnerabilities_medium=vuln_counts.get("medium", 0),
            vulnerabilities_low=vuln_counts.get("low", 0),
            allowed_registries=len(allowlist.registries) if allowlist else 0,
            trusted_signers=len(signers),
            policy_violations_30d=len(violations),
        )

    def _log_event(self, event: SupplyChainEvent) -> None:
        """Log a supply chain event."""
        with self._lock:
            self._events.append(event)

        if self._on_event:
            try:
                self._on_event(event)
            except Exception:
                pass

    def get_events(
        self,
        workspace_id: Optional[str] = None,
        event_type: Optional[SupplyChainEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[SupplyChainEvent]:
        """Get supply chain events."""
        with self._lock:
            events = list(self._events)

        if workspace_id:
            events = [e for e in events if e.workspace_id == workspace_id]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def export_inventory(
        self,
        workspace_id: str,
        include_sbom: bool = True,
        include_vulnerabilities: bool = True,
    ) -> Dict[str, Any]:
        """Export artifact inventory for evidence pack."""
        artifacts = self.list_artifacts()
        pins = self.list_pins(workspace_id=workspace_id)
        signers = self.list_signers()
        allowlist = self.get_registry_allowlist(workspace_id)

        export_data: Dict[str, Any] = {
            "workspace_id": workspace_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": [a.to_dict() for a in artifacts],
            "digest_pins": [p.to_dict() for p in pins],
            "trusted_signers": [s.to_dict() for s in signers],
            "registry_allowlist": allowlist.to_dict() if allowlist else None,
        }

        if include_sbom:
            sboms = self.list_sboms()
            export_data["sboms"] = []
            for sbom in sboms:
                sbom_dict = sbom.to_dict()
                if not include_vulnerabilities:
                    sbom_dict["vulnerabilities"] = []
                export_data["sboms"].append(sbom_dict)

        # Compute export hash
        content = json.dumps(export_data, sort_keys=True, default=str)
        export_data["integrity_hash"] = f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

        return export_data
