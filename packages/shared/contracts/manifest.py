# -*- coding: utf-8 -*-
"""
Artifact Manifest Contracts.

Defines the manifest structure for strategy artifacts.
Manifest contains metadata about the artifact, including:
- Identity and versioning
- Runtime requirements
- Security information (signatures, digests)
- Provenance (build info)

IMPORTANT: Manifest is used by BOTH Cloud (builder) and Agent (verifier).
This is a SHARED contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Set
from uuid import UUID, uuid4


class ArtifactFormat(str, Enum):
    """Format of the artifact."""

    OCI_IMAGE = "oci_image"  # Preferred
    ZIP_BUNDLE = "zip_bundle"  # Fallback
    WHEEL = "wheel"  # Python package


class SandboxType(str, Enum):
    """Type of sandbox for artifact execution."""

    PROCESS = "process"  # Default process-level isolation
    CONTAINER = "container"  # Container-based isolation
    VM = "vm"  # VM-level isolation (enterprise)
    WASM = "wasm"  # WebAssembly sandbox


class Timeframe(str, Enum):
    """Trading timeframe."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class SignatureAlgorithm(str, Enum):
    """
    Signature algorithm used.

    Design Doc: Actual signing uses Ed25519 via ccea.crypto.signing.
    ED25519 is the default and recommended algorithm.
    """

    ED25519 = "ed25519"  # Default: Ed25519 via ccea.crypto.signing
    SIGSTORE = "sigstore"  # Future: Sigstore/Cosign integration
    GPG = "gpg"  # Enterprise/offline
    NONE = "none"  # Only for development


# Minimum schema version this code supports
MIN_SCHEMA_VERSION: Final[str] = "1.0.0"
MAX_SCHEMA_VERSION: Final[str] = "1.99.99"
CURRENT_SCHEMA_VERSION: Final[str] = "1.0.0"


@dataclass
class Signature:
    """
    Artifact signature information.
    """

    algorithm: SignatureAlgorithm = SignatureAlgorithm.ED25519
    signature_value: str = ""
    certificate: Optional[str] = None
    timestamp: Optional[datetime] = None
    key_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "algorithm": self.algorithm.value,
            "signature_value": self.signature_value,
            "certificate": self.certificate,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "key_id": self.key_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Signature:
        """Create from dictionary."""
        algo_str = data.get("algorithm", "ed25519")
        # Handle backwards compatibility: treat "sigstore" as "ed25519" if it's the default
        try:
            algorithm = SignatureAlgorithm(algo_str)
        except ValueError:
            algorithm = SignatureAlgorithm.ED25519

        return cls(
            algorithm=algorithm,
            signature_value=data.get("signature_value", ""),
            certificate=data.get("certificate"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
            key_id=data.get("key_id"),
        )


@dataclass
class Provenance:
    """
    Build provenance information.

    Tracks where and how the artifact was built.
    """

    # Source info
    git_repo: str = ""
    git_sha: str = ""
    git_branch: str = ""
    git_tag: Optional[str] = None

    # Build info
    builder_id: str = ""
    build_timestamp: datetime = field(default_factory=datetime.utcnow)
    build_host: str = ""
    ci_job_id: Optional[str] = None
    ci_pipeline_url: Optional[str] = None

    # Training info (for ML models)
    training_run_id: Optional[str] = None
    dataset_refs: List[str] = field(default_factory=list)
    params_hash: Optional[str] = None

    # Dependencies
    deps_lock_digest: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "git_repo": self.git_repo,
            "git_sha": self.git_sha,
            "git_branch": self.git_branch,
            "git_tag": self.git_tag,
            "builder_id": self.builder_id,
            "build_timestamp": self.build_timestamp.isoformat(),
            "build_host": self.build_host,
            "ci_job_id": self.ci_job_id,
            "ci_pipeline_url": self.ci_pipeline_url,
            "training_run_id": self.training_run_id,
            "dataset_refs": self.dataset_refs,
            "params_hash": self.params_hash,
            "deps_lock_digest": self.deps_lock_digest,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Provenance:
        """Create from dictionary."""
        return cls(
            git_repo=data.get("git_repo", ""),
            git_sha=data.get("git_sha", ""),
            git_branch=data.get("git_branch", ""),
            git_tag=data.get("git_tag"),
            builder_id=data.get("builder_id", ""),
            build_timestamp=(
                datetime.fromisoformat(data["build_timestamp"])
                if "build_timestamp" in data
                else datetime.utcnow()
            ),
            build_host=data.get("build_host", ""),
            ci_job_id=data.get("ci_job_id"),
            ci_pipeline_url=data.get("ci_pipeline_url"),
            training_run_id=data.get("training_run_id"),
            dataset_refs=data.get("dataset_refs", []),
            params_hash=data.get("params_hash"),
            deps_lock_digest=data.get("deps_lock_digest", ""),
        )


@dataclass
class RuntimeRequirements:
    """
    Runtime requirements for the artifact.

    Agent uses this to validate environment before loading artifact.
    """

    # Python version
    python_version: str = ">=3.10"

    # System requirements
    min_memory_mb: int = 512
    min_cpu_cores: int = 1
    gpu_required: bool = False

    # Sandbox requirements
    requires_network: bool = False
    requires_filesystem: bool = False
    egress_domains: List[str] = field(default_factory=list)

    # Broker requirements (for live execution)
    requires_broker_access: bool = False
    supported_brokers: List[str] = field(default_factory=list)

    # Schema version compatibility
    min_schema_version: str = MIN_SCHEMA_VERSION
    max_schema_version: str = MAX_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "python_version": self.python_version,
            "min_memory_mb": self.min_memory_mb,
            "min_cpu_cores": self.min_cpu_cores,
            "gpu_required": self.gpu_required,
            "requires_network": self.requires_network,
            "requires_filesystem": self.requires_filesystem,
            "egress_domains": self.egress_domains,
            "requires_broker_access": self.requires_broker_access,
            "supported_brokers": self.supported_brokers,
            "min_schema_version": self.min_schema_version,
            "max_schema_version": self.max_schema_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RuntimeRequirements:
        """Create from dictionary."""
        return cls(
            python_version=data.get("python_version", ">=3.10"),
            min_memory_mb=data.get("min_memory_mb", 512),
            min_cpu_cores=data.get("min_cpu_cores", 1),
            gpu_required=data.get("gpu_required", False),
            requires_network=data.get("requires_network", False),
            requires_filesystem=data.get("requires_filesystem", False),
            egress_domains=data.get("egress_domains", []),
            requires_broker_access=data.get("requires_broker_access", False),
            supported_brokers=data.get("supported_brokers", []),
            min_schema_version=data.get("min_schema_version", MIN_SCHEMA_VERSION),
            max_schema_version=data.get("max_schema_version", MAX_SCHEMA_VERSION),
        )


@dataclass
class NetworkPermissions:
    """
    Network permission configuration.

    Design Doc: Sandbox network permissions.
    """

    enabled: bool = False
    egress_allowlist: List[str] = field(default_factory=list)
    max_connections: int = 10
    timeout_seconds: int = 30

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "egress_allowlist": self.egress_allowlist,
            "max_connections": self.max_connections,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NetworkPermissions":
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            egress_allowlist=data.get("egress_allowlist", []),
            max_connections=data.get("max_connections", 10),
            timeout_seconds=data.get("timeout_seconds", 30),
        )


@dataclass
class FilesystemPermissions:
    """
    Filesystem permission configuration.

    Design Doc: Sandbox filesystem permissions.
    """

    readonly: bool = True
    allowed_paths: List[str] = field(default_factory=list)
    temp_dir_allowed: bool = True
    max_file_size_mb: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "readonly": self.readonly,
            "allowed_paths": self.allowed_paths,
            "temp_dir_allowed": self.temp_dir_allowed,
            "max_file_size_mb": self.max_file_size_mb,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FilesystemPermissions":
        """Create from dictionary."""
        return cls(
            readonly=data.get("readonly", True),
            allowed_paths=data.get("allowed_paths", []),
            temp_dir_allowed=data.get("temp_dir_allowed", True),
            max_file_size_mb=data.get("max_file_size_mb", 100),
        )


@dataclass
class ResourcePermissions:
    """
    Resource limit configuration.

    Design Doc: Sandbox resource limits.
    """

    max_memory_mb: int = 512
    max_cpu_percent: float = 100.0
    max_execution_time_seconds: int = 3600
    max_threads: int = 4

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_percent": self.max_cpu_percent,
            "max_execution_time_seconds": self.max_execution_time_seconds,
            "max_threads": self.max_threads,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourcePermissions":
        """Create from dictionary."""
        return cls(
            max_memory_mb=data.get("max_memory_mb", 512),
            max_cpu_percent=data.get("max_cpu_percent", 100.0),
            max_execution_time_seconds=data.get("max_execution_time_seconds", 3600),
            max_threads=data.get("max_threads", 4),
        )


@dataclass
class Permissions:
    """
    Complete permission block for artifact manifest.

    Design Doc: Unified permissions contract between Cloud (builder) and Agent (verifier).
    This is the single source of truth for sandbox permissions.

    IMPORTANT: Agent enforces these permissions via SandboxPermissionsEnforcer.
    deny-by-default: If permissions block is missing, agent uses most restrictive defaults.
    """

    network: NetworkPermissions = field(default_factory=NetworkPermissions)
    filesystem: FilesystemPermissions = field(default_factory=FilesystemPermissions)
    resources: ResourcePermissions = field(default_factory=ResourcePermissions)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "network": self.network.to_dict(),
            "filesystem": self.filesystem.to_dict(),
            "resources": self.resources.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Permissions":
        """Create from dictionary."""
        return cls(
            network=NetworkPermissions.from_dict(data.get("network", {})),
            filesystem=FilesystemPermissions.from_dict(data.get("filesystem", {})),
            resources=ResourcePermissions.from_dict(data.get("resources", {})),
        )

    @classmethod
    def default_restrictive(cls) -> "Permissions":
        """Create default restrictive permissions (deny-by-default)."""
        return cls(
            network=NetworkPermissions(enabled=False, egress_allowlist=[]),
            filesystem=FilesystemPermissions(readonly=True, allowed_paths=[]),
            resources=ResourcePermissions(
                max_memory_mb=256,
                max_cpu_percent=50.0,
                max_execution_time_seconds=300,
            ),
        )


@dataclass
class RiskProfileSuggested:
    """
    Suggested risk profile from artifact.

    IMPORTANT: This is a SUGGESTION only. Agent's local policy
    has priority. Cloud cannot enforce risk limits.
    """

    max_position_size: Optional[str] = None
    max_daily_loss: Optional[str] = None
    max_drawdown_pct: Optional[str] = None
    risk_level: str = "moderate"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_position_size": self.max_position_size,
            "max_daily_loss": self.max_daily_loss,
            "max_drawdown_pct": self.max_drawdown_pct,
            "risk_level": self.risk_level,
        }


@dataclass
class ArtifactManifest:
    """
    Complete artifact manifest.

    This is the primary metadata structure for strategy artifacts.
    Agent verifies manifest before loading artifact.
    """

    # Schema version (for compatibility)
    schema_version: str = CURRENT_SCHEMA_VERSION

    # Identity
    artifact_id: UUID = field(default_factory=uuid4)
    strategy_id: str = ""
    strategy_name: str = ""
    version: str = "1.0.0"
    description: str = ""

    # Format and location
    format: ArtifactFormat = ArtifactFormat.OCI_IMAGE
    artifact_digest: str = ""  # SHA256 digest
    registry_url: str = ""
    entrypoint: str = ""

    # Security
    signature: Signature = field(default_factory=Signature)
    sbom_ref: Optional[str] = None  # Reference to SBOM

    # Provenance
    provenance: Provenance = field(default_factory=Provenance)

    # Runtime
    runtime: RuntimeRequirements = field(default_factory=RuntimeRequirements)

    # Permissions (Design Doc: unified contract for sandbox enforcement)
    permissions: Permissions = field(default_factory=Permissions)

    # Risk (suggested, not enforced)
    risk_profile_suggested: RiskProfileSuggested = field(default_factory=RiskProfileSuggested)

    # Data contracts
    data_contract: Dict[str, Any] = field(default_factory=dict)
    telemetry_schema_version: str = "1.0.0"

    # Change classification
    change_class: str = "trading_impacting"

    # Sandbox configuration (Design Doc 8.2)
    sandbox_type: SandboxType = SandboxType.PROCESS

    # Trading timeframe (Design Doc 8.2)
    timeframe: Optional[Timeframe] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    built_at: Optional[datetime] = None  # When artifact was built

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "schema_version": self.schema_version,
            "artifact_id": str(self.artifact_id),
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "version": self.version,
            "description": self.description,
            "format": self.format.value,
            "artifact_digest": self.artifact_digest,
            "registry_url": self.registry_url,
            "entrypoint": self.entrypoint,
            "signature": self.signature.to_dict(),
            "sbom_ref": self.sbom_ref,
            "provenance": self.provenance.to_dict(),
            "runtime": self.runtime.to_dict(),
            "permissions": self.permissions.to_dict(),
            "risk_profile_suggested": self.risk_profile_suggested.to_dict(),
            "data_contract": self.data_contract,
            "telemetry_schema_version": self.telemetry_schema_version,
            "change_class": self.change_class,
            "sandbox_type": self.sandbox_type.value,
            "timeframe": self.timeframe.value if self.timeframe else None,
            "created_at": self.created_at.isoformat(),
            "built_at": self.built_at.isoformat() if self.built_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ArtifactManifest:
        """Create from dictionary."""
        # Parse sandbox_type with fallback
        sandbox_type_str = data.get("sandbox_type", "process")
        try:
            sandbox_type = SandboxType(sandbox_type_str)
        except ValueError:
            sandbox_type = SandboxType.PROCESS

        # Parse timeframe with fallback
        timeframe_str = data.get("timeframe")
        timeframe = None
        if timeframe_str:
            try:
                timeframe = Timeframe(timeframe_str)
            except ValueError:
                pass

        return cls(
            schema_version=data.get("schema_version", CURRENT_SCHEMA_VERSION),
            artifact_id=UUID(data["artifact_id"]) if "artifact_id" in data else uuid4(),
            strategy_id=data.get("strategy_id", ""),
            strategy_name=data.get("strategy_name", ""),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            format=ArtifactFormat(data.get("format", "oci_image")),
            artifact_digest=data.get("artifact_digest", ""),
            registry_url=data.get("registry_url", ""),
            entrypoint=data.get("entrypoint", ""),
            signature=Signature.from_dict(data.get("signature", {})),
            sbom_ref=data.get("sbom_ref"),
            provenance=Provenance.from_dict(data.get("provenance", {})),
            runtime=RuntimeRequirements.from_dict(data.get("runtime", {})),
            permissions=Permissions.from_dict(data.get("permissions", {})),
            risk_profile_suggested=RiskProfileSuggested(**data.get("risk_profile_suggested", {})),
            data_contract=data.get("data_contract", {}),
            telemetry_schema_version=data.get("telemetry_schema_version", "1.0.0"),
            change_class=data.get("change_class", "trading_impacting"),
            sandbox_type=sandbox_type,
            timeframe=timeframe,
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else datetime.utcnow()
            ),
            built_at=datetime.fromisoformat(data["built_at"]) if data.get("built_at") else None,
        )

    def is_schema_compatible(
        self, min_version: str = MIN_SCHEMA_VERSION, max_version: str = MAX_SCHEMA_VERSION
    ) -> bool:
        """Check if manifest schema version is compatible."""
        from packaging import version

        manifest_ver = version.parse(self.schema_version)
        min_ver = version.parse(min_version)
        max_ver = version.parse(max_version)
        return min_ver <= manifest_ver <= max_ver

    def has_valid_signature(self) -> bool:
        """Check if manifest has a non-empty signature."""
        return self.signature.algorithm != SignatureAlgorithm.NONE and bool(
            self.signature.signature_value
        )


# Type alias
ManifestSchema = ArtifactManifest
