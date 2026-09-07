# -*- coding: utf-8 -*-
"""
Artifact Publisher - Publishes artifacts to registry with digest pinning.

CLOUD ZONE ONLY.

Phase 3 (3.1): End-to-end supply chain publishing.
- Publishes built artifacts to registry
- Ensures digest pinning
- Handles both local and remote registries

Design Doc:
- Artifact pinned by digest (immutable)
- Registry stores metadata + blob reference
- Agent pulls by digest (never "latest")
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Set
from uuid import UUID, uuid4

from packages.shared.contracts.manifest import ArtifactFormat, ArtifactManifest


class RegistryType(str, Enum):
    """Supported registry types."""
    LOCAL = "local"           # Local filesystem
    S3 = "s3"                 # S3-compatible object store
    OCI = "oci"               # OCI-compatible registry (e.g., Docker Hub)
    DATABASE = "database"     # Database-backed (for metadata)


class PublishStatus(str, Enum):
    """Artifact publish status."""
    PENDING = "pending"
    UPLOADING = "uploading"
    VERIFYING = "verifying"
    PUBLISHED = "published"
    FAILED = "failed"


@dataclass
class RegistryConfig:
    """Registry configuration."""
    registry_type: RegistryType = RegistryType.LOCAL
    registry_url: str = ""
    registry_name: str = "ccea-registry"

    # Local registry settings
    local_path: Optional[Path] = None

    # S3 settings
    s3_bucket: Optional[str] = None
    s3_prefix: str = "artifacts/"
    s3_region: str = "us-east-1"

    # OCI settings
    oci_repository: Optional[str] = None

    # Authentication (never stored in cloud - passed at runtime)
    # These are for registry auth, NOT broker credentials
    auth_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (safe, no secrets)."""
        return {
            "registry_type": self.registry_type.value,
            "registry_url": self.registry_url,
            "registry_name": self.registry_name,
            "local_path": str(self.local_path) if self.local_path else None,
            "s3_bucket": self.s3_bucket,
            "s3_prefix": self.s3_prefix,
            "s3_region": self.s3_region,
            "oci_repository": self.oci_repository,
            "auth_required": self.auth_required,
        }


@dataclass
class PublishResult:
    """Result of artifact publishing."""
    success: bool = False
    artifact_digest: str = ""
    manifest_digest: str = ""
    registry_url: str = ""
    registry_ref: str = ""  # Full reference: registry/name@digest
    status: PublishStatus = PublishStatus.PENDING
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    published_at: Optional[datetime] = None
    upload_time_ms: int = 0
    artifact_size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "artifact_digest": self.artifact_digest,
            "manifest_digest": self.manifest_digest,
            "registry_url": self.registry_url,
            "registry_ref": self.registry_ref,
            "status": self.status.value,
            "errors": self.errors,
            "warnings": self.warnings,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "upload_time_ms": self.upload_time_ms,
            "artifact_size_bytes": self.artifact_size_bytes,
        }


@dataclass
class ArtifactReference:
    """
    Artifact reference - how to fetch an artifact.

    Always uses digest for immutable reference.
    """
    registry_name: str
    artifact_digest: str  # sha256:...
    manifest_digest: str  # sha256:...
    strategy_id: str
    version: str
    format: ArtifactFormat
    artifact_url: Optional[str] = None  # Direct URL if available

    @property
    def digest_ref(self) -> str:
        """Get digest reference (registry/strategy@digest)."""
        return f"{self.registry_name}/{self.strategy_id}@{self.artifact_digest}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "registry_name": self.registry_name,
            "artifact_digest": self.artifact_digest,
            "manifest_digest": self.manifest_digest,
            "strategy_id": self.strategy_id,
            "version": self.version,
            "format": self.format.value,
            "artifact_url": self.artifact_url,
            "digest_ref": self.digest_ref,
        }


class ArtifactPublisher:
    """
    Publishes artifacts to registry with digest pinning.

    Design Doc Phase 3 (3.1):
    - Build → Sign → Publish → Pull → Verify → Run
    - Artifact accessible by digest ref
    - No "latest" tags in production

    Usage:
        publisher = ArtifactPublisher(config)

        # Publish built artifact
        result = publisher.publish(
            artifact_path=Path("/path/to/artifact.zip"),
            manifest=manifest,
        )

        if result.success:
            print(f"Published: {result.registry_ref}")
    """

    def __init__(self, config: Optional[RegistryConfig] = None):
        """Initialize publisher."""
        self._config = config or RegistryConfig()

        # Ensure local path exists for local registry
        if self._config.registry_type == RegistryType.LOCAL:
            if self._config.local_path is None:
                self._config.local_path = Path.home() / ".ccea" / "registry"
            self._config.local_path.mkdir(parents=True, exist_ok=True)

    def publish(
        self,
        artifact_path: Path,
        manifest: ArtifactManifest,
        sbom_path: Optional[Path] = None,
    ) -> PublishResult:
        """
        Publish artifact to registry.

        Args:
            artifact_path: Path to artifact file (ZIP, OCI tarball, etc.)
            manifest: Artifact manifest
            sbom_path: Optional path to SBOM file

        Returns:
            PublishResult with details
        """
        import time
        start_time = time.time()

        result = PublishResult()
        result.status = PublishStatus.UPLOADING

        try:
            # Validate artifact exists
            if not artifact_path.exists():
                result.errors.append(f"Artifact not found: {artifact_path}")
                result.status = PublishStatus.FAILED
                return result

            # Verify manifest has required fields
            if not manifest.artifact_digest:
                result.errors.append("Manifest missing artifact_digest")
                result.status = PublishStatus.FAILED
                return result

            # Verify signature exists (per Design Doc - unsigned not allowed)
            if not manifest.has_valid_signature():
                result.errors.append(
                    "Artifact must be signed before publishing. "
                    "Unsigned artifacts are not allowed."
                )
                result.status = PublishStatus.FAILED
                return result

            # Compute and verify digest
            actual_digest = self._compute_file_digest(artifact_path)
            if actual_digest != manifest.artifact_digest:
                result.errors.append(
                    f"Digest mismatch: manifest says {manifest.artifact_digest}, "
                    f"actual is {actual_digest}"
                )
                result.status = PublishStatus.FAILED
                return result

            result.artifact_digest = actual_digest
            result.artifact_size_bytes = artifact_path.stat().st_size

            # Publish based on registry type
            result.status = PublishStatus.VERIFYING

            if self._config.registry_type == RegistryType.LOCAL:
                self._publish_local(artifact_path, manifest, sbom_path, result)
            elif self._config.registry_type == RegistryType.S3:
                self._publish_s3(artifact_path, manifest, sbom_path, result)
            elif self._config.registry_type == RegistryType.OCI:
                self._publish_oci(artifact_path, manifest, sbom_path, result)
            else:
                result.errors.append(f"Unsupported registry type: {self._config.registry_type}")
                result.status = PublishStatus.FAILED
                return result

            if result.errors:
                result.status = PublishStatus.FAILED
                return result

            # Success
            result.success = True
            result.status = PublishStatus.PUBLISHED
            result.published_at = datetime.now(timezone.utc)
            result.registry_url = self._config.registry_url or str(self._config.local_path)
            result.registry_ref = f"{self._config.registry_name}/{manifest.strategy_id}@{actual_digest}"

        except Exception as e:
            result.errors.append(f"Publish failed: {str(e)}")
            result.status = PublishStatus.FAILED

        result.upload_time_ms = int((time.time() - start_time) * 1000)
        return result

    def get_artifact(
        self,
        digest: str,
        strategy_id: str,
        output_path: Path,
    ) -> Optional[ArtifactReference]:
        """
        Get artifact by digest.

        Args:
            digest: Artifact digest (sha256:...)
            strategy_id: Strategy identifier
            output_path: Where to save the artifact

        Returns:
            ArtifactReference if found, None otherwise
        """
        if self._config.registry_type == RegistryType.LOCAL:
            return self._get_local(digest, strategy_id, output_path)
        elif self._config.registry_type == RegistryType.S3:
            return self._get_s3(digest, strategy_id, output_path)
        elif self._config.registry_type == RegistryType.OCI:
            return self._get_oci(digest, strategy_id, output_path)
        return None

    def artifact_exists(self, digest: str, strategy_id: str) -> bool:
        """Check if artifact exists in registry."""
        if self._config.registry_type == RegistryType.LOCAL:
            artifact_dir = self._get_artifact_dir(strategy_id, digest)
            return artifact_dir.exists() and (artifact_dir / "artifact.zip").exists()
        # For S3/OCI, would need actual implementation
        return False

    def list_versions(self, strategy_id: str) -> List[ArtifactReference]:
        """List all versions of a strategy in registry."""
        refs = []

        if self._config.registry_type == RegistryType.LOCAL:
            strategy_dir = self._config.local_path / "artifacts" / strategy_id
            if strategy_dir.exists():
                for digest_dir in strategy_dir.iterdir():
                    if digest_dir.is_dir():
                        manifest_path = digest_dir / "manifest.json"
                        if manifest_path.exists():
                            try:
                                with open(manifest_path) as f:
                                    manifest_data = json.load(f)
                                refs.append(ArtifactReference(
                                    registry_name=self._config.registry_name,
                                    artifact_digest=manifest_data.get("artifact_digest", ""),
                                    manifest_digest=self._compute_file_digest(manifest_path),
                                    strategy_id=strategy_id,
                                    version=manifest_data.get("version", ""),
                                    format=ArtifactFormat(manifest_data.get("format", "zip_bundle")),
                                ))
                            except Exception:
                                continue

        return refs

    # ========================================================================
    # Private Methods
    # ========================================================================

    def _compute_file_digest(self, file_path: Path) -> str:
        """Compute SHA256 digest of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return f"sha256:{sha256.hexdigest()}"

    def _compute_json_digest(self, data: Dict[str, Any]) -> str:
        """Compute SHA256 digest of JSON data."""
        content = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def _get_artifact_dir(self, strategy_id: str, digest: str) -> Path:
        """Get artifact directory path."""
        # Sanitize digest for filesystem
        safe_digest = digest.replace(":", "_")
        return self._config.local_path / "artifacts" / strategy_id / safe_digest

    def _publish_local(
        self,
        artifact_path: Path,
        manifest: ArtifactManifest,
        sbom_path: Optional[Path],
        result: PublishResult,
    ) -> None:
        """Publish to local filesystem registry."""
        artifact_dir = self._get_artifact_dir(
            manifest.strategy_id,
            manifest.artifact_digest,
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Copy artifact
        dest_artifact = artifact_dir / "artifact.zip"
        shutil.copy2(artifact_path, dest_artifact)

        # Write manifest
        manifest_path = artifact_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        result.manifest_digest = self._compute_file_digest(manifest_path)

        # Copy SBOM if provided
        if sbom_path and sbom_path.exists():
            shutil.copy2(sbom_path, artifact_dir / "sbom.json")

        # Write metadata
        metadata = {
            "strategy_id": manifest.strategy_id,
            "version": manifest.version,
            "artifact_digest": manifest.artifact_digest,
            "manifest_digest": result.manifest_digest,
            "published_at": datetime.now(timezone.utc).isoformat(),
            "format": manifest.format.value,
            "signed": manifest.has_valid_signature(),
        }
        with open(artifact_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _get_local(
        self,
        digest: str,
        strategy_id: str,
        output_path: Path,
    ) -> Optional[ArtifactReference]:
        """Get artifact from local registry."""
        artifact_dir = self._get_artifact_dir(strategy_id, digest)
        artifact_file = artifact_dir / "artifact.zip"
        manifest_file = artifact_dir / "manifest.json"

        if not artifact_file.exists() or not manifest_file.exists():
            return None

        # Copy to output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact_file, output_path)

        # Load manifest
        with open(manifest_file) as f:
            manifest_data = json.load(f)

        return ArtifactReference(
            registry_name=self._config.registry_name,
            artifact_digest=digest,
            manifest_digest=self._compute_file_digest(manifest_file),
            strategy_id=strategy_id,
            version=manifest_data.get("version", ""),
            format=ArtifactFormat(manifest_data.get("format", "zip_bundle")),
            artifact_url=str(artifact_file),
        )

    def _publish_s3(
        self,
        artifact_path: Path,
        manifest: ArtifactManifest,
        sbom_path: Optional[Path],
        result: PublishResult,
    ) -> None:
        """Publish to S3-compatible storage."""
        # Placeholder - would use boto3 in production
        result.warnings.append("S3 publishing not fully implemented - using local fallback")
        self._publish_local(artifact_path, manifest, sbom_path, result)

    def _get_s3(
        self,
        digest: str,
        strategy_id: str,
        output_path: Path,
    ) -> Optional[ArtifactReference]:
        """Get artifact from S3."""
        # Placeholder
        return self._get_local(digest, strategy_id, output_path)

    def _publish_oci(
        self,
        artifact_path: Path,
        manifest: ArtifactManifest,
        sbom_path: Optional[Path],
        result: PublishResult,
    ) -> None:
        """Publish to OCI registry."""
        # Placeholder - would use OCI client in production
        result.warnings.append("OCI publishing not fully implemented - using local fallback")
        self._publish_local(artifact_path, manifest, sbom_path, result)

    def _get_oci(
        self,
        digest: str,
        strategy_id: str,
        output_path: Path,
    ) -> Optional[ArtifactReference]:
        """Get artifact from OCI registry."""
        # Placeholder
        return self._get_local(digest, strategy_id, output_path)


def create_artifact_reference(
    manifest: ArtifactManifest,
    registry_name: str = "ccea-registry",
) -> ArtifactReference:
    """
    Create artifact reference from manifest.

    Args:
        manifest: Artifact manifest
        registry_name: Registry name

    Returns:
        ArtifactReference
    """
    return ArtifactReference(
        registry_name=registry_name,
        artifact_digest=manifest.artifact_digest,
        manifest_digest="",  # Computed on publish
        strategy_id=manifest.strategy_id,
        version=manifest.version,
        format=manifest.format,
    )
