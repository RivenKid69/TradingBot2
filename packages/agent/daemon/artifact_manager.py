# -*- coding: utf-8 -*-
"""
Artifact Manager - Download, verify, and manage strategy artifacts.

AGENT ZONE ONLY - handles all artifact lifecycle operations:
- Download from Cloud with digest verification
- Local caching and version management
- Hot-swap preparation for strategy updates
- Manifest permission extraction

Design Doc Phase 4:
- Agent downloads artifact from presigned URL
- Verifies SHA-256 digest before use
- Extracts permissions from manifest.yaml
- Supports atomic swap for live updates
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Set
from uuid import uuid4

import yaml


# Constants
ARTIFACT_CACHE_DIR: Final[str] = ".ccea/artifacts"
MAX_ARTIFACT_SIZE_MB: Final[int] = 500
MANIFEST_FILENAME: Final[str] = "manifest.yaml"
SUPPORTED_FORMATS: Final[Set[str]] = {".tar.gz", ".tgz", ".zip", ".whl"}


class ArtifactType(Enum):
    """Types of artifacts."""
    STRATEGY = auto()
    MODEL = auto()
    CONFIG = auto()
    DATA = auto()


class ArtifactState(Enum):
    """Artifact lifecycle state."""
    DOWNLOADING = auto()
    VERIFYING = auto()
    EXTRACTING = auto()
    READY = auto()
    ACTIVE = auto()
    INACTIVE = auto()
    FAILED = auto()
    DELETED = auto()


@dataclass
class ArtifactManifest:
    """
    Parsed artifact manifest.

    Contains permissions and metadata from manifest.yaml.
    """
    name: str = ""
    version: str = ""
    artifact_type: ArtifactType = ArtifactType.STRATEGY
    entrypoint: str = ""

    # Permissions - CRITICAL for sandbox enforcement
    permissions: Dict[str, Any] = field(default_factory=dict)
    network_allowed: bool = False
    egress_allowlist: List[str] = field(default_factory=list)
    filesystem_readonly: bool = True
    allowed_paths: List[str] = field(default_factory=list)
    max_memory_mb: int = 512
    max_cpu_percent: float = 100.0
    max_execution_time_seconds: int = 3600

    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    python_version: str = ">=3.10"

    # Metadata
    author: str = ""
    description: str = ""
    created_at: Optional[datetime] = None
    checksum: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "artifact_type": self.artifact_type.name,
            "entrypoint": self.entrypoint,
            "permissions": self.permissions,
            "network_allowed": self.network_allowed,
            "egress_allowlist": self.egress_allowlist,
            "filesystem_readonly": self.filesystem_readonly,
            "allowed_paths": self.allowed_paths,
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_percent": self.max_cpu_percent,
            "max_execution_time_seconds": self.max_execution_time_seconds,
            "dependencies": self.dependencies,
            "python_version": self.python_version,
            "author": self.author,
            "description": self.description,
            "checksum": self.checksum,
        }

    @classmethod
    def from_yaml(cls, yaml_content: str) -> "ArtifactManifest":
        """Parse manifest from YAML content."""
        data = yaml.safe_load(yaml_content) or {}
        permissions = data.get("permissions", {})

        artifact_type_str = data.get("type", "strategy").upper()
        artifact_type = getattr(ArtifactType, artifact_type_str, ArtifactType.STRATEGY)

        return cls(
            name=data.get("name", ""),
            version=data.get("version", ""),
            artifact_type=artifact_type,
            entrypoint=data.get("entrypoint", ""),
            permissions=permissions,
            network_allowed=permissions.get("network", {}).get("enabled", False),
            egress_allowlist=permissions.get("network", {}).get("egress_allowlist", []),
            filesystem_readonly=permissions.get("filesystem", {}).get("readonly", True),
            allowed_paths=permissions.get("filesystem", {}).get("allowed_paths", []),
            max_memory_mb=permissions.get("resources", {}).get("max_memory_mb", 512),
            max_cpu_percent=permissions.get("resources", {}).get("max_cpu_percent", 100.0),
            max_execution_time_seconds=permissions.get("resources", {}).get(
                "max_execution_time_seconds", 3600
            ),
            dependencies=data.get("dependencies", []),
            python_version=data.get("python_version", ">=3.10"),
            author=data.get("author", ""),
            description=data.get("description", ""),
        )

    @classmethod
    def default_restrictive(cls) -> "ArtifactManifest":
        """Create default restrictive manifest (deny-by-default)."""
        return cls(
            name="unknown",
            version="0.0.0",
            network_allowed=False,
            egress_allowlist=[],
            filesystem_readonly=True,
            allowed_paths=[],
            max_memory_mb=256,
            max_cpu_percent=50.0,
            max_execution_time_seconds=300,
        )


@dataclass
class Artifact:
    """
    Represents a downloaded and verified artifact.
    """
    artifact_id: str = field(default_factory=lambda: str(uuid4()))
    deployment_id: Optional[str] = None
    name: str = ""
    version: str = ""
    artifact_type: ArtifactType = ArtifactType.STRATEGY
    state: ArtifactState = ArtifactState.DOWNLOADING

    # Paths
    archive_path: Optional[Path] = None
    extracted_path: Optional[Path] = None

    # Verification
    expected_digest: str = ""
    actual_digest: str = ""
    verified: bool = False

    # Manifest
    manifest: Optional[ArtifactManifest] = None

    # Timestamps
    downloaded_at: Optional[datetime] = None
    verified_at: Optional[datetime] = None
    activated_at: Optional[datetime] = None

    # Metadata
    source_url: str = ""
    size_bytes: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "deployment_id": self.deployment_id,
            "name": self.name,
            "version": self.version,
            "artifact_type": self.artifact_type.name,
            "state": self.state.name,
            "archive_path": str(self.archive_path) if self.archive_path else None,
            "extracted_path": str(self.extracted_path) if self.extracted_path else None,
            "expected_digest": self.expected_digest,
            "actual_digest": self.actual_digest,
            "verified": self.verified,
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "downloaded_at": self.downloaded_at.isoformat() if self.downloaded_at else None,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "source_url": self.source_url,
            "size_bytes": self.size_bytes,
            "error": self.error,
        }


class ArtifactVerificationError(Exception):
    """Artifact verification failed."""
    pass


class ArtifactDownloadError(Exception):
    """Artifact download failed."""
    pass


class ArtifactManager:
    """
    Manages artifact lifecycle: download, verify, extract, activate.

    SECURITY CRITICAL:
    - All artifacts must pass digest verification before use
    - Manifest permissions are extracted and enforced by sandbox
    - No artifact execution without verification

    Usage:
        manager = ArtifactManager(cache_dir)

        # Download and verify
        artifact = manager.download_and_verify(
            url="https://storage.../artifact.tar.gz",
            expected_digest="sha256:abc123...",
            deployment_id="deploy-123",
        )

        # Get permissions for sandbox
        permissions = manager.get_permissions(artifact.artifact_id)

        # Activate for use
        manager.activate(artifact.artifact_id)

        # Atomic swap for live updates
        manager.swap(old_artifact_id, new_artifact_id)
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_cache_size_mb: int = 2000,
        on_progress: Optional[Callable[[str, float], None]] = None,
    ):
        """
        Initialize artifact manager.

        Args:
            cache_dir: Directory for artifact cache
            max_cache_size_mb: Maximum cache size in MB
            on_progress: Progress callback (artifact_id, progress_pct)
        """
        self._cache_dir = cache_dir or Path.home() / ARTIFACT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._max_cache_size_mb = max_cache_size_mb
        self._on_progress = on_progress

        # Artifact registry
        self._artifacts: Dict[str, Artifact] = {}
        self._active_artifact: Optional[str] = None

        # Index file for persistence
        self._index_path = self._cache_dir / "index.json"
        self._load_index()

    def _load_index(self) -> None:
        """Load artifact index from disk."""
        if not self._index_path.exists():
            return

        try:
            with open(self._index_path) as f:
                data = json.load(f)

            for artifact_data in data.get("artifacts", []):
                artifact_id = artifact_data["artifact_id"]
                artifact = Artifact(
                    artifact_id=artifact_id,
                    deployment_id=artifact_data.get("deployment_id"),
                    name=artifact_data.get("name", ""),
                    version=artifact_data.get("version", ""),
                    state=ArtifactState[artifact_data.get("state", "INACTIVE")],
                    expected_digest=artifact_data.get("expected_digest", ""),
                    actual_digest=artifact_data.get("actual_digest", ""),
                    verified=artifact_data.get("verified", False),
                )

                # Restore paths
                if artifact_data.get("archive_path"):
                    artifact.archive_path = Path(artifact_data["archive_path"])
                if artifact_data.get("extracted_path"):
                    artifact.extracted_path = Path(artifact_data["extracted_path"])
                    # Reload manifest
                    manifest_path = artifact.extracted_path / MANIFEST_FILENAME
                    if manifest_path.exists():
                        with open(manifest_path) as mf:
                            artifact.manifest = ArtifactManifest.from_yaml(mf.read())

                if artifact.extracted_path and artifact.extracted_path.exists():
                    self._artifacts[artifact_id] = artifact

            self._active_artifact = data.get("active_artifact")

        except Exception:
            # Corrupted index, start fresh
            pass

    def _save_index(self) -> None:
        """Save artifact index to disk."""
        data = {
            "artifacts": [a.to_dict() for a in self._artifacts.values()],
            "active_artifact": self._active_artifact,
            "saved_at": datetime.utcnow().isoformat(),
        }

        with open(self._index_path, "w") as f:
            json.dump(data, f, indent=2)

    def download_and_verify(
        self,
        url: str,
        expected_digest: str,
        deployment_id: Optional[str] = None,
        artifact_name: Optional[str] = None,
    ) -> Artifact:
        """
        Download artifact and verify digest.

        Args:
            url: Download URL (presigned)
            expected_digest: Expected SHA-256 digest (sha256:xxx format)
            deployment_id: Associated deployment ID
            artifact_name: Optional artifact name

        Returns:
            Verified Artifact

        Raises:
            ArtifactDownloadError: Download failed
            ArtifactVerificationError: Digest mismatch
        """
        import httpx

        artifact = Artifact(
            deployment_id=deployment_id,
            name=artifact_name or "",
            source_url=url,
            expected_digest=expected_digest,
            state=ArtifactState.DOWNLOADING,
        )
        self._artifacts[artifact.artifact_id] = artifact

        try:
            # Create temp file for download
            archive_path = self._cache_dir / f"{artifact.artifact_id}.download"

            # Download with progress tracking
            hasher = hashlib.sha256()

            with httpx.stream("GET", url, follow_redirects=True, timeout=300) as response:
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0

                # Check size limit
                if total_size > MAX_ARTIFACT_SIZE_MB * 1024 * 1024:
                    raise ArtifactDownloadError(
                        f"Artifact too large: {total_size / 1024 / 1024:.1f}MB > {MAX_ARTIFACT_SIZE_MB}MB"
                    )

                with open(archive_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=65536):
                        f.write(chunk)
                        hasher.update(chunk)
                        downloaded += len(chunk)

                        if self._on_progress and total_size > 0:
                            progress = (downloaded / total_size) * 50  # 0-50%
                            self._on_progress(artifact.artifact_id, progress)

            artifact.downloaded_at = datetime.utcnow()
            artifact.size_bytes = downloaded
            artifact.archive_path = archive_path
            artifact.state = ArtifactState.VERIFYING

            # Verify digest
            actual_digest = f"sha256:{hasher.hexdigest()}"
            artifact.actual_digest = actual_digest

            # Normalize expected digest format
            expected = expected_digest
            if not expected.startswith("sha256:"):
                expected = f"sha256:{expected}"

            if actual_digest != expected:
                artifact.state = ArtifactState.FAILED
                artifact.error = f"Digest mismatch: expected {expected}, got {actual_digest}"
                raise ArtifactVerificationError(artifact.error)

            artifact.verified = True
            artifact.verified_at = datetime.utcnow()
            artifact.state = ArtifactState.EXTRACTING

            if self._on_progress:
                self._on_progress(artifact.artifact_id, 60)

            # Extract artifact
            extracted_path = self._extract_artifact(artifact)
            artifact.extracted_path = extracted_path

            # Parse manifest
            manifest_path = extracted_path / MANIFEST_FILENAME
            if manifest_path.exists():
                with open(manifest_path) as f:
                    artifact.manifest = ArtifactManifest.from_yaml(f.read())
                artifact.name = artifact.manifest.name or artifact.name
                artifact.version = artifact.manifest.version
                artifact.artifact_type = artifact.manifest.artifact_type
            else:
                # No manifest - use default restrictive permissions
                artifact.manifest = ArtifactManifest.default_restrictive()

            artifact.state = ArtifactState.READY

            if self._on_progress:
                self._on_progress(artifact.artifact_id, 100)

            # Cleanup download file
            if archive_path.exists():
                archive_path.unlink()

            self._save_index()
            return artifact

        except ArtifactVerificationError:
            raise
        except httpx.HTTPError as e:
            artifact.state = ArtifactState.FAILED
            artifact.error = f"Download failed: {e}"
            raise ArtifactDownloadError(artifact.error) from e
        except Exception as e:
            artifact.state = ArtifactState.FAILED
            artifact.error = str(e)
            raise ArtifactDownloadError(f"Artifact download failed: {e}") from e

    def _extract_artifact(self, artifact: Artifact) -> Path:
        """Extract artifact archive."""
        if not artifact.archive_path or not artifact.archive_path.exists():
            raise ValueError("No archive to extract")

        extracted_path = self._cache_dir / artifact.artifact_id

        # Detect format and extract
        archive_str = str(artifact.archive_path)

        if archive_str.endswith((".tar.gz", ".tgz")):
            with tarfile.open(artifact.archive_path, "r:gz") as tar:
                # Security: check for path traversal
                for member in tar.getmembers():
                    if member.name.startswith("/") or ".." in member.name:
                        raise ValueError(f"Unsafe path in archive: {member.name}")
                tar.extractall(extracted_path)

        elif archive_str.endswith(".zip") or archive_str.endswith(".whl"):
            with zipfile.ZipFile(artifact.archive_path, "r") as zf:
                # Security: check for path traversal
                for name in zf.namelist():
                    if name.startswith("/") or ".." in name:
                        raise ValueError(f"Unsafe path in archive: {name}")
                zf.extractall(extracted_path)

        else:
            # Unknown format - copy as-is
            extracted_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact.archive_path, extracted_path / "artifact")

        return extracted_path

    def verify_local(self, artifact_id: str) -> bool:
        """
        Re-verify local artifact integrity.

        Args:
            artifact_id: Artifact ID

        Returns:
            True if verification passed
        """
        artifact = self._artifacts.get(artifact_id)
        if not artifact or not artifact.extracted_path:
            return False

        # Recompute digest of extracted files
        # This is a simplified check - production would use full tree hash
        manifest_path = artifact.extracted_path / MANIFEST_FILENAME
        if manifest_path.exists():
            with open(manifest_path, "rb") as f:
                content_hash = hashlib.sha256(f.read()).hexdigest()
            # Compare with stored manifest checksum
            if artifact.manifest and artifact.manifest.checksum:
                return content_hash == artifact.manifest.checksum

        return artifact.verified

    def get_permissions(self, artifact_id: str) -> Optional[ArtifactManifest]:
        """
        Get artifact permissions from manifest.

        Args:
            artifact_id: Artifact ID

        Returns:
            ArtifactManifest with permissions, or default restrictive
        """
        artifact = self._artifacts.get(artifact_id)
        if not artifact:
            return None

        return artifact.manifest or ArtifactManifest.default_restrictive()

    def activate(self, artifact_id: str) -> bool:
        """
        Mark artifact as active (currently running).

        Args:
            artifact_id: Artifact ID

        Returns:
            True if activated
        """
        artifact = self._artifacts.get(artifact_id)
        if not artifact or artifact.state not in (ArtifactState.READY, ArtifactState.INACTIVE):
            return False

        # Deactivate current
        if self._active_artifact and self._active_artifact != artifact_id:
            old = self._artifacts.get(self._active_artifact)
            if old:
                old.state = ArtifactState.INACTIVE

        artifact.state = ArtifactState.ACTIVE
        artifact.activated_at = datetime.utcnow()
        self._active_artifact = artifact_id

        self._save_index()
        return True

    def deactivate(self, artifact_id: str) -> bool:
        """Deactivate artifact."""
        artifact = self._artifacts.get(artifact_id)
        if not artifact:
            return False

        artifact.state = ArtifactState.INACTIVE
        if self._active_artifact == artifact_id:
            self._active_artifact = None

        self._save_index()
        return True

    def swap(self, old_artifact_id: str, new_artifact_id: str) -> bool:
        """
        Atomic swap of artifacts for live update.

        Args:
            old_artifact_id: Currently active artifact
            new_artifact_id: New artifact to activate

        Returns:
            True if swap successful
        """
        old = self._artifacts.get(old_artifact_id)
        new = self._artifacts.get(new_artifact_id)

        if not new or new.state != ArtifactState.READY:
            return False

        # Deactivate old
        if old:
            old.state = ArtifactState.INACTIVE

        # Activate new
        new.state = ArtifactState.ACTIVE
        new.activated_at = datetime.utcnow()
        self._active_artifact = new_artifact_id

        self._save_index()
        return True

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Get artifact by ID."""
        return self._artifacts.get(artifact_id)

    def get_active_artifact(self) -> Optional[Artifact]:
        """Get currently active artifact."""
        if self._active_artifact:
            return self._artifacts.get(self._active_artifact)
        return None

    def get_artifact_path(self, artifact_id: str) -> Optional[Path]:
        """Get extracted path for artifact."""
        artifact = self._artifacts.get(artifact_id)
        if artifact and artifact.extracted_path:
            return artifact.extracted_path
        return None

    def delete(self, artifact_id: str) -> bool:
        """
        Delete artifact from cache.

        Args:
            artifact_id: Artifact ID

        Returns:
            True if deleted
        """
        artifact = self._artifacts.get(artifact_id)
        if not artifact:
            return False

        if artifact.state == ArtifactState.ACTIVE:
            return False  # Cannot delete active artifact

        # Remove files
        if artifact.extracted_path and artifact.extracted_path.exists():
            shutil.rmtree(artifact.extracted_path, ignore_errors=True)
        if artifact.archive_path and artifact.archive_path.exists():
            artifact.archive_path.unlink(missing_ok=True)

        artifact.state = ArtifactState.DELETED
        del self._artifacts[artifact_id]

        self._save_index()
        return True

    def cleanup_old(self, keep_count: int = 3) -> int:
        """
        Cleanup old inactive artifacts.

        Args:
            keep_count: Number of recent artifacts to keep

        Returns:
            Number of artifacts deleted
        """
        # Sort by activation time, keep recent
        inactive = [
            a for a in self._artifacts.values()
            if a.state == ArtifactState.INACTIVE
        ]
        inactive.sort(key=lambda a: a.activated_at or datetime.min, reverse=True)

        deleted = 0
        for artifact in inactive[keep_count:]:
            if self.delete(artifact.artifact_id):
                deleted += 1

        return deleted

    def get_cache_size_mb(self) -> float:
        """Get total cache size in MB."""
        total_size = 0
        for artifact in self._artifacts.values():
            if artifact.extracted_path and artifact.extracted_path.exists():
                for path in artifact.extracted_path.rglob("*"):
                    if path.is_file():
                        total_size += path.stat().st_size

        return total_size / (1024 * 1024)

    def list_artifacts(self) -> List[Artifact]:
        """List all artifacts."""
        return list(self._artifacts.values())

    def get_status(self) -> Dict[str, Any]:
        """Get artifact manager status."""
        return {
            "cache_dir": str(self._cache_dir),
            "cache_size_mb": self.get_cache_size_mb(),
            "max_cache_size_mb": self._max_cache_size_mb,
            "total_artifacts": len(self._artifacts),
            "active_artifact": self._active_artifact,
            "artifacts_by_state": {
                state.name: sum(1 for a in self._artifacts.values() if a.state == state)
                for state in ArtifactState
            },
        }
