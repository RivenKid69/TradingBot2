# -*- coding: utf-8 -*-
"""
Registry Mirror - Local artifact mirroring for enterprise deployments.

CLOUD ZONE ONLY.

Provides local artifact registry mirroring for:
- Air-gapped deployments (no internet access)
- Reduced latency in distributed deployments
- Enterprise compliance (artifact control)
- Offline verification support

Design Doc Reference:
    - Phase 9: Enterprise/on-prem pack
    - Design Doc 4.4: Artifact registry
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

# Constants
DEFAULT_SYNC_INTERVAL: Final[int] = 3600  # 1 hour
MAX_CONCURRENT_DOWNLOADS: Final[int] = 5
CHUNK_SIZE: Final[int] = 8192  # 8KB chunks


class MirrorStatus(Enum):
    """Registry mirror status."""
    IDLE = auto()
    SYNCING = auto()
    SYNCED = auto()
    ERROR = auto()
    OFFLINE = auto()


class SyncMode(Enum):
    """Synchronization mode."""
    FULL = "full"  # Sync all artifacts
    INCREMENTAL = "incremental"  # Sync only new/updated
    SELECTIVE = "selective"  # Sync specific artifacts only


@dataclass
class MirrorSyncResult:
    """Result of a mirror sync operation."""
    success: bool = False
    synced_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    bytes_transferred: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    synced_artifacts: List[str] = field(default_factory=list)
    failed_artifacts: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "synced_count": self.synced_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "bytes_transferred": self.bytes_transferred,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors,
            "synced_artifacts": self.synced_artifacts,
            "failed_artifacts": self.failed_artifacts,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ArtifactInfo:
    """Information about a mirrored artifact."""
    digest: str
    name: str
    tag: str
    size_bytes: int
    created_at: datetime
    synced_at: Optional[datetime] = None
    signature_verified: bool = False
    sbom_available: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "digest": self.digest,
            "name": self.name,
            "tag": self.tag,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat(),
            "synced_at": self.synced_at.isoformat() if self.synced_at else None,
            "signature_verified": self.signature_verified,
            "sbom_available": self.sbom_available,
            "metadata": self.metadata,
        }


@dataclass
class RegistryMirrorConfig:
    """Configuration for registry mirror."""
    # Storage
    storage_path: Path = field(default_factory=lambda: Path.home() / ".ccea" / "mirror")
    max_storage_bytes: int = 50 * 1024 * 1024 * 1024  # 50GB default

    # Upstream registry
    upstream_url: str = ""
    upstream_username: Optional[str] = None
    upstream_password: Optional[str] = None
    upstream_insecure: bool = False

    # Local registry
    local_port: int = 5001
    local_host: str = "0.0.0.0"

    # Sync settings
    sync_interval_seconds: int = DEFAULT_SYNC_INTERVAL
    sync_mode: SyncMode = SyncMode.INCREMENTAL
    auto_sync: bool = True

    # Air-gapped mode
    air_gapped: bool = False
    offline_verification: bool = True

    # Filtering
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)

    # Verification
    require_signatures: bool = True
    trusted_keys: List[str] = field(default_factory=list)

    # Retention
    retention_days: int = 90
    keep_last_n_versions: int = 5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "storage_path": str(self.storage_path),
            "max_storage_bytes": self.max_storage_bytes,
            "upstream_url": self.upstream_url,
            "sync_interval_seconds": self.sync_interval_seconds,
            "sync_mode": self.sync_mode.value,
            "auto_sync": self.auto_sync,
            "air_gapped": self.air_gapped,
            "require_signatures": self.require_signatures,
            "retention_days": self.retention_days,
        }


class RegistryMirror:
    """
    Registry Mirror for enterprise artifact mirroring.

    Provides local caching and distribution of artifacts for:
    - Air-gapped deployments
    - Offline verification
    - Compliance requirements
    - Reduced latency

    Usage:
        config = RegistryMirrorConfig(
            upstream_url="https://registry.ccea.io",
            storage_path=Path("/data/mirror"),
        )
        mirror = RegistryMirror(config)

        # Start mirror service
        await mirror.start()

        # Sync artifacts
        result = await mirror.sync()

        # Get artifact
        artifact = mirror.get_artifact("sha256:abc123...")
    """

    def __init__(
        self,
        config: Optional[RegistryMirrorConfig] = None,
        on_sync_complete: Optional[Callable[[MirrorSyncResult], None]] = None,
    ):
        """
        Initialize registry mirror.

        Args:
            config: Mirror configuration
            on_sync_complete: Callback when sync completes
        """
        self.config = config or RegistryMirrorConfig()
        self._on_sync_complete = on_sync_complete

        # State
        self._status = MirrorStatus.IDLE
        self._artifacts: Dict[str, ArtifactInfo] = {}
        self._last_sync: Optional[datetime] = None
        self._last_sync_result: Optional[MirrorSyncResult] = None

        # Storage
        self._storage_path = self.config.storage_path
        self._blobs_path = self._storage_path / "blobs"
        self._manifests_path = self._storage_path / "manifests"
        self._index_path = self._storage_path / "index.json"

        # Background tasks
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False
        self._sync_lock = asyncio.Lock()

    @property
    def status(self) -> MirrorStatus:
        """Get current mirror status."""
        return self._status

    @property
    def last_sync(self) -> Optional[datetime]:
        """Get last sync timestamp."""
        return self._last_sync

    @property
    def artifact_count(self) -> int:
        """Get number of mirrored artifacts."""
        return len(self._artifacts)

    @property
    def storage_used_bytes(self) -> int:
        """Get storage used in bytes."""
        total = 0
        if self._blobs_path.exists():
            for f in self._blobs_path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        return total

    async def start(self) -> None:
        """Start the mirror service."""
        if self._running:
            return

        logger.info("Starting registry mirror service")

        # Initialize storage
        self._init_storage()

        # Load index
        self._load_index()

        self._running = True
        self._status = MirrorStatus.IDLE

        # Start auto-sync if enabled
        if self.config.auto_sync and not self.config.air_gapped:
            self._sync_task = asyncio.create_task(self._auto_sync_loop())

        logger.info(
            f"Registry mirror started with {self.artifact_count} artifacts, "
            f"storage used: {self.storage_used_bytes / 1024 / 1024:.1f}MB"
        )

    async def stop(self) -> None:
        """Stop the mirror service."""
        if not self._running:
            return

        logger.info("Stopping registry mirror service")

        self._running = False

        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        # Save index
        self._save_index()

        self._status = MirrorStatus.OFFLINE
        logger.info("Registry mirror stopped")

    async def sync(
        self,
        artifacts: Optional[List[str]] = None,
        force: bool = False,
    ) -> MirrorSyncResult:
        """
        Synchronize artifacts from upstream registry.

        Args:
            artifacts: Specific artifacts to sync (None = all)
            force: Force re-sync even if already present

        Returns:
            Sync result
        """
        if self.config.air_gapped:
            return MirrorSyncResult(
                success=False,
                errors=["Cannot sync in air-gapped mode"],
            )

        async with self._sync_lock:
            self._status = MirrorStatus.SYNCING
            start_time = time.time()

            result = MirrorSyncResult()

            try:
                # Get list of artifacts to sync
                if artifacts:
                    to_sync = artifacts
                else:
                    to_sync = await self._list_upstream_artifacts()

                # Filter artifacts
                to_sync = self._filter_artifacts(to_sync)

                # Sync each artifact
                for artifact_ref in to_sync:
                    try:
                        synced = await self._sync_artifact(artifact_ref, force)
                        if synced:
                            result.synced_count += 1
                            result.synced_artifacts.append(artifact_ref)
                        else:
                            result.skipped_count += 1
                    except Exception as e:
                        result.failed_count += 1
                        result.failed_artifacts.append(artifact_ref)
                        result.errors.append(f"Failed to sync {artifact_ref}: {e}")
                        logger.error(f"Failed to sync artifact {artifact_ref}: {e}")

                result.success = result.failed_count == 0
                self._status = MirrorStatus.SYNCED if result.success else MirrorStatus.ERROR

            except Exception as e:
                result.success = False
                result.errors.append(str(e))
                self._status = MirrorStatus.ERROR
                logger.error(f"Sync failed: {e}")

            result.duration_seconds = time.time() - start_time
            self._last_sync = datetime.utcnow()
            self._last_sync_result = result

            # Save index
            self._save_index()

            # Callback
            if self._on_sync_complete:
                try:
                    self._on_sync_complete(result)
                except Exception as e:
                    logger.error(f"Sync callback error: {e}")

            logger.info(
                f"Sync completed: {result.synced_count} synced, "
                f"{result.skipped_count} skipped, {result.failed_count} failed"
            )

            return result

    def get_artifact(self, digest: str) -> Optional[ArtifactInfo]:
        """
        Get artifact information by digest.

        Args:
            digest: Artifact digest (sha256:...)

        Returns:
            Artifact info or None
        """
        return self._artifacts.get(digest)

    def get_artifact_path(self, digest: str) -> Optional[Path]:
        """
        Get local path for artifact blob.

        Args:
            digest: Artifact digest

        Returns:
            Path to blob or None
        """
        if digest not in self._artifacts:
            return None

        # Convert digest to path
        algo, hash_value = digest.split(":", 1)
        blob_path = self._blobs_path / algo / hash_value[:2] / hash_value

        if blob_path.exists():
            return blob_path
        return None

    def list_artifacts(
        self,
        name_filter: Optional[str] = None,
        tag_filter: Optional[str] = None,
    ) -> List[ArtifactInfo]:
        """
        List all mirrored artifacts.

        Args:
            name_filter: Filter by name (glob pattern)
            tag_filter: Filter by tag (glob pattern)

        Returns:
            List of artifact info
        """
        import fnmatch

        artifacts = list(self._artifacts.values())

        if name_filter:
            artifacts = [a for a in artifacts if fnmatch.fnmatch(a.name, name_filter)]

        if tag_filter:
            artifacts = [a for a in artifacts if fnmatch.fnmatch(a.tag, tag_filter)]

        return artifacts

    async def verify_artifact(self, digest: str) -> Tuple[bool, Optional[str]]:
        """
        Verify artifact integrity and signature.

        Args:
            digest: Artifact digest

        Returns:
            (is_valid, error_message)
        """
        artifact = self._artifacts.get(digest)
        if not artifact:
            return False, "Artifact not found"

        blob_path = self.get_artifact_path(digest)
        if not blob_path or not blob_path.exists():
            return False, "Artifact blob not found"

        # Verify digest
        computed_digest = await self._compute_digest(blob_path)
        if computed_digest != digest:
            return False, "Digest mismatch"

        # Verify signature if required
        if self.config.require_signatures:
            if not artifact.signature_verified:
                # Try to verify signature
                sig_valid = await self._verify_signature(digest)
                if not sig_valid:
                    return False, "Signature verification failed"
                artifact.signature_verified = True

        return True, None

    async def import_artifact(
        self,
        source_path: Path,
        name: str,
        tag: str,
        signature_path: Optional[Path] = None,
    ) -> Optional[ArtifactInfo]:
        """
        Import artifact from local file (for air-gapped).

        Args:
            source_path: Path to artifact file
            name: Artifact name
            tag: Artifact tag
            signature_path: Optional path to signature file

        Returns:
            Artifact info or None
        """
        if not source_path.exists():
            logger.error(f"Source file not found: {source_path}")
            return None

        # Compute digest
        digest = await self._compute_digest(source_path)

        # Copy to blob storage
        algo, hash_value = digest.split(":", 1)
        blob_dir = self._blobs_path / algo / hash_value[:2]
        blob_dir.mkdir(parents=True, exist_ok=True)
        blob_path = blob_dir / hash_value

        shutil.copy2(source_path, blob_path)

        # Import signature if provided
        sig_verified = False
        if signature_path and signature_path.exists():
            sig_verified = await self._import_and_verify_signature(
                digest, signature_path
            )

        # Create artifact info
        artifact = ArtifactInfo(
            digest=digest,
            name=name,
            tag=tag,
            size_bytes=source_path.stat().st_size,
            created_at=datetime.utcnow(),
            synced_at=datetime.utcnow(),
            signature_verified=sig_verified,
        )

        self._artifacts[digest] = artifact
        self._save_index()

        logger.info(f"Imported artifact: {name}:{tag} ({digest})")
        return artifact

    async def delete_artifact(self, digest: str) -> bool:
        """
        Delete artifact from mirror.

        Args:
            digest: Artifact digest

        Returns:
            True if deleted
        """
        if digest not in self._artifacts:
            return False

        # Remove blob
        blob_path = self.get_artifact_path(digest)
        if blob_path and blob_path.exists():
            blob_path.unlink()

        # Remove from index
        del self._artifacts[digest]
        self._save_index()

        logger.info(f"Deleted artifact: {digest}")
        return True

    async def cleanup_old_artifacts(self) -> int:
        """
        Clean up artifacts older than retention period.

        Returns:
            Number of artifacts deleted
        """
        cutoff = datetime.utcnow() - timedelta(days=self.config.retention_days)
        to_delete = []

        for digest, artifact in self._artifacts.items():
            if artifact.synced_at and artifact.synced_at < cutoff:
                to_delete.append(digest)

        for digest in to_delete:
            await self.delete_artifact(digest)

        logger.info(f"Cleaned up {len(to_delete)} old artifacts")
        return len(to_delete)

    def get_status(self) -> Dict[str, Any]:
        """Get mirror status information."""
        return {
            "status": self._status.name,
            "artifact_count": self.artifact_count,
            "storage_used_bytes": self.storage_used_bytes,
            "storage_max_bytes": self.config.max_storage_bytes,
            "storage_used_percent": (
                self.storage_used_bytes / self.config.max_storage_bytes * 100
                if self.config.max_storage_bytes > 0
                else 0
            ),
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "last_sync_result": (
                self._last_sync_result.to_dict() if self._last_sync_result else None
            ),
            "air_gapped": self.config.air_gapped,
            "auto_sync": self.config.auto_sync,
            "upstream_url": self.config.upstream_url if not self.config.air_gapped else None,
        }

    # ===== Private Methods =====

    def _init_storage(self) -> None:
        """Initialize storage directories."""
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._blobs_path.mkdir(parents=True, exist_ok=True)
        self._manifests_path.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> None:
        """Load artifact index from disk."""
        if not self._index_path.exists():
            return

        try:
            with open(self._index_path, "r") as f:
                data = json.load(f)

            for item in data.get("artifacts", []):
                artifact = ArtifactInfo(
                    digest=item["digest"],
                    name=item["name"],
                    tag=item["tag"],
                    size_bytes=item["size_bytes"],
                    created_at=datetime.fromisoformat(item["created_at"]),
                    synced_at=(
                        datetime.fromisoformat(item["synced_at"])
                        if item.get("synced_at")
                        else None
                    ),
                    signature_verified=item.get("signature_verified", False),
                    sbom_available=item.get("sbom_available", False),
                    metadata=item.get("metadata", {}),
                )
                self._artifacts[artifact.digest] = artifact

            if data.get("last_sync"):
                self._last_sync = datetime.fromisoformat(data["last_sync"])

        except Exception as e:
            logger.error(f"Failed to load index: {e}")

    def _save_index(self) -> None:
        """Save artifact index to disk."""
        data = {
            "version": "1.0",
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "artifacts": [a.to_dict() for a in self._artifacts.values()],
        }

        with open(self._index_path, "w") as f:
            json.dump(data, f, indent=2)

    async def _auto_sync_loop(self) -> None:
        """Background auto-sync loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.sync_interval_seconds)
                if self._running and not self.config.air_gapped:
                    await self.sync()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-sync error: {e}")

    async def _list_upstream_artifacts(self) -> List[str]:
        """List artifacts from upstream registry."""
        # Placeholder - would use registry API
        # In real implementation: GET /v2/_catalog, then /v2/{name}/tags/list
        return []

    def _filter_artifacts(self, artifacts: List[str]) -> List[str]:
        """Filter artifacts based on include/exclude patterns."""
        import fnmatch

        result = artifacts

        if self.config.include_patterns:
            result = [
                a for a in result
                if any(fnmatch.fnmatch(a, p) for p in self.config.include_patterns)
            ]

        if self.config.exclude_patterns:
            result = [
                a for a in result
                if not any(fnmatch.fnmatch(a, p) for p in self.config.exclude_patterns)
            ]

        return result

    async def _sync_artifact(self, artifact_ref: str, force: bool) -> bool:
        """
        Sync a single artifact from upstream.

        Returns:
            True if synced (new or updated), False if skipped
        """
        # Parse artifact reference
        if "@" in artifact_ref:
            name, digest = artifact_ref.rsplit("@", 1)
            tag = ""
        elif ":" in artifact_ref:
            name, tag = artifact_ref.rsplit(":", 1)
            digest = ""
        else:
            name = artifact_ref
            tag = "latest"
            digest = ""

        # Check if already present
        if digest and digest in self._artifacts and not force:
            return False

        # Download manifest to get digest
        # Placeholder - would use registry API
        # GET /v2/{name}/manifests/{tag}

        # Download blob
        # GET /v2/{name}/blobs/{digest}

        # Store in local storage
        # ...

        return True

    async def _compute_digest(self, path: Path) -> str:
        """Compute SHA256 digest of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                sha256.update(chunk)
        return f"sha256:{sha256.hexdigest()}"

    async def _verify_signature(self, digest: str) -> bool:
        """
        Verify artifact signature using cosign/sigstore.

        SECURITY: This method implements fail-closed behavior per CCEA Design Doc Section 8.3.
        Artifacts without valid signatures are rejected to prevent supply chain attacks.

        Args:
            digest: The artifact digest to verify (sha256:...)

        Returns:
            True if signature is valid, False otherwise

        Raises:
            SignatureVerificationError: If verification infrastructure is unavailable
        """
        # Check for development bypass (NEVER enable in production)
        if os.environ.get("CCEA_SKIP_SIGNATURE_VERIFICATION") == "DEVELOPMENT_ONLY":
            logger.warning(
                "SECURITY: Signature verification SKIPPED (DEVELOPMENT_ONLY mode). "
                "This MUST NOT be used in production. Digest: %s",
                digest,
            )
            # Emit metric for security monitoring
            self._emit_signature_skip_metric(digest)
            return True

        # SECURITY: Fail-closed signature verification per CCEA Design Doc Section 8.3
        # Tech Debt Tracking: CCEA-SEC-001
        # Status: CONTROLLED - fail-closed rejects unsigned artifacts
        # Production implementation requires cosign/sigstore integration
        # Acceptance criteria:
        #   - All artifacts rejected until signing infrastructure deployed
        #   - Metrics emitted for security alerting (signature_verification_failed)
        #   - Development bypass requires explicit env var (auditable)
        logger.error(
            "SECURITY: Signature verification not implemented. "
            "Artifact rejected per fail-closed policy. Digest: %s. "
            "Tracking: CCEA-SEC-001",
            digest,
        )
        # Emit metric for security alerting
        self._emit_signature_failure_metric(digest, reason="not_implemented")
        return False

    def _emit_signature_skip_metric(self, digest: str) -> None:
        """Emit metric when signature verification is skipped (dev only)."""
        # Integration point for metrics/alerting system
        logger.info("METRIC: signature_verification_skipped{digest=%s}", digest)

    def _emit_signature_failure_metric(self, digest: str, reason: str) -> None:
        """Emit metric when signature verification fails."""
        # Integration point for metrics/alerting system
        logger.info("METRIC: signature_verification_failed{digest=%s,reason=%s}", digest, reason)

    async def _import_and_verify_signature(
        self,
        digest: str,
        signature_path: Path,
    ) -> bool:
        """
        Import and verify a signature file against trusted keys.

        SECURITY: This method implements fail-closed behavior per CCEA Design Doc Section 8.3.
        Signatures that cannot be verified are rejected.

        Args:
            digest: The artifact digest
            signature_path: Path to the signature file

        Returns:
            True if signature is valid, False otherwise
        """
        # Check for development bypass
        if os.environ.get("CCEA_SKIP_SIGNATURE_VERIFICATION") == "DEVELOPMENT_ONLY":
            logger.warning(
                "SECURITY: Signature import/verification SKIPPED (DEVELOPMENT_ONLY mode). "
                "Digest: %s, Signature: %s",
                digest,
                signature_path,
            )
            self._emit_signature_skip_metric(digest)
            return True

        # Fail-closed: verification is mandatory
        if not signature_path.exists():
            logger.error(
                "SECURITY: Signature file not found. Artifact rejected. "
                "Digest: %s, Expected signature: %s",
                digest,
                signature_path,
            )
            self._emit_signature_failure_metric(digest, reason="signature_file_missing")
            return False

        # Production implementation should verify signature against trusted keys
        logger.error(
            "SECURITY: Signature verification not fully implemented. "
            "Artifact rejected per fail-closed policy. Digest: %s",
            digest,
        )
        self._emit_signature_failure_metric(digest, reason="not_implemented")
        return False
