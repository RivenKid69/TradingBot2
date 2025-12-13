# -*- coding: utf-8 -*-
"""
CCEA Immutable Blob Storage.

Provides content-addressable storage for:
- Configuration blobs
- Artifact manifests
- Any immutable data referenced by digest

Security:
- Content is immutable once stored
- Referenced by SHA256 digest only
- No secrets stored in blobs
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import threading

from ccea.crypto.digest import compute_digest, compute_json_digest, verify_digest


logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================

class BlobError(Exception):
    """Base blob storage error."""
    pass


class BlobNotFoundError(BlobError):
    """Blob not found."""
    pass


class BlobIntegrityError(BlobError):
    """Blob integrity check failed."""
    pass


class BlobAlreadyExistsError(BlobError):
    """Blob with this digest already exists."""
    pass


# ============================================================================
# Blob Metadata
# ============================================================================

@dataclass
class BlobMetadata:
    """Metadata for a stored blob."""
    digest: str
    size_bytes: int
    content_type: str = "application/octet-stream"
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    workspace_id: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)


# ============================================================================
# Blob Store Interface
# ============================================================================

class BlobStore(ABC):
    """
    Abstract immutable blob storage interface.

    All blobs are stored and retrieved by digest.
    Content is immutable - same digest always returns same content.
    """

    @abstractmethod
    def put(
        self,
        content: bytes,
        content_type: str = "application/octet-stream",
        workspace_id: Optional[str] = None,
        created_by: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> BlobMetadata:
        """
        Store a blob.

        Args:
            content: Blob content
            content_type: MIME type
            workspace_id: Owning workspace
            created_by: Creator identifier
            labels: Optional labels

        Returns:
            BlobMetadata with digest

        Raises:
            BlobAlreadyExistsError: If blob exists (idempotent - returns existing)
        """
        pass

    @abstractmethod
    def get(self, digest: str) -> bytes:
        """
        Retrieve blob content by digest.

        Args:
            digest: SHA256 digest

        Returns:
            Blob content bytes

        Raises:
            BlobNotFoundError: If blob doesn't exist
        """
        pass

    @abstractmethod
    def get_metadata(self, digest: str) -> BlobMetadata:
        """
        Get blob metadata.

        Args:
            digest: SHA256 digest

        Returns:
            BlobMetadata

        Raises:
            BlobNotFoundError: If blob doesn't exist
        """
        pass

    @abstractmethod
    def exists(self, digest: str) -> bool:
        """Check if blob exists."""
        pass

    @abstractmethod
    def delete(self, digest: str) -> bool:
        """
        Delete a blob.

        Note: In production, blobs may be reference-counted or garbage-collected.

        Args:
            digest: SHA256 digest

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def list_by_workspace(
        self,
        workspace_id: str,
        limit: int = 100,
    ) -> List[BlobMetadata]:
        """List blobs in a workspace."""
        pass


# ============================================================================
# In-Memory Blob Store
# ============================================================================

class InMemoryBlobStore(BlobStore):
    """
    In-memory blob store for development/testing.

    Not suitable for production - use persistent storage.
    """

    def __init__(self):
        self._blobs: Dict[str, bytes] = {}
        self._metadata: Dict[str, BlobMetadata] = {}
        self._lock = threading.Lock()

    def put(
        self,
        content: bytes,
        content_type: str = "application/octet-stream",
        workspace_id: Optional[str] = None,
        created_by: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> BlobMetadata:
        digest = compute_digest(content)

        with self._lock:
            # Idempotent - if exists, return existing
            if digest in self._metadata:
                return self._metadata[digest]

            metadata = BlobMetadata(
                digest=digest,
                size_bytes=len(content),
                content_type=content_type,
                workspace_id=workspace_id,
                created_by=created_by,
                labels=labels or {},
            )

            self._blobs[digest] = content
            self._metadata[digest] = metadata

        logger.debug(
            "Blob stored",
            extra={
                "digest": digest[:32] + "...",
                "size": len(content),
            }
        )

        return metadata

    def get(self, digest: str) -> bytes:
        with self._lock:
            if digest not in self._blobs:
                raise BlobNotFoundError(f"Blob not found: {digest}")
            return self._blobs[digest]

    def get_metadata(self, digest: str) -> BlobMetadata:
        with self._lock:
            if digest not in self._metadata:
                raise BlobNotFoundError(f"Blob not found: {digest}")
            return self._metadata[digest]

    def exists(self, digest: str) -> bool:
        with self._lock:
            return digest in self._blobs

    def delete(self, digest: str) -> bool:
        with self._lock:
            if digest in self._blobs:
                del self._blobs[digest]
                del self._metadata[digest]
                return True
            return False

    def list_by_workspace(
        self,
        workspace_id: str,
        limit: int = 100,
    ) -> List[BlobMetadata]:
        with self._lock:
            blobs = [
                m for m in self._metadata.values()
                if m.workspace_id == workspace_id
            ]
            blobs.sort(key=lambda m: m.created_at, reverse=True)
            return blobs[:limit]


# ============================================================================
# File-based Blob Store
# ============================================================================

class FileBlobStore(BlobStore):
    """
    File-system based blob store.

    Stores blobs as files named by digest.
    Suitable for development and small-scale deployments.
    """

    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize file blob store.

        Args:
            base_path: Base directory for blob storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._metadata_path = self.base_path / "metadata"
        self._metadata_path.mkdir(exist_ok=True)
        self._lock = threading.Lock()

    def _blob_path(self, digest: str) -> Path:
        """Get path for blob file."""
        # Use first 4 chars of hash for sharding
        hash_part = digest.split(":")[-1]
        shard = hash_part[:4]
        shard_dir = self.base_path / shard
        shard_dir.mkdir(exist_ok=True)
        return shard_dir / hash_part

    def _metadata_file_path(self, digest: str) -> Path:
        """Get path for metadata file."""
        hash_part = digest.split(":")[-1]
        return self._metadata_path / f"{hash_part}.json"

    def put(
        self,
        content: bytes,
        content_type: str = "application/octet-stream",
        workspace_id: Optional[str] = None,
        created_by: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> BlobMetadata:
        digest = compute_digest(content)

        with self._lock:
            blob_path = self._blob_path(digest)
            meta_path = self._metadata_file_path(digest)

            # Idempotent - if exists, return existing
            if blob_path.exists():
                return self._load_metadata(digest)

            # Write blob
            blob_path.write_bytes(content)

            # Write metadata
            metadata = BlobMetadata(
                digest=digest,
                size_bytes=len(content),
                content_type=content_type,
                workspace_id=workspace_id,
                created_by=created_by,
                labels=labels or {},
            )
            self._save_metadata(metadata)

        return metadata

    def get(self, digest: str) -> bytes:
        blob_path = self._blob_path(digest)
        if not blob_path.exists():
            raise BlobNotFoundError(f"Blob not found: {digest}")

        content = blob_path.read_bytes()

        # Verify integrity
        if not verify_digest(content, digest):
            raise BlobIntegrityError(f"Blob integrity check failed: {digest}")

        return content

    def get_metadata(self, digest: str) -> BlobMetadata:
        return self._load_metadata(digest)

    def exists(self, digest: str) -> bool:
        return self._blob_path(digest).exists()

    def delete(self, digest: str) -> bool:
        with self._lock:
            blob_path = self._blob_path(digest)
            meta_path = self._metadata_file_path(digest)

            if not blob_path.exists():
                return False

            blob_path.unlink()
            if meta_path.exists():
                meta_path.unlink()

            return True

    def list_by_workspace(
        self,
        workspace_id: str,
        limit: int = 100,
    ) -> List[BlobMetadata]:
        results = []
        with self._lock:
            for meta_file in self._metadata_path.glob("*.json"):
                try:
                    with open(meta_file, "r") as f:
                        data = json.load(f)
                    if data.get("workspace_id") == workspace_id:
                        results.append(BlobMetadata(
                            digest=data["digest"],
                            size_bytes=data["size_bytes"],
                            content_type=data.get("content_type", "application/octet-stream"),
                            created_at=datetime.fromisoformat(data["created_at"]),
                            created_by=data.get("created_by"),
                            workspace_id=data.get("workspace_id"),
                            labels=data.get("labels", {}),
                        ))
                except Exception:
                    continue

        results.sort(key=lambda m: m.created_at, reverse=True)
        return results[:limit]

    def _save_metadata(self, metadata: BlobMetadata) -> None:
        """Save metadata to file."""
        meta_path = self._metadata_file_path(metadata.digest)
        data = {
            "digest": metadata.digest,
            "size_bytes": metadata.size_bytes,
            "content_type": metadata.content_type,
            "created_at": metadata.created_at.isoformat(),
            "created_by": metadata.created_by,
            "workspace_id": metadata.workspace_id,
            "labels": metadata.labels,
        }
        with open(meta_path, "w") as f:
            json.dump(data, f)

    def _load_metadata(self, digest: str) -> BlobMetadata:
        """Load metadata from file."""
        meta_path = self._metadata_file_path(digest)
        if not meta_path.exists():
            raise BlobNotFoundError(f"Metadata not found: {digest}")

        with open(meta_path, "r") as f:
            data = json.load(f)

        return BlobMetadata(
            digest=data["digest"],
            size_bytes=data["size_bytes"],
            content_type=data.get("content_type", "application/octet-stream"),
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data.get("created_by"),
            workspace_id=data.get("workspace_id"),
            labels=data.get("labels", {}),
        )


# ============================================================================
# Config Blob Helper
# ============================================================================

class ConfigBlobStore:
    """
    Helper for storing configuration as JSON blobs.

    Configuration is stored as immutable JSON blobs.
    Any change creates a new blob with new digest.
    """

    def __init__(self, blob_store: BlobStore):
        """Initialize config blob store."""
        self.blob_store = blob_store

    def put_config(
        self,
        config: Dict[str, Any],
        workspace_id: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> str:
        """
        Store configuration blob.

        Args:
            config: Configuration dictionary
            workspace_id: Owning workspace
            created_by: Creator identifier

        Returns:
            Digest of stored config
        """
        content = json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        metadata = self.blob_store.put(
            content=content,
            content_type="application/json",
            workspace_id=workspace_id,
            created_by=created_by,
            labels={"type": "config"},
        )

        return metadata.digest

    def get_config(self, digest: str) -> Dict[str, Any]:
        """
        Retrieve configuration by digest.

        Args:
            digest: Config blob digest

        Returns:
            Configuration dictionary
        """
        content = self.blob_store.get(digest)
        return json.loads(content.decode("utf-8"))

    def config_exists(self, digest: str) -> bool:
        """Check if config exists."""
        return self.blob_store.exists(digest)
