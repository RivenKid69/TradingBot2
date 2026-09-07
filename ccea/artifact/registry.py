# -*- coding: utf-8 -*-
"""
CCEA Artifact Registry.

Provides artifact storage and retrieval:
- Stores artifacts by digest
- Maintains artifact metadata
- Supports allowlisted registries

Security:
- Agent only pulls from allowlisted registries
- All artifacts verified by digest
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import threading

from ccea.crypto.digest import compute_file_digest, verify_file_digest
from ccea.artifact.signer import SignatureInfo


@dataclass
class RegistryEntry:
    """Registry entry for an artifact."""

    artifact_id: str
    version: str
    digest: str
    manifest_digest: str
    created_at: datetime
    signature: Optional[SignatureInfo] = None
    sbom_digest: Optional[str] = None
    workspace_id: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)


class ArtifactRegistry:
    """
    Local artifact registry.

    Stores artifacts in a content-addressable storage.
    """

    def __init__(
        self,
        storage_path: Path,
        allowlist: Optional[List[str]] = None,
    ):
        """
        Initialize registry.

        Args:
            storage_path: Base path for storage
            allowlist: Allowlisted registry URLs (for remote registries)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.blobs_path = self.storage_path / "blobs"
        self.blobs_path.mkdir(exist_ok=True)

        self.manifests_path = self.storage_path / "manifests"
        self.manifests_path.mkdir(exist_ok=True)

        self.index_path = self.storage_path / "index.json"

        self.allowlist = set(allowlist or [])
        self._index: Dict[str, RegistryEntry] = {}
        self._lock = threading.Lock()

        self._load_index()

    def push(
        self,
        artifact_path: Path,
        manifest_path: Path,
        artifact_id: str,
        version: str,
        signature: Optional[SignatureInfo] = None,
        sbom_path: Optional[Path] = None,
        workspace_id: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> RegistryEntry:
        """
        Push artifact to registry.

        Args:
            artifact_path: Path to artifact package
            manifest_path: Path to manifest
            artifact_id: Artifact identifier
            version: Artifact version
            signature: Optional signature
            sbom_path: Optional SBOM path
            workspace_id: Optional workspace
            labels: Optional labels

        Returns:
            RegistryEntry

        Raises:
            ValueError: If artifact verification fails
        """
        # Compute digests
        artifact_digest = compute_file_digest(artifact_path)
        manifest_digest = compute_file_digest(manifest_path)

        # Verify signature if provided
        if signature and signature.signed_digest:
            if signature.signed_digest != artifact_digest:
                raise ValueError("Signature digest mismatch")

        # Copy files to blobs
        artifact_blob = self._blob_path(artifact_digest)
        manifest_blob = self._blob_path(manifest_digest)

        with self._lock:
            if not artifact_blob.exists():
                shutil.copy2(artifact_path, artifact_blob)

            if not manifest_blob.exists():
                shutil.copy2(manifest_path, manifest_blob)

            # Handle SBOM
            sbom_digest = None
            if sbom_path and sbom_path.exists():
                sbom_digest = compute_file_digest(sbom_path)
                sbom_blob = self._blob_path(sbom_digest)
                if not sbom_blob.exists():
                    shutil.copy2(sbom_path, sbom_blob)

            # Create entry
            entry = RegistryEntry(
                artifact_id=artifact_id,
                version=version,
                digest=artifact_digest,
                manifest_digest=manifest_digest,
                created_at=datetime.utcnow(),
                signature=signature,
                sbom_digest=sbom_digest,
                workspace_id=workspace_id,
                labels=labels or {},
            )

            # Add to index
            key = f"{artifact_id}:{version}"
            self._index[key] = entry

            # Also index by digest
            self._index[artifact_digest] = entry

            self._save_index()

        return entry

    def pull(self, digest: str, output_path: Path) -> bool:
        """
        Pull artifact by digest.

        Args:
            digest: Artifact digest
            output_path: Where to write artifact

        Returns:
            True if pulled, False if not found
        """
        blob_path = self._blob_path(digest)

        if not blob_path.exists():
            return False

        # Verify integrity
        if not verify_file_digest(blob_path, digest):
            return False

        shutil.copy2(blob_path, output_path)
        return True

    def get(self, artifact_id: str, version: str) -> Optional[RegistryEntry]:
        """Get registry entry."""
        key = f"{artifact_id}:{version}"
        with self._lock:
            return self._index.get(key)

    def get_by_digest(self, digest: str) -> Optional[RegistryEntry]:
        """Get entry by digest."""
        with self._lock:
            return self._index.get(digest)

    def list(
        self,
        workspace_id: Optional[str] = None,
        artifact_id: Optional[str] = None,
    ) -> List[RegistryEntry]:
        """List registry entries."""
        with self._lock:
            entries = list(self._index.values())

        # Filter
        if workspace_id:
            entries = [e for e in entries if e.workspace_id == workspace_id]
        if artifact_id:
            entries = [e for e in entries if e.artifact_id == artifact_id]

        # Deduplicate (same entry may be indexed by key and digest)
        seen = set()
        unique = []
        for e in entries:
            if e.digest not in seen:
                seen.add(e.digest)
                unique.append(e)

        return sorted(unique, key=lambda e: e.created_at, reverse=True)

    def exists(self, digest: str) -> bool:
        """Check if artifact exists."""
        return self._blob_path(digest).exists()

    def delete(self, artifact_id: str, version: str) -> bool:
        """Delete artifact from registry."""
        key = f"{artifact_id}:{version}"

        with self._lock:
            entry = self._index.get(key)
            if entry is None:
                return False

            # Remove from index
            del self._index[key]
            if entry.digest in self._index:
                del self._index[entry.digest]

            # Don't delete blobs (may be referenced by other entries)
            # Implement garbage collection separately if needed

            self._save_index()

        return True

    def _blob_path(self, digest: str) -> Path:
        """Get path for blob."""
        hash_part = digest.split(":")[-1]
        shard = hash_part[:4]
        shard_dir = self.blobs_path / shard
        shard_dir.mkdir(exist_ok=True)
        return shard_dir / hash_part

    def _save_index(self) -> None:
        """Save index to disk."""
        data = {}
        for key, entry in self._index.items():
            data[key] = {
                "artifact_id": entry.artifact_id,
                "version": entry.version,
                "digest": entry.digest,
                "manifest_digest": entry.manifest_digest,
                "created_at": entry.created_at.isoformat(),
                "signature": entry.signature.to_dict() if entry.signature else None,
                "sbom_digest": entry.sbom_digest,
                "workspace_id": entry.workspace_id,
                "labels": entry.labels,
            }

        with open(self.index_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_index(self) -> None:
        """Load index from disk."""
        if not self.index_path.exists():
            return

        try:
            with open(self.index_path, "r") as f:
                data = json.load(f)

            for key, entry_data in data.items():
                signature = None
                if entry_data.get("signature"):
                    signature = SignatureInfo.from_dict(entry_data["signature"])

                entry = RegistryEntry(
                    artifact_id=entry_data["artifact_id"],
                    version=entry_data["version"],
                    digest=entry_data["digest"],
                    manifest_digest=entry_data["manifest_digest"],
                    created_at=datetime.fromisoformat(entry_data["created_at"]),
                    signature=signature,
                    sbom_digest=entry_data.get("sbom_digest"),
                    workspace_id=entry_data.get("workspace_id"),
                    labels=entry_data.get("labels", {}),
                )
                self._index[key] = entry
        except Exception as e:
            # If index is corrupted, start fresh
            self._index = {}
