# -*- coding: utf-8 -*-
"""
Strategy Registry - Stores artifact metadata.

CLOUD ZONE ONLY.

Registry stores only metadata and references.
Actual artifacts are stored in artifact storage (OCI registry, S3, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from packages.shared.contracts.manifest import ArtifactManifest


@dataclass
class RegistryEntry:
    """Entry in the strategy registry."""

    entry_id: UUID = field(default_factory=uuid4)
    strategy_id: str = ""
    version: str = ""
    artifact_digest: str = ""
    manifest: Optional[ArtifactManifest] = None

    # Status
    is_active: bool = True
    is_deprecated: bool = False

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    deprecated_at: Optional[datetime] = None

    # Usage
    deployment_count: int = 0
    last_deployed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": str(self.entry_id),
            "strategy_id": self.strategy_id,
            "version": self.version,
            "artifact_digest": self.artifact_digest,
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "is_active": self.is_active,
            "is_deprecated": self.is_deprecated,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "deprecated_at": self.deprecated_at.isoformat() if self.deprecated_at else None,
            "deployment_count": self.deployment_count,
            "last_deployed_at": self.last_deployed_at.isoformat() if self.last_deployed_at else None,
        }


class StrategyRegistry:
    """
    Registry for strategy artifacts.

    Stores artifact metadata and provides lookup.

    Usage:
        registry = StrategyRegistry()

        # Register artifact
        entry = registry.register(manifest)

        # Look up by digest
        entry = registry.get_by_digest("sha256:abc...")

        # Get latest version
        entry = registry.get_latest("momentum_btc")
    """

    def __init__(self):
        """Initialize registry."""
        self._entries: Dict[UUID, RegistryEntry] = {}
        self._by_digest: Dict[str, RegistryEntry] = {}
        self._by_strategy: Dict[str, List[RegistryEntry]] = {}

    def register(self, manifest: ArtifactManifest) -> RegistryEntry:
        """
        Register artifact in registry.

        Args:
            manifest: Artifact manifest

        Returns:
            Created registry entry
        """
        # Check for existing entry with same digest
        if manifest.artifact_digest in self._by_digest:
            return self._by_digest[manifest.artifact_digest]

        # Create entry
        entry = RegistryEntry(
            strategy_id=manifest.strategy_id,
            version=manifest.version,
            artifact_digest=manifest.artifact_digest,
            manifest=manifest,
        )

        # Store
        self._entries[entry.entry_id] = entry
        self._by_digest[manifest.artifact_digest] = entry

        if manifest.strategy_id not in self._by_strategy:
            self._by_strategy[manifest.strategy_id] = []
        self._by_strategy[manifest.strategy_id].append(entry)

        return entry

    def get_by_digest(self, digest: str) -> Optional[RegistryEntry]:
        """Get entry by artifact digest."""
        return self._by_digest.get(digest)

    def get_by_id(self, entry_id: UUID) -> Optional[RegistryEntry]:
        """Get entry by ID."""
        return self._entries.get(entry_id)

    def get_latest(self, strategy_id: str) -> Optional[RegistryEntry]:
        """
        Get latest version of strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Latest non-deprecated entry or None
        """
        entries = self._by_strategy.get(strategy_id, [])
        active = [e for e in entries if e.is_active and not e.is_deprecated]
        if not active:
            return None
        # Sort by version (simple string sort - in production use semver)
        active.sort(key=lambda e: e.version, reverse=True)
        return active[0]

    def get_versions(self, strategy_id: str) -> List[RegistryEntry]:
        """Get all versions of strategy."""
        return self._by_strategy.get(strategy_id, [])

    def deprecate(self, digest: str, reason: str = "") -> bool:
        """
        Deprecate an artifact version.

        Args:
            digest: Artifact digest
            reason: Deprecation reason

        Returns:
            True if deprecated
        """
        entry = self._by_digest.get(digest)
        if not entry:
            return False

        entry.is_deprecated = True
        entry.deprecated_at = datetime.utcnow()
        entry.updated_at = datetime.utcnow()
        return True

    def record_deployment(self, digest: str) -> bool:
        """
        Record deployment of artifact.

        Args:
            digest: Artifact digest

        Returns:
            True if recorded
        """
        entry = self._by_digest.get(digest)
        if not entry:
            return False

        entry.deployment_count += 1
        entry.last_deployed_at = datetime.utcnow()
        entry.updated_at = datetime.utcnow()
        return True

    def list_strategies(self) -> List[str]:
        """List all registered strategy IDs."""
        return list(self._by_strategy.keys())

    def count(self) -> int:
        """Get total entry count."""
        return len(self._entries)
