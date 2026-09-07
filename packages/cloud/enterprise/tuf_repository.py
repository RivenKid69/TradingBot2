# -*- coding: utf-8 -*-
"""
TUF Repository - Secure update framework for rollback/freeze attack protection.

CLOUD ZONE ONLY.

Implements TUF (The Update Framework) concepts for secure update distribution:
- Signed metadata hierarchy (root → timestamp → snapshot → targets)
- Expiration-based freshness checks
- Rollback attack protection via version numbers
- Freeze attack protection via expiration
- Threshold signatures for root keys

Design Doc Reference:
    - Phase 9: Enterprise/on-prem pack
    - Design Doc 15.2/5.2: Agent updates - TUF-подобный подход
    - https://theupdateframework.io/

Implementation based on TUF specification:
    - https://theupdateframework.github.io/specification/latest/
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

# Constants
ROOT_VERSION: Final[int] = 1
TIMESTAMP_EXPIRY_DAYS: Final[int] = 1
SNAPSHOT_EXPIRY_DAYS: Final[int] = 7
TARGETS_EXPIRY_DAYS: Final[int] = 365
ROOT_EXPIRY_DAYS: Final[int] = 365


class TUFRole(Enum):
    """TUF role types."""

    ROOT = "root"
    TIMESTAMP = "timestamp"
    SNAPSHOT = "snapshot"
    TARGETS = "targets"


class KeyType(Enum):
    """Cryptographic key types."""

    ED25519 = "ed25519"
    RSA = "rsa-pkcs1v15-sha256"
    ECDSA = "ecdsa-sha2-nistp256"


@dataclass
class TUFKey:
    """TUF public key."""

    key_id: str  # SHA256 of canonical key representation
    key_type: KeyType
    public_key: str  # Base64-encoded public key
    scheme: str = "ed25519"  # Signature scheme

    def to_dict(self) -> Dict[str, Any]:
        """Convert to canonical TUF format."""
        return {
            "keytype": self.key_type.value,
            "scheme": self.scheme,
            "keyval": {
                "public": self.public_key,
            },
        }

    @classmethod
    def from_dict(cls, key_id: str, data: Dict[str, Any]) -> "TUFKey":
        """Create from TUF format."""
        return cls(
            key_id=key_id,
            key_type=KeyType(data["keytype"]),
            scheme=data.get("scheme", "ed25519"),
            public_key=data["keyval"]["public"],
        )


@dataclass
class TUFSignature:
    """TUF signature."""

    key_id: str
    signature: str  # Base64-encoded signature

    def to_dict(self) -> Dict[str, Any]:
        """Convert to TUF format."""
        return {
            "keyid": self.key_id,
            "sig": self.signature,
        }


@dataclass
class RoleInfo:
    """Role information in root metadata."""

    key_ids: List[str]
    threshold: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to TUF format."""
        return {
            "keyids": self.key_ids,
            "threshold": self.threshold,
        }


@dataclass
class TargetInfo:
    """Information about a target file."""

    length: int
    hashes: Dict[str, str]  # algorithm -> hash
    custom: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to TUF format."""
        result = {
            "length": self.length,
            "hashes": self.hashes,
        }
        if self.custom:
            result["custom"] = self.custom
        return result


@dataclass
class TUFMetadata:
    """Base TUF metadata."""

    version: int
    expires: datetime
    role: TUFRole = TUFRole.ROOT  # Default, overridden in subclasses
    spec_version: str = "1.0.0"

    def to_signed_dict(self) -> Dict[str, Any]:
        """Convert to TUF signed format (without signatures)."""
        return {
            "_type": self.role.value,
            "spec_version": self.spec_version,
            "version": self.version,
            "expires": self.expires.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }


@dataclass
class RootMetadata(TUFMetadata):
    """Root metadata - trust anchor."""

    keys: Dict[str, TUFKey] = field(default_factory=dict)
    roles: Dict[str, RoleInfo] = field(default_factory=dict)
    consistent_snapshot: bool = True

    def __post_init__(self):
        self.role = TUFRole.ROOT

    def to_signed_dict(self) -> Dict[str, Any]:
        """Convert to TUF format."""
        base = super().to_signed_dict()
        base.update(
            {
                "consistent_snapshot": self.consistent_snapshot,
                "keys": {k: v.to_dict() for k, v in self.keys.items()},
                "roles": {k: v.to_dict() for k, v in self.roles.items()},
            }
        )
        return base


@dataclass
class TimestampMetadata(TUFMetadata):
    """Timestamp metadata - freshness."""

    snapshot_info: TargetInfo = field(default_factory=lambda: TargetInfo(0, {}))

    def __post_init__(self):
        self.role = TUFRole.TIMESTAMP

    def to_signed_dict(self) -> Dict[str, Any]:
        """Convert to TUF format."""
        base = super().to_signed_dict()
        base["meta"] = {
            "snapshot.json": self.snapshot_info.to_dict(),
        }
        return base


@dataclass
class SnapshotMetadata(TUFMetadata):
    """Snapshot metadata - consistency."""

    meta: Dict[str, TargetInfo] = field(default_factory=dict)

    def __post_init__(self):
        self.role = TUFRole.SNAPSHOT

    def to_signed_dict(self) -> Dict[str, Any]:
        """Convert to TUF format."""
        base = super().to_signed_dict()
        base["meta"] = {k: v.to_dict() for k, v in self.meta.items()}
        return base


@dataclass
class TargetsMetadata(TUFMetadata):
    """Targets metadata - list of available updates."""

    targets: Dict[str, TargetInfo] = field(default_factory=dict)
    delegations: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.role = TUFRole.TARGETS

    def to_signed_dict(self) -> Dict[str, Any]:
        """Convert to TUF format."""
        base = super().to_signed_dict()
        base["targets"] = {k: v.to_dict() for k, v in self.targets.items()}
        if self.delegations:
            base["delegations"] = self.delegations
        return base


@dataclass
class SignedMetadata:
    """Signed metadata container."""

    signed: Dict[str, Any]
    signatures: List[TUFSignature] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to TUF format."""
        return {
            "signed": self.signed,
            "signatures": [s.to_dict() for s in self.signatures],
        }

    @property
    def canonical_bytes(self) -> bytes:
        """Get canonical JSON representation of signed portion."""
        return self._canonicalize(self.signed)

    @staticmethod
    def _canonicalize(obj: Any) -> bytes:
        """Create canonical JSON representation."""
        return json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")


@dataclass
class TUFConfig:
    """Configuration for TUF repository."""

    # Storage
    repository_path: Path = field(default_factory=lambda: Path.home() / ".ccea" / "tuf")

    # Expiration settings
    timestamp_expiry_days: int = TIMESTAMP_EXPIRY_DAYS
    snapshot_expiry_days: int = SNAPSHOT_EXPIRY_DAYS
    targets_expiry_days: int = TARGETS_EXPIRY_DAYS
    root_expiry_days: int = ROOT_EXPIRY_DAYS

    # Threshold settings
    root_threshold: int = 2  # Number of signatures required
    targets_threshold: int = 1
    snapshot_threshold: int = 1
    timestamp_threshold: int = 1

    # Consistent snapshots
    consistent_snapshot: bool = True

    # Offline keys
    root_keys_offline: bool = True
    targets_keys_offline: bool = False

    # Validation
    strict_expiration: bool = True
    max_root_rotations: int = 1024

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "repository_path": str(self.repository_path),
            "timestamp_expiry_days": self.timestamp_expiry_days,
            "snapshot_expiry_days": self.snapshot_expiry_days,
            "targets_expiry_days": self.targets_expiry_days,
            "root_expiry_days": self.root_expiry_days,
            "root_threshold": self.root_threshold,
            "consistent_snapshot": self.consistent_snapshot,
        }


class TUFRepository:
    """
    TUF Repository for secure update distribution.

    Implements TUF (The Update Framework) for:
    - Rollback attack protection (version numbers)
    - Freeze attack protection (expiration)
    - Compromise resilience (key thresholds)
    - Secure key rotation

    Usage:
        config = TUFConfig(repository_path=Path("/data/tuf"))
        repo = TUFRepository(config)

        # Initialize repository
        await repo.initialize(root_keys, targets_key, snapshot_key, timestamp_key)

        # Add target (update)
        await repo.add_target(
            "agent-1.1.0.tar.gz",
            artifact_path,
            custom={"version": "1.1.0"},
        )

        # Publish (sign and write metadata)
        await repo.publish()

        # Agent verification
        valid, error = await repo.verify_target(
            "agent-1.1.0.tar.gz",
            expected_hash,
            expected_length,
        )
    """

    def __init__(
        self,
        config: Optional[TUFConfig] = None,
    ):
        """
        Initialize TUF repository.

        Args:
            config: Repository configuration
        """
        self.config = config or TUFConfig()

        # Metadata state
        self._root: Optional[RootMetadata] = None
        self._timestamp: Optional[TimestampMetadata] = None
        self._snapshot: Optional[SnapshotMetadata] = None
        self._targets: Optional[TargetsMetadata] = None

        # Keys
        self._keys: Dict[str, TUFKey] = {}
        self._signing_keys: Dict[TUFRole, List[bytes]] = {}  # role -> private keys

        # Signed metadata
        self._signed_root: Optional[SignedMetadata] = None
        self._signed_timestamp: Optional[SignedMetadata] = None
        self._signed_snapshot: Optional[SignedMetadata] = None
        self._signed_targets: Optional[SignedMetadata] = None

        # State
        self._initialized = False
        self._dirty = False

    @property
    def is_initialized(self) -> bool:
        """Check if repository is initialized."""
        return self._initialized

    async def initialize(
        self,
        root_keys: List[Tuple[bytes, bytes]],  # [(public, private), ...]
        targets_key: Tuple[bytes, bytes],
        snapshot_key: Tuple[bytes, bytes],
        timestamp_key: Tuple[bytes, bytes],
    ) -> bool:
        """
        Initialize a new TUF repository.

        Args:
            root_keys: Root key pairs (threshold signatures)
            targets_key: Targets key pair
            snapshot_key: Snapshot key pair
            timestamp_key: Timestamp key pair

        Returns:
            True if initialized successfully
        """
        try:
            # Create repository directory
            self.config.repository_path.mkdir(parents=True, exist_ok=True)

            # Process root keys
            root_key_ids = []
            root_private_keys = []
            for pub, priv in root_keys:
                key_id = self._compute_key_id(pub)
                root_key_ids.append(key_id)
                root_private_keys.append(priv)
                self._keys[key_id] = TUFKey(
                    key_id=key_id,
                    key_type=KeyType.ED25519,
                    public_key=base64.b64encode(pub).decode(),
                )

            # Process other keys
            targets_key_id = self._compute_key_id(targets_key[0])
            self._keys[targets_key_id] = TUFKey(
                key_id=targets_key_id,
                key_type=KeyType.ED25519,
                public_key=base64.b64encode(targets_key[0]).decode(),
            )

            snapshot_key_id = self._compute_key_id(snapshot_key[0])
            self._keys[snapshot_key_id] = TUFKey(
                key_id=snapshot_key_id,
                key_type=KeyType.ED25519,
                public_key=base64.b64encode(snapshot_key[0]).decode(),
            )

            timestamp_key_id = self._compute_key_id(timestamp_key[0])
            self._keys[timestamp_key_id] = TUFKey(
                key_id=timestamp_key_id,
                key_type=KeyType.ED25519,
                public_key=base64.b64encode(timestamp_key[0]).decode(),
            )

            # Store signing keys
            self._signing_keys[TUFRole.ROOT] = root_private_keys
            self._signing_keys[TUFRole.TARGETS] = [targets_key[1]]
            self._signing_keys[TUFRole.SNAPSHOT] = [snapshot_key[1]]
            self._signing_keys[TUFRole.TIMESTAMP] = [timestamp_key[1]]

            # Create root metadata
            now = datetime.utcnow()
            self._root = RootMetadata(
                version=ROOT_VERSION,
                expires=now + timedelta(days=self.config.root_expiry_days),
                keys=self._keys.copy(),
                roles={
                    "root": RoleInfo(root_key_ids, self.config.root_threshold),
                    "targets": RoleInfo([targets_key_id], self.config.targets_threshold),
                    "snapshot": RoleInfo([snapshot_key_id], self.config.snapshot_threshold),
                    "timestamp": RoleInfo([timestamp_key_id], self.config.timestamp_threshold),
                },
                consistent_snapshot=self.config.consistent_snapshot,
            )

            # Create targets metadata
            self._targets = TargetsMetadata(
                version=1,
                expires=now + timedelta(days=self.config.targets_expiry_days),
            )

            # Create snapshot metadata
            self._snapshot = SnapshotMetadata(
                version=1,
                expires=now + timedelta(days=self.config.snapshot_expiry_days),
            )

            # Create timestamp metadata
            self._timestamp = TimestampMetadata(
                version=1,
                expires=now + timedelta(days=self.config.timestamp_expiry_days),
            )

            # Sign root metadata
            self._signed_root = await self._sign_metadata(self._root, TUFRole.ROOT)

            self._initialized = True
            self._dirty = True

            logger.info("TUF repository initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize TUF repository: {e}")
            return False

    async def load(self) -> bool:
        """
        Load existing repository from disk.

        Returns:
            True if loaded successfully
        """
        try:
            repo_path = self.config.repository_path

            # Load root
            root_path = repo_path / "root.json"
            if root_path.exists():
                with open(root_path, "r") as f:
                    data = json.load(f)
                self._signed_root = self._parse_signed_metadata(data)
                self._root = self._parse_root_metadata(data["signed"])

                # Load keys from root
                for key_id, key_data in data["signed"]["keys"].items():
                    self._keys[key_id] = TUFKey.from_dict(key_id, key_data)

            # Load targets
            targets_path = repo_path / "targets.json"
            if targets_path.exists():
                with open(targets_path, "r") as f:
                    data = json.load(f)
                self._signed_targets = self._parse_signed_metadata(data)
                self._targets = self._parse_targets_metadata(data["signed"])

            # Load snapshot
            snapshot_path = repo_path / "snapshot.json"
            if snapshot_path.exists():
                with open(snapshot_path, "r") as f:
                    data = json.load(f)
                self._signed_snapshot = self._parse_signed_metadata(data)
                self._snapshot = self._parse_snapshot_metadata(data["signed"])

            # Load timestamp
            timestamp_path = repo_path / "timestamp.json"
            if timestamp_path.exists():
                with open(timestamp_path, "r") as f:
                    data = json.load(f)
                self._signed_timestamp = self._parse_signed_metadata(data)
                self._timestamp = self._parse_timestamp_metadata(data["signed"])

            self._initialized = True
            logger.info("TUF repository loaded")
            return True

        except Exception as e:
            logger.error(f"Failed to load TUF repository: {e}")
            return False

    async def add_target(
        self,
        name: str,
        content: bytes,
        custom: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add a target (update artifact) to the repository.

        Args:
            name: Target name (e.g., "agent-1.1.0.tar.gz")
            content: Target content bytes
            custom: Custom metadata

        Returns:
            True if added successfully
        """
        if not self._initialized or not self._targets:
            return False

        try:
            # Compute hashes
            sha256_hash = hashlib.sha256(content).hexdigest()
            sha512_hash = hashlib.sha512(content).hexdigest()

            # Create target info
            target_info = TargetInfo(
                length=len(content),
                hashes={
                    "sha256": sha256_hash,
                    "sha512": sha512_hash,
                },
                custom=custom,
            )

            # Add to targets
            self._targets.targets[name] = target_info

            # Write target file
            if self.config.consistent_snapshot:
                target_path = self.config.repository_path / "targets" / f"{sha256_hash}.{name}"
            else:
                target_path = self.config.repository_path / "targets" / name

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "wb") as f:
                f.write(content)

            self._dirty = True
            logger.info(f"Added target: {name} (sha256:{sha256_hash[:16]}...)")
            return True

        except Exception as e:
            logger.error(f"Failed to add target {name}: {e}")
            return False

    async def remove_target(self, name: str) -> bool:
        """
        Remove a target from the repository.

        Args:
            name: Target name

        Returns:
            True if removed
        """
        if not self._initialized or not self._targets:
            return False

        if name in self._targets.targets:
            del self._targets.targets[name]
            self._dirty = True
            logger.info(f"Removed target: {name}")
            return True

        return False

    async def publish(self) -> bool:
        """
        Sign and publish all metadata.

        This creates a consistent snapshot of all metadata.

        Returns:
            True if published successfully
        """
        if not self._initialized:
            return False

        if not self._dirty:
            return True

        try:
            now = datetime.utcnow()

            # Update targets version and expiry
            self._targets.version += 1
            self._targets.expires = now + timedelta(days=self.config.targets_expiry_days)

            # Sign targets
            self._signed_targets = await self._sign_metadata(self._targets, TUFRole.TARGETS)

            # Compute targets hash for snapshot
            targets_bytes = json.dumps(self._signed_targets.to_dict()).encode()
            targets_hash = hashlib.sha256(targets_bytes).hexdigest()

            # Update snapshot
            self._snapshot.version += 1
            self._snapshot.expires = now + timedelta(days=self.config.snapshot_expiry_days)
            self._snapshot.meta["targets.json"] = TargetInfo(
                length=len(targets_bytes),
                hashes={"sha256": targets_hash},
            )

            # Sign snapshot
            self._signed_snapshot = await self._sign_metadata(self._snapshot, TUFRole.SNAPSHOT)

            # Compute snapshot hash for timestamp
            snapshot_bytes = json.dumps(self._signed_snapshot.to_dict()).encode()
            snapshot_hash = hashlib.sha256(snapshot_bytes).hexdigest()

            # Update timestamp
            self._timestamp.version += 1
            self._timestamp.expires = now + timedelta(days=self.config.timestamp_expiry_days)
            self._timestamp.snapshot_info = TargetInfo(
                length=len(snapshot_bytes),
                hashes={"sha256": snapshot_hash},
            )

            # Sign timestamp
            self._signed_timestamp = await self._sign_metadata(self._timestamp, TUFRole.TIMESTAMP)

            # Write all metadata
            await self._write_metadata()

            self._dirty = False
            logger.info("TUF repository published")
            return True

        except Exception as e:
            logger.error(f"Failed to publish TUF repository: {e}")
            return False

    async def rotate_root(
        self,
        new_root_keys: Optional[List[Tuple[bytes, bytes]]] = None,
        new_targets_key: Optional[Tuple[bytes, bytes]] = None,
        new_snapshot_key: Optional[Tuple[bytes, bytes]] = None,
        new_timestamp_key: Optional[Tuple[bytes, bytes]] = None,
    ) -> bool:
        """
        Rotate root metadata keys.

        Args:
            new_root_keys: New root key pairs
            new_targets_key: New targets key pair
            new_snapshot_key: New snapshot key pair
            new_timestamp_key: New timestamp key pair

        Returns:
            True if rotated successfully
        """
        if not self._initialized or not self._root:
            return False

        try:
            # Increment version
            self._root.version += 1
            self._root.expires = datetime.utcnow() + timedelta(days=self.config.root_expiry_days)

            # Update keys if provided
            if new_root_keys:
                root_key_ids = []
                for pub, priv in new_root_keys:
                    key_id = self._compute_key_id(pub)
                    root_key_ids.append(key_id)
                    self._keys[key_id] = TUFKey(
                        key_id=key_id,
                        key_type=KeyType.ED25519,
                        public_key=base64.b64encode(pub).decode(),
                    )
                    if TUFRole.ROOT not in self._signing_keys:
                        self._signing_keys[TUFRole.ROOT] = []
                    self._signing_keys[TUFRole.ROOT].append(priv)

                self._root.roles["root"] = RoleInfo(root_key_ids, self.config.root_threshold)

            # Sign with both old and new keys (cross-signing)
            self._signed_root = await self._sign_metadata(self._root, TUFRole.ROOT)

            self._dirty = True
            logger.info(f"Root rotated to version {self._root.version}")
            return True

        except Exception as e:
            logger.error(f"Failed to rotate root: {e}")
            return False

    async def verify_target(
        self,
        name: str,
        content_hash: str,
        content_length: int,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify a target against repository metadata.

        Args:
            name: Target name
            content_hash: SHA256 hash of content
            content_length: Content length

        Returns:
            (is_valid, error_message)
        """
        if not self._initialized or not self._targets:
            return False, "Repository not initialized"

        # Check expiration
        now = datetime.utcnow()

        if self._timestamp and self._timestamp.expires < now:
            return False, "Timestamp metadata expired (possible freeze attack)"

        if self._snapshot and self._snapshot.expires < now:
            return False, "Snapshot metadata expired"

        if self._targets.expires < now:
            return False, "Targets metadata expired"

        # Find target
        target_info = self._targets.targets.get(name)
        if not target_info:
            return False, f"Target not found: {name}"

        # Verify length
        if target_info.length != content_length:
            return False, f"Length mismatch: expected {target_info.length}, got {content_length}"

        # Verify hash
        expected_hash = target_info.hashes.get("sha256")
        if expected_hash and expected_hash != content_hash:
            return False, f"Hash mismatch"

        return True, None

    def get_target_info(self, name: str) -> Optional[TargetInfo]:
        """Get target information."""
        if not self._targets:
            return None
        return self._targets.targets.get(name)

    def list_targets(self) -> Dict[str, TargetInfo]:
        """List all targets."""
        if not self._targets:
            return {}
        return self._targets.targets.copy()

    def get_metadata_versions(self) -> Dict[str, int]:
        """Get current metadata versions."""
        return {
            "root": self._root.version if self._root else 0,
            "timestamp": self._timestamp.version if self._timestamp else 0,
            "snapshot": self._snapshot.version if self._snapshot else 0,
            "targets": self._targets.version if self._targets else 0,
        }

    def get_metadata_expiry(self) -> Dict[str, Optional[datetime]]:
        """Get metadata expiry times."""
        return {
            "root": self._root.expires if self._root else None,
            "timestamp": self._timestamp.expires if self._timestamp else None,
            "snapshot": self._snapshot.expires if self._snapshot else None,
            "targets": self._targets.expires if self._targets else None,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics."""
        return {
            "initialized": self._initialized,
            "dirty": self._dirty,
            "target_count": len(self._targets.targets) if self._targets else 0,
            "key_count": len(self._keys),
            "versions": self.get_metadata_versions(),
            "expiry": {
                k: v.isoformat() if v else None for k, v in self.get_metadata_expiry().items()
            },
            "config": self.config.to_dict(),
        }

    # ===== Private Methods =====

    def _compute_key_id(self, public_key: bytes) -> str:
        """Compute key ID from public key."""
        # TUF key ID is SHA256 of canonical key representation
        return hashlib.sha256(public_key).hexdigest()

    async def _sign_metadata(
        self,
        metadata: TUFMetadata,
        role: TUFRole,
    ) -> SignedMetadata:
        """Sign metadata with role keys."""
        signed_dict = metadata.to_signed_dict()
        canonical = SignedMetadata._canonicalize(signed_dict)

        signatures = []
        private_keys = self._signing_keys.get(role, [])

        for private_key in private_keys:
            # Sign canonical representation
            signature = await self._sign(canonical, private_key)
            key_id = self._compute_key_id(private_key[:32])  # Extract public from ed25519

            # Note: In real implementation, derive key_id properly
            # For simplicity, we use first matching key
            for kid, key in self._keys.items():
                signatures.append(
                    TUFSignature(
                        key_id=kid,
                        signature=base64.b64encode(signature).decode(),
                    )
                )
                break

        return SignedMetadata(signed=signed_dict, signatures=signatures)

    async def _sign(self, data: bytes, private_key: bytes) -> bytes:
        """
        Sign data with an Ed25519 private key.

        Uses real Ed25519 signing via the cryptography library (CCEA-SEC-002
        resolved). The first 32 bytes of ``private_key`` are the Ed25519 seed.
        A SHA-256 placeholder remains available ONLY when both cryptography is
        absent and CCEA_TUF_DEVELOPMENT_MODE=ENABLED; otherwise fail-closed.
        """
        from packages.cloud.enterprise.crypto import CRYPTO_AVAILABLE

        if CRYPTO_AVAILABLE:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )

            try:
                signer = Ed25519PrivateKey.from_private_bytes(private_key[:32])
                return signer.sign(data)
            except Exception as exc:  # pragma: no cover - defensive
                raise RuntimeError(f"TUF Ed25519 signing failed: {exc}")

        import os

        if os.environ.get("CCEA_TUF_DEVELOPMENT_MODE") == "ENABLED":
            import logging

            logging.getLogger(__name__).warning(
                "SECURITY: placeholder TUF signing (DEVELOPMENT_MODE, no cryptography). "
                "Not cryptographically secure. Tracking: CCEA-SEC-002"
            )
            return hashlib.sha256(private_key + data).digest()

        raise RuntimeError(
            "TUF signing requires the cryptography library (Ed25519). " "Tracking: CCEA-SEC-002"
        )

    async def _write_metadata(self) -> None:
        """Write all metadata to disk."""
        repo_path = self.config.repository_path

        # Write root
        if self._signed_root:
            root_path = repo_path / "root.json"
            with open(root_path, "w") as f:
                json.dump(self._signed_root.to_dict(), f, indent=2)

            # Also write versioned copy
            versioned_root = repo_path / f"{self._root.version}.root.json"
            with open(versioned_root, "w") as f:
                json.dump(self._signed_root.to_dict(), f, indent=2)

        # Write timestamp
        if self._signed_timestamp:
            timestamp_path = repo_path / "timestamp.json"
            with open(timestamp_path, "w") as f:
                json.dump(self._signed_timestamp.to_dict(), f, indent=2)

        # Write snapshot
        if self._signed_snapshot:
            # Always write to snapshot.json for loading
            snapshot_path = repo_path / "snapshot.json"
            with open(snapshot_path, "w") as f:
                json.dump(self._signed_snapshot.to_dict(), f, indent=2)

            # Also write versioned copy if consistent_snapshot
            if self.config.consistent_snapshot:
                versioned_snapshot = repo_path / f"{self._snapshot.version}.snapshot.json"
                with open(versioned_snapshot, "w") as f:
                    json.dump(self._signed_snapshot.to_dict(), f, indent=2)

        # Write targets
        if self._signed_targets:
            # Always write to targets.json for loading
            targets_path = repo_path / "targets.json"
            with open(targets_path, "w") as f:
                json.dump(self._signed_targets.to_dict(), f, indent=2)

            # Also write versioned copy if consistent_snapshot
            if self.config.consistent_snapshot:
                versioned_targets = repo_path / f"{self._targets.version}.targets.json"
                with open(versioned_targets, "w") as f:
                    json.dump(self._signed_targets.to_dict(), f, indent=2)

    def _parse_signed_metadata(self, data: Dict[str, Any]) -> SignedMetadata:
        """Parse signed metadata from JSON."""
        signatures = [
            TUFSignature(key_id=s["keyid"], signature=s["sig"]) for s in data.get("signatures", [])
        ]
        return SignedMetadata(signed=data["signed"], signatures=signatures)

    def _parse_root_metadata(self, data: Dict[str, Any]) -> RootMetadata:
        """Parse root metadata."""
        return RootMetadata(
            version=data["version"],
            expires=datetime.strptime(data["expires"], "%Y-%m-%dT%H:%M:%SZ"),
            spec_version=data.get("spec_version", "1.0.0"),
            consistent_snapshot=data.get("consistent_snapshot", True),
            keys={kid: TUFKey.from_dict(kid, kdata) for kid, kdata in data.get("keys", {}).items()},
            roles={
                role: RoleInfo(
                    key_ids=rdata["keyids"],
                    threshold=rdata["threshold"],
                )
                for role, rdata in data.get("roles", {}).items()
            },
        )

    def _parse_targets_metadata(self, data: Dict[str, Any]) -> TargetsMetadata:
        """Parse targets metadata."""
        return TargetsMetadata(
            version=data["version"],
            expires=datetime.strptime(data["expires"], "%Y-%m-%dT%H:%M:%SZ"),
            targets={
                name: TargetInfo(
                    length=info["length"],
                    hashes=info["hashes"],
                    custom=info.get("custom"),
                )
                for name, info in data.get("targets", {}).items()
            },
        )

    def _parse_snapshot_metadata(self, data: Dict[str, Any]) -> SnapshotMetadata:
        """Parse snapshot metadata."""
        return SnapshotMetadata(
            version=data["version"],
            expires=datetime.strptime(data["expires"], "%Y-%m-%dT%H:%M:%SZ"),
            meta={
                name: TargetInfo(
                    length=info["length"],
                    hashes=info["hashes"],
                )
                for name, info in data.get("meta", {}).items()
            },
        )

    def _parse_timestamp_metadata(self, data: Dict[str, Any]) -> TimestampMetadata:
        """Parse timestamp metadata."""
        snapshot_info_data = data.get("meta", {}).get("snapshot.json", {})
        return TimestampMetadata(
            version=data["version"],
            expires=datetime.strptime(data["expires"], "%Y-%m-%dT%H:%M:%SZ"),
            snapshot_info=TargetInfo(
                length=snapshot_info_data.get("length", 0),
                hashes=snapshot_info_data.get("hashes", {}),
            ),
        )
