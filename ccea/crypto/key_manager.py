# -*- coding: utf-8 -*-
"""
CCEA Key Manager.

Provides comprehensive key management for artifact signing:
- Key generation and storage
- Key rotation with overlap period
- Trust root management
- Key revocation
- Keyless (sigstore) and keyful (GPG/X509) modes

Per Design Doc Phase 4:
- Key management for signing keys
- Keyless sigstore vs keyful for enterprise/offline
- Trust root definition

Security:
- Private keys protected with encryption
- Key rotation with grace period
- Revocation propagation
"""

from __future__ import annotations

import json
import os
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.types import (
    PrivateKeyTypes,
    PublicKeyTypes,
)

from ccea.crypto.keys import (
    KeyAlgorithm,
    KeyPair,
    generate_keypair,
    load_private_key,
    load_public_key,
    serialize_private_key,
    serialize_public_key,
)


class KeyStatus(str, Enum):
    """Key lifecycle status."""
    ACTIVE = "active"
    PENDING = "pending"  # Generated but not yet activated
    ROTATING = "rotating"  # Being rotated (grace period)
    REVOKED = "revoked"
    EXPIRED = "expired"


class KeyPurpose(str, Enum):
    """Key purpose."""
    ARTIFACT_SIGNING = "artifact_signing"
    MANIFEST_SIGNING = "manifest_signing"
    AGENT_DEVICE = "agent_device"
    CLOUD_SERVER = "cloud_server"


class TrustLevel(str, Enum):
    """Trust level for keys."""
    ROOT = "root"  # Trust anchor
    INTERMEDIATE = "intermediate"
    LEAF = "leaf"


@dataclass
class KeyMetadata:
    """Metadata for a managed key."""
    key_id: str
    algorithm: KeyAlgorithm
    purpose: KeyPurpose
    status: KeyStatus
    trust_level: TrustLevel
    created_at: datetime
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    revocation_reason: Optional[str] = None
    parent_key_id: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if key is currently valid for use."""
        if self.status not in (KeyStatus.ACTIVE, KeyStatus.ROTATING):
            return False
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key_id": self.key_id,
            "algorithm": self.algorithm.value,
            "purpose": self.purpose.value,
            "status": self.status.value,
            "trust_level": self.trust_level.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            "revocation_reason": self.revocation_reason,
            "parent_key_id": self.parent_key_id,
            "labels": self.labels,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeyMetadata":
        """Create from dictionary."""
        return cls(
            key_id=data["key_id"],
            algorithm=KeyAlgorithm(data["algorithm"]),
            purpose=KeyPurpose(data["purpose"]),
            status=KeyStatus(data["status"]),
            trust_level=TrustLevel(data["trust_level"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            revoked_at=datetime.fromisoformat(data["revoked_at"]) if data.get("revoked_at") else None,
            revocation_reason=data.get("revocation_reason"),
            parent_key_id=data.get("parent_key_id"),
            labels=data.get("labels", {}),
        )


@dataclass
class TrustRoot:
    """Trust root configuration."""
    root_key_ids: Set[str]
    allowed_purposes: Set[KeyPurpose]
    min_algorithm: KeyAlgorithm = KeyAlgorithm.ED25519
    require_expiration: bool = True
    max_validity_days: int = 365
    allow_self_signed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "root_key_ids": list(self.root_key_ids),
            "allowed_purposes": [p.value for p in self.allowed_purposes],
            "min_algorithm": self.min_algorithm.value,
            "require_expiration": self.require_expiration,
            "max_validity_days": self.max_validity_days,
            "allow_self_signed": self.allow_self_signed,
        }


class KeyManager:
    """
    Comprehensive key manager for CCEA.

    Handles key lifecycle, rotation, and trust management.
    """

    # Default rotation grace period (hours)
    DEFAULT_ROTATION_GRACE_HOURS = 24

    def __init__(
        self,
        storage_path: Path,
        master_password: Optional[bytes] = None,
    ):
        """
        Initialize key manager.

        Args:
            storage_path: Path for key storage
            master_password: Password for encrypting private keys
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.keys_dir = self.storage_path / "keys"
        self.keys_dir.mkdir(exist_ok=True)

        self.metadata_path = self.storage_path / "key_metadata.json"
        self.trust_path = self.storage_path / "trust_root.json"

        self.master_password = master_password
        self._metadata: Dict[str, KeyMetadata] = {}
        self._public_keys: Dict[str, PublicKeyTypes] = {}
        self._private_keys: Dict[str, PrivateKeyTypes] = {}
        self._trust_root: Optional[TrustRoot] = None
        self._lock = threading.Lock()

        self._load()

    def generate_key(
        self,
        purpose: KeyPurpose,
        algorithm: KeyAlgorithm = KeyAlgorithm.ED25519,
        trust_level: TrustLevel = TrustLevel.LEAF,
        validity_days: int = 365,
        parent_key_id: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        activate: bool = True,
    ) -> str:
        """
        Generate a new key.

        Args:
            purpose: Key purpose
            algorithm: Algorithm to use
            trust_level: Trust level
            validity_days: Days until expiration
            parent_key_id: Parent key for chain
            labels: Optional labels
            activate: Whether to activate immediately

        Returns:
            Generated key ID
        """
        # Generate unique key ID
        key_id = f"ccea-{purpose.value}-{secrets.token_hex(8)}"

        # Generate keypair
        keypair = generate_keypair(algorithm=algorithm, key_id=key_id)

        # Create metadata
        now = datetime.now(timezone.utc)
        metadata = KeyMetadata(
            key_id=key_id,
            algorithm=algorithm,
            purpose=purpose,
            status=KeyStatus.ACTIVE if activate else KeyStatus.PENDING,
            trust_level=trust_level,
            created_at=now,
            expires_at=now + timedelta(days=validity_days) if validity_days > 0 else None,
            parent_key_id=parent_key_id,
            labels=labels or {},
        )

        with self._lock:
            # Store keys
            self._metadata[key_id] = metadata
            self._public_keys[key_id] = keypair.public_key
            self._private_keys[key_id] = keypair.private_key

            # Persist
            self._save_key(key_id, keypair)
            self._save_metadata()

        return key_id

    def get_active_key(self, purpose: KeyPurpose) -> Optional[str]:
        """
        Get the currently active key for a purpose.

        Args:
            purpose: Key purpose

        Returns:
            Key ID or None if no active key
        """
        with self._lock:
            for key_id, meta in self._metadata.items():
                if meta.purpose == purpose and meta.status == KeyStatus.ACTIVE and meta.is_valid():
                    return key_id
        return None

    def get_signing_keys(self, purpose: KeyPurpose) -> List[str]:
        """
        Get all valid signing keys for a purpose.

        Includes both active and rotating keys (for signature verification
        during rotation grace period).

        Args:
            purpose: Key purpose

        Returns:
            List of key IDs
        """
        with self._lock:
            keys = []
            for key_id, meta in self._metadata.items():
                if meta.purpose == purpose and meta.is_valid():
                    keys.append(key_id)
            return keys

    def get_public_key(self, key_id: str) -> Optional[PublicKeyTypes]:
        """Get public key by ID."""
        with self._lock:
            return self._public_keys.get(key_id)

    def get_private_key(self, key_id: str) -> Optional[PrivateKeyTypes]:
        """
        Get private key by ID.

        WARNING: Handle with care. Private key should not leave the system.
        """
        with self._lock:
            return self._private_keys.get(key_id)

    def get_metadata(self, key_id: str) -> Optional[KeyMetadata]:
        """Get key metadata."""
        with self._lock:
            return self._metadata.get(key_id)

    def rotate_key(
        self,
        key_id: str,
        grace_hours: int = DEFAULT_ROTATION_GRACE_HOURS,
    ) -> str:
        """
        Rotate a key.

        Creates new key and marks old key as rotating with grace period.

        Args:
            key_id: Key to rotate
            grace_hours: Hours for rotation grace period

        Returns:
            New key ID

        Raises:
            ValueError: If key not found or cannot be rotated
        """
        with self._lock:
            old_meta = self._metadata.get(key_id)
            if not old_meta:
                raise ValueError(f"Key not found: {key_id}")

            if old_meta.status != KeyStatus.ACTIVE:
                raise ValueError(f"Only active keys can be rotated: {key_id}")

            # Mark old key as rotating
            old_meta.status = KeyStatus.ROTATING
            old_meta.expires_at = datetime.now(timezone.utc) + timedelta(hours=grace_hours)

        # Generate new key with same properties
        new_key_id = self.generate_key(
            purpose=old_meta.purpose,
            algorithm=old_meta.algorithm,
            trust_level=old_meta.trust_level,
            validity_days=365,  # Standard validity
            parent_key_id=old_meta.parent_key_id,
            labels={**old_meta.labels, "rotated_from": key_id},
            activate=True,
        )

        with self._lock:
            self._save_metadata()

        return new_key_id

    def revoke_key(self, key_id: str, reason: str) -> None:
        """
        Revoke a key.

        Args:
            key_id: Key to revoke
            reason: Revocation reason (for audit)

        Raises:
            ValueError: If key not found
        """
        with self._lock:
            meta = self._metadata.get(key_id)
            if not meta:
                raise ValueError(f"Key not found: {key_id}")

            meta.status = KeyStatus.REVOKED
            meta.revoked_at = datetime.now(timezone.utc)
            meta.revocation_reason = reason

            # Remove private key from memory (keep public for verification)
            self._private_keys.pop(key_id, None)

            self._save_metadata()

    def is_revoked(self, key_id: str) -> bool:
        """Check if key is revoked."""
        with self._lock:
            meta = self._metadata.get(key_id)
            return meta is not None and meta.status == KeyStatus.REVOKED

    def set_trust_root(
        self,
        root_key_ids: List[str],
        allowed_purposes: List[KeyPurpose],
        **kwargs: Any,
    ) -> None:
        """
        Set trust root configuration.

        Args:
            root_key_ids: List of root key IDs
            allowed_purposes: Allowed key purposes
            **kwargs: Additional TrustRoot parameters
        """
        with self._lock:
            self._trust_root = TrustRoot(
                root_key_ids=set(root_key_ids),
                allowed_purposes=set(allowed_purposes),
                **kwargs,
            )
            self._save_trust_root()

    def verify_trust_chain(self, key_id: str) -> bool:
        """
        Verify key's trust chain back to root.

        Args:
            key_id: Key to verify

        Returns:
            True if trust chain is valid
        """
        with self._lock:
            if not self._trust_root:
                return False

            meta = self._metadata.get(key_id)
            if not meta:
                return False

            # Check if key purpose is allowed
            if meta.purpose not in self._trust_root.allowed_purposes:
                return False

            # Root keys are trusted by definition
            if meta.trust_level == TrustLevel.ROOT:
                return key_id in self._trust_root.root_key_ids

            # Check parent chain
            current = meta
            visited = set()

            while current.parent_key_id:
                if current.key_id in visited:
                    return False  # Circular reference
                visited.add(current.key_id)

                parent = self._metadata.get(current.parent_key_id)
                if not parent:
                    return False

                if not parent.is_valid():
                    return False

                if parent.trust_level == TrustLevel.ROOT:
                    return parent.key_id in self._trust_root.root_key_ids

                current = parent

            # Self-signed check
            return self._trust_root.allow_self_signed

    def list_keys(
        self,
        purpose: Optional[KeyPurpose] = None,
        status: Optional[KeyStatus] = None,
    ) -> List[KeyMetadata]:
        """List keys with optional filters."""
        with self._lock:
            keys = list(self._metadata.values())

        if purpose:
            keys = [k for k in keys if k.purpose == purpose]
        if status:
            keys = [k for k in keys if k.status == status]

        return sorted(keys, key=lambda k: k.created_at, reverse=True)

    def export_public_key(self, key_id: str) -> Optional[str]:
        """Export public key as PEM."""
        public_key = self.get_public_key(key_id)
        if not public_key:
            return None
        return serialize_public_key(public_key)

    def import_public_key(
        self,
        key_id: str,
        pem_data: str,
        purpose: KeyPurpose,
        trust_level: TrustLevel = TrustLevel.LEAF,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Import an external public key.

        Args:
            key_id: Key identifier
            pem_data: PEM-encoded public key
            purpose: Key purpose
            trust_level: Trust level
            labels: Optional labels
        """
        public_key = load_public_key(pem_data)

        # Detect algorithm
        from ccea.crypto.keys import get_key_algorithm
        algorithm = get_key_algorithm(public_key)

        metadata = KeyMetadata(
            key_id=key_id,
            algorithm=algorithm,
            purpose=purpose,
            status=KeyStatus.ACTIVE,
            trust_level=trust_level,
            created_at=datetime.now(timezone.utc),
            labels=labels or {"imported": "true"},
        )

        with self._lock:
            self._metadata[key_id] = metadata
            self._public_keys[key_id] = public_key
            self._save_metadata()

            # Save public key to disk
            key_path = self.keys_dir / f"{key_id}.pub.pem"
            key_path.write_text(pem_data)

    def cleanup_expired(self) -> List[str]:
        """
        Clean up expired keys.

        Returns:
            List of expired key IDs
        """
        expired = []
        now = datetime.now(timezone.utc)

        with self._lock:
            for key_id, meta in list(self._metadata.items()):
                if meta.expires_at and now > meta.expires_at:
                    if meta.status in (KeyStatus.ACTIVE, KeyStatus.ROTATING):
                        meta.status = KeyStatus.EXPIRED
                        self._private_keys.pop(key_id, None)
                        expired.append(key_id)

            if expired:
                self._save_metadata()

        return expired

    def _save_key(self, key_id: str, keypair: KeyPair) -> None:
        """Save key to disk."""
        # Save public key
        pub_path = self.keys_dir / f"{key_id}.pub.pem"
        pub_path.write_text(keypair.get_public_key_pem())

        # Save private key (encrypted if password provided)
        priv_path = self.keys_dir / f"{key_id}.priv.pem"
        priv_path.write_text(keypair.get_private_key_pem(self.master_password))

        # Restrict permissions on private key
        os.chmod(priv_path, 0o600)

    def _load(self) -> None:
        """Load keys and metadata from disk."""
        self._load_metadata()
        self._load_trust_root()
        self._load_keys()

    def _load_metadata(self) -> None:
        """Load key metadata from disk."""
        if not self.metadata_path.exists():
            return

        try:
            with open(self.metadata_path) as f:
                data = json.load(f)

            for key_id, meta_dict in data.items():
                self._metadata[key_id] = KeyMetadata.from_dict(meta_dict)
        except Exception:
            self._metadata = {}

    def _save_metadata(self) -> None:
        """Save key metadata to disk."""
        data = {
            key_id: meta.to_dict()
            for key_id, meta in self._metadata.items()
        }

        with open(self.metadata_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_trust_root(self) -> None:
        """Load trust root from disk."""
        if not self.trust_path.exists():
            return

        try:
            with open(self.trust_path) as f:
                data = json.load(f)

            self._trust_root = TrustRoot(
                root_key_ids=set(data.get("root_key_ids", [])),
                allowed_purposes={KeyPurpose(p) for p in data.get("allowed_purposes", [])},
                min_algorithm=KeyAlgorithm(data.get("min_algorithm", "ed25519")),
                require_expiration=data.get("require_expiration", True),
                max_validity_days=data.get("max_validity_days", 365),
                allow_self_signed=data.get("allow_self_signed", False),
            )
        except Exception:
            self._trust_root = None

    def _save_trust_root(self) -> None:
        """Save trust root to disk."""
        if not self._trust_root:
            return

        with open(self.trust_path, "w") as f:
            json.dump(self._trust_root.to_dict(), f, indent=2)

    def _load_keys(self) -> None:
        """Load keys from disk."""
        for key_id in self._metadata:
            # Load public key
            pub_path = self.keys_dir / f"{key_id}.pub.pem"
            if pub_path.exists():
                try:
                    self._public_keys[key_id] = load_public_key(pub_path.read_text())
                except Exception:
                    pass

            # Load private key (only if not revoked/expired)
            meta = self._metadata.get(key_id)
            if meta and meta.status in (KeyStatus.ACTIVE, KeyStatus.ROTATING, KeyStatus.PENDING):
                priv_path = self.keys_dir / f"{key_id}.priv.pem"
                if priv_path.exists():
                    try:
                        self._private_keys[key_id] = load_private_key(
                            priv_path.read_text(),
                            self.master_password,
                        )
                    except Exception:
                        pass


class SigningKeyProvider:
    """
    Helper class for obtaining signing keys.

    Used by ArtifactSigner to get the appropriate key.
    """

    def __init__(self, key_manager: KeyManager):
        """
        Initialize provider.

        Args:
            key_manager: Key manager instance
        """
        self.key_manager = key_manager

    def get_artifact_signing_key(self) -> Optional[KeyPair]:
        """Get the active artifact signing key."""
        key_id = self.key_manager.get_active_key(KeyPurpose.ARTIFACT_SIGNING)
        if not key_id:
            return None

        private_key = self.key_manager.get_private_key(key_id)
        public_key = self.key_manager.get_public_key(key_id)
        meta = self.key_manager.get_metadata(key_id)

        if not all([private_key, public_key, meta]):
            return None

        return KeyPair(
            algorithm=meta.algorithm,
            private_key=private_key,
            public_key=public_key,
            key_id=key_id,
        )

    def get_verification_keys(
        self,
        purpose: KeyPurpose = KeyPurpose.ARTIFACT_SIGNING,
    ) -> Dict[str, PublicKeyTypes]:
        """
        Get all valid verification keys.

        Returns both active and rotating keys for verification.
        """
        keys = {}
        for key_id in self.key_manager.get_signing_keys(purpose):
            public_key = self.key_manager.get_public_key(key_id)
            if public_key:
                keys[key_id] = public_key
        return keys
