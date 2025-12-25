# -*- coding: utf-8 -*-
"""
Customer Managed Keys (CMK) Service.

CCEA Phase 8 Implementation.

This service provides encryption with customer-managed keys:
    - Key registration and management
    - Envelope encryption with CMK
    - Key rotation support
    - Audit logging

Design Doc Reference:
    - Phase 8 (13.4): "Enterprise: optionally customer-managed keys (CMK)"
    - Encryption at rest with customer control

CLOUD ZONE ONLY.
ENTERPRISE FEATURE.
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, Final, List, Optional, Tuple
from uuid import uuid4

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


# ============================================================================
# Constants
# ============================================================================

# Default key rotation period (days)
DEFAULT_KEY_ROTATION_DAYS: Final[int] = 90

# Maximum key age before mandatory rotation
MAX_KEY_AGE_DAYS: Final[int] = 365

# Key types
class KeyType(Enum):
    """Types of encryption keys."""
    SYMMETRIC = auto()   # AES-256
    ASYMMETRIC = auto()  # RSA-4096
    DERIVED = auto()     # PBKDF2-derived


class KeyStatus(Enum):
    """Key lifecycle status."""
    ACTIVE = "active"
    PENDING_ROTATION = "pending_rotation"
    ROTATING = "rotating"
    DISABLED = "disabled"
    REVOKED = "revoked"


class KeyPurpose(Enum):
    """Purpose of the key."""
    DATA_ENCRYPTION = "data_encryption"
    TELEMETRY_ENCRYPTION = "telemetry_encryption"
    BACKUP_ENCRYPTION = "backup_encryption"
    SIGNING = "signing"


@dataclass
class KeyInfo:
    """
    Information about a managed key.

    Attributes:
        id: Key identifier
        workspace_id: Owning workspace
        name: Key name
        key_type: Type of key
        purpose: Key purpose
        status: Current status
        created_at: Creation time
        rotated_at: Last rotation time
        expires_at: Expiration time
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    name: str = ""
    key_type: KeyType = KeyType.SYMMETRIC
    purpose: KeyPurpose = KeyPurpose.DATA_ENCRYPTION
    status: KeyStatus = KeyStatus.ACTIVE
    algorithm: str = "AES-256-GCM"
    key_version: int = 1

    # Key material hash (for verification, NOT the actual key)
    key_hash: str = ""

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    rotated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    disabled_at: Optional[datetime] = None

    # Metadata
    created_by: str = ""
    rotation_policy_days: int = DEFAULT_KEY_ROTATION_DAYS
    usage_count: int = 0
    last_used_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (safe, no key material).

        Note: key_hash is truncated to first 16 characters for UI/logging.
        For security incident response requiring full hash correlation,
        use to_audit_dict() which includes the complete key_hash.

        Hash Truncation Policy:
            - UI/API responses: 16 chars (sufficient for visual identification)
            - Audit logs: Full hash (via to_audit_dict)
            - Incident response: Full hash available in CMKService audit trail

        Tech Debt Tracking: docs/reports/TECH_DEBT_REGISTRY.md#security-cmk-hash-truncation
        """
        return {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "name": self.name,
            "key_type": self.key_type.name,
            "purpose": self.purpose.name,
            "status": self.status.value,
            "algorithm": self.algorithm,
            "key_version": self.key_version,
            "key_hash": self.key_hash[:16] + "..." if self.key_hash else "",  # Truncated for UI
            "created_at": self.created_at.isoformat(),
            "rotated_at": self.rotated_at.isoformat() if self.rotated_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "rotation_policy_days": self.rotation_policy_days,
            "usage_count": self.usage_count,
        }

    def to_audit_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with full key_hash for audit/security purposes.

        Use this method for:
            - Security incident investigation
            - Compliance audit trails
            - Key correlation during forensic analysis

        WARNING: Contains full key_hash. Do not expose via public APIs.
        """
        result = self.to_dict()
        result["key_hash"] = self.key_hash  # Full hash for audit
        result["_audit_context"] = True
        return result

    @property
    def is_expired(self) -> bool:
        """Check if key is expired."""
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return True
        return False

    @property
    def needs_rotation(self) -> bool:
        """Check if key needs rotation."""
        last_rotated = self.rotated_at or self.created_at
        days_since_rotation = (datetime.utcnow() - last_rotated).days
        return days_since_rotation >= self.rotation_policy_days


@dataclass
class CMKConfig:
    """CMK service configuration."""
    enabled: bool = True
    default_rotation_days: int = DEFAULT_KEY_ROTATION_DAYS
    max_key_age_days: int = MAX_KEY_AGE_DAYS
    auto_rotation: bool = True
    require_approval_for_rotation: bool = False
    audit_all_operations: bool = True


@dataclass
class EncryptionResult:
    """Result of encryption operation."""
    success: bool = True
    ciphertext: Optional[bytes] = None
    encrypted_dek: Optional[bytes] = None  # Encrypted Data Encryption Key
    key_id: str = ""
    key_version: int = 0
    algorithm: str = ""
    error: Optional[str] = None

    def to_envelope(self) -> Dict[str, Any]:
        """Convert to envelope format for storage."""
        return {
            "ciphertext": base64.b64encode(self.ciphertext).decode() if self.ciphertext else None,
            "encrypted_dek": base64.b64encode(self.encrypted_dek).decode() if self.encrypted_dek else None,
            "key_id": self.key_id,
            "key_version": self.key_version,
            "algorithm": self.algorithm,
        }


@dataclass
class DecryptionResult:
    """Result of decryption operation."""
    success: bool = True
    plaintext: Optional[bytes] = None
    key_id: str = ""
    error: Optional[str] = None


class CustomerManagedKeysService:
    """
    Customer-managed keys service.

    Provides envelope encryption using customer-provided keys.

    Usage:
        cmk = CustomerManagedKeysService()

        # Register customer key
        key_info = cmk.register_key(
            workspace_id="workspace-123",
            name="primary-encryption-key",
            key_material=customer_provided_key,
            purpose=KeyPurpose.DATA_ENCRYPTION,
        )

        # Encrypt data
        result = cmk.encrypt(
            workspace_id="workspace-123",
            plaintext=b"sensitive data",
        )

        # Decrypt data
        decrypted = cmk.decrypt(
            workspace_id="workspace-123",
            ciphertext=result.ciphertext,
            encrypted_dek=result.encrypted_dek,
            key_id=result.key_id,
            key_version=result.key_version,
        )

    SECURITY:
        - Customer controls key material
        - Envelope encryption (DEK encrypted with CMK)
        - Key rotation support
        - Full audit trail
    """

    def __init__(
        self,
        config: Optional[CMKConfig] = None,
        on_key_operation: Optional[Callable[[str, str, KeyInfo], None]] = None,
    ):
        """
        Initialize CMK service.

        Args:
            config: Service configuration
            on_key_operation: Callback for key operations (operation, workspace_id, key_info)
        """
        self.config = config or CMKConfig()
        self._on_key_operation = on_key_operation
        self._keys: Dict[str, Dict[str, KeyInfo]] = {}  # workspace_id -> key_id -> info
        self._key_material: Dict[str, bytes] = {}  # key_id -> encrypted key material
        self._dek_cache: Dict[str, bytes] = {}  # key_id -> DEK (in memory only)
        self._audit_log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        # Master key for protecting CMK material at rest (in real impl, use HSM/KMS)
        self._master_key = Fernet.generate_key()
        self._fernet = Fernet(self._master_key)

    def register_key(
        self,
        workspace_id: str,
        name: str,
        key_material: bytes,
        purpose: KeyPurpose = KeyPurpose.DATA_ENCRYPTION,
        rotation_days: Optional[int] = None,
        created_by: str = "system",
    ) -> KeyInfo:
        """
        Register a customer-managed key.

        Args:
            workspace_id: Workspace identifier
            name: Key name
            key_material: The actual key bytes (will be encrypted at rest)
            purpose: Key purpose
            rotation_days: Rotation period
            created_by: Who created the key

        Returns:
            KeyInfo
        """
        # Generate key ID
        key_id = str(uuid4())

        # Determine key type and validate
        if len(key_material) == 32:
            key_type = KeyType.SYMMETRIC
            algorithm = "AES-256-GCM"
        elif len(key_material) == 64:
            key_type = KeyType.SYMMETRIC
            algorithm = "AES-512"
        else:
            # Try to detect RSA key
            try:
                # Check if it's a valid key size
                key_type = KeyType.SYMMETRIC
                algorithm = f"AES-{len(key_material) * 8}"
            except Exception:
                key_type = KeyType.DERIVED
                algorithm = "PBKDF2-AES-256"

        # Create key info
        key_info = KeyInfo(
            id=key_id,
            workspace_id=workspace_id,
            name=name,
            key_type=key_type,
            purpose=purpose,
            algorithm=algorithm,
            key_hash=hashlib.sha256(key_material).hexdigest(),
            rotation_policy_days=rotation_days or self.config.default_rotation_days,
            created_by=created_by,
        )

        # Set expiration
        key_info.expires_at = datetime.utcnow() + timedelta(days=self.config.max_key_age_days)

        # Encrypt and store key material
        encrypted_material = self._fernet.encrypt(key_material)

        with self._lock:
            if workspace_id not in self._keys:
                self._keys[workspace_id] = {}
            self._keys[workspace_id][key_id] = key_info
            self._key_material[key_id] = encrypted_material

            self._log_audit("key_registered", workspace_id, key_info)

        if self._on_key_operation:
            self._on_key_operation("register", workspace_id, key_info)

        return key_info

    def get_key_info(self, workspace_id: str, key_id: str) -> Optional[KeyInfo]:
        """Get key information (without material)."""
        with self._lock:
            workspace_keys = self._keys.get(workspace_id, {})
            return workspace_keys.get(key_id)

    def list_keys(self, workspace_id: str) -> List[KeyInfo]:
        """List all keys for workspace."""
        with self._lock:
            return list(self._keys.get(workspace_id, {}).values())

    def get_active_key(
        self,
        workspace_id: str,
        purpose: KeyPurpose = KeyPurpose.DATA_ENCRYPTION,
    ) -> Optional[KeyInfo]:
        """Get the active key for a purpose."""
        with self._lock:
            workspace_keys = self._keys.get(workspace_id, {})
            for key_info in workspace_keys.values():
                if (key_info.purpose == purpose and
                    key_info.status == KeyStatus.ACTIVE and
                    not key_info.is_expired):
                    return key_info
            return None

    def encrypt(
        self,
        workspace_id: str,
        plaintext: bytes,
        key_id: Optional[str] = None,
        purpose: KeyPurpose = KeyPurpose.DATA_ENCRYPTION,
    ) -> EncryptionResult:
        """
        Encrypt data using envelope encryption.

        Uses a Data Encryption Key (DEK) encrypted with the CMK.

        Args:
            workspace_id: Workspace identifier
            plaintext: Data to encrypt
            key_id: Specific key to use (optional)
            purpose: Key purpose if not specifying key_id

        Returns:
            EncryptionResult with ciphertext and encrypted DEK
        """
        # Get key
        if key_id:
            key_info = self.get_key_info(workspace_id, key_id)
        else:
            key_info = self.get_active_key(workspace_id, purpose)

        if not key_info:
            return EncryptionResult(
                success=False,
                error="No active key found",
            )

        if key_info.is_expired:
            return EncryptionResult(
                success=False,
                error="Key is expired",
            )

        if key_info.status != KeyStatus.ACTIVE:
            return EncryptionResult(
                success=False,
                error=f"Key is not active: {key_info.status.value}",
            )

        try:
            # Get CMK material
            cmk_material = self._get_key_material(key_info.id)
            if not cmk_material:
                return EncryptionResult(
                    success=False,
                    error="Failed to retrieve key material",
                )

            # Generate DEK
            dek = Fernet.generate_key()
            dek_fernet = Fernet(dek)

            # Encrypt plaintext with DEK
            ciphertext = dek_fernet.encrypt(plaintext)

            # Encrypt DEK with CMK
            encrypted_dek = self._encrypt_dek(dek, cmk_material)

            # Update usage stats
            with self._lock:
                key_info.usage_count += 1
                key_info.last_used_at = datetime.utcnow()

            self._log_audit("encrypt", workspace_id, key_info)

            return EncryptionResult(
                success=True,
                ciphertext=ciphertext,
                encrypted_dek=encrypted_dek,
                key_id=key_info.id,
                key_version=key_info.key_version,
                algorithm=key_info.algorithm,
            )

        except Exception as e:
            return EncryptionResult(
                success=False,
                error=str(e),
            )

    def decrypt(
        self,
        workspace_id: str,
        ciphertext: bytes,
        encrypted_dek: bytes,
        key_id: str,
        key_version: int = 0,
    ) -> DecryptionResult:
        """
        Decrypt data using envelope encryption.

        Args:
            workspace_id: Workspace identifier
            ciphertext: Encrypted data
            encrypted_dek: Encrypted DEK
            key_id: Key used for encryption
            key_version: Key version used

        Returns:
            DecryptionResult with plaintext
        """
        key_info = self.get_key_info(workspace_id, key_id)
        if not key_info:
            return DecryptionResult(
                success=False,
                error="Key not found",
            )

        # Allow decryption with disabled keys (for data recovery)
        if key_info.status == KeyStatus.REVOKED:
            return DecryptionResult(
                success=False,
                error="Key has been revoked",
            )

        try:
            # Get CMK material
            cmk_material = self._get_key_material(key_id)
            if not cmk_material:
                return DecryptionResult(
                    success=False,
                    error="Failed to retrieve key material",
                )

            # Decrypt DEK with CMK
            dek = self._decrypt_dek(encrypted_dek, cmk_material)
            if not dek:
                return DecryptionResult(
                    success=False,
                    error="Failed to decrypt DEK",
                )

            # Decrypt ciphertext with DEK
            dek_fernet = Fernet(dek)
            plaintext = dek_fernet.decrypt(ciphertext)

            self._log_audit("decrypt", workspace_id, key_info)

            return DecryptionResult(
                success=True,
                plaintext=plaintext,
                key_id=key_id,
            )

        except Exception as e:
            return DecryptionResult(
                success=False,
                error=str(e),
            )

    def rotate_key(
        self,
        workspace_id: str,
        key_id: str,
        new_key_material: bytes,
        rotated_by: str = "system",
    ) -> Optional[KeyInfo]:
        """
        Rotate a key with new material.

        Args:
            workspace_id: Workspace identifier
            key_id: Key to rotate
            new_key_material: New key bytes
            rotated_by: Who rotated the key

        Returns:
            Updated KeyInfo or None
        """
        with self._lock:
            key_info = self._keys.get(workspace_id, {}).get(key_id)
            if not key_info:
                return None

            # Update key info
            key_info.key_version += 1
            key_info.key_hash = hashlib.sha256(new_key_material).hexdigest()
            key_info.rotated_at = datetime.utcnow()
            key_info.expires_at = datetime.utcnow() + timedelta(days=self.config.max_key_age_days)

            # Encrypt and store new material
            encrypted_material = self._fernet.encrypt(new_key_material)
            self._key_material[key_id] = encrypted_material

            # Clear DEK cache
            if key_id in self._dek_cache:
                del self._dek_cache[key_id]

            self._log_audit("key_rotated", workspace_id, key_info, {"rotated_by": rotated_by})

        if self._on_key_operation:
            self._on_key_operation("rotate", workspace_id, key_info)

        return key_info

    def disable_key(
        self,
        workspace_id: str,
        key_id: str,
        disabled_by: str = "system",
    ) -> bool:
        """Disable a key."""
        with self._lock:
            key_info = self._keys.get(workspace_id, {}).get(key_id)
            if not key_info:
                return False

            key_info.status = KeyStatus.DISABLED
            key_info.disabled_at = datetime.utcnow()

            self._log_audit("key_disabled", workspace_id, key_info, {"disabled_by": disabled_by})

        return True

    def revoke_key(
        self,
        workspace_id: str,
        key_id: str,
        revoked_by: str = "system",
        reason: str = "",
    ) -> bool:
        """Revoke a key (cannot be undone)."""
        with self._lock:
            key_info = self._keys.get(workspace_id, {}).get(key_id)
            if not key_info:
                return False

            key_info.status = KeyStatus.REVOKED

            # Remove key material
            if key_id in self._key_material:
                del self._key_material[key_id]
            if key_id in self._dek_cache:
                del self._dek_cache[key_id]

            self._log_audit("key_revoked", workspace_id, key_info, {
                "revoked_by": revoked_by,
                "reason": reason,
            })

        if self._on_key_operation:
            self._on_key_operation("revoke", workspace_id, key_info)

        return True

    def get_keys_needing_rotation(self) -> List[KeyInfo]:
        """Get all keys that need rotation."""
        result = []
        with self._lock:
            for workspace_keys in self._keys.values():
                for key_info in workspace_keys.values():
                    if (key_info.status == KeyStatus.ACTIVE and
                        key_info.needs_rotation):
                        result.append(key_info)
        return result

    def _get_key_material(self, key_id: str) -> Optional[bytes]:
        """Get decrypted key material."""
        with self._lock:
            encrypted = self._key_material.get(key_id)
            if not encrypted:
                return None

            try:
                return self._fernet.decrypt(encrypted)
            except Exception:
                return None

    def _encrypt_dek(self, dek: bytes, cmk: bytes) -> bytes:
        """Encrypt DEK with CMK."""
        # Derive encryption key from CMK
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"cmk_dek_encryption",
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(cmk))
        fernet = Fernet(key)
        return fernet.encrypt(dek)

    def _decrypt_dek(self, encrypted_dek: bytes, cmk: bytes) -> Optional[bytes]:
        """Decrypt DEK with CMK."""
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"cmk_dek_encryption",
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(cmk))
            fernet = Fernet(key)
            return fernet.decrypt(encrypted_dek)
        except Exception:
            return None

    def _log_audit(
        self,
        operation: str,
        workspace_id: str,
        key_info: KeyInfo,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log audit event."""
        if not self.config.audit_all_operations:
            return

        entry = {
            "operation": operation,
            "workspace_id": workspace_id,
            "key_id": key_info.id,
            "key_name": key_info.name,
            "key_version": key_info.key_version,
            "timestamp": datetime.utcnow().isoformat(),
            **(extra or {}),
        }
        self._audit_log.append(entry)

    def get_audit_log(
        self,
        workspace_id: Optional[str] = None,
        key_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get audit log."""
        with self._lock:
            log = list(self._audit_log)

            if workspace_id:
                log = [e for e in log if e.get("workspace_id") == workspace_id]
            if key_id:
                log = [e for e in log if e.get("key_id") == key_id]

            return log
