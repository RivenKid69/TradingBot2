# -*- coding: utf-8 -*-
"""
Local Vault - Encrypted credential storage.

Stores broker credentials securely on the local machine.
Cloud NEVER has access to these credentials.

Security Architecture:
1. Master key derived from OS keychain or environment variable
2. Credentials encrypted with AES-256-GCM
3. Each credential has unique IV
4. Vault file is encrypted JSON
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Final, List, Optional

# Try to import cryptography, but provide fallback
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


# Constants
VAULT_VERSION: Final[str] = "1.0.0"
KEY_SIZE: Final[int] = 32  # 256 bits
NONCE_SIZE: Final[int] = 12  # 96 bits for GCM
SALT_SIZE: Final[int] = 16
PBKDF2_ITERATIONS: Final[int] = 100000


class VaultError(Exception):
    """Base exception for vault errors."""
    pass


class CredentialNotFoundError(VaultError):
    """Credential not found in vault."""
    pass


class VaultLockedError(VaultError):
    """Vault is locked and cannot be accessed."""
    pass


class VaultCorruptedError(VaultError):
    """Vault data is corrupted."""
    pass


@dataclass
class VaultConfig:
    """
    Vault configuration.
    """
    # Storage path
    vault_path: Path = field(default_factory=lambda: Path.home() / ".ccea" / "vault.enc")

    # Key source
    use_keychain: bool = True  # Prefer OS keychain
    env_var_name: str = "CCEA_VAULT_KEY"  # Fallback env var

    # Security settings
    pbkdf2_iterations: int = PBKDF2_ITERATIONS

    # Auto-lock
    auto_lock_seconds: int = 300  # Lock after 5 minutes of inactivity

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vault_path": str(self.vault_path),
            "use_keychain": self.use_keychain,
            "env_var_name": self.env_var_name,
            "pbkdf2_iterations": self.pbkdf2_iterations,
            "auto_lock_seconds": self.auto_lock_seconds,
        }


@dataclass
class CredentialEntry:
    """
    Single credential entry in vault.
    """
    credential_id: str
    broker: str
    credential_type: str  # api_key, api_secret, oauth_token, etc.
    encrypted_value: bytes = b""
    nonce: bytes = b""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "credential_id": self.credential_id,
            "broker": self.broker,
            "credential_type": self.credential_type,
            "encrypted_value": base64.b64encode(self.encrypted_value).decode(),
            "nonce": base64.b64encode(self.nonce).decode(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CredentialEntry:
        """Create from dictionary."""
        return cls(
            credential_id=data["credential_id"],
            broker=data["broker"],
            credential_type=data["credential_type"],
            encrypted_value=base64.b64decode(data["encrypted_value"]),
            nonce=base64.b64decode(data["nonce"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )


class LocalVault:
    """
    Local Vault for secure credential storage.

    Credentials are encrypted and stored locally.
    Cloud NEVER has access to these credentials.

    Usage:
        vault = LocalVault()
        vault.unlock(master_key)
        vault.store("binance", "api_key", "my_key_value")
        key = vault.retrieve("binance", "api_key")
    """

    def __init__(self, config: Optional[VaultConfig] = None):
        """Initialize vault."""
        if not CRYPTO_AVAILABLE:
            raise VaultError(
                "cryptography package required. Install with: pip install cryptography"
            )

        self.config = config or VaultConfig()
        self._master_key: Optional[bytes] = None
        self._entries: Dict[str, CredentialEntry] = {}
        self._salt: Optional[bytes] = None
        self._locked = True
        self._last_access: Optional[datetime] = None

    @property
    def is_locked(self) -> bool:
        """Check if vault is locked."""
        return self._locked

    @property
    def is_initialized(self) -> bool:
        """Check if vault file exists."""
        return self.config.vault_path.exists()

    def initialize(self, master_password: str) -> None:
        """
        Initialize new vault with master password.

        Args:
            master_password: Master password for vault
        """
        if self.is_initialized:
            raise VaultError("Vault already initialized")

        # Generate salt
        self._salt = secrets.token_bytes(SALT_SIZE)

        # Derive key from password
        self._master_key = self._derive_key(master_password, self._salt)

        # Create empty vault
        self._entries = {}
        self._locked = False
        self._last_access = datetime.utcnow()

        # Save vault
        self._save()

    def unlock(self, master_password: str) -> None:
        """
        Unlock vault with master password.

        Args:
            master_password: Master password for vault
        """
        if not self.is_initialized:
            raise VaultError("Vault not initialized. Call initialize() first.")

        # Load vault data
        with open(self.config.vault_path, "rb") as f:
            vault_data = f.read()

        # Parse header
        try:
            header_end = vault_data.index(b"\n")
            header = json.loads(vault_data[:header_end])
            encrypted_data = vault_data[header_end + 1:]
        except (ValueError, json.JSONDecodeError) as e:
            raise VaultCorruptedError(f"Invalid vault format: {e}")

        # Get salt
        self._salt = base64.b64decode(header["salt"])

        # Derive key
        self._master_key = self._derive_key(master_password, self._salt)

        # Decrypt vault
        try:
            nonce = base64.b64decode(header["nonce"])
            aesgcm = AESGCM(self._master_key)
            decrypted = aesgcm.decrypt(nonce, encrypted_data, None)
            entries_data = json.loads(decrypted)
        except Exception as e:
            self._master_key = None
            raise VaultError(f"Failed to unlock vault: {e}")

        # Load entries
        self._entries = {
            k: CredentialEntry.from_dict(v) for k, v in entries_data.items()
        }

        self._locked = False
        self._last_access = datetime.utcnow()

    def lock(self) -> None:
        """Lock vault and clear sensitive data from memory."""
        self._master_key = None
        self._entries.clear()
        self._locked = True

    def store(
        self,
        broker: str,
        credential_type: str,
        value: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store credential in vault.

        Args:
            broker: Broker identifier (e.g., "binance", "alpaca")
            credential_type: Type of credential (e.g., "api_key", "api_secret")
            value: Credential value (will be encrypted)
            metadata: Optional metadata

        Returns:
            Credential ID
        """
        self._check_unlocked()

        # Generate credential ID
        credential_id = f"{broker}:{credential_type}"

        # Encrypt value
        nonce = secrets.token_bytes(NONCE_SIZE)
        aesgcm = AESGCM(self._master_key)
        encrypted = aesgcm.encrypt(nonce, value.encode(), None)

        # Create or update entry
        now = datetime.utcnow()
        if credential_id in self._entries:
            entry = self._entries[credential_id]
            entry.encrypted_value = encrypted
            entry.nonce = nonce
            entry.updated_at = now
            if metadata:
                entry.metadata.update(metadata)
        else:
            entry = CredentialEntry(
                credential_id=credential_id,
                broker=broker,
                credential_type=credential_type,
                encrypted_value=encrypted,
                nonce=nonce,
                created_at=now,
                updated_at=now,
                metadata=metadata or {},
            )
            self._entries[credential_id] = entry

        # Save vault
        self._save()
        self._last_access = datetime.utcnow()

        return credential_id

    def retrieve(self, broker: str, credential_type: str) -> str:
        """
        Retrieve credential from vault.

        Args:
            broker: Broker identifier
            credential_type: Type of credential

        Returns:
            Decrypted credential value
        """
        self._check_unlocked()

        credential_id = f"{broker}:{credential_type}"

        if credential_id not in self._entries:
            raise CredentialNotFoundError(
                f"Credential not found: {credential_id}"
            )

        entry = self._entries[credential_id]

        # Decrypt
        aesgcm = AESGCM(self._master_key)
        decrypted = aesgcm.decrypt(entry.nonce, entry.encrypted_value, None)

        self._last_access = datetime.utcnow()

        return decrypted.decode()

    def delete(self, broker: str, credential_type: str) -> bool:
        """
        Delete credential from vault.

        Args:
            broker: Broker identifier
            credential_type: Type of credential

        Returns:
            True if deleted, False if not found
        """
        self._check_unlocked()

        credential_id = f"{broker}:{credential_type}"

        if credential_id in self._entries:
            del self._entries[credential_id]
            self._save()
            return True
        return False

    def list_credentials(self, broker: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List credentials (without values).

        Args:
            broker: Optional broker filter

        Returns:
            List of credential info (no values)
        """
        self._check_unlocked()

        results = []
        for entry in self._entries.values():
            if broker is None or entry.broker == broker:
                results.append({
                    "credential_id": entry.credential_id,
                    "broker": entry.broker,
                    "credential_type": entry.credential_type,
                    "created_at": entry.created_at.isoformat(),
                    "updated_at": entry.updated_at.isoformat(),
                    "metadata": entry.metadata,
                })
        return results

    def rotate_master_key(self, old_password: str, new_password: str) -> None:
        """
        Rotate master key.

        Args:
            old_password: Current master password
            new_password: New master password
        """
        # First unlock with old password to verify
        if self._locked:
            self.unlock(old_password)

        # Generate new salt and key
        new_salt = secrets.token_bytes(SALT_SIZE)
        new_key = self._derive_key(new_password, new_salt)

        # Re-encrypt all credentials with new key
        for entry in self._entries.values():
            # Decrypt with old key
            aesgcm_old = AESGCM(self._master_key)
            decrypted = aesgcm_old.decrypt(entry.nonce, entry.encrypted_value, None)

            # Encrypt with new key
            new_nonce = secrets.token_bytes(NONCE_SIZE)
            aesgcm_new = AESGCM(new_key)
            entry.encrypted_value = aesgcm_new.encrypt(new_nonce, decrypted, None)
            entry.nonce = new_nonce
            entry.updated_at = datetime.utcnow()

        # Update master key and salt
        self._master_key = new_key
        self._salt = new_salt

        # Save with new encryption
        self._save()

    def _check_unlocked(self) -> None:
        """Check vault is unlocked."""
        if self._locked:
            raise VaultLockedError("Vault is locked")
        if self._master_key is None:
            raise VaultLockedError("Master key not available")

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=KEY_SIZE,
            salt=salt,
            iterations=self.config.pbkdf2_iterations,
        )
        return kdf.derive(password.encode())

    def _save(self) -> None:
        """Save vault to disk."""
        if self._locked or self._master_key is None:
            raise VaultLockedError("Cannot save locked vault")

        # Ensure directory exists
        self.config.vault_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize entries
        entries_json = json.dumps({
            k: v.to_dict() for k, v in self._entries.items()
        })

        # Encrypt
        nonce = secrets.token_bytes(NONCE_SIZE)
        aesgcm = AESGCM(self._master_key)
        encrypted = aesgcm.encrypt(nonce, entries_json.encode(), None)

        # Create header
        header = {
            "version": VAULT_VERSION,
            "salt": base64.b64encode(self._salt).decode(),
            "nonce": base64.b64encode(nonce).decode(),
            "created_at": datetime.utcnow().isoformat(),
        }

        # Write file
        with open(self.config.vault_path, "wb") as f:
            f.write(json.dumps(header).encode())
            f.write(b"\n")
            f.write(encrypted)

        # Set restrictive permissions
        os.chmod(self.config.vault_path, 0o600)
