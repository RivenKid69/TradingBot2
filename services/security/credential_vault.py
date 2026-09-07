"""
Broker API Key Vault with AES-256-GCM encryption.

Provides secure storage for broker API credentials with:
- AES-256-GCM authenticated encryption
- Per-user key derivation for isolation
- Access logging for audit trail
- Secure deletion per GDPR Art. 17

References:
    - NIST SP 800-57 Part 1 Rev 5: Key Management
    - OWASP Cryptographic Storage Cheat Sheet
    - PCI DSS Requirement 3.4: Render sensitive data unreadable
    - RFC 5116: AEAD (Authenticated Encryption with Associated Data)

Security Model:
    Master Key (32 bytes) -> PBKDF2 with user_id salt -> User Key -> AES-256-GCM

Example:
    >>> vault = CredentialVault(master_key=os.urandom(32))
    >>> encrypted = vault.encrypt(
    ...     user_id="user_001",
    ...     credential_type=CredentialType.BROKER_API_KEY,
    ...     broker="alpaca",
    ...     plaintext="sk-live-abc123xyz"
    ... )
    >>> decrypted = vault.decrypt(encrypted, purpose="order_execution")
"""

from __future__ import annotations

import hashlib
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes


class CredentialType(Enum):
    """Types of broker credentials that can be stored."""

    BROKER_API_KEY = "broker_api_key"
    BROKER_API_SECRET = "broker_api_secret"
    BROKER_PASSPHRASE = "broker_passphrase"
    BROKER_ACCESS_TOKEN = "broker_access_token"
    BROKER_REFRESH_TOKEN = "broker_refresh_token"


class CredentialVaultError(Exception):
    """Base exception for credential vault errors."""

    pass


class InvalidMasterKeyError(CredentialVaultError):
    """Raised when master key is invalid."""

    pass


class CredentialDecryptionError(CredentialVaultError):
    """Raised when decryption fails (tampering, wrong key, etc.)."""

    pass


@dataclass
class EncryptedCredential:
    """
    Represents an encrypted broker credential.

    Attributes:
        credential_id: Unique identifier for the credential
        user_id: Owner of the credential
        credential_type: Type of credential (API key, secret, etc.)
        broker: Broker identifier (alpaca, binance, etc.)
        ciphertext: Encrypted credential data
        nonce: Unique nonce used for encryption
        created_at: Timestamp of creation
        last_accessed: Timestamp of last access
        access_count: Number of times credential was accessed
        metadata: Optional additional metadata
    """

    credential_id: str
    user_id: str
    credential_type: CredentialType
    broker: str
    ciphertext: bytes
    nonce: bytes
    created_at: datetime
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "credential_id": self.credential_id,
            "user_id": self.user_id,
            "credential_type": self.credential_type.value,
            "broker": self.broker,
            "ciphertext": self.ciphertext.hex(),
            "nonce": self.nonce.hex(),
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "access_count": self.access_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EncryptedCredential":
        """Deserialize from dictionary."""
        return cls(
            credential_id=data["credential_id"],
            user_id=data["user_id"],
            credential_type=CredentialType(data["credential_type"]),
            broker=data["broker"],
            ciphertext=bytes.fromhex(data["ciphertext"]),
            nonce=bytes.fromhex(data["nonce"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=(
                datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None
            ),
            access_count=data.get("access_count", 0),
            metadata=data.get("metadata", {}),
        )


class CredentialVault:
    """
    Secure storage for broker API credentials.

    Security features:
        - AES-256-GCM encryption (authenticated encryption)
        - Unique nonce per encryption (96-bit)
        - Key derivation from master key + user_id (user isolation)
        - Associated data binding (prevents credential swapping)
        - Access logging for audit trail
        - Secure deletion with overwrite

    Args:
        master_key: 32-byte master encryption key.
                   Should be sourced from environment variable or KMS.
        kdf_iterations: Number of PBKDF2 iterations (default: 100,000)

    Raises:
        InvalidMasterKeyError: If master key is not exactly 32 bytes

    Example:
        >>> import os
        >>> master_key = os.environ.get("CREDENTIAL_MASTER_KEY")
        >>> if master_key:
        ...     vault = CredentialVault(bytes.fromhex(master_key))
    """

    # NIST recommends minimum 10,000 iterations, we use 100,000 for extra security
    DEFAULT_KDF_ITERATIONS = 100_000

    # GCM nonce size (96 bits / 12 bytes is standard for AES-GCM)
    NONCE_SIZE = 12

    # Master key size (256 bits / 32 bytes for AES-256)
    MASTER_KEY_SIZE = 32

    def __init__(self, master_key: bytes, kdf_iterations: int = DEFAULT_KDF_ITERATIONS):
        """Initialize the credential vault with a master key."""
        if not isinstance(master_key, bytes):
            raise InvalidMasterKeyError("Master key must be bytes")

        if len(master_key) != self.MASTER_KEY_SIZE:
            raise InvalidMasterKeyError(
                f"Master key must be {self.MASTER_KEY_SIZE} bytes (256 bits), "
                f"got {len(master_key)} bytes"
            )

        self._master_key = master_key
        self._kdf_iterations = kdf_iterations
        self._access_log: List[Dict[str, Any]] = []

    def _derive_user_key(self, user_id: str) -> bytes:
        """
        Derive a user-specific encryption key from the master key.

        Uses PBKDF2-HMAC-SHA256 with user_id as salt for user isolation.
        This ensures that even if one user's derived key is compromised,
        other users' credentials remain protected.

        Args:
            user_id: Unique user identifier

        Returns:
            32-byte derived key for the user
        """
        # Use user_id as salt - provides user isolation
        salt = user_id.encode("utf-8")

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.MASTER_KEY_SIZE,
            salt=salt,
            iterations=self._kdf_iterations,
            backend=default_backend(),
        )

        return kdf.derive(self._master_key)

    def _generate_credential_id(
        self, user_id: str, broker: str, credential_type: CredentialType
    ) -> str:
        """
        Generate a deterministic credential ID.

        This allows looking up credentials without storing the plaintext
        combination of user_id, broker, and credential_type.
        """
        data = f"{user_id}:{broker}:{credential_type.value}".encode("utf-8")
        return hashlib.sha256(data).hexdigest()[:16]

    def _build_associated_data(
        self, user_id: str, broker: str, credential_type: CredentialType
    ) -> bytes:
        """
        Build associated data for authenticated encryption.

        This data is authenticated but not encrypted, binding the
        ciphertext to the context and preventing credential swapping attacks.
        """
        return f"{user_id}:{broker}:{credential_type.value}".encode("utf-8")

    def encrypt(
        self,
        user_id: str,
        credential_type: CredentialType,
        broker: str,
        plaintext: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EncryptedCredential:
        """
        Encrypt a broker credential.

        Uses AES-256-GCM for authenticated encryption:
        - Confidentiality: Ciphertext reveals nothing about plaintext
        - Integrity: Any tampering is detected on decryption
        - Authenticity: Verifies the ciphertext was created with the correct key

        Args:
            user_id: Owner of the credential
            credential_type: Type of credential
            broker: Broker identifier
            plaintext: The credential value to encrypt
            metadata: Optional metadata to store with the credential

        Returns:
            EncryptedCredential object ready for storage

        Example:
            >>> encrypted = vault.encrypt(
            ...     user_id="user_001",
            ...     credential_type=CredentialType.BROKER_API_KEY,
            ...     broker="alpaca",
            ...     plaintext="sk-live-abc123xyz"
            ... )
        """
        # Derive user-specific key
        user_key = self._derive_user_key(user_id)

        # Create AES-GCM cipher
        aesgcm = AESGCM(user_key)

        # Generate random nonce (CRITICAL: must be unique per encryption)
        nonce = secrets.token_bytes(self.NONCE_SIZE)

        # Build associated data for authentication
        associated_data = self._build_associated_data(user_id, broker, credential_type)

        # Encrypt with authentication
        ciphertext = aesgcm.encrypt(
            nonce=nonce, data=plaintext.encode("utf-8"), associated_data=associated_data
        )

        # Generate credential ID
        credential_id = self._generate_credential_id(user_id, broker, credential_type)

        return EncryptedCredential(
            credential_id=credential_id,
            user_id=user_id,
            credential_type=credential_type,
            broker=broker,
            ciphertext=ciphertext,
            nonce=nonce,
            created_at=datetime.now(timezone.utc),
            metadata=metadata or {},
        )

    def decrypt(
        self, credential: EncryptedCredential, purpose: str, source_ip: Optional[str] = None
    ) -> str:
        """
        Decrypt a broker credential.

        Verifies authenticity and integrity before returning plaintext.
        All access is logged for audit purposes.

        Args:
            credential: The encrypted credential to decrypt
            purpose: Reason for access (logged for audit)
            source_ip: Source IP address (optional, for audit)

        Returns:
            Decrypted plaintext credential

        Raises:
            CredentialDecryptionError: If decryption fails (tampering detected,
                                       wrong key, corrupted data)

        Example:
            >>> plaintext = vault.decrypt(encrypted, purpose="order_execution")
        """
        # Derive user-specific key
        user_key = self._derive_user_key(credential.user_id)

        # Create AES-GCM cipher
        aesgcm = AESGCM(user_key)

        # Build associated data (must match encryption)
        associated_data = self._build_associated_data(
            credential.user_id, credential.broker, credential.credential_type
        )

        try:
            # Decrypt and verify authentication tag
            plaintext_bytes = aesgcm.decrypt(
                nonce=credential.nonce, data=credential.ciphertext, associated_data=associated_data
            )
        except Exception as e:
            # Log failed attempt
            self._log_access(
                credential=credential,
                purpose=purpose,
                source_ip=source_ip,
                success=False,
                error=str(e),
            )
            raise CredentialDecryptionError(
                f"Failed to decrypt credential {credential.credential_id}: "
                "authentication failed (possible tampering or wrong key)"
            ) from e

        # Log successful access
        self._log_access(credential=credential, purpose=purpose, source_ip=source_ip, success=True)

        # Update credential metadata
        credential.last_accessed = datetime.now(timezone.utc)
        credential.access_count += 1

        return plaintext_bytes.decode("utf-8")

    def _log_access(
        self,
        credential: EncryptedCredential,
        purpose: str,
        source_ip: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Log credential access for audit trail."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "credential_id": credential.credential_id,
            "user_id": credential.user_id,
            "broker": credential.broker,
            "credential_type": credential.credential_type.value,
            "purpose": purpose,
            "source_ip": source_ip,
            "success": success,
        }

        if error:
            log_entry["error"] = error

        self._access_log.append(log_entry)

    def delete(self, credential: EncryptedCredential) -> bool:
        """
        Securely delete a credential (GDPR Art. 17 - Right to Erasure).

        Overwrites ciphertext and nonce with random data before
        releasing the object for garbage collection.

        Args:
            credential: The credential to delete

        Returns:
            True if deletion was successful

        Note:
            The caller is responsible for removing the credential
            from persistent storage after this operation.
        """
        # Log the deletion
        self._log_access(credential=credential, purpose="DELETION", success=True)

        # Overwrite sensitive data with random bytes
        credential.ciphertext = secrets.token_bytes(len(credential.ciphertext))
        credential.nonce = secrets.token_bytes(self.NONCE_SIZE)

        return True

    def rotate_credential(
        self,
        credential: EncryptedCredential,
        new_plaintext: str,
        purpose: str = "credential_rotation",
    ) -> EncryptedCredential:
        """
        Rotate a credential with a new value.

        This creates a new encrypted credential with the same
        identifiers but new ciphertext and nonce.

        Args:
            credential: The existing credential to rotate
            new_plaintext: The new credential value
            purpose: Reason for rotation (logged)

        Returns:
            New EncryptedCredential with updated ciphertext
        """
        # Log the rotation
        self._log_access(credential=credential, purpose=purpose, success=True)

        # Create new encrypted credential
        new_credential = self.encrypt(
            user_id=credential.user_id,
            credential_type=credential.credential_type,
            broker=credential.broker,
            plaintext=new_plaintext,
            metadata=credential.metadata,
        )

        # Preserve access count history
        new_credential.access_count = credential.access_count

        # Securely delete old credential data
        self.delete(credential)

        return new_credential

    def get_access_log(
        self,
        user_id: Optional[str] = None,
        credential_id: Optional[str] = None,
        since: Optional[datetime] = None,
        success_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get access log entries, optionally filtered.

        Args:
            user_id: Filter by user ID
            credential_id: Filter by credential ID
            since: Filter entries after this timestamp
            success_only: Only return successful accesses

        Returns:
            List of access log entries matching filters
        """
        result = self._access_log

        if user_id:
            result = [e for e in result if e.get("user_id") == user_id]

        if credential_id:
            result = [e for e in result if e.get("credential_id") == credential_id]

        if since:
            since_iso = since.isoformat()
            result = [e for e in result if e.get("timestamp", "") >= since_iso]

        if success_only:
            result = [e for e in result if e.get("success", False)]

        return result

    def clear_access_log(self) -> int:
        """
        Clear the in-memory access log.

        Note: This only clears the in-memory log. Persistent audit
        logs should be managed separately.

        Returns:
            Number of entries cleared
        """
        count = len(self._access_log)
        self._access_log = []
        return count

    @staticmethod
    def generate_master_key() -> bytes:
        """
        Generate a cryptographically secure master key.

        Use this method to create a new master key for the vault.
        Store the key securely (e.g., in environment variable or KMS).

        Returns:
            32-byte random master key

        Example:
            >>> key = CredentialVault.generate_master_key()
            >>> print(key.hex())  # Store this securely!
        """
        return secrets.token_bytes(CredentialVault.MASTER_KEY_SIZE)

    def verify_credential(self, credential: EncryptedCredential, expected_plaintext: str) -> bool:
        """
        Verify a credential matches an expected value.

        Useful for validating credentials without exposing the plaintext.
        Uses constant-time comparison to prevent timing attacks.

        Args:
            credential: The encrypted credential to verify
            expected_plaintext: The expected plaintext value

        Returns:
            True if the credential matches, False otherwise
        """
        try:
            decrypted = self.decrypt(credential, purpose="verification")
            return secrets.compare_digest(
                decrypted.encode("utf-8"), expected_plaintext.encode("utf-8")
            )
        except CredentialDecryptionError:
            return False
