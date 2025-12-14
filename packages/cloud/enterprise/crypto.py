# -*- coding: utf-8 -*-
"""
Enterprise Cryptographic Operations - Ed25519 signing and verification.

CLOUD ZONE ONLY.

Provides cryptographic signing for:
- Evidence packs
- Agent update packages
- Artifact signatures
- Provenance attestations

Standards:
- Ed25519 (RFC 8032) for signatures
- SHA-256 for hashing
- Base64 encoding for serialization

Design Doc Reference:
    - Phase 10: Enterprise signing
    - Design Doc 16.1: Evidence pack signing
    - Design Doc 15.2: Agent update signing
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Constants
SIGNATURE_ALGORITHM: Final[str] = "ed25519"
SIGNATURE_VERSION: Final[str] = "1.0"
KEY_SIZE: Final[int] = 32  # Ed25519 key size

# Check if cryptography library is available
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.exceptions import InvalidSignature
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    Ed25519PrivateKey = None
    Ed25519PublicKey = None
    InvalidSignature = Exception


class CryptoError(Exception):
    """Cryptographic operation error."""
    pass


class SignatureError(CryptoError):
    """Signature error."""
    pass


class VerificationError(CryptoError):
    """Signature verification error."""
    pass


class KeyError(CryptoError):
    """Key management error."""
    pass


@dataclass
class SigningKey:
    """
    Ed25519 signing key pair.

    Stores both private and public keys for signing operations.
    """
    key_id: str = ""
    private_key_bytes: bytes = field(default=b"", repr=False)
    public_key_bytes: bytes = b""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    algorithm: str = SIGNATURE_ALGORITHM

    _private_key: Optional[Any] = field(default=None, repr=False)
    _public_key: Optional[Any] = field(default=None, repr=False)

    @property
    def private_key(self) -> Any:
        """Get the private key object."""
        if self._private_key is None and self.private_key_bytes:
            self._private_key = Ed25519PrivateKey.from_private_bytes(self.private_key_bytes)
        return self._private_key

    @property
    def public_key(self) -> Any:
        """Get the public key object."""
        if self._public_key is None:
            if self.public_key_bytes:
                self._public_key = Ed25519PublicKey.from_public_bytes(self.public_key_bytes)
            elif self.private_key:
                self._public_key = self.private_key.public_key()
        return self._public_key

    @property
    def public_key_base64(self) -> str:
        """Get public key as base64 string."""
        if self.public_key:
            return base64.b64encode(
                self.public_key.public_bytes_raw()
            ).decode('ascii')
        return ""

    @property
    def is_valid(self) -> bool:
        """Check if key is valid and not expired."""
        if not self.private_key_bytes and not self.public_key_bytes:
            return False
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without private key)."""
        return {
            "key_id": self.key_id,
            "public_key": self.public_key_base64,
            "algorithm": self.algorithm,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class Signature:
    """
    Digital signature with metadata.
    """
    signature_bytes: bytes = b""
    algorithm: str = SIGNATURE_ALGORITHM
    key_id: str = ""
    signed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    signer_id: str = ""
    version: str = SIGNATURE_VERSION

    # Optional attestation data
    payload_digest: str = ""  # SHA-256 of signed payload
    payload_type: str = ""  # Type of signed data

    @property
    def signature_base64(self) -> str:
        """Get signature as base64 string."""
        return base64.b64encode(self.signature_bytes).decode('ascii')

    @classmethod
    def from_base64(cls, sig_b64: str, **kwargs) -> 'Signature':
        """Create signature from base64 string."""
        return cls(
            signature_bytes=base64.b64decode(sig_b64),
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "signature": self.signature_base64,
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "signed_at": self.signed_at.isoformat(),
            "signer_id": self.signer_id,
            "version": self.version,
            "payload_digest": self.payload_digest,
            "payload_type": self.payload_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signature':
        """Create signature from dictionary."""
        return cls(
            signature_bytes=base64.b64decode(data["signature"]),
            algorithm=data.get("algorithm", SIGNATURE_ALGORITHM),
            key_id=data.get("key_id", ""),
            signed_at=datetime.fromisoformat(data["signed_at"]) if data.get("signed_at") else datetime.now(timezone.utc),
            signer_id=data.get("signer_id", ""),
            version=data.get("version", SIGNATURE_VERSION),
            payload_digest=data.get("payload_digest", ""),
            payload_type=data.get("payload_type", ""),
        )


class Ed25519Signer:
    """
    Ed25519 digital signature provider.

    Provides signing and verification using Ed25519 (RFC 8032).

    Usage:
        # Generate new key
        signer = Ed25519Signer()
        key = signer.generate_key("my-key-id")

        # Sign data
        signature = signer.sign(b"data to sign", key)

        # Verify signature
        is_valid = signer.verify(b"data to sign", signature, key)
    """

    def __init__(
        self,
        trusted_keys: Optional[List[SigningKey]] = None,
        default_signer_id: str = "ccea-signer",
    ):
        """
        Initialize signer.

        Args:
            trusted_keys: List of trusted public keys for verification
            default_signer_id: Default signer identifier
        """
        if not CRYPTO_AVAILABLE:
            raise CryptoError(
                "cryptography library not available. "
                "Install with: pip install cryptography"
            )

        self._trusted_keys: Dict[str, SigningKey] = {}
        self.default_signer_id = default_signer_id

        if trusted_keys:
            for key in trusted_keys:
                self._trusted_keys[key.key_id] = key

    def generate_key(
        self,
        key_id: str = "",
        expires_in_days: Optional[int] = None,
    ) -> SigningKey:
        """
        Generate a new Ed25519 key pair.

        Args:
            key_id: Key identifier
            expires_in_days: Key expiration in days

        Returns:
            SigningKey with private and public keys
        """
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        now = datetime.now(timezone.utc)
        expires_at = None
        if expires_in_days:
            from datetime import timedelta
            expires_at = now + timedelta(days=expires_in_days)

        # Generate key_id from public key if not provided
        if not key_id:
            pub_bytes = public_key.public_bytes_raw()
            key_id = hashlib.sha256(pub_bytes).hexdigest()[:16]

        return SigningKey(
            key_id=key_id,
            private_key_bytes=private_key.private_bytes_raw(),
            public_key_bytes=public_key.public_bytes_raw(),
            created_at=now,
            expires_at=expires_at,
            _private_key=private_key,
            _public_key=public_key,
        )

    def sign(
        self,
        data: bytes,
        key: SigningKey,
        payload_type: str = "",
        signer_id: Optional[str] = None,
    ) -> Signature:
        """
        Sign data with Ed25519 key.

        Args:
            data: Data to sign
            key: Signing key
            payload_type: Type of payload being signed
            signer_id: Signer identifier

        Returns:
            Signature object
        """
        if not key.private_key:
            raise SignatureError("Private key not available for signing")

        if not key.is_valid:
            raise SignatureError("Key is invalid or expired")

        # Compute digest
        digest = hashlib.sha256(data).hexdigest()

        # Sign
        signature_bytes = key.private_key.sign(data)

        return Signature(
            signature_bytes=signature_bytes,
            algorithm=SIGNATURE_ALGORITHM,
            key_id=key.key_id,
            signed_at=datetime.now(timezone.utc),
            signer_id=signer_id or self.default_signer_id,
            payload_digest=f"sha256:{digest}",
            payload_type=payload_type,
        )

    def verify(
        self,
        data: bytes,
        signature: Signature,
        key: Optional[SigningKey] = None,
    ) -> bool:
        """
        Verify Ed25519 signature.

        Args:
            data: Original data
            signature: Signature to verify
            key: Public key (uses trusted keys if not provided)

        Returns:
            True if signature is valid
        """
        # Get public key
        public_key = None

        if key:
            public_key = key.public_key
        elif signature.key_id and signature.key_id in self._trusted_keys:
            public_key = self._trusted_keys[signature.key_id].public_key

        if not public_key:
            raise VerificationError(
                f"No public key available for key_id: {signature.key_id}"
            )

        try:
            public_key.verify(signature.signature_bytes, data)

            # Optionally verify digest
            if signature.payload_digest:
                computed_digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
                if signature.payload_digest != computed_digest:
                    logger.warning(
                        f"Payload digest mismatch: {signature.payload_digest} != {computed_digest}"
                    )
                    return False

            return True

        except InvalidSignature:
            return False
        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return False

    def add_trusted_key(self, key: SigningKey) -> None:
        """Add a trusted public key for verification."""
        self._trusted_keys[key.key_id] = key

    def remove_trusted_key(self, key_id: str) -> bool:
        """Remove a trusted key."""
        if key_id in self._trusted_keys:
            del self._trusted_keys[key_id]
            return True
        return False

    def get_trusted_keys(self) -> List[SigningKey]:
        """Get list of trusted keys."""
        return list(self._trusted_keys.values())

    def export_public_key(
        self,
        key: SigningKey,
        format: str = "raw",
    ) -> bytes:
        """
        Export public key in specified format.

        Args:
            key: Key to export
            format: 'raw', 'pem', or 'openssh'

        Returns:
            Public key bytes
        """
        public_key = key.public_key
        if not public_key:
            raise KeyError("No public key available")

        if format == "raw":
            return public_key.public_bytes_raw()
        elif format == "pem":
            return public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        elif format == "openssh":
            return public_key.public_bytes(
                encoding=serialization.Encoding.OpenSSH,
                format=serialization.PublicFormat.OpenSSH,
            )
        else:
            raise ValueError(f"Unknown format: {format}")

    def import_private_key(
        self,
        key_bytes: bytes,
        key_id: str = "",
        format: str = "raw",
    ) -> SigningKey:
        """
        Import private key from bytes.

        Args:
            key_bytes: Key bytes
            key_id: Key identifier
            format: 'raw' or 'pem'

        Returns:
            SigningKey
        """
        if format == "raw":
            if len(key_bytes) != KEY_SIZE:
                raise KeyError(f"Invalid key size: {len(key_bytes)}, expected {KEY_SIZE}")
            private_key = Ed25519PrivateKey.from_private_bytes(key_bytes)
        elif format == "pem":
            private_key = serialization.load_pem_private_key(key_bytes, password=None)
            if not isinstance(private_key, Ed25519PrivateKey):
                raise KeyError("Not an Ed25519 private key")
        else:
            raise ValueError(f"Unknown format: {format}")

        public_key = private_key.public_key()

        if not key_id:
            key_id = hashlib.sha256(public_key.public_bytes_raw()).hexdigest()[:16]

        return SigningKey(
            key_id=key_id,
            private_key_bytes=private_key.private_bytes_raw(),
            public_key_bytes=public_key.public_bytes_raw(),
            _private_key=private_key,
            _public_key=public_key,
        )

    def import_public_key(
        self,
        key_bytes: bytes,
        key_id: str = "",
        format: str = "raw",
    ) -> SigningKey:
        """
        Import public key from bytes.

        Args:
            key_bytes: Key bytes
            key_id: Key identifier
            format: 'raw', 'pem', or 'openssh'

        Returns:
            SigningKey (public only)
        """
        if format == "raw":
            if len(key_bytes) != KEY_SIZE:
                raise KeyError(f"Invalid key size: {len(key_bytes)}, expected {KEY_SIZE}")
            public_key = Ed25519PublicKey.from_public_bytes(key_bytes)
        elif format == "pem":
            public_key = serialization.load_pem_public_key(key_bytes)
            if not isinstance(public_key, Ed25519PublicKey):
                raise KeyError("Not an Ed25519 public key")
        elif format == "openssh":
            public_key = serialization.load_ssh_public_key(key_bytes)
            if not isinstance(public_key, Ed25519PublicKey):
                raise KeyError("Not an Ed25519 public key")
        else:
            raise ValueError(f"Unknown format: {format}")

        if not key_id:
            key_id = hashlib.sha256(public_key.public_bytes_raw()).hexdigest()[:16]

        return SigningKey(
            key_id=key_id,
            public_key_bytes=public_key.public_bytes_raw(),
            _public_key=public_key,
        )

    def save_key(
        self,
        key: SigningKey,
        path: Path,
        include_private: bool = False,
        password: Optional[bytes] = None,
    ) -> None:
        """
        Save key to file.

        Args:
            key: Key to save
            path: File path
            include_private: Include private key
            password: Password for encryption (private key only)
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        if include_private and key.private_key:
            # Save private key
            if password:
                encryption = serialization.BestAvailableEncryption(password)
            else:
                encryption = serialization.NoEncryption()

            key_bytes = key.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=encryption,
            )
        else:
            # Save public key only
            key_bytes = key.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )

        with open(path, 'wb') as f:
            f.write(key_bytes)

        # Restrictive permissions
        os.chmod(path, 0o600 if include_private else 0o644)

    def load_key(
        self,
        path: Path,
        key_id: str = "",
        password: Optional[bytes] = None,
    ) -> SigningKey:
        """
        Load key from file.

        Args:
            path: File path
            key_id: Key identifier
            password: Password for decryption

        Returns:
            SigningKey
        """
        with open(path, 'rb') as f:
            key_bytes = f.read()

        # Try private key first
        try:
            private_key = serialization.load_pem_private_key(key_bytes, password=password)
            if isinstance(private_key, Ed25519PrivateKey):
                public_key = private_key.public_key()

                if not key_id:
                    key_id = hashlib.sha256(public_key.public_bytes_raw()).hexdigest()[:16]

                return SigningKey(
                    key_id=key_id,
                    private_key_bytes=private_key.private_bytes_raw(),
                    public_key_bytes=public_key.public_bytes_raw(),
                    _private_key=private_key,
                    _public_key=public_key,
                )
        except Exception:
            pass

        # Try public key
        try:
            public_key = serialization.load_pem_public_key(key_bytes)
            if isinstance(public_key, Ed25519PublicKey):
                if not key_id:
                    key_id = hashlib.sha256(public_key.public_bytes_raw()).hexdigest()[:16]

                return SigningKey(
                    key_id=key_id,
                    public_key_bytes=public_key.public_bytes_raw(),
                    _public_key=public_key,
                )
        except Exception:
            pass

        raise KeyError(f"Could not load key from {path}")


def create_signer(
    signing_key_path: Optional[Path] = None,
    signing_key_bytes: Optional[bytes] = None,
    trusted_keys_dir: Optional[Path] = None,
    signer_id: str = "ccea-signer",
) -> Ed25519Signer:
    """
    Factory function to create a configured Ed25519Signer.

    Args:
        signing_key_path: Path to signing key
        signing_key_bytes: Raw signing key bytes
        trusted_keys_dir: Directory containing trusted public keys
        signer_id: Default signer identifier

    Returns:
        Configured Ed25519Signer
    """
    signer = Ed25519Signer(default_signer_id=signer_id)

    # Load trusted keys
    if trusted_keys_dir and trusted_keys_dir.exists():
        for key_file in trusted_keys_dir.glob("*.pem"):
            try:
                key = signer.load_key(key_file, key_id=key_file.stem)
                signer.add_trusted_key(key)
                logger.info(f"Loaded trusted key: {key.key_id}")
            except Exception as e:
                logger.warning(f"Failed to load trusted key {key_file}: {e}")

    return signer


# Utility functions for simple use cases

def sign_data(
    data: bytes,
    private_key: bytes,
    signer_id: str = "ccea-signer",
    payload_type: str = "",
) -> str:
    """
    Sign data and return base64-encoded signature.

    Simple utility function for one-off signing.

    Args:
        data: Data to sign
        private_key: Raw Ed25519 private key (32 bytes)
        signer_id: Signer identifier
        payload_type: Type of payload

    Returns:
        Base64-encoded signature JSON
    """
    signer = Ed25519Signer(default_signer_id=signer_id)
    key = signer.import_private_key(private_key, format="raw")
    signature = signer.sign(data, key, payload_type=payload_type)
    return json.dumps(signature.to_dict())


def verify_data(
    data: bytes,
    signature_json: str,
    public_key: bytes,
) -> bool:
    """
    Verify signature on data.

    Simple utility function for one-off verification.

    Args:
        data: Original data
        signature_json: JSON-encoded signature
        public_key: Raw Ed25519 public key (32 bytes)

    Returns:
        True if signature is valid
    """
    sig_data = json.loads(signature_json)
    signature = Signature.from_dict(sig_data)

    signer = Ed25519Signer()
    key = signer.import_public_key(public_key, key_id=signature.key_id, format="raw")

    return signer.verify(data, signature, key)
