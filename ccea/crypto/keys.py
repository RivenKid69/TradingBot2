# -*- coding: utf-8 -*-
"""
CCEA Key Management.

Provides Ed25519 key generation, loading, and serialization.
Agent generates keypair locally; only public key is shared with cloud.

Security (design intent):
- Private keys designed to remain in the agent (verify via security review)
- Keys are generated with cryptographically secure random
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, ec
from cryptography.hazmat.primitives.asymmetric.types import (
    PrivateKeyTypes,
    PublicKeyTypes,
)
from cryptography.hazmat.backends import default_backend


class KeyAlgorithm(str, Enum):
    """Supported key algorithms."""

    ED25519 = "ed25519"
    ECDSA_P256 = "ecdsa-p256"


@dataclass
class KeyPair:
    """
    Cryptographic key pair.

    Attributes:
        algorithm: Key algorithm used
        private_key: Private key object (NEVER share)
        public_key: Public key object (safe to share)
        key_id: Optional key identifier
    """

    algorithm: KeyAlgorithm
    private_key: PrivateKeyTypes
    public_key: PublicKeyTypes
    key_id: Optional[str] = None

    def get_public_key_pem(self) -> str:
        """Get public key in PEM format."""
        return serialize_public_key(self.public_key)

    def get_private_key_pem(self, password: Optional[bytes] = None) -> str:
        """
        Get private key in PEM format.

        Args:
            password: Optional password for encryption

        Returns:
            PEM-encoded private key

        WARNING: Handle with care. Private key should stay local.
        """
        return serialize_private_key(self.private_key, password)


def generate_keypair(
    algorithm: KeyAlgorithm = KeyAlgorithm.ED25519,
    key_id: Optional[str] = None,
) -> KeyPair:
    """
    Generate a new cryptographic key pair.

    Args:
        algorithm: Key algorithm to use (default: Ed25519)
        key_id: Optional key identifier

    Returns:
        KeyPair with private and public keys

    Example:
        keypair = generate_keypair()
        public_pem = keypair.get_public_key_pem()
        # Share public_pem with cloud during enrollment
    """
    if algorithm == KeyAlgorithm.ED25519:
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
    elif algorithm == KeyAlgorithm.ECDSA_P256:
        private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
        public_key = private_key.public_key()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return KeyPair(
        algorithm=algorithm,
        private_key=private_key,
        public_key=public_key,
        key_id=key_id,
    )


def serialize_public_key(public_key: PublicKeyTypes) -> str:
    """
    Serialize public key to PEM format.

    Args:
        public_key: Public key object

    Returns:
        PEM-encoded public key string
    """
    pem_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return pem_bytes.decode("utf-8")


def serialize_private_key(
    private_key: PrivateKeyTypes,
    password: Optional[bytes] = None,
) -> str:
    """
    Serialize private key to PEM format.

    Args:
        private_key: Private key object
        password: Optional password for encryption

    Returns:
        PEM-encoded private key string

    WARNING: Handle private keys with extreme care.
    """
    if password:
        encryption = serialization.BestAvailableEncryption(password)
    else:
        encryption = serialization.NoEncryption()

    pem_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=encryption,
    )
    return pem_bytes.decode("utf-8")


def load_public_key(pem_data: str) -> PublicKeyTypes:
    """
    Load public key from PEM format.

    Args:
        pem_data: PEM-encoded public key string

    Returns:
        Public key object

    Raises:
        ValueError: If PEM data is invalid
    """
    try:
        return serialization.load_pem_public_key(
            pem_data.encode("utf-8"),
            backend=default_backend(),
        )
    except Exception as e:
        raise ValueError(f"Invalid public key PEM: {e}") from e


def load_private_key(
    pem_data: str,
    password: Optional[bytes] = None,
) -> PrivateKeyTypes:
    """
    Load private key from PEM format.

    Args:
        pem_data: PEM-encoded private key string
        password: Optional password if key is encrypted

    Returns:
        Private key object

    Raises:
        ValueError: If PEM data is invalid or password is wrong
    """
    try:
        return serialization.load_pem_private_key(
            pem_data.encode("utf-8"),
            password=password,
            backend=default_backend(),
        )
    except Exception as e:
        raise ValueError(f"Invalid private key PEM: {e}") from e


def get_key_algorithm(key: PublicKeyTypes) -> KeyAlgorithm:
    """
    Detect algorithm from public key.

    Args:
        key: Public key object

    Returns:
        KeyAlgorithm enum value
    """
    if isinstance(key, ed25519.Ed25519PublicKey):
        return KeyAlgorithm.ED25519
    elif isinstance(key, ec.EllipticCurvePublicKey):
        if isinstance(key.curve, ec.SECP256R1):
            return KeyAlgorithm.ECDSA_P256
    raise ValueError(f"Unknown key type: {type(key)}")
