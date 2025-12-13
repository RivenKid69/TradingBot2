# -*- coding: utf-8 -*-
"""
CCEA Cryptographic Utilities.

Provides:
- Key generation (Ed25519, ECDSA)
- Key management with rotation and trust root
- Message signing and verification
- Digest computation
- Secure token generation

Security: All cryptographic operations follow best practices.
"""

from ccea.crypto.keys import (
    KeyPair,
    KeyAlgorithm,
    generate_keypair,
    load_private_key,
    load_public_key,
    serialize_private_key,
    serialize_public_key,
)

from ccea.crypto.signing import (
    sign_message,
    verify_signature,
    sign_json,
    verify_json_signature,
    MessageSigner,
    MessageVerifier,
)

from ccea.crypto.digest import (
    compute_digest,
    compute_file_digest,
    verify_digest,
    verify_file_digest,
    compute_json_digest,
    DigestVerifier,
)

from ccea.crypto.tokens import (
    generate_enrollment_token,
    generate_idempotency_key,
    generate_agent_id,
    generate_command_id,
)

from ccea.crypto.key_manager import (
    KeyManager,
    KeyMetadata,
    KeyStatus,
    KeyPurpose,
    TrustLevel,
    TrustRoot,
    SigningKeyProvider,
)

__all__ = [
    # Keys
    "KeyPair",
    "KeyAlgorithm",
    "generate_keypair",
    "load_private_key",
    "load_public_key",
    "serialize_private_key",
    "serialize_public_key",
    # Key Manager
    "KeyManager",
    "KeyMetadata",
    "KeyStatus",
    "KeyPurpose",
    "TrustLevel",
    "TrustRoot",
    "SigningKeyProvider",
    # Signing
    "sign_message",
    "verify_signature",
    "sign_json",
    "verify_json_signature",
    "MessageSigner",
    "MessageVerifier",
    # Digest
    "compute_digest",
    "compute_file_digest",
    "verify_digest",
    "verify_file_digest",
    "compute_json_digest",
    "DigestVerifier",
    # Tokens
    "generate_enrollment_token",
    "generate_idempotency_key",
    "generate_agent_id",
    "generate_command_id",
]
