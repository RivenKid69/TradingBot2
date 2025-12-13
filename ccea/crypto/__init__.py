# -*- coding: utf-8 -*-
"""
CCEA Cryptographic Utilities.

Provides:
- Key generation (Ed25519, ECDSA)
- Message signing and verification
- Digest computation
- Secure token generation

Security: All cryptographic operations follow best practices.
"""

from ccea.crypto.keys import (
    KeyPair,
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
)

from ccea.crypto.digest import (
    compute_digest,
    compute_file_digest,
    verify_digest,
    compute_json_digest,
)

from ccea.crypto.tokens import (
    generate_enrollment_token,
    generate_idempotency_key,
    generate_agent_id,
    generate_command_id,
)

__all__ = [
    # Keys
    "KeyPair",
    "generate_keypair",
    "load_private_key",
    "load_public_key",
    "serialize_private_key",
    "serialize_public_key",
    # Signing
    "sign_message",
    "verify_signature",
    "sign_json",
    "verify_json_signature",
    # Digest
    "compute_digest",
    "compute_file_digest",
    "verify_digest",
    "compute_json_digest",
    # Tokens
    "generate_enrollment_token",
    "generate_idempotency_key",
    "generate_agent_id",
    "generate_command_id",
]
