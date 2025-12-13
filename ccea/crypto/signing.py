# -*- coding: utf-8 -*-
"""
CCEA Message Signing and Verification.

Provides cryptographic signing for Cloud-Agent protocol messages.
All messages must be signed for authentication (Design Doc 10.2).

Security:
- Ed25519 signatures for integrity and authenticity
- Verification required before processing any message
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from typing import Any, Dict, Optional, Union

from cryptography.hazmat.primitives.asymmetric import ed25519, ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.types import (
    PrivateKeyTypes,
    PublicKeyTypes,
)
from cryptography.exceptions import InvalidSignature

from ccea.crypto.keys import KeyAlgorithm, get_key_algorithm


def sign_message(
    message: bytes,
    private_key: PrivateKeyTypes,
) -> str:
    """
    Sign a message with private key.

    Args:
        message: Message bytes to sign
        private_key: Private key for signing

    Returns:
        Base64-encoded signature string
    """
    if isinstance(private_key, ed25519.Ed25519PrivateKey):
        signature = private_key.sign(message)
    elif isinstance(private_key, ec.EllipticCurvePrivateKey):
        signature = private_key.sign(message, ec.ECDSA(hashes.SHA256()))
    else:
        raise ValueError(f"Unsupported private key type: {type(private_key)}")

    return base64.b64encode(signature).decode("utf-8")


def verify_signature(
    message: bytes,
    signature: str,
    public_key: PublicKeyTypes,
) -> bool:
    """
    Verify a message signature.

    Args:
        message: Original message bytes
        signature: Base64-encoded signature string
        public_key: Public key for verification

    Returns:
        True if signature is valid, False otherwise
    """
    try:
        sig_bytes = base64.b64decode(signature)

        if isinstance(public_key, ed25519.Ed25519PublicKey):
            public_key.verify(sig_bytes, message)
        elif isinstance(public_key, ec.EllipticCurvePublicKey):
            public_key.verify(sig_bytes, message, ec.ECDSA(hashes.SHA256()))
        else:
            raise ValueError(f"Unsupported public key type: {type(public_key)}")

        return True
    except (InvalidSignature, ValueError):
        return False


def sign_json(
    data: Dict[str, Any],
    private_key: PrivateKeyTypes,
    key_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Sign JSON data and return data with signature field.

    Args:
        data: Dictionary to sign
        private_key: Private key for signing
        key_id: Optional key identifier

    Returns:
        Dictionary with 'signature' field added
    """
    # Remove any existing signature
    data_to_sign = {k: v for k, v in data.items() if k != "signature"}

    # Serialize deterministically
    message = json.dumps(
        data_to_sign,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_serializer,
    ).encode("utf-8")

    signature_value = sign_message(message, private_key)
    algorithm = get_key_algorithm(private_key.public_key())

    signature_obj = {
        "algorithm": algorithm.value,
        "value": signature_value,
    }
    if key_id:
        signature_obj["key_id"] = key_id

    return {**data_to_sign, "signature": signature_obj}


def verify_json_signature(
    data: Dict[str, Any],
    public_key: PublicKeyTypes,
) -> bool:
    """
    Verify signature on JSON data.

    Args:
        data: Dictionary with 'signature' field
        public_key: Public key for verification

    Returns:
        True if signature is valid, False otherwise
    """
    if "signature" not in data:
        return False

    signature_obj = data["signature"]
    if not isinstance(signature_obj, dict):
        return False

    signature_value = signature_obj.get("value")
    if not signature_value:
        return False

    # Reconstruct data without signature
    data_to_verify = {k: v for k, v in data.items() if k != "signature"}

    # Serialize deterministically
    message = json.dumps(
        data_to_verify,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_serializer,
    ).encode("utf-8")

    return verify_signature(message, signature_value, public_key)


def _json_serializer(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default json."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class MessageSigner:
    """
    Helper class for signing protocol messages.

    Usage:
        signer = MessageSigner(private_key, key_id="agent_key_1")
        signed_msg = signer.sign(message_dict)
    """

    def __init__(
        self,
        private_key: PrivateKeyTypes,
        key_id: Optional[str] = None,
    ):
        """
        Initialize signer.

        Args:
            private_key: Private key for signing
            key_id: Optional key identifier
        """
        self.private_key = private_key
        self.key_id = key_id
        self.algorithm = get_key_algorithm(private_key.public_key())

    def sign(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sign message data."""
        return sign_json(data, self.private_key, self.key_id)


class MessageVerifier:
    """
    Helper class for verifying protocol messages.

    Usage:
        verifier = MessageVerifier()
        verifier.add_key("agent_123", public_key)
        is_valid = verifier.verify(message_dict, "agent_123")
    """

    def __init__(self):
        """Initialize verifier with empty key store."""
        self._keys: Dict[str, PublicKeyTypes] = {}

    def add_key(self, key_id: str, public_key: PublicKeyTypes) -> None:
        """Add a public key to the verifier."""
        self._keys[key_id] = public_key

    def remove_key(self, key_id: str) -> None:
        """Remove a public key from the verifier."""
        self._keys.pop(key_id, None)

    def get_key(self, key_id: str) -> Optional[PublicKeyTypes]:
        """Get a public key by ID."""
        return self._keys.get(key_id)

    def verify(
        self,
        data: Dict[str, Any],
        key_id: Optional[str] = None,
    ) -> bool:
        """
        Verify message signature.

        Args:
            data: Message with signature
            key_id: Key ID to use (or extracted from signature)

        Returns:
            True if valid, False otherwise
        """
        # Extract key_id from signature if not provided
        if key_id is None:
            sig = data.get("signature", {})
            key_id = sig.get("key_id")

        if key_id is None:
            return False

        public_key = self._keys.get(key_id)
        if public_key is None:
            return False

        return verify_json_signature(data, public_key)
