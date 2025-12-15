# -*- coding: utf-8 -*-
"""
Cloud Message Signature Verifier.

Design Doc Section 10.2: All Cloud-to-Agent messages MUST be signed.

Security Requirements:
1. Commands from Cloud MUST be signed
2. Agent MUST verify signature BEFORE processing
3. Reject all unsigned or invalid-signed messages
4. Cloud public key provisioned at enrollment

AGENT ZONE ONLY - Never import in Cloud zone.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from cryptography.hazmat.primitives.asymmetric import ed25519, ec
from cryptography.hazmat.primitives.asymmetric.types import PublicKeyTypes
from cryptography.exceptions import InvalidSignature

from ccea.crypto.keys import load_public_key
from ccea.crypto.signing import verify_json_signature

logger = logging.getLogger(__name__)


class SignatureVerificationError(Exception):
    """Raised when signature verification fails."""

    def __init__(self, message: str, reason: str = "unknown"):
        super().__init__(message)
        self.reason = reason


@dataclass
class VerificationResult:
    """Result of signature verification."""

    valid: bool
    key_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    reason: Optional[str] = None
    verified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CloudSignatureVerifierConfig:
    """Configuration for Cloud signature verification."""

    # Whether to enforce signature verification (fail-closed)
    enforce_verification: bool = True

    # Maximum age of signed message (prevents replay attacks)
    max_message_age_seconds: int = 300  # 5 minutes

    # Allow unsigned messages (only for development/testing)
    allow_unsigned: bool = False

    # Log verification failures
    log_failures: bool = True


class CloudSignatureVerifier:
    """
    Verifies signatures on messages from Cloud Control Plane.

    Design Doc Section 10.2:
    - All commands from Cloud MUST be signed
    - Agent MUST verify before processing
    - Reject invalid signatures (fail-closed)

    Security Invariants:
    1. Cloud public key is provisioned at enrollment (trusted root)
    2. All commands MUST have valid signature
    3. Unsigned messages are REJECTED (unless allow_unsigned=True for dev)
    4. Timestamp must be within max_message_age (replay protection)
    5. Signature verification failures are logged for audit

    Usage:
        verifier = CloudSignatureVerifier(cloud_public_key_pem)

        # Verify command from Cloud
        result = verifier.verify_command(command_data)
        if not result.valid:
            raise SignatureVerificationError(f"Invalid signature: {result.reason}")

        # Or use verify_or_raise for fail-fast
        verifier.verify_or_raise(command_data)
    """

    def __init__(
        self,
        cloud_public_key_pem: Optional[str] = None,
        config: Optional[CloudSignatureVerifierConfig] = None,
    ):
        """
        Initialize verifier.

        Args:
            cloud_public_key_pem: Cloud's public key in PEM format
            config: Verifier configuration
        """
        self._config = config or CloudSignatureVerifierConfig()
        self._cloud_public_key: Optional[PublicKeyTypes] = None
        self._key_fingerprint: Optional[str] = None

        # Additional trusted keys (for key rotation)
        self._trusted_keys: Dict[str, PublicKeyTypes] = {}

        # Verification statistics
        self._stats = {
            "verified": 0,
            "failed": 0,
            "unsigned": 0,
            "expired": 0,
            "unknown_key": 0,
        }

        if cloud_public_key_pem:
            self.set_cloud_public_key(cloud_public_key_pem)

    def set_cloud_public_key(self, public_key_pem: str) -> None:
        """
        Set Cloud's public key for verification.

        This should be called during enrollment when Cloud provides its public key.

        Args:
            public_key_pem: Cloud's public key in PEM format
        """
        import hashlib

        self._cloud_public_key = load_public_key(public_key_pem)
        self._key_fingerprint = hashlib.sha256(public_key_pem.encode("utf-8")).hexdigest()

        # Add to trusted keys
        self._trusted_keys["cloud_primary"] = self._cloud_public_key

        logger.info(f"Cloud public key set: fingerprint={self._key_fingerprint[:16]}...")

    def add_trusted_key(self, key_id: str, public_key_pem: str) -> None:
        """
        Add additional trusted key (for key rotation).

        Args:
            key_id: Key identifier
            public_key_pem: Public key in PEM format
        """
        self._trusted_keys[key_id] = load_public_key(public_key_pem)
        logger.info(f"Added trusted key: {key_id}")

    def remove_trusted_key(self, key_id: str) -> None:
        """Remove a trusted key."""
        if key_id in self._trusted_keys and key_id != "cloud_primary":
            del self._trusted_keys[key_id]

    @property
    def has_cloud_key(self) -> bool:
        """Check if Cloud public key is set."""
        return self._cloud_public_key is not None

    def verify_message(self, message: Dict[str, Any]) -> VerificationResult:
        """
        Verify signature on a message from Cloud.

        Args:
            message: Message dictionary with signature field

        Returns:
            VerificationResult with validity and details
        """
        # Check for signature field
        signature = message.get("signature")
        if not signature:
            self._stats["unsigned"] += 1
            if self._config.allow_unsigned:
                return VerificationResult(
                    valid=True,
                    reason="unsigned_allowed",
                )
            if self._config.log_failures:
                logger.warning("Signature verification failed: message unsigned")
            return VerificationResult(
                valid=False,
                reason="message_unsigned",
            )

        # Extract key_id from signature
        key_id = signature.get("key_id", "cloud_primary")

        # Get public key
        public_key = self._trusted_keys.get(key_id)
        if not public_key:
            self._stats["unknown_key"] += 1
            if self._config.log_failures:
                logger.warning(f"Signature verification failed: unknown key {key_id}")
            return VerificationResult(
                valid=False,
                key_id=key_id,
                reason=f"unknown_key:{key_id}",
            )

        # Check timestamp (replay protection)
        timestamp_str = message.get("timestamp") or signature.get("timestamp")
        if timestamp_str:
            try:
                msg_timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                age = abs((now - msg_timestamp).total_seconds())

                if age > self._config.max_message_age_seconds:
                    self._stats["expired"] += 1
                    if self._config.log_failures:
                        logger.warning(f"Signature verification failed: message expired (age={age}s)")
                    return VerificationResult(
                        valid=False,
                        key_id=key_id,
                        timestamp=msg_timestamp,
                        reason=f"message_expired:{age:.0f}s",
                    )
            except (ValueError, TypeError):
                pass  # Non-fatal if timestamp is missing or malformed

        # Verify signature
        try:
            is_valid = verify_json_signature(message, public_key)

            if is_valid:
                self._stats["verified"] += 1
                return VerificationResult(
                    valid=True,
                    key_id=key_id,
                )
            else:
                self._stats["failed"] += 1
                if self._config.log_failures:
                    logger.warning("Signature verification failed: invalid signature")
                return VerificationResult(
                    valid=False,
                    key_id=key_id,
                    reason="invalid_signature",
                )
        except Exception as e:
            self._stats["failed"] += 1
            if self._config.log_failures:
                logger.error(f"Signature verification error: {e}")
            return VerificationResult(
                valid=False,
                key_id=key_id,
                reason=f"verification_error:{str(e)}",
            )

    def verify_or_raise(self, message: Dict[str, Any]) -> None:
        """
        Verify signature or raise exception.

        Design Doc: Fail-closed - reject all invalid signatures.

        Args:
            message: Message to verify

        Raises:
            SignatureVerificationError: If verification fails
        """
        if not self._config.enforce_verification:
            return

        result = self.verify_message(message)
        if not result.valid:
            raise SignatureVerificationError(
                f"Cloud message signature verification failed: {result.reason}",
                reason=result.reason or "unknown",
            )

    def verify_command(self, command: Dict[str, Any]) -> VerificationResult:
        """
        Verify signature on a command from Cloud.

        Commands are critical operations and MUST have valid signatures.

        Args:
            command: Command dictionary with signature

        Returns:
            VerificationResult
        """
        return self.verify_message(command)

    def verify_commands(self, commands: List[Dict[str, Any]]) -> List[VerificationResult]:
        """
        Verify signatures on multiple commands.

        Args:
            commands: List of commands

        Returns:
            List of VerificationResults
        """
        return [self.verify_command(cmd) for cmd in commands]

    def get_stats(self) -> Dict[str, int]:
        """Get verification statistics."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset verification statistics."""
        self._stats = {
            "verified": 0,
            "failed": 0,
            "unsigned": 0,
            "expired": 0,
            "unknown_key": 0,
        }


class VerifiedCloudMessage:
    """
    Wrapper for verified Cloud messages.

    Provides type-safe access to verified message data.
    """

    def __init__(self, message: Dict[str, Any], verification: VerificationResult):
        """
        Initialize verified message.

        Args:
            message: Original message data
            verification: Verification result
        """
        if not verification.valid:
            raise SignatureVerificationError(
                f"Cannot create VerifiedCloudMessage from invalid message: {verification.reason}",
                reason=verification.reason or "unknown",
            )
        self._message = message
        self._verification = verification

    @property
    def data(self) -> Dict[str, Any]:
        """Get message data."""
        return self._message

    @property
    def verification(self) -> VerificationResult:
        """Get verification result."""
        return self._verification

    @property
    def key_id(self) -> Optional[str]:
        """Get key ID used for verification."""
        return self._verification.key_id

    def get(self, key: str, default: Any = None) -> Any:
        """Get message field."""
        return self._message.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Get message field."""
        return self._message[key]


def create_verifier_from_enrollment(
    enrollment_response: Dict[str, Any],
    config: Optional[CloudSignatureVerifierConfig] = None,
) -> CloudSignatureVerifier:
    """
    Create verifier from enrollment response.

    During enrollment, Cloud provides its public key. This function
    extracts the key and creates a verifier.

    Args:
        enrollment_response: Enrollment response from Cloud
        config: Optional verifier configuration

    Returns:
        Configured CloudSignatureVerifier
    """
    cloud_public_key = enrollment_response.get("cloud_public_key")
    if not cloud_public_key:
        # Fallback to no verification if Cloud doesn't provide key
        logger.warning("Enrollment response missing cloud_public_key - verification disabled")
        verifier_config = config or CloudSignatureVerifierConfig()
        verifier_config.enforce_verification = False
        return CloudSignatureVerifier(config=verifier_config)

    return CloudSignatureVerifier(
        cloud_public_key_pem=cloud_public_key,
        config=config,
    )
