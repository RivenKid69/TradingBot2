# -*- coding: utf-8 -*-
"""
Cloud Command Signer - Signs commands before sending to agents.

Design Doc 10.2: All Cloud-to-Agent messages MUST be signed.

CLOUD ZONE ONLY.

This module provides:
- Cloud keypair generation/loading
- Command signing before dispatch
- Public key distribution to agents at enrollment

Security:
- Private key stored securely (env var or secrets manager)
- All commands signed with Ed25519
- Key rotation support via key_id
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.asymmetric.types import PrivateKeyTypes, PublicKeyTypes

from ccea.crypto.keys import (
    KeyAlgorithm,
    KeyPair,
    generate_keypair,
    load_private_key,
    load_public_key,
    serialize_public_key,
)
from ccea.crypto.signing import sign_json

logger = logging.getLogger(__name__)


# Environment variables for Cloud keys
ENV_CLOUD_PRIVATE_KEY = "CCEA_CLOUD_PRIVATE_KEY"
ENV_CLOUD_PRIVATE_KEY_FILE = "CCEA_CLOUD_PRIVATE_KEY_FILE"
ENV_CLOUD_KEY_ID = "CCEA_CLOUD_KEY_ID"


class CloudKeyNotConfiguredError(RuntimeError):
    """Cloud signing key is not configured."""

    pass


@dataclass
class CloudKeyInfo:
    """Information about Cloud's signing key."""

    key_id: str
    algorithm: str
    public_key_pem: str
    fingerprint: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CloudCommandSigner:
    """
    Signs commands for Cloud-to-Agent communication.

    Design Doc 10.2:
    - All commands from Cloud MUST be signed
    - Agent MUST verify signature before processing
    - Cloud public key provisioned at enrollment

    Usage:
        # Initialize (typically once at app startup)
        signer = CloudCommandSigner.from_env()

        # Sign a command before sending to agent
        signed_cmd = signer.sign_command(command_dict)

        # Get public key for agent enrollment
        public_key_pem = signer.get_public_key_pem()

    Key Management:
        Keys can be provided via:
        1. CCEA_CLOUD_PRIVATE_KEY env var (PEM string)
        2. CCEA_CLOUD_PRIVATE_KEY_FILE env var (path to PEM file)
        3. Auto-generated (development only, NOT for production)
    """

    def __init__(
        self,
        private_key: PrivateKeyTypes,
        key_id: str = "cloud_primary",
    ):
        """
        Initialize signer with private key.

        Args:
            private_key: Cloud's private signing key
            key_id: Identifier for this key (for rotation support)
        """
        self._private_key = private_key
        self._public_key = private_key.public_key()
        self._key_id = key_id

        # Compute fingerprint
        public_key_pem = serialize_public_key(self._public_key)
        self._fingerprint = hashlib.sha256(public_key_pem.encode("utf-8")).hexdigest()

        # Stats
        self._commands_signed = 0

        logger.info(
            f"CloudCommandSigner initialized: key_id={key_id}, "
            f"fingerprint={self._fingerprint[:16]}..."
        )

    @classmethod
    def from_env(cls, auto_generate: bool = False) -> "CloudCommandSigner":
        """
        Create signer from environment variables.

        Args:
            auto_generate: If True, generate key if not found (dev only)

        Returns:
            Configured CloudCommandSigner

        Raises:
            CloudKeyNotConfiguredError: If key not found and auto_generate=False
        """
        key_id = os.environ.get(ENV_CLOUD_KEY_ID, "cloud_primary")

        # Try loading from env var (PEM string)
        pem_string = os.environ.get(ENV_CLOUD_PRIVATE_KEY)
        if pem_string:
            private_key = load_private_key(pem_string)
            return cls(private_key, key_id)

        # Try loading from file
        key_file = os.environ.get(ENV_CLOUD_PRIVATE_KEY_FILE)
        if key_file:
            key_path = Path(key_file)
            if key_path.exists():
                pem_string = key_path.read_text()
                private_key = load_private_key(pem_string)
                return cls(private_key, key_id)
            else:
                logger.warning(f"Cloud key file not found: {key_file}")

        # Auto-generate for development
        if auto_generate:
            logger.warning(
                "Cloud signing key not configured. Auto-generating for development. "
                "DO NOT USE IN PRODUCTION!"
            )
            keypair = generate_keypair(KeyAlgorithm.ED25519)
            return cls(keypair.private_key, key_id)

        raise CloudKeyNotConfiguredError(
            "Cloud signing key not configured. Set CCEA_CLOUD_PRIVATE_KEY or "
            "CCEA_CLOUD_PRIVATE_KEY_FILE environment variable."
        )

    @classmethod
    def from_pem(cls, pem_data: str, key_id: str = "cloud_primary") -> "CloudCommandSigner":
        """
        Create signer from PEM-encoded private key.

        Args:
            pem_data: PEM-encoded private key string
            key_id: Key identifier

        Returns:
            CloudCommandSigner instance
        """
        private_key = load_private_key(pem_data)
        return cls(private_key, key_id)

    @classmethod
    def generate(cls, key_id: str = "cloud_primary") -> "CloudCommandSigner":
        """
        Generate a new signing key.

        WARNING: For development/testing only.
        Production should use pre-generated keys stored securely.

        Args:
            key_id: Key identifier

        Returns:
            CloudCommandSigner with new keypair
        """
        keypair = generate_keypair(KeyAlgorithm.ED25519)
        return cls(keypair.private_key, key_id)

    @property
    def key_id(self) -> str:
        """Get key identifier."""
        return self._key_id

    @property
    def fingerprint(self) -> str:
        """Get key fingerprint."""
        return self._fingerprint

    @property
    def commands_signed(self) -> int:
        """Get count of commands signed."""
        return self._commands_signed

    def get_public_key_pem(self) -> str:
        """
        Get Cloud's public key in PEM format.

        This should be provided to agents during enrollment
        so they can verify command signatures.

        Returns:
            PEM-encoded public key string
        """
        return serialize_public_key(self._public_key)

    def get_key_info(self) -> CloudKeyInfo:
        """
        Get information about Cloud's signing key.

        Returns:
            CloudKeyInfo with public key and metadata
        """
        return CloudKeyInfo(
            key_id=self._key_id,
            algorithm=KeyAlgorithm.ED25519.value,
            public_key_pem=self.get_public_key_pem(),
            fingerprint=self._fingerprint,
        )

    def sign_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sign a command for agent delivery.

        Design Doc 10.2: All commands MUST be signed.

        Args:
            command: Command dictionary to sign

        Returns:
            Command with 'signature' field added

        Example:
            cmd = {
                "id": "...",
                "command_type": "REQUEST_START_RUN",
                "payload_ref": "sha256:...",
                ...
            }
            signed_cmd = signer.sign_command(cmd)
            # signed_cmd now has 'signature' field
        """
        # Add timestamp for replay protection
        if "timestamp" not in command:
            command = {
                **command,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Sign the command
        signed = sign_json(command, self._private_key, self._key_id)

        self._commands_signed += 1
        return signed

    def sign_commands(self, commands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sign multiple commands.

        Args:
            commands: List of command dictionaries

        Returns:
            List of signed commands
        """
        return [self.sign_command(cmd) for cmd in commands]

    def sign_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sign a generic message (not just commands).

        Can be used for other Cloud-to-Agent messages like heartbeat responses.

        Args:
            message: Message dictionary

        Returns:
            Message with 'signature' field
        """
        if "timestamp" not in message:
            message = {
                **message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        return sign_json(message, self._private_key, self._key_id)


# Singleton instance for application-wide use
_cloud_signer: Optional[CloudCommandSigner] = None


def get_cloud_signer(auto_init: bool = True) -> CloudCommandSigner:
    """
    Get the application-wide Cloud command signer.

    Args:
        auto_init: If True, initialize from environment if not already set

    Returns:
        CloudCommandSigner instance

    Raises:
        CloudKeyNotConfiguredError: If not configured and auto_init=False
    """
    global _cloud_signer

    if _cloud_signer is None:
        if auto_init:
            # In development, auto-generate; in production, require configuration
            is_dev = os.environ.get("CCEA_ENV", "development") == "development"
            _cloud_signer = CloudCommandSigner.from_env(auto_generate=is_dev)
        else:
            raise CloudKeyNotConfiguredError("Cloud signer not initialized")

    return _cloud_signer


def set_cloud_signer(signer: CloudCommandSigner) -> None:
    """
    Set the application-wide Cloud command signer.

    Use this for testing or custom initialization.

    Args:
        signer: CloudCommandSigner instance to use
    """
    global _cloud_signer
    _cloud_signer = signer


def reset_cloud_signer() -> None:
    """Reset the global signer (for testing)."""
    global _cloud_signer
    _cloud_signer = None


def sign_command_for_agent(command: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to sign a command using global signer.

    Args:
        command: Command dictionary to sign

    Returns:
        Signed command

    Raises:
        CloudKeyNotConfiguredError: If signer not configured
    """
    signer = get_cloud_signer()
    return signer.sign_command(command)


def get_cloud_public_key_pem() -> str:
    """
    Get Cloud's public key PEM for agent enrollment.

    Returns:
        PEM-encoded public key

    Raises:
        CloudKeyNotConfiguredError: If signer not configured
    """
    signer = get_cloud_signer()
    return signer.get_public_key_pem()
