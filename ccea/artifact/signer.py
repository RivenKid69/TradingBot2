# -*- coding: utf-8 -*-
"""
CCEA Artifact Signer.

Handles artifact signing and signature verification.
All artifacts MUST be signed before deployment.

Security:
- Agent rejects unsigned artifacts
- Pipeline blocks unpublished unsigned artifacts
- Supports multiple signing algorithms
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from cryptography.hazmat.primitives.asymmetric.types import (
    PrivateKeyTypes,
    PublicKeyTypes,
)

from ccea.crypto.keys import KeyPair, KeyAlgorithm, load_public_key
from ccea.crypto.signing import sign_message, verify_signature
from ccea.crypto.digest import compute_file_digest


@dataclass
class SignatureInfo:
    """Signature information."""

    algorithm: str
    signature: str
    key_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    signed_digest: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "algorithm": self.algorithm,
            "signature_value": self.signature,
            "key_id": self.key_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "signed_digest": self.signed_digest,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignatureInfo":
        """Create from dictionary."""
        return cls(
            algorithm=data["algorithm"],
            signature=data["signature_value"],
            key_id=data.get("key_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
            signed_digest=data.get("signed_digest"),
        )


class ArtifactSigner:
    """
    Artifact signer.

    Signs artifacts and manifests.
    """

    def __init__(
        self,
        private_key: PrivateKeyTypes,
        key_id: Optional[str] = None,
        algorithm: KeyAlgorithm = KeyAlgorithm.ED25519,
    ):
        """
        Initialize signer.

        Args:
            private_key: Private key for signing
            key_id: Optional key identifier
            algorithm: Signing algorithm
        """
        self.private_key = private_key
        self.key_id = key_id
        self.algorithm = algorithm

    @classmethod
    def from_keypair(cls, keypair: KeyPair) -> "ArtifactSigner":
        """Create signer from keypair."""
        return cls(
            private_key=keypair.private_key,
            key_id=keypair.key_id,
            algorithm=keypair.algorithm,
        )

    def sign_file(self, file_path: Path) -> SignatureInfo:
        """
        Sign a file.

        Args:
            file_path: Path to file

        Returns:
            SignatureInfo with signature

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file content
        content = file_path.read_bytes()

        # Compute digest
        digest = compute_file_digest(file_path)

        # Sign
        signature = sign_message(content, self.private_key)

        return SignatureInfo(
            algorithm=self.algorithm.value,
            signature=signature,
            key_id=self.key_id,
            timestamp=datetime.utcnow(),
            signed_digest=digest,
        )

    def sign_bytes(self, data: bytes) -> SignatureInfo:
        """
        Sign raw bytes.

        Args:
            data: Bytes to sign

        Returns:
            SignatureInfo
        """
        from ccea.crypto.digest import compute_digest

        digest = compute_digest(data)
        signature = sign_message(data, self.private_key)

        return SignatureInfo(
            algorithm=self.algorithm.value,
            signature=signature,
            key_id=self.key_id,
            timestamp=datetime.utcnow(),
            signed_digest=digest,
        )

    def sign_manifest(self, manifest_path: Path) -> SignatureInfo:
        """
        Sign a manifest file.

        Args:
            manifest_path: Path to manifest JSON

        Returns:
            SignatureInfo
        """
        return self.sign_file(manifest_path)


class SignatureVerifier:
    """
    Artifact signature verifier.

    Verifies artifact signatures against known public keys.
    """

    def __init__(self):
        """Initialize verifier."""
        self._keys: Dict[str, PublicKeyTypes] = {}
        self._trusted_digests: Dict[str, str] = {}

    def add_trusted_key(self, key_id: str, public_key: PublicKeyTypes) -> None:
        """
        Add a trusted public key.

        Args:
            key_id: Key identifier
            public_key: Public key object
        """
        self._keys[key_id] = public_key

    def add_trusted_key_pem(self, key_id: str, pem_data: str) -> None:
        """
        Add a trusted public key from PEM.

        Args:
            key_id: Key identifier
            pem_data: PEM-encoded public key
        """
        public_key = load_public_key(pem_data)
        self._keys[key_id] = public_key

    def add_trusted_digest(self, name: str, digest: str) -> None:
        """
        Add a trusted digest.

        Args:
            name: Artifact name/identifier
            digest: Expected SHA256 digest
        """
        self._trusted_digests[name] = digest

    def verify_file(
        self,
        file_path: Path,
        signature_info: SignatureInfo,
        key_id: Optional[str] = None,
    ) -> bool:
        """
        Verify file signature.

        Args:
            file_path: Path to file
            signature_info: Signature information
            key_id: Optional key ID (uses signature_info.key_id if not provided)

        Returns:
            True if verified, False otherwise
        """
        if not file_path.exists():
            return False

        # Get key
        kid = key_id or signature_info.key_id
        if kid is None or kid not in self._keys:
            return False

        public_key = self._keys[kid]

        # Read content
        content = file_path.read_bytes()

        # Verify digest if provided
        if signature_info.signed_digest:
            actual_digest = compute_file_digest(file_path)
            if actual_digest != signature_info.signed_digest:
                return False

        # Verify signature
        return verify_signature(content, signature_info.signature, public_key)

    def verify_bytes(
        self,
        data: bytes,
        signature_info: SignatureInfo,
        key_id: Optional[str] = None,
    ) -> bool:
        """
        Verify bytes signature.

        Args:
            data: Data bytes
            signature_info: Signature information
            key_id: Optional key ID

        Returns:
            True if verified
        """
        kid = key_id or signature_info.key_id
        if kid is None or kid not in self._keys:
            return False

        public_key = self._keys[kid]

        # Verify digest if provided
        if signature_info.signed_digest:
            from ccea.crypto.digest import compute_digest

            actual_digest = compute_digest(data)
            if actual_digest != signature_info.signed_digest:
                return False

        return verify_signature(data, signature_info.signature, public_key)

    def verify_artifact(
        self,
        artifact_path: Path,
        manifest_path: Path,
        artifact_signature: SignatureInfo,
        manifest_signature: SignatureInfo,
    ) -> tuple[bool, str]:
        """
        Verify complete artifact (package + manifest).

        Args:
            artifact_path: Path to artifact package
            manifest_path: Path to manifest
            artifact_signature: Artifact signature
            manifest_signature: Manifest signature

        Returns:
            Tuple of (is_valid, reason)
        """
        # Verify manifest signature
        if not self.verify_file(manifest_path, manifest_signature):
            return False, "Invalid manifest signature"

        # Verify artifact signature
        if not self.verify_file(artifact_path, artifact_signature):
            return False, "Invalid artifact signature"

        # Verify artifact digest matches manifest
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            # Check digest consistency
            actual_digest = compute_file_digest(artifact_path)
            manifest_digest = manifest.get("artifact_digest") or artifact_signature.signed_digest

            if manifest_digest and actual_digest != manifest_digest:
                return False, "Artifact digest mismatch"

        except Exception as e:
            return False, f"Verification error: {e}"

        return True, "Verified"

    def is_trusted_digest(self, name: str, digest: str) -> bool:
        """Check if digest is trusted for artifact."""
        trusted = self._trusted_digests.get(name)
        return trusted == digest if trusted else False
