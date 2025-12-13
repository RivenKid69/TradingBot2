# -*- coding: utf-8 -*-
"""
CCEA Digest Computation.

Provides SHA256 digest computation for:
- Artifact content
- Configuration blobs
- Manifest verification

Security:
- All artifacts and configs are digest-pinned
- Agent verifies digests before execution
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, BinaryIO, Dict, Union


def compute_digest(data: bytes) -> str:
    """
    Compute SHA256 digest of bytes.

    Args:
        data: Bytes to hash

    Returns:
        Digest string in format 'sha256:<hex>'
    """
    digest = hashlib.sha256(data).hexdigest()
    return f"sha256:{digest}"


def compute_file_digest(file_path: Union[str, Path]) -> str:
    """
    Compute SHA256 digest of a file.

    Args:
        file_path: Path to file

    Returns:
        Digest string in format 'sha256:<hex>'

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return f"sha256:{sha256.hexdigest()}"


def compute_stream_digest(stream: BinaryIO) -> str:
    """
    Compute SHA256 digest of a binary stream.

    Args:
        stream: Binary stream to hash

    Returns:
        Digest string in format 'sha256:<hex>'
    """
    sha256 = hashlib.sha256()
    for chunk in iter(lambda: stream.read(8192), b""):
        sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"


def compute_json_digest(data: Dict[str, Any]) -> str:
    """
    Compute SHA256 digest of JSON data.

    Uses deterministic serialization (sorted keys, no whitespace).

    Args:
        data: Dictionary to hash

    Returns:
        Digest string in format 'sha256:<hex>'
    """
    content = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return compute_digest(content)


def verify_digest(data: bytes, expected_digest: str) -> bool:
    """
    Verify that data matches expected digest.

    Args:
        data: Bytes to verify
        expected_digest: Expected digest string

    Returns:
        True if digest matches, False otherwise
    """
    actual_digest = compute_digest(data)
    return actual_digest == expected_digest


def verify_file_digest(file_path: Union[str, Path], expected_digest: str) -> bool:
    """
    Verify that a file matches expected digest.

    Args:
        file_path: Path to file
        expected_digest: Expected digest string

    Returns:
        True if digest matches, False otherwise
    """
    try:
        actual_digest = compute_file_digest(file_path)
        return actual_digest == expected_digest
    except FileNotFoundError:
        return False


def extract_hash_from_digest(digest: str) -> str:
    """
    Extract hex hash from digest string.

    Args:
        digest: Digest string in format 'sha256:<hex>'

    Returns:
        Hex hash string

    Raises:
        ValueError: If digest format is invalid
    """
    if not digest.startswith("sha256:"):
        raise ValueError(f"Invalid digest format: {digest}")
    return digest[7:]


def _json_default(obj: Any) -> Any:
    """Default JSON serializer for non-standard types."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class DigestVerifier:
    """
    Helper class for digest verification.

    Maintains a registry of known digests.
    """

    def __init__(self):
        """Initialize with empty registry."""
        self._registry: Dict[str, str] = {}

    def register(self, name: str, digest: str) -> None:
        """Register a known digest."""
        if not digest.startswith("sha256:"):
            raise ValueError(f"Invalid digest format: {digest}")
        self._registry[name] = digest

    def get(self, name: str) -> str:
        """Get registered digest by name."""
        digest = self._registry.get(name)
        if digest is None:
            raise KeyError(f"Unknown digest: {name}")
        return digest

    def verify(self, name: str, data: bytes) -> bool:
        """Verify data against registered digest."""
        expected = self._registry.get(name)
        if expected is None:
            return False
        return verify_digest(data, expected)

    def verify_file(self, name: str, file_path: Union[str, Path]) -> bool:
        """Verify file against registered digest."""
        expected = self._registry.get(name)
        if expected is None:
            return False
        return verify_file_digest(file_path, expected)
