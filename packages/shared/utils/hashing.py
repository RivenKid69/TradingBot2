# -*- coding: utf-8 -*-
"""
Hashing Utilities.

Secure hashing functions for content integrity verification.
Safe for use in both Cloud and Agent zones.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Union


def compute_sha256(
    data: Union[bytes, str, BinaryIO],
    as_hex: bool = True,
    with_prefix: bool = False,
) -> str:
    """
    Compute SHA256 hash of data.

    Args:
        data: Data to hash (bytes, string, or file-like object)
        as_hex: Return hex string (True) or bytes (False)
        with_prefix: Include 'sha256:' prefix

    Returns:
        SHA256 hash as hex string or bytes

    Examples:
        >>> compute_sha256(b"hello")
        '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
        >>> compute_sha256("hello", with_prefix=True)
        'sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    """
    hasher = hashlib.sha256()

    if isinstance(data, str):
        hasher.update(data.encode("utf-8"))
    elif isinstance(data, bytes):
        hasher.update(data)
    elif hasattr(data, "read"):
        # File-like object
        while chunk := data.read(8192):
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            hasher.update(chunk)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    if as_hex:
        hex_digest = hasher.hexdigest()
        if with_prefix:
            return f"sha256:{hex_digest}"
        return hex_digest

    return hasher.digest()


def compute_file_hash(
    file_path: Union[str, Path],
    with_prefix: bool = False,
) -> str:
    """
    Compute SHA256 hash of a file.

    Args:
        file_path: Path to file
        with_prefix: Include 'sha256:' prefix

    Returns:
        SHA256 hash of file contents
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, "rb") as f:
        return compute_sha256(f, with_prefix=with_prefix)


def compute_content_hash(
    content: Dict[str, Any],
    exclude_keys: Optional[set] = None,
    with_prefix: bool = False,
) -> str:
    """
    Compute deterministic hash of dictionary content.

    Useful for computing hashes of configs, manifests, etc.

    Args:
        content: Dictionary to hash
        exclude_keys: Keys to exclude from hashing
        with_prefix: Include 'sha256:' prefix

    Returns:
        SHA256 hash of JSON-serialized content

    Examples:
        >>> compute_content_hash({"a": 1, "b": 2})
        '7a38bf81f383f69433ad6e900d35b3e2385593f76a7b7ab5d4355b8ba41ee24b'
    """
    # Create a copy to avoid modifying original
    to_hash = dict(content)

    # Remove excluded keys
    if exclude_keys:
        for key in exclude_keys:
            to_hash.pop(key, None)

    # Sort keys for deterministic output
    json_str = json.dumps(to_hash, sort_keys=True, separators=(",", ":"))

    return compute_sha256(json_str, with_prefix=with_prefix)


def verify_digest(
    data: Union[bytes, str, BinaryIO],
    expected_digest: str,
) -> bool:
    """
    Verify data matches expected digest.

    Args:
        data: Data to verify
        expected_digest: Expected SHA256 digest (with or without prefix)

    Returns:
        True if digest matches, False otherwise
    """
    # Handle prefix
    if expected_digest.startswith("sha256:"):
        expected = expected_digest[7:]  # Remove prefix
    else:
        expected = expected_digest

    computed = compute_sha256(data, as_hex=True, with_prefix=False)
    return computed.lower() == expected.lower()


def verify_file_digest(
    file_path: Union[str, Path],
    expected_digest: str,
) -> bool:
    """
    Verify file matches expected digest.

    Args:
        file_path: Path to file
        expected_digest: Expected SHA256 digest

    Returns:
        True if digest matches, False otherwise
    """
    try:
        computed = compute_file_hash(file_path)
        expected = expected_digest.replace("sha256:", "")
        return computed.lower() == expected.lower()
    except FileNotFoundError:
        return False


class IncrementalHasher:
    """
    Incremental hasher for streaming data.

    Useful for hashing large files or streams.
    """

    def __init__(self):
        """Initialize incremental hasher."""
        self._hasher = hashlib.sha256()
        self._bytes_processed = 0

    def update(self, data: Union[bytes, str]) -> None:
        """
        Add data to hash computation.

        Args:
            data: Data to add
        """
        if isinstance(data, str):
            data = data.encode("utf-8")
        self._hasher.update(data)
        self._bytes_processed += len(data)

    def digest(self, with_prefix: bool = False) -> str:
        """
        Get current hash value.

        Args:
            with_prefix: Include 'sha256:' prefix

        Returns:
            Current SHA256 hash
        """
        hex_digest = self._hasher.hexdigest()
        if with_prefix:
            return f"sha256:{hex_digest}"
        return hex_digest

    @property
    def bytes_processed(self) -> int:
        """Get number of bytes processed."""
        return self._bytes_processed

    def reset(self) -> None:
        """Reset hasher state."""
        self._hasher = hashlib.sha256()
        self._bytes_processed = 0
