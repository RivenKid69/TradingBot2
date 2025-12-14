# -*- coding: utf-8 -*-
"""
Password Hasher - WI-AUTH-01.

CLOUD ZONE ONLY.

Implements Argon2id password hashing per OWASP recommendations.
Argon2id is the recommended algorithm for password hashing:
- Memory-hard (resistant to GPU attacks)
- Time-hard (configurable iterations)
- Resistant to side-channel attacks

References:
- OWASP Password Storage Cheat Sheet
- NIST 800-63B
- RFC 9106 (Argon2)
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Final, Optional, Tuple

# Try to import argon2, fallback to bcrypt, then to PBKDF2
_ARGON2_AVAILABLE = False
_BCRYPT_AVAILABLE = False

try:
    import argon2
    from argon2 import PasswordHasher as Argon2Hasher
    from argon2.exceptions import (
        HashingError,
        InvalidHashError,
        VerificationError,
        VerifyMismatchError,
    )
    _ARGON2_AVAILABLE = True
except ImportError:
    pass

try:
    import bcrypt
    _BCRYPT_AVAILABLE = True
except ImportError:
    pass


# ============================================================================
# Constants
# ============================================================================

# Argon2id parameters (OWASP recommended)
# These provide ~1 second hash time on modern hardware
ARGON2_TIME_COST: Final[int] = 3  # iterations
ARGON2_MEMORY_COST: Final[int] = 65536  # 64 MB
ARGON2_PARALLELISM: Final[int] = 4  # threads
ARGON2_HASH_LENGTH: Final[int] = 32  # bytes
ARGON2_SALT_LENGTH: Final[int] = 16  # bytes

# Bcrypt parameters (fallback)
BCRYPT_ROUNDS: Final[int] = 12  # ~250ms on modern hardware

# PBKDF2 parameters (last resort fallback)
PBKDF2_ITERATIONS: Final[int] = 600000  # OWASP 2023 recommendation
PBKDF2_HASH_ALGORITHM: Final[str] = "sha256"
PBKDF2_HASH_LENGTH: Final[int] = 32  # bytes
PBKDF2_SALT_LENGTH: Final[int] = 16  # bytes


class HashAlgorithm(str, Enum):
    """Supported hash algorithms."""
    ARGON2ID = "argon2id"
    BCRYPT = "bcrypt"
    PBKDF2 = "pbkdf2"


# ============================================================================
# Hash Result
# ============================================================================

@dataclass
class HashResult:
    """Result of password hashing."""
    hash: str
    algorithm: HashAlgorithm
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hash": self.hash,
            "algorithm": self.algorithm.value,
            "parameters": self.parameters,
        }


# ============================================================================
# Password Hasher
# ============================================================================

class PasswordHasher:
    """
    Production-grade password hasher using Argon2id.

    Falls back to bcrypt, then PBKDF2 if Argon2 is not available.

    Usage:
        hasher = PasswordHasher()
        hash = hasher.hash("password123")
        is_valid = hasher.verify("password123", hash)
    """

    def __init__(
        self,
        *,
        algorithm: Optional[HashAlgorithm] = None,
        time_cost: int = ARGON2_TIME_COST,
        memory_cost: int = ARGON2_MEMORY_COST,
        parallelism: int = ARGON2_PARALLELISM,
        bcrypt_rounds: int = BCRYPT_ROUNDS,
        pbkdf2_iterations: int = PBKDF2_ITERATIONS,
    ):
        """
        Initialize hasher.

        Args:
            algorithm: Force specific algorithm (auto-detect if None)
            time_cost: Argon2 time cost (iterations)
            memory_cost: Argon2 memory cost (KB)
            parallelism: Argon2 parallelism (threads)
            bcrypt_rounds: bcrypt cost factor
            pbkdf2_iterations: PBKDF2 iterations
        """
        self._algorithm = algorithm or self._detect_algorithm()
        self._time_cost = time_cost
        self._memory_cost = memory_cost
        self._parallelism = parallelism
        self._bcrypt_rounds = bcrypt_rounds
        self._pbkdf2_iterations = pbkdf2_iterations

        # Initialize Argon2 hasher if available
        if self._algorithm == HashAlgorithm.ARGON2ID and _ARGON2_AVAILABLE:
            self._argon2_hasher = Argon2Hasher(
                time_cost=time_cost,
                memory_cost=memory_cost,
                parallelism=parallelism,
                hash_len=ARGON2_HASH_LENGTH,
                salt_len=ARGON2_SALT_LENGTH,
                type=argon2.Type.ID,  # Argon2id
            )
        else:
            self._argon2_hasher = None

    def _detect_algorithm(self) -> HashAlgorithm:
        """Detect best available algorithm."""
        if _ARGON2_AVAILABLE:
            return HashAlgorithm.ARGON2ID
        elif _BCRYPT_AVAILABLE:
            return HashAlgorithm.BCRYPT
        else:
            return HashAlgorithm.PBKDF2

    def hash(self, password: str) -> str:
        """
        Hash a password.

        Args:
            password: Plain text password

        Returns:
            Hash string (format depends on algorithm)
        """
        if self._algorithm == HashAlgorithm.ARGON2ID:
            return self._hash_argon2(password)
        elif self._algorithm == HashAlgorithm.BCRYPT:
            return self._hash_bcrypt(password)
        else:
            return self._hash_pbkdf2(password)

    def verify(self, password: str, hash: str) -> bool:
        """
        Verify a password against a hash.

        Args:
            password: Plain text password
            hash: Hash to verify against

        Returns:
            True if password matches hash
        """
        # Detect algorithm from hash format
        if hash.startswith("$argon2"):
            return self._verify_argon2(password, hash)
        elif hash.startswith("$2"):
            return self._verify_bcrypt(password, hash)
        elif hash.startswith("pbkdf2:"):
            return self._verify_pbkdf2(password, hash)
        else:
            # Legacy SHA256 hash (for migration)
            return self._verify_legacy_sha256(password, hash)

    def needs_rehash(self, hash: str) -> bool:
        """
        Check if hash needs to be rehashed (algorithm upgrade).

        Args:
            hash: Hash to check

        Returns:
            True if hash should be rehashed
        """
        # Always rehash legacy SHA256
        if not hash.startswith(("$argon2", "$2", "pbkdf2:")):
            return True

        # Rehash if using weaker algorithm than current
        if self._algorithm == HashAlgorithm.ARGON2ID:
            if not hash.startswith("$argon2"):
                return True
            # Check if Argon2 parameters need upgrade
            if _ARGON2_AVAILABLE and self._argon2_hasher:
                try:
                    return self._argon2_hasher.check_needs_rehash(hash)
                except (InvalidHashError, Exception):
                    return True

        return False

    # ========================================================================
    # Argon2id Implementation
    # ========================================================================

    def _hash_argon2(self, password: str) -> str:
        """Hash using Argon2id."""
        if not _ARGON2_AVAILABLE or not self._argon2_hasher:
            raise RuntimeError("Argon2 not available")
        return self._argon2_hasher.hash(password)

    def _verify_argon2(self, password: str, hash: str) -> bool:
        """Verify Argon2id hash."""
        if not _ARGON2_AVAILABLE:
            raise RuntimeError("Argon2 not available for verification")

        hasher = self._argon2_hasher or Argon2Hasher()
        try:
            hasher.verify(hash, password)
            return True
        except VerifyMismatchError:
            return False
        except (InvalidHashError, VerificationError):
            return False

    # ========================================================================
    # Bcrypt Implementation
    # ========================================================================

    def _hash_bcrypt(self, password: str) -> str:
        """Hash using bcrypt."""
        if not _BCRYPT_AVAILABLE:
            raise RuntimeError("bcrypt not available")
        salt = bcrypt.gensalt(rounds=self._bcrypt_rounds)
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    def _verify_bcrypt(self, password: str, hash: str) -> bool:
        """Verify bcrypt hash."""
        if not _BCRYPT_AVAILABLE:
            raise RuntimeError("bcrypt not available for verification")
        try:
            return bcrypt.checkpw(password.encode("utf-8"), hash.encode("utf-8"))
        except (ValueError, Exception):
            return False

    # ========================================================================
    # PBKDF2 Implementation (Fallback)
    # ========================================================================

    def _hash_pbkdf2(self, password: str) -> str:
        """Hash using PBKDF2-SHA256."""
        salt = os.urandom(PBKDF2_SALT_LENGTH)
        hash_bytes = hashlib.pbkdf2_hmac(
            PBKDF2_HASH_ALGORITHM,
            password.encode("utf-8"),
            salt,
            self._pbkdf2_iterations,
            dklen=PBKDF2_HASH_LENGTH,
        )
        # Format: pbkdf2:sha256:iterations$salt$hash
        salt_hex = salt.hex()
        hash_hex = hash_bytes.hex()
        return f"pbkdf2:{PBKDF2_HASH_ALGORITHM}:{self._pbkdf2_iterations}${salt_hex}${hash_hex}"

    def _verify_pbkdf2(self, password: str, hash: str) -> bool:
        """Verify PBKDF2 hash."""
        try:
            # Parse format: pbkdf2:sha256:iterations$salt$hash
            parts = hash.split("$")
            if len(parts) != 3:
                return False

            header, salt_hex, hash_hex = parts
            header_parts = header.split(":")
            if len(header_parts) != 3 or header_parts[0] != "pbkdf2":
                return False

            algorithm = header_parts[1]
            iterations = int(header_parts[2])

            salt = bytes.fromhex(salt_hex)
            expected_hash = bytes.fromhex(hash_hex)

            computed_hash = hashlib.pbkdf2_hmac(
                algorithm,
                password.encode("utf-8"),
                salt,
                iterations,
                dklen=len(expected_hash),
            )

            return hmac.compare_digest(computed_hash, expected_hash)
        except (ValueError, Exception):
            return False

    # ========================================================================
    # Legacy SHA256 (for migration only)
    # ========================================================================

    def _verify_legacy_sha256(self, password: str, hash: str) -> bool:
        """
        Verify legacy SHA256 hash (for migration).

        WARNING: SHA256 is NOT suitable for password hashing.
        This is only for migrating existing passwords.
        """
        try:
            computed = hashlib.sha256(password.encode()).hexdigest()
            return hmac.compare_digest(computed, hash)
        except (ValueError, Exception):
            return False

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def algorithm(self) -> HashAlgorithm:
        """Get current algorithm."""
        return self._algorithm

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        if self._algorithm == HashAlgorithm.ARGON2ID:
            return {
                "time_cost": self._time_cost,
                "memory_cost": self._memory_cost,
                "parallelism": self._parallelism,
            }
        elif self._algorithm == HashAlgorithm.BCRYPT:
            return {"rounds": self._bcrypt_rounds}
        else:
            return {"iterations": self._pbkdf2_iterations}


# ============================================================================
# Convenience Functions
# ============================================================================

# Singleton hasher instance
_default_hasher: Optional[PasswordHasher] = None


def get_hasher() -> PasswordHasher:
    """Get the default hasher instance."""
    global _default_hasher
    if _default_hasher is None:
        _default_hasher = PasswordHasher()
    return _default_hasher


def hash_password(password: str) -> str:
    """
    Hash a password using the default hasher.

    Args:
        password: Plain text password

    Returns:
        Hash string
    """
    return get_hasher().hash(password)


def verify_password(password: str, hash: str) -> bool:
    """
    Verify a password against a hash.

    Args:
        password: Plain text password
        hash: Hash to verify against

    Returns:
        True if password matches hash
    """
    return get_hasher().verify(password, hash)


def needs_rehash(hash: str) -> bool:
    """
    Check if hash needs to be rehashed.

    Args:
        hash: Hash to check

    Returns:
        True if hash should be rehashed
    """
    return get_hasher().needs_rehash(hash)
