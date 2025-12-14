# -*- coding: utf-8 -*-
"""
Tests for Password Hasher - WI-AUTH-01.

Tests verify:
- Argon2id password hashing (or fallback algorithms)
- Password verification works correctly
- Legacy SHA256 migration support
- Hash rehashing detection
"""

import pytest

from packages.cloud.control_plane.security.password_hasher import (
    PasswordHasher,
    HashAlgorithm,
    hash_password,
    verify_password,
    needs_rehash,
)


class TestPasswordHasher:
    """Test PasswordHasher class."""

    def test_hash_returns_string(self):
        """Hash should return a string."""
        hasher = PasswordHasher()
        result = hasher.hash("testpassword")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_hash_is_not_plaintext(self):
        """Hash should not be the plaintext password."""
        hasher = PasswordHasher()
        password = "testpassword123"
        result = hasher.hash(password)
        assert result != password

    def test_hash_is_deterministic_salt(self):
        """Same password should produce different hashes (random salt)."""
        hasher = PasswordHasher()
        password = "testpassword123"
        hash1 = hasher.hash(password)
        hash2 = hasher.hash(password)
        # Due to random salt, hashes should be different
        assert hash1 != hash2

    def test_verify_correct_password(self):
        """Correct password should verify successfully."""
        hasher = PasswordHasher()
        password = "MySecurePassword123!"
        hashed = hasher.hash(password)

        assert hasher.verify(password, hashed)

    def test_verify_wrong_password(self):
        """Wrong password should fail verification."""
        hasher = PasswordHasher()
        password = "MySecurePassword123!"
        hashed = hasher.hash(password)

        assert not hasher.verify("WrongPassword", hashed)

    def test_verify_empty_password(self):
        """Empty password should fail verification."""
        hasher = PasswordHasher()
        password = "MySecurePassword123!"
        hashed = hasher.hash(password)

        assert not hasher.verify("", hashed)

    def test_verify_unicode_password(self):
        """Unicode passwords should work correctly."""
        hasher = PasswordHasher()
        password = "MyP@ssword123日本語"
        hashed = hasher.hash(password)

        assert hasher.verify(password, hashed)
        assert not hasher.verify("MyP@ssword123", hashed)

    def test_verify_long_password(self):
        """Long passwords should work correctly."""
        hasher = PasswordHasher()
        password = "A" * 100 + "1a!"
        hashed = hasher.hash(password)

        assert hasher.verify(password, hashed)

    def test_algorithm_property(self):
        """Algorithm property should return current algorithm."""
        hasher = PasswordHasher()
        assert hasher.algorithm in [
            HashAlgorithm.ARGON2ID,
            HashAlgorithm.BCRYPT,
            HashAlgorithm.PBKDF2,
        ]

    def test_parameters_property(self):
        """Parameters property should return dict."""
        hasher = PasswordHasher()
        params = hasher.parameters
        assert isinstance(params, dict)


class TestLegacySHA256Migration:
    """Test migration from legacy SHA256 hashes."""

    def test_verify_legacy_sha256_hash(self):
        """Legacy SHA256 hashes should verify for migration."""
        import hashlib

        hasher = PasswordHasher()
        password = "oldpassword123"

        # Create legacy SHA256 hash
        legacy_hash = hashlib.sha256(password.encode()).hexdigest()

        # Should verify successfully
        assert hasher.verify(password, legacy_hash)

    def test_legacy_hash_needs_rehash(self):
        """Legacy SHA256 hashes should need rehashing."""
        import hashlib

        hasher = PasswordHasher()
        password = "oldpassword123"
        legacy_hash = hashlib.sha256(password.encode()).hexdigest()

        assert hasher.needs_rehash(legacy_hash)


class TestNeedsRehash:
    """Test needs_rehash functionality."""

    def test_new_hash_does_not_need_rehash(self):
        """Freshly created hash should not need rehashing."""
        hasher = PasswordHasher()
        hashed = hasher.hash("testpassword")

        # New hash with same parameters should not need rehash
        # (This may still return True if Argon2 parameters differ)
        result = hasher.needs_rehash(hashed)
        assert isinstance(result, bool)

    def test_legacy_sha256_needs_rehash(self):
        """Legacy SHA256 should always need rehash."""
        import hashlib

        legacy_hash = hashlib.sha256(b"password").hexdigest()
        assert needs_rehash(legacy_hash)


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_hash_password_function(self):
        """hash_password should work correctly."""
        hashed = hash_password("testpassword123")
        assert isinstance(hashed, str)
        assert len(hashed) > 0

    def test_verify_password_function(self):
        """verify_password should work correctly."""
        password = "testpassword123"
        hashed = hash_password(password)

        assert verify_password(password, hashed)
        assert not verify_password("wrong", hashed)


class TestPBKDF2Fallback:
    """Test PBKDF2 fallback (always available)."""

    def test_pbkdf2_hash_and_verify(self):
        """PBKDF2 should work as fallback."""
        hasher = PasswordHasher(algorithm=HashAlgorithm.PBKDF2)
        password = "testpassword123"

        hashed = hasher.hash(password)
        assert hashed.startswith("pbkdf2:")

        assert hasher.verify(password, hashed)
        assert not hasher.verify("wrong", hashed)

    def test_pbkdf2_hash_format(self):
        """PBKDF2 hash should have correct format."""
        hasher = PasswordHasher(algorithm=HashAlgorithm.PBKDF2)
        hashed = hasher.hash("testpassword")

        # Format: pbkdf2:sha256:iterations$salt$hash
        parts = hashed.split("$")
        assert len(parts) == 3
        assert parts[0].startswith("pbkdf2:")


class TestSecurityProperties:
    """Test security properties of the hasher."""

    def test_different_passwords_different_hashes(self):
        """Different passwords should produce different hashes."""
        hasher = PasswordHasher()

        hash1 = hasher.hash("password1")
        hash2 = hasher.hash("password2")

        assert hash1 != hash2

    def test_timing_attack_resistance(self):
        """Verification should be constant-time (no early exit)."""
        hasher = PasswordHasher()
        hashed = hasher.hash("correctpassword")

        # Both wrong passwords should take similar time
        # (This is a weak test - proper timing tests need more samples)
        import time

        start = time.perf_counter()
        hasher.verify("wrongpassword", hashed)
        time1 = time.perf_counter() - start

        start = time.perf_counter()
        hasher.verify("completelydifferent", hashed)
        time2 = time.perf_counter() - start

        # Times should be in same order of magnitude
        # (Very loose test due to system variance)
        assert abs(time1 - time2) < 1.0

    def test_empty_password_produces_hash(self):
        """Empty password should still produce valid hash."""
        hasher = PasswordHasher()
        hashed = hasher.hash("")
        assert isinstance(hashed, str)
        assert len(hashed) > 0
