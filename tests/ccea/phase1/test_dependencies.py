# -*- coding: utf-8 -*-
"""
WI-DEPS-01: Reproducible Dependencies Baseline Tests

Verifies that all required CCEA runtime dependencies are:
1. Installed and importable
2. At the correct minimum version
3. Functional for their intended purposes

Reference:
- CCEA_MASTER_REMEDIATION_PLAN.md - Phase 1 (P0)
- SLSA/SBOM basics
- OWASP supply-chain hygiene

Standards:
- cryptography: NIST-approved algorithms, FIPS-capable
- PyJWT: RFC 7519 compliant
- argon2-cffi: OWASP recommended, NIST SP 800-63B
"""

from __future__ import annotations

import importlib
import sys
from typing import Tuple

import pytest


# =============================================================================
# Dependency Import Tests
# =============================================================================


class TestCCEAAgentDependencies:
    """Test CCEA Agent dependencies (ccea-agent optional group)."""

    def test_cryptography_installed(self) -> None:
        """Verify cryptography package is installed."""
        try:
            import cryptography

            assert cryptography.__version__ is not None
        except ImportError:
            pytest.fail(
                "cryptography package not installed. "
                "Install with: pip install 'cryptography>=42.0.0'"
            )

    def test_cryptography_version(self) -> None:
        """Verify cryptography is at minimum required version (42.0.0)."""
        import cryptography

        version = cryptography.__version__
        major, minor, *_ = version.split(".")

        assert int(major) >= 42, (
            f"cryptography version {version} is below minimum 42.0.0. "
            "Upgrade with: pip install 'cryptography>=42.0.0'"
        )

    def test_cryptography_aesgcm_available(self) -> None:
        """Verify AES-GCM is available (required by local_vault.py)."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        # Test basic functionality
        key = AESGCM.generate_key(bit_length=256)
        assert len(key) == 32, "AES-256 key should be 32 bytes"

    def test_cryptography_ed25519_available(self) -> None:
        """Verify Ed25519 is available (required by keys.py)."""
        from cryptography.hazmat.primitives.asymmetric import ed25519

        # Test key generation
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        assert private_key is not None
        assert public_key is not None

    def test_cryptography_pbkdf2_available(self) -> None:
        """Verify PBKDF2 is available (required by local_vault.py)."""
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives import hashes

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"test_salt_12345",
            iterations=1000,
        )
        key = kdf.derive(b"password")
        assert len(key) == 32

    def test_cryptography_ecdsa_p256_available(self) -> None:
        """Verify ECDSA P-256 is available (required by keys.py)."""
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.backends import default_backend

        private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
        public_key = private_key.public_key()

        assert private_key is not None
        assert public_key is not None


class TestCCEACloudDependencies:
    """Test CCEA Cloud Control Plane dependencies (ccea-cloud optional group)."""

    def test_asyncpg_installed(self) -> None:
        """Verify asyncpg package is installed."""
        try:
            import asyncpg

            assert asyncpg.__version__ is not None
        except ImportError:
            pytest.fail(
                "asyncpg package not installed. " "Install with: pip install 'asyncpg>=0.29.0'"
            )

    def test_asyncpg_version(self) -> None:
        """Verify asyncpg is at minimum required version (0.29.0)."""
        import asyncpg

        version = asyncpg.__version__
        parts = version.split(".")
        major = int(parts[0])
        minor = int(parts[1])

        assert major > 0 or (major == 0 and minor >= 29), (
            f"asyncpg version {version} is below minimum 0.29.0. "
            "Upgrade with: pip install 'asyncpg>=0.29.0'"
        )

    def test_pyjwt_installed(self) -> None:
        """Verify PyJWT package is installed."""
        try:
            import jwt

            assert jwt.__version__ is not None
        except ImportError:
            pytest.fail("PyJWT package not installed. " "Install with: pip install 'PyJWT>=2.8.0'")

    def test_pyjwt_version(self) -> None:
        """Verify PyJWT is at minimum required version (2.8.0)."""
        import jwt

        version = jwt.__version__
        parts = version.split(".")
        major = int(parts[0])
        minor = int(parts[1])

        assert major > 2 or (major == 2 and minor >= 8), (
            f"PyJWT version {version} is below minimum 2.8.0. "
            "Upgrade with: pip install 'PyJWT>=2.8.0'"
        )

    def test_pyjwt_encode_decode(self) -> None:
        """Verify PyJWT can encode/decode tokens (RFC 7519 compliance)."""
        import jwt

        # Test basic encode/decode
        payload = {"sub": "test_user", "exp": 9999999999}
        secret = "test_secret"

        token = jwt.encode(payload, secret, algorithm="HS256")
        decoded = jwt.decode(token, secret, algorithms=["HS256"])

        assert decoded["sub"] == "test_user"

    def test_argon2_installed(self) -> None:
        """Verify argon2-cffi package is installed."""
        try:
            import argon2

            assert hasattr(argon2, "PasswordHasher")
        except ImportError:
            pytest.fail(
                "argon2-cffi package not installed. "
                "Install with: pip install 'argon2-cffi>=23.1.0'"
            )

    def test_argon2_version(self) -> None:
        """Verify argon2-cffi is at minimum required version (23.1.0)."""
        import argon2

        # argon2-cffi uses __version__ from argon2 module
        version = argon2.__version__
        parts = version.split(".")
        major = int(parts[0])
        minor = int(parts[1])

        assert major > 23 or (major == 23 and minor >= 1), (
            f"argon2-cffi version {version} is below minimum 23.1.0. "
            "Upgrade with: pip install 'argon2-cffi>=23.1.0'"
        )

    def test_argon2_password_hashing(self) -> None:
        """Verify Argon2 password hashing works (OWASP recommended)."""
        from argon2 import PasswordHasher

        ph = PasswordHasher()
        password = "secure_password_123"

        # Hash password
        hash_result = ph.hash(password)
        assert hash_result.startswith("$argon2")

        # Verify password
        assert ph.verify(hash_result, password) is True

        # Verify wrong password fails
        with pytest.raises(Exception):  # argon2.exceptions.VerifyMismatchError
            ph.verify(hash_result, "wrong_password")

    def test_aiosqlite_installed(self) -> None:
        """Verify aiosqlite is installed (dev/test database driver)."""
        try:
            import aiosqlite

            assert aiosqlite.__version__ is not None
        except ImportError:
            pytest.fail(
                "aiosqlite package not installed. " "Install with: pip install 'aiosqlite>=0.19.0'"
            )


class TestCCEADatabaseConfiguration:
    """Test database configuration works correctly."""

    def test_database_module_imports(self) -> None:
        """Verify database module can be imported."""
        try:
            from packages.cloud.control_plane import database

            assert hasattr(database, "create_engine")
            assert hasattr(database, "DATABASE_URL")
        except ImportError as e:
            pytest.fail(f"Failed to import database module: {e}")

    def test_database_default_is_sqlite(self) -> None:
        """Verify default database URL is SQLite for dev/test."""
        import os

        # Temporarily unset any existing database URL
        original = os.environ.get("CCEA_DATABASE_URL")
        if "CCEA_DATABASE_URL" in os.environ:
            del os.environ["CCEA_DATABASE_URL"]

        try:
            # Re-import to get fresh default
            import importlib
            from packages.cloud.control_plane import database

            importlib.reload(database)

            assert database._DEFAULT_DATABASE_URL.startswith(
                "sqlite+aiosqlite"
            ), "Default database URL should use SQLite for dev/test"
        finally:
            # Restore original value
            if original is not None:
                os.environ["CCEA_DATABASE_URL"] = original

    def test_database_validation_asyncpg(self) -> None:
        """Verify database validation works for asyncpg URLs."""
        from packages.cloud.control_plane.database import _validate_database_url

        # This should not raise (asyncpg is installed)
        _validate_database_url("postgresql+asyncpg://user:pass@localhost/db")

    def test_database_validation_aiosqlite(self) -> None:
        """Verify database validation works for aiosqlite URLs."""
        from packages.cloud.control_plane.database import _validate_database_url

        # This should not raise (aiosqlite is installed)
        _validate_database_url("sqlite+aiosqlite:///test.db")


class TestCCEACryptoModule:
    """Test CCEA crypto module with cryptography dependency."""

    def test_keys_module_imports(self) -> None:
        """Verify keys module can be imported."""
        try:
            from ccea.crypto import keys

            assert hasattr(keys, "generate_keypair")
            assert hasattr(keys, "KeyAlgorithm")
        except ImportError as e:
            pytest.fail(f"Failed to import keys module: {e}")

    def test_keys_generate_ed25519(self) -> None:
        """Verify Ed25519 key generation works."""
        from ccea.crypto.keys import generate_keypair, KeyAlgorithm

        keypair = generate_keypair(algorithm=KeyAlgorithm.ED25519)

        assert keypair is not None
        assert keypair.algorithm == KeyAlgorithm.ED25519
        assert keypair.private_key is not None
        assert keypair.public_key is not None

    def test_keys_generate_ecdsa_p256(self) -> None:
        """Verify ECDSA P-256 key generation works."""
        from ccea.crypto.keys import generate_keypair, KeyAlgorithm

        keypair = generate_keypair(algorithm=KeyAlgorithm.ECDSA_P256)

        assert keypair is not None
        assert keypair.algorithm == KeyAlgorithm.ECDSA_P256

    def test_keys_serialization(self) -> None:
        """Verify key serialization works."""
        from ccea.crypto.keys import (
            generate_keypair,
            serialize_public_key,
            serialize_private_key,
            load_public_key,
            load_private_key,
        )

        keypair = generate_keypair()

        # Serialize
        public_pem = serialize_public_key(keypair.public_key)
        private_pem = serialize_private_key(keypair.private_key)

        assert public_pem.startswith("-----BEGIN PUBLIC KEY-----")
        assert private_pem.startswith("-----BEGIN PRIVATE KEY-----")

        # Deserialize
        loaded_public = load_public_key(public_pem)
        loaded_private = load_private_key(private_pem)

        assert loaded_public is not None
        assert loaded_private is not None


class TestCCEAVaultModule:
    """Test CCEA vault module with cryptography dependency."""

    def test_vault_module_imports(self) -> None:
        """Verify local_vault module can be imported."""
        try:
            from packages.agent.vault import local_vault

            assert hasattr(local_vault, "LocalVault")
            assert hasattr(local_vault, "CRYPTO_AVAILABLE")
        except ImportError as e:
            pytest.fail(f"Failed to import local_vault module: {e}")

    def test_vault_crypto_available(self) -> None:
        """Verify cryptography is available for vault."""
        from packages.agent.vault.local_vault import CRYPTO_AVAILABLE

        assert CRYPTO_AVAILABLE is True, (
            "cryptography package required for vault operations. "
            "Install with: pip install cryptography"
        )

    def test_vault_initialization(self, tmp_path) -> None:
        """Verify vault can be initialized and used."""
        from packages.agent.vault.local_vault import LocalVault, VaultConfig

        config = VaultConfig(vault_path=tmp_path / "test_vault.enc")
        vault = LocalVault(config)

        # Initialize vault
        vault.initialize("master_password_123")

        assert vault.is_initialized is True
        assert vault.is_locked is False

    def test_vault_store_retrieve(self, tmp_path) -> None:
        """Verify vault can store and retrieve credentials."""
        from packages.agent.vault.local_vault import LocalVault, VaultConfig

        config = VaultConfig(vault_path=tmp_path / "test_vault.enc")
        vault = LocalVault(config)

        # Initialize and store
        vault.initialize("master_password_123")
        vault.store("binance", "api_key", "test_api_key_12345")
        vault.store("binance", "api_secret", "test_api_secret_67890")

        # Retrieve
        api_key = vault.retrieve("binance", "api_key")
        api_secret = vault.retrieve("binance", "api_secret")

        assert api_key == "test_api_key_12345"
        assert api_secret == "test_api_secret_67890"

    def test_vault_unlock(self, tmp_path) -> None:
        """Verify vault can be locked and unlocked."""
        from packages.agent.vault.local_vault import LocalVault, VaultConfig

        config = VaultConfig(vault_path=tmp_path / "test_vault.enc")
        vault = LocalVault(config)

        # Initialize and store
        vault.initialize("master_password_123")
        vault.store("test_broker", "api_key", "test_value")

        # Lock
        vault.lock()
        assert vault.is_locked is True

        # Unlock
        vault.unlock("master_password_123")
        assert vault.is_locked is False

        # Verify data persists
        value = vault.retrieve("test_broker", "api_key")
        assert value == "test_value"


class TestCCEAJWTDependencies:
    """Test JWT authentication dependencies."""

    def test_dependencies_module_imports(self) -> None:
        """Verify dependencies module can be imported."""
        try:
            from packages.cloud.control_plane import dependencies

            assert hasattr(dependencies, "create_access_token")
            assert hasattr(dependencies, "decode_token")
        except ImportError as e:
            pytest.fail(f"Failed to import dependencies module: {e}")

    def test_create_access_token(self) -> None:
        """Verify JWT token creation works."""
        from uuid import uuid4
        from packages.cloud.control_plane.dependencies import create_access_token

        user_id = uuid4()
        token = create_access_token(
            user_id=user_id,
            email="test@example.com",
            permissions=["read", "write"],
        )

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

        # Should be a valid JWT (3 parts separated by dots)
        parts = token.split(".")
        assert len(parts) == 3

    def test_decode_token(self) -> None:
        """Verify JWT token decoding works."""
        from uuid import uuid4
        from packages.cloud.control_plane.dependencies import (
            create_access_token,
            decode_token,
        )

        user_id = uuid4()
        email = "test@example.com"
        permissions = ["read", "write"]

        token = create_access_token(
            user_id=user_id,
            email=email,
            permissions=permissions,
        )

        decoded = decode_token(token)

        assert decoded["sub"] == str(user_id)
        assert decoded["email"] == email
        assert decoded["permissions"] == permissions


# =============================================================================
# Version Matrix Test
# =============================================================================


class TestDependencyVersionMatrix:
    """Test all dependencies are at correct versions per pyproject.toml."""

    REQUIRED_DEPS = {
        # CCEA Agent
        "cryptography": ("42.0.0", "44.0.0"),
        # CCEA Cloud
        "asyncpg": ("0.29.0", "1.0.0"),
        "PyJWT": ("2.8.0", "3.0.0"),
        "argon2-cffi": ("23.1.0", "24.0.0"),
        "aiosqlite": ("0.19.0", "1.0.0"),
    }

    def _parse_version(self, version_str: str) -> Tuple[int, ...]:
        """Parse version string to tuple of integers."""
        parts = version_str.split(".")
        result = []
        for part in parts:
            # Handle pre-release versions like "42.0.8"
            num_part = ""
            for char in part:
                if char.isdigit():
                    num_part += char
                else:
                    break
            if num_part:
                result.append(int(num_part))
        return tuple(result)

    def _version_in_range(
        self,
        version: str,
        min_version: str,
        max_version: str,
    ) -> bool:
        """Check if version is within range [min, max)."""
        v = self._parse_version(version)
        v_min = self._parse_version(min_version)
        v_max = self._parse_version(max_version)

        return v_min <= v < v_max

    @pytest.mark.parametrize(
        "package,version_range",
        [
            ("cryptography", ("42.0.0", "44.0.0")),
            ("asyncpg", ("0.29.0", "1.0.0")),
            ("PyJWT", ("2.8.0", "3.0.0")),
            ("argon2-cffi", ("23.1.0", "24.0.0")),
            ("aiosqlite", ("0.19.0", "1.0.0")),
        ],
    )
    def test_dependency_version_in_range(
        self,
        package: str,
        version_range: Tuple[str, str],
    ) -> None:
        """Verify each dependency is within required version range."""
        min_ver, max_ver = version_range

        # Handle special import names
        import_names = {
            "PyJWT": "jwt",
            "argon2-cffi": "argon2",
        }
        import_name = import_names.get(package, package)

        try:
            module = importlib.import_module(import_name)
            version = module.__version__
        except ImportError:
            pytest.fail(f"{package} is not installed")
        except AttributeError:
            pytest.skip(f"{package} does not expose __version__")

        assert self._version_in_range(
            version, min_ver, max_ver
        ), f"{package} version {version} is not in required range [{min_ver}, {max_ver})"


# =============================================================================
# Integration Test
# =============================================================================


class TestDependencyIntegration:
    """Integration tests ensuring dependencies work together."""

    def test_vault_with_jwt_workflow(self, tmp_path) -> None:
        """Test realistic workflow: create vault, generate token for access."""
        from uuid import uuid4
        from packages.agent.vault.local_vault import LocalVault, VaultConfig
        from packages.cloud.control_plane.dependencies import (
            create_access_token,
            decode_token,
        )

        # Setup vault
        config = VaultConfig(vault_path=tmp_path / "test_vault.enc")
        vault = LocalVault(config)
        vault.initialize("master_password_123")

        # Store API credentials
        vault.store("binance", "api_key", "binance_api_key_abc123")
        vault.store("binance", "api_secret", "binance_api_secret_xyz789")

        # Create JWT for cloud authentication
        user_id = uuid4()
        token = create_access_token(
            user_id=user_id,
            email="trader@example.com",
            permissions=["agent:read", "agent:write", "commands:execute"],
        )

        # Decode and verify
        decoded = decode_token(token)
        assert decoded["email"] == "trader@example.com"

        # Retrieve credentials from vault (agent-side operation)
        api_key = vault.retrieve("binance", "api_key")
        assert api_key == "binance_api_key_abc123"

    def test_argon2_with_jwt_auth_workflow(self) -> None:
        """Test auth workflow: hash password, create JWT."""
        from uuid import uuid4
        from argon2 import PasswordHasher
        from packages.cloud.control_plane.dependencies import create_access_token

        # Hash password (simulating user registration)
        ph = PasswordHasher()
        password = "secure_user_password"
        password_hash = ph.hash(password)

        # Verify password (simulating login)
        assert ph.verify(password_hash, password)

        # Create JWT after successful login
        user_id = uuid4()
        token = create_access_token(
            user_id=user_id,
            email="user@example.com",
            permissions=["user:profile"],
        )

        assert token is not None
        parts = token.split(".")
        assert len(parts) == 3


# =============================================================================
# CI Acceptance Tests
# =============================================================================


class TestCIAcceptance:
    """
    Acceptance tests for WI-DEPS-01.

    These tests verify the acceptance criteria from the remediation plan:
    - .github/workflows/build-and-test.yml passes dependency install
    - .github/workflows/security-sast.yml dependency-audit stays green
    """

    def test_all_ccea_imports_succeed(self) -> None:
        """Verify all CCEA modules can be imported without missing deps."""
        imports_to_test = [
            "ccea.crypto.keys",
            "packages.agent.vault.local_vault",
            "packages.cloud.control_plane.database",
            "packages.cloud.control_plane.dependencies",
        ]

        for module_path in imports_to_test:
            try:
                importlib.import_module(module_path)
            except ImportError as e:
                pytest.fail(f"Failed to import {module_path}: {e}")

    def test_no_dependency_deprecation_warnings(self) -> None:
        """Verify no deprecation warnings from dependencies."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)

            # Import all modules
            import cryptography
            import asyncpg
            import jwt
            import argon2
            import aiosqlite

            # Filter for dependency-related deprecations
            dep_warnings = [
                warning
                for warning in w
                if "deprecat" in str(warning.message).lower()
                and any(
                    pkg in warning.filename
                    for pkg in ["cryptography", "asyncpg", "jwt", "argon2", "aiosqlite"]
                )
            ]

            assert (
                len(dep_warnings) == 0
            ), f"Found deprecation warnings in dependencies: {dep_warnings}"
