# -*- coding: utf-8 -*-
"""
Tests for Keychain Manager.

Design Doc Phase 5: OS keychain integration for master key.
"""

import pytest
import base64
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from packages.agent.daemon.keychain import (
    KeychainManager,
    KeychainConfig,
    KeychainError,
    KeychainNotAvailableError,
    KeyNotFoundError,
    KEY_SIZE,
)


class TestKeychainConfig:
    """Tests for KeychainConfig."""

    def test_default_config(self):
        """Test default values."""
        config = KeychainConfig()

        assert config.service_name == "ccea-agent"
        assert config.account_name == "vault-master-key"
        assert config.use_keychain is True
        assert config.fallback_to_env is True
        assert config.fallback_to_file is True

    def test_custom_config(self):
        """Test custom values."""
        config = KeychainConfig(
            service_name="my-app",
            account_name="my-key",
            use_keychain=False,
        )

        assert config.service_name == "my-app"
        assert config.use_keychain is False


class TestKeychainManager:
    """Tests for KeychainManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create KeychainManager with file fallback."""
        config = KeychainConfig(
            use_keychain=False,  # Disable keychain for tests
            fallback_to_env=False,
            fallback_to_file=True,
            key_file_path=temp_dir / "vault.key",
        )
        return KeychainManager(config)

    def test_initial_state(self, manager):
        """Test initial state."""
        # Keychain availability depends on platform
        info = manager.get_key_info()
        assert "platform" in info
        assert "keychain_available" in info

    def test_generate_and_store_key(self, manager):
        """Test key generation and storage."""
        key = manager.get_master_key()

        assert len(key) == KEY_SIZE
        assert manager.config.key_file_path.exists()

    def test_retrieve_stored_key(self, manager):
        """Test retrieving stored key."""
        key1 = manager.get_master_key()
        key2 = manager.get_master_key()

        assert key1 == key2

    def test_store_and_retrieve(self, manager):
        """Test explicit store and retrieve."""
        test_key = b"x" * KEY_SIZE

        assert manager.store_master_key(test_key) is True

        retrieved = manager.get_master_key()
        assert retrieved == test_key

    def test_invalid_key_size(self, manager):
        """Test invalid key size rejection."""
        with pytest.raises(ValueError):
            manager.store_master_key(b"short_key")

    def test_delete_key(self, manager):
        """Test key deletion."""
        manager.get_master_key()  # Generate
        assert manager.config.key_file_path.exists()

        assert manager.delete_master_key() is True
        assert not manager.config.key_file_path.exists()

    def test_rotate_key(self, manager):
        """Test key rotation."""
        old_key = manager.get_master_key()
        new_key = manager.rotate_master_key()

        assert old_key != new_key
        assert len(new_key) == KEY_SIZE

        # New key should be retrievable
        retrieved = manager.get_master_key()
        assert retrieved == new_key

    def test_get_key_info(self, manager):
        """Test key info retrieval."""
        manager.get_master_key()  # Ensure key exists

        info = manager.get_key_info()

        assert "platform" in info
        assert "keychain_enabled" in info
        assert "key_file_exists" in info
        assert info["key_file_exists"] is True

    def test_env_fallback(self, temp_dir, monkeypatch):
        """Test environment variable fallback."""
        test_key = b"y" * KEY_SIZE
        key_b64 = base64.b64encode(test_key).decode()

        monkeypatch.setenv("CCEA_VAULT_KEY", key_b64)

        config = KeychainConfig(
            use_keychain=False,
            fallback_to_env=True,
            fallback_to_file=False,
            key_file_path=temp_dir / "vault.key",
            allow_key_generation=False,
        )
        manager = KeychainManager(config)

        key = manager.get_master_key()
        assert key == test_key

    def test_no_key_found(self, temp_dir):
        """Test error when no key found."""
        config = KeychainConfig(
            use_keychain=False,
            fallback_to_env=False,
            fallback_to_file=False,
            key_file_path=temp_dir / "vault.key",
            allow_key_generation=False,
        )
        manager = KeychainManager(config)

        with pytest.raises(KeyNotFoundError):
            manager.get_master_key()

    def test_key_file_permissions(self, manager):
        """Test key file has restrictive permissions."""
        import os
        import stat

        manager.get_master_key()

        # Check permissions (Unix only)
        if os.name != "nt":
            mode = os.stat(manager.config.key_file_path).st_mode
            # Should be 0o600 (owner read/write only)
            assert (mode & stat.S_IRWXG) == 0  # No group permissions
            assert (mode & stat.S_IRWXO) == 0  # No other permissions


class TestKeychainPlatformSpecific:
    """Platform-specific keychain tests."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager_with_keychain(self, temp_dir):
        """Create manager that tries to use keychain."""
        config = KeychainConfig(
            use_keychain=True,
            fallback_to_file=True,
            key_file_path=temp_dir / "vault.key",
        )
        return KeychainManager(config)

    def test_keychain_availability(self, manager_with_keychain):
        """Test keychain availability check."""
        available = manager_with_keychain.is_keychain_available

        # Result depends on platform
        assert isinstance(available, bool)

    @pytest.mark.skipif(True, reason="Requires actual keychain access")
    def test_keychain_store_retrieve(self, manager_with_keychain):
        """Test actual keychain operations (requires keychain)."""
        if not manager_with_keychain.is_keychain_available:
            pytest.skip("Keychain not available")

        test_key = b"z" * KEY_SIZE
        manager_with_keychain.store_master_key(test_key)

        retrieved = manager_with_keychain.get_master_key()
        assert retrieved == test_key

        manager_with_keychain.delete_master_key()
