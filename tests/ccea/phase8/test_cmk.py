# -*- coding: utf-8 -*-
"""
Tests for CustomerManagedKeysService.

CCEA Phase 8 - Enterprise CMK (Customer Managed Keys) tests.
"""

import pytest
import os
from datetime import datetime, timedelta

# Skip entire module if cryptography is not available
try:
    from packages.cloud.governance.cmk import (
        CustomerManagedKeysService,
        CMKConfig,
        KeyInfo,
        KeyType,
        KeyStatus,
        KeyPurpose,
        EncryptionResult,
        DecryptionResult,
        DEFAULT_KEY_ROTATION_DAYS,
        MAX_KEY_AGE_DAYS,
    )
except ImportError:
    pytestmark = pytest.mark.skip(reason="cryptography not available")


@pytest.fixture
def cmk_service():
    """Create CMK service."""
    return CustomerManagedKeysService()


@pytest.fixture
def sample_key():
    """Generate sample 32-byte key."""
    return os.urandom(32)


class TestCMKServiceBasic:
    """Basic CMK service tests."""

    def test_create_service(self):
        """Test creating CMK service."""
        service = CustomerManagedKeysService()
        assert service is not None
        assert service.config is not None

    def test_create_with_config(self):
        """Test creating with custom config."""
        config = CMKConfig(
            default_rotation_days=60,
            auto_rotation=True,
        )
        service = CustomerManagedKeysService(config)

        assert service.config.default_rotation_days == 60


class TestKeyRegistration:
    """Key registration tests."""

    def test_register_key(self, cmk_service, sample_key):
        """Test registering a key."""
        key_info = cmk_service.register_key(
            workspace_id="ws-123",
            name="primary-key",
            key_material=sample_key,
            purpose=KeyPurpose.DATA_ENCRYPTION,
        )

        assert key_info.id is not None
        assert key_info.workspace_id == "ws-123"
        assert key_info.name == "primary-key"
        assert key_info.purpose == KeyPurpose.DATA_ENCRYPTION
        assert key_info.status == KeyStatus.ACTIVE

    def test_key_hash_generated(self, cmk_service, sample_key):
        """Test key hash is generated on registration."""
        key_info = cmk_service.register_key(
            workspace_id="ws-123",
            name="test-key",
            key_material=sample_key,
        )

        assert key_info.key_hash is not None
        assert len(key_info.key_hash) == 64  # SHA-256 hex

    def test_key_expiration_set(self, cmk_service, sample_key):
        """Test key expiration is set."""
        key_info = cmk_service.register_key(
            workspace_id="ws-123",
            name="test-key",
            key_material=sample_key,
        )

        expected_expiry = datetime.utcnow() + timedelta(days=MAX_KEY_AGE_DAYS)
        delta = abs((key_info.expires_at - expected_expiry).total_seconds())
        assert delta < 60  # Within a minute

    def test_custom_rotation_days(self, cmk_service, sample_key):
        """Test custom rotation days."""
        key_info = cmk_service.register_key(
            workspace_id="ws-123",
            name="test-key",
            key_material=sample_key,
            rotation_days=30,
        )

        assert key_info.rotation_policy_days == 30


class TestKeyManagement:
    """Key management tests."""

    def test_get_key_info(self, cmk_service, sample_key):
        """Test getting key info."""
        created = cmk_service.register_key(
            workspace_id="ws-123",
            name="test-key",
            key_material=sample_key,
        )

        retrieved = cmk_service.get_key_info("ws-123", created.id)

        assert retrieved is not None
        assert retrieved.id == created.id

    def test_list_keys(self, cmk_service, sample_key):
        """Test listing keys for workspace."""
        cmk_service.register_key("ws-123", "key-1", sample_key)
        cmk_service.register_key("ws-123", "key-2", os.urandom(32))

        keys = cmk_service.list_keys("ws-123")

        assert len(keys) == 2

    def test_get_active_key(self, cmk_service, sample_key):
        """Test getting active key for purpose."""
        cmk_service.register_key(
            workspace_id="ws-123",
            name="data-key",
            key_material=sample_key,
            purpose=KeyPurpose.DATA_ENCRYPTION,
        )

        active = cmk_service.get_active_key("ws-123", KeyPurpose.DATA_ENCRYPTION)

        assert active is not None
        assert active.purpose == KeyPurpose.DATA_ENCRYPTION
        assert active.status == KeyStatus.ACTIVE


class TestEncryption:
    """Encryption tests."""

    def test_encrypt_data(self, cmk_service, sample_key):
        """Test encrypting data."""
        cmk_service.register_key("ws-123", "key", sample_key)

        result = cmk_service.encrypt(
            workspace_id="ws-123",
            plaintext=b"Hello, World!",
        )

        assert result.success is True
        assert result.ciphertext is not None
        assert result.encrypted_dek is not None
        assert result.key_id is not None

    def test_encrypt_updates_usage(self, cmk_service, sample_key):
        """Test encryption updates usage count."""
        key_info = cmk_service.register_key("ws-123", "key", sample_key)

        cmk_service.encrypt("ws-123", b"data")
        cmk_service.encrypt("ws-123", b"more data")

        updated = cmk_service.get_key_info("ws-123", key_info.id)
        assert updated.usage_count >= 2

    def test_encrypt_no_key_fails(self, cmk_service):
        """Test encryption fails without key."""
        result = cmk_service.encrypt("ws-no-key", b"data")

        assert result.success is False
        assert "No active key" in result.error


class TestDecryption:
    """Decryption tests."""

    def test_decrypt_data(self, cmk_service, sample_key):
        """Test decrypting data."""
        cmk_service.register_key("ws-123", "key", sample_key)

        plaintext = b"Secret message here!"

        # Encrypt
        encrypted = cmk_service.encrypt("ws-123", plaintext)

        # Decrypt
        decrypted = cmk_service.decrypt(
            workspace_id="ws-123",
            ciphertext=encrypted.ciphertext,
            encrypted_dek=encrypted.encrypted_dek,
            key_id=encrypted.key_id,
        )

        assert decrypted.success is True
        assert decrypted.plaintext == plaintext

    def test_decrypt_with_wrong_key_fails(self, cmk_service):
        """Test decryption with wrong key fails."""
        key1 = os.urandom(32)
        key2 = os.urandom(32)

        cmk_service.register_key("ws-1", "key1", key1)
        key2_info = cmk_service.register_key("ws-2", "key2", key2)

        encrypted = cmk_service.encrypt("ws-1", b"data")

        # Try to decrypt with different key
        result = cmk_service.decrypt(
            workspace_id="ws-2",
            ciphertext=encrypted.ciphertext,
            encrypted_dek=encrypted.encrypted_dek,
            key_id=key2_info.id,
        )

        assert result.success is False


class TestKeyRotation:
    """Key rotation tests."""

    def test_rotate_key(self, cmk_service, sample_key):
        """Test rotating a key."""
        key_info = cmk_service.register_key("ws-123", "key", sample_key)
        original_version = key_info.key_version
        original_hash = key_info.key_hash

        new_key = os.urandom(32)
        rotated = cmk_service.rotate_key(
            workspace_id="ws-123",
            key_id=key_info.id,
            new_key_material=new_key,
            rotated_by="admin",
        )

        assert rotated is not None
        assert rotated.key_version == original_version + 1
        assert rotated.key_hash != original_hash
        assert rotated.rotated_at is not None

    def test_needs_rotation(self, cmk_service, sample_key):
        """Test needs_rotation detection."""
        key_info = cmk_service.register_key(
            workspace_id="ws-123",
            name="key",
            key_material=sample_key,
            rotation_days=30,
        )

        # Newly created key should not need rotation
        assert key_info.needs_rotation is False

        # Simulate old key
        key_info.rotated_at = None
        key_info.created_at = datetime.utcnow() - timedelta(days=35)

        assert key_info.needs_rotation is True

    def test_get_keys_needing_rotation(self, cmk_service, sample_key):
        """Test getting keys needing rotation."""
        key_info = cmk_service.register_key(
            workspace_id="ws-123",
            name="old-key",
            key_material=sample_key,
            rotation_days=30,
        )

        # Simulate old key
        key_info.created_at = datetime.utcnow() - timedelta(days=35)

        keys = cmk_service.get_keys_needing_rotation()

        assert any(k.id == key_info.id for k in keys)

    def test_data_decryptable_after_rotation(self, cmk_service, sample_key):
        """Test data encrypted after rotation uses new key."""
        # Note: In this implementation, key rotation replaces the key material.
        # Data encrypted with the old key would need the old key to decrypt.
        # A production system would need key versioning to retain old keys for decryption.
        # This test verifies new encryption works after rotation.
        key_info = cmk_service.register_key("ws-123", "key", sample_key)

        # Rotate key
        new_key = os.urandom(32)
        cmk_service.rotate_key("ws-123", key_info.id, new_key)

        # Encrypt with new rotated key
        encrypted = cmk_service.encrypt("ws-123", b"new data after rotation")

        # Decrypt should work with the new key
        decrypted = cmk_service.decrypt(
            workspace_id="ws-123",
            ciphertext=encrypted.ciphertext,
            encrypted_dek=encrypted.encrypted_dek,
            key_id=encrypted.key_id,
        )

        assert decrypted.success is True
        assert decrypted.plaintext == b"new data after rotation"


class TestKeyDisabling:
    """Key disabling tests."""

    def test_disable_key(self, cmk_service, sample_key):
        """Test disabling a key."""
        key_info = cmk_service.register_key("ws-123", "key", sample_key)

        result = cmk_service.disable_key("ws-123", key_info.id, "admin")

        assert result is True

        updated = cmk_service.get_key_info("ws-123", key_info.id)
        assert updated.status == KeyStatus.DISABLED

    def test_encrypt_with_disabled_key_fails(self, cmk_service, sample_key):
        """Test encryption fails with disabled key."""
        key_info = cmk_service.register_key("ws-123", "key", sample_key)
        cmk_service.disable_key("ws-123", key_info.id)

        result = cmk_service.encrypt(
            workspace_id="ws-123",
            plaintext=b"data",
            key_id=key_info.id,
        )

        assert result.success is False
        assert "not active" in result.error.lower()


class TestKeyRevocation:
    """Key revocation tests."""

    def test_revoke_key(self, cmk_service, sample_key):
        """Test revoking a key."""
        key_info = cmk_service.register_key("ws-123", "key", sample_key)

        result = cmk_service.revoke_key(
            workspace_id="ws-123",
            key_id=key_info.id,
            revoked_by="admin",
            reason="Compromised",
        )

        assert result is True

        updated = cmk_service.get_key_info("ws-123", key_info.id)
        assert updated.status == KeyStatus.REVOKED

    def test_decrypt_with_revoked_key_fails(self, cmk_service, sample_key):
        """Test decryption fails with revoked key."""
        key_info = cmk_service.register_key("ws-123", "key", sample_key)
        encrypted = cmk_service.encrypt("ws-123", b"data")

        cmk_service.revoke_key("ws-123", key_info.id)

        result = cmk_service.decrypt(
            workspace_id="ws-123",
            ciphertext=encrypted.ciphertext,
            encrypted_dek=encrypted.encrypted_dek,
            key_id=encrypted.key_id,
        )

        assert result.success is False
        assert "revoked" in result.error.lower()


class TestKeyExpiration:
    """Key expiration tests."""

    def test_is_expired(self, cmk_service, sample_key):
        """Test is_expired detection."""
        key_info = cmk_service.register_key("ws-123", "key", sample_key)

        # New key should not be expired
        assert key_info.is_expired is False

        # Simulate expired key
        key_info.expires_at = datetime.utcnow() - timedelta(days=1)
        assert key_info.is_expired is True

    def test_encrypt_with_expired_key_fails(self, cmk_service, sample_key):
        """Test encryption fails with expired key."""
        key_info = cmk_service.register_key("ws-123", "key", sample_key)
        key_info.expires_at = datetime.utcnow() - timedelta(days=1)

        result = cmk_service.encrypt(
            workspace_id="ws-123",
            plaintext=b"data",
            key_id=key_info.id,
        )

        assert result.success is False
        assert "expired" in result.error.lower()


class TestAuditLog:
    """Audit log tests."""

    def test_audit_log_on_register(self, cmk_service, sample_key):
        """Test audit log on key registration."""
        key_info = cmk_service.register_key("ws-123", "key", sample_key)

        log = cmk_service.get_audit_log(key_id=key_info.id)

        assert len(log) > 0
        assert log[0]["operation"] == "key_registered"

    def test_audit_log_on_encrypt(self, cmk_service, sample_key):
        """Test audit log on encryption."""
        key_info = cmk_service.register_key("ws-123", "key", sample_key)
        cmk_service.encrypt("ws-123", b"data")

        log = cmk_service.get_audit_log(key_id=key_info.id)
        operations = [e["operation"] for e in log]

        assert "encrypt" in operations

    def test_audit_log_filtering(self, cmk_service, sample_key):
        """Test audit log filtering."""
        cmk_service.register_key("ws-1", "key1", sample_key)
        cmk_service.register_key("ws-2", "key2", os.urandom(32))

        log = cmk_service.get_audit_log(workspace_id="ws-1")

        assert all(e["workspace_id"] == "ws-1" for e in log)


class TestKeyInfoSerialization:
    """Key info serialization tests."""

    def test_to_dict_no_key_material(self, sample_key):
        """Test serialization doesn't include key material."""
        key_info = KeyInfo(
            workspace_id="ws-123",
            name="test-key",
            key_hash="abc123" * 10,  # Full hash
        )

        data = key_info.to_dict()

        assert data["name"] == "test-key"
        # Hash should be truncated
        assert len(data["key_hash"]) < len(key_info.key_hash)
        assert "..." in data["key_hash"]


class TestEncryptionResult:
    """Encryption result tests."""

    def test_to_envelope(self, cmk_service, sample_key):
        """Test converting result to envelope format."""
        cmk_service.register_key("ws-123", "key", sample_key)
        result = cmk_service.encrypt("ws-123", b"data")

        envelope = result.to_envelope()

        assert "ciphertext" in envelope
        assert "encrypted_dek" in envelope
        assert "key_id" in envelope
        assert "key_version" in envelope

        # Base64 encoded
        import base64

        decoded = base64.b64decode(envelope["ciphertext"])
        assert decoded == result.ciphertext


class TestKeyCallback:
    """Key operation callback tests."""

    def test_callback_on_register(self, sample_key):
        """Test callback on key registration."""
        operations = []

        def on_op(op, ws, key):
            operations.append((op, ws, key.name))

        service = CustomerManagedKeysService(on_key_operation=on_op)
        service.register_key("ws-123", "my-key", sample_key)

        assert len(operations) >= 1
        assert operations[0] == ("register", "ws-123", "my-key")

    def test_callback_on_rotate(self, sample_key):
        """Test callback on key rotation."""
        operations = []

        def on_op(op, ws, key):
            operations.append(op)

        service = CustomerManagedKeysService(on_key_operation=on_op)
        key_info = service.register_key("ws-123", "key", sample_key)
        service.rotate_key("ws-123", key_info.id, os.urandom(32))

        assert "rotate" in operations


class TestConfigOptions:
    """Configuration options tests."""

    def test_disable_audit_log(self, sample_key):
        """Test disabling audit log."""
        config = CMKConfig(audit_all_operations=False)
        service = CustomerManagedKeysService(config)

        service.register_key("ws-123", "key", sample_key)

        log = service.get_audit_log()
        assert len(log) == 0
