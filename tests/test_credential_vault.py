"""
Tests for CredentialVault - AES-256-GCM encrypted credential storage.

Comprehensive tests covering:
    - Encryption/decryption roundtrip
    - User isolation (different users, different ciphertext)
    - Access logging
    - Tamper detection
    - Secure deletion
    - Key derivation
    - Error handling

References:
    - NIST SP 800-57 Part 1 Rev 5: Key Management
    - OWASP Cryptographic Storage Cheat Sheet
"""

import os
import secrets
import pytest
from datetime import datetime, timezone

from services.security.credential_vault import (
    CredentialVault,
    CredentialType,
    EncryptedCredential,
    CredentialVaultError,
    InvalidMasterKeyError,
    CredentialDecryptionError,
)


class TestCredentialVaultInitialization:
    """Tests for vault initialization and master key handling."""

    def test_valid_master_key_accepted(self):
        """Verify 32-byte master key is accepted."""
        master_key = secrets.token_bytes(32)
        vault = CredentialVault(master_key)
        assert vault is not None

    def test_invalid_master_key_length_rejected(self):
        """Verify keys of wrong length are rejected."""
        for length in [0, 16, 24, 31, 33, 64]:
            with pytest.raises(InvalidMasterKeyError):
                CredentialVault(secrets.token_bytes(length))

    def test_non_bytes_master_key_rejected(self):
        """Verify non-bytes master key is rejected."""
        with pytest.raises(InvalidMasterKeyError):
            CredentialVault("not_bytes_32_characters_long_x")  # type: ignore

    def test_generate_master_key(self):
        """Verify generate_master_key produces valid key."""
        key = CredentialVault.generate_master_key()
        assert isinstance(key, bytes)
        assert len(key) == 32
        # Should be able to use it
        vault = CredentialVault(key)
        assert vault is not None

    def test_custom_kdf_iterations(self):
        """Verify custom KDF iterations are accepted."""
        master_key = secrets.token_bytes(32)
        vault = CredentialVault(master_key, kdf_iterations=50_000)
        assert vault is not None


class TestEncryptionDecryption:
    """Tests for encrypt/decrypt functionality."""

    @pytest.fixture
    def vault(self):
        """Create a vault with random master key."""
        return CredentialVault(secrets.token_bytes(32))

    @pytest.fixture
    def sample_credential(self, vault):
        """Create an encrypted sample credential."""
        return vault.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="alpaca",
            plaintext="sk-live-abc123xyz789"
        )

    def test_encrypt_decrypt_roundtrip(self, vault):
        """Verify encryption followed by decryption returns original."""
        plaintext = "sk-live-abc123xyz"
        encrypted = vault.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="alpaca",
            plaintext=plaintext
        )
        decrypted = vault.decrypt(encrypted, purpose="test")
        assert decrypted == plaintext

    def test_encrypt_decrypt_roundtrip_all_credential_types(self, vault):
        """Verify all credential types can be encrypted/decrypted."""
        for cred_type in CredentialType:
            plaintext = f"secret_for_{cred_type.value}"
            encrypted = vault.encrypt(
                user_id="user_001",
                credential_type=cred_type,
                broker="test_broker",
                plaintext=plaintext
            )
            decrypted = vault.decrypt(encrypted, purpose="test")
            assert decrypted == plaintext

    def test_encrypted_credential_has_required_fields(self, sample_credential):
        """Verify encrypted credential has all required fields."""
        assert sample_credential.credential_id is not None
        assert sample_credential.user_id == "user_001"
        assert sample_credential.credential_type == CredentialType.BROKER_API_KEY
        assert sample_credential.broker == "alpaca"
        assert sample_credential.ciphertext is not None
        assert sample_credential.nonce is not None
        assert sample_credential.created_at is not None

    def test_nonce_is_unique_per_encryption(self, vault):
        """Verify each encryption uses a unique nonce."""
        nonces = set()
        for _ in range(100):
            encrypted = vault.encrypt(
                user_id="user_001",
                credential_type=CredentialType.BROKER_API_KEY,
                broker="alpaca",
                plaintext="same_key"
            )
            nonces.add(encrypted.nonce)
        # All nonces should be unique
        assert len(nonces) == 100

    def test_ciphertext_changes_each_encryption(self, vault):
        """Verify ciphertext differs even for same plaintext."""
        ciphertexts = set()
        for _ in range(10):
            encrypted = vault.encrypt(
                user_id="user_001",
                credential_type=CredentialType.BROKER_API_KEY,
                broker="alpaca",
                plaintext="same_key"
            )
            ciphertexts.add(encrypted.ciphertext)
        # All ciphertexts should be unique (due to random nonce)
        assert len(ciphertexts) == 10

    def test_special_characters_in_plaintext(self, vault):
        """Verify special characters are handled correctly."""
        special_texts = [
            "key-with-dashes",
            "key_with_underscores",
            "key with spaces",
            "key\nwith\nnewlines",
            "key\twith\ttabs",
            "key=with=equals",
            "key&with&ampersand",
            "日本語キー",  # Japanese
            "🔑🔐",  # Emoji
        ]
        for plaintext in special_texts:
            encrypted = vault.encrypt(
                user_id="user_001",
                credential_type=CredentialType.BROKER_API_KEY,
                broker="test",
                plaintext=plaintext
            )
            decrypted = vault.decrypt(encrypted, purpose="test")
            assert decrypted == plaintext

    def test_long_plaintext(self, vault):
        """Verify long plaintext is handled correctly."""
        plaintext = "x" * 10000
        encrypted = vault.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="test",
            plaintext=plaintext
        )
        decrypted = vault.decrypt(encrypted, purpose="test")
        assert decrypted == plaintext

    def test_empty_plaintext(self, vault):
        """Verify empty plaintext is handled correctly."""
        plaintext = ""
        encrypted = vault.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="test",
            plaintext=plaintext
        )
        decrypted = vault.decrypt(encrypted, purpose="test")
        assert decrypted == plaintext


class TestUserIsolation:
    """Tests for user isolation - each user's keys should be isolated."""

    @pytest.fixture
    def vault(self):
        """Create a vault with random master key."""
        return CredentialVault(secrets.token_bytes(32))

    def test_different_users_different_ciphertext(self, vault):
        """Verify same plaintext for different users produces different ciphertext."""
        plaintext = "same_key"
        enc1 = vault.encrypt("user_1", CredentialType.BROKER_API_KEY, "alpaca", plaintext)
        enc2 = vault.encrypt("user_2", CredentialType.BROKER_API_KEY, "alpaca", plaintext)
        assert enc1.ciphertext != enc2.ciphertext

    def test_user_cannot_decrypt_other_user_credential(self, vault):
        """Verify user cannot decrypt another user's credential."""
        encrypted = vault.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="alpaca",
            plaintext="user_001_key"
        )
        # Modify user_id (simulating cross-user access attempt)
        encrypted.user_id = "user_002"

        with pytest.raises(CredentialDecryptionError):
            vault.decrypt(encrypted, purpose="test")

    def test_credential_bound_to_broker(self, vault):
        """Verify credential is bound to specific broker."""
        encrypted = vault.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="alpaca",
            plaintext="alpaca_key"
        )
        # Modify broker (simulating cross-broker access attempt)
        encrypted.broker = "binance"

        with pytest.raises(CredentialDecryptionError):
            vault.decrypt(encrypted, purpose="test")

    def test_credential_bound_to_type(self, vault):
        """Verify credential is bound to credential type."""
        encrypted = vault.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="alpaca",
            plaintext="api_key"
        )
        # Modify type (simulating cross-type access attempt)
        encrypted.credential_type = CredentialType.BROKER_API_SECRET

        with pytest.raises(CredentialDecryptionError):
            vault.decrypt(encrypted, purpose="test")


class TestTamperDetection:
    """Tests for tamper detection (authenticated encryption)."""

    @pytest.fixture
    def vault(self):
        """Create a vault with random master key."""
        return CredentialVault(secrets.token_bytes(32))

    def test_tampered_ciphertext_detected(self, vault):
        """Verify tampering with ciphertext is detected."""
        encrypted = vault.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="alpaca",
            plaintext="original_key"
        )
        # Tamper with ciphertext
        tampered = bytearray(encrypted.ciphertext)
        tampered[0] ^= 0xFF  # Flip bits
        encrypted.ciphertext = bytes(tampered)

        with pytest.raises(CredentialDecryptionError):
            vault.decrypt(encrypted, purpose="test")

    def test_tampered_nonce_detected(self, vault):
        """Verify tampering with nonce is detected."""
        encrypted = vault.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="alpaca",
            plaintext="original_key"
        )
        # Tamper with nonce
        tampered = bytearray(encrypted.nonce)
        tampered[0] ^= 0xFF
        encrypted.nonce = bytes(tampered)

        with pytest.raises(CredentialDecryptionError):
            vault.decrypt(encrypted, purpose="test")

    def test_truncated_ciphertext_detected(self, vault):
        """Verify truncated ciphertext is detected."""
        encrypted = vault.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="alpaca",
            plaintext="original_key"
        )
        # Truncate ciphertext
        encrypted.ciphertext = encrypted.ciphertext[:10]

        with pytest.raises(CredentialDecryptionError):
            vault.decrypt(encrypted, purpose="test")


class TestAccessLogging:
    """Tests for access logging functionality."""

    @pytest.fixture
    def vault(self):
        """Create a vault with random master key."""
        return CredentialVault(secrets.token_bytes(32))

    def test_decrypt_logs_access(self, vault):
        """Verify decryption is logged."""
        encrypted = vault.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="alpaca",
            plaintext="test_key"
        )
        vault.decrypt(encrypted, purpose="order_execution")

        log = vault.get_access_log(user_id="user_001")
        assert len(log) == 1
        assert log[0]["purpose"] == "order_execution"
        assert log[0]["success"] is True

    def test_failed_decrypt_logs_access(self, vault):
        """Verify failed decryption is logged."""
        encrypted = vault.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="alpaca",
            plaintext="test_key"
        )
        encrypted.ciphertext = b"tampered"

        with pytest.raises(CredentialDecryptionError):
            vault.decrypt(encrypted, purpose="test")

        log = vault.get_access_log(user_id="user_001")
        assert len(log) == 1
        assert log[0]["success"] is False

    def test_access_count_increments(self, vault):
        """Verify access count increments on each decrypt."""
        encrypted = vault.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="alpaca",
            plaintext="test_key"
        )
        assert encrypted.access_count == 0

        vault.decrypt(encrypted, purpose="test1")
        assert encrypted.access_count == 1

        vault.decrypt(encrypted, purpose="test2")
        assert encrypted.access_count == 2

    def test_last_accessed_updated(self, vault):
        """Verify last_accessed timestamp is updated."""
        encrypted = vault.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="alpaca",
            plaintext="test_key"
        )
        assert encrypted.last_accessed is None

        vault.decrypt(encrypted, purpose="test")
        assert encrypted.last_accessed is not None

    def test_log_filter_by_user(self, vault):
        """Verify log can be filtered by user."""
        enc1 = vault.encrypt("user_001", CredentialType.BROKER_API_KEY, "alpaca", "key1")
        enc2 = vault.encrypt("user_002", CredentialType.BROKER_API_KEY, "alpaca", "key2")

        vault.decrypt(enc1, purpose="test")
        vault.decrypt(enc2, purpose="test")

        log_user1 = vault.get_access_log(user_id="user_001")
        assert len(log_user1) == 1
        assert log_user1[0]["user_id"] == "user_001"

    def test_clear_access_log(self, vault):
        """Verify access log can be cleared."""
        encrypted = vault.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="alpaca",
            plaintext="test_key"
        )
        vault.decrypt(encrypted, purpose="test")

        count = vault.clear_access_log()
        assert count == 1
        assert len(vault.get_access_log()) == 0


class TestSecureDeletion:
    """Tests for secure deletion functionality."""

    @pytest.fixture
    def vault(self):
        """Create a vault with random master key."""
        return CredentialVault(secrets.token_bytes(32))

    def test_delete_overwrites_ciphertext(self, vault):
        """Verify delete overwrites ciphertext with random data."""
        encrypted = vault.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="alpaca",
            plaintext="secret_key"
        )
        original_ciphertext = encrypted.ciphertext

        result = vault.delete(encrypted)
        assert result is True
        assert encrypted.ciphertext != original_ciphertext
        assert len(encrypted.ciphertext) == len(original_ciphertext)

    def test_delete_overwrites_nonce(self, vault):
        """Verify delete overwrites nonce with random data."""
        encrypted = vault.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="alpaca",
            plaintext="secret_key"
        )
        original_nonce = encrypted.nonce

        vault.delete(encrypted)
        assert encrypted.nonce != original_nonce

    def test_delete_logs_access(self, vault):
        """Verify deletion is logged."""
        encrypted = vault.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="alpaca",
            plaintext="secret_key"
        )
        vault.delete(encrypted)

        log = vault.get_access_log(user_id="user_001")
        assert len(log) == 1
        assert log[0]["purpose"] == "DELETION"


class TestCredentialRotation:
    """Tests for credential rotation functionality."""

    @pytest.fixture
    def vault(self):
        """Create a vault with random master key."""
        return CredentialVault(secrets.token_bytes(32))

    def test_rotate_credential(self, vault):
        """Verify credential rotation works correctly."""
        original = vault.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="alpaca",
            plaintext="old_key"
        )
        vault.decrypt(original, purpose="test")  # Access to increment count
        original_access_count = original.access_count

        rotated = vault.rotate_credential(original, "new_key", "rotation")

        # Verify access count preserved (before any decrypt on rotated)
        assert rotated.access_count == original_access_count

        # Verify new credential decrypts correctly
        decrypted = vault.decrypt(rotated, purpose="test")
        assert decrypted == "new_key"

        # After decrypt, access count should be incremented
        assert rotated.access_count == original_access_count + 1

        # Verify old credential was securely deleted
        # (original ciphertext should be overwritten)
        with pytest.raises(CredentialDecryptionError):
            vault.decrypt(original, purpose="test")


class TestCredentialVerification:
    """Tests for credential verification functionality."""

    @pytest.fixture
    def vault(self):
        """Create a vault with random master key."""
        return CredentialVault(secrets.token_bytes(32))

    def test_verify_correct_credential(self, vault):
        """Verify correct credential passes verification."""
        encrypted = vault.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="alpaca",
            plaintext="correct_key"
        )
        assert vault.verify_credential(encrypted, "correct_key") is True

    def test_verify_incorrect_credential(self, vault):
        """Verify incorrect credential fails verification."""
        encrypted = vault.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="alpaca",
            plaintext="correct_key"
        )
        assert vault.verify_credential(encrypted, "wrong_key") is False

    def test_verify_tampered_credential(self, vault):
        """Verify tampered credential fails verification."""
        encrypted = vault.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="alpaca",
            plaintext="correct_key"
        )
        encrypted.ciphertext = b"tampered"
        assert vault.verify_credential(encrypted, "correct_key") is False


class TestSerialization:
    """Tests for credential serialization/deserialization."""

    @pytest.fixture
    def vault(self):
        """Create a vault with random master key."""
        return CredentialVault(secrets.token_bytes(32))

    def test_to_dict_from_dict_roundtrip(self, vault):
        """Verify credential can be serialized and deserialized."""
        encrypted = vault.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="alpaca",
            plaintext="test_key",
            metadata={"label": "primary"}
        )
        data = encrypted.to_dict()
        restored = EncryptedCredential.from_dict(data)

        # Verify fields match
        assert restored.credential_id == encrypted.credential_id
        assert restored.user_id == encrypted.user_id
        assert restored.credential_type == encrypted.credential_type
        assert restored.broker == encrypted.broker
        assert restored.ciphertext == encrypted.ciphertext
        assert restored.nonce == encrypted.nonce
        assert restored.metadata == encrypted.metadata

        # Verify can still decrypt
        decrypted = vault.decrypt(restored, purpose="test")
        assert decrypted == "test_key"


class TestKeyDerivation:
    """Tests for key derivation behavior."""

    def test_same_user_same_derived_key(self):
        """Verify same user always derives same key (deterministic)."""
        master_key = secrets.token_bytes(32)
        vault1 = CredentialVault(master_key)
        vault2 = CredentialVault(master_key)

        # Encrypt with vault1, decrypt with vault2
        encrypted = vault1.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="alpaca",
            plaintext="test_key"
        )
        decrypted = vault2.decrypt(encrypted, purpose="test")
        assert decrypted == "test_key"

    def test_different_master_key_cannot_decrypt(self):
        """Verify different master key cannot decrypt."""
        vault1 = CredentialVault(secrets.token_bytes(32))
        vault2 = CredentialVault(secrets.token_bytes(32))

        encrypted = vault1.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="alpaca",
            plaintext="test_key"
        )

        with pytest.raises(CredentialDecryptionError):
            vault2.decrypt(encrypted, purpose="test")
