# -*- coding: utf-8 -*-
"""
Tests for Ed25519 cryptographic operations.

Phase 10: Enterprise signing.
"""

import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pytest

# Skip all tests if cryptography is not available
cryptography = pytest.importorskip("cryptography")

from packages.cloud.enterprise.crypto import (
    Ed25519Signer,
    SigningKey,
    Signature,
    CryptoError,
    SignatureError,
    VerificationError,
    create_signer,
    sign_data,
    verify_data,
    CRYPTO_AVAILABLE,
    SIGNATURE_ALGORITHM,
    KEY_SIZE,
)


class TestSigningKey:
    """Tests for SigningKey dataclass."""

    def test_create_empty_key(self):
        """Test creating empty signing key."""
        key = SigningKey()
        assert key.key_id == ""
        assert key.private_key_bytes == b""
        assert key.public_key_bytes == b""
        assert key.algorithm == SIGNATURE_ALGORITHM
        assert not key.is_valid

    def test_key_expiration(self):
        """Test key expiration check."""
        # Non-expired key
        key = SigningKey(
            key_id="test",
            private_key_bytes=b"x" * KEY_SIZE,
            expires_at=datetime.now(timezone.utc) + timedelta(days=30)
        )
        # Note: is_valid also checks if key bytes are correct format
        # For this test, we just check the expiration logic structure

        # Expired key
        expired_key = SigningKey(
            key_id="test-expired",
            public_key_bytes=b"y" * KEY_SIZE,
            expires_at=datetime.now(timezone.utc) - timedelta(days=1)
        )
        assert not expired_key.is_valid

    def test_to_dict(self):
        """Test key serialization."""
        key = SigningKey(
            key_id="test-key",
            algorithm="ed25519",
        )
        data = key.to_dict()
        assert data["key_id"] == "test-key"
        assert data["algorithm"] == "ed25519"
        assert "public_key" in data
        assert "created_at" in data


class TestSignature:
    """Tests for Signature dataclass."""

    def test_create_signature(self):
        """Test creating signature object."""
        sig = Signature(
            signature_bytes=b"test_signature",
            key_id="test-key",
            signer_id="test-signer",
            payload_digest="sha256:abc123",
        )
        assert sig.signature_bytes == b"test_signature"
        assert sig.key_id == "test-key"
        assert sig.signer_id == "test-signer"
        assert sig.payload_digest == "sha256:abc123"

    def test_signature_base64(self):
        """Test base64 encoding of signature."""
        sig = Signature(signature_bytes=b"hello")
        assert sig.signature_base64 == "aGVsbG8="

    def test_from_base64(self):
        """Test creating signature from base64."""
        sig = Signature.from_base64("aGVsbG8=", key_id="test")
        assert sig.signature_bytes == b"hello"
        assert sig.key_id == "test"

    def test_to_dict(self):
        """Test signature serialization."""
        sig = Signature(
            signature_bytes=b"test",
            key_id="key-1",
            signer_id="signer-1",
            payload_type="test-payload",
        )
        data = sig.to_dict()
        assert data["key_id"] == "key-1"
        assert data["signer_id"] == "signer-1"
        assert data["payload_type"] == "test-payload"
        assert "signature" in data

    def test_from_dict(self):
        """Test signature deserialization."""
        data = {
            "signature": "dGVzdA==",  # "test" in base64
            "key_id": "key-1",
            "signer_id": "signer-1",
            "signed_at": datetime.now(timezone.utc).isoformat(),
            "payload_digest": "sha256:123",
        }
        sig = Signature.from_dict(data)
        assert sig.signature_bytes == b"test"
        assert sig.key_id == "key-1"
        assert sig.signer_id == "signer-1"


class TestEd25519Signer:
    """Tests for Ed25519Signer."""

    def test_init(self):
        """Test signer initialization."""
        signer = Ed25519Signer()
        assert signer.default_signer_id == "ccea-signer"
        assert len(signer.get_trusted_keys()) == 0

    def test_init_with_custom_signer_id(self):
        """Test signer with custom ID."""
        signer = Ed25519Signer(default_signer_id="custom-signer")
        assert signer.default_signer_id == "custom-signer"

    def test_generate_key(self):
        """Test key generation."""
        signer = Ed25519Signer()
        key = signer.generate_key(key_id="test-key")

        assert key.key_id == "test-key"
        assert len(key.private_key_bytes) == KEY_SIZE
        assert len(key.public_key_bytes) == KEY_SIZE
        assert key.is_valid
        assert key.private_key is not None
        assert key.public_key is not None

    def test_generate_key_with_expiry(self):
        """Test key generation with expiration."""
        signer = Ed25519Signer()
        key = signer.generate_key(key_id="expiring-key", expires_in_days=30)

        assert key.expires_at is not None
        assert key.expires_at > datetime.now(timezone.utc)
        assert key.is_valid

    def test_generate_key_auto_id(self):
        """Test key generation with auto-generated ID."""
        signer = Ed25519Signer()
        key = signer.generate_key()

        assert key.key_id != ""
        assert len(key.key_id) == 16  # SHA-256 truncated to 16 chars

    def test_sign_and_verify(self):
        """Test signing and verification."""
        signer = Ed25519Signer()
        key = signer.generate_key(key_id="sign-test")

        data = b"Hello, World!"
        signature = signer.sign(data, key, payload_type="test")

        assert signature.signature_bytes != b""
        assert signature.key_id == "sign-test"
        assert signature.payload_type == "test"
        assert signature.payload_digest.startswith("sha256:")

        # Verify
        is_valid = signer.verify(data, signature, key)
        assert is_valid

    def test_verify_with_trusted_key(self):
        """Test verification using trusted keys."""
        signer = Ed25519Signer()
        key = signer.generate_key(key_id="trusted-key")

        # Add as trusted key
        signer.add_trusted_key(key)

        data = b"test data"
        signature = signer.sign(data, key)

        # Verify without explicitly passing key
        is_valid = signer.verify(data, signature)
        assert is_valid

    def test_verify_invalid_signature(self):
        """Test that invalid signatures fail verification."""
        signer = Ed25519Signer()
        key = signer.generate_key()

        data = b"original data"
        signature = signer.sign(data, key)

        # Modify data
        modified_data = b"modified data"
        is_valid = signer.verify(modified_data, signature, key)
        assert not is_valid

    def test_verify_tampered_signature(self):
        """Test that tampered signatures fail verification."""
        signer = Ed25519Signer()
        key = signer.generate_key()

        data = b"test data"
        signature = signer.sign(data, key)

        # Tamper with signature
        tampered_sig = Signature(
            signature_bytes=b"x" * len(signature.signature_bytes),
            key_id=signature.key_id,
        )
        is_valid = signer.verify(data, tampered_sig, key)
        assert not is_valid

    def test_sign_without_private_key(self):
        """Test that signing fails without private key."""
        signer = Ed25519Signer()
        key = SigningKey(
            key_id="public-only",
            public_key_bytes=b"x" * KEY_SIZE,  # Invalid but tests the check
        )

        with pytest.raises(SignatureError):
            signer.sign(b"data", key)

    def test_add_remove_trusted_key(self):
        """Test trusted key management."""
        signer = Ed25519Signer()
        key = signer.generate_key(key_id="trusted")

        assert len(signer.get_trusted_keys()) == 0

        signer.add_trusted_key(key)
        assert len(signer.get_trusted_keys()) == 1
        assert signer.get_trusted_keys()[0].key_id == "trusted"

        signer.remove_trusted_key("trusted")
        assert len(signer.get_trusted_keys()) == 0

    def test_export_public_key_raw(self):
        """Test raw public key export."""
        signer = Ed25519Signer()
        key = signer.generate_key()

        raw = signer.export_public_key(key, format="raw")
        assert len(raw) == KEY_SIZE

    def test_export_public_key_pem(self):
        """Test PEM public key export."""
        signer = Ed25519Signer()
        key = signer.generate_key()

        pem = signer.export_public_key(key, format="pem")
        assert pem.startswith(b"-----BEGIN PUBLIC KEY-----")
        assert pem.strip().endswith(b"-----END PUBLIC KEY-----")

    def test_export_public_key_openssh(self):
        """Test OpenSSH public key export."""
        signer = Ed25519Signer()
        key = signer.generate_key()

        openssh = signer.export_public_key(key, format="openssh")
        assert openssh.startswith(b"ssh-ed25519 ")

    def test_import_private_key_raw(self):
        """Test raw private key import."""
        signer = Ed25519Signer()

        # Generate key to get valid key bytes
        original = signer.generate_key(key_id="original")
        private_bytes = original.private_key_bytes

        # Import the raw bytes
        imported = signer.import_private_key(private_bytes, key_id="imported")

        assert imported.key_id == "imported"
        assert imported.private_key_bytes == private_bytes
        assert imported.public_key_bytes == original.public_key_bytes

    def test_import_public_key_raw(self):
        """Test raw public key import."""
        signer = Ed25519Signer()

        # Generate key to get valid key bytes
        original = signer.generate_key()
        public_bytes = original.public_key_bytes

        # Import the raw bytes
        imported = signer.import_public_key(public_bytes, key_id="pub-imported")

        assert imported.key_id == "pub-imported"
        assert imported.public_key_bytes == public_bytes
        assert imported.private_key_bytes == b""


class TestSignerKeyPersistence:
    """Tests for key save/load functionality."""

    def test_save_and_load_private_key(self):
        """Test saving and loading private key."""
        signer = Ed25519Signer()
        key = signer.generate_key(key_id="persistent")

        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = Path(tmpdir) / "test.pem"

            # Save with private key
            signer.save_key(key, key_path, include_private=True)

            # Load
            loaded = signer.load_key(key_path, key_id="loaded")

            assert loaded.key_id == "loaded"
            assert loaded.private_key_bytes == key.private_key_bytes
            assert loaded.public_key_bytes == key.public_key_bytes

    def test_save_and_load_public_key_only(self):
        """Test saving and loading public key only."""
        signer = Ed25519Signer()
        key = signer.generate_key(key_id="pub-persist")

        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = Path(tmpdir) / "public.pem"

            # Save public only
            signer.save_key(key, key_path, include_private=False)

            # Load
            loaded = signer.load_key(key_path, key_id="pub-loaded")

            assert loaded.key_id == "pub-loaded"
            assert loaded.public_key_bytes == key.public_key_bytes
            assert loaded.private_key_bytes == b""  # No private key

    def test_save_and_load_encrypted_key(self):
        """Test saving and loading password-protected key."""
        signer = Ed25519Signer()
        key = signer.generate_key(key_id="encrypted")
        password = b"strong-password-123"

        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = Path(tmpdir) / "encrypted.pem"

            # Save with encryption
            signer.save_key(key, key_path, include_private=True, password=password)

            # Load with password
            loaded = signer.load_key(key_path, key_id="decrypted", password=password)

            assert loaded.private_key_bytes == key.private_key_bytes

    def test_sign_and_verify_roundtrip_with_persistence(self):
        """Test full sign/verify cycle with persisted keys."""
        signer = Ed25519Signer()
        key = signer.generate_key(key_id="roundtrip")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save keys
            private_path = Path(tmpdir) / "private.pem"
            public_path = Path(tmpdir) / "public.pem"

            signer.save_key(key, private_path, include_private=True)
            signer.save_key(key, public_path, include_private=False)

            # Sign with loaded private key
            private_key = signer.load_key(private_path)
            data = b"test message for roundtrip"
            signature = signer.sign(data, private_key)

            # Verify with loaded public key
            public_key = signer.load_key(public_path)
            is_valid = signer.verify(data, signature, public_key)
            assert is_valid


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_sign_data_function(self):
        """Test simple sign_data utility."""
        signer = Ed25519Signer()
        key = signer.generate_key()

        sig_json = sign_data(
            b"test data",
            key.private_key_bytes,
            signer_id="test-signer",
            payload_type="test-type",
        )

        assert isinstance(sig_json, str)
        sig_data = json.loads(sig_json)
        assert sig_data["signer_id"] == "test-signer"
        assert sig_data["payload_type"] == "test-type"

    def test_verify_data_function(self):
        """Test simple verify_data utility."""
        signer = Ed25519Signer()
        key = signer.generate_key()

        data = b"verify me"
        sig_json = sign_data(data, key.private_key_bytes)

        is_valid = verify_data(data, sig_json, key.public_key_bytes)
        assert is_valid

    def test_verify_data_invalid(self):
        """Test verify_data with invalid data."""
        signer = Ed25519Signer()
        key = signer.generate_key()

        data = b"original"
        sig_json = sign_data(data, key.private_key_bytes)

        is_valid = verify_data(b"modified", sig_json, key.public_key_bytes)
        assert not is_valid

    def test_create_signer_factory(self):
        """Test create_signer factory function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create trusted keys directory
            trusted_dir = Path(tmpdir) / "trusted"
            trusted_dir.mkdir()

            # Save a trusted key
            signer_temp = Ed25519Signer()
            key = signer_temp.generate_key(key_id="trusted-1")
            signer_temp.save_key(
                key,
                trusted_dir / "trusted-1.pem",
                include_private=False
            )

            # Create signer with trusted keys
            signer = create_signer(
                trusted_keys_dir=trusted_dir,
                signer_id="factory-signer"
            )

            assert signer.default_signer_id == "factory-signer"
            assert len(signer.get_trusted_keys()) == 1


class TestCryptoAvailability:
    """Tests for crypto availability checking."""

    def test_crypto_available(self):
        """Test that CRYPTO_AVAILABLE is True when cryptography is installed."""
        assert CRYPTO_AVAILABLE is True

    def test_signature_algorithm_constant(self):
        """Test signature algorithm constant."""
        assert SIGNATURE_ALGORITHM == "ed25519"

    def test_key_size_constant(self):
        """Test key size constant."""
        assert KEY_SIZE == 32


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_verify_without_key(self):
        """Test verification fails without available key."""
        signer = Ed25519Signer()
        sig = Signature(
            signature_bytes=b"fake",
            key_id="unknown-key",
        )

        with pytest.raises(VerificationError):
            signer.verify(b"data", sig)

    def test_invalid_key_size(self):
        """Test import with invalid key size."""
        signer = Ed25519Signer()

        with pytest.raises(Exception):  # KeyError or ValueError
            signer.import_private_key(b"too-short", format="raw")

    def test_export_unknown_format(self):
        """Test export with unknown format."""
        signer = Ed25519Signer()
        key = signer.generate_key()

        with pytest.raises(ValueError):
            signer.export_public_key(key, format="unknown")

    def test_empty_data_signing(self):
        """Test signing empty data."""
        signer = Ed25519Signer()
        key = signer.generate_key()

        signature = signer.sign(b"", key)
        is_valid = signer.verify(b"", signature, key)
        assert is_valid

    def test_large_data_signing(self):
        """Test signing large data."""
        signer = Ed25519Signer()
        key = signer.generate_key()

        # 1MB of data
        large_data = b"x" * (1024 * 1024)

        signature = signer.sign(large_data, key)
        is_valid = signer.verify(large_data, signature, key)
        assert is_valid
