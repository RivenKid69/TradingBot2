# -*- coding: utf-8 -*-
"""
Tests for Key Manager.

Per Design Doc Phase 4:
- Key management (keyless sigstore vs keyful for enterprise/offline)
- Trust root definition
- Key rotation
- Key revocation
"""

import pytest
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ccea.crypto.key_manager import (
    KeyManager,
    KeyMetadata,
    KeyStatus,
    KeyPurpose,
    TrustLevel,
    TrustRoot,
    SigningKeyProvider,
)
from ccea.crypto.keys import KeyAlgorithm


class TestKeyMetadata:
    """Tests for KeyMetadata."""

    def test_metadata_creation(self):
        """Test metadata creation."""
        now = datetime.now(timezone.utc)
        metadata = KeyMetadata(
            key_id="test-key-1",
            algorithm=KeyAlgorithm.ED25519,
            purpose=KeyPurpose.ARTIFACT_SIGNING,
            status=KeyStatus.ACTIVE,
            trust_level=TrustLevel.LEAF,
            created_at=now,
        )

        assert metadata.key_id == "test-key-1"
        assert metadata.algorithm == KeyAlgorithm.ED25519
        assert metadata.purpose == KeyPurpose.ARTIFACT_SIGNING
        assert metadata.status == KeyStatus.ACTIVE

    def test_is_valid_active_key(self):
        """Test is_valid for active key."""
        metadata = KeyMetadata(
            key_id="test-key",
            algorithm=KeyAlgorithm.ED25519,
            purpose=KeyPurpose.ARTIFACT_SIGNING,
            status=KeyStatus.ACTIVE,
            trust_level=TrustLevel.LEAF,
            created_at=datetime.now(timezone.utc),
        )

        assert metadata.is_valid() is True

    def test_is_valid_revoked_key(self):
        """Test is_valid for revoked key."""
        metadata = KeyMetadata(
            key_id="test-key",
            algorithm=KeyAlgorithm.ED25519,
            purpose=KeyPurpose.ARTIFACT_SIGNING,
            status=KeyStatus.REVOKED,
            trust_level=TrustLevel.LEAF,
            created_at=datetime.now(timezone.utc),
            revoked_at=datetime.now(timezone.utc),
        )

        assert metadata.is_valid() is False

    def test_is_valid_expired_key(self):
        """Test is_valid for expired key."""
        past = datetime.now(timezone.utc) - timedelta(days=1)
        metadata = KeyMetadata(
            key_id="test-key",
            algorithm=KeyAlgorithm.ED25519,
            purpose=KeyPurpose.ARTIFACT_SIGNING,
            status=KeyStatus.ACTIVE,
            trust_level=TrustLevel.LEAF,
            created_at=datetime.now(timezone.utc) - timedelta(days=365),
            expires_at=past,
        )

        assert metadata.is_valid() is False

    def test_serialization(self):
        """Test metadata serialization."""
        now = datetime.now(timezone.utc)
        metadata = KeyMetadata(
            key_id="test-key",
            algorithm=KeyAlgorithm.ED25519,
            purpose=KeyPurpose.ARTIFACT_SIGNING,
            status=KeyStatus.ACTIVE,
            trust_level=TrustLevel.LEAF,
            created_at=now,
            labels={"env": "test"},
        )

        data = metadata.to_dict()
        restored = KeyMetadata.from_dict(data)

        assert restored.key_id == metadata.key_id
        assert restored.algorithm == metadata.algorithm
        assert restored.purpose == metadata.purpose
        assert restored.labels == metadata.labels


class TestTrustRoot:
    """Tests for TrustRoot."""

    def test_trust_root_creation(self):
        """Test trust root creation."""
        trust_root = TrustRoot(
            root_key_ids={"root-key-1", "root-key-2"},
            allowed_purposes={KeyPurpose.ARTIFACT_SIGNING},
        )

        assert len(trust_root.root_key_ids) == 2
        assert KeyPurpose.ARTIFACT_SIGNING in trust_root.allowed_purposes

    def test_trust_root_serialization(self):
        """Test trust root serialization."""
        trust_root = TrustRoot(
            root_key_ids={"root-key-1"},
            allowed_purposes={KeyPurpose.ARTIFACT_SIGNING, KeyPurpose.MANIFEST_SIGNING},
            max_validity_days=180,
        )

        data = trust_root.to_dict()

        assert "root-key-1" in data["root_key_ids"]
        assert data["max_validity_days"] == 180


class TestKeyManager:
    """Tests for KeyManager."""

    @pytest.fixture
    def key_manager(self, tmp_path):
        """Create KeyManager instance."""
        return KeyManager(storage_path=tmp_path / "keys")

    def test_initialization(self, tmp_path):
        """Test KeyManager initialization."""
        km = KeyManager(storage_path=tmp_path / "keys")

        assert km.storage_path.exists()
        assert km.keys_dir.exists()

    def test_generate_key(self, key_manager):
        """Test key generation."""
        key_id = key_manager.generate_key(
            purpose=KeyPurpose.ARTIFACT_SIGNING,
            algorithm=KeyAlgorithm.ED25519,
        )

        assert key_id is not None
        assert key_id.startswith("ccea-artifact_signing-")

        # Check key exists
        public_key = key_manager.get_public_key(key_id)
        assert public_key is not None

    def test_generate_key_with_expiration(self, key_manager):
        """Test key generation with expiration."""
        key_id = key_manager.generate_key(
            purpose=KeyPurpose.ARTIFACT_SIGNING,
            validity_days=30,
        )

        metadata = key_manager.get_metadata(key_id)
        assert metadata.expires_at is not None
        assert metadata.expires_at > datetime.now(timezone.utc)

    def test_get_active_key(self, key_manager):
        """Test getting active key."""
        # Generate key
        key_id = key_manager.generate_key(
            purpose=KeyPurpose.ARTIFACT_SIGNING,
        )

        # Get active key
        active = key_manager.get_active_key(KeyPurpose.ARTIFACT_SIGNING)
        assert active == key_id

    def test_get_signing_keys(self, key_manager):
        """Test getting all signing keys."""
        # Generate multiple keys
        key1 = key_manager.generate_key(purpose=KeyPurpose.ARTIFACT_SIGNING)
        key2 = key_manager.generate_key(purpose=KeyPurpose.ARTIFACT_SIGNING)

        keys = key_manager.get_signing_keys(KeyPurpose.ARTIFACT_SIGNING)
        assert key1 in keys
        assert key2 in keys

    def test_rotate_key(self, key_manager):
        """Test key rotation."""
        # Generate initial key
        old_key_id = key_manager.generate_key(
            purpose=KeyPurpose.ARTIFACT_SIGNING,
        )

        # Rotate
        new_key_id = key_manager.rotate_key(old_key_id, grace_hours=1)

        assert new_key_id != old_key_id

        # Old key should be in rotating status
        old_meta = key_manager.get_metadata(old_key_id)
        assert old_meta.status == KeyStatus.ROTATING

        # New key should be active
        new_meta = key_manager.get_metadata(new_key_id)
        assert new_meta.status == KeyStatus.ACTIVE

        # Both keys should be valid during grace period
        assert old_meta.is_valid() is True
        assert new_meta.is_valid() is True

    def test_revoke_key(self, key_manager):
        """Test key revocation."""
        key_id = key_manager.generate_key(
            purpose=KeyPurpose.ARTIFACT_SIGNING,
        )

        key_manager.revoke_key(key_id, reason="Compromised")

        metadata = key_manager.get_metadata(key_id)
        assert metadata.status == KeyStatus.REVOKED
        assert metadata.revocation_reason == "Compromised"
        assert metadata.is_valid() is False

        # Private key should be removed
        private = key_manager.get_private_key(key_id)
        assert private is None

        # Public key should still exist (for verification)
        public = key_manager.get_public_key(key_id)
        assert public is not None

    def test_is_revoked(self, key_manager):
        """Test is_revoked check."""
        key_id = key_manager.generate_key(
            purpose=KeyPurpose.ARTIFACT_SIGNING,
        )

        assert key_manager.is_revoked(key_id) is False

        key_manager.revoke_key(key_id, reason="test")
        assert key_manager.is_revoked(key_id) is True

    def test_set_trust_root(self, key_manager):
        """Test setting trust root."""
        root_key = key_manager.generate_key(
            purpose=KeyPurpose.ARTIFACT_SIGNING,
            trust_level=TrustLevel.ROOT,
        )

        key_manager.set_trust_root(
            root_key_ids=[root_key],
            allowed_purposes=[KeyPurpose.ARTIFACT_SIGNING],
        )

        # Create leaf key with parent
        leaf_key = key_manager.generate_key(
            purpose=KeyPurpose.ARTIFACT_SIGNING,
            trust_level=TrustLevel.LEAF,
            parent_key_id=root_key,
        )

        # Verify trust chain
        assert key_manager.verify_trust_chain(leaf_key) is True

    def test_verify_trust_chain_no_root(self, key_manager):
        """Test trust chain verification without root."""
        key_id = key_manager.generate_key(
            purpose=KeyPurpose.ARTIFACT_SIGNING,
        )

        # No trust root set
        assert key_manager.verify_trust_chain(key_id) is False

    def test_list_keys(self, key_manager):
        """Test listing keys with filters."""
        # Generate keys with different purposes
        art_key = key_manager.generate_key(purpose=KeyPurpose.ARTIFACT_SIGNING)
        man_key = key_manager.generate_key(purpose=KeyPurpose.MANIFEST_SIGNING)

        # List all
        all_keys = key_manager.list_keys()
        assert len(all_keys) == 2

        # List by purpose
        art_keys = key_manager.list_keys(purpose=KeyPurpose.ARTIFACT_SIGNING)
        assert len(art_keys) == 1
        assert art_keys[0].key_id == art_key

    def test_export_public_key(self, key_manager):
        """Test exporting public key."""
        key_id = key_manager.generate_key(
            purpose=KeyPurpose.ARTIFACT_SIGNING,
        )

        pem = key_manager.export_public_key(key_id)
        assert pem is not None
        assert "PUBLIC KEY" in pem

    def test_import_public_key(self, key_manager):
        """Test importing external public key."""
        # Generate a key first to get valid PEM
        temp_key = key_manager.generate_key(
            purpose=KeyPurpose.ARTIFACT_SIGNING,
        )
        pem = key_manager.export_public_key(temp_key)

        # Import as external key
        key_manager.import_public_key(
            key_id="external-key-1",
            pem_data=pem,
            purpose=KeyPurpose.ARTIFACT_SIGNING,
        )

        # Verify it's imported
        public = key_manager.get_public_key("external-key-1")
        assert public is not None

        metadata = key_manager.get_metadata("external-key-1")
        assert metadata.labels.get("imported") == "true"

    def test_cleanup_expired(self, key_manager):
        """Test cleanup of expired keys."""
        # Generate key that expires immediately
        key_id = key_manager.generate_key(
            purpose=KeyPurpose.ARTIFACT_SIGNING,
            validity_days=0,  # Already expired
        )

        # Manually set expiration to past
        meta = key_manager.get_metadata(key_id)
        meta.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        key_manager._save_metadata()

        # Cleanup
        expired = key_manager.cleanup_expired()
        assert key_id in expired

        # Key should be expired now
        meta = key_manager.get_metadata(key_id)
        assert meta.status == KeyStatus.EXPIRED

    def test_persistence(self, tmp_path):
        """Test key persistence across instances."""
        storage = tmp_path / "keys"

        # Create first instance and generate key
        km1 = KeyManager(storage_path=storage)
        key_id = km1.generate_key(
            purpose=KeyPurpose.ARTIFACT_SIGNING,
        )
        pem1 = km1.export_public_key(key_id)

        # Create second instance
        km2 = KeyManager(storage_path=storage)

        # Key should be loaded
        public = km2.get_public_key(key_id)
        assert public is not None

        pem2 = km2.export_public_key(key_id)
        assert pem1 == pem2


class TestSigningKeyProvider:
    """Tests for SigningKeyProvider."""

    @pytest.fixture
    def key_manager(self, tmp_path):
        """Create KeyManager instance."""
        return KeyManager(storage_path=tmp_path / "keys")

    def test_get_artifact_signing_key(self, key_manager):
        """Test getting artifact signing key."""
        # Generate key first
        key_id = key_manager.generate_key(
            purpose=KeyPurpose.ARTIFACT_SIGNING,
        )

        provider = SigningKeyProvider(key_manager)
        keypair = provider.get_artifact_signing_key()

        assert keypair is not None
        assert keypair.key_id == key_id
        assert keypair.private_key is not None
        assert keypair.public_key is not None

    def test_get_artifact_signing_key_no_key(self, key_manager):
        """Test getting artifact signing key when none exists."""
        provider = SigningKeyProvider(key_manager)
        keypair = provider.get_artifact_signing_key()

        assert keypair is None

    def test_get_verification_keys(self, key_manager):
        """Test getting verification keys."""
        # Generate multiple keys
        key1 = key_manager.generate_key(purpose=KeyPurpose.ARTIFACT_SIGNING)
        key2 = key_manager.generate_key(purpose=KeyPurpose.ARTIFACT_SIGNING)

        provider = SigningKeyProvider(key_manager)
        keys = provider.get_verification_keys()

        assert len(keys) == 2
        assert key1 in keys
        assert key2 in keys
