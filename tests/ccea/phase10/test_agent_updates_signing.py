# -*- coding: utf-8 -*-
"""
Tests for Agent Updates signing with Ed25519.

Phase 10: Enterprise signing.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from uuid import uuid4
import pytest

# Skip tests if cryptography is not available
cryptography = pytest.importorskip("cryptography")

from packages.cloud.enterprise.agent_updates import (
    AgentUpdateManager,
    AgentUpdateConfig,
    AgentUpdate,
    UpdateChannel,
    UpdateState,
    UpdatePriority,
    AgentUpdateStatus,
)
from packages.cloud.enterprise.crypto import (
    Ed25519Signer,
    CRYPTO_AVAILABLE,
)


@pytest.fixture
def signing_key():
    """Create a signing key for tests."""
    signer = Ed25519Signer()
    return signer.generate_key(key_id="agent-update-signer")


@pytest.fixture
def update_config():
    """Create update manager configuration."""
    return AgentUpdateConfig(
        default_channel=UpdateChannel.STABLE,
        require_signatures=True,
        enable_staged_rollout=False,
    )


@pytest.fixture
def update_manager(update_config):
    """Create agent update manager."""
    return AgentUpdateManager(update_config)


class TestAgentUpdateSigning:
    """Tests for agent update signing."""

    @pytest.mark.asyncio
    async def test_sign_update_with_ed25519(self, update_manager, signing_key):
        """Test signing update with real Ed25519."""
        update = update_manager.create_update(
            version="1.1.0",
            artifact_digest="sha256:abc123",
            artifact_url="https://example.com/update.tar.gz",
        )

        success = await update_manager.sign_update(
            update.id,
            signing_key.private_key_bytes,
            signer_id="test-signer",
        )

        assert success is True
        assert update.signature != ""
        assert update.signed_by == "test-signer"
        assert update.signed_at is not None

    @pytest.mark.asyncio
    async def test_signed_update_contains_json_signature(
        self, update_manager, signing_key
    ):
        """Test that signature is proper JSON format."""
        update = update_manager.create_update(
            version="1.2.0",
            artifact_digest="sha256:def456",
            artifact_url="https://example.com/update.tar.gz",
        )

        await update_manager.sign_update(update.id, signing_key.private_key_bytes)

        sig_data = json.loads(update.signature)
        assert "signature" in sig_data
        assert sig_data["algorithm"] == "ed25519"
        assert sig_data["payload_type"] == "agent-update"

    @pytest.mark.asyncio
    async def test_verify_signed_update(self, update_manager, signing_key):
        """Test verifying signed update."""
        update = update_manager.create_update(
            version="1.3.0",
            artifact_digest="sha256:ghi789",
            artifact_url="https://example.com/update.tar.gz",
        )

        await update_manager.sign_update(update.id, signing_key.private_key_bytes)

        # Verify using public key
        is_valid, error = await update_manager.verify_update_signature(
            update.id,
            signing_key.public_key_bytes,
        )

        assert is_valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_verify_tampered_update(self, update_manager, signing_key):
        """Test that tampered updates fail verification."""
        update = update_manager.create_update(
            version="1.4.0",
            artifact_digest="sha256:original",
            artifact_url="https://example.com/update.tar.gz",
        )

        await update_manager.sign_update(update.id, signing_key.private_key_bytes)

        # Tamper with the artifact digest
        update.artifact_digest = "sha256:tampered"

        # Verification should fail
        is_valid, error = await update_manager.verify_update_signature(
            update.id,
            signing_key.public_key_bytes,
        )

        assert is_valid is False

    @pytest.mark.asyncio
    async def test_sign_update_not_found(self, update_manager, signing_key):
        """Test signing non-existent update."""
        fake_id = uuid4()
        success = await update_manager.sign_update(
            fake_id,
            signing_key.private_key_bytes,
        )
        assert success is False

    @pytest.mark.asyncio
    async def test_verify_unsigned_update(self, update_manager):
        """Test verification of unsigned update."""
        update = update_manager.create_update(
            version="1.5.0",
            artifact_digest="sha256:xyz",
            artifact_url="https://example.com/update.tar.gz",
        )

        is_valid, error = await update_manager.verify_update_signature(update.id)

        assert is_valid is False
        assert "not signed" in error


class TestUpdateReleaseWithSignature:
    """Tests for releasing signed updates."""

    @pytest.mark.asyncio
    async def test_release_requires_signature(self, signing_key):
        """Test that release fails without signature when required."""
        config = AgentUpdateConfig(require_signatures=True)
        manager = AgentUpdateManager(config)

        update = manager.create_update(
            version="2.0.0",
            artifact_digest="sha256:unsigned",
            artifact_url="https://example.com/update.tar.gz",
        )

        # Try to release without signing
        success = await manager.release_update(update.id)
        assert success is False

    @pytest.mark.asyncio
    async def test_release_signed_update_succeeds(self, signing_key):
        """Test releasing signed update succeeds."""
        import base64

        config = AgentUpdateConfig(
            require_signatures=True,
            enable_staged_rollout=False,
            trusted_signing_keys=[
                base64.b64encode(signing_key.public_key_bytes).decode()
            ],
        )
        manager = AgentUpdateManager(config)

        update = manager.create_update(
            version="2.1.0",
            artifact_digest="sha256:signed",
            artifact_url="https://example.com/update.tar.gz",
        )

        # Sign first
        await manager.sign_update(update.id, signing_key.private_key_bytes)

        # Release should succeed
        success = await manager.release_update(update.id)
        assert success is True
        assert update.released_at is not None

    @pytest.mark.asyncio
    async def test_release_without_signature_when_not_required(self):
        """Test release succeeds without signature when not required."""
        config = AgentUpdateConfig(require_signatures=False)
        manager = AgentUpdateManager(config)

        update = manager.create_update(
            version="2.2.0",
            artifact_digest="sha256:nosig",
            artifact_url="https://example.com/update.tar.gz",
        )

        success = await manager.release_update(update.id)
        assert success is True


class TestTrustedSigningKeys:
    """Tests for trusted signing key management."""

    @pytest.mark.asyncio
    async def test_verify_with_trusted_keys_config(self, signing_key):
        """Test verification using trusted keys from config."""
        import base64

        config = AgentUpdateConfig(
            require_signatures=True,
            trusted_signing_keys=[
                base64.b64encode(signing_key.public_key_bytes).decode()
            ],
        )
        manager = AgentUpdateManager(config)

        update = manager.create_update(
            version="3.0.0",
            artifact_digest="sha256:trusted",
            artifact_url="https://example.com/update.tar.gz",
        )

        await manager.sign_update(update.id, signing_key.private_key_bytes)

        # Verify without explicitly passing key - uses trusted keys
        is_valid, error = await manager.verify_update_signature(update.id)

        assert is_valid is True


class TestSignaturePayloadFormat:
    """Tests for update signature payload format."""

    @pytest.mark.asyncio
    async def test_signature_payload_structure(self, update_manager, signing_key):
        """Test the structure of signature payload."""
        update = update_manager.create_update(
            version="4.0.0",
            artifact_digest="sha256:payload",
            artifact_url="https://example.com/update.tar.gz",
        )

        # Create the payload that gets signed
        payload = update_manager._create_signature_payload(update)
        payload_data = json.loads(payload.decode())

        assert payload_data["version"] == "4.0.0"
        assert payload_data["artifact_digest"] == "sha256:payload"
        assert payload_data["artifact_url"] == "https://example.com/update.tar.gz"
        assert "created_at" in payload_data

    @pytest.mark.asyncio
    async def test_payload_is_deterministic(self, update_manager):
        """Test that payload generation is deterministic."""
        update = update_manager.create_update(
            version="4.1.0",
            artifact_digest="sha256:deterministic",
            artifact_url="https://example.com/update.tar.gz",
        )

        payload1 = update_manager._create_signature_payload(update)
        payload2 = update_manager._create_signature_payload(update)

        assert payload1 == payload2


class TestCryptoFallbackInUpdates:
    """Tests for crypto fallback behavior in updates."""

    @pytest.mark.asyncio
    async def test_sign_fallback_without_crypto(self):
        """Test signing falls back when crypto unavailable."""
        manager = AgentUpdateManager()

        update = manager.create_update(
            version="5.0.0",
            artifact_digest="sha256:fallback",
            artifact_url="https://example.com/update.tar.gz",
        )

        with patch('packages.cloud.enterprise.agent_updates.CRYPTO_AVAILABLE', False):
            success = await manager.sign_update(
                update.id,
                b"fake-key-bytes" + b"\x00" * 20,  # 32 bytes total
            )

        # Should still produce a signature (placeholder)
        assert success is True
        assert update.signature != ""

    @pytest.mark.asyncio
    async def test_verify_fallback_without_crypto(self):
        """Test verification returns True when crypto unavailable."""
        manager = AgentUpdateManager()

        update = manager.create_update(
            version="5.1.0",
            artifact_digest="sha256:fallback2",
            artifact_url="https://example.com/update.tar.gz",
        )
        update.signature = "fake-signature"

        with patch('packages.cloud.enterprise.agent_updates.CRYPTO_AVAILABLE', False):
            is_valid, error = await manager.verify_update_signature(update.id)

        # Should return True (verification skipped)
        assert is_valid is True


class TestUpdateWorkflowWithSigning:
    """Integration tests for update workflow with signing."""

    @pytest.mark.asyncio
    async def test_full_update_workflow(self, signing_key):
        """Test complete update workflow with signing."""
        import base64

        config = AgentUpdateConfig(
            require_signatures=True,
            enable_staged_rollout=True,
            trusted_signing_keys=[
                base64.b64encode(signing_key.public_key_bytes).decode()
            ],
        )
        manager = AgentUpdateManager(config)

        agent_id = uuid4()

        # Step 1: Create update
        update = manager.create_update(
            version="6.0.0",
            artifact_digest="sha256:workflow",
            artifact_url="https://example.com/update.tar.gz",
            priority=UpdatePriority.HIGH,
        )

        # Step 2: Sign update
        await manager.sign_update(update.id, signing_key.private_key_bytes)
        assert update.signed_at is not None

        # Step 3: Verify signature
        is_valid, _ = await manager.verify_update_signature(
            update.id,
            signing_key.public_key_bytes,
        )
        assert is_valid

        # Step 4: Release update
        await manager.release_update(update.id, initial_rollout_percentage=5.0)
        assert update.released_at is not None
        assert update.rollout_percentage == 5.0

        # Step 5: Check for updates
        available = await manager.check_for_updates(
            agent_id,
            current_version="5.9.0",
        )
        # May or may not be available depending on rollout hash

    @pytest.mark.asyncio
    async def test_reject_unsigned_during_install(self, signing_key):
        """Test that unsigned updates are rejected during install check."""
        config = AgentUpdateConfig(require_signatures=True)
        manager = AgentUpdateManager(config)

        update = manager.create_update(
            version="6.1.0",
            artifact_digest="sha256:unsigned-install",
            artifact_url="https://example.com/update.tar.gz",
        )

        # Try to release unsigned - should fail
        success = await manager.release_update(update.id)
        assert success is False


class TestSigningAlgorithmDetails:
    """Tests for Ed25519 signing algorithm specifics."""

    @pytest.mark.asyncio
    async def test_signature_is_64_bytes(self, update_manager, signing_key):
        """Test that Ed25519 signature is exactly 64 bytes."""
        update = update_manager.create_update(
            version="7.0.0",
            artifact_digest="sha256:64bytes",
            artifact_url="https://example.com/update.tar.gz",
        )

        await update_manager.sign_update(update.id, signing_key.private_key_bytes)

        sig_data = json.loads(update.signature)
        import base64
        sig_bytes = base64.b64decode(sig_data["signature"])

        assert len(sig_bytes) == 64  # Ed25519 signature size

    @pytest.mark.asyncio
    async def test_different_keys_produce_different_signatures(self):
        """Test that different keys produce different signatures."""
        signer = Ed25519Signer()
        key1 = signer.generate_key(key_id="key1")
        key2 = signer.generate_key(key_id="key2")

        manager = AgentUpdateManager()

        # Create two identical updates
        update1 = manager.create_update(
            version="7.1.0",
            artifact_digest="sha256:same",
            artifact_url="https://example.com/update.tar.gz",
        )
        update2 = manager.create_update(
            version="7.1.0",
            artifact_digest="sha256:same",
            artifact_url="https://example.com/update.tar.gz",
        )

        # Sign with different keys
        await manager.sign_update(update1.id, key1.private_key_bytes)
        await manager.sign_update(update2.id, key2.private_key_bytes)

        # Signatures should be different
        assert update1.signature != update2.signature
