# -*- coding: utf-8 -*-
"""
Tests for Evidence Pack signing with Ed25519.

Phase 10: Enterprise signing.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
import pytest

# Skip tests if cryptography is not available
cryptography = pytest.importorskip("cryptography")

from packages.cloud.enterprise.evidence_pack import (
    EvidencePackExporter,
    EvidencePackConfig,
    EvidencePack,
    EvidenceType,
    ExportDestination,
    ExportFormat,
)
from packages.cloud.enterprise.crypto import (
    Ed25519Signer,
    CRYPTO_AVAILABLE,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def signing_key(temp_dir):
    """Create a signing key for tests."""
    signer = Ed25519Signer()
    key = signer.generate_key(key_id="test-evidence-signer")
    key_path = temp_dir / "signing-key.pem"
    signer.save_key(key, key_path, include_private=True)
    return key_path


@pytest.fixture
def exporter_config(temp_dir, signing_key):
    """Create exporter configuration."""
    return EvidencePackConfig(
        output_path=temp_dir / "packs",
        sign_pack=True,
        signing_key_path=signing_key,
        destination=ExportDestination.LOCAL,
        output_format=ExportFormat.JSON,
        compress=False,
    )


@pytest.fixture
def exporter(exporter_config):
    """Create evidence pack exporter."""
    return EvidencePackExporter(exporter_config)


class TestEvidencePackSigning:
    """Tests for evidence pack signing."""

    @pytest.mark.asyncio
    async def test_sign_pack_with_crypto(self, exporter, temp_dir):
        """Test signing evidence pack with real Ed25519."""
        # Create a simple pack
        pack = EvidencePack(
            version="1.0.0",
            created_at=datetime.now(timezone.utc),
            checksum="sha256:abc123def456",
        )

        # Sign the pack
        signature = await exporter._sign_pack(pack.checksum)

        assert signature is not None
        sig_data = json.loads(signature)
        assert "signature" in sig_data
        assert sig_data["algorithm"] == "ed25519"
        assert sig_data["signer_id"] == "ccea-evidence-pack-signer"
        assert sig_data["payload_type"] == "evidence-pack-checksum"

    @pytest.mark.asyncio
    async def test_verify_pack_signature(self, exporter, temp_dir):
        """Test verifying evidence pack signature."""
        checksum = "sha256:test123"

        # Sign
        signature = await exporter._sign_pack(checksum)
        assert signature is not None

        # Verify
        is_valid = await exporter._verify_signature(checksum, signature)
        assert is_valid

    @pytest.mark.asyncio
    async def test_verify_invalid_signature(self, exporter):
        """Test that invalid signatures fail verification."""
        checksum = "sha256:original"
        signature = await exporter._sign_pack(checksum)

        # Verify with different checksum
        is_valid = await exporter._verify_signature("sha256:modified", signature)
        assert not is_valid

    @pytest.mark.asyncio
    async def test_sign_pack_without_key(self, temp_dir):
        """Test signing without configured key uses ephemeral key."""
        config = EvidencePackConfig(
            output_path=temp_dir / "packs",
            sign_pack=True,
            # No signing_key_path
        )
        exporter = EvidencePackExporter(config)

        # Call _sign_pack directly without mocking
        signature = await exporter._sign_pack("sha256:test")

        # Should still sign with ephemeral key
        assert signature is not None

    @pytest.mark.asyncio
    async def test_legacy_signature_format_verification(self, exporter):
        """Test that legacy signature formats are handled."""
        # Legacy format (base64 encoded string)
        import base64

        legacy_sig = base64.b64encode(b"CCEA-EVIDENCE-SIG::test::2025-01-01").decode()

        # Should not crash, may return True for legacy
        is_valid = await exporter._verify_signature("test", legacy_sig)
        # Legacy format is accepted (returns True for backwards compatibility)


class TestEvidencePackSigningIntegration:
    """Integration tests for evidence pack signing."""

    @pytest.mark.asyncio
    async def test_sign_and_verify_roundtrip(self, exporter, temp_dir):
        """Test complete sign/verify roundtrip."""
        # Test the signing and verification flow
        checksum = "sha256:integration_test_checksum"

        # Sign
        signature = await exporter._sign_pack(checksum)
        assert signature is not None

        # Parse signature
        sig_data = json.loads(signature)
        assert "signature" in sig_data
        assert sig_data["algorithm"] == "ed25519"

        # Verify
        is_valid = await exporter._verify_signature(checksum, signature)
        assert is_valid

    @pytest.mark.asyncio
    async def test_signature_changes_with_different_data(self, exporter, temp_dir):
        """Test that different data produces different signatures."""
        sig1 = await exporter._sign_pack("sha256:data1")
        sig2 = await exporter._sign_pack("sha256:data2")

        # Signatures should be different
        assert sig1 != sig2


class TestSigningKeyManagement:
    """Tests for signing key management in evidence packs."""

    def test_config_with_signing_key_path(self, temp_dir, signing_key):
        """Test configuration with signing key path."""
        config = EvidencePackConfig(
            output_path=temp_dir,
            sign_pack=True,
            signing_key_path=signing_key,
        )
        assert config.sign_pack is True
        assert config.signing_key_path == signing_key

    def test_config_without_signing(self, temp_dir):
        """Test configuration without signing."""
        config = EvidencePackConfig(
            output_path=temp_dir,
            sign_pack=False,
        )
        assert config.sign_pack is False

    @pytest.mark.asyncio
    async def test_sign_with_missing_key_file(self, temp_dir):
        """Test signing with non-existent key file."""
        config = EvidencePackConfig(
            output_path=temp_dir,
            sign_pack=True,
            signing_key_path=temp_dir / "nonexistent.pem",
        )
        exporter = EvidencePackExporter(config)

        # Should use ephemeral key and still produce signature
        signature = await exporter._sign_pack("sha256:test")
        assert signature is not None


class TestSignaturePayload:
    """Tests for signature payload structure."""

    @pytest.mark.asyncio
    async def test_signature_contains_required_fields(self, exporter):
        """Test that signature contains all required fields."""
        signature = await exporter._sign_pack("sha256:test123")
        sig_data = json.loads(signature)

        required_fields = [
            "signature",
            "algorithm",
            "key_id",
            "signed_at",
            "signer_id",
            "payload_digest",
            "payload_type",
        ]

        for field in required_fields:
            assert field in sig_data, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_signature_algorithm_is_ed25519(self, exporter):
        """Test that signature uses Ed25519 algorithm."""
        signature = await exporter._sign_pack("sha256:test")
        sig_data = json.loads(signature)
        assert sig_data["algorithm"] == "ed25519"

    @pytest.mark.asyncio
    async def test_payload_digest_matches(self, exporter):
        """Test that payload digest in signature matches input."""
        checksum = "sha256:abc123"
        signature = await exporter._sign_pack(checksum)
        sig_data = json.loads(signature)

        # The payload digest should be SHA-256 of the checksum bytes
        import hashlib

        expected_digest = f"sha256:{hashlib.sha256(checksum.encode()).hexdigest()}"
        assert sig_data["payload_digest"] == expected_digest


class TestCryptoFallback:
    """Tests for crypto fallback behavior."""

    @pytest.mark.asyncio
    async def test_fallback_when_crypto_unavailable(self, temp_dir):
        """Test fallback signature when cryptography is unavailable."""
        config = EvidencePackConfig(output_path=temp_dir)
        exporter = EvidencePackExporter(config)

        # Mock CRYPTO_AVAILABLE to False
        with patch("packages.cloud.enterprise.evidence_pack.CRYPTO_AVAILABLE", False):
            signature = await exporter._sign_pack("sha256:test")

        # Should return base64-encoded placeholder
        assert signature is not None
        import base64

        decoded = base64.b64decode(signature).decode()
        assert "CCEA-EVIDENCE-SIG::" in decoded

    @pytest.mark.asyncio
    async def test_verify_fallback_signature(self, temp_dir):
        """Test verification of fallback signature."""
        config = EvidencePackConfig(output_path=temp_dir)
        exporter = EvidencePackExporter(config)

        with patch("packages.cloud.enterprise.evidence_pack.CRYPTO_AVAILABLE", False):
            # Verification should return True when crypto unavailable
            is_valid = await exporter._verify_signature("test", "anything")
            assert is_valid is True
