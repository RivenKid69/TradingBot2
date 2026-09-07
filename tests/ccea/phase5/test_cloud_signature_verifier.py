# -*- coding: utf-8 -*-
"""
Tests for Cloud Signature Verification.

Design Doc Section 10.2:
- All Cloud-to-Agent messages MUST be signed
- Agent MUST verify signature BEFORE processing
- Reject all unsigned or invalid-signed messages
"""

import pytest
from datetime import datetime, timedelta, timezone

from ccea.crypto.keys import generate_keypair, KeyAlgorithm
from ccea.crypto.signing import sign_json

from packages.agent.cloud.signature_verifier import (
    CloudSignatureVerifier,
    CloudSignatureVerifierConfig,
    SignatureVerificationError,
    VerificationResult,
    VerifiedCloudMessage,
    create_verifier_from_enrollment,
)


class TestCloudSignatureVerifier:
    """Tests for CloudSignatureVerifier."""

    @pytest.fixture
    def cloud_keypair(self):
        """Generate Cloud keypair for testing."""
        return generate_keypair(KeyAlgorithm.ED25519)

    @pytest.fixture
    def verifier(self, cloud_keypair):
        """Create verifier with Cloud public key."""
        return CloudSignatureVerifier(cloud_public_key_pem=cloud_keypair.get_public_key_pem())

    @pytest.fixture
    def signed_message(self, cloud_keypair):
        """Create a signed message."""
        message = {
            "command_id": "123",
            "command_type": "UPDATE_POLICY",
            "payload": {"max_position_size": 100},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return sign_json(
            message,
            cloud_keypair.private_key,
            key_id="cloud_primary",
        )

    def test_verify_valid_signature(self, verifier, signed_message):
        """Test verifying a valid signature."""
        result = verifier.verify_message(signed_message)

        assert result.valid
        assert result.key_id == "cloud_primary"
        assert result.reason is None

    def test_verify_unsigned_message_fails_by_default(self, verifier):
        """Test that unsigned messages are rejected by default."""
        message = {
            "command_id": "123",
            "command_type": "UPDATE_POLICY",
        }

        result = verifier.verify_message(message)

        assert not result.valid
        assert result.reason == "message_unsigned"

    def test_verify_unsigned_message_allowed_in_dev_mode(self, cloud_keypair):
        """Test that unsigned messages can be allowed in development mode."""
        verifier = CloudSignatureVerifier(
            cloud_public_key_pem=cloud_keypair.get_public_key_pem(),
            config=CloudSignatureVerifierConfig(allow_unsigned=True),
        )

        message = {"command_id": "123"}

        result = verifier.verify_message(message)

        assert result.valid
        assert result.reason == "unsigned_allowed"

    def test_verify_invalid_signature_fails(self, verifier, cloud_keypair):
        """Test that invalid signatures are rejected."""
        # Create a message signed with different key
        other_keypair = generate_keypair(KeyAlgorithm.ED25519)
        message = sign_json(
            {"command_id": "123"},
            other_keypair.private_key,
            key_id="cloud_primary",
        )

        result = verifier.verify_message(message)

        assert not result.valid
        assert result.reason == "invalid_signature"

    def test_verify_unknown_key_fails(self, verifier, cloud_keypair):
        """Test that unknown key_id fails."""
        message = sign_json(
            {"command_id": "123"},
            cloud_keypair.private_key,
            key_id="unknown_key",
        )

        result = verifier.verify_message(message)

        assert not result.valid
        assert "unknown_key" in result.reason

    def test_verify_expired_message_fails(self, cloud_keypair):
        """Test that expired messages are rejected."""
        verifier = CloudSignatureVerifier(
            cloud_public_key_pem=cloud_keypair.get_public_key_pem(),
            config=CloudSignatureVerifierConfig(max_message_age_seconds=60),
        )

        # Create message with old timestamp
        old_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        message = sign_json(
            {
                "command_id": "123",
                "timestamp": old_time.isoformat(),
            },
            cloud_keypair.private_key,
            key_id="cloud_primary",
        )

        result = verifier.verify_message(message)

        assert not result.valid
        assert "message_expired" in result.reason

    def test_verify_or_raise_raises_on_invalid(self, verifier):
        """Test verify_or_raise raises exception on invalid signature."""
        message = {"command_id": "123"}  # Unsigned

        with pytest.raises(SignatureVerificationError) as exc_info:
            verifier.verify_or_raise(message)

        assert "message_unsigned" in exc_info.value.reason

    def test_verify_or_raise_passes_on_valid(self, verifier, signed_message):
        """Test verify_or_raise passes on valid signature."""
        # Should not raise
        verifier.verify_or_raise(signed_message)

    def test_add_trusted_key(self, verifier):
        """Test adding additional trusted keys."""
        other_keypair = generate_keypair(KeyAlgorithm.ED25519)
        verifier.add_trusted_key("rotation_key", other_keypair.get_public_key_pem())

        # Sign with rotation key
        message = sign_json(
            {"command_id": "456"},
            other_keypair.private_key,
            key_id="rotation_key",
        )

        result = verifier.verify_message(message)

        assert result.valid
        assert result.key_id == "rotation_key"

    def test_remove_trusted_key(self, verifier):
        """Test removing trusted keys."""
        other_keypair = generate_keypair(KeyAlgorithm.ED25519)
        verifier.add_trusted_key("temp_key", other_keypair.get_public_key_pem())
        verifier.remove_trusted_key("temp_key")

        message = sign_json(
            {"command_id": "789"},
            other_keypair.private_key,
            key_id="temp_key",
        )

        result = verifier.verify_message(message)

        assert not result.valid
        assert "unknown_key" in result.reason

    def test_cannot_remove_primary_key(self, verifier):
        """Test that primary key cannot be removed."""
        verifier.remove_trusted_key("cloud_primary")

        # Primary key should still work
        assert verifier.has_cloud_key

    def test_stats_tracking(self, verifier, signed_message, cloud_keypair):
        """Test verification statistics."""
        # Verify valid
        verifier.verify_message(signed_message)

        # Verify invalid (unsigned)
        verifier.verify_message({"command_id": "123"})

        stats = verifier.get_stats()

        assert stats["verified"] == 1
        assert stats["unsigned"] == 1

    def test_reset_stats(self, verifier, signed_message):
        """Test resetting statistics."""
        verifier.verify_message(signed_message)
        verifier.reset_stats()

        stats = verifier.get_stats()

        assert stats["verified"] == 0

    def test_has_cloud_key_false_when_not_set(self):
        """Test has_cloud_key returns False when no key is set."""
        verifier = CloudSignatureVerifier()

        assert not verifier.has_cloud_key

    def test_verify_command(self, verifier, signed_message):
        """Test verify_command method."""
        result = verifier.verify_command(signed_message)

        assert result.valid

    def test_verify_commands(self, verifier, cloud_keypair):
        """Test verifying multiple commands."""
        commands = []
        for i in range(3):
            cmd = sign_json(
                {"command_id": str(i)},
                cloud_keypair.private_key,
                key_id="cloud_primary",
            )
            commands.append(cmd)

        results = verifier.verify_commands(commands)

        assert len(results) == 3
        assert all(r.valid for r in results)


class TestVerifiedCloudMessage:
    """Tests for VerifiedCloudMessage wrapper."""

    @pytest.fixture
    def cloud_keypair(self):
        """Generate Cloud keypair."""
        return generate_keypair(KeyAlgorithm.ED25519)

    def test_create_from_valid_verification(self, cloud_keypair):
        """Test creating VerifiedCloudMessage from valid verification."""
        message = {"command_id": "123", "data": "test"}
        verification = VerificationResult(valid=True, key_id="cloud_primary")

        verified = VerifiedCloudMessage(message, verification)

        assert verified.data == message
        assert verified["command_id"] == "123"
        assert verified.get("data") == "test"
        assert verified.key_id == "cloud_primary"

    def test_create_from_invalid_verification_raises(self):
        """Test that invalid verification raises exception."""
        message = {"command_id": "123"}
        verification = VerificationResult(valid=False, reason="invalid_signature")

        with pytest.raises(SignatureVerificationError):
            VerifiedCloudMessage(message, verification)


class TestCreateVerifierFromEnrollment:
    """Tests for create_verifier_from_enrollment factory."""

    @pytest.fixture
    def cloud_keypair(self):
        """Generate Cloud keypair."""
        return generate_keypair(KeyAlgorithm.ED25519)

    def test_creates_verifier_with_cloud_key(self, cloud_keypair):
        """Test creating verifier from enrollment response with Cloud key."""
        enrollment_response = {
            "agent_id": "agent_123",
            "access_token": "token",
            "cloud_public_key": cloud_keypair.get_public_key_pem(),
        }

        verifier = create_verifier_from_enrollment(enrollment_response)

        assert verifier.has_cloud_key

    def test_creates_disabled_verifier_without_cloud_key(self):
        """Test creating verifier when no Cloud key provided."""
        enrollment_response = {
            "agent_id": "agent_123",
            "access_token": "token",
            # No cloud_public_key
        }

        verifier = create_verifier_from_enrollment(enrollment_response)

        assert not verifier.has_cloud_key
        # Should not enforce verification
        result = verifier.verify_message({"command_id": "123"})
        # Enforcement is disabled, so should pass
        assert result.valid or not verifier._config.enforce_verification


class TestCloudSignatureVerifierConfig:
    """Tests for configuration options."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CloudSignatureVerifierConfig()

        assert config.enforce_verification is True
        assert config.allow_unsigned is False
        assert config.max_message_age_seconds == 300
        assert config.log_failures is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = CloudSignatureVerifierConfig(
            enforce_verification=False,
            allow_unsigned=True,
            max_message_age_seconds=600,
        )

        assert config.enforce_verification is False
        assert config.allow_unsigned is True
        assert config.max_message_age_seconds == 600


class TestSignatureVerificationSecurity:
    """Security-focused tests for signature verification."""

    @pytest.fixture
    def cloud_keypair(self):
        """Generate Cloud keypair."""
        return generate_keypair(KeyAlgorithm.ED25519)

    def test_tampered_message_fails(self, cloud_keypair):
        """Test that tampered messages fail verification."""
        verifier = CloudSignatureVerifier(cloud_public_key_pem=cloud_keypair.get_public_key_pem())

        # Sign original message
        signed = sign_json(
            {"command_id": "123", "amount": 100},
            cloud_keypair.private_key,
            key_id="cloud_primary",
        )

        # Tamper with message
        signed["amount"] = 99999999

        result = verifier.verify_message(signed)

        assert not result.valid
        assert result.reason == "invalid_signature"

    def test_signature_replay_protection(self, cloud_keypair):
        """Test replay protection via timestamp check."""
        verifier = CloudSignatureVerifier(
            cloud_public_key_pem=cloud_keypair.get_public_key_pem(),
            config=CloudSignatureVerifierConfig(max_message_age_seconds=60),
        )

        # Create message with old timestamp
        old_timestamp = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        signed = sign_json(
            {"command_id": "123", "timestamp": old_timestamp},
            cloud_keypair.private_key,
            key_id="cloud_primary",
        )

        result = verifier.verify_message(signed)

        assert not result.valid
        assert "expired" in result.reason

    def test_fail_closed_on_missing_key(self):
        """Test fail-closed when no Cloud key is set."""
        verifier = CloudSignatureVerifier(
            config=CloudSignatureVerifierConfig(enforce_verification=True)
        )

        keypair = generate_keypair(KeyAlgorithm.ED25519)
        signed = sign_json(
            {"command_id": "123"},
            keypair.private_key,
            key_id="cloud_primary",
        )

        result = verifier.verify_message(signed)

        assert not result.valid
        assert "unknown_key" in result.reason

    def test_signature_algorithm_mismatch_detected(self, cloud_keypair):
        """Test that algorithm mismatches are detected."""
        verifier = CloudSignatureVerifier(cloud_public_key_pem=cloud_keypair.get_public_key_pem())

        # Try to verify with EC key against Ed25519 public key
        ec_keypair = generate_keypair(KeyAlgorithm.ECDSA_P256)
        signed = sign_json(
            {"command_id": "123"},
            ec_keypair.private_key,
            key_id="cloud_primary",
        )

        result = verifier.verify_message(signed)

        # Should fail due to algorithm/key mismatch
        assert not result.valid
