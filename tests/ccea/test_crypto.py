# -*- coding: utf-8 -*-
"""
Tests for CCEA Cryptographic Module.

Tests:
- Key generation and serialization
- Message signing and verification
- Digest computation
- Token generation
"""

import pytest
from datetime import datetime

from ccea.crypto.keys import (
    KeyPair,
    KeyAlgorithm,
    generate_keypair,
    serialize_public_key,
    serialize_private_key,
    load_public_key,
    load_private_key,
    get_key_algorithm,
)
from ccea.crypto.signing import (
    sign_message,
    verify_signature,
    sign_json,
    verify_json_signature,
    MessageSigner,
    MessageVerifier,
)
from ccea.crypto.digest import (
    compute_digest,
    compute_file_digest,
    verify_digest,
    compute_json_digest,
    DigestVerifier,
)
from ccea.crypto.tokens import (
    generate_enrollment_token,
    generate_idempotency_key,
    generate_agent_id,
    generate_command_id,
    is_valid_enrollment_token,
    is_valid_agent_id,
    is_valid_idempotency_key,
)


class TestKeyGeneration:
    """Tests for key generation."""

    def test_generate_ed25519_keypair(self):
        """Test Ed25519 keypair generation."""
        keypair = generate_keypair(KeyAlgorithm.ED25519)

        assert keypair is not None
        assert keypair.algorithm == KeyAlgorithm.ED25519
        assert keypair.private_key is not None
        assert keypair.public_key is not None

    def test_generate_ecdsa_keypair(self):
        """Test ECDSA P-256 keypair generation."""
        keypair = generate_keypair(KeyAlgorithm.ECDSA_P256)

        assert keypair is not None
        assert keypair.algorithm == KeyAlgorithm.ECDSA_P256

    def test_keypair_with_key_id(self):
        """Test keypair with key ID."""
        keypair = generate_keypair(key_id="test_key_123")

        assert keypair.key_id == "test_key_123"

    def test_public_key_pem_serialization(self):
        """Test public key PEM serialization."""
        keypair = generate_keypair()
        pem = keypair.get_public_key_pem()

        assert "-----BEGIN PUBLIC KEY-----" in pem
        assert "-----END PUBLIC KEY-----" in pem

    def test_private_key_pem_serialization(self):
        """Test private key PEM serialization."""
        keypair = generate_keypair()
        pem = keypair.get_private_key_pem()

        assert "-----BEGIN PRIVATE KEY-----" in pem
        assert "-----END PRIVATE KEY-----" in pem

    def test_key_roundtrip(self):
        """Test key serialization and loading roundtrip."""
        original = generate_keypair()

        # Serialize
        pub_pem = serialize_public_key(original.public_key)
        priv_pem = serialize_private_key(original.private_key)

        # Load
        loaded_pub = load_public_key(pub_pem)
        loaded_priv = load_private_key(priv_pem)

        # Verify algorithm detection
        assert get_key_algorithm(loaded_pub) == original.algorithm

    def test_encrypted_private_key(self):
        """Test encrypted private key."""
        keypair = generate_keypair()
        password = b"test_password_123"

        encrypted_pem = keypair.get_private_key_pem(password)
        assert "ENCRYPTED" in encrypted_pem

        # Load with password
        loaded = load_private_key(encrypted_pem, password)
        assert loaded is not None


class TestSigning:
    """Tests for message signing."""

    def test_sign_and_verify_message(self):
        """Test basic message signing and verification."""
        keypair = generate_keypair()
        message = b"Hello, CCEA!"

        signature = sign_message(message, keypair.private_key)
        assert signature is not None

        is_valid = verify_signature(message, signature, keypair.public_key)
        assert is_valid is True

    def test_invalid_signature_rejected(self):
        """Test that invalid signature is rejected."""
        keypair = generate_keypair()
        message = b"Original message"
        tampered = b"Tampered message"

        signature = sign_message(message, keypair.private_key)
        is_valid = verify_signature(tampered, signature, keypair.public_key)

        assert is_valid is False

    def test_wrong_key_rejected(self):
        """Test that wrong key is rejected."""
        keypair1 = generate_keypair()
        keypair2 = generate_keypair()
        message = b"Test message"

        signature = sign_message(message, keypair1.private_key)
        is_valid = verify_signature(message, signature, keypair2.public_key)

        assert is_valid is False

    def test_sign_json_data(self):
        """Test JSON data signing."""
        keypair = generate_keypair()
        data = {
            "message_type": "HEARTBEAT",
            "agent_id": "agent_test123456789012",
            "timestamp": "2024-01-01T00:00:00Z",
        }

        signed = sign_json(data, keypair.private_key, "key_1")

        assert "signature" in signed
        assert signed["signature"]["algorithm"] == "ed25519"
        assert signed["signature"]["key_id"] == "key_1"

    def test_verify_json_signature(self):
        """Test JSON signature verification."""
        keypair = generate_keypair()
        data = {"test": "data", "number": 123}

        signed = sign_json(data, keypair.private_key)
        is_valid = verify_json_signature(signed, keypair.public_key)

        assert is_valid is True

    def test_message_signer_verifier(self):
        """Test MessageSigner and MessageVerifier classes."""
        keypair = generate_keypair(key_id="agent_key")

        signer = MessageSigner(keypair.private_key, "agent_key")
        verifier = MessageVerifier()
        verifier.add_key("agent_key", keypair.public_key)

        data = {"command": "test", "value": 42}
        signed = signer.sign(data)

        assert verifier.verify(signed, "agent_key") is True
        assert verifier.verify(signed) is True  # Uses key_id from signature


class TestDigest:
    """Tests for digest computation."""

    def test_compute_digest(self):
        """Test basic digest computation."""
        data = b"Test data for hashing"
        digest = compute_digest(data)

        assert digest.startswith("sha256:")
        assert len(digest) == 71  # sha256: + 64 hex chars

    def test_digest_consistency(self):
        """Test that same data produces same digest."""
        data = b"Consistent data"

        digest1 = compute_digest(data)
        digest2 = compute_digest(data)

        assert digest1 == digest2

    def test_different_data_different_digest(self):
        """Test that different data produces different digest."""
        data1 = b"Data one"
        data2 = b"Data two"

        assert compute_digest(data1) != compute_digest(data2)

    def test_verify_digest(self):
        """Test digest verification."""
        data = b"Verification test"
        digest = compute_digest(data)

        assert verify_digest(data, digest) is True
        assert verify_digest(b"Wrong data", digest) is False

    def test_compute_json_digest(self):
        """Test JSON digest computation."""
        data = {"key": "value", "number": 42}
        digest = compute_json_digest(data)

        assert digest.startswith("sha256:")

    def test_json_digest_order_independent(self):
        """Test that JSON digest is key-order independent."""
        data1 = {"a": 1, "b": 2}
        data2 = {"b": 2, "a": 1}

        assert compute_json_digest(data1) == compute_json_digest(data2)

    def test_digest_verifier(self):
        """Test DigestVerifier class."""
        verifier = DigestVerifier()
        data = b"Test content"
        digest = compute_digest(data)

        verifier.register("test_artifact", digest)

        assert verifier.verify("test_artifact", data) is True
        assert verifier.verify("test_artifact", b"Wrong") is False
        assert verifier.verify("unknown", data) is False


class TestTokenGeneration:
    """Tests for token generation."""

    def test_generate_enrollment_token(self):
        """Test enrollment token generation."""
        token = generate_enrollment_token()

        assert token.startswith("enroll_")
        assert len(token) >= 39  # enroll_ + 32 chars min
        assert is_valid_enrollment_token(token) is True

    def test_enrollment_token_custom_length(self):
        """Test enrollment token with custom length."""
        token = generate_enrollment_token(length=48)

        assert len(token) == 7 + 48  # enroll_ + 48 chars

    def test_generate_agent_id(self):
        """Test agent ID generation."""
        agent_id = generate_agent_id()

        assert agent_id.startswith("agent_")
        assert len(agent_id) >= 22  # agent_ + 16 chars min
        assert is_valid_agent_id(agent_id) is True

    def test_generate_idempotency_key(self):
        """Test idempotency key generation."""
        key = generate_idempotency_key()

        assert len(key) >= 16
        assert is_valid_idempotency_key(key) is True

    def test_generate_command_id(self):
        """Test command ID generation."""
        cmd_id = generate_command_id()

        assert cmd_id.startswith("cmd_")
        assert len(cmd_id) > 20

    def test_token_uniqueness(self):
        """Test that generated tokens are unique."""
        tokens = [generate_enrollment_token() for _ in range(100)]

        assert len(set(tokens)) == 100

    def test_invalid_tokens(self):
        """Test validation of invalid tokens."""
        assert is_valid_enrollment_token("invalid") is False
        assert is_valid_enrollment_token("enroll_short") is False
        assert is_valid_agent_id("not_agent") is False
        assert is_valid_idempotency_key("short") is False


class TestKeyAlgorithms:
    """Tests for different key algorithms."""

    @pytest.mark.parametrize("algorithm", [KeyAlgorithm.ED25519, KeyAlgorithm.ECDSA_P256])
    def test_sign_verify_all_algorithms(self, algorithm):
        """Test signing/verification with all algorithms."""
        keypair = generate_keypair(algorithm)
        message = b"Test message for all algorithms"

        signature = sign_message(message, keypair.private_key)
        assert verify_signature(message, signature, keypair.public_key) is True
