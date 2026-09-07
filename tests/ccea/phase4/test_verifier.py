# -*- coding: utf-8 -*-
"""
Tests for Artifact Verifier.

Per Design Doc Phase 4:
- verify digest + signature + allowlist registry + schema_version compatibility
- strict reject: unsigned/unknown registry/unknown schema_version
"""

import json
import pytest
from pathlib import Path

from ccea.artifact.verifier import (
    ArtifactVerifier,
    VerificationReport,
    VerificationResult,
    RejectionReason,
    SchemaVersionPolicy,
    VerificationCache,
    create_verifier_from_key_manager,
)
from ccea.artifact.signer import SignatureInfo, ArtifactSigner
from ccea.crypto.keys import generate_keypair, KeyAlgorithm
from ccea.crypto.digest import compute_file_digest


class TestSchemaVersionPolicy:
    """Tests for SchemaVersionPolicy."""

    def test_default_policy(self):
        """Test default schema policy."""
        policy = SchemaVersionPolicy()

        assert policy.min_supported == "1.0.0"
        assert policy.max_supported == "1.99.99"

    def test_compatible_version(self):
        """Test compatible version."""
        policy = SchemaVersionPolicy(
            min_supported="1.0.0",
            max_supported="2.0.0",
        )

        compatible, warning = policy.is_compatible("1.5.0")
        assert compatible is True
        assert warning is None

    def test_version_below_minimum(self):
        """Test version below minimum."""
        policy = SchemaVersionPolicy(
            min_supported="1.0.0",
            max_supported="2.0.0",
        )

        compatible, warning = policy.is_compatible("0.9.0")
        assert compatible is False
        assert "below minimum" in warning

    def test_version_above_maximum(self):
        """Test version above maximum."""
        policy = SchemaVersionPolicy(
            min_supported="1.0.0",
            max_supported="2.0.0",
        )

        compatible, warning = policy.is_compatible("3.0.0")
        assert compatible is False
        assert "above maximum" in warning

    def test_deprecated_version(self):
        """Test deprecated version warning."""
        policy = SchemaVersionPolicy(
            min_supported="1.0.0",
            max_supported="2.0.0",
            deprecated_versions={"1.0.0"},
        )

        compatible, warning = policy.is_compatible("1.0.0")
        assert compatible is True
        assert "deprecated" in warning

    def test_invalid_version_format(self):
        """Test invalid version format."""
        policy = SchemaVersionPolicy()

        compatible, warning = policy.is_compatible("invalid")
        assert compatible is False
        assert "Invalid" in warning


class TestArtifactVerifier:
    """Tests for ArtifactVerifier."""

    @pytest.fixture
    def keypair(self):
        """Generate test keypair."""
        return generate_keypair(algorithm=KeyAlgorithm.ED25519, key_id="test-key")

    @pytest.fixture
    def sample_artifact(self, tmp_path, keypair):
        """Create sample artifact with manifest."""
        # Create artifact
        artifact_path = tmp_path / "artifact.zip"
        artifact_path.write_bytes(b"test artifact content")

        # Create manifest
        manifest = {
            "schema_version": "1.0.0",
            "artifact_id": "test-artifact",
            "entrypoint": {
                "module": "strategy",
                "class": "TestStrategy",
            },
            "runtime": {
                "python_version": "3.11",
            },
            "deps_lock_digest": "sha256:" + "a" * 64,
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        return artifact_path, manifest_path

    def test_verifier_initialization(self):
        """Test verifier initialization."""
        verifier = ArtifactVerifier()

        assert verifier._strict_mode is True
        assert "local" in verifier._allowed_registries

    def test_add_trusted_key(self, keypair):
        """Test adding trusted key."""
        verifier = ArtifactVerifier()
        verifier.add_trusted_key("test-key", keypair.public_key)

        assert "test-key" in verifier._trusted_keys

    def test_verify_unsigned_strict_mode(self, sample_artifact):
        """Test that unsigned artifacts are rejected in strict mode."""
        artifact_path, manifest_path = sample_artifact

        verifier = ArtifactVerifier(strict_mode=True)

        report = verifier.verify(
            artifact_path=artifact_path,
            manifest_path=manifest_path,
        )

        assert report.result == VerificationResult.REJECTED
        assert report.rejection_reason == RejectionReason.UNSIGNED

    def test_verify_unsigned_non_strict_mode(self, sample_artifact):
        """Test unsigned artifacts with warning in non-strict mode."""
        artifact_path, manifest_path = sample_artifact

        # Disable SBOM requirement for this test (testing signature behavior only)
        verifier = ArtifactVerifier(strict_mode=False, require_sbom=False)

        report = verifier.verify(
            artifact_path=artifact_path,
            manifest_path=manifest_path,
        )

        assert report.result == VerificationResult.VERIFIED
        assert len(report.warnings) > 0
        assert any("unsigned" in w.lower() for w in report.warnings)

    def test_verify_signed_artifact(self, sample_artifact, keypair):
        """Test verification of signed artifact."""
        artifact_path, manifest_path = sample_artifact

        # Sign artifact
        signer = ArtifactSigner.from_keypair(keypair)
        signature = signer.sign_file(artifact_path)

        # Update manifest with signature and sbom_ref (for SBOM enforcement)
        manifest = json.loads(manifest_path.read_text())
        manifest["signature"] = signature.to_dict()
        manifest["sbom_ref"] = "sha256:dummy_sbom_ref"  # Add SBOM reference
        manifest_path.write_text(json.dumps(manifest))

        # Verify
        verifier = ArtifactVerifier()
        verifier.add_trusted_key("test-key", keypair.public_key)

        report = verifier.verify(
            artifact_path=artifact_path,
            manifest_path=manifest_path,
            signature_info=signature,
        )

        assert report.result == VerificationResult.VERIFIED
        assert report.signature_verified is True

    def test_verify_invalid_signature(self, sample_artifact, keypair):
        """Test rejection of invalid signature."""
        artifact_path, manifest_path = sample_artifact

        # Create invalid signature
        signature = SignatureInfo(
            algorithm="ed25519",
            signature="invalid_signature_base64",
            key_id="test-key",
        )

        verifier = ArtifactVerifier()
        verifier.add_trusted_key("test-key", keypair.public_key)

        report = verifier.verify(
            artifact_path=artifact_path,
            manifest_path=manifest_path,
            signature_info=signature,
        )

        assert report.result == VerificationResult.REJECTED
        assert report.rejection_reason == RejectionReason.INVALID_SIGNATURE

    def test_verify_digest_mismatch(self, sample_artifact, keypair):
        """Test rejection on digest mismatch."""
        artifact_path, manifest_path = sample_artifact

        verifier = ArtifactVerifier(strict_mode=False)

        report = verifier.verify(
            artifact_path=artifact_path,
            manifest_path=manifest_path,
            expected_digest="sha256:" + "f" * 64,
        )

        assert report.result == VerificationResult.REJECTED
        assert report.rejection_reason == RejectionReason.DIGEST_MISMATCH

    def test_verify_unknown_registry(self, sample_artifact):
        """Test rejection of unknown registry."""
        artifact_path, manifest_path = sample_artifact

        verifier = ArtifactVerifier(
            allowed_registries={"trusted-registry"},
            strict_mode=False,
        )

        report = verifier.verify(
            artifact_path=artifact_path,
            manifest_path=manifest_path,
            registry_source="unknown-registry",
        )

        assert report.result == VerificationResult.REJECTED
        assert report.rejection_reason == RejectionReason.UNKNOWN_REGISTRY

    def test_verify_incompatible_schema(self, tmp_path):
        """Test rejection of incompatible schema version."""
        # Create artifact with old schema
        artifact_path = tmp_path / "artifact.zip"
        artifact_path.write_bytes(b"test")

        manifest = {
            "schema_version": "0.1.0",  # Too old
            "artifact_id": "test",
            "entrypoint": {"module": "m", "class": "C"},
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        verifier = ArtifactVerifier(strict_mode=False)

        report = verifier.verify(
            artifact_path=artifact_path,
            manifest_path=manifest_path,
        )

        assert report.result == VerificationResult.REJECTED
        assert report.rejection_reason == RejectionReason.INCOMPATIBLE_SCHEMA

    def test_verify_missing_schema_version(self, tmp_path):
        """Test rejection when schema version is missing."""
        artifact_path = tmp_path / "artifact.zip"
        artifact_path.write_bytes(b"test")

        manifest = {
            "artifact_id": "test",
            "entrypoint": {"module": "m", "class": "C"},
            # Missing schema_version
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        verifier = ArtifactVerifier(strict_mode=False)

        report = verifier.verify(
            artifact_path=artifact_path,
            manifest_path=manifest_path,
        )

        assert report.result == VerificationResult.REJECTED
        assert report.rejection_reason == RejectionReason.UNKNOWN_SCHEMA_VERSION

    def test_verify_prohibited_content(self, tmp_path):
        """Test rejection of manifest with order-like fields."""
        artifact_path = tmp_path / "artifact.zip"
        artifact_path.write_bytes(b"test")

        manifest = {
            "schema_version": "1.0.0",
            "artifact_id": "test",
            "entrypoint": {"module": "m", "class": "C"},
            "order": {  # Prohibited!
                "side": "BUY",
                "quantity": 100,
            },
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        verifier = ArtifactVerifier(strict_mode=False)

        report = verifier.verify(
            artifact_path=artifact_path,
            manifest_path=manifest_path,
        )

        assert report.result == VerificationResult.REJECTED
        assert report.rejection_reason == RejectionReason.PROHIBITED_CONTENT

    def test_verify_revoked_key(self, sample_artifact, keypair):
        """Test rejection when key is revoked."""
        artifact_path, manifest_path = sample_artifact

        # Sign artifact
        signer = ArtifactSigner.from_keypair(keypair)
        signature = signer.sign_file(artifact_path)

        # Setup verifier with revoked key
        verifier = ArtifactVerifier()
        verifier.add_trusted_key("test-key", keypair.public_key)
        verifier.revoke_key("test-key")

        report = verifier.verify(
            artifact_path=artifact_path,
            manifest_path=manifest_path,
            signature_info=signature,
        )

        assert report.result == VerificationResult.REJECTED
        assert report.rejection_reason == RejectionReason.REVOKED_KEY

    def test_verify_quick(self, sample_artifact, keypair):
        """Test quick verification."""
        artifact_path, manifest_path = sample_artifact

        # Sign artifact
        signer = ArtifactSigner.from_keypair(keypair)
        signature = signer.sign_file(artifact_path)

        # Quick verify
        verifier = ArtifactVerifier()
        verifier.add_trusted_key("test-key", keypair.public_key)

        expected_digest = compute_file_digest(artifact_path)
        result = verifier.verify_quick(artifact_path, expected_digest, signature)

        assert result is True

    def test_verify_quick_fails_wrong_digest(self, sample_artifact, keypair):
        """Test quick verification fails on wrong digest."""
        artifact_path, manifest_path = sample_artifact

        signer = ArtifactSigner.from_keypair(keypair)
        signature = signer.sign_file(artifact_path)

        verifier = ArtifactVerifier()
        verifier.add_trusted_key("test-key", keypair.public_key)

        result = verifier.verify_quick(artifact_path, "sha256:" + "0" * 64, signature)

        assert result is False


class TestVerificationCache:
    """Tests for VerificationCache."""

    def test_cache_put_get(self):
        """Test putting and getting from cache."""
        cache = VerificationCache()

        report = VerificationReport(
            result=VerificationResult.VERIFIED,
            artifact_digest="sha256:abc",
        )

        cache.put("sha256:abc", report)
        cached = cache.get("sha256:abc")

        assert cached is not None
        assert cached.artifact_digest == "sha256:abc"

    def test_cache_miss(self):
        """Test cache miss."""
        cache = VerificationCache()

        result = cache.get("sha256:nonexistent")
        assert result is None

    def test_cache_eviction(self):
        """Test cache eviction when full."""
        cache = VerificationCache(max_entries=2)

        for i in range(3):
            report = VerificationReport(
                result=VerificationResult.VERIFIED,
                artifact_digest=f"sha256:{i}",
            )
            cache.put(f"sha256:{i}", report)

        # First entry should be evicted
        assert cache.get("sha256:0") is None
        assert cache.get("sha256:1") is not None
        assert cache.get("sha256:2") is not None

    def test_cache_invalidate(self):
        """Test cache invalidation."""
        cache = VerificationCache()

        report = VerificationReport(
            result=VerificationResult.VERIFIED,
            artifact_digest="sha256:abc",
        )
        cache.put("sha256:abc", report)

        cache.invalidate("sha256:abc")

        assert cache.get("sha256:abc") is None

    def test_cache_clear(self):
        """Test clearing cache."""
        cache = VerificationCache()

        for i in range(3):
            report = VerificationReport(
                result=VerificationResult.VERIFIED,
                artifact_digest=f"sha256:{i}",
            )
            cache.put(f"sha256:{i}", report)

        cache.clear()

        for i in range(3):
            assert cache.get(f"sha256:{i}") is None


class TestVerificationReport:
    """Tests for VerificationReport."""

    def test_report_creation(self):
        """Test report creation."""
        report = VerificationReport(
            result=VerificationResult.VERIFIED,
            artifact_digest="sha256:abc",
            signature_verified=True,
        )

        assert report.result == VerificationResult.VERIFIED
        assert report.signature_verified is True

    def test_report_to_dict(self):
        """Test report serialization."""
        report = VerificationReport(
            result=VerificationResult.REJECTED,
            artifact_digest="sha256:abc",
            rejection_reason=RejectionReason.UNSIGNED,
            rejection_details="No signature",
        )

        data = report.to_dict()

        assert data["result"] == "rejected"
        assert data["rejection_reason"] == "unsigned"
        assert data["artifact_digest"] == "sha256:abc"
