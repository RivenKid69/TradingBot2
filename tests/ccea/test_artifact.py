# -*- coding: utf-8 -*-
"""
Tests for CCEA Artifact Module.

Tests:
- Artifact building
- Manifest creation and validation
- Artifact signing and verification
- Artifact registry
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from ccea.artifact.builder import (
    ArtifactBuilder,
    BuildConfig,
    build_hello_strategy,
)
from ccea.artifact.manifest import (
    ManifestBuilder,
    ManifestValidator,
)
from ccea.artifact.signer import (
    ArtifactSigner,
    SignatureVerifier,
    SignatureInfo,
)
from ccea.artifact.registry import (
    ArtifactRegistry,
    RegistryEntry,
)
from ccea.models.manifest import ArtifactType, ChangeClass
from ccea.crypto.keys import generate_keypair


class TestArtifactBuilder:
    """Tests for ArtifactBuilder."""

    def test_build_hello_strategy(self):
        """Test building hello strategy artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            package = build_hello_strategy(output_dir)

            assert package.artifact_id == "hello_strategy"
            assert package.package_path.exists()
            assert package.manifest_path.exists()
            assert package.digest.startswith("sha256:")

    def test_build_with_signing(self):
        """Test building artifact with signing."""
        keypair = generate_keypair()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            package = build_hello_strategy(output_dir, signing_key=keypair)

            assert package.signature is not None
            assert package.manifest.signature is not None

    def test_build_creates_sbom(self):
        """Test that build creates SBOM."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            package = build_hello_strategy(output_dir)

            assert package.sbom_path is not None
            assert package.sbom_path.exists()

    def test_manifest_has_no_broker_requirement(self):
        """Test hello strategy doesn't require broker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            package = build_hello_strategy(output_dir)

            if package.manifest.live_capabilities:
                assert package.manifest.live_capabilities.requires_broker_access is False


class TestManifestBuilder:
    """Tests for ManifestBuilder."""

    def test_build_manifest(self):
        """Test building manifest with builder."""
        manifest = (
            ManifestBuilder()
            .set_artifact_id("test_artifact")
            .set_type(ArtifactType.STRATEGY)
            .set_name("Test Strategy")
            .set_entrypoint("test.module", "TestClass")
            .set_runtime("3.11")
            .set_deps_lock_digest("sha256:0000000000000000000000000000000000000000000000000000000000000000")
            .build()
        )

        assert manifest.artifact_id == "test_artifact"
        assert manifest.artifact_type == ArtifactType.STRATEGY
        assert manifest.name == "Test Strategy"

    def test_build_manifest_with_provenance(self):
        """Test manifest with provenance."""
        manifest = (
            ManifestBuilder()
            .set_artifact_id("provenance_test")
            .set_type(ArtifactType.MODEL)
            .set_entrypoint("model.module", "Model")
            .set_runtime("3.11")
            .set_deps_lock_digest("sha256:1111111111111111111111111111111111111111111111111111111111111111")
            .set_provenance(
                git_sha="a" * 40,
                git_branch="main",
            )
            .build()
        )

        assert manifest.provenance is not None
        assert manifest.provenance.git_sha == "a" * 40


class TestManifestValidator:
    """Tests for ManifestValidator."""

    def test_validate_valid_manifest(self):
        """Test validation of valid manifest."""
        validator = ManifestValidator()

        manifest = (
            ManifestBuilder()
            .set_artifact_id("valid_artifact")
            .set_type(ArtifactType.STRATEGY)
            .set_entrypoint("module", "Class")
            .set_runtime("3.11")
            .set_deps_lock_digest("sha256:2222222222222222222222222222222222222222222222222222222222222222")
            .build()
        )

        errors = validator.validate(manifest)
        assert len(errors) == 0

    def test_validate_missing_artifact_id(self):
        """Test validation catches missing artifact_id."""
        from pydantic import ValidationError
        from ccea.models.manifest import ArtifactManifest, Entrypoint, Runtime

        # Pydantic validation should catch empty artifact_id
        with pytest.raises(ValidationError) as exc_info:
            ArtifactManifest(
                schema_version="1.0.0",
                artifact_id="",  # Empty - should be rejected
                artifact_type=ArtifactType.STRATEGY,
                entrypoint=Entrypoint(module="test", class_name="Test"),
                runtime=Runtime(python_version="3.11"),
                deps_lock_digest="sha256:3333333333333333333333333333333333333333333333333333333333333333",
                created_at=datetime.utcnow(),
            )

        assert "artifact_id" in str(exc_info.value)

    def test_validate_prohibited_fields(self):
        """Test validator detects prohibited fields."""
        validator = ManifestValidator()

        # Simulate a manifest with prohibited field
        # (Real manifests prevent this, but we test the validator logic)
        prohibited = validator._find_prohibited_fields({
            "normal": "data",
            "side": "BUY",  # Prohibited
        })

        assert "side" in prohibited


class TestArtifactSigner:
    """Tests for ArtifactSigner."""

    def test_sign_file(self):
        """Test signing a file."""
        keypair = generate_keypair()
        signer = ArtifactSigner.from_keypair(keypair)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"Test content for signing")
            f.flush()

            sig_info = signer.sign_file(Path(f.name))

            assert sig_info.signature is not None
            assert sig_info.signed_digest.startswith("sha256:")

    def test_sign_bytes(self):
        """Test signing raw bytes."""
        keypair = generate_keypair()
        signer = ArtifactSigner.from_keypair(keypair)

        data = b"Raw bytes to sign"
        sig_info = signer.sign_bytes(data)

        assert sig_info.signature is not None

    def test_signature_info_serialization(self):
        """Test SignatureInfo serialization."""
        sig = SignatureInfo(
            algorithm="ed25519",
            signature="base64signature",
            key_id="key_1",
            timestamp=datetime.utcnow(),
            signed_digest="sha256:abc123",
        )

        data = sig.to_dict()
        restored = SignatureInfo.from_dict(data)

        assert restored.algorithm == sig.algorithm
        assert restored.signature == sig.signature


class TestSignatureVerifier:
    """Tests for SignatureVerifier."""

    def test_verify_signed_file(self):
        """Test verifying signed file."""
        keypair = generate_keypair(key_id="test_key")
        signer = ArtifactSigner.from_keypair(keypair)
        verifier = SignatureVerifier()

        verifier.add_trusted_key("test_key", keypair.public_key)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"Content to verify")
            f.flush()

            sig_info = signer.sign_file(Path(f.name))
            is_valid = verifier.verify_file(Path(f.name), sig_info, "test_key")

            assert is_valid is True

    def test_verify_tampered_file(self):
        """Test that tampered file fails verification."""
        keypair = generate_keypair(key_id="tamper_key")
        signer = ArtifactSigner.from_keypair(keypair)
        verifier = SignatureVerifier()

        verifier.add_trusted_key("tamper_key", keypair.public_key)

        with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
            f.write("Original content")
            f.flush()

            sig_info = signer.sign_file(Path(f.name))

            # Tamper with file
            f.seek(0)
            f.write("Tampered content")
            f.flush()

            is_valid = verifier.verify_file(Path(f.name), sig_info, "tamper_key")

            assert is_valid is False

    def test_verify_with_wrong_key(self):
        """Test verification with wrong key fails."""
        keypair1 = generate_keypair(key_id="key_1")
        keypair2 = generate_keypair(key_id="key_2")

        signer = ArtifactSigner.from_keypair(keypair1)
        verifier = SignatureVerifier()

        # Add wrong key
        verifier.add_trusted_key("key_1", keypair2.public_key)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"Test content")
            f.flush()

            sig_info = signer.sign_file(Path(f.name))
            is_valid = verifier.verify_file(Path(f.name), sig_info, "key_1")

            assert is_valid is False


class TestArtifactRegistry:
    """Tests for ArtifactRegistry."""

    def test_push_and_pull_artifact(self):
        """Test pushing and pulling artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(Path(tmpdir) / "registry")

            # Create test artifact
            artifact_path = Path(tmpdir) / "test_artifact.zip"
            artifact_path.write_bytes(b"Test artifact content")

            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text('{"test": "manifest"}')

            entry = registry.push(
                artifact_path=artifact_path,
                manifest_path=manifest_path,
                artifact_id="test_artifact",
                version="1.0.0",
            )

            assert entry.artifact_id == "test_artifact"
            assert entry.digest.startswith("sha256:")

            # Pull artifact
            output_path = Path(tmpdir) / "pulled_artifact.zip"
            result = registry.pull(entry.digest, output_path)

            assert result is True
            assert output_path.exists()
            assert output_path.read_bytes() == b"Test artifact content"

    def test_get_artifact_by_version(self):
        """Test getting artifact by ID and version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(Path(tmpdir) / "registry")

            artifact_path = Path(tmpdir) / "artifact.zip"
            artifact_path.write_bytes(b"Content")

            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text('{}')

            registry.push(
                artifact_path=artifact_path,
                manifest_path=manifest_path,
                artifact_id="my_artifact",
                version="2.0.0",
            )

            entry = registry.get("my_artifact", "2.0.0")

            assert entry is not None
            assert entry.artifact_id == "my_artifact"
            assert entry.version == "2.0.0"

    def test_artifact_exists(self):
        """Test checking artifact existence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(Path(tmpdir) / "registry")

            artifact_path = Path(tmpdir) / "artifact.zip"
            artifact_path.write_bytes(b"Exists test")

            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text('{}')

            entry = registry.push(
                artifact_path=artifact_path,
                manifest_path=manifest_path,
                artifact_id="exists_test",
                version="1.0.0",
            )

            assert registry.exists(entry.digest) is True
            assert registry.exists("sha256:nonexistent") is False

    def test_list_artifacts(self):
        """Test listing artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(Path(tmpdir) / "registry")

            for i in range(3):
                artifact_path = Path(tmpdir) / f"artifact_{i}.zip"
                artifact_path.write_bytes(f"Content {i}".encode())

                manifest_path = Path(tmpdir) / f"manifest_{i}.json"
                manifest_path.write_text('{}')

                registry.push(
                    artifact_path=artifact_path,
                    manifest_path=manifest_path,
                    artifact_id=f"artifact_{i}",
                    version="1.0.0",
                    workspace_id="ws_test",
                )

            entries = registry.list(workspace_id="ws_test")

            assert len(entries) == 3

    def test_delete_artifact(self):
        """Test deleting artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(Path(tmpdir) / "registry")

            artifact_path = Path(tmpdir) / "deletable.zip"
            artifact_path.write_bytes(b"Deletable")

            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text('{}')

            registry.push(
                artifact_path=artifact_path,
                manifest_path=manifest_path,
                artifact_id="deletable",
                version="1.0.0",
            )

            result = registry.delete("deletable", "1.0.0")

            assert result is True
            assert registry.get("deletable", "1.0.0") is None
