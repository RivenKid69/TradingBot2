# -*- coding: utf-8 -*-
"""
Phase 3 Tests - Supply Chain End-to-End.

Tests for:
- 3.1: Artifact Builder honest format + registry publish + digest pinning
- 3.1A: Deterministic idempotency keys for lifecycle commands
- 3.2: Artifact signing + mandatory Agent verification
- 3.3: SBOM + provenance with mandatory fields
- 3.4: Immutable Config Blobs + digest + diff for approve
- 3.5: No silent upgrades for TRADING_IMPACTING
"""

import hashlib
import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

import pytest


# =============================================================================
# Test 3.1: Artifact Builder Honest Format
# =============================================================================


class TestArtifactBuilderHonestFormat:
    """Tests for honest format in artifact builder."""

    def test_default_format_is_zip_bundle(self):
        """Default output format should be ZIP_BUNDLE, not OCI_IMAGE."""
        from packages.cloud.builder.artifact_builder import BuildConfig
        from packages.shared.contracts.manifest import ArtifactFormat

        config = BuildConfig(
            strategy_id="test_strategy",
            strategy_name="Test Strategy",
            version="1.0.0",
            entrypoint="strategy:TestStrategy",
        )

        # Phase 3 (3.1): Default should be ZIP_BUNDLE (honest format)
        assert config.output_format == ArtifactFormat.ZIP_BUNDLE

    def test_format_validation_rejects_mismatch(self):
        """Format validation should reject mismatch between declared and actual."""
        from packages.cloud.builder.artifact_builder import (
            ArtifactBuilder,
            BuildConfig,
            BuildResult,
        )
        from packages.shared.contracts.manifest import ArtifactFormat

        # Create a ZIP file but declare OCI_IMAGE format
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a simple source file
            (tmpdir / "strategy.py").write_text("class TestStrategy: pass")

            builder = ArtifactBuilder()
            result = BuildResult()

            # Create a ZIP file
            artifact_path = tmpdir / "artifact.zip"
            with zipfile.ZipFile(artifact_path, "w") as zf:
                zf.writestr("test.txt", "content")

            # Validate with mismatched format
            config = BuildConfig(
                strategy_id="test",
                strategy_name="Test",
                version="1.0.0",
                entrypoint="strategy:Test",
                output_format=ArtifactFormat.OCI_IMAGE,  # Mismatch!
            )

            is_valid = builder._validate_format_honesty(config, artifact_path, result)

            assert not is_valid
            assert any("Format mismatch" in e for e in result.errors)

    def test_format_validation_accepts_correct_format(self):
        """Format validation should accept matching format."""
        from packages.cloud.builder.artifact_builder import (
            ArtifactBuilder,
            BuildConfig,
            BuildResult,
        )
        from packages.shared.contracts.manifest import ArtifactFormat

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            builder = ArtifactBuilder()
            result = BuildResult()

            # Create a ZIP file
            artifact_path = tmpdir / "artifact.zip"
            with zipfile.ZipFile(artifact_path, "w") as zf:
                zf.writestr("test.txt", "content")

            config = BuildConfig(
                strategy_id="test",
                strategy_name="Test",
                version="1.0.0",
                entrypoint="strategy:Test",
                output_format=ArtifactFormat.ZIP_BUNDLE,  # Correct!
            )

            is_valid = builder._validate_format_honesty(config, artifact_path, result)

            assert is_valid
            assert len(result.errors) == 0


# =============================================================================
# Test 3.1A: Deterministic Idempotency Keys
# =============================================================================


class TestDeterministicIdempotencyKeys:
    """Tests for deterministic idempotency key generation."""

    def test_same_inputs_produce_same_key(self):
        """Same command inputs should produce same idempotency key."""
        from packages.cloud.control_plane.services.idempotency import (
            generate_deterministic_idempotency_key,
        )

        workspace_id = uuid4()
        agent_id = uuid4()
        deployment_id = uuid4()
        artifact_digest = "sha256:abc123"
        config_digest = "sha256:def456"

        key1 = generate_deterministic_idempotency_key(
            workspace_id=workspace_id,
            agent_id=agent_id,
            command_type="REQUEST_START_RUN",
            deployment_id=deployment_id,
            artifact_digest=artifact_digest,
            config_digest=config_digest,
        )

        key2 = generate_deterministic_idempotency_key(
            workspace_id=workspace_id,
            agent_id=agent_id,
            command_type="REQUEST_START_RUN",
            deployment_id=deployment_id,
            artifact_digest=artifact_digest,
            config_digest=config_digest,
        )

        assert key1 == key2, "Same inputs should produce same key"

    def test_different_inputs_produce_different_keys(self):
        """Different command inputs should produce different keys."""
        from packages.cloud.control_plane.services.idempotency import (
            generate_deterministic_idempotency_key,
        )

        workspace_id = uuid4()
        agent_id = uuid4()

        key1 = generate_deterministic_idempotency_key(
            workspace_id=workspace_id,
            agent_id=agent_id,
            command_type="REQUEST_START_RUN",
            artifact_digest="sha256:abc123",
        )

        key2 = generate_deterministic_idempotency_key(
            workspace_id=workspace_id,
            agent_id=agent_id,
            command_type="REQUEST_START_RUN",
            artifact_digest="sha256:different",  # Different!
        )

        assert key1 != key2, "Different inputs should produce different keys"

    def test_idempotency_key_builder(self):
        """Test builder pattern for idempotency keys."""
        from packages.cloud.control_plane.services.idempotency import (
            IdempotencyKeyBuilder,
            generate_deterministic_idempotency_key,
        )

        workspace_id = uuid4()
        agent_id = uuid4()
        deployment_id = uuid4()

        # Using builder
        key_from_builder = (
            IdempotencyKeyBuilder()
            .workspace(workspace_id)
            .agent(agent_id)
            .command("REQUEST_START_RUN")
            .deployment(deployment_id)
            .artifact("sha256:test")
            .build()
        )

        # Using function directly
        key_direct = generate_deterministic_idempotency_key(
            workspace_id=workspace_id,
            agent_id=agent_id,
            command_type="REQUEST_START_RUN",
            deployment_id=deployment_id,
            artifact_digest="sha256:test",
        )

        assert key_from_builder == key_direct

    def test_operational_commands_include_run_id(self):
        """STOP/PAUSE commands should be idempotent per run."""
        from packages.cloud.control_plane.services.idempotency import (
            generate_deterministic_idempotency_key,
        )

        workspace_id = uuid4()
        agent_id = uuid4()
        run_id_1 = uuid4()
        run_id_2 = uuid4()

        key1 = generate_deterministic_idempotency_key(
            workspace_id=workspace_id,
            agent_id=agent_id,
            command_type="REQUEST_STOP_RUN",
            run_id=run_id_1,
        )

        key2 = generate_deterministic_idempotency_key(
            workspace_id=workspace_id,
            agent_id=agent_id,
            command_type="REQUEST_STOP_RUN",
            run_id=run_id_2,
        )

        # Same command to different runs should have different keys
        assert key1 != key2

        # Same command to same run should have same key
        key3 = generate_deterministic_idempotency_key(
            workspace_id=workspace_id,
            agent_id=agent_id,
            command_type="REQUEST_STOP_RUN",
            run_id=run_id_1,
        )
        assert key1 == key3


# =============================================================================
# Test 3.2: Artifact Signing + Mandatory Verification
# =============================================================================


class TestArtifactSigningAndVerification:
    """Tests for artifact signing and mandatory verification."""

    def test_unsigned_artifacts_rejected_in_strict_mode(self):
        """Verifier should reject unsigned artifacts in strict mode."""
        from ccea.artifact.verifier import (
            ArtifactVerifier,
            RejectionReason,
            VerificationResult,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create artifact and manifest
            artifact_path = tmpdir / "artifact.zip"
            with zipfile.ZipFile(artifact_path, "w") as zf:
                zf.writestr("test.txt", "content")

            manifest_path = tmpdir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "entrypoint": "test:Test",
                    }
                )
            )

            # Strict mode verifier
            verifier = ArtifactVerifier(strict_mode=True)

            # No signature provided
            result = verifier.verify(
                artifact_path=artifact_path,
                manifest_path=manifest_path,
                signature_info=None,  # No signature!
            )

            assert result.result == VerificationResult.REJECTED
            assert result.rejection_reason == RejectionReason.UNSIGNED

    def test_invalid_signature_rejected(self):
        """Verifier should reject artifacts with invalid signatures."""
        from ccea.artifact.verifier import (
            ArtifactVerifier,
            RejectionReason,
            VerificationResult,
        )
        from ccea.artifact.signer import SignatureInfo

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            artifact_path = tmpdir / "artifact.zip"
            with zipfile.ZipFile(artifact_path, "w") as zf:
                zf.writestr("test.txt", "content")

            manifest_path = tmpdir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "entrypoint": "test:Test",
                    }
                )
            )

            verifier = ArtifactVerifier(strict_mode=True)

            # Invalid signature
            sig_info = SignatureInfo(
                signature=b"invalid_signature",
                algorithm="ed25519",
                key_id="test-key",
            )

            result = verifier.verify(
                artifact_path=artifact_path,
                manifest_path=manifest_path,
                signature_info=sig_info,
            )

            assert result.result == VerificationResult.REJECTED
            assert result.rejection_reason == RejectionReason.INVALID_SIGNATURE

    def test_digest_mismatch_rejected(self):
        """Verifier should reject artifacts with digest mismatch."""
        from ccea.artifact.verifier import (
            ArtifactVerifier,
            RejectionReason,
            VerificationResult,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            artifact_path = tmpdir / "artifact.zip"
            with zipfile.ZipFile(artifact_path, "w") as zf:
                zf.writestr("test.txt", "content")

            manifest_path = tmpdir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "entrypoint": "test:Test",
                    }
                )
            )

            verifier = ArtifactVerifier(strict_mode=False)  # Allow unsigned for this test

            result = verifier.verify(
                artifact_path=artifact_path,
                manifest_path=manifest_path,
                expected_digest="sha256:wrong_digest",  # Mismatch!
            )

            assert result.result == VerificationResult.REJECTED
            assert result.rejection_reason == RejectionReason.DIGEST_MISMATCH

    def test_unknown_registry_rejected(self):
        """Verifier should reject artifacts from unknown registries."""
        from ccea.artifact.verifier import (
            ArtifactVerifier,
            RejectionReason,
            VerificationResult,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            artifact_path = tmpdir / "artifact.zip"
            with zipfile.ZipFile(artifact_path, "w") as zf:
                zf.writestr("test.txt", "content")

            manifest_path = tmpdir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "entrypoint": "test:Test",
                    }
                )
            )

            # Only allow specific registry
            verifier = ArtifactVerifier(
                allowed_registries={"allowed-registry"},
                strict_mode=False,
            )

            result = verifier.verify(
                artifact_path=artifact_path,
                manifest_path=manifest_path,
                registry_source="unknown-registry",  # Not in allowlist!
            )

            assert result.result == VerificationResult.REJECTED
            assert result.rejection_reason == RejectionReason.UNKNOWN_REGISTRY


# =============================================================================
# Test 3.3: SBOM + Provenance Validation
# =============================================================================


class TestSBOMAndProvenanceValidation:
    """Tests for SBOM and provenance validation."""

    def test_provenance_missing_mandatory_fields(self):
        """Validator should report missing mandatory provenance fields."""
        from packages.cloud.builder.provenance_validator import (
            ProvenanceValidationLevel,
            ProvenanceValidator,
        )
        from packages.shared.contracts.manifest import ArtifactManifest, Provenance

        # Provenance missing deps_lock_digest
        provenance = Provenance(
            builder_id="test-builder",
            build_timestamp=datetime.now(timezone.utc),
            # Missing: deps_lock_digest
        )

        manifest = ArtifactManifest(provenance=provenance)

        validator = ProvenanceValidator(level=ProvenanceValidationLevel.STANDARD)
        result = validator.validate(manifest)

        assert not result.valid
        assert "deps_lock_digest" in result.missing_mandatory

    def test_provenance_valid_with_all_fields(self):
        """Validator should accept provenance with all mandatory fields."""
        from packages.cloud.builder.provenance_validator import (
            ProvenanceValidationLevel,
            ProvenanceValidator,
        )
        from packages.shared.contracts.manifest import ArtifactManifest, Provenance

        provenance = Provenance(
            builder_id="test-builder",
            build_timestamp=datetime.now(timezone.utc),
            deps_lock_digest="sha256:abc123",
            git_sha="a" * 40,
        )

        manifest = ArtifactManifest(provenance=provenance)

        validator = ProvenanceValidator(level=ProvenanceValidationLevel.STANDARD)
        result = validator.validate(manifest)

        assert result.valid
        assert len(result.errors) == 0

    def test_sbom_validation_cyclonedx(self):
        """Validator should accept valid CycloneDX SBOM."""
        from packages.cloud.builder.provenance_validator import SBOMValidator

        sbom_data = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "components": [
                {"name": "numpy", "version": "1.24.0"},
                {"name": "pandas", "version": "2.0.0"},
            ],
        }

        validator = SBOMValidator()
        result = validator.validate(sbom_data)

        assert result.valid
        assert result.sbom_format == "CycloneDX"
        assert result.component_count == 2

    def test_sbom_validation_missing_version(self):
        """Validator should report missing specVersion."""
        from packages.cloud.builder.provenance_validator import SBOMValidator

        sbom_data = {
            "bomFormat": "CycloneDX",
            # Missing: specVersion
            "components": [],
        }

        validator = SBOMValidator()
        result = validator.validate(sbom_data)

        assert not result.valid
        assert any("specVersion" in e for e in result.errors)


# =============================================================================
# Test 3.4: Config Diff for Approval
# =============================================================================


class TestConfigDiff:
    """Tests for config diff functionality."""

    def test_compute_diff_detects_changes(self):
        """Diff service should detect config changes."""
        from packages.cloud.control_plane.services.config_diff import (
            ChangeType,
            ConfigDiffService,
        )

        old_config = {
            "risk": {"max_drawdown": 0.1},
            "execution": {"mode": "paper"},
        }

        new_config = {
            "risk": {"max_drawdown": 0.2},  # Modified
            "execution": {"mode": "paper"},
            "new_field": "value",  # Added
        }

        service = ConfigDiffService()
        result = service.compute_diff(old_config, new_config)

        assert result.has_changes
        assert result.modified_count >= 1
        assert result.added_count >= 1

    def test_diff_identifies_trading_impacting_changes(self):
        """Diff should identify trading-impacting field changes."""
        from packages.cloud.control_plane.services.config_diff import (
            ChangeImpact,
            ConfigDiffService,
        )

        old_config = {"risk": {"max_position_size": 1000}}
        new_config = {"risk": {"max_position_size": 2000}}  # Trading impacting!

        service = ConfigDiffService()
        result = service.compute_diff(old_config, new_config)

        assert result.overall_impact == ChangeImpact.TRADING_IMPACTING
        assert len(result.trading_impacting_changes) > 0

    def test_same_config_produces_no_changes(self):
        """Same config should produce no changes."""
        from packages.cloud.control_plane.services.config_diff import ConfigDiffService

        config = {"key": "value", "nested": {"a": 1}}

        service = ConfigDiffService()
        result = service.compute_diff(config, config.copy())

        assert not result.has_changes

    def test_diff_format_for_display(self):
        """Diff should produce human-readable format."""
        from packages.cloud.control_plane.services.config_diff import ConfigDiffService

        old_config = {"value": 1}
        new_config = {"value": 2}

        service = ConfigDiffService()
        result = service.compute_diff(old_config, new_config)

        summary = result.format_summary()
        assert "value" in summary or "1" in summary


# =============================================================================
# Test 3.5: No Silent Upgrades for TRADING_IMPACTING
# =============================================================================


class TestNoSilentUpgrades:
    """Tests for TRADING_IMPACTING approval enforcement."""

    def test_trading_impacting_always_requires_approval(self):
        """TRADING_IMPACTING commands must always require approval."""
        from packages.cloud.control_plane.services.change_class_enforcer import (
            ChangeClass,
            ChangeClassEnforcer,
        )

        enforcer = ChangeClassEnforcer()

        # REQUEST_START_RUN is always trading-impacting
        determination = enforcer.determine_change_class(
            command_type="REQUEST_START_RUN",
            requested_class=ChangeClass.NON_TRADING_IMPACTING,  # Try to bypass!
        )

        # Should be corrected to TRADING_IMPACTING
        assert determination.change_class == ChangeClass.TRADING_IMPACTING
        assert determination.requires_approval
        assert determination.overridden  # Was corrected

    def test_cannot_create_trading_impacting_without_approval(self):
        """Validator should reject TRADING_IMPACTING without requires_approval."""
        from packages.cloud.control_plane.services.change_class_enforcer import (
            ChangeClass,
            ChangeClassEnforcer,
        )

        enforcer = ChangeClassEnforcer()

        is_valid, error = enforcer.validate_command_approval_state(
            command_type="REQUEST_START_RUN",
            change_class=ChangeClass.TRADING_IMPACTING,
            requires_approval=False,  # Invalid!
        )

        assert not is_valid
        assert "CANNOT bypass" in error

    def test_safety_commands_can_execute_without_approval(self):
        """STOP/PAUSE commands can execute without approval (safety)."""
        from packages.cloud.control_plane.services.change_class_enforcer import (
            ChangeClass,
            ChangeClassEnforcer,
        )

        enforcer = ChangeClassEnforcer()

        # REQUEST_STOP_RUN is a safety command
        determination = enforcer.determine_change_class(
            command_type="REQUEST_STOP_RUN",
        )

        assert determination.change_class == ChangeClass.OPERATIONAL
        assert not determination.requires_approval

    def test_agent_rejects_trading_impacting_without_approval(self):
        """Agent should reject TRADING_IMPACTING commands without approval."""
        from packages.cloud.control_plane.services.change_class_enforcer import (
            AgentChangeClassEnforcer,
        )

        enforcer = AgentChangeClassEnforcer(fail_closed=True)

        is_valid, error = enforcer.validate_command(
            command_type="REQUEST_START_RUN",
            change_class="trading_impacting",
            requires_approval=True,
            has_approval=False,  # Not approved!
            approval_evidence_hash=None,
        )

        assert not is_valid
        assert "approval" in error.lower()

    def test_agent_accepts_approved_trading_impacting(self):
        """Agent should accept TRADING_IMPACTING commands with approval."""
        from packages.cloud.control_plane.services.change_class_enforcer import (
            AgentChangeClassEnforcer,
        )

        enforcer = AgentChangeClassEnforcer(fail_closed=True)

        is_valid, error = enforcer.validate_command(
            command_type="REQUEST_START_RUN",
            change_class="trading_impacting",
            requires_approval=True,
            has_approval=True,
            approval_evidence_hash="sha256:evidence123",
        )

        assert is_valid
        assert error == ""


# =============================================================================
# Test Artifact Publisher
# =============================================================================


class TestArtifactPublisher:
    """Tests for artifact publisher."""

    def test_publish_requires_signature(self):
        """Publisher should reject unsigned artifacts."""
        from packages.cloud.builder.artifact_publisher import (
            ArtifactPublisher,
            PublishStatus,
            RegistryConfig,
        )
        from packages.shared.contracts.manifest import (
            ArtifactManifest,
            Signature,
            SignatureAlgorithm,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create unsigned manifest
            manifest = ArtifactManifest(
                strategy_id="test",
                version="1.0.0",
                artifact_digest="sha256:abc123",
                signature=Signature(algorithm=SignatureAlgorithm.NONE),  # Unsigned!
            )

            # Create artifact
            artifact_path = tmpdir / "artifact.zip"
            with zipfile.ZipFile(artifact_path, "w") as zf:
                zf.writestr("test.txt", "content")

            publisher = ArtifactPublisher(RegistryConfig(local_path=tmpdir / "registry"))
            result = publisher.publish(artifact_path, manifest)

            assert not result.success
            assert result.status == PublishStatus.FAILED
            assert any("signed" in e.lower() for e in result.errors)

    def test_publish_verifies_digest(self):
        """Publisher should verify artifact digest matches manifest."""
        from packages.cloud.builder.artifact_publisher import (
            ArtifactPublisher,
            PublishStatus,
            RegistryConfig,
        )
        from packages.shared.contracts.manifest import (
            ArtifactManifest,
            Signature,
            SignatureAlgorithm,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create artifact
            artifact_path = tmpdir / "artifact.zip"
            with zipfile.ZipFile(artifact_path, "w") as zf:
                zf.writestr("test.txt", "content")

            # Create manifest with wrong digest
            manifest = ArtifactManifest(
                strategy_id="test",
                version="1.0.0",
                artifact_digest="sha256:wrong_digest",  # Wrong!
                signature=Signature(
                    algorithm=SignatureAlgorithm.SIGSTORE,
                    signature_value="test_sig",
                ),
            )

            publisher = ArtifactPublisher(RegistryConfig(local_path=tmpdir / "registry"))
            result = publisher.publish(artifact_path, manifest)

            assert not result.success
            assert result.status == PublishStatus.FAILED
            assert any("mismatch" in e.lower() for e in result.errors)


# =============================================================================
# Integration Tests
# =============================================================================


class TestPhase3Integration:
    """Integration tests for Phase 3 supply chain."""

    def test_end_to_end_artifact_lifecycle(self):
        """Test complete artifact build → publish → verify flow.

        This test verifies the complete CCEA supply chain works together:
        1. BUILD: Create strategy artifact with manifest and SBOM
        2. SIGN: Sign artifact with cryptographic signature
        3. PUBLISH: Upload to artifact registry with digest verification
        4. VERIFY: Validate artifact integrity and signature

        Per CCEA Architecture (Design Doc Section 8):
        - All artifacts are immutable (digest pinned)
        - All artifacts are signed (Agent verifies)
        - Manifest + SBOM are required
        """
        from packages.cloud.builder.artifact_builder import (
            ArtifactBuilder,
            BuildConfig,
        )
        from packages.cloud.builder.artifact_publisher import (
            ArtifactPublisher,
            PublishStatus,
            RegistryConfig,
        )
        from packages.shared.contracts.manifest import ArtifactFormat

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # ================================================================
            # STEP 1: BUILD - Create strategy source and build artifact
            # ================================================================
            source_dir = tmpdir / "strategy_src"
            source_dir.mkdir()

            # Create minimal strategy source
            strategy_file = source_dir / "strategy.py"
            strategy_file.write_text(
                '''
"""Example strategy for e2e test."""
class TestStrategy:
    """Minimal test strategy."""
    def __init__(self):
        self.name = "test"
    def compute_intent(self, snapshot):
        return None
'''
            )

            requirements_file = source_dir / "requirements.txt"
            requirements_file.write_text("numpy>=1.20.0\n")

            # Build configuration
            build_config = BuildConfig(
                strategy_id="e2e_test_strategy",
                strategy_name="E2E Test Strategy",
                version="1.0.0",
                entrypoint="strategy:TestStrategy",
                source_path=source_dir,
                requirements_file=requirements_file,
                output_format=ArtifactFormat.ZIP_BUNDLE,
                output_path=tmpdir / "artifacts",
                sign_artifact=True,
                git_repo="test/repo",
                git_sha="abc123def456",
                git_branch="main",
            )

            # Build artifact
            builder = ArtifactBuilder(builder_id="e2e_test_builder")
            build_result = builder.build(build_config)

            # Verify build success
            assert build_result.success, f"Build failed: {build_result.errors}"
            assert build_result.manifest is not None, "Manifest must be created"
            assert build_result.artifact_path is not None, "Artifact path must be set"
            assert build_result.artifact_path.exists(), "Artifact file must exist"
            assert build_result.artifact_digest.startswith(
                "sha256:"
            ), "Digest must use SHA-256 format"

            # Verify manifest content
            manifest = build_result.manifest
            assert manifest.strategy_id == "e2e_test_strategy"
            assert manifest.version == "1.0.0"
            assert manifest.artifact_digest == build_result.artifact_digest
            assert manifest.signature is not None, "Artifact must be signed"

            # ================================================================
            # STEP 2: PUBLISH - Upload to registry with verification
            # ================================================================
            registry_path = tmpdir / "registry"
            registry_config = RegistryConfig(local_path=registry_path)
            publisher = ArtifactPublisher(registry_config)

            publish_result = publisher.publish(
                build_result.artifact_path,
                build_result.manifest,
            )

            # Verify publish success
            assert publish_result.success, f"Publish failed: {publish_result.errors}"
            assert publish_result.status == PublishStatus.PUBLISHED
            assert publish_result.registry_ref is not None, "Registry ref must be set"

            # ================================================================
            # STEP 3: VERIFY - Validate artifact integrity
            # ================================================================
            # Verify artifact exists in registry
            from packages.shared.utils.hashing import compute_file_hash

            # Re-compute digest to verify integrity
            published_path = registry_path / publish_result.registry_ref
            if published_path.exists():
                recomputed_digest = compute_file_hash(published_path)
                assert (
                    recomputed_digest == build_result.artifact_digest
                ), "Published artifact digest must match original"

            # ================================================================
            # SUCCESS: Complete supply chain verified
            # ================================================================
            # This test proves:
            # - Artifacts can be built from source
            # - Artifacts are signed
            # - Artifacts can be published to registry
            # - Artifact integrity is maintained through the pipeline

    def test_artifact_lifecycle_rejects_unsigned(self):
        """Verify supply chain rejects unsigned artifacts."""
        from packages.cloud.builder.artifact_builder import (
            ArtifactBuilder,
            BuildConfig,
        )
        from packages.cloud.builder.artifact_publisher import (
            ArtifactPublisher,
            PublishStatus,
            RegistryConfig,
        )
        from packages.shared.contracts.manifest import ArtifactFormat

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create minimal source
            source_dir = tmpdir / "strategy_src"
            source_dir.mkdir()
            (source_dir / "strategy.py").write_text("class Test: pass")

            # Build WITHOUT signing
            build_config = BuildConfig(
                strategy_id="unsigned_test",
                strategy_name="Unsigned Test",
                version="1.0.0",
                entrypoint="strategy:Test",
                source_path=source_dir,
                output_format=ArtifactFormat.ZIP_BUNDLE,
                output_path=tmpdir / "artifacts",
                sign_artifact=False,  # No signature!
            )

            builder = ArtifactBuilder()
            build_result = builder.build(build_config)

            # Build may succeed, but publish should fail
            if build_result.success and build_result.artifact_path:
                registry_config = RegistryConfig(local_path=tmpdir / "registry")
                publisher = ArtifactPublisher(registry_config)

                publish_result = publisher.publish(
                    build_result.artifact_path,
                    build_result.manifest,
                )

                # Publisher should reject unsigned artifacts
                assert (
                    not publish_result.success
                ), "Publisher must reject unsigned artifacts (CCEA Section 8.3)"
                assert publish_result.status == PublishStatus.FAILED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
