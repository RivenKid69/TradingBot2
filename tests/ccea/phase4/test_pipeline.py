# -*- coding: utf-8 -*-
"""
Tests for Publish Pipeline.

Per Design Doc Phase 4:
- Without signature, artifact is NOT published
- Agent does NOT run artifact without successful verification
- Complete pipeline: build → sbom → sign → validate → publish
"""

import json
import pytest
import tempfile
from pathlib import Path

from ccea.artifact.pipeline import (
    PublishPipeline,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    ArtifactFormat,
    build_and_publish,
)
from ccea.artifact.registry import ArtifactRegistry
from ccea.crypto.keys import generate_keypair, KeyAlgorithm


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_config(self, tmp_path):
        """Test default configuration."""
        (tmp_path / "strategy.py").write_text("class Strategy: pass")

        config = PipelineConfig(
            source_dir=tmp_path,
            entrypoint_module="strategy",
            entrypoint_class="Strategy",
        )

        assert config.output_format == ArtifactFormat.OCI
        assert config.version == "1.0.0"
        assert config.generate_sbom is True
        assert config.sign_artifact is True

    def test_custom_config(self, tmp_path):
        """Test custom configuration."""
        (tmp_path / "my_strategy.py").write_text("class MyStrategy: pass")

        config = PipelineConfig(
            source_dir=tmp_path,
            entrypoint_module="my_strategy",
            entrypoint_class="MyStrategy",
            artifact_id="custom-artifact",
            version="2.0.0",
            output_format=ArtifactFormat.ZIP,
            labels={"env": "test"},
        )

        assert config.artifact_id == "custom-artifact"
        assert config.version == "2.0.0"
        assert config.output_format == ArtifactFormat.ZIP


class TestPublishPipeline:
    """Tests for PublishPipeline."""

    @pytest.fixture
    def keypair(self):
        """Generate test keypair."""
        return generate_keypair(algorithm=KeyAlgorithm.ED25519, key_id="test-key")

    @pytest.fixture
    def sample_source(self, tmp_path):
        """Create sample source directory."""
        src = tmp_path / "src"
        src.mkdir()

        (src / "__init__.py").write_text("# Package\n")
        (src / "strategy.py").write_text(
            '''
"""Test strategy."""
class TestStrategy:
    """Simple test strategy."""
    def __init__(self, config=None):
        self.config = config or {}

    def run(self):
        return {"status": "ok"}
'''
        )
        return src

    @pytest.fixture
    def registry(self, tmp_path):
        """Create registry."""
        return ArtifactRegistry(storage_path=tmp_path / "registry")

    def test_pipeline_initialization(self, keypair, registry):
        """Test pipeline initialization."""
        pipeline = PublishPipeline(
            registry=registry,
            signing_key=keypair,
        )

        assert pipeline.registry is registry
        assert pipeline.signing_key is keypair

    def test_execute_with_signature(self, sample_source, keypair, tmp_path):
        """Test full pipeline execution with signature."""
        config = PipelineConfig(
            source_dir=sample_source,
            entrypoint_module="strategy",
            entrypoint_class="TestStrategy",
            version="1.0.0",
            output_format=ArtifactFormat.ZIP,
            output_dir=tmp_path / "output",
        )

        pipeline = PublishPipeline(signing_key=keypair)
        result = pipeline.execute(config)

        assert result.success is True
        assert result.stage == PipelineStage.COMPLETE
        assert result.signature is not None
        assert result.artifact_digest.startswith("sha256:")

    def test_execute_without_signature_fails(self, sample_source, tmp_path):
        """Test that pipeline fails without signature when required."""
        config = PipelineConfig(
            source_dir=sample_source,
            entrypoint_module="strategy",
            entrypoint_class="TestStrategy",
            output_format=ArtifactFormat.ZIP,
            output_dir=tmp_path / "output",
        )

        # No signing key provided
        pipeline = PublishPipeline(signing_key=None, require_signature=True)
        result = pipeline.execute(config)

        assert result.success is False
        assert result.stage == PipelineStage.SIGN
        assert any("signature" in e.lower() for e in result.errors)

    def test_execute_sign_disabled_fails_when_required(self, sample_source, keypair, tmp_path):
        """Test that disabling sign fails when signature required."""
        config = PipelineConfig(
            source_dir=sample_source,
            entrypoint_module="strategy",
            entrypoint_class="TestStrategy",
            output_format=ArtifactFormat.ZIP,
            output_dir=tmp_path / "output",
            sign_artifact=False,  # Disabled
        )

        pipeline = PublishPipeline(signing_key=keypair, require_signature=True)
        result = pipeline.execute(config)

        assert result.success is False
        assert "unsigned" in str(result.errors).lower()

    def test_execute_generates_sbom(self, sample_source, keypair, tmp_path):
        """Test that SBOM is generated."""
        config = PipelineConfig(
            source_dir=sample_source,
            entrypoint_module="strategy",
            entrypoint_class="TestStrategy",
            output_format=ArtifactFormat.ZIP,
            output_dir=tmp_path / "output",
            generate_sbom=True,
        )

        pipeline = PublishPipeline(signing_key=keypair)
        result = pipeline.execute(config)

        assert result.success is True
        assert result.sbom_path is not None
        assert result.sbom_path.exists()
        assert result.sbom_digest is not None

    def test_execute_validates_manifest(self, sample_source, keypair, tmp_path):
        """Test that manifest is validated."""
        config = PipelineConfig(
            source_dir=sample_source,
            entrypoint_module="strategy",
            entrypoint_class="TestStrategy",
            output_format=ArtifactFormat.ZIP,
            output_dir=tmp_path / "output",
            validate_manifest=True,
        )

        pipeline = PublishPipeline(signing_key=keypair)
        result = pipeline.execute(config)

        assert result.success is True

    def test_execute_publishes_to_registry(self, sample_source, keypair, tmp_path, registry):
        """Test publishing to registry."""
        config = PipelineConfig(
            source_dir=sample_source,
            entrypoint_module="strategy",
            entrypoint_class="TestStrategy",
            artifact_id="test-strategy",
            version="1.0.0",
            output_format=ArtifactFormat.ZIP,
            output_dir=tmp_path / "output",
        )

        pipeline = PublishPipeline(registry=registry, signing_key=keypair)
        result = pipeline.execute(config)

        assert result.success is True
        assert result.registry_entry is not None
        assert result.registry_entry.artifact_id == "test-strategy"

        # Verify in registry
        entry = registry.get("test-strategy", "1.0.0")
        assert entry is not None

    def test_execute_with_requirements(self, sample_source, keypair, tmp_path):
        """Test with requirements file."""
        req_file = sample_source / "requirements.txt"
        req_file.write_text("numpy==1.24.0\n")

        config = PipelineConfig(
            source_dir=sample_source,
            entrypoint_module="strategy",
            entrypoint_class="TestStrategy",
            requirements_file=req_file,
            output_format=ArtifactFormat.ZIP,
            output_dir=tmp_path / "output",
        )

        pipeline = PublishPipeline(signing_key=keypair)
        result = pipeline.execute(config)

        assert result.success is True

    def test_execute_oci_format(self, sample_source, keypair, tmp_path):
        """Test OCI format output."""
        config = PipelineConfig(
            source_dir=sample_source,
            entrypoint_module="strategy",
            entrypoint_class="TestStrategy",
            output_format=ArtifactFormat.OCI,
            output_dir=tmp_path / "output",
        )

        pipeline = PublishPipeline(signing_key=keypair)
        result = pipeline.execute(config)

        # Should succeed (with fallback to ZIP if OCI fails)
        assert result.success is True

    def test_execute_invalid_source(self, keypair, tmp_path):
        """Test with invalid source directory."""
        config = PipelineConfig(
            source_dir=tmp_path / "nonexistent",
            entrypoint_module="strategy",
            entrypoint_class="Strategy",
            output_dir=tmp_path / "output",
        )

        pipeline = PublishPipeline(signing_key=keypair)
        result = pipeline.execute(config)

        assert result.success is False
        assert result.stage == PipelineStage.BUILD

    def test_result_to_dict(self, sample_source, keypair, tmp_path):
        """Test result serialization."""
        config = PipelineConfig(
            source_dir=sample_source,
            entrypoint_module="strategy",
            entrypoint_class="TestStrategy",
            output_format=ArtifactFormat.ZIP,
            output_dir=tmp_path / "output",
        )

        pipeline = PublishPipeline(signing_key=keypair)
        result = pipeline.execute(config)

        data = result.to_dict()

        assert "success" in data
        assert "artifact_digest" in data
        assert "started_at" in data


class TestBuildAndPublish:
    """Tests for build_and_publish convenience function."""

    def test_build_and_publish(self, tmp_path):
        """Test convenience function."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "strategy.py").write_text("class Strategy: pass")

        keypair = generate_keypair()

        result = build_and_publish(
            source_dir=src,
            entrypoint_module="strategy",
            entrypoint_class="Strategy",
            signing_key=keypair,
            output_format=ArtifactFormat.ZIP,
            output_dir=tmp_path / "output",
        )

        assert result.success is True
        assert result.signature is not None
