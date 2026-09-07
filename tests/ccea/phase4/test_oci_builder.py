# -*- coding: utf-8 -*-
"""
Tests for OCI Image Builder.

Per Design Doc Phase 4:
- OCI image (digest-pinned) as primary format
- Digest pinning for immutability
- Proper OCI layout structure
"""

import json
import pytest
import tempfile
from pathlib import Path

from ccea.artifact.oci_builder import (
    OCIImageBuilder,
    OCIBuildConfig,
    OCIBuildResult,
    OCIManifest,
    OCILayer,
    OCIMediaType,
    Platform,
    create_oci_artifact,
)
from ccea.crypto.digest import compute_file_digest


class TestOCIBuildConfig:
    """Tests for OCI build configuration."""

    def test_default_config(self, tmp_path):
        """Test default configuration values."""
        (tmp_path / "strategy.py").write_text("class Strategy: pass")

        config = OCIBuildConfig(
            source_dir=tmp_path,
            entrypoint_module="strategy",
            entrypoint_class="Strategy",
        )

        assert config.image_name == "strategy"
        assert config.tag == "latest"
        assert config.platform == Platform.LINUX_AMD64
        assert config.python_version == "3.11"
        assert "*.py" in config.include_patterns

    def test_custom_config(self, tmp_path):
        """Test custom configuration."""
        (tmp_path / "my_strategy.py").write_text("class MyStrategy: pass")

        config = OCIBuildConfig(
            source_dir=tmp_path,
            entrypoint_module="my_strategy",
            entrypoint_class="MyStrategy",
            image_name="custom-image",
            tag="v1.0.0",
            platform=Platform.LINUX_ARM64,
            labels={"env": "production"},
        )

        assert config.image_name == "custom-image"
        assert config.tag == "v1.0.0"
        assert config.platform == Platform.LINUX_ARM64
        assert config.labels["env"] == "production"


class TestOCIManifest:
    """Tests for OCI manifest."""

    def test_manifest_creation(self):
        """Test manifest creation."""
        manifest = OCIManifest(
            config={
                "mediaType": OCIMediaType.CONFIG.value,
                "digest": "sha256:abc123",
                "size": 1000,
            },
            layers=[
                {
                    "mediaType": OCIMediaType.LAYER_TAR_GZIP.value,
                    "digest": "sha256:def456",
                    "size": 5000,
                }
            ],
        )

        assert manifest.schema_version == 2
        assert manifest.media_type == OCIMediaType.MANIFEST.value

    def test_manifest_to_dict(self):
        """Test manifest serialization."""
        manifest = OCIManifest(
            config={"digest": "sha256:abc"},
            layers=[{"digest": "sha256:def"}],
            annotations={"key": "value"},
        )

        data = manifest.to_dict()

        assert data["schemaVersion"] == 2
        assert data["config"]["digest"] == "sha256:abc"
        assert len(data["layers"]) == 1
        assert data["annotations"]["key"] == "value"

    def test_manifest_digest_computation(self):
        """Test manifest digest is deterministic."""
        manifest = OCIManifest(
            config={"digest": "sha256:abc"},
            layers=[{"digest": "sha256:def"}],
        )

        digest1 = manifest.compute_digest()
        digest2 = manifest.compute_digest()

        assert digest1 == digest2
        assert digest1.startswith("sha256:")


class TestOCIImageBuilder:
    """Tests for OCI Image Builder."""

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

    def test_builder_initialization(self):
        """Test builder initialization."""
        builder = OCIImageBuilder()
        assert builder.cache_dir.exists()
        assert builder.verify_base_images is True

    def test_build_basic_image(self, sample_source, tmp_path):
        """Test building basic OCI image."""
        output = tmp_path / "output"

        config = OCIBuildConfig(
            source_dir=sample_source,
            entrypoint_module="strategy",
            entrypoint_class="TestStrategy",
            image_name="test-strategy",
            tag="1.0.0",
            output_dir=output,
        )

        builder = OCIImageBuilder()
        result = builder.build(config)

        assert result is not None
        assert result.manifest_digest.startswith("sha256:")
        assert result.config_digest.startswith("sha256:")
        assert len(result.layers) >= 1
        assert result.output_dir.exists()

    def test_oci_layout_structure(self, sample_source, tmp_path):
        """Test OCI layout structure is correct."""
        output = tmp_path / "output"

        config = OCIBuildConfig(
            source_dir=sample_source,
            entrypoint_module="strategy",
            entrypoint_class="TestStrategy",
            output_dir=output,
        )

        builder = OCIImageBuilder()
        result = builder.build(config)

        # Check OCI layout files
        assert (result.output_dir / "oci-layout").exists()
        assert (result.output_dir / "index.json").exists()
        assert (result.output_dir / "blobs" / "sha256").exists()

        # Check oci-layout content
        oci_layout = json.loads((result.output_dir / "oci-layout").read_text())
        assert oci_layout["imageLayoutVersion"] == "1.0.0"

        # Check index.json
        index = json.loads((result.output_dir / "index.json").read_text())
        assert index["schemaVersion"] == 2
        assert len(index["manifests"]) == 1

    def test_digest_pinning(self, sample_source, tmp_path):
        """Test that digests are correctly pinned."""
        output = tmp_path / "output"

        config = OCIBuildConfig(
            source_dir=sample_source,
            entrypoint_module="strategy",
            entrypoint_class="TestStrategy",
            output_dir=output,
        )

        builder = OCIImageBuilder()
        result = builder.build(config)

        # All layer digests should be sha256
        for layer in result.layers:
            assert layer.digest.startswith("sha256:")
            # Verify blob exists
            hash_part = layer.digest.split(":")[1]
            blob_path = result.output_dir / "blobs" / "sha256" / hash_part
            assert blob_path.exists()

    def test_image_with_requirements(self, sample_source, tmp_path):
        """Test building image with requirements.txt."""
        # Create requirements.txt
        req_file = sample_source / "requirements.txt"
        req_file.write_text("numpy==1.24.0\npandas>=2.0.0\n")

        output = tmp_path / "output"

        config = OCIBuildConfig(
            source_dir=sample_source,
            entrypoint_module="strategy",
            entrypoint_class="TestStrategy",
            requirements_file=req_file,
            output_dir=output,
        )

        builder = OCIImageBuilder()
        result = builder.build(config)

        # Should have dependency layer
        assert len(result.layers) >= 2

        # Check that deps layer has correct annotation
        deps_layer = [
            l for l in result.layers if l.annotations.get("io.ccea.layer.type") == "dependencies"
        ]
        assert len(deps_layer) == 1

    def test_image_labels(self, sample_source, tmp_path):
        """Test custom labels in image."""
        output = tmp_path / "output"

        config = OCIBuildConfig(
            source_dir=sample_source,
            entrypoint_module="strategy",
            entrypoint_class="TestStrategy",
            labels={"env": "test", "version": "1.0"},
            output_dir=output,
        )

        builder = OCIImageBuilder()
        result = builder.build(config)

        # Check index.json has annotations
        index = json.loads((result.output_dir / "index.json").read_text())
        manifest_ref = index["manifests"][0]

        # Read manifest
        manifest_hash = manifest_ref["digest"].split(":")[1]
        manifest_path = result.output_dir / "blobs" / "sha256" / manifest_hash
        manifest = json.loads(manifest_path.read_text())

        assert "env" in manifest["annotations"]
        assert manifest["annotations"]["env"] == "test"

    def test_export_tar(self, sample_source, tmp_path):
        """Test exporting OCI image as tar."""
        output = tmp_path / "output"

        config = OCIBuildConfig(
            source_dir=sample_source,
            entrypoint_module="strategy",
            entrypoint_class="TestStrategy",
            output_dir=output,
        )

        builder = OCIImageBuilder()
        result = builder.build(config)

        tar_path = tmp_path / "image.tar"
        builder.export_tar(result, tar_path)

        assert tar_path.exists()
        assert tar_path.stat().st_size > 0

    def test_validation_errors(self, tmp_path):
        """Test validation errors for invalid config."""
        builder = OCIImageBuilder()

        # Non-existent source directory
        config = OCIBuildConfig(
            source_dir=tmp_path / "nonexistent",
            entrypoint_module="strategy",
            entrypoint_class="Strategy",
        )

        with pytest.raises(ValueError, match="Source directory not found"):
            builder.build(config)

    def test_convenience_function(self, sample_source, tmp_path):
        """Test create_oci_artifact convenience function."""
        output = tmp_path / "output"

        result = create_oci_artifact(
            source_dir=sample_source,
            entrypoint_module="strategy",
            entrypoint_class="TestStrategy",
            output_dir=output,
        )

        assert result is not None
        assert result.manifest_digest.startswith("sha256:")


class TestOCILayer:
    """Tests for OCI Layer."""

    def test_layer_creation(self):
        """Test layer creation."""
        layer = OCILayer(
            digest="sha256:abc123",
            size=1000,
            media_type=OCIMediaType.LAYER_TAR_GZIP,
        )

        assert layer.digest == "sha256:abc123"
        assert layer.size == 1000
        assert layer.media_type == OCIMediaType.LAYER_TAR_GZIP

    def test_layer_annotations(self):
        """Test layer with annotations."""
        layer = OCILayer(
            digest="sha256:abc123",
            size=1000,
            annotations={"io.ccea.layer.type": "application"},
        )

        assert layer.annotations["io.ccea.layer.type"] == "application"
