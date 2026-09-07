# -*- coding: utf-8 -*-
"""
Tests for SBOM Generator.

Per Design Doc Phase 4:
- SBOM (CycloneDX/SPDX) generation
- sbom_ref in manifest
- Dependency analysis
"""

import json
import pytest
from pathlib import Path

from ccea.artifact.sbom import (
    SBOMGenerator,
    SBOMFormat,
    SBOM,
    Component,
    ComponentType,
    ComponentScope,
    License,
    Dependency,
    generate_sbom,
)


class TestComponent:
    """Tests for Component."""

    def test_component_creation(self):
        """Test component creation."""
        component = Component(
            name="numpy",
            version="1.24.0",
            type=ComponentType.LIBRARY,
        )

        assert component.name == "numpy"
        assert component.version == "1.24.0"
        assert component.type == ComponentType.LIBRARY
        assert component.bom_ref == "numpy@1.24.0"

    def test_component_with_purl(self):
        """Test component with Package URL."""
        component = Component(
            name="numpy",
            version="1.24.0",
            purl="pkg:pypi/numpy@1.24.0",
        )

        assert component.purl == "pkg:pypi/numpy@1.24.0"

    def test_component_to_cyclonedx(self):
        """Test CycloneDX serialization."""
        component = Component(
            name="numpy",
            version="1.24.0",
            type=ComponentType.LIBRARY,
            description="Numerical Python",
            purl="pkg:pypi/numpy@1.24.0",
            licenses=[License(id="BSD-3-Clause")],
        )

        data = component.to_cyclonedx()

        assert data["type"] == "library"
        assert data["name"] == "numpy"
        assert data["version"] == "1.24.0"
        assert data["purl"] == "pkg:pypi/numpy@1.24.0"
        assert "licenses" in data

    def test_component_to_spdx(self):
        """Test SPDX serialization."""
        component = Component(
            name="numpy",
            version="1.24.0",
            purl="pkg:pypi/numpy@1.24.0",
            licenses=[License(id="BSD-3-Clause")],
        )

        data = component.to_spdx()

        assert data["name"] == "numpy"
        assert data["versionInfo"] == "1.24.0"
        assert data["licenseConcluded"] == "BSD-3-Clause"
        assert "externalRefs" in data


class TestSBOM:
    """Tests for SBOM."""

    def test_sbom_creation(self):
        """Test SBOM creation."""
        sbom = SBOM(serial_number="urn:uuid:test-123")

        assert sbom.serial_number == "urn:uuid:test-123"
        assert sbom.version == 1
        assert len(sbom.components) == 0

    def test_add_component(self):
        """Test adding component to SBOM."""
        sbom = SBOM(serial_number="urn:uuid:test")

        component = Component(name="test-lib", version="1.0.0")
        sbom.add_component(component)

        assert len(sbom.components) == 1
        assert sbom.components[0].name == "test-lib"

    def test_add_dependency(self):
        """Test adding dependency."""
        sbom = SBOM(serial_number="urn:uuid:test")

        sbom.add_dependency("app@1.0", ["lib1@1.0", "lib2@2.0"])

        assert len(sbom.dependencies) == 1
        assert sbom.dependencies[0].ref == "app@1.0"
        assert len(sbom.dependencies[0].depends_on) == 2

    def test_to_cyclonedx(self):
        """Test CycloneDX format output."""
        sbom = SBOM(
            serial_number="urn:uuid:test-123",
            metadata={"component": {"name": "test-app", "version": "1.0.0"}},
        )

        sbom.add_component(Component(name="numpy", version="1.24.0"))
        sbom.add_dependency("test-app@1.0.0", ["numpy@1.24.0"])

        data = sbom.to_cyclonedx()

        assert data["bomFormat"] == "CycloneDX"
        assert data["specVersion"] == "1.5"
        assert data["serialNumber"] == "urn:uuid:test-123"
        assert len(data["components"]) == 1
        assert len(data["dependencies"]) == 1

    def test_to_spdx(self):
        """Test SPDX format output."""
        sbom = SBOM(
            serial_number="test-123",
            metadata={"component": {"name": "test-app"}},
        )

        sbom.add_component(
            Component(
                name="test-app",
                version="1.0.0",
                type=ComponentType.APPLICATION,
            )
        )

        data = sbom.to_spdx()

        assert data["spdxVersion"] == "SPDX-2.3"
        assert data["dataLicense"] == "CC0-1.0"
        assert "packages" in data
        assert "relationships" in data


class TestSBOMGenerator:
    """Tests for SBOMGenerator."""

    @pytest.fixture
    def sample_source(self, tmp_path):
        """Create sample source directory."""
        src = tmp_path / "src"
        src.mkdir()

        (src / "strategy.py").write_text("class Strategy: pass")
        (src / "requirements.txt").write_text(
            """
# Dependencies
numpy==1.24.0
pandas>=2.0.0,<3.0.0
scikit-learn
"""
        )
        return src

    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = SBOMGenerator()
        assert generator.include_transitive is True

    def test_generate_basic_sbom(self, sample_source):
        """Test basic SBOM generation."""
        generator = SBOMGenerator()

        sbom = generator.generate(
            source_dir=sample_source,
            artifact_name="test-strategy",
            artifact_version="1.0.0",
        )

        assert sbom is not None
        assert sbom.serial_number.startswith("urn:uuid:")
        assert len(sbom.components) > 0

        # Check main component
        main = sbom.metadata["component"]
        assert main["name"] == "test-strategy"
        assert main["version"] == "1.0.0"

    def test_parse_requirements_txt(self, sample_source):
        """Test parsing requirements.txt."""
        generator = SBOMGenerator()

        sbom = generator.generate(
            source_dir=sample_source,
            artifact_name="test",
            artifact_version="1.0.0",
            requirements_file=sample_source / "requirements.txt",
        )

        # Should have numpy, pandas, scikit-learn components
        names = {c.name for c in sbom.components}
        assert "numpy" in names
        assert "pandas" in names
        assert "scikit-learn" in names

    def test_parse_pyproject_toml(self, tmp_path):
        """Test parsing pyproject.toml."""
        src = tmp_path / "src"
        src.mkdir()

        pyproject = src / "pyproject.toml"
        pyproject.write_text(
            """
[project]
name = "my-project"
version = "1.0.0"
dependencies = [
    "requests>=2.28.0",
    "pydantic>=2.0.0",
]
"""
        )

        generator = SBOMGenerator()
        sbom = generator.generate(
            source_dir=src,
            artifact_name="test",
            artifact_version="1.0.0",
        )

        names = {c.name for c in sbom.components}
        assert "requests" in names
        assert "pydantic" in names

    def test_includes_python_runtime(self, tmp_path):
        """Test that Python runtime is included."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "app.py").write_text("print('hello')")

        generator = SBOMGenerator()
        sbom = generator.generate(
            source_dir=src,
            artifact_name="test",
            artifact_version="1.0.0",
        )

        names = {c.name for c in sbom.components}
        assert "python" in names

    def test_write_cyclonedx_json(self, sample_source, tmp_path):
        """Test writing CycloneDX JSON."""
        generator = SBOMGenerator()

        sbom = generator.generate(
            source_dir=sample_source,
            artifact_name="test",
            artifact_version="1.0.0",
        )

        output = tmp_path / "sbom.json"
        digest = generator.write(sbom, output, SBOMFormat.CYCLONEDX_JSON)

        assert output.exists()
        assert digest.startswith("sha256:")

        # Verify JSON is valid
        data = json.loads(output.read_text())
        assert data["bomFormat"] == "CycloneDX"

    def test_write_spdx_json(self, sample_source, tmp_path):
        """Test writing SPDX JSON."""
        generator = SBOMGenerator()

        sbom = generator.generate(
            source_dir=sample_source,
            artifact_name="test",
            artifact_version="1.0.0",
        )

        output = tmp_path / "sbom.spdx.json"
        generator.write(sbom, output, SBOMFormat.SPDX_JSON)

        assert output.exists()

        data = json.loads(output.read_text())
        assert data["spdxVersion"] == "SPDX-2.3"


class TestConvenienceFunction:
    """Tests for generate_sbom convenience function."""

    def test_generate_sbom_function(self, tmp_path):
        """Test generate_sbom convenience function."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "app.py").write_text("class App: pass")
        (src / "requirements.txt").write_text("requests==2.28.0\n")

        output = tmp_path / "sbom.json"

        digest = generate_sbom(
            source_dir=src,
            artifact_name="my-app",
            artifact_version="1.0.0",
            output_path=output,
            requirements_file=src / "requirements.txt",
        )

        assert output.exists()
        assert digest.startswith("sha256:")

        data = json.loads(output.read_text())
        assert data["bomFormat"] == "CycloneDX"

        # Check component is present
        components = data["components"]
        names = {c["name"] for c in components}
        assert "requests" in names
