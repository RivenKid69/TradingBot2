# -*- coding: utf-8 -*-
"""
Tests for EU AI Act Technical Documentation System (Article 11, Annex IV).

Tests cover:
- TechnicalDocumentationGenerator - Main documentation functionality
- DocumentationSection - Section data structures
- ComplianceEvidence - Evidence management
- ChangeRecord - Change tracking
- Export functionality
"""

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services.ai_act.technical_documentation import (
    # Enums
    DocumentationSectionType,
    ComplianceStatus,
    ExportFormat,
    # Data structures
    DocumentationMetadata,
    ComplianceEvidence,
    DocumentationSection,
    ChangeRecord,
    # Configuration
    TechnicalDocumentationConfig,
    # Main class
    TechnicalDocumentationGenerator,
    # Factory
    create_technical_documentation_generator,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for documentation output."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def doc_config(temp_output_dir):
    """Create a test documentation configuration."""
    return TechnicalDocumentationConfig(
        output_path=temp_output_dir,
        auto_generate_from_code=False,
        include_code_references=False,
        include_git_history=False,
        include_test_results=False,
        export_formats=["markdown", "json"],
    )


@pytest.fixture
def doc_generator(doc_config):
    """Create a test documentation generator."""
    return TechnicalDocumentationGenerator(doc_config)


# =============================================================================
# DocumentationSectionType Tests
# =============================================================================


class TestDocumentationSectionType:
    """Tests for DocumentationSectionType enum."""

    def test_section_type_values(self):
        """Test section type values match Annex IV requirements."""
        assert DocumentationSectionType.GENERAL_DESCRIPTION.value == "general_description"
        assert DocumentationSectionType.ALGORITHM_AND_DATA.value == "algorithm_and_data"
        assert DocumentationSectionType.MONITORING_AND_CONTROL.value == "monitoring_and_control"
        assert DocumentationSectionType.PERFORMANCE_METRICS.value == "performance_metrics"
        assert DocumentationSectionType.RISK_MANAGEMENT.value == "risk_management"
        assert DocumentationSectionType.CHANGE_LOG.value == "change_log"

    def test_all_annex_iv_sections_covered(self):
        """Test that all Annex IV sections are covered."""
        # Annex IV has 6 main sections
        required_sections = [
            "general_description",
            "algorithm_and_data",
            "monitoring_and_control",
            "performance_metrics",
            "risk_management",
            "change_log",
        ]

        for section in required_sections:
            assert any(s.value == section for s in DocumentationSectionType)


class TestComplianceStatus:
    """Tests for ComplianceStatus enum."""

    def test_compliance_status_values(self):
        """Test compliance status values."""
        assert ComplianceStatus.COMPLIANT.value == "compliant"
        assert ComplianceStatus.PARTIALLY_COMPLIANT.value == "partially_compliant"
        assert ComplianceStatus.NON_COMPLIANT.value == "non_compliant"
        assert ComplianceStatus.UNDER_REVIEW.value == "under_review"


class TestExportFormat:
    """Tests for ExportFormat enum."""

    def test_export_format_values(self):
        """Test export format values."""
        assert ExportFormat.MARKDOWN.value == "markdown"
        assert ExportFormat.HTML.value == "html"
        assert ExportFormat.JSON.value == "json"


# =============================================================================
# DocumentationMetadata Tests
# =============================================================================


class TestDocumentationMetadata:
    """Tests for DocumentationMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating documentation metadata."""
        metadata = DocumentationMetadata(
            system_name="Test System",
            system_version="1.0.0",
            provider_name="Test Provider",
        )

        assert metadata.document_id.startswith("DOC-")
        assert metadata.system_name == "Test System"
        assert metadata.system_version == "1.0.0"
        assert metadata.provider_name == "Test Provider"

    def test_metadata_auto_timestamps(self):
        """Test metadata auto-generates timestamps."""
        metadata = DocumentationMetadata()

        assert metadata.created_at != ""
        assert metadata.last_updated != ""

    def test_metadata_default_classification(self):
        """Test metadata defaults to high-risk classification."""
        metadata = DocumentationMetadata()

        assert metadata.risk_classification == "high_risk"
        assert metadata.annex_iv_compliance is True


# =============================================================================
# ComplianceEvidence Tests
# =============================================================================


class TestComplianceEvidence:
    """Tests for ComplianceEvidence dataclass."""

    def test_create_evidence(self):
        """Test creating compliance evidence."""
        evidence = ComplianceEvidence(
            article_reference="Article 9",
            requirement_text="Risk management system required",
            evidence_description="Implementation of risk_management.py",
            evidence_type="code_reference",
            evidence_location="services/ai_act/risk_management.py",
        )

        assert evidence.evidence_id.startswith("EVD-")
        assert evidence.article_reference == "Article 9"
        assert evidence.evidence_type == "code_reference"

    def test_evidence_default_status(self):
        """Test evidence defaults to under review."""
        evidence = ComplianceEvidence()

        assert evidence.compliance_status == ComplianceStatus.UNDER_REVIEW.value


# =============================================================================
# DocumentationSection Tests
# =============================================================================


class TestDocumentationSection:
    """Tests for DocumentationSection dataclass."""

    def test_create_section(self):
        """Test creating a documentation section."""
        section = DocumentationSection(
            section_type=DocumentationSectionType.GENERAL_DESCRIPTION.value,
            title="General Description",
            content="This is the general description.",
        )

        assert section.section_id.startswith("SEC-")
        assert section.section_type == "general_description"
        assert section.title == "General Description"

    def test_section_to_markdown(self):
        """Test section conversion to Markdown."""
        section = DocumentationSection(
            title="Test Section",
            content="Test content",
            subsections=[
                {"level": 2, "title": "Subsection", "content": "Subsection content"}
            ],
        )

        markdown = section.to_markdown()

        assert "# Test Section" in markdown
        assert "Test content" in markdown
        assert "## Subsection" in markdown

    def test_section_evidence_refs(self):
        """Test section evidence references."""
        section = DocumentationSection(
            title="Test",
            evidence_refs=["EVD-001", "EVD-002"],
        )

        assert len(section.evidence_refs) == 2
        assert "EVD-001" in section.evidence_refs


# =============================================================================
# ChangeRecord Tests
# =============================================================================


class TestChangeRecord:
    """Tests for ChangeRecord dataclass."""

    def test_create_change_record(self):
        """Test creating a change record."""
        change = ChangeRecord(
            change_type="algorithm",
            description="Updated policy network architecture",
            rationale="Improve performance",
            impact_assessment="Low risk - backwards compatible",
        )

        assert change.change_id.startswith("CHG-")
        assert change.change_type == "algorithm"
        assert change.timestamp != ""

    def test_substantial_modification_flag(self):
        """Test substantial modification flag."""
        change = ChangeRecord(
            description="Major algorithm change",
            is_substantial_modification=True,
            requires_reassessment=True,
        )

        assert change.is_substantial_modification is True
        assert change.requires_reassessment is True


# =============================================================================
# TechnicalDocumentationConfig Tests
# =============================================================================


class TestTechnicalDocumentationConfig:
    """Tests for TechnicalDocumentationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TechnicalDocumentationConfig()

        assert config.output_path == "docs/compliance/technical_documentation"
        assert config.auto_generate_from_code is True
        assert "markdown" in config.export_formats
        assert "json" in config.export_formats

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TechnicalDocumentationConfig(
            output_path="/custom/path",
            max_change_history=50,
        )

        assert config.output_path == "/custom/path"
        assert config.max_change_history == 50


# =============================================================================
# TechnicalDocumentationGenerator Tests
# =============================================================================


class TestTechnicalDocumentationGenerator:
    """Tests for TechnicalDocumentationGenerator class."""

    def test_generator_initialization(self, doc_generator, temp_output_dir):
        """Test generator initialization."""
        assert doc_generator is not None
        assert Path(temp_output_dir).exists()

    def test_generate_general_description(self, doc_generator):
        """Test generating general description section."""
        section = doc_generator.generate_general_description(
            system_name="Test AI System",
            system_version="2.0.0",
            provider_name="Test Corp",
        )

        assert section is not None
        assert section.section_type == "general_description"
        assert "Test AI System" in section.content
        assert "2.0.0" in section.content

    def test_generate_algorithm_description(self, doc_generator):
        """Test generating algorithm description section."""
        section = doc_generator.generate_algorithm_description(
            algorithm_name="Custom RL Algorithm",
            algorithm_type="Reinforcement Learning",
        )

        assert section is not None
        assert section.section_type == "algorithm_and_data"
        assert "Custom RL Algorithm" in section.content

    def test_generate_monitoring_description(self, doc_generator):
        """Test generating monitoring description section."""
        section = doc_generator.generate_monitoring_description()

        assert section is not None
        assert section.section_type == "monitoring_and_control"
        assert "Human Oversight" in section.content

    def test_generate_performance_metrics(self, doc_generator):
        """Test generating performance metrics section."""
        section = doc_generator.generate_performance_metrics_description()

        assert section is not None
        assert section.section_type == "performance_metrics"
        assert "Sharpe" in section.content

    def test_generate_risk_management(self, doc_generator):
        """Test generating risk management section."""
        section = doc_generator.generate_risk_management_description()

        assert section is not None
        assert section.section_type == "risk_management"
        assert "Article 9" in section.content

    def test_generate_change_log(self, doc_generator):
        """Test generating change log section."""
        # Add some changes first
        doc_generator.record_change(ChangeRecord(
            change_type="config",
            description="Updated parameters",
        ))

        section = doc_generator.generate_change_log_section()

        assert section is not None
        assert section.section_type == "change_log"

    def test_generate_all(self, doc_generator):
        """Test generating all documentation sections."""
        sections = doc_generator.generate_all()

        assert len(sections) >= 6  # All Annex IV sections
        section_types = [s.section_type for s in sections.values()]
        assert "general_description" in section_types
        assert "algorithm_and_data" in section_types
        assert "risk_management" in section_types

    def test_add_evidence(self, doc_generator):
        """Test adding compliance evidence."""
        evidence = ComplianceEvidence(
            article_reference="Article 9",
            requirement_text="Risk management required",
            evidence_description="Implementation exists",
        )

        evidence_id = doc_generator.add_evidence(evidence)

        assert evidence_id in doc_generator.evidence
        assert doc_generator.evidence[evidence_id].article_reference == "Article 9"

    def test_link_evidence_to_section(self, doc_generator):
        """Test linking evidence to a section."""
        # Generate a section
        section = doc_generator.generate_general_description()

        # Add evidence
        evidence = ComplianceEvidence(article_reference="Annex IV.1")
        evidence_id = doc_generator.add_evidence(evidence)

        # Link them
        result = doc_generator.link_evidence_to_section(section.section_id, evidence_id)

        assert result is True
        assert evidence_id in section.evidence_refs

    def test_record_change(self, doc_generator):
        """Test recording a change."""
        change = ChangeRecord(
            change_type="algorithm",
            description="Updated model architecture",
        )

        change_id = doc_generator.record_change(change)

        assert len(doc_generator.change_log) == 1
        assert doc_generator.change_log[0].change_id == change_id

    def test_get_substantial_modifications(self, doc_generator):
        """Test getting substantial modifications."""
        # Add a regular change
        doc_generator.record_change(ChangeRecord(
            description="Minor fix",
            is_substantial_modification=False,
        ))

        # Add a substantial modification
        doc_generator.record_change(ChangeRecord(
            description="Major change",
            is_substantial_modification=True,
        ))

        substantial = doc_generator.get_substantial_modifications()

        assert len(substantial) == 1
        assert substantial[0].description == "Major change"

    def test_generate_compliance_matrix(self, doc_generator):
        """Test generating compliance matrix."""
        # Generate all sections
        doc_generator.generate_all()

        matrix = doc_generator.generate_compliance_matrix()

        assert "Annex IV.1 - General Description" in matrix
        assert "Annex IV.5 - Risk Management" in matrix


# =============================================================================
# Export Tests
# =============================================================================


class TestExport:
    """Tests for documentation export functionality."""

    def test_export_markdown(self, doc_generator, temp_output_dir):
        """Test exporting to Markdown format."""
        doc_generator.generate_all()
        output_path = doc_generator.export(ExportFormat.MARKDOWN, temp_output_dir)

        assert Path(output_path).exists()
        assert (Path(output_path) / "README.md").exists()

    def test_export_json(self, doc_generator, temp_output_dir):
        """Test exporting to JSON format."""
        doc_generator.generate_all()
        output_path = doc_generator.export(ExportFormat.JSON, temp_output_dir)

        json_file = Path(output_path)
        assert json_file.exists()

        # Verify JSON structure
        with open(json_file) as f:
            data = json.load(f)
            assert "metadata" in data
            assert "sections" in data
            assert "compliance_matrix" in data

    def test_export_html(self, doc_generator, temp_output_dir):
        """Test exporting to HTML format."""
        doc_generator.generate_all()
        output_path = doc_generator.export(ExportFormat.HTML, temp_output_dir)

        assert Path(output_path).exists()
        with open(output_path) as f:
            content = f.read()
            assert "<html" in content
            assert "Technical Documentation" in content

    def test_export_with_string_format(self, doc_generator, temp_output_dir):
        """Test exporting with string format argument."""
        doc_generator.generate_all()
        output_path = doc_generator.export("json", temp_output_dir)

        assert Path(output_path).exists()


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_generator_default(self):
        """Test creating generator with default config."""
        generator = create_technical_documentation_generator()

        assert generator is not None
        assert isinstance(generator, TechnicalDocumentationGenerator)

    def test_create_generator_with_config(self, temp_output_dir):
        """Test creating generator with custom config."""
        config = TechnicalDocumentationConfig(
            output_path=temp_output_dir,
            max_change_history=50,
        )

        generator = create_technical_documentation_generator(config)

        assert generator.config.max_change_history == 50

    def test_create_generator_with_dict(self, temp_output_dir):
        """Test creating generator with dict config."""
        generator = create_technical_documentation_generator({
            "output_path": temp_output_dir,
            "include_git_history": False,
        })

        assert generator.config.include_git_history is False


# =============================================================================
# Article 11 Compliance Tests
# =============================================================================


class TestArticle11Compliance:
    """Tests for Article 11 compliance requirements."""

    def test_documentation_kept_up_to_date(self, doc_generator):
        """Test that documentation can be updated (Article 11(1))."""
        # Generate initial documentation
        doc_generator.generate_all()
        initial_sections = len(doc_generator.sections)

        # Re-generate (should update)
        doc_generator.generate_all()

        # Sections should still exist
        assert len(doc_generator.sections) >= initial_sections

    def test_documentation_available_to_authorities(self, doc_generator, temp_output_dir):
        """Test documentation can be exported for authorities (Article 11(2))."""
        doc_generator.generate_all()

        # Export in multiple formats for authority access
        md_path = doc_generator.export(ExportFormat.MARKDOWN, temp_output_dir)
        json_path = doc_generator.export(ExportFormat.JSON, temp_output_dir)

        assert Path(md_path).exists()
        assert Path(json_path).exists()

    def test_annex_iv_complete_documentation(self, doc_generator):
        """Test all Annex IV elements are documented."""
        doc_generator.generate_all()

        # Check all required Annex IV sections exist
        section_types = {s.section_type for s in doc_generator.sections.values()}

        required = [
            "general_description",  # Annex IV.1
            "algorithm_and_data",   # Annex IV.2
            "monitoring_and_control",  # Annex IV.3
            "performance_metrics",  # Annex IV.4
            "risk_management",      # Annex IV.5
            "change_log",           # Annex IV.6
        ]

        for req in required:
            assert req in section_types, f"Missing required section: {req}"


# =============================================================================
# Annex IV Compliance Tests
# =============================================================================


class TestAnnexIVCompliance:
    """Tests for Annex IV specific requirements."""

    def test_general_description_content(self, doc_generator):
        """Test general description has required elements (Annex IV.1)."""
        section = doc_generator.generate_general_description(
            system_name="Test System",
            system_version="1.0.0",
            provider_name="Test Provider",
        )

        content = section.content

        # Check required Annex IV.1 elements
        assert "Test System" in content  # System name
        assert "1.0.0" in content  # Version
        assert "Intended Purpose" in content
        assert "Target Users" in content

    def test_algorithm_description_content(self, doc_generator):
        """Test algorithm description has required elements (Annex IV.2)."""
        section = doc_generator.generate_algorithm_description()
        content = section.content

        # Check required Annex IV.2 elements
        assert "Algorithm" in content
        assert "Training" in content
        assert "Data" in content

    def test_risk_management_content(self, doc_generator):
        """Test risk management description has required elements (Annex IV.5)."""
        section = doc_generator.generate_risk_management_description()
        content = section.content

        # Check required Annex IV.5 elements
        assert "Risk" in content
        assert "Mitigation" in content
        assert "Article 9" in content

    def test_change_log_records_modifications(self, doc_generator):
        """Test change log properly records modifications (Annex IV.6)."""
        # Record several changes
        for i in range(5):
            doc_generator.record_change(ChangeRecord(
                change_type="config",
                description=f"Change {i}",
            ))

        section = doc_generator.generate_change_log_section()

        # Change log section should exist and reference change process
        assert section is not None
        assert "Change Management" in section.content


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_generator(self):
        """Test generator with no sections."""
        generator = create_technical_documentation_generator()

        matrix = generator.generate_compliance_matrix()
        assert matrix is not None

    def test_link_nonexistent_section(self, doc_generator):
        """Test linking evidence to nonexistent section."""
        evidence = ComplianceEvidence()
        evidence_id = doc_generator.add_evidence(evidence)

        result = doc_generator.link_evidence_to_section("nonexistent", evidence_id)
        assert result is False

    def test_link_nonexistent_evidence(self, doc_generator):
        """Test linking nonexistent evidence to section."""
        section = doc_generator.generate_general_description()

        result = doc_generator.link_evidence_to_section(section.section_id, "nonexistent")
        assert result is False

    def test_change_log_limit(self, doc_generator):
        """Test change log respects maximum limit."""
        doc_generator.config.max_change_history = 5

        # Record more than max changes
        for i in range(10):
            doc_generator.record_change(ChangeRecord(description=f"Change {i}"))

        assert len(doc_generator.change_log) == 5

    def test_metadata_update(self, doc_generator):
        """Test metadata is updated during generation."""
        doc_generator.generate_general_description(
            system_name="Updated System",
            system_version="3.0.0",
        )

        assert doc_generator.metadata.system_name == "Updated System"
        assert doc_generator.metadata.system_version == "3.0.0"
