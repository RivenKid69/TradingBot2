# -*- coding: utf-8 -*-
"""
Tests for EU AI Act Conformity Assessment Module (Phase 4).

Tests cover:
- ConformitySelfAssessment - Self-assessment functionality
- ChecklistItem - Checklist management
- ComplianceGap - Gap tracking
- AssessmentReport - Report generation
- EUDeclaration - Declaration generation (Article 47)
- InstructionsForUse - IFU generation (Article 13)
- RegistrationInfo - Registration preparation (Article 49)

Target: 100% code coverage for services/ai_act/conformity_assessment.py
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services.ai_act.conformity_assessment import (
    # Enums
    ConformityStatus,
    ChecklistItemStatus,
    RequirementCategory,
    AssessmentPhase,
    GapSeverity,
    # Data structures
    ChecklistItem,
    ComplianceGap,
    AssessmentReport,
    EUDeclaration,
    InstructionsForUse,
    RegistrationInfo,
    ConformityAssessmentConfig,
    # Main class
    ConformitySelfAssessment,
    # Factory functions
    create_conformity_assessment,
    get_default_checklist,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def config(temp_dir):
    """Create test configuration."""
    return ConformityAssessmentConfig(
        organization_name="Test Organization",
        organization_address="123 Test Street",
        organization_contact="test@example.com",
        system_name="Test AI System",
        system_version="1.0.0",
        passing_score_threshold=80.0,
        critical_gap_tolerance=0,
        major_gap_tolerance=2,
        output_path=os.path.join(temp_dir, "output"),
        log_path=os.path.join(temp_dir, "logs"),
    )


@pytest.fixture
def assessment(config):
    """Create ConformitySelfAssessment instance."""
    return ConformitySelfAssessment(config)


@pytest.fixture
def initialized_assessment(assessment):
    """Create assessment with initialized checklist."""
    assessment.initialize_checklist()
    return assessment


# =============================================================================
# Test Enumerations
# =============================================================================


class TestConformityStatus:
    """Tests for ConformityStatus enum."""

    def test_all_statuses_exist(self):
        """Verify all conformity statuses exist."""
        assert ConformityStatus.NOT_STARTED.value == "not_started"
        assert ConformityStatus.IN_PROGRESS.value == "in_progress"
        assert ConformityStatus.PASSED.value == "passed"
        assert ConformityStatus.FAILED.value == "failed"
        assert ConformityStatus.CONDITIONAL.value == "conditional"
        assert ConformityStatus.REQUIRES_REMEDIATION.value == "requires_remediation"

    def test_status_count(self):
        """Verify correct number of statuses."""
        assert len(ConformityStatus) == 6


class TestChecklistItemStatus:
    """Tests for ChecklistItemStatus enum."""

    def test_all_item_statuses_exist(self):
        """Verify all checklist item statuses exist."""
        assert ChecklistItemStatus.NOT_CHECKED.value == "not_checked"
        assert ChecklistItemStatus.COMPLIANT.value == "compliant"
        assert ChecklistItemStatus.PARTIALLY_COMPLIANT.value == "partially_compliant"
        assert ChecklistItemStatus.NON_COMPLIANT.value == "non_compliant"
        assert ChecklistItemStatus.NOT_APPLICABLE.value == "not_applicable"

    def test_item_status_count(self):
        """Verify correct number of item statuses."""
        assert len(ChecklistItemStatus) == 5


class TestRequirementCategory:
    """Tests for RequirementCategory enum."""

    def test_all_categories_exist(self):
        """Verify all requirement categories exist."""
        assert RequirementCategory.RISK_MANAGEMENT.value == "risk_management"
        assert RequirementCategory.DATA_GOVERNANCE.value == "data_governance"
        assert RequirementCategory.TECHNICAL_DOCUMENTATION.value == "technical_documentation"
        assert RequirementCategory.RECORD_KEEPING.value == "record_keeping"
        assert RequirementCategory.TRANSPARENCY.value == "transparency"
        assert RequirementCategory.HUMAN_OVERSIGHT.value == "human_oversight"
        assert RequirementCategory.ACCURACY_ROBUSTNESS.value == "accuracy_robustness"
        assert RequirementCategory.QMS.value == "quality_management_system"
        assert RequirementCategory.CYBERSECURITY.value == "cybersecurity"

    def test_category_count(self):
        """Verify correct number of categories."""
        assert len(RequirementCategory) == 9


class TestAssessmentPhase:
    """Tests for AssessmentPhase enum."""

    def test_all_phases_exist(self):
        """Verify all assessment phases exist."""
        assert AssessmentPhase.PRE_ASSESSMENT.value == "pre_assessment"
        assert AssessmentPhase.DOCUMENTATION_REVIEW.value == "documentation_review"
        assert AssessmentPhase.TECHNICAL_VERIFICATION.value == "technical_verification"
        assert AssessmentPhase.TESTING.value == "testing"
        assert AssessmentPhase.FINAL_REVIEW.value == "final_review"
        assert AssessmentPhase.DECLARATION.value == "declaration"

    def test_phase_count(self):
        """Verify correct number of phases."""
        assert len(AssessmentPhase) == 6


class TestGapSeverity:
    """Tests for GapSeverity enum."""

    def test_all_severities_exist(self):
        """Verify all gap severities exist."""
        assert GapSeverity.CRITICAL.value == "critical"
        assert GapSeverity.MAJOR.value == "major"
        assert GapSeverity.MINOR.value == "minor"
        assert GapSeverity.OBSERVATION.value == "observation"

    def test_severity_count(self):
        """Verify correct number of severities."""
        assert len(GapSeverity) == 4


# =============================================================================
# Test Data Structures
# =============================================================================


class TestChecklistItem:
    """Tests for ChecklistItem dataclass."""

    def test_create_with_defaults(self):
        """Test creating checklist item with defaults."""
        item = ChecklistItem()
        assert item.item_id.startswith("CHK-")
        assert item.category == RequirementCategory.RISK_MANAGEMENT
        assert item.status == ChecklistItemStatus.NOT_CHECKED

    def test_create_with_values(self):
        """Test creating checklist item with values."""
        item = ChecklistItem(
            item_id="CHK-001",
            category=RequirementCategory.DATA_GOVERNANCE,
            article_reference="Article 10",
            requirement_text="Data quality requirement",
            description="Ensure data quality",
            status=ChecklistItemStatus.COMPLIANT,
        )
        assert item.item_id == "CHK-001"
        assert item.category == RequirementCategory.DATA_GOVERNANCE
        assert item.article_reference == "Article 10"
        assert item.status == ChecklistItemStatus.COMPLIANT

    def test_auto_generated_id(self):
        """Test auto-generated item ID."""
        item1 = ChecklistItem()
        item2 = ChecklistItem()
        assert item1.item_id != item2.item_id  # Should be unique


class TestComplianceGap:
    """Tests for ComplianceGap dataclass."""

    def test_create_with_defaults(self):
        """Test creating gap with defaults."""
        gap = ComplianceGap()
        assert gap.gap_id.startswith("GAP-")
        assert gap.severity == GapSeverity.MINOR
        assert gap.is_resolved == False
        assert gap.blocks_conformity == False  # MINOR doesn't block

    def test_critical_gap_blocks_conformity(self):
        """Test that critical gaps block conformity."""
        gap = ComplianceGap(severity=GapSeverity.CRITICAL)
        assert gap.blocks_conformity == True

    def test_major_gap_blocks_conformity(self):
        """Test that major gaps block conformity."""
        gap = ComplianceGap(severity=GapSeverity.MAJOR)
        assert gap.blocks_conformity == True

    def test_minor_gap_does_not_block(self):
        """Test that minor gaps don't block conformity."""
        gap = ComplianceGap(severity=GapSeverity.MINOR)
        assert gap.blocks_conformity == False

    def test_observation_does_not_block(self):
        """Test that observations don't block conformity."""
        gap = ComplianceGap(severity=GapSeverity.OBSERVATION)
        assert gap.blocks_conformity == False

    def test_auto_timestamp(self):
        """Test auto-generated timestamp."""
        gap = ComplianceGap()
        assert gap.identified_date is not None
        # Should be valid ISO format
        datetime.fromisoformat(gap.identified_date.replace("Z", "+00:00"))


class TestAssessmentReport:
    """Tests for AssessmentReport dataclass."""

    def test_create_with_defaults(self):
        """Test creating report with defaults."""
        report = AssessmentReport()
        assert report.report_id.startswith("AR-")
        assert report.overall_status == ConformityStatus.NOT_STARTED
        assert report.compliance_score == 0.0
        assert report.approved == False

    def test_create_with_values(self):
        """Test creating report with values."""
        report = AssessmentReport(
            assessor="Test Assessor",
            system_name="Test System",
            system_version="1.0.0",
            overall_status=ConformityStatus.PASSED,
            compliance_score=95.0,
            total_items=100,
            compliant_items=95,
        )
        assert report.assessor == "Test Assessor"
        assert report.compliance_score == 95.0
        assert report.compliant_items == 95


class TestEUDeclaration:
    """Tests for EUDeclaration dataclass."""

    def test_create_with_defaults(self):
        """Test creating declaration with defaults."""
        declaration = EUDeclaration()
        assert declaration.declaration_id.startswith("EU-DOC-")
        assert declaration.declaration_number.startswith("DOC-")
        assert declaration.is_valid == True

    def test_create_with_values(self):
        """Test creating declaration with values."""
        declaration = EUDeclaration(
            provider_name="Test Provider",
            system_name="Test System",
            signatory_name="John Doe",
            signatory_title="CEO",
        )
        assert declaration.provider_name == "Test Provider"
        assert declaration.signatory_name == "John Doe"


class TestInstructionsForUse:
    """Tests for InstructionsForUse dataclass."""

    def test_create_with_defaults(self):
        """Test creating IFU with defaults."""
        ifu = InstructionsForUse()
        assert ifu.document_id.startswith("IFU-")
        assert ifu.version == "1.0"

    def test_create_with_values(self):
        """Test creating IFU with values."""
        ifu = InstructionsForUse(
            provider_name="Test Provider",
            system_name="Test System",
            intended_purpose="Test purpose",
            capabilities=["Cap 1", "Cap 2"],
            limitations=["Lim 1", "Lim 2"],
        )
        assert ifu.provider_name == "Test Provider"
        assert len(ifu.capabilities) == 2
        assert len(ifu.limitations) == 2


class TestRegistrationInfo:
    """Tests for RegistrationInfo dataclass."""

    def test_create_with_defaults(self):
        """Test creating registration info with defaults."""
        reg = RegistrationInfo()
        assert reg.registration_id.startswith("REG-")
        assert reg.registration_status == "pending"
        assert reg.risk_classification == "high_risk"
        assert reg.ce_marking_affixed == False

    def test_create_with_values(self):
        """Test creating registration info with values."""
        reg = RegistrationInfo(
            provider_name="Test Provider",
            system_name="Test System",
            member_states_available=["Germany", "France"],
        )
        assert reg.provider_name == "Test Provider"
        assert len(reg.member_states_available) == 2


class TestConformityAssessmentConfig:
    """Tests for ConformityAssessmentConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ConformityAssessmentConfig()
        assert config.passing_score_threshold == 80.0
        assert config.critical_gap_tolerance == 0
        assert config.major_gap_tolerance == 2

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ConformityAssessmentConfig(
            organization_name="Custom Org",
            passing_score_threshold=90.0,
        )
        assert config.organization_name == "Custom Org"
        assert config.passing_score_threshold == 90.0


# =============================================================================
# Test ConformitySelfAssessment
# =============================================================================


class TestConformitySelfAssessmentInit:
    """Tests for ConformitySelfAssessment initialization."""

    def test_init_with_default_config(self, temp_dir):
        """Test initialization with default config."""
        with patch.object(Path, "mkdir"):
            assessment = ConformitySelfAssessment()
        assert assessment.config is not None

    def test_init_with_custom_config(self, config):
        """Test initialization with custom config."""
        assessment = ConformitySelfAssessment(config)
        assert assessment.config.organization_name == "Test Organization"

    def test_directories_created(self, config):
        """Test that output directories are created."""
        assessment = ConformitySelfAssessment(config)
        assert Path(config.output_path).exists()
        assert Path(config.log_path).exists()


class TestChecklistManagement:
    """Tests for checklist management."""

    def test_initialize_checklist(self, assessment):
        """Test checklist initialization."""
        items = assessment.initialize_checklist()
        assert len(items) > 0
        # Should have items for all categories
        categories = set(item.category for item in items)
        assert RequirementCategory.RISK_MANAGEMENT in categories
        assert RequirementCategory.DATA_GOVERNANCE in categories
        assert RequirementCategory.QMS in categories

    def test_get_checklist(self, initialized_assessment):
        """Test getting all checklist items."""
        items = initialized_assessment.get_checklist()
        assert len(items) > 0

    def test_get_checklist_by_category(self, initialized_assessment):
        """Test getting items by category."""
        items = initialized_assessment.get_checklist_by_category(
            RequirementCategory.RISK_MANAGEMENT
        )
        assert len(items) > 0
        for item in items:
            assert item.category == RequirementCategory.RISK_MANAGEMENT

    def test_assess_item_compliant(self, initialized_assessment):
        """Test assessing item as compliant."""
        items = initialized_assessment.get_checklist()
        item_id = items[0].item_id

        result = initialized_assessment.assess_item(
            item_id=item_id,
            status=ChecklistItemStatus.COMPLIANT,
            evidence_description="Test evidence",
            evidence_references=["doc1.pdf", "doc2.pdf"],
            assessed_by="Test Assessor",
            notes="Test notes",
        )

        assert result.status == ChecklistItemStatus.COMPLIANT
        assert result.evidence_description == "Test evidence"
        assert len(result.evidence_references) == 2
        assert result.assessed_by == "Test Assessor"
        assert result.assessment_date is not None

    def test_assess_item_non_compliant_creates_gap(self, initialized_assessment):
        """Test that non-compliant assessment creates a gap."""
        items = initialized_assessment.get_checklist()
        item_id = items[0].item_id

        initialized_assessment.assess_item(
            item_id=item_id,
            status=ChecklistItemStatus.NON_COMPLIANT,
        )

        gaps = initialized_assessment.get_gaps()
        assert len(gaps) > 0

    def test_assess_item_partially_compliant_creates_gap(self, initialized_assessment):
        """Test that partial compliance creates a gap."""
        items = initialized_assessment.get_checklist()
        item_id = items[0].item_id

        initialized_assessment.assess_item(
            item_id=item_id,
            status=ChecklistItemStatus.PARTIALLY_COMPLIANT,
        )

        gaps = initialized_assessment.get_gaps()
        assert len(gaps) > 0

    def test_assess_invalid_item_raises_error(self, initialized_assessment):
        """Test that assessing invalid item raises error."""
        with pytest.raises(ValueError, match="not found"):
            initialized_assessment.assess_item(
                item_id="INVALID-ID",
                status=ChecklistItemStatus.COMPLIANT,
            )


class TestGapManagement:
    """Tests for gap management."""

    def test_add_gap(self, assessment):
        """Test adding a gap."""
        gap = assessment.add_gap(
            category=RequirementCategory.RISK_MANAGEMENT,
            severity=GapSeverity.MAJOR,
            title="Test Gap",
            description="Test description",
            article_reference="Article 9",
            remediation_plan="Fix it",
            responsible_party="Test Team",
        )

        assert gap.gap_id.startswith("GAP-")
        assert gap.severity == GapSeverity.MAJOR
        assert gap.title == "Test Gap"

    def test_resolve_gap(self, assessment):
        """Test resolving a gap."""
        gap = assessment.add_gap(
            category=RequirementCategory.RISK_MANAGEMENT,
            severity=GapSeverity.MINOR,
            title="Test Gap",
            description="Test",
        )

        resolved = assessment.resolve_gap(
            gap_id=gap.gap_id,
            resolution_evidence="Fixed via update",
        )

        assert resolved.is_resolved == True
        assert resolved.resolution_date is not None
        assert resolved.resolution_evidence == "Fixed via update"

    def test_resolve_invalid_gap_raises_error(self, assessment):
        """Test resolving invalid gap raises error."""
        with pytest.raises(ValueError, match="not found"):
            assessment.resolve_gap("INVALID-GAP", "evidence")

    def test_get_gaps(self, assessment):
        """Test getting all gaps."""
        assessment.add_gap(
            category=RequirementCategory.RISK_MANAGEMENT,
            severity=GapSeverity.MINOR,
            title="Gap 1",
            description="Desc 1",
        )
        assessment.add_gap(
            category=RequirementCategory.DATA_GOVERNANCE,
            severity=GapSeverity.MAJOR,
            title="Gap 2",
            description="Desc 2",
        )

        gaps = assessment.get_gaps()
        assert len(gaps) == 2

    def test_get_open_gaps(self, assessment):
        """Test getting open gaps."""
        gap1 = assessment.add_gap(
            category=RequirementCategory.RISK_MANAGEMENT,
            severity=GapSeverity.MINOR,
            title="Gap 1",
            description="Desc 1",
        )
        gap2 = assessment.add_gap(
            category=RequirementCategory.DATA_GOVERNANCE,
            severity=GapSeverity.MAJOR,
            title="Gap 2",
            description="Desc 2",
        )

        assessment.resolve_gap(gap1.gap_id, "Resolved")

        open_gaps = assessment.get_open_gaps()
        assert len(open_gaps) == 1
        assert open_gaps[0].gap_id == gap2.gap_id

    def test_get_blocking_gaps(self, assessment):
        """Test getting blocking gaps."""
        assessment.add_gap(
            category=RequirementCategory.RISK_MANAGEMENT,
            severity=GapSeverity.CRITICAL,
            title="Critical Gap",
            description="Desc",
        )
        assessment.add_gap(
            category=RequirementCategory.DATA_GOVERNANCE,
            severity=GapSeverity.MINOR,
            title="Minor Gap",
            description="Desc",
        )

        blocking = assessment.get_blocking_gaps()
        assert len(blocking) == 1
        assert blocking[0].severity == GapSeverity.CRITICAL


class TestAssessmentExecution:
    """Tests for assessment execution."""

    def test_run_full_assessment(self, initialized_assessment):
        """Test running full assessment."""
        report = initialized_assessment.run_full_assessment(
            assessor="Test Assessor",
            auto_assess=True,
        )

        assert report.report_id is not None
        assert report.assessor == "Test Assessor"
        assert report.total_items > 0
        assert report.compliance_score >= 0

    def test_assessment_passed(self, initialized_assessment):
        """Test assessment that passes."""
        # Auto-assess will mark all as compliant
        report = initialized_assessment.run_full_assessment(auto_assess=True)
        assert report.overall_status == ConformityStatus.PASSED
        assert report.compliance_score >= 80.0

    def test_assessment_failed_with_critical_gap(self, initialized_assessment):
        """Test assessment fails with critical gap."""
        # Add critical gap
        initialized_assessment.add_gap(
            category=RequirementCategory.RISK_MANAGEMENT,
            severity=GapSeverity.CRITICAL,
            title="Critical issue",
            description="Blocks conformity",
        )

        report = initialized_assessment.run_full_assessment(auto_assess=True)
        assert report.overall_status == ConformityStatus.FAILED
        assert report.critical_gaps > 0

    def test_assessment_requires_remediation(self, initialized_assessment):
        """Test assessment requires remediation."""
        # Add multiple major gaps beyond tolerance
        for i in range(3):
            initialized_assessment.add_gap(
                category=RequirementCategory.RISK_MANAGEMENT,
                severity=GapSeverity.MAJOR,
                title=f"Major issue {i}",
                description="Requires remediation",
            )

        report = initialized_assessment.run_full_assessment(auto_assess=True)
        assert report.overall_status == ConformityStatus.REQUIRES_REMEDIATION

    def test_report_saved_to_file(self, initialized_assessment):
        """Test that report is saved to file."""
        report = initialized_assessment.run_full_assessment(auto_assess=True)

        # Check file exists
        expected_path = (
            Path(initialized_assessment.config.output_path)
            / f"assessment_report_{report.report_id}.json"
        )
        assert expected_path.exists()

        # Verify content
        with open(expected_path) as f:
            data = json.load(f)
        assert data["report_id"] == report.report_id

    def test_assessment_without_initialization(self, assessment):
        """Test that assessment initializes checklist if needed."""
        report = assessment.run_full_assessment(auto_assess=True)
        assert report.total_items > 0


class TestEUDeclarationGeneration:
    """Tests for EU Declaration generation."""

    def test_generate_declaration(self, initialized_assessment):
        """Test generating EU Declaration."""
        # First run assessment
        initialized_assessment.run_full_assessment(auto_assess=True)

        declaration = initialized_assessment.generate_eu_declaration(
            signatory_name="John Doe",
            signatory_title="CEO",
            signature_place="Berlin, Germany",
            applied_standards=["ISO 9001", "ISO 27001"],
        )

        assert declaration.declaration_id is not None
        assert declaration.signatory_name == "John Doe"
        assert declaration.signatory_title == "CEO"
        assert declaration.signature_place == "Berlin, Germany"
        assert len(declaration.applied_standards) == 2

    def test_declaration_has_default_standards(self, initialized_assessment):
        """Test declaration has default standards if none provided."""
        initialized_assessment.run_full_assessment(auto_assess=True)
        declaration = initialized_assessment.generate_eu_declaration()
        assert len(declaration.applied_standards) > 0

    def test_declaration_statement_content(self, initialized_assessment):
        """Test declaration statement contains required text."""
        initialized_assessment.run_full_assessment(auto_assess=True)
        declaration = initialized_assessment.generate_eu_declaration()

        assert "Regulation (EU) 2024/1689" in declaration.declaration_statement
        assert "sole responsibility" in declaration.declaration_statement

    def test_declaration_saved_to_file(self, initialized_assessment):
        """Test declaration is saved to file."""
        initialized_assessment.run_full_assessment(auto_assess=True)
        declaration = initialized_assessment.generate_eu_declaration()

        expected_path = (
            Path(initialized_assessment.config.output_path)
            / f"eu_declaration_{declaration.declaration_id}.json"
        )
        assert expected_path.exists()


class TestInstructionsForUseGeneration:
    """Tests for Instructions for Use generation."""

    def test_generate_instructions(self, assessment):
        """Test generating Instructions for Use."""
        ifu = assessment.generate_instructions_for_use(
            capabilities=["Capability 1", "Capability 2"],
            limitations=["Limitation 1"],
            known_risks=[
                {
                    "risk": "Market risk",
                    "description": "Loss potential",
                    "mitigation": "Position limits",
                }
            ],
        )

        assert ifu.document_id is not None
        assert len(ifu.capabilities) == 2
        assert len(ifu.limitations) == 1
        assert len(ifu.known_risks) == 1

    def test_instructions_has_defaults(self, assessment):
        """Test instructions has defaults if none provided."""
        ifu = assessment.generate_instructions_for_use()

        assert len(ifu.capabilities) > 0
        assert len(ifu.limitations) > 0
        assert len(ifu.known_risks) > 0
        assert len(ifu.human_oversight_measures) > 0

    def test_instructions_has_hardware_requirements(self, assessment):
        """Test instructions has hardware requirements."""
        ifu = assessment.generate_instructions_for_use()
        assert "CPU" in ifu.hardware_requirements
        assert "RAM" in ifu.hardware_requirements

    def test_instructions_has_logging_capabilities(self, assessment):
        """Test instructions has logging capabilities."""
        ifu = assessment.generate_instructions_for_use()
        assert len(ifu.logging_capabilities) > 0

    def test_instructions_saved_to_file(self, assessment):
        """Test instructions saved to file."""
        ifu = assessment.generate_instructions_for_use()

        expected_path = (
            Path(assessment.config.output_path) / f"instructions_for_use_{ifu.document_id}.json"
        )
        assert expected_path.exists()


class TestRegistrationPreparation:
    """Tests for registration preparation."""

    def test_prepare_registration(self, initialized_assessment):
        """Test preparing registration info."""
        initialized_assessment.run_full_assessment(auto_assess=True)

        reg = initialized_assessment.prepare_registration(
            provider_id="LEI-12345",
            member_states=["Germany", "France", "Netherlands"],
        )

        assert reg.registration_id is not None
        assert reg.provider_id == "LEI-12345"
        assert len(reg.member_states_available) == 3
        assert reg.risk_classification == "high_risk"

    def test_registration_has_defaults(self, initialized_assessment):
        """Test registration has defaults."""
        initialized_assessment.run_full_assessment(auto_assess=True)
        reg = initialized_assessment.prepare_registration()

        assert reg.conformity_assessment_procedure == "internal_control"
        assert len(reg.member_states_available) > 0

    def test_registration_saved_to_file(self, initialized_assessment):
        """Test registration saved to file."""
        initialized_assessment.run_full_assessment(auto_assess=True)
        reg = initialized_assessment.prepare_registration()

        expected_path = (
            Path(initialized_assessment.config.output_path)
            / f"registration_{reg.registration_id}.json"
        )
        assert expected_path.exists()


class TestExportAndReporting:
    """Tests for export and reporting."""

    def test_export_full_package(self, initialized_assessment):
        """Test exporting full conformity package."""
        initialized_assessment.run_full_assessment(auto_assess=True)

        package = initialized_assessment.export_full_package()

        assert "checklist" in package
        assert "gaps" in package
        assert "reports" in package

        # Verify files exist
        assert Path(package["checklist"]).exists()
        assert Path(package["gaps"]).exists()
        assert Path(package["reports"]).exists()

    def test_get_compliance_summary(self, initialized_assessment):
        """Test getting compliance summary."""
        initialized_assessment.run_full_assessment(auto_assess=True)

        summary = initialized_assessment.get_compliance_summary()

        assert "total_items" in summary
        assert "compliant_items" in summary
        assert "compliance_percentage" in summary
        assert "open_gaps" in summary
        assert "current_phase" in summary

    def test_compliance_summary_before_assessment(self, assessment):
        """Test compliance summary before assessment."""
        summary = assessment.get_compliance_summary()
        assert summary["status"] == "not_assessed"


class TestDefaultChecklist:
    """Tests for default checklist items."""

    def test_default_checklist_has_risk_management(self, assessment):
        """Test default checklist has risk management items."""
        items = assessment._get_default_checklist_items()
        risk_items = [i for i in items if i.category == RequirementCategory.RISK_MANAGEMENT]
        assert len(risk_items) >= 5

    def test_default_checklist_has_data_governance(self, assessment):
        """Test default checklist has data governance items."""
        items = assessment._get_default_checklist_items()
        data_items = [i for i in items if i.category == RequirementCategory.DATA_GOVERNANCE]
        assert len(data_items) >= 3

    def test_default_checklist_has_article_references(self, assessment):
        """Test default checklist items have article references."""
        items = assessment._get_default_checklist_items()
        for item in items:
            assert item.article_reference != ""

    def test_default_checklist_has_requirement_text(self, assessment):
        """Test default checklist items have requirement text."""
        items = assessment._get_default_checklist_items()
        for item in items:
            assert item.requirement_text != ""


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_conformity_assessment_no_config(self, temp_dir):
        """Test creating assessment without config."""
        with patch.object(Path, "mkdir"):
            assessment = create_conformity_assessment()
        assert assessment is not None

    def test_create_conformity_assessment_with_config(self, config):
        """Test creating assessment with config object."""
        assessment = create_conformity_assessment(config)
        assert assessment.config.organization_name == "Test Organization"

    def test_create_conformity_assessment_with_dict(self, temp_dir):
        """Test creating assessment with dict config."""
        config_dict = {
            "organization_name": "Dict Org",
            "system_name": "Dict System",
            "output_path": os.path.join(temp_dir, "output"),
            "log_path": os.path.join(temp_dir, "logs"),
        }
        assessment = create_conformity_assessment(config_dict)
        assert assessment.config.organization_name == "Dict Org"

    def test_get_default_checklist(self):
        """Test getting default checklist as dict."""
        checklist = get_default_checklist()
        assert isinstance(checklist, list)
        assert len(checklist) > 0
        assert isinstance(checklist[0], dict)
        assert "item_id" in checklist[0]
        assert "category" in checklist[0]


# =============================================================================
# Test Event Logging
# =============================================================================


class TestEventLogging:
    """Tests for event logging."""

    def test_log_event_creates_file(self, initialized_assessment):
        """Test that log events create log files."""
        initialized_assessment.assess_item(
            item_id=initialized_assessment.get_checklist()[0].item_id,
            status=ChecklistItemStatus.COMPLIANT,
        )

        log_path = Path(initialized_assessment.config.log_path)
        log_files = list(log_path.glob("*.jsonl"))
        assert len(log_files) > 0

    def test_log_event_content(self, initialized_assessment):
        """Test log event content format."""
        items = initialized_assessment.get_checklist()
        item_id = items[0].item_id

        initialized_assessment.assess_item(
            item_id=item_id,
            status=ChecklistItemStatus.COMPLIANT,
        )

        log_path = Path(initialized_assessment.config.log_path)
        log_files = list(log_path.glob("*.jsonl"))

        with open(log_files[0]) as f:
            line = f.readline()
            event = json.loads(line)

        assert "timestamp" in event
        assert "event_type" in event
        assert "data" in event


# =============================================================================
# Test Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_assess_items(self, initialized_assessment):
        """Test concurrent item assessments."""
        import threading

        items = initialized_assessment.get_checklist()[:5]
        results = []

        def assess_item(item):
            try:
                initialized_assessment.assess_item(
                    item_id=item.item_id,
                    status=ChecklistItemStatus.COMPLIANT,
                )
                results.append(True)
            except Exception:
                results.append(False)

        threads = [threading.Thread(target=assess_item, args=(item,)) for item in items]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results)
        assert len(results) == 5


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_checklist_assessment(self, assessment):
        """Test assessment with empty checklist."""
        # Don't initialize checklist, but assessment should handle it
        report = assessment.run_full_assessment(auto_assess=True)
        assert report.total_items > 0  # Should auto-initialize

    def test_all_not_applicable(self, initialized_assessment):
        """Test when all items are N/A."""
        for item in initialized_assessment.get_checklist():
            initialized_assessment.assess_item(
                item_id=item.item_id,
                status=ChecklistItemStatus.NOT_APPLICABLE,
            )

        report = initialized_assessment.run_full_assessment(auto_assess=False)
        assert report.compliance_score == 100.0  # All N/A means full compliance

    def test_mixed_statuses(self, initialized_assessment):
        """Test with mixed statuses."""
        items = initialized_assessment.get_checklist()

        # First third compliant
        for item in items[: len(items) // 3]:
            initialized_assessment.assess_item(
                item_id=item.item_id,
                status=ChecklistItemStatus.COMPLIANT,
            )

        # Second third partial
        for item in items[len(items) // 3 : 2 * len(items) // 3]:
            initialized_assessment.assess_item(
                item_id=item.item_id,
                status=ChecklistItemStatus.PARTIALLY_COMPLIANT,
            )

        # Rest N/A
        for item in items[2 * len(items) // 3 :]:
            initialized_assessment.assess_item(
                item_id=item.item_id,
                status=ChecklistItemStatus.NOT_APPLICABLE,
            )

        report = initialized_assessment.run_full_assessment(auto_assess=False)
        assert 0 < report.compliance_score < 100


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_conformity_workflow(self, initialized_assessment):
        """Test complete conformity assessment workflow."""
        # 1. Run assessment
        report = initialized_assessment.run_full_assessment(
            assessor="Integration Test",
            auto_assess=True,
        )
        assert report.overall_status == ConformityStatus.PASSED

        # 2. Generate declaration
        declaration = initialized_assessment.generate_eu_declaration(
            signatory_name="Test Signatory",
            signatory_title="Compliance Officer",
            signature_place="Test City",
        )
        assert declaration.is_valid

        # 3. Generate instructions
        ifu = initialized_assessment.generate_instructions_for_use()
        assert ifu.document_id is not None

        # 4. Prepare registration
        reg = initialized_assessment.prepare_registration(
            provider_id="TEST-LEI",
            member_states=["All EU Member States"],
        )
        assert reg.registration_id is not None

        # 5. Export package
        package = initialized_assessment.export_full_package()
        assert len(package) >= 3

        # 6. Get summary
        summary = initialized_assessment.get_compliance_summary()
        assert summary["compliance_percentage"] == 100.0


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
