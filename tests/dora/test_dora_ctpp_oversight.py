# -*- coding: utf-8 -*-
"""
Tests for DORA Critical Third-Party Provider (CTPP) Oversight Module (Articles 31-44).

Comprehensive tests for CTPP oversight management including:
- CTPP designation tracking
- Lead Overseer coordination
- Oversight recommendation compliance
- CTPP-specific risk assessment
- Enhanced contractual requirements

Tests Reference:
    - Articles 31-44 DORA: CTPP Oversight Framework
    - ESAs Joint Statement on CTPP Designation
    - Lead Overseers: EBA, ESMA, EIOPA
"""

import pytest
import threading
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from services.dora_integration.third_party import (
    # Enumerations
    LeadOverseer,
    CTPPStatus,
    OversightRecommendationType,
    RecommendationStatus,
    ComplianceLevel,
    OversightExerciseType,
    # Data structures
    CTPPDesignation,
    OversightRecommendation,
    OversightExercise,
    CTPPRiskAssessment,
    CTPPContractRequirement,
    EntityCTPPRelationship,
    # Constants
    DESIGNATED_CTPPS_2025,
    # Configuration
    CTPPOversightConfig,
    # Functions
    get_ctpp_contract_requirements,
    # Main implementation
    DORACtppOversight,
    # Factory functions
    create_ctpp_oversight,
    get_lead_overseers,
    get_designated_ctpps_list,
    get_ctpp_requirements,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def ctpp_config():
    """Create CTPP oversight configuration."""
    return CTPPOversightConfig(
        ctpp_assessment_frequency_months=6,
        require_ctpp_exit_plan=True,
        require_ctpp_alternatives=True,
        min_alternatives_for_ctpp=1,
        notify_on_new_recommendation=False,  # Disable for tests
        notify_on_oversight_exercise=False,
        log_all_events=False,  # Disable logging for tests
    )


@pytest.fixture
def ctpp_manager(ctpp_config):
    """Create CTPP oversight manager."""
    return DORACtppOversight(config=ctpp_config)


@pytest.fixture
def manager_with_relationship(ctpp_manager):
    """Create manager with sample relationship."""
    relationship = ctpp_manager.register_ctpp_relationship(
        ctpp_name="AWS",
        services_used=["cloud_computing", "storage"],
        entity_name="Test Financial Entity",
        contract_reference="CTR-2024-001",
        critical_functions=["trading_platform"],
        dependency_level="high",
        exit_plan_in_place=False,
        alternative_providers=["Azure"],
    )
    return ctpp_manager, relationship


# =============================================================================
# Test Enumerations
# =============================================================================

class TestCTPPEnumerations:
    """Tests for CTPP oversight enumerations."""

    def test_lead_overseer_values(self):
        """Test LeadOverseer enum values."""
        assert LeadOverseer.EBA.value == "eba"
        assert LeadOverseer.ESMA.value == "esma"
        assert LeadOverseer.EIOPA.value == "eiopa"

    def test_ctpp_status_values(self):
        """Test CTPPStatus enum values."""
        assert CTPPStatus.NOT_DESIGNATED.value == "not_designated"
        assert CTPPStatus.PENDING_DESIGNATION.value == "pending_designation"
        assert CTPPStatus.DESIGNATED.value == "designated"
        assert CTPPStatus.UNDER_REVIEW.value == "under_review"
        assert CTPPStatus.DESIGNATION_REMOVED.value == "designation_removed"

    def test_oversight_recommendation_type_values(self):
        """Test OversightRecommendationType enum values."""
        assert OversightRecommendationType.SECURITY.value == "security"
        assert OversightRecommendationType.RISK_MANAGEMENT.value == "risk_management"
        assert OversightRecommendationType.GOVERNANCE.value == "governance"
        assert OversightRecommendationType.ICT_OPERATIONS.value == "ict_operations"
        assert OversightRecommendationType.BUSINESS_CONTINUITY.value == "business_continuity"
        assert OversightRecommendationType.EXIT_STRATEGY.value == "exit_strategy"

    def test_recommendation_status_values(self):
        """Test RecommendationStatus enum values."""
        assert RecommendationStatus.RECEIVED.value == "received"
        assert RecommendationStatus.UNDER_REVIEW.value == "under_review"
        assert RecommendationStatus.ACTION_REQUIRED.value == "action_required"
        assert RecommendationStatus.IN_PROGRESS.value == "in_progress"
        assert RecommendationStatus.IMPLEMENTED.value == "implemented"
        assert RecommendationStatus.NOT_APPLICABLE.value == "not_applicable"

    def test_compliance_level_values(self):
        """Test ComplianceLevel enum values."""
        assert ComplianceLevel.COMPLIANT.value == "compliant"
        assert ComplianceLevel.PARTIALLY_COMPLIANT.value == "partially_compliant"
        assert ComplianceLevel.NON_COMPLIANT.value == "non_compliant"
        assert ComplianceLevel.NOT_ASSESSED.value == "not_assessed"

    def test_oversight_exercise_type_values(self):
        """Test OversightExerciseType enum values."""
        assert OversightExerciseType.GENERAL_INVESTIGATION.value == "general_investigation"
        assert OversightExerciseType.INSPECTION.value == "inspection"
        assert OversightExerciseType.OFF_SITE_REVIEW.value == "off_site_review"
        assert OversightExerciseType.INFORMATION_REQUEST.value == "information_request"


# =============================================================================
# Test Data Structures
# =============================================================================

class TestDataStructures:
    """Tests for CTPP data structures."""

    def test_ctpp_designation_auto_init(self):
        """Test CTPPDesignation auto-initialization."""
        designation = CTPPDesignation(
            provider_name="Test CTPP",
            provider_lei="123456789",
            status=CTPPStatus.DESIGNATED,
        )

        assert designation.designation_id.startswith("CTD-")
        assert designation.created_date

    def test_oversight_recommendation_auto_init(self):
        """Test OversightRecommendation auto-initialization."""
        recommendation = OversightRecommendation(
            ctpp_id="CTD-001",
            provider_name="Test CTPP",
            title="Security Enhancement",
        )

        assert recommendation.recommendation_id.startswith("REC-")
        assert recommendation.issue_date
        assert recommendation.status == RecommendationStatus.RECEIVED

    def test_oversight_exercise_auto_init(self):
        """Test OversightExercise auto-initialization."""
        exercise = OversightExercise(
            ctpp_id="CTD-001",
            title="Annual Review",
        )

        assert exercise.exercise_id.startswith("OEX-")

    def test_ctpp_risk_assessment_auto_init(self):
        """Test CTPPRiskAssessment auto-initialization."""
        assessment = CTPPRiskAssessment(
            ctpp_id="CTD-001",
            provider_name="Test CTPP",
        )

        assert assessment.assessment_id.startswith("CRA-")
        assert assessment.assessment_date

    def test_ctpp_contract_requirement_auto_init(self):
        """Test CTPPContractRequirement auto-initialization."""
        requirement = CTPPContractRequirement(
            article_reference="Article 33",
            requirement_name="Lead Overseer Access",
        )

        assert requirement.requirement_id.startswith("CTR-")
        assert requirement.is_mandatory is True

    def test_entity_ctpp_relationship_auto_init(self):
        """Test EntityCTPPRelationship auto-initialization."""
        relationship = EntityCTPPRelationship(
            ctpp_id="CTD-001",
            ctpp_name="AWS",
            entity_name="Test Entity",
        )

        assert relationship.relationship_id.startswith("REL-")
        assert relationship.contract_compliance == ComplianceLevel.NOT_ASSESSED


# =============================================================================
# Test Configuration
# =============================================================================

class TestCTPPOversightConfig:
    """Tests for CTPP oversight configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CTPPOversightConfig()

        assert config.ctpp_assessment_frequency_months == 6
        assert config.require_ctpp_exit_plan is True
        assert config.require_ctpp_alternatives is True
        assert config.min_alternatives_for_ctpp == 1
        assert config.notify_on_new_recommendation is True
        assert config.require_annual_ctpp_review is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = CTPPOversightConfig(
            ctpp_assessment_frequency_months=12,
            min_alternatives_for_ctpp=2,
            notify_on_new_recommendation=False,
        )

        assert config.ctpp_assessment_frequency_months == 12
        assert config.min_alternatives_for_ctpp == 2
        assert config.notify_on_new_recommendation is False

    def test_config_with_callbacks(self):
        """Test configuration with callbacks."""
        notification_fn = MagicMock()
        escalation_fn = MagicMock()

        config = CTPPOversightConfig(
            notification_callback=notification_fn,
            escalation_callback=escalation_fn,
        )

        assert config.notification_callback == notification_fn
        assert config.escalation_callback == escalation_fn


# =============================================================================
# Test CTPP Contract Requirements
# =============================================================================

class TestCTPPContractRequirements:
    """Tests for CTPP contractual requirements."""

    def test_get_ctpp_contract_requirements(self):
        """Test getting CTPP contract requirements."""
        requirements = get_ctpp_contract_requirements()

        assert len(requirements) >= 8
        # All should be mandatory
        assert all(r.is_mandatory for r in requirements)

    def test_requirements_have_article_references(self):
        """Test requirements have Article references."""
        requirements = get_ctpp_contract_requirements()

        for req in requirements:
            assert req.article_reference
            assert "Article" in req.article_reference

    def test_specific_requirements_present(self):
        """Test specific requirements are present."""
        requirements = get_ctpp_contract_requirements()
        requirement_names = [r.requirement_name for r in requirements]

        assert "CTPP Status Acknowledgment" in requirement_names
        assert "Lead Overseer Access" in requirement_names
        assert "Cooperation with Oversight Exercises" in requirement_names


# =============================================================================
# Test Known CTPPs
# =============================================================================

class TestKnownCTPPs:
    """Tests for known CTPP data."""

    def test_designated_ctpps_2025_structure(self):
        """Test DESIGNATED_CTPPS_2025 data structure."""
        assert len(DESIGNATED_CTPPS_2025) > 0

        for ctpp in DESIGNATED_CTPPS_2025:
            assert "name" in ctpp
            assert "lei" in ctpp
            assert "lead_overseer" in ctpp
            assert "headquarters_country" in ctpp
            assert "services" in ctpp

    def test_known_ctpps_include_major_providers(self):
        """Test known CTPPs include major cloud providers."""
        names = [c["name"].lower() for c in DESIGNATED_CTPPS_2025]

        # Major cloud providers should be included
        assert any("amazon" in name or "aws" in name for name in names)
        assert any("google" in name for name in names)
        assert any("microsoft" in name for name in names)


# =============================================================================
# Test CTPP Status Checking
# =============================================================================

class TestCTPPStatusChecking:
    """Tests for CTPP status checking."""

    def test_check_ctpp_status_aws(self, ctpp_manager):
        """Test checking AWS as CTPP."""
        is_ctpp, designation = ctpp_manager.check_ctpp_status("AWS")

        assert is_ctpp is True
        assert designation is not None
        assert "amazon" in designation.provider_name.lower() or "aws" in designation.provider_name.lower()

    def test_check_ctpp_status_google(self, ctpp_manager):
        """Test checking Google Cloud as CTPP."""
        is_ctpp, designation = ctpp_manager.check_ctpp_status("Google")

        assert is_ctpp is True
        assert designation is not None

    def test_check_ctpp_status_azure(self, ctpp_manager):
        """Test checking Azure as CTPP."""
        is_ctpp, designation = ctpp_manager.check_ctpp_status("Azure")

        assert is_ctpp is True
        assert designation is not None

    def test_check_ctpp_status_unknown_provider(self, ctpp_manager):
        """Test checking unknown provider."""
        is_ctpp, designation = ctpp_manager.check_ctpp_status("Unknown Provider XYZ")

        assert is_ctpp is False
        assert designation is None

    def test_check_ctpp_status_case_insensitive(self, ctpp_manager):
        """Test CTPP status check is case insensitive."""
        is_ctpp1, _ = ctpp_manager.check_ctpp_status("aws")
        is_ctpp2, _ = ctpp_manager.check_ctpp_status("AWS")
        is_ctpp3, _ = ctpp_manager.check_ctpp_status("Aws")

        assert is_ctpp1 == is_ctpp2 == is_ctpp3


# =============================================================================
# Test CTPP Designation Management
# =============================================================================

class TestCTPPDesignationManagement:
    """Tests for CTPP designation management."""

    def test_get_all_designated_ctpps(self, ctpp_manager):
        """Test getting all designated CTPPs."""
        ctpps = ctpp_manager.get_all_designated_ctpps()

        assert len(ctpps) == len(DESIGNATED_CTPPS_2025)
        assert all(c.status == CTPPStatus.DESIGNATED for c in ctpps)

    def test_get_ctpps_by_lead_overseer_eba(self, ctpp_manager):
        """Test getting CTPPs by EBA lead overseer."""
        ctpps = ctpp_manager.get_ctpps_by_lead_overseer(LeadOverseer.EBA)

        assert len(ctpps) > 0
        assert all(c.lead_overseer == LeadOverseer.EBA for c in ctpps)

    def test_get_ctpps_by_lead_overseer_esma(self, ctpp_manager):
        """Test getting CTPPs by ESMA lead overseer."""
        ctpps = ctpp_manager.get_ctpps_by_lead_overseer(LeadOverseer.ESMA)

        assert isinstance(ctpps, list)
        assert all(c.lead_overseer == LeadOverseer.ESMA for c in ctpps)

    def test_add_new_ctpp_designation(self, ctpp_manager):
        """Test adding new CTPP designation."""
        designation = ctpp_manager.add_ctpp_designation(
            provider_name="New Cloud Provider",
            provider_lei="549300TESTLEI123",
            lead_overseer=LeadOverseer.EIOPA,
            designation_reference="ESA/2025/NEW",
            services_in_scope=["cloud_storage"],
            headquarters_country="FR",
        )

        assert designation.designation_id.startswith("CTD-")
        assert designation.status == CTPPStatus.DESIGNATED
        assert designation.lead_overseer == LeadOverseer.EIOPA

        # Should be findable now
        is_ctpp, found = ctpp_manager.check_ctpp_status("New Cloud Provider")
        assert is_ctpp is True

    def test_get_ctpp_designation_by_id(self, ctpp_manager):
        """Test getting CTPP designation by ID."""
        ctpps = ctpp_manager.get_all_designated_ctpps()
        if ctpps:
            ctpp_id = ctpps[0].designation_id
            retrieved = ctpp_manager.get_ctpp_designation(ctpp_id)

            assert retrieved is not None
            assert retrieved.designation_id == ctpp_id


# =============================================================================
# Test Entity CTPP Relationship Management
# =============================================================================

class TestEntityCTPPRelationshipManagement:
    """Tests for entity CTPP relationship management."""

    def test_register_ctpp_relationship(self, ctpp_manager):
        """Test registering CTPP relationship."""
        relationship = ctpp_manager.register_ctpp_relationship(
            ctpp_name="AWS",
            services_used=["compute", "storage"],
            entity_name="Test Entity",
            critical_functions=["trading"],
            dependency_level="high",
        )

        assert relationship is not None
        assert relationship.relationship_id.startswith("REL-")
        assert len(relationship.services_used) == 2

    def test_register_relationship_generates_actions(self, ctpp_manager):
        """Test relationship registration generates required actions."""
        relationship = ctpp_manager.register_ctpp_relationship(
            ctpp_name="AWS",
            services_used=["compute"],
            critical_functions=["trading"],
            exit_plan_in_place=False,
            alternative_providers=[],
        )

        assert len(relationship.actions_required) > 0
        # Should require exit plan and alternatives
        actions_text = " ".join(relationship.actions_required).lower()
        assert "exit" in actions_text or "alternative" in actions_text

    def test_register_relationship_unknown_ctpp(self, ctpp_manager):
        """Test registering relationship with unknown CTPP."""
        result = ctpp_manager.register_ctpp_relationship(
            ctpp_name="Unknown Provider",
            services_used=["service"],
        )

        assert result is None

    def test_get_ctpp_relationship(self, manager_with_relationship):
        """Test getting CTPP relationship."""
        manager, relationship = manager_with_relationship

        retrieved = manager.get_ctpp_relationship(relationship.relationship_id)

        assert retrieved is not None
        assert retrieved.relationship_id == relationship.relationship_id

    def test_get_all_ctpp_relationships(self, manager_with_relationship):
        """Test getting all CTPP relationships."""
        manager, _ = manager_with_relationship

        relationships = manager.get_all_ctpp_relationships()

        assert len(relationships) >= 1

    def test_update_relationship(self, manager_with_relationship):
        """Test updating CTPP relationship."""
        manager, relationship = manager_with_relationship

        updated = manager.update_relationship(
            relationship.relationship_id,
            exit_plan_in_place=True,
            dependency_level="critical",
        )

        assert updated.exit_plan_in_place is True
        assert updated.dependency_level == "critical"


# =============================================================================
# Test Oversight Recommendations
# =============================================================================

class TestOversightRecommendations:
    """Tests for oversight recommendation management."""

    def test_record_oversight_recommendation(self, ctpp_manager):
        """Test recording oversight recommendation."""
        recommendation = ctpp_manager.record_oversight_recommendation(
            ctpp_name="AWS",
            recommendation_type=OversightRecommendationType.SECURITY,
            title="Security Enhancement Required",
            description="Implement enhanced encryption",
            issuing_authority=LeadOverseer.EBA,
            deadline="2025-06-01",
            entity_actions_required=["Review security controls"],
        )

        assert recommendation is not None
        assert recommendation.recommendation_id.startswith("REC-")
        assert recommendation.status == RecommendationStatus.RECEIVED

    def test_record_recommendation_unknown_ctpp(self, ctpp_manager):
        """Test recording recommendation for unknown CTPP."""
        result = ctpp_manager.record_oversight_recommendation(
            ctpp_name="Unknown Provider",
            recommendation_type=OversightRecommendationType.SECURITY,
            title="Test",
            description="Test",
            issuing_authority=LeadOverseer.EBA,
        )

        assert result is None

    def test_get_recommendations_requiring_action(self, ctpp_manager):
        """Test getting recommendations requiring action."""
        ctpp_manager.record_oversight_recommendation(
            ctpp_name="AWS",
            recommendation_type=OversightRecommendationType.RISK_MANAGEMENT,
            title="Risk Review",
            description="Conduct risk review",
            issuing_authority=LeadOverseer.EBA,
            entity_actions_required=["Update risk assessment"],
        )

        requiring_action = ctpp_manager.get_recommendations_requiring_action()

        assert len(requiring_action) >= 1

    def test_update_recommendation_status(self, ctpp_manager):
        """Test updating recommendation status."""
        recommendation = ctpp_manager.record_oversight_recommendation(
            ctpp_name="AWS",
            recommendation_type=OversightRecommendationType.GOVERNANCE,
            title="Governance Update",
            description="Update governance",
            issuing_authority=LeadOverseer.EBA,
        )

        updated = ctpp_manager.update_recommendation_status(
            recommendation.recommendation_id,
            status=RecommendationStatus.IN_PROGRESS,
            progress_pct=50.0,
            actions_taken=[{"action": "Started review", "date": "2025-01-01"}],
        )

        assert updated.status == RecommendationStatus.IN_PROGRESS
        assert updated.implementation_progress_pct == 50.0
        assert len(updated.actions_taken) > 0


# =============================================================================
# Test Oversight Exercises
# =============================================================================

class TestOversightExercises:
    """Tests for oversight exercise management."""

    def test_record_oversight_exercise(self, ctpp_manager):
        """Test recording oversight exercise."""
        exercise = ctpp_manager.record_oversight_exercise(
            ctpp_name="AWS",
            exercise_type=OversightExerciseType.OFF_SITE_REVIEW,
            title="Annual Security Review",
            scope="Security controls assessment",
            lead_overseer=LeadOverseer.EBA,
            entity_participation_required=True,
        )

        assert exercise is not None
        assert exercise.exercise_id.startswith("OEX-")
        assert exercise.status == "notified"

    def test_record_exercise_unknown_ctpp(self, ctpp_manager):
        """Test recording exercise for unknown CTPP."""
        result = ctpp_manager.record_oversight_exercise(
            ctpp_name="Unknown",
            exercise_type=OversightExerciseType.INSPECTION,
            title="Test",
            scope="Test",
            lead_overseer=LeadOverseer.EBA,
        )

        assert result is None

    def test_get_active_exercises(self, ctpp_manager):
        """Test getting active exercises."""
        ctpp_manager.record_oversight_exercise(
            ctpp_name="AWS",
            exercise_type=OversightExerciseType.INFORMATION_REQUEST,
            title="Info Request",
            scope="Data gathering",
            lead_overseer=LeadOverseer.EBA,
        )

        active = ctpp_manager.get_active_exercises()

        assert len(active) >= 1
        assert all(e.status in ["notified", "in_progress"] for e in active)

    def test_get_exercise_by_id(self, ctpp_manager):
        """Test getting exercise by ID."""
        exercise = ctpp_manager.record_oversight_exercise(
            ctpp_name="AWS",
            exercise_type=OversightExerciseType.GENERAL_INVESTIGATION,
            title="Investigation",
            scope="Full scope",
            lead_overseer=LeadOverseer.EBA,
        )

        retrieved = ctpp_manager.get_exercise(exercise.exercise_id)

        assert retrieved is not None
        assert retrieved.exercise_id == exercise.exercise_id


# =============================================================================
# Test CTPP Risk Assessment
# =============================================================================

class TestCTPPRiskAssessment:
    """Tests for CTPP risk assessment."""

    def test_assess_ctpp_risks(self, manager_with_relationship):
        """Test CTPP risk assessment."""
        manager, relationship = manager_with_relationship

        assessment = manager.assess_ctpp_risks(
            relationship.relationship_id,
            assessed_by="Risk Manager",
        )

        assert assessment is not None
        assert assessment.assessment_id.startswith("CRA-")
        assert assessment.overall_risk_level in ["low", "medium", "high", "critical"]

    def test_assess_ctpp_risks_invalid_relationship(self, ctpp_manager):
        """Test risk assessment with invalid relationship."""
        result = ctpp_manager.assess_ctpp_risks("INVALID-ID")

        assert result is None

    def test_assessment_identifies_missing_alternatives(self, ctpp_manager):
        """Test assessment identifies missing alternatives."""
        relationship = ctpp_manager.register_ctpp_relationship(
            ctpp_name="AWS",
            services_used=["compute"],
            alternative_providers=[],
        )

        assessment = ctpp_manager.assess_ctpp_risks(relationship.relationship_id)

        assert assessment.substitutability_risk_level == "critical"
        assert "alternative" in " ".join(assessment.recommendations).lower()

    def test_get_ctpp_assessment(self, manager_with_relationship):
        """Test getting CTPP assessment by ID."""
        manager, relationship = manager_with_relationship

        assessment = manager.assess_ctpp_risks(relationship.relationship_id)
        retrieved = manager.get_ctpp_assessment(assessment.assessment_id)

        assert retrieved is not None
        assert retrieved.assessment_id == assessment.assessment_id


# =============================================================================
# Test Contract Compliance
# =============================================================================

class TestContractCompliance:
    """Tests for CTPP contract compliance checking."""

    def test_check_contract_compliance_full(self, manager_with_relationship):
        """Test full contract compliance."""
        manager, relationship = manager_with_relationship

        requirements = get_ctpp_contract_requirements()
        provisions = {req.requirement_name: True for req in requirements}

        results = manager.check_ctpp_contract_compliance(
            relationship.relationship_id,
            contract_provisions=provisions,
        )

        assert results["overall_compliance"] == ComplianceLevel.COMPLIANT
        assert results["non_compliant_count"] == 0

    def test_check_contract_compliance_partial(self, manager_with_relationship):
        """Test partial contract compliance."""
        manager, relationship = manager_with_relationship

        # Only provide some requirements
        provisions = {
            "CTPP Status Acknowledgment": True,
            "Lead Overseer Access": True,
            "Cooperation with Oversight Exercises": False,
        }

        results = manager.check_ctpp_contract_compliance(
            relationship.relationship_id,
            contract_provisions=provisions,
        )

        assert results["non_compliant_count"] > 0
        assert len(results["gaps"]) > 0

    def test_check_contract_compliance_invalid_relationship(self, ctpp_manager):
        """Test compliance check with invalid relationship."""
        results = ctpp_manager.check_ctpp_contract_compliance(
            "INVALID-ID",
            contract_provisions={},
        )

        assert "error" in results


# =============================================================================
# Test Reporting
# =============================================================================

class TestCTPPReporting:
    """Tests for CTPP reporting."""

    def test_get_ctpp_oversight_status(self, ctpp_manager):
        """Test getting CTPP oversight status."""
        status = ctpp_manager.get_ctpp_oversight_status()

        assert "timestamp" in status
        assert "ctpp_landscape" in status
        assert "entity_relationships" in status
        assert "oversight_activity" in status
        assert "compliance_indicators" in status

    def test_get_ctpp_oversight_status_with_data(self, manager_with_relationship):
        """Test status with relationship data."""
        manager, _ = manager_with_relationship

        status = manager.get_ctpp_oversight_status()

        assert status["entity_relationships"]["total"] >= 1

    def test_generate_ctpp_report(self, ctpp_manager):
        """Test generating CTPP report."""
        report = ctpp_manager.generate_ctpp_report()

        assert "report_date" in report
        assert "report_type" in report
        assert report["report_type"] == "ctpp_oversight_summary"
        assert "designated_ctpps" in report
        assert "entity_ctpp_usage" in report

    def test_generate_ctpp_report_with_data(self, manager_with_relationship):
        """Test report with relationship data."""
        manager, _ = manager_with_relationship

        report = manager.generate_ctpp_report()

        assert len(report["entity_ctpp_usage"]) >= 1


# =============================================================================
# Test Thread Safety
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_relationship_registration(self, ctpp_config):
        """Test concurrent relationship registration."""
        manager = DORACtppOversight(config=ctpp_config)
        errors = []
        results = []

        def register_relationship(index):
            try:
                relationship = manager.register_ctpp_relationship(
                    ctpp_name="AWS",
                    services_used=[f"service_{index}"],
                    entity_name=f"Entity {index}",
                )
                if relationship:
                    results.append(relationship.relationship_id)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_relationship, args=(i,))
            for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10

    def test_concurrent_status_checks(self, ctpp_manager):
        """Test concurrent CTPP status checks."""
        errors = []
        results = []

        def check_status():
            try:
                is_ctpp, designation = ctpp_manager.check_ctpp_status("AWS")
                results.append(is_ctpp)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=check_status)
            for _ in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(r is True for r in results)


# =============================================================================
# Test Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_ctpp_oversight_default(self):
        """Test create_ctpp_oversight with defaults."""
        manager = create_ctpp_oversight()

        assert isinstance(manager, DORACtppOversight)
        assert manager.config is not None

    def test_create_ctpp_oversight_with_config(self, ctpp_config):
        """Test create_ctpp_oversight with custom config."""
        manager = create_ctpp_oversight(config=ctpp_config)

        assert manager.config == ctpp_config

    def test_get_lead_overseers(self):
        """Test get_lead_overseers returns all overseers."""
        overseers = get_lead_overseers()

        assert len(overseers) == 3
        assert LeadOverseer.EBA in overseers
        assert LeadOverseer.ESMA in overseers
        assert LeadOverseer.EIOPA in overseers

    def test_get_designated_ctpps_list(self):
        """Test get_designated_ctpps_list returns known CTPPs."""
        ctpps = get_designated_ctpps_list()

        assert len(ctpps) == len(DESIGNATED_CTPPS_2025)

    def test_get_ctpp_requirements(self):
        """Test get_ctpp_requirements returns requirements."""
        requirements = get_ctpp_requirements()

        assert len(requirements) >= 8
        assert all(isinstance(r, CTPPContractRequirement) for r in requirements)


# =============================================================================
# Test Integration Scenarios
# =============================================================================

class TestIntegrationScenarios:
    """Integration tests for complete workflows."""

    def test_complete_ctpp_management_workflow(self, ctpp_config):
        """Test complete CTPP management workflow."""
        manager = DORACtppOversight(config=ctpp_config)

        # 1. Check if provider is CTPP
        is_ctpp, designation = manager.check_ctpp_status("AWS")
        assert is_ctpp is True

        # 2. Register relationship
        relationship = manager.register_ctpp_relationship(
            ctpp_name="AWS",
            services_used=["compute", "storage", "database"],
            entity_name="Financial Entity Ltd",
            contract_reference="CTR-2024-001",
            critical_functions=["trading_platform", "risk_engine"],
            dependency_level="high",
            exit_plan_in_place=True,
            alternative_providers=["Azure", "GCP"],
        )
        assert relationship is not None

        # 3. Assess CTPP risks
        assessment = manager.assess_ctpp_risks(
            relationship.relationship_id,
            assessed_by="Chief Risk Officer",
        )
        assert assessment is not None
        assert assessment.overall_risk_level

        # 4. Check contract compliance
        requirements = get_ctpp_contract_requirements()
        provisions = {req.requirement_name: True for req in requirements}
        compliance = manager.check_ctpp_contract_compliance(
            relationship.relationship_id,
            contract_provisions=provisions,
        )
        assert compliance["overall_compliance"] == ComplianceLevel.COMPLIANT

        # 5. Record oversight recommendation
        recommendation = manager.record_oversight_recommendation(
            ctpp_name="AWS",
            recommendation_type=OversightRecommendationType.SECURITY,
            title="Enhanced Encryption",
            description="Implement enhanced encryption",
            issuing_authority=LeadOverseer.EBA,
            entity_actions_required=["Review encryption standards"],
        )
        assert recommendation is not None

        # 6. Update recommendation status
        manager.update_recommendation_status(
            recommendation.recommendation_id,
            status=RecommendationStatus.IMPLEMENTED,
            progress_pct=100.0,
        )

        # 7. Generate report
        report = manager.generate_ctpp_report()
        assert len(report["entity_ctpp_usage"]) >= 1

        # 8. Get status
        status = manager.get_ctpp_oversight_status()
        assert status["entity_relationships"]["total"] >= 1

    def test_oversight_exercise_workflow(self, ctpp_config):
        """Test oversight exercise workflow."""
        manager = DORACtppOversight(config=ctpp_config)

        # Register relationship first
        relationship = manager.register_ctpp_relationship(
            ctpp_name="AWS",
            services_used=["compute"],
        )

        # Record exercise
        exercise = manager.record_oversight_exercise(
            ctpp_name="AWS",
            exercise_type=OversightExerciseType.INSPECTION,
            title="Annual Inspection 2025",
            scope="Full operational resilience assessment",
            lead_overseer=LeadOverseer.EBA,
            entity_participation_required=True,
        )

        assert exercise is not None
        assert exercise.entity_participation_required is True

        # Check active exercises
        active = manager.get_active_exercises()
        assert len(active) >= 1

        # Get status
        status = manager.get_ctpp_oversight_status()
        assert status["oversight_activity"]["active_exercises"] >= 1
