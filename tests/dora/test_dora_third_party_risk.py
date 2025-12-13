# -*- coding: utf-8 -*-
"""
Tests for DORA Third-Party ICT Risk Management Module (Article 28).

Comprehensive test coverage for:
- Enumerations
- Data structures (ICTService, ICTProvider, ThirdPartyRisk, etc.)
- ThirdPartyRiskConfig
- DORAThirdPartyRiskManagement
- Factory functions

References:
    - Article 28 DORA (EU 2022/2554)
    - RTS on ICT Risk Management CDR 2024/1774
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch
import threading
import tempfile
import os

from services.dora_integration.third_party import (
    # Enumerations
    ProviderType,
    ProviderCriticality,
    ServiceCriticality,
    ProviderStatus,
    RiskCategory,
    RiskLevel,
    DueDiligenceStatus,
    AssessmentType,
    SubstitutabilityLevel,
    # Data structures
    ICTService,
    ICTProvider,
    ThirdPartyRisk,
    ThirdPartyRiskAssessment,
    DueDiligenceCheck,
    ProviderRelationshipEvent,
    # Configuration
    ThirdPartyRiskConfig,
    # Main class
    DORAThirdPartyRiskManagement,
    # Factory functions
    create_third_party_risk_management,
    get_provider_types,
    get_risk_categories,
    get_criticality_levels,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def config():
    """Create test configuration."""
    return ThirdPartyRiskConfig(
        periodic_assessment_months=12,
        critical_provider_assessment_months=6,
        require_due_diligence_before_onboarding=False,  # For easier testing
        log_all_events=False,
    )


@pytest.fixture
def manager(config):
    """Create test manager instance."""
    return DORAThirdPartyRiskManagement(config=config)


@pytest.fixture
def sample_service():
    """Create sample ICT service."""
    return ICTService(
        name="Cloud Compute",
        description="Virtual machine hosting",
        service_type="compute",
        criticality=ServiceCriticality.CRITICAL_OR_IMPORTANT,
        supports_critical_function=True,
        supported_functions=["trading", "risk_calculation"],
        availability_target_pct=99.9,
        data_classification="confidential",
    )


@pytest.fixture
def sample_provider(manager, sample_service):
    """Create and register sample provider."""
    provider = manager.register_provider(
        name="AWS",
        provider_type=ProviderType.CLOUD_SERVICE,
        country="US",
        legal_name="Amazon Web Services, Inc.",
        lei="LESI123456789",
        services=[sample_service],
    )
    return provider


# =============================================================================
# Enumeration Tests
# =============================================================================

class TestEnumerations:
    """Test all enumeration classes."""

    def test_provider_type_values(self):
        """Test ProviderType enumeration values."""
        assert ProviderType.CLOUD_SERVICE.value == "cloud_service"
        assert ProviderType.INFRASTRUCTURE.value == "infrastructure"
        assert ProviderType.DATA_PROVIDER.value == "data_provider"
        assert ProviderType.PAYMENT_PROCESSOR.value == "payment_processor"
        assert len(ProviderType) == 12

    def test_provider_criticality_values(self):
        """Test ProviderCriticality enumeration values."""
        assert ProviderCriticality.CRITICAL.value == "critical"
        assert ProviderCriticality.HIGH.value == "high"
        assert ProviderCriticality.MEDIUM.value == "medium"
        assert ProviderCriticality.LOW.value == "low"
        assert len(ProviderCriticality) == 4

    def test_service_criticality_values(self):
        """Test ServiceCriticality enumeration values."""
        assert ServiceCriticality.CRITICAL_OR_IMPORTANT.value == "critical_or_important"
        assert ServiceCriticality.STANDARD.value == "standard"
        assert len(ServiceCriticality) == 4

    def test_provider_status_values(self):
        """Test ProviderStatus enumeration values."""
        assert ProviderStatus.ACTIVE.value == "active"
        assert ProviderStatus.SUSPENDED.value == "suspended"
        assert ProviderStatus.TERMINATED.value == "terminated"
        assert ProviderStatus.PENDING_EXIT.value == "pending_exit"

    def test_risk_category_values(self):
        """Test RiskCategory enumeration values."""
        assert RiskCategory.OPERATIONAL.value == "operational"
        assert RiskCategory.SECURITY.value == "security"
        assert RiskCategory.CONCENTRATION.value == "concentration"
        assert RiskCategory.DATA_PROTECTION.value == "data_protection"
        assert len(RiskCategory) == 10

    def test_risk_level_values(self):
        """Test RiskLevel enumeration values."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"

    def test_due_diligence_status_values(self):
        """Test DueDiligenceStatus enumeration values."""
        assert DueDiligenceStatus.NOT_STARTED.value == "not_started"
        assert DueDiligenceStatus.IN_PROGRESS.value == "in_progress"
        assert DueDiligenceStatus.COMPLETED.value == "completed"
        assert DueDiligenceStatus.EXPIRED.value == "expired"

    def test_assessment_type_values(self):
        """Test AssessmentType enumeration values."""
        assert AssessmentType.INITIAL.value == "initial"
        assert AssessmentType.PERIODIC.value == "periodic"
        assert AssessmentType.INCIDENT_TRIGGERED.value == "incident_triggered"

    def test_substitutability_level_values(self):
        """Test SubstitutabilityLevel enumeration values."""
        assert SubstitutabilityLevel.EASILY_SUBSTITUTABLE.value == "easily_substitutable"
        assert SubstitutabilityLevel.NOT_SUBSTITUTABLE.value == "not_substitutable"


# =============================================================================
# Data Structure Tests
# =============================================================================

class TestICTService:
    """Test ICTService data structure."""

    def test_ict_service_creation(self):
        """Test ICTService creation with defaults."""
        service = ICTService(name="Test Service")
        assert service.name == "Test Service"
        assert service.service_id.startswith("SVC-")
        assert service.criticality == ServiceCriticality.STANDARD
        assert service.availability_target_pct == 99.0

    def test_ict_service_with_all_fields(self, sample_service):
        """Test ICTService with all fields populated."""
        assert sample_service.name == "Cloud Compute"
        assert sample_service.supports_critical_function is True
        assert "trading" in sample_service.supported_functions
        assert sample_service.availability_target_pct == 99.9

    def test_ict_service_auto_id_generation(self):
        """Test automatic service ID generation."""
        service1 = ICTService(name="Service 1")
        service2 = ICTService(name="Service 2")
        assert service1.service_id != service2.service_id
        assert len(service1.service_id) == 12  # SVC-XXXXXXXX


class TestICTProvider:
    """Test ICTProvider data structure."""

    def test_ict_provider_creation(self):
        """Test ICTProvider creation with defaults."""
        provider = ICTProvider(name="Test Provider", country="DE")
        assert provider.name == "Test Provider"
        assert provider.provider_id.startswith("PRV-")
        assert provider.is_eu_provider is True  # Germany is EU
        assert provider.criticality == ProviderCriticality.LOW
        assert provider.status == ProviderStatus.ACTIVE

    def test_ict_provider_eu_detection(self):
        """Test automatic EU provider detection."""
        eu_provider = ICTProvider(name="EU Provider", country="FR")
        assert eu_provider.is_eu_provider is True

        non_eu_provider = ICTProvider(name="Non-EU Provider", country="US")
        assert non_eu_provider.is_eu_provider is False

    def test_ict_provider_id_uniqueness(self):
        """Test provider ID uniqueness."""
        provider1 = ICTProvider(name="Provider 1", country="DE")
        provider2 = ICTProvider(name="Provider 2", country="DE")
        assert provider1.provider_id != provider2.provider_id


class TestThirdPartyRisk:
    """Test ThirdPartyRisk data structure."""

    def test_third_party_risk_creation(self):
        """Test ThirdPartyRisk creation."""
        risk = ThirdPartyRisk(
            provider_id="PRV-12345678",
            category=RiskCategory.SECURITY,
            name="Security Control Gap",
            likelihood=4,
            impact=5,
        )
        assert risk.risk_id.startswith("TPR-")
        assert risk.category == RiskCategory.SECURITY
        assert risk.calculate_risk_score() == 20
        assert risk.risk_level == RiskLevel.CRITICAL

    def test_risk_score_calculation(self):
        """Test risk score calculation."""
        # Low risk (score <= 4)
        low_risk = ThirdPartyRisk(likelihood=1, impact=2)
        assert low_risk.calculate_risk_score() == 2
        assert low_risk.risk_level == RiskLevel.LOW

        # Medium risk (score 5-9)
        medium_risk = ThirdPartyRisk(likelihood=3, impact=3)
        assert medium_risk.calculate_risk_score() == 9
        assert medium_risk.risk_level == RiskLevel.MEDIUM

        # High risk (score 10-16)
        high_risk = ThirdPartyRisk(likelihood=4, impact=4)
        assert high_risk.calculate_risk_score() == 16
        assert high_risk.risk_level == RiskLevel.HIGH

        # Critical risk (score > 16)
        critical_risk = ThirdPartyRisk(likelihood=5, impact=5)
        assert critical_risk.calculate_risk_score() == 25
        assert critical_risk.risk_level == RiskLevel.CRITICAL


class TestThirdPartyRiskAssessment:
    """Test ThirdPartyRiskAssessment data structure."""

    def test_assessment_creation(self):
        """Test assessment creation with defaults."""
        assessment = ThirdPartyRiskAssessment(
            provider_id="PRV-12345678",
            assessment_type=AssessmentType.INITIAL,
        )
        assert assessment.assessment_id.startswith("TPA-")
        assert assessment.assessment_type == AssessmentType.INITIAL
        assert assessment.continue_relationship is True

    def test_assessment_risk_count(self):
        """Test assessment risk count by level."""
        assessment = ThirdPartyRiskAssessment(
            provider_id="PRV-12345678",
            risk_count_by_level={
                "critical": 1,
                "high": 2,
                "medium": 3,
                "low": 4,
            }
        )
        assert assessment.risk_count_by_level["critical"] == 1
        assert assessment.risk_count_by_level["high"] == 2


class TestDueDiligenceCheck:
    """Test DueDiligenceCheck data structure."""

    def test_due_diligence_check_creation(self):
        """Test due diligence check creation."""
        check = DueDiligenceCheck(
            provider_id="PRV-12345678",
            check_type="security_controls",
        )
        assert check.check_id.startswith("DDC-")
        assert check.status == DueDiligenceStatus.NOT_STARTED
        assert check.remediation_required is False


class TestProviderRelationshipEvent:
    """Test ProviderRelationshipEvent data structure."""

    def test_event_creation(self):
        """Test event creation."""
        event = ProviderRelationshipEvent(
            provider_id="PRV-12345678",
            event_type="onboarding",
            description="Provider onboarded",
        )
        assert event.event_id.startswith("PRE-")
        assert event.event_type == "onboarding"
        assert event.event_date is not None


# =============================================================================
# Configuration Tests
# =============================================================================

class TestThirdPartyRiskConfig:
    """Test ThirdPartyRiskConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ThirdPartyRiskConfig()
        assert config.periodic_assessment_months == 12
        assert config.critical_provider_assessment_months == 6
        assert config.require_due_diligence_before_onboarding is True
        assert config.max_providers_per_critical_function == 3

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ThirdPartyRiskConfig(
            periodic_assessment_months=6,
            require_due_diligence_before_onboarding=False,
            max_providers_per_critical_function=5,
        )
        assert config.periodic_assessment_months == 6
        assert config.require_due_diligence_before_onboarding is False
        assert config.max_providers_per_critical_function == 5


# =============================================================================
# Provider Management Tests
# =============================================================================

class TestProviderManagement:
    """Test provider management functionality."""

    def test_register_provider(self, manager):
        """Test provider registration."""
        provider = manager.register_provider(
            name="Test Provider",
            provider_type=ProviderType.CLOUD_SERVICE,
            country="DE",
        )
        assert provider is not None
        assert provider.name == "Test Provider"
        assert provider.provider_type == ProviderType.CLOUD_SERVICE
        assert provider.status == ProviderStatus.ONBOARDING

    def test_register_provider_with_critical_service(self, manager):
        """Test provider registration with critical service."""
        critical_service = ICTService(
            name="Critical Service",
            supports_critical_function=True,
        )
        provider = manager.register_provider(
            name="Critical Provider",
            provider_type=ProviderType.CLOUD_SERVICE,
            country="US",
            services=[critical_service],
        )
        assert provider.criticality == ProviderCriticality.CRITICAL
        assert provider.supports_critical_function is True

    def test_get_provider(self, manager, sample_provider):
        """Test getting provider by ID."""
        retrieved = manager.get_provider(sample_provider.provider_id)
        assert retrieved is not None
        assert retrieved.provider_id == sample_provider.provider_id

    def test_get_provider_not_found(self, manager):
        """Test getting non-existent provider."""
        result = manager.get_provider("PRV-NONEXIST")
        assert result is None

    def test_get_provider_by_name(self, manager, sample_provider):
        """Test getting provider by name."""
        retrieved = manager.get_provider_by_name("AWS")
        assert retrieved is not None
        assert retrieved.name == "AWS"

    def test_update_provider(self, manager, sample_provider):
        """Test updating provider."""
        updated = manager.update_provider(
            sample_provider.provider_id,
            primary_contact_email="test@example.com",
        )
        assert updated is not None
        assert updated.primary_contact_email == "test@example.com"

    def test_get_providers_by_criticality(self, manager):
        """Test getting providers by criticality."""
        # Register providers with different criticality
        manager.register_provider(
            name="Critical Provider",
            provider_type=ProviderType.CLOUD_SERVICE,
            country="DE",
            criticality=ProviderCriticality.CRITICAL,
        )
        manager.register_provider(
            name="Low Provider",
            provider_type=ProviderType.CONSULTING,
            country="DE",
            criticality=ProviderCriticality.LOW,
        )

        critical_providers = manager.get_providers_by_criticality(ProviderCriticality.CRITICAL)
        assert len(critical_providers) == 1
        assert critical_providers[0].name == "Critical Provider"

    def test_get_providers_by_type(self, manager):
        """Test getting providers by type."""
        manager.register_provider(
            name="Cloud Provider",
            provider_type=ProviderType.CLOUD_SERVICE,
            country="US",
        )
        manager.register_provider(
            name="Data Provider",
            provider_type=ProviderType.DATA_PROVIDER,
            country="UK",
        )

        cloud_providers = manager.get_providers_by_type(ProviderType.CLOUD_SERVICE)
        assert len(cloud_providers) == 1
        assert cloud_providers[0].provider_type == ProviderType.CLOUD_SERVICE

    def test_activate_provider(self, manager):
        """Test provider activation."""
        provider = manager.register_provider(
            name="Test Provider",
            provider_type=ProviderType.CLOUD_SERVICE,
            country="DE",
        )
        # Manager has require_due_diligence_before_onboarding=False
        activated = manager.activate_provider(provider.provider_id)
        assert activated is not None
        assert activated.status == ProviderStatus.ACTIVE

    def test_suspend_provider(self, manager, sample_provider):
        """Test provider suspension."""
        suspended = manager.suspend_provider(
            sample_provider.provider_id,
            reason="Security incident",
        )
        assert suspended is not None
        assert suspended.status == ProviderStatus.SUSPENDED

    def test_get_critical_providers(self, manager):
        """Test getting critical providers."""
        critical_service = ICTService(supports_critical_function=True)
        provider = manager.register_provider(
            name="Critical Provider",
            provider_type=ProviderType.CLOUD_SERVICE,
            country="DE",
            services=[critical_service],
        )
        # Activate the provider
        manager.activate_provider(provider.provider_id)

        critical_providers = manager.get_critical_providers()
        assert len(critical_providers) == 1


# =============================================================================
# Service Management Tests
# =============================================================================

class TestServiceManagement:
    """Test service management functionality."""

    def test_add_service_to_provider(self, manager, sample_provider):
        """Test adding service to provider."""
        new_service = ICTService(
            name="Storage Service",
            service_type="storage",
        )
        updated = manager.add_service_to_provider(
            sample_provider.provider_id,
            new_service,
        )
        assert updated is not None
        assert len(updated.services) == 2

    def test_get_services_by_provider(self, manager, sample_provider):
        """Test getting services for a provider."""
        services = manager.get_services_by_provider(sample_provider.provider_id)
        assert len(services) == 1
        assert services[0].name == "Cloud Compute"

    def test_get_critical_services(self, manager, sample_provider):
        """Test getting critical services."""
        critical_services = manager.get_critical_services()
        assert len(critical_services) == 1
        provider, service = critical_services[0]
        assert service.criticality == ServiceCriticality.CRITICAL_OR_IMPORTANT


# =============================================================================
# Risk Management Tests
# =============================================================================

class TestRiskManagement:
    """Test risk management functionality."""

    def test_identify_risk(self, manager, sample_provider):
        """Test risk identification."""
        risk = manager.identify_risk(
            provider_id=sample_provider.provider_id,
            category=RiskCategory.SECURITY,
            name="Weak Authentication",
            description="Provider uses weak authentication",
            likelihood=4,
            impact=4,
        )
        assert risk is not None
        assert risk.category == RiskCategory.SECURITY
        assert risk.risk_level == RiskLevel.HIGH

    def test_identify_risk_invalid_provider(self, manager):
        """Test risk identification with invalid provider."""
        risk = manager.identify_risk(
            provider_id="PRV-INVALID",
            category=RiskCategory.SECURITY,
            name="Test Risk",
        )
        assert risk is None

    def test_update_risk(self, manager, sample_provider):
        """Test risk update."""
        risk = manager.identify_risk(
            provider_id=sample_provider.provider_id,
            category=RiskCategory.OPERATIONAL,
            name="Test Risk",
            likelihood=2,
            impact=2,
        )
        updated = manager.update_risk(
            risk.risk_id,
            likelihood=5,
            impact=5,
        )
        assert updated is not None
        assert updated.risk_level == RiskLevel.CRITICAL

    def test_get_risks_for_provider(self, manager, sample_provider):
        """Test getting risks for a provider."""
        manager.identify_risk(
            provider_id=sample_provider.provider_id,
            category=RiskCategory.SECURITY,
            name="Risk 1",
        )
        manager.identify_risk(
            provider_id=sample_provider.provider_id,
            category=RiskCategory.OPERATIONAL,
            name="Risk 2",
        )

        risks = manager.get_risks_for_provider(sample_provider.provider_id)
        assert len(risks) == 2

    def test_get_risks_by_level(self, manager, sample_provider):
        """Test getting risks by level."""
        manager.identify_risk(
            provider_id=sample_provider.provider_id,
            category=RiskCategory.SECURITY,
            name="High Risk",
            likelihood=4,
            impact=4,
        )
        manager.identify_risk(
            provider_id=sample_provider.provider_id,
            category=RiskCategory.OPERATIONAL,
            name="Low Risk",
            likelihood=1,
            impact=2,
        )

        high_risks = manager.get_risks_by_level(RiskLevel.HIGH)
        assert len(high_risks) == 1
        assert high_risks[0].name == "High Risk"

    def test_accept_risk(self, manager, sample_provider):
        """Test risk acceptance."""
        risk = manager.identify_risk(
            provider_id=sample_provider.provider_id,
            category=RiskCategory.OPERATIONAL,
            name="Accepted Risk",
        )
        accepted = manager.accept_risk(
            risk.risk_id,
            accepted_by="Risk Manager",
            justification="Within tolerance",
        )
        assert accepted is not None
        assert accepted.treatment_strategy == "accept"
        assert accepted.accepted_by == "Risk Manager"


# =============================================================================
# Risk Assessment Tests
# =============================================================================

class TestRiskAssessment:
    """Test risk assessment functionality."""

    def test_perform_risk_assessment(self, manager, sample_provider):
        """Test performing risk assessment."""
        assessment = manager.perform_risk_assessment(
            provider_id=sample_provider.provider_id,
            assessment_type=AssessmentType.INITIAL,
            assessed_by="Test Assessor",
        )
        assert assessment is not None
        assert assessment.assessment_type == AssessmentType.INITIAL
        assert assessment.assessed_by == "Test Assessor"
        assert assessment.overall_rating in ["satisfactory", "needs_improvement", "unsatisfactory"]

    def test_assessment_with_critical_risks(self, manager, sample_provider):
        """Test assessment with critical risks."""
        manager.identify_risk(
            provider_id=sample_provider.provider_id,
            category=RiskCategory.SECURITY,
            name="Critical Risk",
            likelihood=5,
            impact=5,
        )
        assessment = manager.perform_risk_assessment(
            provider_id=sample_provider.provider_id,
        )
        assert assessment.overall_risk_level == RiskLevel.CRITICAL
        assert assessment.overall_rating == "unsatisfactory"
        assert assessment.continue_relationship is False

    def test_approve_assessment(self, manager, sample_provider):
        """Test assessment approval."""
        assessment = manager.perform_risk_assessment(
            provider_id=sample_provider.provider_id,
        )
        approved = manager.approve_assessment(
            assessment.assessment_id,
            approved_by="Senior Manager",
            conditions=["Implement MFA within 30 days"],
        )
        assert approved is not None
        assert approved.approved_by == "Senior Manager"
        assert len(approved.conditions_for_continuation) == 1

    def test_get_latest_assessment(self, manager, sample_provider):
        """Test getting latest assessment."""
        manager.perform_risk_assessment(
            provider_id=sample_provider.provider_id,
            assessment_type=AssessmentType.INITIAL,
        )
        manager.perform_risk_assessment(
            provider_id=sample_provider.provider_id,
            assessment_type=AssessmentType.PERIODIC,
        )

        latest = manager.get_latest_assessment(sample_provider.provider_id)
        assert latest is not None
        assert latest.assessment_type == AssessmentType.PERIODIC


# =============================================================================
# Due Diligence Tests
# =============================================================================

class TestDueDiligence:
    """Test due diligence functionality."""

    def test_start_due_diligence(self, manager, sample_provider):
        """Test starting due diligence process."""
        checks = manager.start_due_diligence(sample_provider.provider_id)
        assert len(checks) == 7  # Standard check types
        assert all(c.status == DueDiligenceStatus.IN_PROGRESS for c in checks)

    def test_start_due_diligence_specific_checks(self, manager, sample_provider):
        """Test starting due diligence with specific checks."""
        checks = manager.start_due_diligence(
            sample_provider.provider_id,
            check_types=["security_controls", "data_protection"],
        )
        assert len(checks) == 2

    def test_complete_due_diligence_check(self, manager, sample_provider):
        """Test completing due diligence check."""
        checks = manager.start_due_diligence(
            sample_provider.provider_id,
            check_types=["security_controls"],
        )
        check = checks[0]

        completed = manager.complete_due_diligence_check(
            check.check_id,
            result="passed",
            findings=["ISO 27001 certified"],
            performed_by="Security Team",
        )
        assert completed is not None
        assert completed.status == DueDiligenceStatus.COMPLETED
        assert completed.result == "passed"

    def test_due_diligence_with_remediation(self, manager, sample_provider):
        """Test due diligence requiring remediation."""
        checks = manager.start_due_diligence(
            sample_provider.provider_id,
            check_types=["security_controls"],
        )
        check = checks[0]

        completed = manager.complete_due_diligence_check(
            check.check_id,
            result="conditional",
            findings=["Gap in encryption"],
            remediation_required=True,
            remediation_actions=["Implement AES-256"],
        )
        assert completed.status == DueDiligenceStatus.REQUIRES_REMEDIATION
        assert completed.remediation_required is True

    def test_get_due_diligence_status(self, manager, sample_provider):
        """Test getting due diligence status."""
        manager.start_due_diligence(sample_provider.provider_id)

        status = manager.get_due_diligence_status(sample_provider.provider_id)
        assert status["total_checks"] == 7
        assert status["in_progress"] == 7
        assert status["overall_status"] == "in_progress"


# =============================================================================
# Control and Oversight Tests
# =============================================================================

class TestControlAndOversight:
    """Test control and oversight functionality."""

    def test_assess_control_over_provider(self, manager, sample_provider):
        """Test assessing control over provider."""
        assessment = manager.assess_control_over_provider(sample_provider.provider_id)
        assert "provider_id" in assessment
        assert "control_areas" in assessment
        assert "overall_control_level" in assessment

    def test_verify_supervisory_access(self, manager, sample_provider):
        """Test verifying supervisory access."""
        verification = manager.verify_supervisory_access(sample_provider.provider_id)
        assert "supervisory_access_ensured" in verification
        assert "areas" in verification


# =============================================================================
# Status and Reporting Tests
# =============================================================================

class TestStatusAndReporting:
    """Test status and reporting functionality."""

    def test_get_third_party_risk_status(self, manager, sample_provider):
        """Test getting overall third-party risk status."""
        status = manager.get_third_party_risk_status()
        assert "providers" in status
        assert "risks" in status
        assert "assessments" in status
        assert "compliance_indicators" in status
        assert status["providers"]["total"] == 1

    def test_export_provider_inventory(self, manager, sample_provider):
        """Test exporting provider inventory."""
        inventory = manager.export_provider_inventory()
        assert len(inventory) == 1
        assert inventory[0]["name"] == "AWS"
        assert "criticality" in inventory[0]
        assert "status" in inventory[0]


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_third_party_risk_management(self):
        """Test factory function."""
        manager = create_third_party_risk_management()
        assert manager is not None
        assert isinstance(manager, DORAThirdPartyRiskManagement)

    def test_create_with_config(self):
        """Test factory function with config."""
        config = ThirdPartyRiskConfig(periodic_assessment_months=24)
        manager = create_third_party_risk_management(config=config)
        assert manager.config.periodic_assessment_months == 24

    def test_get_provider_types(self):
        """Test getting provider types."""
        types = get_provider_types()
        assert len(types) == 12
        assert ProviderType.CLOUD_SERVICE in types

    def test_get_risk_categories(self):
        """Test getting risk categories."""
        categories = get_risk_categories()
        assert len(categories) == 10
        assert RiskCategory.SECURITY in categories

    def test_get_criticality_levels(self):
        """Test getting criticality levels."""
        levels = get_criticality_levels()
        assert len(levels) == 4
        assert ProviderCriticality.CRITICAL in levels


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Test thread safety of manager operations."""

    def test_concurrent_provider_registration(self, manager):
        """Test concurrent provider registration."""
        results = []
        errors = []

        def register_provider(name):
            try:
                provider = manager.register_provider(
                    name=name,
                    provider_type=ProviderType.CLOUD_SERVICE,
                    country="DE",
                )
                results.append(provider)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            t = threading.Thread(target=register_provider, args=(f"Provider_{i}",))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        # All providers should have unique IDs
        ids = [p.provider_id for p in results]
        assert len(set(ids)) == 10

    def test_concurrent_risk_identification(self, manager, sample_provider):
        """Test concurrent risk identification."""
        results = []
        errors = []

        def identify_risk(name):
            try:
                risk = manager.identify_risk(
                    provider_id=sample_provider.provider_id,
                    category=RiskCategory.SECURITY,
                    name=name,
                )
                results.append(risk)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            t = threading.Thread(target=identify_risk, args=(f"Risk_{i}",))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_provider_lifecycle(self, manager):
        """Test full provider lifecycle."""
        # 1. Register provider
        provider = manager.register_provider(
            name="Full Lifecycle Provider",
            provider_type=ProviderType.CLOUD_SERVICE,
            country="DE",
        )
        assert provider.status == ProviderStatus.ONBOARDING

        # 2. Start due diligence
        checks = manager.start_due_diligence(provider.provider_id)
        assert len(checks) > 0

        # 3. Complete all due diligence checks
        for check in checks:
            manager.complete_due_diligence_check(
                check.check_id,
                result="passed",
                performed_by="Compliance Team",
            )

        # 4. Verify due diligence completed
        status = manager.get_due_diligence_status(provider.provider_id)
        assert status["overall_status"] == "completed"

        # 5. Activate provider
        activated = manager.activate_provider(provider.provider_id)
        assert activated.status == ProviderStatus.ACTIVE

        # 6. Perform risk assessment
        assessment = manager.perform_risk_assessment(
            provider.provider_id,
            assessment_type=AssessmentType.INITIAL,
        )
        assert assessment is not None

        # 7. Suspend provider
        suspended = manager.suspend_provider(
            provider.provider_id,
            reason="Planned maintenance",
        )
        assert suspended.status == ProviderStatus.SUSPENDED

    def test_callback_integration(self):
        """Test callback integration."""
        escalation_calls = []
        notification_calls = []

        def escalation_callback(event_type, data):
            escalation_calls.append((event_type, data))

        def notification_callback(event_type, data):
            notification_calls.append((event_type, data))

        config = ThirdPartyRiskConfig(
            escalation_callback=escalation_callback,
            notification_callback=notification_callback,
            notify_on_risk_increase=True,
            log_all_events=False,
        )
        manager = DORAThirdPartyRiskManagement(config=config)

        # Register provider
        provider = manager.register_provider(
            name="Callback Test Provider",
            provider_type=ProviderType.CLOUD_SERVICE,
            country="DE",
        )

        # Create high risk (should trigger escalation)
        manager.identify_risk(
            provider_id=provider.provider_id,
            category=RiskCategory.SECURITY,
            name="Critical Security Gap",
            likelihood=5,
            impact=5,
        )

        assert len(escalation_calls) == 1
        assert escalation_calls[0][0] == "high_risk_identified"

        # Suspend provider (should trigger notification)
        manager.suspend_provider(provider.provider_id, reason="Test")
        assert len(notification_calls) == 1
        assert notification_calls[0][0] == "provider_suspended"
