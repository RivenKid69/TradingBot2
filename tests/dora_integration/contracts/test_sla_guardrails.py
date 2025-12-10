# -*- coding: utf-8 -*-
"""
Tests for DORA SLA Guardrails Module.

Tests SLA tier definitions, capacity validation, and commitment approval.
"""

import pytest
from datetime import datetime, timezone, timedelta
from dataclasses import asdict

from services.dora_integration.contracts.sla_guardrails import (
    # Main class
    SLAGuardrails,
    # Configuration
    SLAGuardrailsConfig,
    # Enumerations
    SLATier,
    CapacityStatus,
    ApprovalStatus,
    InfrastructureRequirement,
    OnCallRequirement,
    # Data structures
    SLATierDefinition,
    CapacityValidation,
    SLACommitmentRequest,
    CurrentCapacityState,
    # Factory and utility functions
    create_sla_guardrails,
    get_sla_tier_definitions,
    get_sla_tiers,
)


class TestEnumerations:
    """Test all enumerations."""

    def test_sla_tier_values(self):
        """Test SLATier enum values."""
        assert SLATier.STANDARD.value == "standard"
        assert SLATier.PROFESSIONAL.value == "professional"
        assert SLATier.ENTERPRISE.value == "enterprise"
        assert SLATier.CRITICAL.value == "critical"

    def test_capacity_status_values(self):
        """Test CapacityStatus enum values."""
        assert CapacityStatus.NOT_VALIDATED.value == "not_validated"
        assert CapacityStatus.VALIDATING.value == "validating"
        assert CapacityStatus.VALIDATED.value == "validated"
        assert CapacityStatus.VALIDATION_FAILED.value == "validation_failed"
        assert CapacityStatus.EXPIRED.value == "expired"

    def test_approval_status_values(self):
        """Test ApprovalStatus enum values."""
        assert ApprovalStatus.PENDING.value == "pending"
        assert ApprovalStatus.APPROVED.value == "approved"
        assert ApprovalStatus.REJECTED.value == "rejected"
        assert ApprovalStatus.CONDITIONAL.value == "conditional"
        assert ApprovalStatus.EXPIRED.value == "expired"

    def test_infrastructure_requirement_values(self):
        """Test InfrastructureRequirement enum values."""
        assert InfrastructureRequirement.SINGLE_AZ.value == "single_az"
        assert InfrastructureRequirement.MULTI_AZ.value == "multi_az"
        assert InfrastructureRequirement.MULTI_REGION.value == "multi_region"
        assert InfrastructureRequirement.DEDICATED_REGION.value == "dedicated_region"

    def test_oncall_requirement_values(self):
        """Test OnCallRequirement enum values."""
        assert OnCallRequirement.BUSINESS_HOURS.value == "business_hours"
        assert OnCallRequirement.EXTENDED_HOURS.value == "extended_hours"
        assert OnCallRequirement.ONCALL_ROTATION.value == "oncall_rotation"
        assert OnCallRequirement.DEDICATED_24_7.value == "dedicated_24_7"


class TestDataStructures:
    """Test data structures."""

    def test_sla_tier_definition_creation(self):
        """Test SLATierDefinition dataclass."""
        defn = SLATierDefinition(
            tier=SLATier.STANDARD,
            name="Standard",
            description="Basic SLA",
            availability_target_pct=99.5,
        )
        assert defn.tier == SLATier.STANDARD
        assert defn.availability_target_pct == 99.5
        assert defn.tier_id == "TIER-STANDARD"

    def test_capacity_validation_creation(self):
        """Test CapacityValidation dataclass."""
        validation = CapacityValidation(
            tier=SLATier.PROFESSIONAL,
            validated_by="Test Engineer",
        )
        assert validation.validation_id.startswith("VAL-")
        assert validation.tier == SLATier.PROFESSIONAL
        assert validation.status == CapacityStatus.NOT_VALIDATED

    def test_capacity_validation_is_valid(self):
        """Test CapacityValidation is_valid property."""
        # Create valid validation
        validation = CapacityValidation(
            tier=SLATier.STANDARD,
            status=CapacityStatus.VALIDATED,
            expiry_date=(datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
        )
        assert validation.is_valid is True

        # Create expired validation
        expired = CapacityValidation(
            tier=SLATier.STANDARD,
            status=CapacityStatus.VALIDATED,
            expiry_date=(datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
        )
        assert expired.is_valid is False

    def test_capacity_validation_all_checks_passed(self):
        """Test CapacityValidation all_checks_passed property."""
        validation = CapacityValidation(
            tier=SLATier.STANDARD,
            infrastructure_check=True,
            replication_check=True,
            backup_check=True,
            oncall_check=True,
            certification_check=True,
        )
        assert validation.all_checks_passed is True

        validation.oncall_check = False
        assert validation.all_checks_passed is False

    def test_sla_commitment_request_creation(self):
        """Test SLACommitmentRequest dataclass."""
        request = SLACommitmentRequest(
            client_id="CLT-001",
            client_name="Test Client",
            requested_tier=SLATier.PROFESSIONAL,
            requested_by="Sales Rep",
        )
        assert request.request_id.startswith("SLA-")
        assert request.client_id == "CLT-001"
        assert request.approval_status == ApprovalStatus.PENDING

    def test_current_capacity_state_creation(self):
        """Test CurrentCapacityState dataclass."""
        state = CurrentCapacityState()
        assert state.has_multi_az is True
        assert state.has_multi_region is False
        assert state.max_clients_per_tier["standard"] == 100


class TestFactoryFunctions:
    """Test factory and utility functions."""

    def test_create_sla_guardrails(self):
        """Test factory function creates instance."""
        guardrails = create_sla_guardrails()
        assert isinstance(guardrails, SLAGuardrails)

    def test_create_with_config(self):
        """Test factory with custom config."""
        config = SLAGuardrailsConfig(
            validation_expiry_days=60,
            approval_expiry_days=14,
        )
        guardrails = create_sla_guardrails(config)
        assert guardrails.config.validation_expiry_days == 60
        assert guardrails.config.approval_expiry_days == 14

    def test_get_sla_tier_definitions(self):
        """Test get_sla_tier_definitions returns all tiers."""
        definitions = get_sla_tier_definitions()
        assert len(definitions) == len(SLATier)
        assert SLATier.STANDARD in definitions
        assert SLATier.PROFESSIONAL in definitions
        assert SLATier.ENTERPRISE in definitions
        assert SLATier.CRITICAL in definitions

    def test_get_sla_tiers_summary(self):
        """Test get_sla_tiers returns summaries."""
        tiers = get_sla_tiers()
        assert len(tiers) == len(SLATier)
        for tier in tiers:
            assert "tier" in tier
            assert "name" in tier
            assert "availability" in tier
            assert "rto_hours" in tier


class TestSLATierDefinitions:
    """Test SLA tier definitions match DORA requirements."""

    def test_standard_tier(self):
        """Test standard tier definition."""
        definitions = get_sla_tier_definitions()
        standard = definitions[SLATier.STANDARD]
        assert standard.availability_target_pct == 99.5
        assert standard.requires_multi_az is False
        assert standard.requires_multi_region is False
        assert standard.requires_24_7_oncall is False

    def test_professional_tier(self):
        """Test professional tier definition."""
        definitions = get_sla_tier_definitions()
        professional = definitions[SLATier.PROFESSIONAL]
        assert professional.availability_target_pct == 99.9
        assert professional.requires_multi_az is True
        assert professional.requires_sync_replication is True
        assert professional.requires_soc2 is True

    def test_enterprise_tier(self):
        """Test enterprise tier definition."""
        definitions = get_sla_tier_definitions()
        enterprise = definitions[SLATier.ENTERPRISE]
        assert enterprise.availability_target_pct == 99.95
        assert enterprise.requires_multi_region is True
        assert enterprise.requires_24_7_oncall is True
        assert enterprise.requires_iso27001 is True

    def test_critical_tier(self):
        """Test critical tier definition (DORA critical functions)."""
        definitions = get_sla_tier_definitions()
        critical = definitions[SLATier.CRITICAL]
        assert critical.availability_target_pct == 99.99
        assert critical.rto_hours == 1.0
        assert critical.rpo_minutes == 5
        assert critical.requires_multi_region is True
        assert critical.requires_24_7_oncall is True
        assert critical.min_oncall_engineers == 6

    def test_tier_progression(self):
        """Test tiers have progressively higher requirements."""
        definitions = get_sla_tier_definitions()
        tiers = [
            definitions[SLATier.STANDARD],
            definitions[SLATier.PROFESSIONAL],
            definitions[SLATier.ENTERPRISE],
            definitions[SLATier.CRITICAL],
        ]
        # Availability should increase
        for i in range(len(tiers) - 1):
            assert tiers[i].availability_target_pct < tiers[i + 1].availability_target_pct

        # RTO should decrease
        for i in range(len(tiers) - 1):
            assert tiers[i].rto_hours >= tiers[i + 1].rto_hours


class TestSLAGuardrails:
    """Test SLAGuardrails main class."""

    @pytest.fixture
    def guardrails(self):
        """Create guardrails instance for testing."""
        return SLAGuardrails()

    def test_initialization(self, guardrails):
        """Test guardrails initialization."""
        assert guardrails is not None
        assert len(guardrails.tier_definitions) == len(SLATier)

    def test_get_tier_definition(self, guardrails):
        """Test get_tier_definition."""
        defn = guardrails.get_tier_definition(SLATier.STANDARD)
        assert defn.tier == SLATier.STANDARD

    def test_get_available_tiers_default(self, guardrails):
        """Test get_available_tiers with default capacity."""
        # Default capacity has multi-az and sync replication
        available = guardrails.get_available_tiers()
        assert SLATier.STANDARD in available
        # Professional requires multi-az + sync replication + 2 engineers
        assert SLATier.PROFESSIONAL in available
        # Enterprise requires multi-region which is false by default
        assert SLATier.ENTERPRISE not in available
        # Critical requires multi-region
        assert SLATier.CRITICAL not in available

    def test_get_available_tiers_full_capacity(self, guardrails):
        """Test get_available_tiers with full capacity."""
        guardrails.update_capacity_state(
            has_multi_region=True,
            has_24_7_coverage=True,
            current_oncall_engineers=6,
            has_iso27001=True,
            current_backup_frequency_minutes=5,
        )
        available = guardrails.get_available_tiers()
        assert SLATier.STANDARD in available
        assert SLATier.PROFESSIONAL in available
        assert SLATier.ENTERPRISE in available
        assert SLATier.CRITICAL in available


class TestCapacityValidation:
    """Test capacity validation functionality."""

    @pytest.fixture
    def guardrails(self):
        """Create guardrails instance for testing."""
        return SLAGuardrails()

    def test_validate_standard_tier(self, guardrails):
        """Test validation for standard tier."""
        validation = guardrails.validate_capacity_for_tier(
            tier=SLATier.STANDARD,
            validated_by="Test Engineer",
        )
        assert validation.status == CapacityStatus.VALIDATED
        assert validation.all_checks_passed is True

    def test_validate_professional_tier(self, guardrails):
        """Test validation for professional tier."""
        validation = guardrails.validate_capacity_for_tier(
            tier=SLATier.PROFESSIONAL,
            validated_by="Test Engineer",
        )
        # Default capacity should pass professional
        assert validation.infrastructure_check is True
        assert validation.replication_check is True

    def test_validate_enterprise_tier_fails(self, guardrails):
        """Test validation for enterprise tier fails without multi-region."""
        validation = guardrails.validate_capacity_for_tier(
            tier=SLATier.ENTERPRISE,
            validated_by="Test Engineer",
        )
        # Should fail because no multi-region
        assert validation.status == CapacityStatus.VALIDATION_FAILED
        assert validation.infrastructure_check is False

    def test_validate_critical_tier_fails(self, guardrails):
        """Test validation for critical tier fails without requirements."""
        validation = guardrails.validate_capacity_for_tier(
            tier=SLATier.CRITICAL,
            validated_by="Test Engineer",
        )
        assert validation.status == CapacityStatus.VALIDATION_FAILED
        assert len(validation.issues_found) > 0

    def test_validation_expiry_date(self, guardrails):
        """Test validation has expiry date."""
        validation = guardrails.validate_capacity_for_tier(
            tier=SLATier.STANDARD,
            validated_by="Test Engineer",
        )
        assert validation.expiry_date is not None
        expiry = datetime.fromisoformat(validation.expiry_date.replace("Z", "+00:00"))
        assert expiry > datetime.now(timezone.utc)

    def test_validation_check_details(self, guardrails):
        """Test validation includes check details."""
        validation = guardrails.validate_capacity_for_tier(
            tier=SLATier.STANDARD,
            validated_by="Test Engineer",
        )
        assert "infrastructure" in validation.check_details
        assert "replication" in validation.check_details
        assert "backup" in validation.check_details
        assert "oncall" in validation.check_details
        assert "certifications" in validation.check_details


class TestCommitmentRequests:
    """Test SLA commitment request functionality."""

    @pytest.fixture
    def guardrails(self):
        """Create guardrails instance for testing."""
        return SLAGuardrails()

    def test_request_sla_commitment_available_tier(self, guardrails):
        """Test requesting commitment for available tier."""
        request = guardrails.request_sla_commitment(
            client_id="CLT-001",
            client_name="Test Client",
            requested_tier=SLATier.STANDARD,
            requested_by="Sales Rep",
            services_in_scope=["api", "data"],
        )
        assert request.approval_status == ApprovalStatus.PENDING
        assert request.client_name == "Test Client"

    def test_request_sla_commitment_unavailable_tier(self, guardrails):
        """Test requesting commitment for unavailable tier."""
        request = guardrails.request_sla_commitment(
            client_id="CLT-001",
            client_name="Test Client",
            requested_tier=SLATier.ENTERPRISE,
            requested_by="Sales Rep",
            services_in_scope=["api"],
        )
        # Should be auto-rejected because tier not available
        assert request.approval_status == ApprovalStatus.REJECTED
        assert "not currently available" in request.approval_notes

    def test_request_for_critical_function(self, guardrails):
        """Test request marked as critical function."""
        request = guardrails.request_sla_commitment(
            client_id="CLT-001",
            client_name="Bank Client",
            requested_tier=SLATier.PROFESSIONAL,
            requested_by="Sales Rep",
            services_in_scope=["trading"],
            is_critical_function=True,
            business_justification="Critical trading function",
        )
        assert request.is_critical_function is True
        assert request.business_justification == "Critical trading function"


class TestCommitmentApproval:
    """Test commitment approval workflow."""

    @pytest.fixture
    def guardrails_with_request(self):
        """Create guardrails with pending request."""
        guardrails = SLAGuardrails()
        request = guardrails.request_sla_commitment(
            client_id="CLT-001",
            client_name="Test Client",
            requested_tier=SLATier.STANDARD,
            requested_by="Sales Rep",
            services_in_scope=["api"],
        )
        return guardrails, request

    def test_approve_commitment(self, guardrails_with_request):
        """Test approving commitment."""
        guardrails, request = guardrails_with_request
        approved = guardrails.approve_commitment(
            request_id=request.request_id,
            approved_by="Engineering Lead",
        )
        assert approved.approval_status == ApprovalStatus.APPROVED
        assert approved.approved_by == "Engineering Lead"
        assert approved.approval_date is not None

    def test_approve_with_conditions(self, guardrails_with_request):
        """Test conditional approval."""
        guardrails, request = guardrails_with_request
        approved = guardrails.approve_commitment(
            request_id=request.request_id,
            approved_by="Engineering Lead",
            conditions=["Client must complete onboarding", "SLA effective in 30 days"],
            notes="Approved with conditions",
        )
        assert approved.approval_status == ApprovalStatus.CONDITIONAL
        assert len(approved.conditions) == 2

    def test_approve_unknown_request(self, guardrails_with_request):
        """Test approving unknown request raises error."""
        guardrails, _ = guardrails_with_request
        with pytest.raises(ValueError, match="not found"):
            guardrails.approve_commitment(
                request_id="UNKNOWN-ID",
                approved_by="Engineering Lead",
            )

    def test_reject_commitment(self, guardrails_with_request):
        """Test rejecting commitment."""
        guardrails, request = guardrails_with_request
        rejected = guardrails.reject_commitment(
            request_id=request.request_id,
            rejected_by="Engineering Lead",
            reason="Insufficient capacity for client's requirements",
        )
        assert rejected.approval_status == ApprovalStatus.REJECTED
        assert "Insufficient capacity" in rejected.approval_notes

    def test_commitment_expiry_date(self, guardrails_with_request):
        """Test approved commitment has expiry date."""
        guardrails, request = guardrails_with_request
        approved = guardrails.approve_commitment(
            request_id=request.request_id,
            approved_by="Engineering Lead",
        )
        assert approved.commitment_expiry_date is not None


class TestCapacityStateManagement:
    """Test capacity state management."""

    @pytest.fixture
    def guardrails(self):
        """Create guardrails instance for testing."""
        return SLAGuardrails()

    def test_update_capacity_state(self, guardrails):
        """Test updating capacity state."""
        state = guardrails.update_capacity_state(
            has_multi_region=True,
            validated_by="Ops Engineer",
        )
        assert state.has_multi_region is True
        assert state.validated_by == "Ops Engineer"

    def test_update_capacity_enables_tiers(self, guardrails):
        """Test updating capacity enables more tiers."""
        # Initially enterprise not available
        available_before = guardrails.get_available_tiers()
        assert SLATier.ENTERPRISE not in available_before

        # Enable multi-region, 24/7, and other requirements
        guardrails.update_capacity_state(
            has_multi_region=True,
            has_24_7_coverage=True,
            current_oncall_engineers=4,
            has_iso27001=True,
            current_backup_frequency_minutes=15,
        )

        available_after = guardrails.get_available_tiers()
        assert SLATier.ENTERPRISE in available_after

    def test_update_partial_state(self, guardrails):
        """Test partial state update preserves other values."""
        original = guardrails.current_capacity.has_multi_az
        guardrails.update_capacity_state(has_multi_region=True)
        assert guardrails.current_capacity.has_multi_az == original


class TestPendingApprovals:
    """Test pending approval tracking."""

    @pytest.fixture
    def guardrails_with_requests(self):
        """Create guardrails with multiple requests."""
        guardrails = SLAGuardrails()
        r1 = guardrails.request_sla_commitment(
            client_id="CLT-001",
            client_name="Client 1",
            requested_tier=SLATier.STANDARD,
            requested_by="Sales 1",
            services_in_scope=["api"],
        )
        r2 = guardrails.request_sla_commitment(
            client_id="CLT-002",
            client_name="Client 2",
            requested_tier=SLATier.STANDARD,
            requested_by="Sales 2",
            services_in_scope=["data"],
        )
        return guardrails, r1, r2

    def test_get_pending_approvals(self, guardrails_with_requests):
        """Test get_pending_approvals."""
        guardrails, r1, r2 = guardrails_with_requests
        pending = guardrails.get_pending_approvals()
        assert len(pending) == 2

    def test_pending_decreases_after_approval(self, guardrails_with_requests):
        """Test pending count decreases after approval."""
        guardrails, r1, r2 = guardrails_with_requests
        guardrails.approve_commitment(r1.request_id, "Lead")
        pending = guardrails.get_pending_approvals()
        assert len(pending) == 1


class TestExpiringValidations:
    """Test expiring validation tracking."""

    @pytest.fixture
    def guardrails(self):
        """Create guardrails instance for testing."""
        config = SLAGuardrailsConfig(validation_expiry_days=30)
        return SLAGuardrails(config)

    def test_get_expiring_validations(self, guardrails):
        """Test get_expiring_validations."""
        # Create a validation
        guardrails.validate_capacity_for_tier(
            tier=SLATier.STANDARD,
            validated_by="Engineer",
        )
        # All new validations expire in 30 days
        expiring = guardrails.get_expiring_validations(days=31)
        assert len(expiring) == 1

    def test_expiring_within_window(self, guardrails):
        """Test expiring within specific window."""
        guardrails.validate_capacity_for_tier(
            tier=SLATier.STANDARD,
            validated_by="Engineer",
        )
        # Check expiring within 14 days - should be empty as expiry is 30 days
        expiring = guardrails.get_expiring_validations(days=14)
        assert len(expiring) == 0


class TestCapacityReporting:
    """Test capacity reporting functionality."""

    @pytest.fixture
    def guardrails_with_data(self):
        """Create guardrails with various data."""
        guardrails = SLAGuardrails()
        # Create validation
        guardrails.validate_capacity_for_tier(SLATier.STANDARD, "Engineer")
        # Create request
        guardrails.request_sla_commitment(
            client_id="CLT-001",
            client_name="Client",
            requested_tier=SLATier.STANDARD,
            requested_by="Sales",
            services_in_scope=["api"],
        )
        return guardrails

    def test_generate_capacity_report(self, guardrails_with_data):
        """Test generate_capacity_report."""
        report = guardrails_with_data.generate_capacity_report()
        assert "report_date" in report
        assert "capacity_state" in report
        assert "available_tiers" in report
        assert "unavailable_tiers" in report
        assert "pending_approvals" in report
        assert "total_validations" in report
        assert "tier_definitions" in report

    def test_report_tier_availability(self, guardrails_with_data):
        """Test report shows tier availability correctly."""
        report = guardrails_with_data.generate_capacity_report()
        assert "standard" in report["available_tiers"]
        assert "enterprise" in report["unavailable_tiers"]

    def test_report_tier_definitions(self, guardrails_with_data):
        """Test report includes tier definitions."""
        report = guardrails_with_data.generate_capacity_report()
        assert len(report["tier_definitions"]) == len(SLATier)
        for tier_name, defn in report["tier_definitions"].items():
            assert "name" in defn
            assert "availability_target" in defn
            assert "available" in defn


class TestConfiguration:
    """Test configuration options."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SLAGuardrailsConfig()
        assert config.validation_expiry_days == 90
        assert config.approval_expiry_days == 30
        assert config.require_engineering_approval is True
        assert config.require_revalidation_on_tier_change is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SLAGuardrailsConfig(
            validation_expiry_days=60,
            max_tier_without_multi_az=SLATier.PROFESSIONAL,
            notify_on_validation_expiry=False,
        )
        assert config.validation_expiry_days == 60
        assert config.max_tier_without_multi_az == SLATier.PROFESSIONAL
        assert config.notify_on_validation_expiry is False

    def test_callbacks(self):
        """Test callback configuration."""
        callback_data = {}

        def approval_callback(event_type, data):
            callback_data["event"] = event_type
            callback_data["data"] = data

        config = SLAGuardrailsConfig(approval_callback=approval_callback)
        guardrails = SLAGuardrails(config)
        guardrails.request_sla_commitment(
            client_id="CLT-001",
            client_name="Test",
            requested_tier=SLATier.STANDARD,
            requested_by="Sales",
            services_in_scope=["api"],
        )
        assert callback_data.get("event") == "request_created"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_approve_unavailable_tier(self):
        """Test approval rejected if tier becomes unavailable."""
        guardrails = SLAGuardrails()
        # Create request for available tier
        request = guardrails.request_sla_commitment(
            client_id="CLT-001",
            client_name="Client",
            requested_tier=SLATier.PROFESSIONAL,
            requested_by="Sales",
            services_in_scope=["api"],
        )
        # Reduce capacity to make tier unavailable
        guardrails.update_capacity_state(
            has_multi_az=False,
            has_sync_replication=False,
        )
        # Try to approve
        result = guardrails.approve_commitment(
            request_id=request.request_id,
            approved_by="Lead",
        )
        assert result.approval_status == ApprovalStatus.REJECTED
        assert "no longer available" in result.approval_notes

    def test_validation_failed_status(self):
        """Test validation failed status set correctly."""
        guardrails = SLAGuardrails()
        # Reduce capacity
        guardrails.update_capacity_state(
            has_multi_az=False,
            has_soc2=False,
        )
        validation = guardrails.validate_capacity_for_tier(
            tier=SLATier.PROFESSIONAL,
            validated_by="Engineer",
        )
        assert validation.status == CapacityStatus.VALIDATION_FAILED
        assert len(validation.issues_found) > 0
