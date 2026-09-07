# -*- coding: utf-8 -*-
"""
Tests for DORA SLA Guardrails Module.

Comprehensive test coverage for:
- SLA tier definitions and validation
- Capacity validation workflows
- Engineering approval process
- Tier availability checks

Reference: DORA Article 30(2)(e), 30(3)(a)
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, MagicMock

from services.dora_integration.contracts import (
    # Enums
    SLATier,
    CapacityStatus,
    ApprovalStatus,
    InfrastructureRequirement,
    OnCallRequirement,
    # Data structures
    SLATierDefinition,
    CapacityValidation,
    SLACommitmentRequest,
    SLAGuardrailsConfig,
    CurrentCapacityState,
    # Main class
    SLAGuardrails,
    # Factory functions
    create_sla_guardrails,
    get_sla_tier_definitions,
    get_sla_tiers,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Create test configuration."""
    return SLAGuardrailsConfig(
        validation_expiry_days=90,
        approval_expiry_days=30,
        require_engineering_approval=True,
    )


@pytest.fixture
def guardrails(config):
    """Create SLA Guardrails instance."""
    return SLAGuardrails(config=config)


@pytest.fixture
def guardrails_with_capacity(guardrails):
    """Create guardrails with standard capacity state."""
    guardrails.update_capacity_state(
        has_multi_az=True,
        has_multi_region=False,
        has_sync_replication=True,
        current_backup_frequency_minutes=30,
        current_oncall_mode=OnCallRequirement.EXTENDED_HOURS,
        current_oncall_engineers=2,
        has_24_7_coverage=False,
        has_soc2=True,
        has_iso27001=False,
        validated_by="test_engineer",
    )
    return guardrails


# =============================================================================
# Enumeration Tests
# =============================================================================


class TestEnumerations:
    """Test all enumeration classes."""

    def test_sla_tier_values(self):
        """Test SLATier enum values."""
        assert SLATier.STANDARD.value == "standard"
        assert SLATier.PROFESSIONAL.value == "professional"
        assert SLATier.ENTERPRISE.value == "enterprise"
        assert SLATier.CRITICAL.value == "critical"
        assert len(SLATier) == 4

    def test_capacity_status_values(self):
        """Test CapacityStatus enum values."""
        assert CapacityStatus.NOT_VALIDATED.value == "not_validated"
        assert CapacityStatus.VALIDATING.value == "validating"
        assert CapacityStatus.VALIDATED.value == "validated"
        assert CapacityStatus.VALIDATION_FAILED.value == "validation_failed"
        assert CapacityStatus.EXPIRED.value == "expired"
        assert len(CapacityStatus) == 5

    def test_approval_status_values(self):
        """Test ApprovalStatus enum values."""
        assert ApprovalStatus.PENDING.value == "pending"
        assert ApprovalStatus.APPROVED.value == "approved"
        assert ApprovalStatus.REJECTED.value == "rejected"
        assert ApprovalStatus.CONDITIONAL.value == "conditional"
        assert ApprovalStatus.EXPIRED.value == "expired"
        assert len(ApprovalStatus) == 5

    def test_infrastructure_requirement_values(self):
        """Test InfrastructureRequirement enum values."""
        assert InfrastructureRequirement.SINGLE_AZ.value == "single_az"
        assert InfrastructureRequirement.MULTI_AZ.value == "multi_az"
        assert InfrastructureRequirement.MULTI_REGION.value == "multi_region"
        assert InfrastructureRequirement.DEDICATED_REGION.value == "dedicated_region"
        assert len(InfrastructureRequirement) == 4

    def test_oncall_requirement_values(self):
        """Test OnCallRequirement enum values."""
        assert OnCallRequirement.BUSINESS_HOURS.value == "business_hours"
        assert OnCallRequirement.EXTENDED_HOURS.value == "extended_hours"
        assert OnCallRequirement.ONCALL_ROTATION.value == "oncall_rotation"
        assert OnCallRequirement.DEDICATED_24_7.value == "dedicated_24_7"
        assert len(OnCallRequirement) == 4


# =============================================================================
# Data Structure Tests
# =============================================================================


class TestDataStructures:
    """Tests for data structures."""

    def test_sla_tier_definition_creation(self):
        """Test SLATierDefinition creation."""
        defn = SLATierDefinition(
            tier=SLATier.STANDARD,
            name="Standard",
            description="Basic SLA",
            availability_target_pct=99.5,
        )
        assert defn.tier == SLATier.STANDARD
        assert defn.name == "Standard"
        assert defn.availability_target_pct == 99.5
        assert defn.tier_id == "TIER-STANDARD"

    def test_capacity_validation_auto_init(self):
        """Test CapacityValidation auto-initialization."""
        validation = CapacityValidation(
            tier=SLATier.PROFESSIONAL,
            validated_by="test_user",
        )
        assert validation.validation_id.startswith("VAL-")
        assert validation.validation_date
        assert validation.status == CapacityStatus.NOT_VALIDATED

    def test_capacity_validation_is_valid(self):
        """Test CapacityValidation is_valid property."""
        # Valid validation
        validation = CapacityValidation(
            tier=SLATier.STANDARD,
            status=CapacityStatus.VALIDATED,
            expiry_date=(datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
        )
        assert validation.is_valid is True

        # Expired validation
        expired = CapacityValidation(
            tier=SLATier.STANDARD,
            status=CapacityStatus.VALIDATED,
            expiry_date=(datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
        )
        assert expired.is_valid is False

        # Not validated
        not_validated = CapacityValidation(
            tier=SLATier.STANDARD,
            status=CapacityStatus.NOT_VALIDATED,
        )
        assert not_validated.is_valid is False

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

        # One check failed
        validation.oncall_check = False
        assert validation.all_checks_passed is False

    def test_sla_commitment_request_auto_init(self):
        """Test SLACommitmentRequest auto-initialization."""
        request = SLACommitmentRequest(
            client_id="CLIENT-001",
            client_name="Test Bank",
            requested_tier=SLATier.PROFESSIONAL,
            requested_by="sales_rep",
        )
        assert request.request_id.startswith("SLA-")
        assert request.request_date
        assert request.approval_status == ApprovalStatus.PENDING

    def test_current_capacity_state_defaults(self):
        """Test CurrentCapacityState default values."""
        state = CurrentCapacityState()
        assert state.has_multi_az is True
        assert state.has_multi_region is False
        assert state.has_24_7_coverage is False
        assert state.state_date
        assert "standard" in state.max_clients_per_tier


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfiguration:
    """Tests for configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SLAGuardrailsConfig()
        assert config.validation_expiry_days == 90
        assert config.approval_expiry_days == 30
        assert config.require_engineering_approval is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SLAGuardrailsConfig(
            validation_expiry_days=60,
            approval_expiry_days=14,
        )
        assert config.validation_expiry_days == 60
        assert config.approval_expiry_days == 14

    def test_config_with_callbacks(self):
        """Test configuration with callbacks."""
        approval_fn = MagicMock()
        validation_fn = MagicMock()
        config = SLAGuardrailsConfig(
            approval_callback=approval_fn,
            validation_callback=validation_fn,
        )
        assert config.approval_callback == approval_fn
        assert config.validation_callback == validation_fn


# =============================================================================
# SLA Tier Definitions Tests
# =============================================================================


class TestSLATierDefinitions:
    """Tests for SLA tier definitions."""

    def test_get_sla_tier_definitions_returns_all_tiers(self):
        """Test that all tiers are defined."""
        definitions = get_sla_tier_definitions()
        assert SLATier.STANDARD in definitions
        assert SLATier.PROFESSIONAL in definitions
        assert SLATier.ENTERPRISE in definitions
        assert SLATier.CRITICAL in definitions

    def test_standard_tier_definition(self):
        """Test Standard tier definition."""
        definitions = get_sla_tier_definitions()
        standard = definitions[SLATier.STANDARD]
        assert standard.availability_target_pct == 99.5
        assert standard.requires_multi_az is False
        assert standard.requires_24_7_oncall is False

    def test_professional_tier_definition(self):
        """Test Professional tier definition."""
        definitions = get_sla_tier_definitions()
        professional = definitions[SLATier.PROFESSIONAL]
        assert professional.availability_target_pct == 99.9
        assert professional.requires_multi_az is True
        assert professional.requires_sync_replication is True
        assert professional.requires_24_7_oncall is False

    def test_enterprise_tier_definition(self):
        """Test Enterprise tier definition."""
        definitions = get_sla_tier_definitions()
        enterprise = definitions[SLATier.ENTERPRISE]
        assert enterprise.availability_target_pct == 99.95
        assert enterprise.requires_multi_region is True
        assert enterprise.requires_24_7_oncall is True
        assert enterprise.requires_soc2 is True
        assert enterprise.requires_iso27001 is True

    def test_critical_tier_definition(self):
        """Test Critical tier definition."""
        definitions = get_sla_tier_definitions()
        critical = definitions[SLATier.CRITICAL]
        assert critical.availability_target_pct == 99.99
        assert critical.min_oncall_engineers == 6
        assert critical.rto_hours == 1.0
        assert critical.rpo_minutes == 5

    def test_get_sla_tiers_returns_list(self):
        """Test get_sla_tiers factory function."""
        tiers = get_sla_tiers()
        assert isinstance(tiers, list)
        assert len(tiers) == 4
        tier_names = [t["tier"] for t in tiers]
        assert "standard" in tier_names
        assert "professional" in tier_names


# =============================================================================
# SLA Guardrails Service Tests
# =============================================================================


class TestSLAGuardrailsService:
    """Tests for SLA Guardrails service."""

    def test_create_sla_guardrails_factory(self):
        """Test factory function creates instance."""
        guardrails = create_sla_guardrails()
        assert isinstance(guardrails, SLAGuardrails)
        assert guardrails.tier_definitions is not None

    def test_get_tier_definition(self, guardrails):
        """Test getting specific tier definition."""
        defn = guardrails.get_tier_definition(SLATier.STANDARD)
        assert defn.tier == SLATier.STANDARD
        assert defn.name == "Standard"

    def test_get_available_tiers_default(self, guardrails):
        """Test getting available tiers with default capacity."""
        # Default capacity should allow Standard
        available = guardrails.get_available_tiers()
        assert SLATier.STANDARD in available

    def test_get_available_tiers_with_capacity(self, guardrails_with_capacity):
        """Test available tiers with specific capacity."""
        available = guardrails_with_capacity.get_available_tiers()
        # Standard should be available (has multi-az, sync replication)
        assert SLATier.STANDARD in available
        # Professional should be available (has multi-az, sync replication, 2 engineers)
        assert SLATier.PROFESSIONAL in available
        # Enterprise should NOT be available (no multi-region, no 24/7)
        assert SLATier.ENTERPRISE not in available
        # Critical should NOT be available
        assert SLATier.CRITICAL not in available

    def test_update_capacity_state(self, guardrails):
        """Test updating capacity state."""
        state = guardrails.update_capacity_state(
            has_multi_az=True,
            has_24_7_coverage=True,
            current_oncall_engineers=4,
            validated_by="admin",
        )
        assert state.has_multi_az is True
        assert state.has_24_7_coverage is True
        assert state.current_oncall_engineers == 4
        assert state.validated_by == "admin"


# =============================================================================
# Capacity Validation Tests
# =============================================================================


class TestCapacityValidation:
    """Tests for capacity validation workflow."""

    def test_validate_capacity_standard_tier(self, guardrails_with_capacity):
        """Test validation for Standard tier succeeds."""
        validation = guardrails_with_capacity.validate_capacity_for_tier(
            tier=SLATier.STANDARD,
            validated_by="engineer",
        )
        assert validation.status == CapacityStatus.VALIDATED
        assert validation.all_checks_passed is True

    def test_validate_capacity_professional_tier(self, guardrails_with_capacity):
        """Test validation for Professional tier."""
        validation = guardrails_with_capacity.validate_capacity_for_tier(
            tier=SLATier.PROFESSIONAL,
            validated_by="engineer",
        )
        assert validation.status == CapacityStatus.VALIDATED
        assert validation.infrastructure_check is True
        assert validation.replication_check is True

    def test_validate_capacity_enterprise_fails(self, guardrails_with_capacity):
        """Test validation for Enterprise tier fails (no multi-region)."""
        validation = guardrails_with_capacity.validate_capacity_for_tier(
            tier=SLATier.ENTERPRISE,
            validated_by="engineer",
        )
        assert validation.status == CapacityStatus.VALIDATION_FAILED
        assert len(validation.issues_found) > 0

    def test_validation_stored(self, guardrails_with_capacity):
        """Test that validations are stored."""
        validation = guardrails_with_capacity.validate_capacity_for_tier(
            tier=SLATier.STANDARD,
            validated_by="engineer",
        )
        assert validation.validation_id in guardrails_with_capacity.capacity_validations

    def test_validation_callback_called(self, config):
        """Test validation callback is called."""
        callback = MagicMock()
        config.validation_callback = callback
        guardrails = SLAGuardrails(config=config)

        guardrails.validate_capacity_for_tier(
            tier=SLATier.STANDARD,
            validated_by="engineer",
        )

        callback.assert_called_once()
        assert callback.call_args[0][0] == "validation_completed"


# =============================================================================
# SLA Commitment Request Tests
# =============================================================================


class TestSLACommitmentRequests:
    """Tests for SLA commitment request workflow."""

    def test_request_sla_commitment(self, guardrails_with_capacity):
        """Test creating SLA commitment request."""
        request = guardrails_with_capacity.request_sla_commitment(
            client_id="CLIENT-001",
            client_name="Test Bank",
            requested_tier=SLATier.STANDARD,
            requested_by="sales_rep",
            services_in_scope=["trading", "analytics"],
            is_critical_function=False,
        )
        assert request.request_id.startswith("SLA-")
        assert request.client_name == "Test Bank"
        assert request.requested_tier == SLATier.STANDARD
        assert request.approval_status == ApprovalStatus.PENDING

    def test_request_unavailable_tier_rejected(self, guardrails_with_capacity):
        """Test requesting unavailable tier is rejected."""
        request = guardrails_with_capacity.request_sla_commitment(
            client_id="CLIENT-001",
            client_name="Test Bank",
            requested_tier=SLATier.ENTERPRISE,  # Not available
            requested_by="sales_rep",
            services_in_scope=["trading"],
        )
        assert request.approval_status == ApprovalStatus.REJECTED
        assert "not currently available" in request.approval_notes

    def test_approve_commitment(self, guardrails_with_capacity):
        """Test approving SLA commitment."""
        request = guardrails_with_capacity.request_sla_commitment(
            client_id="CLIENT-001",
            client_name="Test Bank",
            requested_tier=SLATier.STANDARD,
            requested_by="sales_rep",
            services_in_scope=["trading"],
        )

        approved = guardrails_with_capacity.approve_commitment(
            request_id=request.request_id,
            approved_by="engineer",
            notes="Capacity verified",
        )

        assert approved.approval_status == ApprovalStatus.APPROVED
        assert approved.approved_by == "engineer"
        assert approved.commitment_expiry_date

    def test_approve_commitment_with_conditions(self, guardrails_with_capacity):
        """Test conditional approval."""
        request = guardrails_with_capacity.request_sla_commitment(
            client_id="CLIENT-001",
            client_name="Test Bank",
            requested_tier=SLATier.PROFESSIONAL,
            requested_by="sales_rep",
            services_in_scope=["trading"],
        )

        approved = guardrails_with_capacity.approve_commitment(
            request_id=request.request_id,
            approved_by="engineer",
            conditions=["Maintain on-call rotation", "Monitor SLA weekly"],
        )

        assert approved.approval_status == ApprovalStatus.CONDITIONAL
        assert len(approved.conditions) == 2

    def test_reject_commitment(self, guardrails_with_capacity):
        """Test rejecting SLA commitment."""
        request = guardrails_with_capacity.request_sla_commitment(
            client_id="CLIENT-001",
            client_name="Test Bank",
            requested_tier=SLATier.STANDARD,
            requested_by="sales_rep",
            services_in_scope=["trading"],
        )

        rejected = guardrails_with_capacity.reject_commitment(
            request_id=request.request_id,
            rejected_by="engineer",
            reason="Insufficient capacity for new clients",
        )

        assert rejected.approval_status == ApprovalStatus.REJECTED
        assert "Insufficient capacity" in rejected.approval_notes

    def test_get_pending_approvals(self, guardrails_with_capacity):
        """Test getting pending approvals."""
        # Create some requests
        guardrails_with_capacity.request_sla_commitment(
            client_id="CLIENT-001",
            client_name="Bank A",
            requested_tier=SLATier.STANDARD,
            requested_by="sales",
            services_in_scope=["trading"],
        )
        guardrails_with_capacity.request_sla_commitment(
            client_id="CLIENT-002",
            client_name="Bank B",
            requested_tier=SLATier.STANDARD,
            requested_by="sales",
            services_in_scope=["trading"],
        )

        pending = guardrails_with_capacity.get_pending_approvals()
        assert len(pending) == 2


# =============================================================================
# Reporting Tests
# =============================================================================


class TestReporting:
    """Tests for reporting functionality."""

    def test_generate_capacity_report(self, guardrails_with_capacity):
        """Test generating capacity report."""
        report = guardrails_with_capacity.generate_capacity_report()

        assert "report_date" in report
        assert "capacity_state" in report
        assert "available_tiers" in report
        assert "unavailable_tiers" in report
        assert "tier_definitions" in report

        # Check available tiers
        assert "standard" in report["available_tiers"]
        assert "professional" in report["available_tiers"]
        assert "enterprise" in report["unavailable_tiers"]

    def test_get_expiring_validations(self, guardrails_with_capacity):
        """Test getting expiring validations."""
        # Create validation that expires soon
        validation = guardrails_with_capacity.validate_capacity_for_tier(
            tier=SLATier.STANDARD,
            validated_by="engineer",
        )

        # Manually set expiry to soon
        validation.expiry_date = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()

        expiring = guardrails_with_capacity.get_expiring_validations(days=14)
        assert len(expiring) >= 1


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_approve_nonexistent_request(self, guardrails):
        """Test approving non-existent request raises error."""
        with pytest.raises(ValueError) as exc_info:
            guardrails.approve_commitment(
                request_id="NONEXISTENT",
                approved_by="engineer",
            )
        assert "not found" in str(exc_info.value)

    def test_reject_nonexistent_request(self, guardrails):
        """Test rejecting non-existent request raises error."""
        with pytest.raises(ValueError) as exc_info:
            guardrails.reject_commitment(
                request_id="NONEXISTENT",
                rejected_by="engineer",
                reason="Test",
            )
        assert "not found" in str(exc_info.value)
