# -*- coding: utf-8 -*-
"""
Comprehensive tests for ISO 27001 Framework Service.

Tests ISO 27001:2022 certification framework per ISMS requirements.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from services.enterprise.iso27001_framework import (
    # Enums
    ISO27001Domain,
    ControlObjective,
    ImplementationStatus,
    # Data structures
    ISO27001Control,
    ControlImplementation,
    RiskAssessment,
    ISO27001Audit,
    CertificationStatus,
    ISO27001Config,
    # Service
    ISO27001FrameworkService,
    # Factory
    create_iso27001_framework,
    get_iso27001_control_library,
)


# =============================================================================
# ISO27001Control Tests
# =============================================================================


class TestISO27001Control:
    """Tests for ISO27001Control dataclass."""

    def test_create_control(self) -> None:
        """Test creating an ISO 27001 control."""
        control = ISO27001Control(
            control_id="ctrl-1",
            reference="A.5.1",
            domain=ISO27001Domain.A5_ORGANIZATIONAL,
            name="Information Security Policy",
            description="Policies for information security",
            objective="Management direction",
        )
        assert control.reference == "A.5.1"
        assert control.domain == ISO27001Domain.A5_ORGANIZATIONAL

    def test_is_implemented_fully(self) -> None:
        """Test is_implemented for fully implemented control."""
        control = ISO27001Control(
            control_id="ctrl-1",
            reference="A.5.1",
            domain=ISO27001Domain.A5_ORGANIZATIONAL,
            name="Test",
            description="Test",
            objective="Test",
            implementation_status=ImplementationStatus.FULLY_IMPLEMENTED,
        )
        assert control.is_implemented is True

    def test_is_implemented_partially(self) -> None:
        """Test is_implemented for partially implemented control."""
        control = ISO27001Control(
            control_id="ctrl-1",
            reference="A.5.1",
            domain=ISO27001Domain.A5_ORGANIZATIONAL,
            name="Test",
            description="Test",
            objective="Test",
            implementation_status=ImplementationStatus.PARTIALLY_IMPLEMENTED,
        )
        assert control.is_implemented is True

    def test_is_implemented_false(self) -> None:
        """Test is_implemented for not implemented control."""
        control = ISO27001Control(
            control_id="ctrl-1",
            reference="A.5.1",
            domain=ISO27001Domain.A5_ORGANIZATIONAL,
            name="Test",
            description="Test",
            objective="Test",
            implementation_status=ImplementationStatus.NOT_IMPLEMENTED,
        )
        assert control.is_implemented is False


# =============================================================================
# RiskAssessment Tests
# =============================================================================


class TestRiskAssessment:
    """Tests for RiskAssessment dataclass."""

    def test_create_assessment(self) -> None:
        """Test creating a risk assessment."""
        assessment = RiskAssessment(
            assessment_id="risk-1",
            title="Data Breach Risk",
            description="Risk of data breach",
            asset="Customer Database",
            threat="External Attack",
            vulnerability="Weak Authentication",
            likelihood=4,
            impact=5,
            assessed_by="security@example.com",
        )
        assert assessment.likelihood == 4
        assert assessment.impact == 5

    def test_calculate_risks(self) -> None:
        """Test risk calculation."""
        assessment = RiskAssessment(
            assessment_id="risk-1",
            title="Test Risk",
            description="Test",
            asset="Asset",
            threat="Threat",
            vulnerability="Vulnerability",
            likelihood=4,
            impact=5,
            assessed_by="assessor",
        )
        assessment.residual_likelihood = 2
        assessment.residual_impact = 3
        assessment.calculate_risks()

        assert assessment.inherent_risk == 20  # 4 * 5
        assert assessment.residual_risk == 6  # 2 * 3


# =============================================================================
# ISO27001FrameworkService Tests
# =============================================================================


class TestISO27001FrameworkService:
    """Tests for ISO27001FrameworkService."""

    def test_create_service(self) -> None:
        """Test creating ISO 27001 framework service."""
        config = ISO27001Config(
            organization_name="Test Corp",
            isms_scope="IT Services",
        )
        service = ISO27001FrameworkService(config)
        assert service.config.organization_name == "Test Corp"

    def test_controls_initialized(self) -> None:
        """Test that controls are initialized from library."""
        config = ISO27001Config(
            organization_name="Test",
            isms_scope="Test",
        )
        service = ISO27001FrameworkService(config)

        controls = service.list_controls()
        assert len(controls) > 0

    def test_get_control_by_reference(self) -> None:
        """Test getting control by reference."""
        config = ISO27001Config(
            organization_name="Test",
            isms_scope="Test",
        )
        service = ISO27001FrameworkService(config)

        control = service.get_control_by_reference("A.5.1")
        assert control is not None
        assert control.reference == "A.5.1"

    def test_list_controls(self) -> None:
        """Test listing all controls."""
        config = ISO27001Config(
            organization_name="Test",
            isms_scope="Test",
        )
        service = ISO27001FrameworkService(config)

        controls = service.list_controls()
        assert len(controls) >= 20  # Default library size

    def test_list_controls_by_domain(self) -> None:
        """Test listing controls by domain."""
        config = ISO27001Config(
            organization_name="Test",
            isms_scope="Test",
        )
        service = ISO27001FrameworkService(config)

        org_controls = service.list_controls(domain=ISO27001Domain.A5_ORGANIZATIONAL)
        assert all(c.domain == ISO27001Domain.A5_ORGANIZATIONAL for c in org_controls)

    def test_list_controls_by_status(self) -> None:
        """Test listing controls by status."""
        config = ISO27001Config(
            organization_name="Test",
            isms_scope="Test",
        )
        service = ISO27001FrameworkService(config)

        not_implemented = service.list_controls(status=ImplementationStatus.NOT_IMPLEMENTED)
        assert all(
            c.implementation_status == ImplementationStatus.NOT_IMPLEMENTED for c in not_implemented
        )

    def test_update_control_status(self) -> None:
        """Test updating control status."""
        config = ISO27001Config(
            organization_name="Test",
            isms_scope="Test",
        )
        service = ISO27001FrameworkService(config)

        controls = service.list_controls()
        control_id = controls[0].control_id

        updated = service.update_control_status(
            control_id,
            ImplementationStatus.FULLY_IMPLEMENTED,
            notes="Implemented per policy",
        )
        assert updated is not None
        assert updated.implementation_status == ImplementationStatus.FULLY_IMPLEMENTED
        assert updated.last_review is not None

    def test_update_control_status_not_found(self) -> None:
        """Test updating non-existent control."""
        config = ISO27001Config(
            organization_name="Test",
            isms_scope="Test",
        )
        service = ISO27001FrameworkService(config)

        result = service.update_control_status(
            "nonexistent", ImplementationStatus.FULLY_IMPLEMENTED
        )
        assert result is None

    def test_assign_control_owner(self) -> None:
        """Test assigning owner to control."""
        config = ISO27001Config(
            organization_name="Test",
            isms_scope="Test",
        )
        service = ISO27001FrameworkService(config)

        controls = service.list_controls()
        control_id = controls[0].control_id

        result = service.assign_control_owner(control_id, "security@example.com")
        assert result is True

        control = service.get_control(control_id)
        assert control is not None
        assert control.owner == "security@example.com"

    def test_add_implementation(self) -> None:
        """Test adding implementation details."""
        config = ISO27001Config(
            organization_name="Test",
            isms_scope="Test",
        )
        service = ISO27001FrameworkService(config)

        controls = service.list_controls()
        control_id = controls[0].control_id

        implementation = service.add_implementation(
            control_id=control_id,
            description="Policy documented and communicated",
            procedures=["Policy creation", "Annual review"],
            technologies=["Document management system"],
            responsible_parties=["CISO", "Security team"],
        )
        assert implementation.control_id == control_id
        assert len(implementation.procedures) == 2

    def test_add_implementation_control_not_found(self) -> None:
        """Test adding implementation to non-existent control."""
        config = ISO27001Config(
            organization_name="Test",
            isms_scope="Test",
        )
        service = ISO27001FrameworkService(config)

        with pytest.raises(ValueError, match="Control not found"):
            service.add_implementation(
                "nonexistent",
                "Description",
                [],
                [],
                [],
            )

    def test_list_implementations(self) -> None:
        """Test listing implementations."""
        config = ISO27001Config(
            organization_name="Test",
            isms_scope="Test",
        )
        service = ISO27001FrameworkService(config)

        controls = service.list_controls()
        control_id = controls[0].control_id

        service.add_implementation(control_id, "Impl 1", [], [], [])
        service.add_implementation(control_id, "Impl 2", [], [], [])

        implementations = service.list_implementations(control_id=control_id)
        assert len(implementations) == 2

    def test_create_risk_assessment(self) -> None:
        """Test creating risk assessment."""
        config = ISO27001Config(
            organization_name="Test",
            isms_scope="Test",
        )
        service = ISO27001FrameworkService(config)

        assessment = service.create_risk_assessment(
            title="Data Breach Risk",
            description="Risk assessment for data breach",
            asset="Customer Database",
            threat="External Attack",
            vulnerability="Weak Authentication",
            likelihood=4,
            impact=5,
            assessed_by="risk@example.com",
        )
        assert assessment.inherent_risk == 20  # Auto-calculated

    def test_list_risk_assessments(self) -> None:
        """Test listing risk assessments."""
        config = ISO27001Config(
            organization_name="Test",
            isms_scope="Test",
        )
        service = ISO27001FrameworkService(config)

        service.create_risk_assessment("Risk 1", "Desc", "Asset", "Threat", "Vuln", 4, 5, "user")
        service.create_risk_assessment("Risk 2", "Desc", "Asset", "Threat", "Vuln", 2, 3, "user")

        assessments = service.list_risk_assessments()
        assert len(assessments) == 2
        # Should be sorted by risk (highest first)
        assert assessments[0].inherent_risk >= assessments[1].inherent_risk

    def test_list_risk_assessments_min_risk(self) -> None:
        """Test listing risk assessments with minimum risk filter."""
        config = ISO27001Config(
            organization_name="Test",
            isms_scope="Test",
        )
        service = ISO27001FrameworkService(config)

        service.create_risk_assessment(
            "High Risk", "Desc", "Asset", "Threat", "Vuln", 5, 5, "user"
        )  # 25
        service.create_risk_assessment(
            "Low Risk", "Desc", "Asset", "Threat", "Vuln", 2, 2, "user"
        )  # 4

        high_risks = service.list_risk_assessments(min_risk=15)
        assert len(high_risks) == 1
        assert high_risks[0].title == "High Risk"

    def test_apply_controls_to_risk(self) -> None:
        """Test applying controls to reduce risk."""
        config = ISO27001Config(
            organization_name="Test",
            isms_scope="Test",
        )
        service = ISO27001FrameworkService(config)

        assessment = service.create_risk_assessment(
            "Risk", "Desc", "Asset", "Threat", "Vuln", 4, 5, "user"
        )
        controls = service.list_controls()

        updated = service.apply_controls_to_risk(
            assessment.assessment_id,
            control_ids=[controls[0].control_id],
            residual_likelihood=2,
            residual_impact=3,
            treatment="mitigate",
        )
        assert updated is not None
        assert updated.residual_risk == 6
        assert updated.risk_treatment == "mitigate"

    def test_create_audit(self) -> None:
        """Test creating an audit."""
        config = ISO27001Config(
            organization_name="Test",
            isms_scope="Test",
        )
        service = ISO27001FrameworkService(config)

        audit = service.create_audit(
            audit_type="internal",
            auditor="auditor@example.com",
            audit_date=datetime.utcnow(),
            scope=["A.5", "A.6"],
        )
        assert audit.audit_type == "internal"
        assert len(audit.scope) == 2

    def test_get_audit(self) -> None:
        """Test getting audit by ID."""
        config = ISO27001Config(
            organization_name="Test",
            isms_scope="Test",
        )
        service = ISO27001FrameworkService(config)

        audit = service.create_audit("internal", "auditor", datetime.utcnow(), ["A.5"])

        retrieved = service.get_audit(audit.audit_id)
        assert retrieved is not None

    def test_add_audit_finding(self) -> None:
        """Test adding audit finding."""
        config = ISO27001Config(
            organization_name="Test",
            isms_scope="Test",
        )
        service = ISO27001FrameworkService(config)

        audit = service.create_audit("internal", "auditor", datetime.utcnow(), ["A.5"])

        result = service.add_audit_finding(
            audit_id=audit.audit_id,
            finding_type="nonconformity",
            description="Missing access reviews",
            control_reference="A.5.15",
            severity="major",
        )
        assert result is True
        assert audit.nonconformities == 1
        assert len(audit.findings) == 1

    def test_add_audit_finding_observation(self) -> None:
        """Test adding observation finding."""
        config = ISO27001Config(
            organization_name="Test",
            isms_scope="Test",
        )
        service = ISO27001FrameworkService(config)

        audit = service.create_audit("internal", "auditor", datetime.utcnow(), ["A.5"])

        service.add_audit_finding(
            audit.audit_id, "observation", "Could improve documentation", "A.5.1", "minor"
        )
        assert audit.observations == 1

    def test_get_compliance_status(self) -> None:
        """Test getting compliance status."""
        config = ISO27001Config(
            organization_name="Test",
            isms_scope="Test",
        )
        service = ISO27001FrameworkService(config)

        # Update some controls to implemented
        controls = service.list_controls()
        for i in range(5):
            service.update_control_status(
                controls[i].control_id, ImplementationStatus.FULLY_IMPLEMENTED
            )

        status = service.get_compliance_status()

        assert status["total_controls"] > 0
        assert status["implemented_controls"] == 5
        assert "overall_compliance_percent" in status
        assert "by_domain" in status

    def test_get_dora_mapping_report(self) -> None:
        """Test getting DORA mapping report."""
        config = ISO27001Config(
            organization_name="Test",
            isms_scope="Test",
        )
        service = ISO27001FrameworkService(config)

        mapping = service.get_dora_mapping_report()
        assert isinstance(mapping, dict)
        # Should have DORA article references
        assert any("Art." in key for key in mapping.keys())


# =============================================================================
# Control Library Tests
# =============================================================================


class TestControlLibrary:
    """Tests for ISO 27001 control library."""

    def test_get_control_library(self) -> None:
        """Test getting control library."""
        library = get_iso27001_control_library()
        assert len(library) > 0

    def test_library_has_all_domains(self) -> None:
        """Test library covers all domains."""
        library = get_iso27001_control_library()
        domains = {ctrl["domain"] for ctrl in library}

        assert ISO27001Domain.A5_ORGANIZATIONAL in domains
        assert ISO27001Domain.A6_PEOPLE in domains
        assert ISO27001Domain.A7_PHYSICAL in domains
        assert ISO27001Domain.A8_TECHNOLOGICAL in domains

    def test_library_has_dora_mappings(self) -> None:
        """Test library has DORA mappings."""
        library = get_iso27001_control_library()

        controls_with_dora = [ctrl for ctrl in library if ctrl.get("dora_mapping")]
        assert len(controls_with_dora) > 0


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_iso27001_framework_basic(self) -> None:
        """Test creating service with factory function."""
        service = create_iso27001_framework(
            organization_name="Test Corp",
            isms_scope="IT Services",
        )
        assert isinstance(service, ISO27001FrameworkService)
        assert service.config.organization_name == "Test Corp"

    def test_create_iso27001_framework_with_cert_body(self) -> None:
        """Test creating service with certification body."""
        service = create_iso27001_framework(
            organization_name="Test Corp",
            isms_scope="IT Services",
            certification_body="BSI",
        )
        assert service.config.certification_body == "BSI"


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_domain_values(self) -> None:
        """Test all domain values."""
        assert ISO27001Domain.A5_ORGANIZATIONAL.value == "A.5"
        assert ISO27001Domain.A6_PEOPLE.value == "A.6"
        assert ISO27001Domain.A7_PHYSICAL.value == "A.7"
        assert ISO27001Domain.A8_TECHNOLOGICAL.value == "A.8"

    def test_control_objective_values(self) -> None:
        """Test all control objective values."""
        assert ControlObjective.NOT_STARTED.value == "not_started"
        assert ControlObjective.DOCUMENTED.value == "documented"
        assert ControlObjective.IMPLEMENTED.value == "implemented"
        assert ControlObjective.OPERATING.value == "operating"
        assert ControlObjective.OPTIMIZING.value == "optimizing"

    def test_implementation_status_values(self) -> None:
        """Test all implementation status values."""
        assert ImplementationStatus.NOT_APPLICABLE.value == "not_applicable"
        assert ImplementationStatus.NOT_IMPLEMENTED.value == "not_implemented"
        assert ImplementationStatus.PARTIALLY_IMPLEMENTED.value == "partially_implemented"
        assert ImplementationStatus.FULLY_IMPLEMENTED.value == "fully_implemented"
        assert ImplementationStatus.NEEDS_IMPROVEMENT.value == "needs_improvement"
