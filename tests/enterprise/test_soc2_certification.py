# -*- coding: utf-8 -*-
"""
Comprehensive tests for SOC2 Type II Certification Framework.

Tests SOC2 certification management per AICPA standards.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from services.enterprise.soc2_certification import (
    # Enums
    SOC2TrustPrinciple,
    ControlStatus,
    EvidenceType,
    AuditStatus,
    # Data structures
    SOC2Control,
    ControlEvidence,
    AuditFinding,
    SOC2Report,
    SOC2Config,
    # Service
    SOC2CertificationService,
    # Factory
    create_soc2_certification,
    get_soc2_control_library,
)


# =============================================================================
# SOC2Control Tests
# =============================================================================


class TestSOC2Control:
    """Tests for SOC2Control dataclass."""

    def test_create_control(self) -> None:
        """Test creating a SOC2 control."""
        control = SOC2Control(
            control_id="ctrl-1",
            trust_principle=SOC2TrustPrinciple.SECURITY,
            criteria_id="CC6.1",
            name="Logical Access Controls",
            description="Access control for systems",
        )
        assert control.criteria_id == "CC6.1"
        assert control.trust_principle == SOC2TrustPrinciple.SECURITY

    def test_is_effective_true(self) -> None:
        """Test is_effective for effective control."""
        control = SOC2Control(
            control_id="ctrl-1",
            trust_principle=SOC2TrustPrinciple.SECURITY,
            criteria_id="CC6.1",
            name="Test",
            description="Test",
            status=ControlStatus.EFFECTIVE,
        )
        assert control.is_effective is True

    def test_is_effective_tested(self) -> None:
        """Test is_effective for tested control."""
        control = SOC2Control(
            control_id="ctrl-1",
            trust_principle=SOC2TrustPrinciple.SECURITY,
            criteria_id="CC6.1",
            name="Test",
            description="Test",
            status=ControlStatus.TESTED,
        )
        assert control.is_effective is True

    def test_is_effective_false(self) -> None:
        """Test is_effective for non-effective control."""
        control = SOC2Control(
            control_id="ctrl-1",
            trust_principle=SOC2TrustPrinciple.SECURITY,
            criteria_id="CC6.1",
            name="Test",
            description="Test",
            status=ControlStatus.IN_PROGRESS,
        )
        assert control.is_effective is False


# =============================================================================
# ControlEvidence Tests
# =============================================================================


class TestControlEvidence:
    """Tests for ControlEvidence dataclass."""

    def test_create_evidence(self) -> None:
        """Test creating control evidence."""
        evidence = ControlEvidence(
            evidence_id="ev-1",
            control_id="ctrl-1",
            evidence_type=EvidenceType.POLICY,
            title="Access Control Policy",
            description="Documentation of access control policy",
        )
        assert evidence.evidence_type == EvidenceType.POLICY
        assert evidence.is_current is True

    def test_expire_evidence(self) -> None:
        """Test expiring evidence."""
        evidence = ControlEvidence(
            evidence_id="ev-1",
            control_id="ctrl-1",
            evidence_type=EvidenceType.SCREENSHOT,
            title="Config Screenshot",
            description="Screenshot of configuration",
        )
        evidence.expire()
        assert evidence.is_current is False


# =============================================================================
# AuditFinding Tests
# =============================================================================


class TestAuditFinding:
    """Tests for AuditFinding dataclass."""

    def test_create_finding(self) -> None:
        """Test creating an audit finding."""
        finding = AuditFinding(
            finding_id="find-1",
            audit_id="audit-1",
            control_id="ctrl-1",
            title="Missing Access Review",
            description="Access reviews not performed quarterly",
            severity="high",
            identified_at=datetime.utcnow(),
            identified_by="auditor@firm.com",
        )
        assert finding.severity == "high"
        assert finding.status == "open"

    def test_start_remediation(self) -> None:
        """Test starting remediation."""
        finding = AuditFinding(
            finding_id="find-1",
            audit_id="audit-1",
            control_id="ctrl-1",
            title="Test Finding",
            description="Test",
            severity="medium",
            identified_at=datetime.utcnow(),
            identified_by="auditor",
        )
        due_date = datetime.utcnow() + timedelta(days=30)
        finding.start_remediation("Implement quarterly access reviews", due_date)

        assert finding.status == "in_remediation"
        assert finding.remediation_plan is not None
        assert finding.remediation_due == due_date

    def test_complete_remediation(self) -> None:
        """Test completing remediation."""
        finding = AuditFinding(
            finding_id="find-1",
            audit_id="audit-1",
            control_id="ctrl-1",
            title="Test Finding",
            description="Test",
            severity="low",
            identified_at=datetime.utcnow(),
            identified_by="auditor",
        )
        finding.complete_remediation()

        assert finding.status == "remediated"
        assert finding.remediated_at is not None


# =============================================================================
# SOC2Config Tests
# =============================================================================


class TestSOC2Config:
    """Tests for SOC2Config dataclass."""

    def test_create_config(self) -> None:
        """Test creating SOC2 config."""
        config = SOC2Config(
            organization_name="Test Corp",
            system_description="SaaS Platform",
            audit_firm="Big4 LLP",
        )
        assert config.organization_name == "Test Corp"
        assert config.evidence_retention_years == 7

    def test_config_defaults(self) -> None:
        """Test SOC2 config default values."""
        config = SOC2Config(
            organization_name="Test",
            system_description="Test",
        )
        assert config.evidence_retention_years == 7
        assert config.control_test_frequency_days == 90
        assert config.auto_remind_before_days == 14


# =============================================================================
# SOC2CertificationService Tests
# =============================================================================


class TestSOC2CertificationService:
    """Tests for SOC2CertificationService."""

    def test_create_service(self) -> None:
        """Test creating SOC2 certification service."""
        config = SOC2Config(
            organization_name="Test Corp",
            system_description="Platform",
        )
        service = SOC2CertificationService(config)
        assert service.config.organization_name == "Test Corp"

    def test_controls_initialized(self) -> None:
        """Test that controls are initialized from library."""
        config = SOC2Config(
            organization_name="Test",
            system_description="Test",
        )
        service = SOC2CertificationService(config)

        controls = service.list_controls()
        assert len(controls) > 0

    def test_get_control_by_criteria(self) -> None:
        """Test getting control by criteria ID."""
        config = SOC2Config(
            organization_name="Test",
            system_description="Test",
        )
        service = SOC2CertificationService(config)

        control = service.get_control_by_criteria("CC6.1")
        assert control is not None
        assert control.criteria_id == "CC6.1"

    def test_list_controls(self) -> None:
        """Test listing all controls."""
        config = SOC2Config(
            organization_name="Test",
            system_description="Test",
        )
        service = SOC2CertificationService(config)

        controls = service.list_controls()
        assert len(controls) >= 15  # Default library size

    def test_list_controls_by_principle(self) -> None:
        """Test listing controls by trust principle."""
        config = SOC2Config(
            organization_name="Test",
            system_description="Test",
        )
        service = SOC2CertificationService(config)

        security_controls = service.list_controls(
            trust_principle=SOC2TrustPrinciple.SECURITY
        )
        assert all(c.trust_principle == SOC2TrustPrinciple.SECURITY for c in security_controls)

    def test_list_controls_by_status(self) -> None:
        """Test listing controls by status."""
        config = SOC2Config(
            organization_name="Test",
            system_description="Test",
        )
        service = SOC2CertificationService(config)

        not_started = service.list_controls(status=ControlStatus.NOT_STARTED)
        assert all(c.status == ControlStatus.NOT_STARTED for c in not_started)

    def test_update_control_status(self) -> None:
        """Test updating control status."""
        config = SOC2Config(
            organization_name="Test",
            system_description="Test",
        )
        service = SOC2CertificationService(config)

        controls = service.list_controls()
        control_id = controls[0].control_id

        updated = service.update_control_status(
            control_id,
            ControlStatus.IMPLEMENTED,
            notes="Implemented per policy",
        )
        assert updated is not None
        assert updated.status == ControlStatus.IMPLEMENTED
        assert updated.implementation_notes == "Implemented per policy"

    def test_update_control_status_effective(self) -> None:
        """Test updating control to effective sets test date."""
        config = SOC2Config(
            organization_name="Test",
            system_description="Test",
        )
        service = SOC2CertificationService(config)

        controls = service.list_controls()
        control_id = controls[0].control_id

        updated = service.update_control_status(control_id, ControlStatus.EFFECTIVE)
        assert updated is not None
        assert updated.last_tested is not None
        assert updated.next_test_due is not None

    def test_update_control_status_not_found(self) -> None:
        """Test updating non-existent control."""
        config = SOC2Config(
            organization_name="Test",
            system_description="Test",
        )
        service = SOC2CertificationService(config)

        result = service.update_control_status("nonexistent", ControlStatus.IMPLEMENTED)
        assert result is None

    def test_assign_control_owner(self) -> None:
        """Test assigning owner to control."""
        config = SOC2Config(
            organization_name="Test",
            system_description="Test",
        )
        service = SOC2CertificationService(config)

        controls = service.list_controls()
        control_id = controls[0].control_id

        result = service.assign_control_owner(control_id, "security@example.com")
        assert result is True

        control = service.get_control(control_id)
        assert control is not None
        assert control.owner == "security@example.com"

    def test_add_evidence(self) -> None:
        """Test adding evidence to control."""
        config = SOC2Config(
            organization_name="Test",
            system_description="Test",
        )
        service = SOC2CertificationService(config)

        controls = service.list_controls()
        control_id = controls[0].control_id

        evidence = service.add_evidence(
            control_id=control_id,
            evidence_type=EvidenceType.POLICY,
            title="Security Policy",
            description="Information security policy document",
            collected_by="compliance@example.com",
        )
        assert evidence.control_id == control_id
        assert evidence.evidence_type == EvidenceType.POLICY

    def test_add_evidence_control_not_found(self) -> None:
        """Test adding evidence to non-existent control."""
        config = SOC2Config(
            organization_name="Test",
            system_description="Test",
        )
        service = SOC2CertificationService(config)

        with pytest.raises(ValueError, match="Control not found"):
            service.add_evidence(
                "nonexistent",
                EvidenceType.POLICY,
                "Test",
                "Test",
                "user",
            )

    def test_list_evidence(self) -> None:
        """Test listing evidence."""
        config = SOC2Config(
            organization_name="Test",
            system_description="Test",
        )
        service = SOC2CertificationService(config)

        controls = service.list_controls()
        control_id = controls[0].control_id

        service.add_evidence(control_id, EvidenceType.POLICY, "Policy", "Desc", "user")
        service.add_evidence(control_id, EvidenceType.SCREENSHOT, "Screenshot", "Desc", "user")

        evidence = service.list_evidence(control_id=control_id)
        assert len(evidence) == 2

    def test_get_evidence_coverage(self) -> None:
        """Test getting evidence coverage."""
        config = SOC2Config(
            organization_name="Test",
            system_description="Test",
        )
        service = SOC2CertificationService(config)

        controls = service.list_controls()

        # Add evidence to some controls
        for i in range(3):
            service.add_evidence(
                controls[i].control_id,
                EvidenceType.POLICY,
                f"Evidence {i}",
                "Desc",
                "user",
            )

        coverage = service.get_evidence_coverage()
        assert coverage["controls_with_evidence"] == 3
        assert coverage["coverage_percent"] > 0

    def test_create_audit_report(self) -> None:
        """Test creating audit report."""
        config = SOC2Config(
            organization_name="Test",
            system_description="Test",
        )
        service = SOC2CertificationService(config)

        report = service.create_audit_report(
            report_type="Type II",
            audit_period_start=datetime.utcnow() - timedelta(days=365),
            audit_period_end=datetime.utcnow(),
            auditor_firm="Big4 LLP",
            lead_auditor="john.auditor@big4.com",
        )
        assert report.report_type == "Type II"
        assert report.auditor_firm == "Big4 LLP"

    def test_get_report(self) -> None:
        """Test getting report by ID."""
        config = SOC2Config(
            organization_name="Test",
            system_description="Test",
        )
        service = SOC2CertificationService(config)

        report = service.create_audit_report(
            "Type II",
            datetime.utcnow() - timedelta(days=365),
            datetime.utcnow(),
            "Big4",
            "auditor",
        )

        retrieved = service.get_report(report.report_id)
        assert retrieved is not None
        assert retrieved.report_id == report.report_id

    def test_add_finding(self) -> None:
        """Test adding audit finding."""
        config = SOC2Config(
            organization_name="Test",
            system_description="Test",
        )
        service = SOC2CertificationService(config)

        report = service.create_audit_report(
            "Type II",
            datetime.utcnow() - timedelta(days=365),
            datetime.utcnow(),
            "Big4",
            "auditor",
        )
        controls = service.list_controls()

        finding = service.add_finding(
            audit_id=report.report_id,
            control_id=controls[0].control_id,
            title="Access Review Gap",
            description="Quarterly access reviews not documented",
            severity="high",
            identified_by="auditor@firm.com",
        )
        assert finding.severity == "high"

        # Check report findings count updated
        updated_report = service.get_report(report.report_id)
        assert updated_report is not None
        assert updated_report.findings_count == 1

    def test_list_findings(self) -> None:
        """Test listing findings."""
        config = SOC2Config(
            organization_name="Test",
            system_description="Test",
        )
        service = SOC2CertificationService(config)

        report = service.create_audit_report(
            "Type II",
            datetime.utcnow() - timedelta(days=365),
            datetime.utcnow(),
            "Big4",
            "auditor",
        )
        controls = service.list_controls()

        service.add_finding(report.report_id, controls[0].control_id, "Find 1", "Desc", "high", "auditor")
        service.add_finding(report.report_id, controls[1].control_id, "Find 2", "Desc", "medium", "auditor")

        findings = service.list_findings(audit_id=report.report_id)
        assert len(findings) == 2

    def test_get_compliance_status(self) -> None:
        """Test getting compliance status."""
        config = SOC2Config(
            organization_name="Test",
            system_description="Test",
        )
        service = SOC2CertificationService(config)

        # Update some controls to effective
        controls = service.list_controls()
        for i in range(5):
            service.update_control_status(controls[i].control_id, ControlStatus.EFFECTIVE)

        status = service.get_compliance_status()
        assert status["total_controls"] > 0
        assert status["effective_controls"] == 5
        assert "overall_compliance_percent" in status
        assert "by_principle" in status

    def test_get_dora_mapping_report(self) -> None:
        """Test getting DORA mapping report."""
        config = SOC2Config(
            organization_name="Test",
            system_description="Test",
        )
        service = SOC2CertificationService(config)

        mapping = service.get_dora_mapping_report()
        assert isinstance(mapping, dict)
        # Should have DORA article references
        assert any("Art." in key for key in mapping.keys())


# =============================================================================
# Control Library Tests
# =============================================================================


class TestControlLibrary:
    """Tests for SOC2 control library."""

    def test_get_control_library(self) -> None:
        """Test getting control library."""
        library = get_soc2_control_library()
        assert len(library) > 0

    def test_library_has_all_principles(self) -> None:
        """Test library covers all trust principles."""
        library = get_soc2_control_library()
        principles = {ctrl["trust_principle"] for ctrl in library}

        assert SOC2TrustPrinciple.SECURITY in principles
        assert SOC2TrustPrinciple.AVAILABILITY in principles
        assert SOC2TrustPrinciple.CONFIDENTIALITY in principles

    def test_library_has_dora_mappings(self) -> None:
        """Test library has DORA mappings."""
        library = get_soc2_control_library()

        controls_with_dora = [
            ctrl for ctrl in library
            if ctrl.get("dora_mapping")
        ]
        assert len(controls_with_dora) > 0


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_soc2_certification_basic(self) -> None:
        """Test creating service with factory function."""
        service = create_soc2_certification(
            organization_name="Test Corp",
            system_description="SaaS Platform",
        )
        assert isinstance(service, SOC2CertificationService)
        assert service.config.organization_name == "Test Corp"

    def test_create_soc2_certification_with_audit_firm(self) -> None:
        """Test creating service with audit firm."""
        service = create_soc2_certification(
            organization_name="Test Corp",
            system_description="SaaS Platform",
            audit_firm="Deloitte",
        )
        assert service.config.audit_firm == "Deloitte"


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_trust_principle_values(self) -> None:
        """Test all trust principle values."""
        assert SOC2TrustPrinciple.SECURITY.value == "security"
        assert SOC2TrustPrinciple.AVAILABILITY.value == "availability"
        assert SOC2TrustPrinciple.PROCESSING_INTEGRITY.value == "processing_integrity"
        assert SOC2TrustPrinciple.CONFIDENTIALITY.value == "confidentiality"
        assert SOC2TrustPrinciple.PRIVACY.value == "privacy"

    def test_control_status_values(self) -> None:
        """Test all control status values."""
        assert ControlStatus.NOT_STARTED.value == "not_started"
        assert ControlStatus.IN_PROGRESS.value == "in_progress"
        assert ControlStatus.IMPLEMENTED.value == "implemented"
        assert ControlStatus.TESTED.value == "tested"
        assert ControlStatus.EFFECTIVE.value == "effective"
        assert ControlStatus.NOT_APPLICABLE.value == "not_applicable"

    def test_evidence_type_values(self) -> None:
        """Test all evidence type values."""
        assert EvidenceType.POLICY.value == "policy"
        assert EvidenceType.PROCEDURE.value == "procedure"
        assert EvidenceType.SCREENSHOT.value == "screenshot"
        assert EvidenceType.LOG.value == "log"
        assert EvidenceType.CONFIGURATION.value == "configuration"

    def test_audit_status_values(self) -> None:
        """Test all audit status values."""
        assert AuditStatus.PLANNING.value == "planning"
        assert AuditStatus.FIELDWORK.value == "fieldwork"
        assert AuditStatus.REVIEW.value == "review"
        assert AuditStatus.COMPLETED.value == "completed"
        assert AuditStatus.REPORT_ISSUED.value == "report_issued"
