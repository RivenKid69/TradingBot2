# -*- coding: utf-8 -*-
"""
Tests for DORA Pooled Audit Support Module.

Comprehensive test coverage for:
- Certification management
- Pooled audit engagements
- Participant management
- Finding tracking
- Report access control

Reference: DORA Article 30(4)
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, MagicMock

from services.dora_integration.due_diligence import (
    # Enums
    AuditReportType,
    PooledAuditStatus,
    ParticipationStatus,
    AuditScopeArea,
    FindingSeverity,
    RemediationStatus,
    # Data structures
    CertificationRecord,
    PooledAuditParticipant,
    AuditFinding,
    PooledAuditEngagement,
    AuditReportAccess,
    PooledAuditConfig,
    # Main class
    PooledAuditSupport,
    # Factory functions
    create_pooled_audit_support,
    get_audit_scope_areas,
    get_report_types,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def config():
    """Create test configuration."""
    return PooledAuditConfig(
        min_participants_for_pooled=2,
        max_participants=10,
        nda_required_for_full_report=True,
    )


@pytest.fixture
def support(config):
    """Create Pooled Audit Support instance."""
    return PooledAuditSupport(config=config)


@pytest.fixture
def support_with_certification(support):
    """Create support with a registered certification."""
    support.register_certification(
        certification_type=AuditReportType.SOC2_TYPE_II,
        certifying_body="Big 4 Auditor",
        issue_date="2024-10-01",
        expiry_date=(datetime.now(timezone.utc) + timedelta(days=180)).isoformat(),
        scope_areas=[AuditScopeArea.ICT_SECURITY, AuditScopeArea.ICT_OPERATIONS],
        scope_description="Full platform security and operations",
    )
    return support


@pytest.fixture
def support_with_engagement(support):
    """Create support with a pooled audit engagement."""
    support.create_pooled_audit(
        name="Q1 2025 Security Assessment",
        audit_type=AuditReportType.PENETRATION_TEST,
        scope_areas=[AuditScopeArea.ICT_SECURITY],
        auditor_name="John Smith",
        auditor_firm="Security Auditors Ltd",
        planned_start_date="2025-02-01",
        planned_end_date="2025-02-15",
        created_by="compliance_manager",
    )
    return support


# =============================================================================
# Enumeration Tests
# =============================================================================

class TestEnumerations:
    """Test all enumeration classes."""

    def test_audit_report_type_values(self):
        """Test AuditReportType enum values."""
        assert AuditReportType.SOC2_TYPE_I.value == "soc2_type_i"
        assert AuditReportType.SOC2_TYPE_II.value == "soc2_type_ii"
        assert AuditReportType.ISO27001.value == "iso27001"
        assert AuditReportType.PENETRATION_TEST.value == "penetration_test"
        assert len(AuditReportType) == 9

    def test_pooled_audit_status_values(self):
        """Test PooledAuditStatus enum values."""
        assert PooledAuditStatus.PLANNING.value == "planning"
        assert PooledAuditStatus.RECRUITING.value == "recruiting"
        assert PooledAuditStatus.SCHEDULED.value == "scheduled"
        assert PooledAuditStatus.IN_PROGRESS.value == "in_progress"
        assert PooledAuditStatus.COMPLETED.value == "completed"
        assert len(PooledAuditStatus) == 7

    def test_participation_status_values(self):
        """Test ParticipationStatus enum values."""
        assert ParticipationStatus.INVITED.value == "invited"
        assert ParticipationStatus.INTERESTED.value == "interested"
        assert ParticipationStatus.CONFIRMED.value == "confirmed"
        assert ParticipationStatus.DECLINED.value == "declined"
        assert len(ParticipationStatus) == 6

    def test_audit_scope_area_values(self):
        """Test AuditScopeArea enum values."""
        assert AuditScopeArea.ICT_GOVERNANCE.value == "ict_governance"
        assert AuditScopeArea.ICT_SECURITY.value == "ict_security"
        assert AuditScopeArea.INCIDENT_MANAGEMENT.value == "incident_management"
        assert len(AuditScopeArea) == 10

    def test_finding_severity_values(self):
        """Test FindingSeverity enum values."""
        assert FindingSeverity.CRITICAL.value == "critical"
        assert FindingSeverity.HIGH.value == "high"
        assert FindingSeverity.MEDIUM.value == "medium"
        assert FindingSeverity.LOW.value == "low"
        assert FindingSeverity.INFORMATIONAL.value == "informational"
        assert len(FindingSeverity) == 5

    def test_remediation_status_values(self):
        """Test RemediationStatus enum values."""
        assert RemediationStatus.OPEN.value == "open"
        assert RemediationStatus.IN_PROGRESS.value == "in_progress"
        assert RemediationStatus.REMEDIATED.value == "remediated"
        assert RemediationStatus.CLOSED.value == "closed"
        assert len(RemediationStatus) == 5


# =============================================================================
# Data Structure Tests
# =============================================================================

class TestDataStructures:
    """Tests for data structures."""

    def test_certification_record_creation(self):
        """Test CertificationRecord creation."""
        cert = CertificationRecord(
            certification_type=AuditReportType.SOC2_TYPE_II,
            certifying_body="Auditor Inc",
            issue_date="2024-10-01",
            expiry_date="2025-10-01",
        )
        assert cert.certification_id.startswith("CERT-")
        assert cert.certification_type == AuditReportType.SOC2_TYPE_II
        assert cert.nda_required is True

    def test_certification_record_is_valid(self):
        """Test CertificationRecord is_valid property."""
        # Valid certification
        valid_cert = CertificationRecord(
            certification_type=AuditReportType.SOC2_TYPE_II,
            expiry_date=(datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
        )
        assert valid_cert.is_valid is True

        # Expired certification
        expired_cert = CertificationRecord(
            certification_type=AuditReportType.SOC2_TYPE_II,
            expiry_date=(datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
        )
        assert expired_cert.is_valid is False

    def test_pooled_audit_participant_creation(self):
        """Test PooledAuditParticipant creation."""
        participant = PooledAuditParticipant(
            client_id="CLIENT-001",
            client_name="Test Bank",
            contact_name="John Doe",
            contact_email="john@testbank.com",
        )
        assert participant.participant_id.startswith("PART-")
        assert participant.status == ParticipationStatus.INVITED
        assert participant.invited_date

    def test_audit_finding_creation(self):
        """Test AuditFinding creation."""
        finding = AuditFinding(
            audit_id="POOL-001",
            title="Weak Password Policy",
            description="Password complexity requirements are insufficient",
            severity=FindingSeverity.HIGH,
            scope_area=AuditScopeArea.ACCESS_CONTROL,
            recommendation="Implement stronger password requirements",
        )
        assert finding.finding_id.startswith("FND-")
        assert finding.remediation_status == RemediationStatus.OPEN
        assert finding.identified_date

    def test_pooled_audit_engagement_creation(self):
        """Test PooledAuditEngagement creation."""
        engagement = PooledAuditEngagement(
            engagement_name="Q1 Security Audit",
            audit_type=AuditReportType.THIRD_PARTY_AUDIT,
            scope_areas=[AuditScopeArea.ICT_SECURITY],
        )
        assert engagement.engagement_id.startswith("POOL-")
        assert engagement.status == PooledAuditStatus.PLANNING
        assert engagement.created_date

    def test_audit_report_access_creation(self):
        """Test AuditReportAccess creation."""
        access = AuditReportAccess(
            client_id="CLIENT-001",
            client_name="Test Bank",
            report_type="certification",
            report_id="CERT-001",
        )
        assert access.access_id.startswith("ACC-")
        assert access.access_count == 0


# =============================================================================
# Configuration Tests
# =============================================================================

class TestConfiguration:
    """Tests for configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PooledAuditConfig()
        assert config.min_participants_for_pooled == 2
        assert config.max_participants == 20
        assert config.nda_required_for_full_report is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = PooledAuditConfig(
            min_participants_for_pooled=3,
            max_participants=15,
            annual_pentest=False,
        )
        assert config.min_participants_for_pooled == 3
        assert config.max_participants == 15
        assert config.annual_pentest is False

    def test_config_with_callback(self):
        """Test configuration with notification callback."""
        callback = MagicMock()
        config = PooledAuditConfig(notification_callback=callback)
        assert config.notification_callback == callback


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_pooled_audit_support(self):
        """Test factory function creates instance."""
        support = create_pooled_audit_support()
        assert isinstance(support, PooledAuditSupport)

    def test_get_audit_scope_areas(self):
        """Test get_audit_scope_areas returns list."""
        areas = get_audit_scope_areas()
        assert isinstance(areas, list)
        assert "ict_security" in areas
        assert "ict_governance" in areas

    def test_get_report_types(self):
        """Test get_report_types returns list."""
        types = get_report_types()
        assert isinstance(types, list)
        assert "soc2_type_ii" in types
        assert "penetration_test" in types


# =============================================================================
# Certification Management Tests
# =============================================================================

class TestCertificationManagement:
    """Tests for certification management."""

    def test_register_certification(self, support):
        """Test registering a new certification."""
        cert = support.register_certification(
            certification_type=AuditReportType.SOC2_TYPE_II,
            certifying_body="Big 4 Auditor",
            issue_date="2024-10-01",
            expiry_date=(datetime.now(timezone.utc) + timedelta(days=365)).isoformat(),
            scope_areas=[AuditScopeArea.ICT_SECURITY],
            scope_description="Security controls",
        )
        assert cert.certification_id in support.certifications
        assert cert.certifying_body == "Big 4 Auditor"

    def test_get_valid_certifications(self, support_with_certification):
        """Test getting valid certifications."""
        valid = support_with_certification.get_valid_certifications()
        assert len(valid) == 1
        assert valid[0].certification_type == AuditReportType.SOC2_TYPE_II

    def test_get_expiring_certifications(self, support):
        """Test getting expiring certifications."""
        # Register cert expiring soon
        support.register_certification(
            certification_type=AuditReportType.ISO27001,
            certifying_body="ISO Registrar",
            issue_date="2024-01-01",
            expiry_date=(datetime.now(timezone.utc) + timedelta(days=20)).isoformat(),
            scope_areas=[AuditScopeArea.ICT_GOVERNANCE],
        )

        expiring = support.get_expiring_certifications(days=30)
        assert len(expiring) == 1

    def test_certification_callback_called(self, config):
        """Test notification callback is called on registration."""
        callback = MagicMock()
        config.notification_callback = callback
        support = PooledAuditSupport(config=config)

        support.register_certification(
            certification_type=AuditReportType.SOC2_TYPE_II,
            certifying_body="Auditor",
            issue_date="2024-10-01",
            expiry_date=(datetime.now(timezone.utc) + timedelta(days=365)).isoformat(),
            scope_areas=[AuditScopeArea.ICT_SECURITY],
        )

        callback.assert_called_once()
        assert callback.call_args[0][0] == "certification_registered"


# =============================================================================
# Pooled Audit Management Tests
# =============================================================================

class TestPooledAuditManagement:
    """Tests for pooled audit management."""

    def test_create_pooled_audit(self, support):
        """Test creating a pooled audit engagement."""
        engagement = support.create_pooled_audit(
            name="Q1 2025 Pentest",
            audit_type=AuditReportType.PENETRATION_TEST,
            scope_areas=[AuditScopeArea.ICT_SECURITY],
            auditor_name="Jane Smith",
            auditor_firm="Pentest Corp",
            planned_start_date="2025-02-01",
            planned_end_date="2025-02-15",
            created_by="manager",
        )
        assert engagement.engagement_id in support.engagements
        assert engagement.engagement_name == "Q1 2025 Pentest"
        assert engagement.status == PooledAuditStatus.PLANNING

    def test_invite_participant(self, support_with_engagement):
        """Test inviting participant to pooled audit."""
        engagement_id = list(support_with_engagement.engagements.keys())[0]

        participant = support_with_engagement.invite_participant(
            engagement_id=engagement_id,
            client_id="CLIENT-001",
            client_name="Test Bank",
            contact_name="John Doe",
            contact_email="john@testbank.com",
        )

        assert participant.participant_id in support_with_engagement.participants
        assert participant.status == ParticipationStatus.INVITED
        # Check participant added to engagement
        engagement = support_with_engagement.engagements[engagement_id]
        assert participant.participant_id in engagement.participant_ids

    def test_confirm_participation(self, support_with_engagement):
        """Test confirming participation."""
        engagement_id = list(support_with_engagement.engagements.keys())[0]

        participant = support_with_engagement.invite_participant(
            engagement_id=engagement_id,
            client_id="CLIENT-001",
            client_name="Test Bank",
            contact_name="John Doe",
            contact_email="john@testbank.com",
        )

        confirmed = support_with_engagement.confirm_participation(
            participant_id=participant.participant_id,
            specific_requirements=["Need API testing", "Include mobile app"],
        )

        assert confirmed.status == ParticipationStatus.CONFIRMED
        assert len(confirmed.specific_requirements) == 2

    def test_decline_participation(self, support_with_engagement):
        """Test declining participation."""
        engagement_id = list(support_with_engagement.engagements.keys())[0]

        participant = support_with_engagement.invite_participant(
            engagement_id=engagement_id,
            client_id="CLIENT-001",
            client_name="Test Bank",
            contact_name="John Doe",
            contact_email="john@testbank.com",
        )

        declined = support_with_engagement.decline_participation(
            participant_id=participant.participant_id,
            reason="Budget constraints",
        )

        assert declined.status == ParticipationStatus.DECLINED

    def test_update_engagement_status(self, support_with_engagement):
        """Test updating engagement status."""
        engagement_id = list(support_with_engagement.engagements.keys())[0]

        updated = support_with_engagement.update_engagement_status(
            engagement_id=engagement_id,
            status=PooledAuditStatus.IN_PROGRESS,
            actual_start_date="2025-02-01",
        )

        assert updated.status == PooledAuditStatus.IN_PROGRESS
        assert updated.actual_start_date == "2025-02-01"


# =============================================================================
# Findings Management Tests
# =============================================================================

class TestFindingsManagement:
    """Tests for audit findings management."""

    def test_add_finding(self, support_with_engagement):
        """Test adding a finding."""
        engagement_id = list(support_with_engagement.engagements.keys())[0]

        finding = support_with_engagement.add_finding(
            audit_id=engagement_id,
            title="SQL Injection Vulnerability",
            description="Input validation missing on search endpoint",
            severity=FindingSeverity.CRITICAL,
            scope_area=AuditScopeArea.ICT_SECURITY,
            recommendation="Implement parameterized queries",
            dora_article_reference="Art. 9",
        )

        assert finding.finding_id in support_with_engagement.findings
        assert finding.severity == FindingSeverity.CRITICAL

        # Check finding added to engagement
        engagement = support_with_engagement.engagements[engagement_id]
        assert finding.finding_id in engagement.finding_ids
        assert engagement.critical_findings == 1

    def test_update_finding_remediation(self, support_with_engagement):
        """Test updating finding remediation status."""
        engagement_id = list(support_with_engagement.engagements.keys())[0]

        finding = support_with_engagement.add_finding(
            audit_id=engagement_id,
            title="Test Finding",
            description="Test description",
            severity=FindingSeverity.MEDIUM,
            scope_area=AuditScopeArea.ICT_SECURITY,
            recommendation="Fix it",
        )

        updated = support_with_engagement.update_finding_remediation(
            finding_id=finding.finding_id,
            status=RemediationStatus.IN_PROGRESS,
            owner="security_team",
            deadline="2025-03-01",
        )

        assert updated.remediation_status == RemediationStatus.IN_PROGRESS
        assert updated.remediation_owner == "security_team"

    def test_update_finding_to_remediated(self, support_with_engagement):
        """Test marking finding as remediated."""
        engagement_id = list(support_with_engagement.engagements.keys())[0]

        finding = support_with_engagement.add_finding(
            audit_id=engagement_id,
            title="Test Finding",
            description="Test description",
            severity=FindingSeverity.LOW,
            scope_area=AuditScopeArea.ICT_OPERATIONS,
            recommendation="Fix it",
        )

        updated = support_with_engagement.update_finding_remediation(
            finding_id=finding.finding_id,
            status=RemediationStatus.REMEDIATED,
            evidence="Fix deployed in release v1.2.3",
        )

        assert updated.remediation_status == RemediationStatus.REMEDIATED
        assert updated.remediation_completed_date

    def test_get_open_findings(self, support_with_engagement):
        """Test getting open findings."""
        engagement_id = list(support_with_engagement.engagements.keys())[0]

        # Add open finding
        support_with_engagement.add_finding(
            audit_id=engagement_id,
            title="Open Finding",
            description="Still open",
            severity=FindingSeverity.HIGH,
            scope_area=AuditScopeArea.ICT_SECURITY,
            recommendation="Fix",
        )

        # Add closed finding
        closed = support_with_engagement.add_finding(
            audit_id=engagement_id,
            title="Closed Finding",
            description="Already fixed",
            severity=FindingSeverity.LOW,
            scope_area=AuditScopeArea.ICT_OPERATIONS,
            recommendation="Fixed",
        )
        support_with_engagement.update_finding_remediation(
            finding_id=closed.finding_id,
            status=RemediationStatus.CLOSED,
        )

        open_findings = support_with_engagement.get_open_findings()
        assert len(open_findings) == 1
        assert open_findings[0].title == "Open Finding"


# =============================================================================
# Report Access Tests
# =============================================================================

class TestReportAccess:
    """Tests for report access management."""

    def test_grant_report_access(self, support_with_certification):
        """Test granting report access."""
        cert_id = list(support_with_certification.certifications.keys())[0]

        access = support_with_certification.grant_report_access(
            client_id="CLIENT-001",
            client_name="Test Bank",
            report_type="certification",
            report_id=cert_id,
            granted_by="compliance_manager",
            nda_signed=True,
            nda_signed_date="2025-01-15",
        )

        assert access.access_id in support_with_certification.report_access
        assert access.nda_signed is True
        assert access.access_expiry_date

    def test_record_report_access(self, support_with_certification):
        """Test recording report access."""
        cert_id = list(support_with_certification.certifications.keys())[0]

        access = support_with_certification.grant_report_access(
            client_id="CLIENT-001",
            client_name="Test Bank",
            report_type="certification",
            report_id=cert_id,
            granted_by="manager",
        )

        # Record access
        updated = support_with_certification.record_report_access(access.access_id)
        assert updated.access_count == 1
        assert updated.first_accessed_date

        # Record again
        updated = support_with_certification.record_report_access(access.access_id)
        assert updated.access_count == 2

    def test_get_client_report_access(self, support_with_certification):
        """Test getting client's report access records."""
        cert_id = list(support_with_certification.certifications.keys())[0]

        support_with_certification.grant_report_access(
            client_id="CLIENT-001",
            client_name="Test Bank",
            report_type="certification",
            report_id=cert_id,
            granted_by="manager",
        )

        access_records = support_with_certification.get_client_report_access("CLIENT-001")
        assert len(access_records) == 1


# =============================================================================
# Reporting Tests
# =============================================================================

class TestReporting:
    """Tests for reporting functionality."""

    def test_get_available_reports(self, support_with_certification):
        """Test getting available reports for a client."""
        cert_id = list(support_with_certification.certifications.keys())[0]

        # Grant access
        support_with_certification.grant_report_access(
            client_id="CLIENT-001",
            client_name="Test Bank",
            report_type="certification",
            report_id=cert_id,
            granted_by="manager",
        )

        reports = support_with_certification.get_available_reports("CLIENT-001")

        assert "certifications" in reports
        assert len(reports["certifications"]) == 1
        assert reports["certifications"][0]["has_access"] is True

    def test_generate_pooled_audit_summary(self, support_with_certification):
        """Test generating summary report."""
        summary = support_with_certification.generate_pooled_audit_summary()

        assert "report_date" in summary
        assert "certifications" in summary
        assert "pooled_audits" in summary
        assert "findings" in summary
        assert "report_access" in summary

        assert summary["certifications"]["valid"] == 1


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_invite_to_nonexistent_engagement(self, support):
        """Test inviting to non-existent engagement raises error."""
        with pytest.raises(ValueError) as exc_info:
            support.invite_participant(
                engagement_id="NONEXISTENT",
                client_id="CLIENT-001",
                client_name="Test",
                contact_name="John",
                contact_email="john@test.com",
            )
        assert "not found" in str(exc_info.value)

    def test_confirm_nonexistent_participant(self, support):
        """Test confirming non-existent participant raises error."""
        with pytest.raises(ValueError) as exc_info:
            support.confirm_participation(participant_id="NONEXISTENT")
        assert "not found" in str(exc_info.value)

    def test_update_nonexistent_engagement(self, support):
        """Test updating non-existent engagement raises error."""
        with pytest.raises(ValueError) as exc_info:
            support.update_engagement_status(
                engagement_id="NONEXISTENT",
                status=PooledAuditStatus.IN_PROGRESS,
            )
        assert "not found" in str(exc_info.value)

    def test_update_nonexistent_finding(self, support):
        """Test updating non-existent finding raises error."""
        with pytest.raises(ValueError) as exc_info:
            support.update_finding_remediation(
                finding_id="NONEXISTENT",
                status=RemediationStatus.REMEDIATED,
            )
        assert "not found" in str(exc_info.value)

    def test_record_access_nonexistent(self, support):
        """Test recording access for non-existent access record raises error."""
        with pytest.raises(ValueError) as exc_info:
            support.record_report_access(access_id="NONEXISTENT")
        assert "not found" in str(exc_info.value)
