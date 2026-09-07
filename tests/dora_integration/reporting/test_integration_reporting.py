# -*- coding: utf-8 -*-
"""
Comprehensive tests for DORA Integration Layer - Unified Reporting Layer (Phase 5).

Tests cover:
1. UnifiedReportingManager - Report lifecycle, validation, packaging
2. DORAReportingTemplates - ITS template generation and validation
3. DORARegisterOfInformation - ROI data generation

Target: 100% coverage of all public methods and branches.
"""

import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

# =============================================================================
# Unified Reporting Manager Tests
# =============================================================================

from services.dora_integration.reporting.unified_reporting import (
    ReportType,
    ReportStatus,
    ReportChannel,
    PackageFormat,
    ClientType,
    ReportDestination,
    ReportValidationResult,
    UnifiedReport,
    SubmissionPackage,
    UnifiedReportingConfig,
    UnifiedReportingManager,
    create_unified_reporting_manager,
    create_report_destination,
    get_report_types,
    get_report_statuses,
    _escape_xml,
)


class TestUnifiedReportingManager:
    """Tests for UnifiedReportingManager class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return UnifiedReportingConfig(
            provider_lei="549300TESTLEI00001",
            provider_name="Test ICT Provider",
            validate_on_register=True,
            require_all_mandatory_fields=True,
            default_format=PackageFormat.JSON,
            encryption_enabled=True,
            max_retry_attempts=3,
        )

    @pytest.fixture
    def manager(self, config):
        """Create test manager instance."""
        return UnifiedReportingManager(config)

    @pytest.fixture
    def destination(self):
        """Create test destination."""
        return ReportDestination(
            name="Test Client",
            client_id="CLIENT-001",
            client_type=ClientType.INVESTMENT_FIRM,
            channel=ReportChannel.API,
            endpoint="https://api.client.com/reports",
            encryption_required=True,
            contact_email="contact@client.com",
            preferred_format=PackageFormat.JSON,
        )

    def test_manager_initialization(self, config):
        """Test manager initialization."""
        manager = UnifiedReportingManager(config)

        assert manager.config == config
        assert len(manager._reports) == 0
        assert len(manager._packages) == 0

    def test_manager_initialization_default_config(self):
        """Test manager with default config."""
        manager = UnifiedReportingManager()

        assert manager.config is not None
        assert isinstance(manager.config, UnifiedReportingConfig)

    def test_create_report(self, manager, destination):
        """Test report creation."""
        report = manager.create_report(
            report_type=ReportType.DORA_MAJOR_INCIDENT,
            content={
                "incident_id": "INC-001",
                "classification": "major",
                "services_affected": ["trading", "market_data"],
            },
            destination=destination,
            due_at=datetime.now(timezone.utc) + timedelta(hours=4),
            reference_id="REF-001",
            classification="confidential",
            attachments=["doc1.pdf"],
        )

        assert report.report_id.startswith("REPORT-")
        assert report.report_type == ReportType.DORA_MAJOR_INCIDENT
        assert report.status == ReportStatus.DRAFT
        assert "incident_id" in report.content
        assert report.destination.client_id == "CLIENT-001"

    def test_create_report_with_validation_errors(self, manager, destination):
        """Test report creation with missing mandatory fields."""
        report = manager.create_report(
            report_type=ReportType.DORA_MAJOR_INCIDENT,
            content={
                "incident_id": "INC-001",
                # Missing: classification, services_affected
            },
            destination=destination,
            due_at=datetime.now(timezone.utc) + timedelta(hours=4),
        )

        assert report.validation_result is not None
        assert not report.validation_result.is_valid
        assert len(report.validation_result.errors) > 0

    def test_register_report(self, manager, destination):
        """Test report registration."""
        report = UnifiedReport(
            report_type=ReportType.DORA_REGISTER_UPDATE,
            content={
                "arrangement_reference": "ARR-001",
                "provider_name": "Test Provider",
            },
            destination=destination,
            due_at=datetime.now(timezone.utc) + timedelta(days=1),
        )

        registered = manager.register_report(report)

        assert registered.report_id == report.report_id
        assert report.report_id in manager._reports

    def test_mark_ready_success(self, manager, destination):
        """Test marking report as ready."""
        report = manager.create_report(
            report_type=ReportType.DORA_MAJOR_INCIDENT,
            content={
                "incident_id": "INC-001",
                "classification": "major",
                "services_affected": ["trading"],
            },
            destination=destination,
            due_at=datetime.now(timezone.utc) + timedelta(hours=4),
        )

        result = manager.mark_ready(report.report_id)

        assert result is True
        assert report.status == ReportStatus.READY
        assert report.validated_at is not None

    def test_mark_ready_validation_failure(self, manager, destination):
        """Test mark_ready fails on validation error."""
        report = manager.create_report(
            report_type=ReportType.DORA_MAJOR_INCIDENT,
            content={
                "incident_id": "INC-001",
                # Missing mandatory fields
            },
            destination=destination,
            due_at=datetime.now(timezone.utc) + timedelta(hours=4),
        )

        result = manager.mark_ready(report.report_id)

        assert result is False
        assert report.status != ReportStatus.READY

    def test_mark_ready_nonexistent_report(self, manager):
        """Test mark_ready with nonexistent report."""
        result = manager.mark_ready("NONEXISTENT-ID")

        assert result is False

    def test_mark_delivered(self, manager, destination):
        """Test marking report as delivered."""
        report = manager.create_report(
            report_type=ReportType.DORA_MAJOR_INCIDENT,
            content={
                "incident_id": "INC-001",
                "classification": "major",
                "services_affected": ["trading"],
            },
            destination=destination,
            due_at=datetime.now(timezone.utc) + timedelta(hours=4),
        )
        manager.mark_ready(report.report_id)

        result = manager.mark_delivered(report.report_id, "ACK-12345")

        assert result is True
        assert report.status == ReportStatus.DELIVERED
        assert report.delivered_at is not None
        assert report.acknowledgment_id == "ACK-12345"

    def test_mark_delivered_nonexistent(self, manager):
        """Test mark_delivered with nonexistent report."""
        result = manager.mark_delivered("NONEXISTENT", "ACK")

        assert result is False

    def test_mark_submitted(self, manager, destination):
        """Test marking report as submitted to NCA."""
        report = manager.create_report(
            report_type=ReportType.DORA_MAJOR_INCIDENT,
            content={
                "incident_id": "INC-001",
                "classification": "major",
                "services_affected": ["trading"],
            },
            destination=destination,
            due_at=datetime.now(timezone.utc) + timedelta(hours=4),
        )
        manager.mark_ready(report.report_id)
        manager.mark_delivered(report.report_id)

        result = manager.mark_submitted(report.report_id)

        assert result is True
        assert report.status == ReportStatus.SUBMITTED
        assert report.submitted_at is not None

    def test_mark_submitted_nonexistent(self, manager):
        """Test mark_submitted with nonexistent report."""
        result = manager.mark_submitted("NONEXISTENT")

        assert result is False

    def test_get_report(self, manager, destination):
        """Test getting report by ID."""
        report = manager.create_report(
            report_type=ReportType.INTERNAL_RESILIENCE,
            content={"summary": "Test", "owner": "IT"},
            destination=destination,
            due_at=datetime.now(timezone.utc) + timedelta(days=7),
        )

        retrieved = manager.get_report(report.report_id)

        assert retrieved is not None
        assert retrieved.report_id == report.report_id

    def test_get_report_nonexistent(self, manager):
        """Test getting nonexistent report."""
        result = manager.get_report("NONEXISTENT")

        assert result is None

    def test_get_reports_for_client(self, manager, destination):
        """Test getting reports for specific client."""
        for i in range(3):
            manager.create_report(
                report_type=ReportType.DORA_REGISTER_UPDATE,
                content={
                    "arrangement_reference": f"ARR-{i}",
                    "provider_name": "Provider",
                },
                destination=destination,
                due_at=datetime.now(timezone.utc) + timedelta(days=i),
            )

        reports = manager.get_reports_for_client("CLIENT-001")

        assert len(reports) == 3

    def test_get_reports_for_client_with_status_filter(self, manager, destination):
        """Test filtering client reports by status."""
        report = manager.create_report(
            report_type=ReportType.DORA_MAJOR_INCIDENT,
            content={
                "incident_id": "INC-001",
                "classification": "major",
                "services_affected": ["trading"],
            },
            destination=destination,
            due_at=datetime.now(timezone.utc) + timedelta(hours=4),
        )
        manager.mark_ready(report.report_id)

        ready_reports = manager.get_reports_for_client("CLIENT-001", status=ReportStatus.READY)

        assert len(ready_reports) == 1
        assert ready_reports[0].status == ReportStatus.READY

    def test_get_pending_reports(self, manager, destination):
        """Test getting pending reports."""
        report1 = manager.create_report(
            report_type=ReportType.DORA_MAJOR_INCIDENT,
            content={
                "incident_id": "INC-001",
                "classification": "major",
                "services_affected": ["trading"],
            },
            destination=destination,
            due_at=datetime.now(timezone.utc) + timedelta(hours=2),
        )

        pending = manager.get_pending_reports()

        assert len(pending) == 1
        assert pending[0].status in {ReportStatus.DRAFT, ReportStatus.READY}

    def test_get_pending_reports_with_type_filter(self, manager, destination):
        """Test filtering pending reports by type."""
        manager.create_report(
            report_type=ReportType.DORA_MAJOR_INCIDENT,
            content={
                "incident_id": "INC-001",
                "classification": "major",
                "services_affected": ["trading"],
            },
            destination=destination,
            due_at=datetime.now(timezone.utc) + timedelta(hours=4),
        )
        manager.create_report(
            report_type=ReportType.INTERNAL_RESILIENCE,
            content={"summary": "Test", "owner": "IT"},
            destination=destination,
            due_at=datetime.now(timezone.utc) + timedelta(days=7),
        )

        pending = manager.get_pending_reports(report_type=ReportType.DORA_MAJOR_INCIDENT)

        assert len(pending) == 1
        assert pending[0].report_type == ReportType.DORA_MAJOR_INCIDENT

    def test_get_ready_reports_for_client(self, manager, destination):
        """Test getting ready reports for client."""
        report = manager.create_report(
            report_type=ReportType.DORA_MAJOR_INCIDENT,
            content={
                "incident_id": "INC-001",
                "classification": "major",
                "services_affected": ["trading"],
            },
            destination=destination,
            due_at=datetime.now(timezone.utc) + timedelta(hours=4),
        )
        manager.mark_ready(report.report_id)

        ready = manager.get_ready_reports_for_client("CLIENT-001")

        assert len(ready) == 1
        assert ready[0].status == ReportStatus.READY

    def test_get_overdue_reports(self, manager, destination):
        """Test getting overdue reports."""
        report = manager.create_report(
            report_type=ReportType.DORA_MAJOR_INCIDENT,
            content={
                "incident_id": "INC-001",
                "classification": "major",
                "services_affected": ["trading"],
            },
            destination=destination,
            due_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )

        overdue = manager.get_overdue_reports()

        assert len(overdue) == 1
        assert overdue[0].report_id == report.report_id

    def test_generate_submission_package_json(self, manager, destination):
        """Test generating JSON submission package."""
        report = manager.create_report(
            report_type=ReportType.DORA_MAJOR_INCIDENT,
            content={
                "incident_id": "INC-001",
                "classification": "major",
                "services_affected": ["trading"],
            },
            destination=destination,
            due_at=datetime.now(timezone.utc) + timedelta(hours=4),
        )
        manager.mark_ready(report.report_id)

        package = manager.generate_submission_package("CLIENT-001")

        assert package.package_id.startswith("PKG-")
        assert len(package.reports) == 1
        assert package.package_format == PackageFormat.JSON
        assert len(package.package_content) > 0
        assert report.status == ReportStatus.PACKAGED

    def test_generate_submission_package_csv(self, manager, destination):
        """Test generating CSV submission package."""
        report = manager.create_report(
            report_type=ReportType.DORA_MAJOR_INCIDENT,
            content={
                "incident_id": "INC-001",
                "classification": "major",
                "services_affected": ["trading"],
            },
            destination=destination,
            due_at=datetime.now(timezone.utc) + timedelta(hours=4),
        )
        manager.mark_ready(report.report_id)

        package = manager.generate_submission_package(
            "CLIENT-001",
            package_format=PackageFormat.CSV,
        )

        assert package.package_format == PackageFormat.CSV
        content = package.package_content.decode("utf-8")
        assert "report_id" in content

    def test_generate_submission_package_xml(self, manager, destination):
        """Test generating XML submission package."""
        report = manager.create_report(
            report_type=ReportType.DORA_MAJOR_INCIDENT,
            content={
                "incident_id": "INC-001",
                "classification": "major",
                "services_affected": ["trading"],
            },
            destination=destination,
            due_at=datetime.now(timezone.utc) + timedelta(hours=4),
        )
        manager.mark_ready(report.report_id)

        package = manager.generate_submission_package(
            "CLIENT-001",
            package_format=PackageFormat.XML,
        )

        assert package.package_format == PackageFormat.XML
        content = package.package_content.decode("utf-8")
        assert "<?xml" in content
        assert "<ReportPackage>" in content

    def test_generate_submission_package_no_ready_reports(self, manager):
        """Test package generation fails without ready reports."""
        with pytest.raises(ValueError):
            manager.generate_submission_package("CLIENT-001")

    def test_export_summary(self, manager, destination):
        """Test export summary."""
        manager.create_report(
            report_type=ReportType.DORA_REGISTER_UPDATE,
            content={
                "arrangement_reference": "ARR-001",
                "provider_name": "Provider",
            },
            destination=destination,
            due_at=datetime.now(timezone.utc) + timedelta(days=1),
        )

        summary = manager.export_summary()

        assert len(summary) == 1
        assert "report_id" in summary[0]
        assert "type" in summary[0]
        assert "status" in summary[0]

    def test_get_statistics(self, manager, destination):
        """Test getting statistics."""
        manager.create_report(
            report_type=ReportType.DORA_MAJOR_INCIDENT,
            content={
                "incident_id": "INC-001",
                "classification": "major",
                "services_affected": ["trading"],
            },
            destination=destination,
            due_at=datetime.now(timezone.utc) + timedelta(hours=4),
        )

        stats = manager.get_statistics()

        assert stats["total_reports"] == 1
        assert "by_status" in stats
        assert "by_type" in stats
        assert "overdue_count" in stats
        assert "clients_count" in stats

    def test_report_to_dict(self, destination):
        """Test report to_dict method."""
        report = UnifiedReport(
            report_type=ReportType.DORA_MAJOR_INCIDENT,
            content={"test": "data"},
            destination=destination,
            due_at=datetime.now(timezone.utc) + timedelta(hours=4),
        )

        data = report.to_dict()

        assert "report_id" in data
        assert "report_type" in data
        assert "content" in data
        assert "destination" in data

    def test_report_mark_methods(self, destination):
        """Test report mark_* methods."""
        report = UnifiedReport(
            report_type=ReportType.DORA_MAJOR_INCIDENT,
            content={"test": "data"},
            destination=destination,
            due_at=datetime.now(timezone.utc) + timedelta(hours=4),
        )

        # Test mark_validated
        validation = ReportValidationResult(is_valid=True)
        report.mark_validated(validation)
        assert report.status == ReportStatus.READY
        assert report.validated_at is not None

        # Test mark_packaged
        report.mark_packaged()
        assert report.status == ReportStatus.PACKAGED
        assert report.packaged_at is not None

        # Test mark_delivered
        report.mark_delivered("ACK-123")
        assert report.status == ReportStatus.DELIVERED
        assert report.delivered_at is not None
        assert report.acknowledgment_id == "ACK-123"

        # Test mark_submitted
        report.mark_submitted()
        assert report.status == ReportStatus.SUBMITTED
        assert report.submitted_at is not None

    def test_submission_package_to_dict(self, destination):
        """Test submission package to_dict method."""
        report = UnifiedReport(
            report_type=ReportType.DORA_MAJOR_INCIDENT,
            content={"test": "data"},
            destination=destination,
            due_at=datetime.now(timezone.utc) + timedelta(hours=4),
        )
        package = SubmissionPackage(
            destination=destination,
            reports=[report],
            package_format=PackageFormat.JSON,
        )

        data = package.to_dict()

        assert "package_id" in data
        assert "reports_count" in data
        assert data["reports_count"] == 1


class TestUnifiedReportingHelpers:
    """Tests for helper functions."""

    def test_create_unified_reporting_manager(self):
        """Test factory function."""
        manager = create_unified_reporting_manager()

        assert isinstance(manager, UnifiedReportingManager)

    def test_create_unified_reporting_manager_with_config(self):
        """Test factory with config."""
        config = UnifiedReportingConfig(provider_lei="TEST")
        manager = create_unified_reporting_manager(config)

        assert manager.config.provider_lei == "TEST"

    def test_create_report_destination(self):
        """Test destination factory function."""
        dest = create_report_destination(
            name="Test",
            client_id="CLI-001",
            channel=ReportChannel.PORTAL,
            endpoint="/portal",
        )

        assert dest.name == "Test"
        assert dest.client_id == "CLI-001"
        assert dest.channel == ReportChannel.PORTAL

    def test_get_report_types(self):
        """Test getting report types."""
        types = get_report_types()

        assert len(types) > 0
        assert ReportType.DORA_MAJOR_INCIDENT in types

    def test_get_report_statuses(self):
        """Test getting report statuses."""
        statuses = get_report_statuses()

        assert len(statuses) > 0
        assert ReportStatus.DRAFT in statuses

    def test_escape_xml(self):
        """Test XML escaping."""
        text = "<test attr=\"value\">&'data'"

        escaped = _escape_xml(text)

        assert "&lt;" in escaped
        assert "&gt;" in escaped
        assert "&amp;" in escaped
        assert "&quot;" in escaped
        assert "&apos;" in escaped


# =============================================================================
# Reporting Templates Tests
# =============================================================================

from services.dora_integration.reporting.reporting_templates import (
    IncidentTypeCode,
    DataTypeCode,
    ClientTypeCode,
    ServiceTypeCode,
    ResponseEffectivenessCode,
    ITSInitialNotificationTemplate,
    ITSIntermediateReportTemplate,
    ITSFinalReportTemplate,
    TimelineEvent,
    ClientIncidentDataPackage,
    DORAReportingTemplates,
    create_reporting_templates,
    get_incident_type_codes,
    get_data_type_codes,
    get_service_type_codes,
    get_client_type_codes,
    create_timeline_event,
)


class TestDORAReportingTemplates:
    """Tests for DORAReportingTemplates class."""

    @pytest.fixture
    def templates(self):
        """Create test templates instance."""
        return DORAReportingTemplates(
            provider_lei="549300TESTLEI00001",
            provider_name="Test ICT Provider",
            entity_lei="549300CLIENTLEI001",
            entity_name="Test Financial Entity",
            entity_type="investment_firm",
            entity_country="DE",
        )

    def test_templates_initialization(self, templates):
        """Test templates initialization."""
        assert templates.provider_lei == "549300TESTLEI00001"
        assert templates.provider_name == "Test ICT Provider"
        assert templates.entity_lei == "549300CLIENTLEI001"

    def test_create_initial_notification(self, templates):
        """Test creating initial notification template."""
        template = templates.create_initial_notification(
            incident_reference="INC-2025-001",
            detection_datetime="2025-01-15T10:00:00Z",
            classification_datetime="2025-01-15T10:30:00Z",
            brief_description="System failure affecting trading services",
            incident_type_code="SYSF",
            critical_services_affected=["trading", "market_data"],
            estimated_clients_affected=100,
            member_states_affected=["DE", "FR"],
            contact_person_name="John Doe",
            contact_person_email="john.doe@client.com",
            contact_person_phone="+49123456789",
            is_recurring=False,
            ict_services_affected=["cloud_hosting"],
        )

        assert template.report_reference.startswith("INIT-")
        assert template.incident_reference == "INC-2025-001"
        assert template.reporting_entity_lei == "549300CLIENTLEI001"
        assert template.ict_provider_lei == "549300TESTLEI00001"
        assert len(template.critical_services_affected) == 2
        assert template.estimated_clients_affected == 100

    def test_create_initial_notification_truncates_description(self, templates):
        """Test that brief description is truncated to 1000 chars."""
        long_desc = "A" * 2000

        template = templates.create_initial_notification(
            incident_reference="INC-001",
            detection_datetime="2025-01-15T10:00:00Z",
            classification_datetime="2025-01-15T10:30:00Z",
            brief_description=long_desc,
        )

        assert len(template.brief_description) == 1000

    def test_create_intermediate_report(self, templates):
        """Test creating intermediate report template."""
        template = templates.create_intermediate_report(
            initial_report_reference="INIT-20250115-ABC123",
            incident_reference="INC-2025-001",
            detailed_description="Detailed analysis of the system failure...",
            preliminary_root_cause="Hardware malfunction in storage array",
            incident_type_code="SYSF",
            affected_clients_count=150,
            data_compromised=False,
            immediate_actions_taken=["Failover to backup systems"],
            ongoing_actions=["Root cause analysis"],
            is_ongoing=True,
            contact_person_email="john.doe@client.com",
            ict_provider_analysis="Storage array failure confirmed",
        )

        assert template.report_reference.startswith("INTM-")
        assert template.initial_report_reference == "INIT-20250115-ABC123"
        assert template.preliminary_root_cause == "Hardware malfunction in storage array"
        assert template.ict_provider_preliminary_analysis == "Storage array failure confirmed"

    def test_create_intermediate_report_default_root_cause(self, templates):
        """Test intermediate report with no root cause provided."""
        template = templates.create_intermediate_report(
            initial_report_reference="INIT-001",
            incident_reference="INC-001",
            detailed_description="Details...",
            # No preliminary_root_cause
        )

        assert template.preliminary_root_cause == "Under investigation"

    def test_create_final_report(self, templates):
        """Test creating final report template."""
        template = templates.create_final_report(
            initial_report_reference="INIT-20250115-ABC123",
            intermediate_report_reference="INTM-20250116-DEF456",
            incident_reference="INC-2025-001",
            incident_title="Storage Array Failure",
            comprehensive_description="Complete analysis of incident...",
            final_root_cause="Hardware malfunction due to firmware bug",
            incident_resolved=True,
            resolution_datetime="2025-01-17T14:00:00Z",
            total_duration_hours=52.0,
            total_clients_affected=150,
            total_economic_impact_eur=50000.0,
            lessons_learned=["Improve monitoring", "Update firmware"],
            remediation_measures=[{"description": "Replace storage arrays"}],
            preventive_measures=[{"description": "Enhanced monitoring"}],
            response_effectiveness_code="EFFC",
            contact_person_email="john.doe@client.com",
            ict_provider_final_analysis="Root cause confirmed",
            ict_provider_corrective_actions=["Firmware update deployed"],
        )

        assert template.report_reference.startswith("FINL-")
        assert template.incident_resolved is True
        assert template.total_duration_hours == 52.0
        assert len(template.lessons_learned) == 2
        assert len(template.ict_provider_corrective_actions) == 1

    def test_create_final_report_defaults(self, templates):
        """Test final report with defaults."""
        template = templates.create_final_report(
            initial_report_reference="INIT-001",
            intermediate_report_reference="INTM-001",
            incident_reference="INC-001",
            incident_title="Test",
            comprehensive_description="Details...",
            final_root_cause="Cause",
        )

        assert len(template.lessons_learned) == 1
        assert len(template.remediation_measures) == 1

    def test_validate_initial_notification_valid(self, templates):
        """Test validation of valid initial notification."""
        template = templates.create_initial_notification(
            incident_reference="INC-001",
            detection_datetime="2025-01-15T10:00:00Z",
            classification_datetime="2025-01-15T10:30:00Z",
            brief_description="Test incident",
            member_states_affected=["DE"],
            contact_person_email="test@example.com",
        )

        is_valid, errors = template.validate()

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_initial_notification_invalid(self, templates):
        """Test validation of invalid initial notification."""
        template = ITSInitialNotificationTemplate(
            # Missing required fields
        )

        is_valid, errors = template.validate()

        assert is_valid is False
        assert len(errors) > 0
        assert any("reporting_entity_lei" in e for e in errors)

    def test_validate_initial_notification_description_too_long(self, templates):
        """Test validation catches too long description."""
        template = ITSInitialNotificationTemplate(
            reporting_entity_lei="TEST",
            reporting_entity_name="Test",
            incident_reference="INC-001",
            detection_datetime="2025-01-15T10:00:00Z",
            classification_datetime="2025-01-15T10:30:00Z",
            brief_description="A" * 1001,
            contact_person_email="test@example.com",
            member_states_affected=["DE"],
        )

        is_valid, errors = template.validate()

        assert is_valid is False
        assert any("1000 characters" in e for e in errors)

    def test_validate_intermediate_report_invalid(self):
        """Test validation of invalid intermediate report."""
        template = ITSIntermediateReportTemplate(
            # Missing required fields
        )

        is_valid, errors = template.validate()

        assert is_valid is False
        assert any("initial_report_reference" in e for e in errors)

    def test_validate_final_report_invalid(self):
        """Test validation of invalid final report."""
        template = ITSFinalReportTemplate(
            # Missing required fields
        )

        is_valid, errors = template.validate()

        assert is_valid is False
        assert any("initial_report_reference" in e for e in errors)
        assert any("final_root_cause" in e for e in errors)

    def test_populate_initial_from_incident(self, templates):
        """Test populating template from incident data."""
        incident_data = {
            "incident_id": "INC-2025-001",
            "detected_at": "2025-01-15T10:00:00Z",
            "classified_at": "2025-01-15T10:30:00Z",
            "description": "System failure",
            "incident_type": "system_failure",
            "affected_services": ["trading"],
            "affected_clients_count": 100,
            "geographic_spread": ["DE", "FR"],
            "is_recurring": False,
            "provider_services_affected": ["cloud"],
        }
        contact = {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "+123",
        }

        template = templates.populate_initial_from_incident(incident_data, contact)

        assert template.incident_reference == "INC-2025-001"
        assert template.incident_type_code == "SYSF"
        assert template.contact_person_name == "John Doe"

    def test_map_incident_type(self, templates):
        """Test incident type mapping."""
        assert templates._map_incident_type("system_failure") == "SYSF"
        assert templates._map_incident_type("cyber_attack") == "CYBA"
        assert templates._map_incident_type("security_breach") == "CYBA"
        assert templates._map_incident_type("third_party_failure") == "TPFA"
        assert templates._map_incident_type("human_error") == "HUMA"
        assert templates._map_incident_type("external_event") == "EXTE"
        assert templates._map_incident_type("process_failure") == "PROC"
        assert templates._map_incident_type("unknown_type") == "UNKN"

    def test_generate_client_data_package(self, templates):
        """Test generating client data package."""
        incident_data = {
            "incident_id": "INC-001",
            "detected_at": "2025-01-15T10:00:00Z",
            "classified_at": "2025-01-15T10:30:00Z",
            "description": "Test",
            "timeline": [
                {
                    "timestamp": "2025-01-15T10:00:00Z",
                    "type": "detection",
                    "description": "Detected",
                },
            ],
        }

        package = templates.generate_client_data_package(
            incident_id="INC-001",
            incident_data=incident_data,
            include_templates=True,
        )

        assert package.package_id.startswith("INCPKG-")
        assert package.incident_id == "INC-001"
        assert package.initial_template is not None
        assert len(package.timeline_events) == 1

    def test_generate_client_data_package_without_templates(self, templates):
        """Test generating package without templates."""
        package = templates.generate_client_data_package(
            incident_id="INC-001",
            incident_data={"test": "data"},
            include_templates=False,
        )

        assert package.initial_template is None

    def test_export_to_json(self, templates):
        """Test JSON export."""
        template = templates.create_initial_notification(
            incident_reference="INC-001",
            detection_datetime="2025-01-15T10:00:00Z",
            classification_datetime="2025-01-15T10:30:00Z",
            brief_description="Test",
        )

        json_str = templates.export_to_json(template)

        data = json.loads(json_str)
        assert "incident_reference" in data
        assert data["incident_reference"] == "INC-001"

    def test_export_to_dict(self, templates):
        """Test dict export."""
        template = templates.create_initial_notification(
            incident_reference="INC-001",
            detection_datetime="2025-01-15T10:00:00Z",
            classification_datetime="2025-01-15T10:30:00Z",
            brief_description="Test",
        )

        data = templates.export_to_dict(template)

        assert isinstance(data, dict)
        assert data["incident_reference"] == "INC-001"

    def test_export_to_csv(self, templates):
        """Test CSV export."""
        template = templates.create_initial_notification(
            incident_reference="INC-001",
            detection_datetime="2025-01-15T10:00:00Z",
            classification_datetime="2025-01-15T10:30:00Z",
            brief_description="Test",
        )

        csv_str = templates.export_to_csv(template)

        assert "incident_reference" in csv_str
        assert "INC-001" in csv_str

    def test_export_to_xml(self, templates):
        """Test XML export."""
        template = templates.create_initial_notification(
            incident_reference="INC-001",
            detection_datetime="2025-01-15T10:00:00Z",
            classification_datetime="2025-01-15T10:30:00Z",
            brief_description="Test",
        )

        xml_str = templates.export_to_xml(template)

        assert "<?xml" in xml_str
        assert "<incident_reference>" in xml_str

    def test_export_package_to_json(self, templates):
        """Test package JSON export."""
        package = templates.generate_client_data_package(
            incident_id="INC-001",
            incident_data={"test": "data"},
        )

        json_str = templates.export_package_to_json(package)

        data = json.loads(json_str)
        assert "package_id" in data
        assert "provider" in data

    def test_validate_template(self, templates):
        """Test template validation helper."""
        template = templates.create_initial_notification(
            incident_reference="INC-001",
            detection_datetime="2025-01-15T10:00:00Z",
            classification_datetime="2025-01-15T10:30:00Z",
            brief_description="Test",
            member_states_affected=["DE"],
            contact_person_email="test@example.com",
        )

        is_valid, errors = templates.validate_template(template)

        assert is_valid is True

    def test_validate_all_mandatory_fields(self, templates):
        """Test mandatory fields validation."""
        template = templates.create_initial_notification(
            incident_reference="INC-001",
            detection_datetime="2025-01-15T10:00:00Z",
            classification_datetime="2025-01-15T10:30:00Z",
            brief_description="Test",
        )

        is_valid, errors = templates.validate_all_mandatory_fields(
            template,
            ["incident_reference", "brief_description", "nonexistent_field"],
        )

        assert is_valid is False
        assert any("nonexistent_field" in e for e in errors)


class TestReportingTemplatesHelpers:
    """Tests for reporting templates helper functions."""

    def test_create_reporting_templates(self):
        """Test factory function."""
        templates = create_reporting_templates(
            provider_lei="TEST",
            provider_name="Test Provider",
        )

        assert isinstance(templates, DORAReportingTemplates)
        assert templates.provider_lei == "TEST"

    def test_get_incident_type_codes(self):
        """Test getting incident type codes."""
        codes = get_incident_type_codes()

        assert "CYBA" in codes
        assert "SYSF" in codes

    def test_get_data_type_codes(self):
        """Test getting data type codes."""
        codes = get_data_type_codes()

        assert "PERS" in codes
        assert "FINA" in codes

    def test_get_service_type_codes(self):
        """Test getting service type codes."""
        codes = get_service_type_codes()

        assert "OREX" in codes
        assert "MKTD" in codes

    def test_get_client_type_codes(self):
        """Test getting client type codes."""
        codes = get_client_type_codes()

        assert "RETA" in codes
        assert "PROF" in codes

    def test_create_timeline_event(self):
        """Test creating timeline event."""
        event = create_timeline_event(
            timestamp="2025-01-15T10:00:00Z",
            event_type="detection",
            description="Incident detected",
            actor="SOC",
            system_affected="Trading",
        )

        assert event.event_id.startswith("EVT-")
        assert event.timestamp == "2025-01-15T10:00:00Z"
        assert event.event_type == "detection"


# =============================================================================
# Register of Information Tests
# =============================================================================

from services.dora_integration.reporting.register_of_information import (
    ContractType,
    ServiceType,
    FunctionType,
    DataLocation,
    ProviderLocationType,
    SubcontractingLevel,
    ExportFormat,
    ProviderIdentification,
    ContractReferenceData,
    SubcontractorData,
    ServiceRecord,
    ROIDataPackage,
    ROIDataGeneratorConfig,
    DORARegisterOfInformation,
    create_register_of_information,
    create_roi_data_generator,
    get_contract_types,
    get_service_types,
    get_subcontracting_levels,
    get_its_templates_provided,
    get_its_templates_client_provides,
)


class TestDORARegisterOfInformation:
    """Tests for DORARegisterOfInformation (ROI Data Generator) class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ROIDataGeneratorConfig(
            provider_lei="549300TESTPROVIDER",
            provider_name="Test ICT Provider GmbH",
            provider_country="DE",
            provider_address="Test Street 1, Berlin",
            parent_lei="549300PARENTLEI001",
            parent_name="Parent Corp",
            default_contact_name="John Doe",
            default_contact_email="john@provider.com",
            default_contact_phone="+49123456789",
            is_designated_ctpp=False,
            require_lei=True,
            validate_countries=True,
        )

    @pytest.fixture
    def generator(self, config):
        """Create test generator instance."""
        return DORARegisterOfInformation(config)

    def test_generator_initialization(self, config):
        """Test generator initialization."""
        generator = DORARegisterOfInformation(config)

        assert generator.config == config
        assert generator._provider_identification is not None
        assert generator._provider_identification.lei == "549300TESTPROVIDER"

    def test_generator_initialization_default_config(self):
        """Test generator with default config."""
        generator = DORARegisterOfInformation()

        assert generator.config is not None
        assert generator._provider_identification is not None

    def test_get_provider_identification(self, generator):
        """Test getting provider identification."""
        provider = generator.get_provider_identification()

        assert provider.lei == "549300TESTPROVIDER"
        assert provider.legal_name == "Test ICT Provider GmbH"
        assert provider.headquarters_country == "DE"
        assert provider.location_type == ProviderLocationType.EU_MEMBER_STATE

    def test_update_provider_identification(self, generator):
        """Test updating provider identification."""
        updated = generator.update_provider_identification(
            trading_name="New Trading Name",
            primary_contact_email="new@provider.com",
        )

        assert updated.trading_name == "New Trading Name"
        assert updated.primary_contact_email == "new@provider.com"

    def test_determine_location_type_eu(self, generator):
        """Test EU country detection."""
        location = generator._determine_location_type("DE")
        assert location == ProviderLocationType.EU_MEMBER_STATE

        location = generator._determine_location_type("FR")
        assert location == ProviderLocationType.EU_MEMBER_STATE

    def test_determine_location_type_eea(self, generator):
        """Test EEA country detection."""
        location = generator._determine_location_type("NO")
        assert location == ProviderLocationType.EEA_COUNTRY

        location = generator._determine_location_type("IS")
        assert location == ProviderLocationType.EEA_COUNTRY

    def test_determine_location_type_third_country(self, generator):
        """Test third country detection."""
        location = generator._determine_location_type("US")
        assert location == ProviderLocationType.THIRD_COUNTRY

    def test_add_contract(self, generator):
        """Test adding contract."""
        contract = generator.add_contract(
            contract_type=ContractType.PROCUREMENT,
            service_types_provided=[ServiceType.CLOUD_COMPUTING.value],
            contract_start_date="2025-01-01",
            contract_end_date="2026-12-31",
            annual_value_eur=100000.0,
            notice_period_days=90,
            data_processing_countries=["DE", "NL"],
            data_storage_countries=["DE"],
            personal_data_processed=True,
            subcontracting_permitted=True,
            audit_rights_granted=True,
            exit_plan_provided=True,
        )

        assert contract.contract_reference.startswith("CTR-")
        assert contract.provider_lei == "549300TESTPROVIDER"
        assert contract.contract_type == ContractType.PROCUREMENT
        assert contract.annual_value_eur == 100000.0
        assert len(contract.data_processing_countries) == 2
        assert "DE" in contract.data_processing_countries

    def test_get_contract(self, generator):
        """Test getting contract by reference."""
        contract = generator.add_contract(
            contract_type=ContractType.OUTSOURCING,
        )

        retrieved = generator.get_contract(contract.contract_reference)

        assert retrieved is not None
        assert retrieved.contract_reference == contract.contract_reference

    def test_get_contract_nonexistent(self, generator):
        """Test getting nonexistent contract."""
        result = generator.get_contract("NONEXISTENT")

        assert result is None

    def test_get_all_contracts(self, generator):
        """Test getting all contracts."""
        generator.add_contract(contract_type=ContractType.PROCUREMENT)
        generator.add_contract(contract_type=ContractType.OUTSOURCING)

        contracts = generator.get_all_contracts()

        assert len(contracts) == 2

    def test_update_contract(self, generator):
        """Test updating contract."""
        contract = generator.add_contract(
            contract_type=ContractType.PROCUREMENT,
            annual_value_eur=100000.0,
        )

        updated = generator.update_contract(
            contract.contract_reference,
            annual_value_eur=150000.0,
            notice_period_days=60,
        )

        assert updated is not None
        assert updated.annual_value_eur == 150000.0
        assert updated.notice_period_days == 60

    def test_update_contract_nonexistent(self, generator):
        """Test updating nonexistent contract."""
        result = generator.update_contract("NONEXISTENT", annual_value_eur=50000)

        assert result is None

    def test_add_service(self, generator):
        """Test adding service."""
        contract = generator.add_contract(contract_type=ContractType.PROCUREMENT)

        service = generator.add_service(
            contract_reference=contract.contract_reference,
            service_name="Cloud Hosting",
            service_type=ServiceType.CLOUD_COMPUTING,
            service_description="IaaS cloud hosting services",
            availability_target_pct=99.9,
            rpo_hours=4,
            rto_hours=2,
            supports_trading_functions=True,
            supports_payment_functions=False,
            personal_data_involved=True,
        )

        assert service is not None
        assert service.service_id.startswith("SVC-")
        assert service.service_name == "Cloud Hosting"
        assert service.service_type == ServiceType.CLOUD_COMPUTING
        assert service.supports_trading_functions is True

    def test_add_service_nonexistent_contract(self, generator):
        """Test adding service to nonexistent contract."""
        result = generator.add_service(
            contract_reference="NONEXISTENT",
            service_name="Test",
            service_type=ServiceType.OTHER,
        )

        assert result is None

    def test_get_services_for_contract(self, generator):
        """Test getting services for contract."""
        contract = generator.add_contract(contract_type=ContractType.PROCUREMENT)

        generator.add_service(
            contract_reference=contract.contract_reference,
            service_name="Service 1",
            service_type=ServiceType.CLOUD_COMPUTING,
        )
        generator.add_service(
            contract_reference=contract.contract_reference,
            service_name="Service 2",
            service_type=ServiceType.DATA_ANALYTICS,
        )

        services = generator.get_services_for_contract(contract.contract_reference)

        assert len(services) == 2

    def test_get_all_services(self, generator):
        """Test getting all services."""
        contract = generator.add_contract(contract_type=ContractType.PROCUREMENT)
        generator.add_service(
            contract_reference=contract.contract_reference,
            service_name="Service 1",
            service_type=ServiceType.CLOUD_COMPUTING,
        )

        services = generator.get_all_services()

        assert len(services) == 1

    def test_add_subcontractor(self, generator):
        """Test adding subcontractor."""
        contract = generator.add_contract(
            contract_type=ContractType.PROCUREMENT,
            subcontracting_permitted=True,
        )

        subcontractor = generator.add_subcontractor(
            parent_contract_reference=contract.contract_reference,
            legal_name="Subcontractor Ltd",
            country="NL",
            lei="549300SUBLEI000001",
            subcontracting_level=SubcontractingLevel.LEVEL_1,
            services_subcontracted=["Infrastructure"],
            data_processing_countries=["NL"],
            personal_data_access=True,
        )

        assert subcontractor is not None
        assert subcontractor.subcontractor_id.startswith("SUB-")
        assert subcontractor.legal_name == "Subcontractor Ltd"
        assert subcontractor.country == "NL"
        assert subcontractor.personal_data_access is True

    def test_add_subcontractor_nonexistent_contract(self, generator):
        """Test adding subcontractor to nonexistent contract."""
        result = generator.add_subcontractor(
            parent_contract_reference="NONEXISTENT",
            legal_name="Test",
            country="DE",
        )

        assert result is None

    def test_get_subcontractors_for_contract(self, generator):
        """Test getting subcontractors for contract."""
        contract = generator.add_contract(
            contract_type=ContractType.PROCUREMENT,
            subcontracting_permitted=True,
        )

        generator.add_subcontractor(
            parent_contract_reference=contract.contract_reference,
            legal_name="Sub 1",
            country="NL",
        )
        generator.add_subcontractor(
            parent_contract_reference=contract.contract_reference,
            legal_name="Sub 2",
            country="BE",
        )

        subcontractors = generator.get_subcontractors_for_contract(contract.contract_reference)

        assert len(subcontractors) == 2

    def test_get_full_subcontracting_chain(self, generator):
        """Test getting full subcontracting chain."""
        contract = generator.add_contract(
            contract_type=ContractType.PROCUREMENT,
            subcontracting_permitted=True,
        )

        generator.add_subcontractor(
            parent_contract_reference=contract.contract_reference,
            legal_name="Sub 1",
            country="NL",
            services_subcontracted=["Storage"],
        )

        chain = generator.get_full_subcontracting_chain(contract.contract_reference)

        assert "contract_reference" in chain
        assert "chain" in chain
        assert len(chain["chain"]) == 1

    def test_get_full_subcontracting_chain_nonexistent(self, generator):
        """Test chain for nonexistent contract."""
        result = generator.get_full_subcontracting_chain("NONEXISTENT")

        assert result == {}

    def test_get_all_subcontractors(self, generator):
        """Test getting all subcontractors."""
        contract = generator.add_contract(
            contract_type=ContractType.PROCUREMENT,
            subcontracting_permitted=True,
        )
        generator.add_subcontractor(
            parent_contract_reference=contract.contract_reference,
            legal_name="Sub 1",
            country="NL",
        )

        subcontractors = generator.get_all_subcontractors()

        assert len(subcontractors) == 1

    def test_generate_roi_data_package(self, generator):
        """Test generating ROI data package."""
        contract = generator.add_contract(
            contract_type=ContractType.PROCUREMENT,
            service_types_provided=[ServiceType.CLOUD_COMPUTING.value],
            subcontracting_permitted=True,
        )

        generator.add_service(
            contract_reference=contract.contract_reference,
            service_name="Cloud",
            service_type=ServiceType.CLOUD_COMPUTING,
        )

        generator.add_subcontractor(
            parent_contract_reference=contract.contract_reference,
            legal_name="Sub",
            country="NL",
        )

        package = generator.generate_roi_data_package(
            reference_date="2025-03-31",
        )

        assert package.package_id.startswith("ROI-PKG-")
        assert package.reference_date == "2025-03-31"
        assert package.provider is not None
        assert len(package.contracts) == 1
        assert len(package.services) == 1
        assert len(package.subcontractors) == 1
        assert package.is_validated is True

    def test_generate_roi_data_package_specific_contracts(self, generator):
        """Test package with specific contracts."""
        contract1 = generator.add_contract(contract_type=ContractType.PROCUREMENT)
        contract2 = generator.add_contract(contract_type=ContractType.OUTSOURCING)

        package = generator.generate_roi_data_package(
            contract_references=[contract1.contract_reference],
        )

        assert len(package.contracts) == 1
        assert package.contracts[0].contract_reference == contract1.contract_reference

    def test_generate_roi_data_package_validation_warnings(self, generator):
        """Test package validation generates warnings."""
        contract = generator.add_contract(
            contract_type=ContractType.PROCUREMENT,
            subcontracting_permitted=True,
            # No subcontractors added
        )

        package = generator.generate_roi_data_package()

        assert len(package.validation_warnings) > 0

    def test_validate_package_missing_service_types(self, generator):
        """Test validation with missing service types."""
        generator.add_contract(
            contract_type=ContractType.PROCUREMENT,
            service_types_provided=[],  # Empty
        )

        package = generator.generate_roi_data_package()

        assert any("service types" in w.lower() for w in package.validation_warnings)

    def test_export_package_to_json(self, generator):
        """Test JSON export."""
        contract = generator.add_contract(contract_type=ContractType.PROCUREMENT)
        generator.add_service(
            contract_reference=contract.contract_reference,
            service_name="Test",
            service_type=ServiceType.CLOUD_COMPUTING,
        )

        package = generator.generate_roi_data_package()
        json_str = generator.export_package_to_json(package)

        data = json.loads(json_str)
        assert "package_id" in data
        assert "provider" in data
        assert "contracts" in data

    def test_export_package_to_csv(self, generator):
        """Test CSV export."""
        contract = generator.add_contract(contract_type=ContractType.PROCUREMENT)
        generator.add_service(
            contract_reference=contract.contract_reference,
            service_name="Test",
            service_type=ServiceType.CLOUD_COMPUTING,
        )
        generator.add_subcontractor(
            parent_contract_reference=contract.contract_reference,
            legal_name="Sub",
            country="NL",
        )

        package = generator.generate_roi_data_package()
        csv_exports = generator.export_package_to_csv(package)

        assert "B_03_01_Provider" in csv_exports
        assert "B_02_01_Contracts" in csv_exports
        assert "B_06_01_Services" in csv_exports
        assert "B_04_01_Subcontractors" in csv_exports

        # Check content
        assert "Provider_ID" in csv_exports["B_03_01_Provider"]
        assert "Contract_Reference" in csv_exports["B_02_01_Contracts"]

    def test_export_package_to_xml(self, generator):
        """Test XML export."""
        contract = generator.add_contract(contract_type=ContractType.PROCUREMENT)
        package = generator.generate_roi_data_package()

        xml_str = generator.export_package_to_xml(package)

        assert "<?xml" in xml_str
        assert "<ROI_DataPackage>" in xml_str

    def test_get_statistics(self, generator):
        """Test getting statistics."""
        contract1 = generator.add_contract(contract_type=ContractType.PROCUREMENT)
        contract2 = generator.add_contract(contract_type=ContractType.OUTSOURCING)

        generator.add_service(
            contract_reference=contract1.contract_reference,
            service_name="Cloud",
            service_type=ServiceType.CLOUD_COMPUTING,
        )

        stats = generator.get_statistics()

        assert stats["provider"]["lei"] == "549300TESTPROVIDER"
        assert stats["data_counts"]["contracts"] == 2
        assert stats["data_counts"]["services"] == 1
        assert "procurement" in stats["contracts_by_type"]
        assert "cloud_computing" in stats["services_by_type"]

    def test_provider_identification_to_dict(self, generator):
        """Test provider identification to_dict."""
        provider = generator.get_provider_identification()

        data = provider.to_dict()

        assert "provider_id" in data
        assert "lei" in data
        assert "legal_name" in data

    def test_contract_reference_data_to_dict(self, generator):
        """Test contract reference data to_dict."""
        contract = generator.add_contract(
            contract_type=ContractType.PROCUREMENT,
            data_processing_countries=["DE"],
        )

        data = contract.to_dict()

        assert "contract_reference" in data
        assert "contract_type" in data
        assert data["data_processing_countries"] == ["DE"]

    def test_service_record_to_dict(self, generator):
        """Test service record to_dict."""
        contract = generator.add_contract(contract_type=ContractType.PROCUREMENT)
        service = generator.add_service(
            contract_reference=contract.contract_reference,
            service_name="Test",
            service_type=ServiceType.CLOUD_COMPUTING,
        )

        data = service.to_dict()

        assert "service_id" in data
        assert "service_type" in data
        assert data["service_type"] == "cloud_computing"

    def test_subcontractor_data_to_dict(self, generator):
        """Test subcontractor data to_dict."""
        contract = generator.add_contract(
            contract_type=ContractType.PROCUREMENT,
            subcontracting_permitted=True,
        )
        sub = generator.add_subcontractor(
            parent_contract_reference=contract.contract_reference,
            legal_name="Sub",
            country="NL",
        )

        data = sub.to_dict()

        assert "subcontractor_id" in data
        assert "subcontracting_level" in data
        assert data["subcontracting_level"] == "level_1"

    def test_roi_data_package_to_dict(self, generator):
        """Test ROI data package to_dict."""
        contract = generator.add_contract(contract_type=ContractType.PROCUREMENT)
        package = generator.generate_roi_data_package()

        data = package.to_dict()

        assert "package_id" in data
        assert "provider" in data
        assert "contracts" in data
        assert "services" in data
        assert "subcontractors" in data
        assert "is_validated" in data

    def test_is_valid_country(self, generator):
        """Test country validation."""
        assert generator._is_valid_country("DE") is True
        assert generator._is_valid_country("US") is True
        assert generator._is_valid_country("123") is False
        assert generator._is_valid_country("TOOLONG") is False


class TestROIDataGeneratorHelpers:
    """Tests for ROI data generator helper functions."""

    def test_create_register_of_information(self):
        """Test factory function."""
        generator = create_register_of_information()

        assert isinstance(generator, DORARegisterOfInformation)

    def test_create_register_of_information_with_config(self):
        """Test factory with config."""
        config = ROIDataGeneratorConfig(provider_lei="TEST")
        generator = create_register_of_information(config)

        assert generator.config.provider_lei == "TEST"

    def test_create_roi_data_generator(self):
        """Test convenience factory function."""
        generator = create_roi_data_generator(
            provider_lei="TEST",
            provider_name="Test Provider",
            provider_country="DE",
        )

        assert generator.config.provider_lei == "TEST"
        assert generator.config.provider_name == "Test Provider"
        assert generator.config.provider_country == "DE"

    def test_get_contract_types(self):
        """Test getting contract types."""
        types = get_contract_types()

        assert len(types) > 0
        assert ContractType.PROCUREMENT in types
        assert ContractType.OUTSOURCING in types

    def test_get_service_types(self):
        """Test getting service types."""
        types = get_service_types()

        assert len(types) > 0
        assert ServiceType.CLOUD_COMPUTING in types
        assert ServiceType.DATA_ANALYTICS in types

    def test_get_subcontracting_levels(self):
        """Test getting subcontracting levels."""
        levels = get_subcontracting_levels()

        assert len(levels) > 0
        assert SubcontractingLevel.LEVEL_1 in levels
        assert SubcontractingLevel.DIRECT in levels

    def test_get_its_templates_provided(self):
        """Test getting ITS templates we provide."""
        templates = get_its_templates_provided()

        assert "B_03_01" in templates
        assert "B_06_01" in templates

    def test_get_its_templates_client_provides(self):
        """Test getting ITS templates clients provide."""
        templates = get_its_templates_client_provides()

        assert "B_01_01" in templates
        assert "B_99_01" in templates


# =============================================================================
# Integration Tests
# =============================================================================


class TestReportingModuleIntegration:
    """Integration tests for the reporting module."""

    def test_module_imports(self):
        """Test all module imports work."""
        from services.dora_integration.reporting import (
            # Unified Reporting
            UnifiedReportingManager,
            ReportType,
            ReportStatus,
            # Templates
            DORAReportingTemplates,
            ITSInitialNotificationTemplate,
            # ROI
            DORARegisterOfInformation,
            ContractType,
            ServiceType,
        )

        assert UnifiedReportingManager is not None
        assert DORAReportingTemplates is not None
        assert DORARegisterOfInformation is not None

    def test_full_reporting_workflow(self):
        """Test complete reporting workflow."""
        from services.dora_integration.reporting import (
            UnifiedReportingManager,
            UnifiedReportingConfig,
            ReportType,
            ReportDestination,
            ReportChannel,
            ClientType,
            PackageFormat,
        )
        from datetime import datetime, timezone, timedelta

        # Setup manager
        config = UnifiedReportingConfig(
            provider_lei="549300WORKFLOW0001",
            provider_name="Workflow Test Provider",
        )
        manager = UnifiedReportingManager(config)

        # Create destination
        destination = ReportDestination(
            name="Integration Client",
            client_id="INT-001",
            client_type=ClientType.INVESTMENT_FIRM,
            channel=ReportChannel.API,
            endpoint="https://api.test.com",
            preferred_format=PackageFormat.JSON,
        )

        # Create report
        report = manager.create_report(
            report_type=ReportType.DORA_MAJOR_INCIDENT,
            content={
                "incident_id": "INT-INC-001",
                "classification": "major",
                "services_affected": ["trading"],
            },
            destination=destination,
            due_at=datetime.now(timezone.utc) + timedelta(hours=4),
        )

        # Mark ready
        assert manager.mark_ready(report.report_id) is True

        # Generate package
        package = manager.generate_submission_package("INT-001")
        assert package is not None
        assert len(package.reports) == 1

        # Mark delivered
        assert manager.mark_delivered(report.report_id, "ACK-INT") is True

        # Mark submitted
        assert manager.mark_submitted(report.report_id) is True

        # Check final state
        final_report = manager.get_report(report.report_id)
        assert final_report.status.value == "submitted"

    def test_full_roi_generation_workflow(self):
        """Test complete ROI generation workflow."""
        from services.dora_integration.reporting import (
            DORARegisterOfInformation,
            ROIDataGeneratorConfig,
            ContractType,
            ServiceType,
            SubcontractingLevel,
        )

        # Setup generator
        config = ROIDataGeneratorConfig(
            provider_lei="549300ROITEST0001",
            provider_name="ROI Test Provider",
            provider_country="DE",
        )
        generator = DORARegisterOfInformation(config)

        # Add contract
        contract = generator.add_contract(
            contract_type=ContractType.PROCUREMENT,
            service_types_provided=[ServiceType.CLOUD_COMPUTING.value],
            data_processing_countries=["DE", "NL"],
            subcontracting_permitted=True,
        )

        # Add services
        generator.add_service(
            contract_reference=contract.contract_reference,
            service_name="Cloud Hosting",
            service_type=ServiceType.CLOUD_COMPUTING,
            availability_target_pct=99.9,
            supports_trading_functions=True,
        )

        # Add subcontractor
        generator.add_subcontractor(
            parent_contract_reference=contract.contract_reference,
            legal_name="Subcontractor NL",
            country="NL",
            subcontracting_level=SubcontractingLevel.LEVEL_1,
            services_subcontracted=["Storage"],
        )

        # Generate package
        package = generator.generate_roi_data_package(
            reference_date="2025-03-31",
        )

        assert package.is_validated is True
        assert package.provider.lei == "549300ROITEST0001"
        assert len(package.contracts) == 1
        assert len(package.services) == 1
        assert len(package.subcontractors) == 1

        # Export to all formats
        json_data = generator.export_package_to_json(package)
        assert "package_id" in json_data

        csv_data = generator.export_package_to_csv(package)
        assert len(csv_data) == 4

        xml_data = generator.export_package_to_xml(package)
        assert "<?xml" in xml_data

    def test_incident_template_generation_workflow(self):
        """Test complete incident template workflow."""
        from services.dora_integration.reporting import (
            DORAReportingTemplates,
            IncidentTypeCode,
        )

        # Setup templates
        templates = DORAReportingTemplates(
            provider_lei="549300TMPLTEST001",
            provider_name="Template Test Provider",
            entity_lei="549300CLIENTLE001",
            entity_name="Test Client",
            entity_type="investment_firm",
            entity_country="DE",
        )

        # Create initial notification
        initial = templates.create_initial_notification(
            incident_reference="TMPL-INC-001",
            detection_datetime="2025-01-15T10:00:00Z",
            classification_datetime="2025-01-15T10:30:00Z",
            brief_description="Test incident for workflow",
            incident_type_code=IncidentTypeCode.SYSF.value,
            member_states_affected=["DE"],
            contact_person_email="test@example.com",
        )

        is_valid, _ = initial.validate()
        assert is_valid is True

        # Create intermediate report
        intermediate = templates.create_intermediate_report(
            initial_report_reference=initial.report_reference,
            incident_reference="TMPL-INC-001",
            detailed_description="Detailed analysis...",
            preliminary_root_cause="Hardware failure",
        )

        is_valid, _ = intermediate.validate()
        assert is_valid is True

        # Create final report
        final = templates.create_final_report(
            initial_report_reference=initial.report_reference,
            intermediate_report_reference=intermediate.report_reference,
            incident_reference="TMPL-INC-001",
            incident_title="System Failure",
            comprehensive_description="Complete analysis...",
            final_root_cause="Disk failure",
            lessons_learned=["Improve monitoring"],
            remediation_measures=[{"description": "Replace hardware"}],
        )

        is_valid, _ = final.validate()
        assert is_valid is True

        # Generate client data package
        package = templates.generate_client_data_package(
            incident_id="TMPL-INC-001",
            incident_data={
                "incident_id": "TMPL-INC-001",
                "detected_at": "2025-01-15T10:00:00Z",
                "classified_at": "2025-01-15T10:30:00Z",
                "description": "Test",
            },
        )

        assert package.initial_template is not None
        json_output = templates.export_package_to_json(package)
        assert "package_id" in json_output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
