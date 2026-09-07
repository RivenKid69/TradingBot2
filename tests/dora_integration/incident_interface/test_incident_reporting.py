# -*- coding: utf-8 -*-
"""
Tests for Incident Reporting Module (Article 19) - Export Templates.

Tests cover:
- Client data package generation (primary export function)
- Deadline calculation
- Report template generation
- Query and statistics
"""

import pytest
from datetime import datetime, timezone, timedelta

from services.dora_integration.incident_interface.incident_reporting import (
    DORAIncidentReporter,
    IncidentReportingConfig,
    ReportType,
    ReportStatus,
    IncidentTypeCode,
    RootCauseCategory,
    CompetentAuthorityType,
    CompetentAuthority,
    InitialNotificationReport,
    IntermediateReport,
    FinalReport,
    ClientDataPackage,
    ReportSubmission,
    create_incident_reporter,
    get_report_deadlines,
    get_report_types,
)


class TestDORAIncidentReporter:
    """Test suite for DORAIncidentReporter."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return IncidentReportingConfig(
            initial_notification_hours_from_classification=4,
            initial_notification_hours_from_detection=24,
            intermediate_report_hours=72,
            final_report_days=30,
            entity_lei="549300TEST000000",
            entity_name="Test ICT Provider",
            entity_type="ict_third_party_provider",
            log_all_reports=False,
        )

    @pytest.fixture
    def reporter(self, config):
        """Create reporter instance."""
        return DORAIncidentReporter(config)

    # =========================================================================
    # Client Data Package Tests (Primary Export Function)
    # =========================================================================

    def test_generate_client_data_package(self, reporter):
        """Test generating client data package."""
        package = reporter.generate_client_data_package(
            incident_id="INC-001",
            client_id="CLIENT-001",
            client_name="Test Client",
            incident_data={
                "title": "Service Disruption",
                "severity": "high",
            },
            timeline_events=[
                {"time": "2025-01-15T10:00:00Z", "event": "Detected"},
                {"time": "2025-01-15T11:00:00Z", "event": "Classified"},
            ],
            affected_services=["trading", "reporting"],
            root_cause_summary="Hardware failure",
            resolution_status="resolved",
            resolution_description="Hardware replaced",
            resolution_datetime="2025-01-15T14:00:00+00:00",  # Required for final_report
            provider_actions=[
                {"action": "Isolated affected system"},
                {"action": "Replaced hardware"},
            ],
            preventive_measures=["Redundancy added"],
            generated_by="ops_team",
        )

        assert package.package_id.startswith("PKG-")
        assert package.incident_id == "INC-001"
        assert package.client_id == "CLIENT-001"
        assert package.ict_provider_lei == "549300TEST000000"
        assert len(package.affected_services) == 2
        assert package.suggested_report_type == "final_report"

    def test_generate_client_data_package_ongoing(self, reporter):
        """Test package for ongoing incident suggests initial notification."""
        package = reporter.generate_client_data_package(
            incident_id="INC-002",
            client_id="CLIENT-002",
            resolution_status="ongoing",
        )

        assert package.suggested_report_type == "initial_notification"

    def test_export_client_package(self, reporter):
        """Test exporting client package."""
        package = reporter.generate_client_data_package(
            incident_id="INC-003",
            client_id="CLIENT-003",
        )

        export = reporter.export_client_package(package.package_id)

        assert export["export_type"] == "client_nca_data_package"
        assert "package" in export
        assert "usage_instructions" in export

    def test_get_packages_for_client(self, reporter):
        """Test getting packages for a client."""
        reporter.generate_client_data_package("INC-A", "CLIENT-X")
        reporter.generate_client_data_package("INC-B", "CLIENT-X")
        reporter.generate_client_data_package("INC-C", "CLIENT-Y")

        packages = reporter.get_packages_for_client("CLIENT-X")
        assert len(packages) == 2

    def test_get_packages_for_incident(self, reporter):
        """Test getting packages for an incident."""
        reporter.generate_client_data_package("INC-MULTI", "CLIENT-1")
        reporter.generate_client_data_package("INC-MULTI", "CLIENT-2")

        packages = reporter.get_packages_for_incident("INC-MULTI")
        assert len(packages) == 2

    # =========================================================================
    # Deadline Calculation Tests
    # =========================================================================

    def test_calculate_initial_notification_deadline(self, reporter):
        """Test initial notification deadline calculation."""
        detection = "2025-01-15T10:00:00+00:00"
        classification = "2025-01-15T11:00:00+00:00"

        deadline = reporter.calculate_initial_notification_deadline(detection, classification)

        # 4h from classification (11:00) = 15:00
        # 24h from detection (10:00) = next day 10:00
        # Whichever is earlier = 15:00 same day
        assert "2025-01-15T15:00:00" in deadline

    def test_calculate_intermediate_deadline(self, reporter):
        """Test intermediate report deadline calculation."""
        initial_submitted = "2025-01-15T15:00:00+00:00"

        deadline = reporter.calculate_intermediate_deadline(initial_submitted)

        # 72h from initial = 3 days later (2025-01-18)
        # But 2025-01-18 is Saturday, so weekend extension pushes to Monday noon
        # Expected: 2025-01-20T12:00:00 (Monday noon)
        assert "2025-01-20" in deadline

    def test_calculate_final_report_deadline_resolved(self, reporter):
        """Test final report deadline from resolution."""
        resolution = "2025-01-20T10:00:00+00:00"

        deadline = reporter.calculate_final_report_deadline(
            resolution_datetime=resolution,
        )

        # 30 days from resolution
        assert "2025-02-19" in deadline

    def test_weekend_extension(self, reporter):
        """Test weekend extension for deadlines."""
        # Saturday deadline
        detection = "2025-01-11T10:00:00+00:00"  # Saturday
        classification = "2025-01-11T11:00:00+00:00"

        deadline = reporter.calculate_initial_notification_deadline(detection, classification)

        # Should extend to Monday noon
        deadline_dt = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
        # weekday() == 0 is Monday
        assert deadline_dt.weekday() == 0 or deadline_dt.hour == 12

    # =========================================================================
    # Report Template Tests
    # =========================================================================

    def test_generate_initial_notification(self, reporter):
        """Test initial notification generation."""
        report = reporter.generate_initial_notification(
            incident_id="INC-INIT",
            detection_datetime="2025-01-15T10:00:00+00:00",
            classification_datetime="2025-01-15T11:00:00+00:00",
            brief_description="System failure affecting trading services",
            incident_type_code=IncidentTypeCode.SYSTEM_FAILURE,
            critical_services_affected=["trading"],
            estimated_clients_affected=100,
            member_states_affected=["DE", "FR"],
            contact_person_name="John Smith",
            contact_person_email="john@example.com",
        )

        assert report.report_id.startswith("RPT-INIT-")
        assert report.report_type == ReportType.INITIAL_NOTIFICATION
        assert report.incident_type_code == IncidentTypeCode.SYSTEM_FAILURE
        assert report.deadline is not None

    def test_generate_intermediate_report(self, reporter):
        """Test intermediate report generation."""
        # First create initial
        initial = reporter.generate_initial_notification(
            incident_id="INC-INT",
            detection_datetime="2025-01-15T10:00:00+00:00",
            classification_datetime="2025-01-15T11:00:00+00:00",
            brief_description="System failure",
        )
        initial.submitted_at = "2025-01-15T15:00:00+00:00"

        report = reporter.generate_intermediate_report(
            incident_id="INC-INT",
            initial_notification_id=initial.report_id,
            detailed_description="Detailed analysis of the system failure",
            preliminary_root_cause="Database corruption",
            root_cause_category=RootCauseCategory.SYSTEM_SOFTWARE,
            affected_ict_services=["trading", "reporting"],
            affected_clients_count=150,
            immediate_actions_taken=["Isolated system", "Started recovery"],
        )

        assert report.report_id.startswith("RPT-INT-")
        assert report.report_type == ReportType.INTERMEDIATE_REPORT
        assert report.initial_notification_id == initial.report_id

    def test_generate_final_report(self, reporter):
        """Test final report generation."""
        # Create initial and intermediate
        initial = reporter.generate_initial_notification(
            incident_id="INC-FIN",
            detection_datetime="2025-01-15T10:00:00+00:00",
            classification_datetime="2025-01-15T11:00:00+00:00",
            brief_description="System failure",
        )

        intermediate = reporter.generate_intermediate_report(
            incident_id="INC-FIN",
            initial_notification_id=initial.report_id,
            detailed_description="Detailed analysis",
        )
        intermediate.submitted_at = "2025-01-18T15:00:00+00:00"

        report = reporter.generate_final_report(
            incident_id="INC-FIN",
            initial_notification_id=initial.report_id,
            intermediate_report_id=intermediate.report_id,
            incident_resolved=True,
            resolution_datetime="2025-01-20T10:00:00+00:00",
            resolution_description="Issue fully resolved",
            comprehensive_description="Complete incident description",
            final_root_cause="Database corruption due to disk failure",
            lessons_learned=["Improve monitoring", "Add redundancy"],
            remediation_measures=[{"measure": "Added redundancy"}],
            preventive_measures=[{"measure": "Enhanced monitoring"}],
        )

        assert report.report_id.startswith("RPT-FIN-")
        assert report.report_type == ReportType.FINAL_REPORT
        assert report.incident_resolved is True

    # =========================================================================
    # Query Tests
    # =========================================================================

    def test_get_report(self, reporter):
        """Test getting a report."""
        initial = reporter.generate_initial_notification(
            incident_id="INC-QUERY",
            detection_datetime="2025-01-15T10:00:00+00:00",
            classification_datetime="2025-01-15T11:00:00+00:00",
            brief_description="Test",
        )

        retrieved = reporter.get_report(initial.report_id)
        assert retrieved is not None
        assert retrieved.incident_id == "INC-QUERY"

    def test_get_reports_for_incident(self, reporter):
        """Test getting all reports for an incident."""
        initial = reporter.generate_initial_notification(
            incident_id="INC-ALL",
            detection_datetime="2025-01-15T10:00:00+00:00",
            classification_datetime="2025-01-15T11:00:00+00:00",
            brief_description="Test",
        )

        reporter.generate_intermediate_report(
            incident_id="INC-ALL",
            initial_notification_id=initial.report_id,
            detailed_description="Details",
        )

        reports = reporter.get_reports_for_incident("INC-ALL")
        assert len(reports["initial"]) == 1
        assert len(reports["intermediate"]) == 1

    def test_get_pending_reports(self, reporter):
        """Test getting pending reports."""
        reporter.generate_initial_notification(
            incident_id="INC-PENDING",
            detection_datetime="2025-01-15T10:00:00+00:00",
            classification_datetime="2025-01-15T11:00:00+00:00",
            brief_description="Test",
        )

        pending = reporter.get_pending_reports()
        assert len(pending) >= 1

    def test_export_report(self, reporter):
        """Test exporting a report."""
        initial = reporter.generate_initial_notification(
            incident_id="INC-EXPORT",
            detection_datetime="2025-01-15T10:00:00+00:00",
            classification_datetime="2025-01-15T11:00:00+00:00",
            brief_description="Test",
        )

        export = reporter.export_report(initial.report_id)

        assert export["export_type"] == "report_template"
        assert export["article_reference"] == "Article 19"
        assert "note" in export

    def test_get_reporting_statistics(self, reporter):
        """Test reporting statistics."""
        reporter.generate_initial_notification(
            incident_id="INC-STATS",
            detection_datetime="2025-01-15T10:00:00+00:00",
            classification_datetime="2025-01-15T11:00:00+00:00",
            brief_description="Test",
        )

        reporter.generate_client_data_package("INC-STATS", "CLIENT-1")

        stats = reporter.get_reporting_statistics()

        assert "report_templates" in stats
        assert "client_data_packages" in stats

    # =========================================================================
    # Authority Management Tests
    # =========================================================================

    def test_register_authority(self, reporter):
        """Test registering competent authority."""
        authority = CompetentAuthority(
            name="BaFin",
            country_code="DE",
            authority_type=CompetentAuthorityType.NCA_PRIMARY,
        )

        registered = reporter.register_authority(authority)

        assert registered.name == "BaFin"
        assert registered.authority_type == CompetentAuthorityType.NCA_PRIMARY

    def test_get_authority(self, reporter):
        """Test getting authority."""
        authority = CompetentAuthority(
            name="AMF",
            country_code="FR",
        )
        reporter.register_authority(authority)

        retrieved = reporter.get_authority(authority.authority_id)
        assert retrieved is not None
        assert retrieved.name == "AMF"


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_incident_reporter(self):
        """Test reporter factory."""
        reporter = create_incident_reporter()
        assert isinstance(reporter, DORAIncidentReporter)

    def test_get_report_deadlines(self):
        """Test deadline info."""
        deadlines = get_report_deadlines()
        assert "initial_notification" in deadlines
        assert "intermediate_report" in deadlines
        assert "final_report" in deadlines

    def test_get_report_types(self):
        """Test report types list."""
        types = get_report_types()
        assert ReportType.INITIAL_NOTIFICATION in types
        assert ReportType.INTERMEDIATE_REPORT in types
        assert ReportType.FINAL_REPORT in types


class TestDataStructures:
    """Test data structures."""

    def test_initial_notification_auto_id(self):
        """Test InitialNotificationReport auto ID."""
        report = InitialNotificationReport()
        assert report.report_id.startswith("RPT-INIT-")
        assert report.created_at is not None

    def test_intermediate_report_auto_id(self):
        """Test IntermediateReport auto ID."""
        report = IntermediateReport()
        assert report.report_id.startswith("RPT-INT-")

    def test_final_report_auto_id(self):
        """Test FinalReport auto ID."""
        report = FinalReport()
        assert report.report_id.startswith("RPT-FIN-")

    def test_client_data_package_auto_id(self):
        """Test ClientDataPackage auto ID."""
        package = ClientDataPackage()
        assert package.package_id.startswith("PKG-")
        assert package.generated_at is not None

    def test_report_submission_auto_id(self):
        """Test ReportSubmission auto ID."""
        submission = ReportSubmission()
        assert submission.submission_id.startswith("SUB-")

    def test_competent_authority_auto_id(self):
        """Test CompetentAuthority auto ID."""
        authority = CompetentAuthority(country_code="DE")
        assert authority.authority_id.startswith("NCA-DE-")
