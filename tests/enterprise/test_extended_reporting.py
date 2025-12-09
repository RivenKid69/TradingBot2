# -*- coding: utf-8 -*-
"""
Tests for Extended Incident Reporting Service.

DORA Phase 3 Block 3.2: Extended incident report formats (PDF/JSON)
"""

import json
import pytest
from datetime import datetime, timedelta

from services.enterprise.extended_reporting import (
    ReportFormat,
    ReportTemplate,
    ReportSeverity,
    ReportStatus,
    DeliveryMethod,
    ReportMetadata,
    IncidentSummary,
    TechnicalDetails,
    ImpactAssessment,
    RemediationPlan,
    ExtendedIncidentReport,
    ReportDelivery,
    ReportingConfig,
    PDFReportGenerator,
    JSONReportGenerator,
    ExtendedReportingService,
    create_extended_reporting,
    generate_pdf_report,
    generate_json_report,
)


class TestReportEnums:
    """Tests for report enums."""

    def test_report_format_values(self) -> None:
        """Test ReportFormat enum values."""
        assert ReportFormat.PDF.value == "pdf"
        assert ReportFormat.JSON.value == "json"
        assert ReportFormat.HTML.value == "html"
        assert ReportFormat.XML.value == "xml"

    def test_report_template_values(self) -> None:
        """Test ReportTemplate enum values."""
        assert ReportTemplate.REGULATORY.value == "regulatory"
        assert ReportTemplate.EXECUTIVE.value == "executive"
        assert ReportTemplate.TECHNICAL.value == "technical"
        assert ReportTemplate.CLIENT.value == "client"
        assert ReportTemplate.INITIAL.value == "initial"
        assert ReportTemplate.INTERMEDIATE.value == "intermediate"
        assert ReportTemplate.FINAL.value == "final"

    def test_report_severity_values(self) -> None:
        """Test ReportSeverity enum values."""
        assert ReportSeverity.CRITICAL.value == "critical"
        assert ReportSeverity.HIGH.value == "high"
        assert ReportSeverity.MEDIUM.value == "medium"
        assert ReportSeverity.LOW.value == "low"
        assert ReportSeverity.INFORMATIONAL.value == "informational"

    def test_report_status_values(self) -> None:
        """Test ReportStatus enum values."""
        assert ReportStatus.DRAFT.value == "draft"
        assert ReportStatus.PENDING_REVIEW.value == "pending_review"
        assert ReportStatus.APPROVED.value == "approved"
        assert ReportStatus.SUBMITTED.value == "submitted"

    def test_delivery_method_values(self) -> None:
        """Test DeliveryMethod enum values."""
        assert DeliveryMethod.EMAIL.value == "email"
        assert DeliveryMethod.API.value == "api"
        assert DeliveryMethod.PORTAL.value == "portal"


class TestReportMetadata:
    """Tests for ReportMetadata dataclass."""

    def test_metadata_creation(self) -> None:
        """Test creating report metadata."""
        metadata = ReportMetadata(
            report_id="report-001",
            incident_id="INC-001",
            template=ReportTemplate.REGULATORY,
            format=ReportFormat.JSON,
            version="1.0.0",
            created_at=datetime.utcnow(),
            created_by="test-user",
            entity_lei="549300EXAMPLE",
            entity_name="Test Entity",
            reporting_nca="BaFin",
        )
        assert metadata.report_id == "report-001"
        assert metadata.incident_id == "INC-001"
        assert metadata.template == ReportTemplate.REGULATORY
        assert "INC-001" in metadata.reference_number

    def test_metadata_auto_reference_number(self) -> None:
        """Test auto-generated reference number."""
        metadata = ReportMetadata(
            report_id="report-001",
            incident_id="INC-12345678",
            template=ReportTemplate.INITIAL,
            format=ReportFormat.PDF,
            version="1.0.0",
            created_at=datetime.utcnow(),
            created_by="test-user",
            entity_lei="549300EXAMPLE",
            entity_name="Test Entity",
            reporting_nca="BaFin",
        )
        # Reference number format: INC-{incident_id[:8]}-{version}
        assert metadata.reference_number == "INC-INC-1234-1.0.0"


class TestIncidentSummary:
    """Tests for IncidentSummary dataclass."""

    def test_incident_summary_creation(self) -> None:
        """Test creating incident summary."""
        detection = datetime.utcnow() - timedelta(hours=5)
        resolution = datetime.utcnow()

        summary = IncidentSummary(
            incident_id="INC-001",
            title="Database Outage",
            description="Primary database server became unresponsive",
            incident_type="SYSTEM_FAILURE",
            detection_time=detection,
            classification_time=detection + timedelta(minutes=15),
            resolution_time=resolution,
            severity=ReportSeverity.HIGH,
            is_major_incident=True,
            affected_services=["API", "Web App"],
            affected_clients=100,
            affected_transactions=5000,
            geographic_scope=["DE", "FR"],
            root_cause_category="HARDWARE_FAILURE",
        )
        assert summary.incident_id == "INC-001"
        assert summary.is_major_incident is True
        assert len(summary.affected_services) == 2

    def test_duration_hours_calculation(self) -> None:
        """Test incident duration calculation."""
        detection = datetime.utcnow() - timedelta(hours=3)
        resolution = datetime.utcnow()

        summary = IncidentSummary(
            incident_id="INC-001",
            title="Test Incident",
            description="Test",
            incident_type="TEST",
            detection_time=detection,
            classification_time=detection,
            resolution_time=resolution,
            severity=ReportSeverity.MEDIUM,
            is_major_incident=False,
            affected_services=[],
            affected_clients=0,
            affected_transactions=0,
            geographic_scope=[],
            root_cause_category="TEST",
        )
        assert summary.duration_hours is not None
        assert 2.9 <= summary.duration_hours <= 3.1

    def test_duration_hours_unresolved(self) -> None:
        """Test duration is None for unresolved incidents."""
        summary = IncidentSummary(
            incident_id="INC-001",
            title="Test",
            description="Test",
            incident_type="TEST",
            detection_time=datetime.utcnow(),
            classification_time=datetime.utcnow(),
            resolution_time=None,
            severity=ReportSeverity.LOW,
            is_major_incident=False,
            affected_services=[],
            affected_clients=0,
            affected_transactions=0,
            geographic_scope=[],
            root_cause_category="TEST",
        )
        assert summary.duration_hours is None


class TestImpactAssessment:
    """Tests for ImpactAssessment dataclass."""

    def test_impact_assessment_creation(self) -> None:
        """Test creating impact assessment."""
        impact = ImpactAssessment(
            clients_affected_count=150,
            clients_affected_percentage=5.5,
            client_types_affected=["retail", "corporate"],
            transactions_affected=10000,
            transaction_value_affected=1500000.0,
            service_downtime_hours=2.5,
            degraded_service_hours=1.0,
            member_states_affected=["DE", "FR", "NL"],
            direct_costs=50000.0,
            indirect_costs=25000.0,
            recovery_costs=10000.0,
            data_breach=False,
            data_types_affected=[],
            records_affected=0,
            critical_services_affected=["trading"],
            media_coverage=False,
            regulatory_inquiries=0,
        )
        assert impact.clients_affected_count == 150
        assert len(impact.member_states_affected) == 3

    def test_total_economic_impact(self) -> None:
        """Test total economic impact calculation."""
        impact = ImpactAssessment(
            clients_affected_count=0,
            clients_affected_percentage=0.0,
            client_types_affected=[],
            transactions_affected=0,
            transaction_value_affected=0.0,
            service_downtime_hours=0.0,
            degraded_service_hours=0.0,
            member_states_affected=[],
            direct_costs=50000.0,
            indirect_costs=25000.0,
            recovery_costs=10000.0,
            data_breach=False,
            data_types_affected=[],
            records_affected=0,
            critical_services_affected=[],
            media_coverage=False,
            regulatory_inquiries=0,
        )
        assert impact.total_economic_impact == 85000.0


class TestExtendedIncidentReport:
    """Tests for ExtendedIncidentReport dataclass."""

    @pytest.fixture
    def sample_report(self) -> ExtendedIncidentReport:
        """Create sample report for testing."""
        metadata = ReportMetadata(
            report_id="report-001",
            incident_id="INC-001",
            template=ReportTemplate.REGULATORY,
            format=ReportFormat.JSON,
            version="1.0.0",
            created_at=datetime.utcnow(),
            created_by="test-user",
            entity_lei="549300EXAMPLE",
            entity_name="Test Entity",
            reporting_nca="BaFin",
        )

        summary = IncidentSummary(
            incident_id="INC-001",
            title="Test Incident",
            description="Test",
            incident_type="TEST",
            detection_time=datetime.utcnow(),
            classification_time=datetime.utcnow(),
            resolution_time=None,
            severity=ReportSeverity.MEDIUM,
            is_major_incident=False,
            affected_services=[],
            affected_clients=0,
            affected_transactions=0,
            geographic_scope=[],
            root_cause_category="TEST",
        )

        impact = ImpactAssessment(
            clients_affected_count=0,
            clients_affected_percentage=0.0,
            client_types_affected=[],
            transactions_affected=0,
            transaction_value_affected=0.0,
            service_downtime_hours=0.0,
            degraded_service_hours=0.0,
            member_states_affected=[],
            direct_costs=0.0,
            indirect_costs=0.0,
            recovery_costs=0.0,
            data_breach=False,
            data_types_affected=[],
            records_affected=0,
            critical_services_affected=[],
            media_coverage=False,
            regulatory_inquiries=0,
        )

        remediation = RemediationPlan(
            immediate_actions=[],
            short_term_actions=[],
            long_term_actions=[],
            lessons_learned=[],
            control_improvements=[],
            policy_updates=[],
            training_requirements=[],
            third_party_notifications=[],
            estimated_completion=datetime.utcnow() + timedelta(days=30),
            responsible_parties=["IT Team"],
        )

        return ExtendedIncidentReport(
            metadata=metadata,
            summary=summary,
            technical_details=None,
            impact_assessment=impact,
            remediation_plan=remediation,
        )

    def test_report_creation(self, sample_report: ExtendedIncidentReport) -> None:
        """Test report creation."""
        assert sample_report.status == ReportStatus.DRAFT
        assert len(sample_report.approvals) == 0

    def test_add_approval(self, sample_report: ExtendedIncidentReport) -> None:
        """Test adding approval."""
        sample_report.add_approval(
            approver="manager@example.com",
            role="Security Manager",
            approved=True,
            comments="Looks good",
        )
        assert len(sample_report.approvals) == 1
        assert sample_report.approvals[0]["approved"] is True
        assert sample_report.status == ReportStatus.APPROVED

    def test_add_revision(self, sample_report: ExtendedIncidentReport) -> None:
        """Test adding revision."""
        original_version = sample_report.metadata.version
        sample_report.add_revision("editor", "Fixed typo")
        assert sample_report.metadata.version != original_version
        assert len(sample_report.revision_history) == 1

    def test_calculate_checksum(self, sample_report: ExtendedIncidentReport) -> None:
        """Test checksum calculation."""
        checksum = sample_report.calculate_checksum()
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA-256 hex length


class TestReportDelivery:
    """Tests for ReportDelivery dataclass."""

    def test_delivery_creation(self) -> None:
        """Test creating delivery record."""
        delivery = ReportDelivery(
            delivery_id="del-001",
            report_id="report-001",
            method=DeliveryMethod.EMAIL,
            recipient="nca@bafin.de",
            recipient_type="NCA",
        )
        assert delivery.delivery_status == "pending"
        assert delivery.sent_at is None

    def test_mark_sent(self) -> None:
        """Test marking delivery as sent."""
        delivery = ReportDelivery(
            delivery_id="del-001",
            report_id="report-001",
            method=DeliveryMethod.API,
            recipient="api.endpoint",
            recipient_type="NCA",
        )
        delivery.mark_sent()
        assert delivery.delivery_status == "sent"
        assert delivery.sent_at is not None

    def test_mark_acknowledged(self) -> None:
        """Test marking delivery as acknowledged."""
        delivery = ReportDelivery(
            delivery_id="del-001",
            report_id="report-001",
            method=DeliveryMethod.API,
            recipient="api.endpoint",
            recipient_type="NCA",
        )
        delivery.mark_acknowledged()
        assert delivery.delivery_status == "acknowledged"
        assert delivery.acknowledged_at is not None

    def test_mark_failed(self) -> None:
        """Test marking delivery as failed."""
        delivery = ReportDelivery(
            delivery_id="del-001",
            report_id="report-001",
            method=DeliveryMethod.API,
            recipient="api.endpoint",
            recipient_type="NCA",
        )
        delivery.mark_failed("Connection timeout")
        assert delivery.delivery_status == "pending_retry"
        assert delivery.retry_count == 1


class TestPDFReportGenerator:
    """Tests for PDF report generator."""

    def test_pdf_generation(self) -> None:
        """Test PDF report generation."""
        config = ReportingConfig(
            entity_lei="549300EXAMPLE",
            entity_name="Test Entity",
            default_nca="BaFin",
        )
        generator = PDFReportGenerator(config)

        metadata = ReportMetadata(
            report_id="report-001",
            incident_id="INC-001",
            template=ReportTemplate.REGULATORY,
            format=ReportFormat.PDF,
            version="1.0.0",
            created_at=datetime.utcnow(),
            created_by="test-user",
            entity_lei="549300EXAMPLE",
            entity_name="Test Entity",
            reporting_nca="BaFin",
        )

        summary = IncidentSummary(
            incident_id="INC-001",
            title="Test",
            description="Test",
            incident_type="TEST",
            detection_time=datetime.utcnow(),
            classification_time=datetime.utcnow(),
            resolution_time=None,
            severity=ReportSeverity.MEDIUM,
            is_major_incident=False,
            affected_services=[],
            affected_clients=0,
            affected_transactions=0,
            geographic_scope=[],
            root_cause_category="TEST",
        )

        impact = ImpactAssessment(
            clients_affected_count=0,
            clients_affected_percentage=0.0,
            client_types_affected=[],
            transactions_affected=0,
            transaction_value_affected=0.0,
            service_downtime_hours=0.0,
            degraded_service_hours=0.0,
            member_states_affected=[],
            direct_costs=0.0,
            indirect_costs=0.0,
            recovery_costs=0.0,
            data_breach=False,
            data_types_affected=[],
            records_affected=0,
            critical_services_affected=[],
            media_coverage=False,
            regulatory_inquiries=0,
        )

        remediation = RemediationPlan(
            immediate_actions=[],
            short_term_actions=[],
            long_term_actions=[],
            lessons_learned=[],
            control_improvements=[],
            policy_updates=[],
            training_requirements=[],
            third_party_notifications=[],
            estimated_completion=datetime.utcnow() + timedelta(days=30),
            responsible_parties=[],
        )

        report = ExtendedIncidentReport(
            metadata=metadata,
            summary=summary,
            technical_details=None,
            impact_assessment=impact,
            remediation_plan=remediation,
        )

        pdf_bytes = generator.generate(report)
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF")


class TestJSONReportGenerator:
    """Tests for JSON report generator."""

    def test_json_generation(self) -> None:
        """Test JSON report generation."""
        config = ReportingConfig(
            entity_lei="549300EXAMPLE",
            entity_name="Test Entity",
            default_nca="BaFin",
        )
        generator = JSONReportGenerator(config)

        metadata = ReportMetadata(
            report_id="report-001",
            incident_id="INC-001",
            template=ReportTemplate.REGULATORY,
            format=ReportFormat.JSON,
            version="1.0.0",
            created_at=datetime.utcnow(),
            created_by="test-user",
            entity_lei="549300EXAMPLE",
            entity_name="Test Entity",
            reporting_nca="BaFin",
        )

        summary = IncidentSummary(
            incident_id="INC-001",
            title="Test",
            description="Test",
            incident_type="TEST",
            detection_time=datetime.utcnow(),
            classification_time=datetime.utcnow(),
            resolution_time=None,
            severity=ReportSeverity.MEDIUM,
            is_major_incident=False,
            affected_services=[],
            affected_clients=0,
            affected_transactions=0,
            geographic_scope=[],
            root_cause_category="TEST",
        )

        impact = ImpactAssessment(
            clients_affected_count=0,
            clients_affected_percentage=0.0,
            client_types_affected=[],
            transactions_affected=0,
            transaction_value_affected=0.0,
            service_downtime_hours=0.0,
            degraded_service_hours=0.0,
            member_states_affected=[],
            direct_costs=0.0,
            indirect_costs=0.0,
            recovery_costs=0.0,
            data_breach=False,
            data_types_affected=[],
            records_affected=0,
            critical_services_affected=[],
            media_coverage=False,
            regulatory_inquiries=0,
        )

        remediation = RemediationPlan(
            immediate_actions=[],
            short_term_actions=[],
            long_term_actions=[],
            lessons_learned=[],
            control_improvements=[],
            policy_updates=[],
            training_requirements=[],
            third_party_notifications=[],
            estimated_completion=datetime.utcnow() + timedelta(days=30),
            responsible_parties=[],
        )

        report = ExtendedIncidentReport(
            metadata=metadata,
            summary=summary,
            technical_details=None,
            impact_assessment=impact,
            remediation_plan=remediation,
        )

        json_str = generator.generate(report)
        assert isinstance(json_str, str)

        parsed = json.loads(json_str)
        assert "$schema" in parsed
        assert "metadata" in parsed
        assert "incident" in parsed


class TestExtendedReportingService:
    """Tests for ExtendedReportingService."""

    @pytest.fixture
    def service(self) -> ExtendedReportingService:
        """Create service instance."""
        return create_extended_reporting(
            entity_lei="549300EXAMPLE",
            entity_name="Test Entity",
            default_nca="BaFin",
        )

    def test_service_creation(self, service: ExtendedReportingService) -> None:
        """Test service creation."""
        assert service.config.entity_lei == "549300EXAMPLE"
        assert service.config.entity_name == "Test Entity"

    def test_create_report(self, service: ExtendedReportingService) -> None:
        """Test creating a report."""
        summary = IncidentSummary(
            incident_id="INC-001",
            title="Test",
            description="Test",
            incident_type="TEST",
            detection_time=datetime.utcnow(),
            classification_time=datetime.utcnow(),
            resolution_time=None,
            severity=ReportSeverity.MEDIUM,
            is_major_incident=False,
            affected_services=[],
            affected_clients=0,
            affected_transactions=0,
            geographic_scope=[],
            root_cause_category="TEST",
        )

        impact = ImpactAssessment(
            clients_affected_count=0,
            clients_affected_percentage=0.0,
            client_types_affected=[],
            transactions_affected=0,
            transaction_value_affected=0.0,
            service_downtime_hours=0.0,
            degraded_service_hours=0.0,
            member_states_affected=[],
            direct_costs=0.0,
            indirect_costs=0.0,
            recovery_costs=0.0,
            data_breach=False,
            data_types_affected=[],
            records_affected=0,
            critical_services_affected=[],
            media_coverage=False,
            regulatory_inquiries=0,
        )

        remediation = RemediationPlan(
            immediate_actions=[],
            short_term_actions=[],
            long_term_actions=[],
            lessons_learned=[],
            control_improvements=[],
            policy_updates=[],
            training_requirements=[],
            third_party_notifications=[],
            estimated_completion=datetime.utcnow() + timedelta(days=30),
            responsible_parties=[],
        )

        report = service.create_report(
            incident_id="INC-001",
            template=ReportTemplate.REGULATORY,
            summary=summary,
            impact_assessment=impact,
            remediation_plan=remediation,
        )

        assert report is not None
        assert report.metadata.incident_id == "INC-001"

    def test_get_report(self, service: ExtendedReportingService) -> None:
        """Test getting a report."""
        summary = IncidentSummary(
            incident_id="INC-001",
            title="Test",
            description="Test",
            incident_type="TEST",
            detection_time=datetime.utcnow(),
            classification_time=datetime.utcnow(),
            resolution_time=None,
            severity=ReportSeverity.MEDIUM,
            is_major_incident=False,
            affected_services=[],
            affected_clients=0,
            affected_transactions=0,
            geographic_scope=[],
            root_cause_category="TEST",
        )

        impact = ImpactAssessment(
            clients_affected_count=0,
            clients_affected_percentage=0.0,
            client_types_affected=[],
            transactions_affected=0,
            transaction_value_affected=0.0,
            service_downtime_hours=0.0,
            degraded_service_hours=0.0,
            member_states_affected=[],
            direct_costs=0.0,
            indirect_costs=0.0,
            recovery_costs=0.0,
            data_breach=False,
            data_types_affected=[],
            records_affected=0,
            critical_services_affected=[],
            media_coverage=False,
            regulatory_inquiries=0,
        )

        remediation = RemediationPlan(
            immediate_actions=[],
            short_term_actions=[],
            long_term_actions=[],
            lessons_learned=[],
            control_improvements=[],
            policy_updates=[],
            training_requirements=[],
            third_party_notifications=[],
            estimated_completion=datetime.utcnow() + timedelta(days=30),
            responsible_parties=[],
        )

        report = service.create_report(
            incident_id="INC-001",
            template=ReportTemplate.REGULATORY,
            summary=summary,
            impact_assessment=impact,
            remediation_plan=remediation,
        )

        retrieved = service.get_report(report.metadata.report_id)
        assert retrieved is not None
        assert retrieved.metadata.report_id == report.metadata.report_id

    def test_create_initial_notification(self, service: ExtendedReportingService) -> None:
        """Test creating initial notification."""
        report = service.create_initial_notification(
            incident_id="INC-001",
            title="Critical System Outage",
            description="Primary system became unresponsive",
            incident_type="SYSTEM_FAILURE",
            detection_time=datetime.utcnow(),
            severity=ReportSeverity.CRITICAL,
            affected_services=["API", "Web"],
        )

        assert report is not None
        assert report.metadata.template == ReportTemplate.INITIAL
        assert report.summary.is_major_incident is True

    def test_reporting_deadlines(self, service: ExtendedReportingService) -> None:
        """Test reporting deadline calculation."""
        report = service.create_initial_notification(
            incident_id="INC-001",
            title="Test",
            description="Test",
            incident_type="TEST",
            detection_time=datetime.utcnow(),
            severity=ReportSeverity.HIGH,
            affected_services=["API"],
        )

        deadlines = service.get_reporting_deadlines("INC-001")
        assert "initial_notification" in deadlines
        assert "intermediate_report" in deadlines
        assert "final_report" in deadlines

    def test_generate_pdf(self, service: ExtendedReportingService) -> None:
        """Test PDF generation via service."""
        summary = IncidentSummary(
            incident_id="INC-001",
            title="Test",
            description="Test",
            incident_type="TEST",
            detection_time=datetime.utcnow(),
            classification_time=datetime.utcnow(),
            resolution_time=None,
            severity=ReportSeverity.MEDIUM,
            is_major_incident=False,
            affected_services=[],
            affected_clients=0,
            affected_transactions=0,
            geographic_scope=[],
            root_cause_category="TEST",
        )

        impact = ImpactAssessment(
            clients_affected_count=0,
            clients_affected_percentage=0.0,
            client_types_affected=[],
            transactions_affected=0,
            transaction_value_affected=0.0,
            service_downtime_hours=0.0,
            degraded_service_hours=0.0,
            member_states_affected=[],
            direct_costs=0.0,
            indirect_costs=0.0,
            recovery_costs=0.0,
            data_breach=False,
            data_types_affected=[],
            records_affected=0,
            critical_services_affected=[],
            media_coverage=False,
            regulatory_inquiries=0,
        )

        remediation = RemediationPlan(
            immediate_actions=[],
            short_term_actions=[],
            long_term_actions=[],
            lessons_learned=[],
            control_improvements=[],
            policy_updates=[],
            training_requirements=[],
            third_party_notifications=[],
            estimated_completion=datetime.utcnow() + timedelta(days=30),
            responsible_parties=[],
        )

        report = service.create_report(
            incident_id="INC-001",
            template=ReportTemplate.REGULATORY,
            summary=summary,
            impact_assessment=impact,
            remediation_plan=remediation,
        )

        pdf_bytes = service.generate_pdf(report.metadata.report_id)
        assert isinstance(pdf_bytes, bytes)

    def test_generate_json(self, service: ExtendedReportingService) -> None:
        """Test JSON generation via service."""
        summary = IncidentSummary(
            incident_id="INC-001",
            title="Test",
            description="Test",
            incident_type="TEST",
            detection_time=datetime.utcnow(),
            classification_time=datetime.utcnow(),
            resolution_time=None,
            severity=ReportSeverity.MEDIUM,
            is_major_incident=False,
            affected_services=[],
            affected_clients=0,
            affected_transactions=0,
            geographic_scope=[],
            root_cause_category="TEST",
        )

        impact = ImpactAssessment(
            clients_affected_count=0,
            clients_affected_percentage=0.0,
            client_types_affected=[],
            transactions_affected=0,
            transaction_value_affected=0.0,
            service_downtime_hours=0.0,
            degraded_service_hours=0.0,
            member_states_affected=[],
            direct_costs=0.0,
            indirect_costs=0.0,
            recovery_costs=0.0,
            data_breach=False,
            data_types_affected=[],
            records_affected=0,
            critical_services_affected=[],
            media_coverage=False,
            regulatory_inquiries=0,
        )

        remediation = RemediationPlan(
            immediate_actions=[],
            short_term_actions=[],
            long_term_actions=[],
            lessons_learned=[],
            control_improvements=[],
            policy_updates=[],
            training_requirements=[],
            third_party_notifications=[],
            estimated_completion=datetime.utcnow() + timedelta(days=30),
            responsible_parties=[],
        )

        report = service.create_report(
            incident_id="INC-001",
            template=ReportTemplate.REGULATORY,
            summary=summary,
            impact_assessment=impact,
            remediation_plan=remediation,
        )

        json_str = service.generate_json(report.metadata.report_id)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert "metadata" in parsed


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_extended_reporting(self) -> None:
        """Test create_extended_reporting factory."""
        service = create_extended_reporting(
            entity_lei="549300TEST",
            entity_name="Test Company",
            default_nca="BaFin",
        )
        assert isinstance(service, ExtendedReportingService)

    def test_generate_pdf_report_function(self) -> None:
        """Test generate_pdf_report convenience function."""
        service = create_extended_reporting(
            entity_lei="549300TEST",
            entity_name="Test Company",
        )

        report = service.create_initial_notification(
            incident_id="INC-001",
            title="Test",
            description="Test",
            incident_type="TEST",
            detection_time=datetime.utcnow(),
            severity=ReportSeverity.MEDIUM,
            affected_services=[],
        )

        pdf = generate_pdf_report(service, report.metadata.report_id)
        assert isinstance(pdf, bytes)

    def test_generate_json_report_function(self) -> None:
        """Test generate_json_report convenience function."""
        service = create_extended_reporting(
            entity_lei="549300TEST",
            entity_name="Test Company",
        )

        report = service.create_initial_notification(
            incident_id="INC-001",
            title="Test",
            description="Test",
            incident_type="TEST",
            detection_time=datetime.utcnow(),
            severity=ReportSeverity.MEDIUM,
            affected_services=[],
        )

        json_str = generate_json_report(service, report.metadata.report_id)
        assert isinstance(json_str, str)
