# -*- coding: utf-8 -*-
"""
Extended Incident Reporting Service for Enterprise Clients.

DORA Phase 3 Block 3.2: Extended incident report formats (PDF/JSON)

Provides enterprise-grade incident reporting capabilities per DORA Art. 19-20:
- PDF report generation for formal submissions
- JSON report generation for API integrations
- Multiple report templates (regulatory, executive, technical)
- Automated delivery mechanisms

DORA References:
    - Art. 19: Major ICT-related incident reporting
    - Art. 20: Harmonised reporting content and templates
    - CDR 2024/1772: Incident classification criteria
    - CDR 2025/301: Incident reporting technical standards

Report Templates:
    - REGULATORY: DORA-aligned report package to support NCA submission
    - EXECUTIVE: High-level summary for management
    - TECHNICAL: Detailed technical analysis for IT teams
    - CLIENT: Client-facing incident notification
"""

from __future__ import annotations

import json
import hashlib
import base64
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4


# =============================================================================
# Enums
# =============================================================================


class ReportFormat(Enum):
    """Supported report output formats."""

    PDF = "pdf"
    JSON = "json"
    HTML = "html"
    XML = "xml"


class ReportTemplate(Enum):
    """Report template types per DORA requirements."""

    REGULATORY = "regulatory"  # DORA-aligned package to support NCA submission
    EXECUTIVE = "executive"  # Management summary
    TECHNICAL = "technical"  # IT team detailed analysis
    CLIENT = "client"  # Client-facing notification
    INITIAL = "initial"  # Initial notification (4 hours)
    INTERMEDIATE = "intermediate"  # Intermediate report (72 hours)
    FINAL = "final"  # Final report (1 month)


class ReportSeverity(Enum):
    """Incident severity levels per CDR 2024/1772."""

    CRITICAL = "critical"  # Major incident - immediate reporting
    HIGH = "high"  # Significant impact
    MEDIUM = "medium"  # Moderate impact
    LOW = "low"  # Minor impact
    INFORMATIONAL = "informational"  # For awareness only


class ReportStatus(Enum):
    """Report lifecycle status."""

    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    ARCHIVED = "archived"


class DeliveryMethod(Enum):
    """Report delivery methods."""

    EMAIL = "email"
    API = "api"
    PORTAL = "portal"
    SECURE_TRANSFER = "secure_transfer"
    MANUAL = "manual"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class ReportMetadata:
    """Report metadata per CDR 2025/301."""

    report_id: str
    incident_id: str
    template: ReportTemplate
    format: ReportFormat
    version: str
    created_at: datetime
    created_by: str
    entity_lei: str  # Legal Entity Identifier
    entity_name: str
    reporting_nca: str  # National Competent Authority
    confidentiality_level: str = "CONFIDENTIAL"
    language: str = "EN"
    reference_number: str = ""

    def __post_init__(self) -> None:
        if not self.reference_number:
            self.reference_number = f"INC-{self.incident_id[:8]}-{self.version}"


@dataclass
class IncidentSummary:
    """Incident summary section per CDR 2024/1772."""

    incident_id: str
    title: str
    description: str
    incident_type: str  # Per Art. 18 classification
    detection_time: datetime
    classification_time: datetime
    resolution_time: datetime | None
    severity: ReportSeverity
    is_major_incident: bool
    affected_services: list[str]
    affected_clients: int
    affected_transactions: int
    geographic_scope: list[str]
    root_cause_category: str
    attack_vector: str | None = None

    @property
    def duration_hours(self) -> float | None:
        """Calculate incident duration in hours."""
        if self.resolution_time:
            delta = self.resolution_time - self.detection_time
            return delta.total_seconds() / 3600
        return None


@dataclass
class TechnicalDetails:
    """Technical details section for IT teams."""

    affected_systems: list[str]
    affected_components: list[str]
    error_codes: list[str]
    log_references: list[str]
    network_indicators: list[str]
    malware_indicators: list[str]
    vulnerability_ids: list[str]  # CVE IDs if applicable
    attack_techniques: list[str]  # MITRE ATT&CK if applicable
    containment_actions: list[str]
    eradication_actions: list[str]
    recovery_actions: list[str]
    forensic_artifacts: list[str] = field(default_factory=list)
    ioc_list: list[dict[str, Any]] = field(default_factory=list)  # Indicators of Compromise


@dataclass
class ImpactAssessment:
    """Impact assessment per CDR 2024/1772 criteria."""

    # Client impact (Art. 18(1)(a))
    clients_affected_count: int
    clients_affected_percentage: float
    client_types_affected: list[str]

    # Transaction impact (Art. 18(1)(b))
    transactions_affected: int
    transaction_value_affected: float

    # Duration impact (Art. 18(1)(c))
    service_downtime_hours: float
    degraded_service_hours: float

    # Geographic impact (Art. 18(1)(d))
    member_states_affected: list[str]

    # Economic impact (Art. 18(1)(e))
    direct_costs: float
    indirect_costs: float
    recovery_costs: float

    # Data impact (Art. 18(1)(f))
    data_breach: bool
    data_types_affected: list[str]
    records_affected: int

    # Critical services impact (Art. 18(1)(g))
    critical_services_affected: list[str]

    # Reputational impact
    media_coverage: bool
    regulatory_inquiries: int

    # Currency (with default)
    currency: str = "EUR"

    @property
    def total_economic_impact(self) -> float:
        """Calculate total economic impact."""
        return self.direct_costs + self.indirect_costs + self.recovery_costs


@dataclass
class RemediationPlan:
    """Remediation plan section."""

    immediate_actions: list[dict[str, Any]]
    short_term_actions: list[dict[str, Any]]
    long_term_actions: list[dict[str, Any]]
    lessons_learned: list[str]
    control_improvements: list[str]
    policy_updates: list[str]
    training_requirements: list[str]
    third_party_notifications: list[str]
    estimated_completion: datetime
    responsible_parties: list[str]
    budget_allocated: float = 0.0
    currency: str = "EUR"


@dataclass
class ExtendedIncidentReport:
    """Complete extended incident report."""

    metadata: ReportMetadata
    summary: IncidentSummary
    technical_details: TechnicalDetails | None
    impact_assessment: ImpactAssessment
    remediation_plan: RemediationPlan
    status: ReportStatus = ReportStatus.DRAFT
    approvals: list[dict[str, Any]] = field(default_factory=list)
    attachments: list[dict[str, Any]] = field(default_factory=list)
    revision_history: list[dict[str, Any]] = field(default_factory=list)

    def add_approval(self, approver: str, role: str, approved: bool, comments: str = "") -> None:
        """Add approval record."""
        self.approvals.append(
            {
                "approver": approver,
                "role": role,
                "approved": approved,
                "comments": comments,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        if approved and all(a["approved"] for a in self.approvals):
            self.status = ReportStatus.APPROVED

    def add_revision(self, author: str, changes: str) -> None:
        """Add revision history entry."""
        version_parts = self.metadata.version.split(".")
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        self.metadata.version = ".".join(version_parts)
        self.revision_history.append(
            {
                "version": self.metadata.version,
                "author": author,
                "changes": changes,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    def calculate_checksum(self) -> str:
        """Calculate report checksum for integrity verification."""
        content = json.dumps(
            {
                "incident_id": self.summary.incident_id,
                "version": self.metadata.version,
                "created_at": self.metadata.created_at.isoformat(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class ReportDelivery:
    """Report delivery record."""

    delivery_id: str
    report_id: str
    method: DeliveryMethod
    recipient: str
    recipient_type: str  # NCA, CLIENT, INTERNAL
    sent_at: datetime | None = None
    acknowledged_at: datetime | None = None
    delivery_status: str = "pending"
    error_message: str | None = None
    retry_count: int = 0
    max_retries: int = 3

    def mark_sent(self) -> None:
        """Mark delivery as sent."""
        self.sent_at = datetime.utcnow()
        self.delivery_status = "sent"

    def mark_acknowledged(self) -> None:
        """Mark delivery as acknowledged."""
        self.acknowledged_at = datetime.utcnow()
        self.delivery_status = "acknowledged"

    def mark_failed(self, error: str) -> None:
        """Mark delivery as failed."""
        self.error_message = error
        self.retry_count += 1
        if self.retry_count >= self.max_retries:
            self.delivery_status = "failed"
        else:
            self.delivery_status = "pending_retry"


@dataclass
class ReportingConfig:
    """Extended reporting service configuration."""

    entity_lei: str
    entity_name: str
    default_nca: str
    pdf_template_path: str = "templates/reports/"
    json_schema_version: str = "1.0"
    auto_submit_major_incidents: bool = False
    require_dual_approval: bool = True
    retention_period_days: int = 2555  # 7 years per DORA
    encryption_enabled: bool = True


# =============================================================================
# PDF Report Generator
# =============================================================================


class PDFReportGenerator:
    """Generate PDF reports for incident reporting."""

    def __init__(self, config: ReportingConfig) -> None:
        """Initialize PDF generator."""
        self.config = config

    def generate(self, report: ExtendedIncidentReport) -> bytes:
        """
        Generate PDF report.

        In production, this would use a library like ReportLab, WeasyPrint,
        or similar. For now, we generate a structured representation.
        """
        # Build PDF content structure
        content = self._build_pdf_structure(report)

        # Encode as bytes (simulating PDF output)
        pdf_content = json.dumps(content, indent=2, default=str).encode("utf-8")

        # Add PDF header marker for identification
        header = b"%PDF-1.7\n% Extended Incident Report\n"
        return header + base64.b64encode(pdf_content)

    def _build_pdf_structure(self, report: ExtendedIncidentReport) -> dict[str, Any]:
        """Build PDF document structure."""
        return {
            "document_type": "DORA_INCIDENT_REPORT",
            "format_version": "1.0",
            "header": {
                "title": f"ICT Incident Report - {report.summary.incident_id}",
                "subtitle": report.metadata.template.value.upper(),
                "reference": report.metadata.reference_number,
                "classification": report.metadata.confidentiality_level,
                "date": report.metadata.created_at.isoformat(),
            },
            "entity_information": {
                "lei": report.metadata.entity_lei,
                "name": report.metadata.entity_name,
                "nca": report.metadata.reporting_nca,
            },
            "incident_summary": {
                "id": report.summary.incident_id,
                "title": report.summary.title,
                "type": report.summary.incident_type,
                "severity": report.summary.severity.value,
                "is_major": report.summary.is_major_incident,
                "detection_time": report.summary.detection_time.isoformat(),
                "classification_time": report.summary.classification_time.isoformat(),
                "resolution_time": (
                    report.summary.resolution_time.isoformat() if report.summary.resolution_time else None
                ),
                "duration_hours": report.summary.duration_hours,
                "description": report.summary.description,
                "root_cause": report.summary.root_cause_category,
            },
            "impact_assessment": {
                "clients_affected": report.impact_assessment.clients_affected_count,
                "transactions_affected": report.impact_assessment.transactions_affected,
                "economic_impact": {
                    "direct": report.impact_assessment.direct_costs,
                    "indirect": report.impact_assessment.indirect_costs,
                    "recovery": report.impact_assessment.recovery_costs,
                    "total": report.impact_assessment.total_economic_impact,
                    "currency": report.impact_assessment.currency,
                },
                "geographic_scope": report.impact_assessment.member_states_affected,
                "data_breach": report.impact_assessment.data_breach,
            },
            "remediation": {
                "immediate_actions": report.remediation_plan.immediate_actions,
                "short_term_actions": report.remediation_plan.short_term_actions,
                "long_term_actions": report.remediation_plan.long_term_actions,
                "lessons_learned": report.remediation_plan.lessons_learned,
                "estimated_completion": report.remediation_plan.estimated_completion.isoformat(),
            },
            "approvals": report.approvals,
            "checksum": report.calculate_checksum(),
        }


# =============================================================================
# JSON Report Generator
# =============================================================================


class JSONReportGenerator:
    """Generate JSON reports for API integration."""

    def __init__(self, config: ReportingConfig) -> None:
        """Initialize JSON generator."""
        self.config = config

    def generate(self, report: ExtendedIncidentReport) -> str:
        """Generate JSON report."""
        return json.dumps(self._build_json_structure(report), indent=2, default=str)

    def generate_compact(self, report: ExtendedIncidentReport) -> str:
        """Generate compact JSON report (no indentation)."""
        return json.dumps(self._build_json_structure(report), default=str)

    def _build_json_structure(self, report: ExtendedIncidentReport) -> dict[str, Any]:
        """Build JSON document structure per CDR 2025/301."""
        structure: dict[str, Any] = {
            "$schema": f"dora-incident-report-v{self.config.json_schema_version}",
            "metadata": {
                "reportId": report.metadata.report_id,
                "incidentId": report.metadata.incident_id,
                "template": report.metadata.template.value,
                "version": report.metadata.version,
                "createdAt": report.metadata.created_at.isoformat(),
                "createdBy": report.metadata.created_by,
                "language": report.metadata.language,
                "referenceNumber": report.metadata.reference_number,
            },
            "reportingEntity": {
                "lei": report.metadata.entity_lei,
                "name": report.metadata.entity_name,
                "nca": report.metadata.reporting_nca,
            },
            "incident": {
                "id": report.summary.incident_id,
                "title": report.summary.title,
                "description": report.summary.description,
                "type": report.summary.incident_type,
                "severity": report.summary.severity.value,
                "isMajor": report.summary.is_major_incident,
                "timeline": {
                    "detected": report.summary.detection_time.isoformat(),
                    "classified": report.summary.classification_time.isoformat(),
                    "resolved": (
                        report.summary.resolution_time.isoformat() if report.summary.resolution_time else None
                    ),
                    "durationHours": report.summary.duration_hours,
                },
                "affectedServices": report.summary.affected_services,
                "geographicScope": report.summary.geographic_scope,
                "rootCause": report.summary.root_cause_category,
                "attackVector": report.summary.attack_vector,
            },
            "impact": {
                "clients": {
                    "count": report.impact_assessment.clients_affected_count,
                    "percentage": report.impact_assessment.clients_affected_percentage,
                    "types": report.impact_assessment.client_types_affected,
                },
                "transactions": {
                    "count": report.impact_assessment.transactions_affected,
                    "value": report.impact_assessment.transaction_value_affected,
                    "currency": report.impact_assessment.currency,
                },
                "duration": {
                    "downtimeHours": report.impact_assessment.service_downtime_hours,
                    "degradedHours": report.impact_assessment.degraded_service_hours,
                },
                "geographic": {"memberStates": report.impact_assessment.member_states_affected},
                "economic": {
                    "directCosts": report.impact_assessment.direct_costs,
                    "indirectCosts": report.impact_assessment.indirect_costs,
                    "recoveryCosts": report.impact_assessment.recovery_costs,
                    "totalImpact": report.impact_assessment.total_economic_impact,
                    "currency": report.impact_assessment.currency,
                },
                "data": {
                    "breachOccurred": report.impact_assessment.data_breach,
                    "typesAffected": report.impact_assessment.data_types_affected,
                    "recordsAffected": report.impact_assessment.records_affected,
                },
                "criticalServices": report.impact_assessment.critical_services_affected,
            },
            "remediation": {
                "immediateActions": report.remediation_plan.immediate_actions,
                "shortTermActions": report.remediation_plan.short_term_actions,
                "longTermActions": report.remediation_plan.long_term_actions,
                "lessonsLearned": report.remediation_plan.lessons_learned,
                "controlImprovements": report.remediation_plan.control_improvements,
                "estimatedCompletion": report.remediation_plan.estimated_completion.isoformat(),
                "responsibleParties": report.remediation_plan.responsible_parties,
            },
            "status": report.status.value,
            "approvals": report.approvals,
            "revisionHistory": report.revision_history,
            "integrity": {"checksum": report.calculate_checksum(), "algorithm": "SHA-256"},
        }

        # Add technical details if present
        if report.technical_details:
            structure["technicalDetails"] = {
                "affectedSystems": report.technical_details.affected_systems,
                "affectedComponents": report.technical_details.affected_components,
                "errorCodes": report.technical_details.error_codes,
                "logReferences": report.technical_details.log_references,
                "vulnerabilityIds": report.technical_details.vulnerability_ids,
                "attackTechniques": report.technical_details.attack_techniques,
                "containmentActions": report.technical_details.containment_actions,
                "eradicationActions": report.technical_details.eradication_actions,
                "recoveryActions": report.technical_details.recovery_actions,
                "indicators": {
                    "network": report.technical_details.network_indicators,
                    "malware": report.technical_details.malware_indicators,
                    "ioc": report.technical_details.ioc_list,
                },
            }

        return structure


# =============================================================================
# Main Service Class
# =============================================================================


class ExtendedReportingService:
    """
    Extended Incident Reporting Service.

    Provides enterprise-grade incident reporting capabilities per DORA Art. 19-20.
    """

    def __init__(self, config: ReportingConfig) -> None:
        """Initialize extended reporting service."""
        self.config = config
        self.pdf_generator = PDFReportGenerator(config)
        self.json_generator = JSONReportGenerator(config)
        self._reports: dict[str, ExtendedIncidentReport] = {}
        self._deliveries: dict[str, ReportDelivery] = {}

    def create_report(
        self,
        incident_id: str,
        template: ReportTemplate,
        summary: IncidentSummary,
        impact_assessment: ImpactAssessment,
        remediation_plan: RemediationPlan,
        technical_details: TechnicalDetails | None = None,
        created_by: str = "system",
    ) -> ExtendedIncidentReport:
        """Create a new incident report."""
        report_id = str(uuid4())
        metadata = ReportMetadata(
            report_id=report_id,
            incident_id=incident_id,
            template=template,
            format=ReportFormat.JSON,  # Default format
            version="1.0.0",
            created_at=datetime.utcnow(),
            created_by=created_by,
            entity_lei=self.config.entity_lei,
            entity_name=self.config.entity_name,
            reporting_nca=self.config.default_nca,
        )

        report = ExtendedIncidentReport(
            metadata=metadata,
            summary=summary,
            technical_details=technical_details,
            impact_assessment=impact_assessment,
            remediation_plan=remediation_plan,
        )

        self._reports[report_id] = report
        return report

    def get_report(self, report_id: str) -> ExtendedIncidentReport | None:
        """Get report by ID."""
        return self._reports.get(report_id)

    def list_reports(
        self,
        incident_id: str | None = None,
        template: ReportTemplate | None = None,
        status: ReportStatus | None = None,
    ) -> list[ExtendedIncidentReport]:
        """List reports with optional filters."""
        reports = list(self._reports.values())

        if incident_id:
            reports = [r for r in reports if r.summary.incident_id == incident_id]
        if template:
            reports = [r for r in reports if r.metadata.template == template]
        if status:
            reports = [r for r in reports if r.status == status]

        return reports

    def generate_pdf(self, report_id: str) -> bytes:
        """Generate PDF version of report."""
        report = self._reports.get(report_id)
        if not report:
            raise ValueError(f"Report not found: {report_id}")
        return self.pdf_generator.generate(report)

    def generate_json(self, report_id: str, compact: bool = False) -> str:
        """Generate JSON version of report."""
        report = self._reports.get(report_id)
        if not report:
            raise ValueError(f"Report not found: {report_id}")
        if compact:
            return self.json_generator.generate_compact(report)
        return self.json_generator.generate(report)

    def submit_report(
        self,
        report_id: str,
        delivery_method: DeliveryMethod,
        recipient: str,
        recipient_type: str = "NCA",
    ) -> ReportDelivery:
        """Submit report to recipient."""
        report = self._reports.get(report_id)
        if not report:
            raise ValueError(f"Report not found: {report_id}")

        if report.status != ReportStatus.APPROVED:
            raise ValueError(f"Report must be approved before submission: {report.status}")

        delivery_id = str(uuid4())
        delivery = ReportDelivery(
            delivery_id=delivery_id,
            report_id=report_id,
            method=delivery_method,
            recipient=recipient,
            recipient_type=recipient_type,
        )

        # Simulate delivery
        delivery.mark_sent()
        report.status = ReportStatus.SUBMITTED

        self._deliveries[delivery_id] = delivery
        return delivery

    def get_delivery(self, delivery_id: str) -> ReportDelivery | None:
        """Get delivery record by ID."""
        return self._deliveries.get(delivery_id)

    def create_initial_notification(
        self,
        incident_id: str,
        title: str,
        description: str,
        incident_type: str,
        detection_time: datetime,
        severity: ReportSeverity,
        affected_services: list[str],
        created_by: str = "system",
    ) -> ExtendedIncidentReport:
        """
        Create initial notification report (4-hour deadline per DORA).

        Per Art. 19(4)(a), initial notification must be submitted within
        4 hours of classifying an incident as major.
        """
        summary = IncidentSummary(
            incident_id=incident_id,
            title=title,
            description=description,
            incident_type=incident_type,
            detection_time=detection_time,
            classification_time=datetime.utcnow(),
            resolution_time=None,
            severity=severity,
            is_major_incident=severity in (ReportSeverity.CRITICAL, ReportSeverity.HIGH),
            affected_services=affected_services,
            affected_clients=0,  # To be determined
            affected_transactions=0,  # To be determined
            geographic_scope=["EU"],
            root_cause_category="UNDER_INVESTIGATION",
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
            immediate_actions=[{"action": "Investigation initiated", "status": "in_progress"}],
            short_term_actions=[],
            long_term_actions=[],
            lessons_learned=[],
            control_improvements=[],
            policy_updates=[],
            training_requirements=[],
            third_party_notifications=[],
            estimated_completion=datetime.utcnow() + timedelta(days=30),
            responsible_parties=["Incident Response Team"],
        )

        return self.create_report(
            incident_id=incident_id,
            template=ReportTemplate.INITIAL,
            summary=summary,
            impact_assessment=impact,
            remediation_plan=remediation,
            created_by=created_by,
        )

    def create_intermediate_report(
        self,
        initial_report_id: str,
        updated_summary: IncidentSummary,
        updated_impact: ImpactAssessment,
        updated_remediation: RemediationPlan,
        technical_details: TechnicalDetails | None = None,
        created_by: str = "system",
    ) -> ExtendedIncidentReport:
        """
        Create intermediate report (72-hour deadline per DORA).

        Per Art. 19(4)(b), intermediate report must be submitted within
        72 hours of initial notification with updated information.
        """
        initial_report = self._reports.get(initial_report_id)
        if not initial_report:
            raise ValueError(f"Initial report not found: {initial_report_id}")

        return self.create_report(
            incident_id=initial_report.summary.incident_id,
            template=ReportTemplate.INTERMEDIATE,
            summary=updated_summary,
            impact_assessment=updated_impact,
            remediation_plan=updated_remediation,
            technical_details=technical_details,
            created_by=created_by,
        )

    def create_final_report(
        self,
        incident_id: str,
        final_summary: IncidentSummary,
        final_impact: ImpactAssessment,
        final_remediation: RemediationPlan,
        technical_details: TechnicalDetails,
        created_by: str = "system",
    ) -> ExtendedIncidentReport:
        """
        Create final report (1-month deadline per DORA).

        Per Art. 19(4)(c), final report must be submitted within 1 month
        of the intermediate report (or earlier if incident is resolved).
        """
        return self.create_report(
            incident_id=incident_id,
            template=ReportTemplate.FINAL,
            summary=final_summary,
            impact_assessment=final_impact,
            remediation_plan=final_remediation,
            technical_details=technical_details,
            created_by=created_by,
        )

    def get_reporting_deadlines(self, incident_id: str) -> dict[str, datetime]:
        """
        Calculate reporting deadlines for an incident.

        Per DORA Art. 19(4):
        - Initial notification: 4 hours from classification
        - Intermediate report: 72 hours from initial notification
        - Final report: 1 month from intermediate report
        """
        reports = self.list_reports(incident_id=incident_id)
        if not reports:
            return {}

        initial = next((r for r in reports if r.metadata.template == ReportTemplate.INITIAL), None)
        if not initial:
            return {}

        classification_time = initial.summary.classification_time
        deadlines = {
            "initial_notification": classification_time + timedelta(hours=4),
            "intermediate_report": classification_time + timedelta(hours=72),
            "final_report": classification_time + timedelta(days=30),
        }

        return deadlines


# =============================================================================
# Factory Functions
# =============================================================================


def create_extended_reporting(
    entity_lei: str,
    entity_name: str,
    default_nca: str = "BaFin",  # Default to German NCA
    **kwargs: Any,
) -> ExtendedReportingService:
    """Create extended reporting service instance."""
    config = ReportingConfig(entity_lei=entity_lei, entity_name=entity_name, default_nca=default_nca, **kwargs)
    return ExtendedReportingService(config)


def generate_pdf_report(
    service: ExtendedReportingService,
    report_id: str,
) -> bytes:
    """Generate PDF report (convenience function)."""
    return service.generate_pdf(report_id)


def generate_json_report(
    service: ExtendedReportingService,
    report_id: str,
    compact: bool = False,
) -> str:
    """Generate JSON report (convenience function)."""
    return service.generate_json(report_id, compact=compact)
