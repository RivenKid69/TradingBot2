# -*- coding: utf-8 -*-
"""
DORA Unified Reporting Module - Integration Layer (Phase 5).

Provides unified reporting capabilities for ICT service providers to generate
client-facing report packages across multiple regulatory frameworks.

Key Distinction (ICT Provider Role):
    - We GENERATE report data packages for clients
    - We DO NOT submit reports to NCAs directly
    - Clients use our data packages for their regulatory submissions

Cross-Regulatory Support:
    - DORA: Major incident reports, register updates
    - AI Act: Serious incident reports (if AI services provided)
    - Internal: Resilience testing results

Report Lifecycle:
    1. Report created (DRAFT)
    2. Content validated
    3. Report marked ready (READY)
    4. Packaged for client delivery
    5. Client receives package for their NCA submission

References:
    - DORA Article 28(3): Register of Information
    - DORA Article 19: Incident reporting via financial entities
    - CDR 2025/301: RTS on incident reporting content
    - CIR 2024/2956: ITS on Register of Information

Migration: services/dora/unified_reporting.py -> services/dora_integration/reporting/
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class ReportType(Enum):
    """Report types supported by the unified manager."""
    # DORA-specific
    DORA_MAJOR_INCIDENT = "dora_major_incident"
    DORA_REGISTER_UPDATE = "dora_register_update"
    DORA_CTPP_NOTIFICATION = "dora_ctpp_notification"

    # Cross-regulatory
    AI_ACT_SERIOUS_INCIDENT = "ai_act_serious_incident"

    # Internal resilience
    TLPT_RESULT = "tlpt_result"
    INTERNAL_RESILIENCE = "internal_resilience"

    # Client data packages
    CLIENT_ROI_PACKAGE = "client_roi_package"
    CLIENT_INCIDENT_PACKAGE = "client_incident_package"


class ReportStatus(Enum):
    """Lifecycle of a report."""
    DRAFT = "draft"
    VALIDATING = "validating"
    READY = "ready"
    PACKAGED = "packaged"
    DELIVERED = "delivered"
    SUBMITTED = "submitted"  # Client has submitted to NCA
    FAILED = "failed"


class ReportChannel(Enum):
    """Delivery channels for client packages."""
    API = "api"
    EMAIL = "email"
    PORTAL = "portal"
    SECURE_FILE_TRANSFER = "sftp"
    WEBHOOK = "webhook"


class PackageFormat(Enum):
    """Export formats for report packages."""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    PDF = "pdf"


class ClientType(Enum):
    """Client types for report routing."""
    FINANCIAL_ENTITY = "financial_entity"
    CREDIT_INSTITUTION = "credit_institution"
    INVESTMENT_FIRM = "investment_firm"
    INSURANCE_UNDERTAKING = "insurance_undertaking"
    CRYPTO_ASSET_PROVIDER = "crypto_asset_provider"
    OTHER = "other"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ReportDestination:
    """Destination metadata for report delivery."""
    name: str
    client_id: str
    client_type: ClientType
    channel: ReportChannel
    endpoint: str
    encryption_required: bool = True
    contact_email: str = ""
    preferred_format: PackageFormat = PackageFormat.JSON

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "client_id": self.client_id,
            "client_type": self.client_type.value,
            "channel": self.channel.value,
            "endpoint": self.endpoint,
            "encryption_required": self.encryption_required,
            "contact_email": self.contact_email,
            "preferred_format": self.preferred_format.value,
        }


@dataclass
class ReportValidationResult:
    """Result of report content validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validated_at: str = ""

    def __post_init__(self):
        if not self.validated_at:
            self.validated_at = datetime.now(timezone.utc).isoformat()


@dataclass
class UnifiedReport:
    """A single report entry for client delivery."""
    report_type: ReportType
    content: Dict[str, Any]
    destination: ReportDestination
    due_at: datetime

    # Identifiers
    report_id: str = ""
    reference_id: str = ""  # Internal reference

    # Lifecycle
    status: ReportStatus = ReportStatus.DRAFT
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    validated_at: Optional[datetime] = None
    packaged_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    submitted_at: Optional[datetime] = None  # When client submitted to NCA

    # Metadata
    classification: str = "confidential"
    attachments: List[str] = field(default_factory=list)
    validation_result: Optional[ReportValidationResult] = None

    # Delivery tracking
    delivery_attempts: int = 0
    last_error: str = ""
    acknowledgment_id: str = ""  # Client acknowledgment

    def __post_init__(self):
        if not self.report_id:
            self.report_id = f"REPORT-{uuid4().hex[:10].upper()}"
        if not self.reference_id:
            self.reference_id = f"REF-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    def mark_validated(self, result: ReportValidationResult) -> None:
        """Mark report as validated."""
        self.validation_result = result
        self.validated_at = datetime.now(timezone.utc)
        if result.is_valid:
            self.status = ReportStatus.READY
        else:
            self.status = ReportStatus.FAILED

    def mark_packaged(self) -> None:
        """Mark report as packaged for delivery."""
        self.status = ReportStatus.PACKAGED
        self.packaged_at = datetime.now(timezone.utc)

    def mark_delivered(self, acknowledgment_id: str = "") -> None:
        """Mark report as delivered to client."""
        self.status = ReportStatus.DELIVERED
        self.delivered_at = datetime.now(timezone.utc)
        self.acknowledgment_id = acknowledgment_id

    def mark_submitted(self, submitted_at: Optional[datetime] = None) -> None:
        """Mark as submitted by client to NCA."""
        self.status = ReportStatus.SUBMITTED
        self.submitted_at = submitted_at or datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "reference_id": self.reference_id,
            "report_type": self.report_type.value,
            "status": self.status.value,
            "content": self.content,
            "destination": self.destination.to_dict(),
            "due_at": self.due_at.isoformat(),
            "created_at": self.created_at.isoformat(),
            "validated_at": self.validated_at.isoformat() if self.validated_at else None,
            "packaged_at": self.packaged_at.isoformat() if self.packaged_at else None,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "classification": self.classification,
            "attachments": self.attachments,
            "acknowledgment_id": self.acknowledgment_id,
        }


@dataclass
class SubmissionPackage:
    """Aggregated submission package for client delivery."""
    destination: ReportDestination
    reports: List[UnifiedReport]
    package_format: PackageFormat = PackageFormat.JSON

    # Package metadata
    package_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    encrypted: bool = True
    checksum: str = ""

    # Package content
    package_content: bytes = field(default=b"", repr=False)

    def __post_init__(self):
        if not self.package_id:
            self.package_id = f"PKG-{uuid4().hex[:8].upper()}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without binary content)."""
        return {
            "package_id": self.package_id,
            "destination": self.destination.to_dict(),
            "reports_count": len(self.reports),
            "report_ids": [r.report_id for r in self.reports],
            "package_format": self.package_format.value,
            "created_at": self.created_at.isoformat(),
            "encrypted": self.encrypted,
            "checksum": self.checksum,
        }


@dataclass
class DeliveryRecord:
    """Record of package delivery to client."""
    package_id: str
    client_id: str
    delivery_channel: ReportChannel
    delivered_at: datetime
    acknowledgment_id: str = ""
    status: str = "delivered"
    error_message: str = ""
    retry_count: int = 0


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class UnifiedReportingConfig:
    """Configuration for unified reporting."""

    # Provider identification
    provider_lei: str = ""
    provider_name: str = ""

    # Validation settings
    validate_on_register: bool = True
    require_all_mandatory_fields: bool = True

    # Packaging settings
    default_format: PackageFormat = PackageFormat.JSON
    encryption_enabled: bool = True

    # Delivery settings
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 60

    # Storage
    storage_path: str = "state/dora/unified_reporting"
    log_path: str = "logs/dora/unified_reporting"


# =============================================================================
# Main Implementation
# =============================================================================

class UnifiedReportingManager:
    """
    Unified Reporting Manager for ICT Service Providers.

    Manages report lifecycle from creation to client delivery:
    1. Report registration and validation
    2. Content packaging per client requirements
    3. Multi-channel delivery
    4. Submission tracking

    Key Principle:
        We generate DATA PACKAGES for clients.
        Clients submit to their NCAs using our data.
        We track delivery status, not NCA submission.

    Usage:
        manager = UnifiedReportingManager(config)

        # Create report
        report = manager.create_report(
            report_type=ReportType.DORA_MAJOR_INCIDENT,
            content={"incident_id": "INC-001", ...},
            destination=destination,
            due_at=datetime.now() + timedelta(hours=4),
        )

        # Validate and mark ready
        if manager.mark_ready(report.report_id):
            # Generate package
            package = manager.generate_submission_package(
                client_id="CLIENT-001",
            )
    """

    # Required fields per report type
    REQUIRED_FIELDS: Dict[ReportType, set] = {
        ReportType.DORA_MAJOR_INCIDENT: {
            "incident_id", "classification", "services_affected"
        },
        ReportType.DORA_REGISTER_UPDATE: {
            "arrangement_reference", "provider_name"
        },
        ReportType.DORA_CTPP_NOTIFICATION: {
            "notification_type", "ctpp_status"
        },
        ReportType.AI_ACT_SERIOUS_INCIDENT: {
            "ai_system_id", "incident_description", "harm_assessment"
        },
        ReportType.TLPT_RESULT: {
            "scope", "threat_scenarios", "tester"
        },
        ReportType.INTERNAL_RESILIENCE: {
            "summary", "owner"
        },
        ReportType.CLIENT_ROI_PACKAGE: {
            "provider_identification", "service_records"
        },
        ReportType.CLIENT_INCIDENT_PACKAGE: {
            "incident_id", "incident_data"
        },
    }

    def __init__(self, config: Optional[UnifiedReportingConfig] = None):
        """Initialize unified reporting manager."""
        self.config = config or UnifiedReportingConfig()

        # Report storage
        self._reports: Dict[str, UnifiedReport] = {}

        # Package storage
        self._packages: Dict[str, SubmissionPackage] = {}

        # Delivery records
        self._delivery_records: List[DeliveryRecord] = []

        # Indexes
        self._reports_by_client: Dict[str, set] = {}
        self._reports_by_type: Dict[ReportType, set] = {}
        self._reports_by_status: Dict[ReportStatus, set] = {}

        logger.info("UnifiedReportingManager initialized")

    # =========================================================================
    # Report Creation and Registration
    # =========================================================================

    def create_report(
        self,
        report_type: ReportType,
        content: Dict[str, Any],
        destination: ReportDestination,
        due_at: datetime,
        reference_id: str = "",
        classification: str = "confidential",
        attachments: Optional[List[str]] = None,
    ) -> UnifiedReport:
        """
        Create a new report.

        Args:
            report_type: Type of report
            content: Report content dictionary
            destination: Delivery destination
            due_at: Due date for delivery
            reference_id: Internal reference
            classification: Data classification
            attachments: List of attachment paths

        Returns:
            Created UnifiedReport
        """
        report = UnifiedReport(
            report_type=report_type,
            content=content,
            destination=destination,
            due_at=due_at,
            reference_id=reference_id,
            classification=classification,
            attachments=attachments or [],
        )

        # Store report
        self._reports[report.report_id] = report

        # Update indexes
        client_id = destination.client_id
        if client_id not in self._reports_by_client:
            self._reports_by_client[client_id] = set()
        self._reports_by_client[client_id].add(report.report_id)

        if report_type not in self._reports_by_type:
            self._reports_by_type[report_type] = set()
        self._reports_by_type[report_type].add(report.report_id)

        if report.status not in self._reports_by_status:
            self._reports_by_status[report.status] = set()
        self._reports_by_status[report.status].add(report.report_id)

        # Validate if configured
        if self.config.validate_on_register:
            self._validate_content(report)

        logger.info(f"Report created: {report.report_id} ({report_type.value})")

        return report

    def register_report(self, report: UnifiedReport) -> UnifiedReport:
        """
        Register an existing report.

        Args:
            report: Report to register

        Returns:
            Registered report
        """
        # Validate if configured
        if self.config.validate_on_register:
            self._validate_content(report)

        # Store
        self._reports[report.report_id] = report

        # Update indexes
        client_id = report.destination.client_id
        if client_id not in self._reports_by_client:
            self._reports_by_client[client_id] = set()
        self._reports_by_client[client_id].add(report.report_id)

        if report.report_type not in self._reports_by_type:
            self._reports_by_type[report.report_type] = set()
        self._reports_by_type[report.report_type].add(report.report_id)

        if report.status not in self._reports_by_status:
            self._reports_by_status[report.status] = set()
        self._reports_by_status[report.status].add(report.report_id)

        return report

    def _validate_content(self, report: UnifiedReport) -> ReportValidationResult:
        """Validate report content."""
        errors = []
        warnings = []

        required = self.REQUIRED_FIELDS.get(report.report_type, set())
        missing = required - set(report.content.keys())

        if missing and self.config.require_all_mandatory_fields:
            errors.append(f"Missing required fields: {', '.join(sorted(missing))}")
        elif missing:
            warnings.append(f"Missing recommended fields: {', '.join(sorted(missing))}")

        # Additional validation by report type
        if report.report_type == ReportType.DORA_MAJOR_INCIDENT:
            if not report.content.get("services_affected"):
                warnings.append("services_affected should not be empty")

        result = ReportValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

        report.validation_result = result

        return result

    # =========================================================================
    # Status Management
    # =========================================================================

    def mark_ready(self, report_id: str) -> bool:
        """
        Mark report as ready for packaging.

        Args:
            report_id: Report ID

        Returns:
            True if successful
        """
        report = self._reports.get(report_id)
        if not report:
            return False

        # Validate first
        result = self._validate_content(report)
        if not result.is_valid:
            logger.warning(f"Report {report_id} validation failed: {result.errors}")
            return False

        # Update status
        old_status = report.status
        report.status = ReportStatus.READY
        report.validated_at = datetime.now(timezone.utc)

        # Update index
        if old_status in self._reports_by_status:
            self._reports_by_status[old_status].discard(report_id)
        if ReportStatus.READY not in self._reports_by_status:
            self._reports_by_status[ReportStatus.READY] = set()
        self._reports_by_status[ReportStatus.READY].add(report_id)

        logger.info(f"Report {report_id} marked ready")
        return True

    def mark_delivered(
        self,
        report_id: str,
        acknowledgment_id: str = "",
    ) -> bool:
        """
        Mark report as delivered to client.

        Args:
            report_id: Report ID
            acknowledgment_id: Client acknowledgment ID

        Returns:
            True if successful
        """
        report = self._reports.get(report_id)
        if not report:
            return False

        old_status = report.status
        report.mark_delivered(acknowledgment_id)

        # Update index
        if old_status in self._reports_by_status:
            self._reports_by_status[old_status].discard(report_id)
        if ReportStatus.DELIVERED not in self._reports_by_status:
            self._reports_by_status[ReportStatus.DELIVERED] = set()
        self._reports_by_status[ReportStatus.DELIVERED].add(report_id)

        logger.info(f"Report {report_id} delivered to client")
        return True

    def mark_submitted(
        self,
        report_id: str,
        submitted_at: Optional[datetime] = None,
    ) -> bool:
        """
        Mark report as submitted by client to NCA.

        Args:
            report_id: Report ID
            submitted_at: Submission timestamp

        Returns:
            True if successful
        """
        report = self._reports.get(report_id)
        if not report:
            return False

        old_status = report.status
        report.mark_submitted(submitted_at)

        # Update index
        if old_status in self._reports_by_status:
            self._reports_by_status[old_status].discard(report_id)
        if ReportStatus.SUBMITTED not in self._reports_by_status:
            self._reports_by_status[ReportStatus.SUBMITTED] = set()
        self._reports_by_status[ReportStatus.SUBMITTED].add(report_id)

        logger.info(f"Report {report_id} marked as submitted to NCA")
        return True

    # =========================================================================
    # Queries
    # =========================================================================

    def get_report(self, report_id: str) -> Optional[UnifiedReport]:
        """Get report by ID."""
        return self._reports.get(report_id)

    def get_reports_for_client(
        self,
        client_id: str,
        status: Optional[ReportStatus] = None,
    ) -> List[UnifiedReport]:
        """Get reports for a specific client."""
        report_ids = self._reports_by_client.get(client_id, set())
        reports = [self._reports[rid] for rid in report_ids if rid in self._reports]

        if status:
            reports = [r for r in reports if r.status == status]

        return sorted(reports, key=lambda r: r.due_at)

    def get_pending_reports(
        self,
        report_type: Optional[ReportType] = None,
    ) -> List[UnifiedReport]:
        """
        Get pending reports (DRAFT or READY status).

        Args:
            report_type: Filter by type

        Returns:
            List of pending reports sorted by due date
        """
        pending_statuses = {ReportStatus.DRAFT, ReportStatus.READY}
        reports = [
            r for r in self._reports.values()
            if r.status in pending_statuses
        ]

        if report_type:
            reports = [r for r in reports if r.report_type == report_type]

        return sorted(reports, key=lambda r: r.due_at)

    def get_ready_reports_for_client(
        self,
        client_id: str,
    ) -> List[UnifiedReport]:
        """Get ready reports for packaging."""
        return self.get_reports_for_client(client_id, ReportStatus.READY)

    def get_overdue_reports(self) -> List[UnifiedReport]:
        """Get reports past their due date."""
        now = datetime.now(timezone.utc)
        pending_statuses = {ReportStatus.DRAFT, ReportStatus.READY, ReportStatus.PACKAGED}

        return [
            r for r in self._reports.values()
            if r.status in pending_statuses and r.due_at < now
        ]

    # =========================================================================
    # Package Generation
    # =========================================================================

    def generate_submission_package(
        self,
        client_id: str,
        package_format: Optional[PackageFormat] = None,
    ) -> SubmissionPackage:
        """
        Generate submission package for client.

        Aggregates all READY reports for the client into a single package.

        Args:
            client_id: Client identifier
            package_format: Override format (default: client preference)

        Returns:
            SubmissionPackage ready for delivery
        """
        ready_reports = self.get_ready_reports_for_client(client_id)

        if not ready_reports:
            raise ValueError(f"No ready reports for client {client_id}")

        destination = ready_reports[0].destination
        fmt = package_format or destination.preferred_format

        # Generate package content
        package_data = {
            "provider": {
                "lei": self.config.provider_lei,
                "name": self.config.provider_name,
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "reports_count": len(ready_reports),
            "reports": [r.to_dict() for r in ready_reports],
        }

        if fmt == PackageFormat.JSON:
            content = json.dumps(package_data, indent=2, default=str).encode("utf-8")
        elif fmt == PackageFormat.CSV:
            content = self._generate_csv_content(ready_reports)
        elif fmt == PackageFormat.XML:
            content = self._generate_xml_content(package_data)
        else:
            content = json.dumps(package_data, default=str).encode("utf-8")

        # Create package
        package = SubmissionPackage(
            destination=destination,
            reports=ready_reports,
            package_format=fmt,
            encrypted=destination.encryption_required and self.config.encryption_enabled,
            package_content=content,
        )

        # Update report statuses
        for report in ready_reports:
            report.mark_packaged()

            # Update status index
            self._reports_by_status[ReportStatus.READY].discard(report.report_id)
            if ReportStatus.PACKAGED not in self._reports_by_status:
                self._reports_by_status[ReportStatus.PACKAGED] = set()
            self._reports_by_status[ReportStatus.PACKAGED].add(report.report_id)

        # Store package
        self._packages[package.package_id] = package

        logger.info(
            f"Generated package {package.package_id} for client {client_id} "
            f"with {len(ready_reports)} reports"
        )

        return package

    def _generate_csv_content(self, reports: List[UnifiedReport]) -> bytes:
        """Generate CSV content from reports."""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "report_id", "report_type", "status", "due_at",
            "created_at", "reference_id"
        ])

        for report in reports:
            writer.writerow([
                report.report_id,
                report.report_type.value,
                report.status.value,
                report.due_at.isoformat(),
                report.created_at.isoformat(),
                report.reference_id,
            ])

        return output.getvalue().encode("utf-8")

    def _generate_xml_content(self, data: Dict[str, Any]) -> bytes:
        """Generate XML content from data."""
        def dict_to_xml(d: Dict, root: str) -> str:
            parts = [f"<{root}>"]
            for key, value in d.items():
                if isinstance(value, dict):
                    parts.append(dict_to_xml(value, key))
                elif isinstance(value, list):
                    parts.append(f"<{key}>")
                    for item in value:
                        if isinstance(item, dict):
                            parts.append(dict_to_xml(item, "item"))
                        else:
                            parts.append(f"<item>{_escape_xml(str(item))}</item>")
                    parts.append(f"</{key}>")
                else:
                    parts.append(f"<{key}>{_escape_xml(str(value))}</{key}>")
            parts.append(f"</{root}>")
            return "".join(parts)

        xml_content = dict_to_xml(data, "ReportPackage")
        return f'<?xml version="1.0" encoding="UTF-8"?>\n{xml_content}'.encode("utf-8")

    # =========================================================================
    # Statistics and Summary
    # =========================================================================

    def export_summary(self) -> List[Dict[str, str]]:
        """Provide a lightweight view for dashboards."""
        summary = []
        for report in self._reports.values():
            summary.append({
                "report_id": report.report_id,
                "type": report.report_type.value,
                "status": report.status.value,
                "destination": report.destination.name,
                "due_at": report.due_at.isoformat(),
                "delivered_at": report.delivered_at.isoformat() if report.delivered_at else "",
                "submitted_at": report.submitted_at.isoformat() if report.submitted_at else "",
            })
        return summary

    def get_statistics(self) -> Dict[str, Any]:
        """Get reporting statistics."""
        by_status = {}
        for status in ReportStatus:
            by_status[status.value] = len(self._reports_by_status.get(status, set()))

        by_type = {}
        for report_type in ReportType:
            by_type[report_type.value] = len(self._reports_by_type.get(report_type, set()))

        overdue = self.get_overdue_reports()

        return {
            "total_reports": len(self._reports),
            "total_packages": len(self._packages),
            "by_status": by_status,
            "by_type": by_type,
            "overdue_count": len(overdue),
            "clients_count": len(self._reports_by_client),
        }


# =============================================================================
# Helper Functions
# =============================================================================

def _escape_xml(text: str) -> str:
    """Escape special XML characters."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
    )


def create_unified_reporting_manager(
    config: Optional[UnifiedReportingConfig] = None,
) -> UnifiedReportingManager:
    """
    Create a UnifiedReportingManager instance.

    Args:
        config: Optional configuration

    Returns:
        Configured UnifiedReportingManager instance
    """
    return UnifiedReportingManager(config=config)


def create_report_destination(
    name: str,
    client_id: str,
    channel: ReportChannel = ReportChannel.API,
    endpoint: str = "",
    client_type: ClientType = ClientType.FINANCIAL_ENTITY,
    encryption_required: bool = True,
    preferred_format: PackageFormat = PackageFormat.JSON,
) -> ReportDestination:
    """
    Create a report destination.

    Args:
        name: Destination name
        client_id: Client identifier
        channel: Delivery channel
        endpoint: Delivery endpoint
        client_type: Type of client
        encryption_required: Whether encryption is required
        preferred_format: Preferred package format

    Returns:
        Configured ReportDestination
    """
    return ReportDestination(
        name=name,
        client_id=client_id,
        client_type=client_type,
        channel=channel,
        endpoint=endpoint,
        encryption_required=encryption_required,
        preferred_format=preferred_format,
    )


def get_report_types() -> List[ReportType]:
    """Get all report types."""
    return list(ReportType)


def get_report_statuses() -> List[ReportStatus]:
    """Get all report statuses."""
    return list(ReportStatus)
