# -*- coding: utf-8 -*-
"""
DORA Unified Reporting (Phase 5).

Unifies reporting artefacts across DORA, AI Act, NIS2 and MiFID II by
providing a consistent abstraction for report registration, validation,
submission tracking and packaging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
from uuid import uuid4


class ReportType(Enum):
    """Report types supported by the unified manager."""
    DORA_MAJOR_INCIDENT = "dora_major_incident"
    DORA_REGISTER_UPDATE = "dora_register_update"
    AI_ACT_SERIOUS_INCIDENT = "ai_act_serious_incident"
    TLPT_RESULT = "tlpt_result"
    INTERNAL_RESILIENCE = "internal_resilience"


class ReportStatus(Enum):
    """Lifecycle of a report."""
    DRAFT = "draft"
    READY = "ready"
    SUBMITTED = "submitted"
    FAILED = "failed"


class ReportChannel(Enum):
    """Delivery channels for submissions."""
    API = "api"
    EMAIL = "email"
    PORTAL = "portal"


@dataclass
class ReportDestination:
    """Destination metadata."""
    name: str
    channel: ReportChannel
    endpoint: str
    encryption_required: bool = True


@dataclass
class UnifiedReport:
    """A single report entry."""
    report_type: ReportType
    content: Dict[str, str]
    destination: ReportDestination
    due_at: datetime
    report_id: str = ""
    status: ReportStatus = ReportStatus.DRAFT
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    submitted_at: Optional[datetime] = None
    classification: str = "confidential"
    attachments: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.report_id:
            self.report_id = f"REPORT-{uuid4().hex[:10].upper()}"

    def mark_submitted(self, submitted_at: Optional[datetime] = None) -> None:
        self.status = ReportStatus.SUBMITTED
        self.submitted_at = submitted_at or datetime.now(timezone.utc)


@dataclass
class SubmissionPackage:
    """Aggregated submission package for a destination."""
    destination: ReportDestination
    reports: List[UnifiedReport]
    package_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    encrypted: bool = True

    def __post_init__(self):
        if not self.package_id:
            self.package_id = f"PKG-{uuid4().hex[:8].upper()}"


class UnifiedReportingManager:
    """Manage unified reporting lifecycle."""

    REQUIRED_FIELDS = {
        ReportType.DORA_MAJOR_INCIDENT: {"incident_id", "classification", "services_affected"},
        ReportType.DORA_REGISTER_UPDATE: {"arrangement_reference", "provider_name"},
        ReportType.AI_ACT_SERIOUS_INCIDENT: {"ai_system_id", "incident_description", "harm_assessment"},
        ReportType.TLPT_RESULT: {"scope", "threat_scenarios", "tester"},
        ReportType.INTERNAL_RESILIENCE: {"summary", "owner"},
    }

    def __init__(self):
        self.reports: Dict[str, UnifiedReport] = {}

    # ------------------------------------------------------------------ #
    # Registration and Validation
    # ------------------------------------------------------------------ #
    def _validate_content(self, report_type: ReportType, content: Dict[str, str]) -> None:
        required = self.REQUIRED_FIELDS[report_type]
        missing = required - set(content.keys())
        if missing:
            raise ValueError(f"Missing required fields: {','.join(sorted(missing))}")

    def register_report(self, report: UnifiedReport) -> UnifiedReport:
        self._validate_content(report.report_type, report.content)
        self.reports[report.report_id] = report
        return report

    # ------------------------------------------------------------------ #
    # Status management
    # ------------------------------------------------------------------ #
    def mark_ready(self, report_id: str) -> bool:
        report = self.reports.get(report_id)
        if not report:
            return False
        report.status = ReportStatus.READY
        return True

    def mark_submitted(self, report_id: str, submitted_at: Optional[datetime] = None) -> bool:
        report = self.reports.get(report_id)
        if not report:
            return False
        report.mark_submitted(submitted_at=submitted_at)
        return True

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def get_pending_reports(self, report_type: Optional[ReportType] = None) -> List[UnifiedReport]:
        reports = [
            report for report in self.reports.values()
            if report.status in {ReportStatus.DRAFT, ReportStatus.READY}
        ]
        if report_type:
            reports = [report for report in reports if report.report_type == report_type]
        return sorted(reports, key=lambda report: report.due_at)

    def generate_submission_package(self, destination_name: str) -> SubmissionPackage:
        ready_reports = [
            report for report in self.reports.values()
            if report.status == ReportStatus.READY and report.destination.name == destination_name
        ]
        if not ready_reports:
            raise ValueError(f"No ready reports for destination {destination_name}")
        destination = ready_reports[0].destination
        encrypted = destination.encryption_required or destination.channel != ReportChannel.EMAIL
        return SubmissionPackage(destination=destination, reports=ready_reports, encrypted=encrypted)

    def export_summary(self) -> List[Dict[str, str]]:
        """Provide a lightweight view for dashboards."""
        summary = []
        for report in self.reports.values():
            summary.append(
                {
                    "report_id": report.report_id,
                    "type": report.report_type.value,
                    "status": report.status.value,
                    "destination": report.destination.name,
                    "due_at": report.due_at.isoformat(),
                    "submitted_at": report.submitted_at.isoformat() if report.submitted_at else "",
                }
            )
        return summary

