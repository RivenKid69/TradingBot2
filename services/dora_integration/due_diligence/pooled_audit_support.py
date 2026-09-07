# -*- coding: utf-8 -*-
"""
DORA Pooled Audit Support Module.

For ICT Third-Party Service Providers: Supports pooled audits per Article 30(4)
where multiple financial entity clients can rely on shared third-party audits.

DORA Context:
    - Art. 30(4): Allows financial entities to make use of pooled audits
    - Art. 30(4)(a): Third-party certifications or internal/external audit reports
    - Art. 30(4)(b): Pooled audits organized jointly with other financial entities

Key Features:
    - Organize and coordinate pooled audit engagements
    - Share audit reports with participating clients
    - Manage audit schedules and participation
    - Track audit findings and remediation across clients

References:
    - DORA Article 30(4): Pooled audit provisions
    - DORA Article 30(3)(e): Audit rights foundation
    - DORA_OPERATIONAL_RESILIENCE_PLAN.md Section 6.12
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class AuditReportType(Enum):
    """Types of audit reports available for pooled use."""
    SOC2_TYPE_I = "soc2_type_i"
    SOC2_TYPE_II = "soc2_type_ii"
    ISO27001 = "iso27001"
    ISAE3402_TYPE_I = "isae3402_type_i"
    ISAE3402_TYPE_II = "isae3402_type_ii"
    PENETRATION_TEST = "penetration_test"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"
    THIRD_PARTY_AUDIT = "third_party_audit"
    INTERNAL_AUDIT = "internal_audit"


class PooledAuditStatus(Enum):
    """Status of a pooled audit engagement."""
    PLANNING = "planning"
    RECRUITING = "recruiting"  # Recruiting participants
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    REPORT_DRAFTING = "report_drafting"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ParticipationStatus(Enum):
    """Client participation status in pooled audit."""
    INVITED = "invited"
    INTERESTED = "interested"
    CONFIRMED = "confirmed"
    DECLINED = "declined"
    WITHDRAWN = "withdrawn"
    PARTICIPATED = "participated"


class AuditScopeArea(Enum):
    """Audit scope areas per DORA requirements."""
    ICT_GOVERNANCE = "ict_governance"
    ICT_RISK_MANAGEMENT = "ict_risk_management"
    ICT_SECURITY = "ict_security"
    ICT_OPERATIONS = "ict_operations"
    ICT_CONTINUITY = "ict_continuity"
    INCIDENT_MANAGEMENT = "incident_management"
    THIRD_PARTY_MANAGEMENT = "third_party_management"
    CHANGE_MANAGEMENT = "change_management"
    ACCESS_CONTROL = "access_control"
    DATA_PROTECTION = "data_protection"


class FindingSeverity(Enum):
    """Severity of audit findings."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class RemediationStatus(Enum):
    """Status of finding remediation."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    REMEDIATED = "remediated"
    ACCEPTED = "accepted"  # Risk accepted
    CLOSED = "closed"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class CertificationRecord:
    """
    Record of a certification or attestation.

    Per Art. 30(4)(a), clients can rely on third-party certifications.
    """
    certification_id: str = ""
    certification_type: AuditReportType = AuditReportType.SOC2_TYPE_II
    certifying_body: str = ""

    # Dates
    issue_date: str = ""
    expiry_date: str = ""
    next_audit_date: str = ""

    # Scope
    scope_description: str = ""
    scope_areas: List[AuditScopeArea] = field(default_factory=list)
    exceptions_noted: List[str] = field(default_factory=list)

    # Access
    report_available: bool = True
    nda_required: bool = True
    summary_available: bool = True

    # Document reference
    document_path: str = ""
    document_hash: str = ""

    def __post_init__(self):
        if not self.certification_id:
            self.certification_id = f"CERT-{uuid.uuid4().hex[:8].upper()}"

    @property
    def is_valid(self) -> bool:
        """Check if certification is currently valid."""
        if not self.expiry_date:
            return False
        expiry = datetime.fromisoformat(self.expiry_date.replace('Z', '+00:00'))
        return datetime.now(timezone.utc) < expiry


@dataclass
class PooledAuditParticipant:
    """
    A client participating in a pooled audit.
    """
    participant_id: str = ""
    client_id: str = ""
    client_name: str = ""
    contact_name: str = ""
    contact_email: str = ""

    # Participation
    status: ParticipationStatus = ParticipationStatus.INVITED
    invited_date: str = ""
    confirmed_date: str = ""

    # Contribution
    cost_share_pct: float = 0.0
    specific_requirements: List[str] = field(default_factory=list)

    # Access
    report_access_granted: bool = False
    report_accessed_date: str = ""

    def __post_init__(self):
        if not self.participant_id:
            self.participant_id = f"PART-{uuid.uuid4().hex[:8].upper()}"
        if not self.invited_date:
            self.invited_date = datetime.now(timezone.utc).isoformat()


@dataclass
class AuditFinding:
    """
    A finding from an audit engagement.
    """
    finding_id: str = ""
    audit_id: str = ""

    # Finding details
    title: str = ""
    description: str = ""
    severity: FindingSeverity = FindingSeverity.MEDIUM
    scope_area: AuditScopeArea = AuditScopeArea.ICT_SECURITY

    # Impact
    dora_article_reference: str = ""
    risk_description: str = ""

    # Remediation
    recommendation: str = ""
    remediation_status: RemediationStatus = RemediationStatus.OPEN
    remediation_owner: str = ""
    remediation_deadline: str = ""
    remediation_completed_date: str = ""
    remediation_evidence: str = ""

    # Tracking
    identified_date: str = ""
    last_updated: str = ""

    def __post_init__(self):
        if not self.finding_id:
            self.finding_id = f"FND-{uuid.uuid4().hex[:8].upper()}"
        if not self.identified_date:
            self.identified_date = datetime.now(timezone.utc).isoformat()
        if not self.last_updated:
            self.last_updated = self.identified_date


@dataclass
class PooledAuditEngagement:
    """
    A pooled audit engagement with multiple client participants.

    Per Art. 30(4)(b), clients may jointly organize audits.
    """
    engagement_id: str = ""
    engagement_name: str = ""
    engagement_description: str = ""

    # Audit details
    audit_type: AuditReportType = AuditReportType.THIRD_PARTY_AUDIT
    auditor_name: str = ""
    auditor_firm: str = ""

    # Scope
    scope_areas: List[AuditScopeArea] = field(default_factory=list)
    scope_description: str = ""

    # Schedule
    status: PooledAuditStatus = PooledAuditStatus.PLANNING
    planned_start_date: str = ""
    planned_end_date: str = ""
    actual_start_date: str = ""
    actual_end_date: str = ""

    # Participants
    min_participants: int = 2
    max_participants: int = 10
    participant_ids: List[str] = field(default_factory=list)

    # Cost sharing
    total_cost: float = 0.0
    cost_sharing_model: str = "equal"  # equal, proportional, tiered
    cost_per_participant: float = 0.0

    # Findings
    finding_ids: List[str] = field(default_factory=list)
    total_findings: int = 0
    critical_findings: int = 0
    high_findings: int = 0

    # Report
    report_issued_date: str = ""
    report_path: str = ""
    report_summary_path: str = ""
    report_valid_until: str = ""

    # Tracking
    created_date: str = ""
    created_by: str = ""
    last_updated: str = ""

    def __post_init__(self):
        if not self.engagement_id:
            self.engagement_id = f"POOL-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_date:
            self.created_date = datetime.now(timezone.utc).isoformat()
        if not self.last_updated:
            self.last_updated = self.created_date


@dataclass
class AuditReportAccess:
    """
    Record of client access to an audit report.
    """
    access_id: str = ""
    client_id: str = ""
    client_name: str = ""

    # Report reference
    report_type: str = ""  # certification or pooled_audit
    report_id: str = ""

    # Access details
    nda_signed: bool = False
    nda_signed_date: str = ""
    access_granted_date: str = ""
    access_granted_by: str = ""
    access_expiry_date: str = ""

    # Usage
    first_accessed_date: str = ""
    last_accessed_date: str = ""
    access_count: int = 0

    def __post_init__(self):
        if not self.access_id:
            self.access_id = f"ACC-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class PooledAuditConfig:
    """Configuration for Pooled Audit Support."""

    # Scheduling
    annual_soc2_month: int = 10  # October
    quarterly_vuln_scan: bool = True
    annual_pentest: bool = True

    # Participation
    min_participants_for_pooled: int = 2
    max_participants: int = 20
    invitation_response_days: int = 14

    # Cost sharing
    default_cost_model: str = "equal"
    provider_contribution_pct: float = 20.0  # Provider pays 20% as organizer

    # Report access
    report_retention_years: int = 7
    nda_required_for_full_report: bool = True
    summary_available_without_nda: bool = True

    # Notifications
    notify_on_new_certification: bool = True
    notify_on_pooled_audit_invite: bool = True
    notify_on_report_available: bool = True
    expiry_warning_days: int = 30

    # Logging
    log_all_access: bool = True
    log_path: str = "logs/dora/pooled_audit"

    # Callbacks
    notification_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# Main Service Class
# =============================================================================

class PooledAuditSupport:
    """
    Pooled Audit Support service for DORA Article 30(4) compliance.

    Enables financial entity clients to:
    - Access third-party certifications and audit reports
    - Participate in pooled audits with other clients
    - Track audit findings and remediation
    """

    def __init__(self, config: Optional[PooledAuditConfig] = None):
        """Initialize Pooled Audit Support service."""
        self.config = config or PooledAuditConfig()

        # Storage
        self.certifications: Dict[str, CertificationRecord] = {}
        self.engagements: Dict[str, PooledAuditEngagement] = {}
        self.participants: Dict[str, PooledAuditParticipant] = {}
        self.findings: Dict[str, AuditFinding] = {}
        self.report_access: Dict[str, AuditReportAccess] = {}

        self._lock = __import__('threading').Lock()

        logger.info("Pooled Audit Support service initialized")

    # =========================================================================
    # Certification Management (Art. 30(4)(a))
    # =========================================================================

    def register_certification(
        self,
        certification_type: AuditReportType,
        certifying_body: str,
        issue_date: str,
        expiry_date: str,
        scope_areas: List[AuditScopeArea],
        scope_description: str = "",
        document_path: str = "",
    ) -> CertificationRecord:
        """
        Register a new certification or attestation.

        Args:
            certification_type: Type of certification
            certifying_body: Auditor/certifier name
            issue_date: Date issued
            expiry_date: Expiration date
            scope_areas: Areas covered
            scope_description: Detailed scope
            document_path: Path to report document

        Returns:
            CertificationRecord
        """
        cert = CertificationRecord(
            certification_type=certification_type,
            certifying_body=certifying_body,
            issue_date=issue_date,
            expiry_date=expiry_date,
            scope_areas=scope_areas,
            scope_description=scope_description,
            document_path=document_path,
        )

        with self._lock:
            self.certifications[cert.certification_id] = cert

        logger.info(
            f"Certification {cert.certification_id} registered: "
            f"{certification_type.value} by {certifying_body}"
        )

        if self.config.notification_callback:
            self.config.notification_callback(
                "certification_registered",
                asdict(cert)
            )

        return cert

    def get_valid_certifications(self) -> List[CertificationRecord]:
        """Get all currently valid certifications."""
        with self._lock:
            return [
                cert for cert in self.certifications.values()
                if cert.is_valid
            ]

    def get_expiring_certifications(self, days: int = 30) -> List[CertificationRecord]:
        """Get certifications expiring within specified days."""
        threshold = datetime.now(timezone.utc) + timedelta(days=days)
        expiring = []

        with self._lock:
            for cert in self.certifications.values():
                if cert.expiry_date:
                    expiry = datetime.fromisoformat(
                        cert.expiry_date.replace('Z', '+00:00')
                    )
                    if expiry <= threshold and cert.is_valid:
                        expiring.append(cert)

        return expiring

    # =========================================================================
    # Pooled Audit Management (Art. 30(4)(b))
    # =========================================================================

    def create_pooled_audit(
        self,
        name: str,
        audit_type: AuditReportType,
        scope_areas: List[AuditScopeArea],
        auditor_name: str = "",
        auditor_firm: str = "",
        planned_start_date: str = "",
        planned_end_date: str = "",
        description: str = "",
        created_by: str = "",
    ) -> PooledAuditEngagement:
        """
        Create a new pooled audit engagement.

        Args:
            name: Engagement name
            audit_type: Type of audit
            scope_areas: Areas to cover
            auditor_name: Lead auditor
            auditor_firm: Audit firm
            planned_start_date: Planned start
            planned_end_date: Planned end
            description: Detailed description
            created_by: Creator

        Returns:
            PooledAuditEngagement
        """
        engagement = PooledAuditEngagement(
            engagement_name=name,
            engagement_description=description,
            audit_type=audit_type,
            auditor_name=auditor_name,
            auditor_firm=auditor_firm,
            scope_areas=scope_areas,
            planned_start_date=planned_start_date,
            planned_end_date=planned_end_date,
            created_by=created_by,
        )

        with self._lock:
            self.engagements[engagement.engagement_id] = engagement

        logger.info(
            f"Pooled audit {engagement.engagement_id} created: {name}"
        )

        return engagement

    def invite_participant(
        self,
        engagement_id: str,
        client_id: str,
        client_name: str,
        contact_name: str,
        contact_email: str,
    ) -> PooledAuditParticipant:
        """
        Invite a client to participate in pooled audit.

        Args:
            engagement_id: Pooled audit engagement
            client_id: Client identifier
            client_name: Client name
            contact_name: Contact person
            contact_email: Contact email

        Returns:
            PooledAuditParticipant
        """
        with self._lock:
            if engagement_id not in self.engagements:
                raise ValueError(f"Engagement {engagement_id} not found")

            engagement = self.engagements[engagement_id]

            participant = PooledAuditParticipant(
                client_id=client_id,
                client_name=client_name,
                contact_name=contact_name,
                contact_email=contact_email,
                status=ParticipationStatus.INVITED,
            )

            self.participants[participant.participant_id] = participant
            engagement.participant_ids.append(participant.participant_id)
            engagement.last_updated = datetime.now(timezone.utc).isoformat()

        logger.info(
            f"Client {client_name} invited to pooled audit {engagement_id}"
        )

        if self.config.notification_callback:
            self.config.notification_callback(
                "participant_invited",
                {
                    "engagement_id": engagement_id,
                    "participant": asdict(participant),
                }
            )

        return participant

    def confirm_participation(
        self,
        participant_id: str,
        specific_requirements: Optional[List[str]] = None,
    ) -> PooledAuditParticipant:
        """
        Confirm client participation in pooled audit.

        Args:
            participant_id: Participant to confirm
            specific_requirements: Client-specific requirements

        Returns:
            Updated participant
        """
        with self._lock:
            if participant_id not in self.participants:
                raise ValueError(f"Participant {participant_id} not found")

            participant = self.participants[participant_id]
            participant.status = ParticipationStatus.CONFIRMED
            participant.confirmed_date = datetime.now(timezone.utc).isoformat()
            if specific_requirements:
                participant.specific_requirements = specific_requirements

        logger.info(f"Participation confirmed: {participant_id}")

        return participant

    def decline_participation(
        self,
        participant_id: str,
        reason: str = "",
    ) -> PooledAuditParticipant:
        """
        Decline participation in pooled audit.

        Args:
            participant_id: Participant declining
            reason: Reason for declining

        Returns:
            Updated participant
        """
        with self._lock:
            if participant_id not in self.participants:
                raise ValueError(f"Participant {participant_id} not found")

            participant = self.participants[participant_id]
            participant.status = ParticipationStatus.DECLINED

        logger.info(f"Participation declined: {participant_id}")

        return participant

    def update_engagement_status(
        self,
        engagement_id: str,
        status: PooledAuditStatus,
        actual_start_date: str = "",
        actual_end_date: str = "",
    ) -> PooledAuditEngagement:
        """
        Update pooled audit engagement status.

        Args:
            engagement_id: Engagement to update
            status: New status
            actual_start_date: Actual start if applicable
            actual_end_date: Actual end if applicable

        Returns:
            Updated engagement
        """
        with self._lock:
            if engagement_id not in self.engagements:
                raise ValueError(f"Engagement {engagement_id} not found")

            engagement = self.engagements[engagement_id]
            engagement.status = status
            if actual_start_date:
                engagement.actual_start_date = actual_start_date
            if actual_end_date:
                engagement.actual_end_date = actual_end_date
            engagement.last_updated = datetime.now(timezone.utc).isoformat()

        logger.info(f"Engagement {engagement_id} status updated to {status.value}")

        return engagement

    # =========================================================================
    # Findings Management
    # =========================================================================

    def add_finding(
        self,
        audit_id: str,
        title: str,
        description: str,
        severity: FindingSeverity,
        scope_area: AuditScopeArea,
        recommendation: str,
        dora_article_reference: str = "",
        risk_description: str = "",
    ) -> AuditFinding:
        """
        Add a finding from an audit.

        Args:
            audit_id: Source audit (certification or engagement)
            title: Finding title
            description: Finding description
            severity: Severity level
            scope_area: Affected area
            recommendation: Remediation recommendation
            dora_article_reference: Related DORA article
            risk_description: Risk if not remediated

        Returns:
            AuditFinding
        """
        finding = AuditFinding(
            audit_id=audit_id,
            title=title,
            description=description,
            severity=severity,
            scope_area=scope_area,
            recommendation=recommendation,
            dora_article_reference=dora_article_reference,
            risk_description=risk_description,
        )

        with self._lock:
            self.findings[finding.finding_id] = finding

            # Update engagement if applicable
            if audit_id in self.engagements:
                engagement = self.engagements[audit_id]
                engagement.finding_ids.append(finding.finding_id)
                engagement.total_findings += 1
                if severity == FindingSeverity.CRITICAL:
                    engagement.critical_findings += 1
                elif severity == FindingSeverity.HIGH:
                    engagement.high_findings += 1

        logger.info(f"Finding {finding.finding_id} added: {title}")

        return finding

    def update_finding_remediation(
        self,
        finding_id: str,
        status: RemediationStatus,
        owner: str = "",
        deadline: str = "",
        evidence: str = "",
    ) -> AuditFinding:
        """
        Update finding remediation status.

        Args:
            finding_id: Finding to update
            status: New remediation status
            owner: Remediation owner
            deadline: Remediation deadline
            evidence: Evidence of remediation

        Returns:
            Updated finding
        """
        with self._lock:
            if finding_id not in self.findings:
                raise ValueError(f"Finding {finding_id} not found")

            finding = self.findings[finding_id]
            finding.remediation_status = status
            if owner:
                finding.remediation_owner = owner
            if deadline:
                finding.remediation_deadline = deadline
            if evidence:
                finding.remediation_evidence = evidence
            if status in [RemediationStatus.REMEDIATED, RemediationStatus.CLOSED]:
                finding.remediation_completed_date = datetime.now(timezone.utc).isoformat()
            finding.last_updated = datetime.now(timezone.utc).isoformat()

        logger.info(f"Finding {finding_id} remediation updated: {status.value}")

        return finding

    def get_open_findings(self) -> List[AuditFinding]:
        """Get all open findings."""
        with self._lock:
            return [
                f for f in self.findings.values()
                if f.remediation_status in [
                    RemediationStatus.OPEN,
                    RemediationStatus.IN_PROGRESS
                ]
            ]

    # =========================================================================
    # Report Access Management
    # =========================================================================

    def grant_report_access(
        self,
        client_id: str,
        client_name: str,
        report_type: str,
        report_id: str,
        granted_by: str,
        nda_signed: bool = False,
        nda_signed_date: str = "",
    ) -> AuditReportAccess:
        """
        Grant client access to an audit report.

        Args:
            client_id: Client identifier
            client_name: Client name
            report_type: Type of report (certification/pooled_audit)
            report_id: Report identifier
            granted_by: Who granted access
            nda_signed: Whether NDA is signed
            nda_signed_date: Date NDA signed

        Returns:
            AuditReportAccess record
        """
        access = AuditReportAccess(
            client_id=client_id,
            client_name=client_name,
            report_type=report_type,
            report_id=report_id,
            nda_signed=nda_signed,
            nda_signed_date=nda_signed_date,
            access_granted_date=datetime.now(timezone.utc).isoformat(),
            access_granted_by=granted_by,
            access_expiry_date=(
                datetime.now(timezone.utc) +
                timedelta(days=365)  # 1 year access
            ).isoformat(),
        )

        with self._lock:
            self.report_access[access.access_id] = access

        logger.info(
            f"Report access granted: {client_name} -> {report_type}/{report_id}"
        )

        if self.config.notification_callback:
            self.config.notification_callback(
                "report_access_granted",
                asdict(access)
            )

        return access

    def record_report_access(self, access_id: str) -> AuditReportAccess:
        """
        Record that a client accessed a report.

        Args:
            access_id: Access record

        Returns:
            Updated access record
        """
        with self._lock:
            if access_id not in self.report_access:
                raise ValueError(f"Access record {access_id} not found")

            access = self.report_access[access_id]
            now = datetime.now(timezone.utc).isoformat()
            if not access.first_accessed_date:
                access.first_accessed_date = now
            access.last_accessed_date = now
            access.access_count += 1

        return access

    def get_client_report_access(self, client_id: str) -> List[AuditReportAccess]:
        """Get all report access records for a client."""
        with self._lock:
            return [
                a for a in self.report_access.values()
                if a.client_id == client_id
            ]

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_available_reports(self, client_id: str) -> Dict[str, Any]:
        """
        Get all reports available to a client.

        Args:
            client_id: Client identifier

        Returns:
            Dict with available certifications and audit reports
        """
        client_access = self.get_client_report_access(client_id)
        access_report_ids = {a.report_id for a in client_access}

        valid_certs = self.get_valid_certifications()

        return {
            "client_id": client_id,
            "certifications": [
                {
                    "id": c.certification_id,
                    "type": c.certification_type.value,
                    "certifier": c.certifying_body,
                    "valid_until": c.expiry_date,
                    "has_access": c.certification_id in access_report_ids,
                    "summary_available": c.summary_available,
                    "nda_required": c.nda_required,
                }
                for c in valid_certs
            ],
            "pooled_audits": [
                {
                    "id": e.engagement_id,
                    "name": e.engagement_name,
                    "type": e.audit_type.value,
                    "status": e.status.value,
                    "has_access": e.engagement_id in access_report_ids,
                    "report_available": bool(e.report_issued_date),
                }
                for e in self.engagements.values()
                if e.status == PooledAuditStatus.COMPLETED
            ],
        }

    def generate_pooled_audit_summary(self) -> Dict[str, Any]:
        """
        Generate summary of pooled audit program.

        Returns:
            Summary statistics
        """
        valid_certs = self.get_valid_certifications()
        expiring_certs = self.get_expiring_certifications()
        open_findings = self.get_open_findings()

        active_engagements = [
            e for e in self.engagements.values()
            if e.status not in [
                PooledAuditStatus.COMPLETED,
                PooledAuditStatus.CANCELLED
            ]
        ]

        return {
            "report_date": datetime.now(timezone.utc).isoformat(),
            "certifications": {
                "total": len(self.certifications),
                "valid": len(valid_certs),
                "expiring_soon": len(expiring_certs),
                "types": [c.certification_type.value for c in valid_certs],
            },
            "pooled_audits": {
                "total": len(self.engagements),
                "active": len(active_engagements),
                "completed": len([
                    e for e in self.engagements.values()
                    if e.status == PooledAuditStatus.COMPLETED
                ]),
            },
            "findings": {
                "total": len(self.findings),
                "open": len(open_findings),
                "critical_open": len([
                    f for f in open_findings
                    if f.severity == FindingSeverity.CRITICAL
                ]),
                "high_open": len([
                    f for f in open_findings
                    if f.severity == FindingSeverity.HIGH
                ]),
            },
            "report_access": {
                "total_grants": len(self.report_access),
                "unique_clients": len({a.client_id for a in self.report_access.values()}),
            },
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_pooled_audit_support(
    config: Optional[PooledAuditConfig] = None,
) -> PooledAuditSupport:
    """
    Create Pooled Audit Support service instance.

    Args:
        config: Optional configuration

    Returns:
        Configured PooledAuditSupport instance
    """
    return PooledAuditSupport(config=config)


def get_audit_scope_areas() -> List[str]:
    """Get list of audit scope areas."""
    return [area.value for area in AuditScopeArea]


def get_report_types() -> List[str]:
    """Get list of report types."""
    return [rt.value for rt in AuditReportType]
