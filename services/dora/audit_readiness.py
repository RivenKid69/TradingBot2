# -*- coding: utf-8 -*-
"""
DORA Audit Readiness Module.

For ICT Third-Party Service Providers: Implements audit and inspection support
per DORA Article 30(3)(d) and 30(3)(e).

DORA Context:
    - Financial entity clients have audit rights (Art. 30(3)(d))
    - NCAs have inspection rights (Art. 30(3)(e))
    - We must support both without impeding supervision
    - Evidence must be readily available and verifiable

Supported Audit Types:
    - Client operational audits
    - Client security assessments
    - NCA inspections (direct or via client's NCA)
    - Third-party auditor assessments (pooled audits)
    - SOC2/ISO27001 certification audits

References:
    - DORA Article 30(3)(d): Financial entity audit rights
    - DORA Article 30(3)(e): NCA access and inspection rights
    - DORA Article 28(6): No impediment to supervision
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class AuditType(Enum):
    """Types of audits we support."""
    CLIENT_OPERATIONAL = "client_operational"
    CLIENT_SECURITY = "client_security"
    NCA_INSPECTION = "nca_inspection"
    THIRD_PARTY = "third_party"  # Pooled audit by external auditor
    CERTIFICATION = "certification"  # SOC2, ISO27001, etc.
    INTERNAL = "internal"


class AuditScope(Enum):
    """Scope of audit coverage."""
    FULL = "full"  # Complete system review
    SECURITY = "security"  # Security controls only
    AVAILABILITY = "availability"  # BCP/DR only
    DATA_PROTECTION = "data_protection"  # GDPR/data handling
    SPECIFIC_INCIDENT = "specific_incident"  # Post-incident review
    CONTROLS_TESTING = "controls_testing"  # Control effectiveness


class AuditStatus(Enum):
    """Audit request status."""
    REQUESTED = "requested"
    ACKNOWLEDGED = "acknowledged"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    EVIDENCE_SUBMITTED = "evidence_submitted"
    FINDINGS_RECEIVED = "findings_received"
    REMEDIATION_PLANNED = "remediation_planned"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class EvidenceType(Enum):
    """Types of audit evidence."""
    DOCUMENT = "document"  # Policies, procedures
    LOG = "log"  # System logs, audit trails
    CONFIGURATION = "configuration"  # System configs
    SCREENSHOT = "screenshot"  # UI evidence
    REPORT = "report"  # Generated reports
    ATTESTATION = "attestation"  # Signed statements
    INTERVIEW = "interview"  # Interview notes
    TEST_RESULT = "test_result"  # Penetration test, DR test


class EvidenceCategory(Enum):
    """DORA-aligned evidence categories."""
    ICT_GOVERNANCE = "ict_governance"
    ICT_RISK_MANAGEMENT = "ict_risk_management"
    ICT_SECURITY = "ict_security"
    ICT_OPERATIONS = "ict_operations"
    ICT_CONTINUITY = "ict_continuity"
    ICT_TESTING = "ict_testing"
    INCIDENT_MANAGEMENT = "incident_management"
    THIRD_PARTY_MANAGEMENT = "third_party_management"
    CHANGE_MANAGEMENT = "change_management"
    ACCESS_CONTROL = "access_control"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class AuditRequest:
    """Audit request from client or regulator."""
    request_id: str = ""
    request_date: str = ""
    requesting_entity: str = ""  # Client name or NCA
    requesting_entity_type: str = ""  # client, nca, third_party

    # Audit details
    audit_type: AuditType = AuditType.CLIENT_OPERATIONAL
    audit_scope: AuditScope = AuditScope.FULL
    audit_title: str = ""
    audit_description: str = ""

    # Scheduling
    proposed_start_date: str = ""
    proposed_end_date: str = ""
    confirmed_start_date: str = ""
    confirmed_end_date: str = ""

    # Participants
    auditor_organization: str = ""
    auditor_contacts: List[str] = field(default_factory=list)
    our_audit_coordinator: str = ""
    our_participants: List[str] = field(default_factory=list)

    # Requirements
    evidence_requested: List[str] = field(default_factory=list)
    access_requirements: List[str] = field(default_factory=list)
    on_site_required: bool = False

    # Status
    status: AuditStatus = AuditStatus.REQUESTED
    acknowledgment_date: str = ""
    completion_date: str = ""

    # SLA tracking
    sla_deadline: str = ""  # Response SLA
    sla_met: Optional[bool] = None

    # Notes
    notes: str = ""

    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"AUD-{uuid.uuid4().hex[:8].upper()}"
        if not self.request_date:
            self.request_date = datetime.now(timezone.utc).isoformat()


@dataclass
class EvidenceItem:
    """Audit evidence item."""
    evidence_id: str = ""
    audit_id: str = ""

    # Evidence details
    evidence_type: EvidenceType = EvidenceType.DOCUMENT
    category: EvidenceCategory = EvidenceCategory.ICT_SECURITY
    title: str = ""
    description: str = ""

    # Source
    source_system: str = ""
    source_path: str = ""
    generated_date: str = ""

    # Coverage
    coverage_period_start: str = ""
    coverage_period_end: str = ""

    # File info
    file_path: str = ""
    file_format: str = ""
    file_size_bytes: int = 0
    file_hash: str = ""  # SHA256 for integrity

    # Verification
    prepared_by: str = ""
    verified_by: str = ""
    verification_date: str = ""

    # Delivery
    delivered_date: str = ""
    delivery_method: str = ""  # secure_portal, encrypted_email, in_person

    def __post_init__(self):
        if not self.evidence_id:
            self.evidence_id = f"EVD-{uuid.uuid4().hex[:8].upper()}"
        if not self.generated_date:
            self.generated_date = datetime.now(timezone.utc).isoformat()


@dataclass
class AuditFinding:
    """Finding from an audit."""
    finding_id: str = ""
    audit_id: str = ""

    # Finding details
    severity: str = ""  # critical, high, medium, low, observation
    category: EvidenceCategory = EvidenceCategory.ICT_SECURITY
    title: str = ""
    description: str = ""
    recommendation: str = ""

    # Reference
    dora_article_reference: str = ""
    control_reference: str = ""

    # Remediation
    remediation_plan: str = ""
    remediation_owner: str = ""
    remediation_deadline: str = ""
    remediation_status: str = ""  # not_started, in_progress, completed, deferred
    remediation_evidence: str = ""

    # Tracking
    identified_date: str = ""
    acknowledged_date: str = ""
    completed_date: str = ""

    def __post_init__(self):
        if not self.finding_id:
            self.finding_id = f"FND-{uuid.uuid4().hex[:8].upper()}"
        if not self.identified_date:
            self.identified_date = datetime.now(timezone.utc).isoformat()


@dataclass
class EvidenceTemplate:
    """Template for generating standard evidence."""
    template_id: str = ""
    template_name: str = ""
    category: EvidenceCategory = EvidenceCategory.ICT_SECURITY
    evidence_type: EvidenceType = EvidenceType.DOCUMENT

    # Content
    description: str = ""
    source_systems: List[str] = field(default_factory=list)
    generation_method: str = ""  # manual, automated, query

    # For automated generation
    query_template: str = ""
    data_sources: List[str] = field(default_factory=list)

    # Standard coverage
    default_period_days: int = 365

    def __post_init__(self):
        if not self.template_id:
            self.template_id = f"TPL-{uuid.uuid4().hex[:8].upper()}"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AuditReadinessConfig:
    """Configuration for audit readiness system."""

    # SLA settings
    acknowledgment_sla_business_days: int = 2
    scheduling_sla_business_days: int = 5
    evidence_delivery_sla_days: int = 10

    # Access settings
    allow_remote_access: bool = True
    allow_on_site_access: bool = True
    require_nda_for_access: bool = True

    # Evidence settings
    evidence_retention_years: int = 7
    auto_generate_standard_evidence: bool = True

    # Notifications
    notify_on_request: bool = True
    notify_on_finding: bool = True

    # Storage
    evidence_path: str = "data/dora/audit_evidence"
    log_path: str = "logs/dora/audits"


# =============================================================================
# Standard Evidence Templates
# =============================================================================

def get_standard_evidence_templates() -> List[EvidenceTemplate]:
    """Get standard evidence templates for common audit requests."""
    return [
        # ICT Governance
        EvidenceTemplate(
            template_name="ICT Strategy Document",
            category=EvidenceCategory.ICT_GOVERNANCE,
            evidence_type=EvidenceType.DOCUMENT,
            description="ICT strategy and digital operational resilience objectives",
            generation_method="manual",
        ),
        EvidenceTemplate(
            template_name="ICT Governance Structure",
            category=EvidenceCategory.ICT_GOVERNANCE,
            evidence_type=EvidenceType.DOCUMENT,
            description="Organization chart with ICT responsibilities",
            generation_method="manual",
        ),

        # ICT Risk Management
        EvidenceTemplate(
            template_name="ICT Risk Register",
            category=EvidenceCategory.ICT_RISK_MANAGEMENT,
            evidence_type=EvidenceType.REPORT,
            description="Current ICT risk register with assessments",
            generation_method="automated",
            query_template="SELECT * FROM risk_register WHERE status = 'active'",
        ),
        EvidenceTemplate(
            template_name="Risk Assessment Report",
            category=EvidenceCategory.ICT_RISK_MANAGEMENT,
            evidence_type=EvidenceType.REPORT,
            description="Annual ICT risk assessment results",
            generation_method="manual",
        ),

        # ICT Security
        EvidenceTemplate(
            template_name="Security Policy",
            category=EvidenceCategory.ICT_SECURITY,
            evidence_type=EvidenceType.DOCUMENT,
            description="Information security policy",
            generation_method="manual",
        ),
        EvidenceTemplate(
            template_name="Vulnerability Scan Results",
            category=EvidenceCategory.ICT_SECURITY,
            evidence_type=EvidenceType.REPORT,
            description="Recent vulnerability scan reports",
            generation_method="automated",
            default_period_days=90,
        ),
        EvidenceTemplate(
            template_name="Penetration Test Report",
            category=EvidenceCategory.ICT_SECURITY,
            evidence_type=EvidenceType.REPORT,
            description="Annual penetration test results and remediation",
            generation_method="manual",
        ),

        # Access Control
        EvidenceTemplate(
            template_name="Access Control Policy",
            category=EvidenceCategory.ACCESS_CONTROL,
            evidence_type=EvidenceType.DOCUMENT,
            description="Access control and authentication policy",
            generation_method="manual",
        ),
        EvidenceTemplate(
            template_name="User Access Review",
            category=EvidenceCategory.ACCESS_CONTROL,
            evidence_type=EvidenceType.REPORT,
            description="Quarterly user access review results",
            generation_method="automated",
            default_period_days=90,
        ),
        EvidenceTemplate(
            template_name="Privileged Access Log",
            category=EvidenceCategory.ACCESS_CONTROL,
            evidence_type=EvidenceType.LOG,
            description="Privileged account activity logs",
            generation_method="automated",
        ),

        # ICT Continuity
        EvidenceTemplate(
            template_name="Business Continuity Plan",
            category=EvidenceCategory.ICT_CONTINUITY,
            evidence_type=EvidenceType.DOCUMENT,
            description="ICT business continuity plan",
            generation_method="manual",
        ),
        EvidenceTemplate(
            template_name="DR Test Results",
            category=EvidenceCategory.ICT_CONTINUITY,
            evidence_type=EvidenceType.TEST_RESULT,
            description="Disaster recovery test results",
            generation_method="manual",
        ),
        EvidenceTemplate(
            template_name="Backup Verification Logs",
            category=EvidenceCategory.ICT_CONTINUITY,
            evidence_type=EvidenceType.LOG,
            description="Backup completion and verification records",
            generation_method="automated",
        ),

        # Incident Management
        EvidenceTemplate(
            template_name="Incident Management Policy",
            category=EvidenceCategory.INCIDENT_MANAGEMENT,
            evidence_type=EvidenceType.DOCUMENT,
            description="ICT incident management procedures",
            generation_method="manual",
        ),
        EvidenceTemplate(
            template_name="Incident Log",
            category=EvidenceCategory.INCIDENT_MANAGEMENT,
            evidence_type=EvidenceType.LOG,
            description="ICT incident records and resolution",
            generation_method="automated",
        ),
        EvidenceTemplate(
            template_name="Major Incident Reports",
            category=EvidenceCategory.INCIDENT_MANAGEMENT,
            evidence_type=EvidenceType.REPORT,
            description="Post-incident review reports for major incidents",
            generation_method="manual",
        ),

        # Change Management
        EvidenceTemplate(
            template_name="Change Management Policy",
            category=EvidenceCategory.CHANGE_MANAGEMENT,
            evidence_type=EvidenceType.DOCUMENT,
            description="ICT change management procedures",
            generation_method="manual",
        ),
        EvidenceTemplate(
            template_name="Change Log",
            category=EvidenceCategory.CHANGE_MANAGEMENT,
            evidence_type=EvidenceType.LOG,
            description="Production change records",
            generation_method="automated",
        ),

        # Third-Party Management
        EvidenceTemplate(
            template_name="Vendor Management Policy",
            category=EvidenceCategory.THIRD_PARTY_MANAGEMENT,
            evidence_type=EvidenceType.DOCUMENT,
            description="Third-party/subcontractor management policy",
            generation_method="manual",
        ),
        EvidenceTemplate(
            template_name="Subcontractor Register",
            category=EvidenceCategory.THIRD_PARTY_MANAGEMENT,
            evidence_type=EvidenceType.REPORT,
            description="List of subcontractors with risk assessments",
            generation_method="automated",
        ),

        # ICT Testing
        EvidenceTemplate(
            template_name="Testing Strategy",
            category=EvidenceCategory.ICT_TESTING,
            evidence_type=EvidenceType.DOCUMENT,
            description="ICT testing and resilience testing strategy",
            generation_method="manual",
        ),
        EvidenceTemplate(
            template_name="Test Results Summary",
            category=EvidenceCategory.ICT_TESTING,
            evidence_type=EvidenceType.REPORT,
            description="Summary of resilience and security testing",
            generation_method="automated",
        ),
    ]


# =============================================================================
# Main Implementation
# =============================================================================

class DORAuditReadiness:
    """
    DORA Audit Readiness System.

    Manages audit requests, evidence generation, and finding remediation
    to support client and NCA audit rights per DORA Article 30.

    Key Features:
    - Audit request tracking with SLA monitoring
    - Standard evidence template library
    - Automated evidence generation
    - Finding and remediation tracking
    - Audit history and reporting

    Usage:
        config = AuditReadinessConfig()
        audit_system = DORAuditReadiness(config)

        # Receive audit request
        request = audit_system.create_audit_request(
            requesting_entity="Bank XYZ",
            audit_type=AuditType.CLIENT_OPERATIONAL,
            ...
        )

        # Acknowledge and schedule
        audit_system.acknowledge_request(request.request_id)
        audit_system.schedule_audit(request.request_id, start_date, end_date)

        # Generate evidence package
        evidence = audit_system.generate_evidence_package(request.request_id)
    """

    def __init__(self, config: Optional[AuditReadinessConfig] = None):
        """Initialize audit readiness system."""
        self.config = config or AuditReadinessConfig()

        # Data stores
        self._requests: Dict[str, AuditRequest] = {}
        self._evidence: Dict[str, EvidenceItem] = {}
        self._findings: Dict[str, AuditFinding] = {}
        self._templates: Dict[str, EvidenceTemplate] = {}

        # Indexes
        self._evidence_by_audit: Dict[str, Set[str]] = {}
        self._findings_by_audit: Dict[str, Set[str]] = {}

        # Load standard templates
        for template in get_standard_evidence_templates():
            self._templates[template.template_id] = template

        # Setup directories
        self._evidence_path = Path(self.config.evidence_path)
        self._evidence_path.mkdir(parents=True, exist_ok=True)
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("DORAuditReadiness initialized")

    # =========================================================================
    # Audit Request Management
    # =========================================================================

    def create_audit_request(
        self,
        requesting_entity: str,
        audit_type: AuditType,
        audit_scope: AuditScope = AuditScope.FULL,
        audit_title: str = "",
        audit_description: str = "",
        proposed_start_date: str = "",
        proposed_end_date: str = "",
        auditor_organization: str = "",
        on_site_required: bool = False,
        requesting_entity_type: str = "client",
    ) -> AuditRequest:
        """
        Create a new audit request.

        Args:
            requesting_entity: Name of requesting client or NCA
            audit_type: Type of audit
            audit_scope: Scope of audit
            audit_title: Title/name of audit
            audit_description: Description of audit objectives
            proposed_start_date: Proposed start date
            proposed_end_date: Proposed end date
            auditor_organization: Organization performing audit
            on_site_required: Whether on-site access needed
            requesting_entity_type: Type of requester (client, nca, third_party)

        Returns:
            Created AuditRequest
        """
        request = AuditRequest(
            requesting_entity=requesting_entity,
            requesting_entity_type=requesting_entity_type,
            audit_type=audit_type,
            audit_scope=audit_scope,
            audit_title=audit_title or f"{audit_type.value} Audit - {requesting_entity}",
            audit_description=audit_description,
            proposed_start_date=proposed_start_date,
            proposed_end_date=proposed_end_date,
            auditor_organization=auditor_organization or requesting_entity,
            on_site_required=on_site_required,
            status=AuditStatus.REQUESTED,
        )

        # Calculate SLA deadline
        sla_deadline = datetime.now(timezone.utc) + timedelta(
            days=self.config.acknowledgment_sla_business_days
        )
        request.sla_deadline = sla_deadline.isoformat()

        self._requests[request.request_id] = request
        self._evidence_by_audit[request.request_id] = set()
        self._findings_by_audit[request.request_id] = set()

        self._log_event("audit_request_created", {
            "request_id": request.request_id,
            "requesting_entity": requesting_entity,
            "audit_type": audit_type.value,
        })

        return request

    def acknowledge_request(
        self,
        request_id: str,
        coordinator: str = "",
        notes: str = "",
    ) -> Optional[AuditRequest]:
        """Acknowledge receipt of audit request."""
        if request_id not in self._requests:
            return None

        request = self._requests[request_id]
        request.status = AuditStatus.ACKNOWLEDGED
        request.acknowledgment_date = datetime.now(timezone.utc).isoformat()
        request.our_audit_coordinator = coordinator
        if notes:
            request.notes = notes

        # Check SLA
        ack_time = datetime.fromisoformat(request.acknowledgment_date.replace("Z", "+00:00"))
        deadline = datetime.fromisoformat(request.sla_deadline.replace("Z", "+00:00"))
        request.sla_met = ack_time <= deadline

        self._log_event("audit_acknowledged", {
            "request_id": request_id,
            "sla_met": request.sla_met,
        })

        return request

    def schedule_audit(
        self,
        request_id: str,
        start_date: str,
        end_date: str,
        participants: Optional[List[str]] = None,
    ) -> Optional[AuditRequest]:
        """Schedule audit dates."""
        if request_id not in self._requests:
            return None

        request = self._requests[request_id]
        request.status = AuditStatus.SCHEDULED
        request.confirmed_start_date = start_date
        request.confirmed_end_date = end_date
        if participants:
            request.our_participants = participants

        self._log_event("audit_scheduled", {
            "request_id": request_id,
            "start_date": start_date,
            "end_date": end_date,
        })

        return request

    def start_audit(self, request_id: str) -> Optional[AuditRequest]:
        """Mark audit as in progress."""
        if request_id not in self._requests:
            return None

        request = self._requests[request_id]
        request.status = AuditStatus.IN_PROGRESS
        return request

    def complete_audit(
        self,
        request_id: str,
        notes: str = "",
    ) -> Optional[AuditRequest]:
        """Mark audit as completed."""
        if request_id not in self._requests:
            return None

        request = self._requests[request_id]
        request.status = AuditStatus.COMPLETED
        request.completion_date = datetime.now(timezone.utc).isoformat()
        if notes:
            request.notes = notes

        self._log_event("audit_completed", {
            "request_id": request_id,
            "findings_count": len(self._findings_by_audit.get(request_id, set())),
        })

        return request

    def get_request(self, request_id: str) -> Optional[AuditRequest]:
        """Get audit request by ID."""
        return self._requests.get(request_id)

    def get_all_requests(
        self,
        status_filter: Optional[List[AuditStatus]] = None,
    ) -> List[AuditRequest]:
        """Get all audit requests."""
        requests = list(self._requests.values())
        if status_filter:
            requests = [r for r in requests if r.status in status_filter]
        return requests

    def get_active_audits(self) -> List[AuditRequest]:
        """Get currently active audits."""
        active_statuses = [
            AuditStatus.REQUESTED,
            AuditStatus.ACKNOWLEDGED,
            AuditStatus.SCHEDULED,
            AuditStatus.IN_PROGRESS,
            AuditStatus.EVIDENCE_SUBMITTED,
        ]
        return self.get_all_requests(status_filter=active_statuses)

    # =========================================================================
    # Evidence Management
    # =========================================================================

    def add_evidence(
        self,
        audit_id: str,
        evidence_type: EvidenceType,
        category: EvidenceCategory,
        title: str,
        description: str = "",
        file_path: str = "",
        prepared_by: str = "",
        coverage_start: str = "",
        coverage_end: str = "",
    ) -> Optional[EvidenceItem]:
        """Add evidence item to an audit."""
        if audit_id not in self._requests:
            return None

        evidence = EvidenceItem(
            audit_id=audit_id,
            evidence_type=evidence_type,
            category=category,
            title=title,
            description=description,
            file_path=file_path,
            prepared_by=prepared_by,
            coverage_period_start=coverage_start,
            coverage_period_end=coverage_end,
        )

        self._evidence[evidence.evidence_id] = evidence
        self._evidence_by_audit[audit_id].add(evidence.evidence_id)

        return evidence

    def generate_evidence_package(
        self,
        request_id: str,
        categories: Optional[List[EvidenceCategory]] = None,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate standard evidence package for an audit.

        Args:
            request_id: Audit request ID
            categories: Categories to include (all if None)
            period_start: Coverage period start
            period_end: Coverage period end

        Returns:
            Dict with evidence package details
        """
        if request_id not in self._requests:
            return {"error": "Request not found"}

        request = self._requests[request_id]
        package = {
            "request_id": request_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "requesting_entity": request.requesting_entity,
            "audit_type": request.audit_type.value,
            "evidence_items": [],
        }

        # Get applicable templates
        templates = list(self._templates.values())
        if categories:
            templates = [t for t in templates if t.category in categories]

        # Generate evidence from templates
        for template in templates:
            evidence = self._generate_from_template(
                request_id=request_id,
                template=template,
                period_start=period_start,
                period_end=period_end,
            )
            if evidence:
                package["evidence_items"].append({
                    "evidence_id": evidence.evidence_id,
                    "title": evidence.title,
                    "category": evidence.category.value,
                    "type": evidence.evidence_type.value,
                })

        # Update request status
        request.status = AuditStatus.EVIDENCE_SUBMITTED

        self._log_event("evidence_package_generated", {
            "request_id": request_id,
            "items_count": len(package["evidence_items"]),
        })

        return package

    def _generate_from_template(
        self,
        request_id: str,
        template: EvidenceTemplate,
        period_start: Optional[str],
        period_end: Optional[str],
    ) -> Optional[EvidenceItem]:
        """Generate evidence from template."""
        # Set default period
        if not period_end:
            period_end = datetime.now(timezone.utc).isoformat()
        if not period_start:
            period_start = (
                datetime.now(timezone.utc) - timedelta(days=template.default_period_days)
            ).isoformat()

        evidence = EvidenceItem(
            audit_id=request_id,
            evidence_type=template.evidence_type,
            category=template.category,
            title=template.template_name,
            description=template.description,
            source_system=", ".join(template.source_systems),
            coverage_period_start=period_start,
            coverage_period_end=period_end,
        )

        self._evidence[evidence.evidence_id] = evidence
        self._evidence_by_audit[request_id].add(evidence.evidence_id)

        return evidence

    def get_evidence_for_audit(self, audit_id: str) -> List[EvidenceItem]:
        """Get all evidence for an audit."""
        evidence_ids = self._evidence_by_audit.get(audit_id, set())
        return [self._evidence[eid] for eid in evidence_ids if eid in self._evidence]

    def verify_evidence(
        self,
        evidence_id: str,
        verified_by: str,
    ) -> Optional[EvidenceItem]:
        """Mark evidence as verified."""
        if evidence_id not in self._evidence:
            return None

        evidence = self._evidence[evidence_id]
        evidence.verified_by = verified_by
        evidence.verification_date = datetime.now(timezone.utc).isoformat()
        return evidence

    def deliver_evidence(
        self,
        evidence_id: str,
        delivery_method: str,
    ) -> Optional[EvidenceItem]:
        """Mark evidence as delivered."""
        if evidence_id not in self._evidence:
            return None

        evidence = self._evidence[evidence_id]
        evidence.delivered_date = datetime.now(timezone.utc).isoformat()
        evidence.delivery_method = delivery_method
        return evidence

    # =========================================================================
    # Finding Management
    # =========================================================================

    def add_finding(
        self,
        audit_id: str,
        severity: str,
        category: EvidenceCategory,
        title: str,
        description: str,
        recommendation: str = "",
        dora_reference: str = "",
    ) -> Optional[AuditFinding]:
        """Record a finding from an audit."""
        if audit_id not in self._requests:
            return None

        finding = AuditFinding(
            audit_id=audit_id,
            severity=severity,
            category=category,
            title=title,
            description=description,
            recommendation=recommendation,
            dora_article_reference=dora_reference,
            remediation_status="not_started",
        )

        self._findings[finding.finding_id] = finding
        self._findings_by_audit[audit_id].add(finding.finding_id)

        # Update request status
        request = self._requests[audit_id]
        if request.status == AuditStatus.EVIDENCE_SUBMITTED:
            request.status = AuditStatus.FINDINGS_RECEIVED

        self._log_event("finding_added", {
            "finding_id": finding.finding_id,
            "audit_id": audit_id,
            "severity": severity,
        })

        return finding

    def plan_remediation(
        self,
        finding_id: str,
        remediation_plan: str,
        owner: str,
        deadline: str,
    ) -> Optional[AuditFinding]:
        """Plan remediation for a finding."""
        if finding_id not in self._findings:
            return None

        finding = self._findings[finding_id]
        finding.remediation_plan = remediation_plan
        finding.remediation_owner = owner
        finding.remediation_deadline = deadline
        finding.remediation_status = "in_progress"
        finding.acknowledged_date = datetime.now(timezone.utc).isoformat()

        return finding

    def complete_remediation(
        self,
        finding_id: str,
        evidence: str = "",
    ) -> Optional[AuditFinding]:
        """Mark remediation as completed."""
        if finding_id not in self._findings:
            return None

        finding = self._findings[finding_id]
        finding.remediation_status = "completed"
        finding.completed_date = datetime.now(timezone.utc).isoformat()
        finding.remediation_evidence = evidence

        self._log_event("remediation_completed", {
            "finding_id": finding_id,
            "audit_id": finding.audit_id,
        })

        return finding

    def get_findings_for_audit(self, audit_id: str) -> List[AuditFinding]:
        """Get all findings for an audit."""
        finding_ids = self._findings_by_audit.get(audit_id, set())
        return [self._findings[fid] for fid in finding_ids if fid in self._findings]

    def get_open_findings(self) -> List[AuditFinding]:
        """Get all open (unremediated) findings."""
        return [
            f for f in self._findings.values()
            if f.remediation_status not in ["completed", "deferred"]
        ]

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_audit_summary(self) -> Dict[str, Any]:
        """Get summary of audit activities."""
        all_requests = list(self._requests.values())
        all_findings = list(self._findings.values())

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "requests": {
                "total": len(all_requests),
                "by_status": {
                    status.value: sum(1 for r in all_requests if r.status == status)
                    for status in AuditStatus
                },
                "by_type": {
                    audit_type.value: sum(1 for r in all_requests if r.audit_type == audit_type)
                    for audit_type in AuditType
                },
                "active": len(self.get_active_audits()),
            },
            "findings": {
                "total": len(all_findings),
                "open": len(self.get_open_findings()),
                "by_severity": {
                    severity: sum(1 for f in all_findings if f.severity == severity)
                    for severity in ["critical", "high", "medium", "low", "observation"]
                },
            },
            "evidence": {
                "total_items": len(self._evidence),
            },
            "templates": {
                "total": len(self._templates),
            },
        }

    def generate_audit_report(
        self,
        audit_id: str,
    ) -> Dict[str, Any]:
        """Generate comprehensive report for an audit."""
        if audit_id not in self._requests:
            return {"error": "Audit not found"}

        request = self._requests[audit_id]
        evidence = self.get_evidence_for_audit(audit_id)
        findings = self.get_findings_for_audit(audit_id)

        return {
            "report_generated": datetime.now(timezone.utc).isoformat(),
            "audit": {
                "request_id": request.request_id,
                "type": request.audit_type.value,
                "scope": request.audit_scope.value,
                "requesting_entity": request.requesting_entity,
                "status": request.status.value,
                "request_date": request.request_date,
                "completion_date": request.completion_date,
            },
            "timeline": {
                "requested": request.request_date,
                "acknowledged": request.acknowledgment_date,
                "scheduled_start": request.confirmed_start_date,
                "scheduled_end": request.confirmed_end_date,
                "completed": request.completion_date,
                "sla_met": request.sla_met,
            },
            "evidence_summary": {
                "total_items": len(evidence),
                "by_category": {
                    cat.value: sum(1 for e in evidence if e.category == cat)
                    for cat in EvidenceCategory
                },
                "items": [
                    {
                        "id": e.evidence_id,
                        "title": e.title,
                        "category": e.category.value,
                        "delivered": e.delivered_date is not None,
                    }
                    for e in evidence
                ],
            },
            "findings_summary": {
                "total": len(findings),
                "by_severity": {
                    sev: sum(1 for f in findings if f.severity == sev)
                    for sev in ["critical", "high", "medium", "low"]
                },
                "remediation_status": {
                    status: sum(1 for f in findings if f.remediation_status == status)
                    for status in ["not_started", "in_progress", "completed", "deferred"]
                },
                "findings": [
                    {
                        "id": f.finding_id,
                        "severity": f.severity,
                        "title": f.title,
                        "status": f.remediation_status,
                    }
                    for f in findings
                ],
            },
        }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"audits_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_audit_readiness(
    config: Optional[AuditReadinessConfig] = None,
) -> DORAuditReadiness:
    """
    Create a DORAuditReadiness instance.

    Args:
        config: Optional configuration

    Returns:
        Configured DORAuditReadiness instance
    """
    return DORAuditReadiness(config=config)
