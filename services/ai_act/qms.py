# -*- coding: utf-8 -*-
"""
AI Act Quality Management System (Article 17).

EU AI Act Article 17 requires providers of high-risk AI systems to put
a quality management system in place that ensures compliance.

This module provides:
1. QualityManagementSystem - Main QMS orchestrator
2. QMSProcedure - Procedure documentation and tracking
3. QMSAudit - Internal audit management
4. DesignControl - Design review and control
5. ChangeManagement - Change control processes
6. CorrectiveAction - CAPA (Corrective and Preventive Actions)

Article 17 Requirements Implemented:
    - (a) Strategy for regulatory compliance
    - (b) Techniques for design, design control, design verification
    - (c) Techniques for development, quality control, quality assurance
    - (d) Examination, test and validation procedures
    - (e) Data management procedures
    - (f) Risk management system linkage
    - (g) Post-market monitoring system
    - (h) Serious incident and malfunctioning reporting
    - (i) Communication with competent authorities
    - (j) Systems and procedures for record keeping
    - (k) Resource management and supply chain
    - (l) Accountability framework

References:
    - Article 17: https://artificialintelligenceact.eu/article/17/
    - ISO 9001:2015: Quality management principles
    - ISO 13485:2016: Medical devices QMS (reference framework)
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class QMSElementType(Enum):
    """
    QMS elements as defined by EU AI Act Article 17(1).

    Each element corresponds to a specific requirement in the regulation.
    """
    REGULATORY_COMPLIANCE = "regulatory_compliance"  # (a)
    DESIGN_CONTROL = "design_control"  # (b)
    DEVELOPMENT_QA = "development_qa"  # (c)
    TESTING_VALIDATION = "testing_validation"  # (d)
    DATA_MANAGEMENT = "data_management"  # (e)
    RISK_MANAGEMENT = "risk_management"  # (f)
    POST_MARKET_MONITORING = "post_market_monitoring"  # (g)
    INCIDENT_REPORTING = "incident_reporting"  # (h)
    AUTHORITY_COMMUNICATION = "authority_communication"  # (i)
    RECORD_KEEPING = "record_keeping"  # (j)
    RESOURCE_MANAGEMENT = "resource_management"  # (k)
    ACCOUNTABILITY = "accountability"  # (l)


class ProcedureStatus(Enum):
    """Status of QMS procedures."""
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    EFFECTIVE = "effective"
    SUPERSEDED = "superseded"
    RETIRED = "retired"


class AuditType(Enum):
    """Types of QMS audits."""
    INTERNAL = "internal"
    EXTERNAL = "external"
    REGULATORY = "regulatory"
    SUPPLIER = "supplier"
    CUSTOMER = "customer"


class AuditStatus(Enum):
    """Status of QMS audits."""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class FindingSeverity(Enum):
    """Severity of audit findings."""
    OBSERVATION = 1
    MINOR = 2
    MAJOR = 3
    CRITICAL = 4


class ChangeType(Enum):
    """Types of changes requiring control."""
    ALGORITHM = "algorithm"
    DATA_SOURCE = "data_source"
    TRAINING_PROCESS = "training_process"
    INFRASTRUCTURE = "infrastructure"
    CONFIGURATION = "configuration"
    DOCUMENTATION = "documentation"
    PROCEDURE = "procedure"


class ChangeImpact(Enum):
    """Impact assessment of changes."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    SUBSTANTIAL = 4  # Requires re-conformity assessment


class CAPAType(Enum):
    """Types of corrective/preventive actions."""
    CORRECTIVE = "corrective"
    PREVENTIVE = "preventive"
    IMPROVEMENT = "improvement"


class CAPAStatus(Enum):
    """Status of CAPA actions."""
    INITIATED = "initiated"
    INVESTIGATING = "investigating"
    ACTION_PLANNED = "action_planned"
    IMPLEMENTING = "implementing"
    VERIFYING = "verifying"
    CLOSED = "closed"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class QMSProcedure:
    """
    QMS procedure document.

    Represents a documented procedure that forms part of the QMS.
    """
    procedure_id: str
    element_type: QMSElementType
    title: str
    description: str

    # Document control
    version: str = "1.0"
    status: ProcedureStatus = ProcedureStatus.DRAFT
    effective_date: Optional[str] = None
    review_date: Optional[str] = None

    # Content
    scope: str = ""
    responsibilities: Dict[str, str] = field(default_factory=dict)
    procedure_steps: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)

    # Approval
    author: str = ""
    reviewer: str = ""
    approver: str = ""
    approval_date: Optional[str] = None

    # Metadata
    created_at: str = ""
    updated_at: str = ""
    revision_history: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        if not self.procedure_id:
            self.procedure_id = f"QMS-{self.element_type.value.upper()[:3]}-{uuid.uuid4().hex[:6].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = datetime.now(timezone.utc).isoformat()


@dataclass
class AuditFinding:
    """
    Audit finding record.

    Documents non-conformances or observations from audits.
    """
    finding_id: str
    audit_id: str
    severity: FindingSeverity
    title: str
    description: str

    # Details
    element_type: QMSElementType = QMSElementType.REGULATORY_COMPLIANCE
    requirement_reference: str = ""
    evidence: str = ""

    # Response
    root_cause: str = ""
    corrective_action: str = ""
    responsible_party: str = ""
    due_date: Optional[str] = None

    # Status
    is_closed: bool = False
    closure_date: Optional[str] = None
    closure_evidence: str = ""

    # Metadata
    created_at: str = ""

    def __post_init__(self):
        if not self.finding_id:
            self.finding_id = f"FIND-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class QMSAudit:
    """
    QMS audit record.

    Documents internal and external audits of the quality management system.
    """
    audit_id: str
    audit_type: AuditType
    title: str
    scope: str

    # Schedule
    planned_date: str = ""
    actual_date: Optional[str] = None

    # Team
    lead_auditor: str = ""
    audit_team: List[str] = field(default_factory=list)
    auditee: str = ""

    # Status
    status: AuditStatus = AuditStatus.PLANNED

    # Results
    findings: List[AuditFinding] = field(default_factory=list)
    conclusions: str = ""
    recommendations: List[str] = field(default_factory=list)

    # Follow-up
    follow_up_required: bool = False
    follow_up_date: Optional[str] = None

    # Metadata
    created_at: str = ""
    completed_at: Optional[str] = None

    def __post_init__(self):
        if not self.audit_id:
            self.audit_id = f"AUDIT-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    @property
    def critical_findings(self) -> List[AuditFinding]:
        """Get critical severity findings."""
        return [f for f in self.findings if f.severity == FindingSeverity.CRITICAL]

    @property
    def major_findings(self) -> List[AuditFinding]:
        """Get major severity findings."""
        return [f for f in self.findings if f.severity == FindingSeverity.MAJOR]

    @property
    def open_findings(self) -> List[AuditFinding]:
        """Get open (not closed) findings."""
        return [f for f in self.findings if not f.is_closed]


@dataclass
class DesignReview:
    """
    Design review record per Article 17(1)(b).

    Documents design control activities including verification and validation.
    """
    review_id: str
    title: str
    design_phase: str  # concept, preliminary, detailed, final

    # Scope
    components_reviewed: List[str] = field(default_factory=list)
    design_requirements: List[str] = field(default_factory=list)

    # Participants
    reviewers: List[str] = field(default_factory=list)
    designer: str = ""

    # Review details
    review_date: str = ""
    review_criteria: List[str] = field(default_factory=list)

    # Results
    status: str = "pending"  # pending, passed, conditional, failed
    findings: List[str] = field(default_factory=list)
    actions_required: List[str] = field(default_factory=list)

    # Approval
    approved: bool = False
    approver: str = ""
    approval_date: Optional[str] = None

    # Metadata
    created_at: str = ""

    def __post_init__(self):
        if not self.review_id:
            self.review_id = f"DR-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class ChangeRequest:
    """
    Change request for controlled changes per Article 17(1)(b).

    Documents and tracks changes that require evaluation and approval.
    """
    change_id: str
    change_type: ChangeType
    title: str
    description: str

    # Requestor
    requestor: str = ""
    request_date: str = ""

    # Justification
    reason: str = ""
    expected_benefits: List[str] = field(default_factory=list)

    # Impact assessment
    impact: ChangeImpact = ChangeImpact.MEDIUM
    affected_components: List[str] = field(default_factory=list)
    risk_assessment: str = ""
    requires_revalidation: bool = False
    requires_conformity_reassessment: bool = False

    # Approval
    status: str = "submitted"  # submitted, under_review, approved, rejected, implemented
    reviewers: List[str] = field(default_factory=list)
    approver: str = ""
    approval_date: Optional[str] = None
    rejection_reason: str = ""

    # Implementation
    implementation_plan: str = ""
    implementation_date: Optional[str] = None
    verification_results: str = ""

    # Metadata
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.change_id:
            self.change_id = f"CR-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.request_date:
            self.request_date = datetime.now(timezone.utc).isoformat()
        self.updated_at = datetime.now(timezone.utc).isoformat()


@dataclass
class CAPARecord:
    """
    Corrective and Preventive Action (CAPA) record.

    Documents actions taken to address non-conformances and prevent recurrence.
    """
    capa_id: str
    capa_type: CAPAType
    title: str
    description: str

    # Source
    source: str = ""  # audit, incident, customer_feedback, internal_review
    source_reference: str = ""  # E.g., audit_id, incident_id

    # Problem definition
    problem_statement: str = ""
    affected_processes: List[str] = field(default_factory=list)

    # Root cause analysis
    root_cause_analysis: str = ""
    root_causes: List[str] = field(default_factory=list)
    analysis_method: str = ""  # 5-why, fishbone, fault tree

    # Actions
    planned_actions: List[Dict[str, Any]] = field(default_factory=list)
    implemented_actions: List[Dict[str, Any]] = field(default_factory=list)

    # Responsibility
    owner: str = ""
    team_members: List[str] = field(default_factory=list)

    # Timeline
    status: CAPAStatus = CAPAStatus.INITIATED
    target_date: Optional[str] = None
    completion_date: Optional[str] = None

    # Verification
    effectiveness_criteria: List[str] = field(default_factory=list)
    verification_method: str = ""
    verification_results: str = ""
    is_effective: Optional[bool] = None

    # Metadata
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.capa_id:
            self.capa_id = f"CAPA-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = datetime.now(timezone.utc).isoformat()


@dataclass
class ResourceRecord:
    """
    Resource management record per Article 17(1)(k).

    Documents resources needed for AI system lifecycle.
    """
    resource_id: str
    resource_type: str  # human, infrastructure, tool, data, supplier
    name: str
    description: str

    # Details
    specifications: Dict[str, Any] = field(default_factory=dict)
    quantity: float = 1.0
    unit: str = ""

    # Supplier (if applicable)
    supplier_name: str = ""
    supplier_id: str = ""
    supplier_qualified: bool = False

    # Status
    is_available: bool = True
    availability_date: Optional[str] = None

    # Competence (for human resources)
    required_competencies: List[str] = field(default_factory=list)
    training_records: List[str] = field(default_factory=list)

    # Metadata
    created_at: str = ""

    def __post_init__(self):
        if not self.resource_id:
            self.resource_id = f"RES-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class AccountabilityRecord:
    """
    Accountability framework record per Article 17(1)(l).

    Documents roles and responsibilities for AI system compliance.
    """
    role_id: str
    role_name: str
    role_description: str

    # Responsibilities
    responsibilities: List[str] = field(default_factory=list)
    authorities: List[str] = field(default_factory=list)

    # Assignment
    assigned_to: str = ""
    department: str = ""
    reporting_to: str = ""

    # Competence requirements
    required_qualifications: List[str] = field(default_factory=list)
    required_training: List[str] = field(default_factory=list)
    required_experience: str = ""

    # Delegation
    can_delegate: bool = False
    delegates: List[str] = field(default_factory=list)

    # Status
    is_active: bool = True
    effective_date: str = ""

    # Metadata
    created_at: str = ""

    def __post_init__(self):
        if not self.role_id:
            self.role_id = f"ROLE-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class QMSConfig:
    """
    Configuration for Quality Management System.
    """
    # Organization info
    organization_name: str = ""
    organization_id: str = ""

    # QMS identification
    qms_version: str = "1.0"
    qms_effective_date: str = ""

    # Review cycles
    procedure_review_months: int = 12
    audit_frequency_months: int = 6
    management_review_months: int = 12

    # Thresholds
    max_open_major_findings: int = 5
    max_open_critical_findings: int = 0
    capa_completion_days: int = 90

    # Paths
    log_path: str = "logs/ai_act/qms"
    documents_path: str = "docs/compliance/qms"

    # Notifications
    notification_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# Main QMS Class
# =============================================================================

class QualityManagementSystem:
    """
    AI Act compliant Quality Management System per Article 17.

    Provides comprehensive QMS functionality including:
    - Procedure documentation and control
    - Design control and verification
    - Change management
    - Audit management
    - CAPA (Corrective and Preventive Actions)
    - Resource management
    - Accountability framework

    Usage:
        config = QMSConfig(
            organization_name="TradingBot2",
            qms_version="1.0",
        )
        qms = QualityManagementSystem(config)

        # Create a procedure
        procedure = qms.create_procedure(
            element_type=QMSElementType.RISK_MANAGEMENT,
            title="Risk Assessment Procedure",
            description="Procedure for conducting risk assessments",
        )

        # Approve procedure
        qms.approve_procedure(procedure.procedure_id, approver="Quality Manager")

        # Schedule an audit
        audit = qms.create_audit(
            audit_type=AuditType.INTERNAL,
            title="Q1 Internal Audit",
            scope="Risk Management System",
        )
    """

    def __init__(self, config: Optional[QMSConfig] = None):
        """Initialize QMS with configuration."""
        self.config = config or QMSConfig()

        # Data stores
        self._procedures: Dict[str, QMSProcedure] = {}
        self._audits: Dict[str, QMSAudit] = {}
        self._design_reviews: Dict[str, DesignReview] = {}
        self._change_requests: Dict[str, ChangeRequest] = {}
        self._capas: Dict[str, CAPARecord] = {}
        self._resources: Dict[str, ResourceRecord] = {}
        self._roles: Dict[str, AccountabilityRecord] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("QualityManagementSystem initialized")

    # =========================================================================
    # Procedure Management
    # =========================================================================

    def create_procedure(
        self,
        element_type: QMSElementType,
        title: str,
        description: str,
        scope: str = "",
        responsibilities: Optional[Dict[str, str]] = None,
        procedure_steps: Optional[List[str]] = None,
        author: str = "",
    ) -> QMSProcedure:
        """
        Create a new QMS procedure document.

        Args:
            element_type: QMS element this procedure addresses
            title: Procedure title
            description: Procedure description
            scope: Scope of the procedure
            responsibilities: Role -> responsibility mapping
            procedure_steps: Steps in the procedure
            author: Author of the procedure

        Returns:
            Created QMSProcedure
        """
        procedure = QMSProcedure(
            procedure_id="",
            element_type=element_type,
            title=title,
            description=description,
            scope=scope,
            responsibilities=responsibilities or {},
            procedure_steps=procedure_steps or [],
            author=author,
        )

        with self._lock:
            self._procedures[procedure.procedure_id] = procedure

        self._log_event("procedure_created", asdict(procedure))
        logger.info(f"Procedure created: {procedure.procedure_id}")
        return procedure

    def submit_procedure_for_review(
        self,
        procedure_id: str,
        reviewer: str,
    ) -> QMSProcedure:
        """Submit procedure for review."""
        with self._lock:
            if procedure_id not in self._procedures:
                raise ValueError(f"Procedure {procedure_id} not found")

            procedure = self._procedures[procedure_id]
            procedure.status = ProcedureStatus.UNDER_REVIEW
            procedure.reviewer = reviewer
            procedure.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("procedure_submitted_for_review", {
            "procedure_id": procedure_id,
            "reviewer": reviewer,
        })
        return procedure

    def approve_procedure(
        self,
        procedure_id: str,
        approver: str,
        effective_date: Optional[str] = None,
    ) -> QMSProcedure:
        """Approve and make procedure effective."""
        with self._lock:
            if procedure_id not in self._procedures:
                raise ValueError(f"Procedure {procedure_id} not found")

            procedure = self._procedures[procedure_id]
            procedure.status = ProcedureStatus.EFFECTIVE
            procedure.approver = approver
            procedure.approval_date = datetime.now(timezone.utc).isoformat()
            procedure.effective_date = effective_date or datetime.now(timezone.utc).isoformat()

            # Set review date
            review_delta = timedelta(days=self.config.procedure_review_months * 30)
            procedure.review_date = (
                datetime.now(timezone.utc) + review_delta
            ).isoformat()

            procedure.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("procedure_approved", {
            "procedure_id": procedure_id,
            "approver": approver,
        })
        logger.info(f"Procedure approved: {procedure_id}")
        return procedure

    def revise_procedure(
        self,
        procedure_id: str,
        changes: str,
        author: str,
    ) -> QMSProcedure:
        """Create a new revision of a procedure."""
        with self._lock:
            if procedure_id not in self._procedures:
                raise ValueError(f"Procedure {procedure_id} not found")

            procedure = self._procedures[procedure_id]

            # Record revision
            procedure.revision_history.append({
                "version": procedure.version,
                "date": procedure.updated_at,
                "changes": changes,
                "author": author,
            })

            # Increment version
            major, minor = procedure.version.split(".")
            procedure.version = f"{major}.{int(minor) + 1}"
            procedure.status = ProcedureStatus.DRAFT
            procedure.author = author
            procedure.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("procedure_revised", {
            "procedure_id": procedure_id,
            "new_version": procedure.version,
        })
        return procedure

    def get_procedure(self, procedure_id: str) -> Optional[QMSProcedure]:
        """Get procedure by ID."""
        with self._lock:
            return self._procedures.get(procedure_id)

    def get_procedures_by_element(self, element_type: QMSElementType) -> List[QMSProcedure]:
        """Get all procedures for a QMS element."""
        with self._lock:
            return [p for p in self._procedures.values() if p.element_type == element_type]

    def get_effective_procedures(self) -> List[QMSProcedure]:
        """Get all effective procedures."""
        with self._lock:
            return [p for p in self._procedures.values()
                    if p.status == ProcedureStatus.EFFECTIVE]

    def get_procedures_due_for_review(self) -> List[QMSProcedure]:
        """Get procedures that are due for review."""
        now = datetime.now(timezone.utc)
        with self._lock:
            due = []
            for p in self._procedures.values():
                if p.review_date:
                    review_dt = datetime.fromisoformat(p.review_date.replace("Z", "+00:00"))
                    if review_dt <= now:
                        due.append(p)
            return due

    # =========================================================================
    # Audit Management
    # =========================================================================

    def create_audit(
        self,
        audit_type: AuditType,
        title: str,
        scope: str,
        planned_date: str,
        lead_auditor: str = "",
        audit_team: Optional[List[str]] = None,
        auditee: str = "",
    ) -> QMSAudit:
        """
        Create a new audit.

        Args:
            audit_type: Type of audit
            title: Audit title
            scope: Scope of audit
            planned_date: Planned audit date (ISO format)
            lead_auditor: Lead auditor name
            audit_team: Audit team members
            auditee: Organization/person being audited

        Returns:
            Created QMSAudit
        """
        audit = QMSAudit(
            audit_id="",
            audit_type=audit_type,
            title=title,
            scope=scope,
            planned_date=planned_date,
            lead_auditor=lead_auditor,
            audit_team=audit_team or [],
            auditee=auditee,
        )

        with self._lock:
            self._audits[audit.audit_id] = audit

        self._log_event("audit_created", {
            "audit_id": audit.audit_id,
            "type": audit_type.value,
            "title": title,
        })
        logger.info(f"Audit created: {audit.audit_id}")
        return audit

    def start_audit(self, audit_id: str, actual_date: Optional[str] = None) -> QMSAudit:
        """Start an audit."""
        with self._lock:
            if audit_id not in self._audits:
                raise ValueError(f"Audit {audit_id} not found")

            audit = self._audits[audit_id]
            audit.status = AuditStatus.IN_PROGRESS
            audit.actual_date = actual_date or datetime.now(timezone.utc).isoformat()

        self._log_event("audit_started", {"audit_id": audit_id})
        return audit

    def add_finding(
        self,
        audit_id: str,
        severity: FindingSeverity,
        title: str,
        description: str,
        element_type: QMSElementType = QMSElementType.REGULATORY_COMPLIANCE,
        requirement_reference: str = "",
        evidence: str = "",
    ) -> AuditFinding:
        """Add a finding to an audit."""
        finding = AuditFinding(
            finding_id="",
            audit_id=audit_id,
            severity=severity,
            title=title,
            description=description,
            element_type=element_type,
            requirement_reference=requirement_reference,
            evidence=evidence,
        )

        with self._lock:
            if audit_id not in self._audits:
                raise ValueError(f"Audit {audit_id} not found")

            self._audits[audit_id].findings.append(finding)

        self._log_event("finding_added", {
            "audit_id": audit_id,
            "finding_id": finding.finding_id,
            "severity": severity.name,
        })
        return finding

    def close_finding(
        self,
        audit_id: str,
        finding_id: str,
        closure_evidence: str,
    ) -> AuditFinding:
        """Close a finding with evidence."""
        with self._lock:
            if audit_id not in self._audits:
                raise ValueError(f"Audit {audit_id} not found")

            audit = self._audits[audit_id]
            for finding in audit.findings:
                if finding.finding_id == finding_id:
                    finding.is_closed = True
                    finding.closure_date = datetime.now(timezone.utc).isoformat()
                    finding.closure_evidence = closure_evidence

                    self._log_event("finding_closed", {
                        "audit_id": audit_id,
                        "finding_id": finding_id,
                    })
                    return finding

            raise ValueError(f"Finding {finding_id} not found in audit {audit_id}")

    def complete_audit(
        self,
        audit_id: str,
        conclusions: str,
        recommendations: Optional[List[str]] = None,
    ) -> QMSAudit:
        """Complete an audit."""
        with self._lock:
            if audit_id not in self._audits:
                raise ValueError(f"Audit {audit_id} not found")

            audit = self._audits[audit_id]
            audit.status = AuditStatus.COMPLETED
            audit.conclusions = conclusions
            audit.recommendations = recommendations or []
            audit.completed_at = datetime.now(timezone.utc).isoformat()

            # Check if follow-up needed
            if audit.critical_findings or audit.major_findings:
                audit.follow_up_required = True

        self._log_event("audit_completed", {
            "audit_id": audit_id,
            "total_findings": len(audit.findings),
            "critical": len(audit.critical_findings),
            "major": len(audit.major_findings),
        })
        logger.info(f"Audit completed: {audit_id}")
        return audit

    def get_audit(self, audit_id: str) -> Optional[QMSAudit]:
        """Get audit by ID."""
        with self._lock:
            return self._audits.get(audit_id)

    def get_open_audits(self) -> List[QMSAudit]:
        """Get all open audits."""
        with self._lock:
            return [a for a in self._audits.values()
                    if a.status in (AuditStatus.PLANNED, AuditStatus.IN_PROGRESS)]

    def get_audits_with_open_findings(self) -> List[QMSAudit]:
        """Get audits that have open findings."""
        with self._lock:
            return [a for a in self._audits.values() if a.open_findings]

    # =========================================================================
    # Design Control
    # =========================================================================

    def create_design_review(
        self,
        title: str,
        design_phase: str,
        components_reviewed: Optional[List[str]] = None,
        design_requirements: Optional[List[str]] = None,
        reviewers: Optional[List[str]] = None,
        designer: str = "",
    ) -> DesignReview:
        """
        Create a design review record per Article 17(1)(b).

        Args:
            title: Review title
            design_phase: Phase of design (concept, preliminary, detailed, final)
            components_reviewed: Components being reviewed
            design_requirements: Requirements being verified
            reviewers: List of reviewers
            designer: Designer/developer

        Returns:
            Created DesignReview
        """
        review = DesignReview(
            review_id="",
            title=title,
            design_phase=design_phase,
            components_reviewed=components_reviewed or [],
            design_requirements=design_requirements or [],
            reviewers=reviewers or [],
            designer=designer,
        )

        with self._lock:
            self._design_reviews[review.review_id] = review

        self._log_event("design_review_created", {
            "review_id": review.review_id,
            "phase": design_phase,
        })
        return review

    def complete_design_review(
        self,
        review_id: str,
        status: str,
        findings: Optional[List[str]] = None,
        actions_required: Optional[List[str]] = None,
        approver: str = "",
    ) -> DesignReview:
        """Complete a design review with results."""
        with self._lock:
            if review_id not in self._design_reviews:
                raise ValueError(f"Design review {review_id} not found")

            review = self._design_reviews[review_id]
            review.status = status
            review.findings = findings or []
            review.actions_required = actions_required or []
            review.review_date = datetime.now(timezone.utc).isoformat()

            if status == "passed":
                review.approved = True
                review.approver = approver
                review.approval_date = datetime.now(timezone.utc).isoformat()

        self._log_event("design_review_completed", {
            "review_id": review_id,
            "status": status,
        })
        return review

    def get_design_review(self, review_id: str) -> Optional[DesignReview]:
        """Get design review by ID."""
        with self._lock:
            return self._design_reviews.get(review_id)

    # =========================================================================
    # Change Management
    # =========================================================================

    def create_change_request(
        self,
        change_type: ChangeType,
        title: str,
        description: str,
        reason: str = "",
        expected_benefits: Optional[List[str]] = None,
        requestor: str = "",
    ) -> ChangeRequest:
        """
        Create a change request.

        Args:
            change_type: Type of change
            title: Change request title
            description: Detailed description
            reason: Reason for change
            expected_benefits: Expected benefits
            requestor: Person requesting the change

        Returns:
            Created ChangeRequest
        """
        change = ChangeRequest(
            change_id="",
            change_type=change_type,
            title=title,
            description=description,
            reason=reason,
            expected_benefits=expected_benefits or [],
            requestor=requestor,
        )

        with self._lock:
            self._change_requests[change.change_id] = change

        self._log_event("change_request_created", {
            "change_id": change.change_id,
            "type": change_type.value,
        })
        logger.info(f"Change request created: {change.change_id}")
        return change

    def assess_change_impact(
        self,
        change_id: str,
        impact: ChangeImpact,
        affected_components: Optional[List[str]] = None,
        risk_assessment: str = "",
        requires_revalidation: bool = False,
    ) -> ChangeRequest:
        """Assess impact of a change request."""
        with self._lock:
            if change_id not in self._change_requests:
                raise ValueError(f"Change request {change_id} not found")

            change = self._change_requests[change_id]
            change.impact = impact
            change.affected_components = affected_components or []
            change.risk_assessment = risk_assessment
            change.requires_revalidation = requires_revalidation
            change.requires_conformity_reassessment = (impact == ChangeImpact.SUBSTANTIAL)
            change.status = "under_review"
            change.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("change_impact_assessed", {
            "change_id": change_id,
            "impact": impact.name,
            "requires_reassessment": change.requires_conformity_reassessment,
        })
        return change

    def approve_change(
        self,
        change_id: str,
        approver: str,
        implementation_plan: str = "",
    ) -> ChangeRequest:
        """Approve a change request."""
        with self._lock:
            if change_id not in self._change_requests:
                raise ValueError(f"Change request {change_id} not found")

            change = self._change_requests[change_id]
            change.status = "approved"
            change.approver = approver
            change.approval_date = datetime.now(timezone.utc).isoformat()
            change.implementation_plan = implementation_plan
            change.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("change_approved", {
            "change_id": change_id,
            "approver": approver,
        })
        logger.info(f"Change approved: {change_id}")
        return change

    def reject_change(
        self,
        change_id: str,
        rejection_reason: str,
    ) -> ChangeRequest:
        """Reject a change request."""
        with self._lock:
            if change_id not in self._change_requests:
                raise ValueError(f"Change request {change_id} not found")

            change = self._change_requests[change_id]
            change.status = "rejected"
            change.rejection_reason = rejection_reason
            change.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("change_rejected", {
            "change_id": change_id,
            "reason": rejection_reason,
        })
        return change

    def implement_change(
        self,
        change_id: str,
        verification_results: str = "",
    ) -> ChangeRequest:
        """Mark a change as implemented."""
        with self._lock:
            if change_id not in self._change_requests:
                raise ValueError(f"Change request {change_id} not found")

            change = self._change_requests[change_id]
            if change.status != "approved":
                raise ValueError(f"Change {change_id} is not approved")

            change.status = "implemented"
            change.implementation_date = datetime.now(timezone.utc).isoformat()
            change.verification_results = verification_results
            change.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("change_implemented", {
            "change_id": change_id,
        })
        logger.info(f"Change implemented: {change_id}")
        return change

    def get_change_request(self, change_id: str) -> Optional[ChangeRequest]:
        """Get change request by ID."""
        with self._lock:
            return self._change_requests.get(change_id)

    def get_pending_changes(self) -> List[ChangeRequest]:
        """Get pending change requests."""
        with self._lock:
            return [c for c in self._change_requests.values()
                    if c.status in ("submitted", "under_review")]

    def get_substantial_changes(self) -> List[ChangeRequest]:
        """Get changes that require conformity reassessment."""
        with self._lock:
            return [c for c in self._change_requests.values()
                    if c.requires_conformity_reassessment]

    # =========================================================================
    # CAPA Management
    # =========================================================================

    def create_capa(
        self,
        capa_type: CAPAType,
        title: str,
        description: str,
        source: str = "",
        source_reference: str = "",
        problem_statement: str = "",
        owner: str = "",
    ) -> CAPARecord:
        """
        Create a CAPA (Corrective and Preventive Action).

        Args:
            capa_type: Type of action
            title: CAPA title
            description: Description
            source: Source of CAPA (audit, incident, etc.)
            source_reference: Reference ID
            problem_statement: Problem description
            owner: CAPA owner

        Returns:
            Created CAPARecord
        """
        capa = CAPARecord(
            capa_id="",
            capa_type=capa_type,
            title=title,
            description=description,
            source=source,
            source_reference=source_reference,
            problem_statement=problem_statement,
            owner=owner,
        )

        with self._lock:
            self._capas[capa.capa_id] = capa

        self._log_event("capa_created", {
            "capa_id": capa.capa_id,
            "type": capa_type.value,
            "source": source,
        })
        logger.info(f"CAPA created: {capa.capa_id}")
        return capa

    def perform_root_cause_analysis(
        self,
        capa_id: str,
        analysis_method: str,
        root_causes: List[str],
        analysis_description: str = "",
    ) -> CAPARecord:
        """Perform root cause analysis for a CAPA."""
        with self._lock:
            if capa_id not in self._capas:
                raise ValueError(f"CAPA {capa_id} not found")

            capa = self._capas[capa_id]
            capa.analysis_method = analysis_method
            capa.root_causes = root_causes
            capa.root_cause_analysis = analysis_description
            capa.status = CAPAStatus.INVESTIGATING
            capa.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("capa_root_cause_analyzed", {
            "capa_id": capa_id,
            "method": analysis_method,
            "root_causes_count": len(root_causes),
        })
        return capa

    def plan_capa_actions(
        self,
        capa_id: str,
        actions: List[Dict[str, Any]],
        effectiveness_criteria: Optional[List[str]] = None,
        target_date: Optional[str] = None,
    ) -> CAPARecord:
        """Plan actions for a CAPA."""
        with self._lock:
            if capa_id not in self._capas:
                raise ValueError(f"CAPA {capa_id} not found")

            capa = self._capas[capa_id]
            capa.planned_actions = actions
            capa.effectiveness_criteria = effectiveness_criteria or []
            capa.target_date = target_date
            capa.status = CAPAStatus.ACTION_PLANNED
            capa.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("capa_actions_planned", {
            "capa_id": capa_id,
            "actions_count": len(actions),
        })
        return capa

    def implement_capa_action(
        self,
        capa_id: str,
        action_index: int,
        implementation_details: str = "",
    ) -> CAPARecord:
        """Mark a CAPA action as implemented."""
        with self._lock:
            if capa_id not in self._capas:
                raise ValueError(f"CAPA {capa_id} not found")

            capa = self._capas[capa_id]
            if action_index >= len(capa.planned_actions):
                raise ValueError(f"Action index {action_index} out of range")

            action = capa.planned_actions[action_index].copy()
            action["implemented"] = True
            action["implementation_date"] = datetime.now(timezone.utc).isoformat()
            action["implementation_details"] = implementation_details
            capa.implemented_actions.append(action)

            # Update status if all actions implemented
            if len(capa.implemented_actions) == len(capa.planned_actions):
                capa.status = CAPAStatus.VERIFYING
            else:
                capa.status = CAPAStatus.IMPLEMENTING

            capa.updated_at = datetime.now(timezone.utc).isoformat()

        return capa

    def verify_capa(
        self,
        capa_id: str,
        verification_method: str,
        verification_results: str,
        is_effective: bool,
    ) -> CAPARecord:
        """Verify CAPA effectiveness."""
        with self._lock:
            if capa_id not in self._capas:
                raise ValueError(f"CAPA {capa_id} not found")

            capa = self._capas[capa_id]
            capa.verification_method = verification_method
            capa.verification_results = verification_results
            capa.is_effective = is_effective

            if is_effective:
                capa.status = CAPAStatus.CLOSED
                capa.completion_date = datetime.now(timezone.utc).isoformat()

            capa.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("capa_verified", {
            "capa_id": capa_id,
            "is_effective": is_effective,
        })
        logger.info(f"CAPA verified: {capa_id} - Effective: {is_effective}")
        return capa

    def get_capa(self, capa_id: str) -> Optional[CAPARecord]:
        """Get CAPA by ID."""
        with self._lock:
            return self._capas.get(capa_id)

    def get_open_capas(self) -> List[CAPARecord]:
        """Get all open CAPAs."""
        with self._lock:
            return [c for c in self._capas.values()
                    if c.status != CAPAStatus.CLOSED]

    def get_overdue_capas(self) -> List[CAPARecord]:
        """Get CAPAs that are past their target date."""
        now = datetime.now(timezone.utc)
        with self._lock:
            overdue = []
            for capa in self._capas.values():
                if capa.target_date and capa.status != CAPAStatus.CLOSED:
                    target_dt = datetime.fromisoformat(capa.target_date.replace("Z", "+00:00"))
                    if target_dt < now:
                        overdue.append(capa)
            return overdue

    # =========================================================================
    # Resource Management (Article 17(1)(k))
    # =========================================================================

    def register_resource(
        self,
        resource_type: str,
        name: str,
        description: str,
        specifications: Optional[Dict[str, Any]] = None,
        quantity: float = 1.0,
        unit: str = "",
    ) -> ResourceRecord:
        """Register a resource."""
        resource = ResourceRecord(
            resource_id="",
            resource_type=resource_type,
            name=name,
            description=description,
            specifications=specifications or {},
            quantity=quantity,
            unit=unit,
        )

        with self._lock:
            self._resources[resource.resource_id] = resource

        return resource

    def qualify_supplier(
        self,
        resource_id: str,
        supplier_name: str,
        supplier_id: str,
    ) -> ResourceRecord:
        """Qualify a supplier for a resource."""
        with self._lock:
            if resource_id not in self._resources:
                raise ValueError(f"Resource {resource_id} not found")

            resource = self._resources[resource_id]
            resource.supplier_name = supplier_name
            resource.supplier_id = supplier_id
            resource.supplier_qualified = True

        return resource

    def get_resource(self, resource_id: str) -> Optional[ResourceRecord]:
        """Get resource by ID."""
        with self._lock:
            return self._resources.get(resource_id)

    # =========================================================================
    # Accountability Framework (Article 17(1)(l))
    # =========================================================================

    def define_role(
        self,
        role_name: str,
        role_description: str,
        responsibilities: Optional[List[str]] = None,
        authorities: Optional[List[str]] = None,
        department: str = "",
    ) -> AccountabilityRecord:
        """Define an accountability role."""
        role = AccountabilityRecord(
            role_id="",
            role_name=role_name,
            role_description=role_description,
            responsibilities=responsibilities or [],
            authorities=authorities or [],
            department=department,
        )

        with self._lock:
            self._roles[role.role_id] = role

        return role

    def assign_role(
        self,
        role_id: str,
        assigned_to: str,
        effective_date: Optional[str] = None,
    ) -> AccountabilityRecord:
        """Assign a role to a person."""
        with self._lock:
            if role_id not in self._roles:
                raise ValueError(f"Role {role_id} not found")

            role = self._roles[role_id]
            role.assigned_to = assigned_to
            role.effective_date = effective_date or datetime.now(timezone.utc).isoformat()
            role.is_active = True

        return role

    def get_role(self, role_id: str) -> Optional[AccountabilityRecord]:
        """Get role by ID."""
        with self._lock:
            return self._roles.get(role_id)

    def get_all_roles(self) -> List[AccountabilityRecord]:
        """Get all defined roles."""
        with self._lock:
            return list(self._roles.values())

    # =========================================================================
    # Compliance Status
    # =========================================================================

    def get_compliance_status(self) -> Dict[str, Any]:
        """
        Get overall QMS compliance status.

        Returns:
            Dictionary with compliance metrics
        """
        with self._lock:
            # Procedure metrics
            total_procedures = len(self._procedures)
            effective_procedures = len([p for p in self._procedures.values()
                                       if p.status == ProcedureStatus.EFFECTIVE])
            due_for_review = len(self.get_procedures_due_for_review())

            # Audit metrics
            total_audits = len(self._audits)
            open_audits = len(self.get_open_audits())
            audits_with_findings = len(self.get_audits_with_open_findings())

            total_findings = sum(len(a.findings) for a in self._audits.values())
            open_findings = sum(len(a.open_findings) for a in self._audits.values())
            critical_findings = sum(len(a.critical_findings) for a in self._audits.values())
            major_findings = sum(len(a.major_findings) for a in self._audits.values())

            # CAPA metrics
            total_capas = len(self._capas)
            open_capas = len(self.get_open_capas())
            overdue_capas = len(self.get_overdue_capas())

            # Change metrics
            total_changes = len(self._change_requests)
            pending_changes = len(self.get_pending_changes())
            substantial_changes = len(self.get_substantial_changes())

            # Element coverage
            covered_elements = set()
            for p in self._procedures.values():
                if p.status == ProcedureStatus.EFFECTIVE:
                    covered_elements.add(p.element_type)

            element_coverage = len(covered_elements) / len(QMSElementType)

            # Compliance score (0-100)
            compliance_score = self._calculate_compliance_score(
                effective_procedures=effective_procedures,
                total_procedures=max(1, total_procedures),
                open_critical=critical_findings,
                open_major=major_findings,
                overdue_capas=overdue_capas,
                element_coverage=element_coverage,
            )

        return {
            "compliance_score": compliance_score,
            "element_coverage": element_coverage,
            "covered_elements": [e.value for e in covered_elements],
            "procedures": {
                "total": total_procedures,
                "effective": effective_procedures,
                "due_for_review": due_for_review,
            },
            "audits": {
                "total": total_audits,
                "open": open_audits,
                "with_open_findings": audits_with_findings,
            },
            "findings": {
                "total": total_findings,
                "open": open_findings,
                "critical": critical_findings,
                "major": major_findings,
            },
            "capas": {
                "total": total_capas,
                "open": open_capas,
                "overdue": overdue_capas,
            },
            "changes": {
                "total": total_changes,
                "pending": pending_changes,
                "substantial": substantial_changes,
            },
            "thresholds": {
                "max_critical_findings": self.config.max_open_critical_findings,
                "max_major_findings": self.config.max_open_major_findings,
            },
            "compliance_issues": self._identify_compliance_issues(
                critical_findings, major_findings, overdue_capas
            ),
        }

    def _calculate_compliance_score(
        self,
        effective_procedures: int,
        total_procedures: int,
        open_critical: int,
        open_major: int,
        overdue_capas: int,
        element_coverage: float,
    ) -> float:
        """Calculate overall compliance score."""
        score = 100.0

        # Procedure effectiveness (20%)
        procedure_ratio = effective_procedures / total_procedures
        score -= (1 - procedure_ratio) * 20

        # Element coverage (30%)
        score -= (1 - element_coverage) * 30

        # Critical findings (-20 each, max -40)
        score -= min(open_critical * 20, 40)

        # Major findings (-5 each, max -20)
        score -= min(open_major * 5, 20)

        # Overdue CAPAs (-5 each, max -10)
        score -= min(overdue_capas * 5, 10)

        return max(0, score)

    def _identify_compliance_issues(
        self,
        critical_findings: int,
        major_findings: int,
        overdue_capas: int,
    ) -> List[str]:
        """Identify compliance issues."""
        issues = []

        if critical_findings > self.config.max_open_critical_findings:
            issues.append(f"Open critical findings ({critical_findings}) exceed threshold")

        if major_findings > self.config.max_open_major_findings:
            issues.append(f"Open major findings ({major_findings}) exceed threshold")

        if overdue_capas > 0:
            issues.append(f"{overdue_capas} CAPA(s) are overdue")

        return issues

    # =========================================================================
    # Export and Reporting
    # =========================================================================

    def export_qms_summary(self) -> Dict[str, Any]:
        """Export complete QMS summary for documentation."""
        with self._lock:
            return {
                "export_date": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "organization": self.config.organization_name,
                    "version": self.config.qms_version,
                },
                "compliance_status": self.get_compliance_status(),
                "procedures": [asdict(p) for p in self._procedures.values()],
                "audits": [asdict(a) for a in self._audits.values()],
                "design_reviews": [asdict(r) for r in self._design_reviews.values()],
                "change_requests": [asdict(c) for c in self._change_requests.values()],
                "capas": [asdict(c) for c in self._capas.values()],
                "resources": [asdict(r) for r in self._resources.values()],
                "roles": [asdict(r) for r in self._roles.values()],
            }

    def generate_management_review_report(self) -> Dict[str, Any]:
        """Generate management review report per Article 17."""
        status = self.get_compliance_status()

        with self._lock:
            # Get audit summary
            recent_audits = [a for a in self._audits.values()
                           if a.status == AuditStatus.COMPLETED][-5:]

            # Get CAPA effectiveness
            closed_capas = [c for c in self._capas.values()
                          if c.status == CAPAStatus.CLOSED]
            effective_capas = [c for c in closed_capas if c.is_effective]

            capa_effectiveness = (
                len(effective_capas) / len(closed_capas) * 100
                if closed_capas else 100
            )

        return {
            "report_date": datetime.now(timezone.utc).isoformat(),
            "period_covered": f"Last {self.config.management_review_months} months",
            "overall_compliance": status,
            "recent_audits": [{
                "id": a.audit_id,
                "title": a.title,
                "findings_count": len(a.findings),
                "status": a.status.value,
            } for a in recent_audits],
            "capa_effectiveness_percent": capa_effectiveness,
            "recommendations": self._generate_recommendations(status),
        }

    def _generate_recommendations(self, status: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []

        if status["element_coverage"] < 1.0:
            missing = set(QMSElementType) - set(
                QMSElementType(e) for e in status["covered_elements"]
            )
            for element in missing:
                recommendations.append(
                    f"Develop procedure for {element.value} element"
                )

        if status["procedures"]["due_for_review"] > 0:
            recommendations.append(
                f"Review {status['procedures']['due_for_review']} procedures due for review"
            )

        if status["capas"]["overdue"] > 0:
            recommendations.append(
                f"Address {status['capas']['overdue']} overdue CAPA(s)"
            )

        return recommendations

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a QMS event."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"qms_events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                # Convert enums to strings for JSON serialization
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log QMS event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_qms(
    config: Optional[Union[Dict[str, Any], QMSConfig]] = None,
) -> QualityManagementSystem:
    """
    Create a QualityManagementSystem instance.

    Args:
        config: Optional configuration

    Returns:
        Configured QualityManagementSystem instance
    """
    if config is None:
        qms_config = QMSConfig()
    elif isinstance(config, QMSConfig):
        qms_config = config
    else:
        qms_config = QMSConfig(
            organization_name=config.get("organization_name", ""),
            organization_id=config.get("organization_id", ""),
            qms_version=config.get("qms_version", "1.0"),
            qms_effective_date=config.get("qms_effective_date", ""),
            procedure_review_months=config.get("procedure_review_months", 12),
            audit_frequency_months=config.get("audit_frequency_months", 6),
            management_review_months=config.get("management_review_months", 12),
            max_open_major_findings=config.get("max_open_major_findings", 5),
            max_open_critical_findings=config.get("max_open_critical_findings", 0),
            capa_completion_days=config.get("capa_completion_days", 90),
            log_path=config.get("log_path", "logs/ai_act/qms"),
            documents_path=config.get("documents_path", "docs/compliance/qms"),
        )

    return QualityManagementSystem(qms_config)


def get_default_qms_procedures() -> List[Dict[str, Any]]:
    """
    Get default QMS procedures based on Article 17 requirements.

    Returns:
        List of procedure definitions for all 12 QMS elements
    """
    return [
        {
            "element_type": QMSElementType.REGULATORY_COMPLIANCE,
            "title": "Regulatory Compliance Strategy",
            "description": "Strategy for ensuring compliance with EU AI Act and related regulations",
            "article_reference": "Article 17(1)(a)",
        },
        {
            "element_type": QMSElementType.DESIGN_CONTROL,
            "title": "Design and Development Control",
            "description": "Techniques for design, design control, and design verification",
            "article_reference": "Article 17(1)(b)",
        },
        {
            "element_type": QMSElementType.DEVELOPMENT_QA,
            "title": "Development Quality Assurance",
            "description": "Quality control and quality assurance during development",
            "article_reference": "Article 17(1)(c)",
        },
        {
            "element_type": QMSElementType.TESTING_VALIDATION,
            "title": "Testing and Validation Procedures",
            "description": "Examination, test and validation procedures before and after deployment",
            "article_reference": "Article 17(1)(d)",
        },
        {
            "element_type": QMSElementType.DATA_MANAGEMENT,
            "title": "Data Management Procedures",
            "description": "Procedures for managing training, validation and testing data",
            "article_reference": "Article 17(1)(e)",
        },
        {
            "element_type": QMSElementType.RISK_MANAGEMENT,
            "title": "Risk Management System",
            "description": "Linkage to risk management system per Article 9",
            "article_reference": "Article 17(1)(f)",
        },
        {
            "element_type": QMSElementType.POST_MARKET_MONITORING,
            "title": "Post-Market Monitoring",
            "description": "Post-market monitoring system per Article 72",
            "article_reference": "Article 17(1)(g)",
        },
        {
            "element_type": QMSElementType.INCIDENT_REPORTING,
            "title": "Incident Reporting Procedures",
            "description": "Procedures for reporting serious incidents and malfunctioning",
            "article_reference": "Article 17(1)(h)",
        },
        {
            "element_type": QMSElementType.AUTHORITY_COMMUNICATION,
            "title": "Authority Communication",
            "description": "Communication with competent authorities and notified bodies",
            "article_reference": "Article 17(1)(i)",
        },
        {
            "element_type": QMSElementType.RECORD_KEEPING,
            "title": "Record Keeping Procedures",
            "description": "Systems and procedures for record keeping and documentation",
            "article_reference": "Article 17(1)(j)",
        },
        {
            "element_type": QMSElementType.RESOURCE_MANAGEMENT,
            "title": "Resource Management",
            "description": "Management of resources including supply chain",
            "article_reference": "Article 17(1)(k)",
        },
        {
            "element_type": QMSElementType.ACCOUNTABILITY,
            "title": "Accountability Framework",
            "description": "Framework for management responsibility and accountability",
            "article_reference": "Article 17(1)(l)",
        },
    ]
