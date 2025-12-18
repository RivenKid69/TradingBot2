# -*- coding: utf-8 -*-
"""
DORA Governance Framework Module (Article 5).

Regulation (EU) 2022/2554 Article 5 defines governance requirements:
    - Management body ultimate responsibility for ICT risk management
    - Definition, approval, oversight of ICT risk management framework
    - Approval of digital operational resilience strategy
    - Setting clear roles and responsibilities
    - Mandatory ICT training for management body (except microenterprises)

This module provides:
    1. ManagementBodyOversight - Management body ICT responsibilities
    2. RoleAssignment - ICT roles and segregation of duties
    3. TrainingRequirements - Mandatory ICT training tracking
    4. AuditIntegration - Internal audit requirements
    5. DORAGovernanceFramework - Main governance orchestrator

References:
    - Article 5 DORA: https://www.digital-operational-resilience-act.com/Article_5.html
    - RTS on ICT Risk Management: CDR 2024/1774
    - Three Lines of Defence Model: IIA Position Paper
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class GovernanceRole(Enum):
    """ICT governance roles per DORA Article 5."""
    MANAGEMENT_BODY = "management_body"  # Ultimate responsibility
    ICT_RISK_OFFICER = "ict_risk_officer"  # Control function
    CHIEF_INFORMATION_SECURITY_OFFICER = "ciso"
    ICT_OPERATIONS_MANAGER = "ict_operations_manager"
    COMPLIANCE_OFFICER = "compliance_officer"
    INTERNAL_AUDITOR = "internal_auditor"
    DATA_PROTECTION_OFFICER = "dpo"
    BUSINESS_CONTINUITY_MANAGER = "bcm"
    INCIDENT_RESPONSE_LEAD = "incident_response_lead"


class DefenceLine(Enum):
    """Three Lines of Defence model."""
    FIRST_LINE = "first_line"  # Business operations and risk ownership
    SECOND_LINE = "second_line"  # Risk management and compliance
    THIRD_LINE = "third_line"  # Internal audit


class TrainingStatus(Enum):
    """Training completion status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXPIRED = "expired"
    EXEMPTED = "exempted"


class ApprovalStatus(Enum):
    """Approval workflow status."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class AuditFindingSeverity(Enum):
    """Internal audit finding severity."""
    OBSERVATION = "observation"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class GovernanceRoleAssignment:
    """
    Role assignment per DORA Article 5(2).

    Documents who is responsible for ICT risk management functions.
    """
    assignment_id: str = ""
    role: GovernanceRole = GovernanceRole.ICT_RISK_OFFICER
    person_name: str = ""
    person_id: str = ""
    email: str = ""
    phone: str = ""

    # Role details
    defence_line: DefenceLine = DefenceLine.SECOND_LINE
    department: str = ""
    reports_to: str = ""

    # Responsibilities
    responsibilities: List[str] = field(default_factory=list)
    delegated_authorities: List[str] = field(default_factory=list)

    # Status
    is_active: bool = True
    assignment_date: str = ""
    end_date: Optional[str] = None

    # Documentation
    appointment_document: str = ""
    job_description: str = ""

    def __post_init__(self):
        if not self.assignment_id:
            self.assignment_id = f"ROLE-{uuid.uuid4().hex[:8].upper()}"
        if not self.assignment_date:
            self.assignment_date = datetime.now(timezone.utc).isoformat()


@dataclass
class ICTTrainingRecord:
    """
    Training record per DORA Article 5(4).

    Documents ICT training for management body and staff.
    """
    training_id: str = ""
    person_id: str = ""
    person_name: str = ""
    role: GovernanceRole = GovernanceRole.MANAGEMENT_BODY

    # Training details
    training_name: str = ""
    training_type: str = ""  # "mandatory", "recommended", "specialized"
    training_provider: str = ""
    training_description: str = ""

    # Topics covered (per Article 5(4))
    topics: List[str] = field(default_factory=list)

    # Status
    status: TrainingStatus = TrainingStatus.NOT_STARTED
    completion_date: Optional[str] = None
    expiry_date: Optional[str] = None
    score: Optional[float] = None  # Assessment score if applicable
    certificate_reference: str = ""

    # Scheduling
    scheduled_date: Optional[str] = None
    duration_hours: float = 0.0

    # Exemption (for microenterprises per Article 5(4))
    is_exempted: bool = False
    exemption_reason: str = ""

    def __post_init__(self):
        if not self.training_id:
            self.training_id = f"TRN-{uuid.uuid4().hex[:8].upper()}"

    @property
    def is_valid(self) -> bool:
        """Check if training is current and valid."""
        if self.status != TrainingStatus.COMPLETED:
            return False
        if self.expiry_date:
            expiry = datetime.fromisoformat(self.expiry_date.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) > expiry:
                return False
        return True


@dataclass
class FrameworkApproval:
    """
    Framework/strategy approval record per DORA Article 5(1).

    Documents management body approval of ICT risk management framework.
    """
    approval_id: str = ""
    document_type: str = ""  # "ict_risk_framework", "resilience_strategy", "policy"
    document_name: str = ""
    document_version: str = ""
    document_reference: str = ""

    # Approval workflow
    status: ApprovalStatus = ApprovalStatus.DRAFT
    submitted_by: str = ""
    submitted_date: Optional[str] = None
    reviewed_by: str = ""
    reviewed_date: Optional[str] = None
    approved_by: str = ""  # Management body member
    approved_date: Optional[str] = None
    rejection_reason: str = ""

    # Validity
    effective_date: Optional[str] = None
    next_review_date: Optional[str] = None
    supersedes_version: str = ""

    # Content summary
    key_changes: List[str] = field(default_factory=list)
    approval_comments: str = ""

    def __post_init__(self):
        if not self.approval_id:
            self.approval_id = f"APR-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class AuditFinding:
    """
    Internal audit finding related to ICT.

    Per DORA Article 5 and Article 6(5) - framework subject to internal audit.
    """
    finding_id: str = ""
    audit_id: str = ""
    audit_date: str = ""

    # Finding details
    title: str = ""
    description: str = ""
    severity: AuditFindingSeverity = AuditFindingSeverity.MEDIUM
    dora_article_reference: str = ""

    # Affected areas
    affected_controls: List[str] = field(default_factory=list)
    affected_processes: List[str] = field(default_factory=list)

    # Root cause
    root_cause: str = ""
    contributing_factors: List[str] = field(default_factory=list)

    # Recommendation
    recommendation: str = ""
    management_response: str = ""
    agreed_action: str = ""
    action_owner: str = ""
    target_date: Optional[str] = None

    # Status
    is_open: bool = True
    closed_date: Optional[str] = None
    verified_by: str = ""

    def __post_init__(self):
        if not self.finding_id:
            self.finding_id = f"FIND-{uuid.uuid4().hex[:8].upper()}"
        if not self.audit_date:
            self.audit_date = datetime.now(timezone.utc).isoformat()


@dataclass
class ICTBudgetAllocation:
    """
    ICT budget allocation per DORA Article 5(2)(e).

    Management body must approve ICT budget for digital resilience.
    """
    budget_id: str = ""
    fiscal_year: int = 0
    budget_period_start: str = ""
    budget_period_end: str = ""

    # Budget amounts (EUR)
    total_ict_budget_eur: float = 0.0
    security_budget_eur: float = 0.0
    resilience_budget_eur: float = 0.0
    training_budget_eur: float = 0.0
    testing_budget_eur: float = 0.0
    third_party_budget_eur: float = 0.0

    # Approval
    approved_by: str = ""
    approved_date: Optional[str] = None

    # Utilization tracking
    actual_spend_eur: float = 0.0
    utilization_pct: float = 0.0

    def __post_init__(self):
        if not self.budget_id:
            self.budget_id = f"BUD-{uuid.uuid4().hex[:8].upper()}"

    def update_utilization(self, spend: float) -> None:
        """Update actual spend and calculate utilization."""
        self.actual_spend_eur = spend
        if self.total_ict_budget_eur > 0:
            self.utilization_pct = (spend / self.total_ict_budget_eur) * 100


# =============================================================================
# Mandatory Training Topics (Article 5(4))
# =============================================================================

MANDATORY_TRAINING_TOPICS = [
    {
        "topic": "ICT Risk Awareness",
        "description": "Understanding ICT risks and their impact on financial services",
        "duration_hours": 2.0,
        "frequency": "annual",
    },
    {
        "topic": "Cyber Security Fundamentals",
        "description": "Basic cybersecurity concepts and threats",
        "duration_hours": 3.0,
        "frequency": "annual",
    },
    {
        "topic": "DORA Regulatory Overview",
        "description": "Understanding DORA requirements and obligations",
        "duration_hours": 2.0,
        "frequency": "annual",
    },
    {
        "topic": "Incident Response Procedures",
        "description": "How to respond to and report ICT incidents",
        "duration_hours": 1.5,
        "frequency": "annual",
    },
    {
        "topic": "Third-Party ICT Risk",
        "description": "Understanding and managing third-party ICT risks",
        "duration_hours": 1.5,
        "frequency": "annual",
    },
    {
        "topic": "Business Continuity",
        "description": "ICT business continuity and disaster recovery",
        "duration_hours": 2.0,
        "frequency": "annual",
    },
    {
        "topic": "Data Protection and Privacy",
        "description": "GDPR and data protection in ICT context",
        "duration_hours": 2.0,
        "frequency": "biennial",
    },
]


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GovernanceConfig:
    """Configuration for DORA Governance Framework."""
    # Training requirements
    enable_training_tracking: bool = True
    training_expiry_months: int = 12
    training_reminder_days: int = 30
    exempt_microenterprises_from_training: bool = True

    # Approval workflow
    require_management_body_approval: bool = True
    require_dual_approval: bool = False
    approval_timeout_days: int = 14

    # Audit requirements
    enable_audit_tracking: bool = True
    audit_finding_escalation_days: int = 30
    critical_finding_escalation_immediate: bool = True

    # Budget tracking
    enable_budget_tracking: bool = True
    budget_utilization_warning_pct: float = 80.0
    budget_utilization_critical_pct: float = 95.0

    # Logging
    log_all_events: bool = True
    log_path: str = "logs/dora/governance"

    # Callbacks
    escalation_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# Main Governance Framework
# =============================================================================

class DORAGovernanceFramework:
    """
    DORA Article 5 compliant Governance Framework.

    Manages:
    - Management body oversight responsibilities
    - ICT role assignments and segregation of duties
    - Mandatory training tracking
    - Framework/strategy approval workflows
    - Internal audit integration
    - ICT budget approval and tracking

    Usage:
        config = GovernanceConfig(enable_training_tracking=True)
        governance = DORAGovernanceFramework(config)

        # Assign ICT roles
        role = governance.assign_role(
            role=GovernanceRole.ICT_RISK_OFFICER,
            person_name="John Smith",
            email="john.smith@example.com",
        )

        # Track training
        training = governance.record_training(
            person_id="P001",
            training_name="DORA Regulatory Overview",
            status=TrainingStatus.COMPLETED,
        )

        # Approve framework
        approval = governance.submit_for_approval(
            document_type="ict_risk_framework",
            document_name="ICT Risk Management Framework v1.0",
        )
    """

    def __init__(
        self,
        config: Optional[GovernanceConfig] = None,
        is_microenterprise: bool = False,
    ):
        """
        Initialize governance framework.

        Args:
            config: Governance configuration
            is_microenterprise: If True, certain requirements relaxed per Article 5(4)
        """
        self.config = config or GovernanceConfig()
        self.is_microenterprise = is_microenterprise

        # Data stores
        self._roles: Dict[str, GovernanceRoleAssignment] = {}
        self._training_records: Dict[str, List[ICTTrainingRecord]] = {}  # person_id -> records
        self._approvals: Dict[str, FrameworkApproval] = {}
        self._audit_findings: Dict[str, AuditFinding] = {}
        self._budgets: Dict[str, ICTBudgetAllocation] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"DORAGovernanceFramework initialized "
            f"(microenterprise={is_microenterprise})"
        )

    # =========================================================================
    # Role Management (Article 5(2))
    # =========================================================================

    def assign_role(
        self,
        role: GovernanceRole,
        person_name: str,
        email: str,
        person_id: Optional[str] = None,
        phone: str = "",
        department: str = "",
        reports_to: str = "",
        responsibilities: Optional[List[str]] = None,
        delegated_authorities: Optional[List[str]] = None,
    ) -> GovernanceRoleAssignment:
        """
        Assign an ICT governance role.

        Per Article 5(2), roles must be clearly defined with
        appropriate segregation of duties.

        Args:
            role: Governance role to assign
            person_name: Name of the person
            email: Contact email
            person_id: Optional person identifier
            phone: Contact phone
            department: Department name
            reports_to: Reporting line
            responsibilities: List of responsibilities
            delegated_authorities: Delegated authorities

        Returns:
            GovernanceRoleAssignment record
        """
        # Determine defence line based on role
        defence_line = self._get_defence_line(role)

        assignment = GovernanceRoleAssignment(
            role=role,
            person_name=person_name,
            person_id=person_id or f"P-{uuid.uuid4().hex[:8].upper()}",
            email=email,
            phone=phone,
            department=department,
            reports_to=reports_to,
            responsibilities=responsibilities or [],
            delegated_authorities=delegated_authorities or [],
            defence_line=defence_line,
        )

        with self._lock:
            self._roles[assignment.assignment_id] = assignment

        self._log_event("role_assigned", {
            "assignment_id": assignment.assignment_id,
            "role": role.value,
            "person_name": person_name,
        })

        logger.info(f"Role assigned: {role.value} -> {person_name}")
        return assignment

    def get_role_assignment(
        self,
        assignment_id: Optional[str] = None,
        role: Optional[GovernanceRole] = None,
    ) -> Optional[GovernanceRoleAssignment]:
        """Get role assignment by ID or role type."""
        with self._lock:
            if assignment_id:
                return self._roles.get(assignment_id)
            if role:
                for assignment in self._roles.values():
                    if assignment.role == role and assignment.is_active:
                        return assignment
        return None

    def get_all_active_roles(self) -> List[GovernanceRoleAssignment]:
        """Get all active role assignments."""
        with self._lock:
            return [r for r in self._roles.values() if r.is_active]

    def revoke_role(self, assignment_id: str, end_date: Optional[str] = None) -> bool:
        """Revoke a role assignment."""
        with self._lock:
            if assignment_id in self._roles:
                self._roles[assignment_id].is_active = False
                self._roles[assignment_id].end_date = (
                    end_date or datetime.now(timezone.utc).isoformat()
                )
                return True
        return False

    def check_segregation_of_duties(self) -> List[str]:
        """
        Check for segregation of duties violations.

        Returns:
            List of violation descriptions
        """
        violations = []
        active_roles = self.get_all_active_roles()

        # Group roles by person
        person_roles: Dict[str, List[GovernanceRole]] = {}
        for assignment in active_roles:
            if assignment.person_id not in person_roles:
                person_roles[assignment.person_id] = []
            person_roles[assignment.person_id].append(assignment.role)

        # Check incompatible role combinations
        incompatible_pairs = [
            (GovernanceRole.ICT_RISK_OFFICER, GovernanceRole.ICT_OPERATIONS_MANAGER),
            (GovernanceRole.INTERNAL_AUDITOR, GovernanceRole.ICT_OPERATIONS_MANAGER),
            (GovernanceRole.INTERNAL_AUDITOR, GovernanceRole.ICT_RISK_OFFICER),
        ]

        for person_id, roles in person_roles.items():
            for role1, role2 in incompatible_pairs:
                if role1 in roles and role2 in roles:
                    violations.append(
                        f"Person {person_id} has incompatible roles: "
                        f"{role1.value} and {role2.value}"
                    )

        return violations

    def _get_defence_line(self, role: GovernanceRole) -> DefenceLine:
        """Determine defence line for a role."""
        first_line_roles = {
            GovernanceRole.ICT_OPERATIONS_MANAGER,
            GovernanceRole.INCIDENT_RESPONSE_LEAD,
        }
        third_line_roles = {
            GovernanceRole.INTERNAL_AUDITOR,
        }

        if role in first_line_roles:
            return DefenceLine.FIRST_LINE
        if role in third_line_roles:
            return DefenceLine.THIRD_LINE
        return DefenceLine.SECOND_LINE

    # =========================================================================
    # Training Management (Article 5(4))
    # =========================================================================

    def record_training(
        self,
        person_id: str,
        person_name: str,
        training_name: str,
        status: TrainingStatus = TrainingStatus.NOT_STARTED,
        role: GovernanceRole = GovernanceRole.MANAGEMENT_BODY,
        training_type: str = "mandatory",
        training_provider: str = "",
        topics: Optional[List[str]] = None,
        completion_date: Optional[str] = None,
        duration_hours: float = 0.0,
        score: Optional[float] = None,
    ) -> ICTTrainingRecord:
        """
        Record ICT training for a person.

        Per Article 5(4), management body members must maintain sufficient
        knowledge and skills to understand ICT risk.

        Args:
            person_id: Person identifier
            person_name: Person name
            training_name: Name of training
            status: Training status
            role: Person's governance role
            training_type: Type of training
            training_provider: Training provider
            topics: Topics covered
            completion_date: When training was completed
            duration_hours: Training duration
            score: Assessment score

        Returns:
            ICTTrainingRecord
        """
        # Check for microenterprise exemption
        is_exempted = (
            self.is_microenterprise and
            self.config.exempt_microenterprises_from_training and
            role == GovernanceRole.MANAGEMENT_BODY
        )

        # Calculate expiry date
        expiry_date = None
        if completion_date and status == TrainingStatus.COMPLETED:
            completion_dt = datetime.fromisoformat(completion_date.replace("Z", "+00:00"))
            expiry_dt = completion_dt + timedelta(days=self.config.training_expiry_months * 30)
            expiry_date = expiry_dt.isoformat()

        record = ICTTrainingRecord(
            person_id=person_id,
            person_name=person_name,
            role=role,
            training_name=training_name,
            training_type=training_type,
            training_provider=training_provider,
            topics=topics or [],
            status=TrainingStatus.EXEMPTED if is_exempted else status,
            completion_date=completion_date,
            expiry_date=expiry_date,
            duration_hours=duration_hours,
            score=score,
            is_exempted=is_exempted,
            exemption_reason="Microenterprise exemption per Article 5(4)" if is_exempted else "",
        )

        with self._lock:
            if person_id not in self._training_records:
                self._training_records[person_id] = []
            self._training_records[person_id].append(record)

        self._log_event("training_recorded", {
            "training_id": record.training_id,
            "person_id": person_id,
            "training_name": training_name,
            "status": record.status.value,
        })

        logger.info(f"Training recorded: {training_name} for {person_name}")
        return record

    def get_training_records(
        self,
        person_id: Optional[str] = None,
        status: Optional[TrainingStatus] = None,
    ) -> List[ICTTrainingRecord]:
        """Get training records with optional filters."""
        with self._lock:
            if person_id:
                records = list(self._training_records.get(person_id, []))
            else:
                records = []
                for person_records in self._training_records.values():
                    records.extend(person_records)

            if status:
                records = [r for r in records if r.status == status]

            return records

    def get_training_compliance_status(self) -> Dict[str, Any]:
        """
        Get overall training compliance status.

        Returns:
            Dictionary with compliance metrics
        """
        with self._lock:
            all_records = []
            for person_records in self._training_records.values():
                all_records.extend(person_records)

            total = len(all_records)
            completed = sum(1 for r in all_records if r.status == TrainingStatus.COMPLETED)
            expired = sum(1 for r in all_records if r.status == TrainingStatus.EXPIRED)
            exempted = sum(1 for r in all_records if r.status == TrainingStatus.EXEMPTED)
            valid = sum(1 for r in all_records if r.is_valid)

            return {
                "total_records": total,
                "completed": completed,
                "expired": expired,
                "exempted": exempted,
                "valid_current": valid,
                "compliance_rate": (valid / (total - exempted) * 100) if (total - exempted) > 0 else 100.0,
                "microenterprise_exemption_applied": self.is_microenterprise,
            }

    def get_expiring_training(self, days_ahead: int = 30) -> List[ICTTrainingRecord]:
        """Get training records expiring within specified days."""
        cutoff = datetime.now(timezone.utc) + timedelta(days=days_ahead)

        with self._lock:
            expiring = []
            for person_records in self._training_records.values():
                for record in person_records:
                    if record.expiry_date and record.status == TrainingStatus.COMPLETED:
                        expiry = datetime.fromisoformat(record.expiry_date.replace("Z", "+00:00"))
                        if expiry <= cutoff:
                            expiring.append(record)

            return expiring

    def get_mandatory_training_gaps(self) -> Dict[str, List[str]]:
        """
        Identify gaps in mandatory training.

        Returns:
            Dictionary of person_id -> list of missing training topics
        """
        mandatory_topics = {t["topic"] for t in MANDATORY_TRAINING_TOPICS}
        gaps: Dict[str, List[str]] = {}

        with self._lock:
            for assignment in self._roles.values():
                if not assignment.is_active:
                    continue
                if assignment.role == GovernanceRole.MANAGEMENT_BODY:
                    person_id = assignment.person_id
                    completed_topics: set = set()

                    for record in self._training_records.get(person_id, []):
                        if record.is_valid or record.status == TrainingStatus.EXEMPTED:
                            completed_topics.update(record.topics)

                    missing = mandatory_topics - completed_topics
                    if missing:
                        gaps[person_id] = list(missing)

        return gaps

    # =========================================================================
    # Approval Workflow (Article 5(1))
    # =========================================================================

    def submit_for_approval(
        self,
        document_type: str,
        document_name: str,
        document_version: str = "1.0",
        submitted_by: str = "",
        key_changes: Optional[List[str]] = None,
    ) -> FrameworkApproval:
        """
        Submit a document for management body approval.

        Per Article 5(1), management body must approve:
        - ICT risk management framework
        - Digital operational resilience strategy
        - ICT-related policies

        Args:
            document_type: Type of document
            document_name: Document name
            document_version: Version number
            submitted_by: Submitter name
            key_changes: List of key changes

        Returns:
            FrameworkApproval record
        """
        approval = FrameworkApproval(
            document_type=document_type,
            document_name=document_name,
            document_version=document_version,
            submitted_by=submitted_by,
            submitted_date=datetime.now(timezone.utc).isoformat(),
            status=ApprovalStatus.PENDING_REVIEW,
            key_changes=key_changes or [],
        )

        with self._lock:
            self._approvals[approval.approval_id] = approval

        self._log_event("approval_submitted", {
            "approval_id": approval.approval_id,
            "document_type": document_type,
            "document_name": document_name,
        })

        logger.info(f"Document submitted for approval: {document_name}")
        return approval

    def review_approval(
        self,
        approval_id: str,
        reviewed_by: str,
        comments: str = "",
        recommend_approval: bool = True,
    ) -> Optional[FrameworkApproval]:
        """Review a submitted document."""
        with self._lock:
            if approval_id not in self._approvals:
                return None

            approval = self._approvals[approval_id]
            approval.reviewed_by = reviewed_by
            approval.reviewed_date = datetime.now(timezone.utc).isoformat()
            approval.status = (
                ApprovalStatus.PENDING_APPROVAL if recommend_approval
                else ApprovalStatus.REJECTED
            )
            if not recommend_approval:
                approval.rejection_reason = comments
            else:
                approval.approval_comments = comments

        return approval

    def approve_document(
        self,
        approval_id: str,
        approved_by: str,
        effective_date: Optional[str] = None,
        next_review_date: Optional[str] = None,
    ) -> Optional[FrameworkApproval]:
        """
        Approve a document (management body action).

        Args:
            approval_id: Approval ID
            approved_by: Management body member name
            effective_date: When document takes effect
            next_review_date: Next review date

        Returns:
            Updated FrameworkApproval or None if not found
        """
        with self._lock:
            if approval_id not in self._approvals:
                return None

            approval = self._approvals[approval_id]

            if self.config.require_management_body_approval:
                # Verify approver has management body role
                has_authority = any(
                    r.role == GovernanceRole.MANAGEMENT_BODY and r.is_active
                    for r in self._roles.values()
                    if r.person_name == approved_by
                )
                if not has_authority:
                    logger.warning(f"Approver {approved_by} does not have management body role")

            approval.approved_by = approved_by
            approval.approved_date = datetime.now(timezone.utc).isoformat()
            approval.status = ApprovalStatus.APPROVED
            approval.effective_date = effective_date or datetime.now(timezone.utc).isoformat()
            approval.next_review_date = next_review_date

        self._log_event("document_approved", {
            "approval_id": approval_id,
            "approved_by": approved_by,
        })

        logger.info(f"Document approved: {approval.document_name} by {approved_by}")
        return approval

    def reject_document(
        self,
        approval_id: str,
        rejected_by: str,
        rejection_reason: str,
    ) -> Optional[FrameworkApproval]:
        """Reject a document."""
        with self._lock:
            if approval_id not in self._approvals:
                return None

            approval = self._approvals[approval_id]
            approval.status = ApprovalStatus.REJECTED
            approval.rejection_reason = rejection_reason

        return approval

    def get_approval(self, approval_id: str) -> Optional[FrameworkApproval]:
        """Get approval by ID."""
        with self._lock:
            return self._approvals.get(approval_id)

    def get_pending_approvals(self) -> List[FrameworkApproval]:
        """Get all pending approvals."""
        with self._lock:
            return [
                a for a in self._approvals.values()
                if a.status in (ApprovalStatus.PENDING_REVIEW, ApprovalStatus.PENDING_APPROVAL)
            ]

    # =========================================================================
    # Audit Integration (Article 6(5))
    # =========================================================================

    def record_audit_finding(
        self,
        audit_id: str,
        title: str,
        description: str,
        severity: AuditFindingSeverity,
        dora_article_reference: str = "",
        affected_controls: Optional[List[str]] = None,
        recommendation: str = "",
        root_cause: str = "",
    ) -> AuditFinding:
        """
        Record an internal audit finding.

        Per Article 6(5), the ICT risk management framework is
        subject to internal audit.

        Args:
            audit_id: Parent audit identifier
            title: Finding title
            description: Finding description
            severity: Finding severity
            dora_article_reference: Related DORA article
            affected_controls: Affected control areas
            recommendation: Auditor recommendation
            root_cause: Root cause analysis

        Returns:
            AuditFinding record
        """
        finding = AuditFinding(
            audit_id=audit_id,
            title=title,
            description=description,
            severity=severity,
            dora_article_reference=dora_article_reference,
            affected_controls=affected_controls or [],
            recommendation=recommendation,
            root_cause=root_cause,
        )

        with self._lock:
            self._audit_findings[finding.finding_id] = finding

        # Escalate critical findings immediately
        if (
            severity == AuditFindingSeverity.CRITICAL and
            self.config.critical_finding_escalation_immediate
        ):
            self._escalate_finding(finding)

        self._log_event("audit_finding_recorded", {
            "finding_id": finding.finding_id,
            "title": title,
            "severity": severity.value,
        })

        logger.info(f"Audit finding recorded: {title} ({severity.value})")
        return finding

    def update_finding_response(
        self,
        finding_id: str,
        management_response: str,
        agreed_action: str,
        action_owner: str,
        target_date: str,
    ) -> Optional[AuditFinding]:
        """Update management response to an audit finding."""
        with self._lock:
            if finding_id not in self._audit_findings:
                return None

            finding = self._audit_findings[finding_id]
            finding.management_response = management_response
            finding.agreed_action = agreed_action
            finding.action_owner = action_owner
            finding.target_date = target_date

        return finding

    def close_finding(
        self,
        finding_id: str,
        verified_by: str,
    ) -> Optional[AuditFinding]:
        """Close a resolved audit finding."""
        with self._lock:
            if finding_id not in self._audit_findings:
                return None

            finding = self._audit_findings[finding_id]
            finding.is_open = False
            finding.closed_date = datetime.now(timezone.utc).isoformat()
            finding.verified_by = verified_by

        return finding

    def get_open_findings(self) -> List[AuditFinding]:
        """Get all open audit findings."""
        with self._lock:
            return [f for f in self._audit_findings.values() if f.is_open]

    def get_overdue_findings(self) -> List[AuditFinding]:
        """Get findings past their target date."""
        now = datetime.now(timezone.utc)
        overdue = []

        with self._lock:
            for finding in self._audit_findings.values():
                if finding.is_open and finding.target_date:
                    target = datetime.fromisoformat(finding.target_date.replace("Z", "+00:00"))
                    if now > target:
                        overdue.append(finding)

        return overdue

    def _escalate_finding(self, finding: AuditFinding) -> None:
        """Escalate a critical finding."""
        if self.config.escalation_callback:
            try:
                self.config.escalation_callback("critical_audit_finding", asdict(finding))
            except Exception as e:
                logger.error(f"Escalation callback failed: {e}")

        logger.warning(f"Critical audit finding escalated: {finding.finding_id}")

    # =========================================================================
    # Budget Management (Article 5(2)(e))
    # =========================================================================

    def create_budget_allocation(
        self,
        fiscal_year: int,
        total_ict_budget_eur: float,
        security_budget_eur: float = 0.0,
        resilience_budget_eur: float = 0.0,
        training_budget_eur: float = 0.0,
        testing_budget_eur: float = 0.0,
        third_party_budget_eur: float = 0.0,
    ) -> ICTBudgetAllocation:
        """
        Create ICT budget allocation for management body approval.

        Per Article 5(2)(e), management body must approve and oversee
        the ICT budget for digital resilience.

        Args:
            fiscal_year: Fiscal year
            total_ict_budget_eur: Total ICT budget in EUR
            security_budget_eur: Security-specific budget
            resilience_budget_eur: Resilience-specific budget
            training_budget_eur: Training budget
            testing_budget_eur: Testing budget
            third_party_budget_eur: Third-party management budget

        Returns:
            ICTBudgetAllocation record
        """
        budget = ICTBudgetAllocation(
            fiscal_year=fiscal_year,
            budget_period_start=f"{fiscal_year}-01-01T00:00:00Z",
            budget_period_end=f"{fiscal_year}-12-31T23:59:59Z",
            total_ict_budget_eur=total_ict_budget_eur,
            security_budget_eur=security_budget_eur,
            resilience_budget_eur=resilience_budget_eur,
            training_budget_eur=training_budget_eur,
            testing_budget_eur=testing_budget_eur,
            third_party_budget_eur=third_party_budget_eur,
        )

        with self._lock:
            self._budgets[budget.budget_id] = budget

        return budget

    def approve_budget(
        self,
        budget_id: str,
        approved_by: str,
    ) -> Optional[ICTBudgetAllocation]:
        """Approve ICT budget allocation."""
        with self._lock:
            if budget_id not in self._budgets:
                return None

            budget = self._budgets[budget_id]
            budget.approved_by = approved_by
            budget.approved_date = datetime.now(timezone.utc).isoformat()

        return budget

    def update_budget_spend(
        self,
        budget_id: str,
        actual_spend_eur: float,
    ) -> Optional[ICTBudgetAllocation]:
        """Update actual budget spend."""
        with self._lock:
            if budget_id not in self._budgets:
                return None

            budget = self._budgets[budget_id]
            budget.update_utilization(actual_spend_eur)

            # Check thresholds
            if budget.utilization_pct >= self.config.budget_utilization_critical_pct:
                logger.warning(
                    f"Budget utilization critical: {budget.utilization_pct:.1f}%"
                )
            elif budget.utilization_pct >= self.config.budget_utilization_warning_pct:
                logger.info(
                    f"Budget utilization warning: {budget.utilization_pct:.1f}%"
                )

        return budget

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_governance_summary(self) -> Dict[str, Any]:
        """Get governance framework summary."""
        with self._lock:
            active_roles = self.get_all_active_roles()
            segregation_violations = self.check_segregation_of_duties()
            training_status = self.get_training_compliance_status()
            pending_approvals = self.get_pending_approvals()
            open_findings = self.get_open_findings()
            overdue_findings = self.get_overdue_findings()

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "is_microenterprise": self.is_microenterprise,
                "roles": {
                    "total_active": len(active_roles),
                    "by_defence_line": {
                        "first_line": sum(1 for r in active_roles if r.defence_line == DefenceLine.FIRST_LINE),
                        "second_line": sum(1 for r in active_roles if r.defence_line == DefenceLine.SECOND_LINE),
                        "third_line": sum(1 for r in active_roles if r.defence_line == DefenceLine.THIRD_LINE),
                    },
                    "segregation_violations": segregation_violations,
                },
                "training": training_status,
                "approvals": {
                    "pending": len(pending_approvals),
                    "total": len(self._approvals),
                },
                "audit": {
                    "open_findings": len(open_findings),
                    "overdue_findings": len(overdue_findings),
                    "critical_open": sum(
                        1 for f in open_findings
                        if f.severity == AuditFindingSeverity.CRITICAL
                    ),
                },
                "budgets": {
                    "total": len(self._budgets),
                    "approved": sum(1 for b in self._budgets.values() if b.approved_by),
                },
            }

    def export_governance_report(self) -> Dict[str, Any]:
        """Export comprehensive governance report."""
        with self._lock:
            return {
                "report_date": datetime.now(timezone.utc).isoformat(),
                "article_reference": "Article 5",
                "summary": self.get_governance_summary(),
                "roles": [asdict(r) for r in self._roles.values()],
                "training_records": [
                    asdict(r)
                    for records in self._training_records.values()
                    for r in records
                ],
                "approvals": [asdict(a) for a in self._approvals.values()],
                "audit_findings": [asdict(f) for f in self._audit_findings.values()],
                "budgets": [asdict(b) for b in self._budgets.values()],
                "mandatory_training_topics": MANDATORY_TRAINING_TOPICS,
            }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a governance event."""
        if not self.config.log_all_events:
            return

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"governance_events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log governance event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_governance_framework(
    config: Optional[GovernanceConfig] = None,
    is_microenterprise: bool = False,
) -> DORAGovernanceFramework:
    """
    Create a DORAGovernanceFramework instance.

    Args:
        config: Optional governance configuration
        is_microenterprise: If True, apply microenterprise exemptions

    Returns:
        Configured DORAGovernanceFramework instance
    """
    return DORAGovernanceFramework(config=config, is_microenterprise=is_microenterprise)


def get_mandatory_training_topics() -> List[Dict[str, Any]]:
    """
    Get list of mandatory training topics per Article 5(4).

    Returns:
        List of training topic definitions
    """
    return MANDATORY_TRAINING_TOPICS.copy()
