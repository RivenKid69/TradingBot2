# -*- coding: utf-8 -*-
"""
GDPR Phase 8: Quarterly Review Service.

Implements the quarterly compliance review cadence per GDPR Article 5(2) accountability:
- Retention schedule review and validation
- Subprocessor list review
- DSAR metrics analysis
- Incident learnings and remediation tracking
- Data inventory audit
- Policy change tracking

Design Doc Reference:
    - Phase 8: "Quarterly review cadence: retention schedule, subprocessor list,
               DSAR metrics, incident learnings"
    - GDPR Article 5(2): Accountability principle
    - GDPR Article 30: Records of processing activities
    - GDPR Article 28: Processor requirements (subprocessors)

CLOUD ZONE ONLY.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple
from uuid import uuid4


logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

REVIEW_CYCLE_DAYS: Final[int] = 90  # Quarterly = 90 days
ADVANCE_NOTICE_DAYS: Final[int] = 14  # Notify 2 weeks before due
OVERDUE_THRESHOLD_DAYS: Final[int] = 7  # Overdue after 7 days past due date


class ReviewType(str, Enum):
    """Type of quarterly review."""
    RETENTION = "retention"              # Retention schedule review
    SUBPROCESSOR = "subprocessor"        # Subprocessor list review
    DSAR = "dsar"                        # DSAR metrics review
    INCIDENT = "incident"                # Incident learnings
    INVENTORY = "inventory"              # Data inventory audit
    POLICY = "policy"                    # Policy changes review
    FULL_QUARTERLY = "full_quarterly"    # Full quarterly review


class ReviewStatus(str, Enum):
    """Status of a review."""
    SCHEDULED = "scheduled"
    DUE_SOON = "due_soon"
    IN_PROGRESS = "in_progress"
    PENDING_APPROVAL = "pending_approval"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


class FindingSeverity(str, Enum):
    """Severity of a review finding."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingStatus(str, Enum):
    """Status of a finding."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ACCEPTED = "accepted"  # Risk accepted
    DEFERRED = "deferred"


class SubprocessorStatus(str, Enum):
    """Status of a subprocessor."""
    ACTIVE = "active"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    UNDER_REVIEW = "under_review"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ReviewFinding:
    """A finding from a compliance review."""
    id: str = field(default_factory=lambda: str(uuid4()))
    review_id: str = ""
    finding_type: str = ""
    severity: FindingSeverity = FindingSeverity.MEDIUM
    status: FindingStatus = FindingStatus.OPEN
    title: str = ""
    description: str = ""
    evidence: str = ""
    recommendation: str = ""
    owner: Optional[str] = None
    due_date: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    resolution_notes: str = ""

    @property
    def is_overdue(self) -> bool:
        """Check if finding resolution is overdue."""
        if self.status in (FindingStatus.RESOLVED, FindingStatus.ACCEPTED):
            return False
        if self.due_date and datetime.now(timezone.utc) > self.due_date:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "review_id": self.review_id,
            "finding_type": self.finding_type,
            "severity": self.severity.value,
            "status": self.status.value,
            "title": self.title,
            "description": self.description,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
            "owner": self.owner,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "is_overdue": self.is_overdue,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_notes": self.resolution_notes,
        }


@dataclass
class Subprocessor:
    """A subprocessor record per GDPR Art. 28."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    category: str = ""                   # cloud, analytics, support, etc.
    data_processed: List[str] = field(default_factory=list)  # Categories of data
    processing_location: str = ""        # EU, US, etc.
    eu_compliant: bool = True
    dpa_signed: bool = False
    dpa_signed_date: Optional[datetime] = None
    scc_in_place: bool = False           # Standard Contractual Clauses
    security_certified: List[str] = field(default_factory=list)  # ISO27001, SOC2, etc.
    status: SubprocessorStatus = SubprocessorStatus.ACTIVE
    last_reviewed_at: Optional[datetime] = None
    last_reviewed_by: Optional[str] = None
    next_review_due: Optional[datetime] = None
    notes: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_review_due(self) -> bool:
        """Check if review is due."""
        if not self.next_review_due:
            return True
        return datetime.now(timezone.utc) >= self.next_review_due

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "data_processed": self.data_processed,
            "processing_location": self.processing_location,
            "compliance": {
                "eu_compliant": self.eu_compliant,
                "dpa_signed": self.dpa_signed,
                "dpa_signed_date": self.dpa_signed_date.isoformat() if self.dpa_signed_date else None,
                "scc_in_place": self.scc_in_place,
                "security_certified": self.security_certified,
            },
            "status": self.status.value,
            "review": {
                "last_reviewed_at": self.last_reviewed_at.isoformat() if self.last_reviewed_at else None,
                "last_reviewed_by": self.last_reviewed_by,
                "next_review_due": self.next_review_due.isoformat() if self.next_review_due else None,
                "is_review_due": self.is_review_due,
            },
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class IncidentLearning:
    """An incident learning record."""
    id: str = field(default_factory=lambda: str(uuid4()))
    incident_id: str = ""
    incident_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    incident_type: str = ""              # data_breach, access_violation, etc.
    severity: str = ""                   # P1, P2, P3, P4
    description: str = ""
    root_cause: str = ""
    impact_summary: str = ""
    data_affected: List[str] = field(default_factory=list)
    users_affected: int = 0
    remediation_taken: str = ""
    preventive_measures: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    status: str = "open"                 # open, in_progress, closed
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "incident_id": self.incident_id,
            "incident_date": self.incident_date.isoformat(),
            "incident_type": self.incident_type,
            "severity": self.severity,
            "description": self.description,
            "root_cause": self.root_cause,
            "impact": {
                "summary": self.impact_summary,
                "data_affected": self.data_affected,
                "users_affected": self.users_affected,
            },
            "response": {
                "remediation_taken": self.remediation_taken,
                "preventive_measures": self.preventive_measures,
                "lessons_learned": self.lessons_learned,
            },
            "status": self.status,
            "review": {
                "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
                "reviewed_by": self.reviewed_by,
            },
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class RetentionReviewItem:
    """Item in retention schedule review."""
    data_type: str = ""
    current_retention_days: int = 0
    min_retention_days: int = 0
    max_retention_days: Optional[int] = None
    recommended_retention_days: int = 0
    change_proposed: bool = False
    change_reason: str = ""
    approved: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_type": self.data_type,
            "retention": {
                "current_days": self.current_retention_days,
                "min_days": self.min_retention_days,
                "max_days": self.max_retention_days,
                "recommended_days": self.recommended_retention_days,
            },
            "change": {
                "proposed": self.change_proposed,
                "reason": self.change_reason,
            },
            "approval": {
                "approved": self.approved,
                "approved_by": self.approved_by,
                "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            },
        }


@dataclass
class QuarterlyReview:
    """A quarterly compliance review."""
    id: str = field(default_factory=lambda: str(uuid4()))
    review_type: ReviewType = ReviewType.FULL_QUARTERLY
    status: ReviewStatus = ReviewStatus.SCHEDULED
    quarter: str = ""                    # e.g., "2025-Q1"
    scheduled_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    due_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=REVIEW_CYCLE_DAYS))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Participants
    lead_reviewer: Optional[str] = None
    reviewers: List[str] = field(default_factory=list)
    approvers: List[str] = field(default_factory=list)

    # Scope
    workspaces_in_scope: List[str] = field(default_factory=list)
    data_types_in_scope: List[str] = field(default_factory=list)

    # Results
    findings: List[ReviewFinding] = field(default_factory=list)
    retention_items: List[RetentionReviewItem] = field(default_factory=list)
    subprocessors_reviewed: List[str] = field(default_factory=list)
    incidents_reviewed: List[str] = field(default_factory=list)

    # Summary
    summary: str = ""
    executive_summary: str = ""
    risk_assessment: str = ""
    recommendations: List[str] = field(default_factory=list)

    # Metrics snapshot
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)

    # Approval
    approved: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    approval_notes: str = ""

    # Audit
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None

    # Integrity
    integrity_hash: str = ""

    def __post_init__(self):
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute integrity hash."""
        content = json.dumps({
            "id": self.id,
            "review_type": self.review_type.value,
            "quarter": self.quarter,
            "findings_count": len(self.findings),
            "status": self.status.value,
        }, sort_keys=True)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    @property
    def is_overdue(self) -> bool:
        """Check if review is overdue."""
        if self.status == ReviewStatus.COMPLETED:
            return False
        return datetime.now(timezone.utc) > self.due_date + timedelta(days=OVERDUE_THRESHOLD_DAYS)

    @property
    def days_until_due(self) -> int:
        """Days until due date."""
        delta = self.due_date - datetime.now(timezone.utc)
        return delta.days

    @property
    def critical_findings(self) -> int:
        """Count of critical findings."""
        return sum(1 for f in self.findings if f.severity == FindingSeverity.CRITICAL)

    @property
    def open_findings(self) -> int:
        """Count of open findings."""
        return sum(1 for f in self.findings if f.status in (FindingStatus.OPEN, FindingStatus.IN_PROGRESS))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "review_type": self.review_type.value,
            "status": self.status.value,
            "quarter": self.quarter,
            "schedule": {
                "scheduled_date": self.scheduled_date.isoformat(),
                "due_date": self.due_date.isoformat(),
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "is_overdue": self.is_overdue,
                "days_until_due": self.days_until_due,
            },
            "participants": {
                "lead_reviewer": self.lead_reviewer,
                "reviewers": self.reviewers,
                "approvers": self.approvers,
            },
            "scope": {
                "workspaces": self.workspaces_in_scope,
                "data_types": self.data_types_in_scope,
            },
            "results": {
                "findings_count": len(self.findings),
                "critical_findings": self.critical_findings,
                "open_findings": self.open_findings,
                "findings": [f.to_dict() for f in self.findings],
                "retention_items": [r.to_dict() for r in self.retention_items],
                "subprocessors_reviewed": self.subprocessors_reviewed,
                "incidents_reviewed": self.incidents_reviewed,
            },
            "summary": {
                "summary": self.summary,
                "executive_summary": self.executive_summary,
                "risk_assessment": self.risk_assessment,
                "recommendations": self.recommendations,
            },
            "metrics_snapshot": self.metrics_snapshot,
            "approval": {
                "approved": self.approved,
                "approved_by": self.approved_by,
                "approved_at": self.approved_at.isoformat() if self.approved_at else None,
                "approval_notes": self.approval_notes,
            },
            "audit": {
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
                "created_by": self.created_by,
            },
            "integrity_hash": self.integrity_hash,
        }


@dataclass
class ReviewSchedule:
    """Schedule of upcoming reviews."""
    reviews: List[QuarterlyReview] = field(default_factory=list)
    next_review: Optional[QuarterlyReview] = None
    overdue_reviews: List[QuarterlyReview] = field(default_factory=list)
    due_soon_reviews: List[QuarterlyReview] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_scheduled": len(self.reviews),
            "next_review": self.next_review.to_dict() if self.next_review else None,
            "overdue_count": len(self.overdue_reviews),
            "overdue": [r.to_dict() for r in self.overdue_reviews],
            "due_soon_count": len(self.due_soon_reviews),
            "due_soon": [r.to_dict() for r in self.due_soon_reviews],
        }


# ============================================================================
# Quarterly Review Service
# ============================================================================

class QuarterlyReviewService:
    """
    Quarterly Review Service.

    Manages the quarterly compliance review cadence:
    - Schedules reviews
    - Tracks review progress
    - Records findings
    - Manages subprocessor list
    - Tracks incident learnings
    - Generates review reports

    Usage:
        service = QuarterlyReviewService(
            retention_registry=retention,
            dashboard_service=dashboard,
            inventory_registry=inventory,
        )

        # Schedule next quarterly review
        review = service.schedule_review(
            review_type=ReviewType.FULL_QUARTERLY,
            lead_reviewer="dpo@company.com",
        )

        # Start review
        service.start_review(review.id, started_by="dpo@company.com")

        # Add findings
        service.add_finding(review.id, ReviewFinding(
            title="Missing retention policy",
            severity=FindingSeverity.HIGH,
        ))

        # Complete review
        service.complete_review(review.id, summary="...")
    """

    def __init__(
        self,
        retention_registry: Optional[Any] = None,
        dashboard_service: Optional[Any] = None,
        inventory_registry: Optional[Any] = None,
        on_review_due: Optional[Callable[[QuarterlyReview], None]] = None,
        on_finding_created: Optional[Callable[[ReviewFinding], None]] = None,
    ):
        """
        Initialize the service.

        Args:
            retention_registry: Retention policy registry
            dashboard_service: Compliance dashboard service
            inventory_registry: Data inventory registry
            on_review_due: Callback when review becomes due
            on_finding_created: Callback when finding is created
        """
        self._retention = retention_registry
        self._dashboard = dashboard_service
        self._inventory = inventory_registry
        self._on_review_due = on_review_due
        self._on_finding_created = on_finding_created

        self._reviews: Dict[str, QuarterlyReview] = {}
        self._findings: Dict[str, ReviewFinding] = {}
        self._subprocessors: Dict[str, Subprocessor] = {}
        self._incidents: Dict[str, IncidentLearning] = {}
        self._audit_log: List[Dict[str, Any]] = []

        self._lock = threading.Lock()

    # ========================================================================
    # Review Management
    # ========================================================================

    def schedule_review(
        self,
        review_type: ReviewType = ReviewType.FULL_QUARTERLY,
        scheduled_date: Optional[datetime] = None,
        lead_reviewer: Optional[str] = None,
        reviewers: Optional[List[str]] = None,
        approvers: Optional[List[str]] = None,
        created_by: Optional[str] = None,
    ) -> QuarterlyReview:
        """
        Schedule a new quarterly review.

        Args:
            review_type: Type of review
            scheduled_date: When to schedule (defaults to now)
            lead_reviewer: Lead reviewer email
            reviewers: List of reviewer emails
            approvers: List of approver emails
            created_by: User creating the review

        Returns:
            Created QuarterlyReview
        """
        scheduled_date = scheduled_date or datetime.now(timezone.utc)

        # Calculate quarter
        quarter = f"{scheduled_date.year}-Q{(scheduled_date.month - 1) // 3 + 1}"

        review = QuarterlyReview(
            review_type=review_type,
            quarter=quarter,
            scheduled_date=scheduled_date,
            due_date=scheduled_date + timedelta(days=REVIEW_CYCLE_DAYS),
            lead_reviewer=lead_reviewer,
            reviewers=reviewers or [],
            approvers=approvers or [],
            created_by=created_by,
        )

        with self._lock:
            self._reviews[review.id] = review

        self._log_audit("review_scheduled", {
            "review_id": review.id,
            "review_type": review_type.value,
            "quarter": quarter,
            "created_by": created_by,
        })

        logger.info(f"Scheduled quarterly review: {review.id} ({quarter})")
        return review

    def start_review(
        self,
        review_id: str,
        started_by: str,
    ) -> Optional[QuarterlyReview]:
        """Start a scheduled review."""
        with self._lock:
            review = self._reviews.get(review_id)
            if not review:
                return None

            if review.status not in (ReviewStatus.SCHEDULED, ReviewStatus.DUE_SOON, ReviewStatus.OVERDUE):
                raise ValueError(f"Cannot start review in status: {review.status.value}")

            review.status = ReviewStatus.IN_PROGRESS
            review.started_at = datetime.now(timezone.utc)
            review.updated_at = datetime.now(timezone.utc)

        self._log_audit("review_started", {
            "review_id": review_id,
            "started_by": started_by,
        })

        # Collect initial metrics snapshot
        self._collect_metrics_snapshot(review)

        logger.info(f"Started quarterly review: {review_id}")
        return review

    def complete_review(
        self,
        review_id: str,
        completed_by: str,
        summary: str,
        executive_summary: str = "",
        risk_assessment: str = "",
        recommendations: Optional[List[str]] = None,
    ) -> Optional[QuarterlyReview]:
        """Complete a review."""
        with self._lock:
            review = self._reviews.get(review_id)
            if not review:
                return None

            if review.status != ReviewStatus.IN_PROGRESS:
                raise ValueError(f"Cannot complete review in status: {review.status.value}")

            review.status = ReviewStatus.PENDING_APPROVAL
            review.completed_at = datetime.now(timezone.utc)
            review.updated_at = datetime.now(timezone.utc)
            review.summary = summary
            review.executive_summary = executive_summary
            review.risk_assessment = risk_assessment
            review.recommendations = recommendations or []
            review.integrity_hash = review._compute_hash()

        self._log_audit("review_completed", {
            "review_id": review_id,
            "completed_by": completed_by,
            "findings_count": len(review.findings),
        })

        logger.info(f"Completed quarterly review: {review_id}")
        return review

    def approve_review(
        self,
        review_id: str,
        approved_by: str,
        approval_notes: str = "",
    ) -> Optional[QuarterlyReview]:
        """Approve a completed review."""
        with self._lock:
            review = self._reviews.get(review_id)
            if not review:
                return None

            if review.status != ReviewStatus.PENDING_APPROVAL:
                raise ValueError(f"Cannot approve review in status: {review.status.value}")

            review.status = ReviewStatus.COMPLETED
            review.approved = True
            review.approved_by = approved_by
            review.approved_at = datetime.now(timezone.utc)
            review.approval_notes = approval_notes
            review.updated_at = datetime.now(timezone.utc)
            review.integrity_hash = review._compute_hash()

        self._log_audit("review_approved", {
            "review_id": review_id,
            "approved_by": approved_by,
        })

        # Schedule next review
        self._schedule_next_review(review)

        logger.info(f"Approved quarterly review: {review_id}")
        return review

    def get_review(self, review_id: str) -> Optional[QuarterlyReview]:
        """Get a review by ID."""
        with self._lock:
            return self._reviews.get(review_id)

    def get_reviews(
        self,
        status: Optional[ReviewStatus] = None,
        review_type: Optional[ReviewType] = None,
        quarter: Optional[str] = None,
    ) -> List[QuarterlyReview]:
        """Get reviews with optional filters."""
        with self._lock:
            reviews = list(self._reviews.values())

        if status:
            reviews = [r for r in reviews if r.status == status]
        if review_type:
            reviews = [r for r in reviews if r.review_type == review_type]
        if quarter:
            reviews = [r for r in reviews if r.quarter == quarter]

        return sorted(reviews, key=lambda r: r.due_date)

    def get_schedule(self) -> ReviewSchedule:
        """Get the review schedule."""
        now = datetime.now(timezone.utc)

        with self._lock:
            reviews = list(self._reviews.values())

        # Update status based on due dates
        for review in reviews:
            if review.status == ReviewStatus.SCHEDULED:
                if review.days_until_due <= ADVANCE_NOTICE_DAYS:
                    review.status = ReviewStatus.DUE_SOON
                if review.is_overdue:
                    review.status = ReviewStatus.OVERDUE

        # Categorize
        schedule = ReviewSchedule()
        schedule.reviews = sorted(reviews, key=lambda r: r.due_date)

        for review in schedule.reviews:
            if review.status == ReviewStatus.OVERDUE:
                schedule.overdue_reviews.append(review)
            elif review.status == ReviewStatus.DUE_SOON:
                schedule.due_soon_reviews.append(review)

        # Next review
        upcoming = [r for r in schedule.reviews if r.status in (
            ReviewStatus.SCHEDULED,
            ReviewStatus.DUE_SOON,
            ReviewStatus.OVERDUE,
            ReviewStatus.IN_PROGRESS,
        )]
        if upcoming:
            schedule.next_review = upcoming[0]

        return schedule

    # ========================================================================
    # Findings Management
    # ========================================================================

    def add_finding(
        self,
        review_id: str,
        finding: ReviewFinding,
    ) -> ReviewFinding:
        """Add a finding to a review."""
        with self._lock:
            review = self._reviews.get(review_id)
            if not review:
                raise ValueError(f"Review {review_id} not found")

            finding.review_id = review_id
            review.findings.append(finding)
            review.updated_at = datetime.now(timezone.utc)
            self._findings[finding.id] = finding

        self._log_audit("finding_created", {
            "finding_id": finding.id,
            "review_id": review_id,
            "severity": finding.severity.value,
            "title": finding.title,
        })

        if self._on_finding_created:
            try:
                self._on_finding_created(finding)
            except Exception as e:
                logger.error(f"Finding callback error: {e}")

        logger.info(f"Added finding to review {review_id}: {finding.title}")
        return finding

    def update_finding(
        self,
        finding_id: str,
        status: Optional[FindingStatus] = None,
        owner: Optional[str] = None,
        due_date: Optional[datetime] = None,
        resolution_notes: Optional[str] = None,
    ) -> Optional[ReviewFinding]:
        """Update a finding."""
        with self._lock:
            finding = self._findings.get(finding_id)
            if not finding:
                return None

            if status:
                finding.status = status
                if status == FindingStatus.RESOLVED:
                    finding.resolved_at = datetime.now(timezone.utc)
            if owner:
                finding.owner = owner
            if due_date:
                finding.due_date = due_date
            if resolution_notes:
                finding.resolution_notes = resolution_notes

        self._log_audit("finding_updated", {
            "finding_id": finding_id,
            "new_status": status.value if status else None,
        })

        return finding

    def get_findings(
        self,
        review_id: Optional[str] = None,
        status: Optional[FindingStatus] = None,
        severity: Optional[FindingSeverity] = None,
    ) -> List[ReviewFinding]:
        """Get findings with optional filters."""
        with self._lock:
            findings = list(self._findings.values())

        if review_id:
            findings = [f for f in findings if f.review_id == review_id]
        if status:
            findings = [f for f in findings if f.status == status]
        if severity:
            findings = [f for f in findings if f.severity == severity]

        return findings

    # ========================================================================
    # Subprocessor Management
    # ========================================================================

    def add_subprocessor(
        self,
        subprocessor: Subprocessor,
    ) -> Subprocessor:
        """Add a subprocessor."""
        with self._lock:
            self._subprocessors[subprocessor.id] = subprocessor

        self._log_audit("subprocessor_added", {
            "subprocessor_id": subprocessor.id,
            "name": subprocessor.name,
            "eu_compliant": subprocessor.eu_compliant,
        })

        logger.info(f"Added subprocessor: {subprocessor.name}")
        return subprocessor

    def update_subprocessor(
        self,
        subprocessor_id: str,
        reviewed_by: str,
        status: Optional[SubprocessorStatus] = None,
        notes: Optional[str] = None,
    ) -> Optional[Subprocessor]:
        """Update/review a subprocessor."""
        with self._lock:
            sub = self._subprocessors.get(subprocessor_id)
            if not sub:
                return None

            sub.last_reviewed_at = datetime.now(timezone.utc)
            sub.last_reviewed_by = reviewed_by
            sub.next_review_due = datetime.now(timezone.utc) + timedelta(days=REVIEW_CYCLE_DAYS)
            sub.updated_at = datetime.now(timezone.utc)

            if status:
                sub.status = status
            if notes:
                sub.notes = notes

        self._log_audit("subprocessor_reviewed", {
            "subprocessor_id": subprocessor_id,
            "reviewed_by": reviewed_by,
        })

        return sub

    def get_subprocessors(
        self,
        status: Optional[SubprocessorStatus] = None,
        eu_compliant: Optional[bool] = None,
        review_due: bool = False,
    ) -> List[Subprocessor]:
        """Get subprocessors with optional filters."""
        with self._lock:
            subs = list(self._subprocessors.values())

        if status:
            subs = [s for s in subs if s.status == status]
        if eu_compliant is not None:
            subs = [s for s in subs if s.eu_compliant == eu_compliant]
        if review_due:
            subs = [s for s in subs if s.is_review_due]

        return subs

    def get_subprocessor_summary(self) -> Dict[str, Any]:
        """Get summary of subprocessors."""
        with self._lock:
            subs = list(self._subprocessors.values())

        return {
            "total": len(subs),
            "active": sum(1 for s in subs if s.status == SubprocessorStatus.ACTIVE),
            "eu_compliant": sum(1 for s in subs if s.eu_compliant),
            "non_eu": sum(1 for s in subs if not s.eu_compliant),
            "review_due": sum(1 for s in subs if s.is_review_due),
            "with_dpa": sum(1 for s in subs if s.dpa_signed),
            "with_scc": sum(1 for s in subs if s.scc_in_place),
            "by_category": self._group_by(subs, lambda s: s.category),
            "by_location": self._group_by(subs, lambda s: s.processing_location),
        }

    def _group_by(self, items: list, key_fn: Callable) -> Dict[str, int]:
        """Group items by key."""
        result: Dict[str, int] = {}
        for item in items:
            key = key_fn(item)
            result[key] = result.get(key, 0) + 1
        return result

    # ========================================================================
    # Incident Learnings
    # ========================================================================

    def add_incident_learning(
        self,
        incident: IncidentLearning,
    ) -> IncidentLearning:
        """Add an incident learning."""
        with self._lock:
            self._incidents[incident.id] = incident

        self._log_audit("incident_learning_added", {
            "incident_id": incident.incident_id,
            "incident_type": incident.incident_type,
            "severity": incident.severity,
        })

        logger.info(f"Added incident learning: {incident.incident_id}")
        return incident

    def review_incident(
        self,
        incident_id: str,
        reviewed_by: str,
        lessons_learned: Optional[List[str]] = None,
        preventive_measures: Optional[List[str]] = None,
    ) -> Optional[IncidentLearning]:
        """Review an incident and document learnings."""
        with self._lock:
            incident = self._incidents.get(incident_id)
            if not incident:
                return None

            incident.reviewed_at = datetime.now(timezone.utc)
            incident.reviewed_by = reviewed_by
            incident.status = "closed"

            if lessons_learned:
                incident.lessons_learned.extend(lessons_learned)
            if preventive_measures:
                incident.preventive_measures.extend(preventive_measures)

        self._log_audit("incident_reviewed", {
            "incident_id": incident_id,
            "reviewed_by": reviewed_by,
        })

        return incident

    def get_incidents(
        self,
        status: Optional[str] = None,
        incident_type: Optional[str] = None,
        reviewed: Optional[bool] = None,
    ) -> List[IncidentLearning]:
        """Get incidents with optional filters."""
        with self._lock:
            incidents = list(self._incidents.values())

        if status:
            incidents = [i for i in incidents if i.status == status]
        if incident_type:
            incidents = [i for i in incidents if i.incident_type == incident_type]
        if reviewed is not None:
            if reviewed:
                incidents = [i for i in incidents if i.reviewed_at is not None]
            else:
                incidents = [i for i in incidents if i.reviewed_at is None]

        return incidents

    # ========================================================================
    # Retention Review
    # ========================================================================

    def review_retention_schedule(
        self,
        review_id: str,
    ) -> List[RetentionReviewItem]:
        """Review retention schedule as part of quarterly review."""
        items = []

        if self._retention and hasattr(self._retention, 'get_all_policies'):
            try:
                policies = self._retention.get_all_policies()

                for policy in policies:
                    item = RetentionReviewItem(
                        data_type=policy.data_type,
                        current_retention_days=policy.retention_days,
                    )

                    # Get min/max from constants if available
                    if hasattr(self._retention, 'MIN_RETENTION_DAYS'):
                        min_days = self._retention.MIN_RETENTION_DAYS.get(policy.data_type, 1)
                        item.min_retention_days = min_days

                    if hasattr(self._retention, 'MAX_RETENTION_DAYS'):
                        max_days = self._retention.MAX_RETENTION_DAYS.get(policy.data_type)
                        item.max_retention_days = max_days

                    items.append(item)
            except Exception as e:
                logger.warning(f"Could not review retention: {e}")

        # Store in review
        with self._lock:
            review = self._reviews.get(review_id)
            if review:
                review.retention_items = items
                review.updated_at = datetime.now(timezone.utc)

        return items

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _collect_metrics_snapshot(self, review: QuarterlyReview) -> None:
        """Collect metrics snapshot for review."""
        snapshot = {
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }

        if self._dashboard:
            try:
                dashboard = self._dashboard.generate_dashboard()
                snapshot["compliance_dashboard"] = dashboard.to_dict()
            except Exception as e:
                logger.warning(f"Could not collect dashboard metrics: {e}")

        if self._inventory:
            try:
                if hasattr(self._inventory, 'generate_report'):
                    report = self._inventory.generate_report()
                    snapshot["inventory_report"] = report.to_dict() if hasattr(report, 'to_dict') else report
            except Exception as e:
                logger.warning(f"Could not collect inventory metrics: {e}")

        review.metrics_snapshot = snapshot

    def _schedule_next_review(self, completed_review: QuarterlyReview) -> None:
        """Schedule the next quarterly review."""
        next_date = completed_review.completed_at + timedelta(days=REVIEW_CYCLE_DAYS)

        self.schedule_review(
            review_type=completed_review.review_type,
            scheduled_date=next_date,
            lead_reviewer=completed_review.lead_reviewer,
            reviewers=completed_review.reviewers,
            approvers=completed_review.approvers,
            created_by="system",
        )

    def _log_audit(self, action: str, details: Dict[str, Any]) -> None:
        """Log an audit event."""
        event = {
            "action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **details,
        }
        self._audit_log.append(event)

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit log."""
        return self._audit_log[-limit:]

    # ========================================================================
    # Report Generation
    # ========================================================================

    def generate_review_report(
        self,
        review_id: str,
    ) -> Dict[str, Any]:
        """Generate a comprehensive review report."""
        review = self.get_review(review_id)
        if not review:
            raise ValueError(f"Review {review_id} not found")

        return {
            "report_type": "quarterly_compliance_review",
            "report_version": "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "review": review.to_dict(),
            "subprocessor_summary": self.get_subprocessor_summary(),
            "open_findings": [f.to_dict() for f in self.get_findings(status=FindingStatus.OPEN)],
            "unreviewed_incidents": [i.to_dict() for i in self.get_incidents(reviewed=False)],
        }

    def export_review(
        self,
        review_id: str,
        output_path: Path,
    ) -> None:
        """Export review report to file."""
        report = self.generate_review_report(review_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"Exported review report to {output_path}")


# ============================================================================
# Pre-populated Subprocessors
# ============================================================================

def create_default_subprocessors() -> List[Subprocessor]:
    """Create default subprocessor list for CCEA platform."""
    return [
        Subprocessor(
            name="AWS (Amazon Web Services)",
            description="Cloud infrastructure provider",
            category="cloud",
            data_processed=["telemetry", "config", "logs", "backups"],
            processing_location="EU (Frankfurt, Ireland)",
            eu_compliant=True,
            dpa_signed=True,
            scc_in_place=True,
            security_certified=["ISO27001", "SOC2", "C5"],
            status=SubprocessorStatus.APPROVED,
        ),
        Subprocessor(
            name="PostgreSQL (AWS RDS)",
            description="Managed database service",
            category="database",
            data_processed=["user_data", "workspace_data", "audit_logs"],
            processing_location="EU (Frankfurt)",
            eu_compliant=True,
            dpa_signed=True,
            security_certified=["ISO27001", "SOC2"],
            status=SubprocessorStatus.APPROVED,
        ),
        Subprocessor(
            name="Redis (AWS ElastiCache)",
            description="In-memory cache",
            category="cache",
            data_processed=["session_data", "cache"],
            processing_location="EU (Frankfurt)",
            eu_compliant=True,
            dpa_signed=True,
            status=SubprocessorStatus.APPROVED,
        ),
    ]


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Constants
    "REVIEW_CYCLE_DAYS",
    "ADVANCE_NOTICE_DAYS",
    "OVERDUE_THRESHOLD_DAYS",
    # Enums
    "ReviewType",
    "ReviewStatus",
    "FindingSeverity",
    "FindingStatus",
    "SubprocessorStatus",
    # Data classes
    "ReviewFinding",
    "Subprocessor",
    "IncidentLearning",
    "RetentionReviewItem",
    "QuarterlyReview",
    "ReviewSchedule",
    # Service
    "QuarterlyReviewService",
    # Factory
    "create_default_subprocessors",
]
