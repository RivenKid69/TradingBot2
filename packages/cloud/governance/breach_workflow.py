"""
Breach Workflow Service - Phase 7 GDPR Implementation

Implements GDPR Articles 33-34 breach notification requirements:
- Personal data breach assessment
- 72-hour notification to supervisory authority
- Data subject notification when required
- Decision tree for notification requirements
- Tabletop exercises for breach response testing

Reference: GDPR Articles 33-34, EDPB Guidelines WP250 rev.01
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4


# =============================================================================
# Enums and Constants
# =============================================================================

class BreachCategory(str, Enum):
    """Category of data breach per GDPR Art. 4(12)."""
    CONFIDENTIALITY = "confidentiality"  # Unauthorized disclosure/access
    INTEGRITY = "integrity"              # Unauthorized modification
    AVAILABILITY = "availability"        # Loss of access/destruction


class BreachSeverity(str, Enum):
    """Severity levels for breach assessment."""
    LOW = "low"             # Unlikely risk to individuals
    MEDIUM = "medium"       # Some risk, may require notification
    HIGH = "high"           # Likely significant risk
    CRITICAL = "critical"   # High risk, immediate notification required


class BreachStatus(str, Enum):
    """Status of a breach incident."""
    DETECTED = "detected"
    CONFIRMED = "confirmed"
    ASSESSING = "assessing"
    ASSESSED = "assessed"
    NOTIFYING = "notifying"
    CONTAINED = "contained"
    REMEDIATED = "remediated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class NotificationStatus(str, Enum):
    """Status of breach notification."""
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    DRAFT = "draft"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    COMPLETE = "complete"


class ExemptionType(str, Enum):
    """Art. 34(3) exemptions from data subject notification."""
    ENCRYPTION = "encryption"               # (a) Data rendered unintelligible
    SUBSEQUENT_MEASURES = "subsequent_measures"  # (b) Risk no longer likely
    DISPROPORTIONATE_EFFORT = "disproportionate_effort"  # (c) Use public communication


class RiskFactor(str, Enum):
    """Risk factors for breach assessment."""
    DATA_SENSITIVITY = "data_sensitivity"
    DATA_VOLUME = "data_volume"
    IDENTIFIABILITY = "identifiability"
    SPECIAL_CATEGORIES = "special_categories"
    VULNERABLE_INDIVIDUALS = "vulnerable_individuals"
    POTENTIAL_HARM = "potential_harm"
    CROSS_BORDER = "cross_border"
    UNAUTHORIZED_ACCESS_TYPE = "unauthorized_access_type"


class TimelineEventType(str, Enum):
    """Types of events in breach timeline."""
    DETECTED = "detected"
    AWARENESS = "awareness"  # Starts 72h clock
    REPORTED_INTERNALLY = "reported_internally"
    ASSESSMENT_STARTED = "assessment_started"
    ASSESSMENT_COMPLETED = "assessment_completed"
    DECISION_MADE = "decision_made"
    AUTHORITY_NOTIFIED = "authority_notified"
    AUTHORITY_ACKNOWLEDGED = "authority_acknowledged"
    SUBJECTS_NOTIFIED = "subjects_notified"
    CONTAINMENT_STARTED = "containment_started"
    CONTAINMENT_COMPLETED = "containment_completed"
    REMEDIATION_STARTED = "remediation_started"
    REMEDIATION_COMPLETED = "remediation_completed"
    ROOT_CAUSE_IDENTIFIED = "root_cause_identified"
    LESSONS_LEARNED = "lessons_learned"
    RESOLVED = "resolved"
    CLOSED = "closed"


class BreachEventType(str, Enum):
    """Types of breach workflow events."""
    BREACH_REPORTED = "breach_reported"
    BREACH_CONFIRMED = "breach_confirmed"
    ASSESSMENT_UPDATED = "assessment_updated"
    DECISION_CHANGED = "decision_changed"
    NOTIFICATION_SENT = "notification_sent"
    NOTIFICATION_ACKNOWLEDGED = "notification_acknowledged"
    STATUS_CHANGED = "status_changed"
    TIMELINE_EVENT_ADDED = "timeline_event_added"
    TABLETOP_STARTED = "tabletop_started"
    TABLETOP_COMPLETED = "tabletop_completed"


# GDPR notification deadlines
AUTHORITY_NOTIFICATION_HOURS = 72
TABLETOP_DRAFT_PACKAGE_HOURS = 24
QUARTERLY_TABLETOP_DAYS = 90

# Risk score thresholds
LOW_RISK_THRESHOLD = 3.0
HIGH_RISK_THRESHOLD = 6.0


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DPOContact:
    """Data Protection Officer contact information."""
    name: str = ""
    email: str = ""
    phone: str = ""
    organization: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "organization": self.organization,
        }


@dataclass
class SupervisoryAuthority:
    """Supervisory authority information."""
    authority_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""  # e.g., "ICO", "CNIL", "BfDI"
    country: str = ""
    email: str = ""
    portal_url: str = ""
    phone: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "authority_id": self.authority_id,
            "name": self.name,
            "country": self.country,
            "email": self.email,
            "portal_url": self.portal_url,
            "phone": self.phone,
        }


@dataclass
class RiskAssessment:
    """Risk assessment for a breach."""
    assessment_id: str = field(default_factory=lambda: str(uuid4()))
    breach_id: str = ""
    assessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    assessed_by: str = ""

    # Risk factors (0.0 - 1.0 each)
    data_sensitivity_score: float = 0.0
    data_volume_score: float = 0.0
    identifiability_score: float = 0.0
    special_categories_present: bool = False
    vulnerable_individuals_affected: bool = False
    potential_harm_score: float = 0.0
    cross_border: bool = False
    malicious_intent: bool = False

    # Calculated
    overall_risk_score: float = 0.0
    severity: BreachSeverity = BreachSeverity.LOW

    # Narrative
    risk_factors_narrative: str = ""
    mitigating_factors: List[str] = field(default_factory=list)
    aggravating_factors: List[str] = field(default_factory=list)

    def calculate_risk_score(self) -> float:
        """Calculate overall risk score (0.0 - 10.0)."""
        base_score = (
            self.data_sensitivity_score * 2.5 +
            self.data_volume_score * 1.5 +
            self.identifiability_score * 2.0 +
            self.potential_harm_score * 2.5 +
            (1.0 if self.special_categories_present else 0.0) +
            (0.5 if self.vulnerable_individuals_affected else 0.0)
        )

        # Adjustments
        if self.malicious_intent:
            base_score += 1.0
        if self.cross_border:
            base_score += 0.5

        # Mitigating factors reduce score
        base_score -= len(self.mitigating_factors) * 0.5

        # Aggravating factors increase score
        base_score += len(self.aggravating_factors) * 0.5

        self.overall_risk_score = max(0.0, min(10.0, base_score))

        # Determine severity
        if self.overall_risk_score < LOW_RISK_THRESHOLD:
            self.severity = BreachSeverity.LOW
        elif self.overall_risk_score < HIGH_RISK_THRESHOLD:
            self.severity = BreachSeverity.MEDIUM if self.overall_risk_score < 5.0 else BreachSeverity.HIGH
        else:
            self.severity = BreachSeverity.CRITICAL

        return self.overall_risk_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "breach_id": self.breach_id,
            "assessed_at": self.assessed_at.isoformat(),
            "assessed_by": self.assessed_by,
            "data_sensitivity_score": self.data_sensitivity_score,
            "data_volume_score": self.data_volume_score,
            "identifiability_score": self.identifiability_score,
            "special_categories_present": self.special_categories_present,
            "vulnerable_individuals_affected": self.vulnerable_individuals_affected,
            "potential_harm_score": self.potential_harm_score,
            "cross_border": self.cross_border,
            "malicious_intent": self.malicious_intent,
            "overall_risk_score": self.overall_risk_score,
            "severity": self.severity.value,
            "risk_factors_narrative": self.risk_factors_narrative,
            "mitigating_factors": self.mitigating_factors,
            "aggravating_factors": self.aggravating_factors,
        }


@dataclass
class NotificationDecision:
    """Decision on notification requirements."""
    decision_id: str = field(default_factory=lambda: str(uuid4()))
    breach_id: str = ""

    # Authority notification (Art. 33)
    authority_notification_required: bool = False
    authority_notification_deadline: Optional[datetime] = None
    authority_notification_reason: str = ""

    # Subject notification (Art. 34)
    subject_notification_required: bool = False
    subject_notification_reason: str = ""
    exemption_applied: Optional[ExemptionType] = None
    exemption_justification: str = ""

    # Decision metadata
    decision_rationale: str = ""
    decided_by: str = ""
    decided_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    evidence_hash: str = ""

    def __post_init__(self) -> None:
        if not self.evidence_hash:
            self.evidence_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = json.dumps({
            "decision_id": self.decision_id,
            "breach_id": self.breach_id,
            "authority_notification_required": self.authority_notification_required,
            "subject_notification_required": self.subject_notification_required,
            "decided_at": self.decided_at.isoformat(),
            "decided_by": self.decided_by,
        }, sort_keys=True)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "breach_id": self.breach_id,
            "authority_notification_required": self.authority_notification_required,
            "authority_notification_deadline": self.authority_notification_deadline.isoformat() if self.authority_notification_deadline else None,
            "authority_notification_reason": self.authority_notification_reason,
            "subject_notification_required": self.subject_notification_required,
            "subject_notification_reason": self.subject_notification_reason,
            "exemption_applied": self.exemption_applied.value if self.exemption_applied else None,
            "exemption_justification": self.exemption_justification,
            "decision_rationale": self.decision_rationale,
            "decided_by": self.decided_by,
            "decided_at": self.decided_at.isoformat(),
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "evidence_hash": self.evidence_hash,
        }


@dataclass
class AuthorityNotification:
    """Notification to supervisory authority per GDPR Art. 33."""
    notification_id: str = field(default_factory=lambda: str(uuid4()))
    breach_id: str = ""
    authority: Optional[SupervisoryAuthority] = None

    # Art. 33(3) required content
    nature_of_breach: str = ""
    categories_of_data: List[str] = field(default_factory=list)
    categories_of_subjects: List[str] = field(default_factory=list)
    approximate_number_of_subjects: str = ""
    approximate_number_of_records: str = ""
    dpo_contact: Optional[DPOContact] = None
    likely_consequences: List[str] = field(default_factory=list)
    measures_taken: List[str] = field(default_factory=list)
    measures_proposed: List[str] = field(default_factory=list)

    # Metadata
    status: NotificationStatus = NotificationStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""
    submitted_at: Optional[datetime] = None
    submission_reference: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "notification_id": self.notification_id,
            "breach_id": self.breach_id,
            "authority": self.authority.to_dict() if self.authority else None,
            "nature_of_breach": self.nature_of_breach,
            "categories_of_data": self.categories_of_data,
            "categories_of_subjects": self.categories_of_subjects,
            "approximate_number_of_subjects": self.approximate_number_of_subjects,
            "approximate_number_of_records": self.approximate_number_of_records,
            "dpo_contact": self.dpo_contact.to_dict() if self.dpo_contact else None,
            "likely_consequences": self.likely_consequences,
            "measures_taken": self.measures_taken,
            "measures_proposed": self.measures_proposed,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "submission_reference": self.submission_reference,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
        }


@dataclass
class SubjectNotification:
    """Notification to data subjects per GDPR Art. 34."""
    notification_id: str = field(default_factory=lambda: str(uuid4()))
    breach_id: str = ""

    # Art. 34(2) required content
    plain_language_description: str = ""
    dpo_contact: Optional[DPOContact] = None
    likely_consequences: List[str] = field(default_factory=list)
    measures_taken: List[str] = field(default_factory=list)
    recommendations_for_subject: List[str] = field(default_factory=list)

    # Delivery
    notification_method: str = "email"  # email, postal, public
    affected_subjects_count: int = 0
    subjects_notified_count: int = 0

    # Metadata
    status: NotificationStatus = NotificationStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""
    sent_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "notification_id": self.notification_id,
            "breach_id": self.breach_id,
            "plain_language_description": self.plain_language_description,
            "dpo_contact": self.dpo_contact.to_dict() if self.dpo_contact else None,
            "likely_consequences": self.likely_consequences,
            "measures_taken": self.measures_taken,
            "recommendations_for_subject": self.recommendations_for_subject,
            "notification_method": self.notification_method,
            "affected_subjects_count": self.affected_subjects_count,
            "subjects_notified_count": self.subjects_notified_count,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
        }


@dataclass
class TimelineEvent:
    """An event in the breach timeline."""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    breach_id: str = ""
    event_type: TimelineEventType = TimelineEventType.DETECTED
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""
    actor_id: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "breach_id": self.breach_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "actor_id": self.actor_id,
            "evidence": self.evidence,
        }


@dataclass
class DataBreach:
    """A personal data breach incident."""
    breach_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    title: str = ""
    description: str = ""
    category: BreachCategory = BreachCategory.CONFIDENTIALITY
    status: BreachStatus = BreachStatus.DETECTED

    # Discovery
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    detected_by: str = ""
    awareness_at: Optional[datetime] = None  # When org became "aware" - starts 72h clock
    reported_by: str = ""

    # Scope
    data_categories_affected: List[str] = field(default_factory=list)
    individuals_affected_count: Optional[int] = None
    individuals_affected_estimate: str = ""
    records_affected_count: Optional[int] = None
    systems_affected: List[str] = field(default_factory=list)

    # Assessment
    assessment: Optional[RiskAssessment] = None
    decision: Optional[NotificationDecision] = None

    # Notifications
    authority_notification: Optional[AuthorityNotification] = None
    subject_notification: Optional[SubjectNotification] = None

    # Timeline
    timeline: List[TimelineEvent] = field(default_factory=list)

    # Resolution
    root_cause: str = ""
    containment_measures: List[str] = field(default_factory=list)
    remediation_measures: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

    # Metadata
    evidence_hash: str = ""

    def __post_init__(self) -> None:
        if not self.awareness_at:
            self.awareness_at = self.detected_at
        if not self.evidence_hash:
            self.evidence_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = json.dumps({
            "breach_id": self.breach_id,
            "workspace_id": self.workspace_id,
            "title": self.title,
            "category": self.category.value,
            "detected_at": self.detected_at.isoformat(),
        }, sort_keys=True)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def get_authority_deadline(self) -> datetime:
        """Get the 72-hour authority notification deadline."""
        awareness = self.awareness_at or self.detected_at
        return awareness + timedelta(hours=AUTHORITY_NOTIFICATION_HOURS)

    def is_authority_deadline_passed(self) -> bool:
        """Check if authority notification deadline has passed."""
        return datetime.now(timezone.utc) > self.get_authority_deadline()

    def get_hours_until_deadline(self) -> float:
        """Get hours remaining until authority notification deadline."""
        delta = self.get_authority_deadline() - datetime.now(timezone.utc)
        return max(0.0, delta.total_seconds() / 3600)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "breach_id": self.breach_id,
            "workspace_id": self.workspace_id,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "status": self.status.value,
            "detected_at": self.detected_at.isoformat(),
            "detected_by": self.detected_by,
            "awareness_at": self.awareness_at.isoformat() if self.awareness_at else None,
            "reported_by": self.reported_by,
            "data_categories_affected": self.data_categories_affected,
            "individuals_affected_count": self.individuals_affected_count,
            "individuals_affected_estimate": self.individuals_affected_estimate,
            "records_affected_count": self.records_affected_count,
            "systems_affected": self.systems_affected,
            "assessment": self.assessment.to_dict() if self.assessment else None,
            "decision": self.decision.to_dict() if self.decision else None,
            "authority_notification": self.authority_notification.to_dict() if self.authority_notification else None,
            "subject_notification": self.subject_notification.to_dict() if self.subject_notification else None,
            "timeline": [e.to_dict() for e in self.timeline],
            "root_cause": self.root_cause,
            "containment_measures": self.containment_measures,
            "remediation_measures": self.remediation_measures,
            "lessons_learned": self.lessons_learned,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "authority_deadline": self.get_authority_deadline().isoformat(),
            "hours_until_deadline": self.get_hours_until_deadline(),
            "evidence_hash": self.evidence_hash,
        }


@dataclass
class TabletopScenario:
    """A scenario for tabletop exercise."""
    scenario_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    breach_category: BreachCategory = BreachCategory.CONFIDENTIALITY
    severity_simulated: BreachSeverity = BreachSeverity.HIGH
    data_types_involved: List[str] = field(default_factory=list)
    expected_affected_count: str = ""
    injects: List[Dict[str, Any]] = field(default_factory=list)  # Time-sequenced events

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "breach_category": self.breach_category.value,
            "severity_simulated": self.severity_simulated.value,
            "data_types_involved": self.data_types_involved,
            "expected_affected_count": self.expected_affected_count,
            "injects": self.injects,
        }


@dataclass
class TabletopExercise:
    """A tabletop exercise for breach response."""
    exercise_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    scenario: Optional[TabletopScenario] = None
    conducted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    conducted_by: str = ""
    participants: List[str] = field(default_factory=list)
    duration_minutes: int = 0

    # Evaluation
    draft_package_produced: bool = False
    draft_package_time_minutes: int = 0  # Time to produce draft package
    timeline_achieved: Dict[str, int] = field(default_factory=dict)  # Event -> minutes
    decisions_made: List[str] = field(default_factory=list)

    # Findings
    gaps_identified: List[str] = field(default_factory=list)
    improvements_recommended: List[str] = field(default_factory=list)
    action_items: List[Dict[str, str]] = field(default_factory=list)

    # Metadata
    status: str = "scheduled"  # scheduled, in_progress, completed
    completed_at: Optional[datetime] = None
    next_exercise_due: Optional[datetime] = None
    evidence_hash: str = ""

    def __post_init__(self) -> None:
        if not self.next_exercise_due:
            self.next_exercise_due = self.conducted_at + timedelta(days=QUARTERLY_TABLETOP_DAYS)
        if not self.evidence_hash:
            self.evidence_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = json.dumps({
            "exercise_id": self.exercise_id,
            "workspace_id": self.workspace_id,
            "scenario_name": self.scenario.name if self.scenario else "",
            "conducted_at": self.conducted_at.isoformat(),
            "participants": self.participants,
        }, sort_keys=True)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exercise_id": self.exercise_id,
            "workspace_id": self.workspace_id,
            "scenario": self.scenario.to_dict() if self.scenario else None,
            "conducted_at": self.conducted_at.isoformat(),
            "conducted_by": self.conducted_by,
            "participants": self.participants,
            "duration_minutes": self.duration_minutes,
            "draft_package_produced": self.draft_package_produced,
            "draft_package_time_minutes": self.draft_package_time_minutes,
            "timeline_achieved": self.timeline_achieved,
            "decisions_made": self.decisions_made,
            "gaps_identified": self.gaps_identified,
            "improvements_recommended": self.improvements_recommended,
            "action_items": self.action_items,
            "status": self.status,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "next_exercise_due": self.next_exercise_due.isoformat() if self.next_exercise_due else None,
            "evidence_hash": self.evidence_hash,
        }


@dataclass
class BreachWorkflowEvent:
    """Event in the breach workflow audit log."""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    breach_id: Optional[str] = None
    event_type: BreachEventType = BreachEventType.BREACH_REPORTED
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    actor_id: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    result: str = "success"
    integrity_hash: str = ""

    def __post_init__(self) -> None:
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = json.dumps({
            "event_id": self.event_id,
            "workspace_id": self.workspace_id,
            "breach_id": self.breach_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "actor_id": self.actor_id,
            "result": self.result,
        }, sort_keys=True)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "workspace_id": self.workspace_id,
            "breach_id": self.breach_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "actor_id": self.actor_id,
            "details": self.details,
            "result": self.result,
            "integrity_hash": self.integrity_hash,
        }


@dataclass
class BreachStats:
    """Statistics for breach management."""
    workspace_id: str = ""
    total_breaches: int = 0
    breaches_open: int = 0
    breaches_resolved: int = 0
    authority_notifications_sent: int = 0
    subject_notifications_sent: int = 0
    average_response_hours: float = 0.0
    average_resolution_days: float = 0.0
    tabletop_exercises_completed: int = 0
    next_tabletop_due: Optional[datetime] = None
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "total_breaches": self.total_breaches,
            "breaches_open": self.breaches_open,
            "breaches_resolved": self.breaches_resolved,
            "authority_notifications_sent": self.authority_notifications_sent,
            "subject_notifications_sent": self.subject_notifications_sent,
            "average_response_hours": self.average_response_hours,
            "average_resolution_days": self.average_resolution_days,
            "tabletop_exercises_completed": self.tabletop_exercises_completed,
            "next_tabletop_due": self.next_tabletop_due.isoformat() if self.next_tabletop_due else None,
            "evaluated_at": self.evaluated_at.isoformat(),
        }


# =============================================================================
# Breach Workflow Service
# =============================================================================

class BreachWorkflowService:
    """
    Breach Workflow Service.

    Manages:
    - Breach reporting and tracking
    - Risk assessment
    - Notification decision tree
    - Authority and subject notifications
    - Tabletop exercises

    Reference: GDPR Articles 33-34
    """

    def __init__(
        self,
        dpo_contact: Optional[DPOContact] = None,
        default_authority: Optional[SupervisoryAuthority] = None,
        on_event: Optional[Callable[[BreachWorkflowEvent], None]] = None,
        on_breach_reported: Optional[Callable[[DataBreach], None]] = None,
        on_deadline_approaching: Optional[Callable[[DataBreach, float], None]] = None,
    ):
        """
        Initialize Breach Workflow Service.

        Args:
            dpo_contact: Default DPO contact information
            default_authority: Default supervisory authority
            on_event: Callback for workflow events
            on_breach_reported: Callback when breach is reported
            on_deadline_approaching: Callback when deadline is approaching
        """
        self._lock = threading.RLock()  # Re-entrant lock for nested calls

        # Configuration
        self._dpo_contact = dpo_contact
        self._default_authority = default_authority

        # Callbacks
        self._on_event = on_event
        self._on_breach_reported = on_breach_reported
        self._on_deadline_approaching = on_deadline_approaching

        # Storage
        self._breaches: Dict[str, DataBreach] = {}  # breach_id -> breach
        self._authorities: Dict[str, SupervisoryAuthority] = {}  # authority_id -> authority
        self._scenarios: Dict[str, TabletopScenario] = {}  # scenario_id -> scenario
        self._exercises: Dict[str, TabletopExercise] = {}  # exercise_id -> exercise
        self._events: List[BreachWorkflowEvent] = []

        # Initialize default scenarios
        self._initialize_default_scenarios()

    def _initialize_default_scenarios(self) -> None:
        """Initialize default tabletop scenarios."""
        scenarios = [
            TabletopScenario(
                name="Ransomware Attack",
                description="Ransomware encrypts database containing customer PII",
                breach_category=BreachCategory.AVAILABILITY,
                severity_simulated=BreachSeverity.CRITICAL,
                data_types_involved=["customer_names", "email_addresses", "payment_info"],
                expected_affected_count="10,000-50,000",
            ),
            TabletopScenario(
                name="Unauthorized Access",
                description="Employee accesses customer data without authorization",
                breach_category=BreachCategory.CONFIDENTIALITY,
                severity_simulated=BreachSeverity.MEDIUM,
                data_types_involved=["customer_names", "account_ids", "trading_history"],
                expected_affected_count="100-500",
            ),
            TabletopScenario(
                name="Data Exfiltration",
                description="External attacker exfiltrates telemetry data via API vulnerability",
                breach_category=BreachCategory.CONFIDENTIALITY,
                severity_simulated=BreachSeverity.HIGH,
                data_types_involved=["telemetry_events", "workspace_ids", "user_ids"],
                expected_affected_count="1,000-5,000",
            ),
            TabletopScenario(
                name="Accidental Disclosure",
                description="Support export accidentally sent to wrong customer",
                breach_category=BreachCategory.CONFIDENTIALITY,
                severity_simulated=BreachSeverity.LOW,
                data_types_involved=["support_logs", "user_ids", "config_data"],
                expected_affected_count="1-10",
            ),
        ]

        for scenario in scenarios:
            self._scenarios[scenario.scenario_id] = scenario

    # =========================================================================
    # Breach Management
    # =========================================================================

    def report_breach(
        self,
        workspace_id: str,
        title: str,
        description: str,
        category: BreachCategory,
        reported_by: str,
        detected_at: Optional[datetime] = None,
        data_categories_affected: Optional[List[str]] = None,
        systems_affected: Optional[List[str]] = None,
        individuals_affected_estimate: str = "",
    ) -> DataBreach:
        """Report a new data breach."""
        breach = DataBreach(
            workspace_id=workspace_id,
            title=title,
            description=description,
            category=category,
            detected_at=detected_at or datetime.now(timezone.utc),
            detected_by=reported_by,
            reported_by=reported_by,
            data_categories_affected=data_categories_affected or [],
            systems_affected=systems_affected or [],
            individuals_affected_estimate=individuals_affected_estimate,
        )

        # Add initial timeline event
        breach.timeline.append(TimelineEvent(
            breach_id=breach.breach_id,
            event_type=TimelineEventType.DETECTED,
            timestamp=breach.detected_at,
            description="Breach initially detected",
            actor_id=reported_by,
        ))

        breach.timeline.append(TimelineEvent(
            breach_id=breach.breach_id,
            event_type=TimelineEventType.AWARENESS,
            timestamp=datetime.now(timezone.utc),
            description="Organization became aware of breach (72h clock started)",
            actor_id=reported_by,
        ))

        with self._lock:
            self._breaches[breach.breach_id] = breach

        # Log event
        event = BreachWorkflowEvent(
            workspace_id=workspace_id,
            breach_id=breach.breach_id,
            event_type=BreachEventType.BREACH_REPORTED,
            actor_id=reported_by,
            details={
                "title": title,
                "category": category.value,
                "deadline": breach.get_authority_deadline().isoformat(),
            },
        )
        self._log_event(event)

        if self._on_breach_reported:
            try:
                self._on_breach_reported(breach)
            except Exception:
                pass

        return breach

    def confirm_breach(
        self,
        breach_id: str,
        confirmed_by: str,
        additional_details: str = "",
    ) -> DataBreach:
        """Confirm a reported breach."""
        with self._lock:
            breach = self._breaches.get(breach_id)
            if not breach:
                raise ValueError(f"Breach {breach_id} not found")

            breach.status = BreachStatus.CONFIRMED
            breach.timeline.append(TimelineEvent(
                breach_id=breach_id,
                event_type=TimelineEventType.REPORTED_INTERNALLY,
                description=f"Breach confirmed. {additional_details}",
                actor_id=confirmed_by,
            ))

        # Log event
        event = BreachWorkflowEvent(
            workspace_id=breach.workspace_id,
            breach_id=breach_id,
            event_type=BreachEventType.BREACH_CONFIRMED,
            actor_id=confirmed_by,
        )
        self._log_event(event)

        return breach

    def assess_breach(
        self,
        breach_id: str,
        assessment: RiskAssessment,
        assessed_by: str,
    ) -> DataBreach:
        """Complete risk assessment for a breach."""
        assessment.breach_id = breach_id
        assessment.assessed_by = assessed_by
        assessment.assessed_at = datetime.now(timezone.utc)
        assessment.calculate_risk_score()

        with self._lock:
            breach = self._breaches.get(breach_id)
            if not breach:
                raise ValueError(f"Breach {breach_id} not found")

            breach.assessment = assessment
            breach.status = BreachStatus.ASSESSED

            breach.timeline.append(TimelineEvent(
                breach_id=breach_id,
                event_type=TimelineEventType.ASSESSMENT_COMPLETED,
                description=f"Risk assessment completed. Score: {assessment.overall_risk_score:.1f}, Severity: {assessment.severity.value}",
                actor_id=assessed_by,
                evidence={"risk_score": assessment.overall_risk_score, "severity": assessment.severity.value},
            ))

        # Log event
        event = BreachWorkflowEvent(
            workspace_id=breach.workspace_id,
            breach_id=breach_id,
            event_type=BreachEventType.ASSESSMENT_UPDATED,
            actor_id=assessed_by,
            details={
                "risk_score": assessment.overall_risk_score,
                "severity": assessment.severity.value,
            },
        )
        self._log_event(event)

        return breach

    def make_notification_decision(
        self,
        breach_id: str,
        decided_by: str,
        authority_notification_required: Optional[bool] = None,
        subject_notification_required: Optional[bool] = None,
        exemption: Optional[ExemptionType] = None,
        exemption_justification: str = "",
        decision_rationale: str = "",
    ) -> DataBreach:
        """Make notification decision based on assessment."""
        breach = self.get_breach(breach_id)
        if not breach:
            raise ValueError(f"Breach {breach_id} not found")

        if not breach.assessment:
            raise ValueError("Breach must be assessed before making notification decision")

        # Auto-determine if not specified
        if authority_notification_required is None:
            # Art. 33: Notify unless unlikely to result in risk
            authority_notification_required = breach.assessment.overall_risk_score >= LOW_RISK_THRESHOLD

        if subject_notification_required is None:
            # Art. 34: Notify if high risk
            subject_notification_required = breach.assessment.overall_risk_score >= HIGH_RISK_THRESHOLD

        # Check exemption
        if exemption and subject_notification_required:
            subject_notification_required = False

        decision = NotificationDecision(
            breach_id=breach_id,
            authority_notification_required=authority_notification_required,
            authority_notification_deadline=breach.get_authority_deadline() if authority_notification_required else None,
            authority_notification_reason=f"Risk score {breach.assessment.overall_risk_score:.1f} {'≥' if authority_notification_required else '<'} threshold {LOW_RISK_THRESHOLD}",
            subject_notification_required=subject_notification_required,
            subject_notification_reason=f"Risk score {breach.assessment.overall_risk_score:.1f} {'≥' if subject_notification_required else '<'} threshold {HIGH_RISK_THRESHOLD}",
            exemption_applied=exemption,
            exemption_justification=exemption_justification,
            decision_rationale=decision_rationale or f"Based on risk assessment with score {breach.assessment.overall_risk_score:.1f}",
            decided_by=decided_by,
        )

        with self._lock:
            breach.decision = decision
            breach.timeline.append(TimelineEvent(
                breach_id=breach_id,
                event_type=TimelineEventType.DECISION_MADE,
                description=f"Notification decision: Authority={authority_notification_required}, Subjects={subject_notification_required}",
                actor_id=decided_by,
                evidence={
                    "authority_required": authority_notification_required,
                    "subject_required": subject_notification_required,
                    "exemption": exemption.value if exemption else None,
                },
            ))

        # Log event
        event = BreachWorkflowEvent(
            workspace_id=breach.workspace_id,
            breach_id=breach_id,
            event_type=BreachEventType.DECISION_CHANGED,
            actor_id=decided_by,
            details={
                "authority_required": authority_notification_required,
                "subject_required": subject_notification_required,
            },
        )
        self._log_event(event)

        return breach

    def approve_decision(
        self,
        breach_id: str,
        approved_by: str,
    ) -> DataBreach:
        """Approve the notification decision (e.g., by DPO)."""
        with self._lock:
            breach = self._breaches.get(breach_id)
            if not breach:
                raise ValueError(f"Breach {breach_id} not found")

            if not breach.decision:
                raise ValueError("No decision to approve")

            if breach.decision.decided_by == approved_by:
                raise ValueError("Self-approval is not allowed")

            breach.decision.approved_by = approved_by
            breach.decision.approved_at = datetime.now(timezone.utc)

        return breach

    # =========================================================================
    # Notifications
    # =========================================================================

    def create_authority_notification(
        self,
        breach_id: str,
        created_by: str,
        authority: Optional[SupervisoryAuthority] = None,
    ) -> AuthorityNotification:
        """Create authority notification draft."""
        breach = self.get_breach(breach_id)
        if not breach:
            raise ValueError(f"Breach {breach_id} not found")

        notification = AuthorityNotification(
            breach_id=breach_id,
            authority=authority or self._default_authority,
            nature_of_breach=f"{breach.category.value.title()} breach: {breach.title}",
            categories_of_data=breach.data_categories_affected,
            approximate_number_of_subjects=breach.individuals_affected_estimate or str(breach.individuals_affected_count or "Unknown"),
            approximate_number_of_records=str(breach.records_affected_count or "Unknown"),
            dpo_contact=self._dpo_contact,
            likely_consequences=breach.assessment.aggravating_factors if breach.assessment else [],
            measures_taken=breach.containment_measures,
            measures_proposed=breach.remediation_measures,
            status=NotificationStatus.DRAFT,
            created_by=created_by,
        )

        with self._lock:
            breach.authority_notification = notification

        return notification

    def submit_authority_notification(
        self,
        breach_id: str,
        submitted_by: str,
        submission_reference: str = "",
    ) -> DataBreach:
        """Mark authority notification as submitted."""
        with self._lock:
            breach = self._breaches.get(breach_id)
            if not breach:
                raise ValueError(f"Breach {breach_id} not found")

            if not breach.authority_notification:
                raise ValueError("No authority notification to submit")

            breach.authority_notification.status = NotificationStatus.SUBMITTED
            breach.authority_notification.submitted_at = datetime.now(timezone.utc)
            breach.authority_notification.submission_reference = submission_reference

            breach.status = BreachStatus.NOTIFYING

            breach.timeline.append(TimelineEvent(
                breach_id=breach_id,
                event_type=TimelineEventType.AUTHORITY_NOTIFIED,
                description=f"Notification submitted to {breach.authority_notification.authority.name if breach.authority_notification.authority else 'authority'}",
                actor_id=submitted_by,
                evidence={"reference": submission_reference},
            ))

        # Log event
        event = BreachWorkflowEvent(
            workspace_id=breach.workspace_id,
            breach_id=breach_id,
            event_type=BreachEventType.NOTIFICATION_SENT,
            actor_id=submitted_by,
            details={"notification_type": "authority", "reference": submission_reference},
        )
        self._log_event(event)

        return breach

    def create_subject_notification(
        self,
        breach_id: str,
        created_by: str,
        plain_language_description: str,
        recommendations: Optional[List[str]] = None,
        notification_method: str = "email",
    ) -> SubjectNotification:
        """Create subject notification draft."""
        breach = self.get_breach(breach_id)
        if not breach:
            raise ValueError(f"Breach {breach_id} not found")

        notification = SubjectNotification(
            breach_id=breach_id,
            plain_language_description=plain_language_description,
            dpo_contact=self._dpo_contact,
            likely_consequences=breach.assessment.aggravating_factors if breach.assessment else [],
            measures_taken=breach.containment_measures + breach.remediation_measures,
            recommendations_for_subject=recommendations or [
                "Monitor your accounts for suspicious activity",
                "Change your passwords if you haven't recently",
                "Be cautious of phishing emails",
            ],
            notification_method=notification_method,
            affected_subjects_count=breach.individuals_affected_count or 0,
            status=NotificationStatus.DRAFT,
            created_by=created_by,
        )

        with self._lock:
            breach.subject_notification = notification

        return notification

    def send_subject_notification(
        self,
        breach_id: str,
        sent_by: str,
        subjects_notified_count: int,
    ) -> DataBreach:
        """Mark subject notification as sent."""
        with self._lock:
            breach = self._breaches.get(breach_id)
            if not breach:
                raise ValueError(f"Breach {breach_id} not found")

            if not breach.subject_notification:
                raise ValueError("No subject notification to send")

            breach.subject_notification.status = NotificationStatus.COMPLETE
            breach.subject_notification.sent_at = datetime.now(timezone.utc)
            breach.subject_notification.subjects_notified_count = subjects_notified_count

            breach.timeline.append(TimelineEvent(
                breach_id=breach_id,
                event_type=TimelineEventType.SUBJECTS_NOTIFIED,
                description=f"Notified {subjects_notified_count} data subjects via {breach.subject_notification.notification_method}",
                actor_id=sent_by,
                evidence={"count": subjects_notified_count, "method": breach.subject_notification.notification_method},
            ))

        # Log event
        event = BreachWorkflowEvent(
            workspace_id=breach.workspace_id,
            breach_id=breach_id,
            event_type=BreachEventType.NOTIFICATION_SENT,
            actor_id=sent_by,
            details={"notification_type": "subject", "count": subjects_notified_count},
        )
        self._log_event(event)

        return breach

    # =========================================================================
    # Containment and Resolution
    # =========================================================================

    def record_containment(
        self,
        breach_id: str,
        measures: List[str],
        recorded_by: str,
    ) -> DataBreach:
        """Record containment measures."""
        with self._lock:
            breach = self._breaches.get(breach_id)
            if not breach:
                raise ValueError(f"Breach {breach_id} not found")

            breach.containment_measures.extend(measures)
            breach.status = BreachStatus.CONTAINED

            breach.timeline.append(TimelineEvent(
                breach_id=breach_id,
                event_type=TimelineEventType.CONTAINMENT_COMPLETED,
                description=f"Containment measures implemented: {', '.join(measures)}",
                actor_id=recorded_by,
            ))

        return breach

    def record_remediation(
        self,
        breach_id: str,
        measures: List[str],
        root_cause: str,
        recorded_by: str,
    ) -> DataBreach:
        """Record remediation measures and root cause."""
        with self._lock:
            breach = self._breaches.get(breach_id)
            if not breach:
                raise ValueError(f"Breach {breach_id} not found")

            breach.remediation_measures.extend(measures)
            breach.root_cause = root_cause
            breach.status = BreachStatus.REMEDIATED

            breach.timeline.append(TimelineEvent(
                breach_id=breach_id,
                event_type=TimelineEventType.ROOT_CAUSE_IDENTIFIED,
                description=f"Root cause: {root_cause}",
                actor_id=recorded_by,
            ))

            breach.timeline.append(TimelineEvent(
                breach_id=breach_id,
                event_type=TimelineEventType.REMEDIATION_COMPLETED,
                description=f"Remediation measures: {', '.join(measures)}",
                actor_id=recorded_by,
            ))

        return breach

    def resolve_breach(
        self,
        breach_id: str,
        lessons_learned: List[str],
        resolved_by: str,
    ) -> DataBreach:
        """Mark breach as resolved."""
        with self._lock:
            breach = self._breaches.get(breach_id)
            if not breach:
                raise ValueError(f"Breach {breach_id} not found")

            breach.lessons_learned = lessons_learned
            breach.status = BreachStatus.RESOLVED
            breach.resolved_at = datetime.now(timezone.utc)

            breach.timeline.append(TimelineEvent(
                breach_id=breach_id,
                event_type=TimelineEventType.LESSONS_LEARNED,
                description=f"Lessons learned documented: {len(lessons_learned)} items",
                actor_id=resolved_by,
            ))

            breach.timeline.append(TimelineEvent(
                breach_id=breach_id,
                event_type=TimelineEventType.RESOLVED,
                description="Breach marked as resolved",
                actor_id=resolved_by,
            ))

        return breach

    def close_breach(
        self,
        breach_id: str,
        closed_by: str,
    ) -> DataBreach:
        """Close a resolved breach."""
        with self._lock:
            breach = self._breaches.get(breach_id)
            if not breach:
                raise ValueError(f"Breach {breach_id} not found")

            if breach.status != BreachStatus.RESOLVED:
                raise ValueError("Breach must be resolved before closing")

            breach.status = BreachStatus.CLOSED
            breach.closed_at = datetime.now(timezone.utc)

            breach.timeline.append(TimelineEvent(
                breach_id=breach_id,
                event_type=TimelineEventType.CLOSED,
                description="Breach incident closed",
                actor_id=closed_by,
            ))

        return breach

    def get_breach(self, breach_id: str) -> Optional[DataBreach]:
        """Get breach by ID."""
        with self._lock:
            return self._breaches.get(breach_id)

    def list_breaches(
        self,
        workspace_id: Optional[str] = None,
        status: Optional[BreachStatus] = None,
        severity: Optional[BreachSeverity] = None,
        open_only: bool = False,
        limit: int = 100,
    ) -> List[DataBreach]:
        """List breaches with filtering."""
        with self._lock:
            breaches = list(self._breaches.values())

        if workspace_id:
            breaches = [b for b in breaches if b.workspace_id == workspace_id]
        if status:
            breaches = [b for b in breaches if b.status == status]
        if severity:
            breaches = [b for b in breaches if b.assessment and b.assessment.severity == severity]
        if open_only:
            closed_statuses = {BreachStatus.RESOLVED, BreachStatus.CLOSED}
            breaches = [b for b in breaches if b.status not in closed_statuses]

        breaches.sort(key=lambda b: b.detected_at, reverse=True)
        return breaches[:limit]

    def get_breaches_approaching_deadline(
        self,
        hours_threshold: float = 24.0,
    ) -> List[DataBreach]:
        """Get breaches with authority notification deadline approaching."""
        breaches = self.list_breaches(open_only=True)
        approaching = []

        for breach in breaches:
            if breach.decision and breach.decision.authority_notification_required:
                if not breach.authority_notification or breach.authority_notification.status != NotificationStatus.SUBMITTED:
                    hours_remaining = breach.get_hours_until_deadline()
                    if hours_remaining <= hours_threshold:
                        approaching.append(breach)

                        if self._on_deadline_approaching:
                            try:
                                self._on_deadline_approaching(breach, hours_remaining)
                            except Exception:
                                pass

        return approaching

    # =========================================================================
    # Tabletop Exercises
    # =========================================================================

    def get_scenario(self, scenario_id: str) -> Optional[TabletopScenario]:
        """Get a tabletop scenario."""
        with self._lock:
            return self._scenarios.get(scenario_id)

    def list_scenarios(self) -> List[TabletopScenario]:
        """List available tabletop scenarios."""
        with self._lock:
            return list(self._scenarios.values())

    def create_exercise(
        self,
        workspace_id: str,
        scenario_id: str,
        conducted_by: str,
        participants: List[str],
        scheduled_at: Optional[datetime] = None,
    ) -> TabletopExercise:
        """Create a tabletop exercise."""
        scenario = self.get_scenario(scenario_id)
        if not scenario:
            raise ValueError(f"Scenario {scenario_id} not found")

        exercise = TabletopExercise(
            workspace_id=workspace_id,
            scenario=scenario,
            conducted_at=scheduled_at or datetime.now(timezone.utc),
            conducted_by=conducted_by,
            participants=participants,
            status="scheduled" if scheduled_at and scheduled_at > datetime.now(timezone.utc) else "in_progress",
        )

        with self._lock:
            self._exercises[exercise.exercise_id] = exercise

        # Log event
        event = BreachWorkflowEvent(
            workspace_id=workspace_id,
            event_type=BreachEventType.TABLETOP_STARTED,
            actor_id=conducted_by,
            details={
                "exercise_id": exercise.exercise_id,
                "scenario": scenario.name,
                "participants_count": len(participants),
            },
        )
        self._log_event(event)

        return exercise

    def complete_exercise(
        self,
        exercise_id: str,
        completed_by: str,
        duration_minutes: int,
        draft_package_produced: bool,
        draft_package_time_minutes: int,
        timeline_achieved: Dict[str, int],
        decisions_made: List[str],
        gaps_identified: List[str],
        improvements_recommended: List[str],
        action_items: Optional[List[Dict[str, str]]] = None,
    ) -> TabletopExercise:
        """Complete a tabletop exercise with findings."""
        with self._lock:
            exercise = self._exercises.get(exercise_id)
            if not exercise:
                raise ValueError(f"Exercise {exercise_id} not found")

            exercise.status = "completed"
            exercise.completed_at = datetime.now(timezone.utc)
            exercise.duration_minutes = duration_minutes
            exercise.draft_package_produced = draft_package_produced
            exercise.draft_package_time_minutes = draft_package_time_minutes
            exercise.timeline_achieved = timeline_achieved
            exercise.decisions_made = decisions_made
            exercise.gaps_identified = gaps_identified
            exercise.improvements_recommended = improvements_recommended
            exercise.action_items = action_items or []

            # Set next exercise due (quarterly)
            exercise.next_exercise_due = datetime.now(timezone.utc) + timedelta(days=QUARTERLY_TABLETOP_DAYS)

            # Update evidence hash
            exercise.evidence_hash = exercise._compute_hash()

        # Log event
        event = BreachWorkflowEvent(
            workspace_id=exercise.workspace_id,
            event_type=BreachEventType.TABLETOP_COMPLETED,
            actor_id=completed_by,
            details={
                "exercise_id": exercise_id,
                "duration_minutes": duration_minutes,
                "draft_package_produced": draft_package_produced,
                "gaps_count": len(gaps_identified),
            },
        )
        self._log_event(event)

        return exercise

    def get_exercise(self, exercise_id: str) -> Optional[TabletopExercise]:
        """Get exercise by ID."""
        with self._lock:
            return self._exercises.get(exercise_id)

    def list_exercises(
        self,
        workspace_id: Optional[str] = None,
        completed_only: bool = False,
        limit: int = 100,
    ) -> List[TabletopExercise]:
        """List tabletop exercises."""
        with self._lock:
            exercises = list(self._exercises.values())

        if workspace_id:
            exercises = [e for e in exercises if e.workspace_id == workspace_id]
        if completed_only:
            exercises = [e for e in exercises if e.status == "completed"]

        exercises.sort(key=lambda e: e.conducted_at, reverse=True)
        return exercises[:limit]

    def is_tabletop_due(self, workspace_id: str) -> bool:
        """Check if a tabletop exercise is due."""
        exercises = self.list_exercises(workspace_id=workspace_id, completed_only=True, limit=1)
        if not exercises:
            return True  # Never done, so due

        last_exercise = exercises[0]
        if last_exercise.next_exercise_due:
            return datetime.now(timezone.utc) >= last_exercise.next_exercise_due

        return True

    # =========================================================================
    # Statistics and Export
    # =========================================================================

    def get_stats(self, workspace_id: str) -> BreachStats:
        """Get breach management statistics."""
        breaches = self.list_breaches(workspace_id=workspace_id, limit=10000)
        exercises = self.list_exercises(workspace_id=workspace_id, completed_only=True, limit=10000)

        closed_statuses = {BreachStatus.RESOLVED, BreachStatus.CLOSED}
        open_breaches = [b for b in breaches if b.status not in closed_statuses]
        resolved_breaches = [b for b in breaches if b.status in closed_statuses]

        authority_sent = len([b for b in breaches
                            if b.authority_notification and b.authority_notification.status == NotificationStatus.SUBMITTED])
        subject_sent = len([b for b in breaches
                           if b.subject_notification and b.subject_notification.status == NotificationStatus.COMPLETE])

        # Calculate averages
        response_times = []
        for breach in breaches:
            if breach.authority_notification and breach.authority_notification.submitted_at and breach.awareness_at:
                delta = (breach.authority_notification.submitted_at - breach.awareness_at).total_seconds() / 3600
                response_times.append(delta)

        resolution_times = []
        for breach in resolved_breaches:
            if breach.resolved_at and breach.detected_at:
                delta = (breach.resolved_at - breach.detected_at).total_seconds() / (3600 * 24)
                resolution_times.append(delta)

        avg_response = sum(response_times) / len(response_times) if response_times else 0.0
        avg_resolution = sum(resolution_times) / len(resolution_times) if resolution_times else 0.0

        # Next tabletop
        recent_exercises = self.list_exercises(workspace_id=workspace_id, completed_only=True, limit=1)
        next_tabletop = None
        if recent_exercises:
            next_tabletop = recent_exercises[0].next_exercise_due

        return BreachStats(
            workspace_id=workspace_id,
            total_breaches=len(breaches),
            breaches_open=len(open_breaches),
            breaches_resolved=len(resolved_breaches),
            authority_notifications_sent=authority_sent,
            subject_notifications_sent=subject_sent,
            average_response_hours=avg_response,
            average_resolution_days=avg_resolution,
            tabletop_exercises_completed=len(exercises),
            next_tabletop_due=next_tabletop,
        )

    def _log_event(self, event: BreachWorkflowEvent) -> None:
        """Log a breach workflow event."""
        with self._lock:
            self._events.append(event)

        if self._on_event:
            try:
                self._on_event(event)
            except Exception:
                pass

    def get_events(
        self,
        workspace_id: Optional[str] = None,
        breach_id: Optional[str] = None,
        event_type: Optional[BreachEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[BreachWorkflowEvent]:
        """Get breach workflow events."""
        with self._lock:
            events = list(self._events)

        if workspace_id:
            events = [e for e in events if e.workspace_id == workspace_id]
        if breach_id:
            events = [e for e in events if e.breach_id == breach_id]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def export_breach_records(
        self,
        workspace_id: str,
        include_events: bool = True,
        include_exercises: bool = True,
    ) -> Dict[str, Any]:
        """Export breach records for evidence pack."""
        breaches = self.list_breaches(workspace_id=workspace_id, limit=10000)
        stats = self.get_stats(workspace_id)

        export_data: Dict[str, Any] = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "workspace_id": workspace_id,
            "breaches": [b.to_dict() for b in breaches],
            "stats": stats.to_dict(),
        }

        if include_events:
            events = self.get_events(workspace_id=workspace_id, limit=10000)
            export_data["events"] = [e.to_dict() for e in events]

        if include_exercises:
            exercises = self.list_exercises(workspace_id=workspace_id, limit=10000)
            export_data["tabletop_exercises"] = [e.to_dict() for e in exercises]
            export_data["scenarios"] = [s.to_dict() for s in self.list_scenarios()]

        # Compute hash
        content = json.dumps(export_data, sort_keys=True, default=str)
        export_data["integrity_hash"] = f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

        return export_data
