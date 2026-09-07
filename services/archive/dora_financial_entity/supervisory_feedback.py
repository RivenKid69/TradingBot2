"""
DORA Article 22 - Supervisory Feedback on Incident Reports

This module implements the supervisory feedback mechanism as required by
DORA (Regulation (EU) 2022/2554) Article 22, which mandates competent
authorities to provide feedback and guidance on incident reports.

Key Requirements:
- Process feedback from competent authorities
- Acknowledge receipt of supervisory guidance
- Implement corrective actions based on feedback
- Track feedback resolution status
- Maintain audit trail of all feedback interactions

References:
- DORA Article 22: Supervisory feedback
- CDR 2025/301: RTS on incident reporting content and time limits
- CIR 2025/302: ITS on standard forms and templates

Author: DORA Compliance Module
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of supervisory feedback per Article 22."""
    ACKNOWLEDGEMENT = "acknowledgement"
    CLARIFICATION_REQUEST = "clarification_request"
    ADDITIONAL_INFORMATION_REQUEST = "additional_information_request"
    CORRECTIVE_ACTION_REQUIRED = "corrective_action_required"
    GUIDANCE = "guidance"
    WARNING = "warning"
    COMMENDATION = "commendation"
    CROSS_BORDER_COORDINATION = "cross_border_coordination"
    ANONYMISED_AGGREGATION = "anonymised_aggregation"


class FeedbackPriority(Enum):
    """Priority levels for supervisory feedback."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FeedbackStatus(Enum):
    """Status of supervisory feedback processing."""
    RECEIVED = "received"
    ACKNOWLEDGED = "acknowledged"
    UNDER_REVIEW = "under_review"
    ACTION_IN_PROGRESS = "action_in_progress"
    PENDING_CLARIFICATION = "pending_clarification"
    RESOLVED = "resolved"
    CLOSED = "closed"


class CorrectiveActionType(Enum):
    """Types of corrective actions that may be required."""
    INCIDENT_RECLASSIFICATION = "incident_reclassification"
    ADDITIONAL_REPORTING = "additional_reporting"
    PROCESS_IMPROVEMENT = "process_improvement"
    CONTROL_ENHANCEMENT = "control_enhancement"
    THIRD_PARTY_ENGAGEMENT = "third_party_engagement"
    COMMUNICATION_UPDATE = "communication_update"
    DOCUMENTATION_UPDATE = "documentation_update"
    TRAINING_REQUIRED = "training_required"
    TECHNICAL_REMEDIATION = "technical_remediation"


class ResponseType(Enum):
    """Types of responses to supervisory feedback."""
    ACKNOWLEDGEMENT = "acknowledgement"
    CLARIFICATION = "clarification"
    ADDITIONAL_INFORMATION = "additional_information"
    ACTION_PLAN = "action_plan"
    COMPLETION_REPORT = "completion_report"
    EXTENSION_REQUEST = "extension_request"


@dataclass
class CompetentAuthority:
    """Competent authority issuing feedback."""
    authority_id: str
    name: str
    country_code: str
    contact_email: str
    contact_phone: Optional[str] = None
    department: Optional[str] = None
    jurisdiction: Optional[str] = None


@dataclass
class SupervisoryFeedback:
    """Individual supervisory feedback item per Article 22."""
    feedback_id: str
    incident_id: str
    report_id: str
    authority: CompetentAuthority
    feedback_type: FeedbackType
    priority: FeedbackPriority
    subject: str
    content: str
    received_at: datetime
    response_deadline: Optional[datetime] = None
    status: FeedbackStatus = FeedbackStatus.RECEIVED
    required_actions: list[str] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)
    reference_number: Optional[str] = None
    related_feedback_ids: list[str] = field(default_factory=list)
    cross_border_implications: bool = False
    affected_member_states: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.feedback_id:
            self.feedback_id = f"FEEDBACK-{uuid4().hex[:12].upper()}"


@dataclass
class CorrectiveAction:
    """Corrective action required by supervisory feedback."""
    action_id: str
    feedback_id: str
    action_type: CorrectiveActionType
    description: str
    assigned_to: str
    deadline: datetime
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    evidence_required: bool = True
    evidence_provided: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    verification_required: bool = False
    verified_by: Optional[str] = None
    verified_at: Optional[datetime] = None

    def __post_init__(self):
        if not self.action_id:
            self.action_id = f"ACTION-{uuid4().hex[:12].upper()}"

    @property
    def is_overdue(self) -> bool:
        """Check if the action is overdue."""
        if self.status == "completed":
            return False
        return datetime.utcnow() > self.deadline

    @property
    def days_until_deadline(self) -> int:
        """Calculate days until deadline."""
        delta = self.deadline - datetime.utcnow()
        return delta.days


@dataclass
class FeedbackResponse:
    """Response to supervisory feedback."""
    response_id: str
    feedback_id: str
    response_type: ResponseType
    content: str
    submitted_at: datetime
    submitted_by: str
    attachments: list[str] = field(default_factory=list)
    action_references: list[str] = field(default_factory=list)
    acknowledged_by_authority: bool = False
    authority_response: Optional[str] = None

    def __post_init__(self):
        if not self.response_id:
            self.response_id = f"RESPONSE-{uuid4().hex[:12].upper()}"


@dataclass
class FeedbackAuditEntry:
    """Audit trail entry for feedback processing."""
    entry_id: str
    feedback_id: str
    action: str
    actor: str
    timestamp: datetime
    details: dict[str, Any] = field(default_factory=dict)
    previous_status: Optional[FeedbackStatus] = None
    new_status: Optional[FeedbackStatus] = None

    def __post_init__(self):
        if not self.entry_id:
            self.entry_id = f"AUDIT-{uuid4().hex[:12].upper()}"


@dataclass
class AnonymisedInsight:
    """Anonymised and aggregated insight from supervisory feedback per Article 22(2)."""
    insight_id: str
    title: str
    description: str
    threat_category: str
    affected_sector: str
    recommendations: list[str]
    publication_date: datetime
    source_authority: str
    severity_level: str
    geographic_scope: list[str] = field(default_factory=list)
    related_vulnerabilities: list[str] = field(default_factory=list)
    mitigation_guidance: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.insight_id:
            self.insight_id = f"INSIGHT-{uuid4().hex[:12].upper()}"


class DORASupervisioryFeedback:
    """
    DORA Article 22 Supervisory Feedback Handler.

    Manages the receipt, processing, and response to supervisory feedback
    from competent authorities regarding ICT incident reports.

    Article 22 Key Requirements:
    1. Competent authorities shall provide feedback and guidance
    2. Feedback may include anonymised and aggregated information
    3. Financial entities shall acknowledge and act on feedback
    4. Cross-border coordination when multiple authorities involved
    """

    def __init__(
        self,
        entity_id: str,
        entity_name: str,
        primary_authority: Optional[CompetentAuthority] = None,
        response_sla_hours: int = 24,
        auto_acknowledge: bool = True
    ):
        """
        Initialize the supervisory feedback handler.

        Args:
            entity_id: Unique identifier for the financial entity
            entity_name: Name of the financial entity
            primary_authority: Primary competent authority
            response_sla_hours: SLA for responding to feedback (default 24h)
            auto_acknowledge: Automatically acknowledge received feedback
        """
        self.entity_id = entity_id
        self.entity_name = entity_name
        self.primary_authority = primary_authority
        self.response_sla_hours = response_sla_hours
        self.auto_acknowledge = auto_acknowledge

        # Storage
        self._feedback: dict[str, SupervisoryFeedback] = {}
        self._corrective_actions: dict[str, CorrectiveAction] = {}
        self._responses: dict[str, list[FeedbackResponse]] = {}
        self._audit_trail: dict[str, list[FeedbackAuditEntry]] = {}
        self._insights: dict[str, AnonymisedInsight] = {}

        # Metrics
        self._metrics = {
            "total_feedback_received": 0,
            "feedback_by_type": {},
            "feedback_by_priority": {},
            "average_response_time_hours": 0.0,
            "on_time_response_rate": 0.0,
            "corrective_actions_completed": 0,
            "corrective_actions_overdue": 0
        }

        logger.info(
            f"Initialized DORASupervisioryFeedback for entity {entity_id}"
        )

    def receive_feedback(
        self,
        incident_id: str,
        report_id: str,
        authority: CompetentAuthority,
        feedback_type: FeedbackType,
        priority: FeedbackPriority,
        subject: str,
        content: str,
        response_deadline: Optional[datetime] = None,
        required_actions: Optional[list[str]] = None,
        attachments: Optional[list[str]] = None,
        reference_number: Optional[str] = None,
        cross_border_implications: bool = False,
        affected_member_states: Optional[list[str]] = None
    ) -> SupervisoryFeedback:
        """
        Receive and register supervisory feedback.

        Args:
            incident_id: ID of the related incident
            report_id: ID of the incident report
            authority: Competent authority issuing feedback
            feedback_type: Type of feedback
            priority: Priority level
            subject: Subject of feedback
            content: Feedback content
            response_deadline: Deadline for response
            required_actions: List of required actions
            attachments: List of attachment references
            reference_number: Authority's reference number
            cross_border_implications: Whether cross-border coordination needed
            affected_member_states: List of affected EU member states

        Returns:
            SupervisoryFeedback: The registered feedback
        """
        feedback = SupervisoryFeedback(
            feedback_id=f"FEEDBACK-{uuid4().hex[:12].upper()}",
            incident_id=incident_id,
            report_id=report_id,
            authority=authority,
            feedback_type=feedback_type,
            priority=priority,
            subject=subject,
            content=content,
            received_at=datetime.utcnow(),
            response_deadline=response_deadline or self._calculate_default_deadline(priority),
            required_actions=required_actions or [],
            attachments=attachments or [],
            reference_number=reference_number,
            cross_border_implications=cross_border_implications,
            affected_member_states=affected_member_states or []
        )

        self._feedback[feedback.feedback_id] = feedback
        self._responses[feedback.feedback_id] = []
        self._audit_trail[feedback.feedback_id] = []

        # Record audit entry
        self._add_audit_entry(
            feedback.feedback_id,
            "feedback_received",
            "system",
            {"authority": authority.name, "type": feedback_type.value}
        )

        # Update metrics
        self._metrics["total_feedback_received"] += 1
        self._metrics["feedback_by_type"][feedback_type.value] = \
            self._metrics["feedback_by_type"].get(feedback_type.value, 0) + 1
        self._metrics["feedback_by_priority"][priority.value] = \
            self._metrics["feedback_by_priority"].get(priority.value, 0) + 1

        # Auto-acknowledge if enabled
        if self.auto_acknowledge:
            self.acknowledge_feedback(feedback.feedback_id, "system")

        logger.info(
            f"Received supervisory feedback {feedback.feedback_id} "
            f"from {authority.name} for incident {incident_id}"
        )

        return feedback

    def _calculate_default_deadline(self, priority: FeedbackPriority) -> datetime:
        """Calculate default response deadline based on priority."""
        hours_by_priority = {
            FeedbackPriority.CRITICAL: 4,
            FeedbackPriority.HIGH: 24,
            FeedbackPriority.MEDIUM: 72,
            FeedbackPriority.LOW: 168  # 7 days
        }
        hours = hours_by_priority.get(priority, self.response_sla_hours)
        return datetime.utcnow() + timedelta(hours=hours)

    def acknowledge_feedback(
        self,
        feedback_id: str,
        acknowledged_by: str
    ) -> FeedbackResponse:
        """
        Acknowledge receipt of supervisory feedback.

        Args:
            feedback_id: ID of the feedback to acknowledge
            acknowledged_by: Person/system acknowledging

        Returns:
            FeedbackResponse: The acknowledgement response
        """
        feedback = self._feedback.get(feedback_id)
        if not feedback:
            raise ValueError(f"Feedback {feedback_id} not found")

        response = FeedbackResponse(
            response_id=f"RESPONSE-{uuid4().hex[:12].upper()}",
            feedback_id=feedback_id,
            response_type=ResponseType.ACKNOWLEDGEMENT,
            content=f"Feedback acknowledged by {self.entity_name}. "
                    f"We will review and respond within the specified deadline.",
            submitted_at=datetime.utcnow(),
            submitted_by=acknowledged_by
        )

        self._responses[feedback_id].append(response)

        # Update status
        old_status = feedback.status
        feedback.status = FeedbackStatus.ACKNOWLEDGED

        self._add_audit_entry(
            feedback_id,
            "feedback_acknowledged",
            acknowledged_by,
            {},
            old_status,
            FeedbackStatus.ACKNOWLEDGED
        )

        logger.info(f"Acknowledged feedback {feedback_id}")

        return response

    def start_review(
        self,
        feedback_id: str,
        reviewer: str,
        notes: Optional[str] = None
    ) -> SupervisoryFeedback:
        """
        Start reviewing supervisory feedback.

        Args:
            feedback_id: ID of the feedback
            reviewer: Person starting review
            notes: Optional review notes

        Returns:
            SupervisoryFeedback: Updated feedback
        """
        feedback = self._feedback.get(feedback_id)
        if not feedback:
            raise ValueError(f"Feedback {feedback_id} not found")

        old_status = feedback.status
        feedback.status = FeedbackStatus.UNDER_REVIEW

        self._add_audit_entry(
            feedback_id,
            "review_started",
            reviewer,
            {"notes": notes} if notes else {},
            old_status,
            FeedbackStatus.UNDER_REVIEW
        )

        logger.info(f"Started review of feedback {feedback_id} by {reviewer}")

        return feedback

    def create_corrective_action(
        self,
        feedback_id: str,
        action_type: CorrectiveActionType,
        description: str,
        assigned_to: str,
        deadline: datetime,
        evidence_required: bool = True,
        verification_required: bool = False
    ) -> CorrectiveAction:
        """
        Create a corrective action based on supervisory feedback.

        Args:
            feedback_id: ID of the related feedback
            action_type: Type of corrective action
            description: Description of the action
            assigned_to: Person assigned to complete action
            deadline: Deadline for completion
            evidence_required: Whether evidence is required
            verification_required: Whether independent verification needed

        Returns:
            CorrectiveAction: The created action
        """
        feedback = self._feedback.get(feedback_id)
        if not feedback:
            raise ValueError(f"Feedback {feedback_id} not found")

        action = CorrectiveAction(
            action_id=f"ACTION-{uuid4().hex[:12].upper()}",
            feedback_id=feedback_id,
            action_type=action_type,
            description=description,
            assigned_to=assigned_to,
            deadline=deadline,
            evidence_required=evidence_required,
            verification_required=verification_required
        )

        self._corrective_actions[action.action_id] = action

        # Update feedback status
        old_status = feedback.status
        feedback.status = FeedbackStatus.ACTION_IN_PROGRESS

        self._add_audit_entry(
            feedback_id,
            "corrective_action_created",
            assigned_to,
            {
                "action_id": action.action_id,
                "action_type": action_type.value,
                "deadline": deadline.isoformat()
            },
            old_status,
            FeedbackStatus.ACTION_IN_PROGRESS
        )

        logger.info(
            f"Created corrective action {action.action_id} "
            f"for feedback {feedback_id}"
        )

        return action

    def start_corrective_action(
        self,
        action_id: str,
        started_by: str
    ) -> CorrectiveAction:
        """
        Mark a corrective action as started.

        Args:
            action_id: ID of the action
            started_by: Person starting the action

        Returns:
            CorrectiveAction: Updated action
        """
        action = self._corrective_actions.get(action_id)
        if not action:
            raise ValueError(f"Action {action_id} not found")

        action.status = "in_progress"
        action.started_at = datetime.utcnow()

        self._add_audit_entry(
            action.feedback_id,
            "corrective_action_started",
            started_by,
            {"action_id": action_id}
        )

        logger.info(f"Started corrective action {action_id}")

        return action

    def complete_corrective_action(
        self,
        action_id: str,
        completed_by: str,
        evidence: Optional[list[str]] = None,
        notes: Optional[str] = None
    ) -> CorrectiveAction:
        """
        Mark a corrective action as completed.

        Args:
            action_id: ID of the action
            completed_by: Person completing the action
            evidence: List of evidence references
            notes: Completion notes

        Returns:
            CorrectiveAction: Updated action
        """
        action = self._corrective_actions.get(action_id)
        if not action:
            raise ValueError(f"Action {action_id} not found")

        if action.evidence_required and not evidence:
            raise ValueError("Evidence is required for this action")

        action.status = "completed"
        action.completed_at = datetime.utcnow()
        if evidence:
            action.evidence_provided = evidence
        if notes:
            action.notes.append(notes)

        self._metrics["corrective_actions_completed"] += 1

        self._add_audit_entry(
            action.feedback_id,
            "corrective_action_completed",
            completed_by,
            {
                "action_id": action_id,
                "evidence_count": len(evidence) if evidence else 0
            }
        )

        logger.info(f"Completed corrective action {action_id}")

        # Check if all actions for feedback are complete
        self._check_feedback_resolution(action.feedback_id)

        return action

    def verify_corrective_action(
        self,
        action_id: str,
        verified_by: str,
        approved: bool,
        notes: Optional[str] = None
    ) -> CorrectiveAction:
        """
        Verify a completed corrective action.

        Args:
            action_id: ID of the action
            verified_by: Person verifying the action
            approved: Whether verification approved
            notes: Verification notes

        Returns:
            CorrectiveAction: Updated action
        """
        action = self._corrective_actions.get(action_id)
        if not action:
            raise ValueError(f"Action {action_id} not found")

        if action.status != "completed":
            raise ValueError("Action must be completed before verification")

        action.verified_by = verified_by
        action.verified_at = datetime.utcnow()

        if approved:
            action.status = "verified"
        else:
            action.status = "verification_failed"
            if notes:
                action.notes.append(f"Verification failed: {notes}")

        self._add_audit_entry(
            action.feedback_id,
            "corrective_action_verified",
            verified_by,
            {
                "action_id": action_id,
                "approved": approved,
                "notes": notes
            }
        )

        logger.info(
            f"Verified corrective action {action_id}: "
            f"{'approved' if approved else 'failed'}"
        )

        return action

    def _check_feedback_resolution(self, feedback_id: str) -> None:
        """Check if all corrective actions are complete and resolve feedback."""
        feedback = self._feedback.get(feedback_id)
        if not feedback:
            return

        # Get all actions for this feedback
        actions = [
            a for a in self._corrective_actions.values()
            if a.feedback_id == feedback_id
        ]

        if not actions:
            return

        # Check if all actions are completed/verified
        all_complete = all(
            a.status in ["completed", "verified"] for a in actions
        )

        if all_complete:
            old_status = feedback.status
            feedback.status = FeedbackStatus.RESOLVED

            self._add_audit_entry(
                feedback_id,
                "feedback_resolved",
                "system",
                {"actions_completed": len(actions)},
                old_status,
                FeedbackStatus.RESOLVED
            )

            logger.info(f"Feedback {feedback_id} resolved - all actions complete")

    def submit_response(
        self,
        feedback_id: str,
        response_type: ResponseType,
        content: str,
        submitted_by: str,
        attachments: Optional[list[str]] = None,
        action_references: Optional[list[str]] = None
    ) -> FeedbackResponse:
        """
        Submit a response to supervisory feedback.

        Args:
            feedback_id: ID of the feedback
            response_type: Type of response
            content: Response content
            submitted_by: Person submitting response
            attachments: List of attachment references
            action_references: References to corrective actions

        Returns:
            FeedbackResponse: The submitted response
        """
        feedback = self._feedback.get(feedback_id)
        if not feedback:
            raise ValueError(f"Feedback {feedback_id} not found")

        response = FeedbackResponse(
            response_id=f"RESPONSE-{uuid4().hex[:12].upper()}",
            feedback_id=feedback_id,
            response_type=response_type,
            content=content,
            submitted_at=datetime.utcnow(),
            submitted_by=submitted_by,
            attachments=attachments or [],
            action_references=action_references or []
        )

        self._responses[feedback_id].append(response)

        self._add_audit_entry(
            feedback_id,
            "response_submitted",
            submitted_by,
            {
                "response_type": response_type.value,
                "response_id": response.response_id
            }
        )

        # Update response time metrics
        self._update_response_metrics(feedback, response)

        logger.info(
            f"Submitted {response_type.value} response to feedback {feedback_id}"
        )

        return response

    def _update_response_metrics(
        self,
        feedback: SupervisoryFeedback,
        response: FeedbackResponse
    ) -> None:
        """Update response time metrics."""
        response_time = (response.submitted_at - feedback.received_at).total_seconds() / 3600

        # Update average response time
        total = self._metrics["total_feedback_received"]
        current_avg = self._metrics["average_response_time_hours"]
        self._metrics["average_response_time_hours"] = \
            (current_avg * (total - 1) + response_time) / total

        # Update on-time rate
        if feedback.response_deadline and response.submitted_at <= feedback.response_deadline:
            on_time_count = self._metrics.get("on_time_count", 0) + 1
            self._metrics["on_time_count"] = on_time_count
            self._metrics["on_time_response_rate"] = on_time_count / total

    def request_deadline_extension(
        self,
        feedback_id: str,
        requested_by: str,
        new_deadline: datetime,
        justification: str
    ) -> FeedbackResponse:
        """
        Request an extension for feedback response deadline.

        Args:
            feedback_id: ID of the feedback
            requested_by: Person requesting extension
            new_deadline: Requested new deadline
            justification: Justification for extension

        Returns:
            FeedbackResponse: The extension request
        """
        feedback = self._feedback.get(feedback_id)
        if not feedback:
            raise ValueError(f"Feedback {feedback_id} not found")

        content = (
            f"Extension Request\n\n"
            f"Original Deadline: {feedback.response_deadline.isoformat() if feedback.response_deadline else 'N/A'}\n"
            f"Requested Deadline: {new_deadline.isoformat()}\n\n"
            f"Justification:\n{justification}\n\n"
            f"We respectfully request consideration of this extension."
        )

        response = self.submit_response(
            feedback_id=feedback_id,
            response_type=ResponseType.EXTENSION_REQUEST,
            content=content,
            submitted_by=requested_by
        )

        feedback.status = FeedbackStatus.PENDING_CLARIFICATION

        logger.info(f"Requested deadline extension for feedback {feedback_id}")

        return response

    def close_feedback(
        self,
        feedback_id: str,
        closed_by: str,
        resolution_summary: str
    ) -> SupervisoryFeedback:
        """
        Close resolved feedback.

        Args:
            feedback_id: ID of the feedback
            closed_by: Person closing the feedback
            resolution_summary: Summary of resolution

        Returns:
            SupervisoryFeedback: Updated feedback
        """
        feedback = self._feedback.get(feedback_id)
        if not feedback:
            raise ValueError(f"Feedback {feedback_id} not found")

        if feedback.status not in [FeedbackStatus.RESOLVED, FeedbackStatus.ACKNOWLEDGED]:
            # Check for acknowledgement-only feedback types
            if feedback.feedback_type not in [
                FeedbackType.ACKNOWLEDGEMENT,
                FeedbackType.COMMENDATION,
                FeedbackType.ANONYMISED_AGGREGATION
            ]:
                raise ValueError(
                    f"Feedback must be resolved before closing. "
                    f"Current status: {feedback.status.value}"
                )

        old_status = feedback.status
        feedback.status = FeedbackStatus.CLOSED

        self._add_audit_entry(
            feedback_id,
            "feedback_closed",
            closed_by,
            {"resolution_summary": resolution_summary},
            old_status,
            FeedbackStatus.CLOSED
        )

        logger.info(f"Closed feedback {feedback_id}")

        return feedback

    def register_insight(
        self,
        title: str,
        description: str,
        threat_category: str,
        affected_sector: str,
        recommendations: list[str],
        source_authority: str,
        severity_level: str,
        geographic_scope: Optional[list[str]] = None,
        related_vulnerabilities: Optional[list[str]] = None,
        mitigation_guidance: Optional[list[str]] = None
    ) -> AnonymisedInsight:
        """
        Register anonymised and aggregated insight per Article 22(2).

        Args:
            title: Insight title
            description: Insight description
            threat_category: Category of threat
            affected_sector: Affected financial sector
            recommendations: List of recommendations
            source_authority: Authority publishing insight
            severity_level: Severity level
            geographic_scope: Geographic scope
            related_vulnerabilities: Related CVEs or vulnerabilities
            mitigation_guidance: Mitigation guidance

        Returns:
            AnonymisedInsight: The registered insight
        """
        insight = AnonymisedInsight(
            insight_id=f"INSIGHT-{uuid4().hex[:12].upper()}",
            title=title,
            description=description,
            threat_category=threat_category,
            affected_sector=affected_sector,
            recommendations=recommendations,
            publication_date=datetime.utcnow(),
            source_authority=source_authority,
            severity_level=severity_level,
            geographic_scope=geographic_scope or [],
            related_vulnerabilities=related_vulnerabilities or [],
            mitigation_guidance=mitigation_guidance or []
        )

        self._insights[insight.insight_id] = insight

        logger.info(f"Registered anonymised insight {insight.insight_id}")

        return insight

    def _add_audit_entry(
        self,
        feedback_id: str,
        action: str,
        actor: str,
        details: dict[str, Any],
        previous_status: Optional[FeedbackStatus] = None,
        new_status: Optional[FeedbackStatus] = None
    ) -> FeedbackAuditEntry:
        """Add an audit trail entry."""
        entry = FeedbackAuditEntry(
            entry_id=f"AUDIT-{uuid4().hex[:12].upper()}",
            feedback_id=feedback_id,
            action=action,
            actor=actor,
            timestamp=datetime.utcnow(),
            details=details,
            previous_status=previous_status,
            new_status=new_status
        )

        if feedback_id not in self._audit_trail:
            self._audit_trail[feedback_id] = []

        self._audit_trail[feedback_id].append(entry)

        return entry

    def get_feedback(self, feedback_id: str) -> Optional[SupervisoryFeedback]:
        """Get feedback by ID."""
        return self._feedback.get(feedback_id)

    def get_feedback_by_incident(self, incident_id: str) -> list[SupervisoryFeedback]:
        """Get all feedback for an incident."""
        return [
            f for f in self._feedback.values()
            if f.incident_id == incident_id
        ]

    def get_pending_feedback(self) -> list[SupervisoryFeedback]:
        """Get all pending (unresolved) feedback."""
        return [
            f for f in self._feedback.values()
            if f.status not in [FeedbackStatus.RESOLVED, FeedbackStatus.CLOSED]
        ]

    def get_overdue_feedback(self) -> list[SupervisoryFeedback]:
        """Get all feedback with overdue responses."""
        now = datetime.utcnow()
        return [
            f for f in self._feedback.values()
            if f.response_deadline and now > f.response_deadline
            and f.status not in [FeedbackStatus.RESOLVED, FeedbackStatus.CLOSED]
        ]

    def get_corrective_actions(self, feedback_id: str) -> list[CorrectiveAction]:
        """Get all corrective actions for a feedback."""
        return [
            a for a in self._corrective_actions.values()
            if a.feedback_id == feedback_id
        ]

    def get_overdue_actions(self) -> list[CorrectiveAction]:
        """Get all overdue corrective actions."""
        return [a for a in self._corrective_actions.values() if a.is_overdue]

    def get_responses(self, feedback_id: str) -> list[FeedbackResponse]:
        """Get all responses for a feedback."""
        return self._responses.get(feedback_id, [])

    def get_audit_trail(self, feedback_id: str) -> list[FeedbackAuditEntry]:
        """Get audit trail for a feedback."""
        return self._audit_trail.get(feedback_id, [])

    def get_insights(self) -> list[AnonymisedInsight]:
        """Get all registered insights."""
        return list(self._insights.values())

    def get_insights_by_threat_category(self, category: str) -> list[AnonymisedInsight]:
        """Get insights by threat category."""
        return [
            i for i in self._insights.values()
            if i.threat_category.lower() == category.lower()
        ]

    def get_metrics(self) -> dict[str, Any]:
        """Get supervisory feedback metrics."""
        # Update overdue actions count
        self._metrics["corrective_actions_overdue"] = len(self.get_overdue_actions())

        return {
            **self._metrics,
            "total_pending_feedback": len(self.get_pending_feedback()),
            "total_overdue_feedback": len(self.get_overdue_feedback()),
            "total_corrective_actions": len(self._corrective_actions),
            "total_insights_registered": len(self._insights)
        }

    def generate_feedback_summary(self, feedback_id: str) -> dict[str, Any]:
        """
        Generate a comprehensive summary of feedback processing.

        Args:
            feedback_id: ID of the feedback

        Returns:
            dict: Comprehensive feedback summary
        """
        feedback = self._feedback.get(feedback_id)
        if not feedback:
            raise ValueError(f"Feedback {feedback_id} not found")

        actions = self.get_corrective_actions(feedback_id)
        responses = self.get_responses(feedback_id)
        audit_trail = self.get_audit_trail(feedback_id)

        return {
            "feedback": {
                "id": feedback.feedback_id,
                "incident_id": feedback.incident_id,
                "authority": feedback.authority.name,
                "type": feedback.feedback_type.value,
                "priority": feedback.priority.value,
                "status": feedback.status.value,
                "subject": feedback.subject,
                "received_at": feedback.received_at.isoformat(),
                "response_deadline": feedback.response_deadline.isoformat() if feedback.response_deadline else None,
                "cross_border": feedback.cross_border_implications
            },
            "corrective_actions": {
                "total": len(actions),
                "completed": len([a for a in actions if a.status == "completed"]),
                "overdue": len([a for a in actions if a.is_overdue]),
                "actions": [
                    {
                        "id": a.action_id,
                        "type": a.action_type.value,
                        "status": a.status,
                        "deadline": a.deadline.isoformat(),
                        "is_overdue": a.is_overdue
                    }
                    for a in actions
                ]
            },
            "responses": {
                "total": len(responses),
                "response_history": [
                    {
                        "id": r.response_id,
                        "type": r.response_type.value,
                        "submitted_at": r.submitted_at.isoformat()
                    }
                    for r in responses
                ]
            },
            "audit_trail": {
                "entries": len(audit_trail),
                "timeline": [
                    {
                        "action": e.action,
                        "actor": e.actor,
                        "timestamp": e.timestamp.isoformat()
                    }
                    for e in audit_trail
                ]
            }
        }


def create_supervisory_feedback(
    entity_id: str,
    entity_name: str,
    primary_authority: Optional[CompetentAuthority] = None,
    response_sla_hours: int = 24,
    auto_acknowledge: bool = True
) -> DORASupervisioryFeedback:
    """
    Factory function to create a DORASupervisioryFeedback instance.

    Args:
        entity_id: Unique identifier for the financial entity
        entity_name: Name of the financial entity
        primary_authority: Primary competent authority
        response_sla_hours: SLA for responding to feedback
        auto_acknowledge: Automatically acknowledge received feedback

    Returns:
        DORASupervisioryFeedback: Configured instance
    """
    return DORASupervisioryFeedback(
        entity_id=entity_id,
        entity_name=entity_name,
        primary_authority=primary_authority,
        response_sla_hours=response_sla_hours,
        auto_acknowledge=auto_acknowledge
    )
