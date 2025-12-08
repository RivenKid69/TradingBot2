"""
DORA Article 14 - Communication Module.

This module implements the communication requirements as specified in
DORA Article 14, covering internal/external communication policies,
incident notifications, stakeholder management, and crisis communication.

DORA Article 14 Requirements:
- Communication policies and procedures
- Internal communication channels
- External stakeholder communication
- Incident notification processes
- Regulatory authority reporting
- Crisis communication plans
- Client and counterparty notifications

Reference: Regulation (EU) 2022/2554, Article 14
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable
from pathlib import Path
import threading
import json
import uuid


class CommunicationType(Enum):
    """Types of communication per Article 14."""
    INTERNAL = "internal"
    EXTERNAL = "external"
    REGULATORY = "regulatory"
    CLIENT = "client"
    COUNTERPARTY = "counterparty"
    PUBLIC = "public"
    MEDIA = "media"
    VENDOR = "vendor"


class CommunicationChannel(Enum):
    """Communication channels."""
    EMAIL = "email"
    SECURE_PORTAL = "secure_portal"
    PHONE = "phone"
    SMS = "sms"
    SECURE_MESSAGING = "secure_messaging"
    VIDEO_CONFERENCE = "video_conference"
    POSTAL = "postal"
    PRESS_RELEASE = "press_release"
    WEBSITE = "website"
    API = "api"
    REGULATORY_SYSTEM = "regulatory_system"


class CommunicationPriority(Enum):
    """Priority levels for communications."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ROUTINE = "routine"


class CommunicationStatus(Enum):
    """Status of communications."""
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    SENT = "sent"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NotificationType(Enum):
    """Types of notifications."""
    INCIDENT = "incident"
    OUTAGE = "outage"
    MAINTENANCE = "maintenance"
    SECURITY_ALERT = "security_alert"
    COMPLIANCE = "compliance"
    POLICY_CHANGE = "policy_change"
    SERVICE_UPDATE = "service_update"
    REGULATORY_REPORT = "regulatory_report"
    CRISIS = "crisis"


class StakeholderType(Enum):
    """Types of stakeholders."""
    MANAGEMENT_BODY = "management_body"
    SENIOR_MANAGEMENT = "senior_management"
    STAFF = "staff"
    REGULATOR = "regulator"
    CLIENT = "client"
    COUNTERPARTY = "counterparty"
    VENDOR = "vendor"
    MEDIA = "media"
    PUBLIC = "public"
    AUDITOR = "auditor"


class EscalationTrigger(Enum):
    """Triggers for communication escalation."""
    NO_RESPONSE = "no_response"
    INCIDENT_SEVERITY = "incident_severity"
    TIME_THRESHOLD = "time_threshold"
    IMPACT_THRESHOLD = "impact_threshold"
    REGULATORY_DEADLINE = "regulatory_deadline"
    CRISIS_DECLARATION = "crisis_declaration"


@dataclass
class CommunicationPolicy:
    """
    Communication policy per Article 14(1).

    Defines communication requirements for different scenarios
    and stakeholder types.
    """
    policy_id: str
    name: str
    description: str
    communication_type: CommunicationType
    applicable_scenarios: list[str]
    target_stakeholders: list[StakeholderType]
    required_channels: list[CommunicationChannel]
    response_time_hours: int
    escalation_time_hours: int
    approval_required: bool
    approver_roles: list[str]
    content_requirements: list[str]
    template_id: str | None = None
    regulatory_basis: str | None = None
    effective_date: datetime = field(default_factory=datetime.now)
    review_date: datetime | None = None
    owner: str = ""
    status: str = "active"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Stakeholder:
    """
    Stakeholder record per Article 14(2).

    Maintains information about communication stakeholders
    and their preferences.
    """
    stakeholder_id: str
    name: str
    stakeholder_type: StakeholderType
    organization: str
    role: str
    primary_contact: str
    secondary_contact: str | None
    preferred_channel: CommunicationChannel
    alternative_channels: list[CommunicationChannel]
    language: str
    timezone: str
    notification_preferences: dict
    escalation_contact: str | None = None
    regulatory_id: str | None = None
    sla_hours: int | None = None
    tags: list[str] = field(default_factory=list)
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class CommunicationTemplate:
    """
    Communication template per Article 14(3).

    Pre-approved templates for consistent communication.
    """
    template_id: str
    name: str
    description: str
    communication_type: CommunicationType
    notification_type: NotificationType
    subject_template: str
    body_template: str
    required_fields: list[str]
    optional_fields: list[str]
    format: str  # html, text, markdown
    language: str
    applicable_channels: list[CommunicationChannel]
    approval_status: str
    approved_by: str | None = None
    approval_date: datetime | None = None
    version: str = "1.0"
    usage_count: int = 0
    last_used: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Communication:
    """
    Communication record per Article 14.

    Tracks individual communications including content,
    delivery status, and acknowledgment.
    """
    communication_id: str
    subject: str
    content: str
    communication_type: CommunicationType
    notification_type: NotificationType
    priority: CommunicationPriority
    status: CommunicationStatus
    channel: CommunicationChannel
    sender: str
    recipients: list[str]
    cc_recipients: list[str]
    template_id: str | None
    policy_id: str | None
    related_incident_id: str | None
    attachments: list[str]
    scheduled_time: datetime | None = None
    sent_time: datetime | None = None
    delivery_time: datetime | None = None
    acknowledgment_time: datetime | None = None
    acknowledged_by: str | None = None
    approval_required: bool = False
    approved_by: str | None = None
    approval_time: datetime | None = None
    delivery_status: dict = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    response_received: bool = False
    response_content: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class EscalationRule:
    """
    Communication escalation rule per Article 14(4).

    Defines when and how to escalate communications.
    """
    rule_id: str
    name: str
    description: str
    trigger: EscalationTrigger
    trigger_threshold: int  # hours or count depending on trigger
    applicable_communication_types: list[CommunicationType]
    applicable_priorities: list[CommunicationPriority]
    escalation_stakeholders: list[str]
    escalation_channel: CommunicationChannel
    escalation_message_template: str
    notification_to_original_sender: bool
    auto_escalate: bool
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CrisisCommunicationPlan:
    """
    Crisis communication plan per Article 14(5).

    Defines communication procedures during crisis events.
    """
    plan_id: str
    name: str
    description: str
    crisis_types: list[str]
    activation_criteria: list[str]
    communication_lead: str
    communication_team: list[str]
    spokespersons: list[str]
    internal_communication_sequence: list[dict]
    external_communication_sequence: list[dict]
    regulatory_notification_requirements: list[dict]
    media_handling_guidelines: list[str]
    social_media_guidelines: list[str]
    holding_statements: list[dict]
    key_messages: list[str]
    qa_document_id: str | None = None
    escalation_matrix: list[dict] = field(default_factory=list)
    contact_list: list[dict] = field(default_factory=list)
    review_date: datetime | None = None
    last_drill_date: datetime | None = None
    status: str = "approved"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class RegulatoryNotification:
    """
    Regulatory notification record per Article 14(6).

    Tracks mandatory notifications to competent authorities.
    """
    notification_id: str
    notification_type: str
    regulatory_authority: str
    authority_contact: str
    submission_channel: CommunicationChannel
    incident_id: str | None
    notification_content: str
    supporting_documents: list[str]
    deadline: datetime
    submission_time: datetime | None = None
    confirmation_received: bool = False
    confirmation_reference: str | None = None
    confirmation_time: datetime | None = None
    follow_up_required: bool = False
    follow_up_deadline: datetime | None = None
    follow_up_submitted: bool = False
    regulatory_feedback: str | None = None
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class CommunicationLog:
    """
    Communication audit log entry.

    Maintains audit trail of all communications.
    """
    log_id: str
    communication_id: str
    action: str
    actor: str
    timestamp: datetime
    details: dict
    ip_address: str | None = None
    system_info: str | None = None


class DORACommunication:
    """
    DORA Article 14 Communication Framework.

    Implements comprehensive communication management including
    policies, stakeholder management, template management,
    escalation, crisis communication, and regulatory notifications.

    Per Article 14 Requirements:
    - Communication policies and procedures
    - Stakeholder management
    - Template management
    - Escalation mechanisms
    - Crisis communication plans
    - Regulatory notification tracking
    - Complete audit trail
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize DORA Communication Framework.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._lock = threading.RLock()

        # Communication policies
        self._policies: dict[str, CommunicationPolicy] = {}

        # Stakeholders
        self._stakeholders: dict[str, Stakeholder] = {}

        # Templates
        self._templates: dict[str, CommunicationTemplate] = {}

        # Communications
        self._communications: dict[str, Communication] = {}

        # Escalation rules
        self._escalation_rules: dict[str, EscalationRule] = {}

        # Crisis communication plans
        self._crisis_plans: dict[str, CrisisCommunicationPlan] = {}

        # Regulatory notifications
        self._regulatory_notifications: dict[str, RegulatoryNotification] = {}

        # Communication log
        self._communication_log: list[CommunicationLog] = []

        # Callbacks
        self._send_callbacks: list[Callable] = []
        self._escalation_callbacks: list[Callable] = []

        # Event log
        self._event_log: list[dict] = []

    def _log_event(self, event_type: str, details: dict) -> None:
        """Log framework events."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details
        }
        self._event_log.append(event)

    def _log_communication_action(
        self,
        communication_id: str,
        action: str,
        actor: str,
        details: dict
    ) -> None:
        """Log communication action for audit."""
        log_entry = CommunicationLog(
            log_id=f"LOG-{uuid.uuid4().hex[:8].upper()}",
            communication_id=communication_id,
            action=action,
            actor=actor,
            timestamp=datetime.now(),
            details=details
        )
        self._communication_log.append(log_entry)

    # =========================================================================
    # Communication Policy Management (Article 14(1))
    # =========================================================================

    def create_policy(
        self,
        name: str,
        description: str,
        communication_type: CommunicationType,
        applicable_scenarios: list[str],
        target_stakeholders: list[StakeholderType],
        required_channels: list[CommunicationChannel],
        response_time_hours: int,
        escalation_time_hours: int,
        approval_required: bool,
        approver_roles: list[str],
        content_requirements: list[str],
        **kwargs
    ) -> CommunicationPolicy:
        """
        Create communication policy per Article 14(1).

        Args:
            name: Policy name
            description: Policy description
            communication_type: Type of communication
            applicable_scenarios: Scenarios where policy applies
            target_stakeholders: Target stakeholder types
            required_channels: Required communication channels
            response_time_hours: Required response time
            escalation_time_hours: Time before escalation
            approval_required: Whether approval is required
            approver_roles: Roles that can approve
            content_requirements: Required content elements
            **kwargs: Additional attributes

        Returns:
            Created CommunicationPolicy instance
        """
        with self._lock:
            policy_id = f"CP-{uuid.uuid4().hex[:8].upper()}"

            policy = CommunicationPolicy(
                policy_id=policy_id,
                name=name,
                description=description,
                communication_type=communication_type,
                applicable_scenarios=applicable_scenarios,
                target_stakeholders=target_stakeholders,
                required_channels=required_channels,
                response_time_hours=response_time_hours,
                escalation_time_hours=escalation_time_hours,
                approval_required=approval_required,
                approver_roles=approver_roles,
                content_requirements=content_requirements,
                **kwargs
            )

            self._policies[policy_id] = policy

            self._log_event("policy_created", {
                "policy_id": policy_id,
                "name": name,
                "communication_type": communication_type.value
            })

            return policy

    def get_policy(self, policy_id: str) -> CommunicationPolicy | None:
        """Get communication policy by ID."""
        with self._lock:
            return self._policies.get(policy_id)

    def get_applicable_policies(
        self,
        communication_type: CommunicationType,
        scenario: str
    ) -> list[CommunicationPolicy]:
        """Get policies applicable to communication type and scenario."""
        with self._lock:
            return [
                p for p in self._policies.values()
                if p.communication_type == communication_type
                and scenario in p.applicable_scenarios
                and p.status == "active"
            ]

    def update_policy_status(self, policy_id: str, status: str) -> bool:
        """Update policy status."""
        with self._lock:
            if policy_id not in self._policies:
                return False

            self._policies[policy_id].status = status
            self._policies[policy_id].updated_at = datetime.now()
            return True

    # =========================================================================
    # Stakeholder Management (Article 14(2))
    # =========================================================================

    def register_stakeholder(
        self,
        name: str,
        stakeholder_type: StakeholderType,
        organization: str,
        role: str,
        primary_contact: str,
        preferred_channel: CommunicationChannel,
        alternative_channels: list[CommunicationChannel],
        language: str,
        timezone: str,
        notification_preferences: dict,
        secondary_contact: str | None = None,
        **kwargs
    ) -> Stakeholder:
        """
        Register stakeholder per Article 14(2).

        Args:
            name: Stakeholder name
            stakeholder_type: Type of stakeholder
            organization: Organization name
            role: Role/title
            primary_contact: Primary contact info
            preferred_channel: Preferred communication channel
            alternative_channels: Alternative channels
            language: Preferred language
            timezone: Stakeholder timezone
            notification_preferences: Notification settings
            secondary_contact: Secondary contact info
            **kwargs: Additional attributes

        Returns:
            Created Stakeholder instance
        """
        with self._lock:
            stakeholder_id = f"SH-{uuid.uuid4().hex[:8].upper()}"

            stakeholder = Stakeholder(
                stakeholder_id=stakeholder_id,
                name=name,
                stakeholder_type=stakeholder_type,
                organization=organization,
                role=role,
                primary_contact=primary_contact,
                secondary_contact=secondary_contact,
                preferred_channel=preferred_channel,
                alternative_channels=alternative_channels,
                language=language,
                timezone=timezone,
                notification_preferences=notification_preferences,
                **kwargs
            )

            self._stakeholders[stakeholder_id] = stakeholder

            self._log_event("stakeholder_registered", {
                "stakeholder_id": stakeholder_id,
                "name": name,
                "type": stakeholder_type.value
            })

            return stakeholder

    def get_stakeholder(self, stakeholder_id: str) -> Stakeholder | None:
        """Get stakeholder by ID."""
        with self._lock:
            return self._stakeholders.get(stakeholder_id)

    def get_stakeholders_by_type(
        self,
        stakeholder_type: StakeholderType
    ) -> list[Stakeholder]:
        """Get stakeholders by type."""
        with self._lock:
            return [
                s for s in self._stakeholders.values()
                if s.stakeholder_type == stakeholder_type and s.active
            ]

    def update_stakeholder_contact(
        self,
        stakeholder_id: str,
        primary_contact: str | None = None,
        secondary_contact: str | None = None,
        preferred_channel: CommunicationChannel | None = None
    ) -> bool:
        """Update stakeholder contact information."""
        with self._lock:
            if stakeholder_id not in self._stakeholders:
                return False

            stakeholder = self._stakeholders[stakeholder_id]

            if primary_contact:
                stakeholder.primary_contact = primary_contact
            if secondary_contact:
                stakeholder.secondary_contact = secondary_contact
            if preferred_channel:
                stakeholder.preferred_channel = preferred_channel

            stakeholder.updated_at = datetime.now()
            return True

    def deactivate_stakeholder(self, stakeholder_id: str) -> bool:
        """Deactivate stakeholder."""
        with self._lock:
            if stakeholder_id not in self._stakeholders:
                return False

            self._stakeholders[stakeholder_id].active = False
            self._stakeholders[stakeholder_id].updated_at = datetime.now()
            return True

    # =========================================================================
    # Template Management (Article 14(3))
    # =========================================================================

    def create_template(
        self,
        name: str,
        description: str,
        communication_type: CommunicationType,
        notification_type: NotificationType,
        subject_template: str,
        body_template: str,
        required_fields: list[str],
        optional_fields: list[str],
        format: str,
        language: str,
        applicable_channels: list[CommunicationChannel],
        **kwargs
    ) -> CommunicationTemplate:
        """
        Create communication template per Article 14(3).

        Args:
            name: Template name
            description: Template description
            communication_type: Type of communication
            notification_type: Type of notification
            subject_template: Subject line template
            body_template: Body content template
            required_fields: Required placeholder fields
            optional_fields: Optional placeholder fields
            format: Content format (html, text, markdown)
            language: Template language
            applicable_channels: Applicable channels
            **kwargs: Additional attributes

        Returns:
            Created CommunicationTemplate instance
        """
        with self._lock:
            template_id = f"CT-{uuid.uuid4().hex[:8].upper()}"

            template = CommunicationTemplate(
                template_id=template_id,
                name=name,
                description=description,
                communication_type=communication_type,
                notification_type=notification_type,
                subject_template=subject_template,
                body_template=body_template,
                required_fields=required_fields,
                optional_fields=optional_fields,
                format=format,
                language=language,
                applicable_channels=applicable_channels,
                approval_status="pending",
                **kwargs
            )

            self._templates[template_id] = template

            self._log_event("template_created", {
                "template_id": template_id,
                "name": name
            })

            return template

    def get_template(self, template_id: str) -> CommunicationTemplate | None:
        """Get template by ID."""
        with self._lock:
            return self._templates.get(template_id)

    def approve_template(self, template_id: str, approved_by: str) -> bool:
        """Approve communication template."""
        with self._lock:
            if template_id not in self._templates:
                return False

            template = self._templates[template_id]
            template.approval_status = "approved"
            template.approved_by = approved_by
            template.approval_date = datetime.now()
            template.updated_at = datetime.now()

            self._log_event("template_approved", {
                "template_id": template_id,
                "approved_by": approved_by
            })

            return True

    def render_template(
        self,
        template_id: str,
        field_values: dict
    ) -> tuple[str, str] | None:
        """
        Render template with field values.

        Args:
            template_id: Template ID
            field_values: Values for template fields

        Returns:
            Tuple of (subject, body) or None if template not found
        """
        with self._lock:
            if template_id not in self._templates:
                return None

            template = self._templates[template_id]

            # Check required fields
            missing = [f for f in template.required_fields if f not in field_values]
            if missing:
                raise ValueError(f"Missing required fields: {missing}")

            # Render subject
            subject = template.subject_template
            for key, value in field_values.items():
                subject = subject.replace(f"{{{{{key}}}}}", str(value))

            # Render body
            body = template.body_template
            for key, value in field_values.items():
                body = body.replace(f"{{{{{key}}}}}", str(value))

            # Update usage stats
            template.usage_count += 1
            template.last_used = datetime.now()

            return (subject, body)

    # =========================================================================
    # Communication Management
    # =========================================================================

    def create_communication(
        self,
        subject: str,
        content: str,
        communication_type: CommunicationType,
        notification_type: NotificationType,
        priority: CommunicationPriority,
        channel: CommunicationChannel,
        sender: str,
        recipients: list[str],
        cc_recipients: list[str] | None = None,
        template_id: str | None = None,
        policy_id: str | None = None,
        related_incident_id: str | None = None,
        attachments: list[str] | None = None,
        scheduled_time: datetime | None = None,
        **kwargs
    ) -> Communication:
        """
        Create communication record.

        Args:
            subject: Communication subject
            content: Communication content
            communication_type: Type of communication
            notification_type: Type of notification
            priority: Priority level
            channel: Communication channel
            sender: Sender identifier
            recipients: List of recipients
            cc_recipients: CC recipients
            template_id: Template used
            policy_id: Applicable policy
            related_incident_id: Related incident
            attachments: Attachment references
            scheduled_time: Scheduled send time
            **kwargs: Additional attributes

        Returns:
            Created Communication instance
        """
        with self._lock:
            communication_id = f"COM-{uuid.uuid4().hex[:8].upper()}"

            # Check if approval is required based on policy
            approval_required = False
            if policy_id and policy_id in self._policies:
                approval_required = self._policies[policy_id].approval_required

            communication = Communication(
                communication_id=communication_id,
                subject=subject,
                content=content,
                communication_type=communication_type,
                notification_type=notification_type,
                priority=priority,
                status=CommunicationStatus.DRAFT,
                channel=channel,
                sender=sender,
                recipients=recipients,
                cc_recipients=cc_recipients or [],
                template_id=template_id,
                policy_id=policy_id,
                related_incident_id=related_incident_id,
                attachments=attachments or [],
                scheduled_time=scheduled_time,
                approval_required=approval_required,
                **kwargs
            )

            self._communications[communication_id] = communication

            self._log_communication_action(
                communication_id,
                "created",
                sender,
                {"subject": subject, "recipients": recipients}
            )

            self._log_event("communication_created", {
                "communication_id": communication_id,
                "type": communication_type.value,
                "priority": priority.value
            })

            return communication

    def get_communication(self, communication_id: str) -> Communication | None:
        """Get communication by ID."""
        with self._lock:
            return self._communications.get(communication_id)

    def approve_communication(
        self,
        communication_id: str,
        approved_by: str
    ) -> bool:
        """Approve communication for sending."""
        with self._lock:
            if communication_id not in self._communications:
                return False

            communication = self._communications[communication_id]
            communication.approved_by = approved_by
            communication.approval_time = datetime.now()
            communication.status = CommunicationStatus.APPROVED
            communication.updated_at = datetime.now()

            self._log_communication_action(
                communication_id,
                "approved",
                approved_by,
                {}
            )

            return True

    def send_communication(
        self,
        communication_id: str,
        actor: str
    ) -> bool:
        """
        Send communication.

        Args:
            communication_id: Communication ID
            actor: Person initiating send

        Returns:
            True if sent successfully
        """
        with self._lock:
            if communication_id not in self._communications:
                return False

            communication = self._communications[communication_id]

            # Check approval if required
            if communication.approval_required and not communication.approved_by:
                communication.status = CommunicationStatus.PENDING_APPROVAL
                return False

            # Update status
            communication.status = CommunicationStatus.SENT
            communication.sent_time = datetime.now()
            communication.updated_at = datetime.now()

            self._log_communication_action(
                communication_id,
                "sent",
                actor,
                {"channel": communication.channel.value}
            )

            # Trigger send callbacks
            for callback in self._send_callbacks:
                try:
                    callback(communication)
                except Exception:
                    pass

            return True

    def mark_delivered(
        self,
        communication_id: str,
        delivery_status: dict
    ) -> bool:
        """Mark communication as delivered."""
        with self._lock:
            if communication_id not in self._communications:
                return False

            communication = self._communications[communication_id]
            communication.status = CommunicationStatus.DELIVERED
            communication.delivery_time = datetime.now()
            communication.delivery_status = delivery_status
            communication.updated_at = datetime.now()

            return True

    def record_acknowledgment(
        self,
        communication_id: str,
        acknowledged_by: str,
        response_content: str | None = None
    ) -> bool:
        """Record acknowledgment of communication."""
        with self._lock:
            if communication_id not in self._communications:
                return False

            communication = self._communications[communication_id]
            communication.status = CommunicationStatus.ACKNOWLEDGED
            communication.acknowledgment_time = datetime.now()
            communication.acknowledged_by = acknowledged_by
            communication.response_received = True
            communication.updated_at = datetime.now()

            if response_content:
                communication.response_content = response_content

            self._log_communication_action(
                communication_id,
                "acknowledged",
                acknowledged_by,
                {"response_content": response_content is not None}
            )

            return True

    def add_send_callback(self, callback: Callable) -> None:
        """Register callback for sent communications."""
        self._send_callbacks.append(callback)

    # =========================================================================
    # Escalation Management (Article 14(4))
    # =========================================================================

    def create_escalation_rule(
        self,
        name: str,
        description: str,
        trigger: EscalationTrigger,
        trigger_threshold: int,
        applicable_communication_types: list[CommunicationType],
        applicable_priorities: list[CommunicationPriority],
        escalation_stakeholders: list[str],
        escalation_channel: CommunicationChannel,
        escalation_message_template: str,
        notification_to_original_sender: bool = True,
        auto_escalate: bool = True,
        **kwargs
    ) -> EscalationRule:
        """
        Create escalation rule per Article 14(4).

        Args:
            name: Rule name
            description: Rule description
            trigger: Escalation trigger type
            trigger_threshold: Threshold for trigger
            applicable_communication_types: Applicable types
            applicable_priorities: Applicable priorities
            escalation_stakeholders: Who to escalate to
            escalation_channel: Channel for escalation
            escalation_message_template: Template for message
            notification_to_original_sender: Notify original sender
            auto_escalate: Automatic escalation
            **kwargs: Additional attributes

        Returns:
            Created EscalationRule instance
        """
        with self._lock:
            rule_id = f"ER-{uuid.uuid4().hex[:8].upper()}"

            rule = EscalationRule(
                rule_id=rule_id,
                name=name,
                description=description,
                trigger=trigger,
                trigger_threshold=trigger_threshold,
                applicable_communication_types=applicable_communication_types,
                applicable_priorities=applicable_priorities,
                escalation_stakeholders=escalation_stakeholders,
                escalation_channel=escalation_channel,
                escalation_message_template=escalation_message_template,
                notification_to_original_sender=notification_to_original_sender,
                auto_escalate=auto_escalate,
                **kwargs
            )

            self._escalation_rules[rule_id] = rule

            self._log_event("escalation_rule_created", {
                "rule_id": rule_id,
                "name": name,
                "trigger": trigger.value
            })

            return rule

    def get_escalation_rule(self, rule_id: str) -> EscalationRule | None:
        """Get escalation rule by ID."""
        with self._lock:
            return self._escalation_rules.get(rule_id)

    def check_escalation_needed(
        self,
        communication_id: str
    ) -> list[EscalationRule]:
        """Check if communication needs escalation."""
        with self._lock:
            if communication_id not in self._communications:
                return []

            communication = self._communications[communication_id]
            triggered_rules = []

            for rule in self._escalation_rules.values():
                if not rule.active:
                    continue

                # Check if communication type matches
                if communication.communication_type not in rule.applicable_communication_types:
                    continue

                # Check if priority matches
                if communication.priority not in rule.applicable_priorities:
                    continue

                # Check trigger conditions
                if rule.trigger == EscalationTrigger.NO_RESPONSE:
                    if not communication.response_received:
                        hours_since_sent = 0
                        if communication.sent_time:
                            hours_since_sent = (
                                datetime.now() - communication.sent_time
                            ).total_seconds() / 3600
                        if hours_since_sent >= rule.trigger_threshold:
                            triggered_rules.append(rule)

                elif rule.trigger == EscalationTrigger.TIME_THRESHOLD:
                    hours_since_created = (
                        datetime.now() - communication.created_at
                    ).total_seconds() / 3600
                    if hours_since_created >= rule.trigger_threshold:
                        triggered_rules.append(rule)

            return triggered_rules

    def escalate_communication(
        self,
        communication_id: str,
        rule_id: str,
        escalated_by: str
    ) -> Communication | None:
        """
        Escalate communication according to rule.

        Args:
            communication_id: Original communication ID
            rule_id: Escalation rule ID
            escalated_by: Person escalating

        Returns:
            New escalation communication or None
        """
        with self._lock:
            if communication_id not in self._communications:
                return None
            if rule_id not in self._escalation_rules:
                return None

            original = self._communications[communication_id]
            rule = self._escalation_rules[rule_id]

            # Create escalation communication
            escalation = self.create_communication(
                subject=f"[ESCALATION] {original.subject}",
                content=rule.escalation_message_template.format(
                    original_subject=original.subject,
                    original_sender=original.sender,
                    sent_time=original.sent_time,
                    communication_id=communication_id
                ),
                communication_type=original.communication_type,
                notification_type=NotificationType.INCIDENT,
                priority=CommunicationPriority.CRITICAL,
                channel=rule.escalation_channel,
                sender=escalated_by,
                recipients=rule.escalation_stakeholders,
                related_incident_id=original.related_incident_id
            )

            self._log_event("communication_escalated", {
                "original_id": communication_id,
                "escalation_id": escalation.communication_id,
                "rule_id": rule_id
            })

            # Trigger escalation callbacks
            for callback in self._escalation_callbacks:
                try:
                    callback(original, escalation, rule)
                except Exception:
                    pass

            return escalation

    def add_escalation_callback(self, callback: Callable) -> None:
        """Register callback for escalations."""
        self._escalation_callbacks.append(callback)

    # =========================================================================
    # Crisis Communication Management (Article 14(5))
    # =========================================================================

    def create_crisis_plan(
        self,
        name: str,
        description: str,
        crisis_types: list[str],
        activation_criteria: list[str],
        communication_lead: str,
        communication_team: list[str],
        spokespersons: list[str],
        internal_communication_sequence: list[dict],
        external_communication_sequence: list[dict],
        regulatory_notification_requirements: list[dict],
        media_handling_guidelines: list[str],
        social_media_guidelines: list[str],
        holding_statements: list[dict],
        key_messages: list[str],
        **kwargs
    ) -> CrisisCommunicationPlan:
        """
        Create crisis communication plan per Article 14(5).

        Args:
            name: Plan name
            description: Plan description
            crisis_types: Types of crises covered
            activation_criteria: Criteria for activation
            communication_lead: Lead for communications
            communication_team: Team members
            spokespersons: Authorized spokespersons
            internal_communication_sequence: Internal comm sequence
            external_communication_sequence: External comm sequence
            regulatory_notification_requirements: Regulatory requirements
            media_handling_guidelines: Media guidelines
            social_media_guidelines: Social media guidelines
            holding_statements: Pre-approved statements
            key_messages: Key messages
            **kwargs: Additional attributes

        Returns:
            Created CrisisCommunicationPlan instance
        """
        with self._lock:
            plan_id = f"CCP-{uuid.uuid4().hex[:8].upper()}"

            plan = CrisisCommunicationPlan(
                plan_id=plan_id,
                name=name,
                description=description,
                crisis_types=crisis_types,
                activation_criteria=activation_criteria,
                communication_lead=communication_lead,
                communication_team=communication_team,
                spokespersons=spokespersons,
                internal_communication_sequence=internal_communication_sequence,
                external_communication_sequence=external_communication_sequence,
                regulatory_notification_requirements=regulatory_notification_requirements,
                media_handling_guidelines=media_handling_guidelines,
                social_media_guidelines=social_media_guidelines,
                holding_statements=holding_statements,
                key_messages=key_messages,
                **kwargs
            )

            self._crisis_plans[plan_id] = plan

            self._log_event("crisis_plan_created", {
                "plan_id": plan_id,
                "name": name,
                "crisis_types": crisis_types
            })

            return plan

    def get_crisis_plan(self, plan_id: str) -> CrisisCommunicationPlan | None:
        """Get crisis communication plan by ID."""
        with self._lock:
            return self._crisis_plans.get(plan_id)

    def get_applicable_crisis_plan(
        self,
        crisis_type: str
    ) -> CrisisCommunicationPlan | None:
        """Get applicable crisis plan for crisis type."""
        with self._lock:
            for plan in self._crisis_plans.values():
                if crisis_type in plan.crisis_types and plan.status == "approved":
                    return plan
            return None

    def record_crisis_drill(self, plan_id: str) -> bool:
        """Record crisis communication drill."""
        with self._lock:
            if plan_id not in self._crisis_plans:
                return False

            self._crisis_plans[plan_id].last_drill_date = datetime.now()
            self._crisis_plans[plan_id].updated_at = datetime.now()
            return True

    # =========================================================================
    # Regulatory Notification Management (Article 14(6))
    # =========================================================================

    def create_regulatory_notification(
        self,
        notification_type: str,
        regulatory_authority: str,
        authority_contact: str,
        submission_channel: CommunicationChannel,
        notification_content: str,
        deadline: datetime,
        supporting_documents: list[str] | None = None,
        incident_id: str | None = None,
        **kwargs
    ) -> RegulatoryNotification:
        """
        Create regulatory notification per Article 14(6).

        Args:
            notification_type: Type of notification
            regulatory_authority: Target authority
            authority_contact: Contact information
            submission_channel: Submission channel
            notification_content: Content of notification
            deadline: Submission deadline
            supporting_documents: Supporting docs
            incident_id: Related incident
            **kwargs: Additional attributes

        Returns:
            Created RegulatoryNotification instance
        """
        with self._lock:
            notification_id = f"RN-{uuid.uuid4().hex[:8].upper()}"

            notification = RegulatoryNotification(
                notification_id=notification_id,
                notification_type=notification_type,
                regulatory_authority=regulatory_authority,
                authority_contact=authority_contact,
                submission_channel=submission_channel,
                incident_id=incident_id,
                notification_content=notification_content,
                supporting_documents=supporting_documents or [],
                deadline=deadline,
                **kwargs
            )

            self._regulatory_notifications[notification_id] = notification

            self._log_event("regulatory_notification_created", {
                "notification_id": notification_id,
                "authority": regulatory_authority,
                "deadline": deadline.isoformat()
            })

            return notification

    def get_regulatory_notification(
        self,
        notification_id: str
    ) -> RegulatoryNotification | None:
        """Get regulatory notification by ID."""
        with self._lock:
            return self._regulatory_notifications.get(notification_id)

    def submit_regulatory_notification(
        self,
        notification_id: str,
        confirmation_reference: str | None = None
    ) -> bool:
        """Record submission of regulatory notification."""
        with self._lock:
            if notification_id not in self._regulatory_notifications:
                return False

            notification = self._regulatory_notifications[notification_id]
            notification.submission_time = datetime.now()
            notification.status = "submitted"
            notification.updated_at = datetime.now()

            if confirmation_reference:
                notification.confirmation_received = True
                notification.confirmation_reference = confirmation_reference
                notification.confirmation_time = datetime.now()

            self._log_event("regulatory_notification_submitted", {
                "notification_id": notification_id,
                "confirmation_reference": confirmation_reference
            })

            return True

    def get_pending_regulatory_notifications(self) -> list[RegulatoryNotification]:
        """Get pending regulatory notifications."""
        with self._lock:
            return [
                n for n in self._regulatory_notifications.values()
                if n.status == "pending"
            ]

    def get_overdue_notifications(self) -> list[RegulatoryNotification]:
        """Get overdue regulatory notifications."""
        with self._lock:
            now = datetime.now()
            return [
                n for n in self._regulatory_notifications.values()
                if n.status == "pending" and n.deadline < now
            ]

    # =========================================================================
    # Analytics and Reporting
    # =========================================================================

    def get_communication_metrics(self) -> dict:
        """Get communication framework metrics."""
        with self._lock:
            return {
                "total_policies": len(self._policies),
                "active_policies": len(
                    [p for p in self._policies.values() if p.status == "active"]
                ),
                "total_stakeholders": len(self._stakeholders),
                "active_stakeholders": len(
                    [s for s in self._stakeholders.values() if s.active]
                ),
                "stakeholders_by_type": self._count_by_enum(
                    self._stakeholders.values(), "stakeholder_type"
                ),
                "total_templates": len(self._templates),
                "approved_templates": len(
                    [t for t in self._templates.values() if t.approval_status == "approved"]
                ),
                "total_communications": len(self._communications),
                "communications_by_status": self._count_by_enum(
                    self._communications.values(), "status"
                ),
                "communications_by_type": self._count_by_enum(
                    self._communications.values(), "communication_type"
                ),
                "communications_by_priority": self._count_by_enum(
                    self._communications.values(), "priority"
                ),
                "total_escalation_rules": len(self._escalation_rules),
                "total_crisis_plans": len(self._crisis_plans),
                "total_regulatory_notifications": len(self._regulatory_notifications),
                "pending_regulatory_notifications": len(
                    self.get_pending_regulatory_notifications()
                ),
                "overdue_notifications": len(self.get_overdue_notifications())
            }

    def _count_by_enum(self, items: Any, attr: str) -> dict[str, int]:
        """Count items by enum attribute."""
        counts: dict[str, int] = {}
        for item in items:
            value = getattr(item, attr)
            key = value.value if hasattr(value, "value") else str(value)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def generate_communication_report(
        self,
        period_start: datetime,
        period_end: datetime
    ) -> dict:
        """
        Generate communication report for period.

        Args:
            period_start: Report period start
            period_end: Report period end

        Returns:
            Comprehensive communication report
        """
        with self._lock:
            # Filter communications by period
            comms_in_period = [
                c for c in self._communications.values()
                if period_start <= c.created_at <= period_end
            ]

            # Filter regulatory notifications by period
            reg_notifs_in_period = [
                n for n in self._regulatory_notifications.values()
                if period_start <= n.created_at <= period_end
            ]

            return {
                "report_period": {
                    "start": period_start.isoformat(),
                    "end": period_end.isoformat()
                },
                "communications": {
                    "total": len(comms_in_period),
                    "by_type": self._count_by_enum(comms_in_period, "communication_type"),
                    "by_status": self._count_by_enum(comms_in_period, "status"),
                    "by_priority": self._count_by_enum(comms_in_period, "priority"),
                    "by_channel": self._count_by_enum(comms_in_period, "channel"),
                    "acknowledgment_rate": self._calculate_acknowledgment_rate(
                        comms_in_period
                    )
                },
                "regulatory_notifications": {
                    "total": len(reg_notifs_in_period),
                    "submitted": len(
                        [n for n in reg_notifs_in_period if n.status == "submitted"]
                    ),
                    "pending": len(
                        [n for n in reg_notifs_in_period if n.status == "pending"]
                    ),
                    "on_time_rate": self._calculate_on_time_rate(reg_notifs_in_period)
                },
                "template_usage": self._get_template_usage_stats(),
                "escalations": self._count_escalations(comms_in_period),
                "generated_at": datetime.now().isoformat()
            }

    def _calculate_acknowledgment_rate(self, communications: list) -> float:
        """Calculate acknowledgment rate."""
        if not communications:
            return 0.0
        acknowledged = len(
            [c for c in communications if c.status == CommunicationStatus.ACKNOWLEDGED]
        )
        return acknowledged / len(communications)

    def _calculate_on_time_rate(self, notifications: list) -> float:
        """Calculate on-time submission rate."""
        submitted = [n for n in notifications if n.submission_time]
        if not submitted:
            return 0.0
        on_time = len([n for n in submitted if n.submission_time <= n.deadline])
        return on_time / len(submitted)

    def _get_template_usage_stats(self) -> list[dict]:
        """Get template usage statistics."""
        return [
            {
                "template_id": t.template_id,
                "name": t.name,
                "usage_count": t.usage_count
            }
            for t in sorted(
                self._templates.values(),
                key=lambda x: x.usage_count,
                reverse=True
            )[:10]
        ]

    def _count_escalations(self, communications: list) -> int:
        """Count escalation communications."""
        return len([c for c in communications if "[ESCALATION]" in c.subject])

    def get_communication_log(
        self,
        communication_id: str | None = None
    ) -> list[CommunicationLog]:
        """Get communication audit log."""
        with self._lock:
            if communication_id:
                return [
                    log for log in self._communication_log
                    if log.communication_id == communication_id
                ]
            return list(self._communication_log)

    # =========================================================================
    # Export and Persistence
    # =========================================================================

    def export_to_file(self, filepath: Path) -> bool:
        """Export communication data to JSONL file."""
        try:
            with open(filepath, "w") as f:
                for event in self._event_log:
                    f.write(json.dumps(event) + "\n")
            return True
        except Exception:
            return False

    def get_event_log(self) -> list[dict]:
        """Get event log."""
        with self._lock:
            return list(self._event_log)


# =============================================================================
# Factory Function
# =============================================================================

def create_dora_communication(config: dict | None = None) -> DORACommunication:
    """
    Factory function to create DORA Communication Framework.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured DORACommunication instance
    """
    return DORACommunication(config=config)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "CommunicationType",
    "CommunicationChannel",
    "CommunicationPriority",
    "CommunicationStatus",
    "NotificationType",
    "StakeholderType",
    "EscalationTrigger",
    # Dataclasses
    "CommunicationPolicy",
    "Stakeholder",
    "CommunicationTemplate",
    "Communication",
    "EscalationRule",
    "CrisisCommunicationPlan",
    "RegulatoryNotification",
    "CommunicationLog",
    # Main class
    "DORACommunication",
    # Factory
    "create_dora_communication",
]
