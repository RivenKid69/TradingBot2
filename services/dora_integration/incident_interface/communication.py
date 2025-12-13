# -*- coding: utf-8 -*-
"""
DORA Crisis Communication Management Module (Article 14).

Regulation (EU) 2022/2554 Article 14 requires financial entities to have:
    - Crisis communication policies
    - Internal and external communication arrangements
    - Staff training on communication procedures
    - Established communication channels

Integration Layer Context:
    For ICT Third-Party Providers:
    - We manage communication policies for OUR internal crisis response
    - We provide communication interfaces for CLIENTS during incidents
    - Clients have their own crisis communication to NCAs
    - We support clients with incident data and status updates

This module provides:
    1. Communication policy management
    2. Communication channel configuration
    3. Crisis communication workflow
    4. Stakeholder notification management
    5. Communication audit trail

References:
    - Article 14 DORA: https://www.digital-operational-resilience-act.com/Article_14.html
    - EBA Guidelines on ICT and security risk management
    - ESA Joint Committee Guidelines on outsourcing
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
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class CommunicationChannel(Enum):
    """Communication channel types."""
    EMAIL = "email"
    PHONE = "phone"
    SMS = "sms"
    SECURE_PORTAL = "secure_portal"
    API = "api"
    VIDEO_CONFERENCE = "video_conference"
    SECURE_MESSAGING = "secure_messaging"
    STATUS_PAGE = "status_page"
    WEBHOOK = "webhook"
    INTERNAL_SLACK = "internal_slack"
    INTERNAL_TEAMS = "internal_teams"


class StakeholderType(Enum):
    """Stakeholder types for communication."""
    INTERNAL_STAFF = "internal_staff"
    MANAGEMENT = "management"
    BOARD = "board"
    CLIENT = "client"
    REGULATOR = "regulator"
    ICT_PROVIDER = "ict_provider"
    MEDIA = "media"
    PUBLIC = "public"
    INCIDENT_RESPONSE_TEAM = "incident_response_team"
    LEGAL = "legal"


class CommunicationPriority(Enum):
    """Communication priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class CommunicationStatus(Enum):
    """Communication status."""
    DRAFT = "draft"
    APPROVED = "approved"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"


class CrisisPhase(Enum):
    """Crisis communication phases."""
    PRE_CRISIS = "pre_crisis"
    INITIAL_RESPONSE = "initial_response"
    ONGOING = "ongoing"
    RECOVERY = "recovery"
    POST_CRISIS = "post_crisis"


class PolicyStatus(Enum):
    """Policy status."""
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    ACTIVE = "active"
    ARCHIVED = "archived"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class CommunicationContact:
    """
    Communication contact for stakeholder notification.
    """
    contact_id: str = ""
    name: str = ""
    role: str = ""
    organization: str = ""
    stakeholder_type: StakeholderType = StakeholderType.CLIENT

    # Contact details
    email: str = ""
    phone: str = ""
    mobile: str = ""
    secure_portal_id: str = ""
    api_endpoint: str = ""
    webhook_url: str = ""

    # Preferences
    preferred_channel: CommunicationChannel = CommunicationChannel.EMAIL
    backup_channels: List[CommunicationChannel] = field(default_factory=list)
    language: str = "en"
    timezone: str = "UTC"

    # Escalation
    is_primary_contact: bool = False
    escalation_level: int = 1
    escalation_delay_minutes: int = 30

    # Status
    is_active: bool = True
    last_contacted_at: str = ""
    last_acknowledged_at: str = ""

    # Metadata
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.contact_id:
            self.contact_id = f"CONTACT-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.backup_channels:
            self.backup_channels = [CommunicationChannel.PHONE]


@dataclass
class CommunicationTemplate:
    """
    Communication template for standardized messaging.
    """
    template_id: str = ""
    name: str = ""
    description: str = ""

    # Template targeting
    stakeholder_types: List[StakeholderType] = field(default_factory=list)
    crisis_phases: List[CrisisPhase] = field(default_factory=list)
    priority_levels: List[CommunicationPriority] = field(default_factory=list)

    # Content
    subject_template: str = ""
    body_template: str = ""
    sms_template: str = ""

    # Variables available for substitution
    available_variables: List[str] = field(default_factory=list)

    # Approval
    requires_approval: bool = True
    approved_by: str = ""
    approved_at: str = ""

    # Status
    status: PolicyStatus = PolicyStatus.DRAFT
    version: str = "1.0"

    # Metadata
    created_at: str = ""
    updated_at: str = ""
    created_by: str = ""

    def __post_init__(self):
        if not self.template_id:
            self.template_id = f"TMPL-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.available_variables:
            self.available_variables = [
                "incident_id", "incident_title", "incident_severity",
                "detection_time", "current_status", "next_update_time",
                "contact_name", "organization_name",
            ]


@dataclass
class CommunicationRecord:
    """
    Record of a communication sent or received.
    """
    communication_id: str = ""
    incident_id: str = ""

    # Communication details
    subject: str = ""
    body: str = ""
    channel: CommunicationChannel = CommunicationChannel.EMAIL
    priority: CommunicationPriority = CommunicationPriority.NORMAL

    # Sender
    sender_name: str = ""
    sender_email: str = ""
    sender_role: str = ""

    # Recipients
    recipients: List[str] = field(default_factory=list)  # Contact IDs
    recipient_details: List[Dict[str, Any]] = field(default_factory=list)

    # Template used
    template_id: str = ""
    variables_used: Dict[str, Any] = field(default_factory=dict)

    # Status
    status: CommunicationStatus = CommunicationStatus.DRAFT
    sent_at: str = ""
    delivered_at: str = ""
    read_at: str = ""
    acknowledged_at: str = ""

    # Delivery tracking
    delivery_attempts: int = 0
    last_error: str = ""

    # Response
    response_received: bool = False
    response_content: str = ""
    response_received_at: str = ""

    # Crisis context
    crisis_phase: CrisisPhase = CrisisPhase.INITIAL_RESPONSE
    communication_sequence: int = 1  # Nth communication for this incident

    # Approval
    requires_approval: bool = True
    approved_by: str = ""
    approved_at: str = ""

    # Metadata
    created_at: str = ""
    created_by: str = ""

    def __post_init__(self):
        if not self.communication_id:
            self.communication_id = f"COMM-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class CommunicationPolicy:
    """
    Crisis communication policy per Article 14.
    """
    policy_id: str = ""
    name: str = ""
    description: str = ""
    version: str = "1.0"

    # Policy scope
    applies_to_stakeholders: List[StakeholderType] = field(default_factory=list)
    applies_to_crisis_phases: List[CrisisPhase] = field(default_factory=list)

    # Communication requirements
    notification_timeline_minutes: Dict[str, int] = field(default_factory=dict)
    update_frequency_minutes: Dict[str, int] = field(default_factory=dict)
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)

    # Approval workflow
    requires_management_approval: bool = True
    approval_timeout_minutes: int = 15
    auto_approve_low_priority: bool = False

    # Channels
    allowed_channels: List[CommunicationChannel] = field(default_factory=list)
    primary_channel: CommunicationChannel = CommunicationChannel.EMAIL
    fallback_channels: List[CommunicationChannel] = field(default_factory=list)

    # Templates
    template_ids: List[str] = field(default_factory=list)

    # Retention
    retention_days: int = 2555  # 7 years per DORA

    # Status
    status: PolicyStatus = PolicyStatus.DRAFT
    effective_date: str = ""
    expiry_date: str = ""

    # Approval
    approved_by: str = ""
    approved_at: str = ""

    # Review
    last_reviewed_at: str = ""
    next_review_at: str = ""
    review_frequency_days: int = 365

    # Metadata
    created_at: str = ""
    updated_at: str = ""
    created_by: str = ""

    def __post_init__(self):
        if not self.policy_id:
            self.policy_id = f"POL-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.notification_timeline_minutes:
            # Default timelines per stakeholder
            self.notification_timeline_minutes = {
                "internal_staff": 15,
                "management": 30,
                "board": 60,
                "client": 60,
                "regulator": 240,
            }
        if not self.update_frequency_minutes:
            self.update_frequency_minutes = {
                "critical": 30,
                "high": 60,
                "medium": 120,
                "low": 240,
            }
        if not self.allowed_channels:
            self.allowed_channels = [
                CommunicationChannel.EMAIL,
                CommunicationChannel.PHONE,
                CommunicationChannel.SECURE_PORTAL,
            ]


@dataclass
class CrisisStatus:
    """
    Current crisis status for communication management.
    """
    status_id: str = ""
    incident_id: str = ""

    # Current phase
    crisis_phase: CrisisPhase = CrisisPhase.INITIAL_RESPONSE

    # Status summary
    current_status: str = ""
    current_impact: str = ""
    affected_services: List[str] = field(default_factory=list)

    # Timeline
    incident_started_at: str = ""
    last_update_at: str = ""
    next_update_at: str = ""
    estimated_resolution_at: str = ""

    # Communication metrics
    communications_sent: int = 0
    stakeholders_notified: int = 0
    acknowledgements_received: int = 0

    # Actions
    current_actions: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)

    # Escalation
    escalation_level: int = 0
    escalated_to: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.status_id:
            self.status_id = f"STATUS-{uuid.uuid4().hex[:8].upper()}"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CommunicationConfig:
    """Configuration for crisis communication system."""
    # Organization info
    organization_name: str = ""
    organization_lei: str = ""

    # Default settings
    default_priority: CommunicationPriority = CommunicationPriority.NORMAL
    default_channel: CommunicationChannel = CommunicationChannel.EMAIL

    # Approval workflow
    require_approval_for_external: bool = True
    approval_timeout_minutes: int = 15

    # Escalation
    auto_escalate_on_no_ack_minutes: int = 30
    max_escalation_level: int = 3

    # Retry settings
    max_delivery_attempts: int = 3
    retry_interval_minutes: int = 5

    # Logging
    log_all_communications: bool = True
    log_path: str = "logs/dora/communications"

    # Callbacks
    on_communication_sent: Optional[Callable[[str, Dict], None]] = None
    on_escalation: Optional[Callable[[str, Dict], None]] = None


# =============================================================================
# Main Class Implementation
# =============================================================================

class DORACommunication:
    """
    DORA Article 14 Crisis Communication Management System.

    Provides:
    - Communication policy management
    - Stakeholder contact management
    - Template-based communication
    - Crisis communication workflow
    - Audit trail for all communications

    Integration Layer Purpose:
        As an ICT provider, this system manages:
        1. Internal crisis communication
        2. Client notification during incidents
        3. Communication audit trail for regulatory compliance

    Usage:
        config = CommunicationConfig(
            organization_name="Example Investment Platform",
        )
        comm = DORACommunication(config)

        # Create communication policy
        policy = comm.create_policy(
            name="Client Incident Communication Policy",
            applies_to_stakeholders=[StakeholderType.CLIENT],
        )

        # Register contacts
        contact = comm.register_contact(
            name="Client Contact",
            stakeholder_type=StakeholderType.CLIENT,
            email="client@example.com",
        )

        # During incident, send communication
        communication = comm.create_communication(
            incident_id="INC-001",
            subject="Service Incident Notification",
            body="...",
            recipients=[contact.contact_id],
        )
        comm.send_communication(communication.communication_id)
    """

    def __init__(self, config: Optional[CommunicationConfig] = None):
        """Initialize crisis communication system."""
        self.config = config or CommunicationConfig()

        # Data stores
        self._policies: Dict[str, CommunicationPolicy] = {}
        self._templates: Dict[str, CommunicationTemplate] = {}
        self._contacts: Dict[str, CommunicationContact] = {}
        self._communications: Dict[str, CommunicationRecord] = {}
        self._crisis_statuses: Dict[str, CrisisStatus] = {}

        # Indexes
        self._contacts_by_type: Dict[StakeholderType, Set[str]] = {
            t: set() for t in StakeholderType
        }
        self._communications_by_incident: Dict[str, Set[str]] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        # Initialize default templates
        self._init_default_templates()

        logger.info("DORACommunication initialized")

    def _init_default_templates(self) -> None:
        """Initialize default communication templates."""
        # Initial notification template
        initial = CommunicationTemplate(
            name="Initial Incident Notification",
            description="Template for initial incident notification to stakeholders",
            stakeholder_types=[StakeholderType.CLIENT, StakeholderType.MANAGEMENT],
            crisis_phases=[CrisisPhase.INITIAL_RESPONSE],
            priority_levels=[CommunicationPriority.HIGH, CommunicationPriority.CRITICAL],
            subject_template="[{priority}] Incident Notification - {incident_title}",
            body_template="""
Dear {contact_name},

We are writing to inform you of an incident affecting our services.

INCIDENT DETAILS
----------------
Incident ID: {incident_id}
Title: {incident_title}
Severity: {incident_severity}
Detected: {detection_time}
Current Status: {current_status}

IMPACT
------
{impact_description}

AFFECTED SERVICES
-----------------
{affected_services}

CURRENT ACTIONS
---------------
{current_actions}

NEXT UPDATE
-----------
Our next status update will be provided by {next_update_time}.

For urgent inquiries, please contact our support team.

Best regards,
{organization_name} Operations Team
""",
            status=PolicyStatus.ACTIVE,
        )

        # Status update template
        update = CommunicationTemplate(
            name="Incident Status Update",
            description="Template for ongoing incident status updates",
            stakeholder_types=[StakeholderType.CLIENT, StakeholderType.MANAGEMENT],
            crisis_phases=[CrisisPhase.ONGOING],
            priority_levels=[CommunicationPriority.NORMAL, CommunicationPriority.HIGH],
            subject_template="[Update #{update_number}] {incident_title}",
            body_template="""
Dear {contact_name},

This is an update regarding the ongoing incident.

CURRENT STATUS
--------------
Status: {current_status}
Phase: {crisis_phase}

PROGRESS
--------
{progress_description}

NEXT STEPS
----------
{next_steps}

ESTIMATED RESOLUTION
--------------------
{estimated_resolution}

Next update will be provided by {next_update_time}.

Best regards,
{organization_name} Operations Team
""",
            status=PolicyStatus.ACTIVE,
        )

        # Resolution template
        resolution = CommunicationTemplate(
            name="Incident Resolution Notification",
            description="Template for incident resolution notification",
            stakeholder_types=[StakeholderType.CLIENT, StakeholderType.MANAGEMENT],
            crisis_phases=[CrisisPhase.POST_CRISIS],
            priority_levels=[CommunicationPriority.NORMAL],
            subject_template="[Resolved] {incident_title}",
            body_template="""
Dear {contact_name},

We are pleased to inform you that the incident has been resolved.

RESOLUTION SUMMARY
------------------
Incident ID: {incident_id}
Title: {incident_title}
Resolution Time: {resolution_time}
Total Duration: {total_duration}

ROOT CAUSE
----------
{root_cause}

REMEDIATION
-----------
{remediation_actions}

PREVENTIVE MEASURES
-------------------
{preventive_measures}

We apologize for any inconvenience caused.

Best regards,
{organization_name} Operations Team
""",
            status=PolicyStatus.ACTIVE,
        )

        with self._lock:
            self._templates[initial.template_id] = initial
            self._templates[update.template_id] = update
            self._templates[resolution.template_id] = resolution

    # =========================================================================
    # Policy Management
    # =========================================================================

    def create_policy(
        self,
        name: str,
        description: str = "",
        applies_to_stakeholders: Optional[List[StakeholderType]] = None,
        applies_to_crisis_phases: Optional[List[CrisisPhase]] = None,
        notification_timeline_minutes: Optional[Dict[str, int]] = None,
        update_frequency_minutes: Optional[Dict[str, int]] = None,
        requires_management_approval: bool = True,
        allowed_channels: Optional[List[CommunicationChannel]] = None,
        created_by: str = "",
    ) -> CommunicationPolicy:
        """
        Create a communication policy.

        Args:
            name: Policy name
            description: Policy description
            applies_to_stakeholders: Target stakeholder types
            applies_to_crisis_phases: Applicable crisis phases
            notification_timeline_minutes: Notification timelines
            update_frequency_minutes: Update frequencies
            requires_management_approval: Whether approval is required
            allowed_channels: Allowed communication channels
            created_by: Creator

        Returns:
            Created CommunicationPolicy
        """
        policy = CommunicationPolicy(
            name=name,
            description=description,
            applies_to_stakeholders=applies_to_stakeholders or list(StakeholderType),
            applies_to_crisis_phases=applies_to_crisis_phases or list(CrisisPhase),
            notification_timeline_minutes=notification_timeline_minutes or {},
            update_frequency_minutes=update_frequency_minutes or {},
            requires_management_approval=requires_management_approval,
            allowed_channels=allowed_channels or [
                CommunicationChannel.EMAIL,
                CommunicationChannel.PHONE,
            ],
            created_by=created_by,
        )

        with self._lock:
            self._policies[policy.policy_id] = policy

        self._log_event("policy_created", {
            "policy_id": policy.policy_id,
            "name": name,
        })

        logger.info(f"Communication policy created: {policy.policy_id}")
        return policy

    def approve_policy(
        self,
        policy_id: str,
        approved_by: str,
    ) -> CommunicationPolicy:
        """Approve a policy."""
        with self._lock:
            if policy_id not in self._policies:
                raise ValueError(f"Policy {policy_id} not found")

            policy = self._policies[policy_id]
            policy.status = PolicyStatus.APPROVED
            policy.approved_by = approved_by
            policy.approved_at = datetime.now(timezone.utc).isoformat()

        return policy

    def activate_policy(
        self,
        policy_id: str,
        effective_date: Optional[str] = None,
    ) -> CommunicationPolicy:
        """Activate a policy."""
        with self._lock:
            if policy_id not in self._policies:
                raise ValueError(f"Policy {policy_id} not found")

            policy = self._policies[policy_id]
            policy.status = PolicyStatus.ACTIVE
            policy.effective_date = effective_date or datetime.now(timezone.utc).isoformat()
            policy.updated_at = datetime.now(timezone.utc).isoformat()

        return policy

    def get_policy(self, policy_id: str) -> Optional[CommunicationPolicy]:
        """Get policy by ID."""
        with self._lock:
            return self._policies.get(policy_id)

    def get_active_policies(self) -> List[CommunicationPolicy]:
        """Get all active policies."""
        with self._lock:
            return [p for p in self._policies.values() if p.status == PolicyStatus.ACTIVE]

    def get_policy_for_stakeholder(
        self,
        stakeholder_type: StakeholderType,
        crisis_phase: CrisisPhase,
    ) -> Optional[CommunicationPolicy]:
        """Get applicable policy for stakeholder and phase."""
        with self._lock:
            for policy in self._policies.values():
                if policy.status != PolicyStatus.ACTIVE:
                    continue
                if stakeholder_type in policy.applies_to_stakeholders:
                    if crisis_phase in policy.applies_to_crisis_phases:
                        return policy
        return None

    # =========================================================================
    # Contact Management
    # =========================================================================

    def register_contact(
        self,
        name: str,
        stakeholder_type: StakeholderType,
        email: str = "",
        phone: str = "",
        role: str = "",
        organization: str = "",
        preferred_channel: CommunicationChannel = CommunicationChannel.EMAIL,
        is_primary_contact: bool = False,
        escalation_level: int = 1,
    ) -> CommunicationContact:
        """
        Register a stakeholder contact.

        Args:
            name: Contact name
            stakeholder_type: Stakeholder type
            email: Email address
            phone: Phone number
            role: Role/title
            organization: Organization name
            preferred_channel: Preferred communication channel
            is_primary_contact: Whether this is primary contact
            escalation_level: Escalation level (1=first)

        Returns:
            Created CommunicationContact
        """
        contact = CommunicationContact(
            name=name,
            stakeholder_type=stakeholder_type,
            email=email,
            phone=phone,
            role=role,
            organization=organization,
            preferred_channel=preferred_channel,
            is_primary_contact=is_primary_contact,
            escalation_level=escalation_level,
        )

        with self._lock:
            self._contacts[contact.contact_id] = contact
            self._contacts_by_type[stakeholder_type].add(contact.contact_id)

        logger.info(f"Contact registered: {contact.contact_id} - {name}")
        return contact

    def update_contact(
        self,
        contact_id: str,
        **updates: Any,
    ) -> CommunicationContact:
        """Update a contact."""
        with self._lock:
            if contact_id not in self._contacts:
                raise ValueError(f"Contact {contact_id} not found")

            contact = self._contacts[contact_id]
            for key, value in updates.items():
                if hasattr(contact, key):
                    setattr(contact, key, value)
            contact.updated_at = datetime.now(timezone.utc).isoformat()

        return contact

    def get_contact(self, contact_id: str) -> Optional[CommunicationContact]:
        """Get contact by ID."""
        with self._lock:
            return self._contacts.get(contact_id)

    def get_contacts_by_type(
        self,
        stakeholder_type: StakeholderType,
        active_only: bool = True,
    ) -> List[CommunicationContact]:
        """Get contacts by stakeholder type."""
        with self._lock:
            contact_ids = self._contacts_by_type.get(stakeholder_type, set())
            contacts = [
                self._contacts[cid] for cid in contact_ids
                if cid in self._contacts
            ]
            if active_only:
                contacts = [c for c in contacts if c.is_active]
        return contacts

    def get_primary_contacts(
        self,
        stakeholder_type: Optional[StakeholderType] = None,
    ) -> List[CommunicationContact]:
        """Get primary contacts."""
        with self._lock:
            contacts = [
                c for c in self._contacts.values()
                if c.is_primary_contact and c.is_active
            ]
            if stakeholder_type:
                contacts = [c for c in contacts if c.stakeholder_type == stakeholder_type]
        return contacts

    # =========================================================================
    # Template Management
    # =========================================================================

    def create_template(
        self,
        name: str,
        subject_template: str,
        body_template: str,
        stakeholder_types: Optional[List[StakeholderType]] = None,
        crisis_phases: Optional[List[CrisisPhase]] = None,
        priority_levels: Optional[List[CommunicationPriority]] = None,
        description: str = "",
        sms_template: str = "",
        requires_approval: bool = True,
        created_by: str = "",
    ) -> CommunicationTemplate:
        """
        Create a communication template.

        Args:
            name: Template name
            subject_template: Subject template with {variables}
            body_template: Body template with {variables}
            stakeholder_types: Target stakeholder types
            crisis_phases: Applicable crisis phases
            priority_levels: Applicable priority levels
            description: Template description
            sms_template: Short SMS template
            requires_approval: Whether approval required
            created_by: Creator

        Returns:
            Created CommunicationTemplate
        """
        template = CommunicationTemplate(
            name=name,
            description=description,
            subject_template=subject_template,
            body_template=body_template,
            sms_template=sms_template,
            stakeholder_types=stakeholder_types or list(StakeholderType),
            crisis_phases=crisis_phases or list(CrisisPhase),
            priority_levels=priority_levels or list(CommunicationPriority),
            requires_approval=requires_approval,
            created_by=created_by,
        )

        with self._lock:
            self._templates[template.template_id] = template

        return template

    def get_template(self, template_id: str) -> Optional[CommunicationTemplate]:
        """Get template by ID."""
        with self._lock:
            return self._templates.get(template_id)

    def get_templates_for_context(
        self,
        stakeholder_type: StakeholderType,
        crisis_phase: CrisisPhase,
        priority: CommunicationPriority,
    ) -> List[CommunicationTemplate]:
        """Get templates matching context."""
        with self._lock:
            return [
                t for t in self._templates.values()
                if t.status == PolicyStatus.ACTIVE
                and stakeholder_type in t.stakeholder_types
                and crisis_phase in t.crisis_phases
                and priority in t.priority_levels
            ]

    def render_template(
        self,
        template_id: str,
        variables: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Render a template with variables.

        Args:
            template_id: Template ID
            variables: Variables for substitution

        Returns:
            Dict with 'subject', 'body', 'sms' keys
        """
        with self._lock:
            if template_id not in self._templates:
                raise ValueError(f"Template {template_id} not found")
            template = self._templates[template_id]

        # Simple variable substitution
        subject = template.subject_template
        body = template.body_template
        sms = template.sms_template

        for key, value in variables.items():
            placeholder = "{" + key + "}"
            subject = subject.replace(placeholder, str(value))
            body = body.replace(placeholder, str(value))
            sms = sms.replace(placeholder, str(value))

        return {
            "subject": subject,
            "body": body,
            "sms": sms,
        }

    # =========================================================================
    # Communication Workflow
    # =========================================================================

    def create_communication(
        self,
        incident_id: str,
        subject: str,
        body: str,
        recipients: List[str],
        channel: Optional[CommunicationChannel] = None,
        priority: CommunicationPriority = CommunicationPriority.NORMAL,
        template_id: str = "",
        variables_used: Optional[Dict[str, Any]] = None,
        crisis_phase: CrisisPhase = CrisisPhase.INITIAL_RESPONSE,
        requires_approval: bool = True,
        created_by: str = "",
    ) -> CommunicationRecord:
        """
        Create a communication record.

        Args:
            incident_id: Related incident ID
            subject: Communication subject
            body: Communication body
            recipients: List of contact IDs
            channel: Communication channel
            priority: Priority level
            template_id: Template used
            variables_used: Variables used in template
            crisis_phase: Current crisis phase
            requires_approval: Whether approval needed
            created_by: Creator

        Returns:
            Created CommunicationRecord
        """
        channel = channel or self.config.default_channel

        # Build recipient details
        recipient_details = []
        with self._lock:
            for contact_id in recipients:
                if contact_id in self._contacts:
                    contact = self._contacts[contact_id]
                    recipient_details.append({
                        "contact_id": contact.contact_id,
                        "name": contact.name,
                        "email": contact.email,
                        "phone": contact.phone,
                        "channel": channel.value,
                    })

            # Get sequence number for incident
            incident_comms = self._communications_by_incident.get(incident_id, set())
            sequence = len(incident_comms) + 1

        communication = CommunicationRecord(
            incident_id=incident_id,
            subject=subject,
            body=body,
            channel=channel,
            priority=priority,
            recipients=recipients,
            recipient_details=recipient_details,
            template_id=template_id,
            variables_used=variables_used or {},
            crisis_phase=crisis_phase,
            communication_sequence=sequence,
            requires_approval=requires_approval,
            created_by=created_by,
            sender_name=self.config.organization_name,
        )

        with self._lock:
            self._communications[communication.communication_id] = communication
            if incident_id not in self._communications_by_incident:
                self._communications_by_incident[incident_id] = set()
            self._communications_by_incident[incident_id].add(communication.communication_id)

        logger.info(f"Communication created: {communication.communication_id}")
        return communication

    def approve_communication(
        self,
        communication_id: str,
        approved_by: str,
    ) -> CommunicationRecord:
        """Approve a communication for sending."""
        with self._lock:
            if communication_id not in self._communications:
                raise ValueError(f"Communication {communication_id} not found")

            comm = self._communications[communication_id]
            comm.status = CommunicationStatus.APPROVED
            comm.approved_by = approved_by
            comm.approved_at = datetime.now(timezone.utc).isoformat()

        logger.info(f"Communication approved: {communication_id}")
        return comm

    def send_communication(
        self,
        communication_id: str,
        sent_by: str = "",
    ) -> CommunicationRecord:
        """
        Send a communication.

        Args:
            communication_id: Communication ID
            sent_by: Sender

        Returns:
            Updated CommunicationRecord
        """
        with self._lock:
            if communication_id not in self._communications:
                raise ValueError(f"Communication {communication_id} not found")

            comm = self._communications[communication_id]

            if comm.requires_approval and comm.status != CommunicationStatus.APPROVED:
                raise ValueError("Communication must be approved before sending")

            comm.delivery_attempts += 1
            now = datetime.now(timezone.utc).isoformat()

            # Simulate sending (actual implementation would use email/SMS/webhook services)
            success = self._send_via_channel(comm)

            if success:
                comm.status = CommunicationStatus.SENT
                comm.sent_at = now

                # Update contacts
                for contact_id in comm.recipients:
                    if contact_id in self._contacts:
                        self._contacts[contact_id].last_contacted_at = now

                self._log_event("communication_sent", {
                    "communication_id": communication_id,
                    "incident_id": comm.incident_id,
                    "recipients": len(comm.recipients),
                    "channel": comm.channel.value,
                })
            else:
                if comm.delivery_attempts >= self.config.max_delivery_attempts:
                    comm.status = CommunicationStatus.FAILED

        # Callback
        if success and self.config.on_communication_sent:
            try:
                self.config.on_communication_sent(communication_id, {
                    "incident_id": comm.incident_id,
                    "recipients": len(comm.recipients),
                })
            except Exception as e:
                logger.error(f"Communication callback failed: {e}")

        return comm

    def _send_via_channel(self, comm: CommunicationRecord) -> bool:
        """
        Send communication via the specified channel.

        Actual implementation would integrate with:
        - Email service (SendGrid, SES, etc.)
        - SMS service (Twilio, etc.)
        - Webhook endpoints
        - Portal notification systems
        """
        logger.info(
            f"Sending communication {comm.communication_id} via {comm.channel.value} "
            f"to {len(comm.recipients)} recipients"
        )
        # Simulated success
        return True

    def record_delivery(
        self,
        communication_id: str,
        delivered_at: Optional[str] = None,
    ) -> CommunicationRecord:
        """Record delivery confirmation."""
        with self._lock:
            if communication_id not in self._communications:
                raise ValueError(f"Communication {communication_id} not found")

            comm = self._communications[communication_id]
            comm.status = CommunicationStatus.DELIVERED
            comm.delivered_at = delivered_at or datetime.now(timezone.utc).isoformat()

        return comm

    def record_acknowledgement(
        self,
        communication_id: str,
        acknowledged_by: str = "",
        response_content: str = "",
    ) -> CommunicationRecord:
        """Record acknowledgement from recipient."""
        with self._lock:
            if communication_id not in self._communications:
                raise ValueError(f"Communication {communication_id} not found")

            comm = self._communications[communication_id]
            now = datetime.now(timezone.utc).isoformat()

            comm.status = CommunicationStatus.ACKNOWLEDGED
            comm.acknowledged_at = now
            if response_content:
                comm.response_received = True
                comm.response_content = response_content
                comm.response_received_at = now

            # Update contacts
            for contact_id in comm.recipients:
                if contact_id in self._contacts:
                    self._contacts[contact_id].last_acknowledged_at = now

        self._log_event("communication_acknowledged", {
            "communication_id": communication_id,
            "acknowledged_by": acknowledged_by,
        })

        return comm

    def get_communication(
        self,
        communication_id: str,
    ) -> Optional[CommunicationRecord]:
        """Get communication by ID."""
        with self._lock:
            return self._communications.get(communication_id)

    def get_communications_for_incident(
        self,
        incident_id: str,
    ) -> List[CommunicationRecord]:
        """Get all communications for an incident."""
        with self._lock:
            comm_ids = self._communications_by_incident.get(incident_id, set())
            return [
                self._communications[cid] for cid in comm_ids
                if cid in self._communications
            ]

    # =========================================================================
    # Crisis Status Management
    # =========================================================================

    def create_crisis_status(
        self,
        incident_id: str,
        current_status: str,
        current_impact: str = "",
        affected_services: Optional[List[str]] = None,
        crisis_phase: CrisisPhase = CrisisPhase.INITIAL_RESPONSE,
        current_actions: Optional[List[str]] = None,
        next_steps: Optional[List[str]] = None,
        estimated_resolution_at: str = "",
    ) -> CrisisStatus:
        """Create or update crisis status."""
        with self._lock:
            # Check if status exists for incident
            existing = None
            for status in self._crisis_statuses.values():
                if status.incident_id == incident_id:
                    existing = status
                    break

            now = datetime.now(timezone.utc).isoformat()

            if existing:
                existing.current_status = current_status
                existing.current_impact = current_impact
                existing.affected_services = affected_services or existing.affected_services
                existing.crisis_phase = crisis_phase
                existing.current_actions = current_actions or existing.current_actions
                existing.next_steps = next_steps or existing.next_steps
                existing.estimated_resolution_at = estimated_resolution_at
                existing.last_update_at = now
                return existing

            status = CrisisStatus(
                incident_id=incident_id,
                current_status=current_status,
                current_impact=current_impact,
                affected_services=affected_services or [],
                crisis_phase=crisis_phase,
                current_actions=current_actions or [],
                next_steps=next_steps or [],
                estimated_resolution_at=estimated_resolution_at,
                incident_started_at=now,
                last_update_at=now,
            )

            self._crisis_statuses[status.status_id] = status

        return status

    def get_crisis_status(
        self,
        incident_id: str,
    ) -> Optional[CrisisStatus]:
        """Get crisis status for incident."""
        with self._lock:
            for status in self._crisis_statuses.values():
                if status.incident_id == incident_id:
                    return status
        return None

    def update_crisis_phase(
        self,
        incident_id: str,
        new_phase: CrisisPhase,
    ) -> Optional[CrisisStatus]:
        """Update crisis phase."""
        with self._lock:
            for status in self._crisis_statuses.values():
                if status.incident_id == incident_id:
                    status.crisis_phase = new_phase
                    status.last_update_at = datetime.now(timezone.utc).isoformat()
                    return status
        return None

    # =========================================================================
    # Escalation
    # =========================================================================

    def check_escalation_needed(
        self,
        incident_id: str,
    ) -> bool:
        """Check if escalation is needed based on acknowledgement status."""
        with self._lock:
            comm_ids = self._communications_by_incident.get(incident_id, set())

            for comm_id in comm_ids:
                comm = self._communications.get(comm_id)
                if not comm:
                    continue

                if comm.status == CommunicationStatus.SENT:
                    # Check if sent long enough ago
                    sent_time = datetime.fromisoformat(
                        comm.sent_at.replace("Z", "+00:00")
                    )
                    now = datetime.now(timezone.utc)
                    elapsed_minutes = (now - sent_time).total_seconds() / 60

                    if elapsed_minutes >= self.config.auto_escalate_on_no_ack_minutes:
                        return True

        return False

    def escalate_communication(
        self,
        incident_id: str,
        reason: str = "",
        escalated_by: str = "",
    ) -> List[CommunicationContact]:
        """
        Escalate to next level contacts.

        Returns list of contacts to notify.
        """
        status = self.get_crisis_status(incident_id)
        if not status:
            return []

        current_level = status.escalation_level
        next_level = min(current_level + 1, self.config.max_escalation_level)

        # Get contacts at next escalation level
        escalation_contacts = []
        with self._lock:
            for contact in self._contacts.values():
                if contact.is_active and contact.escalation_level == next_level:
                    escalation_contacts.append(contact)

            # Update status
            status.escalation_level = next_level
            status.escalated_to = [c.contact_id for c in escalation_contacts]

        self._log_event("escalation", {
            "incident_id": incident_id,
            "new_level": next_level,
            "contacts": len(escalation_contacts),
            "reason": reason,
        })

        # Callback
        if self.config.on_escalation:
            try:
                self.config.on_escalation(incident_id, {
                    "level": next_level,
                    "contacts": [c.name for c in escalation_contacts],
                })
            except Exception as e:
                logger.error(f"Escalation callback failed: {e}")

        return escalation_contacts

    # =========================================================================
    # Statistics and Export
    # =========================================================================

    def get_communication_statistics(
        self,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get communication statistics."""
        if not period_end:
            period_end = datetime.now(timezone.utc).isoformat()
        if not period_start:
            period_start = (
                datetime.now(timezone.utc) - timedelta(days=30)
            ).isoformat()

        with self._lock:
            all_comms = list(self._communications.values())

        by_channel = {}
        for ch in CommunicationChannel:
            by_channel[ch.value] = sum(1 for c in all_comms if c.channel == ch)

        by_status = {}
        for st in CommunicationStatus:
            by_status[st.value] = sum(1 for c in all_comms if c.status == st)

        acknowledged = sum(
            1 for c in all_comms
            if c.status == CommunicationStatus.ACKNOWLEDGED
        )
        sent = sum(1 for c in all_comms if c.sent_at)

        return {
            "period_start": period_start,
            "period_end": period_end,
            "total_communications": len(all_comms),
            "by_channel": by_channel,
            "by_status": by_status,
            "acknowledgement_rate": acknowledged / sent if sent > 0 else 0,
            "total_contacts": len(self._contacts),
            "active_policies": sum(
                1 for p in self._policies.values()
                if p.status == PolicyStatus.ACTIVE
            ),
            "active_templates": sum(
                1 for t in self._templates.values()
                if t.status == PolicyStatus.ACTIVE
            ),
        }

    def export_communication_log(
        self,
        incident_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Export communication log for audit."""
        with self._lock:
            if incident_id:
                comm_ids = self._communications_by_incident.get(incident_id, set())
                communications = [
                    asdict(self._communications[cid])
                    for cid in comm_ids
                    if cid in self._communications
                ]
            else:
                communications = [asdict(c) for c in self._communications.values()]

        return {
            "export_date": datetime.now(timezone.utc).isoformat(),
            "article_reference": "Article 14",
            "incident_id": incident_id,
            "communication_count": len(communications),
            "communications": communications,
        }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a communication event."""
        if not self.config.log_all_communications:
            return

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"communications_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log communication event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_communication_service(
    config: Optional[CommunicationConfig] = None,
) -> DORACommunication:
    """Create a DORACommunication instance."""
    return DORACommunication(config=config)


def get_communication_channels() -> List[CommunicationChannel]:
    """Get list of communication channels."""
    return list(CommunicationChannel)


def get_stakeholder_types() -> List[StakeholderType]:
    """Get list of stakeholder types."""
    return list(StakeholderType)


def get_crisis_phases() -> List[CrisisPhase]:
    """Get list of crisis phases."""
    return list(CrisisPhase)
