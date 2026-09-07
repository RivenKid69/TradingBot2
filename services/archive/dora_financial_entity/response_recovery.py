# -*- coding: utf-8 -*-
"""
DORA Response and Recovery Module (Article 11).

Regulation (EU) 2022/2554 Article 11 defines response and recovery requirements:
    - ICT business continuity policy as part of operational continuity
    - Implementation through business continuity plans
    - Quick, appropriate and effective response to all ICT-related incidents
    - Activation, escalation, internal and external communication
    - Crisis management plans with clear procedures

This module provides:
    1. IncidentResponse - Incident response management
    2. CrisisManagement - Crisis management procedures
    3. EscalationManager - Escalation handling
    4. RecoveryManager - System recovery operations
    5. DORAResponseRecovery - Main response/recovery orchestrator

References:
    - Article 11 DORA: https://www.digital-operational-resilience-act.com/Article_11.html
    - RTS on ICT Risk Management: CDR 2024/1774
    - NIST SP 800-61: Computer Security Incident Handling Guide
    - ISO 22301:2019 Business Continuity Management
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


class IncidentSeverity(Enum):
    """Incident severity levels."""

    P1_CRITICAL = "p1_critical"  # Critical business impact
    P2_HIGH = "p2_high"  # Significant impact
    P3_MEDIUM = "p3_medium"  # Moderate impact
    P4_LOW = "p4_low"  # Minor impact


class IncidentStatus(Enum):
    """Incident lifecycle status."""

    NEW = "new"
    TRIAGED = "triaged"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    ERADICATING = "eradicating"
    RECOVERING = "recovering"
    RESOLVED = "resolved"
    POST_INCIDENT = "post_incident"
    CLOSED = "closed"


class IncidentCategory(Enum):
    """ICT incident categories per DORA."""

    CYBER_ATTACK = "cyber_attack"
    SYSTEM_FAILURE = "system_failure"
    DATA_BREACH = "data_breach"
    NETWORK_OUTAGE = "network_outage"
    APPLICATION_FAILURE = "application_failure"
    THIRD_PARTY_FAILURE = "third_party_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    HUMAN_ERROR = "human_error"
    CONFIGURATION_ERROR = "configuration_error"
    CAPACITY_ISSUE = "capacity_issue"


class EscalationLevel(Enum):
    """Escalation levels."""

    L1_OPERATIONS = "l1_operations"
    L2_ENGINEERING = "l2_engineering"
    L3_MANAGEMENT = "l3_management"
    L4_EXECUTIVE = "l4_executive"
    L5_CRISIS = "l5_crisis"


class CrisisStatus(Enum):
    """Crisis status."""

    STANDBY = "standby"
    ACTIVATED = "activated"
    ESCALATED = "escalated"
    DE_ESCALATING = "de_escalating"
    DEACTIVATED = "deactivated"


class RecoveryStatus(Enum):
    """Recovery operation status."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class ICTIncident:
    """
    ICT Incident record per DORA Article 11.

    Documents ICT-related incidents requiring response.
    """

    incident_id: str = ""
    title: str = ""
    description: str = ""
    category: IncidentCategory = IncidentCategory.SYSTEM_FAILURE
    severity: IncidentSeverity = IncidentSeverity.P3_MEDIUM

    # Status and lifecycle
    status: IncidentStatus = IncidentStatus.NEW
    escalation_level: EscalationLevel = EscalationLevel.L1_OPERATIONS

    # Impact
    affected_systems: List[str] = field(default_factory=list)
    affected_functions: List[str] = field(default_factory=list)
    affected_users_count: int = 0
    business_impact: str = ""
    financial_impact_eur: Optional[float] = None

    # Detection
    detection_time: str = ""
    detection_source: str = ""  # "monitoring", "user_report", "alert", etc.
    alert_id: str = ""

    # Response
    response_started: Optional[str] = None
    responders: List[str] = field(default_factory=list)
    incident_commander: str = ""

    # Containment
    containment_time: Optional[str] = None
    containment_actions: List[str] = field(default_factory=list)

    # Resolution
    resolution_time: Optional[str] = None
    resolution_actions: List[str] = field(default_factory=list)
    root_cause: str = ""
    root_cause_category: str = ""

    # Recovery
    recovery_started: Optional[str] = None
    recovery_completed: Optional[str] = None
    recovery_actions: List[str] = field(default_factory=list)

    # Communication
    internal_communications: List[Dict[str, Any]] = field(default_factory=list)
    external_communications: List[Dict[str, Any]] = field(default_factory=list)
    regulatory_notification_required: bool = False
    regulatory_notification_sent: Optional[str] = None

    # Classification
    is_major_incident: bool = False
    is_reportable: bool = False

    # Post-incident
    post_incident_review_completed: bool = False
    post_incident_report_id: str = ""
    lessons_learned: List[str] = field(default_factory=list)
    improvement_actions: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    created_date: str = ""
    last_updated: str = ""
    closed_date: Optional[str] = None

    def __post_init__(self):
        if not self.incident_id:
            self.incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        if not self.detection_time:
            self.detection_time = datetime.now(timezone.utc).isoformat()
        if not self.created_date:
            self.created_date = datetime.now(timezone.utc).isoformat()
        if not self.last_updated:
            self.last_updated = self.created_date

    @property
    def time_to_detect_minutes(self) -> Optional[float]:
        """Calculate time from incident start to detection."""
        return None  # Would require incident start time

    @property
    def time_to_respond_minutes(self) -> Optional[float]:
        """Calculate time from detection to response start."""
        if not self.response_started:
            return None
        detection = datetime.fromisoformat(self.detection_time.replace("Z", "+00:00"))
        response = datetime.fromisoformat(self.response_started.replace("Z", "+00:00"))
        return (response - detection).total_seconds() / 60

    @property
    def time_to_resolve_minutes(self) -> Optional[float]:
        """Calculate time from detection to resolution."""
        if not self.resolution_time:
            return None
        detection = datetime.fromisoformat(self.detection_time.replace("Z", "+00:00"))
        resolution = datetime.fromisoformat(self.resolution_time.replace("Z", "+00:00"))
        return (resolution - detection).total_seconds() / 60


@dataclass
class ResponseProcedure:
    """
    Incident response procedure.
    """

    procedure_id: str = ""
    name: str = ""
    description: str = ""
    incident_category: IncidentCategory = IncidentCategory.SYSTEM_FAILURE
    severity_applicable: List[IncidentSeverity] = field(default_factory=list)

    # Steps
    steps: List[Dict[str, Any]] = field(default_factory=list)
    estimated_duration_minutes: int = 0

    # Roles
    required_roles: List[str] = field(default_factory=list)
    incident_commander_required: bool = True

    # Escalation
    escalation_criteria: List[str] = field(default_factory=list)
    escalation_timeout_minutes: int = 30

    # Communication
    notification_templates: List[str] = field(default_factory=list)
    stakeholders_to_notify: List[str] = field(default_factory=list)

    # Documentation
    documentation_required: List[str] = field(default_factory=list)

    # Status
    is_active: bool = True
    version: str = "1.0"
    last_reviewed: Optional[str] = None

    def __post_init__(self):
        if not self.procedure_id:
            self.procedure_id = f"RESP-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class EscalationRule:
    """
    Escalation rule definition.
    """

    rule_id: str = ""
    name: str = ""
    description: str = ""

    # Trigger conditions
    trigger_severity: List[IncidentSeverity] = field(default_factory=list)
    trigger_category: List[IncidentCategory] = field(default_factory=list)
    trigger_elapsed_minutes: int = 0
    trigger_conditions: List[str] = field(default_factory=list)

    # Escalation target
    from_level: EscalationLevel = EscalationLevel.L1_OPERATIONS
    to_level: EscalationLevel = EscalationLevel.L2_ENGINEERING

    # Notification
    notify_contacts: List[str] = field(default_factory=list)
    notification_method: str = ""  # "email", "sms", "phone", "all"

    # Status
    is_active: bool = True

    def __post_init__(self):
        if not self.rule_id:
            self.rule_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class CrisisEvent:
    """
    Crisis event record.
    """

    crisis_id: str = ""
    title: str = ""
    description: str = ""
    status: CrisisStatus = CrisisStatus.STANDBY

    # Trigger
    trigger_incident_ids: List[str] = field(default_factory=list)
    trigger_reason: str = ""
    activated_time: Optional[str] = None
    activated_by: str = ""

    # Management
    crisis_commander: str = ""
    crisis_team: List[str] = field(default_factory=list)
    war_room_location: str = ""

    # Impact
    affected_functions: List[str] = field(default_factory=list)
    business_impact_level: str = ""

    # Actions
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    decisions_made: List[Dict[str, Any]] = field(default_factory=list)

    # Communications
    internal_comms: List[Dict[str, Any]] = field(default_factory=list)
    external_comms: List[Dict[str, Any]] = field(default_factory=list)
    regulatory_comms: List[Dict[str, Any]] = field(default_factory=list)

    # Resolution
    deactivated_time: Optional[str] = None
    deactivated_by: str = ""
    resolution_summary: str = ""

    def __post_init__(self):
        if not self.crisis_id:
            self.crisis_id = f"CRISIS-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class RecoveryAction:
    """
    Recovery action record.
    """

    action_id: str = ""
    incident_id: str = ""
    action_type: str = ""  # "restore", "failover", "rebuild", "workaround"
    description: str = ""

    # Target
    target_system: str = ""
    target_function: str = ""

    # Execution
    status: RecoveryStatus = RecoveryStatus.NOT_STARTED
    started_time: Optional[str] = None
    completed_time: Optional[str] = None
    executed_by: str = ""

    # Verification
    verification_steps: List[str] = field(default_factory=list)
    verification_status: str = ""
    verified_by: str = ""
    verified_time: Optional[str] = None

    # Rollback
    rollback_available: bool = True
    rollback_procedure: str = ""
    rollback_executed: bool = False

    def __post_init__(self):
        if not self.action_id:
            self.action_id = f"REC-{uuid.uuid4().hex[:8].upper()}"


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ResponseRecoveryConfig:
    """Configuration for Response and Recovery module."""

    # Response times (target in minutes)
    target_response_time_p1_minutes: int = 15
    target_response_time_p2_minutes: int = 30
    target_response_time_p3_minutes: int = 60
    target_response_time_p4_minutes: int = 240

    # Escalation
    auto_escalation_enabled: bool = True
    escalation_check_interval_minutes: int = 5

    # Crisis management
    crisis_activation_threshold: IncidentSeverity = IncidentSeverity.P1_CRITICAL
    crisis_team_notification_minutes: int = 5

    # Recovery
    require_verification_before_close: bool = True
    post_incident_review_required: bool = True
    post_incident_review_deadline_hours: int = 72

    # Notifications
    notification_enabled: bool = True
    regulatory_notification_threshold: IncidentSeverity = IncidentSeverity.P1_CRITICAL

    # Logging
    log_all_events: bool = True
    log_path: str = "logs/dora/response_recovery"

    # Callbacks
    notification_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# Main Response and Recovery Class
# =============================================================================


class DORAResponseRecovery:
    """
    DORA Article 11 compliant Response and Recovery module.

    Manages:
    - Incident lifecycle management
    - Response procedures
    - Escalation handling
    - Crisis management
    - Recovery operations
    - Post-incident review

    Usage:
        config = ResponseRecoveryConfig()
        response = DORAResponseRecovery(config)

        # Create incident
        incident = response.create_incident(
            title="Database Outage",
            category=IncidentCategory.SYSTEM_FAILURE,
            severity=IncidentSeverity.P1_CRITICAL,
        )

        # Start response
        response.start_response(incident.incident_id, responders=["team1"])

        # Escalate if needed
        response.escalate_incident(incident.incident_id, EscalationLevel.L3_MANAGEMENT)
    """

    def __init__(self, config: Optional[ResponseRecoveryConfig] = None):
        """Initialize Response and Recovery module."""
        self.config = config or ResponseRecoveryConfig()

        # Data stores
        self._incidents: Dict[str, ICTIncident] = {}
        self._procedures: Dict[str, ResponseProcedure] = {}
        self._escalation_rules: Dict[str, EscalationRule] = {}
        self._crises: Dict[str, CrisisEvent] = {}
        self._recovery_actions: Dict[str, RecoveryAction] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("DORAResponseRecovery initialized")

    # =========================================================================
    # Incident Management
    # =========================================================================

    def create_incident(
        self,
        title: str,
        category: IncidentCategory,
        severity: IncidentSeverity,
        description: str = "",
        affected_systems: Optional[List[str]] = None,
        affected_functions: Optional[List[str]] = None,
        detection_source: str = "monitoring",
        alert_id: str = "",
    ) -> ICTIncident:
        """
        Create a new ICT incident.

        Per Article 11, entities must respond quickly and effectively
        to all ICT-related incidents.

        Args:
            title: Incident title
            category: Incident category
            severity: Incident severity
            description: Incident description
            affected_systems: Affected systems
            affected_functions: Affected functions
            detection_source: Detection source
            alert_id: Related alert ID

        Returns:
            Created ICTIncident
        """
        incident = ICTIncident(
            title=title,
            category=category,
            severity=severity,
            description=description,
            affected_systems=affected_systems or [],
            affected_functions=affected_functions or [],
            detection_source=detection_source,
            alert_id=alert_id,
        )

        # Determine if major incident
        incident.is_major_incident = severity in (
            IncidentSeverity.P1_CRITICAL,
            IncidentSeverity.P2_HIGH,
        )

        # Determine if reportable
        incident.is_reportable = (
            severity == IncidentSeverity.P1_CRITICAL
            or category == IncidentCategory.CYBER_ATTACK
            or category == IncidentCategory.DATA_BREACH
        )

        incident.regulatory_notification_required = (
            severity.value <= self.config.regulatory_notification_threshold.value
        )

        with self._lock:
            self._incidents[incident.incident_id] = incident

        # Check for crisis activation
        if severity == self.config.crisis_activation_threshold:
            self._consider_crisis_activation(incident)

        self._log_event(
            "incident_created",
            {
                "incident_id": incident.incident_id,
                "title": title,
                "severity": severity.value,
                "category": category.value,
            },
        )

        logger.warning(f"Incident created: {incident.incident_id} - {title} ({severity.value})")
        return incident

    def start_response(
        self,
        incident_id: str,
        responders: Optional[List[str]] = None,
        incident_commander: str = "",
    ) -> Optional[ICTIncident]:
        """Start incident response."""
        with self._lock:
            if incident_id not in self._incidents:
                return None

            incident = self._incidents[incident_id]
            incident.status = IncidentStatus.INVESTIGATING
            incident.response_started = datetime.now(timezone.utc).isoformat()
            incident.responders = responders or []
            incident.incident_commander = incident_commander
            incident.last_updated = datetime.now(timezone.utc).isoformat()

        self._log_event(
            "response_started",
            {
                "incident_id": incident_id,
                "responders": responders,
            },
        )

        return incident

    def contain_incident(
        self,
        incident_id: str,
        containment_actions: Optional[List[str]] = None,
    ) -> Optional[ICTIncident]:
        """Mark incident as contained."""
        with self._lock:
            if incident_id not in self._incidents:
                return None

            incident = self._incidents[incident_id]
            incident.status = IncidentStatus.CONTAINED
            incident.containment_time = datetime.now(timezone.utc).isoformat()
            incident.containment_actions = containment_actions or []
            incident.last_updated = datetime.now(timezone.utc).isoformat()

        self._log_event(
            "incident_contained",
            {
                "incident_id": incident_id,
                "containment_actions": containment_actions,
            },
        )

        return incident

    def resolve_incident(
        self,
        incident_id: str,
        resolution_actions: Optional[List[str]] = None,
        root_cause: str = "",
        root_cause_category: str = "",
    ) -> Optional[ICTIncident]:
        """Mark incident as resolved."""
        with self._lock:
            if incident_id not in self._incidents:
                return None

            incident = self._incidents[incident_id]
            incident.status = IncidentStatus.RESOLVED
            incident.resolution_time = datetime.now(timezone.utc).isoformat()
            incident.resolution_actions = resolution_actions or []
            incident.root_cause = root_cause
            incident.root_cause_category = root_cause_category
            incident.last_updated = datetime.now(timezone.utc).isoformat()

        self._log_event(
            "incident_resolved",
            {
                "incident_id": incident_id,
                "root_cause": root_cause,
            },
        )

        return incident

    def close_incident(
        self,
        incident_id: str,
        require_post_incident_review: bool = True,
    ) -> Optional[ICTIncident]:
        """Close an incident."""
        with self._lock:
            if incident_id not in self._incidents:
                return None

            incident = self._incidents[incident_id]

            if require_post_incident_review and not incident.post_incident_review_completed:
                logger.warning(f"Cannot close {incident_id}: post-incident review required")
                return None

            incident.status = IncidentStatus.CLOSED
            incident.closed_date = datetime.now(timezone.utc).isoformat()
            incident.last_updated = incident.closed_date

        self._log_event("incident_closed", {"incident_id": incident_id})
        return incident

    def get_incident(self, incident_id: str) -> Optional[ICTIncident]:
        """Get incident by ID."""
        with self._lock:
            return self._incidents.get(incident_id)

    def get_open_incidents(self) -> List[ICTIncident]:
        """Get all open incidents."""
        with self._lock:
            return [
                i
                for i in self._incidents.values()
                if i.status not in (IncidentStatus.RESOLVED, IncidentStatus.CLOSED)
            ]

    def get_incidents_by_severity(
        self,
        severity: IncidentSeverity,
    ) -> List[ICTIncident]:
        """Get incidents by severity."""
        with self._lock:
            return [i for i in self._incidents.values() if i.severity == severity]

    # =========================================================================
    # Escalation Management
    # =========================================================================

    def create_escalation_rule(
        self,
        name: str,
        from_level: EscalationLevel,
        to_level: EscalationLevel,
        trigger_elapsed_minutes: int,
        trigger_severity: Optional[List[IncidentSeverity]] = None,
        trigger_category: Optional[List[IncidentCategory]] = None,
        notify_contacts: Optional[List[str]] = None,
        notification_method: str = "all",
    ) -> EscalationRule:
        """Create an escalation rule."""
        rule = EscalationRule(
            name=name,
            from_level=from_level,
            to_level=to_level,
            trigger_elapsed_minutes=trigger_elapsed_minutes,
            trigger_severity=trigger_severity or [],
            trigger_category=trigger_category or [],
            notify_contacts=notify_contacts or [],
            notification_method=notification_method,
        )

        with self._lock:
            self._escalation_rules[rule.rule_id] = rule

        return rule

    def escalate_incident(
        self,
        incident_id: str,
        to_level: EscalationLevel,
        reason: str = "",
        escalated_by: str = "",
    ) -> Optional[ICTIncident]:
        """Escalate an incident to a higher level."""
        with self._lock:
            if incident_id not in self._incidents:
                return None

            incident = self._incidents[incident_id]
            from_level = incident.escalation_level
            incident.escalation_level = to_level
            incident.status = IncidentStatus.INVESTIGATING
            incident.last_updated = datetime.now(timezone.utc).isoformat()

        # Notify
        if self.config.notification_callback:
            try:
                self.config.notification_callback(
                    "incident_escalated",
                    {
                        "incident_id": incident_id,
                        "from_level": from_level.value,
                        "to_level": to_level.value,
                        "reason": reason,
                    },
                )
            except Exception as e:
                logger.error(f"Escalation notification failed: {e}")

        self._log_event(
            "incident_escalated",
            {
                "incident_id": incident_id,
                "from_level": from_level.value,
                "to_level": to_level.value,
                "reason": reason,
            },
        )

        logger.warning(f"Incident {incident_id} escalated to {to_level.value}")
        return incident

    def check_escalation_needed(self, incident_id: str) -> Optional[EscalationLevel]:
        """Check if incident needs escalation based on rules."""
        with self._lock:
            if incident_id not in self._incidents:
                return None

            incident = self._incidents[incident_id]

            if incident.status in (IncidentStatus.RESOLVED, IncidentStatus.CLOSED):
                return None

            # Calculate elapsed time
            detection = datetime.fromisoformat(incident.detection_time.replace("Z", "+00:00"))
            elapsed_minutes = (datetime.now(timezone.utc) - detection).total_seconds() / 60

            for rule in self._escalation_rules.values():
                if not rule.is_active:
                    continue

                if rule.from_level != incident.escalation_level:
                    continue

                if rule.trigger_severity and incident.severity not in rule.trigger_severity:
                    continue

                if elapsed_minutes >= rule.trigger_elapsed_minutes:
                    return rule.to_level

        return None

    # =========================================================================
    # Crisis Management
    # =========================================================================

    def activate_crisis(
        self,
        title: str,
        trigger_incident_ids: Optional[List[str]] = None,
        trigger_reason: str = "",
        crisis_commander: str = "",
        crisis_team: Optional[List[str]] = None,
    ) -> CrisisEvent:
        """
        Activate crisis management.

        Per Article 11, entities must have crisis management plans.

        Args:
            title: Crisis title
            trigger_incident_ids: Triggering incident IDs
            trigger_reason: Reason for activation
            crisis_commander: Crisis commander
            crisis_team: Crisis management team

        Returns:
            Activated CrisisEvent
        """
        crisis = CrisisEvent(
            title=title,
            status=CrisisStatus.ACTIVATED,
            trigger_incident_ids=trigger_incident_ids or [],
            trigger_reason=trigger_reason,
            activated_time=datetime.now(timezone.utc).isoformat(),
            crisis_commander=crisis_commander,
            crisis_team=crisis_team or [],
        )

        with self._lock:
            self._crises[crisis.crisis_id] = crisis

        # Notify
        if self.config.notification_callback:
            try:
                self.config.notification_callback(
                    "crisis_activated",
                    {
                        "crisis_id": crisis.crisis_id,
                        "title": title,
                        "commander": crisis_commander,
                    },
                )
            except Exception as e:
                logger.error(f"Crisis notification failed: {e}")

        self._log_event(
            "crisis_activated",
            {
                "crisis_id": crisis.crisis_id,
                "title": title,
            },
        )

        logger.critical(f"CRISIS ACTIVATED: {crisis.crisis_id} - {title}")
        return crisis

    def add_crisis_action(
        self,
        crisis_id: str,
        action_description: str,
        action_by: str,
    ) -> Optional[CrisisEvent]:
        """Add an action to a crisis."""
        with self._lock:
            if crisis_id not in self._crises:
                return None

            crisis = self._crises[crisis_id]
            crisis.actions_taken.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action": action_description,
                    "by": action_by,
                }
            )

        return crisis

    def add_crisis_decision(
        self,
        crisis_id: str,
        decision: str,
        decided_by: str,
        rationale: str = "",
    ) -> Optional[CrisisEvent]:
        """Add a decision to a crisis."""
        with self._lock:
            if crisis_id not in self._crises:
                return None

            crisis = self._crises[crisis_id]
            crisis.decisions_made.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "decision": decision,
                    "decided_by": decided_by,
                    "rationale": rationale,
                }
            )

        return crisis

    def deactivate_crisis(
        self,
        crisis_id: str,
        deactivated_by: str,
        resolution_summary: str = "",
    ) -> Optional[CrisisEvent]:
        """Deactivate a crisis."""
        with self._lock:
            if crisis_id not in self._crises:
                return None

            crisis = self._crises[crisis_id]
            crisis.status = CrisisStatus.DEACTIVATED
            crisis.deactivated_time = datetime.now(timezone.utc).isoformat()
            crisis.deactivated_by = deactivated_by
            crisis.resolution_summary = resolution_summary

        self._log_event(
            "crisis_deactivated",
            {
                "crisis_id": crisis_id,
                "resolution_summary": resolution_summary,
            },
        )

        logger.info(f"Crisis deactivated: {crisis_id}")
        return crisis

    def get_active_crisis(self) -> Optional[CrisisEvent]:
        """Get active crisis if any."""
        with self._lock:
            for crisis in self._crises.values():
                if crisis.status == CrisisStatus.ACTIVATED:
                    return crisis
        return None

    def _consider_crisis_activation(self, incident: ICTIncident) -> None:
        """Consider if crisis should be activated for an incident."""
        if incident.severity == IncidentSeverity.P1_CRITICAL:
            logger.warning(f"P1 incident {incident.incident_id} may require crisis activation")

    # =========================================================================
    # Recovery Management
    # =========================================================================

    def create_recovery_action(
        self,
        incident_id: str,
        action_type: str,
        description: str,
        target_system: str = "",
        target_function: str = "",
        verification_steps: Optional[List[str]] = None,
        rollback_procedure: str = "",
    ) -> RecoveryAction:
        """Create a recovery action for an incident."""
        action = RecoveryAction(
            incident_id=incident_id,
            action_type=action_type,
            description=description,
            target_system=target_system,
            target_function=target_function,
            verification_steps=verification_steps or [],
            rollback_procedure=rollback_procedure,
            rollback_available=bool(rollback_procedure),
        )

        with self._lock:
            self._recovery_actions[action.action_id] = action

            # Update incident
            if incident_id in self._incidents:
                self._incidents[incident_id].recovery_actions.append(description)

        return action

    def start_recovery_action(
        self,
        action_id: str,
        executed_by: str,
    ) -> Optional[RecoveryAction]:
        """Start a recovery action."""
        with self._lock:
            if action_id not in self._recovery_actions:
                return None

            action = self._recovery_actions[action_id]
            action.status = RecoveryStatus.IN_PROGRESS
            action.started_time = datetime.now(timezone.utc).isoformat()
            action.executed_by = executed_by

        return action

    def complete_recovery_action(
        self,
        action_id: str,
        success: bool,
        verified_by: str = "",
    ) -> Optional[RecoveryAction]:
        """Complete a recovery action."""
        with self._lock:
            if action_id not in self._recovery_actions:
                return None

            action = self._recovery_actions[action_id]
            action.status = RecoveryStatus.COMPLETED if success else RecoveryStatus.FAILED
            action.completed_time = datetime.now(timezone.utc).isoformat()

            if verified_by:
                action.verification_status = "verified"
                action.verified_by = verified_by
                action.verified_time = datetime.now(timezone.utc).isoformat()

        return action

    def rollback_recovery_action(
        self,
        action_id: str,
        rollback_reason: str = "",
    ) -> Optional[RecoveryAction]:
        """Rollback a recovery action."""
        with self._lock:
            if action_id not in self._recovery_actions:
                return None

            action = self._recovery_actions[action_id]

            if not action.rollback_available:
                logger.error(f"Rollback not available for action {action_id}")
                return None

            action.status = RecoveryStatus.ROLLED_BACK
            action.rollback_executed = True

        self._log_event(
            "recovery_rolled_back",
            {
                "action_id": action_id,
                "reason": rollback_reason,
            },
        )

        return action

    # =========================================================================
    # Post-Incident Review
    # =========================================================================

    def complete_post_incident_review(
        self,
        incident_id: str,
        lessons_learned: Optional[List[str]] = None,
        improvement_actions: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[ICTIncident]:
        """Complete post-incident review for an incident."""
        with self._lock:
            if incident_id not in self._incidents:
                return None

            incident = self._incidents[incident_id]
            incident.post_incident_review_completed = True
            incident.lessons_learned = lessons_learned or []
            incident.improvement_actions = improvement_actions or []
            incident.status = IncidentStatus.POST_INCIDENT
            incident.last_updated = datetime.now(timezone.utc).isoformat()

        self._log_event(
            "post_incident_review_completed",
            {
                "incident_id": incident_id,
                "lessons_count": len(lessons_learned or []),
                "improvements_count": len(improvement_actions or []),
            },
        )

        return incident

    def get_incidents_pending_review(self) -> List[ICTIncident]:
        """Get incidents pending post-incident review."""
        with self._lock:
            return [
                i
                for i in self._incidents.values()
                if i.status == IncidentStatus.RESOLVED and not i.post_incident_review_completed
            ]

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_response_summary(self) -> Dict[str, Any]:
        """Get response and recovery summary."""
        with self._lock:
            all_incidents = list(self._incidents.values())
            open_incidents = self.get_open_incidents()

            # Calculate metrics
            resolved_incidents = [i for i in all_incidents if i.resolution_time]

            avg_response_time = None
            avg_resolution_time = None

            if resolved_incidents:
                response_times = [
                    i.time_to_respond_minutes
                    for i in resolved_incidents
                    if i.time_to_respond_minutes is not None
                ]
                resolution_times = [
                    i.time_to_resolve_minutes
                    for i in resolved_incidents
                    if i.time_to_resolve_minutes is not None
                ]

                if response_times:
                    avg_response_time = sum(response_times) / len(response_times)
                if resolution_times:
                    avg_resolution_time = sum(resolution_times) / len(resolution_times)

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "incidents": {
                    "total": len(all_incidents),
                    "open": len(open_incidents),
                    "by_severity": {
                        s.value: sum(1 for i in all_incidents if i.severity == s)
                        for s in IncidentSeverity
                    },
                    "by_status": {
                        s.value: sum(1 for i in all_incidents if i.status == s)
                        for s in IncidentStatus
                    },
                    "major_incidents": sum(1 for i in all_incidents if i.is_major_incident),
                    "pending_review": len(self.get_incidents_pending_review()),
                },
                "metrics": {
                    "avg_response_time_minutes": avg_response_time,
                    "avg_resolution_time_minutes": avg_resolution_time,
                },
                "escalation_rules": {
                    "total": len(self._escalation_rules),
                    "active": sum(1 for r in self._escalation_rules.values() if r.is_active),
                },
                "crises": {
                    "total": len(self._crises),
                    "active": 1 if self.get_active_crisis() else 0,
                },
                "recovery_actions": {
                    "total": len(self._recovery_actions),
                    "in_progress": sum(
                        1
                        for a in self._recovery_actions.values()
                        if a.status == RecoveryStatus.IN_PROGRESS
                    ),
                },
            }

    def export_response_report(self) -> Dict[str, Any]:
        """Export complete response and recovery report."""
        with self._lock:
            return {
                "export_date": datetime.now(timezone.utc).isoformat(),
                "article_reference": "Article 11",
                "summary": self.get_response_summary(),
                "incidents": [asdict(i) for i in self._incidents.values()],
                "procedures": [asdict(p) for p in self._procedures.values()],
                "escalation_rules": [asdict(r) for r in self._escalation_rules.values()],
                "crises": [asdict(c) for c in self._crises.values()],
                "recovery_actions": [asdict(a) for a in self._recovery_actions.values()],
            }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event."""
        if not self.config.log_all_events:
            return

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"response_events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================


def create_response_recovery(
    config: Optional[ResponseRecoveryConfig] = None,
) -> DORAResponseRecovery:
    """
    Create a DORAResponseRecovery instance.

    Args:
        config: Optional configuration

    Returns:
        Configured DORAResponseRecovery instance
    """
    return DORAResponseRecovery(config=config)
