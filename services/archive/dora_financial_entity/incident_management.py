# -*- coding: utf-8 -*-
"""
DORA ICT Incident Management Module (Article 17).

Regulation (EU) 2022/2554 Article 17 defines ICT-related incident management:
    - Process to detect, manage, and notify ICT-related incidents
    - Early warning indicators
    - Allocation of roles and responsibilities
    - Recording and classification of incidents
    - Escalation procedures

This module provides:
    1. DORAIncidentManagement - Main incident management orchestrator
    2. ICTEvent - ICT event detection and recording
    3. DORAIncident - DORA-compliant incident tracking
    4. IncidentWorkflow - Incident lifecycle management
    5. EarlyWarningSystem - Early warning indicator monitoring

References:
    - Article 17 DORA: https://www.digital-operational-resilience-act.com/Article_17.html
    - RTS on Incident Classification: CDR 2024/1772
    - RTS on Incident Reporting: CDR 2025/301
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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class ICTEventType(Enum):
    """Types of ICT events per DORA Article 17."""
    SYSTEM_FAILURE = "system_failure"
    SECURITY_BREACH = "security_breach"
    DATA_BREACH = "data_breach"
    NETWORK_DISRUPTION = "network_disruption"
    SERVICE_DEGRADATION = "service_degradation"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    MALWARE_DETECTION = "malware_detection"
    DENIAL_OF_SERVICE = "denial_of_service"
    CONFIGURATION_ERROR = "configuration_error"
    HARDWARE_FAILURE = "hardware_failure"
    SOFTWARE_FAILURE = "software_failure"
    THIRD_PARTY_FAILURE = "third_party_failure"
    HUMAN_ERROR = "human_error"
    NATURAL_DISASTER = "natural_disaster"
    OTHER = "other"


class IncidentPhase(Enum):
    """Incident management phases per Article 17."""
    DETECTION = "detection"              # Early warning, automated detection
    RECORDING = "recording"              # Record and log incident
    CLASSIFICATION = "classification"    # Classify per Article 18 criteria
    ESCALATION = "escalation"            # Internal escalation
    NOTIFICATION = "notification"        # Regulatory notification if major
    INVESTIGATION = "investigation"      # Root cause analysis
    RESOLUTION = "resolution"            # Corrective actions
    CLOSURE = "closure"                  # Post-incident review


class IncidentPriority(Enum):
    """Incident priority levels."""
    P1_CRITICAL = "P1"      # Immediate action required
    P2_HIGH = "P2"          # Action within 4 hours
    P3_MEDIUM = "P3"        # Action within 24 hours
    P4_LOW = "P4"           # Action within 72 hours


class IncidentStatus(Enum):
    """Incident status tracking."""
    NEW = "new"
    DETECTED = "detected"
    RECORDED = "recorded"
    CLASSIFIED = "classified"
    ESCALATED = "escalated"
    NOTIFIED = "notified"
    UNDER_INVESTIGATION = "under_investigation"
    RESOLUTION_IN_PROGRESS = "resolution_in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    REOPENED = "reopened"


class EscalationLevel(Enum):
    """Escalation levels per Article 17(3)(d)."""
    L1_OPERATIONAL = "L1"   # Operational team
    L2_TECHNICAL = "L2"     # Technical management
    L3_MANAGEMENT = "L3"    # Senior management
    L4_EXECUTIVE = "L4"     # Executive/Board level
    L5_REGULATORY = "L5"    # Regulatory authorities


class EarlyWarningType(Enum):
    """Early warning indicator types per Article 17(3)(a)."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    SECURITY_ALERT = "security_alert"
    CAPACITY_WARNING = "capacity_warning"
    AVAILABILITY_WARNING = "availability_warning"
    ERROR_RATE_SPIKE = "error_rate_spike"
    LATENCY_SPIKE = "latency_spike"
    THIRD_PARTY_ALERT = "third_party_alert"
    THREAT_INTELLIGENCE = "threat_intelligence"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ICTEvent:
    """
    ICT Event per DORA Article 17.

    Represents a detected event that may indicate an incident.
    """
    event_id: str = ""
    event_type: ICTEventType = ICTEventType.OTHER
    source_system: str = ""
    source_component: str = ""

    # Detection
    detected_at: str = ""
    detected_by: str = ""  # automated, manual, third_party
    detection_method: str = ""

    # Event details
    title: str = ""
    description: str = ""
    raw_data: Dict[str, Any] = field(default_factory=dict)

    # Affected resources
    affected_systems: List[str] = field(default_factory=list)
    affected_services: List[str] = field(default_factory=list)
    affected_data_types: List[str] = field(default_factory=list)

    # Indicators
    indicators_of_compromise: List[str] = field(default_factory=list)

    # Processing
    is_incident: Optional[bool] = None
    incident_id: str = ""
    processed_at: str = ""
    processed_by: str = ""

    # Metadata
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"EVT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        if not self.detected_at:
            self.detected_at = datetime.now(timezone.utc).isoformat()


@dataclass
class EarlyWarningIndicator:
    """
    Early warning indicator per Article 17(3)(a).

    Monitors system health and raises alerts before incidents occur.
    """
    indicator_id: str = ""
    indicator_type: EarlyWarningType = EarlyWarningType.ANOMALOUS_BEHAVIOR
    name: str = ""
    description: str = ""

    # Monitoring configuration
    monitored_system: str = ""
    monitored_metric: str = ""
    baseline_value: float = 0.0
    current_value: float = 0.0

    # Thresholds
    warning_threshold: float = 0.0
    critical_threshold: float = 0.0
    threshold_direction: str = "above"  # "above", "below", "deviation"

    # Alert state
    is_triggered: bool = False
    trigger_time: Optional[str] = None
    trigger_value: float = 0.0
    severity: str = "warning"  # "warning", "critical"

    # Actions
    auto_escalate: bool = False
    escalation_level: EscalationLevel = EscalationLevel.L1_OPERATIONAL
    notification_recipients: List[str] = field(default_factory=list)

    # Status
    is_active: bool = True
    last_checked: str = ""
    check_interval_seconds: int = 60

    def __post_init__(self):
        if not self.indicator_id:
            self.indicator_id = f"EWI-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class DORAIncident:
    """
    DORA-compliant incident per Article 17.

    Comprehensive incident tracking with all DORA-required fields.
    """
    incident_id: str = ""

    # Source event
    source_event_id: str = ""
    source_events: List[str] = field(default_factory=list)

    # Classification (will be detailed by incident_classification module)
    incident_type: ICTEventType = ICTEventType.OTHER
    priority: IncidentPriority = IncidentPriority.P3_MEDIUM
    is_major: bool = False
    classification_criteria: Dict[str, Any] = field(default_factory=dict)

    # Status tracking
    current_phase: IncidentPhase = IncidentPhase.DETECTION
    status: IncidentStatus = IncidentStatus.NEW

    # Timeline
    detected_at: str = ""
    recorded_at: str = ""
    classified_at: str = ""
    escalated_at: str = ""
    notified_at: str = ""
    investigation_started_at: str = ""
    resolution_started_at: str = ""
    resolved_at: str = ""
    closed_at: str = ""

    # Description
    title: str = ""
    description: str = ""
    technical_details: str = ""

    # Impact assessment
    affected_systems: List[str] = field(default_factory=list)
    affected_services: List[str] = field(default_factory=list)
    affected_clients_count: int = 0
    affected_client_types: List[str] = field(default_factory=list)  # retail, professional, etc.
    geographic_spread: List[str] = field(default_factory=list)  # country codes
    data_compromised: bool = False
    data_types_affected: List[str] = field(default_factory=list)
    critical_functions_affected: List[str] = field(default_factory=list)

    # Duration tracking
    duration_hours: float = 0.0
    service_downtime_hours: float = 0.0

    # Economic impact
    estimated_economic_impact_eur: float = 0.0
    direct_costs_eur: float = 0.0
    indirect_costs_eur: float = 0.0
    recovery_costs_eur: float = 0.0

    # Escalation
    escalation_level: EscalationLevel = EscalationLevel.L1_OPERATIONAL
    escalation_history: List[Dict[str, Any]] = field(default_factory=list)

    # Assignment
    assigned_team: str = ""
    incident_commander: str = ""
    assigned_to: List[str] = field(default_factory=list)

    # Investigation
    root_cause: str = ""
    root_cause_category: str = ""  # human_error, software_bug, hardware_failure, etc.
    contributing_factors: List[str] = field(default_factory=list)
    investigation_notes: str = ""

    # Actions
    immediate_actions: List[Dict[str, Any]] = field(default_factory=list)
    containment_actions: List[Dict[str, Any]] = field(default_factory=list)
    corrective_actions: List[Dict[str, Any]] = field(default_factory=list)
    preventive_actions: List[Dict[str, Any]] = field(default_factory=list)

    # Regulatory reporting (Article 19)
    requires_notification: bool = False
    notification_submitted: bool = False
    notification_deadline: str = ""
    notification_report_ids: List[str] = field(default_factory=list)

    # Related items
    related_incidents: List[str] = field(default_factory=list)
    related_changes: List[str] = field(default_factory=list)
    related_vulnerabilities: List[str] = field(default_factory=list)

    # Post-incident
    lessons_learned: List[str] = field(default_factory=list)
    post_incident_review_id: str = ""

    # Metadata
    created_at: str = ""
    updated_at: str = ""
    created_by: str = ""
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.incident_id:
            self.incident_id = f"INC-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.detected_at:
            self.detected_at = self.created_at
        if not self.recorded_at:
            self.recorded_at = self.created_at

    @property
    def is_active(self) -> bool:
        """Check if incident is still active."""
        return self.status not in (IncidentStatus.RESOLVED, IncidentStatus.CLOSED)

    @property
    def time_since_detection(self) -> timedelta:
        """Calculate time since detection."""
        detected = datetime.fromisoformat(self.detected_at.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) - detected

    @property
    def notification_overdue(self) -> bool:
        """Check if notification is overdue."""
        if not self.requires_notification or self.notification_submitted:
            return False
        if not self.notification_deadline:
            return False
        deadline = datetime.fromisoformat(self.notification_deadline.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) > deadline


@dataclass
class IncidentAction:
    """
    Incident action tracking.
    """
    action_id: str = ""
    incident_id: str = ""
    action_type: str = ""  # immediate, containment, corrective, preventive
    description: str = ""
    assigned_to: str = ""
    due_date: str = ""

    # Status
    status: str = "pending"  # pending, in_progress, completed, cancelled
    started_at: str = ""
    completed_at: str = ""
    completion_notes: str = ""

    # Verification
    verified: bool = False
    verified_by: str = ""
    verified_at: str = ""

    def __post_init__(self):
        if not self.action_id:
            self.action_id = f"ACT-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class EscalationRule:
    """
    Escalation rule per Article 17(3)(d).
    """
    rule_id: str = ""
    name: str = ""
    description: str = ""

    # Trigger conditions
    trigger_priority: List[IncidentPriority] = field(default_factory=list)
    trigger_incident_types: List[ICTEventType] = field(default_factory=list)
    trigger_time_minutes: int = 0  # Time without resolution
    trigger_conditions: Dict[str, Any] = field(default_factory=dict)

    # Escalation target
    from_level: EscalationLevel = EscalationLevel.L1_OPERATIONAL
    to_level: EscalationLevel = EscalationLevel.L2_TECHNICAL

    # Notifications
    notify_roles: List[str] = field(default_factory=list)
    notify_individuals: List[str] = field(default_factory=list)
    notification_template: str = ""

    # Status
    is_active: bool = True

    def __post_init__(self):
        if not self.rule_id:
            self.rule_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class IncidentManagementConfig:
    """Configuration for ICT Incident Management."""
    # Workflow settings
    auto_classify: bool = True
    auto_escalate: bool = True
    auto_notify: bool = False  # Manual approval for notifications

    # Timeframes (per CDR 2025/301)
    initial_notification_hours_from_classification: int = 4
    initial_notification_hours_from_detection: int = 24
    intermediate_report_hours: int = 72
    final_report_days: int = 30

    # Escalation timeframes
    p1_escalation_minutes: int = 15
    p2_escalation_minutes: int = 60
    p3_escalation_minutes: int = 240
    p4_escalation_minutes: int = 480

    # Early warning
    enable_early_warning: bool = True
    early_warning_check_interval_seconds: int = 60

    # Retention
    incident_retention_days: int = 1825  # 5 years per DORA
    event_retention_days: int = 365

    # Logging
    log_all_events: bool = True
    log_path: str = "logs/dora/incident_management"

    # Callbacks
    escalation_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    notification_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# Main Class Implementation
# =============================================================================

class DORAIncidentManagement:
    """
    DORA Article 17 compliant ICT Incident Management.

    Provides:
    - ICT event detection and recording
    - Incident lifecycle management
    - Escalation procedures
    - Early warning monitoring
    - Regulatory notification triggers

    Usage:
        config = IncidentManagementConfig()
        incident_mgmt = DORAIncidentManagement(config)

        # Detect and record an event
        event = incident_mgmt.detect_event(
            event_type=ICTEventType.SYSTEM_FAILURE,
            source_system="trading_engine",
            title="Trading Engine Unresponsive",
            description="Trading engine not responding to requests",
        )

        # Create incident from event
        incident = incident_mgmt.create_incident_from_event(event)

        # Classify incident (triggers notification workflow if major)
        incident_mgmt.classify_incident(
            incident.incident_id,
            priority=IncidentPriority.P1_CRITICAL,
            is_major=True,
        )
    """

    def __init__(self, config: Optional[IncidentManagementConfig] = None):
        """Initialize ICT Incident Management."""
        self.config = config or IncidentManagementConfig()

        # Data stores
        self._events: Dict[str, ICTEvent] = {}
        self._incidents: Dict[str, DORAIncident] = {}
        self._actions: Dict[str, IncidentAction] = {}
        self._early_warnings: Dict[str, EarlyWarningIndicator] = {}
        self._escalation_rules: Dict[str, EscalationRule] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        # Initialize default escalation rules
        self._init_default_escalation_rules()

        logger.info("DORAIncidentManagement initialized")

    # =========================================================================
    # Event Management (Article 17(2))
    # =========================================================================

    def detect_event(
        self,
        event_type: ICTEventType,
        source_system: str,
        title: str,
        description: str,
        source_component: str = "",
        detected_by: str = "automated",
        detection_method: str = "",
        affected_systems: Optional[List[str]] = None,
        affected_services: Optional[List[str]] = None,
        raw_data: Optional[Dict[str, Any]] = None,
        indicators_of_compromise: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> ICTEvent:
        """
        Detect and record an ICT event per Article 17(2).

        Args:
            event_type: Type of ICT event
            source_system: System where event originated
            title: Event title
            description: Event description
            source_component: Specific component
            detected_by: Detection source
            detection_method: How event was detected
            affected_systems: List of affected systems
            affected_services: List of affected services
            raw_data: Raw event data
            indicators_of_compromise: IOCs if security event
            tags: Event tags

        Returns:
            Created ICTEvent
        """
        event = ICTEvent(
            event_type=event_type,
            source_system=source_system,
            source_component=source_component,
            detected_by=detected_by,
            detection_method=detection_method,
            title=title,
            description=description,
            raw_data=raw_data or {},
            affected_systems=affected_systems or [],
            affected_services=affected_services or [],
            indicators_of_compromise=indicators_of_compromise or [],
            tags=tags or [],
        )

        with self._lock:
            self._events[event.event_id] = event

        self._log_event("event_detected", {
            "event_id": event.event_id,
            "event_type": event_type.value,
            "source_system": source_system,
            "title": title,
        })

        logger.info(f"ICT Event detected: {event.event_id} - {title}")
        return event

    def process_event(
        self,
        event_id: str,
        is_incident: bool,
        processed_by: str = "automated",
        notes: str = "",
    ) -> ICTEvent:
        """
        Process an event and determine if it constitutes an incident.

        Args:
            event_id: Event ID
            is_incident: Whether event is classified as incident
            processed_by: Who processed the event
            notes: Processing notes

        Returns:
            Updated ICTEvent
        """
        with self._lock:
            if event_id not in self._events:
                raise ValueError(f"Event {event_id} not found")

            event = self._events[event_id]
            event.is_incident = is_incident
            event.processed_at = datetime.now(timezone.utc).isoformat()
            event.processed_by = processed_by

        self._log_event("event_processed", {
            "event_id": event_id,
            "is_incident": is_incident,
            "processed_by": processed_by,
        })

        return event

    def get_event(self, event_id: str) -> Optional[ICTEvent]:
        """Get event by ID."""
        with self._lock:
            return self._events.get(event_id)

    def get_unprocessed_events(self) -> List[ICTEvent]:
        """Get events not yet processed."""
        with self._lock:
            return [e for e in self._events.values() if e.is_incident is None]

    # =========================================================================
    # Incident Management (Article 17(3))
    # =========================================================================

    def create_incident_from_event(
        self,
        event: ICTEvent,
        title: Optional[str] = None,
        description: Optional[str] = None,
        assigned_team: str = "",
        incident_commander: str = "",
    ) -> DORAIncident:
        """
        Create an incident from a detected event.

        Args:
            event: Source ICTEvent
            title: Optional incident title override
            description: Optional description override
            assigned_team: Assigned team
            incident_commander: Incident commander

        Returns:
            Created DORAIncident
        """
        incident = DORAIncident(
            source_event_id=event.event_id,
            source_events=[event.event_id],
            incident_type=event.event_type,
            detected_at=event.detected_at,
            title=title or event.title,
            description=description or event.description,
            affected_systems=event.affected_systems.copy(),
            affected_services=event.affected_services.copy(),
            assigned_team=assigned_team,
            incident_commander=incident_commander,
            current_phase=IncidentPhase.RECORDING,
            status=IncidentStatus.RECORDED,
        )

        with self._lock:
            self._incidents[incident.incident_id] = incident

            # Link event to incident
            event.is_incident = True
            event.incident_id = incident.incident_id

        self._log_event("incident_created", {
            "incident_id": incident.incident_id,
            "source_event_id": event.event_id,
            "incident_type": incident.incident_type.value,
        })

        logger.info(f"Incident created: {incident.incident_id}")
        return incident

    def create_incident(
        self,
        incident_type: ICTEventType,
        title: str,
        description: str,
        affected_systems: Optional[List[str]] = None,
        affected_services: Optional[List[str]] = None,
        priority: IncidentPriority = IncidentPriority.P3_MEDIUM,
        assigned_team: str = "",
        incident_commander: str = "",
        created_by: str = "",
    ) -> DORAIncident:
        """
        Create an incident directly without a source event.

        Args:
            incident_type: Type of incident
            title: Incident title
            description: Incident description
            affected_systems: Affected systems
            affected_services: Affected services
            priority: Incident priority
            assigned_team: Assigned team
            incident_commander: Incident commander
            created_by: Creator

        Returns:
            Created DORAIncident
        """
        incident = DORAIncident(
            incident_type=incident_type,
            title=title,
            description=description,
            affected_systems=affected_systems or [],
            affected_services=affected_services or [],
            priority=priority,
            assigned_team=assigned_team,
            incident_commander=incident_commander,
            created_by=created_by,
            current_phase=IncidentPhase.RECORDING,
            status=IncidentStatus.RECORDED,
        )

        with self._lock:
            self._incidents[incident.incident_id] = incident

        self._log_event("incident_created_direct", {
            "incident_id": incident.incident_id,
            "incident_type": incident_type.value,
            "priority": priority.value,
        })

        logger.info(f"Incident created: {incident.incident_id}")
        return incident

    def classify_incident(
        self,
        incident_id: str,
        priority: IncidentPriority,
        is_major: bool = False,
        classification_criteria: Optional[Dict[str, Any]] = None,
        affected_clients_count: int = 0,
        geographic_spread: Optional[List[str]] = None,
        data_compromised: bool = False,
        critical_functions_affected: Optional[List[str]] = None,
        estimated_economic_impact_eur: float = 0.0,
    ) -> DORAIncident:
        """
        Classify an incident per Article 17/18 criteria.

        If classified as major, triggers notification workflow.

        Args:
            incident_id: Incident ID
            priority: Incident priority
            is_major: Whether this is a major incident
            classification_criteria: Classification criteria details
            affected_clients_count: Number of affected clients
            geographic_spread: Affected countries
            data_compromised: Whether data was compromised
            critical_functions_affected: Affected critical functions
            estimated_economic_impact_eur: Estimated economic impact

        Returns:
            Updated DORAIncident
        """
        with self._lock:
            if incident_id not in self._incidents:
                raise ValueError(f"Incident {incident_id} not found")

            incident = self._incidents[incident_id]
            now = datetime.now(timezone.utc).isoformat()

            incident.priority = priority
            incident.is_major = is_major
            incident.classification_criteria = classification_criteria or {}
            incident.affected_clients_count = affected_clients_count
            incident.geographic_spread = geographic_spread or []
            incident.data_compromised = data_compromised
            incident.critical_functions_affected = critical_functions_affected or []
            incident.estimated_economic_impact_eur = estimated_economic_impact_eur
            incident.classified_at = now
            incident.current_phase = IncidentPhase.CLASSIFICATION
            incident.status = IncidentStatus.CLASSIFIED
            incident.updated_at = now

            # If major incident, set notification requirements
            if is_major:
                incident.requires_notification = True

                # Calculate notification deadline per CDR 2025/301
                detected = datetime.fromisoformat(
                    incident.detected_at.replace("Z", "+00:00")
                )
                classified = datetime.fromisoformat(
                    incident.classified_at.replace("Z", "+00:00")
                )

                deadline_from_classification = classified + timedelta(
                    hours=self.config.initial_notification_hours_from_classification
                )
                deadline_from_detection = detected + timedelta(
                    hours=self.config.initial_notification_hours_from_detection
                )

                # Whichever is earlier
                deadline = min(deadline_from_classification, deadline_from_detection)
                incident.notification_deadline = deadline.isoformat()

                logger.warning(
                    f"Major incident classified: {incident_id}. "
                    f"Notification deadline: {incident.notification_deadline}"
                )

        self._log_event("incident_classified", {
            "incident_id": incident_id,
            "priority": priority.value,
            "is_major": is_major,
            "notification_deadline": incident.notification_deadline if is_major else None,
        })

        return incident

    def escalate_incident(
        self,
        incident_id: str,
        to_level: EscalationLevel,
        reason: str,
        escalated_by: str = "",
    ) -> DORAIncident:
        """
        Escalate an incident per Article 17(3)(d).

        Args:
            incident_id: Incident ID
            to_level: Target escalation level
            reason: Reason for escalation
            escalated_by: Who escalated

        Returns:
            Updated DORAIncident
        """
        with self._lock:
            if incident_id not in self._incidents:
                raise ValueError(f"Incident {incident_id} not found")

            incident = self._incidents[incident_id]
            now = datetime.now(timezone.utc).isoformat()

            escalation_record = {
                "from_level": incident.escalation_level.value,
                "to_level": to_level.value,
                "reason": reason,
                "escalated_by": escalated_by,
                "escalated_at": now,
            }
            incident.escalation_history.append(escalation_record)
            incident.escalation_level = to_level

            if incident.current_phase == IncidentPhase.CLASSIFICATION:
                incident.current_phase = IncidentPhase.ESCALATION
                incident.status = IncidentStatus.ESCALATED

            incident.escalated_at = now
            incident.updated_at = now

        # Trigger escalation callback
        if self.config.escalation_callback:
            try:
                self.config.escalation_callback(
                    "escalation",
                    {
                        "incident_id": incident_id,
                        **escalation_record,
                    }
                )
            except Exception as e:
                logger.error(f"Escalation callback failed: {e}")

        self._log_event("incident_escalated", {
            "incident_id": incident_id,
            "to_level": to_level.value,
            "reason": reason,
        })

        logger.info(f"Incident {incident_id} escalated to {to_level.value}")
        return incident

    def start_investigation(
        self,
        incident_id: str,
        assigned_to: Optional[List[str]] = None,
        notes: str = "",
    ) -> DORAIncident:
        """
        Start investigation of an incident.

        Args:
            incident_id: Incident ID
            assigned_to: Investigators assigned
            notes: Initial investigation notes

        Returns:
            Updated DORAIncident
        """
        with self._lock:
            if incident_id not in self._incidents:
                raise ValueError(f"Incident {incident_id} not found")

            incident = self._incidents[incident_id]
            now = datetime.now(timezone.utc).isoformat()

            incident.current_phase = IncidentPhase.INVESTIGATION
            incident.status = IncidentStatus.UNDER_INVESTIGATION
            incident.investigation_started_at = now
            incident.investigation_notes = notes
            incident.updated_at = now

            if assigned_to:
                incident.assigned_to = assigned_to

        self._log_event("investigation_started", {
            "incident_id": incident_id,
            "assigned_to": assigned_to,
        })

        return incident

    def set_root_cause(
        self,
        incident_id: str,
        root_cause: str,
        root_cause_category: str,
        contributing_factors: Optional[List[str]] = None,
    ) -> DORAIncident:
        """
        Set root cause for an incident.

        Args:
            incident_id: Incident ID
            root_cause: Root cause description
            root_cause_category: Root cause category
            contributing_factors: Contributing factors

        Returns:
            Updated DORAIncident
        """
        with self._lock:
            if incident_id not in self._incidents:
                raise ValueError(f"Incident {incident_id} not found")

            incident = self._incidents[incident_id]
            incident.root_cause = root_cause
            incident.root_cause_category = root_cause_category
            incident.contributing_factors = contributing_factors or []
            incident.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("root_cause_identified", {
            "incident_id": incident_id,
            "root_cause_category": root_cause_category,
        })

        return incident

    def add_action(
        self,
        incident_id: str,
        action_type: str,
        description: str,
        assigned_to: str = "",
        due_date: Optional[str] = None,
    ) -> IncidentAction:
        """
        Add an action to an incident.

        Args:
            incident_id: Incident ID
            action_type: Type of action (immediate, containment, corrective, preventive)
            description: Action description
            assigned_to: Person assigned
            due_date: Due date

        Returns:
            Created IncidentAction
        """
        action = IncidentAction(
            incident_id=incident_id,
            action_type=action_type,
            description=description,
            assigned_to=assigned_to,
            due_date=due_date or "",
        )

        with self._lock:
            if incident_id not in self._incidents:
                raise ValueError(f"Incident {incident_id} not found")

            self._actions[action.action_id] = action

            incident = self._incidents[incident_id]
            action_record = {
                "action_id": action.action_id,
                "description": description,
                "assigned_to": assigned_to,
                "due_date": due_date,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            if action_type == "immediate":
                incident.immediate_actions.append(action_record)
            elif action_type == "containment":
                incident.containment_actions.append(action_record)
            elif action_type == "corrective":
                incident.corrective_actions.append(action_record)
            elif action_type == "preventive":
                incident.preventive_actions.append(action_record)

            incident.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("action_added", {
            "incident_id": incident_id,
            "action_id": action.action_id,
            "action_type": action_type,
        })

        return action

    def complete_action(
        self,
        action_id: str,
        completion_notes: str = "",
        verified_by: str = "",
    ) -> IncidentAction:
        """
        Complete an incident action.

        Args:
            action_id: Action ID
            completion_notes: Completion notes
            verified_by: Person who verified completion

        Returns:
            Updated IncidentAction
        """
        with self._lock:
            if action_id not in self._actions:
                raise ValueError(f"Action {action_id} not found")

            action = self._actions[action_id]
            now = datetime.now(timezone.utc).isoformat()
            action.status = "completed"
            action.completed_at = now
            action.completion_notes = completion_notes

            if verified_by:
                action.verified = True
                action.verified_by = verified_by
                action.verified_at = now

        return action

    def start_resolution(
        self,
        incident_id: str,
        notes: str = "",
    ) -> DORAIncident:
        """
        Start resolution phase for an incident.

        Args:
            incident_id: Incident ID
            notes: Resolution notes

        Returns:
            Updated DORAIncident
        """
        with self._lock:
            if incident_id not in self._incidents:
                raise ValueError(f"Incident {incident_id} not found")

            incident = self._incidents[incident_id]
            now = datetime.now(timezone.utc).isoformat()

            incident.current_phase = IncidentPhase.RESOLUTION
            incident.status = IncidentStatus.RESOLUTION_IN_PROGRESS
            incident.resolution_started_at = now
            incident.updated_at = now

        self._log_event("resolution_started", {"incident_id": incident_id})
        return incident

    def resolve_incident(
        self,
        incident_id: str,
        resolution_details: str,
        final_impact_assessment: Optional[Dict[str, Any]] = None,
    ) -> DORAIncident:
        """
        Resolve an incident.

        Args:
            incident_id: Incident ID
            resolution_details: Details of resolution
            final_impact_assessment: Final impact assessment

        Returns:
            Updated DORAIncident
        """
        with self._lock:
            if incident_id not in self._incidents:
                raise ValueError(f"Incident {incident_id} not found")

            incident = self._incidents[incident_id]
            now = datetime.now(timezone.utc).isoformat()

            incident.status = IncidentStatus.RESOLVED
            incident.resolved_at = now
            incident.updated_at = now

            # Calculate duration
            detected = datetime.fromisoformat(
                incident.detected_at.replace("Z", "+00:00")
            )
            resolved = datetime.fromisoformat(now.replace("Z", "+00:00"))
            incident.duration_hours = (resolved - detected).total_seconds() / 3600

            # Update impact if provided
            if final_impact_assessment:
                if "economic_impact_eur" in final_impact_assessment:
                    incident.estimated_economic_impact_eur = final_impact_assessment["economic_impact_eur"]
                if "affected_clients_count" in final_impact_assessment:
                    incident.affected_clients_count = final_impact_assessment["affected_clients_count"]

        self._log_event("incident_resolved", {
            "incident_id": incident_id,
            "duration_hours": incident.duration_hours,
        })

        logger.info(f"Incident resolved: {incident_id}")
        return incident

    def close_incident(
        self,
        incident_id: str,
        lessons_learned: Optional[List[str]] = None,
        post_incident_review_id: str = "",
    ) -> DORAIncident:
        """
        Close an incident after post-incident review.

        Args:
            incident_id: Incident ID
            lessons_learned: Lessons learned from incident
            post_incident_review_id: Reference to post-incident review

        Returns:
            Updated DORAIncident
        """
        with self._lock:
            if incident_id not in self._incidents:
                raise ValueError(f"Incident {incident_id} not found")

            incident = self._incidents[incident_id]

            if incident.status != IncidentStatus.RESOLVED:
                raise ValueError(f"Incident {incident_id} must be resolved before closing")

            now = datetime.now(timezone.utc).isoformat()
            incident.current_phase = IncidentPhase.CLOSURE
            incident.status = IncidentStatus.CLOSED
            incident.closed_at = now
            incident.lessons_learned = lessons_learned or []
            incident.post_incident_review_id = post_incident_review_id
            incident.updated_at = now

        self._log_event("incident_closed", {
            "incident_id": incident_id,
            "lessons_learned": lessons_learned,
        })

        logger.info(f"Incident closed: {incident_id}")
        return incident

    # =========================================================================
    # Notification Management (Article 19 trigger)
    # =========================================================================

    def mark_notification_submitted(
        self,
        incident_id: str,
        report_id: str,
    ) -> DORAIncident:
        """
        Mark that notification has been submitted for an incident.

        Args:
            incident_id: Incident ID
            report_id: Report ID from incident_reporting module

        Returns:
            Updated DORAIncident
        """
        with self._lock:
            if incident_id not in self._incidents:
                raise ValueError(f"Incident {incident_id} not found")

            incident = self._incidents[incident_id]
            incident.notification_submitted = True
            incident.notified_at = datetime.now(timezone.utc).isoformat()
            incident.notification_report_ids.append(report_id)

            if incident.current_phase == IncidentPhase.ESCALATION:
                incident.current_phase = IncidentPhase.NOTIFICATION
                incident.status = IncidentStatus.NOTIFIED

            incident.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("notification_submitted", {
            "incident_id": incident_id,
            "report_id": report_id,
        })

        return incident

    def get_incidents_requiring_notification(self) -> List[DORAIncident]:
        """Get incidents that require notification but haven't been notified."""
        with self._lock:
            return [
                i for i in self._incidents.values()
                if i.requires_notification and not i.notification_submitted
            ]

    def get_overdue_notifications(self) -> List[DORAIncident]:
        """Get incidents with overdue notifications."""
        with self._lock:
            return [
                i for i in self._incidents.values()
                if i.notification_overdue
            ]

    # =========================================================================
    # Early Warning System (Article 17(3)(a))
    # =========================================================================

    def register_early_warning_indicator(
        self,
        indicator_type: EarlyWarningType,
        name: str,
        monitored_system: str,
        monitored_metric: str,
        baseline_value: float,
        warning_threshold: float,
        critical_threshold: float,
        threshold_direction: str = "above",
        auto_escalate: bool = False,
        escalation_level: EscalationLevel = EscalationLevel.L1_OPERATIONAL,
        notification_recipients: Optional[List[str]] = None,
        check_interval_seconds: int = 60,
    ) -> EarlyWarningIndicator:
        """
        Register an early warning indicator per Article 17(3)(a).

        Args:
            indicator_type: Type of early warning
            name: Indicator name
            monitored_system: System being monitored
            monitored_metric: Metric being monitored
            baseline_value: Normal baseline value
            warning_threshold: Warning threshold
            critical_threshold: Critical threshold
            threshold_direction: Direction for threshold comparison
            auto_escalate: Whether to auto-escalate on trigger
            escalation_level: Target escalation level
            notification_recipients: Recipients for notifications
            check_interval_seconds: Check interval

        Returns:
            Created EarlyWarningIndicator
        """
        indicator = EarlyWarningIndicator(
            indicator_type=indicator_type,
            name=name,
            monitored_system=monitored_system,
            monitored_metric=monitored_metric,
            baseline_value=baseline_value,
            current_value=baseline_value,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            threshold_direction=threshold_direction,
            auto_escalate=auto_escalate,
            escalation_level=escalation_level,
            notification_recipients=notification_recipients or [],
            check_interval_seconds=check_interval_seconds,
        )

        with self._lock:
            self._early_warnings[indicator.indicator_id] = indicator

        self._log_event("early_warning_registered", {
            "indicator_id": indicator.indicator_id,
            "name": name,
            "monitored_system": monitored_system,
        })

        return indicator

    def update_early_warning_value(
        self,
        indicator_id: str,
        current_value: float,
    ) -> Tuple[EarlyWarningIndicator, Optional[ICTEvent]]:
        """
        Update early warning indicator value and check thresholds.

        Args:
            indicator_id: Indicator ID
            current_value: Current observed value

        Returns:
            Tuple of (updated indicator, generated event if triggered)
        """
        event = None

        with self._lock:
            if indicator_id not in self._early_warnings:
                raise ValueError(f"Indicator {indicator_id} not found")

            indicator = self._early_warnings[indicator_id]
            indicator.current_value = current_value
            indicator.last_checked = datetime.now(timezone.utc).isoformat()

            # Check thresholds
            triggered = False
            severity = "none"

            if indicator.threshold_direction == "above":
                if current_value >= indicator.critical_threshold:
                    triggered = True
                    severity = "critical"
                elif current_value >= indicator.warning_threshold:
                    triggered = True
                    severity = "warning"
            elif indicator.threshold_direction == "below":
                if current_value <= indicator.critical_threshold:
                    triggered = True
                    severity = "critical"
                elif current_value <= indicator.warning_threshold:
                    triggered = True
                    severity = "warning"
            elif indicator.threshold_direction == "deviation":
                deviation = abs(current_value - indicator.baseline_value)
                if deviation >= indicator.critical_threshold:
                    triggered = True
                    severity = "critical"
                elif deviation >= indicator.warning_threshold:
                    triggered = True
                    severity = "warning"

            # Handle trigger
            if triggered and not indicator.is_triggered:
                indicator.is_triggered = True
                indicator.trigger_time = datetime.now(timezone.utc).isoformat()
                indicator.trigger_value = current_value
                indicator.severity = severity

                # Generate event
                event = self.detect_event(
                    event_type=ICTEventType.SERVICE_DEGRADATION,
                    source_system=indicator.monitored_system,
                    title=f"Early Warning: {indicator.name}",
                    description=(
                        f"Early warning indicator triggered. "
                        f"Metric: {indicator.monitored_metric}, "
                        f"Value: {current_value}, "
                        f"Threshold: {indicator.critical_threshold if severity == 'critical' else indicator.warning_threshold}"
                    ),
                    detection_method="early_warning_system",
                    tags=["early_warning", severity, indicator.indicator_type.value],
                )
            elif not triggered and indicator.is_triggered:
                # Clear trigger
                indicator.is_triggered = False
                indicator.trigger_time = None
                indicator.trigger_value = 0.0
                indicator.severity = "none"

        return indicator, event

    def get_triggered_early_warnings(self) -> List[EarlyWarningIndicator]:
        """Get all currently triggered early warning indicators."""
        with self._lock:
            return [
                i for i in self._early_warnings.values()
                if i.is_triggered and i.is_active
            ]

    def get_early_warning_indicator(
        self,
        indicator_id: str,
    ) -> Optional[EarlyWarningIndicator]:
        """Get early warning indicator by ID."""
        with self._lock:
            return self._early_warnings.get(indicator_id)

    # =========================================================================
    # Escalation Rules (Article 17(3)(d))
    # =========================================================================

    def _init_default_escalation_rules(self) -> None:
        """Initialize default escalation rules."""
        default_rules = [
            EscalationRule(
                name="P1 Auto-Escalation",
                description="Auto-escalate P1 incidents if not resolved within 15 minutes",
                trigger_priority=[IncidentPriority.P1_CRITICAL],
                trigger_time_minutes=15,
                from_level=EscalationLevel.L1_OPERATIONAL,
                to_level=EscalationLevel.L2_TECHNICAL,
            ),
            EscalationRule(
                name="P1 Management Escalation",
                description="Escalate P1 to management if not resolved within 1 hour",
                trigger_priority=[IncidentPriority.P1_CRITICAL],
                trigger_time_minutes=60,
                from_level=EscalationLevel.L2_TECHNICAL,
                to_level=EscalationLevel.L3_MANAGEMENT,
            ),
            EscalationRule(
                name="Major Incident Executive Escalation",
                description="Escalate major incidents to executive level",
                trigger_priority=[IncidentPriority.P1_CRITICAL],
                trigger_conditions={"is_major": True},
                trigger_time_minutes=120,
                from_level=EscalationLevel.L3_MANAGEMENT,
                to_level=EscalationLevel.L4_EXECUTIVE,
            ),
            EscalationRule(
                name="P2 Auto-Escalation",
                description="Auto-escalate P2 incidents if not resolved within 1 hour",
                trigger_priority=[IncidentPriority.P2_HIGH],
                trigger_time_minutes=60,
                from_level=EscalationLevel.L1_OPERATIONAL,
                to_level=EscalationLevel.L2_TECHNICAL,
            ),
        ]

        for rule in default_rules:
            self._escalation_rules[rule.rule_id] = rule

    def add_escalation_rule(
        self,
        rule: EscalationRule,
    ) -> EscalationRule:
        """Add a custom escalation rule."""
        with self._lock:
            self._escalation_rules[rule.rule_id] = rule
        return rule

    def check_escalation_rules(self) -> List[Tuple[DORAIncident, EscalationRule]]:
        """
        Check which incidents need escalation based on rules.

        Returns:
            List of (incident, rule) tuples that should be escalated
        """
        escalations_needed = []
        now = datetime.now(timezone.utc)

        with self._lock:
            for incident in self._incidents.values():
                if not incident.is_active:
                    continue

                for rule in self._escalation_rules.values():
                    if not rule.is_active:
                        continue

                    # Check if incident matches rule criteria
                    if rule.trigger_priority and incident.priority not in rule.trigger_priority:
                        continue

                    if rule.trigger_incident_types and incident.incident_type not in rule.trigger_incident_types:
                        continue

                    if incident.escalation_level != rule.from_level:
                        continue

                    # Check additional conditions
                    if rule.trigger_conditions:
                        if rule.trigger_conditions.get("is_major") and not incident.is_major:
                            continue

                    # Check time threshold
                    if rule.trigger_time_minutes > 0:
                        created = datetime.fromisoformat(
                            incident.created_at.replace("Z", "+00:00")
                        )
                        elapsed_minutes = (now - created).total_seconds() / 60

                        if elapsed_minutes >= rule.trigger_time_minutes:
                            escalations_needed.append((incident, rule))

        return escalations_needed

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_incident(self, incident_id: str) -> Optional[DORAIncident]:
        """Get incident by ID."""
        with self._lock:
            return self._incidents.get(incident_id)

    def get_active_incidents(self) -> List[DORAIncident]:
        """Get all active incidents."""
        with self._lock:
            return [i for i in self._incidents.values() if i.is_active]

    def get_major_incidents(self) -> List[DORAIncident]:
        """Get all major incidents."""
        with self._lock:
            return [i for i in self._incidents.values() if i.is_major]

    def get_incidents_by_priority(
        self,
        priority: IncidentPriority,
    ) -> List[DORAIncident]:
        """Get incidents by priority."""
        with self._lock:
            return [i for i in self._incidents.values() if i.priority == priority]

    def get_incidents_by_status(
        self,
        status: IncidentStatus,
    ) -> List[DORAIncident]:
        """Get incidents by status."""
        with self._lock:
            return [i for i in self._incidents.values() if i.status == status]

    def get_incidents_by_type(
        self,
        incident_type: ICTEventType,
    ) -> List[DORAIncident]:
        """Get incidents by type."""
        with self._lock:
            return [i for i in self._incidents.values() if i.incident_type == incident_type]

    def get_recent_incidents(
        self,
        hours: int = 24,
    ) -> List[DORAIncident]:
        """Get incidents from the last N hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        with self._lock:
            return [
                i for i in self._incidents.values()
                if datetime.fromisoformat(
                    i.created_at.replace("Z", "+00:00")
                ) >= cutoff
            ]

    def get_action(self, action_id: str) -> Optional[IncidentAction]:
        """Get action by ID."""
        with self._lock:
            return self._actions.get(action_id)

    def get_actions_for_incident(
        self,
        incident_id: str,
    ) -> List[IncidentAction]:
        """Get all actions for an incident."""
        with self._lock:
            return [
                a for a in self._actions.values()
                if a.incident_id == incident_id
            ]

    def get_pending_actions(self) -> List[IncidentAction]:
        """Get all pending actions."""
        with self._lock:
            return [
                a for a in self._actions.values()
                if a.status == "pending"
            ]

    # =========================================================================
    # Reporting and Statistics
    # =========================================================================

    def get_incident_statistics(
        self,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get incident statistics for a period.

        Args:
            period_start: Start of period (ISO format)
            period_end: End of period (ISO format)

        Returns:
            Statistics dictionary
        """
        if not period_end:
            period_end = datetime.now(timezone.utc).isoformat()
        if not period_start:
            period_start = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

        start_dt = datetime.fromisoformat(period_start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(period_end.replace("Z", "+00:00"))

        with self._lock:
            period_incidents = [
                i for i in self._incidents.values()
                if start_dt <= datetime.fromisoformat(
                    i.created_at.replace("Z", "+00:00")
                ) <= end_dt
            ]

        # Calculate statistics
        total = len(period_incidents)
        major = sum(1 for i in period_incidents if i.is_major)
        resolved = sum(1 for i in period_incidents if i.status == IncidentStatus.RESOLVED)
        closed = sum(1 for i in period_incidents if i.status == IncidentStatus.CLOSED)

        by_priority = {}
        for priority in IncidentPriority:
            by_priority[priority.value] = sum(
                1 for i in period_incidents if i.priority == priority
            )

        by_type = {}
        for inc_type in ICTEventType:
            count = sum(1 for i in period_incidents if i.incident_type == inc_type)
            if count > 0:
                by_type[inc_type.value] = count

        # Calculate average resolution time
        resolved_incidents = [
            i for i in period_incidents
            if i.resolved_at
        ]

        avg_resolution_hours = 0.0
        if resolved_incidents:
            total_hours = sum(i.duration_hours for i in resolved_incidents)
            avg_resolution_hours = total_hours / len(resolved_incidents)

        return {
            "period_start": period_start,
            "period_end": period_end,
            "total_incidents": total,
            "major_incidents": major,
            "resolved": resolved,
            "closed": closed,
            "open": total - resolved - closed,
            "by_priority": by_priority,
            "by_type": by_type,
            "average_resolution_hours": round(avg_resolution_hours, 2),
            "notification_submitted": sum(
                1 for i in period_incidents if i.notification_submitted
            ),
            "notification_pending": sum(
                1 for i in period_incidents
                if i.requires_notification and not i.notification_submitted
            ),
        }

    def export_incident_data(
        self,
        incident_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Export incident data for reporting.

        Args:
            incident_id: Optional specific incident ID

        Returns:
            Export data dictionary
        """
        with self._lock:
            if incident_id:
                if incident_id not in self._incidents:
                    raise ValueError(f"Incident {incident_id} not found")
                incidents = [asdict(self._incidents[incident_id])]
                actions = [
                    asdict(a) for a in self._actions.values()
                    if a.incident_id == incident_id
                ]
            else:
                incidents = [asdict(i) for i in self._incidents.values()]
                actions = [asdict(a) for a in self._actions.values()]

        return {
            "export_date": datetime.now(timezone.utc).isoformat(),
            "article_reference": "Article 17",
            "incidents": incidents,
            "actions": actions,
            "statistics": self.get_incident_statistics(),
        }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an incident management event."""
        if not self.config.log_all_events:
            return

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"incident_mgmt_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log incident event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_incident_management(
    config: Optional[IncidentManagementConfig] = None,
) -> DORAIncidentManagement:
    """
    Create a DORAIncidentManagement instance.

    Args:
        config: Optional configuration

    Returns:
        Configured DORAIncidentManagement instance
    """
    return DORAIncidentManagement(config=config)


def get_incident_phases() -> List[IncidentPhase]:
    """
    Get list of incident phases per Article 17.

    Returns:
        List of IncidentPhase enums
    """
    return list(IncidentPhase)


def get_escalation_levels() -> List[EscalationLevel]:
    """
    Get list of escalation levels.

    Returns:
        List of EscalationLevel enums
    """
    return list(EscalationLevel)
