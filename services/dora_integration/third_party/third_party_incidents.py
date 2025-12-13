"""
DORA Article 23 - Operational or Security Payment-Related Incidents
and ICT Third-Party Provider Incidents

This module implements the incident management requirements for ICT
third-party service providers as required by DORA (Regulation (EU) 2022/2554)
Article 23.

Key Requirements:
- Track ICT incidents affecting third-party service providers
- Coordinate incident response with third-party providers
- Assess impact on financial entity operations
- Ensure contractual obligations are met during incidents
- Maintain incident records for regulatory reporting

References:
- DORA Article 23: Operational or security payment-related incidents
- DORA Article 28-30: ICT third-party risk management
- CDR 2024/1773: RTS on third-party provider oversight

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


class ThirdPartyProviderType(Enum):
    """Types of ICT third-party service providers."""
    CLOUD_SERVICE_PROVIDER = "cloud_service_provider"
    SOFTWARE_VENDOR = "software_vendor"
    DATA_CENTER = "data_center"
    NETWORK_PROVIDER = "network_provider"
    SECURITY_PROVIDER = "security_provider"
    PAYMENT_PROCESSOR = "payment_processor"
    INFRASTRUCTURE_PROVIDER = "infrastructure_provider"
    MANAGED_SERVICE_PROVIDER = "managed_service_provider"
    CONSULTING_PROVIDER = "consulting_provider"
    OTHER = "other"


class ThirdPartyCriticality(Enum):
    """Criticality levels for third-party providers per DORA Article 28."""
    CRITICAL = "critical"
    IMPORTANT = "important"
    STANDARD = "standard"
    NON_CRITICAL = "non_critical"


class ThirdPartyIncidentType(Enum):
    """Types of third-party related incidents."""
    SERVICE_OUTAGE = "service_outage"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SECURITY_BREACH = "security_breach"
    DATA_BREACH = "data_breach"
    SUPPLY_CHAIN_ATTACK = "supply_chain_attack"
    VENDOR_BANKRUPTCY = "vendor_bankruptcy"
    CONTRACT_VIOLATION = "contract_violation"
    REGULATORY_COMPLIANCE_FAILURE = "regulatory_compliance_failure"
    COMMUNICATION_FAILURE = "communication_failure"
    FOURTH_PARTY_INCIDENT = "fourth_party_incident"
    GEOPOLITICAL_DISRUPTION = "geopolitical_disruption"
    NATURAL_DISASTER = "natural_disaster"


class IncidentSeverity(Enum):
    """Severity levels for third-party incidents."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IncidentStatus(Enum):
    """Status of third-party incidents."""
    DETECTED = "detected"
    CONFIRMED = "confirmed"
    INVESTIGATING = "investigating"
    MITIGATING = "mitigating"
    RESOLVED = "resolved"
    POST_INCIDENT_REVIEW = "post_incident_review"
    CLOSED = "closed"


class ContractualSLAStatus(Enum):
    """Status of contractual SLA compliance during incident."""
    COMPLIANT = "compliant"
    AT_RISK = "at_risk"
    BREACHED = "breached"
    WAIVED = "waived"
    UNDER_REVIEW = "under_review"


class EscalationLevel(Enum):
    """Escalation levels for third-party incidents."""
    OPERATIONAL = "operational"
    MANAGEMENT = "management"
    EXECUTIVE = "executive"
    BOARD = "board"
    REGULATORY = "regulatory"


class CommunicationChannel(Enum):
    """Communication channels with third-party providers."""
    EMAIL = "email"
    PHONE = "phone"
    VIDEO_CONFERENCE = "video_conference"
    INCIDENT_PORTAL = "incident_portal"
    API_STATUS = "api_status"
    DEDICATED_LINE = "dedicated_line"
    IN_PERSON = "in_person"


@dataclass
class ThirdPartyProvider:
    """ICT third-party service provider information."""
    provider_id: str
    name: str
    provider_type: ThirdPartyProviderType
    criticality: ThirdPartyCriticality
    services_provided: list[str]
    contract_reference: str
    primary_contact_name: str
    primary_contact_email: str
    primary_contact_phone: str
    incident_contact_email: Optional[str] = None
    incident_contact_phone: Optional[str] = None
    incident_portal_url: Optional[str] = None
    sla_response_time_minutes: int = 60
    sla_resolution_time_hours: int = 24
    jurisdiction: str = "EU"
    data_locations: list[str] = field(default_factory=list)
    subcontractors: list[str] = field(default_factory=list)
    last_audit_date: Optional[datetime] = None
    certification_status: dict[str, str] = field(default_factory=dict)
    is_critical_or_important: bool = False

    def __post_init__(self):
        if not self.provider_id:
            self.provider_id = f"PROVIDER-{uuid4().hex[:12].upper()}"
        # Critical or important per DORA Article 28(1)(b)
        self.is_critical_or_important = self.criticality in [
            ThirdPartyCriticality.CRITICAL,
            ThirdPartyCriticality.IMPORTANT
        ]


@dataclass
class AffectedService:
    """Service affected by third-party incident."""
    service_id: str
    service_name: str
    business_function: str
    impact_level: str
    affected_users: int
    affected_transactions_per_hour: int
    revenue_impact_per_hour: float
    is_critical_service: bool
    recovery_time_objective_hours: float
    recovery_point_objective_hours: float
    current_status: str = "degraded"
    workaround_available: bool = False
    workaround_description: Optional[str] = None


@dataclass
class SLAAssessment:
    """Assessment of SLA compliance during incident."""
    assessment_id: str
    provider_id: str
    incident_id: str
    sla_metric: str
    sla_target: str
    actual_value: str
    status: ContractualSLAStatus
    assessed_at: datetime
    breach_duration_minutes: int = 0
    financial_penalty_applicable: bool = False
    penalty_amount: float = 0.0
    notes: str = ""

    def __post_init__(self):
        if not self.assessment_id:
            self.assessment_id = f"SLA-{uuid4().hex[:12].upper()}"


@dataclass
class CommunicationRecord:
    """Record of communication with third-party provider during incident."""
    record_id: str
    incident_id: str
    provider_id: str
    channel: CommunicationChannel
    direction: str  # "inbound" or "outbound"
    timestamp: datetime
    participants: list[str]
    summary: str
    full_content: Optional[str] = None
    attachments: list[str] = field(default_factory=list)
    follow_up_required: bool = False
    follow_up_deadline: Optional[datetime] = None
    response_received: bool = False

    def __post_init__(self):
        if not self.record_id:
            self.record_id = f"COMM-{uuid4().hex[:12].upper()}"


@dataclass
class EscalationRecord:
    """Record of incident escalation."""
    escalation_id: str
    incident_id: str
    from_level: EscalationLevel
    to_level: EscalationLevel
    escalated_by: str
    escalated_at: datetime
    reason: str
    stakeholders_notified: list[str]
    acknowledgement_required: bool = True
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    def __post_init__(self):
        if not self.escalation_id:
            self.escalation_id = f"ESC-{uuid4().hex[:12].upper()}"


@dataclass
class MitigationAction:
    """Mitigation action taken during third-party incident."""
    action_id: str
    incident_id: str
    action_type: str
    description: str
    assigned_to: str
    priority: str
    created_at: datetime
    deadline: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    outcome: Optional[str] = None
    requires_provider_action: bool = False
    provider_action_status: Optional[str] = None

    def __post_init__(self):
        if not self.action_id:
            self.action_id = f"MITACTION-{uuid4().hex[:12].upper()}"


@dataclass
class ThirdPartyIncident:
    """Third-party ICT incident record."""
    incident_id: str
    provider: ThirdPartyProvider
    incident_type: ThirdPartyIncidentType
    severity: IncidentSeverity
    title: str
    description: str
    detected_at: datetime
    reported_by_provider_at: Optional[datetime] = None
    confirmed_at: Optional[datetime] = None
    status: IncidentStatus = IncidentStatus.DETECTED
    affected_services: list[AffectedService] = field(default_factory=list)
    root_cause: Optional[str] = None
    provider_incident_reference: Optional[str] = None
    is_major_incident: bool = False
    requires_regulatory_notification: bool = False
    current_escalation_level: EscalationLevel = EscalationLevel.OPERATIONAL
    business_impact_summary: Optional[str] = None
    technical_impact_summary: Optional[str] = None
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    lessons_learned: list[str] = field(default_factory=list)
    related_incidents: list[str] = field(default_factory=list)
    fourth_party_involved: bool = False
    fourth_party_name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.incident_id:
            self.incident_id = f"3P-INC-{uuid4().hex[:12].upper()}"

    @property
    def duration_hours(self) -> Optional[float]:
        """Calculate incident duration in hours."""
        end_time = self.resolved_at or datetime.utcnow()
        delta = end_time - self.detected_at
        return delta.total_seconds() / 3600

    @property
    def is_active(self) -> bool:
        """Check if incident is still active."""
        return self.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]


@dataclass
class PostIncidentReview:
    """Post-incident review for third-party incidents."""
    review_id: str
    incident_id: str
    provider_id: str
    review_date: datetime
    conducted_by: list[str]
    provider_participants: list[str]
    root_cause_analysis: str
    timeline_of_events: list[dict[str, Any]]
    what_went_well: list[str]
    what_needs_improvement: list[str]
    action_items: list[dict[str, Any]]
    sla_review: dict[str, Any]
    contract_review_needed: bool = False
    contract_changes_proposed: list[str] = field(default_factory=list)
    provider_rating_impact: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

    def __post_init__(self):
        if not self.review_id:
            self.review_id = f"PIR-{uuid4().hex[:12].upper()}"


class DORAThirdPartyIncidents:
    """
    DORA Article 23 Third-Party Incident Manager.

    Manages ICT incidents involving third-party service providers,
    ensuring compliance with DORA requirements for incident tracking,
    communication, and regulatory reporting.

    Article 23 Key Requirements:
    1. Report significant incidents to competent authorities
    2. Track incidents affecting critical/important functions
    3. Coordinate response with third-party providers
    4. Maintain incident records for regulatory purposes
    5. Conduct post-incident reviews
    """

    def __init__(
        self,
        entity_id: str,
        entity_name: str,
        default_escalation_timeout_hours: int = 4
    ):
        """
        Initialize the third-party incident manager.

        Args:
            entity_id: Unique identifier for the financial entity
            entity_name: Name of the financial entity
            default_escalation_timeout_hours: Default time before escalation
        """
        self.entity_id = entity_id
        self.entity_name = entity_name
        self.default_escalation_timeout_hours = default_escalation_timeout_hours

        # Storage
        self._providers: dict[str, ThirdPartyProvider] = {}
        self._incidents: dict[str, ThirdPartyIncident] = {}
        self._sla_assessments: dict[str, list[SLAAssessment]] = {}
        self._communications: dict[str, list[CommunicationRecord]] = {}
        self._escalations: dict[str, list[EscalationRecord]] = {}
        self._mitigation_actions: dict[str, list[MitigationAction]] = {}
        self._post_incident_reviews: dict[str, PostIncidentReview] = {}

        # Metrics
        self._metrics = {
            "total_incidents": 0,
            "active_incidents": 0,
            "incidents_by_provider": {},
            "incidents_by_type": {},
            "incidents_by_severity": {},
            "average_resolution_time_hours": 0.0,
            "sla_breach_count": 0,
            "major_incidents_count": 0
        }

        logger.info(
            f"Initialized DORAThirdPartyIncidents for entity {entity_id}"
        )

    def register_provider(
        self,
        name: str,
        provider_type: ThirdPartyProviderType,
        criticality: ThirdPartyCriticality,
        services_provided: list[str],
        contract_reference: str,
        primary_contact_name: str,
        primary_contact_email: str,
        primary_contact_phone: str,
        incident_contact_email: Optional[str] = None,
        incident_contact_phone: Optional[str] = None,
        incident_portal_url: Optional[str] = None,
        sla_response_time_minutes: int = 60,
        sla_resolution_time_hours: int = 24,
        jurisdiction: str = "EU",
        data_locations: Optional[list[str]] = None,
        subcontractors: Optional[list[str]] = None
    ) -> ThirdPartyProvider:
        """
        Register a third-party ICT service provider.

        Args:
            name: Provider name
            provider_type: Type of provider
            criticality: Criticality level per DORA
            services_provided: List of services provided
            contract_reference: Contract reference number
            primary_contact_name: Primary contact name
            primary_contact_email: Primary contact email
            primary_contact_phone: Primary contact phone
            incident_contact_email: Dedicated incident contact email
            incident_contact_phone: Dedicated incident contact phone
            incident_portal_url: URL to incident portal
            sla_response_time_minutes: SLA response time target
            sla_resolution_time_hours: SLA resolution time target
            jurisdiction: Provider jurisdiction
            data_locations: Data storage locations
            subcontractors: Known subcontractors (fourth parties)

        Returns:
            ThirdPartyProvider: Registered provider
        """
        provider = ThirdPartyProvider(
            provider_id=f"PROVIDER-{uuid4().hex[:12].upper()}",
            name=name,
            provider_type=provider_type,
            criticality=criticality,
            services_provided=services_provided,
            contract_reference=contract_reference,
            primary_contact_name=primary_contact_name,
            primary_contact_email=primary_contact_email,
            primary_contact_phone=primary_contact_phone,
            incident_contact_email=incident_contact_email or primary_contact_email,
            incident_contact_phone=incident_contact_phone or primary_contact_phone,
            incident_portal_url=incident_portal_url,
            sla_response_time_minutes=sla_response_time_minutes,
            sla_resolution_time_hours=sla_resolution_time_hours,
            jurisdiction=jurisdiction,
            data_locations=data_locations or [],
            subcontractors=subcontractors or []
        )

        self._providers[provider.provider_id] = provider

        logger.info(f"Registered third-party provider: {name} ({provider.provider_id})")

        return provider

    def report_incident(
        self,
        provider_id: str,
        incident_type: ThirdPartyIncidentType,
        severity: IncidentSeverity,
        title: str,
        description: str,
        affected_services: Optional[list[AffectedService]] = None,
        detected_at: Optional[datetime] = None,
        reported_by_provider_at: Optional[datetime] = None,
        provider_incident_reference: Optional[str] = None,
        fourth_party_involved: bool = False,
        fourth_party_name: Optional[str] = None
    ) -> ThirdPartyIncident:
        """
        Report a third-party ICT incident.

        Args:
            provider_id: ID of the third-party provider
            incident_type: Type of incident
            severity: Severity level
            title: Incident title
            description: Incident description
            affected_services: List of affected services
            detected_at: When incident was detected
            reported_by_provider_at: When provider reported the incident
            provider_incident_reference: Provider's incident reference
            fourth_party_involved: Whether fourth party is involved
            fourth_party_name: Name of fourth party if involved

        Returns:
            ThirdPartyIncident: The reported incident
        """
        provider = self._providers.get(provider_id)
        if not provider:
            raise ValueError(f"Provider {provider_id} not found")

        now = datetime.utcnow()
        incident = ThirdPartyIncident(
            incident_id=f"3P-INC-{uuid4().hex[:12].upper()}",
            provider=provider,
            incident_type=incident_type,
            severity=severity,
            title=title,
            description=description,
            detected_at=detected_at or now,
            reported_by_provider_at=reported_by_provider_at,
            affected_services=affected_services or [],
            provider_incident_reference=provider_incident_reference,
            fourth_party_involved=fourth_party_involved,
            fourth_party_name=fourth_party_name
        )

        # Determine if this is a major incident requiring regulatory notification
        incident.is_major_incident = self._assess_major_incident(incident)
        incident.requires_regulatory_notification = (
            incident.is_major_incident and provider.is_critical_or_important
        )

        # Set initial escalation level based on severity
        incident.current_escalation_level = self._determine_initial_escalation(
            severity, provider.criticality
        )

        self._incidents[incident.incident_id] = incident
        self._sla_assessments[incident.incident_id] = []
        self._communications[incident.incident_id] = []
        self._escalations[incident.incident_id] = []
        self._mitigation_actions[incident.incident_id] = []

        # Update metrics
        self._update_metrics_on_incident_creation(incident)

        logger.info(
            f"Reported third-party incident {incident.incident_id}: "
            f"{title} (Provider: {provider.name}, Severity: {severity.value})"
        )

        return incident

    def _assess_major_incident(self, incident: ThirdPartyIncident) -> bool:
        """Assess if incident qualifies as major per DORA criteria."""
        # Major incident criteria per DORA Article 18
        if incident.severity == IncidentSeverity.CRITICAL:
            return True

        # Check affected services
        if incident.affected_services:
            critical_services_affected = any(
                s.is_critical_service for s in incident.affected_services
            )
            if critical_services_affected:
                return True

            total_users_affected = sum(s.affected_users for s in incident.affected_services)
            if total_users_affected >= 5000:  # CDR 2024/1772 threshold
                return True

            total_revenue_impact = sum(s.revenue_impact_per_hour for s in incident.affected_services)
            if total_revenue_impact >= 100000:  # €100,000 threshold
                return True

        return False

    def _determine_initial_escalation(
        self,
        severity: IncidentSeverity,
        provider_criticality: ThirdPartyCriticality
    ) -> EscalationLevel:
        """Determine initial escalation level based on severity and criticality."""
        if severity == IncidentSeverity.CRITICAL:
            return EscalationLevel.EXECUTIVE
        if severity == IncidentSeverity.HIGH:
            if provider_criticality in [ThirdPartyCriticality.CRITICAL, ThirdPartyCriticality.IMPORTANT]:
                return EscalationLevel.MANAGEMENT
            return EscalationLevel.OPERATIONAL
        return EscalationLevel.OPERATIONAL

    def _update_metrics_on_incident_creation(self, incident: ThirdPartyIncident) -> None:
        """Update metrics when an incident is created."""
        self._metrics["total_incidents"] += 1
        self._metrics["active_incidents"] += 1

        provider_name = incident.provider.name
        self._metrics["incidents_by_provider"][provider_name] = \
            self._metrics["incidents_by_provider"].get(provider_name, 0) + 1

        incident_type = incident.incident_type.value
        self._metrics["incidents_by_type"][incident_type] = \
            self._metrics["incidents_by_type"].get(incident_type, 0) + 1

        severity = incident.severity.value
        self._metrics["incidents_by_severity"][severity] = \
            self._metrics["incidents_by_severity"].get(severity, 0) + 1

        if incident.is_major_incident:
            self._metrics["major_incidents_count"] += 1

    def confirm_incident(
        self,
        incident_id: str,
        confirmed_by: str,
        business_impact_summary: str,
        technical_impact_summary: str
    ) -> ThirdPartyIncident:
        """
        Confirm a reported incident after initial assessment.

        Args:
            incident_id: ID of the incident
            confirmed_by: Person confirming the incident
            business_impact_summary: Summary of business impact
            technical_impact_summary: Summary of technical impact

        Returns:
            ThirdPartyIncident: Updated incident
        """
        incident = self._incidents.get(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")

        incident.status = IncidentStatus.CONFIRMED
        incident.confirmed_at = datetime.utcnow()
        incident.business_impact_summary = business_impact_summary
        incident.technical_impact_summary = technical_impact_summary

        logger.info(f"Confirmed incident {incident_id}")

        return incident

    def update_incident_status(
        self,
        incident_id: str,
        new_status: IncidentStatus,
        updated_by: str,
        notes: Optional[str] = None
    ) -> ThirdPartyIncident:
        """
        Update incident status.

        Args:
            incident_id: ID of the incident
            new_status: New status
            updated_by: Person updating status
            notes: Optional notes

        Returns:
            ThirdPartyIncident: Updated incident
        """
        incident = self._incidents.get(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")

        old_status = incident.status
        incident.status = new_status

        if new_status == IncidentStatus.RESOLVED:
            incident.resolved_at = datetime.utcnow()
            self._metrics["active_incidents"] -= 1
            self._update_resolution_metrics(incident)
        elif new_status == IncidentStatus.CLOSED:
            incident.closed_at = datetime.utcnow()
            if old_status != IncidentStatus.RESOLVED:
                self._metrics["active_incidents"] -= 1

        logger.info(
            f"Updated incident {incident_id} status: "
            f"{old_status.value} -> {new_status.value}"
        )

        return incident

    def _update_resolution_metrics(self, incident: ThirdPartyIncident) -> None:
        """Update resolution time metrics."""
        if incident.duration_hours is not None:
            total = self._metrics["total_incidents"]
            current_avg = self._metrics["average_resolution_time_hours"]
            resolved_count = total - self._metrics["active_incidents"]
            if resolved_count > 0:
                self._metrics["average_resolution_time_hours"] = \
                    (current_avg * (resolved_count - 1) + incident.duration_hours) / resolved_count

    def assess_sla_compliance(
        self,
        incident_id: str,
        sla_metric: str,
        sla_target: str,
        actual_value: str,
        assessed_by: str
    ) -> SLAAssessment:
        """
        Assess SLA compliance during incident.

        Args:
            incident_id: ID of the incident
            sla_metric: SLA metric being assessed
            sla_target: Target value
            actual_value: Actual value
            assessed_by: Person making assessment

        Returns:
            SLAAssessment: The assessment result
        """
        incident = self._incidents.get(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")

        # Determine compliance status
        status = self._evaluate_sla_compliance(sla_metric, sla_target, actual_value)

        assessment = SLAAssessment(
            assessment_id=f"SLA-{uuid4().hex[:12].upper()}",
            provider_id=incident.provider.provider_id,
            incident_id=incident_id,
            sla_metric=sla_metric,
            sla_target=sla_target,
            actual_value=actual_value,
            status=status,
            assessed_at=datetime.utcnow()
        )

        self._sla_assessments[incident_id].append(assessment)

        if status == ContractualSLAStatus.BREACHED:
            self._metrics["sla_breach_count"] += 1

        logger.info(
            f"SLA assessment for incident {incident_id}: "
            f"{sla_metric} - {status.value}"
        )

        return assessment

    def _evaluate_sla_compliance(
        self,
        metric: str,
        target: str,
        actual: str
    ) -> ContractualSLAStatus:
        """Evaluate SLA compliance based on metric and values."""
        # Simple comparison logic - can be enhanced based on metric type
        try:
            target_num = float(target.replace("h", "").replace("m", "").replace("%", ""))
            actual_num = float(actual.replace("h", "").replace("m", "").replace("%", ""))

            if "response" in metric.lower() or "time" in metric.lower():
                # Lower is better for response/time metrics
                if actual_num <= target_num:
                    return ContractualSLAStatus.COMPLIANT
                elif actual_num <= target_num * 1.5:
                    return ContractualSLAStatus.AT_RISK
                else:
                    return ContractualSLAStatus.BREACHED
            else:
                # Higher is better for availability/performance metrics
                if actual_num >= target_num:
                    return ContractualSLAStatus.COMPLIANT
                elif actual_num >= target_num * 0.9:
                    return ContractualSLAStatus.AT_RISK
                else:
                    return ContractualSLAStatus.BREACHED
        except (ValueError, AttributeError):
            return ContractualSLAStatus.UNDER_REVIEW

    def record_communication(
        self,
        incident_id: str,
        channel: CommunicationChannel,
        direction: str,
        participants: list[str],
        summary: str,
        full_content: Optional[str] = None,
        attachments: Optional[list[str]] = None,
        follow_up_required: bool = False,
        follow_up_deadline: Optional[datetime] = None
    ) -> CommunicationRecord:
        """
        Record communication with third-party provider.

        Args:
            incident_id: ID of the incident
            channel: Communication channel used
            direction: "inbound" or "outbound"
            participants: List of participants
            summary: Communication summary
            full_content: Full content if available
            attachments: List of attachments
            follow_up_required: Whether follow-up is required
            follow_up_deadline: Deadline for follow-up

        Returns:
            CommunicationRecord: The recorded communication
        """
        incident = self._incidents.get(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")

        record = CommunicationRecord(
            record_id=f"COMM-{uuid4().hex[:12].upper()}",
            incident_id=incident_id,
            provider_id=incident.provider.provider_id,
            channel=channel,
            direction=direction,
            timestamp=datetime.utcnow(),
            participants=participants,
            summary=summary,
            full_content=full_content,
            attachments=attachments or [],
            follow_up_required=follow_up_required,
            follow_up_deadline=follow_up_deadline
        )

        self._communications[incident_id].append(record)

        logger.info(
            f"Recorded {direction} communication for incident {incident_id} "
            f"via {channel.value}"
        )

        return record

    def escalate_incident(
        self,
        incident_id: str,
        to_level: EscalationLevel,
        escalated_by: str,
        reason: str,
        stakeholders_to_notify: list[str]
    ) -> EscalationRecord:
        """
        Escalate incident to a higher level.

        Args:
            incident_id: ID of the incident
            to_level: Target escalation level
            escalated_by: Person escalating
            reason: Reason for escalation
            stakeholders_to_notify: List of stakeholders to notify

        Returns:
            EscalationRecord: The escalation record
        """
        incident = self._incidents.get(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")

        record = EscalationRecord(
            escalation_id=f"ESC-{uuid4().hex[:12].upper()}",
            incident_id=incident_id,
            from_level=incident.current_escalation_level,
            to_level=to_level,
            escalated_by=escalated_by,
            escalated_at=datetime.utcnow(),
            reason=reason,
            stakeholders_notified=stakeholders_to_notify
        )

        incident.current_escalation_level = to_level
        self._escalations[incident_id].append(record)

        logger.info(
            f"Escalated incident {incident_id} from "
            f"{record.from_level.value} to {to_level.value}"
        )

        return record

    def create_mitigation_action(
        self,
        incident_id: str,
        action_type: str,
        description: str,
        assigned_to: str,
        priority: str,
        deadline: Optional[datetime] = None,
        requires_provider_action: bool = False
    ) -> MitigationAction:
        """
        Create a mitigation action for the incident.

        Args:
            incident_id: ID of the incident
            action_type: Type of action
            description: Action description
            assigned_to: Person assigned
            priority: Priority level
            deadline: Action deadline
            requires_provider_action: Whether provider action is needed

        Returns:
            MitigationAction: The created action
        """
        incident = self._incidents.get(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")

        action = MitigationAction(
            action_id=f"MITACTION-{uuid4().hex[:12].upper()}",
            incident_id=incident_id,
            action_type=action_type,
            description=description,
            assigned_to=assigned_to,
            priority=priority,
            created_at=datetime.utcnow(),
            deadline=deadline,
            requires_provider_action=requires_provider_action
        )

        self._mitigation_actions[incident_id].append(action)

        # Update incident status to mitigating if investigating
        if incident.status == IncidentStatus.INVESTIGATING:
            incident.status = IncidentStatus.MITIGATING

        logger.info(
            f"Created mitigation action {action.action_id} "
            f"for incident {incident_id}"
        )

        return action

    def complete_mitigation_action(
        self,
        incident_id: str,
        action_id: str,
        completed_by: str,
        outcome: str
    ) -> MitigationAction:
        """
        Complete a mitigation action.

        Args:
            incident_id: ID of the incident
            action_id: ID of the action
            completed_by: Person completing action
            outcome: Outcome description

        Returns:
            MitigationAction: Updated action
        """
        actions = self._mitigation_actions.get(incident_id, [])
        action = next((a for a in actions if a.action_id == action_id), None)

        if not action:
            raise ValueError(f"Action {action_id} not found for incident {incident_id}")

        action.status = "completed"
        action.completed_at = datetime.utcnow()
        action.outcome = outcome

        logger.info(f"Completed mitigation action {action_id}")

        return action

    def set_root_cause(
        self,
        incident_id: str,
        root_cause: str,
        determined_by: str
    ) -> ThirdPartyIncident:
        """
        Set the root cause of an incident.

        Args:
            incident_id: ID of the incident
            root_cause: Root cause description
            determined_by: Person determining root cause

        Returns:
            ThirdPartyIncident: Updated incident
        """
        incident = self._incidents.get(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")

        incident.root_cause = root_cause

        logger.info(f"Set root cause for incident {incident_id}")

        return incident

    def create_post_incident_review(
        self,
        incident_id: str,
        conducted_by: list[str],
        provider_participants: list[str],
        root_cause_analysis: str,
        timeline_of_events: list[dict[str, Any]],
        what_went_well: list[str],
        what_needs_improvement: list[str],
        action_items: list[dict[str, Any]],
        sla_review: dict[str, Any],
        contract_review_needed: bool = False,
        contract_changes_proposed: Optional[list[str]] = None
    ) -> PostIncidentReview:
        """
        Create a post-incident review.

        Args:
            incident_id: ID of the incident
            conducted_by: List of reviewers
            provider_participants: Provider participants
            root_cause_analysis: Root cause analysis
            timeline_of_events: Event timeline
            what_went_well: What went well
            what_needs_improvement: Areas for improvement
            action_items: Action items from review
            sla_review: SLA review results
            contract_review_needed: Whether contract review needed
            contract_changes_proposed: Proposed contract changes

        Returns:
            PostIncidentReview: The created review
        """
        incident = self._incidents.get(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")

        review = PostIncidentReview(
            review_id=f"PIR-{uuid4().hex[:12].upper()}",
            incident_id=incident_id,
            provider_id=incident.provider.provider_id,
            review_date=datetime.utcnow(),
            conducted_by=conducted_by,
            provider_participants=provider_participants,
            root_cause_analysis=root_cause_analysis,
            timeline_of_events=timeline_of_events,
            what_went_well=what_went_well,
            what_needs_improvement=what_needs_improvement,
            action_items=action_items,
            sla_review=sla_review,
            contract_review_needed=contract_review_needed,
            contract_changes_proposed=contract_changes_proposed or []
        )

        self._post_incident_reviews[incident_id] = review

        # Update incident status
        incident.status = IncidentStatus.POST_INCIDENT_REVIEW

        logger.info(f"Created post-incident review for incident {incident_id}")

        return review

    def add_lessons_learned(
        self,
        incident_id: str,
        lessons: list[str]
    ) -> ThirdPartyIncident:
        """
        Add lessons learned to an incident.

        Args:
            incident_id: ID of the incident
            lessons: List of lessons learned

        Returns:
            ThirdPartyIncident: Updated incident
        """
        incident = self._incidents.get(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")

        incident.lessons_learned.extend(lessons)

        logger.info(f"Added {len(lessons)} lessons learned to incident {incident_id}")

        return incident

    def get_incident(self, incident_id: str) -> Optional[ThirdPartyIncident]:
        """Get incident by ID."""
        return self._incidents.get(incident_id)

    def get_provider(self, provider_id: str) -> Optional[ThirdPartyProvider]:
        """Get provider by ID."""
        return self._providers.get(provider_id)

    def get_active_incidents(self) -> list[ThirdPartyIncident]:
        """Get all active incidents."""
        return [i for i in self._incidents.values() if i.is_active]

    def get_incidents_by_provider(self, provider_id: str) -> list[ThirdPartyIncident]:
        """Get all incidents for a provider."""
        return [
            i for i in self._incidents.values()
            if i.provider.provider_id == provider_id
        ]

    def get_major_incidents(self) -> list[ThirdPartyIncident]:
        """Get all major incidents."""
        return [i for i in self._incidents.values() if i.is_major_incident]

    def get_incidents_requiring_notification(self) -> list[ThirdPartyIncident]:
        """Get incidents requiring regulatory notification."""
        return [
            i for i in self._incidents.values()
            if i.requires_regulatory_notification and i.is_active
        ]

    def get_critical_providers(self) -> list[ThirdPartyProvider]:
        """Get all critical/important providers per DORA Article 28."""
        return [p for p in self._providers.values() if p.is_critical_or_important]

    def get_communications(self, incident_id: str) -> list[CommunicationRecord]:
        """Get all communications for an incident."""
        return self._communications.get(incident_id, [])

    def get_escalations(self, incident_id: str) -> list[EscalationRecord]:
        """Get all escalations for an incident."""
        return self._escalations.get(incident_id, [])

    def get_mitigation_actions(self, incident_id: str) -> list[MitigationAction]:
        """Get all mitigation actions for an incident."""
        return self._mitigation_actions.get(incident_id, [])

    def get_sla_assessments(self, incident_id: str) -> list[SLAAssessment]:
        """Get all SLA assessments for an incident."""
        return self._sla_assessments.get(incident_id, [])

    def get_post_incident_review(self, incident_id: str) -> Optional[PostIncidentReview]:
        """Get post-incident review for an incident."""
        return self._post_incident_reviews.get(incident_id)

    def get_metrics(self) -> dict[str, Any]:
        """Get third-party incident metrics."""
        return {
            **self._metrics,
            "total_providers": len(self._providers),
            "critical_providers": len(self.get_critical_providers()),
            "pending_notifications": len(self.get_incidents_requiring_notification())
        }

    def generate_incident_report(self, incident_id: str) -> dict[str, Any]:
        """
        Generate a comprehensive incident report.

        Args:
            incident_id: ID of the incident

        Returns:
            dict: Comprehensive incident report
        """
        incident = self._incidents.get(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")

        communications = self.get_communications(incident_id)
        escalations = self.get_escalations(incident_id)
        actions = self.get_mitigation_actions(incident_id)
        sla_assessments = self.get_sla_assessments(incident_id)
        review = self.get_post_incident_review(incident_id)

        return {
            "incident": {
                "id": incident.incident_id,
                "title": incident.title,
                "type": incident.incident_type.value,
                "severity": incident.severity.value,
                "status": incident.status.value,
                "is_major": incident.is_major_incident,
                "requires_notification": incident.requires_regulatory_notification,
                "detected_at": incident.detected_at.isoformat(),
                "resolved_at": incident.resolved_at.isoformat() if incident.resolved_at else None,
                "duration_hours": incident.duration_hours,
                "root_cause": incident.root_cause
            },
            "provider": {
                "id": incident.provider.provider_id,
                "name": incident.provider.name,
                "type": incident.provider.provider_type.value,
                "criticality": incident.provider.criticality.value,
                "is_critical_or_important": incident.provider.is_critical_or_important
            },
            "affected_services": [
                {
                    "name": s.service_name,
                    "impact_level": s.impact_level,
                    "is_critical": s.is_critical_service,
                    "affected_users": s.affected_users
                }
                for s in incident.affected_services
            ],
            "communications": {
                "total": len(communications),
                "records": [
                    {
                        "channel": c.channel.value,
                        "direction": c.direction,
                        "timestamp": c.timestamp.isoformat(),
                        "summary": c.summary
                    }
                    for c in communications
                ]
            },
            "escalations": {
                "current_level": incident.current_escalation_level.value,
                "history": [
                    {
                        "from": e.from_level.value,
                        "to": e.to_level.value,
                        "timestamp": e.escalated_at.isoformat(),
                        "reason": e.reason
                    }
                    for e in escalations
                ]
            },
            "mitigation_actions": {
                "total": len(actions),
                "completed": len([a for a in actions if a.status == "completed"]),
                "actions": [
                    {
                        "id": a.action_id,
                        "type": a.action_type,
                        "status": a.status,
                        "outcome": a.outcome
                    }
                    for a in actions
                ]
            },
            "sla_compliance": {
                "assessments": [
                    {
                        "metric": s.sla_metric,
                        "target": s.sla_target,
                        "actual": s.actual_value,
                        "status": s.status.value
                    }
                    for s in sla_assessments
                ],
                "breaches": len([s for s in sla_assessments if s.status == ContractualSLAStatus.BREACHED])
            },
            "post_incident_review": {
                "completed": review is not None,
                "review_date": review.review_date.isoformat() if review else None,
                "action_items_count": len(review.action_items) if review else 0,
                "contract_review_needed": review.contract_review_needed if review else False
            } if review else None,
            "lessons_learned": incident.lessons_learned
        }


def create_third_party_incidents(
    entity_id: str,
    entity_name: str,
    default_escalation_timeout_hours: int = 4
) -> DORAThirdPartyIncidents:
    """
    Factory function to create a DORAThirdPartyIncidents instance.

    Args:
        entity_id: Unique identifier for the financial entity
        entity_name: Name of the financial entity
        default_escalation_timeout_hours: Default time before escalation

    Returns:
        DORAThirdPartyIncidents: Configured instance
    """
    return DORAThirdPartyIncidents(
        entity_id=entity_id,
        entity_name=entity_name,
        default_escalation_timeout_hours=default_escalation_timeout_hours
    )
