# -*- coding: utf-8 -*-
"""
DORA ICT Incident Reporting Module (Article 19).

Regulation (EU) 2022/2554 Article 19 defines incident reporting requirements:
    - Initial notification within 4 hours from classification / 24 hours from detection
    - Intermediate report within 72 hours
    - Final report within 1 month
    - Reporting to designated competent authority

USAGE CONTEXT:
    This module implements FINANCIAL ENTITY obligations to report to NCAs.

    For ICT THIRD-PARTY PROVIDERS (our primary use case):
    - We do NOT report directly to NCAs (except if designated CTPP)
    - We notify CLIENTS, who then report to their NCAs
    - Use `client_incident_notification.py` for client notifications
    - This module can be used to understand client reporting requirements

    Integration with client_incident_notification.py:
    >>> from services.dora.client_incident_notification import DORAClientNotification
    >>> from services.dora.incident_reporting import DORAIncidentReporter
    >>>
    >>> # Create incident for clients
    >>> client_notifier.create_incident(...)
    >>>
    >>> # For reference: understand what client must report
    >>> report = incident_reporter.create_initial_notification(...)

This module implements reporting requirements from:
    - DORA Article 19-21
    - Commission Delegated Regulation (CDR) 2025/301 - RTS on content and time limits
    - Commission Implementing Regulation (CIR) 2025/302 - ITS on forms and procedures
    - Entry into force: 12 March 2025

References:
    - Article 19 DORA: https://www.digital-operational-resilience-act.com/Article_19.html
    - Article 20 DORA: https://www.digital-operational-resilience-act.com/Article_20.html
    - CDR 2025/301: RTS on major ICT-related incident reporting
    - CIR 2025/302: ITS on reporting templates and procedures

Related Modules (for ICT providers):
    - client_incident_notification.py: Client notification (PRIMARY for ICT providers)
    - audit_readiness.py: Incident documentation for audits
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
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class ReportType(Enum):
    """Report types per CDR 2025/301."""
    INITIAL_NOTIFICATION = "initial_notification"
    INTERMEDIATE_REPORT = "intermediate_report"
    FINAL_REPORT = "final_report"
    VOLUNTARY_NOTIFICATION = "voluntary_notification"  # Per Article 19(4)


class ReportStatus(Enum):
    """Report submission status."""
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class IncidentTypeCode(Enum):
    """Incident type codes per ITS Annex."""
    CYBER_ATTACK = "CYBA"
    SYSTEM_FAILURE = "SYSF"
    EXTERNAL_EVENT = "EXTE"
    PROCESS_FAILURE = "PROC"
    THIRD_PARTY_FAILURE = "TPFA"
    UNKNOWN = "UNKN"


class RootCauseCategory(Enum):
    """Root cause categories per ITS."""
    MALICIOUS_INTERNAL = "malicious_internal"
    MALICIOUS_EXTERNAL = "malicious_external"
    ACCIDENTAL_INTERNAL = "accidental_internal"
    ACCIDENTAL_EXTERNAL = "accidental_external"
    SYSTEM_HARDWARE = "system_hardware"
    SYSTEM_SOFTWARE = "system_software"
    NATURAL_EVENT = "natural_event"
    THIRD_PARTY = "third_party"
    UNKNOWN = "unknown"


class CompetentAuthorityType(Enum):
    """Types of competent authorities."""
    NCA_PRIMARY = "nca_primary"          # Primary national competent authority
    NCA_SECONDARY = "nca_secondary"      # Secondary (if applicable)
    ECB = "ecb"                          # For significant credit institutions
    ESA = "esa"                          # European Supervisory Authority


# =============================================================================
# Data Structures - Competent Authority
# =============================================================================

@dataclass
class CompetentAuthority:
    """
    Competent authority information for incident reporting.
    """
    authority_id: str = ""
    name: str = ""
    authority_type: CompetentAuthorityType = CompetentAuthorityType.NCA_PRIMARY
    country_code: str = ""  # ISO 3166-1 alpha-2

    # Contact information
    reporting_portal_url: str = ""
    reporting_email: str = ""
    emergency_contact: str = ""

    # Technical requirements
    supported_formats: List[str] = field(default_factory=list)  # ["XML", "JSON", "CSV"]
    api_endpoint: str = ""
    uses_central_hub: bool = False

    # Regulatory details
    regulatory_framework: str = ""  # "DORA", "NIS2", etc.
    submission_deadlines: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.authority_id:
            self.authority_id = f"NCA-{self.country_code}-{uuid.uuid4().hex[:4].upper()}"


# =============================================================================
# Data Structures - Reports per ITS Annex
# =============================================================================

@dataclass
class InitialNotificationReport:
    """
    Initial Notification per ITS Annex I (CDR 2025/301).

    Contains limited mandatory fields to minimize burden during active incident.
    Deadline: 4 hours from classification OR 24 hours from detection (whichever earlier).
    """
    report_id: str = ""
    incident_id: str = ""
    report_type: ReportType = ReportType.INITIAL_NOTIFICATION

    # Section 1: Reporting Entity Identification
    reporting_entity_lei: str = ""
    reporting_entity_name: str = ""
    reporting_entity_type: str = ""  # investment_firm, credit_institution, etc.
    contact_person_name: str = ""
    contact_person_email: str = ""
    contact_person_phone: str = ""

    # Section 2: Incident Identification
    incident_reference: str = ""  # Internal reference
    detection_datetime: str = ""
    classification_datetime: str = ""

    # Section 3: Incident Description
    incident_type_code: IncidentTypeCode = IncidentTypeCode.UNKNOWN
    brief_description: str = ""  # Max 1000 characters

    # Section 4: Affected Services (high level)
    critical_services_affected: List[str] = field(default_factory=list)
    estimated_clients_affected: int = 0

    # Section 5: Geographic Scope
    member_states_affected: List[str] = field(default_factory=list)

    # Section 6: Initial Impact Assessment
    estimated_impact_description: str = ""
    is_ongoing: bool = True

    # Section 7: Recurring Incident Flag
    is_recurring: bool = False
    related_incident_references: List[str] = field(default_factory=list)

    # Section 8: Cross-border Services
    cross_border_services_affected: bool = False

    # Metadata
    status: ReportStatus = ReportStatus.DRAFT
    created_at: str = ""
    submitted_at: str = ""
    deadline: str = ""

    def __post_init__(self):
        if not self.report_id:
            self.report_id = f"RPT-INIT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class IntermediateReport:
    """
    Intermediate Report per ITS Annex II (CDR 2025/301).

    Contains more detailed information as investigation progresses.
    Deadline: 72 hours from initial notification.
    """
    report_id: str = ""
    incident_id: str = ""
    report_type: ReportType = ReportType.INTERMEDIATE_REPORT
    initial_notification_id: str = ""  # Reference to initial notification

    # Section 1: Updated Entity Information
    reporting_entity_lei: str = ""
    reporting_entity_name: str = ""
    contact_person_name: str = ""
    contact_person_email: str = ""
    contact_person_phone: str = ""

    # Section 2: Updated Incident Description
    incident_reference: str = ""
    detailed_description: str = ""  # Max 4000 characters
    incident_type_code: IncidentTypeCode = IncidentTypeCode.UNKNOWN

    # Section 3: Detailed Impact Assessment
    affected_ict_services: List[str] = field(default_factory=list)
    affected_business_functions: List[str] = field(default_factory=list)
    affected_clients_count: int = 0
    affected_client_types: List[str] = field(default_factory=list)
    geographic_spread: List[str] = field(default_factory=list)

    # Section 4: Data Impact
    data_compromised: bool = False
    data_types_affected: List[str] = field(default_factory=list)
    records_affected: int = 0

    # Section 5: Root Cause Analysis (Preliminary)
    preliminary_root_cause: str = ""
    root_cause_category: RootCauseCategory = RootCauseCategory.UNKNOWN
    is_malicious: bool = False
    attack_vector: str = ""

    # Section 6: Actions Taken
    immediate_actions_taken: List[str] = field(default_factory=list)
    containment_actions: List[str] = field(default_factory=list)
    recovery_actions_started: List[str] = field(default_factory=list)

    # Section 7: Ongoing Response
    ongoing_actions: List[str] = field(default_factory=list)
    external_support_engaged: bool = False
    external_parties_involved: List[str] = field(default_factory=list)

    # Section 8: Timeline
    incident_start_datetime: str = ""
    service_disruption_start: str = ""
    estimated_resolution_datetime: str = ""

    # Section 9: Current Status
    is_ongoing: bool = True
    current_status_description: str = ""

    # Metadata
    status: ReportStatus = ReportStatus.DRAFT
    created_at: str = ""
    submitted_at: str = ""
    deadline: str = ""

    def __post_init__(self):
        if not self.report_id:
            self.report_id = f"RPT-INT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class FinalReport:
    """
    Final Report per ITS Annex III (CDR 2025/301).

    Complete information after incident resolution.
    Deadline: 1 month after incident resolution (or after intermediate if not resolved).
    """
    report_id: str = ""
    incident_id: str = ""
    report_type: ReportType = ReportType.FINAL_REPORT
    initial_notification_id: str = ""
    intermediate_report_id: str = ""

    # Section 1: Entity Information
    reporting_entity_lei: str = ""
    reporting_entity_name: str = ""
    contact_person_name: str = ""
    contact_person_email: str = ""
    contact_person_phone: str = ""

    # Section 2: Complete Incident Description
    incident_reference: str = ""
    incident_title: str = ""
    comprehensive_description: str = ""

    # Section 3: Resolution Status
    incident_resolved: bool = False
    resolution_datetime: str = ""
    resolution_description: str = ""

    # Section 4: Complete Timeline
    detection_datetime: str = ""
    classification_datetime: str = ""
    incident_start_datetime: str = ""
    service_impact_start: str = ""
    service_impact_end: str = ""
    incident_end_datetime: str = ""
    timeline_events: List[Dict[str, Any]] = field(default_factory=list)

    # Section 5: Final Root Cause Analysis
    final_root_cause: str = ""
    root_cause_category: RootCauseCategory = RootCauseCategory.UNKNOWN
    contributing_factors: List[str] = field(default_factory=list)

    # Section 6: Complete Impact Assessment
    total_duration_hours: float = 0.0
    service_downtime_hours: float = 0.0
    total_clients_affected: int = 0
    clients_by_type: Dict[str, int] = field(default_factory=dict)
    geographic_spread: List[str] = field(default_factory=list)

    # Section 7: Data Impact (Final)
    data_loss_confirmed: bool = False
    data_types_compromised: List[str] = field(default_factory=list)
    total_records_affected: int = 0
    individuals_notified: int = 0

    # Section 8: Economic Impact (Final)
    total_economic_impact_eur: float = 0.0
    direct_costs_eur: float = 0.0
    indirect_costs_eur: float = 0.0
    recovery_costs_eur: float = 0.0

    # Section 9: Response Effectiveness
    response_effectiveness: str = ""  # effective, partially_effective, ineffective
    response_timeline_met: bool = False
    escalation_procedures_followed: bool = False

    # Section 10: Lessons Learned
    lessons_learned: List[str] = field(default_factory=list)
    what_worked_well: List[str] = field(default_factory=list)
    areas_for_improvement: List[str] = field(default_factory=list)

    # Section 11: Remediation Measures
    remediation_measures: List[Dict[str, Any]] = field(default_factory=list)
    remediation_completion_dates: Dict[str, str] = field(default_factory=dict)

    # Section 12: Preventive Measures
    preventive_measures: List[Dict[str, Any]] = field(default_factory=list)
    preventive_implementation_dates: Dict[str, str] = field(default_factory=dict)

    # Section 13: Follow-up Actions
    follow_up_actions: List[str] = field(default_factory=list)
    follow_up_deadlines: Dict[str, str] = field(default_factory=dict)

    # Metadata
    status: ReportStatus = ReportStatus.DRAFT
    created_at: str = ""
    submitted_at: str = ""
    deadline: str = ""

    def __post_init__(self):
        if not self.report_id:
            self.report_id = f"RPT-FIN-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class ReportSubmission:
    """
    Record of report submission to competent authority.
    """
    submission_id: str = ""
    report_id: str = ""
    report_type: ReportType = ReportType.INITIAL_NOTIFICATION

    # Submission details
    authority_id: str = ""
    authority_name: str = ""
    submitted_at: str = ""
    submitted_by: str = ""
    submission_method: str = ""  # portal, email, api

    # Response
    acknowledgement_received: bool = False
    acknowledgement_datetime: str = ""
    acknowledgement_reference: str = ""

    # Feedback
    feedback_received: bool = False
    feedback_datetime: str = ""
    feedback_content: str = ""
    requires_correction: bool = False
    correction_deadline: str = ""

    def __post_init__(self):
        if not self.submission_id:
            self.submission_id = f"SUB-{uuid.uuid4().hex[:8].upper()}"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class IncidentReportingConfig:
    """Configuration for incident reporting."""
    # Deadlines per CDR 2025/301 (in hours)
    initial_notification_hours_from_classification: int = 4
    initial_notification_hours_from_detection: int = 24
    intermediate_report_hours: int = 72
    final_report_days: int = 30

    # Weekend/Holiday extension per CDR 2025/301 Art. 4
    weekend_extension_enabled: bool = True
    extend_to_noon_next_business_day: bool = True

    # Entity information
    entity_lei: str = ""
    entity_name: str = ""
    entity_type: str = ""

    # Primary competent authority
    primary_nca_id: str = ""
    primary_nca_name: str = ""
    primary_nca_country: str = ""

    # Approval workflow
    require_approval_before_submit: bool = True
    approval_roles: List[str] = field(default_factory=list)

    # Logging
    log_all_reports: bool = True
    log_path: str = "logs/dora/incident_reporting"

    # Callbacks
    submission_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# Main Reporting Engine
# =============================================================================

class DORAIncidentReporter:
    """
    DORA Article 19 compliant incident reporting to competent authorities.

    Implements RTS/ITS reporting requirements including:
    - Initial notification (4h from classification / 24h from detection)
    - Intermediate report (72h from initial)
    - Final report (1 month after resolution)
    - Weekend/holiday deadline extensions
    - Report lifecycle management

    Usage:
        config = IncidentReportingConfig(
            entity_lei="549300EXAMPLE0000",
            entity_name="Example Investment Firm",
            primary_nca_name="BaFin",
            primary_nca_country="DE",
        )
        reporter = DORAIncidentReporter(config)

        # Generate initial notification
        initial = reporter.generate_initial_notification(
            incident_id="INC-001",
            detection_datetime="2025-01-15T10:00:00Z",
            classification_datetime="2025-01-15T11:00:00Z",
            brief_description="System failure affecting trading services",
        )

        # Submit to NCA
        submission = reporter.submit_report(initial.report_id)
    """

    def __init__(self, config: Optional[IncidentReportingConfig] = None):
        """Initialize incident reporter."""
        self.config = config or IncidentReportingConfig()

        # Data stores
        self._initial_notifications: Dict[str, InitialNotificationReport] = {}
        self._intermediate_reports: Dict[str, IntermediateReport] = {}
        self._final_reports: Dict[str, FinalReport] = {}
        self._submissions: Dict[str, ReportSubmission] = {}
        self._authorities: Dict[str, CompetentAuthority] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        # Initialize authorities if configured
        if self.config.primary_nca_id:
            self._init_default_authorities()

        logger.info("DORAIncidentReporter initialized")

    def _init_default_authorities(self) -> None:
        """Initialize default competent authorities."""
        primary = CompetentAuthority(
            authority_id=self.config.primary_nca_id,
            name=self.config.primary_nca_name,
            country_code=self.config.primary_nca_country,
            authority_type=CompetentAuthorityType.NCA_PRIMARY,
        )
        self._authorities[primary.authority_id] = primary

    # =========================================================================
    # Deadline Calculation
    # =========================================================================

    def calculate_initial_notification_deadline(
        self,
        detection_datetime: str,
        classification_datetime: str,
    ) -> str:
        """
        Calculate initial notification deadline per CDR 2025/301.

        Deadline is WHICHEVER IS EARLIER:
        - 4 hours from classification as major
        - 24 hours from initial detection

        Per CDR 2025/301 Art. 4, if deadline falls on weekend/holiday,
        extends to noon of next business day.

        Args:
            detection_datetime: When incident was detected (ISO format)
            classification_datetime: When classified as major (ISO format)

        Returns:
            Deadline datetime (ISO format)
        """
        detected = datetime.fromisoformat(detection_datetime.replace("Z", "+00:00"))
        classified = datetime.fromisoformat(classification_datetime.replace("Z", "+00:00"))

        deadline_from_classification = classified + timedelta(
            hours=self.config.initial_notification_hours_from_classification
        )
        deadline_from_detection = detected + timedelta(
            hours=self.config.initial_notification_hours_from_detection
        )

        # Use whichever is earlier
        deadline = min(deadline_from_classification, deadline_from_detection)

        # Apply weekend/holiday extension if enabled
        if self.config.weekend_extension_enabled:
            deadline = self._apply_weekend_extension(deadline)

        return deadline.isoformat()

    def calculate_intermediate_deadline(
        self,
        initial_notification_datetime: str,
    ) -> str:
        """
        Calculate intermediate report deadline.

        72 hours from initial notification submission.

        Args:
            initial_notification_datetime: When initial was submitted

        Returns:
            Deadline datetime (ISO format)
        """
        initial = datetime.fromisoformat(
            initial_notification_datetime.replace("Z", "+00:00")
        )
        deadline = initial + timedelta(hours=self.config.intermediate_report_hours)

        if self.config.weekend_extension_enabled:
            deadline = self._apply_weekend_extension(deadline)

        return deadline.isoformat()

    def calculate_final_report_deadline(
        self,
        resolution_datetime: Optional[str] = None,
        intermediate_datetime: Optional[str] = None,
    ) -> str:
        """
        Calculate final report deadline.

        1 month from:
        - Incident resolution (if resolved)
        - Intermediate report (if not yet resolved)

        Args:
            resolution_datetime: When incident was resolved
            intermediate_datetime: When intermediate was submitted

        Returns:
            Deadline datetime (ISO format)
        """
        if resolution_datetime:
            base = datetime.fromisoformat(resolution_datetime.replace("Z", "+00:00"))
        elif intermediate_datetime:
            base = datetime.fromisoformat(intermediate_datetime.replace("Z", "+00:00"))
        else:
            base = datetime.now(timezone.utc)

        deadline = base + timedelta(days=self.config.final_report_days)
        return deadline.isoformat()

    def _apply_weekend_extension(
        self,
        deadline: datetime,
    ) -> datetime:
        """
        Apply weekend/holiday extension per CDR 2025/301 Art. 4.

        If deadline falls on Saturday/Sunday, extends to noon next Monday.
        """
        if not self.config.extend_to_noon_next_business_day:
            return deadline

        # Check if weekend (Saturday=5, Sunday=6)
        if deadline.weekday() >= 5:
            # Find next Monday
            days_until_monday = (7 - deadline.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 1  # If Sunday, next day is Monday
            next_business_day = deadline + timedelta(days=days_until_monday)

            # Set to noon
            extended = next_business_day.replace(
                hour=12, minute=0, second=0, microsecond=0
            )
            return extended

        return deadline

    # =========================================================================
    # Report Generation
    # =========================================================================

    def generate_initial_notification(
        self,
        incident_id: str,
        detection_datetime: str,
        classification_datetime: str,
        brief_description: str,
        incident_type_code: IncidentTypeCode = IncidentTypeCode.UNKNOWN,
        critical_services_affected: Optional[List[str]] = None,
        estimated_clients_affected: int = 0,
        member_states_affected: Optional[List[str]] = None,
        is_recurring: bool = False,
        related_incident_references: Optional[List[str]] = None,
        contact_person_name: str = "",
        contact_person_email: str = "",
        contact_person_phone: str = "",
    ) -> InitialNotificationReport:
        """
        Generate initial notification per ITS Annex I.

        This is the first report sent within 4h of classification.

        Args:
            incident_id: Internal incident ID
            detection_datetime: When incident was detected
            classification_datetime: When classified as major
            brief_description: Brief description (max 1000 chars)
            incident_type_code: Type of incident
            critical_services_affected: List of affected critical services
            estimated_clients_affected: Estimated client count
            member_states_affected: List of affected member states
            is_recurring: Whether this is part of recurring pattern
            related_incident_references: Related incident IDs
            contact_person_name: Contact person
            contact_person_email: Contact email
            contact_person_phone: Contact phone

        Returns:
            InitialNotificationReport
        """
        deadline = self.calculate_initial_notification_deadline(
            detection_datetime, classification_datetime
        )

        report = InitialNotificationReport(
            incident_id=incident_id,
            incident_reference=incident_id,
            reporting_entity_lei=self.config.entity_lei,
            reporting_entity_name=self.config.entity_name,
            reporting_entity_type=self.config.entity_type,
            detection_datetime=detection_datetime,
            classification_datetime=classification_datetime,
            incident_type_code=incident_type_code,
            brief_description=brief_description[:1000],  # Max 1000 chars
            critical_services_affected=critical_services_affected or [],
            estimated_clients_affected=estimated_clients_affected,
            member_states_affected=member_states_affected or [],
            is_recurring=is_recurring,
            related_incident_references=related_incident_references or [],
            contact_person_name=contact_person_name,
            contact_person_email=contact_person_email,
            contact_person_phone=contact_person_phone,
            deadline=deadline,
        )

        with self._lock:
            self._initial_notifications[report.report_id] = report

        self._log_report("initial_notification_generated", report.report_id, incident_id)
        logger.info(f"Initial notification generated: {report.report_id}")
        return report

    def generate_intermediate_report(
        self,
        incident_id: str,
        initial_notification_id: str,
        detailed_description: str,
        preliminary_root_cause: str = "",
        root_cause_category: RootCauseCategory = RootCauseCategory.UNKNOWN,
        affected_ict_services: Optional[List[str]] = None,
        affected_business_functions: Optional[List[str]] = None,
        affected_clients_count: int = 0,
        affected_client_types: Optional[List[str]] = None,
        geographic_spread: Optional[List[str]] = None,
        data_compromised: bool = False,
        data_types_affected: Optional[List[str]] = None,
        immediate_actions_taken: Optional[List[str]] = None,
        containment_actions: Optional[List[str]] = None,
        ongoing_actions: Optional[List[str]] = None,
        external_support_engaged: bool = False,
        is_ongoing: bool = True,
        estimated_resolution_datetime: str = "",
        contact_person_name: str = "",
        contact_person_email: str = "",
        contact_person_phone: str = "",
    ) -> IntermediateReport:
        """
        Generate intermediate report per ITS Annex II.

        Due 72 hours after initial notification.

        Args:
            incident_id: Internal incident ID
            initial_notification_id: ID of initial notification
            detailed_description: Detailed description (max 4000 chars)
            preliminary_root_cause: Preliminary root cause analysis
            root_cause_category: Root cause category
            affected_ict_services: Affected ICT services
            affected_business_functions: Affected business functions
            affected_clients_count: Updated client count
            affected_client_types: Types of affected clients
            geographic_spread: Affected countries
            data_compromised: Whether data was compromised
            data_types_affected: Types of data affected
            immediate_actions_taken: Immediate actions taken
            containment_actions: Containment actions
            ongoing_actions: Ongoing response actions
            external_support_engaged: Whether external support engaged
            is_ongoing: Whether incident is ongoing
            estimated_resolution_datetime: Estimated resolution time
            contact_person_name: Contact person
            contact_person_email: Contact email
            contact_person_phone: Contact phone

        Returns:
            IntermediateReport
        """
        # Get initial notification for reference
        with self._lock:
            initial = self._initial_notifications.get(initial_notification_id)

        if not initial:
            raise ValueError(f"Initial notification {initial_notification_id} not found")

        initial_submitted = initial.submitted_at or initial.created_at
        deadline = self.calculate_intermediate_deadline(initial_submitted)

        report = IntermediateReport(
            incident_id=incident_id,
            incident_reference=incident_id,
            initial_notification_id=initial_notification_id,
            reporting_entity_lei=self.config.entity_lei,
            reporting_entity_name=self.config.entity_name,
            detailed_description=detailed_description[:4000],  # Max 4000 chars
            incident_type_code=initial.incident_type_code,
            preliminary_root_cause=preliminary_root_cause,
            root_cause_category=root_cause_category,
            affected_ict_services=affected_ict_services or [],
            affected_business_functions=affected_business_functions or [],
            affected_clients_count=affected_clients_count,
            affected_client_types=affected_client_types or [],
            geographic_spread=geographic_spread or [],
            data_compromised=data_compromised,
            data_types_affected=data_types_affected or [],
            immediate_actions_taken=immediate_actions_taken or [],
            containment_actions=containment_actions or [],
            ongoing_actions=ongoing_actions or [],
            external_support_engaged=external_support_engaged,
            is_ongoing=is_ongoing,
            estimated_resolution_datetime=estimated_resolution_datetime,
            contact_person_name=contact_person_name,
            contact_person_email=contact_person_email,
            contact_person_phone=contact_person_phone,
            deadline=deadline,
        )

        with self._lock:
            self._intermediate_reports[report.report_id] = report

        self._log_report("intermediate_report_generated", report.report_id, incident_id)
        logger.info(f"Intermediate report generated: {report.report_id}")
        return report

    def generate_final_report(
        self,
        incident_id: str,
        initial_notification_id: str,
        intermediate_report_id: str,
        incident_resolved: bool,
        resolution_datetime: str = "",
        resolution_description: str = "",
        comprehensive_description: str = "",
        final_root_cause: str = "",
        root_cause_category: RootCauseCategory = RootCauseCategory.UNKNOWN,
        contributing_factors: Optional[List[str]] = None,
        total_duration_hours: float = 0.0,
        service_downtime_hours: float = 0.0,
        total_clients_affected: int = 0,
        geographic_spread: Optional[List[str]] = None,
        total_economic_impact_eur: float = 0.0,
        lessons_learned: Optional[List[str]] = None,
        remediation_measures: Optional[List[Dict[str, Any]]] = None,
        preventive_measures: Optional[List[Dict[str, Any]]] = None,
        timeline_events: Optional[List[Dict[str, Any]]] = None,
        contact_person_name: str = "",
        contact_person_email: str = "",
        contact_person_phone: str = "",
    ) -> FinalReport:
        """
        Generate final report per ITS Annex III.

        Due 1 month after incident resolution.

        Args:
            incident_id: Internal incident ID
            initial_notification_id: ID of initial notification
            intermediate_report_id: ID of intermediate report
            incident_resolved: Whether incident is resolved
            resolution_datetime: When resolved
            resolution_description: Resolution description
            comprehensive_description: Full incident description
            final_root_cause: Final root cause analysis
            root_cause_category: Root cause category
            contributing_factors: Contributing factors
            total_duration_hours: Total incident duration
            service_downtime_hours: Service downtime
            total_clients_affected: Final client count
            geographic_spread: Affected countries
            total_economic_impact_eur: Total economic impact
            lessons_learned: Lessons learned
            remediation_measures: Remediation measures
            preventive_measures: Preventive measures
            timeline_events: Complete timeline
            contact_person_name: Contact person
            contact_person_email: Contact email
            contact_person_phone: Contact phone

        Returns:
            FinalReport
        """
        # Get intermediate for reference
        with self._lock:
            intermediate = self._intermediate_reports.get(intermediate_report_id)

        intermediate_submitted = ""
        if intermediate:
            intermediate_submitted = intermediate.submitted_at or intermediate.created_at

        deadline = self.calculate_final_report_deadline(
            resolution_datetime if incident_resolved else None,
            intermediate_submitted,
        )

        report = FinalReport(
            incident_id=incident_id,
            incident_reference=incident_id,
            initial_notification_id=initial_notification_id,
            intermediate_report_id=intermediate_report_id,
            reporting_entity_lei=self.config.entity_lei,
            reporting_entity_name=self.config.entity_name,
            incident_resolved=incident_resolved,
            resolution_datetime=resolution_datetime,
            resolution_description=resolution_description,
            comprehensive_description=comprehensive_description,
            final_root_cause=final_root_cause,
            root_cause_category=root_cause_category,
            contributing_factors=contributing_factors or [],
            total_duration_hours=total_duration_hours,
            service_downtime_hours=service_downtime_hours,
            total_clients_affected=total_clients_affected,
            geographic_spread=geographic_spread or [],
            total_economic_impact_eur=total_economic_impact_eur,
            lessons_learned=lessons_learned or [],
            remediation_measures=remediation_measures or [],
            preventive_measures=preventive_measures or [],
            timeline_events=timeline_events or [],
            contact_person_name=contact_person_name,
            contact_person_email=contact_person_email,
            contact_person_phone=contact_person_phone,
            deadline=deadline,
        )

        with self._lock:
            self._final_reports[report.report_id] = report

        self._log_report("final_report_generated", report.report_id, incident_id)
        logger.info(f"Final report generated: {report.report_id}")
        return report

    # =========================================================================
    # Report Approval and Submission
    # =========================================================================

    def approve_report(
        self,
        report_id: str,
        approved_by: str,
        notes: str = "",
    ) -> None:
        """
        Approve a report for submission.

        Args:
            report_id: Report ID
            approved_by: Approver name/ID
            notes: Approval notes
        """
        with self._lock:
            report = self._find_report(report_id)
            if not report:
                raise ValueError(f"Report {report_id} not found")

            report.status = ReportStatus.APPROVED

        self._log_report("report_approved", report_id, getattr(report, 'incident_id', ''))
        logger.info(f"Report approved: {report_id} by {approved_by}")

    def submit_report(
        self,
        report_id: str,
        authority_id: Optional[str] = None,
        submitted_by: str = "",
        submission_method: str = "portal",
    ) -> ReportSubmission:
        """
        Submit a report to the competent authority.

        Args:
            report_id: Report ID
            authority_id: Target authority ID (uses primary if not specified)
            submitted_by: Who is submitting
            submission_method: Method of submission

        Returns:
            ReportSubmission record
        """
        with self._lock:
            report = self._find_report(report_id)
            if not report:
                raise ValueError(f"Report {report_id} not found")

            if self.config.require_approval_before_submit:
                if report.status != ReportStatus.APPROVED:
                    raise ValueError(f"Report {report_id} must be approved before submission")

            # Get authority
            auth_id = authority_id or self.config.primary_nca_id
            authority = self._authorities.get(auth_id)

            now = datetime.now(timezone.utc).isoformat()

            # Create submission record
            submission = ReportSubmission(
                report_id=report_id,
                report_type=report.report_type,
                authority_id=auth_id,
                authority_name=authority.name if authority else "",
                submitted_at=now,
                submitted_by=submitted_by,
                submission_method=submission_method,
            )

            # Update report status
            report.status = ReportStatus.SUBMITTED
            report.submitted_at = now

            self._submissions[submission.submission_id] = submission

        # Trigger callback
        if self.config.submission_callback:
            try:
                self.config.submission_callback("report_submitted", {
                    "submission_id": submission.submission_id,
                    "report_id": report_id,
                    "authority_id": auth_id,
                })
            except Exception as e:
                logger.error(f"Submission callback failed: {e}")

        self._log_report("report_submitted", report_id, getattr(report, 'incident_id', ''))
        logger.info(f"Report submitted: {report_id} to {auth_id}")
        return submission

    def record_acknowledgement(
        self,
        submission_id: str,
        acknowledgement_reference: str,
        acknowledgement_datetime: Optional[str] = None,
    ) -> ReportSubmission:
        """
        Record acknowledgement from competent authority.

        Args:
            submission_id: Submission ID
            acknowledgement_reference: NCA's acknowledgement reference
            acknowledgement_datetime: When acknowledged

        Returns:
            Updated ReportSubmission
        """
        with self._lock:
            if submission_id not in self._submissions:
                raise ValueError(f"Submission {submission_id} not found")

            submission = self._submissions[submission_id]
            submission.acknowledgement_received = True
            submission.acknowledgement_reference = acknowledgement_reference
            submission.acknowledgement_datetime = (
                acknowledgement_datetime or datetime.now(timezone.utc).isoformat()
            )

            # Update report status
            report = self._find_report(submission.report_id)
            if report:
                report.status = ReportStatus.ACKNOWLEDGED

        logger.info(f"Acknowledgement recorded for submission {submission_id}")
        return submission

    def record_feedback(
        self,
        submission_id: str,
        feedback_content: str,
        requires_correction: bool = False,
        correction_deadline: str = "",
    ) -> ReportSubmission:
        """
        Record feedback from competent authority.

        Args:
            submission_id: Submission ID
            feedback_content: Feedback content
            requires_correction: Whether correction is required
            correction_deadline: Deadline for correction

        Returns:
            Updated ReportSubmission
        """
        with self._lock:
            if submission_id not in self._submissions:
                raise ValueError(f"Submission {submission_id} not found")

            submission = self._submissions[submission_id]
            submission.feedback_received = True
            submission.feedback_datetime = datetime.now(timezone.utc).isoformat()
            submission.feedback_content = feedback_content
            submission.requires_correction = requires_correction
            submission.correction_deadline = correction_deadline

        logger.info(f"Feedback recorded for submission {submission_id}")
        return submission

    # =========================================================================
    # Query Methods
    # =========================================================================

    def _find_report(self, report_id: str) -> Any:
        """Find a report by ID across all report types."""
        if report_id in self._initial_notifications:
            return self._initial_notifications[report_id]
        if report_id in self._intermediate_reports:
            return self._intermediate_reports[report_id]
        if report_id in self._final_reports:
            return self._final_reports[report_id]
        return None

    def get_report(self, report_id: str) -> Any:
        """Get report by ID."""
        with self._lock:
            return self._find_report(report_id)

    def get_reports_for_incident(
        self,
        incident_id: str,
    ) -> Dict[str, Any]:
        """
        Get all reports for an incident.

        Returns:
            Dictionary with 'initial', 'intermediate', 'final' keys
        """
        with self._lock:
            initial = [
                r for r in self._initial_notifications.values()
                if r.incident_id == incident_id
            ]
            intermediate = [
                r for r in self._intermediate_reports.values()
                if r.incident_id == incident_id
            ]
            final = [
                r for r in self._final_reports.values()
                if r.incident_id == incident_id
            ]

        return {
            "initial": initial,
            "intermediate": intermediate,
            "final": final,
        }

    def get_pending_reports(self) -> List[Any]:
        """Get reports not yet submitted."""
        with self._lock:
            pending = []
            for reports in [
                self._initial_notifications.values(),
                self._intermediate_reports.values(),
                self._final_reports.values(),
            ]:
                pending.extend([r for r in reports if r.status in (
                    ReportStatus.DRAFT, ReportStatus.PENDING_APPROVAL, ReportStatus.APPROVED
                )])
        return pending

    def get_overdue_reports(self) -> List[Any]:
        """Get reports that are past their deadline."""
        now = datetime.now(timezone.utc)
        overdue = []

        with self._lock:
            for reports in [
                self._initial_notifications.values(),
                self._intermediate_reports.values(),
                self._final_reports.values(),
            ]:
                for r in reports:
                    if r.status not in (ReportStatus.SUBMITTED, ReportStatus.ACKNOWLEDGED):
                        if r.deadline:
                            deadline = datetime.fromisoformat(
                                r.deadline.replace("Z", "+00:00")
                            )
                            if now > deadline:
                                overdue.append(r)

        return overdue

    def get_submission(self, submission_id: str) -> Optional[ReportSubmission]:
        """Get submission by ID."""
        with self._lock:
            return self._submissions.get(submission_id)

    def get_submissions_for_report(
        self,
        report_id: str,
    ) -> List[ReportSubmission]:
        """Get all submissions for a report."""
        with self._lock:
            return [s for s in self._submissions.values() if s.report_id == report_id]

    # =========================================================================
    # Authority Management
    # =========================================================================

    def register_authority(
        self,
        authority: CompetentAuthority,
    ) -> CompetentAuthority:
        """Register a competent authority."""
        with self._lock:
            self._authorities[authority.authority_id] = authority
        return authority

    def get_authority(
        self,
        authority_id: str,
    ) -> Optional[CompetentAuthority]:
        """Get authority by ID."""
        with self._lock:
            return self._authorities.get(authority_id)

    def get_primary_authority(self) -> Optional[CompetentAuthority]:
        """Get primary competent authority."""
        with self._lock:
            for auth in self._authorities.values():
                if auth.authority_type == CompetentAuthorityType.NCA_PRIMARY:
                    return auth
        return None

    # =========================================================================
    # Export and Statistics
    # =========================================================================

    def export_report(
        self,
        report_id: str,
        format: str = "json",
    ) -> Dict[str, Any]:
        """
        Export a report for submission.

        Args:
            report_id: Report ID
            format: Export format

        Returns:
            Export data dictionary
        """
        with self._lock:
            report = self._find_report(report_id)
            if not report:
                raise ValueError(f"Report {report_id} not found")

        return {
            "export_date": datetime.now(timezone.utc).isoformat(),
            "format": format,
            "article_reference": "Article 19",
            "report": asdict(report),
        }

    def get_reporting_statistics(
        self,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get reporting statistics.

        Args:
            period_start: Start of period
            period_end: End of period

        Returns:
            Statistics dictionary
        """
        if not period_end:
            period_end = datetime.now(timezone.utc).isoformat()
        if not period_start:
            period_start = (
                datetime.now(timezone.utc) - timedelta(days=30)
            ).isoformat()

        with self._lock:
            total_initial = len(self._initial_notifications)
            total_intermediate = len(self._intermediate_reports)
            total_final = len(self._final_reports)
            total_submissions = len(self._submissions)

            submitted_initial = sum(
                1 for r in self._initial_notifications.values()
                if r.status in (ReportStatus.SUBMITTED, ReportStatus.ACKNOWLEDGED)
            )
            submitted_intermediate = sum(
                1 for r in self._intermediate_reports.values()
                if r.status in (ReportStatus.SUBMITTED, ReportStatus.ACKNOWLEDGED)
            )
            submitted_final = sum(
                1 for r in self._final_reports.values()
                if r.status in (ReportStatus.SUBMITTED, ReportStatus.ACKNOWLEDGED)
            )

        overdue = len(self.get_overdue_reports())

        return {
            "period_start": period_start,
            "period_end": period_end,
            "initial_notifications": {
                "total": total_initial,
                "submitted": submitted_initial,
                "pending": total_initial - submitted_initial,
            },
            "intermediate_reports": {
                "total": total_intermediate,
                "submitted": submitted_intermediate,
                "pending": total_intermediate - submitted_intermediate,
            },
            "final_reports": {
                "total": total_final,
                "submitted": submitted_final,
                "pending": total_final - submitted_final,
            },
            "total_submissions": total_submissions,
            "overdue_reports": overdue,
        }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_report(
        self,
        event_type: str,
        report_id: str,
        incident_id: str,
    ) -> None:
        """Log a reporting event."""
        if not self.config.log_all_reports:
            return

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "report_id": report_id,
            "incident_id": incident_id,
        }

        log_file = self._log_path / f"reporting_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log reporting event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_incident_reporter(
    config: Optional[IncidentReportingConfig] = None,
) -> DORAIncidentReporter:
    """
    Create a DORAIncidentReporter instance.

    Args:
        config: Optional configuration

    Returns:
        Configured DORAIncidentReporter instance
    """
    return DORAIncidentReporter(config=config)


def get_report_deadlines() -> Dict[str, str]:
    """
    Get report deadlines per CDR 2025/301.

    Returns:
        Dictionary of deadline information
    """
    return {
        "initial_notification": "4 hours from classification OR 24 hours from detection (whichever earlier)",
        "intermediate_report": "72 hours from initial notification",
        "final_report": "1 month from resolution (or from intermediate if not resolved)",
        "weekend_extension": "If deadline on weekend, extends to noon next business day",
    }


def get_report_types() -> List[ReportType]:
    """Get list of report types."""
    return list(ReportType)
