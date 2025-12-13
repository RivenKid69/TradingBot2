# -*- coding: utf-8 -*-
"""
DORA ICT Incident Reporting Module (Article 19) - Export Templates.

Regulation (EU) 2022/2554 Article 19 defines incident reporting requirements:
    - Initial notification within 4 hours from classification / 24 hours from detection
    - Intermediate report within 72 hours
    - Final report within 1 month
    - Reporting to designated competent authority

INTEGRATION LAYER CONTEXT - EXPORT-ONLY SEMANTICS:
    This module is designed for ICT THIRD-PARTY PROVIDERS to help their
    FINANCIAL ENTITY CLIENTS with NCA reporting. We provide:

    1. Report TEMPLATES that clients can use
    2. Data EXPORT functions to package incident data
    3. REFERENCE implementation of reporting requirements
    4. CLIENT DATA PACKAGES with all info needed for client's NCA report

    KEY DISTINCTION:
    - ICT providers do NOT report directly to NCAs (unless designated CTPP)
    - We provide DATA to clients; clients submit to their NCAs
    - This module generates EXPORT PACKAGES, not submissions

    Usage Flow:
    >>> from services.dora_integration.incident_interface import (
    ...     DORAIncidentReporter,
    ...     ClientNotificationService,
    ... )
    >>>
    >>> # 1. Notify client about incident
    >>> notifier.notify_affected_clients(incident_id)
    >>>
    >>> # 2. Generate client data package for their NCA report
    >>> package = reporter.generate_client_data_package(incident_id, client_id)
    >>>
    >>> # 3. Client uses package to file their report with NCA

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
    Deadline: 1 month from resolution (or from intermediate if not resolved).
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
class ClientDataPackage:
    """
    Data package for client's NCA reporting.

    This is the primary export for ICT providers - a complete package
    of incident data that clients can use to file their NCA reports.
    """
    package_id: str = ""
    incident_id: str = ""
    client_id: str = ""
    client_name: str = ""

    # Package generation
    generated_at: str = ""
    generated_by: str = ""

    # ICT provider information
    ict_provider_lei: str = ""
    ict_provider_name: str = ""
    contract_reference: str = ""

    # Incident summary
    incident_summary: Dict[str, Any] = field(default_factory=dict)

    # Timeline for client's report
    incident_timeline: List[Dict[str, Any]] = field(default_factory=list)

    # Impact on client
    client_specific_impact: Dict[str, Any] = field(default_factory=dict)

    # Services affected (from client's perspective)
    affected_services: List[str] = field(default_factory=list)

    # Data impact relevant to client
    data_impact: Dict[str, Any] = field(default_factory=dict)

    # Actions taken by provider
    provider_actions: List[Dict[str, Any]] = field(default_factory=list)

    # Root cause (sanitized for client)
    root_cause_summary: str = ""

    # Resolution information
    resolution_status: str = ""
    resolution_description: str = ""
    resolution_datetime: str = ""

    # Preventive measures
    preventive_measures: List[str] = field(default_factory=list)

    # Supporting documentation references
    supporting_documents: List[str] = field(default_factory=list)

    # Export format hints
    suggested_report_type: str = ""  # initial, intermediate, final
    client_deadline_guidance: str = ""

    def __post_init__(self):
        if not self.package_id:
            self.package_id = f"PKG-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()


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

    # Entity information (ICT Provider)
    entity_lei: str = ""
    entity_name: str = ""
    entity_type: str = ""

    # Primary competent authority (for reference)
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
    DORA Article 19 Incident Reporting - Export Template Generator.

    For ICT Third-Party Providers, this class provides:
    - Report TEMPLATES based on ITS requirements
    - CLIENT DATA PACKAGES for client NCA reporting
    - DEADLINE CALCULATION per CDR 2025/301
    - EXPORT functions for client data packages

    Primary Use Case (ICT Provider):
        reporter = DORAIncidentReporter(config)

        # Generate data package for client's NCA report
        package = reporter.generate_client_data_package(
            incident_id="INC-001",
            client_id="CLIENT-001",
        )

        # Client uses package to file their report

    Secondary Use Case (Reference Implementation):
        # Generate template reports for internal tracking
        initial = reporter.generate_initial_notification(...)
        intermediate = reporter.generate_intermediate_report(...)
        final = reporter.generate_final_report(...)
    """

    def __init__(self, config: Optional[IncidentReportingConfig] = None):
        """Initialize incident reporter."""
        self.config = config or IncidentReportingConfig()

        # Data stores
        self._initial_notifications: Dict[str, InitialNotificationReport] = {}
        self._intermediate_reports: Dict[str, IntermediateReport] = {}
        self._final_reports: Dict[str, FinalReport] = {}
        self._client_packages: Dict[str, ClientDataPackage] = {}
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

        logger.info("DORAIncidentReporter initialized (export-only mode)")

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
    # CLIENT DATA PACKAGE GENERATION (Primary Export Function)
    # =========================================================================

    def generate_client_data_package(
        self,
        incident_id: str,
        client_id: str,
        client_name: str = "",
        incident_data: Optional[Dict[str, Any]] = None,
        timeline_events: Optional[List[Dict[str, Any]]] = None,
        affected_services: Optional[List[str]] = None,
        root_cause_summary: str = "",
        resolution_status: str = "",
        resolution_description: str = "",
        resolution_datetime: str = "",
        provider_actions: Optional[List[Dict[str, Any]]] = None,
        preventive_measures: Optional[List[str]] = None,
        generated_by: str = "",
    ) -> ClientDataPackage:
        """
        Generate a client data package for NCA reporting.

        This is the PRIMARY EXPORT FUNCTION for ICT providers.
        Generates a complete data package that clients can use
        to populate their NCA incident reports.

        Args:
            incident_id: Internal incident ID
            client_id: Client identifier
            client_name: Client name
            incident_data: Incident summary data
            timeline_events: List of timeline events
            affected_services: Services affected for this client
            root_cause_summary: Sanitized root cause for client
            resolution_status: Current resolution status
            resolution_description: Resolution details
            resolution_datetime: When resolved
            provider_actions: Actions taken by provider
            preventive_measures: Preventive measures implemented
            generated_by: Who generated the package

        Returns:
            ClientDataPackage ready for client's NCA report
        """
        # Calculate suggested deadline for client
        deadline_guidance = self._calculate_client_deadline_guidance(incident_data or {})

        package = ClientDataPackage(
            incident_id=incident_id,
            client_id=client_id,
            client_name=client_name,
            ict_provider_lei=self.config.entity_lei,
            ict_provider_name=self.config.entity_name,
            generated_by=generated_by,
            incident_summary=incident_data or {},
            incident_timeline=timeline_events or [],
            affected_services=affected_services or [],
            root_cause_summary=root_cause_summary,
            resolution_status=resolution_status,
            resolution_description=resolution_description,
            resolution_datetime=resolution_datetime,
            provider_actions=provider_actions or [],
            preventive_measures=preventive_measures or [],
            client_deadline_guidance=deadline_guidance,
        )

        # Determine suggested report type based on resolution status
        if not resolution_datetime:
            package.suggested_report_type = "initial_notification"
        elif resolution_status in ["resolved", "closed"]:
            package.suggested_report_type = "final_report"
        else:
            package.suggested_report_type = "intermediate_report"

        with self._lock:
            self._client_packages[package.package_id] = package

        self._log_report("client_package_generated", package.package_id, incident_id)
        logger.info(f"Client data package generated: {package.package_id} for client {client_id}")

        return package

    def _calculate_client_deadline_guidance(
        self,
        incident_data: Dict[str, Any],
    ) -> str:
        """Calculate deadline guidance for client's NCA report."""
        detection_datetime = incident_data.get("detection_datetime", "")
        classification_datetime = incident_data.get("classification_datetime", "")

        if not detection_datetime and not classification_datetime:
            return "Client should determine applicable deadline based on their detection/classification timeline"

        try:
            if classification_datetime:
                classified = datetime.fromisoformat(classification_datetime.replace("Z", "+00:00"))
                deadline = classified + timedelta(hours=4)
                return f"Initial notification deadline (4h from classification): {deadline.isoformat()}"
            elif detection_datetime:
                detected = datetime.fromisoformat(detection_datetime.replace("Z", "+00:00"))
                deadline = detected + timedelta(hours=24)
                return f"Initial notification deadline (24h from detection): {deadline.isoformat()}"
        except Exception:
            pass

        return "Client should determine applicable deadline based on DORA Article 19 requirements"

    def export_client_package(
        self,
        package_id: str,
        format: str = "json",
    ) -> Dict[str, Any]:
        """
        Export a client data package.

        Args:
            package_id: Package ID
            format: Export format (json, dict)

        Returns:
            Exported package data
        """
        with self._lock:
            if package_id not in self._client_packages:
                raise ValueError(f"Client package {package_id} not found")

            package = self._client_packages[package_id]

        return {
            "export_type": "client_nca_data_package",
            "export_date": datetime.now(timezone.utc).isoformat(),
            "format": format,
            "package": asdict(package),
            "usage_instructions": {
                "purpose": "This data package is provided to assist with NCA incident reporting",
                "client_responsibility": "Client is responsible for submitting to their NCA",
                "suggested_report_type": package.suggested_report_type,
                "deadline_guidance": package.client_deadline_guidance,
            },
        }

    def get_client_package(
        self,
        package_id: str,
    ) -> Optional[ClientDataPackage]:
        """Get client package by ID."""
        with self._lock:
            return self._client_packages.get(package_id)

    def get_packages_for_client(
        self,
        client_id: str,
    ) -> List[ClientDataPackage]:
        """Get all packages for a client."""
        with self._lock:
            return [
                p for p in self._client_packages.values()
                if p.client_id == client_id
            ]

    def get_packages_for_incident(
        self,
        incident_id: str,
    ) -> List[ClientDataPackage]:
        """Get all packages for an incident."""
        with self._lock:
            return [
                p for p in self._client_packages.values()
                if p.incident_id == incident_id
            ]

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
    # Report Template Generation (Reference Implementation)
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
        Generate initial notification template per ITS Annex I.

        NOTE: For ICT providers, this is a TEMPLATE for reference.
        The actual submission is done by the financial entity client.

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
            InitialNotificationReport template
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
        logger.info(f"Initial notification template generated: {report.report_id}")
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
        Generate intermediate report template per ITS Annex II.

        NOTE: For ICT providers, this is a TEMPLATE for reference.

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
            IntermediateReport template
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
        logger.info(f"Intermediate report template generated: {report.report_id}")
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
        Generate final report template per ITS Annex III.

        NOTE: For ICT providers, this is a TEMPLATE for reference.

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
            FinalReport template
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
        logger.info(f"Final report template generated: {report.report_id}")
        return report

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
        Export a report template.

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
            "export_type": "report_template",
            "export_date": datetime.now(timezone.utc).isoformat(),
            "format": format,
            "article_reference": "Article 19",
            "report": asdict(report),
            "note": "This is a template for reference. Actual submission is client responsibility.",
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
            total_packages = len(self._client_packages)

        overdue = len(self.get_overdue_reports())

        return {
            "period_start": period_start,
            "period_end": period_end,
            "report_templates": {
                "initial_notifications": total_initial,
                "intermediate_reports": total_intermediate,
                "final_reports": total_final,
            },
            "client_data_packages": total_packages,
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
