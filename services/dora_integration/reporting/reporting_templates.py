# -*- coding: utf-8 -*-
"""
DORA Harmonised Reporting Templates Module - Integration Layer (Article 20).

Provides ITS-compliant reporting templates for ICT service providers to generate
incident data packages for client NCA submissions.

Key Distinction (ICT Provider Role):
    - We GENERATE template data packages for clients
    - We DO NOT submit templates to NCAs directly
    - Clients use our pre-populated templates for their Art. 19 obligations

Regulation (EU) 2022/2554 Article 20 mandates harmonised reporting templates
for major ICT-related incidents, developed by ESAs through ITS.

This module implements:
    - ITS Annex I: Initial Notification Template
    - ITS Annex II: Intermediate Report Template
    - ITS Annex III: Final Report Template
    - Template validation and export
    - Pre-population from internal incident data

Based on:
    - CDR 2025/301 - RTS on content and time limits
    - CIR 2025/302 - ITS on standard forms and templates
    - Entry into force: 12 March 2025

References:
    - Article 20 DORA: https://www.digital-operational-resilience-act.com/Article_20.html
    - CIR 2025/302: ITS on incident reporting templates

Migration: services/dora/reporting_templates.py -> services/dora_integration/reporting/
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations for Template Fields (Per ITS Specification)
# =============================================================================

class IncidentTypeCode(Enum):
    """Incident type codes per ITS Annex."""
    CYBA = "CYBA"  # Cyber attack
    SYSF = "SYSF"  # System failure
    EXTE = "EXTE"  # External event
    PROC = "PROC"  # Process failure
    TPFA = "TPFA"  # Third party failure
    HUMA = "HUMA"  # Human error
    UNKN = "UNKN"  # Unknown


class DataTypeCode(Enum):
    """Data type codes per ITS."""
    PERS = "PERS"  # Personal data
    FINA = "FINA"  # Financial data
    CONF = "CONF"  # Confidential business
    TRAD = "TRAD"  # Trading data
    AUTH = "AUTH"  # Authentication credentials
    OTHE = "OTHE"  # Other


class ClientTypeCode(Enum):
    """Client type codes per ITS."""
    RETA = "RETA"  # Retail
    PROF = "PROF"  # Professional
    ELIG = "ELIG"  # Eligible counterparty
    INST = "INST"  # Institutional
    OFIN = "OFIN"  # Other financial entity


class ServiceTypeCode(Enum):
    """Service type codes per ITS."""
    OREX = "OREX"  # Order execution
    MKTD = "MKTD"  # Market data
    RISK = "RISK"  # Risk monitoring
    SETT = "SETT"  # Settlement
    CUST = "CUST"  # Custody
    PAYM = "PAYM"  # Payment processing
    REPT = "REPT"  # Regulatory reporting
    CLPT = "CLPT"  # Client portal
    TRAD = "TRAD"  # Trading infrastructure
    CLOD = "CLOD"  # Cloud services
    DATA = "DATA"  # Data analytics


class ResponseEffectivenessCode(Enum):
    """Response effectiveness codes per ITS."""
    EFFC = "EFFC"  # Effective
    PART = "PART"  # Partially effective
    INEF = "INEF"  # Ineffective


class TemplateExportFormat(Enum):
    """Export formats for templates."""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    DICT = "dict"


# =============================================================================
# ITS Template Data Structures
# =============================================================================

@dataclass
class ITSInitialNotificationTemplate:
    """
    ITS Annex I - Initial Notification Template.

    Contains mandatory fields for initial notification within
    4 hours of classification / 24 hours of detection.

    Note: This is a DATA STRUCTURE for clients to populate
    their NCA submissions. ICT providers generate this for clients.
    """
    # Section 1: Report Identification
    report_reference: str = ""
    report_type: str = "INIT"
    submission_datetime: str = ""
    version_number: int = 1

    # Section 2: Reporting Entity (Client fills their details)
    reporting_entity_lei: str = ""
    reporting_entity_name: str = ""
    reporting_entity_type: str = ""
    reporting_entity_country: str = ""

    # Section 3: Contact Information
    contact_person_name: str = ""
    contact_person_role: str = ""
    contact_person_email: str = ""
    contact_person_phone: str = ""

    # Section 4: Incident Identification
    incident_reference: str = ""
    detection_datetime: str = ""
    classification_datetime: str = ""

    # Section 5: Incident Type
    incident_type_code: str = ""
    incident_type_description: str = ""

    # Section 6: Brief Description
    brief_description: str = ""

    # Section 7: Affected Critical Services
    critical_services_affected: List[str] = field(default_factory=list)
    critical_services_codes: List[str] = field(default_factory=list)

    # Section 8: Estimated Client Impact
    estimated_clients_affected: int = 0
    client_types_affected: List[str] = field(default_factory=list)

    # Section 9: Geographic Scope
    member_states_affected: List[str] = field(default_factory=list)
    third_countries_affected: List[str] = field(default_factory=list)

    # Section 10: Initial Impact Assessment
    estimated_impact_level: str = ""
    estimated_impact_description: str = ""

    # Section 11: Incident Status
    is_ongoing: bool = True

    # Section 12: Recurring Incident Flag
    is_recurring: bool = False
    related_incident_references: List[str] = field(default_factory=list)

    # Section 13: Cross-border Services
    cross_border_services_affected: bool = False

    # Section 14: ICT Provider Information (Our data)
    ict_provider_lei: str = ""
    ict_provider_name: str = ""
    ict_provider_services_affected: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.report_reference:
            self.report_reference = (
                f"INIT-{datetime.now().strftime('%Y%m%d%H%M%S')}-"
                f"{uuid.uuid4().hex[:6].upper()}"
            )
        if not self.submission_datetime:
            self.submission_datetime = datetime.now(timezone.utc).isoformat()

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate template against ITS requirements.

        Returns:
            Tuple of (is_valid, list of validation errors)
        """
        errors = []

        # Mandatory fields per ITS
        if not self.reporting_entity_lei:
            errors.append("reporting_entity_lei is required")
        if not self.reporting_entity_name:
            errors.append("reporting_entity_name is required")
        if not self.incident_reference:
            errors.append("incident_reference is required")
        if not self.detection_datetime:
            errors.append("detection_datetime is required")
        if not self.classification_datetime:
            errors.append("classification_datetime is required")
        if not self.brief_description:
            errors.append("brief_description is required")
        if len(self.brief_description) > 1000:
            errors.append("brief_description exceeds 1000 characters")
        if not self.contact_person_email:
            errors.append("contact_person_email is required")
        if not self.member_states_affected:
            errors.append("at least one member_state is required")

        return len(errors) == 0, errors


@dataclass
class ITSIntermediateReportTemplate:
    """
    ITS Annex II - Intermediate Report Template.

    Contains detailed information for report within 72 hours
    of initial notification.
    """
    # Section 1: Report Identification
    report_reference: str = ""
    report_type: str = "INTM"
    initial_report_reference: str = ""
    submission_datetime: str = ""
    version_number: int = 1

    # Section 2: Reporting Entity
    reporting_entity_lei: str = ""
    reporting_entity_name: str = ""
    contact_person_name: str = ""
    contact_person_email: str = ""
    contact_person_phone: str = ""

    # Section 3: Incident Identification
    incident_reference: str = ""
    incident_type_code: str = ""

    # Section 4: Detailed Description
    detailed_description: str = ""

    # Section 5: Affected ICT Services
    affected_ict_services: List[str] = field(default_factory=list)
    affected_service_codes: List[str] = field(default_factory=list)

    # Section 6: Affected Business Functions
    affected_business_functions: List[str] = field(default_factory=list)
    critical_functions_affected: bool = False

    # Section 7: Updated Client Impact
    affected_clients_count: int = 0
    affected_clients_by_type: Dict[str, int] = field(default_factory=dict)
    client_impact_description: str = ""

    # Section 8: Geographic Spread
    geographic_spread: List[str] = field(default_factory=list)
    cross_border_impact_description: str = ""

    # Section 9: Data Impact
    data_compromised: bool = False
    data_types_affected: List[str] = field(default_factory=list)
    data_type_codes: List[str] = field(default_factory=list)
    records_affected: int = 0
    data_impact_description: str = ""

    # Section 10: Root Cause Analysis (Preliminary)
    preliminary_root_cause: str = ""
    root_cause_category_code: str = ""
    is_malicious: bool = False
    attack_vector: str = ""
    attack_pattern_reference: str = ""

    # Section 11: Immediate Actions Taken
    immediate_actions_taken: List[str] = field(default_factory=list)
    containment_actions: List[str] = field(default_factory=list)

    # Section 12: Recovery Actions
    recovery_actions_started: List[str] = field(default_factory=list)
    recovery_progress_percent: int = 0

    # Section 13: Ongoing Response
    ongoing_actions: List[str] = field(default_factory=list)

    # Section 14: External Support
    external_support_engaged: bool = False
    external_parties_involved: List[str] = field(default_factory=list)

    # Section 15: Timeline
    incident_start_datetime: str = ""
    service_disruption_start: str = ""
    estimated_resolution_datetime: str = ""

    # Section 16: Current Status
    is_ongoing: bool = True
    current_status_description: str = ""

    # Section 17: ICT Provider Data
    ict_provider_lei: str = ""
    ict_provider_name: str = ""
    ict_provider_preliminary_analysis: str = ""

    def __post_init__(self):
        if not self.report_reference:
            self.report_reference = (
                f"INTM-{datetime.now().strftime('%Y%m%d%H%M%S')}-"
                f"{uuid.uuid4().hex[:6].upper()}"
            )
        if not self.submission_datetime:
            self.submission_datetime = datetime.now(timezone.utc).isoformat()

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate template against ITS requirements."""
        errors = []

        if not self.initial_report_reference:
            errors.append("initial_report_reference is required")
        if not self.detailed_description:
            errors.append("detailed_description is required")
        if len(self.detailed_description) > 4000:
            errors.append("detailed_description exceeds 4000 characters")
        if not self.preliminary_root_cause:
            errors.append(
                "preliminary_root_cause is required (even if 'under investigation')"
            )

        return len(errors) == 0, errors


@dataclass
class ITSFinalReportTemplate:
    """
    ITS Annex III - Final Report Template.

    Contains complete information for report within 1 month
    of incident resolution.
    """
    # Section 1: Report Identification
    report_reference: str = ""
    report_type: str = "FINL"
    initial_report_reference: str = ""
    intermediate_report_reference: str = ""
    submission_datetime: str = ""
    version_number: int = 1

    # Section 2: Reporting Entity
    reporting_entity_lei: str = ""
    reporting_entity_name: str = ""
    contact_person_name: str = ""
    contact_person_email: str = ""
    contact_person_phone: str = ""

    # Section 3: Incident Identification
    incident_reference: str = ""
    incident_title: str = ""
    incident_type_code: str = ""

    # Section 4: Comprehensive Description
    comprehensive_description: str = ""

    # Section 5: Resolution Status
    incident_resolved: bool = False
    resolution_datetime: str = ""
    resolution_description: str = ""

    # Section 6: Complete Timeline
    detection_datetime: str = ""
    classification_datetime: str = ""
    incident_start_datetime: str = ""
    service_impact_start: str = ""
    service_impact_end: str = ""
    incident_end_datetime: str = ""
    timeline_events: List[Dict[str, Any]] = field(default_factory=list)

    # Section 7: Final Root Cause Analysis
    final_root_cause: str = ""
    root_cause_category_code: str = ""
    contributing_factors: List[str] = field(default_factory=list)
    root_cause_analysis_method: str = ""

    # Section 8: Complete Impact Assessment
    total_duration_hours: float = 0.0
    service_downtime_hours: float = 0.0
    total_clients_affected: int = 0
    clients_by_type: Dict[str, int] = field(default_factory=dict)
    geographic_spread: List[str] = field(default_factory=list)

    # Section 9: Data Impact (Final)
    data_loss_confirmed: bool = False
    data_types_compromised: List[str] = field(default_factory=list)
    data_type_codes: List[str] = field(default_factory=list)
    total_records_affected: int = 0
    individuals_notified: int = 0
    data_breach_notification_sent: bool = False

    # Section 10: Economic Impact (Final)
    total_economic_impact_eur: float = 0.0
    direct_costs_eur: float = 0.0
    indirect_costs_eur: float = 0.0
    recovery_costs_eur: float = 0.0
    economic_impact_breakdown: Dict[str, float] = field(default_factory=dict)

    # Section 11: Response Effectiveness
    response_effectiveness_code: str = ""
    response_effectiveness_description: str = ""
    response_timeline_met: bool = False
    escalation_procedures_followed: bool = False

    # Section 12: Lessons Learned
    lessons_learned: List[str] = field(default_factory=list)
    what_worked_well: List[str] = field(default_factory=list)
    areas_for_improvement: List[str] = field(default_factory=list)

    # Section 13: Remediation Measures
    remediation_measures: List[Dict[str, Any]] = field(default_factory=list)
    remediation_status: str = ""
    remediation_completion_date: str = ""

    # Section 14: Preventive Measures
    preventive_measures: List[Dict[str, Any]] = field(default_factory=list)
    preventive_implementation_status: str = ""
    preventive_completion_date: str = ""

    # Section 15: Follow-up Actions
    follow_up_actions: List[str] = field(default_factory=list)
    follow_up_deadlines: Dict[str, str] = field(default_factory=dict)

    # Section 16: Attachments
    attachments: List[Dict[str, str]] = field(default_factory=list)

    # Section 17: ICT Provider Final Data
    ict_provider_lei: str = ""
    ict_provider_name: str = ""
    ict_provider_final_analysis: str = ""
    ict_provider_corrective_actions: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.report_reference:
            self.report_reference = (
                f"FINL-{datetime.now().strftime('%Y%m%d%H%M%S')}-"
                f"{uuid.uuid4().hex[:6].upper()}"
            )
        if not self.submission_datetime:
            self.submission_datetime = datetime.now(timezone.utc).isoformat()

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate template against ITS requirements."""
        errors = []

        if not self.initial_report_reference:
            errors.append("initial_report_reference is required")
        if not self.final_root_cause:
            errors.append("final_root_cause is required")
        if not self.comprehensive_description:
            errors.append("comprehensive_description is required")
        if not self.lessons_learned:
            errors.append("at least one lesson_learned is required")
        if not self.remediation_measures:
            errors.append("at least one remediation_measure is required")

        return len(errors) == 0, errors


@dataclass
class TimelineEvent:
    """Timeline event for incident reports."""
    event_id: str = ""
    timestamp: str = ""
    event_type: str = ""
    description: str = ""
    actor: str = ""
    system_affected: str = ""

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"EVT-{uuid.uuid4().hex[:6].upper()}"

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "description": self.description,
            "actor": self.actor,
            "system_affected": self.system_affected,
        }


@dataclass
class ClientIncidentDataPackage:
    """
    Complete incident data package for client.

    ICT providers generate this package containing all data
    the client needs to populate their ITS templates for NCA submission.
    """
    package_id: str = ""
    generated_at: str = ""
    incident_id: str = ""

    # Provider info
    provider_lei: str = ""
    provider_name: str = ""

    # Pre-populated templates
    initial_template: Optional[ITSInitialNotificationTemplate] = None
    intermediate_template: Optional[ITSIntermediateReportTemplate] = None
    final_template: Optional[ITSFinalReportTemplate] = None

    # Raw incident data
    incident_data: Dict[str, Any] = field(default_factory=dict)

    # Timeline
    timeline_events: List[TimelineEvent] = field(default_factory=list)

    # Attachments
    attachments: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        if not self.package_id:
            self.package_id = f"INCPKG-{uuid.uuid4().hex[:8].upper()}"
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Template Factory
# =============================================================================

class DORAReportingTemplates:
    """
    DORA Article 20 Harmonised Reporting Templates Generator.

    Generates ITS-compliant reporting templates for ICT service providers
    to deliver incident data packages to their financial entity clients.

    Key Principle:
        We generate TEMPLATE DATA for clients.
        Clients submit to their NCAs using our pre-populated templates.
        We track generation and delivery, not NCA submission.

    Provides:
    - Template creation
    - Template validation
    - Export to various formats (JSON, CSV, XML)
    - Template pre-population from incident data
    - Client data package generation

    Usage:
        templates = DORAReportingTemplates(
            provider_lei="549300EXAMPLE0000",
            provider_name="ICT Provider Name",
        )

        # Create initial notification template
        initial = templates.create_initial_notification(
            incident_reference="INC-001",
            detection_datetime="2025-01-15T10:00:00Z",
            ...
        )

        # Validate
        is_valid, errors = initial.validate()

        # Export for client
        json_data = templates.export_to_json(initial)

        # Generate complete package
        package = templates.generate_client_data_package(
            incident_id="INC-001",
            incident_data={...},
        )
    """

    def __init__(
        self,
        provider_lei: str = "",
        provider_name: str = "",
        entity_lei: str = "",
        entity_name: str = "",
        entity_type: str = "",
        entity_country: str = "",
    ):
        """
        Initialize reporting templates generator.

        Args:
            provider_lei: ICT Provider LEI (our LEI)
            provider_name: ICT Provider name (our name)
            entity_lei: Default client entity LEI (for pre-population)
            entity_name: Default client entity name
            entity_type: Default client entity type
            entity_country: Default client entity country
        """
        # Provider info (us)
        self.provider_lei = provider_lei
        self.provider_name = provider_name

        # Default client info (can be overridden per template)
        self.entity_lei = entity_lei
        self.entity_name = entity_name
        self.entity_type = entity_type
        self.entity_country = entity_country

        logger.info("DORAReportingTemplates initialized")

    # =========================================================================
    # Template Creation
    # =========================================================================

    def create_initial_notification(
        self,
        incident_reference: str,
        detection_datetime: str,
        classification_datetime: str,
        brief_description: str,
        incident_type_code: str = "UNKN",
        critical_services_affected: Optional[List[str]] = None,
        estimated_clients_affected: int = 0,
        member_states_affected: Optional[List[str]] = None,
        contact_person_name: str = "",
        contact_person_email: str = "",
        contact_person_phone: str = "",
        is_recurring: bool = False,
        ict_services_affected: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ITSInitialNotificationTemplate:
        """
        Create an initial notification template.

        Args:
            incident_reference: Internal incident reference
            detection_datetime: Detection datetime (ISO 8601)
            classification_datetime: Classification datetime (ISO 8601)
            brief_description: Brief description (max 1000 chars)
            incident_type_code: Incident type code
            critical_services_affected: Critical services affected
            estimated_clients_affected: Estimated client count
            member_states_affected: Affected member states
            contact_person_name: Contact name
            contact_person_email: Contact email
            contact_person_phone: Contact phone
            is_recurring: Recurring incident flag
            ict_services_affected: Our services affected

        Returns:
            ITSInitialNotificationTemplate
        """
        template = ITSInitialNotificationTemplate(
            reporting_entity_lei=self.entity_lei,
            reporting_entity_name=self.entity_name,
            reporting_entity_type=self.entity_type,
            reporting_entity_country=self.entity_country,
            incident_reference=incident_reference,
            detection_datetime=detection_datetime,
            classification_datetime=classification_datetime,
            brief_description=brief_description[:1000],
            incident_type_code=incident_type_code,
            critical_services_affected=critical_services_affected or [],
            estimated_clients_affected=estimated_clients_affected,
            member_states_affected=member_states_affected or [],
            contact_person_name=contact_person_name,
            contact_person_email=contact_person_email,
            contact_person_phone=contact_person_phone,
            is_recurring=is_recurring,
            ict_provider_lei=self.provider_lei,
            ict_provider_name=self.provider_name,
            ict_provider_services_affected=ict_services_affected or [],
        )

        return template

    def create_intermediate_report(
        self,
        initial_report_reference: str,
        incident_reference: str,
        detailed_description: str,
        preliminary_root_cause: str = "",
        incident_type_code: str = "UNKN",
        affected_clients_count: int = 0,
        data_compromised: bool = False,
        immediate_actions_taken: Optional[List[str]] = None,
        ongoing_actions: Optional[List[str]] = None,
        is_ongoing: bool = True,
        contact_person_name: str = "",
        contact_person_email: str = "",
        contact_person_phone: str = "",
        ict_provider_analysis: str = "",
        **kwargs: Any,
    ) -> ITSIntermediateReportTemplate:
        """
        Create an intermediate report template.

        Args:
            initial_report_reference: Reference to initial notification
            incident_reference: Internal incident reference
            detailed_description: Detailed description (max 4000 chars)
            preliminary_root_cause: Preliminary root cause
            incident_type_code: Incident type code
            affected_clients_count: Updated client count
            data_compromised: Data compromised flag
            immediate_actions_taken: Immediate actions
            ongoing_actions: Ongoing actions
            is_ongoing: Ongoing flag
            contact_person_name: Contact name
            contact_person_email: Contact email
            contact_person_phone: Contact phone
            ict_provider_analysis: Our preliminary analysis

        Returns:
            ITSIntermediateReportTemplate
        """
        template = ITSIntermediateReportTemplate(
            initial_report_reference=initial_report_reference,
            reporting_entity_lei=self.entity_lei,
            reporting_entity_name=self.entity_name,
            incident_reference=incident_reference,
            incident_type_code=incident_type_code,
            detailed_description=detailed_description[:4000],
            preliminary_root_cause=preliminary_root_cause or "Under investigation",
            affected_clients_count=affected_clients_count,
            data_compromised=data_compromised,
            immediate_actions_taken=immediate_actions_taken or [],
            ongoing_actions=ongoing_actions or [],
            is_ongoing=is_ongoing,
            contact_person_name=contact_person_name,
            contact_person_email=contact_person_email,
            contact_person_phone=contact_person_phone,
            ict_provider_lei=self.provider_lei,
            ict_provider_name=self.provider_name,
            ict_provider_preliminary_analysis=ict_provider_analysis,
        )

        return template

    def create_final_report(
        self,
        initial_report_reference: str,
        intermediate_report_reference: str,
        incident_reference: str,
        incident_title: str,
        comprehensive_description: str,
        final_root_cause: str,
        incident_resolved: bool = True,
        resolution_datetime: str = "",
        total_duration_hours: float = 0.0,
        total_clients_affected: int = 0,
        total_economic_impact_eur: float = 0.0,
        lessons_learned: Optional[List[str]] = None,
        remediation_measures: Optional[List[Dict[str, Any]]] = None,
        preventive_measures: Optional[List[Dict[str, Any]]] = None,
        response_effectiveness_code: str = "EFFC",
        contact_person_name: str = "",
        contact_person_email: str = "",
        contact_person_phone: str = "",
        ict_provider_final_analysis: str = "",
        ict_provider_corrective_actions: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ITSFinalReportTemplate:
        """
        Create a final report template.

        Returns:
            ITSFinalReportTemplate
        """
        template = ITSFinalReportTemplate(
            initial_report_reference=initial_report_reference,
            intermediate_report_reference=intermediate_report_reference,
            reporting_entity_lei=self.entity_lei,
            reporting_entity_name=self.entity_name,
            incident_reference=incident_reference,
            incident_title=incident_title,
            comprehensive_description=comprehensive_description,
            final_root_cause=final_root_cause,
            incident_resolved=incident_resolved,
            resolution_datetime=resolution_datetime,
            total_duration_hours=total_duration_hours,
            total_clients_affected=total_clients_affected,
            total_economic_impact_eur=total_economic_impact_eur,
            lessons_learned=lessons_learned or ["To be documented"],
            remediation_measures=remediation_measures or [
                {"description": "To be defined"}
            ],
            preventive_measures=preventive_measures or [],
            response_effectiveness_code=response_effectiveness_code,
            contact_person_name=contact_person_name,
            contact_person_email=contact_person_email,
            contact_person_phone=contact_person_phone,
            ict_provider_lei=self.provider_lei,
            ict_provider_name=self.provider_name,
            ict_provider_final_analysis=ict_provider_final_analysis,
            ict_provider_corrective_actions=ict_provider_corrective_actions or [],
        )

        return template

    # =========================================================================
    # Template Population from Incident Data
    # =========================================================================

    def populate_initial_from_incident(
        self,
        incident_data: Dict[str, Any],
        contact_info: Optional[Dict[str, str]] = None,
    ) -> ITSInitialNotificationTemplate:
        """
        Populate initial notification from incident data.

        Args:
            incident_data: Incident data dictionary
            contact_info: Contact information

        Returns:
            Pre-populated ITSInitialNotificationTemplate
        """
        contact = contact_info or {}

        return self.create_initial_notification(
            incident_reference=incident_data.get("incident_id", ""),
            detection_datetime=incident_data.get("detected_at", ""),
            classification_datetime=incident_data.get("classified_at", ""),
            brief_description=incident_data.get("description", "")[:1000],
            incident_type_code=self._map_incident_type(
                incident_data.get("incident_type", "")
            ),
            critical_services_affected=incident_data.get("affected_services", []),
            estimated_clients_affected=incident_data.get("affected_clients_count", 0),
            member_states_affected=incident_data.get("geographic_spread", []),
            contact_person_name=contact.get("name", ""),
            contact_person_email=contact.get("email", ""),
            contact_person_phone=contact.get("phone", ""),
            is_recurring=incident_data.get("is_recurring", False),
            ict_services_affected=incident_data.get("provider_services_affected", []),
        )

    def _map_incident_type(self, incident_type: str) -> str:
        """Map internal incident type to ITS code."""
        mapping = {
            "system_failure": "SYSF",
            "security_breach": "CYBA",
            "cyber_attack": "CYBA",
            "data_breach": "CYBA",
            "third_party_failure": "TPFA",
            "human_error": "HUMA",
            "external_event": "EXTE",
            "process_failure": "PROC",
        }
        return mapping.get(incident_type.lower(), "UNKN")

    # =========================================================================
    # Client Data Package Generation
    # =========================================================================

    def generate_client_data_package(
        self,
        incident_id: str,
        incident_data: Dict[str, Any],
        include_templates: bool = True,
        contact_info: Optional[Dict[str, str]] = None,
    ) -> ClientIncidentDataPackage:
        """
        Generate complete incident data package for client.

        Creates a package containing all data the client needs
        to populate their ITS templates for NCA submission.

        Args:
            incident_id: Incident identifier
            incident_data: Raw incident data
            include_templates: Whether to pre-populate templates
            contact_info: Contact information

        Returns:
            ClientIncidentDataPackage
        """
        package = ClientIncidentDataPackage(
            incident_id=incident_id,
            provider_lei=self.provider_lei,
            provider_name=self.provider_name,
            incident_data=incident_data,
        )

        if include_templates:
            # Pre-populate initial template
            package.initial_template = self.populate_initial_from_incident(
                incident_data, contact_info
            )

        # Add timeline events if available
        if "timeline" in incident_data:
            for event_data in incident_data["timeline"]:
                package.timeline_events.append(TimelineEvent(
                    timestamp=event_data.get("timestamp", ""),
                    event_type=event_data.get("type", ""),
                    description=event_data.get("description", ""),
                    actor=event_data.get("actor", ""),
                ))

        return package

    # =========================================================================
    # Export Functions
    # =========================================================================

    def export_to_json(
        self,
        template: Any,
        indent: int = 2,
    ) -> str:
        """
        Export template to JSON format.

        Args:
            template: Template instance
            indent: JSON indentation

        Returns:
            JSON string
        """
        return json.dumps(asdict(template), indent=indent, default=str)

    def export_to_dict(self, template: Any) -> Dict[str, Any]:
        """
        Export template to dictionary.

        Args:
            template: Template instance

        Returns:
            Dictionary
        """
        return asdict(template)

    def export_to_csv(self, template: Any) -> str:
        """
        Export template to CSV format (flat structure).

        Args:
            template: Template instance

        Returns:
            CSV string
        """
        data = asdict(template)

        # Flatten nested structures
        flat_data = {}
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                flat_data[key] = json.dumps(value, default=str)
            else:
                flat_data[key] = value

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=flat_data.keys())
        writer.writeheader()
        writer.writerow(flat_data)

        return output.getvalue()

    def export_to_xml(self, template: Any) -> str:
        """
        Export template to XML format.

        Args:
            template: Template instance

        Returns:
            XML string
        """
        data = asdict(template)

        def dict_to_xml(d: Dict[str, Any], root_tag: str) -> str:
            xml_parts = [f"<{root_tag}>"]
            for key, value in d.items():
                if isinstance(value, list):
                    xml_parts.append(f"<{key}>")
                    for item in value:
                        if isinstance(item, dict):
                            xml_parts.append(dict_to_xml(item, "item"))
                        else:
                            xml_parts.append(
                                f"<item>{_escape_xml(str(item))}</item>"
                            )
                    xml_parts.append(f"</{key}>")
                elif isinstance(value, dict):
                    xml_parts.append(dict_to_xml(value, key))
                else:
                    xml_parts.append(f"<{key}>{_escape_xml(str(value))}</{key}>")
            xml_parts.append(f"</{root_tag}>")
            return "".join(xml_parts)

        report_type = data.get("report_type", "report")
        xml_content = dict_to_xml(data, f"DORA_{report_type}_Report")

        return f'<?xml version="1.0" encoding="UTF-8"?>\n{xml_content}'

    def export_package_to_json(
        self,
        package: ClientIncidentDataPackage,
        indent: int = 2,
    ) -> str:
        """
        Export client data package to JSON.

        Args:
            package: Client data package
            indent: JSON indentation

        Returns:
            JSON string
        """
        data = {
            "package_id": package.package_id,
            "generated_at": package.generated_at,
            "incident_id": package.incident_id,
            "provider": {
                "lei": package.provider_lei,
                "name": package.provider_name,
            },
            "incident_data": package.incident_data,
            "timeline_events": [e.to_dict() for e in package.timeline_events],
            "attachments": package.attachments,
        }

        if package.initial_template:
            data["initial_template"] = asdict(package.initial_template)
        if package.intermediate_template:
            data["intermediate_template"] = asdict(package.intermediate_template)
        if package.final_template:
            data["final_template"] = asdict(package.final_template)

        return json.dumps(data, indent=indent, default=str)

    # =========================================================================
    # Validation
    # =========================================================================

    def validate_template(self, template: Any) -> Tuple[bool, List[str]]:
        """
        Validate a template.

        Args:
            template: Template to validate

        Returns:
            Tuple of (is_valid, errors)
        """
        if hasattr(template, 'validate'):
            return template.validate()
        return True, []

    def validate_all_mandatory_fields(
        self,
        template: Any,
        mandatory_fields: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Validate that all mandatory fields are present.

        Args:
            template: Template to validate
            mandatory_fields: List of mandatory field names

        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        data = (
            asdict(template)
            if hasattr(template, '__dataclass_fields__')
            else template
        )

        for field_name in mandatory_fields:
            if field_name not in data or not data[field_name]:
                errors.append(
                    f"Field '{field_name}' is mandatory but empty or missing"
                )

        return len(errors) == 0, errors


# =============================================================================
# Helper Functions
# =============================================================================

def _escape_xml(text: str) -> str:
    """Escape special XML characters."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
    )


def create_reporting_templates(
    provider_lei: str = "",
    provider_name: str = "",
    entity_lei: str = "",
    entity_name: str = "",
    entity_type: str = "",
    entity_country: str = "",
) -> DORAReportingTemplates:
    """
    Create a DORAReportingTemplates instance.

    Args:
        provider_lei: ICT Provider LEI
        provider_name: ICT Provider name
        entity_lei: Default client entity LEI
        entity_name: Default client entity name
        entity_type: Default client entity type
        entity_country: Default client entity country

    Returns:
        Configured DORAReportingTemplates instance
    """
    return DORAReportingTemplates(
        provider_lei=provider_lei,
        provider_name=provider_name,
        entity_lei=entity_lei,
        entity_name=entity_name,
        entity_type=entity_type,
        entity_country=entity_country,
    )


def get_incident_type_codes() -> Dict[str, str]:
    """Get mapping of incident type codes."""
    return {code.value: code.name for code in IncidentTypeCode}


def get_data_type_codes() -> Dict[str, str]:
    """Get mapping of data type codes."""
    return {code.value: code.name for code in DataTypeCode}


def get_service_type_codes() -> Dict[str, str]:
    """Get mapping of service type codes."""
    return {code.value: code.name for code in ServiceTypeCode}


def get_client_type_codes() -> Dict[str, str]:
    """Get mapping of client type codes."""
    return {code.value: code.name for code in ClientTypeCode}


def create_timeline_event(
    timestamp: str,
    event_type: str,
    description: str,
    actor: str = "",
    system_affected: str = "",
) -> TimelineEvent:
    """Create a timeline event."""
    return TimelineEvent(
        timestamp=timestamp,
        event_type=event_type,
        description=description,
        actor=actor,
        system_affected=system_affected,
    )
