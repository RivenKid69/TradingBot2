# -*- coding: utf-8 -*-
"""
DORA Harmonised Reporting Templates Module (Article 20).

Regulation (EU) 2022/2554 Article 20 mandates harmonised reporting templates
for major ICT-related incidents, developed by ESAs through ITS.

This module implements:
    - ITS Annex I: Initial Notification Template
    - ITS Annex II: Intermediate Report Template
    - ITS Annex III: Final Report Template
    - Template validation and export

Based on:
    - CDR 2025/301 - RTS on content and time limits
    - CIR 2025/302 - ITS on standard forms and templates
    - Entry into force: 12 March 2025

References:
    - Article 20 DORA: https://www.digital-operational-resilience-act.com/Article_20.html
    - CIR 2025/302: ITS on incident reporting templates
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
# Enumerations for Template Fields
# =============================================================================

class IncidentTypeCode(Enum):
    """Incident type codes per ITS."""
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


class ResponseEffectivenessCode(Enum):
    """Response effectiveness codes per ITS."""
    EFFC = "EFFC"  # Effective
    PART = "PART"  # Partially effective
    INEF = "INEF"  # Ineffective


# =============================================================================
# ITS Template Data Structures
# =============================================================================

@dataclass
class ITSInitialNotificationTemplate:
    """
    ITS Annex I - Initial Notification Template.

    Contains mandatory fields for initial notification within
    4 hours of classification / 24 hours of detection.
    """
    # Section 1: Report Identification
    report_reference: str = ""  # Unique report reference
    report_type: str = "INIT"  # INIT for initial
    submission_datetime: str = ""
    version_number: int = 1

    # Section 2: Reporting Entity
    reporting_entity_lei: str = ""
    reporting_entity_name: str = ""
    reporting_entity_type: str = ""  # Entity type code
    reporting_entity_country: str = ""  # ISO 3166-1 alpha-2

    # Section 3: Contact Information
    contact_person_name: str = ""
    contact_person_role: str = ""
    contact_person_email: str = ""
    contact_person_phone: str = ""

    # Section 4: Incident Identification
    incident_reference: str = ""  # Internal reference
    detection_datetime: str = ""  # ISO 8601
    classification_datetime: str = ""  # ISO 8601

    # Section 5: Incident Type
    incident_type_code: str = ""  # IncidentTypeCode value
    incident_type_description: str = ""

    # Section 6: Brief Description
    brief_description: str = ""  # Max 1000 characters

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
    estimated_impact_level: str = ""  # low, medium, high, critical
    estimated_impact_description: str = ""

    # Section 11: Incident Status
    is_ongoing: bool = True

    # Section 12: Recurring Incident Flag
    is_recurring: bool = False
    related_incident_references: List[str] = field(default_factory=list)

    # Section 13: Cross-border Services
    cross_border_services_affected: bool = False

    def __post_init__(self):
        if not self.report_reference:
            self.report_reference = f"INIT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        if not self.submission_datetime:
            self.submission_datetime = datetime.now(timezone.utc).isoformat()

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate template against ITS requirements.

        Returns:
            Tuple of (is_valid, list of validation errors)
        """
        errors = []

        # Mandatory fields
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
    report_type: str = "INTM"  # INTM for intermediate
    initial_report_reference: str = ""
    submission_datetime: str = ""
    version_number: int = 1

    # Section 2: Reporting Entity (same as initial)
    reporting_entity_lei: str = ""
    reporting_entity_name: str = ""
    contact_person_name: str = ""
    contact_person_email: str = ""
    contact_person_phone: str = ""

    # Section 3: Incident Identification
    incident_reference: str = ""
    incident_type_code: str = ""

    # Section 4: Detailed Description
    detailed_description: str = ""  # Max 4000 characters

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
    attack_pattern_reference: str = ""  # MITRE ATT&CK

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

    def __post_init__(self):
        if not self.report_reference:
            self.report_reference = f"INTM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
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
            errors.append("preliminary_root_cause is required (even if 'under investigation')")

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
    report_type: str = "FINL"  # FINL for final
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
    comprehensive_description: str = ""  # No character limit

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
    response_effectiveness_code: str = ""  # EFFC, PART, INEF
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

    def __post_init__(self):
        if not self.report_reference:
            self.report_reference = f"FINL-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
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
    event_type: str = ""  # detection, escalation, containment, recovery, etc.
    description: str = ""
    actor: str = ""  # Who took the action
    system_affected: str = ""

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"EVT-{uuid.uuid4().hex[:6].upper()}"


# =============================================================================
# Template Factory
# =============================================================================

class DORAReportingTemplates:
    """
    DORA Article 20 Harmonised Reporting Templates.

    Provides:
    - Template creation
    - Template validation
    - Export to various formats (JSON, CSV, XML)
    - Template pre-population from incident data

    Usage:
        templates = DORAReportingTemplates()

        # Create initial notification
        initial = templates.create_initial_notification(
            reporting_entity_lei="549300EXAMPLE0000",
            incident_reference="INC-001",
            detection_datetime="2025-01-15T10:00:00Z",
        )

        # Validate
        is_valid, errors = initial.validate()

        # Export
        json_data = templates.export_to_json(initial)
    """

    def __init__(
        self,
        entity_lei: str = "",
        entity_name: str = "",
        entity_type: str = "",
        entity_country: str = "",
    ):
        """
        Initialize reporting templates.

        Args:
            entity_lei: Entity LEI
            entity_name: Entity name
            entity_type: Entity type
            entity_country: Entity country code
        """
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
        **kwargs: Any,
    ) -> ITSFinalReportTemplate:
        """
        Create a final report template.

        Args:
            initial_report_reference: Reference to initial notification
            intermediate_report_reference: Reference to intermediate report
            incident_reference: Internal incident reference
            incident_title: Incident title
            comprehensive_description: Full description
            final_root_cause: Final root cause
            incident_resolved: Resolved flag
            resolution_datetime: Resolution datetime
            total_duration_hours: Total duration
            total_clients_affected: Total clients affected
            total_economic_impact_eur: Total economic impact
            lessons_learned: Lessons learned
            remediation_measures: Remediation measures
            preventive_measures: Preventive measures
            response_effectiveness_code: Response effectiveness
            contact_person_name: Contact name
            contact_person_email: Contact email
            contact_person_phone: Contact phone

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
            remediation_measures=remediation_measures or [{"description": "To be defined"}],
            preventive_measures=preventive_measures or [],
            response_effectiveness_code=response_effectiveness_code,
            contact_person_name=contact_person_name,
            contact_person_email=contact_person_email,
            contact_person_phone=contact_person_phone,
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

    def export_to_csv(
        self,
        template: Any,
    ) -> str:
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

    def export_to_xml(
        self,
        template: Any,
    ) -> str:
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
                            xml_parts.append(f"<item>{_escape_xml(str(item))}</item>")
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

    # =========================================================================
    # Validation
    # =========================================================================

    def validate_template(
        self,
        template: Any,
    ) -> Tuple[bool, List[str]]:
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
        data = asdict(template) if hasattr(template, '__dataclass_fields__') else template

        for field in mandatory_fields:
            if field not in data or not data[field]:
                errors.append(f"Field '{field}' is mandatory but empty or missing")

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
    entity_lei: str = "",
    entity_name: str = "",
    entity_type: str = "",
    entity_country: str = "",
) -> DORAReportingTemplates:
    """
    Create a DORAReportingTemplates instance.

    Args:
        entity_lei: Entity LEI
        entity_name: Entity name
        entity_type: Entity type
        entity_country: Entity country

    Returns:
        Configured DORAReportingTemplates instance
    """
    return DORAReportingTemplates(
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


def create_timeline_event(
    timestamp: str,
    event_type: str,
    description: str,
    actor: str = "",
) -> TimelineEvent:
    """Create a timeline event."""
    return TimelineEvent(
        timestamp=timestamp,
        event_type=event_type,
        description=description,
        actor=actor,
    )
