# -*- coding: utf-8 -*-
"""
DORA Critical Third-Party Provider (CTPP) Oversight Module (Articles 31-44).

Regulation (EU) 2022/2554 Articles 31-44 establish the oversight framework
for Critical Third-Party Providers (CTPPs) designated by ESAs.

ICT Provider Perspective (Why This Module Exists):
==================================================
As an ICT Third-Party Service Provider (Art. 30), this module serves THREE purposes:

    1. FORWARD-LOOKING COMPLIANCE: Track if WE become designated as CTPP (Art. 31).
       If our platform becomes systemically important, we need to be prepared for
       Lead Overseer oversight, including inspections and recommendations.

    2. SUBCONTRACTOR DUE DILIGENCE: Monitor CTPP status of OUR subcontractors
       (AWS, Google Cloud, Microsoft Azure, etc.). Per Art. 30(2)(a), we must
       inform clients about subcontracting arrangements with CTPPs.

    3. CLIENT SUPPORT: Provide CTPP-related data to B2B clients for their
       Art. 28 due diligence and Register of Information (Art. 28(3)) submissions.
       Financial entities must document their CTPP dependencies.

Key Components:
    - Article 31: Designation of CTPPs by ESAs
    - Article 32: Structure of the Oversight Framework
    - Article 33: Tasks of the Lead Overseer
    - Article 34: Powers of the Lead Overseer
    - Article 35: Operational coordination between overseers
    - Article 36: Exercise of oversight powers
    - Articles 37-44: Additional oversight requirements

CTPP Designation Criteria (Article 31(2)):
    a) Systemic impact of service disruption on financial entities
    b) Systemic character or importance of financial entities relying on CTPP
    c) Reliance of financial entities on CTPP services
    d) Degree of substitutability
    e) Number of Member States affected

This module provides:
    1. CTPP designation tracking (for us and our subcontractors)
    2. Lead Overseer coordination (if we become CTPP)
    3. Oversight requirement compliance readiness
    4. CTPP-specific risk management
    5. Client data export for their CTPP documentation needs

Architecture Context:
    This module is part of the INTEGRATION layer (services/dora_integration/),
    providing client-facing interfaces for DORA compliance. It is NOT part of
    the archived Financial Entity modules (services/archive/dora_financial_entity/).

References:
    - Articles 31-44 DORA: https://www.digital-operational-resilience-act.com
    - ESAs Joint Statement: CTPPs designated 19 Nov 2025
    - Designated CTPPs: AWS, Google Cloud, Microsoft Azure, ServiceNow, Oracle, Salesforce
    - Lead Overseers: EBA, ESMA, EIOPA
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


class LeadOverseer(Enum):
    """Lead Overseer ESA assignment."""

    EBA = "eba"  # European Banking Authority
    ESMA = "esma"  # European Securities and Markets Authority
    EIOPA = "eiopa"  # European Insurance and Occupational Pensions Authority


class CTPPStatus(Enum):
    """CTPP designation status."""

    NOT_DESIGNATED = "not_designated"
    PENDING_DESIGNATION = "pending_designation"
    DESIGNATED = "designated"
    UNDER_REVIEW = "under_review"
    DESIGNATION_REMOVED = "designation_removed"


class OversightRecommendationType(Enum):
    """Types of oversight recommendations."""

    SECURITY = "security"
    RISK_MANAGEMENT = "risk_management"
    GOVERNANCE = "governance"
    ICT_OPERATIONS = "ict_operations"
    SUBCONTRACTING = "subcontracting"
    INCIDENT_REPORTING = "incident_reporting"
    BUSINESS_CONTINUITY = "business_continuity"
    EXIT_STRATEGY = "exit_strategy"


class RecommendationStatus(Enum):
    """Status of oversight recommendation."""

    RECEIVED = "received"
    UNDER_REVIEW = "under_review"
    ACTION_REQUIRED = "action_required"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    NOT_APPLICABLE = "not_applicable"


class ComplianceLevel(Enum):
    """CTPP-specific compliance level."""

    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSED = "not_assessed"


class OversightExerciseType(Enum):
    """Types of oversight exercises."""

    GENERAL_INVESTIGATION = "general_investigation"
    INSPECTION = "inspection"
    OFF_SITE_REVIEW = "off_site_review"
    INFORMATION_REQUEST = "information_request"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class CTPPDesignation:
    """
    CTPP designation record per Article 31.
    """

    designation_id: str = ""
    provider_id: str = ""
    provider_name: str = ""
    provider_lei: str = ""

    # Designation details
    status: CTPPStatus = CTPPStatus.NOT_DESIGNATED
    designation_date: Optional[str] = None
    designation_reference: str = ""  # ESA decision reference

    # Lead overseer
    lead_overseer: LeadOverseer = LeadOverseer.ESMA
    lead_overseer_contact: str = ""

    # Designation criteria (Article 31(2))
    systemic_impact_score: float = 0.0
    systemic_importance_score: float = 0.0
    reliance_score: float = 0.0
    substitutability_score: float = 0.0  # Lower = less substitutable
    member_states_affected: int = 0

    # Services provided
    services_in_scope: List[str] = field(default_factory=list)
    critical_services: List[str] = field(default_factory=list)

    # Financial entities using
    financial_entities_count: int = 0
    financial_entity_types: List[str] = field(default_factory=list)

    # Geographic scope
    countries_operating: List[str] = field(default_factory=list)
    headquarters_country: str = ""

    # Review
    last_review_date: Optional[str] = None
    next_review_date: Optional[str] = None
    review_frequency_months: int = 12

    # Metadata
    created_date: str = ""
    modified_date: str = ""

    def __post_init__(self):
        if not self.designation_id:
            self.designation_id = f"CTD-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_date:
            self.created_date = datetime.now(timezone.utc).isoformat()


@dataclass
class OversightRecommendation:
    """
    Oversight recommendation from Lead Overseer.
    """

    recommendation_id: str = ""
    ctpp_id: str = ""  # Designation ID
    provider_name: str = ""

    # Recommendation details
    recommendation_type: OversightRecommendationType = OversightRecommendationType.SECURITY
    title: str = ""
    description: str = ""
    detailed_requirements: List[str] = field(default_factory=list)

    # Source
    issuing_authority: LeadOverseer = LeadOverseer.ESMA
    issue_date: str = ""
    reference_number: str = ""

    # Timeline
    deadline: Optional[str] = None
    extension_requested: bool = False
    extension_granted: bool = False
    new_deadline: Optional[str] = None

    # Status
    status: RecommendationStatus = RecommendationStatus.RECEIVED
    implementation_progress_pct: float = 0.0

    # Actions taken
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    evidence_provided: List[str] = field(default_factory=list)

    # Entity impact
    impacts_financial_entities: bool = True
    entity_actions_required: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.recommendation_id:
            self.recommendation_id = f"REC-{uuid.uuid4().hex[:8].upper()}"
        if not self.issue_date:
            self.issue_date = datetime.now(timezone.utc).isoformat()


@dataclass
class OversightExercise:
    """
    Record of oversight exercise conducted by Lead Overseer.
    """

    exercise_id: str = ""
    ctpp_id: str = ""
    provider_name: str = ""

    # Exercise details
    exercise_type: OversightExerciseType = OversightExerciseType.OFF_SITE_REVIEW
    title: str = ""
    scope: str = ""
    areas_covered: List[str] = field(default_factory=list)

    # Authority
    lead_overseer: LeadOverseer = LeadOverseer.ESMA
    exercise_reference: str = ""

    # Timeline
    notification_date: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    report_date: Optional[str] = None

    # Participation
    entity_participation_required: bool = False
    entity_contacts: List[str] = field(default_factory=list)
    information_provided: List[str] = field(default_factory=list)

    # Findings
    findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations_issued: List[str] = field(default_factory=list)

    # Status
    status: str = ""  # notified, in_progress, completed, findings_issued

    def __post_init__(self):
        if not self.exercise_id:
            self.exercise_id = f"OEX-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class CTPPRiskAssessment:
    """
    CTPP-specific risk assessment.
    """

    assessment_id: str = ""
    ctpp_id: str = ""
    provider_name: str = ""

    # Assessment details
    assessment_date: str = ""
    assessed_by: str = ""

    # Risk categories
    concentration_risk_level: str = ""  # low, medium, high, critical
    substitutability_risk_level: str = ""
    geographic_risk_level: str = ""
    operational_risk_level: str = ""

    # Overall assessment
    overall_risk_level: str = ""

    # Dependency metrics
    services_count: int = 0
    critical_services_count: int = 0
    revenue_dependency_pct: float = 0.0
    transaction_dependency_pct: float = 0.0

    # Alternatives
    alternatives_identified: List[str] = field(default_factory=list)
    alternatives_count: int = 0
    exit_plan_exists: bool = False
    exit_plan_tested: bool = False

    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    required_actions: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.assessment_id:
            self.assessment_id = f"CRA-{uuid.uuid4().hex[:8].upper()}"
        if not self.assessment_date:
            self.assessment_date = datetime.now(timezone.utc).isoformat()


@dataclass
class CTPPContractRequirement:
    """
    Additional contractual requirement for CTPP arrangements.
    """

    requirement_id: str = ""
    article_reference: str = ""
    requirement_name: str = ""
    description: str = ""

    # Compliance
    is_mandatory: bool = True
    compliance_status: ComplianceLevel = ComplianceLevel.NOT_ASSESSED
    compliance_notes: str = ""

    # Evidence
    contract_clause_reference: str = ""
    evidence_available: bool = False

    def __post_init__(self):
        if not self.requirement_id:
            self.requirement_id = f"CTR-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class EntityCTPPRelationship:
    """
    Financial entity's relationship with a CTPP.
    """

    relationship_id: str = ""
    ctpp_id: str = ""
    ctpp_name: str = ""
    entity_name: str = ""

    # Relationship details
    contract_reference: str = ""
    relationship_start_date: str = ""

    # Services used
    services_used: List[str] = field(default_factory=list)
    critical_functions_supported: List[str] = field(default_factory=list)

    # Dependency
    dependency_level: str = ""  # low, medium, high, critical
    substitutability: str = ""

    # Compliance
    contract_compliance: ComplianceLevel = ComplianceLevel.NOT_ASSESSED
    oversight_requirements_met: bool = False

    # Actions
    actions_required: List[str] = field(default_factory=list)
    actions_completed: List[str] = field(default_factory=list)

    # Exit readiness
    exit_plan_in_place: bool = False
    alternative_providers: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.relationship_id:
            self.relationship_id = f"REL-{uuid.uuid4().hex[:8].upper()}"


# =============================================================================
# Known CTPPs (as of November 2025)
# =============================================================================

DESIGNATED_CTPPS_2025 = [
    {
        "name": "Amazon Web Services EMEA SARL",
        "lei": "549300FM8DP0RYK05T08",
        "lead_overseer": LeadOverseer.EBA,
        "headquarters_country": "LU",
        "services": ["cloud_computing", "data_storage", "compute_services"],
    },
    {
        "name": "Google Cloud EMEA Limited",
        "lei": "549300U0LCRNP0LXO379",
        "lead_overseer": LeadOverseer.ESMA,
        "headquarters_country": "IE",
        "services": ["cloud_computing", "data_analytics", "ai_services"],
    },
    {
        "name": "Microsoft Ireland Operations Limited",
        "lei": "635400VB27VF2FPU6G28",
        "lead_overseer": LeadOverseer.EBA,
        "headquarters_country": "IE",
        "services": ["cloud_computing", "productivity_services", "data_storage"],
    },
    {
        "name": "ServiceNow Nederland B.V.",
        "lei": "724500YKNF7N9MTJXJ40",
        "lead_overseer": LeadOverseer.EBA,
        "headquarters_country": "NL",
        "services": ["workflow_automation", "it_service_management"],
    },
    {
        "name": "Oracle Corporation",
        "lei": "1Z4GXXU7ZHVWFCD8TV52",
        "lead_overseer": LeadOverseer.EBA,
        "headquarters_country": "US",
        "services": ["database_services", "cloud_computing", "enterprise_software"],
    },
    {
        "name": "Salesforce Inc.",
        "lei": "549300SZXWL54PRYBO13",
        "lead_overseer": LeadOverseer.EBA,
        "headquarters_country": "US",
        "services": ["crm_services", "cloud_computing"],
    },
]


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CTPPOversightConfig:
    """Configuration for CTPP Oversight management."""

    # Assessment
    ctpp_assessment_frequency_months: int = 6
    require_ctpp_exit_plan: bool = True
    require_ctpp_alternatives: bool = True
    min_alternatives_for_ctpp: int = 1

    # Notifications
    notify_on_new_recommendation: bool = True
    notify_on_oversight_exercise: bool = True
    notify_days_before_deadline: int = 14

    # Compliance
    require_annual_ctpp_review: bool = True
    track_recommendation_compliance: bool = True

    # Documentation
    log_all_events: bool = True
    log_path: str = "logs/dora/ctpp_oversight"

    # Callbacks
    notification_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    escalation_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# CTPP-Specific Requirements (Article 30(3) enhanced)
# =============================================================================


def get_ctpp_contract_requirements() -> List[CTPPContractRequirement]:
    """
    Get additional contractual requirements for CTPP arrangements.

    These are enhanced requirements beyond standard Article 30.
    """
    requirements = []

    requirements.append(
        CTPPContractRequirement(
            article_reference="Article 28(7)",
            requirement_name="CTPP Status Acknowledgment",
            description="Contract acknowledges provider's CTPP designation status",
            is_mandatory=True,
        )
    )

    requirements.append(
        CTPPContractRequirement(
            article_reference="Article 33(2)",
            requirement_name="Lead Overseer Access",
            description="Contract ensures Lead Overseer access rights to CTPP",
            is_mandatory=True,
        )
    )

    requirements.append(
        CTPPContractRequirement(
            article_reference="Article 34",
            requirement_name="Cooperation with Oversight Exercises",
            description="Contract requires CTPP cooperation with oversight exercises",
            is_mandatory=True,
        )
    )

    requirements.append(
        CTPPContractRequirement(
            article_reference="Article 35",
            requirement_name="Cross-Border Coordination",
            description="Provisions for coordination between NCAs and Lead Overseer",
            is_mandatory=True,
        )
    )

    requirements.append(
        CTPPContractRequirement(
            article_reference="Article 36",
            requirement_name="Recommendation Implementation",
            description="CTPP commitment to implement oversight recommendations",
            is_mandatory=True,
        )
    )

    requirements.append(
        CTPPContractRequirement(
            article_reference="Article 31(12)",
            requirement_name="Exit Strategy Enhanced",
            description="Enhanced exit strategy requirements for CTPP services",
            is_mandatory=True,
        )
    )

    requirements.append(
        CTPPContractRequirement(
            article_reference="Article 28(8)",
            requirement_name="Transition Support Commitment",
            description="CTPP commitment to transition support if designation removed",
            is_mandatory=True,
        )
    )

    requirements.append(
        CTPPContractRequirement(
            article_reference="Article 37",
            requirement_name="Information Provision",
            description="CTPP obligation to provide information to entity and overseer",
            is_mandatory=True,
        )
    )

    return requirements


# =============================================================================
# Main Implementation
# =============================================================================


class DORACtppOversight:
    """
    DORA Articles 31-44 CTPP Oversight Management.

    Implements management of Critical Third-Party Provider relationships:
    - CTPP designation tracking
    - Lead Overseer coordination
    - Oversight recommendation compliance
    - CTPP-specific risk assessment
    - Enhanced contractual requirements

    Per Articles 31-44, financial entities using CTPPs must:
    - Track CTPP designation status
    - Comply with Lead Overseer requirements
    - Implement oversight recommendations affecting them
    - Maintain enhanced exit strategies
    - Participate in oversight exercises when required

    Usage:
        config = CTPPOversightConfig()
        manager = DORACtppOversight(config)

        # Check if provider is CTPP
        is_ctpp = manager.check_ctpp_status("AWS")

        # Track CTPP relationship
        relationship = manager.register_ctpp_relationship(
            ctpp_name="Amazon Web Services",
            services_used=["cloud_computing", "storage"],
        )

        # Assess CTPP risks
        assessment = manager.assess_ctpp_risks(relationship.relationship_id)
    """

    def __init__(self, config: Optional[CTPPOversightConfig] = None):
        """Initialize CTPP Oversight Management."""
        self.config = config or CTPPOversightConfig()

        # Data stores
        self._designations: Dict[str, CTPPDesignation] = {}
        self._recommendations: Dict[str, OversightRecommendation] = {}
        self._exercises: Dict[str, OversightExercise] = {}
        self._assessments: Dict[str, CTPPRiskAssessment] = {}
        self._relationships: Dict[str, EntityCTPPRelationship] = {}

        # Indexes
        self._designations_by_name: Dict[str, str] = {}  # name -> designation_id
        self._recommendations_by_ctpp: Dict[str, Set[str]] = {}
        self._relationships_by_ctpp: Dict[str, Set[str]] = {}

        # Initialize known CTPPs
        self._initialize_known_ctpps()

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("DORACtppOversight initialized")

    def _initialize_known_ctpps(self) -> None:
        """Initialize with known designated CTPPs."""
        for ctpp_data in DESIGNATED_CTPPS_2025:
            designation = CTPPDesignation(
                provider_name=ctpp_data["name"],
                provider_lei=ctpp_data["lei"],
                status=CTPPStatus.DESIGNATED,
                designation_date="2025-11-19",
                lead_overseer=ctpp_data["lead_overseer"],
                headquarters_country=ctpp_data["headquarters_country"],
                services_in_scope=ctpp_data["services"],
                designation_reference="ESA/2025/CTPP",
            )

            self._designations[designation.designation_id] = designation
            self._designations_by_name[ctpp_data["name"].lower()] = designation.designation_id
            self._recommendations_by_ctpp[designation.designation_id] = set()
            self._relationships_by_ctpp[designation.designation_id] = set()

    # =========================================================================
    # CTPP Designation Management
    # =========================================================================

    def check_ctpp_status(
        self,
        provider_name: str,
    ) -> Tuple[bool, Optional[CTPPDesignation]]:
        """
        Check if a provider is a designated CTPP.

        Args:
            provider_name: Provider name to check

        Returns:
            Tuple of (is_ctpp, designation_info)
        """
        with self._lock:
            # Search by name (case-insensitive)
            search_name = provider_name.lower()

            # Direct match
            if search_name in self._designations_by_name:
                designation_id = self._designations_by_name[search_name]
                designation = self._designations.get(designation_id)
                if designation and designation.status == CTPPStatus.DESIGNATED:
                    return True, designation

            # Partial match (e.g., "AWS" -> "Amazon Web Services EMEA SARL")
            for name, designation_id in self._designations_by_name.items():
                if search_name in name or name in search_name:
                    designation = self._designations.get(designation_id)
                    if designation and designation.status == CTPPStatus.DESIGNATED:
                        return True, designation

            # Check by common abbreviations
            abbreviations = {
                "aws": "amazon web services",
                "gcp": "google cloud",
                "azure": "microsoft",
                "google": "google cloud",
                "microsoft": "microsoft ireland",
            }
            if search_name in abbreviations:
                full_name = abbreviations[search_name]
                for name, designation_id in self._designations_by_name.items():
                    if full_name in name:
                        designation = self._designations.get(designation_id)
                        if designation and designation.status == CTPPStatus.DESIGNATED:
                            return True, designation

            return False, None

    def get_ctpp_designation(
        self,
        designation_id: str,
    ) -> Optional[CTPPDesignation]:
        """Get CTPP designation by ID."""
        with self._lock:
            return self._designations.get(designation_id)

    def get_all_designated_ctpps(self) -> List[CTPPDesignation]:
        """Get all currently designated CTPPs."""
        with self._lock:
            return [d for d in self._designations.values() if d.status == CTPPStatus.DESIGNATED]

    def get_ctpps_by_lead_overseer(
        self,
        lead_overseer: LeadOverseer,
    ) -> List[CTPPDesignation]:
        """Get CTPPs supervised by a specific lead overseer."""
        with self._lock:
            return [
                d
                for d in self._designations.values()
                if d.status == CTPPStatus.DESIGNATED and d.lead_overseer == lead_overseer
            ]

    def add_ctpp_designation(
        self,
        provider_name: str,
        provider_lei: str = "",
        lead_overseer: LeadOverseer = LeadOverseer.ESMA,
        designation_date: Optional[str] = None,
        designation_reference: str = "",
        services_in_scope: Optional[List[str]] = None,
        headquarters_country: str = "",
    ) -> CTPPDesignation:
        """
        Add a new CTPP designation record.

        Use this to track new CTPP designations by ESAs.
        """
        designation = CTPPDesignation(
            provider_name=provider_name,
            provider_lei=provider_lei,
            status=CTPPStatus.DESIGNATED,
            designation_date=designation_date or datetime.now(timezone.utc).isoformat(),
            designation_reference=designation_reference,
            lead_overseer=lead_overseer,
            services_in_scope=services_in_scope or [],
            headquarters_country=headquarters_country,
        )

        with self._lock:
            self._designations[designation.designation_id] = designation
            self._designations_by_name[provider_name.lower()] = designation.designation_id
            self._recommendations_by_ctpp[designation.designation_id] = set()
            self._relationships_by_ctpp[designation.designation_id] = set()

        self._log_event(
            "ctpp_designation_added",
            {
                "designation_id": designation.designation_id,
                "provider_name": provider_name,
                "lead_overseer": lead_overseer.value,
            },
        )

        return designation

    # =========================================================================
    # Entity CTPP Relationship Management
    # =========================================================================

    def register_ctpp_relationship(
        self,
        ctpp_name: str,
        services_used: List[str],
        entity_name: str = "",
        contract_reference: str = "",
        critical_functions: Optional[List[str]] = None,
        dependency_level: str = "medium",
        exit_plan_in_place: bool = False,
        alternative_providers: Optional[List[str]] = None,
    ) -> Optional[EntityCTPPRelationship]:
        """
        Register entity's relationship with a CTPP.

        Args:
            ctpp_name: CTPP provider name
            services_used: Services used from CTPP
            entity_name: Your entity name
            contract_reference: Contract reference
            critical_functions: Critical functions supported
            dependency_level: Dependency level (low, medium, high, critical)
            exit_plan_in_place: Whether exit plan exists
            alternative_providers: List of alternatives

        Returns:
            EntityCTPPRelationship or None if CTPP not found
        """
        with self._lock:
            # Find CTPP
            is_ctpp, designation = self.check_ctpp_status(ctpp_name)
            if not is_ctpp or not designation:
                logger.warning(f"Provider '{ctpp_name}' is not a designated CTPP")
                return None

            relationship = EntityCTPPRelationship(
                ctpp_id=designation.designation_id,
                ctpp_name=designation.provider_name,
                entity_name=entity_name,
                contract_reference=contract_reference,
                relationship_start_date=datetime.now(timezone.utc).isoformat(),
                services_used=services_used,
                critical_functions_supported=critical_functions or [],
                dependency_level=dependency_level,
                exit_plan_in_place=exit_plan_in_place,
                alternative_providers=alternative_providers or [],
            )

            # Determine required actions
            actions = []
            if not exit_plan_in_place and self.config.require_ctpp_exit_plan:
                actions.append("Develop CTPP-specific exit strategy")
            if not alternative_providers and self.config.require_ctpp_alternatives:
                actions.append("Identify alternative providers")
            if critical_functions:
                actions.append("Conduct enhanced CTPP risk assessment")

            relationship.actions_required = actions

            self._relationships[relationship.relationship_id] = relationship
            self._relationships_by_ctpp[designation.designation_id].add(
                relationship.relationship_id
            )

        self._log_event(
            "ctpp_relationship_registered",
            {
                "relationship_id": relationship.relationship_id,
                "ctpp_name": designation.provider_name,
                "services_count": len(services_used),
            },
        )

        return relationship

    def get_ctpp_relationship(
        self,
        relationship_id: str,
    ) -> Optional[EntityCTPPRelationship]:
        """Get CTPP relationship by ID."""
        with self._lock:
            return self._relationships.get(relationship_id)

    def get_all_ctpp_relationships(self) -> List[EntityCTPPRelationship]:
        """Get all CTPP relationships."""
        with self._lock:
            return list(self._relationships.values())

    def get_relationships_for_ctpp(
        self,
        ctpp_id: str,
    ) -> List[EntityCTPPRelationship]:
        """Get all relationships with a specific CTPP."""
        with self._lock:
            rel_ids = self._relationships_by_ctpp.get(ctpp_id, set())
            return [self._relationships[rid] for rid in rel_ids if rid in self._relationships]

    def update_relationship(
        self,
        relationship_id: str,
        **updates: Any,
    ) -> Optional[EntityCTPPRelationship]:
        """Update CTPP relationship."""
        with self._lock:
            if relationship_id not in self._relationships:
                return None

            relationship = self._relationships[relationship_id]
            for key, value in updates.items():
                if hasattr(relationship, key):
                    setattr(relationship, key, value)

        return relationship

    # =========================================================================
    # Oversight Recommendations
    # =========================================================================

    def record_oversight_recommendation(
        self,
        ctpp_name: str,
        recommendation_type: OversightRecommendationType,
        title: str,
        description: str,
        issuing_authority: LeadOverseer,
        deadline: Optional[str] = None,
        reference_number: str = "",
        entity_actions_required: Optional[List[str]] = None,
    ) -> Optional[OversightRecommendation]:
        """
        Record an oversight recommendation from Lead Overseer.

        Use this to track recommendations that may require entity action.

        Args:
            ctpp_name: CTPP the recommendation is for
            recommendation_type: Type of recommendation
            title: Recommendation title
            description: Full description
            issuing_authority: Lead Overseer issuing
            deadline: Implementation deadline
            reference_number: Official reference
            entity_actions_required: Actions required from entity

        Returns:
            OversightRecommendation or None
        """
        with self._lock:
            is_ctpp, designation = self.check_ctpp_status(ctpp_name)
            if not designation:
                return None

            recommendation = OversightRecommendation(
                ctpp_id=designation.designation_id,
                provider_name=designation.provider_name,
                recommendation_type=recommendation_type,
                title=title,
                description=description,
                issuing_authority=issuing_authority,
                deadline=deadline,
                reference_number=reference_number,
                entity_actions_required=entity_actions_required or [],
                impacts_financial_entities=bool(entity_actions_required),
            )

            self._recommendations[recommendation.recommendation_id] = recommendation
            self._recommendations_by_ctpp[designation.designation_id].add(
                recommendation.recommendation_id
            )

        # Notify if configured
        if self.config.notify_on_new_recommendation:
            if self.config.notification_callback:
                self.config.notification_callback(
                    "oversight_recommendation",
                    {
                        "recommendation_id": recommendation.recommendation_id,
                        "ctpp_name": designation.provider_name,
                        "title": title,
                        "entity_actions": entity_actions_required or [],
                    },
                )

        self._log_event(
            "recommendation_recorded",
            {
                "recommendation_id": recommendation.recommendation_id,
                "ctpp_name": designation.provider_name,
                "type": recommendation_type.value,
            },
        )

        return recommendation

    def get_recommendation(
        self,
        recommendation_id: str,
    ) -> Optional[OversightRecommendation]:
        """Get recommendation by ID."""
        with self._lock:
            return self._recommendations.get(recommendation_id)

    def get_recommendations_for_ctpp(
        self,
        ctpp_id: str,
    ) -> List[OversightRecommendation]:
        """Get all recommendations for a CTPP."""
        with self._lock:
            rec_ids = self._recommendations_by_ctpp.get(ctpp_id, set())
            return [self._recommendations[rid] for rid in rec_ids if rid in self._recommendations]

    def get_recommendations_requiring_action(self) -> List[OversightRecommendation]:
        """Get recommendations requiring entity action."""
        with self._lock:
            return [
                r
                for r in self._recommendations.values()
                if r.impacts_financial_entities
                and r.status
                not in [RecommendationStatus.IMPLEMENTED, RecommendationStatus.NOT_APPLICABLE]
            ]

    def update_recommendation_status(
        self,
        recommendation_id: str,
        status: RecommendationStatus,
        progress_pct: float = 0.0,
        actions_taken: Optional[List[Dict[str, Any]]] = None,
        evidence: Optional[List[str]] = None,
    ) -> Optional[OversightRecommendation]:
        """Update recommendation implementation status."""
        with self._lock:
            if recommendation_id not in self._recommendations:
                return None

            rec = self._recommendations[recommendation_id]
            rec.status = status
            rec.implementation_progress_pct = progress_pct

            if actions_taken:
                rec.actions_taken.extend(actions_taken)
            if evidence:
                rec.evidence_provided.extend(evidence)

        return rec

    # =========================================================================
    # Oversight Exercises
    # =========================================================================

    def record_oversight_exercise(
        self,
        ctpp_name: str,
        exercise_type: OversightExerciseType,
        title: str,
        scope: str,
        lead_overseer: LeadOverseer,
        notification_date: Optional[str] = None,
        start_date: Optional[str] = None,
        entity_participation_required: bool = False,
    ) -> Optional[OversightExercise]:
        """
        Record an oversight exercise notification.

        Args:
            ctpp_name: CTPP being examined
            exercise_type: Type of exercise
            title: Exercise title
            scope: Exercise scope
            lead_overseer: Lead Overseer conducting
            notification_date: When notified
            start_date: Exercise start date
            entity_participation_required: Whether entity must participate

        Returns:
            OversightExercise or None
        """
        with self._lock:
            is_ctpp, designation = self.check_ctpp_status(ctpp_name)
            if not designation:
                return None

            exercise = OversightExercise(
                ctpp_id=designation.designation_id,
                provider_name=designation.provider_name,
                exercise_type=exercise_type,
                title=title,
                scope=scope,
                lead_overseer=lead_overseer,
                notification_date=notification_date or datetime.now(timezone.utc).isoformat(),
                start_date=start_date,
                entity_participation_required=entity_participation_required,
                status="notified",
            )

            self._exercises[exercise.exercise_id] = exercise

        # Notify if participation required
        if entity_participation_required and self.config.notify_on_oversight_exercise:
            if self.config.notification_callback:
                self.config.notification_callback(
                    "oversight_exercise",
                    {
                        "exercise_id": exercise.exercise_id,
                        "ctpp_name": designation.provider_name,
                        "participation_required": True,
                    },
                )

        self._log_event(
            "oversight_exercise_recorded",
            {
                "exercise_id": exercise.exercise_id,
                "ctpp_name": designation.provider_name,
                "type": exercise_type.value,
            },
        )

        return exercise

    def get_exercise(
        self,
        exercise_id: str,
    ) -> Optional[OversightExercise]:
        """Get exercise by ID."""
        with self._lock:
            return self._exercises.get(exercise_id)

    def get_active_exercises(self) -> List[OversightExercise]:
        """Get active oversight exercises."""
        with self._lock:
            return [e for e in self._exercises.values() if e.status in ["notified", "in_progress"]]

    # =========================================================================
    # CTPP Risk Assessment
    # =========================================================================

    def assess_ctpp_risks(
        self,
        relationship_id: str,
        assessed_by: str = "",
    ) -> Optional[CTPPRiskAssessment]:
        """
        Perform CTPP-specific risk assessment.

        Args:
            relationship_id: CTPP relationship to assess
            assessed_by: Assessor name

        Returns:
            CTPPRiskAssessment or None
        """
        with self._lock:
            if relationship_id not in self._relationships:
                return None

            relationship = self._relationships[relationship_id]
            designation = self._designations.get(relationship.ctpp_id)
            if not designation:
                return None

            assessment = CTPPRiskAssessment(
                ctpp_id=relationship.ctpp_id,
                provider_name=relationship.ctpp_name,
                assessed_by=assessed_by,
            )

            # Assess concentration risk
            services_count = len(relationship.services_used)
            critical_count = len(relationship.critical_functions_supported)

            assessment.services_count = services_count
            assessment.critical_services_count = critical_count

            if critical_count > 2:
                assessment.concentration_risk_level = "high"
            elif critical_count > 0:
                assessment.concentration_risk_level = "medium"
            else:
                assessment.concentration_risk_level = "low"

            # Assess substitutability
            alternatives = len(relationship.alternative_providers)
            if alternatives == 0:
                assessment.substitutability_risk_level = "critical"
            elif alternatives == 1:
                assessment.substitutability_risk_level = "high"
            elif alternatives <= 2:
                assessment.substitutability_risk_level = "medium"
            else:
                assessment.substitutability_risk_level = "low"

            assessment.alternatives_count = alternatives
            assessment.alternatives_identified = relationship.alternative_providers

            # Assess exit readiness
            assessment.exit_plan_exists = relationship.exit_plan_in_place

            # Determine overall risk
            risk_levels = [
                assessment.concentration_risk_level,
                assessment.substitutability_risk_level,
            ]

            if "critical" in risk_levels:
                assessment.overall_risk_level = "critical"
            elif "high" in risk_levels:
                assessment.overall_risk_level = "high"
            elif "medium" in risk_levels:
                assessment.overall_risk_level = "medium"
            else:
                assessment.overall_risk_level = "low"

            # Generate recommendations
            recommendations = []
            if not relationship.exit_plan_in_place:
                recommendations.append("Develop CTPP exit strategy")
            if alternatives < self.config.min_alternatives_for_ctpp:
                recommendations.append("Identify additional alternative providers")
            if critical_count > 2:
                recommendations.append("Review concentration on single CTPP")

            assessment.recommendations = recommendations

            self._assessments[assessment.assessment_id] = assessment

        self._log_event(
            "ctpp_risk_assessed",
            {
                "assessment_id": assessment.assessment_id,
                "ctpp_name": relationship.ctpp_name,
                "overall_risk": assessment.overall_risk_level,
            },
        )

        return assessment

    def get_ctpp_assessment(
        self,
        assessment_id: str,
    ) -> Optional[CTPPRiskAssessment]:
        """Get CTPP assessment by ID."""
        with self._lock:
            return self._assessments.get(assessment_id)

    # =========================================================================
    # Contract Compliance
    # =========================================================================

    def check_ctpp_contract_compliance(
        self,
        relationship_id: str,
        contract_provisions: Dict[str, bool],
    ) -> Dict[str, Any]:
        """
        Check contract compliance with CTPP-specific requirements.

        Args:
            relationship_id: CTPP relationship
            contract_provisions: Dict of requirement_name -> is_present

        Returns:
            Compliance check results
        """
        with self._lock:
            if relationship_id not in self._relationships:
                return {"error": "Relationship not found"}

            relationship = self._relationships[relationship_id]
            requirements = get_ctpp_contract_requirements()

            results = {
                "relationship_id": relationship_id,
                "ctpp_name": relationship.ctpp_name,
                "check_date": datetime.now(timezone.utc).isoformat(),
                "requirements_checked": [],
                "compliant_count": 0,
                "non_compliant_count": 0,
                "overall_compliance": ComplianceLevel.NOT_ASSESSED,
                "gaps": [],
            }

            for req in requirements:
                is_present = contract_provisions.get(req.requirement_name, False)
                status = ComplianceLevel.COMPLIANT if is_present else ComplianceLevel.NON_COMPLIANT

                results["requirements_checked"].append(
                    {
                        "requirement": req.requirement_name,
                        "article": req.article_reference,
                        "mandatory": req.is_mandatory,
                        "present": is_present,
                        "status": status.value,
                    }
                )

                if is_present:
                    results["compliant_count"] += 1
                else:
                    results["non_compliant_count"] += 1
                    if req.is_mandatory:
                        results["gaps"].append(
                            {
                                "requirement": req.requirement_name,
                                "article": req.article_reference,
                                "action": f"Add {req.requirement_name} clause to contract",
                            }
                        )

            # Determine overall compliance
            total = len(requirements)
            if results["compliant_count"] == total:
                results["overall_compliance"] = ComplianceLevel.COMPLIANT
            elif results["compliant_count"] > total * 0.7:
                results["overall_compliance"] = ComplianceLevel.PARTIALLY_COMPLIANT
            else:
                results["overall_compliance"] = ComplianceLevel.NON_COMPLIANT

            # Update relationship
            relationship.contract_compliance = results["overall_compliance"]

        return results

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_ctpp_oversight_status(self) -> Dict[str, Any]:
        """Get comprehensive CTPP oversight status."""
        with self._lock:
            designations = list(self._designations.values())
            relationships = list(self._relationships.values())
            recommendations = list(self._recommendations.values())

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ctpp_landscape": {
                    "total_designated": len(
                        [d for d in designations if d.status == CTPPStatus.DESIGNATED]
                    ),
                    "by_lead_overseer": {
                        overseer.value: len(
                            [
                                d
                                for d in designations
                                if d.lead_overseer == overseer and d.status == CTPPStatus.DESIGNATED
                            ]
                        )
                        for overseer in LeadOverseer
                    },
                },
                "entity_relationships": {
                    "total": len(relationships),
                    "with_exit_plan": sum(1 for r in relationships if r.exit_plan_in_place),
                    "with_alternatives": sum(1 for r in relationships if r.alternative_providers),
                    "by_dependency_level": {
                        level: sum(1 for r in relationships if r.dependency_level == level)
                        for level in ["low", "medium", "high", "critical"]
                        if sum(1 for r in relationships if r.dependency_level == level) > 0
                    },
                },
                "oversight_activity": {
                    "total_recommendations": len(recommendations),
                    "requiring_action": len(self.get_recommendations_requiring_action()),
                    "active_exercises": len(self.get_active_exercises()),
                },
                "compliance_indicators": {
                    "all_ctpp_relationships_documented": len(relationships) > 0,
                    "all_have_exit_plans": (
                        all(r.exit_plan_in_place for r in relationships) if relationships else True
                    ),
                    "all_have_alternatives": (
                        all(len(r.alternative_providers) > 0 for r in relationships)
                        if relationships
                        else True
                    ),
                    "no_pending_recommendations": len(self.get_recommendations_requiring_action())
                    == 0,
                },
            }

    def generate_ctpp_report(self) -> Dict[str, Any]:
        """Generate comprehensive CTPP report for management."""
        with self._lock:
            return {
                "report_date": datetime.now(timezone.utc).isoformat(),
                "report_type": "ctpp_oversight_summary",
                "designated_ctpps": [
                    {
                        "name": d.provider_name,
                        "lei": d.provider_lei,
                        "lead_overseer": d.lead_overseer.value,
                        "designation_date": d.designation_date,
                        "services": d.services_in_scope,
                    }
                    for d in self._designations.values()
                    if d.status == CTPPStatus.DESIGNATED
                ],
                "entity_ctpp_usage": [
                    {
                        "ctpp": r.ctpp_name,
                        "services_used": r.services_used,
                        "critical_functions": r.critical_functions_supported,
                        "dependency_level": r.dependency_level,
                        "exit_plan": r.exit_plan_in_place,
                        "alternatives": len(r.alternative_providers),
                        "compliance": r.contract_compliance.value,
                    }
                    for r in self._relationships.values()
                ],
                "outstanding_actions": [
                    {
                        "ctpp": r.provider_name,
                        "recommendation": r.title,
                        "status": r.status.value,
                        "deadline": r.deadline,
                        "entity_actions": r.entity_actions_required,
                    }
                    for r in self._recommendations.values()
                    if r.status
                    not in [RecommendationStatus.IMPLEMENTED, RecommendationStatus.NOT_APPLICABLE]
                ],
                "assessments": [
                    {
                        "ctpp": a.provider_name,
                        "overall_risk": a.overall_risk_level,
                        "concentration_risk": a.concentration_risk_level,
                        "substitutability_risk": a.substitutability_risk_level,
                        "exit_plan_exists": a.exit_plan_exists,
                    }
                    for a in self._assessments.values()
                ],
            }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """Log an event."""
        if not self.config.log_all_events:
            return

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"ctpp_events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================


def create_ctpp_oversight(
    config: Optional[CTPPOversightConfig] = None,
) -> DORACtppOversight:
    """
    Create a DORACtppOversight instance.

    Args:
        config: Optional configuration

    Returns:
        Configured DORACtppOversight instance
    """
    return DORACtppOversight(config=config)


def get_lead_overseers() -> List[LeadOverseer]:
    """Get all Lead Overseer ESAs."""
    return list(LeadOverseer)


def get_designated_ctpps_list() -> List[Dict[str, Any]]:
    """Get list of known designated CTPPs."""
    return DESIGNATED_CTPPS_2025.copy()


def get_ctpp_requirements() -> List[CTPPContractRequirement]:
    """Get CTPP-specific contractual requirements."""
    return get_ctpp_contract_requirements()
