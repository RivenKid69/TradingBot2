# -*- coding: utf-8 -*-
"""
DORA Contractual Requirements Module (Article 30).

Regulation (EU) 2022/2554 Article 30 establishes key contractual provisions
for ICT service contracts, with enhanced requirements for critical functions.

Article 30 Requirements:
    (2) Basic provisions for ALL ICT service contracts:
        - Clear service description
        - Data processing/storage locations
        - Service level descriptions
        - Incident assistance obligations
        - Cooperation with authorities
        - Termination rights

    (3) Additional provisions for CRITICAL/IMPORTANT functions:
        - Full SLAs with quantitative targets
        - Notice periods
        - Reporting obligations
        - Entity and NCA audit access rights
        - Exit strategies and transition support
        - Performance remediation measures
        - Business continuity guarantees
        - Security measures

This module provides:
    1. Contract requirement validation
    2. Gap analysis for existing contracts
    3. Contract compliance tracking
    4. Amendment recommendation generation
    5. SLA monitoring integration

References:
    - Article 30 DORA: https://www.digital-operational-resilience-act.com/Article_30.html
    - RTS on ICT Risk Management: CDR 2024/1774 Chapter V
    - EBA Guidelines on Outsourcing: EBA/GL/2019/02
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

class RequirementCategory(Enum):
    """Requirement category per Article 30."""
    BASIC = "basic"  # Article 30(2) - all contracts
    CRITICAL = "critical"  # Article 30(3) - critical functions


class RequirementType(Enum):
    """Types of contractual requirements."""
    # =========================================================================
    # Article 30(2) Basic Requirements - ALL ICT service contracts
    # Reference: https://www.digital-operational-resilience-act.com/Article_30.html
    # =========================================================================
    SERVICE_DESCRIPTION = "service_description"  # Art. 30(2)(a) - functions & ICT services
    DATA_LOCATION = "data_location"  # Art. 30(2)(b) - processing/storage locations
    SERVICE_LEVELS = "service_levels"  # Art. 30(2)(c) - availability, authenticity, integrity, confidentiality
    DATA_ACCESS_RECOVERY = "data_access_recovery"  # Art. 30(2)(d) - data access, recovery, return upon termination/insolvency
    SERVICE_LEVEL_DESCRIPTIONS = "service_level_descriptions"  # Art. 30(2)(e) - SLA descriptions with targets
    INCIDENT_ASSISTANCE = "incident_assistance"  # Art. 30(2)(f) - assistance during ICT incidents
    AUTHORITY_COOPERATION = "authority_cooperation"  # Art. 30(2)(g) - cooperation with NCAs
    TERMINATION_RIGHTS = "termination_rights"  # Art. 30(2)(h) - termination rights and notice periods
    TRAINING_PARTICIPATION = "training_participation"  # Art. 30(2)(i) - security awareness training

    # =========================================================================
    # Article 30(3) Critical Function Requirements - critical/important functions only
    # =========================================================================
    SLA_TARGETS = "sla_targets"  # Art. 30(3)(a) - full SLAs with quantitative targets
    NOTICE_PERIODS = "notice_periods"  # Art. 30(3)(b) - notice periods
    REPORTING_OBLIGATIONS = "reporting_obligations"  # Art. 30(3)(b) - reporting obligations
    BCP_REQUIREMENTS = "bcp_requirements"  # Art. 30(3)(c) - provider's business contingency plans
    RESILIENCE_TESTING = "resilience_testing"  # Art. 30(3)(d) - participation in client's testing
    AUDIT_RIGHTS = "audit_rights"  # Art. 30(3)(e) - unrestricted audit rights
    EXIT_STRATEGY = "exit_strategy"  # Art. 30(3)(f) - exit strategies
    NCA_ACCESS = "nca_access"  # Art. 30(3)(g) - supervisory cooperation
    BUSINESS_CONTINUITY = "business_continuity"  # Art. 30(3)(h) - BCP implementation and testing
    SECURITY_MEASURES = "security_measures"  # Art. 30(3)(i) - security arrangements
    SUBCONTRACTING = "subcontracting"  # Art. 30(3)(j) - subcontracting conditions

    # Additional derived requirements
    TRANSITION_SUPPORT = "transition_support"  # Part of Art. 30(3)(f)
    PERFORMANCE_REMEDIATION = "performance_remediation"  # SLA breach handling
    DATA_PROTECTION = "data_protection"  # GDPR alignment
    CONFIDENTIALITY = "confidentiality"
    IP_RIGHTS = "ip_rights"


class ComplianceStatus(Enum):
    """Contract compliance status."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSED = "not_assessed"
    REQUIRES_REVIEW = "requires_review"


class GapSeverity(Enum):
    """Severity of identified gap."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RemediationStatus(Enum):
    """Remediation action status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not_applicable"


class ContractStatus(Enum):
    """Contract review status."""
    ACTIVE = "active"
    UNDER_REVIEW = "under_review"
    PENDING_AMENDMENT = "pending_amendment"
    AMENDED = "amended"
    TERMINATED = "terminated"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ContractualRequirement:
    """
    DORA contractual requirement definition.

    Defines requirements per Article 30(2) and 30(3).
    """
    requirement_id: str = ""
    requirement_type: RequirementType = RequirementType.SERVICE_DESCRIPTION
    category: RequirementCategory = RequirementCategory.BASIC
    article_reference: str = ""

    # Description
    name: str = ""
    description: str = ""
    detailed_criteria: List[str] = field(default_factory=list)

    # Applicability
    applies_to_all: bool = True
    applies_to_critical_only: bool = False

    # Validation
    mandatory: bool = True
    verification_method: str = ""  # document_review, interview, evidence

    def __post_init__(self):
        if not self.requirement_id:
            self.requirement_id = f"REQ-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class ContractProvision:
    """
    Actual provision in a contract.

    Maps to a requirement and tracks compliance.
    """
    provision_id: str = ""
    contract_id: str = ""
    requirement_id: str = ""

    # Provision details
    clause_reference: str = ""  # Section/clause in contract
    provision_text: str = ""  # Actual contract text (summary)
    provision_summary: str = ""

    # Compliance
    compliance_status: ComplianceStatus = ComplianceStatus.NOT_ASSESSED
    compliance_notes: str = ""

    # Assessment
    assessed_by: str = ""
    assessed_date: Optional[str] = None
    next_review_date: Optional[str] = None

    def __post_init__(self):
        if not self.provision_id:
            self.provision_id = f"PRV-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class ContractAssessment:
    """
    Assessment of a contract against DORA requirements.
    """
    assessment_id: str = ""
    contract_id: str = ""
    provider_name: str = ""
    is_critical_function: bool = False

    # Assessment details
    assessment_date: str = ""
    assessed_by: str = ""
    assessment_type: str = ""  # initial, periodic, trigger-based

    # Requirements analysis
    applicable_requirements: List[str] = field(default_factory=list)
    assessed_provisions: List[str] = field(default_factory=list)

    # Compliance summary
    overall_compliance: ComplianceStatus = ComplianceStatus.NOT_ASSESSED
    compliance_score_pct: float = 0.0
    compliant_count: int = 0
    partially_compliant_count: int = 0
    non_compliant_count: int = 0

    # Gaps identified
    gaps: List[str] = field(default_factory=list)  # Gap IDs

    # Recommendations
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    required_amendments: List[Dict[str, Any]] = field(default_factory=list)

    # Approval
    reviewed_by: str = ""
    review_date: Optional[str] = None
    approved_by: str = ""
    approval_date: Optional[str] = None

    def __post_init__(self):
        if not self.assessment_id:
            self.assessment_id = f"ASM-{uuid.uuid4().hex[:8].upper()}"
        if not self.assessment_date:
            self.assessment_date = datetime.now(timezone.utc).isoformat()


@dataclass
class ContractGap:
    """
    Identified gap in contract compliance.
    """
    gap_id: str = ""
    assessment_id: str = ""
    contract_id: str = ""
    requirement_id: str = ""

    # Gap details
    gap_description: str = ""
    severity: GapSeverity = GapSeverity.MEDIUM
    article_reference: str = ""

    # Impact
    impact_description: str = ""
    risk_if_not_remediated: str = ""

    # Remediation
    remediation_recommendation: str = ""
    remediation_status: RemediationStatus = RemediationStatus.NOT_STARTED
    remediation_owner: str = ""
    remediation_deadline: Optional[str] = None
    remediation_completed_date: Optional[str] = None

    # Tracking
    identified_date: str = ""
    last_updated: str = ""

    def __post_init__(self):
        if not self.gap_id:
            self.gap_id = f"GAP-{uuid.uuid4().hex[:8].upper()}"
        if not self.identified_date:
            self.identified_date = datetime.now(timezone.utc).isoformat()
        if not self.last_updated:
            self.last_updated = self.identified_date


@dataclass
class ContractAmendment:
    """
    Proposed or completed contract amendment.
    """
    amendment_id: str = ""
    contract_id: str = ""
    provider_name: str = ""

    # Amendment details
    amendment_type: str = ""  # add_clause, modify_clause, remove_clause
    related_gaps: List[str] = field(default_factory=list)
    related_requirements: List[str] = field(default_factory=list)

    # Content
    proposed_changes: List[Dict[str, Any]] = field(default_factory=list)
    justification: str = ""

    # Status
    status: str = ""  # draft, submitted, negotiating, accepted, rejected, implemented
    submitted_date: Optional[str] = None
    provider_response_date: Optional[str] = None
    implementation_date: Optional[str] = None

    # Negotiation
    negotiation_notes: str = ""
    provider_counter_proposal: str = ""

    # Tracking
    created_date: str = ""
    created_by: str = ""

    def __post_init__(self):
        if not self.amendment_id:
            self.amendment_id = f"AMD-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_date:
            self.created_date = datetime.now(timezone.utc).isoformat()


@dataclass
class SLADefinition:
    """
    Service Level Agreement definition.
    """
    sla_id: str = ""
    contract_id: str = ""
    service_name: str = ""

    # SLA metrics
    metric_name: str = ""
    metric_description: str = ""
    target_value: float = 0.0
    target_unit: str = ""  # percent, hours, minutes, etc.
    measurement_period: str = ""  # monthly, quarterly, annually

    # Thresholds
    warning_threshold: float = 0.0
    breach_threshold: float = 0.0

    # Remediation
    remediation_on_breach: str = ""
    service_credits: str = ""

    # Monitoring
    monitoring_method: str = ""
    reporting_frequency: str = ""

    def __post_init__(self):
        if not self.sla_id:
            self.sla_id = f"SLA-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class ICTContract:
    """
    ICT Service Contract record.
    """
    contract_id: str = ""
    contract_reference: str = ""
    provider_id: str = ""
    provider_name: str = ""

    # Contract details
    contract_type: str = ""  # outsourcing, procurement, intra_group
    effective_date: str = ""
    expiry_date: Optional[str] = None
    renewal_type: str = ""  # automatic, manual
    notice_period_days: int = 30

    # Criticality
    supports_critical_function: bool = False
    critical_functions: List[str] = field(default_factory=list)

    # Status
    status: ContractStatus = ContractStatus.ACTIVE
    last_assessment_id: str = ""
    last_assessment_date: Optional[str] = None
    overall_compliance: ComplianceStatus = ComplianceStatus.NOT_ASSESSED

    # Provisions
    provisions: List[str] = field(default_factory=list)  # Provision IDs
    slas: List[str] = field(default_factory=list)  # SLA IDs

    # Gaps and amendments
    open_gaps: List[str] = field(default_factory=list)
    pending_amendments: List[str] = field(default_factory=list)

    # Metadata
    created_date: str = ""
    modified_date: str = ""

    def __post_init__(self):
        if not self.contract_id:
            self.contract_id = f"CTR-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_date:
            self.created_date = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ContractualRequirementsConfig:
    """Configuration for Contractual Requirements management."""

    # Assessment settings
    assessment_frequency_months: int = 12
    critical_contract_assessment_months: int = 6
    assessment_reminder_days: int = 30

    # Compliance thresholds
    compliant_threshold_pct: float = 90.0
    partially_compliant_threshold_pct: float = 70.0

    # Gap management
    critical_gap_escalation: bool = True
    gap_remediation_default_days: int = 90
    critical_gap_remediation_days: int = 30

    # Notifications
    notify_on_new_gap: bool = True
    notify_on_amendment_response: bool = True
    notify_before_review: bool = True

    # Documentation
    log_all_events: bool = True
    log_path: str = "logs/dora/contractual_requirements"

    # Callbacks
    escalation_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    notification_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# DORA Article 30 Requirements Definition
# =============================================================================

def get_article_30_requirements() -> List[ContractualRequirement]:
    """
    Get all Article 30 requirements.

    Returns complete list of DORA Article 30(2) and 30(3) requirements.
    """
    requirements = []

    # ==========================================================================
    # Article 30(2) - Basic Requirements (ALL ICT service contracts)
    # ==========================================================================

    requirements.append(ContractualRequirement(
        requirement_type=RequirementType.SERVICE_DESCRIPTION,
        category=RequirementCategory.BASIC,
        article_reference="Article 30(2)(a)",
        name="Clear Service Description",
        description="Clear and complete description of all functions and ICT services to be provided",
        detailed_criteria=[
            "Complete list of ICT services provided",
            "Description of each service functionality",
            "Service scope and boundaries",
            "Interfaces with other services",
        ],
        mandatory=True,
        verification_method="document_review",
    ))

    requirements.append(ContractualRequirement(
        requirement_type=RequirementType.DATA_LOCATION,
        category=RequirementCategory.BASIC,
        article_reference="Article 30(2)(b)",
        name="Data Processing and Storage Locations",
        description="Locations where data will be processed and stored",
        detailed_criteria=[
            "Specific countries for data processing",
            "Specific countries for data storage",
            "Requirements for location changes notification",
            "Restrictions on data transfers",
        ],
        mandatory=True,
        verification_method="document_review",
    ))

    requirements.append(ContractualRequirement(
        requirement_type=RequirementType.SERVICE_LEVELS,
        category=RequirementCategory.BASIC,
        article_reference="Article 30(2)(c)",
        name="Service Level Descriptions",
        description="Provisions on availability, authenticity, integrity and confidentiality",
        detailed_criteria=[
            "Availability commitments",
            "Data authenticity measures",
            "Data integrity protection",
            "Confidentiality requirements",
            "Personal data protection measures",
        ],
        mandatory=True,
        verification_method="document_review",
    ))

    # Article 30(2)(d) - Data Access, Recovery and Return
    # CRITICAL: This requirement is often missed but is MANDATORY for ALL contracts
    requirements.append(ContractualRequirement(
        requirement_type=RequirementType.DATA_ACCESS_RECOVERY,
        category=RequirementCategory.BASIC,
        article_reference="Article 30(2)(d)",
        name="Data Access, Recovery and Return",
        description="Provisions on data access, recovery and return in easily accessible format upon termination, insolvency, or resolution of ICT provider",
        detailed_criteria=[
            "Data export in easily accessible, non-proprietary format",
            "Data access mechanisms upon contract termination",
            "Data recovery procedures upon provider insolvency",
            "Data return timeline commitments",
            "Data management provisions during provider resolution",
            "Escrow or safeguard arrangements for critical data",
        ],
        mandatory=True,
        verification_method="document_review",
    ))

    # Article 30(2)(e) - Service Level Descriptions
    requirements.append(ContractualRequirement(
        requirement_type=RequirementType.SERVICE_LEVEL_DESCRIPTIONS,
        category=RequirementCategory.BASIC,
        article_reference="Article 30(2)(e)",
        name="Service Level Descriptions",
        description="Service level descriptions including quantitative and qualitative performance targets, with provisions for updates and revisions",
        detailed_criteria=[
            "Quantitative performance targets (uptime %, latency)",
            "Qualitative service descriptions",
            "Measurement methodology",
            "Reporting frequency",
            "Provisions for SLA updates and revisions",
        ],
        mandatory=True,
        verification_method="document_review",
    ))

    # Article 30(2)(f) - Incident Assistance
    requirements.append(ContractualRequirement(
        requirement_type=RequirementType.INCIDENT_ASSISTANCE,
        category=RequirementCategory.BASIC,
        article_reference="Article 30(2)(f)",
        name="Incident Assistance Obligations",
        description="Obligation to provide assistance in the event of ICT incidents at no additional cost or at predetermined cost",
        detailed_criteria=[
            "Incident notification timelines",
            "Support during incident response",
            "Root cause analysis cooperation",
            "Remediation support",
            "Cost provisions for assistance (free or predetermined)",
        ],
        mandatory=True,
        verification_method="document_review",
    ))

    # Article 30(2)(g) - Authority Cooperation
    requirements.append(ContractualRequirement(
        requirement_type=RequirementType.AUTHORITY_COOPERATION,
        category=RequirementCategory.BASIC,
        article_reference="Article 30(2)(g)",
        name="Cooperation with Competent Authorities",
        description="Obligation to fully cooperate with competent authorities and resolution authorities of the financial entity",
        detailed_criteria=[
            "Unrestricted NCA access provisions",
            "Information provision on request",
            "Cooperation in supervisory activities",
            "Resolution authority access rights",
            "No impediment to supervision clause",
        ],
        mandatory=True,
        verification_method="document_review",
    ))

    # Article 30(2)(h) - Termination Rights
    requirements.append(ContractualRequirement(
        requirement_type=RequirementType.TERMINATION_RIGHTS,
        category=RequirementCategory.BASIC,
        article_reference="Article 30(2)(h)",
        name="Termination Rights and Notice Periods",
        description="Termination rights and related minimum notice periods for contract termination, in line with supervisory expectations",
        detailed_criteria=[
            "Clear termination clauses",
            "Minimum notice period requirements",
            "Termination for cause provisions",
            "Regulatory termination rights",
            "Exit transition provisions",
        ],
        mandatory=True,
        verification_method="document_review",
    ))

    # Article 30(2)(i) - Training Participation
    requirements.append(ContractualRequirement(
        requirement_type=RequirementType.TRAINING_PARTICIPATION,
        category=RequirementCategory.BASIC,
        article_reference="Article 30(2)(i)",
        name="Security Awareness and Resilience Training Participation",
        description="Conditions for ICT provider participation in financial entity's security awareness programmes and digital operational resilience training per Article 13(6)",
        detailed_criteria=[
            "Commitment to participate in client security awareness programs",
            "Participation in digital operational resilience training",
            "Key personnel availability for training exercises",
            "Joint incident simulation exercises",
            "Reasonable notice and scheduling provisions",
        ],
        mandatory=True,
        verification_method="document_review",
    ))

    # ==========================================================================
    # Article 30(3) - Additional Requirements (Critical/Important Functions)
    # Reference: https://www.digital-operational-resilience-act.com/Article_30.html
    # ==========================================================================

    # Article 30(3)(a) - Full SLAs with Quantitative Targets
    requirements.append(ContractualRequirement(
        requirement_type=RequirementType.SLA_TARGETS,
        category=RequirementCategory.CRITICAL,
        article_reference="Article 30(3)(a)",
        name="Full SLAs with Quantitative Targets",
        description="Full service level descriptions including quantitative and qualitative performance targets",
        detailed_criteria=[
            "Quantitative availability targets",
            "Response time targets",
            "Resolution time targets",
            "Performance benchmarks",
            "Capacity commitments",
        ],
        applies_to_all=False,
        applies_to_critical_only=True,
        mandatory=True,
        verification_method="document_review",
    ))

    # Article 30(3)(b) - Notice Periods AND Reporting Obligations
    requirements.append(ContractualRequirement(
        requirement_type=RequirementType.NOTICE_PERIODS,
        category=RequirementCategory.CRITICAL,
        article_reference="Article 30(3)(b)",
        name="Notice Periods and Reporting Obligations",
        description="Notice periods and reporting obligations to the financial entity",
        detailed_criteria=[
            "Termination notice period",
            "Change notification requirements",
            "Minimum notice for material changes",
            "Regular performance reporting",
            "Incident reporting frequency",
        ],
        applies_to_all=False,
        applies_to_critical_only=True,
        mandatory=True,
        verification_method="document_review",
    ))

    # Article 30(3)(c) - Provider's Business Contingency Plans
    requirements.append(ContractualRequirement(
        requirement_type=RequirementType.BCP_REQUIREMENTS,
        category=RequirementCategory.CRITICAL,
        article_reference="Article 30(3)(c)",
        name="Provider Business Contingency Plans",
        description="Requirements for ICT provider to maintain appropriate business contingency plans",
        detailed_criteria=[
            "Provider must have documented BCP",
            "Disaster recovery plan requirements",
            "ICT-specific contingency measures",
            "Regular BCP review and updates",
            "BCP summary available to client on request",
        ],
        applies_to_all=False,
        applies_to_critical_only=True,
        mandatory=True,
        verification_method="document_review",
    ))

    # Article 30(3)(d) - Resilience Testing Participation
    requirements.append(ContractualRequirement(
        requirement_type=RequirementType.RESILIENCE_TESTING,
        category=RequirementCategory.CRITICAL,
        article_reference="Article 30(3)(d)",
        name="Resilience Testing Participation",
        description="Participation in testing of business contingency plans per Articles 26-27",
        detailed_criteria=[
            "Obligation to participate in client's resilience testing",
            "Cooperation with penetration tests",
            "Vulnerability assessment support",
            "Threat-led testing cooperation (TLPT if required)",
            "Testing results sharing provisions",
            "Joint DR exercises",
        ],
        applies_to_all=False,
        applies_to_critical_only=True,
        mandatory=True,
        verification_method="document_review",
    ))

    # Article 30(3)(e) - UNRESTRICTED Audit and Access Rights
    requirements.append(ContractualRequirement(
        requirement_type=RequirementType.AUDIT_RIGHTS,
        category=RequirementCategory.CRITICAL,
        article_reference="Article 30(3)(e)",
        name="Unrestricted Audit and Access Rights",
        description="UNRESTRICTED rights of access, inspection and audit by financial entity and their competent authority",
        detailed_criteria=[
            "Unrestricted on-site audit rights",
            "Remote audit capabilities",
            "No cap on audit frequency for cause-based audits",
            "Access to relevant documentation",
            "Third-party auditor acceptance",
            "NCA inspection cooperation",
        ],
        applies_to_all=False,
        applies_to_critical_only=True,
        mandatory=True,
        verification_method="document_review",
    ))

    # Article 30(3)(f) - Exit Strategies
    requirements.append(ContractualRequirement(
        requirement_type=RequirementType.EXIT_STRATEGY,
        category=RequirementCategory.CRITICAL,
        article_reference="Article 30(3)(f)",
        name="Exit Strategies",
        description="Exit strategies including adequate transition periods and data portability",
        detailed_criteria=[
            "Documented exit strategy",
            "Adequate transition period",
            "Data return procedures",
            "Knowledge transfer requirements",
            "No vendor lock-in provisions",
        ],
        applies_to_all=False,
        applies_to_critical_only=True,
        mandatory=True,
        verification_method="document_review",
    ))

    # Article 30(3)(f) continued - Transition Support
    requirements.append(ContractualRequirement(
        requirement_type=RequirementType.TRANSITION_SUPPORT,
        category=RequirementCategory.CRITICAL,
        article_reference="Article 30(3)(f)",
        name="Transition Support",
        description="Transition assistance for migration to alternative provider or in-house",
        detailed_criteria=[
            "Transition support duration",
            "Technical assistance commitment",
            "Parallel run provisions",
            "Documentation transfer",
        ],
        applies_to_all=False,
        applies_to_critical_only=True,
        mandatory=True,
        verification_method="document_review",
    ))

    # Article 30(3)(g) - Supervisory Cooperation
    requirements.append(ContractualRequirement(
        requirement_type=RequirementType.NCA_ACCESS,
        category=RequirementCategory.CRITICAL,
        article_reference="Article 30(3)(g)",
        name="Supervisory Cooperation",
        description="Participation in supervisory oversight activities and cooperation with competent authorities",
        detailed_criteria=[
            "Information provision to NCAs on request",
            "No impediment to supervision",
            "Cooperation in supervisory activities",
            "Support for resolution planning",
        ],
        applies_to_all=False,
        applies_to_critical_only=True,
        mandatory=True,
        verification_method="document_review",
    ))

    # Article 30(3)(h) - Business Continuity Implementation
    requirements.append(ContractualRequirement(
        requirement_type=RequirementType.BUSINESS_CONTINUITY,
        category=RequirementCategory.CRITICAL,
        article_reference="Article 30(3)(h)",
        name="Business Continuity Implementation",
        description="Implementation and testing of business continuity measures ensuring service availability",
        detailed_criteria=[
            "Provider BCP implementation",
            "DR capabilities demonstrated",
            "Testing schedule and evidence",
            "Recovery time commitments (RTO/RPO)",
        ],
        applies_to_all=False,
        applies_to_critical_only=True,
        mandatory=True,
        verification_method="document_review",
    ))

    # Article 30(3)(i) - Security Measures
    requirements.append(ContractualRequirement(
        requirement_type=RequirementType.SECURITY_MEASURES,
        category=RequirementCategory.CRITICAL,
        article_reference="Article 30(3)(i)",
        name="ICT Security Measures",
        description="ICT security-related arrangements including implementation and testing of security measures",
        detailed_criteria=[
            "Security policy implementation",
            "Security testing provisions",
            "Vulnerability management",
            "Security certifications (SOC2, ISO27001)",
            "Security incident handling",
        ],
        applies_to_all=False,
        applies_to_critical_only=True,
        mandatory=True,
        verification_method="document_review",
    ))

    # Article 30(3)(j) - Subcontracting Provisions
    requirements.append(ContractualRequirement(
        requirement_type=RequirementType.SUBCONTRACTING,
        category=RequirementCategory.CRITICAL,
        article_reference="Article 30(3)(j)",
        name="Subcontracting Provisions",
        description="Conditions for subcontracting, including prior approval requirements and chain monitoring",
        detailed_criteria=[
            "Prior approval for critical subcontracting",
            "Notification of subcontractors",
            "Flow-down of requirements",
            "Subcontractor chain monitoring",
            "Right to object to new subcontractors",
        ],
        applies_to_all=False,
        applies_to_critical_only=True,
        mandatory=True,
        verification_method="document_review",
    ))

    return requirements


# =============================================================================
# Main Implementation
# =============================================================================

class DORAContractualRequirements:
    """
    DORA Article 30 Contractual Requirements Management.

    Implements comprehensive contract compliance management including:
    - Requirements tracking per Article 30(2) and 30(3)
    - Contract assessment and gap analysis
    - Amendment tracking
    - SLA definition and monitoring
    - Compliance reporting

    Usage:
        config = ContractualRequirementsConfig()
        manager = DORAContractualRequirements(config)

        # Register a contract
        contract = manager.register_contract(
            provider_name="Cloud Provider",
            supports_critical_function=True,
        )

        # Assess contract compliance
        assessment = manager.assess_contract(contract.contract_id)

        # Generate amendment requests
        amendments = manager.generate_amendments(contract.contract_id)
    """

    def __init__(self, config: Optional[ContractualRequirementsConfig] = None):
        """Initialize Contractual Requirements Management."""
        self.config = config or ContractualRequirementsConfig()

        # Load standard requirements
        self._requirements: Dict[str, ContractualRequirement] = {}
        for req in get_article_30_requirements():
            self._requirements[req.requirement_id] = req

        # Data stores
        self._contracts: Dict[str, ICTContract] = {}
        self._provisions: Dict[str, ContractProvision] = {}
        self._assessments: Dict[str, ContractAssessment] = {}
        self._gaps: Dict[str, ContractGap] = {}
        self._amendments: Dict[str, ContractAmendment] = {}
        self._slas: Dict[str, SLADefinition] = {}

        # Indexes
        self._provisions_by_contract: Dict[str, Set[str]] = {}
        self._gaps_by_contract: Dict[str, Set[str]] = {}
        self._amendments_by_contract: Dict[str, Set[str]] = {}
        self._slas_by_contract: Dict[str, Set[str]] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("DORAContractualRequirements initialized")

    # =========================================================================
    # Requirements Management
    # =========================================================================

    def get_all_requirements(self) -> List[ContractualRequirement]:
        """Get all DORA contractual requirements."""
        with self._lock:
            return list(self._requirements.values())

    def get_basic_requirements(self) -> List[ContractualRequirement]:
        """Get Article 30(2) basic requirements."""
        with self._lock:
            return [
                r for r in self._requirements.values()
                if r.category == RequirementCategory.BASIC
            ]

    def get_critical_requirements(self) -> List[ContractualRequirement]:
        """Get Article 30(3) critical function requirements."""
        with self._lock:
            return [
                r for r in self._requirements.values()
                if r.category == RequirementCategory.CRITICAL
            ]

    def get_applicable_requirements(
        self,
        supports_critical_function: bool,
    ) -> List[ContractualRequirement]:
        """Get requirements applicable to a contract type."""
        with self._lock:
            if supports_critical_function:
                return list(self._requirements.values())
            else:
                return [
                    r for r in self._requirements.values()
                    if r.category == RequirementCategory.BASIC
                ]

    def get_requirement(
        self,
        requirement_id: str,
    ) -> Optional[ContractualRequirement]:
        """Get requirement by ID."""
        with self._lock:
            return self._requirements.get(requirement_id)

    # =========================================================================
    # Contract Management
    # =========================================================================

    def register_contract(
        self,
        provider_name: str,
        provider_id: str = "",
        contract_reference: str = "",
        contract_type: str = "procurement",
        effective_date: Optional[str] = None,
        expiry_date: Optional[str] = None,
        supports_critical_function: bool = False,
        critical_functions: Optional[List[str]] = None,
        notice_period_days: int = 30,
    ) -> ICTContract:
        """
        Register an ICT service contract.

        Args:
            provider_name: Provider name
            provider_id: Provider ID (if in provider registry)
            contract_reference: Contract reference number
            contract_type: Contract type
            effective_date: Contract effective date
            expiry_date: Contract expiry date
            supports_critical_function: Whether supports critical function
            critical_functions: List of critical functions supported
            notice_period_days: Termination notice period

        Returns:
            Registered ICTContract
        """
        contract = ICTContract(
            provider_id=provider_id,
            provider_name=provider_name,
            contract_reference=contract_reference or f"CTR-{datetime.now().strftime('%Y%m%d')}",
            contract_type=contract_type,
            effective_date=effective_date or datetime.now(timezone.utc).isoformat(),
            expiry_date=expiry_date,
            supports_critical_function=supports_critical_function,
            critical_functions=critical_functions or [],
            notice_period_days=notice_period_days,
            status=ContractStatus.ACTIVE,
        )

        with self._lock:
            self._contracts[contract.contract_id] = contract
            self._provisions_by_contract[contract.contract_id] = set()
            self._gaps_by_contract[contract.contract_id] = set()
            self._amendments_by_contract[contract.contract_id] = set()
            self._slas_by_contract[contract.contract_id] = set()

        self._log_event("contract_registered", {
            "contract_id": contract.contract_id,
            "provider_name": provider_name,
            "supports_critical_function": supports_critical_function,
        })

        return contract

    def get_contract(
        self,
        contract_id: str,
    ) -> Optional[ICTContract]:
        """Get contract by ID."""
        with self._lock:
            return self._contracts.get(contract_id)

    def get_all_contracts(
        self,
        include_terminated: bool = False,
    ) -> List[ICTContract]:
        """Get all contracts."""
        with self._lock:
            contracts = list(self._contracts.values())
            if not include_terminated:
                contracts = [
                    c for c in contracts
                    if c.status != ContractStatus.TERMINATED
                ]
            return contracts

    def get_contracts_for_critical_functions(self) -> List[ICTContract]:
        """Get contracts supporting critical functions."""
        with self._lock:
            return [
                c for c in self._contracts.values()
                if c.supports_critical_function and c.status == ContractStatus.ACTIVE
            ]

    def get_contracts_needing_assessment(self) -> List[ICTContract]:
        """Get contracts due for assessment."""
        now = datetime.now(timezone.utc)

        with self._lock:
            result = []
            for contract in self._contracts.values():
                if contract.status != ContractStatus.ACTIVE:
                    continue

                # Check if never assessed
                if not contract.last_assessment_date:
                    result.append(contract)
                    continue

                # Calculate next assessment date
                last_date = datetime.fromisoformat(
                    contract.last_assessment_date.replace("Z", "+00:00")
                )
                if contract.supports_critical_function:
                    months = self.config.critical_contract_assessment_months
                else:
                    months = self.config.assessment_frequency_months

                next_date = last_date + timedelta(days=months * 30)
                reminder_date = next_date - timedelta(days=self.config.assessment_reminder_days)

                if now >= reminder_date:
                    result.append(contract)

            return result

    # =========================================================================
    # Provision Management
    # =========================================================================

    def add_contract_provision(
        self,
        contract_id: str,
        requirement_id: str,
        clause_reference: str,
        provision_summary: str,
        provision_text: str = "",
        compliance_status: ComplianceStatus = ComplianceStatus.NOT_ASSESSED,
        compliance_notes: str = "",
    ) -> Optional[ContractProvision]:
        """
        Add a provision mapping to a contract.

        Args:
            contract_id: Contract ID
            requirement_id: Requirement this provision addresses
            clause_reference: Contract clause reference
            provision_summary: Summary of provision
            provision_text: Full provision text (optional)
            compliance_status: Compliance status
            compliance_notes: Assessment notes

        Returns:
            ContractProvision or None if contract not found
        """
        with self._lock:
            if contract_id not in self._contracts:
                return None

            provision = ContractProvision(
                contract_id=contract_id,
                requirement_id=requirement_id,
                clause_reference=clause_reference,
                provision_summary=provision_summary,
                provision_text=provision_text,
                compliance_status=compliance_status,
                compliance_notes=compliance_notes,
            )

            self._provisions[provision.provision_id] = provision
            self._provisions_by_contract[contract_id].add(provision.provision_id)
            self._contracts[contract_id].provisions.append(provision.provision_id)

        return provision

    def assess_provision(
        self,
        provision_id: str,
        compliance_status: ComplianceStatus,
        compliance_notes: str,
        assessed_by: str,
    ) -> Optional[ContractProvision]:
        """Assess a provision's compliance."""
        with self._lock:
            if provision_id not in self._provisions:
                return None

            provision = self._provisions[provision_id]
            provision.compliance_status = compliance_status
            provision.compliance_notes = compliance_notes
            provision.assessed_by = assessed_by
            provision.assessed_date = datetime.now(timezone.utc).isoformat()

        return provision

    def get_provisions_for_contract(
        self,
        contract_id: str,
    ) -> List[ContractProvision]:
        """Get all provisions for a contract."""
        with self._lock:
            provision_ids = self._provisions_by_contract.get(contract_id, set())
            return [
                self._provisions[pid]
                for pid in provision_ids
                if pid in self._provisions
            ]

    # =========================================================================
    # Contract Assessment
    # =========================================================================

    def assess_contract(
        self,
        contract_id: str,
        assessment_type: str = "periodic",
        assessed_by: str = "",
    ) -> Optional[ContractAssessment]:
        """
        Perform comprehensive contract assessment against DORA requirements.

        Args:
            contract_id: Contract to assess
            assessment_type: Type of assessment
            assessed_by: Assessor name

        Returns:
            ContractAssessment or None if contract not found
        """
        with self._lock:
            if contract_id not in self._contracts:
                return None

            contract = self._contracts[contract_id]

            # Get applicable requirements
            applicable = self.get_applicable_requirements(
                contract.supports_critical_function
            )

            # Get existing provisions
            provisions = self.get_provisions_for_contract(contract_id)
            provisions_by_req = {
                p.requirement_id: p for p in provisions
            }

            assessment = ContractAssessment(
                contract_id=contract_id,
                provider_name=contract.provider_name,
                is_critical_function=contract.supports_critical_function,
                assessed_by=assessed_by,
                assessment_type=assessment_type,
                applicable_requirements=[r.requirement_id for r in applicable],
            )

            gaps_created = []

            # Assess each requirement
            for req in applicable:
                provision = provisions_by_req.get(req.requirement_id)

                if provision:
                    assessment.assessed_provisions.append(provision.provision_id)

                    if provision.compliance_status == ComplianceStatus.COMPLIANT:
                        assessment.compliant_count += 1
                    elif provision.compliance_status == ComplianceStatus.PARTIALLY_COMPLIANT:
                        assessment.partially_compliant_count += 1
                        # Create gap for partial compliance
                        gap = self._create_gap(
                            assessment.assessment_id,
                            contract_id,
                            req.requirement_id,
                            f"Partial compliance: {provision.compliance_notes}",
                            GapSeverity.MEDIUM,
                        )
                        gaps_created.append(gap.gap_id)
                    else:
                        assessment.non_compliant_count += 1
                        # Create gap for non-compliance
                        gap = self._create_gap(
                            assessment.assessment_id,
                            contract_id,
                            req.requirement_id,
                            f"Non-compliant: Requirement {req.name} not met",
                            GapSeverity.HIGH if req.mandatory else GapSeverity.MEDIUM,
                        )
                        gaps_created.append(gap.gap_id)
                else:
                    # No provision for this requirement
                    assessment.non_compliant_count += 1
                    gap = self._create_gap(
                        assessment.assessment_id,
                        contract_id,
                        req.requirement_id,
                        f"Missing provision for: {req.name}",
                        GapSeverity.CRITICAL if req.mandatory else GapSeverity.HIGH,
                    )
                    gaps_created.append(gap.gap_id)

            assessment.gaps = gaps_created

            # Calculate compliance score
            total = len(applicable)
            if total > 0:
                weighted = (
                    assessment.compliant_count +
                    (assessment.partially_compliant_count * 0.5)
                )
                assessment.compliance_score_pct = (weighted / total) * 100

            # Determine overall compliance
            if assessment.compliance_score_pct >= self.config.compliant_threshold_pct:
                assessment.overall_compliance = ComplianceStatus.COMPLIANT
            elif assessment.compliance_score_pct >= self.config.partially_compliant_threshold_pct:
                assessment.overall_compliance = ComplianceStatus.PARTIALLY_COMPLIANT
            else:
                assessment.overall_compliance = ComplianceStatus.NON_COMPLIANT

            # Generate recommendations
            assessment.recommendations = self._generate_recommendations(
                assessment, applicable
            )

            # Store assessment
            self._assessments[assessment.assessment_id] = assessment

            # Update contract
            contract.last_assessment_id = assessment.assessment_id
            contract.last_assessment_date = assessment.assessment_date
            contract.overall_compliance = assessment.overall_compliance
            contract.open_gaps = gaps_created

        self._log_event("assessment_completed", {
            "assessment_id": assessment.assessment_id,
            "contract_id": contract_id,
            "compliance_score": assessment.compliance_score_pct,
            "gaps_count": len(gaps_created),
        })

        return assessment

    def get_assessment(
        self,
        assessment_id: str,
    ) -> Optional[ContractAssessment]:
        """Get assessment by ID."""
        with self._lock:
            return self._assessments.get(assessment_id)

    def get_assessments_for_contract(
        self,
        contract_id: str,
    ) -> List[ContractAssessment]:
        """Get all assessments for a contract."""
        with self._lock:
            return [
                a for a in self._assessments.values()
                if a.contract_id == contract_id
            ]

    def approve_assessment(
        self,
        assessment_id: str,
        approved_by: str,
    ) -> Optional[ContractAssessment]:
        """Approve an assessment."""
        with self._lock:
            if assessment_id not in self._assessments:
                return None

            assessment = self._assessments[assessment_id]
            assessment.approved_by = approved_by
            assessment.approval_date = datetime.now(timezone.utc).isoformat()

        return assessment

    # =========================================================================
    # Gap Management
    # =========================================================================

    def _create_gap(
        self,
        assessment_id: str,
        contract_id: str,
        requirement_id: str,
        gap_description: str,
        severity: GapSeverity,
    ) -> ContractGap:
        """Create a compliance gap."""
        req = self._requirements.get(requirement_id)

        gap = ContractGap(
            assessment_id=assessment_id,
            contract_id=contract_id,
            requirement_id=requirement_id,
            gap_description=gap_description,
            severity=severity,
            article_reference=req.article_reference if req else "",
            remediation_recommendation=self._get_remediation_recommendation(requirement_id),
        )

        # Set deadline based on severity
        if severity == GapSeverity.CRITICAL:
            days = self.config.critical_gap_remediation_days
        else:
            days = self.config.gap_remediation_default_days

        gap.remediation_deadline = (
            datetime.now(timezone.utc) + timedelta(days=days)
        ).isoformat()

        self._gaps[gap.gap_id] = gap
        self._gaps_by_contract[contract_id].add(gap.gap_id)

        # Escalate if critical
        if severity == GapSeverity.CRITICAL and self.config.critical_gap_escalation:
            if self.config.escalation_callback:
                self.config.escalation_callback("critical_gap", {
                    "gap_id": gap.gap_id,
                    "contract_id": contract_id,
                    "description": gap_description,
                })

        return gap

    def get_gap(
        self,
        gap_id: str,
    ) -> Optional[ContractGap]:
        """Get gap by ID."""
        with self._lock:
            return self._gaps.get(gap_id)

    def get_gaps_for_contract(
        self,
        contract_id: str,
    ) -> List[ContractGap]:
        """Get all gaps for a contract."""
        with self._lock:
            gap_ids = self._gaps_by_contract.get(contract_id, set())
            return [
                self._gaps[gid]
                for gid in gap_ids
                if gid in self._gaps
            ]

    def get_open_gaps(self) -> List[ContractGap]:
        """Get all open gaps."""
        with self._lock:
            return [
                g for g in self._gaps.values()
                if g.remediation_status not in [
                    RemediationStatus.COMPLETED,
                    RemediationStatus.NOT_APPLICABLE
                ]
            ]

    def get_gaps_by_severity(
        self,
        severity: GapSeverity,
    ) -> List[ContractGap]:
        """Get gaps by severity."""
        with self._lock:
            return [g for g in self._gaps.values() if g.severity == severity]

    def update_gap_remediation(
        self,
        gap_id: str,
        status: RemediationStatus,
        owner: str = "",
        notes: str = "",
    ) -> Optional[ContractGap]:
        """Update gap remediation status."""
        with self._lock:
            if gap_id not in self._gaps:
                return None

            gap = self._gaps[gap_id]
            gap.remediation_status = status
            if owner:
                gap.remediation_owner = owner
            gap.last_updated = datetime.now(timezone.utc).isoformat()

            if status == RemediationStatus.COMPLETED:
                gap.remediation_completed_date = gap.last_updated

        return gap

    def _get_remediation_recommendation(
        self,
        requirement_id: str,
    ) -> str:
        """Get remediation recommendation for a requirement."""
        req = self._requirements.get(requirement_id)
        if not req:
            return "Review contract and add appropriate clause"

        recommendations = {
            # Article 30(2) Basic Requirements - ALL contracts
            RequirementType.SERVICE_DESCRIPTION: "Add detailed service description clause (Art. 30(2)(a))",
            RequirementType.DATA_LOCATION: "Add data processing and storage location clause (Art. 30(2)(b))",
            RequirementType.SERVICE_LEVELS: "Add availability/integrity/confidentiality provisions (Art. 30(2)(c))",
            RequirementType.DATA_ACCESS_RECOVERY: "Add data access, recovery and return provisions (Art. 30(2)(d)) - CRITICAL",
            RequirementType.SERVICE_LEVEL_DESCRIPTIONS: "Add service level descriptions with targets (Art. 30(2)(e))",
            RequirementType.INCIDENT_ASSISTANCE: "Add incident assistance obligations (Art. 30(2)(f))",
            RequirementType.AUTHORITY_COOPERATION: "Add NCA/resolution authority cooperation clause (Art. 30(2)(g))",
            RequirementType.TERMINATION_RIGHTS: "Add/clarify termination provisions and notice periods (Art. 30(2)(h))",
            RequirementType.TRAINING_PARTICIPATION: "Add security awareness training participation clause (Art. 30(2)(i))",

            # Article 30(3) Critical Function Requirements
            RequirementType.SLA_TARGETS: "Add full SLAs with quantitative targets (Art. 30(3)(a))",
            RequirementType.NOTICE_PERIODS: "Specify notice periods and reporting obligations (Art. 30(3)(b))",
            RequirementType.BCP_REQUIREMENTS: "Add provider BCP requirements clause (Art. 30(3)(c))",
            RequirementType.RESILIENCE_TESTING: "Add resilience testing participation clause (Art. 30(3)(d))",
            RequirementType.AUDIT_RIGHTS: "Add unrestricted audit rights clause (Art. 30(3)(e))",
            RequirementType.EXIT_STRATEGY: "Document exit strategy provisions (Art. 30(3)(f))",
            RequirementType.TRANSITION_SUPPORT: "Add transition assistance clause (Art. 30(3)(f))",
            RequirementType.NCA_ACCESS: "Add supervisory cooperation clause (Art. 30(3)(g))",
            RequirementType.BUSINESS_CONTINUITY: "Add BCP implementation requirements (Art. 30(3)(h))",
            RequirementType.SECURITY_MEASURES: "Add ICT security requirements clause (Art. 30(3)(i))",
            RequirementType.SUBCONTRACTING: "Add subcontracting provisions (Art. 30(3)(j))",

            # Additional
            RequirementType.REPORTING_OBLIGATIONS: "Add reporting obligations (part of Art. 30(3)(b))",
            RequirementType.PERFORMANCE_REMEDIATION: "Add SLA breach remediation clause",
        }

        return recommendations.get(req.requirement_type, "Review and amend contract")

    # =========================================================================
    # Amendment Management
    # =========================================================================

    def create_amendment_request(
        self,
        contract_id: str,
        gap_ids: List[str],
        proposed_changes: List[Dict[str, Any]],
        justification: str,
        created_by: str = "",
    ) -> Optional[ContractAmendment]:
        """
        Create a contract amendment request.

        Args:
            contract_id: Contract to amend
            gap_ids: Gaps this amendment addresses
            proposed_changes: List of proposed changes
            justification: Business justification
            created_by: Creator name

        Returns:
            ContractAmendment or None if contract not found
        """
        with self._lock:
            if contract_id not in self._contracts:
                return None

            contract = self._contracts[contract_id]

            # Get related requirements
            related_reqs = []
            for gap_id in gap_ids:
                gap = self._gaps.get(gap_id)
                if gap and gap.requirement_id not in related_reqs:
                    related_reqs.append(gap.requirement_id)

            amendment = ContractAmendment(
                contract_id=contract_id,
                provider_name=contract.provider_name,
                amendment_type="modify_clause",
                related_gaps=gap_ids,
                related_requirements=related_reqs,
                proposed_changes=proposed_changes,
                justification=justification,
                status="draft",
                created_by=created_by,
            )

            self._amendments[amendment.amendment_id] = amendment
            self._amendments_by_contract[contract_id].add(amendment.amendment_id)
            contract.pending_amendments.append(amendment.amendment_id)

        self._log_event("amendment_created", {
            "amendment_id": amendment.amendment_id,
            "contract_id": contract_id,
            "gaps_addressed": len(gap_ids),
        })

        return amendment

    def generate_amendments(
        self,
        contract_id: str,
    ) -> List[ContractAmendment]:
        """
        Auto-generate amendment requests for all open gaps.

        Returns list of generated amendment requests.
        """
        with self._lock:
            if contract_id not in self._contracts:
                return []

            contract = self._contracts[contract_id]
            gaps = self.get_gaps_for_contract(contract_id)
            open_gaps = [
                g for g in gaps
                if g.remediation_status not in [
                    RemediationStatus.COMPLETED,
                    RemediationStatus.NOT_APPLICABLE
                ]
            ]

            if not open_gaps:
                return []

            # Group gaps by requirement type for efficient amendments
            gaps_by_type: Dict[str, List[ContractGap]] = {}
            for gap in open_gaps:
                req = self._requirements.get(gap.requirement_id)
                if req:
                    key = req.requirement_type.value
                    if key not in gaps_by_type:
                        gaps_by_type[key] = []
                    gaps_by_type[key].append(gap)

            amendments = []
            for req_type, type_gaps in gaps_by_type.items():
                proposed_changes = []
                for gap in type_gaps:
                    proposed_changes.append({
                        "gap_id": gap.gap_id,
                        "requirement": gap.requirement_id,
                        "recommendation": gap.remediation_recommendation,
                    })

                amendment = self.create_amendment_request(
                    contract_id=contract_id,
                    gap_ids=[g.gap_id for g in type_gaps],
                    proposed_changes=proposed_changes,
                    justification=f"DORA Article 30 compliance - {req_type}",
                )
                if amendment:
                    amendments.append(amendment)

            return amendments

    def submit_amendment(
        self,
        amendment_id: str,
    ) -> Optional[ContractAmendment]:
        """Mark amendment as submitted to provider."""
        with self._lock:
            if amendment_id not in self._amendments:
                return None

            amendment = self._amendments[amendment_id]
            amendment.status = "submitted"
            amendment.submitted_date = datetime.now(timezone.utc).isoformat()

        return amendment

    def record_amendment_response(
        self,
        amendment_id: str,
        accepted: bool,
        counter_proposal: str = "",
        notes: str = "",
    ) -> Optional[ContractAmendment]:
        """Record provider response to amendment request."""
        with self._lock:
            if amendment_id not in self._amendments:
                return None

            amendment = self._amendments[amendment_id]
            amendment.provider_response_date = datetime.now(timezone.utc).isoformat()

            if accepted:
                amendment.status = "accepted"
            elif counter_proposal:
                amendment.status = "negotiating"
                amendment.provider_counter_proposal = counter_proposal
            else:
                amendment.status = "rejected"

            amendment.negotiation_notes = notes

        # Notify if configured
        if self.config.notify_on_amendment_response:
            if self.config.notification_callback:
                self.config.notification_callback("amendment_response", {
                    "amendment_id": amendment_id,
                    "status": amendment.status,
                })

        return amendment

    def complete_amendment(
        self,
        amendment_id: str,
    ) -> Optional[ContractAmendment]:
        """Mark amendment as implemented."""
        with self._lock:
            if amendment_id not in self._amendments:
                return None

            amendment = self._amendments[amendment_id]
            amendment.status = "implemented"
            amendment.implementation_date = datetime.now(timezone.utc).isoformat()

            # Mark related gaps as completed
            for gap_id in amendment.related_gaps:
                if gap_id in self._gaps:
                    self._gaps[gap_id].remediation_status = RemediationStatus.COMPLETED
                    self._gaps[gap_id].remediation_completed_date = amendment.implementation_date

            # Remove from pending
            contract = self._contracts.get(amendment.contract_id)
            if contract and amendment_id in contract.pending_amendments:
                contract.pending_amendments.remove(amendment_id)

        return amendment

    def get_amendment(
        self,
        amendment_id: str,
    ) -> Optional[ContractAmendment]:
        """Get amendment by ID."""
        with self._lock:
            return self._amendments.get(amendment_id)

    def get_amendments_for_contract(
        self,
        contract_id: str,
    ) -> List[ContractAmendment]:
        """Get all amendments for a contract."""
        with self._lock:
            amendment_ids = self._amendments_by_contract.get(contract_id, set())
            return [
                self._amendments[aid]
                for aid in amendment_ids
                if aid in self._amendments
            ]

    def get_pending_amendments(self) -> List[ContractAmendment]:
        """Get all pending amendments."""
        with self._lock:
            return [
                a for a in self._amendments.values()
                if a.status not in ["implemented", "rejected"]
            ]

    # =========================================================================
    # SLA Management
    # =========================================================================

    def add_sla(
        self,
        contract_id: str,
        service_name: str,
        metric_name: str,
        target_value: float,
        target_unit: str,
        measurement_period: str = "monthly",
        warning_threshold: float = 0.0,
        breach_threshold: float = 0.0,
        remediation_on_breach: str = "",
        service_credits: str = "",
    ) -> Optional[SLADefinition]:
        """Add SLA definition to a contract."""
        with self._lock:
            if contract_id not in self._contracts:
                return None

            sla = SLADefinition(
                contract_id=contract_id,
                service_name=service_name,
                metric_name=metric_name,
                target_value=target_value,
                target_unit=target_unit,
                measurement_period=measurement_period,
                warning_threshold=warning_threshold,
                breach_threshold=breach_threshold,
                remediation_on_breach=remediation_on_breach,
                service_credits=service_credits,
            )

            self._slas[sla.sla_id] = sla
            self._slas_by_contract[contract_id].add(sla.sla_id)
            self._contracts[contract_id].slas.append(sla.sla_id)

        return sla

    def get_slas_for_contract(
        self,
        contract_id: str,
    ) -> List[SLADefinition]:
        """Get all SLAs for a contract."""
        with self._lock:
            sla_ids = self._slas_by_contract.get(contract_id, set())
            return [
                self._slas[sid]
                for sid in sla_ids
                if sid in self._slas
            ]

    # =========================================================================
    # Reporting and Statistics
    # =========================================================================

    def _generate_recommendations(
        self,
        assessment: ContractAssessment,
        requirements: List[ContractualRequirement],
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on assessment."""
        recommendations = []

        if assessment.compliance_score_pct < 50:
            recommendations.append({
                "priority": "high",
                "recommendation": "Immediate contract review and amendment required",
                "rationale": f"Compliance score ({assessment.compliance_score_pct:.1f}%) is below minimum threshold",
            })

        if assessment.non_compliant_count > 0:
            recommendations.append({
                "priority": "high",
                "recommendation": f"Address {assessment.non_compliant_count} non-compliant requirements",
                "rationale": "Non-compliant requirements create regulatory risk",
            })

        if assessment.partially_compliant_count > 3:
            recommendations.append({
                "priority": "medium",
                "recommendation": "Review partially compliant provisions for strengthening",
                "rationale": "Multiple partial compliance areas indicate systemic gaps",
            })

        return recommendations

    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get overall compliance summary across all contracts."""
        with self._lock:
            active_contracts = [
                c for c in self._contracts.values()
                if c.status == ContractStatus.ACTIVE
            ]

            critical_contracts = [
                c for c in active_contracts
                if c.supports_critical_function
            ]

            all_open_gaps = self.get_open_gaps()

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "contracts": {
                    "total": len(self._contracts),
                    "active": len(active_contracts),
                    "critical_function": len(critical_contracts),
                    "by_compliance_status": {
                        status.value: sum(
                            1 for c in active_contracts
                            if c.overall_compliance == status
                        )
                        for status in ComplianceStatus
                    },
                },
                "assessments": {
                    "total": len(self._assessments),
                    "contracts_needing_assessment": len(self.get_contracts_needing_assessment()),
                },
                "gaps": {
                    "total_open": len(all_open_gaps),
                    "by_severity": {
                        severity.value: sum(
                            1 for g in all_open_gaps
                            if g.severity == severity
                        )
                        for severity in GapSeverity
                    },
                },
                "amendments": {
                    "total": len(self._amendments),
                    "pending": len(self.get_pending_amendments()),
                },
                "compliance_indicators": {
                    "all_contracts_assessed": all(
                        c.last_assessment_date is not None
                        for c in active_contracts
                    ),
                    "no_critical_gaps": sum(
                        1 for g in all_open_gaps
                        if g.severity == GapSeverity.CRITICAL
                    ) == 0,
                    "all_critical_contracts_compliant": all(
                        c.overall_compliance == ComplianceStatus.COMPLIANT
                        for c in critical_contracts
                    ),
                },
            }

    def generate_gap_report(
        self,
        contract_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate detailed gap report."""
        with self._lock:
            if contract_id:
                gaps = self.get_gaps_for_contract(contract_id)
            else:
                gaps = list(self._gaps.values())

            return {
                "report_date": datetime.now(timezone.utc).isoformat(),
                "contract_id": contract_id,
                "total_gaps": len(gaps),
                "gaps_by_severity": {
                    severity.value: [
                        {
                            "gap_id": g.gap_id,
                            "contract_id": g.contract_id,
                            "description": g.gap_description,
                            "article": g.article_reference,
                            "status": g.remediation_status.value,
                            "deadline": g.remediation_deadline,
                        }
                        for g in gaps
                        if g.severity == severity
                    ]
                    for severity in GapSeverity
                },
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

        log_file = self._log_path / f"contractual_events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_contractual_requirements(
    config: Optional[ContractualRequirementsConfig] = None,
) -> DORAContractualRequirements:
    """
    Create a DORAContractualRequirements instance.

    Args:
        config: Optional configuration

    Returns:
        Configured DORAContractualRequirements instance
    """
    return DORAContractualRequirements(config=config)


def get_requirement_types() -> List[RequirementType]:
    """Get all requirement types."""
    return list(RequirementType)


def get_basic_requirement_count() -> int:
    """Get count of basic requirements (Article 30(2))."""
    return len([
        r for r in get_article_30_requirements()
        if r.category == RequirementCategory.BASIC
    ])


def get_critical_requirement_count() -> int:
    """Get count of critical function requirements (Article 30(3))."""
    return len([
        r for r in get_article_30_requirements()
        if r.category == RequirementCategory.CRITICAL
    ])
