# -*- coding: utf-8 -*-
"""
DORA TLPT Tester Requirements Module (Article 27).

Regulation (EU) 2022/2554 Article 27 defines requirements for testers
carrying out threat-led penetration testing (TLPT):
    - Highest suitability and reputability
    - Technical and organisational capabilities
    - Expertise in threat intelligence, penetration testing, red team testing
    - Certification by accreditation body
    - Independence and no conflicts of interest
    - Professional indemnity insurance

Key Requirements per Article 27:
    1. Highest suitability and reputability (Article 27(1)(a))
    2. Technical and organisational capabilities (Article 27(1)(a))
    3. Expertise in threat intelligence (Article 27(1)(a)(i))
    4. Expertise in penetration testing (Article 27(1)(a)(ii))
    5. Expertise in red team testing (Article 27(1)(a)(iii))
    6. Certified by accreditation body (Article 27(1)(a))
    7. Professional indemnity insurance (Article 27(1)(b))
    8. Independence and no conflicts of interest (Article 27(2))

Internal Tester Provisions (Article 27(2)):
    - Internal testers allowed if approved by competent authority
    - No conflicts of interest
    - Threat intelligence MUST be from external provider
    - External testers required every 3rd test

References:
    - Article 27 DORA: https://www.digital-operational-resilience-act.com/Article_27.html
    - RTS on TLPT (CDR 2024/2698): https://eur-lex.europa.eu/eli/reg_del/2024/2698/oj
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class TesterType(Enum):
    """Type of TLPT tester."""
    INTERNAL = "internal"
    EXTERNAL = "external"


class TesterRole(Enum):
    """Roles in TLPT testing."""
    THREAT_INTELLIGENCE = "threat_intelligence"
    PENETRATION_TESTER = "penetration_tester"
    RED_TEAM_LEAD = "red_team_lead"
    RED_TEAM_OPERATOR = "red_team_operator"


class CertificationCategory(Enum):
    """Categories of security certifications."""
    PENETRATION_TESTING = "penetration_testing"
    RED_TEAMING = "red_teaming"
    THREAT_INTELLIGENCE = "threat_intelligence"
    SECURITY_MANAGEMENT = "security_management"
    INCIDENT_RESPONSE = "incident_response"
    CLOUD_SECURITY = "cloud_security"


class QualificationStatus(Enum):
    """Qualification validation status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SUSPENDED = "suspended"


class ConflictCheckResult(Enum):
    """Results of conflict of interest check."""
    NO_CONFLICT = "no_conflict"
    POTENTIAL_CONFLICT = "potential_conflict"
    CONFLICT_IDENTIFIED = "conflict_identified"
    CONFLICT_MITIGATED = "conflict_mitigated"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SecurityCertification:
    """
    Security certification per Article 27(1)(a).
    """
    certification_id: str = ""
    name: str = ""
    issuing_body: str = ""
    category: CertificationCategory = CertificationCategory.PENETRATION_TESTING

    # Certification details
    certification_number: str = ""
    issue_date: str = ""
    expiry_date: str = ""
    is_active: bool = True

    # Verification
    verified: bool = False
    verification_date: str = ""
    verification_method: str = ""  # badge_check, direct_verification, document_review

    # Metadata
    created_at: str = ""

    def __post_init__(self):
        if not self.certification_id:
            self.certification_id = f"CERT-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    @property
    def is_expired(self) -> bool:
        """Check if certification has expired."""
        if not self.expiry_date:
            return False
        expiry = datetime.fromisoformat(self.expiry_date.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) > expiry


@dataclass
class TesterExpertise:
    """
    Tester expertise record per Article 27(1)(a).
    """
    expertise_id: str = ""
    area: str = ""  # threat_intelligence, penetration_testing, red_teaming
    years_experience: float = 0.0

    # Skills
    specific_skills: List[str] = field(default_factory=list)
    tools_expertise: List[str] = field(default_factory=list)
    techniques_expertise: List[str] = field(default_factory=list)

    # Sector experience
    sector_experience: List[str] = field(default_factory=list)  # finance, banking, etc.

    # Evidence
    portfolio_references: List[str] = field(default_factory=list)
    testimonials: List[str] = field(default_factory=list)

    # Assessment
    proficiency_level: str = ""  # expert, advanced, intermediate
    assessed_by: str = ""
    assessment_date: str = ""

    def __post_init__(self):
        if not self.expertise_id:
            self.expertise_id = f"EXP-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class ConflictOfInterestDeclaration:
    """
    Conflict of interest declaration per Article 27(2).
    """
    declaration_id: str = ""
    tester_id: str = ""
    engagement_id: str = ""
    target_entity: str = ""

    # Declaration
    declaration_date: str = ""

    # Potential conflicts
    prior_relationships: List[Dict[str, Any]] = field(default_factory=list)
    # Format: {"entity": str, "relationship": str, "period": str, "details": str}

    financial_interests: List[Dict[str, Any]] = field(default_factory=list)
    # Format: {"type": str, "entity": str, "details": str}

    personal_relationships: List[Dict[str, Any]] = field(default_factory=list)
    other_engagements: List[Dict[str, Any]] = field(default_factory=list)

    # Assessment
    check_result: ConflictCheckResult = ConflictCheckResult.NO_CONFLICT
    assessment_notes: str = ""
    mitigation_measures: List[str] = field(default_factory=list)

    # Approval
    reviewed_by: str = ""
    review_date: str = ""
    approved: bool = False

    # Signature
    tester_signature: str = ""
    signature_date: str = ""

    def __post_init__(self):
        if not self.declaration_id:
            self.declaration_id = f"COI-{uuid.uuid4().hex[:8].upper()}"
        if not self.declaration_date:
            self.declaration_date = datetime.now(timezone.utc).isoformat()


@dataclass
class ProfessionalIndemnityInsurance:
    """
    Professional indemnity insurance per Article 27(1)(b).
    """
    insurance_id: str = ""
    policy_number: str = ""
    insurer: str = ""

    # Coverage
    coverage_amount_eur: float = 0.0
    coverage_type: str = ""  # professional_indemnity, cyber_liability
    coverage_scope: str = ""

    # Period
    effective_date: str = ""
    expiry_date: str = ""

    # Verification
    verified: bool = False
    verification_date: str = ""
    verified_by: str = ""
    certificate_reference: str = ""

    # Status
    is_active: bool = True

    def __post_init__(self):
        if not self.insurance_id:
            self.insurance_id = f"INS-{uuid.uuid4().hex[:8].upper()}"

    @property
    def is_expired(self) -> bool:
        """Check if insurance has expired."""
        if not self.expiry_date:
            return True
        expiry = datetime.fromisoformat(self.expiry_date.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) > expiry


@dataclass
class TLPTTester:
    """
    TLPT Tester per Article 27.
    """
    tester_id: str = ""
    name: str = ""
    organization: str = ""
    tester_type: TesterType = TesterType.EXTERNAL

    # Contact
    email: str = ""
    phone: str = ""

    # Roles
    roles: List[TesterRole] = field(default_factory=list)
    primary_role: TesterRole = TesterRole.PENETRATION_TESTER

    # Qualifications per Article 27(1)(a)
    certifications: List[str] = field(default_factory=list)  # Certification IDs
    expertise_areas: List[str] = field(default_factory=list)  # Expertise IDs

    # Insurance per Article 27(1)(b)
    insurance_id: str = ""

    # Approval
    qualification_status: QualificationStatus = QualificationStatus.PENDING
    qualification_date: str = ""
    qualified_by: str = ""
    qualification_notes: str = ""

    # NCA approval for internal testers per Article 27(2)
    nca_approved: bool = False
    nca_approval_date: str = ""
    nca_approval_reference: str = ""

    # Engagement history
    engagements_completed: List[str] = field(default_factory=list)
    total_engagements: int = 0

    # COI declarations
    coi_declarations: List[str] = field(default_factory=list)

    # Status
    is_active: bool = True

    # Metadata
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.tester_id:
            self.tester_id = f"TESTER-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class TesterOrganization:
    """
    TLPT Testing organization per Article 27.
    """
    org_id: str = ""
    name: str = ""
    legal_name: str = ""

    # Registration
    registration_number: str = ""
    registration_country: str = ""
    lei: str = ""  # Legal Entity Identifier

    # Contact
    address: str = ""
    primary_contact: str = ""
    email: str = ""
    phone: str = ""

    # Accreditation per Article 27(1)(a)
    accreditations: List[Dict[str, Any]] = field(default_factory=list)
    # Format: {"body": str, "accreditation": str, "number": str, "expiry": str}

    # Services
    services_offered: List[str] = field(default_factory=list)  # TI, pentest, red_team

    # Insurance
    insurance_id: str = ""
    insurance_coverage_eur: float = 0.0

    # Team
    team_members: List[str] = field(default_factory=list)  # Tester IDs
    total_testers: int = 0

    # Track record
    engagements_completed: int = 0
    financial_sector_experience: bool = True
    years_in_business: int = 0

    # Qualification
    qualification_status: QualificationStatus = QualificationStatus.PENDING
    qualified_services: List[str] = field(default_factory=list)

    # Status
    is_active: bool = True

    # Metadata
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.org_id:
            self.org_id = f"ORG-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class TesterQualificationAssessment:
    """
    Assessment of tester qualifications per Article 27.
    """
    assessment_id: str = ""
    tester_id: str = ""

    # Assessment date
    assessment_date: str = ""
    assessor: str = ""

    # Article 27(1)(a) checks
    suitability_check: bool = False
    suitability_notes: str = ""

    reputability_check: bool = False
    reputability_notes: str = ""

    # Technical capabilities
    technical_capability_check: bool = False
    technical_capability_notes: str = ""

    # Organisational capabilities
    organisational_capability_check: bool = False
    organisational_capability_notes: str = ""

    # Expertise checks
    ti_expertise_check: bool = False
    ti_expertise_notes: str = ""

    pentest_expertise_check: bool = False
    pentest_expertise_notes: str = ""

    redteam_expertise_check: bool = False
    redteam_expertise_notes: str = ""

    # Certification check
    certification_check: bool = False
    certification_notes: str = ""
    required_certifications: List[str] = field(default_factory=list)
    verified_certifications: List[str] = field(default_factory=list)

    # Insurance check per Article 27(1)(b)
    insurance_check: bool = False
    insurance_notes: str = ""

    # COI check per Article 27(2)
    coi_check: bool = False
    coi_notes: str = ""

    # Overall result
    overall_result: QualificationStatus = QualificationStatus.PENDING
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.assessment_id:
            self.assessment_id = f"ASSESS-{uuid.uuid4().hex[:8].upper()}"
        if not self.assessment_date:
            self.assessment_date = datetime.now(timezone.utc).isoformat()


@dataclass
class InternalTesterApproval:
    """
    Internal tester approval per Article 27(2).
    """
    approval_id: str = ""
    tester_id: str = ""
    entity_name: str = ""

    # Request
    request_date: str = ""
    requested_by: str = ""
    justification: str = ""

    # Article 27(2) conditions
    no_conflict_of_interest: bool = False
    external_ti_committed: bool = False  # TI must be external
    previous_external_engagement: str = ""  # Every 3rd must be external

    # Supporting documentation
    documentation: List[str] = field(default_factory=list)
    coi_declaration_id: str = ""

    # NCA submission
    submitted_to_nca: bool = False
    submission_date: str = ""
    nca_name: str = ""
    submission_reference: str = ""

    # Approval
    approved: bool = False
    approval_date: str = ""
    approval_reference: str = ""
    conditions: List[str] = field(default_factory=list)

    # Validity
    valid_from: str = ""
    valid_until: str = ""

    def __post_init__(self):
        if not self.approval_id:
            self.approval_id = f"ITA-{uuid.uuid4().hex[:8].upper()}"


# =============================================================================
# Recognized Certifications per Article 27
# =============================================================================

RECOGNIZED_CERTIFICATIONS = {
    CertificationCategory.PENETRATION_TESTING: [
        {"name": "OSCP", "full_name": "Offensive Security Certified Professional", "body": "Offensive Security"},
        {"name": "OSCE", "full_name": "Offensive Security Certified Expert", "body": "Offensive Security"},
        {"name": "OSWE", "full_name": "Offensive Security Web Expert", "body": "Offensive Security"},
        {"name": "OSEE", "full_name": "Offensive Security Exploitation Expert", "body": "Offensive Security"},
        {"name": "GPEN", "full_name": "GIAC Penetration Tester", "body": "GIAC/SANS"},
        {"name": "GWAPT", "full_name": "GIAC Web Application Penetration Tester", "body": "GIAC/SANS"},
        {"name": "GXPN", "full_name": "GIAC Exploit Researcher and Advanced Penetration Tester", "body": "GIAC/SANS"},
        {"name": "CEH", "full_name": "Certified Ethical Hacker", "body": "EC-Council"},
        {"name": "LPT", "full_name": "Licensed Penetration Tester", "body": "EC-Council"},
        {"name": "CPT", "full_name": "Certified Penetration Tester", "body": "IACRB"},
    ],
    CertificationCategory.RED_TEAMING: [
        {"name": "CRTO", "full_name": "Certified Red Team Operator", "body": "Zero-Point Security"},
        {"name": "CRTL", "full_name": "Certified Red Team Lead", "body": "Zero-Point Security"},
        {"name": "GRTP", "full_name": "GIAC Red Team Professional", "body": "GIAC/SANS"},
        {"name": "CREST CRT", "full_name": "CREST Certified Red Team", "body": "CREST"},
        {"name": "CREST CCT", "full_name": "CREST Certified Simulated Attack Manager", "body": "CREST"},
    ],
    CertificationCategory.THREAT_INTELLIGENCE: [
        {"name": "GCTI", "full_name": "GIAC Cyber Threat Intelligence", "body": "GIAC/SANS"},
        {"name": "CTIA", "full_name": "Certified Threat Intelligence Analyst", "body": "EC-Council"},
        {"name": "CREST CTIA", "full_name": "CREST Certified Threat Intelligence Analyst", "body": "CREST"},
    ],
    CertificationCategory.SECURITY_MANAGEMENT: [
        {"name": "CISSP", "full_name": "Certified Information Systems Security Professional", "body": "(ISC)2"},
        {"name": "CISM", "full_name": "Certified Information Security Manager", "body": "ISACA"},
        {"name": "CISA", "full_name": "Certified Information Systems Auditor", "body": "ISACA"},
    ],
}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TesterManagementConfig:
    """Configuration for TLPT Tester Management."""
    # Certification requirements
    require_certification: bool = True
    minimum_certifications_pentest: int = 1
    minimum_certifications_redteam: int = 1
    minimum_certifications_ti: int = 1

    # Experience requirements
    minimum_experience_years: float = 3.0
    minimum_financial_sector_engagements: int = 2

    # Insurance requirements
    require_insurance: bool = True
    minimum_coverage_eur: float = 1_000_000

    # Internal tester rules per Article 27(2)
    allow_internal_testers: bool = True
    max_consecutive_internal: int = 2
    require_external_ti: bool = True

    # COI requirements
    coi_declaration_required: bool = True
    coi_lookback_years: int = 3

    # Logging
    log_path: str = "logs/dora/tester_management"


# =============================================================================
# Main Class Implementation
# =============================================================================

class DORATestermanagement:
    """
    DORA Article 27 compliant TLPT Tester Management.

    Provides:
    - Tester registration and qualification
    - Certification verification
    - Conflict of interest management
    - Insurance verification
    - Internal tester approval workflow
    - Organization accreditation

    Article 27 Requirements:
    1. Highest suitability and reputability (Art. 27(1)(a))
    2. Technical and organisational capabilities (Art. 27(1)(a))
    3. Expertise verification (Art. 27(1)(a)(i-iii))
    4. Certification by accreditation body (Art. 27(1)(a))
    5. Professional indemnity insurance (Art. 27(1)(b))
    6. Independence and no conflicts of interest (Art. 27(2))

    Internal Tester Rules (Art. 27(2)):
    - Approved by competent authority
    - No conflicts of interest
    - External TI provider mandatory
    - External testers every 3rd engagement

    Usage:
        config = TesterManagementConfig()
        tester_mgmt = DORATestermanagement(config)

        # Register tester
        tester = tester_mgmt.register_tester(
            name="John Smith",
            organization="RedTeam Corp",
            roles=[TesterRole.RED_TEAM_OPERATOR],
        )

        # Add certification
        cert = tester_mgmt.add_certification(
            tester_id=tester.tester_id,
            name="OSCP",
            issuing_body="Offensive Security",
        )

        # Validate qualifications
        result = tester_mgmt.validate_tester_qualifications(tester.tester_id)
    """

    def __init__(self, config: Optional[TesterManagementConfig] = None):
        """Initialize Tester Management."""
        self.config = config or TesterManagementConfig()

        # Data stores
        self._testers: Dict[str, TLPTTester] = {}
        self._organizations: Dict[str, TesterOrganization] = {}
        self._certifications: Dict[str, SecurityCertification] = {}
        self._expertise: Dict[str, TesterExpertise] = {}
        self._insurance: Dict[str, ProfessionalIndemnityInsurance] = {}
        self._coi_declarations: Dict[str, ConflictOfInterestDeclaration] = {}
        self._assessments: Dict[str, TesterQualificationAssessment] = {}
        self._internal_approvals: Dict[str, InternalTesterApproval] = {}

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("DORATestermanagement initialized")

    # =========================================================================
    # Tester Registration
    # =========================================================================

    def register_tester(
        self,
        name: str,
        organization: str,
        tester_type: TesterType = TesterType.EXTERNAL,
        roles: Optional[List[TesterRole]] = None,
        email: str = "",
        phone: str = "",
    ) -> TLPTTester:
        """
        Register a TLPT tester.

        Args:
            name: Tester name
            organization: Organization name
            tester_type: Internal or external
            roles: Tester roles
            email: Email address
            phone: Phone number

        Returns:
            Created TLPTTester
        """
        tester = TLPTTester(
            name=name,
            organization=organization,
            tester_type=tester_type,
            roles=roles or [TesterRole.PENETRATION_TESTER],
            primary_role=roles[0] if roles else TesterRole.PENETRATION_TESTER,
            email=email,
            phone=phone,
            qualification_status=QualificationStatus.PENDING,
        )

        self._testers[tester.tester_id] = tester

        self._log_event("tester_registered", {
            "tester_id": tester.tester_id,
            "name": name,
            "organization": organization,
            "type": tester_type.value,
        })

        logger.info(f"Tester registered: {tester.tester_id} - {name}")
        return tester

    def get_tester(self, tester_id: str) -> Optional[TLPTTester]:
        """Get tester by ID."""
        return self._testers.get(tester_id)

    def get_testers_by_role(self, role: TesterRole) -> List[TLPTTester]:
        """Get testers by role."""
        return [t for t in self._testers.values() if role in t.roles and t.is_active]

    def get_qualified_testers(
        self,
        role: Optional[TesterRole] = None,
        tester_type: Optional[TesterType] = None,
    ) -> List[TLPTTester]:
        """Get qualified testers."""
        testers = [
            t for t in self._testers.values()
            if t.qualification_status == QualificationStatus.APPROVED and t.is_active
        ]

        if role:
            testers = [t for t in testers if role in t.roles]
        if tester_type:
            testers = [t for t in testers if t.tester_type == tester_type]

        return testers

    # =========================================================================
    # Certification Management (Article 27(1)(a))
    # =========================================================================

    def add_certification(
        self,
        tester_id: str,
        name: str,
        issuing_body: str,
        category: CertificationCategory = CertificationCategory.PENETRATION_TESTING,
        certification_number: str = "",
        issue_date: Optional[str] = None,
        expiry_date: Optional[str] = None,
    ) -> SecurityCertification:
        """
        Add certification for a tester.

        Args:
            tester_id: Tester ID
            name: Certification name (e.g., OSCP)
            issuing_body: Issuing body
            category: Certification category
            certification_number: Certification number
            issue_date: Issue date
            expiry_date: Expiry date

        Returns:
            Created SecurityCertification
        """
        if tester_id not in self._testers:
            raise ValueError(f"Tester {tester_id} not found")

        cert = SecurityCertification(
            name=name,
            issuing_body=issuing_body,
            category=category,
            certification_number=certification_number,
            issue_date=issue_date or datetime.now(timezone.utc).isoformat(),
            expiry_date=expiry_date or "",
        )

        self._certifications[cert.certification_id] = cert

        # Link to tester
        tester = self._testers[tester_id]
        tester.certifications.append(cert.certification_id)
        tester.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("certification_added", {
            "certification_id": cert.certification_id,
            "tester_id": tester_id,
            "name": name,
        })

        return cert

    def verify_certification(
        self,
        certification_id: str,
        verification_method: str = "document_review",
        verified_by: str = "",
    ) -> SecurityCertification:
        """
        Verify a certification.

        Args:
            certification_id: Certification ID
            verification_method: How it was verified
            verified_by: Verifier

        Returns:
            Updated SecurityCertification
        """
        if certification_id not in self._certifications:
            raise ValueError(f"Certification {certification_id} not found")

        cert = self._certifications[certification_id]
        cert.verified = True
        cert.verification_date = datetime.now(timezone.utc).isoformat()
        cert.verification_method = verification_method

        return cert

    def get_certification(self, certification_id: str) -> Optional[SecurityCertification]:
        """Get certification by ID."""
        return self._certifications.get(certification_id)

    def get_certifications_for_tester(
        self,
        tester_id: str,
    ) -> List[SecurityCertification]:
        """Get all certifications for a tester."""
        if tester_id not in self._testers:
            return []

        tester = self._testers[tester_id]
        return [
            self._certifications[cid]
            for cid in tester.certifications
            if cid in self._certifications
        ]

    def check_certification_requirements(
        self,
        tester_id: str,
        role: TesterRole,
    ) -> Dict[str, Any]:
        """
        Check if tester meets certification requirements for a role.

        Args:
            tester_id: Tester ID
            role: Role to check

        Returns:
            Certification check result
        """
        certs = self.get_certifications_for_tester(tester_id)
        valid_certs = [c for c in certs if not c.is_expired and c.verified]

        # Requirements by role
        required = {
            TesterRole.THREAT_INTELLIGENCE: CertificationCategory.THREAT_INTELLIGENCE,
            TesterRole.PENETRATION_TESTER: CertificationCategory.PENETRATION_TESTING,
            TesterRole.RED_TEAM_LEAD: CertificationCategory.RED_TEAMING,
            TesterRole.RED_TEAM_OPERATOR: CertificationCategory.PENETRATION_TESTING,
        }

        required_category = required.get(role)
        matching_certs = [c for c in valid_certs if c.category == required_category]

        return {
            "tester_id": tester_id,
            "role": role.value,
            "required_category": required_category.value if required_category else None,
            "total_certifications": len(certs),
            "valid_certifications": len(valid_certs),
            "matching_certifications": len(matching_certs),
            "meets_requirement": len(matching_certs) >= 1,
            "certificates": [{"name": c.name, "verified": c.verified, "expired": c.is_expired} for c in matching_certs],
        }

    # =========================================================================
    # Expertise Management (Article 27(1)(a)(i-iii))
    # =========================================================================

    def add_expertise(
        self,
        tester_id: str,
        area: str,
        years_experience: float,
        specific_skills: Optional[List[str]] = None,
        tools_expertise: Optional[List[str]] = None,
        sector_experience: Optional[List[str]] = None,
        proficiency_level: str = "advanced",
    ) -> TesterExpertise:
        """
        Add expertise record for a tester.

        Args:
            tester_id: Tester ID
            area: Expertise area
            years_experience: Years of experience
            specific_skills: Specific skills
            tools_expertise: Tools expertise
            sector_experience: Sector experience
            proficiency_level: Proficiency level

        Returns:
            Created TesterExpertise
        """
        if tester_id not in self._testers:
            raise ValueError(f"Tester {tester_id} not found")

        expertise = TesterExpertise(
            area=area,
            years_experience=years_experience,
            specific_skills=specific_skills or [],
            tools_expertise=tools_expertise or [],
            sector_experience=sector_experience or [],
            proficiency_level=proficiency_level,
        )

        self._expertise[expertise.expertise_id] = expertise

        # Link to tester
        tester = self._testers[tester_id]
        tester.expertise_areas.append(expertise.expertise_id)
        tester.updated_at = datetime.now(timezone.utc).isoformat()

        return expertise

    def get_expertise_for_tester(
        self,
        tester_id: str,
    ) -> List[TesterExpertise]:
        """Get all expertise records for a tester."""
        if tester_id not in self._testers:
            return []

        tester = self._testers[tester_id]
        return [
            self._expertise[eid]
            for eid in tester.expertise_areas
            if eid in self._expertise
        ]

    # =========================================================================
    # Insurance Management (Article 27(1)(b))
    # =========================================================================

    def add_insurance(
        self,
        tester_id: str,
        policy_number: str,
        insurer: str,
        coverage_amount_eur: float,
        effective_date: str,
        expiry_date: str,
        coverage_type: str = "professional_indemnity",
    ) -> ProfessionalIndemnityInsurance:
        """
        Add insurance for a tester.

        Args:
            tester_id: Tester ID
            policy_number: Policy number
            insurer: Insurance company
            coverage_amount_eur: Coverage amount
            effective_date: Effective date
            expiry_date: Expiry date
            coverage_type: Coverage type

        Returns:
            Created ProfessionalIndemnityInsurance
        """
        if tester_id not in self._testers:
            raise ValueError(f"Tester {tester_id} not found")

        insurance = ProfessionalIndemnityInsurance(
            policy_number=policy_number,
            insurer=insurer,
            coverage_amount_eur=coverage_amount_eur,
            coverage_type=coverage_type,
            effective_date=effective_date,
            expiry_date=expiry_date,
        )

        self._insurance[insurance.insurance_id] = insurance

        # Link to tester
        tester = self._testers[tester_id]
        tester.insurance_id = insurance.insurance_id
        tester.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("insurance_added", {
            "insurance_id": insurance.insurance_id,
            "tester_id": tester_id,
            "coverage_eur": coverage_amount_eur,
        })

        return insurance

    def verify_insurance(
        self,
        insurance_id: str,
        certificate_reference: str,
        verified_by: str,
    ) -> ProfessionalIndemnityInsurance:
        """Verify insurance policy."""
        if insurance_id not in self._insurance:
            raise ValueError(f"Insurance {insurance_id} not found")

        insurance = self._insurance[insurance_id]
        insurance.verified = True
        insurance.verification_date = datetime.now(timezone.utc).isoformat()
        insurance.verified_by = verified_by
        insurance.certificate_reference = certificate_reference

        return insurance

    def check_insurance_requirements(
        self,
        tester_id: str,
    ) -> Dict[str, Any]:
        """Check if tester meets insurance requirements."""
        if tester_id not in self._testers:
            return {"meets_requirement": False, "reason": "Tester not found"}

        tester = self._testers[tester_id]
        if not tester.insurance_id:
            return {"meets_requirement": False, "reason": "No insurance on file"}

        insurance = self._insurance.get(tester.insurance_id)
        if not insurance:
            return {"meets_requirement": False, "reason": "Insurance record not found"}

        issues = []
        if insurance.is_expired:
            issues.append("Insurance expired")
        if not insurance.verified:
            issues.append("Insurance not verified")
        if insurance.coverage_amount_eur < self.config.minimum_coverage_eur:
            issues.append(f"Coverage below minimum ({self.config.minimum_coverage_eur} EUR)")

        return {
            "meets_requirement": len(issues) == 0,
            "insurance_id": insurance.insurance_id,
            "coverage_eur": insurance.coverage_amount_eur,
            "verified": insurance.verified,
            "expired": insurance.is_expired,
            "issues": issues,
        }

    # =========================================================================
    # Conflict of Interest (Article 27(2))
    # =========================================================================

    def submit_coi_declaration(
        self,
        tester_id: str,
        engagement_id: str,
        target_entity: str,
        prior_relationships: Optional[List[Dict[str, Any]]] = None,
        financial_interests: Optional[List[Dict[str, Any]]] = None,
        personal_relationships: Optional[List[Dict[str, Any]]] = None,
        tester_signature: str = "",
    ) -> ConflictOfInterestDeclaration:
        """
        Submit conflict of interest declaration per Article 27(2).

        Args:
            tester_id: Tester ID
            engagement_id: TLPT engagement ID
            target_entity: Entity being tested
            prior_relationships: Prior relationships with entity
            financial_interests: Financial interests
            personal_relationships: Personal relationships
            tester_signature: Tester signature

        Returns:
            Created ConflictOfInterestDeclaration
        """
        if tester_id not in self._testers:
            raise ValueError(f"Tester {tester_id} not found")

        declaration = ConflictOfInterestDeclaration(
            tester_id=tester_id,
            engagement_id=engagement_id,
            target_entity=target_entity,
            prior_relationships=prior_relationships or [],
            financial_interests=financial_interests or [],
            personal_relationships=personal_relationships or [],
            tester_signature=tester_signature,
            signature_date=datetime.now(timezone.utc).isoformat() if tester_signature else "",
        )

        # Initial assessment
        if prior_relationships or financial_interests or personal_relationships:
            declaration.check_result = ConflictCheckResult.POTENTIAL_CONFLICT
        else:
            declaration.check_result = ConflictCheckResult.NO_CONFLICT

        self._coi_declarations[declaration.declaration_id] = declaration

        # Link to tester
        tester = self._testers[tester_id]
        tester.coi_declarations.append(declaration.declaration_id)
        tester.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("coi_declaration_submitted", {
            "declaration_id": declaration.declaration_id,
            "tester_id": tester_id,
            "engagement_id": engagement_id,
            "initial_result": declaration.check_result.value,
        })

        return declaration

    def review_coi_declaration(
        self,
        declaration_id: str,
        reviewed_by: str,
        check_result: ConflictCheckResult,
        assessment_notes: str = "",
        mitigation_measures: Optional[List[str]] = None,
        approved: bool = True,
    ) -> ConflictOfInterestDeclaration:
        """
        Review and approve COI declaration.

        Args:
            declaration_id: Declaration ID
            reviewed_by: Reviewer
            check_result: Check result
            assessment_notes: Assessment notes
            mitigation_measures: Mitigation measures if conflict
            approved: Whether approved

        Returns:
            Updated ConflictOfInterestDeclaration
        """
        if declaration_id not in self._coi_declarations:
            raise ValueError(f"Declaration {declaration_id} not found")

        declaration = self._coi_declarations[declaration_id]
        declaration.check_result = check_result
        declaration.assessment_notes = assessment_notes
        declaration.mitigation_measures = mitigation_measures or []
        declaration.reviewed_by = reviewed_by
        declaration.review_date = datetime.now(timezone.utc).isoformat()
        declaration.approved = approved

        self._log_event("coi_declaration_reviewed", {
            "declaration_id": declaration_id,
            "result": check_result.value,
            "approved": approved,
        })

        return declaration

    def get_coi_declaration(
        self,
        declaration_id: str,
    ) -> Optional[ConflictOfInterestDeclaration]:
        """Get COI declaration by ID."""
        return self._coi_declarations.get(declaration_id)

    # =========================================================================
    # Tester Qualification (Article 27)
    # =========================================================================

    def validate_tester_qualifications(
        self,
        tester_id: str,
        assessor: str = "",
    ) -> TesterQualificationAssessment:
        """
        Perform full qualification assessment per Article 27.

        Args:
            tester_id: Tester ID
            assessor: Assessor name

        Returns:
            TesterQualificationAssessment
        """
        if tester_id not in self._testers:
            raise ValueError(f"Tester {tester_id} not found")

        tester = self._testers[tester_id]
        assessment = TesterQualificationAssessment(
            tester_id=tester_id,
            assessor=assessor,
        )

        # Get tester data
        certifications = self.get_certifications_for_tester(tester_id)
        valid_certs = [c for c in certifications if not c.is_expired and c.verified]
        expertise = self.get_expertise_for_tester(tester_id)
        insurance_check = self.check_insurance_requirements(tester_id)

        # Suitability check
        assessment.suitability_check = tester.is_active
        assessment.suitability_notes = "Tester is active" if tester.is_active else "Tester is inactive"

        # Reputability check (based on engagement history)
        assessment.reputability_check = tester.total_engagements >= 0
        assessment.reputability_notes = f"{tester.total_engagements} engagements completed"

        # Technical capability check
        tech_expertise = [e for e in expertise if e.years_experience >= self.config.minimum_experience_years]
        assessment.technical_capability_check = len(tech_expertise) > 0
        assessment.technical_capability_notes = f"{len(tech_expertise)} areas with sufficient experience"

        # Certification check
        assessment.required_certifications = [
            cert["name"] for certs in RECOGNIZED_CERTIFICATIONS.values() for cert in certs
        ][:5]  # Top 5 as examples
        assessment.verified_certifications = [c.name for c in valid_certs]
        assessment.certification_check = len(valid_certs) >= self.config.minimum_certifications_pentest
        assessment.certification_notes = f"{len(valid_certs)} valid certifications"

        # Insurance check
        assessment.insurance_check = insurance_check.get("meets_requirement", False)
        assessment.insurance_notes = "; ".join(insurance_check.get("issues", [])) or "Insurance OK"

        # Expertise checks by role
        for role in tester.roles:
            if role == TesterRole.THREAT_INTELLIGENCE:
                ti_exp = [e for e in expertise if "threat_intelligence" in e.area.lower()]
                assessment.ti_expertise_check = len(ti_exp) > 0
                assessment.ti_expertise_notes = f"{len(ti_exp)} TI expertise records"
            elif role == TesterRole.PENETRATION_TESTER:
                pentest_exp = [e for e in expertise if "penetration" in e.area.lower()]
                assessment.pentest_expertise_check = len(pentest_exp) > 0
                assessment.pentest_expertise_notes = f"{len(pentest_exp)} pentest expertise records"
            elif role in (TesterRole.RED_TEAM_LEAD, TesterRole.RED_TEAM_OPERATOR):
                rt_exp = [e for e in expertise if "red_team" in e.area.lower()]
                assessment.redteam_expertise_check = len(rt_exp) > 0
                assessment.redteam_expertise_notes = f"{len(rt_exp)} red team expertise records"

        # COI check
        recent_coi = [
            self._coi_declarations[cid]
            for cid in tester.coi_declarations
            if cid in self._coi_declarations
        ]
        if recent_coi:
            last_coi = recent_coi[-1]
            assessment.coi_check = last_coi.approved and last_coi.check_result != ConflictCheckResult.CONFLICT_IDENTIFIED
            assessment.coi_notes = f"Last COI: {last_coi.check_result.value}"
        else:
            assessment.coi_check = True  # No COI on file
            assessment.coi_notes = "No COI declarations on file"

        # Overall result
        all_checks = [
            assessment.suitability_check,
            assessment.reputability_check,
            assessment.technical_capability_check,
            assessment.certification_check,
        ]

        if self.config.require_insurance:
            all_checks.append(assessment.insurance_check)

        if all(all_checks):
            assessment.overall_result = QualificationStatus.APPROVED
            assessment.summary = "Tester designed to meet Article 27 requirements (pending external validation)"
        else:
            assessment.overall_result = QualificationStatus.REJECTED
            failed = []
            if not assessment.suitability_check:
                failed.append("suitability")
            if not assessment.reputability_check:
                failed.append("reputability")
            if not assessment.technical_capability_check:
                failed.append("technical capability")
            if not assessment.certification_check:
                failed.append("certification")
            if not assessment.insurance_check:
                failed.append("insurance")
            assessment.summary = f"Failed checks: {', '.join(failed)}"

        self._assessments[assessment.assessment_id] = assessment

        # Update tester status
        tester.qualification_status = assessment.overall_result
        if assessment.overall_result == QualificationStatus.APPROVED:
            tester.qualification_date = datetime.now(timezone.utc).isoformat()
            tester.qualified_by = assessor

        tester.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("qualification_assessed", {
            "assessment_id": assessment.assessment_id,
            "tester_id": tester_id,
            "result": assessment.overall_result.value,
        })

        return assessment

    # =========================================================================
    # Internal Tester Approval (Article 27(2))
    # =========================================================================

    def request_internal_tester_approval(
        self,
        tester_id: str,
        entity_name: str,
        justification: str,
        nca_name: str,
        requested_by: str = "",
    ) -> InternalTesterApproval:
        """
        Request approval for internal tester per Article 27(2).

        Args:
            tester_id: Tester ID
            entity_name: Entity name
            justification: Justification for using internal tester
            nca_name: NCA name
            requested_by: Requester

        Returns:
            Created InternalTesterApproval
        """
        if tester_id not in self._testers:
            raise ValueError(f"Tester {tester_id} not found")

        tester = self._testers[tester_id]
        if tester.tester_type != TesterType.INTERNAL:
            raise ValueError("Approval only required for internal testers")

        approval = InternalTesterApproval(
            tester_id=tester_id,
            entity_name=entity_name,
            request_date=datetime.now(timezone.utc).isoformat(),
            requested_by=requested_by,
            justification=justification,
            nca_name=nca_name,
            external_ti_committed=self.config.require_external_ti,
        )

        self._internal_approvals[approval.approval_id] = approval

        self._log_event("internal_approval_requested", {
            "approval_id": approval.approval_id,
            "tester_id": tester_id,
            "entity": entity_name,
        })

        return approval

    def submit_to_nca(
        self,
        approval_id: str,
        submission_reference: str,
    ) -> InternalTesterApproval:
        """Record submission to NCA."""
        if approval_id not in self._internal_approvals:
            raise ValueError(f"Approval {approval_id} not found")

        approval = self._internal_approvals[approval_id]
        approval.submitted_to_nca = True
        approval.submission_date = datetime.now(timezone.utc).isoformat()
        approval.submission_reference = submission_reference

        return approval

    def record_nca_approval(
        self,
        approval_id: str,
        approval_reference: str,
        conditions: Optional[List[str]] = None,
        valid_until: Optional[str] = None,
    ) -> InternalTesterApproval:
        """
        Record NCA approval for internal tester.

        Args:
            approval_id: Approval ID
            approval_reference: NCA approval reference
            conditions: Any conditions attached
            valid_until: Validity period

        Returns:
            Updated InternalTesterApproval
        """
        if approval_id not in self._internal_approvals:
            raise ValueError(f"Approval {approval_id} not found")

        approval = self._internal_approvals[approval_id]
        now = datetime.now(timezone.utc).isoformat()

        approval.approved = True
        approval.approval_date = now
        approval.approval_reference = approval_reference
        approval.conditions = conditions or []
        approval.valid_from = now
        approval.valid_until = valid_until or ""

        # Update tester
        tester = self._testers.get(approval.tester_id)
        if tester:
            tester.nca_approved = True
            tester.nca_approval_date = now
            tester.nca_approval_reference = approval_reference
            tester.updated_at = now

        self._log_event("nca_approval_recorded", {
            "approval_id": approval_id,
            "tester_id": approval.tester_id,
            "reference": approval_reference,
        })

        return approval

    # =========================================================================
    # Organization Management
    # =========================================================================

    def register_organization(
        self,
        name: str,
        legal_name: str = "",
        registration_country: str = "",
        services_offered: Optional[List[str]] = None,
        accreditations: Optional[List[Dict[str, Any]]] = None,
    ) -> TesterOrganization:
        """
        Register a testing organization.

        Args:
            name: Organization name
            legal_name: Legal name
            registration_country: Country
            services_offered: Services offered
            accreditations: Accreditations

        Returns:
            Created TesterOrganization
        """
        org = TesterOrganization(
            name=name,
            legal_name=legal_name or name,
            registration_country=registration_country,
            services_offered=services_offered or [],
            accreditations=accreditations or [],
        )

        self._organizations[org.org_id] = org

        self._log_event("organization_registered", {
            "org_id": org.org_id,
            "name": name,
        })

        return org

    def get_organization(self, org_id: str) -> Optional[TesterOrganization]:
        """Get organization by ID."""
        return self._organizations.get(org_id)

    def qualify_organization(
        self,
        org_id: str,
        qualified_services: List[str],
        qualified_by: str = "",
    ) -> TesterOrganization:
        """Qualify organization for specific services."""
        if org_id not in self._organizations:
            raise ValueError(f"Organization {org_id} not found")

        org = self._organizations[org_id]
        org.qualification_status = QualificationStatus.APPROVED
        org.qualified_services = qualified_services
        org.updated_at = datetime.now(timezone.utc).isoformat()

        return org

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_tester_summary(
        self,
        tester_id: str,
    ) -> Dict[str, Any]:
        """Get comprehensive summary for a tester."""
        if tester_id not in self._testers:
            raise ValueError(f"Tester {tester_id} not found")

        tester = self._testers[tester_id]
        certifications = self.get_certifications_for_tester(tester_id)
        expertise = self.get_expertise_for_tester(tester_id)
        insurance_check = self.check_insurance_requirements(tester_id)

        return {
            "tester_id": tester_id,
            "name": tester.name,
            "organization": tester.organization,
            "type": tester.tester_type.value,
            "roles": [r.value for r in tester.roles],
            "qualification_status": tester.qualification_status.value,
            "certifications": {
                "total": len(certifications),
                "verified": sum(1 for c in certifications if c.verified),
                "valid": sum(1 for c in certifications if not c.is_expired),
                "list": [c.name for c in certifications],
            },
            "expertise": {
                "areas": [e.area for e in expertise],
                "total_years": sum(e.years_experience for e in expertise),
            },
            "insurance": insurance_check,
            "engagements_completed": tester.total_engagements,
            "nca_approved": tester.nca_approved if tester.tester_type == TesterType.INTERNAL else None,
        }

    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get Article 27 compliance summary."""
        total_testers = len(self._testers)
        qualified = sum(1 for t in self._testers.values() if t.qualification_status == QualificationStatus.APPROVED)
        internal = sum(1 for t in self._testers.values() if t.tester_type == TesterType.INTERNAL)
        internal_approved = sum(
            1 for t in self._testers.values()
            if t.tester_type == TesterType.INTERNAL and t.nca_approved
        )

        return {
            "article_reference": "Article 27",
            "assessment_date": datetime.now(timezone.utc).isoformat(),
            "testers": {
                "total": total_testers,
                "qualified": qualified,
                "qualification_rate": (qualified / total_testers * 100) if total_testers > 0 else 0,
            },
            "internal_testers": {
                "total": internal,
                "nca_approved": internal_approved,
            },
            "organizations": {
                "total": len(self._organizations),
                "qualified": sum(1 for o in self._organizations.values() if o.qualification_status == QualificationStatus.APPROVED),
            },
            "certifications": {
                "total": len(self._certifications),
                "verified": sum(1 for c in self._certifications.values() if c.verified),
            },
        }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a tester management event."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"tester_mgmt_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_tester_management(
    config: Optional[TesterManagementConfig] = None,
) -> DORATestermanagement:
    """
    Create a DORATestermanagement instance.

    Args:
        config: Optional configuration

    Returns:
        Configured DORATestermanagement instance
    """
    return DORATestermanagement(config=config)


def get_recognized_certifications() -> Dict[str, List[Dict[str, str]]]:
    """
    Get list of recognized certifications per Article 27.

    Returns:
        Dictionary of certifications by category
    """
    return {
        cat.value: certs
        for cat, certs in RECOGNIZED_CERTIFICATIONS.items()
    }


def get_tester_roles() -> List[TesterRole]:
    """
    Get list of tester roles.

    Returns:
        List of TesterRole enums
    """
    return list(TesterRole)


def get_certification_categories() -> List[CertificationCategory]:
    """
    Get list of certification categories.

    Returns:
        List of CertificationCategory enums
    """
    return list(CertificationCategory)
