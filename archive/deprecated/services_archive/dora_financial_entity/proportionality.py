# -*- coding: utf-8 -*-
"""
DORA Proportionality Assessment Module (Articles 4, 16).

WARNING: THIS MODULE IS FOR FINANCIAL ENTITY (FE) CLIENTS ONLY.
This platform is an ICT Provider / Software Provider (DORA Art. 30),
NOT a Financial Entity. Proportionality assessment (Art. 4, 16) applies to
Financial Entities, not to software providers. Use this module only for
B2B clients who ARE Financial Entities.

Article 4 - Proportionality Principle:
    DORA requirements apply in a manner proportionate to the size, risk profile,
    nature, scale and complexity of operations.

Article 16 - Simplified ICT Risk Management Framework:
    Certain entities may apply a simplified framework instead of Articles 5-15:
    (a) Small and non-interconnected investment firms
    (b) Payment institutions exempted per PSD2 Article 32(1)
    (c) Institutions exempted per CRD Article 2(5)(4-23)
    (d) Electronic money institutions exempted per EMD2 Article 9
    (e) Small IORPs

Microenterprises (EU Recommendation 2003/361):
    Definition: <10 employees AND (<€2M turnover OR <€2M balance sheet)
    Partial exemptions:
    - No third-party ICT risk strategy required (Article 28(2))
    - Simplified ICT risk management (Article 6(6))
    - No recurring incidents assessment (CDR 2024/1772 Article 11(3))
    - No management body training requirements (Article 5(4))

References:
    - Article 4 DORA: https://www.digital-operational-resilience-act.com/Article_4.html
    - Article 16 DORA: https://www.digital-operational-resilience-act.com/Article_16.html
    - EU SME Definition: https://ec.europa.eu/growth/smes/sme-definition_en
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class DORARegime(Enum):
    """Applicable DORA regime based on entity classification."""
    FULL = "full"  # Standard full requirements
    SIMPLIFIED = "simplified"  # Article 16 simplified framework
    MICROENTERPRISE = "microenterprise"  # Partial exemptions for microenterprises


class ExemptionType(Enum):
    """Types of DORA exemptions."""
    # Article 16 simplified framework exemptions
    SMALL_NON_INTERCONNECTED_INVESTMENT_FIRM = "small_non_interconnected_investment_firm"  # Art. 16(1)(a)
    PSD2_EXEMPTED_PAYMENT_INSTITUTION = "psd2_exempted_payment_institution"  # Art. 16(1)(b)
    CRD_EXEMPTED_INSTITUTION = "crd_exempted_institution"  # Art. 16(1)(c)
    EMD2_EXEMPTED_E_MONEY_INSTITUTION = "emd2_exempted_e_money_institution"  # Art. 16(1)(d)
    SMALL_IORP = "small_iorp"  # Art. 16(1)(e)

    # Microenterprise exemptions
    MICROENTERPRISE_NO_TPP_STRATEGY = "microenterprise_no_tpp_strategy"  # Art. 28(2)
    MICROENTERPRISE_SIMPLIFIED_RISK = "microenterprise_simplified_risk"  # Art. 6(6)
    MICROENTERPRISE_NO_RECURRING_INCIDENTS = "microenterprise_no_recurring_incidents"  # CDR 2024/1772 Art. 11(3)
    MICROENTERPRISE_NO_TRAINING_REQUIREMENT = "microenterprise_no_training"  # Art. 5(4)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class RegimeExemption:
    """
    A specific exemption that applies to the entity.

    Documents which DORA requirements are modified or waived.
    """
    exemption_type: ExemptionType
    dora_article: str  # e.g., "Article 16(1)(a)"
    description: str
    affected_requirements: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    documentation_required: List[str] = field(default_factory=list)


@dataclass
class EntityClassification:
    """
    Classification per DORA Article 4 (Proportionality) and Article 16.

    This determines which DORA regime applies to the entity.
    """
    classification_id: str = ""
    classification_date: str = ""

    # Entity identification
    legal_name: str = ""
    lei: Optional[str] = None

    # Size metrics (as of assessment date)
    employee_count: int = 0
    annual_turnover_eur: float = 0.0
    balance_sheet_eur: float = 0.0

    # Entity type
    entity_type: str = "investment_firm"  # or appropriate type
    mifid_authorization: str = ""
    nca_registration: str = ""

    # Article 16 exemption checks
    is_small_non_interconnected: bool = False  # Art. 16(1)(a)
    is_psd2_exempted: bool = False  # Art. 16(1)(b)
    is_crd_exempted: bool = False  # Art. 16(1)(c)
    is_emd_exempted: bool = False  # Art. 16(1)(d)
    is_small_iorp: bool = False  # Art. 16(1)(e)

    # Exemption documentation
    exemption_documentation: str = ""
    exemption_reference: str = ""
    member_state_decision: str = ""

    # Risk profile (Article 4 proportionality factors)
    risk_profile: str = "medium"  # "low", "medium", "high"
    operational_complexity: str = "medium"  # "low", "medium", "high"
    cross_border_operations: bool = False
    systemic_importance: bool = False

    # Assessment
    assessed_by: str = ""
    reviewed_by: str = ""
    nca_confirmed: bool = False
    confirmation_date: Optional[str] = None

    def __post_init__(self):
        if not self.classification_id:
            self.classification_id = f"CLASS-{uuid.uuid4().hex[:8].upper()}"
        if not self.classification_date:
            self.classification_date = datetime.now(timezone.utc).isoformat()

    @property
    def is_microenterprise(self) -> bool:
        """
        EU Recommendation 2003/361 definition.

        Microenterprise = <10 employees AND (<€2M turnover OR <€2M balance sheet)

        NOTE: The condition is OR for turnover/balance sheet, not AND.
        """
        return (
            self.employee_count < 10
            and (
                self.annual_turnover_eur < 2_000_000
                or self.balance_sheet_eur < 2_000_000
            )
        )

    @property
    def is_small_enterprise(self) -> bool:
        """
        EU Recommendation 2003/361 definition.

        Small enterprise = <50 employees AND (<€10M turnover OR <€10M balance sheet)
        """
        return (
            self.employee_count < 50
            and (
                self.annual_turnover_eur < 10_000_000
                or self.balance_sheet_eur < 10_000_000
            )
        )

    @property
    def is_medium_enterprise(self) -> bool:
        """
        EU Recommendation 2003/361 definition.

        Medium enterprise = <250 employees AND (<€50M turnover OR <€43M balance sheet)
        """
        return (
            self.employee_count < 250
            and (
                self.annual_turnover_eur < 50_000_000
                or self.balance_sheet_eur < 43_000_000
            )
        )

    @property
    def qualifies_for_simplified_framework(self) -> bool:
        """Check if entity qualifies for Article 16 simplified framework."""
        return any([
            self.is_small_non_interconnected,
            self.is_psd2_exempted,
            self.is_crd_exempted,
            self.is_emd_exempted,
            self.is_small_iorp,
        ])

    @property
    def applicable_regime(self) -> DORARegime:
        """Determine which DORA regime applies."""
        # Check for Article 16 simplified framework
        if self.qualifies_for_simplified_framework:
            return DORARegime.SIMPLIFIED

        # Check for microenterprise partial exemptions
        if self.is_microenterprise:
            return DORARegime.MICROENTERPRISE

        return DORARegime.FULL


@dataclass
class ProportionalityAssessment:
    """
    Complete proportionality assessment result.

    Documents the entity classification, applicable regime, and exemptions.
    """
    assessment_id: str = ""
    assessment_date: str = ""

    # Entity classification
    entity_classification: Optional[EntityClassification] = None

    # Regime determination
    applicable_regime: DORARegime = DORARegime.FULL
    regime_rationale: str = ""

    # Applicable exemptions
    exemptions: List[RegimeExemption] = field(default_factory=list)

    # Modified requirements
    full_requirements: List[str] = field(default_factory=list)
    simplified_requirements: List[str] = field(default_factory=list)
    exempted_requirements: List[str] = field(default_factory=list)

    # Proportionality factors considered (Article 4)
    proportionality_factors: Dict[str, Any] = field(default_factory=dict)

    # Documentation
    supporting_evidence: List[str] = field(default_factory=list)
    nca_confirmation_status: str = "not_required"  # "not_required", "pending", "confirmed"

    # Review
    assessed_by: str = ""
    legal_review_required: bool = True
    legal_review_completed: bool = False
    legal_reviewer: str = ""

    # Next steps
    recommended_actions: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.assessment_id:
            self.assessment_id = f"PROP-{uuid.uuid4().hex[:8].upper()}"
        if not self.assessment_date:
            self.assessment_date = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Microenterprise Exemptions
# =============================================================================

MICROENTERPRISE_EXEMPTIONS = [
    RegimeExemption(
        exemption_type=ExemptionType.MICROENTERPRISE_NO_TPP_STRATEGY,
        dora_article="Article 28(2)",
        description="No requirement for third-party ICT risk strategy",
        affected_requirements=[
            "Third-party ICT risk strategy (Article 28)",
            "ICT concentration risk strategy",
        ],
        conditions=["Entity qualifies as microenterprise per EU Recommendation 2003/361"],
    ),
    RegimeExemption(
        exemption_type=ExemptionType.MICROENTERPRISE_SIMPLIFIED_RISK,
        dora_article="Article 6(6)",
        description="Simplified ICT risk management framework requirements",
        affected_requirements=[
            "ICT risk management framework complexity",
            "Digital operational resilience strategy detail",
        ],
        conditions=["Entity qualifies as microenterprise per EU Recommendation 2003/361"],
    ),
    RegimeExemption(
        exemption_type=ExemptionType.MICROENTERPRISE_NO_RECURRING_INCIDENTS,
        dora_article="CDR 2024/1772 Article 11(3)",
        description="No recurring incidents assessment requirement",
        affected_requirements=[
            "Recurring incident assessment",
            "Monthly pattern analysis",
        ],
        conditions=["Entity qualifies as microenterprise per EU Recommendation 2003/361"],
    ),
    RegimeExemption(
        exemption_type=ExemptionType.MICROENTERPRISE_NO_TRAINING_REQUIREMENT,
        dora_article="Article 5(4)",
        description="No mandatory management body ICT training requirements",
        affected_requirements=[
            "Management body ICT skills training",
            "Mandatory training tracking",
        ],
        conditions=["Entity qualifies as microenterprise per EU Recommendation 2003/361"],
    ),
]


# =============================================================================
# Simplified Framework Requirements
# =============================================================================

SIMPLIFIED_FRAMEWORK_REQUIREMENTS = {
    "icT_risk_management": [
        "Maintain basic ICT risk management framework",
        "Document ICT systems and key dependencies",
        "Basic incident management procedures",
        "Simplified testing requirements",
    ],
    "incident_management": [
        "Classify ICT incidents",
        "Report major incidents to NCA",
        "Basic root cause analysis",
    ],
    "testing": [
        "Basic vulnerability assessments",
        "No mandatory TLPT requirement",
        "Simplified penetration testing scope",
    ],
    "third_party": [
        "Maintain Register of Information (still required)",
        "Basic due diligence on providers",
        "Standard contractual requirements",
    ],
}


# =============================================================================
# Proportionality Assessor
# =============================================================================

class ProportionalityAssessor:
    """
    Assess which DORA regime applies to an entity.

    Implements Article 4 (Proportionality) and Article 16 (Simplified Framework).

    Usage:
        assessor = ProportionalityAssessor()

        # Create entity classification
        entity = EntityClassification(
            legal_name="Trading Platform Ltd",
            employee_count=8,
            annual_turnover_eur=1_500_000,
            balance_sheet_eur=1_000_000,
            entity_type="investment_firm",
        )

        # Assess proportionality
        assessment = assessor.assess(entity)

        print(f"Applicable regime: {assessment.applicable_regime.value}")
        for exemption in assessment.exemptions:
            print(f"  - {exemption.description}")
    """

    def __init__(self):
        """Initialize proportionality assessor."""
        logger.info("ProportionalityAssessor initialized")

    def assess(
        self,
        entity: EntityClassification,
        include_proportionality_factors: bool = True,
    ) -> ProportionalityAssessment:
        """
        Assess the applicable DORA regime for an entity.

        Args:
            entity: Entity classification data
            include_proportionality_factors: Whether to assess Article 4 factors

        Returns:
            ProportionalityAssessment with detailed findings
        """
        assessment = ProportionalityAssessment(
            entity_classification=entity,
        )

        # Determine applicable regime
        regime = entity.applicable_regime
        assessment.applicable_regime = regime

        # Build regime rationale
        assessment.regime_rationale = self._build_regime_rationale(entity, regime)

        # Identify applicable exemptions
        assessment.exemptions = self._get_applicable_exemptions(entity, regime)

        # Categorize requirements
        self._categorize_requirements(assessment, entity, regime)

        # Assess proportionality factors (Article 4)
        if include_proportionality_factors:
            assessment.proportionality_factors = self._assess_proportionality_factors(entity)

        # Generate recommendations
        assessment.recommended_actions = self._generate_recommendations(entity, regime)

        logger.info(
            f"Proportionality assessment completed: {entity.legal_name} -> {regime.value}"
        )

        return assessment

    def _build_regime_rationale(
        self,
        entity: EntityClassification,
        regime: DORARegime,
    ) -> str:
        """Build explanation for regime determination."""
        if regime == DORARegime.SIMPLIFIED:
            reasons = []
            if entity.is_small_non_interconnected:
                reasons.append("Small and non-interconnected investment firm (Article 16(1)(a))")
            if entity.is_psd2_exempted:
                reasons.append("Payment institution exempted per PSD2 Article 32(1) (Article 16(1)(b))")
            if entity.is_crd_exempted:
                reasons.append("Institution exempted per CRD (Article 16(1)(c))")
            if entity.is_emd_exempted:
                reasons.append("E-money institution exempted per EMD2 (Article 16(1)(d))")
            if entity.is_small_iorp:
                reasons.append("Small IORP (Article 16(1)(e))")
            return f"Simplified framework applies because: {'; '.join(reasons)}"

        elif regime == DORARegime.MICROENTERPRISE:
            return (
                f"Microenterprise regime applies: "
                f"{entity.employee_count} employees (<10), "
                f"€{entity.annual_turnover_eur:,.0f} turnover or "
                f"€{entity.balance_sheet_eur:,.0f} balance sheet (<€2M). "
                f"Partial exemptions apply per Articles 5(4), 6(6), 28(2), and CDR 2024/1772 Art. 11(3)."
            )

        else:
            return (
                f"Full DORA regime applies: "
                f"Entity does not qualify for simplified framework (Article 16) "
                f"or microenterprise exemptions."
            )

    def _get_applicable_exemptions(
        self,
        entity: EntityClassification,
        regime: DORARegime,
    ) -> List[RegimeExemption]:
        """Get list of applicable exemptions."""
        exemptions = []

        # Article 16 simplified framework exemptions
        if regime == DORARegime.SIMPLIFIED:
            if entity.is_small_non_interconnected:
                exemptions.append(RegimeExemption(
                    exemption_type=ExemptionType.SMALL_NON_INTERCONNECTED_INVESTMENT_FIRM,
                    dora_article="Article 16(1)(a)",
                    description="Small and non-interconnected investment firm - simplified ICT framework",
                    affected_requirements=["Articles 5-15 replaced by Article 16 requirements"],
                    conditions=["Qualification as small and non-interconnected per IFR"],
                ))
            # Add other simplified framework exemptions as applicable

        # Microenterprise exemptions
        if entity.is_microenterprise:
            exemptions.extend(MICROENTERPRISE_EXEMPTIONS)

        return exemptions

    def _categorize_requirements(
        self,
        assessment: ProportionalityAssessment,
        entity: EntityClassification,
        regime: DORARegime,
    ) -> None:
        """Categorize requirements into full, simplified, and exempted."""
        # Core requirements that always apply
        always_applicable = [
            "Article 17 - ICT-related incident management",
            "Article 18 - Incident classification",
            "Article 19 - Major incident reporting",
            "Article 28 - Register of Information",
            "Article 30 - Key contractual provisions (limited)",
        ]

        if regime == DORARegime.FULL:
            assessment.full_requirements = [
                "Article 5 - Governance (full)",
                "Article 6 - ICT risk management framework (full)",
                "Article 7-15 - Full ICT requirements",
                "Article 24-25 - Full testing requirements",
                "Article 28 - Third-party risk management (full strategy)",
            ] + always_applicable
            assessment.simplified_requirements = []
            assessment.exempted_requirements = []

        elif regime == DORARegime.SIMPLIFIED:
            assessment.full_requirements = always_applicable
            assessment.simplified_requirements = [
                "Article 16 - Simplified ICT risk management framework",
                "Simplified testing requirements",
                "Reduced documentation requirements",
            ]
            assessment.exempted_requirements = [
                "Articles 5-15 (replaced by Article 16)",
                "Advanced TLPT requirements",
            ]

        elif regime == DORARegime.MICROENTERPRISE:
            assessment.full_requirements = always_applicable + [
                "Article 5 - Governance (except training)",
                "Article 6 - ICT risk management (simplified per Art. 6(6))",
                "Article 24-25 - Testing requirements",
            ]
            assessment.simplified_requirements = [
                "ICT risk management complexity (Article 6(6))",
            ]
            assessment.exempted_requirements = [
                "Management body training (Article 5(4))",
                "Third-party ICT risk strategy (Article 28(2))",
                "Recurring incident assessment (CDR 2024/1772 Art. 11(3))",
            ]

    def _assess_proportionality_factors(
        self,
        entity: EntityClassification,
    ) -> Dict[str, Any]:
        """
        Assess Article 4 proportionality factors.

        These factors should inform how requirements are implemented,
        even within a given regime.
        """
        return {
            "size": {
                "employee_count": entity.employee_count,
                "turnover_eur": entity.annual_turnover_eur,
                "balance_sheet_eur": entity.balance_sheet_eur,
                "category": "microenterprise" if entity.is_microenterprise else (
                    "small" if entity.is_small_enterprise else (
                        "medium" if entity.is_medium_enterprise else "large"
                    )
                ),
            },
            "risk_profile": {
                "assessment": entity.risk_profile,
                "cross_border": entity.cross_border_operations,
                "systemic_importance": entity.systemic_importance,
            },
            "complexity": {
                "operational_complexity": entity.operational_complexity,
            },
            "nature": {
                "entity_type": entity.entity_type,
            },
            "proportionality_notes": [
                "Requirements should be implemented proportionately per Article 4",
                "NCA may request proportionality justification",
                "Document proportionality considerations in compliance files",
            ],
        }

    def _generate_recommendations(
        self,
        entity: EntityClassification,
        regime: DORARegime,
    ) -> List[str]:
        """Generate recommended actions based on assessment."""
        recommendations = []

        # Common recommendations
        recommendations.append("Document this proportionality assessment")
        recommendations.append("Seek legal review of entity classification")

        if regime == DORARegime.MICROENTERPRISE:
            recommendations.extend([
                "Document microenterprise qualification with financial evidence",
                "Note exemptions in compliance documentation",
                "Monitor size metrics - loss of microenterprise status triggers full requirements",
            ])

        elif regime == DORARegime.SIMPLIFIED:
            recommendations.extend([
                "Document simplified framework qualification",
                "Obtain NCA confirmation if required",
                "Implement Article 16 requirements instead of Articles 5-15",
            ])

        else:
            recommendations.extend([
                "Implement full DORA requirements",
                "Consider proportionality in implementation approach",
                "Document proportionality rationale for each requirement",
            ])

        # Register of Information always required
        recommendations.append(
            "Prepare Register of Information - due 30 April 2025 (submitted via NCA to ESAs)"
        )

        return recommendations

    def check_microenterprise_status(
        self,
        employee_count: int,
        annual_turnover_eur: float,
        balance_sheet_eur: float,
    ) -> bool:
        """
        Quick check for microenterprise status.

        Per EU Recommendation 2003/361:
        Microenterprise = <10 employees AND (<€2M turnover OR <€2M balance sheet)

        Args:
            employee_count: Number of employees
            annual_turnover_eur: Annual turnover in EUR
            balance_sheet_eur: Total balance sheet in EUR

        Returns:
            True if entity qualifies as microenterprise
        """
        return (
            employee_count < 10
            and (annual_turnover_eur < 2_000_000 or balance_sheet_eur < 2_000_000)
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_proportionality_assessor() -> ProportionalityAssessor:
    """
    Create a ProportionalityAssessor instance.

    Returns:
        Configured ProportionalityAssessor instance
    """
    return ProportionalityAssessor()


def assess_entity_proportionality(
    legal_name: str,
    employee_count: int,
    annual_turnover_eur: float,
    balance_sheet_eur: float,
    entity_type: str = "investment_firm",
    is_small_non_interconnected: bool = False,
    is_psd2_exempted: bool = False,
    is_crd_exempted: bool = False,
    is_emd_exempted: bool = False,
    is_small_iorp: bool = False,
) -> ProportionalityAssessment:
    """
    Convenience function to assess entity proportionality.

    Args:
        legal_name: Legal name of entity
        employee_count: Number of employees
        annual_turnover_eur: Annual turnover in EUR
        balance_sheet_eur: Total balance sheet in EUR
        entity_type: Type of financial entity
        is_small_non_interconnected: Qualifies per Article 16(1)(a)
        is_psd2_exempted: Qualifies per Article 16(1)(b)
        is_crd_exempted: Qualifies per Article 16(1)(c)
        is_emd_exempted: Qualifies per Article 16(1)(d)
        is_small_iorp: Qualifies per Article 16(1)(e)

    Returns:
        ProportionalityAssessment with detailed findings
    """
    entity = EntityClassification(
        legal_name=legal_name,
        employee_count=employee_count,
        annual_turnover_eur=annual_turnover_eur,
        balance_sheet_eur=balance_sheet_eur,
        entity_type=entity_type,
        is_small_non_interconnected=is_small_non_interconnected,
        is_psd2_exempted=is_psd2_exempted,
        is_crd_exempted=is_crd_exempted,
        is_emd_exempted=is_emd_exempted,
        is_small_iorp=is_small_iorp,
    )

    assessor = ProportionalityAssessor()
    return assessor.assess(entity)
