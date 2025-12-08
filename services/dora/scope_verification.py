# -*- coding: utf-8 -*-
"""
DORA Scope Verification Module (Article 2).

Regulation (EU) 2022/2554 Article 2 defines the scope of application.
This module verifies if an entity falls within DORA scope.

CRITICAL: Verify if DORA applies BEFORE any implementation work.

Article 2(1) lists 21 types of financial entities subject to DORA:
    (a) Credit institutions
    (b) Payment institutions
    (c) Account information service providers
    (d) Electronic money institutions
    (e) Investment firms
    (f) Crypto-asset service providers
    (g) Central securities depositories
    (h) Central counterparties
    (i) Trading venues
    (j) Trade repositories
    (k) Managers of alternative investment funds
    (l) Management companies
    (m) Data reporting service providers
    (n) Insurance and reinsurance undertakings
    (o) Insurance intermediaries
    (p) Institutions for occupational retirement provision
    (q) Credit rating agencies
    (r) Administrators of critical benchmarks
    (s) Crowdfunding service providers
    (t) Securitisation repositories
    (u) ICT third-party service providers (specific oversight rules)

References:
    - Article 2 DORA: https://www.digital-operational-resilience-act.com/Article_2.html
    - EUR-Lex Full Text: https://eur-lex.europa.eu/eli/reg/2022/2554/oj
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class DORAEntityType(Enum):
    """
    Entity types subject to DORA per Article 2(1).

    Each value corresponds to a specific paragraph in Article 2(1).
    """
    # Article 2(1)(a-u)
    CREDIT_INSTITUTION = "credit_institution"  # (a)
    PAYMENT_INSTITUTION = "payment_institution"  # (b)
    ACCOUNT_INFORMATION_SERVICE_PROVIDER = "account_information_service_provider"  # (c)
    ELECTRONIC_MONEY_INSTITUTION = "electronic_money_institution"  # (d)
    INVESTMENT_FIRM = "investment_firm"  # (e) - Most relevant for algo trading
    CRYPTO_ASSET_SERVICE_PROVIDER = "crypto_asset_service_provider"  # (f) - MiCA
    CENTRAL_SECURITIES_DEPOSITORY = "central_securities_depository"  # (g)
    CENTRAL_COUNTERPARTY = "central_counterparty"  # (h)
    TRADING_VENUE = "trading_venue"  # (i)
    TRADE_REPOSITORY = "trade_repository"  # (j)
    AIFM = "manager_of_alternative_investment_funds"  # (k) - AIFM
    MANAGEMENT_COMPANY = "management_company"  # (l) - UCITS
    DATA_REPORTING_SERVICE_PROVIDER = "data_reporting_service_provider"  # (m)
    INSURANCE_UNDERTAKING = "insurance_undertaking"  # (n)
    INSURANCE_INTERMEDIARY = "insurance_intermediary"  # (o)
    IORP = "institution_for_occupational_retirement_provision"  # (p)
    CREDIT_RATING_AGENCY = "credit_rating_agency"  # (q)
    ADMINISTRATOR_OF_CRITICAL_BENCHMARKS = "administrator_of_critical_benchmarks"  # (r)
    CROWDFUNDING_SERVICE_PROVIDER = "crowdfunding_service_provider"  # (s)
    SECURITISATION_REPOSITORY = "securitisation_repository"  # (t)
    ICT_THIRD_PARTY_SERVICE_PROVIDER = "ict_third_party_service_provider"  # (u)

    # Special categories
    NOT_A_FINANCIAL_ENTITY = "not_a_financial_entity"
    UNCLEAR = "unclear"


class DORAScopeResult(Enum):
    """Result of DORA scope verification."""
    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"
    PARTIALLY_IN_SCOPE = "partially_in_scope"  # Some exemptions apply
    UNCLEAR = "unclear"  # Requires legal clarification


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class EntityAuthorization:
    """
    Authorization details for a financial entity.

    Used to determine which DORA entity type applies.
    """
    # Basic identification
    legal_name: str
    lei: Optional[str] = None  # Legal Entity Identifier (ISO 17442)
    national_id: Optional[str] = None

    # Authorization details
    authorization_type: str = ""  # e.g., "MiFID II", "MiCA", "PSD2"
    authorization_reference: str = ""  # e.g., license number
    authorizing_nca: str = ""  # e.g., "BaFin", "AMF", "CBI"
    member_state: str = ""  # ISO 3166-1 alpha-2, e.g., "DE", "FR", "IE"
    authorization_date: Optional[str] = None

    # Entity type determination
    determined_entity_type: Optional[DORAEntityType] = None

    # Additional documentation
    authorization_documents: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ScopeVerification:
    """
    Result of DORA scope verification.

    Documents whether and how DORA applies to an entity.
    """
    verification_id: str = ""
    verification_date: str = ""

    # Entity information
    entity_authorization: Optional[EntityAuthorization] = None
    entity_type: DORAEntityType = DORAEntityType.UNCLEAR

    # Verification result
    result: DORAScopeResult = DORAScopeResult.UNCLEAR
    dora_article_reference: str = ""  # e.g., "Article 2(1)(e)"

    # Detailed findings
    in_scope_activities: List[str] = field(default_factory=list)
    out_of_scope_activities: List[str] = field(default_factory=list)
    partial_exemptions: List[str] = field(default_factory=list)

    # Implications
    applicable_articles: List[str] = field(default_factory=list)
    exempted_articles: List[str] = field(default_factory=list)
    key_requirements: List[str] = field(default_factory=list)

    # Confidence and review
    confidence_level: str = "low"  # "low", "medium", "high"
    legal_review_required: bool = True
    legal_review_completed: bool = False
    legal_reviewer: str = ""
    legal_review_date: Optional[str] = None

    # Notes
    notes: str = ""
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.verification_id:
            self.verification_id = f"SCOPE-{uuid.uuid4().hex[:8].upper()}"
        if not self.verification_date:
            self.verification_date = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Entity Type Information
# =============================================================================

ENTITY_TYPE_INFO = {
    DORAEntityType.CREDIT_INSTITUTION: {
        "article": "Article 2(1)(a)",
        "definition_reference": "Regulation (EU) No 575/2013, Article 4(1)(1)",
        "description": "Credit institutions as defined in CRR",
        "typical_examples": ["Banks", "Building societies"],
        "supervising_authority": "ECB (significant) / NCA (less significant)",
    },
    DORAEntityType.PAYMENT_INSTITUTION: {
        "article": "Article 2(1)(b)",
        "definition_reference": "Directive (EU) 2015/2366 (PSD2), Article 4(4)",
        "description": "Payment institutions authorized under PSD2",
        "typical_examples": ["Payment processors", "Money transfer operators"],
        "supervising_authority": "National payment authority",
    },
    DORAEntityType.ACCOUNT_INFORMATION_SERVICE_PROVIDER: {
        "article": "Article 2(1)(c)",
        "definition_reference": "Directive (EU) 2015/2366 (PSD2), Article 4(19)",
        "description": "Account information service providers under PSD2",
        "typical_examples": ["Account aggregators", "Financial data providers"],
        "supervising_authority": "National payment authority",
    },
    DORAEntityType.ELECTRONIC_MONEY_INSTITUTION: {
        "article": "Article 2(1)(d)",
        "definition_reference": "Directive 2009/110/EC (EMD2), Article 2(1)",
        "description": "Electronic money institutions",
        "typical_examples": ["E-money issuers", "Prepaid card providers"],
        "supervising_authority": "National payment authority",
    },
    DORAEntityType.INVESTMENT_FIRM: {
        "article": "Article 2(1)(e)",
        "definition_reference": "Regulation (EU) No 575/2013, Article 4(1)(2)",
        "description": "Investment firms as defined in CRR, including algorithmic traders",
        "typical_examples": [
            "Broker-dealers",
            "Algorithmic trading firms",
            "Portfolio managers",
            "Investment advisers",
        ],
        "supervising_authority": "National securities regulator (e.g., BaFin, AMF, FCA)",
        "notes": "Most relevant for algorithmic trading platforms under MiFID II",
    },
    DORAEntityType.CRYPTO_ASSET_SERVICE_PROVIDER: {
        "article": "Article 2(1)(f)",
        "definition_reference": "Regulation (EU) 2023/1114 (MiCA), Article 3(1)(15)",
        "description": "Crypto-asset service providers as defined in MiCA",
        "typical_examples": [
            "Crypto exchanges",
            "Crypto custody providers",
            "Crypto trading platforms",
        ],
        "supervising_authority": "National securities/financial regulator",
        "notes": "MiCA authorization required from 30 December 2024",
    },
    DORAEntityType.CENTRAL_SECURITIES_DEPOSITORY: {
        "article": "Article 2(1)(g)",
        "definition_reference": "Regulation (EU) No 909/2014 (CSDR), Article 2(1)(1)",
        "description": "Central securities depositories",
        "typical_examples": ["Euroclear", "Clearstream"],
        "supervising_authority": "National securities regulator + ESMA",
    },
    DORAEntityType.CENTRAL_COUNTERPARTY: {
        "article": "Article 2(1)(h)",
        "definition_reference": "Regulation (EU) No 648/2012 (EMIR), Article 2(1)",
        "description": "Central counterparties (CCPs)",
        "typical_examples": ["LCH", "Eurex Clearing", "ICE Clear"],
        "supervising_authority": "National authority + ESMA supervision",
    },
    DORAEntityType.TRADING_VENUE: {
        "article": "Article 2(1)(i)",
        "definition_reference": "Directive 2014/65/EU (MiFID II), Article 4(1)(24)",
        "description": "Trading venues (regulated markets, MTFs, OTFs)",
        "typical_examples": ["Stock exchanges", "MTFs", "OTFs"],
        "supervising_authority": "National securities regulator",
    },
    DORAEntityType.TRADE_REPOSITORY: {
        "article": "Article 2(1)(j)",
        "definition_reference": "Regulation (EU) No 648/2012 (EMIR), Article 2(2)",
        "description": "Trade repositories",
        "typical_examples": ["DTCC", "Regis-TR"],
        "supervising_authority": "ESMA direct supervision",
    },
    DORAEntityType.AIFM: {
        "article": "Article 2(1)(k)",
        "definition_reference": "Directive 2011/61/EU (AIFMD), Article 4(1)(b)",
        "description": "Managers of alternative investment funds",
        "typical_examples": ["Hedge fund managers", "Private equity managers"],
        "supervising_authority": "National securities regulator",
    },
    DORAEntityType.MANAGEMENT_COMPANY: {
        "article": "Article 2(1)(l)",
        "definition_reference": "Directive 2009/65/EC (UCITS), Article 2(1)(b)",
        "description": "Management companies (UCITS managers)",
        "typical_examples": ["UCITS fund managers"],
        "supervising_authority": "National securities regulator",
    },
    DORAEntityType.DATA_REPORTING_SERVICE_PROVIDER: {
        "article": "Article 2(1)(m)",
        "definition_reference": "Regulation (EU) No 600/2014 (MiFIR), Article 2(1)(36a)",
        "description": "Data reporting service providers (ARMs, APAs, CTPs)",
        "typical_examples": ["Approved reporting mechanisms", "Trade publication services"],
        "supervising_authority": "ESMA direct supervision (from MiFIR refit)",
    },
    DORAEntityType.INSURANCE_UNDERTAKING: {
        "article": "Article 2(1)(n)",
        "definition_reference": "Directive 2009/138/EC (Solvency II), Article 13(1)",
        "description": "Insurance and reinsurance undertakings",
        "typical_examples": ["Insurance companies", "Reinsurers"],
        "supervising_authority": "National insurance authority + EIOPA",
    },
    DORAEntityType.INSURANCE_INTERMEDIARY: {
        "article": "Article 2(1)(o)",
        "definition_reference": "Directive (EU) 2016/97 (IDD), Article 2(1)(3)",
        "description": "Insurance intermediaries (excluding ancillary)",
        "typical_examples": ["Insurance brokers", "Insurance agents"],
        "supervising_authority": "National insurance authority",
    },
    DORAEntityType.IORP: {
        "article": "Article 2(1)(p)",
        "definition_reference": "Directive (EU) 2016/2341 (IORP II), Article 6(1)",
        "description": "Institutions for occupational retirement provision",
        "typical_examples": ["Pension funds", "Occupational pension schemes"],
        "supervising_authority": "National pension authority + EIOPA",
    },
    DORAEntityType.CREDIT_RATING_AGENCY: {
        "article": "Article 2(1)(q)",
        "definition_reference": "Regulation (EC) No 1060/2009 (CRA Regulation), Article 3(1)(b)",
        "description": "Credit rating agencies",
        "typical_examples": ["S&P", "Moody's", "Fitch"],
        "supervising_authority": "ESMA direct supervision",
    },
    DORAEntityType.ADMINISTRATOR_OF_CRITICAL_BENCHMARKS: {
        "article": "Article 2(1)(r)",
        "definition_reference": "Regulation (EU) 2016/1011 (BMR), Article 3(1)(6)",
        "description": "Administrators of critical benchmarks",
        "typical_examples": ["EURIBOR administrator", "EONIA administrator"],
        "supervising_authority": "College of supervisors led by home NCA",
    },
    DORAEntityType.CROWDFUNDING_SERVICE_PROVIDER: {
        "article": "Article 2(1)(s)",
        "definition_reference": "Regulation (EU) 2020/1503 (ECSP), Article 2(1)(e)",
        "description": "Crowdfunding service providers",
        "typical_examples": ["Investment crowdfunding platforms", "Lending platforms"],
        "supervising_authority": "National securities regulator",
    },
    DORAEntityType.SECURITISATION_REPOSITORY: {
        "article": "Article 2(1)(t)",
        "definition_reference": "Regulation (EU) 2017/2402 (Securitisation Regulation), Article 2(23)",
        "description": "Securitisation repositories",
        "typical_examples": ["Securitisation data repositories"],
        "supervising_authority": "ESMA direct supervision",
    },
    DORAEntityType.ICT_THIRD_PARTY_SERVICE_PROVIDER: {
        "article": "Article 2(1)(u)",
        "definition_reference": "DORA Article 3(21)",
        "description": "ICT third-party service providers (oversight framework)",
        "typical_examples": ["Cloud providers", "Data centers", "Managed IT services"],
        "supervising_authority": "ESAs oversight (if designated as CTPP)",
        "notes": "Subject to oversight framework in Articles 31-44 if designated as CTPP",
    },
}


# =============================================================================
# Main DORA Scope Class
# =============================================================================

class DORAScope:
    """
    DORA Article 2 - Scope verification.

    CRITICAL: Verify if DORA applies BEFORE any implementation.

    This class helps determine:
    1. If DORA applies to an entity
    2. Which entity type (Article 2(1)(a-u)) applies
    3. What requirements apply based on entity type

    Usage:
        scope = DORAScope()

        # Check if entity type is in scope
        result = scope.verify_scope(
            entity_type=DORAEntityType.INVESTMENT_FIRM,
            authorization="MiFID II licensed",
        )

        if result.result == DORAScopeResult.IN_SCOPE:
            print("DORA applies")
            print(f"Key requirements: {result.key_requirements}")
    """

    def __init__(self):
        """Initialize DORA scope verifier."""
        self.entity_type_info = ENTITY_TYPE_INFO
        logger.info("DORAScope initialized")

    def verify_scope(
        self,
        entity_type: DORAEntityType,
        authorization: Optional[EntityAuthorization] = None,
        activities: Optional[List[str]] = None,
    ) -> ScopeVerification:
        """
        Verify if an entity is subject to DORA.

        Args:
            entity_type: Type of financial entity
            authorization: Authorization details (optional)
            activities: List of activities performed

        Returns:
            ScopeVerification with detailed findings
        """
        verification = ScopeVerification(
            entity_authorization=authorization,
            entity_type=entity_type,
        )

        # Check if entity type is explicitly in scope
        if entity_type == DORAEntityType.NOT_A_FINANCIAL_ENTITY:
            verification.result = DORAScopeResult.OUT_OF_SCOPE
            verification.confidence_level = "medium"
            verification.notes = (
                "Entity is not a financial entity under Article 2(1). "
                "DORA may not apply. However, consult legal counsel to confirm."
            )
            verification.warnings.append(
                "If the entity provides ICT services to financial entities, "
                "it may be subject to the oversight framework (Articles 31-44)."
            )
            return verification

        if entity_type == DORAEntityType.UNCLEAR:
            verification.result = DORAScopeResult.UNCLEAR
            verification.confidence_level = "low"
            verification.legal_review_required = True
            verification.notes = "Entity type unclear. Legal clarification required."
            return verification

        # Entity type is one of the 21 types in Article 2(1)
        info = self.entity_type_info.get(entity_type, {})
        verification.result = DORAScopeResult.IN_SCOPE
        verification.dora_article_reference = info.get("article", "Article 2(1)")
        verification.confidence_level = "high" if authorization else "medium"

        # Determine applicable requirements
        verification.applicable_articles = self._get_applicable_articles(entity_type)
        verification.key_requirements = self._get_key_requirements(entity_type)

        # Add any specific warnings
        if entity_type == DORAEntityType.ICT_THIRD_PARTY_SERVICE_PROVIDER:
            verification.warnings.append(
                "As ICT third-party service provider, the oversight framework "
                "(Articles 31-44) applies if designated as Critical Third-Party Provider (CTPP)."
            )
            verification.notes = info.get("notes", "")

        if entity_type in (DORAEntityType.INVESTMENT_FIRM, DORAEntityType.CRYPTO_ASSET_SERVICE_PROVIDER):
            verification.notes = info.get("notes", "")

        # Check for activities
        if activities:
            verification.in_scope_activities = activities

        logger.info(
            f"Scope verification completed: {entity_type.value} -> {verification.result.value}"
        )

        return verification

    def verify_authorization(
        self,
        authorization: EntityAuthorization,
    ) -> ScopeVerification:
        """
        Verify DORA scope based on authorization details.

        Determines entity type from authorization and verifies scope.

        Args:
            authorization: Entity authorization details

        Returns:
            ScopeVerification with detailed findings
        """
        # Determine entity type from authorization
        entity_type = self._determine_entity_type(authorization)
        authorization.determined_entity_type = entity_type

        return self.verify_scope(
            entity_type=entity_type,
            authorization=authorization,
        )

    def get_entity_type_for_algo_trading(self) -> DORAEntityType:
        """
        Get the most likely entity type for an algorithmic trading platform.

        Returns:
            DORAEntityType.INVESTMENT_FIRM for MiFID II authorized entities
            DORAEntityType.CRYPTO_ASSET_SERVICE_PROVIDER for MiCA authorized entities
        """
        # For algorithmic trading platforms:
        # - If trading traditional instruments (stocks, bonds, derivatives): INVESTMENT_FIRM
        # - If trading crypto-assets: CRYPTO_ASSET_SERVICE_PROVIDER
        # Most algo trading platforms would be INVESTMENT_FIRM under MiFID II Article 4(1)(1)
        return DORAEntityType.INVESTMENT_FIRM

    def get_entity_info(self, entity_type: DORAEntityType) -> Dict[str, Any]:
        """
        Get detailed information about an entity type.

        Args:
            entity_type: The entity type to get info for

        Returns:
            Dictionary with entity type details
        """
        return self.entity_type_info.get(entity_type, {})

    def _determine_entity_type(
        self,
        authorization: EntityAuthorization,
    ) -> DORAEntityType:
        """Determine entity type from authorization."""
        auth_type = authorization.authorization_type.lower()

        # Map authorization types to entity types
        mapping = {
            "mifid": DORAEntityType.INVESTMENT_FIRM,
            "mifid ii": DORAEntityType.INVESTMENT_FIRM,
            "mifid2": DORAEntityType.INVESTMENT_FIRM,
            "investment firm": DORAEntityType.INVESTMENT_FIRM,
            "mica": DORAEntityType.CRYPTO_ASSET_SERVICE_PROVIDER,
            "crypto": DORAEntityType.CRYPTO_ASSET_SERVICE_PROVIDER,
            "psd2": DORAEntityType.PAYMENT_INSTITUTION,
            "payment": DORAEntityType.PAYMENT_INSTITUTION,
            "emd": DORAEntityType.ELECTRONIC_MONEY_INSTITUTION,
            "e-money": DORAEntityType.ELECTRONIC_MONEY_INSTITUTION,
            "aifmd": DORAEntityType.AIFM,
            "aif": DORAEntityType.AIFM,
            "ucits": DORAEntityType.MANAGEMENT_COMPANY,
            "credit institution": DORAEntityType.CREDIT_INSTITUTION,
            "bank": DORAEntityType.CREDIT_INSTITUTION,
        }

        for key, entity_type in mapping.items():
            if key in auth_type:
                return entity_type

        return DORAEntityType.UNCLEAR

    def _get_applicable_articles(self, entity_type: DORAEntityType) -> List[str]:
        """Get list of DORA articles applicable to entity type."""
        # Core articles applicable to all financial entities
        core_articles = [
            "Article 5 - Governance",
            "Article 6 - ICT risk management framework",
            "Article 7 - ICT systems, protocols and tools",
            "Article 8 - Identification",
            "Article 9 - Protection and prevention",
            "Article 10 - Detection",
            "Article 11 - Response and recovery",
            "Article 12 - Backup policies",
            "Article 13 - Learning and evolving",
            "Article 14 - Communication",
            "Article 17 - ICT-related incident management process",
            "Article 18 - Classification of ICT-related incidents",
            "Article 19 - Reporting of major ICT-related incidents",
            "Article 24 - General requirements for digital operational resilience testing",
            "Article 25 - Testing of ICT tools and systems",
            "Article 28 - General principles on ICT third-party risk management",
            "Article 29 - Preliminary assessment of ICT concentration risk",
            "Article 30 - Key contractual provisions",
        ]

        # ICT third-party service providers have different articles
        if entity_type == DORAEntityType.ICT_THIRD_PARTY_SERVICE_PROVIDER:
            return [
                "Articles 31-44 - Oversight framework (if designated as CTPP)",
            ]

        return core_articles

    def _get_key_requirements(self, entity_type: DORAEntityType) -> List[str]:
        """Get key DORA requirements for entity type."""
        if entity_type == DORAEntityType.ICT_THIRD_PARTY_SERVICE_PROVIDER:
            return [
                "Cooperation with Lead Overseer (if CTPP)",
                "Provision of information to financial entities",
                "Compliance with oversight recommendations",
            ]

        return [
            "ICT risk management framework implementation",
            "Major incident reporting to NCA",
            "Register of Information on ICT third-party providers",
            "Digital operational resilience testing",
            "Third-party risk management and due diligence",
        ]


# =============================================================================
# Factory Functions
# =============================================================================

def create_scope_verifier() -> DORAScope:
    """
    Create a DORAScope instance.

    Returns:
        Configured DORAScope instance
    """
    return DORAScope()


def get_entity_type_description(entity_type: DORAEntityType) -> str:
    """
    Get a human-readable description of an entity type.

    Args:
        entity_type: The entity type

    Returns:
        Description string
    """
    info = ENTITY_TYPE_INFO.get(entity_type, {})
    return info.get("description", entity_type.value)
