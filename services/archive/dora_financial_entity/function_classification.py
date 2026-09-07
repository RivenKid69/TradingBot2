# -*- coding: utf-8 -*-
"""
DORA Function Classification Module (Article 3(22)).

Article 3(22) defines "critical or important function" as a function whose disruption would:
    1. Materially impair the financial performance of a financial entity, OR
    2. Materially impair the soundness or continuity of its services and activities, OR
    3. Materially impair the continuing compliance with authorization conditions

This classification is FUNDAMENTAL to all DORA requirements:
    - Third-party risk assessment prioritization (Article 28)
    - Register of Information focus (Article 28)
    - Exit strategy requirements (Article 28)
    - Testing priority (Article 24)
    - Incident reporting thresholds (Article 18)

References:
    - Article 3(22) DORA: https://www.dora-info.eu/dora/article-3/
    - RTS on Third-Party Risk: CDR 2024/1773
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

class FunctionCriticality(Enum):
    """Criticality level of a business function."""
    CRITICAL = "critical"
    IMPORTANT = "important"
    STANDARD = "standard"  # Neither critical nor important


class ImpairmentType(Enum):
    """
    Types of material impairment per Article 3(22).

    Function is critical/important if disruption causes ANY of these.
    """
    FINANCIAL_PERFORMANCE = "financial_performance"  # Article 3(22)(a)
    SERVICE_SOUNDNESS = "service_soundness"  # Article 3(22)(b)
    REGULATORY_COMPLIANCE = "regulatory_compliance"  # Article 3(22)(c)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ThirdPartyProvider:
    """
    ICT third-party service provider information.

    Per Article 28 and Register of Information requirements.
    """
    provider_id: str = ""
    provider_name: str = ""
    provider_type: str = ""  # "exchange", "broker", "data_provider", "cloud", etc.

    # Identification (for Register of Information)
    lei: Optional[str] = None  # Legal Entity Identifier
    alternative_identifier: Optional[str] = None  # If no LEI
    identifier_type: str = ""  # "lei", "sec_crd", "registration_number", "trade_name"
    country: str = ""  # ISO 3166-1 alpha-2

    # Services provided
    services: List[str] = field(default_factory=list)

    # Criticality assessment
    supports_critical_function: bool = False
    is_substitutable: bool = True
    substitution_difficulty: str = "medium"  # "low", "medium", "high"
    alternative_providers: List[str] = field(default_factory=list)

    # Risk classification
    risk_level: str = "medium"  # "low", "medium", "high", "critical"

    # Contract information
    contract_type: str = ""  # "standard_terms", "negotiated", "regulated"
    contract_reference: str = ""

    def __post_init__(self):
        if not self.provider_id:
            self.provider_id = f"PROV-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class ICTService:
    """
    ICT service supporting a business function.

    Maps ICT services to business functions per Article 8.
    """
    service_id: str = ""
    service_name: str = ""
    service_description: str = ""
    service_type: str = ""  # "market_data", "execution", "storage", "communication", etc.

    # Dependencies
    third_party_providers: List[str] = field(default_factory=list)  # Provider IDs
    internal_systems: List[str] = field(default_factory=list)

    # Criticality
    is_critical: bool = False
    rto_hours: Optional[float] = None  # Recovery Time Objective
    rpo_hours: Optional[float] = None  # Recovery Point Objective

    def __post_init__(self):
        if not self.service_id:
            self.service_id = f"SVC-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class FunctionClassification:
    """
    Article 3(22) - Critical or Important Function classification.

    MUST be done BEFORE third-party risk assessment.
    """
    classification_id: str = ""
    classification_date: str = ""

    # Function identification
    function_name: str = ""
    function_description: str = ""
    function_category: str = ""  # "trading", "risk", "settlement", "reporting", etc.

    # Assessment criteria (Article 3(22))
    impairs_financial_performance: bool = False  # (a)
    impairs_service_soundness: bool = False  # (b)
    impairs_regulatory_compliance: bool = False  # (c)

    # Detailed impact assessment
    financial_performance_impact: str = ""  # Description of financial impact
    service_soundness_impact: str = ""  # Description of service impact
    regulatory_compliance_impact: str = ""  # Description of compliance impact

    # Quantitative estimates (where possible)
    estimated_revenue_impact_eur: Optional[float] = None
    max_acceptable_downtime_hours: Optional[float] = None

    # Supporting ICT services
    ict_services_supporting: List[str] = field(default_factory=list)  # Service IDs

    # Third-party providers
    third_party_providers: List[str] = field(default_factory=list)  # Provider IDs

    # Classification result
    criticality: FunctionCriticality = FunctionCriticality.STANDARD
    classification_rationale: str = ""

    # Review
    classified_by: str = ""
    reviewed_by: str = ""
    review_date: Optional[str] = None
    next_review_date: Optional[str] = None

    def __post_init__(self):
        if not self.classification_id:
            self.classification_id = f"FUNC-{uuid.uuid4().hex[:8].upper()}"
        if not self.classification_date:
            self.classification_date = datetime.now(timezone.utc).isoformat()

        # Determine criticality based on impairment criteria
        self._determine_criticality()

    def _determine_criticality(self) -> None:
        """Determine function criticality based on impairment assessment."""
        # Function is critical/important if ANY criterion is True
        if any([
            self.impairs_financial_performance,
            self.impairs_service_soundness,
            self.impairs_regulatory_compliance,
        ]):
            # Distinguish between critical and important
            # Critical: multiple criteria OR high financial impact
            critical_count = sum([
                self.impairs_financial_performance,
                self.impairs_service_soundness,
                self.impairs_regulatory_compliance,
            ])

            if critical_count >= 2:
                self.criticality = FunctionCriticality.CRITICAL
            elif self.impairs_regulatory_compliance:
                # Regulatory compliance impact alone makes it important
                self.criticality = FunctionCriticality.IMPORTANT
            elif self.impairs_financial_performance and self.estimated_revenue_impact_eur:
                # High financial impact (>10% revenue) makes it critical
                self.criticality = FunctionCriticality.CRITICAL
            else:
                self.criticality = FunctionCriticality.IMPORTANT
        else:
            self.criticality = FunctionCriticality.STANDARD

    @property
    def is_critical_or_important(self) -> bool:
        """Function is critical/important if ANY impairment criterion is True."""
        return self.criticality in (
            FunctionCriticality.CRITICAL,
            FunctionCriticality.IMPORTANT,
        )

    @property
    def impairment_types(self) -> List[ImpairmentType]:
        """Get list of impairment types that apply."""
        types = []
        if self.impairs_financial_performance:
            types.append(ImpairmentType.FINANCIAL_PERFORMANCE)
        if self.impairs_service_soundness:
            types.append(ImpairmentType.SERVICE_SOUNDNESS)
        if self.impairs_regulatory_compliance:
            types.append(ImpairmentType.REGULATORY_COMPLIANCE)
        return types


# =============================================================================
# Default Platform Functions
# =============================================================================

def get_platform_functions() -> Dict[str, FunctionClassification]:
    """
    Get default function classifications for an algorithmic trading platform.

    These are typical functions that should be assessed. Actual classifications
    should be reviewed and adjusted based on the specific platform.

    Returns:
        Dictionary of function name -> FunctionClassification
    """
    return {
        "order_execution": FunctionClassification(
            function_name="Order Execution",
            function_description="Placing and executing trading orders on exchanges/brokers",
            function_category="trading",
            impairs_financial_performance=True,  # Direct revenue impact
            impairs_service_soundness=True,  # Core service
            impairs_regulatory_compliance=True,  # Best execution obligation (MiFID II)
            financial_performance_impact="Direct loss of trading revenue; potential P&L impact from missed trades",
            service_soundness_impact="Core trading capability unavailable",
            regulatory_compliance_impact="Best execution obligations (Article 27 MiFID II); RTS 6 requirements",
            ict_services_supporting=["exchange_api", "order_management", "risk_engine"],
            third_party_providers=["binance", "alpaca", "oanda", "ib"],
            classification_rationale="Core business function; disruption directly impacts revenue and compliance",
        ),

        "market_data": FunctionClassification(
            function_name="Market Data",
            function_description="Receiving real-time and historical market prices",
            function_category="data",
            impairs_financial_performance=True,  # Cannot trade without data
            impairs_service_soundness=True,  # Core service dependency
            impairs_regulatory_compliance=False,  # No direct regulatory requirement
            financial_performance_impact="Trading decisions cannot be made; potential missed opportunities",
            service_soundness_impact="All trading strategies dependent on market data",
            ict_services_supporting=["data_feeds", "websockets", "historical_data"],
            third_party_providers=["binance", "polygon", "alpaca", "yahoo"],
            classification_rationale="Essential input for all trading decisions",
        ),

        "risk_monitoring": FunctionClassification(
            function_name="Risk Monitoring",
            function_description="Monitoring positions, exposure, and risk limits in real-time",
            function_category="risk",
            impairs_financial_performance=True,  # Risk of uncontrolled losses
            impairs_service_soundness=True,  # Safety-critical
            impairs_regulatory_compliance=True,  # MiFID II RTS 6 requirement
            financial_performance_impact="Potential for uncontrolled losses; breach of risk limits",
            service_soundness_impact="Cannot ensure safe operation of trading system",
            regulatory_compliance_impact="RTS 6 Article 17 requires real-time monitoring; capital requirements",
            ict_services_supporting=["risk_engine", "position_tracking", "limit_monitoring"],
            third_party_providers=[],  # Internal function
            classification_rationale="Safety-critical function required by regulation",
        ),

        "kill_switch": FunctionClassification(
            function_name="Kill Switch",
            function_description="Emergency system to halt all trading activities",
            function_category="risk",
            impairs_financial_performance=False,  # Prevents losses
            impairs_service_soundness=True,  # Safety mechanism
            impairs_regulatory_compliance=True,  # RTS 6 Article 12 requirement
            service_soundness_impact="Cannot ensure emergency stop capability",
            regulatory_compliance_impact="RTS 6 Article 12 mandates kill functionality",
            ict_services_supporting=["kill_switch_service", "order_management"],
            third_party_providers=[],  # Internal function
            classification_rationale="Regulatory mandatory safety mechanism",
        ),

        "regulatory_reporting": FunctionClassification(
            function_name="Regulatory Reporting",
            function_description="Submitting transaction reports to regulators (RTS 22)",
            function_category="compliance",
            impairs_financial_performance=False,  # No direct P&L impact
            impairs_service_soundness=False,  # Not core business
            impairs_regulatory_compliance=True,  # Direct compliance impact
            regulatory_compliance_impact="MiFIR Article 26; RTS 22 transaction reporting; potential fines",
            ict_services_supporting=["reporting_system", "arm_client"],
            third_party_providers=[],  # Typically internal or ARM
            classification_rationale="Direct regulatory obligation with penalty for non-compliance",
        ),

        "backtesting": FunctionClassification(
            function_name="Strategy Backtesting",
            function_description="Testing strategies on historical data",
            function_category="research",
            impairs_financial_performance=False,  # No immediate revenue impact
            impairs_service_soundness=False,  # Not operational
            impairs_regulatory_compliance=False,  # No regulatory requirement
            ict_services_supporting=["backtest_engine", "historical_data"],
            third_party_providers=["polygon"],  # Historical data
            classification_rationale="Development/research function; not operationally critical",
        ),

        "model_training": FunctionClassification(
            function_name="AI Model Training",
            function_description="Training and updating machine learning models",
            function_category="research",
            impairs_financial_performance=False,  # No immediate revenue impact
            impairs_service_soundness=False,  # Not operational
            impairs_regulatory_compliance=False,  # No direct requirement (EU AI Act separate)
            ict_services_supporting=["training_infrastructure", "data_pipeline"],
            third_party_providers=[],  # Internal
            classification_rationale="Development function; can be delayed without operational impact",
        ),

        "user_authentication": FunctionClassification(
            function_name="User Authentication",
            function_description="User login and access control",
            function_category="security",
            impairs_financial_performance=False,  # Indirect impact
            impairs_service_soundness=True,  # Cannot operate without access
            impairs_regulatory_compliance=True,  # Access control requirements
            service_soundness_impact="Users cannot access system",
            regulatory_compliance_impact="Access control requirements per RTS on ICT risk management",
            ict_services_supporting=["auth_service", "identity_provider"],
            third_party_providers=[],  # Internal or IdP
            classification_rationale="Security-critical but can have fallback procedures",
        ),

        "audit_trail": FunctionClassification(
            function_name="Audit Trail Recording",
            function_description="Recording all trading decisions and actions",
            function_category="compliance",
            impairs_financial_performance=False,
            impairs_service_soundness=False,  # Trading can continue
            impairs_regulatory_compliance=True,  # MiFIR Article 25
            regulatory_compliance_impact="MiFIR Article 25 requires record keeping; RTS 6 audit requirements",
            ict_services_supporting=["audit_storage", "logging_system"],
            third_party_providers=[],  # Internal
            classification_rationale="Regulatory requirement; short disruption acceptable with recovery",
        ),

        "position_reconciliation": FunctionClassification(
            function_name="Position Reconciliation",
            function_description="Reconciling positions with exchanges/brokers",
            function_category="operations",
            impairs_financial_performance=True,  # Risk of position discrepancies
            impairs_service_soundness=True,  # Operational accuracy
            impairs_regulatory_compliance=False,
            financial_performance_impact="Position discrepancies could lead to unexpected exposure",
            service_soundness_impact="Cannot verify accurate position state",
            ict_services_supporting=["position_sync", "exchange_api"],
            third_party_providers=["binance", "alpaca", "oanda", "ib"],
            classification_rationale="Operational integrity function",
        ),
    }


def get_ict_providers() -> Dict[str, ThirdPartyProvider]:
    """
    Get default ICT third-party providers for an algorithmic trading platform.

    These are typical providers. Actual providers should be documented based on
    the specific platform's integrations.

    Returns:
        Dictionary of provider name -> ThirdPartyProvider
    """
    return {
        "binance": ThirdPartyProvider(
            provider_name="Binance",
            provider_type="exchange",
            alternative_identifier="Binance Holdings Limited (Malta)",
            identifier_type="registration_number",
            country="MT",
            services=["market_data", "order_execution", "account_management"],
            supports_critical_function=True,
            is_substitutable=True,
            substitution_difficulty="medium",
            alternative_providers=["kraken", "coinbase"],
            risk_level="high",
            contract_type="standard_terms",
        ),

        "alpaca": ThirdPartyProvider(
            provider_name="Alpaca Securities LLC",
            provider_type="broker",
            alternative_identifier="CRD# 288202",
            identifier_type="sec_crd",
            country="US",
            services=["market_data", "order_execution", "account_management"],
            supports_critical_function=True,
            is_substitutable=True,
            substitution_difficulty="medium",
            alternative_providers=["ib", "tradier"],
            risk_level="medium",
            contract_type="standard_terms",
        ),

        "polygon": ThirdPartyProvider(
            provider_name="Polygon.io Inc",
            provider_type="data_provider",
            alternative_identifier="Polygon.io Inc (Delaware)",
            identifier_type="registration_number",
            country="US",
            services=["market_data", "historical_data", "options_data"],
            supports_critical_function=True,
            is_substitutable=True,
            substitution_difficulty="low",
            alternative_providers=["iex", "alpha_vantage", "yahoo"],
            risk_level="medium",
            contract_type="standard_terms",
        ),

        "oanda": ThirdPartyProvider(
            provider_name="OANDA Corporation",
            provider_type="forex_broker",
            lei="549300OFS14BSRRXZ270",  # Example LEI
            identifier_type="lei",
            country="US",
            services=["forex_market_data", "forex_execution"],
            supports_critical_function=True,
            is_substitutable=True,
            substitution_difficulty="medium",
            alternative_providers=["ib", "saxo"],
            risk_level="medium",
            contract_type="standard_terms",
        ),

        "ib": ThirdPartyProvider(
            provider_name="Interactive Brokers LLC",
            provider_type="broker",
            lei="549300GYZ1LOSP5FNQ37",
            identifier_type="lei",
            country="US",
            services=["market_data", "order_execution", "options_trading", "account_management"],
            supports_critical_function=True,
            is_substitutable=True,
            substitution_difficulty="high",
            alternative_providers=["alpaca", "schwab"],
            risk_level="medium",
            contract_type="standard_terms",
        ),

        "deribit": ThirdPartyProvider(
            provider_name="Deribit",
            provider_type="crypto_derivatives_exchange",
            alternative_identifier="Deribit B.V. (Netherlands)",
            identifier_type="registration_number",
            country="NL",
            services=["crypto_options", "crypto_futures", "market_data"],
            supports_critical_function=True,
            is_substitutable=True,
            substitution_difficulty="high",
            alternative_providers=["binance"],
            risk_level="high",
            contract_type="standard_terms",
        ),

        "yahoo": ThirdPartyProvider(
            provider_name="Yahoo Finance",
            provider_type="data_provider",
            alternative_identifier="Yahoo! Inc (Verizon Media)",
            identifier_type="trade_name",
            country="US",
            services=["market_data", "historical_data", "fundamentals"],
            supports_critical_function=False,  # Backup data source
            is_substitutable=True,
            substitution_difficulty="low",
            alternative_providers=["polygon", "alpha_vantage"],
            risk_level="low",
            contract_type="standard_terms",
        ),
    }


# =============================================================================
# Function Classifier
# =============================================================================

class FunctionClassifier:
    """
    Classify business functions per DORA Article 3(22).

    Helps determine which functions are "critical or important" and thus
    subject to enhanced DORA requirements.

    Usage:
        classifier = FunctionClassifier()

        # Add a function
        func = classifier.classify_function(
            function_name="Order Execution",
            function_description="Trading order placement and execution",
            impairs_financial_performance=True,
            impairs_service_soundness=True,
            impairs_regulatory_compliance=True,
        )

        if func.is_critical_or_important:
            print(f"{func.function_name} is {func.criticality.value}")
    """

    def __init__(self):
        """Initialize function classifier."""
        self._functions: Dict[str, FunctionClassification] = {}
        self._providers: Dict[str, ThirdPartyProvider] = {}
        self._services: Dict[str, ICTService] = {}
        logger.info("FunctionClassifier initialized")

    def classify_function(
        self,
        function_name: str,
        function_description: str,
        impairs_financial_performance: bool = False,
        impairs_service_soundness: bool = False,
        impairs_regulatory_compliance: bool = False,
        function_category: str = "",
        financial_performance_impact: str = "",
        service_soundness_impact: str = "",
        regulatory_compliance_impact: str = "",
        ict_services_supporting: Optional[List[str]] = None,
        third_party_providers: Optional[List[str]] = None,
        classification_rationale: str = "",
        classified_by: str = "system",
    ) -> FunctionClassification:
        """
        Classify a business function per Article 3(22).

        Args:
            function_name: Name of the function
            function_description: Description of what the function does
            impairs_financial_performance: Would disruption materially impair financial performance?
            impairs_service_soundness: Would disruption materially impair service soundness?
            impairs_regulatory_compliance: Would disruption materially impair compliance?
            function_category: Category (trading, risk, compliance, etc.)
            financial_performance_impact: Description of financial impact
            service_soundness_impact: Description of service impact
            regulatory_compliance_impact: Description of compliance impact
            ict_services_supporting: List of ICT services supporting this function
            third_party_providers: List of third-party providers supporting this function
            classification_rationale: Explanation of classification decision
            classified_by: Who performed the classification

        Returns:
            FunctionClassification result
        """
        classification = FunctionClassification(
            function_name=function_name,
            function_description=function_description,
            function_category=function_category,
            impairs_financial_performance=impairs_financial_performance,
            impairs_service_soundness=impairs_service_soundness,
            impairs_regulatory_compliance=impairs_regulatory_compliance,
            financial_performance_impact=financial_performance_impact,
            service_soundness_impact=service_soundness_impact,
            regulatory_compliance_impact=regulatory_compliance_impact,
            ict_services_supporting=ict_services_supporting or [],
            third_party_providers=third_party_providers or [],
            classification_rationale=classification_rationale,
            classified_by=classified_by,
        )

        self._functions[function_name.lower().replace(" ", "_")] = classification

        logger.info(
            f"Function classified: {function_name} -> {classification.criticality.value}"
        )

        return classification

    def register_provider(self, provider: ThirdPartyProvider) -> None:
        """Register an ICT third-party provider."""
        self._providers[provider.provider_name.lower().replace(" ", "_")] = provider
        logger.info(f"Provider registered: {provider.provider_name}")

    def register_service(self, service: ICTService) -> None:
        """Register an ICT service."""
        self._services[service.service_name.lower().replace(" ", "_")] = service
        logger.info(f"Service registered: {service.service_name}")

    def get_function(self, function_name: str) -> Optional[FunctionClassification]:
        """Get a classified function by name."""
        key = function_name.lower().replace(" ", "_")
        return self._functions.get(key)

    def get_all_functions(self) -> List[FunctionClassification]:
        """Get all classified functions."""
        return list(self._functions.values())

    def get_critical_functions(self) -> List[FunctionClassification]:
        """Get only critical functions."""
        return [f for f in self._functions.values() if f.criticality == FunctionCriticality.CRITICAL]

    def get_important_functions(self) -> List[FunctionClassification]:
        """Get only important functions."""
        return [f for f in self._functions.values() if f.criticality == FunctionCriticality.IMPORTANT]

    def get_critical_or_important_functions(self) -> List[FunctionClassification]:
        """Get all critical or important functions."""
        return [f for f in self._functions.values() if f.is_critical_or_important]

    def get_providers_for_critical_functions(self) -> List[str]:
        """Get list of provider IDs that support critical/important functions."""
        providers = set()
        for func in self.get_critical_or_important_functions():
            providers.update(func.third_party_providers)
        return list(providers)

    def load_platform_defaults(self) -> None:
        """Load default platform function classifications."""
        for name, func in get_platform_functions().items():
            self._functions[name] = func

        for name, provider in get_ict_providers().items():
            self._providers[name] = provider

        logger.info(
            f"Loaded {len(self._functions)} functions and {len(self._providers)} providers"
        )

    def get_classification_summary(self) -> Dict[str, Any]:
        """
        Get summary of function classifications.

        Returns:
            Dictionary with classification statistics
        """
        functions = list(self._functions.values())

        critical_count = sum(1 for f in functions if f.criticality == FunctionCriticality.CRITICAL)
        important_count = sum(1 for f in functions if f.criticality == FunctionCriticality.IMPORTANT)
        standard_count = sum(1 for f in functions if f.criticality == FunctionCriticality.STANDARD)

        critical_providers = self.get_providers_for_critical_functions()

        return {
            "total_functions": len(functions),
            "critical_count": critical_count,
            "important_count": important_count,
            "standard_count": standard_count,
            "critical_or_important_ratio": (critical_count + important_count) / len(functions) if functions else 0,
            "critical_providers": critical_providers,
            "total_providers": len(self._providers),
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_function_classifier() -> FunctionClassifier:
    """
    Create a FunctionClassifier instance.

    Returns:
        Configured FunctionClassifier instance
    """
    return FunctionClassifier()
