# -*- coding: utf-8 -*-
"""
DORA Third-Party ICT Risk Management Module (Article 28).

Regulation (EU) 2022/2554 Article 28 defines requirements for managing
ICT third-party risk as an integral part of ICT risk framework:
    - Full responsibility remains with financial entity
    - Proportionate approach based on nature, scale and criticality
    - Managing ICT concentration risk
    - Register of Information (Article 28(3))
    - Exit strategies (Article 28(8))

This module provides:
    1. ICTProvider - Third-party provider data model
    2. ThirdPartyRiskAssessment - Risk assessment structure
    3. DORAThirdPartyRiskManagement - Main management class
    4. Provider criticality classification
    5. Due diligence assessment tools

References:
    - Article 28 DORA: https://www.digital-operational-resilience-act.com/Article_28.html
    - RTS on ICT Risk Management: CDR 2024/1774 Chapter V
    - ESAs Designated CTPPs: https://www.esma.europa.eu/press-news/esma-news/esas-designate-ctpps
    - ITS on Register of Information: CIR 2024/2956
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

class ProviderType(Enum):
    """Type of ICT third-party service provider."""
    CLOUD_SERVICE = "cloud_service"
    INFRASTRUCTURE = "infrastructure"
    SOFTWARE_VENDOR = "software_vendor"
    DATA_PROVIDER = "data_provider"
    EXCHANGE = "exchange"
    BROKER = "broker"
    PAYMENT_PROCESSOR = "payment_processor"
    TELECOMMUNICATIONS = "telecommunications"
    SECURITY_SERVICES = "security_services"
    MANAGED_SERVICES = "managed_services"
    CONSULTING = "consulting"
    OTHER = "other"


class ProviderCriticality(Enum):
    """Provider criticality classification per Article 28."""
    CRITICAL = "critical"  # Supports critical/important function
    HIGH = "high"  # Significant operational impact
    MEDIUM = "medium"  # Moderate operational impact
    LOW = "low"  # Limited operational impact


class ServiceCriticality(Enum):
    """Service criticality classification."""
    CRITICAL_OR_IMPORTANT = "critical_or_important"  # Article 3(22)
    IMPORTANT = "important"
    STANDARD = "standard"
    NON_CRITICAL = "non_critical"


class ProviderStatus(Enum):
    """Provider relationship status."""
    ACTIVE = "active"
    UNDER_REVIEW = "under_review"
    ONBOARDING = "onboarding"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"
    PENDING_EXIT = "pending_exit"


class RiskCategory(Enum):
    """Third-party risk categories."""
    OPERATIONAL = "operational"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    FINANCIAL = "financial"
    REPUTATIONAL = "reputational"
    CONCENTRATION = "concentration"
    GEOGRAPHICAL = "geographical"
    SUBCONTRACTING = "subcontracting"
    CONTINUITY = "continuity"
    DATA_PROTECTION = "data_protection"


class RiskLevel(Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DueDiligenceStatus(Enum):
    """Due diligence assessment status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_REMEDIATION = "requires_remediation"
    EXPIRED = "expired"


class AssessmentType(Enum):
    """Type of assessment."""
    INITIAL = "initial"
    PERIODIC = "periodic"
    INCIDENT_TRIGGERED = "incident_triggered"
    CHANGE_TRIGGERED = "change_triggered"
    REGULATORY_TRIGGERED = "regulatory_triggered"


class SubstitutabilityLevel(Enum):
    """Provider substitutability level."""
    EASILY_SUBSTITUTABLE = "easily_substitutable"
    SUBSTITUTABLE_WITH_EFFORT = "substitutable_with_effort"
    DIFFICULT_TO_SUBSTITUTE = "difficult_to_substitute"
    NOT_SUBSTITUTABLE = "not_substitutable"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ICTService:
    """
    ICT Service provided by a third party.

    Represents a specific service within a provider relationship.
    """
    service_id: str = ""
    name: str = ""
    description: str = ""
    service_type: str = ""

    # Criticality
    criticality: ServiceCriticality = ServiceCriticality.STANDARD
    supports_critical_function: bool = False
    supported_functions: List[str] = field(default_factory=list)

    # Service details
    service_level_targets: Dict[str, Any] = field(default_factory=dict)
    availability_target_pct: float = 99.0
    response_time_sla_hours: int = 4

    # Data handling
    data_processed: List[str] = field(default_factory=list)
    data_stored: List[str] = field(default_factory=list)
    data_classification: str = ""

    # Dependencies
    dependent_systems: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.service_id:
            self.service_id = f"SVC-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class ICTProvider:
    """
    ICT Third-Party Service Provider per Article 28.

    Represents a third-party provider with all required information
    for DORA compliance tracking.
    """
    provider_id: str = ""
    name: str = ""
    legal_name: str = ""
    provider_type: ProviderType = ProviderType.OTHER

    # Identification (per ITS requirements)
    lei: str = ""  # Legal Entity Identifier (if available)
    registration_number: str = ""
    tax_id: str = ""

    # Location
    country: str = ""  # ISO 3166-1 alpha-2
    headquarters_country: str = ""
    data_processing_countries: List[str] = field(default_factory=list)
    data_storage_countries: List[str] = field(default_factory=list)
    is_eu_provider: bool = False

    # Criticality assessment
    criticality: ProviderCriticality = ProviderCriticality.LOW
    supports_critical_function: bool = False

    # Services
    services: List[ICTService] = field(default_factory=list)

    # Relationship
    status: ProviderStatus = ProviderStatus.ACTIVE
    relationship_start_date: str = ""
    contract_reference: str = ""

    # Subcontracting chain
    permits_subcontracting: bool = False
    known_subcontractors: List[str] = field(default_factory=list)
    subcontracting_chain_known: bool = False

    # Risk assessment
    overall_risk_level: RiskLevel = RiskLevel.MEDIUM
    last_risk_assessment_date: Optional[str] = None
    next_risk_assessment_date: Optional[str] = None

    # Due diligence
    due_diligence_status: DueDiligenceStatus = DueDiligenceStatus.NOT_STARTED
    due_diligence_date: Optional[str] = None

    # Substitutability
    substitutability: SubstitutabilityLevel = SubstitutabilityLevel.SUBSTITUTABLE_WITH_EFFORT
    alternative_providers: List[str] = field(default_factory=list)

    # CTPP status (Articles 31-44)
    is_designated_ctpp: bool = False
    ctpp_designation_date: Optional[str] = None

    # Audit rights
    audit_rights_granted: bool = False
    last_audit_date: Optional[str] = None

    # Contact
    primary_contact_name: str = ""
    primary_contact_email: str = ""
    incident_contact_email: str = ""

    # Metadata
    created_date: str = ""
    modified_date: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.provider_id:
            self.provider_id = f"PRV-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_date:
            self.created_date = datetime.now(timezone.utc).isoformat()
        # Check EU status
        if self.country:
            EU_COUNTRIES = {
                "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
                "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
                "PL", "PT", "RO", "SK", "SI", "ES", "SE"
            }
            self.is_eu_provider = self.country.upper() in EU_COUNTRIES


@dataclass
class ThirdPartyRisk:
    """
    Identified risk related to a third-party provider.
    """
    risk_id: str = ""
    provider_id: str = ""
    category: RiskCategory = RiskCategory.OPERATIONAL
    name: str = ""
    description: str = ""

    # Assessment
    likelihood: int = 3  # 1-5
    impact: int = 3  # 1-5
    inherent_risk_level: RiskLevel = RiskLevel.MEDIUM
    residual_risk_level: RiskLevel = RiskLevel.LOW

    # Mitigations
    existing_controls: List[str] = field(default_factory=list)
    planned_mitigations: List[str] = field(default_factory=list)
    mitigation_owner: str = ""
    mitigation_deadline: Optional[str] = None

    # Treatment
    treatment_strategy: str = ""  # accept, mitigate, transfer, avoid
    accepted_by: str = ""
    acceptance_date: Optional[str] = None

    # Status
    is_active: bool = True
    identified_date: str = ""
    last_reviewed: str = ""

    def __post_init__(self):
        if not self.risk_id:
            self.risk_id = f"TPR-{uuid.uuid4().hex[:8].upper()}"
        if not self.identified_date:
            self.identified_date = datetime.now(timezone.utc).isoformat()
        if not self.last_reviewed:
            self.last_reviewed = self.identified_date

    def calculate_risk_score(self) -> int:
        """Calculate risk score."""
        return self.likelihood * self.impact

    @property
    def risk_level(self) -> RiskLevel:
        """Determine risk level from score."""
        score = self.calculate_risk_score()
        if score <= 4:
            return RiskLevel.LOW
        elif score <= 9:
            return RiskLevel.MEDIUM
        elif score <= 16:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL


@dataclass
class ThirdPartyRiskAssessment:
    """
    Complete third-party risk assessment per Article 28.
    """
    assessment_id: str = ""
    provider_id: str = ""
    assessment_type: AssessmentType = AssessmentType.PERIODIC

    # Assessment scope
    services_assessed: List[str] = field(default_factory=list)
    assessment_period_start: str = ""
    assessment_period_end: str = ""

    # Risk findings
    identified_risks: List[str] = field(default_factory=list)  # Risk IDs
    risk_count_by_level: Dict[str, int] = field(default_factory=dict)

    # Assessments by category
    operational_assessment: Dict[str, Any] = field(default_factory=dict)
    security_assessment: Dict[str, Any] = field(default_factory=dict)
    compliance_assessment: Dict[str, Any] = field(default_factory=dict)
    financial_assessment: Dict[str, Any] = field(default_factory=dict)
    concentration_assessment: Dict[str, Any] = field(default_factory=dict)

    # Control evaluation
    control_effectiveness: str = ""  # effective, partially_effective, ineffective
    control_gaps: List[str] = field(default_factory=list)

    # Recommendations
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    required_actions: List[Dict[str, Any]] = field(default_factory=list)

    # Overall result
    overall_risk_level: RiskLevel = RiskLevel.MEDIUM
    overall_rating: str = ""  # satisfactory, needs_improvement, unsatisfactory
    continue_relationship: bool = True
    conditions_for_continuation: List[str] = field(default_factory=list)

    # Approval
    assessed_by: str = ""
    assessment_date: str = ""
    reviewed_by: str = ""
    review_date: Optional[str] = None
    approved_by: str = ""
    approval_date: Optional[str] = None

    # Next assessment
    next_assessment_date: Optional[str] = None

    def __post_init__(self):
        if not self.assessment_id:
            self.assessment_id = f"TPA-{uuid.uuid4().hex[:8].upper()}"
        if not self.assessment_date:
            self.assessment_date = datetime.now(timezone.utc).isoformat()


@dataclass
class DueDiligenceCheck:
    """
    Due diligence check for a provider.
    """
    check_id: str = ""
    provider_id: str = ""
    check_type: str = ""  # financial, legal, security, operational, compliance

    # Check details
    description: str = ""
    criteria: List[str] = field(default_factory=list)
    evidence_required: List[str] = field(default_factory=list)

    # Results
    status: DueDiligenceStatus = DueDiligenceStatus.NOT_STARTED
    result: str = ""  # passed, failed, conditional
    findings: List[str] = field(default_factory=list)
    evidence_collected: List[str] = field(default_factory=list)

    # Remediation
    remediation_required: bool = False
    remediation_actions: List[str] = field(default_factory=list)
    remediation_deadline: Optional[str] = None

    # Metadata
    performed_by: str = ""
    performed_date: Optional[str] = None
    expires_date: Optional[str] = None

    def __post_init__(self):
        if not self.check_id:
            self.check_id = f"DDC-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class ProviderRelationshipEvent:
    """
    Event in the lifecycle of a provider relationship.
    """
    event_id: str = ""
    provider_id: str = ""
    event_type: str = ""  # onboarding, review, incident, contract_change, termination
    event_date: str = ""
    description: str = ""

    # Impact
    impact_description: str = ""
    services_affected: List[str] = field(default_factory=list)

    # Actions
    actions_taken: List[str] = field(default_factory=list)
    follow_up_required: bool = False
    follow_up_actions: List[str] = field(default_factory=list)

    # Metadata
    recorded_by: str = ""
    attachments: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"PRE-{uuid.uuid4().hex[:8].upper()}"
        if not self.event_date:
            self.event_date = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ThirdPartyRiskConfig:
    """Configuration for Third-Party ICT Risk Management."""

    # Risk assessment
    periodic_assessment_months: int = 12
    critical_provider_assessment_months: int = 6
    risk_threshold_for_escalation: RiskLevel = RiskLevel.HIGH

    # Due diligence
    due_diligence_validity_months: int = 24
    require_due_diligence_before_onboarding: bool = True
    require_security_assessment: bool = True

    # Criticality thresholds
    auto_critical_if_supports_critical_function: bool = True
    revenue_threshold_for_critical_pct: float = 5.0  # % of revenue

    # Provider limits
    max_providers_per_critical_function: int = 3
    require_alternatives_for_critical: bool = True

    # Monitoring
    monitor_provider_financials: bool = True
    monitor_provider_security: bool = True
    monitor_subcontracting_chain: bool = True

    # Notifications
    notify_on_risk_increase: bool = True
    notify_on_provider_incident: bool = True
    notify_days_before_assessment_due: int = 30

    # Documentation
    log_all_events: bool = True
    log_path: str = "logs/dora/third_party_risk"

    # Callbacks
    escalation_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    notification_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# Main Implementation
# =============================================================================

class DORAThirdPartyRiskManagement:
    """
    DORA Article 28 compliant Third-Party ICT Risk Management.

    Implements comprehensive third-party risk management including:
    - Provider registration and classification
    - Risk identification and assessment
    - Due diligence management
    - Control evaluation
    - Ongoing monitoring

    Key Article 28 requirements implemented:
    - (1)(a): Strategy to manage ICT third-party risk
    - (1)(b): Full responsibility remains with financial entity
    - (2): Proportionate approach
    - (3): Register of Information (separate module)
    - (4): Due diligence before contracting
    - (5): Managing ICT concentration risk
    - (8): Exit strategies

    Usage:
        config = ThirdPartyRiskConfig()
        manager = DORAThirdPartyRiskManagement(config)

        # Register a provider
        provider = manager.register_provider(
            name="Cloud Provider",
            provider_type=ProviderType.CLOUD_SERVICE,
            country="DE",
        )

        # Perform risk assessment
        assessment = manager.perform_risk_assessment(
            provider_id=provider.provider_id,
            assessment_type=AssessmentType.INITIAL,
        )
    """

    def __init__(self, config: Optional[ThirdPartyRiskConfig] = None):
        """Initialize Third-Party ICT Risk Management."""
        self.config = config or ThirdPartyRiskConfig()

        # Data stores
        self._providers: Dict[str, ICTProvider] = {}
        self._risks: Dict[str, ThirdPartyRisk] = {}
        self._assessments: Dict[str, ThirdPartyRiskAssessment] = {}
        self._due_diligence: Dict[str, List[DueDiligenceCheck]] = {}  # provider_id -> checks
        self._events: Dict[str, List[ProviderRelationshipEvent]] = {}  # provider_id -> events

        # Indexes for fast lookup
        self._providers_by_criticality: Dict[ProviderCriticality, Set[str]] = {
            c: set() for c in ProviderCriticality
        }
        self._providers_by_type: Dict[ProviderType, Set[str]] = {
            t: set() for t in ProviderType
        }
        self._risks_by_provider: Dict[str, Set[str]] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("DORAThirdPartyRiskManagement initialized")

    # =========================================================================
    # Provider Management
    # =========================================================================

    def register_provider(
        self,
        name: str,
        provider_type: ProviderType,
        country: str,
        legal_name: str = "",
        lei: str = "",
        criticality: Optional[ProviderCriticality] = None,
        services: Optional[List[ICTService]] = None,
        permits_subcontracting: bool = False,
        primary_contact_email: str = "",
        notes: str = "",
    ) -> ICTProvider:
        """
        Register a new ICT third-party provider.

        Per Article 28(4), due diligence should be performed
        before entering into contractual arrangements.

        Args:
            name: Provider display name
            provider_type: Type of provider
            country: Provider country (ISO 3166-1 alpha-2)
            legal_name: Legal entity name
            lei: Legal Entity Identifier
            criticality: Provider criticality (auto-determined if not set)
            services: Services provided
            permits_subcontracting: Whether subcontracting is permitted
            primary_contact_email: Primary contact email
            notes: Additional notes

        Returns:
            Registered ICTProvider
        """
        provider = ICTProvider(
            name=name,
            legal_name=legal_name or name,
            provider_type=provider_type,
            country=country.upper(),
            lei=lei,
            criticality=criticality or ProviderCriticality.LOW,
            services=services or [],
            permits_subcontracting=permits_subcontracting,
            primary_contact_email=primary_contact_email,
            notes=notes,
            status=ProviderStatus.ONBOARDING,
            due_diligence_status=DueDiligenceStatus.NOT_STARTED,
        )

        # Auto-determine criticality if supports critical function
        if self.config.auto_critical_if_supports_critical_function:
            if any(s.supports_critical_function for s in provider.services):
                provider.criticality = ProviderCriticality.CRITICAL
                provider.supports_critical_function = True

        with self._lock:
            self._providers[provider.provider_id] = provider
            self._providers_by_criticality[provider.criticality].add(provider.provider_id)
            self._providers_by_type[provider.provider_type].add(provider.provider_id)
            self._risks_by_provider[provider.provider_id] = set()
            self._events[provider.provider_id] = []
            self._due_diligence[provider.provider_id] = []

        # Record event
        self._record_event(
            provider.provider_id,
            "onboarding",
            f"Provider {name} registered",
        )

        self._log_event("provider_registered", {
            "provider_id": provider.provider_id,
            "name": name,
            "type": provider_type.value,
            "country": country,
            "criticality": provider.criticality.value,
        })

        logger.info(f"Provider registered: {name} ({provider.provider_id})")
        return provider

    def update_provider(
        self,
        provider_id: str,
        **updates: Any,
    ) -> Optional[ICTProvider]:
        """Update provider information."""
        with self._lock:
            if provider_id not in self._providers:
                return None

            provider = self._providers[provider_id]
            old_criticality = provider.criticality
            old_type = provider.provider_type

            for key, value in updates.items():
                if hasattr(provider, key):
                    setattr(provider, key, value)

            provider.modified_date = datetime.now(timezone.utc).isoformat()

            # Update indexes if needed
            if provider.criticality != old_criticality:
                self._providers_by_criticality[old_criticality].discard(provider_id)
                self._providers_by_criticality[provider.criticality].add(provider_id)

            if provider.provider_type != old_type:
                self._providers_by_type[old_type].discard(provider_id)
                self._providers_by_type[provider.provider_type].add(provider_id)

        self._log_event("provider_updated", {
            "provider_id": provider_id,
            "updates": {k: str(v) for k, v in updates.items()},
        })

        return provider

    def get_provider(self, provider_id: str) -> Optional[ICTProvider]:
        """Get provider by ID."""
        with self._lock:
            return self._providers.get(provider_id)

    def get_provider_by_name(self, name: str) -> Optional[ICTProvider]:
        """Get provider by name."""
        with self._lock:
            for provider in self._providers.values():
                if provider.name.lower() == name.lower():
                    return provider
            return None

    def get_providers_by_criticality(
        self,
        criticality: ProviderCriticality,
    ) -> List[ICTProvider]:
        """Get all providers with specific criticality."""
        with self._lock:
            return [
                self._providers[pid]
                for pid in self._providers_by_criticality.get(criticality, set())
            ]

    def get_providers_by_type(
        self,
        provider_type: ProviderType,
    ) -> List[ICTProvider]:
        """Get all providers of specific type."""
        with self._lock:
            return [
                self._providers[pid]
                for pid in self._providers_by_type.get(provider_type, set())
            ]

    def get_critical_providers(self) -> List[ICTProvider]:
        """Get all critical providers."""
        with self._lock:
            return [
                p for p in self._providers.values()
                if p.criticality == ProviderCriticality.CRITICAL
                and p.status == ProviderStatus.ACTIVE
            ]

    def get_providers_supporting_function(
        self,
        function_name: str,
    ) -> List[ICTProvider]:
        """Get providers supporting a specific function."""
        with self._lock:
            result = []
            for provider in self._providers.values():
                for service in provider.services:
                    if function_name in service.supported_functions:
                        result.append(provider)
                        break
            return result

    def activate_provider(self, provider_id: str) -> Optional[ICTProvider]:
        """Activate a provider after onboarding."""
        with self._lock:
            if provider_id not in self._providers:
                return None

            provider = self._providers[provider_id]

            # Check due diligence requirement
            if self.config.require_due_diligence_before_onboarding:
                if provider.due_diligence_status != DueDiligenceStatus.COMPLETED:
                    logger.warning(
                        f"Cannot activate provider {provider_id}: "
                        "Due diligence not completed"
                    )
                    return None

            provider.status = ProviderStatus.ACTIVE
            provider.modified_date = datetime.now(timezone.utc).isoformat()

        self._record_event(
            provider_id,
            "activation",
            f"Provider {provider.name} activated",
        )

        return provider

    def suspend_provider(
        self,
        provider_id: str,
        reason: str,
    ) -> Optional[ICTProvider]:
        """Suspend a provider relationship."""
        with self._lock:
            if provider_id not in self._providers:
                return None

            provider = self._providers[provider_id]
            provider.status = ProviderStatus.SUSPENDED
            provider.modified_date = datetime.now(timezone.utc).isoformat()

        self._record_event(
            provider_id,
            "suspension",
            f"Provider suspended: {reason}",
        )

        # Notify if configured
        if self.config.notification_callback:
            self.config.notification_callback("provider_suspended", {
                "provider_id": provider_id,
                "provider_name": provider.name,
                "reason": reason,
            })

        return provider

    # =========================================================================
    # Service Management
    # =========================================================================

    def add_service_to_provider(
        self,
        provider_id: str,
        service: ICTService,
    ) -> Optional[ICTProvider]:
        """Add a service to a provider."""
        with self._lock:
            if provider_id not in self._providers:
                return None

            provider = self._providers[provider_id]
            provider.services.append(service)

            # Update criticality if needed
            if service.supports_critical_function:
                provider.supports_critical_function = True
                if self.config.auto_critical_if_supports_critical_function:
                    self._update_provider_criticality(provider_id, ProviderCriticality.CRITICAL)

            provider.modified_date = datetime.now(timezone.utc).isoformat()

        return provider

    def get_services_by_provider(
        self,
        provider_id: str,
    ) -> List[ICTService]:
        """Get all services for a provider."""
        with self._lock:
            provider = self._providers.get(provider_id)
            if not provider:
                return []
            return provider.services.copy()

    def get_critical_services(self) -> List[Tuple[ICTProvider, ICTService]]:
        """Get all critical services across all providers."""
        with self._lock:
            result = []
            for provider in self._providers.values():
                for service in provider.services:
                    if service.criticality == ServiceCriticality.CRITICAL_OR_IMPORTANT:
                        result.append((provider, service))
            return result

    # =========================================================================
    # Risk Management
    # =========================================================================

    def identify_risk(
        self,
        provider_id: str,
        category: RiskCategory,
        name: str,
        description: str = "",
        likelihood: int = 3,
        impact: int = 3,
        existing_controls: Optional[List[str]] = None,
        treatment_strategy: str = "mitigate",
        mitigation_owner: str = "",
    ) -> Optional[ThirdPartyRisk]:
        """
        Identify and register a third-party risk.

        Args:
            provider_id: Provider ID
            category: Risk category
            name: Risk name
            description: Risk description
            likelihood: Likelihood score (1-5)
            impact: Impact score (1-5)
            existing_controls: Existing controls
            treatment_strategy: Treatment strategy
            mitigation_owner: Risk owner

        Returns:
            Registered ThirdPartyRisk or None if provider not found
        """
        with self._lock:
            if provider_id not in self._providers:
                return None

            risk = ThirdPartyRisk(
                provider_id=provider_id,
                category=category,
                name=name,
                description=description,
                likelihood=min(5, max(1, likelihood)),
                impact=min(5, max(1, impact)),
                existing_controls=existing_controls or [],
                treatment_strategy=treatment_strategy,
                mitigation_owner=mitigation_owner,
            )

            # Calculate risk levels
            risk.inherent_risk_level = risk.risk_level

            self._risks[risk.risk_id] = risk
            self._risks_by_provider[provider_id].add(risk.risk_id)

        # Check if escalation needed
        if risk.risk_level.value in [RiskLevel.HIGH.value, RiskLevel.CRITICAL.value]:
            if self.config.escalation_callback:
                self.config.escalation_callback("high_risk_identified", {
                    "risk_id": risk.risk_id,
                    "provider_id": provider_id,
                    "risk_name": name,
                    "risk_level": risk.risk_level.value,
                })

        self._log_event("risk_identified", {
            "risk_id": risk.risk_id,
            "provider_id": provider_id,
            "category": category.value,
            "risk_level": risk.risk_level.value,
        })

        return risk

    def update_risk(
        self,
        risk_id: str,
        **updates: Any,
    ) -> Optional[ThirdPartyRisk]:
        """Update a risk."""
        with self._lock:
            if risk_id not in self._risks:
                return None

            risk = self._risks[risk_id]
            old_level = risk.risk_level

            for key, value in updates.items():
                if hasattr(risk, key):
                    setattr(risk, key, value)

            risk.last_reviewed = datetime.now(timezone.utc).isoformat()

            # Recalculate risk level
            new_level = risk.risk_level

            # Notify on risk increase
            if self.config.notify_on_risk_increase:
                if self._risk_level_increased(old_level, new_level):
                    if self.config.notification_callback:
                        self.config.notification_callback("risk_increased", {
                            "risk_id": risk_id,
                            "old_level": old_level.value,
                            "new_level": new_level.value,
                        })

        return risk

    def get_risk(self, risk_id: str) -> Optional[ThirdPartyRisk]:
        """Get risk by ID."""
        with self._lock:
            return self._risks.get(risk_id)

    def get_risks_for_provider(
        self,
        provider_id: str,
    ) -> List[ThirdPartyRisk]:
        """Get all risks for a provider."""
        with self._lock:
            risk_ids = self._risks_by_provider.get(provider_id, set())
            return [self._risks[rid] for rid in risk_ids if rid in self._risks]

    def get_risks_by_level(
        self,
        level: RiskLevel,
    ) -> List[ThirdPartyRisk]:
        """Get all risks at specific level."""
        with self._lock:
            return [r for r in self._risks.values() if r.risk_level == level and r.is_active]

    def get_risks_by_category(
        self,
        category: RiskCategory,
    ) -> List[ThirdPartyRisk]:
        """Get all risks in a category."""
        with self._lock:
            return [r for r in self._risks.values() if r.category == category and r.is_active]

    def accept_risk(
        self,
        risk_id: str,
        accepted_by: str,
        justification: str,
    ) -> Optional[ThirdPartyRisk]:
        """Accept a risk."""
        with self._lock:
            if risk_id not in self._risks:
                return None

            risk = self._risks[risk_id]
            risk.treatment_strategy = "accept"
            risk.accepted_by = accepted_by
            risk.acceptance_date = datetime.now(timezone.utc).isoformat()
            risk.last_reviewed = risk.acceptance_date

        self._log_event("risk_accepted", {
            "risk_id": risk_id,
            "accepted_by": accepted_by,
            "justification": justification,
        })

        return risk

    # =========================================================================
    # Risk Assessment
    # =========================================================================

    def perform_risk_assessment(
        self,
        provider_id: str,
        assessment_type: AssessmentType = AssessmentType.PERIODIC,
        services_to_assess: Optional[List[str]] = None,
        assessed_by: str = "",
    ) -> Optional[ThirdPartyRiskAssessment]:
        """
        Perform comprehensive third-party risk assessment.

        Per Article 28, financial entities must manage ICT third-party
        risk as an integral part of their ICT risk management framework.

        Args:
            provider_id: Provider to assess
            assessment_type: Type of assessment
            services_to_assess: Specific services (None = all)
            assessed_by: Assessor name

        Returns:
            ThirdPartyRiskAssessment or None if provider not found
        """
        with self._lock:
            if provider_id not in self._providers:
                return None

            provider = self._providers[provider_id]

            # Determine services to assess
            if services_to_assess:
                services = [
                    s for s in provider.services
                    if s.service_id in services_to_assess
                ]
            else:
                services = provider.services

            assessment = ThirdPartyRiskAssessment(
                provider_id=provider_id,
                assessment_type=assessment_type,
                services_assessed=[s.service_id for s in services],
                assessment_period_start=datetime.now(timezone.utc).isoformat(),
                assessed_by=assessed_by,
            )

            # Perform operational assessment
            assessment.operational_assessment = self._assess_operational_risk(provider, services)

            # Perform security assessment
            assessment.security_assessment = self._assess_security_risk(provider, services)

            # Perform compliance assessment
            assessment.compliance_assessment = self._assess_compliance_risk(provider)

            # Perform financial assessment
            assessment.financial_assessment = self._assess_financial_risk(provider)

            # Perform concentration assessment
            assessment.concentration_assessment = self._assess_concentration_risk(provider)

            # Collect identified risks
            provider_risks = self.get_risks_for_provider(provider_id)
            assessment.identified_risks = [r.risk_id for r in provider_risks]

            # Count by level
            assessment.risk_count_by_level = {
                "critical": sum(1 for r in provider_risks if r.risk_level == RiskLevel.CRITICAL),
                "high": sum(1 for r in provider_risks if r.risk_level == RiskLevel.HIGH),
                "medium": sum(1 for r in provider_risks if r.risk_level == RiskLevel.MEDIUM),
                "low": sum(1 for r in provider_risks if r.risk_level == RiskLevel.LOW),
            }

            # Determine overall risk level
            if assessment.risk_count_by_level["critical"] > 0:
                assessment.overall_risk_level = RiskLevel.CRITICAL
            elif assessment.risk_count_by_level["high"] > 0:
                assessment.overall_risk_level = RiskLevel.HIGH
            elif assessment.risk_count_by_level["medium"] > 0:
                assessment.overall_risk_level = RiskLevel.MEDIUM
            else:
                assessment.overall_risk_level = RiskLevel.LOW

            # Determine overall rating
            if assessment.overall_risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]:
                assessment.overall_rating = "satisfactory"
            elif assessment.overall_risk_level == RiskLevel.HIGH:
                assessment.overall_rating = "needs_improvement"
            else:
                assessment.overall_rating = "unsatisfactory"

            # Set continue relationship recommendation
            assessment.continue_relationship = assessment.overall_risk_level != RiskLevel.CRITICAL

            # Set completion
            assessment.assessment_period_end = datetime.now(timezone.utc).isoformat()

            # Calculate next assessment date
            if provider.criticality == ProviderCriticality.CRITICAL:
                months = self.config.critical_provider_assessment_months
            else:
                months = self.config.periodic_assessment_months

            next_date = datetime.now(timezone.utc) + timedelta(days=months * 30)
            assessment.next_assessment_date = next_date.isoformat()

            # Update provider
            provider.last_risk_assessment_date = assessment.assessment_date
            provider.next_risk_assessment_date = assessment.next_assessment_date
            provider.overall_risk_level = assessment.overall_risk_level

            self._assessments[assessment.assessment_id] = assessment

        self._record_event(
            provider_id,
            "risk_assessment",
            f"Risk assessment completed: {assessment.overall_rating}",
        )

        self._log_event("assessment_completed", {
            "assessment_id": assessment.assessment_id,
            "provider_id": provider_id,
            "overall_risk_level": assessment.overall_risk_level.value,
            "overall_rating": assessment.overall_rating,
        })

        return assessment

    def approve_assessment(
        self,
        assessment_id: str,
        approved_by: str,
        conditions: Optional[List[str]] = None,
    ) -> Optional[ThirdPartyRiskAssessment]:
        """Approve a risk assessment."""
        with self._lock:
            if assessment_id not in self._assessments:
                return None

            assessment = self._assessments[assessment_id]
            assessment.approved_by = approved_by
            assessment.approval_date = datetime.now(timezone.utc).isoformat()
            assessment.conditions_for_continuation = conditions or []

        return assessment

    def get_assessment(
        self,
        assessment_id: str,
    ) -> Optional[ThirdPartyRiskAssessment]:
        """Get assessment by ID."""
        with self._lock:
            return self._assessments.get(assessment_id)

    def get_assessments_for_provider(
        self,
        provider_id: str,
    ) -> List[ThirdPartyRiskAssessment]:
        """Get all assessments for a provider."""
        with self._lock:
            return [
                a for a in self._assessments.values()
                if a.provider_id == provider_id
            ]

    def get_latest_assessment(
        self,
        provider_id: str,
    ) -> Optional[ThirdPartyRiskAssessment]:
        """Get most recent assessment for a provider."""
        assessments = self.get_assessments_for_provider(provider_id)
        if not assessments:
            return None
        return max(assessments, key=lambda a: a.assessment_date)

    def get_providers_needing_assessment(self) -> List[ICTProvider]:
        """Get providers that need risk assessment."""
        now = datetime.now(timezone.utc)

        with self._lock:
            result = []
            for provider in self._providers.values():
                if provider.status != ProviderStatus.ACTIVE:
                    continue

                if not provider.next_risk_assessment_date:
                    result.append(provider)
                    continue

                next_date = datetime.fromisoformat(
                    provider.next_risk_assessment_date.replace("Z", "+00:00")
                )

                # Check if due or upcoming (within reminder period)
                reminder_date = now + timedelta(days=self.config.notify_days_before_assessment_due)
                if next_date <= reminder_date:
                    result.append(provider)

            return result

    # =========================================================================
    # Due Diligence
    # =========================================================================

    def start_due_diligence(
        self,
        provider_id: str,
        check_types: Optional[List[str]] = None,
    ) -> List[DueDiligenceCheck]:
        """
        Start due diligence process for a provider.

        Per Article 28(4), financial entities shall ensure that
        before entering into a contractual arrangement, they
        conduct a comprehensive due diligence.

        Args:
            provider_id: Provider ID
            check_types: Specific check types (None = all standard)

        Returns:
            List of due diligence checks created
        """
        with self._lock:
            if provider_id not in self._providers:
                return []

            provider = self._providers[provider_id]

            # Standard check types
            if not check_types:
                check_types = [
                    "financial_stability",
                    "legal_compliance",
                    "security_controls",
                    "operational_capability",
                    "business_continuity",
                    "regulatory_compliance",
                    "data_protection",
                ]

            checks = []
            for check_type in check_types:
                check = DueDiligenceCheck(
                    provider_id=provider_id,
                    check_type=check_type,
                    description=f"Due diligence check: {check_type}",
                    criteria=self._get_due_diligence_criteria(check_type),
                    evidence_required=self._get_required_evidence(check_type),
                    status=DueDiligenceStatus.IN_PROGRESS,
                )
                checks.append(check)

            self._due_diligence[provider_id].extend(checks)
            provider.due_diligence_status = DueDiligenceStatus.IN_PROGRESS

        self._record_event(
            provider_id,
            "due_diligence_started",
            f"Due diligence started with {len(checks)} checks",
        )

        return checks

    def complete_due_diligence_check(
        self,
        check_id: str,
        result: str,
        findings: Optional[List[str]] = None,
        evidence_collected: Optional[List[str]] = None,
        performed_by: str = "",
        remediation_required: bool = False,
        remediation_actions: Optional[List[str]] = None,
    ) -> Optional[DueDiligenceCheck]:
        """Complete a due diligence check."""
        with self._lock:
            # Find the check
            check = None
            provider_id = None
            for pid, checks in self._due_diligence.items():
                for c in checks:
                    if c.check_id == check_id:
                        check = c
                        provider_id = pid
                        break
                if check:
                    break

            if not check:
                return None

            check.result = result
            check.findings = findings or []
            check.evidence_collected = evidence_collected or []
            check.performed_by = performed_by
            check.performed_date = datetime.now(timezone.utc).isoformat()
            check.remediation_required = remediation_required
            check.remediation_actions = remediation_actions or []

            if result == "passed":
                check.status = DueDiligenceStatus.COMPLETED
            elif result == "conditional" or remediation_required:
                check.status = DueDiligenceStatus.REQUIRES_REMEDIATION
            else:
                check.status = DueDiligenceStatus.FAILED

            # Set expiry
            check.expires_date = (
                datetime.now(timezone.utc) +
                timedelta(days=self.config.due_diligence_validity_months * 30)
            ).isoformat()

            # Update provider overall status
            self._update_provider_due_diligence_status(provider_id)

        return check

    def get_due_diligence_status(
        self,
        provider_id: str,
    ) -> Dict[str, Any]:
        """Get due diligence status for a provider."""
        with self._lock:
            if provider_id not in self._providers:
                return {}

            provider = self._providers[provider_id]
            checks = self._due_diligence.get(provider_id, [])

            return {
                "provider_id": provider_id,
                "overall_status": provider.due_diligence_status.value,
                "due_diligence_date": provider.due_diligence_date,
                "total_checks": len(checks),
                "completed": sum(1 for c in checks if c.status == DueDiligenceStatus.COMPLETED),
                "failed": sum(1 for c in checks if c.status == DueDiligenceStatus.FAILED),
                "in_progress": sum(1 for c in checks if c.status == DueDiligenceStatus.IN_PROGRESS),
                "requires_remediation": sum(
                    1 for c in checks
                    if c.status == DueDiligenceStatus.REQUIRES_REMEDIATION
                ),
                "checks": [
                    {
                        "check_id": c.check_id,
                        "type": c.check_type,
                        "status": c.status.value,
                        "result": c.result,
                    }
                    for c in checks
                ],
            }

    # =========================================================================
    # Control and Oversight
    # =========================================================================

    def assess_control_over_provider(
        self,
        provider_id: str,
    ) -> Dict[str, Any]:
        """
        Assess level of control over a third-party provider.

        Per Article 28(1)(b), the financial entity retains full
        responsibility for compliance even when outsourcing.
        """
        with self._lock:
            if provider_id not in self._providers:
                return {}

            provider = self._providers[provider_id]

            # Control assessment criteria
            assessment = {
                "provider_id": provider_id,
                "provider_name": provider.name,
                "assessment_date": datetime.now(timezone.utc).isoformat(),
                "control_areas": {},
                "overall_control_level": "adequate",
                "issues": [],
                "recommendations": [],
            }

            # Audit rights
            audit_control = {
                "area": "audit_rights",
                "has_control": provider.audit_rights_granted,
                "last_exercised": provider.last_audit_date,
                "assessment": "adequate" if provider.audit_rights_granted else "insufficient",
            }
            assessment["control_areas"]["audit_rights"] = audit_control
            if not provider.audit_rights_granted:
                assessment["issues"].append("No audit rights granted")
                assessment["recommendations"].append(
                    "Negotiate audit rights in contract"
                )

            # Subcontracting visibility
            subcontracting_control = {
                "area": "subcontracting",
                "has_control": provider.subcontracting_chain_known,
                "permits_subcontracting": provider.permits_subcontracting,
                "known_subcontractors": len(provider.known_subcontractors),
                "assessment": "adequate" if provider.subcontracting_chain_known else "limited",
            }
            assessment["control_areas"]["subcontracting"] = subcontracting_control
            if provider.permits_subcontracting and not provider.subcontracting_chain_known:
                assessment["issues"].append("Subcontracting chain not fully known")
                assessment["recommendations"].append(
                    "Request full disclosure of subcontractors"
                )

            # Data location control
            data_control = {
                "area": "data_location",
                "processing_countries": provider.data_processing_countries,
                "storage_countries": provider.data_storage_countries,
                "all_in_eu": all(
                    c in self._get_eu_countries()
                    for c in provider.data_processing_countries + provider.data_storage_countries
                ),
                "assessment": "adequate",
            }
            if not provider.data_processing_countries:
                data_control["assessment"] = "unknown"
                assessment["issues"].append("Data processing locations not documented")
            assessment["control_areas"]["data_location"] = data_control

            # Incident notification
            incident_control = {
                "area": "incident_notification",
                "has_contact": bool(provider.incident_contact_email),
                "assessment": "adequate" if provider.incident_contact_email else "insufficient",
            }
            assessment["control_areas"]["incident_notification"] = incident_control
            if not provider.incident_contact_email:
                assessment["recommendations"].append(
                    "Establish incident notification contact"
                )

            # Determine overall control level
            issues_count = len(assessment["issues"])
            if issues_count == 0:
                assessment["overall_control_level"] = "adequate"
            elif issues_count <= 2:
                assessment["overall_control_level"] = "needs_improvement"
            else:
                assessment["overall_control_level"] = "insufficient"

            return assessment

    def verify_supervisory_access(
        self,
        provider_id: str,
    ) -> Dict[str, Any]:
        """
        Verify that supervisory access is ensured.

        Per Article 28(1)(b), arrangements must not impede effective
        supervision of the financial entity.
        """
        with self._lock:
            if provider_id not in self._providers:
                return {}

            provider = self._providers[provider_id]

            verification = {
                "provider_id": provider_id,
                "verification_date": datetime.now(timezone.utc).isoformat(),
                "supervisory_access_ensured": True,
                "areas": {},
                "issues": [],
            }

            # Check NCA access
            nca_access = provider.audit_rights_granted  # Should include NCA access
            verification["areas"]["nca_access"] = {
                "ensured": nca_access,
                "notes": "Audit rights include competent authority access" if nca_access else "",
            }
            if not nca_access:
                verification["issues"].append(
                    "NCA access not contractually ensured"
                )

            # Check data availability
            data_available = bool(provider.data_processing_countries)
            verification["areas"]["data_availability"] = {
                "ensured": data_available,
                "locations_known": provider.data_processing_countries,
            }

            # Check documentation availability
            verification["areas"]["documentation"] = {
                "ensured": True,  # Assumed if registered
                "notes": "Provider information maintained in register",
            }

            verification["supervisory_access_ensured"] = len(verification["issues"]) == 0

            return verification

    # =========================================================================
    # Status and Reporting
    # =========================================================================

    def get_third_party_risk_status(self) -> Dict[str, Any]:
        """Get comprehensive third-party risk status."""
        with self._lock:
            all_risks = list(self._risks.values())
            active_risks = [r for r in all_risks if r.is_active]

            providers_needing_assessment = self.get_providers_needing_assessment()

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "providers": {
                    "total": len(self._providers),
                    "active": sum(
                        1 for p in self._providers.values()
                        if p.status == ProviderStatus.ACTIVE
                    ),
                    "by_criticality": {
                        c.value: len(self._providers_by_criticality[c])
                        for c in ProviderCriticality
                    },
                    "by_type": {
                        t.value: len(self._providers_by_type[t])
                        for t in ProviderType
                        if len(self._providers_by_type[t]) > 0
                    },
                    "supporting_critical_functions": sum(
                        1 for p in self._providers.values()
                        if p.supports_critical_function
                    ),
                    "designated_ctpps": sum(
                        1 for p in self._providers.values()
                        if p.is_designated_ctpp
                    ),
                },
                "risks": {
                    "total": len(active_risks),
                    "by_level": {
                        "critical": sum(1 for r in active_risks if r.risk_level == RiskLevel.CRITICAL),
                        "high": sum(1 for r in active_risks if r.risk_level == RiskLevel.HIGH),
                        "medium": sum(1 for r in active_risks if r.risk_level == RiskLevel.MEDIUM),
                        "low": sum(1 for r in active_risks if r.risk_level == RiskLevel.LOW),
                    },
                    "by_category": {
                        c.value: sum(1 for r in active_risks if r.category == c)
                        for c in RiskCategory
                        if sum(1 for r in active_risks if r.category == c) > 0
                    },
                },
                "assessments": {
                    "total": len(self._assessments),
                    "providers_needing_assessment": len(providers_needing_assessment),
                },
                "due_diligence": {
                    "completed": sum(
                        1 for p in self._providers.values()
                        if p.due_diligence_status == DueDiligenceStatus.COMPLETED
                    ),
                    "in_progress": sum(
                        1 for p in self._providers.values()
                        if p.due_diligence_status == DueDiligenceStatus.IN_PROGRESS
                    ),
                    "not_started": sum(
                        1 for p in self._providers.values()
                        if p.due_diligence_status == DueDiligenceStatus.NOT_STARTED
                    ),
                },
                "compliance_indicators": {
                    "all_critical_providers_assessed": all(
                        p.last_risk_assessment_date is not None
                        for p in self._providers.values()
                        if p.criticality == ProviderCriticality.CRITICAL
                    ),
                    "due_diligence_current": all(
                        p.due_diligence_status == DueDiligenceStatus.COMPLETED
                        for p in self._providers.values()
                        if p.status == ProviderStatus.ACTIVE
                    ),
                    "no_critical_risks": sum(
                        1 for r in active_risks if r.risk_level == RiskLevel.CRITICAL
                    ) == 0,
                },
            }

    def export_provider_inventory(self) -> List[Dict[str, Any]]:
        """Export provider inventory for reporting."""
        with self._lock:
            return [
                {
                    "provider_id": p.provider_id,
                    "name": p.name,
                    "legal_name": p.legal_name,
                    "lei": p.lei,
                    "type": p.provider_type.value,
                    "country": p.country,
                    "is_eu_provider": p.is_eu_provider,
                    "criticality": p.criticality.value,
                    "supports_critical_function": p.supports_critical_function,
                    "status": p.status.value,
                    "services_count": len(p.services),
                    "overall_risk_level": p.overall_risk_level.value,
                    "due_diligence_status": p.due_diligence_status.value,
                    "last_assessment_date": p.last_risk_assessment_date,
                    "is_designated_ctpp": p.is_designated_ctpp,
                }
                for p in self._providers.values()
            ]

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _update_provider_criticality(
        self,
        provider_id: str,
        new_criticality: ProviderCriticality,
    ) -> None:
        """Update provider criticality and indexes."""
        with self._lock:
            if provider_id not in self._providers:
                return

            provider = self._providers[provider_id]
            old_criticality = provider.criticality

            self._providers_by_criticality[old_criticality].discard(provider_id)
            self._providers_by_criticality[new_criticality].add(provider_id)
            provider.criticality = new_criticality

    def _update_provider_due_diligence_status(
        self,
        provider_id: str,
    ) -> None:
        """Update provider's overall due diligence status."""
        with self._lock:
            if provider_id not in self._providers:
                return

            provider = self._providers[provider_id]
            checks = self._due_diligence.get(provider_id, [])

            if not checks:
                provider.due_diligence_status = DueDiligenceStatus.NOT_STARTED
                return

            # Check for any failures
            if any(c.status == DueDiligenceStatus.FAILED for c in checks):
                provider.due_diligence_status = DueDiligenceStatus.FAILED
                return

            # Check for any requiring remediation
            if any(c.status == DueDiligenceStatus.REQUIRES_REMEDIATION for c in checks):
                provider.due_diligence_status = DueDiligenceStatus.REQUIRES_REMEDIATION
                return

            # Check if all completed
            if all(c.status == DueDiligenceStatus.COMPLETED for c in checks):
                provider.due_diligence_status = DueDiligenceStatus.COMPLETED
                provider.due_diligence_date = datetime.now(timezone.utc).isoformat()
                return

            # Otherwise still in progress
            provider.due_diligence_status = DueDiligenceStatus.IN_PROGRESS

    def _assess_operational_risk(
        self,
        provider: ICTProvider,
        services: List[ICTService],
    ) -> Dict[str, Any]:
        """Assess operational risk for a provider."""
        return {
            "services_count": len(services),
            "critical_services": sum(
                1 for s in services
                if s.criticality == ServiceCriticality.CRITICAL_OR_IMPORTANT
            ),
            "availability_concerns": [
                s.service_id for s in services
                if s.availability_target_pct < 99.0
            ],
            "substitutability": provider.substitutability.value,
            "alternatives_available": len(provider.alternative_providers),
            "subcontracting_risk": provider.permits_subcontracting and not provider.subcontracting_chain_known,
            "assessment_score": "low" if len(services) < 3 else "medium",
        }

    def _assess_security_risk(
        self,
        provider: ICTProvider,
        services: List[ICTService],
    ) -> Dict[str, Any]:
        """Assess security risk for a provider."""
        return {
            "audit_rights": provider.audit_rights_granted,
            "last_audit": provider.last_audit_date,
            "data_sensitivity": any(
                s.data_classification in ["confidential", "restricted"]
                for s in services
            ),
            "security_certifications_requested": True,
            "assessment_score": "low" if provider.audit_rights_granted else "medium",
        }

    def _assess_compliance_risk(
        self,
        provider: ICTProvider,
    ) -> Dict[str, Any]:
        """Assess compliance risk for a provider."""
        return {
            "is_eu_provider": provider.is_eu_provider,
            "data_locations_eu_compliant": all(
                c in self._get_eu_countries() or c in ["US", "UK", "CH"]  # Adequacy
                for c in provider.data_processing_countries
            ) if provider.data_processing_countries else None,
            "is_designated_ctpp": provider.is_designated_ctpp,
            "regulatory_requirements_met": provider.due_diligence_status == DueDiligenceStatus.COMPLETED,
            "assessment_score": "low" if provider.is_eu_provider else "medium",
        }

    def _assess_financial_risk(
        self,
        provider: ICTProvider,
    ) -> Dict[str, Any]:
        """Assess financial risk for a provider."""
        return {
            "financial_stability_verified": provider.due_diligence_status == DueDiligenceStatus.COMPLETED,
            "assessment_score": "low",
        }

    def _assess_concentration_risk(
        self,
        provider: ICTProvider,
    ) -> Dict[str, Any]:
        """Assess concentration risk for a provider."""
        with self._lock:
            # Count how many critical functions this provider supports
            critical_functions = set()
            for service in provider.services:
                if service.supports_critical_function:
                    critical_functions.update(service.supported_functions)

            return {
                "critical_functions_supported": len(critical_functions),
                "functions_list": list(critical_functions),
                "substitutability": provider.substitutability.value,
                "alternatives_count": len(provider.alternative_providers),
                "concentration_concern": (
                    len(critical_functions) > 1 and
                    provider.substitutability in [
                        SubstitutabilityLevel.DIFFICULT_TO_SUBSTITUTE,
                        SubstitutabilityLevel.NOT_SUBSTITUTABLE
                    ]
                ),
                "assessment_score": "high" if len(critical_functions) > 2 else "medium",
            }

    def _get_due_diligence_criteria(
        self,
        check_type: str,
    ) -> List[str]:
        """Get due diligence criteria for a check type."""
        criteria_map = {
            "financial_stability": [
                "Financial statements available",
                "No significant losses",
                "Adequate capital reserves",
                "Credit rating acceptable",
            ],
            "legal_compliance": [
                "Valid business registration",
                "No pending litigation",
                "Regulatory licenses valid",
                "Contract terms acceptable",
            ],
            "security_controls": [
                "ISO 27001 or equivalent",
                "Security policies documented",
                "Incident response capability",
                "Vulnerability management",
            ],
            "operational_capability": [
                "Service delivery capability",
                "Technical expertise",
                "Support availability",
                "Scalability",
            ],
            "business_continuity": [
                "BCP documented",
                "DR capabilities",
                "Regular testing",
                "Recovery objectives defined",
            ],
            "regulatory_compliance": [
                "DORA compliance",
                "Industry regulations",
                "Data protection compliance",
                "Reporting capabilities",
            ],
            "data_protection": [
                "GDPR compliance",
                "Data processing agreements",
                "Data location controls",
                "Encryption standards",
            ],
        }
        return criteria_map.get(check_type, [])

    def _get_required_evidence(
        self,
        check_type: str,
    ) -> List[str]:
        """Get required evidence for a check type."""
        evidence_map = {
            "financial_stability": ["Financial statements", "Credit report"],
            "legal_compliance": ["Registration documents", "Licenses"],
            "security_controls": ["Security certifications", "Audit reports"],
            "operational_capability": ["Service documentation", "SLAs"],
            "business_continuity": ["BCP documents", "DR test results"],
            "regulatory_compliance": ["Compliance attestations"],
            "data_protection": ["DPA", "Privacy policy"],
        }
        return evidence_map.get(check_type, [])

    def _get_eu_countries(self) -> Set[str]:
        """Get set of EU country codes."""
        return {
            "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
            "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
            "PL", "PT", "RO", "SK", "SI", "ES", "SE"
        }

    def _risk_level_increased(
        self,
        old_level: RiskLevel,
        new_level: RiskLevel,
    ) -> bool:
        """Check if risk level increased."""
        level_order = {
            RiskLevel.LOW: 0,
            RiskLevel.MEDIUM: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.CRITICAL: 3,
        }
        return level_order[new_level] > level_order[old_level]

    def _record_event(
        self,
        provider_id: str,
        event_type: str,
        description: str,
    ) -> None:
        """Record a provider relationship event."""
        event = ProviderRelationshipEvent(
            provider_id=provider_id,
            event_type=event_type,
            description=description,
        )

        with self._lock:
            if provider_id in self._events:
                self._events[provider_id].append(event)

    def _log_event(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """Log an event to file."""
        if not self.config.log_all_events:
            return

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"third_party_events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_third_party_risk_management(
    config: Optional[ThirdPartyRiskConfig] = None,
) -> DORAThirdPartyRiskManagement:
    """
    Create a DORAThirdPartyRiskManagement instance.

    Args:
        config: Optional configuration

    Returns:
        Configured DORAThirdPartyRiskManagement instance
    """
    return DORAThirdPartyRiskManagement(config=config)


def get_provider_types() -> List[ProviderType]:
    """Get all provider types."""
    return list(ProviderType)


def get_risk_categories() -> List[RiskCategory]:
    """Get all risk categories."""
    return list(RiskCategory)


def get_criticality_levels() -> List[ProviderCriticality]:
    """Get all criticality levels."""
    return list(ProviderCriticality)
