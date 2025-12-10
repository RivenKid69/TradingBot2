# -*- coding: utf-8 -*-
"""
DORA ICT Incident Classification Module (Article 18).

Regulation (EU) 2022/2554 Article 18 defines classification criteria for ICT-related incidents:
    - Classification by impact and severity
    - Criteria for major incidents
    - Thresholds for notification requirements
    - Assessment of client impact, duration, geographic spread

This module implements the classification criteria from:
    - DORA Article 18
    - Commission Delegated Regulation (CDR) 2024/1772 - RTS on incident classification
    - CDR 2025/301 - RTS on incident reporting content and time limits

Integration Layer Context:
    As part of the DORA Integration Layer, this module helps both:
    1. ICT providers: Classify incidents to determine notification urgency to clients
    2. Financial entities: Classify incidents for NCA reporting decisions

    The classification result determines client notification SLAs and
    helps clients understand if NCA reporting is required.

References:
    - Article 18 DORA: https://www.digital-operational-resilience-act.com/Article_18.html
    - CDR 2024/1772: RTS on classification criteria for major ICT-related incidents
    - CDR 2025/301: RTS on content and time limits for incident reports
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
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class IncidentClassificationType(Enum):
    """Incident classification types per DORA Article 18."""
    MAJOR = "major"
    SIGNIFICANT = "significant"
    MINOR = "minor"


class ClientType(Enum):
    """Client types for impact assessment."""
    RETAIL = "retail"
    PROFESSIONAL = "professional"
    ELIGIBLE_COUNTERPARTY = "eligible_counterparty"
    INSTITUTIONAL = "institutional"
    OTHER_FINANCIAL_ENTITY = "other_financial_entity"


class DataType(Enum):
    """Data types for breach assessment."""
    PERSONAL_DATA = "personal_data"
    FINANCIAL_DATA = "financial_data"
    CONFIDENTIAL_BUSINESS = "confidential_business"
    TRADING_DATA = "trading_data"
    AUTHENTICATION_CREDENTIALS = "authentication_credentials"
    OTHER_SENSITIVE = "other_sensitive"


class CriticalServiceType(Enum):
    """Critical service types per DORA Article 3(22)."""
    ORDER_EXECUTION = "order_execution"
    MARKET_DATA = "market_data"
    RISK_MONITORING = "risk_monitoring"
    SETTLEMENT = "settlement"
    CLEARING = "clearing"
    CUSTODY = "custody"
    PAYMENT_PROCESSING = "payment_processing"
    REGULATORY_REPORTING = "regulatory_reporting"
    CLIENT_PORTAL = "client_portal"
    TRADING_INFRASTRUCTURE = "trading_infrastructure"


class MajorIncidentTrigger(Enum):
    """Triggers for major incident classification per CDR 2024/1772."""
    CRITICAL_SERVICE_BREACH = "critical_service_breach"
    MULTIPLE_CRITERIA_THRESHOLD = "multiple_criteria_threshold"
    RECURRING_INCIDENTS = "recurring_incidents"
    DATA_BREACH = "data_breach"
    CLIENT_IMPACT_THRESHOLD = "client_impact_threshold"
    ECONOMIC_IMPACT_THRESHOLD = "economic_impact_threshold"
    DURATION_THRESHOLD = "duration_threshold"
    GEOGRAPHIC_SPREAD_THRESHOLD = "geographic_spread_threshold"


class ReputationalImpactLevel(Enum):
    """Reputational impact levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    SEVERE = "severe"


# =============================================================================
# Data Structures - Classification Criteria
# =============================================================================

@dataclass
class ClassificationThresholds:
    """
    Classification thresholds per CDR 2024/1772.

    These are the thresholds used to determine if an incident is major.
    """
    # Client impact thresholds (Article 18(1)(a))
    retail_client_count: int = 5000
    professional_client_count: int = 100
    counterparty_count: int = 10

    # Duration threshold (Article 18(1)(b))
    duration_hours: float = 4.0

    # Geographic spread (Article 18(1)(c))
    countries_affected: int = 2

    # Economic impact (Article 18(1)(g))
    economic_impact_eur: float = 100_000.0

    # Recurring incidents threshold (Article 18(2))
    recurring_count: int = 3
    recurring_period_months: int = 3

    # Data breach - any data loss is material
    data_breach_material: bool = True

    # Critical service + malicious access = automatic major
    critical_service_malicious: bool = True

    # Number of criteria needed for major (if not auto-trigger)
    criteria_threshold_for_major: int = 2


@dataclass
class ClientImpactAssessment:
    """
    Client impact assessment per Article 18(1)(a).
    """
    # Counts by client type
    retail_clients_affected: int = 0
    professional_clients_affected: int = 0
    counterparties_affected: int = 0
    institutional_clients_affected: int = 0
    other_financial_entities_affected: int = 0

    # Total unique clients
    total_clients_affected: int = 0

    # Relevance/importance of affected clients
    includes_systemically_important: bool = False
    affects_client_critical_operations: bool = False

    # Impact on client services
    client_services_unavailable: List[str] = field(default_factory=list)
    client_data_compromised: bool = False

    @property
    def exceeds_threshold(self) -> bool:
        """Check if any client threshold is exceeded."""
        thresholds = ClassificationThresholds()
        return (
            self.retail_clients_affected >= thresholds.retail_client_count or
            self.professional_clients_affected >= thresholds.professional_client_count or
            self.counterparties_affected >= thresholds.counterparty_count
        )


@dataclass
class DurationAssessment:
    """
    Duration assessment per Article 18(1)(b).
    """
    incident_start: str = ""
    incident_end: str = ""
    service_disruption_start: str = ""
    service_disruption_end: str = ""

    # Duration in hours
    total_duration_hours: float = 0.0
    service_unavailability_hours: float = 0.0
    degraded_service_hours: float = 0.0

    # Is incident ongoing
    is_ongoing: bool = False

    @property
    def exceeds_threshold(self) -> bool:
        """Check if duration threshold is exceeded."""
        thresholds = ClassificationThresholds()
        return (
            self.total_duration_hours >= thresholds.duration_hours or
            self.service_unavailability_hours >= thresholds.duration_hours
        )


@dataclass
class GeographicAssessment:
    """
    Geographic spread assessment per Article 18(1)(c).
    """
    # Affected EU member states (ISO 3166-1 alpha-2)
    member_states_affected: List[str] = field(default_factory=list)

    # Affected non-EU countries
    third_countries_affected: List[str] = field(default_factory=list)

    # Total countries
    total_countries_affected: int = 0

    # Cross-border impact
    affects_cross_border_services: bool = False
    affects_eu_wide_infrastructure: bool = False

    @property
    def exceeds_threshold(self) -> bool:
        """Check if geographic threshold is exceeded."""
        thresholds = ClassificationThresholds()
        total = len(self.member_states_affected) + len(self.third_countries_affected)
        return total >= thresholds.countries_affected


@dataclass
class DataLossAssessment:
    """
    Data loss assessment per Article 18(1)(d).
    """
    # Data compromised
    data_compromised: bool = False
    data_types_affected: List[DataType] = field(default_factory=list)

    # Data integrity
    data_integrity_affected: bool = False
    data_confidentiality_affected: bool = False
    data_availability_affected: bool = False

    # Volume
    records_affected: int = 0
    estimated_individuals_affected: int = 0

    # Personal data (GDPR)
    includes_personal_data: bool = False
    includes_special_category_data: bool = False
    gdpr_notification_required: bool = False

    @property
    def is_material(self) -> bool:
        """Check if data loss is material."""
        return (
            self.data_compromised or
            self.data_integrity_affected or
            self.includes_personal_data
        )


@dataclass
class CriticalServiceAssessment:
    """
    Critical service impact assessment per Article 18(1)(e).
    """
    # Affected critical services
    critical_services_affected: List[CriticalServiceType] = field(default_factory=list)

    # Impact level
    services_fully_unavailable: List[str] = field(default_factory=list)
    services_partially_available: List[str] = field(default_factory=list)
    services_degraded: List[str] = field(default_factory=list)

    # Critical function impact (per Article 3(22))
    affects_critical_or_important_functions: bool = False
    critical_function_names: List[str] = field(default_factory=list)

    # Business continuity
    business_continuity_plan_activated: bool = False
    alternative_arrangements_used: bool = False

    @property
    def has_impact(self) -> bool:
        """Check if critical services are affected."""
        return (
            len(self.critical_services_affected) > 0 or
            self.affects_critical_or_important_functions
        )


@dataclass
class EconomicImpactAssessment:
    """
    Economic impact assessment per Article 18(1)(g).
    """
    # Direct costs
    direct_financial_losses_eur: float = 0.0
    remediation_costs_eur: float = 0.0
    recovery_costs_eur: float = 0.0
    regulatory_fines_eur: float = 0.0
    legal_costs_eur: float = 0.0

    # Indirect costs
    lost_revenue_eur: float = 0.0
    compensation_to_clients_eur: float = 0.0
    reputational_damage_estimate_eur: float = 0.0

    # Total
    total_estimated_impact_eur: float = 0.0

    # Assessment status
    is_estimated: bool = True
    assessment_date: str = ""
    assessed_by: str = ""

    def calculate_total(self) -> float:
        """Calculate total economic impact."""
        self.total_estimated_impact_eur = (
            self.direct_financial_losses_eur +
            self.remediation_costs_eur +
            self.recovery_costs_eur +
            self.regulatory_fines_eur +
            self.legal_costs_eur +
            self.lost_revenue_eur +
            self.compensation_to_clients_eur +
            self.reputational_damage_estimate_eur
        )
        return self.total_estimated_impact_eur

    @property
    def exceeds_threshold(self) -> bool:
        """Check if economic threshold is exceeded."""
        thresholds = ClassificationThresholds()
        self.calculate_total()
        return self.total_estimated_impact_eur >= thresholds.economic_impact_eur


@dataclass
class ReputationalAssessment:
    """
    Reputational impact assessment per Article 18(1)(h).
    """
    impact_level: ReputationalImpactLevel = ReputationalImpactLevel.NONE

    # Media/public attention
    media_coverage_expected: bool = False
    media_coverage_level: str = ""  # none, local, national, international
    social_media_impact: bool = False

    # Stakeholder impact
    affects_market_confidence: bool = False
    affects_investor_confidence: bool = False
    affects_regulatory_standing: bool = False

    # Client relationship
    client_complaints_expected: int = 0
    client_attrition_risk: str = ""  # none, low, medium, high

    @property
    def is_significant(self) -> bool:
        """Check if reputational impact is significant."""
        return self.impact_level in (
            ReputationalImpactLevel.HIGH,
            ReputationalImpactLevel.SEVERE
        )


@dataclass
class RecurringIncidentAssessment:
    """
    Recurring incident assessment per Article 18(2).
    """
    # Related incidents
    related_incident_ids: List[str] = field(default_factory=list)
    incidents_same_root_cause: int = 0

    # Period
    assessment_period_start: str = ""
    assessment_period_end: str = ""

    # Root cause analysis
    same_root_cause: bool = False
    root_cause_description: str = ""
    root_cause_category: str = ""

    # Pattern detection
    is_recurring_pattern: bool = False
    recurrence_frequency: str = ""  # daily, weekly, monthly

    @property
    def exceeds_threshold(self) -> bool:
        """Check if recurring threshold is exceeded."""
        thresholds = ClassificationThresholds()
        return (
            self.same_root_cause and
            self.incidents_same_root_cause >= thresholds.recurring_count
        )


@dataclass
class MaliciousAccessAssessment:
    """
    Malicious/unauthorized access assessment per CDR 2024/1772.
    """
    is_malicious: bool = False
    is_unauthorized_access: bool = False

    # Attack details
    attack_type: str = ""  # ransomware, phishing, ddos, etc.
    attack_vector: str = ""
    threat_actor_type: str = ""  # external, internal, nation_state, etc.

    # Indicators of Compromise
    iocs_identified: List[str] = field(default_factory=list)

    # Access level achieved
    system_access_level: str = ""  # none, read, write, admin, root
    lateral_movement: bool = False
    persistence_established: bool = False
    data_exfiltration: bool = False


# =============================================================================
# Main Classification Result
# =============================================================================

@dataclass
class IncidentClassificationResult:
    """
    Complete incident classification result per Article 18.
    """
    classification_id: str = ""
    incident_id: str = ""

    # Classification result
    classification: IncidentClassificationType = IncidentClassificationType.MINOR
    is_major: bool = False

    # Triggers that led to classification
    major_triggers: List[MajorIncidentTrigger] = field(default_factory=list)
    criteria_met: List[str] = field(default_factory=list)
    criteria_count: int = 0

    # Detailed assessments
    client_impact: ClientImpactAssessment = field(default_factory=ClientImpactAssessment)
    duration: DurationAssessment = field(default_factory=DurationAssessment)
    geographic_spread: GeographicAssessment = field(default_factory=GeographicAssessment)
    data_loss: DataLossAssessment = field(default_factory=DataLossAssessment)
    critical_services: CriticalServiceAssessment = field(default_factory=CriticalServiceAssessment)
    economic_impact: EconomicImpactAssessment = field(default_factory=EconomicImpactAssessment)
    reputational_impact: ReputationalAssessment = field(default_factory=ReputationalAssessment)
    recurring_assessment: RecurringIncidentAssessment = field(default_factory=RecurringIncidentAssessment)
    malicious_access: MaliciousAccessAssessment = field(default_factory=MaliciousAccessAssessment)

    # Notification requirements
    requires_notification: bool = False
    notification_urgency: str = ""  # immediate, standard
    notification_deadline_hours: int = 0

    # Classification metadata
    classified_at: str = ""
    classified_by: str = ""
    classification_rationale: str = ""
    confidence_level: str = ""  # low, medium, high

    # Review
    requires_review: bool = False
    review_reason: str = ""
    reviewed_at: str = ""
    reviewed_by: str = ""

    def __post_init__(self):
        if not self.classification_id:
            self.classification_id = f"CLS-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        if not self.classified_at:
            self.classified_at = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class IncidentClassificationConfig:
    """Configuration for incident classification."""
    # Custom thresholds (override defaults)
    thresholds: Optional[ClassificationThresholds] = None

    # Classification behavior
    auto_classify: bool = True
    require_human_review_for_major: bool = True
    conservative_classification: bool = True  # When in doubt, classify as major

    # Logging
    log_all_classifications: bool = True
    log_path: str = "logs/dora/incident_classification"

    # Entity-specific configuration
    entity_type: str = "investment_firm"
    entity_size: str = "small"  # micro, small, medium, large


# =============================================================================
# Main Classification Engine
# =============================================================================

class DORAIncidentClassification:
    """
    DORA Article 18 compliant incident classification engine.

    Implements the classification criteria from CDR 2024/1772 to determine
    if an incident qualifies as a major ICT-related incident requiring
    notification to competent authorities.

    Integration Layer Purpose:
        This classifier helps ICT providers determine notification
        urgency to clients, and helps clients understand if they
        need to report to NCAs.

    Usage:
        config = IncidentClassificationConfig()
        classifier = DORAIncidentClassification(config)

        # Classify an incident
        result = classifier.classify_incident(
            incident_id="INC-001",
            client_impact=ClientImpactAssessment(
                retail_clients_affected=6000,
            ),
            duration=DurationAssessment(
                total_duration_hours=5.0,
            ),
        )

        if result.is_major:
            print(f"Major incident! Triggers: {result.major_triggers}")
    """

    def __init__(self, config: Optional[IncidentClassificationConfig] = None):
        """Initialize incident classification engine."""
        self.config = config or IncidentClassificationConfig()
        self.thresholds = self.config.thresholds or ClassificationThresholds()

        # Classification history
        self._classifications: Dict[str, IncidentClassificationResult] = {}
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("DORAIncidentClassification initialized")

    def classify_incident(
        self,
        incident_id: str,
        client_impact: Optional[ClientImpactAssessment] = None,
        duration: Optional[DurationAssessment] = None,
        geographic_spread: Optional[GeographicAssessment] = None,
        data_loss: Optional[DataLossAssessment] = None,
        critical_services: Optional[CriticalServiceAssessment] = None,
        economic_impact: Optional[EconomicImpactAssessment] = None,
        reputational_impact: Optional[ReputationalAssessment] = None,
        recurring_assessment: Optional[RecurringIncidentAssessment] = None,
        malicious_access: Optional[MaliciousAccessAssessment] = None,
        classified_by: str = "automated",
    ) -> IncidentClassificationResult:
        """
        Classify an incident per Article 18 criteria.

        The classification algorithm:
        1. Check for automatic major triggers (critical service + malicious, data breach)
        2. Evaluate each criterion threshold
        3. Count criteria met
        4. If 2+ criteria met (or auto-trigger), classify as MAJOR
        5. If 1 criterion met, classify as SIGNIFICANT
        6. Otherwise, classify as MINOR

        Args:
            incident_id: Incident identifier
            client_impact: Client impact assessment
            duration: Duration assessment
            geographic_spread: Geographic spread assessment
            data_loss: Data loss assessment
            critical_services: Critical services assessment
            economic_impact: Economic impact assessment
            reputational_impact: Reputational impact assessment
            recurring_assessment: Recurring incident assessment
            malicious_access: Malicious access assessment
            classified_by: Who is performing classification

        Returns:
            IncidentClassificationResult
        """
        result = IncidentClassificationResult(
            incident_id=incident_id,
            client_impact=client_impact or ClientImpactAssessment(),
            duration=duration or DurationAssessment(),
            geographic_spread=geographic_spread or GeographicAssessment(),
            data_loss=data_loss or DataLossAssessment(),
            critical_services=critical_services or CriticalServiceAssessment(),
            economic_impact=economic_impact or EconomicImpactAssessment(),
            reputational_impact=reputational_impact or ReputationalAssessment(),
            recurring_assessment=recurring_assessment or RecurringIncidentAssessment(),
            malicious_access=malicious_access or MaliciousAccessAssessment(),
            classified_by=classified_by,
        )

        # Step 1: Check automatic major triggers
        auto_major, auto_triggers = self._check_auto_major_triggers(result)

        if auto_major:
            result.is_major = True
            result.classification = IncidentClassificationType.MAJOR
            result.major_triggers = auto_triggers
            result.requires_notification = True
            result.notification_urgency = "immediate"
            result.notification_deadline_hours = 4

        else:
            # Step 2: Evaluate each criterion
            criteria_results = self._evaluate_criteria(result)
            result.criteria_met = [c for c, met in criteria_results.items() if met]
            result.criteria_count = len(result.criteria_met)

            # Step 3: Determine classification
            if result.criteria_count >= self.thresholds.criteria_threshold_for_major:
                result.is_major = True
                result.classification = IncidentClassificationType.MAJOR
                result.major_triggers = [MajorIncidentTrigger.MULTIPLE_CRITERIA_THRESHOLD]
                result.requires_notification = True
                result.notification_urgency = "standard"
                result.notification_deadline_hours = 4

            elif result.criteria_count >= 1:
                result.classification = IncidentClassificationType.SIGNIFICANT
                result.requires_notification = False
                # May still want to track

            else:
                result.classification = IncidentClassificationType.MINOR
                result.requires_notification = False

        # Step 4: Build classification rationale
        result.classification_rationale = self._build_rationale(result)

        # Step 5: Set review requirements
        if result.is_major and self.config.require_human_review_for_major:
            result.requires_review = True
            result.review_reason = "Major incident classification requires human review"

        # Conservative mode
        if self.config.conservative_classification and result.criteria_count == 1:
            result.requires_review = True
            result.review_reason = "Borderline classification - review recommended"

        # Store result
        with self._lock:
            self._classifications[result.classification_id] = result

        # Log classification
        self._log_classification(result)

        logger.info(
            f"Incident {incident_id} classified as {result.classification.value}. "
            f"Major: {result.is_major}, Criteria met: {result.criteria_count}"
        )

        return result

    def _check_auto_major_triggers(
        self,
        result: IncidentClassificationResult,
    ) -> Tuple[bool, List[MajorIncidentTrigger]]:
        """
        Check for automatic major incident triggers.

        Per CDR 2024/1772, certain conditions automatically qualify as major:
        1. Critical service affected + malicious/unauthorized access
        2. Any data breach affecting client data
        3. Recurring incidents (3+ in 3 months, same cause)
        """
        triggers = []

        # Check critical service + malicious access
        if (
            result.critical_services.has_impact and
            (result.malicious_access.is_malicious or result.malicious_access.is_unauthorized_access)
        ):
            triggers.append(MajorIncidentTrigger.CRITICAL_SERVICE_BREACH)

        # Check data breach
        if result.data_loss.is_material:
            if result.data_loss.includes_personal_data:
                triggers.append(MajorIncidentTrigger.DATA_BREACH)
            elif result.data_loss.data_compromised:
                triggers.append(MajorIncidentTrigger.DATA_BREACH)

        # Check recurring incidents
        if result.recurring_assessment.exceeds_threshold:
            triggers.append(MajorIncidentTrigger.RECURRING_INCIDENTS)

        return len(triggers) > 0, triggers

    def _evaluate_criteria(
        self,
        result: IncidentClassificationResult,
    ) -> Dict[str, bool]:
        """
        Evaluate each classification criterion.

        Returns dictionary of criterion name to whether it's met.
        """
        criteria = {}

        # Criterion 1: Client impact (Article 18(1)(a))
        criteria["client_impact"] = result.client_impact.exceeds_threshold

        # Criterion 2: Duration (Article 18(1)(b))
        criteria["duration"] = result.duration.exceeds_threshold

        # Criterion 3: Geographic spread (Article 18(1)(c))
        criteria["geographic_spread"] = result.geographic_spread.exceeds_threshold

        # Criterion 4: Data losses (Article 18(1)(d))
        criteria["data_loss"] = result.data_loss.is_material

        # Criterion 5: Critical services (Article 18(1)(e))
        criteria["critical_services"] = result.critical_services.has_impact

        # Criterion 6: Economic impact (Article 18(1)(g))
        criteria["economic_impact"] = result.economic_impact.exceeds_threshold

        # Criterion 7: Reputational impact (Article 18(1)(h))
        criteria["reputational_impact"] = result.reputational_impact.is_significant

        return criteria

    def _build_rationale(
        self,
        result: IncidentClassificationResult,
    ) -> str:
        """Build human-readable classification rationale."""
        parts = []

        parts.append(f"Classification: {result.classification.value.upper()}")

        if result.is_major:
            parts.append("\nMajor incident determination:")
            if result.major_triggers:
                for trigger in result.major_triggers:
                    parts.append(f"  - Trigger: {trigger.value}")
            parts.append(f"\nCriteria met ({result.criteria_count}):")
            for criterion in result.criteria_met:
                parts.append(f"  - {criterion}")
        else:
            if result.criteria_met:
                parts.append(f"\nCriteria met ({result.criteria_count}):")
                for criterion in result.criteria_met:
                    parts.append(f"  - {criterion}")
            else:
                parts.append("\nNo major criteria thresholds exceeded.")

        if result.requires_notification:
            parts.append(f"\nNotification required: Yes (within {result.notification_deadline_hours}h)")
        else:
            parts.append("\nNotification required: No")

        return "\n".join(parts)

    def reclassify_incident(
        self,
        incident_id: str,
        updates: Dict[str, Any],
        reclassified_by: str = "",
    ) -> IncidentClassificationResult:
        """
        Reclassify an incident with updated information.

        This is important for incidents that may be upgraded as more
        information becomes available (per CDR 2025/301).

        Args:
            incident_id: Incident ID
            updates: Dictionary of assessment updates
            reclassified_by: Who is reclassifying

        Returns:
            Updated IncidentClassificationResult
        """
        # Find existing classification
        existing = None
        with self._lock:
            for cls in self._classifications.values():
                if cls.incident_id == incident_id:
                    existing = cls
                    break

        if not existing:
            raise ValueError(f"No classification found for incident {incident_id}")

        # Build updated assessments
        client_impact = existing.client_impact
        duration = existing.duration
        geographic_spread = existing.geographic_spread
        data_loss = existing.data_loss
        critical_services = existing.critical_services
        economic_impact = existing.economic_impact
        reputational_impact = existing.reputational_impact
        recurring_assessment = existing.recurring_assessment
        malicious_access = existing.malicious_access

        # Apply updates (simplified - in production would be more sophisticated)
        if "client_impact" in updates:
            for key, value in updates["client_impact"].items():
                if hasattr(client_impact, key):
                    setattr(client_impact, key, value)

        if "duration" in updates:
            for key, value in updates["duration"].items():
                if hasattr(duration, key):
                    setattr(duration, key, value)

        if "economic_impact" in updates:
            for key, value in updates["economic_impact"].items():
                if hasattr(economic_impact, key):
                    setattr(economic_impact, key, value)

        # Reclassify
        return self.classify_incident(
            incident_id=incident_id,
            client_impact=client_impact,
            duration=duration,
            geographic_spread=geographic_spread,
            data_loss=data_loss,
            critical_services=critical_services,
            economic_impact=economic_impact,
            reputational_impact=reputational_impact,
            recurring_assessment=recurring_assessment,
            malicious_access=malicious_access,
            classified_by=reclassified_by or "reclassification",
        )

    def approve_classification(
        self,
        classification_id: str,
        approved_by: str,
        notes: str = "",
    ) -> IncidentClassificationResult:
        """
        Approve a classification after human review.

        Args:
            classification_id: Classification ID
            approved_by: Reviewer who approved
            notes: Review notes

        Returns:
            Updated IncidentClassificationResult
        """
        with self._lock:
            if classification_id not in self._classifications:
                raise ValueError(f"Classification {classification_id} not found")

            result = self._classifications[classification_id]
            result.requires_review = False
            result.reviewed_at = datetime.now(timezone.utc).isoformat()
            result.reviewed_by = approved_by

            if notes:
                result.classification_rationale += f"\n\nReview notes: {notes}"

        self._log_classification(result, event_type="classification_approved")
        return result

    def override_classification(
        self,
        classification_id: str,
        new_classification: IncidentClassificationType,
        override_by: str,
        override_reason: str,
    ) -> IncidentClassificationResult:
        """
        Override an automated classification.

        Args:
            classification_id: Classification ID
            new_classification: New classification
            override_by: Who is overriding
            override_reason: Reason for override

        Returns:
            Updated IncidentClassificationResult
        """
        with self._lock:
            if classification_id not in self._classifications:
                raise ValueError(f"Classification {classification_id} not found")

            result = self._classifications[classification_id]
            old_classification = result.classification

            result.classification = new_classification
            result.is_major = new_classification == IncidentClassificationType.MAJOR
            result.requires_notification = result.is_major
            result.requires_review = False
            result.reviewed_at = datetime.now(timezone.utc).isoformat()
            result.reviewed_by = override_by
            result.classification_rationale += (
                f"\n\nMANUAL OVERRIDE by {override_by}:\n"
                f"Changed from {old_classification.value} to {new_classification.value}\n"
                f"Reason: {override_reason}"
            )

        self._log_classification(result, event_type="classification_overridden")
        return result

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_classification(
        self,
        classification_id: str,
    ) -> Optional[IncidentClassificationResult]:
        """Get classification by ID."""
        with self._lock:
            return self._classifications.get(classification_id)

    def get_classification_for_incident(
        self,
        incident_id: str,
    ) -> Optional[IncidentClassificationResult]:
        """Get most recent classification for an incident."""
        with self._lock:
            for cls in reversed(list(self._classifications.values())):
                if cls.incident_id == incident_id:
                    return cls
        return None

    def get_major_classifications(self) -> List[IncidentClassificationResult]:
        """Get all major incident classifications."""
        with self._lock:
            return [c for c in self._classifications.values() if c.is_major]

    def get_pending_reviews(self) -> List[IncidentClassificationResult]:
        """Get classifications pending human review."""
        with self._lock:
            return [c for c in self._classifications.values() if c.requires_review]

    def get_classifications_requiring_notification(
        self,
    ) -> List[IncidentClassificationResult]:
        """Get classifications requiring regulatory notification."""
        with self._lock:
            return [c for c in self._classifications.values() if c.requires_notification]

    # =========================================================================
    # Recurring Incident Detection
    # =========================================================================

    def check_recurring_incidents(
        self,
        incident_id: str,
        root_cause: str,
        root_cause_category: str,
        lookback_months: int = 3,
    ) -> RecurringIncidentAssessment:
        """
        Check if incident is part of a recurring pattern.

        Per Article 18(2), recurring incidents (3+ in 3 months with same cause)
        are automatically classified as major.

        Args:
            incident_id: Current incident ID
            root_cause: Root cause description
            root_cause_category: Root cause category
            lookback_months: Months to look back

        Returns:
            RecurringIncidentAssessment
        """
        assessment = RecurringIncidentAssessment(
            root_cause_description=root_cause,
            root_cause_category=root_cause_category,
        )

        # Set assessment period
        now = datetime.now(timezone.utc)
        period_start = now - timedelta(days=lookback_months * 30)
        assessment.assessment_period_start = period_start.isoformat()
        assessment.assessment_period_end = now.isoformat()

        # In a real implementation, this would query the incident database
        # For now, we check our classification history
        with self._lock:
            related = []
            for cls in self._classifications.values():
                if cls.incident_id == incident_id:
                    continue

                cls_time = datetime.fromisoformat(
                    cls.classified_at.replace("Z", "+00:00")
                )

                if cls_time < period_start:
                    continue

                # Check if same root cause category
                # In production, would do more sophisticated matching
                if cls.recurring_assessment.root_cause_category == root_cause_category:
                    related.append(cls.incident_id)

            assessment.related_incident_ids = related
            assessment.incidents_same_root_cause = len(related) + 1  # Include current

            if assessment.incidents_same_root_cause >= self.thresholds.recurring_count:
                assessment.same_root_cause = True
                assessment.is_recurring_pattern = True

        return assessment

    # =========================================================================
    # Threshold Management
    # =========================================================================

    def update_thresholds(
        self,
        new_thresholds: ClassificationThresholds,
    ) -> None:
        """
        Update classification thresholds.

        May be needed for entity-specific requirements or regulatory updates.

        Args:
            new_thresholds: New threshold configuration
        """
        self.thresholds = new_thresholds
        logger.info("Classification thresholds updated")

    def get_thresholds(self) -> ClassificationThresholds:
        """Get current classification thresholds."""
        return self.thresholds

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_classification_statistics(
        self,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get classification statistics for a period.

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

        start_dt = datetime.fromisoformat(period_start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(period_end.replace("Z", "+00:00"))

        with self._lock:
            period_classifications = [
                c for c in self._classifications.values()
                if start_dt <= datetime.fromisoformat(
                    c.classified_at.replace("Z", "+00:00")
                ) <= end_dt
            ]

        by_type = {
            IncidentClassificationType.MAJOR.value: 0,
            IncidentClassificationType.SIGNIFICANT.value: 0,
            IncidentClassificationType.MINOR.value: 0,
        }

        for c in period_classifications:
            by_type[c.classification.value] += 1

        # Most common triggers
        trigger_counts: Dict[str, int] = {}
        for c in period_classifications:
            for trigger in c.major_triggers:
                trigger_counts[trigger.value] = trigger_counts.get(trigger.value, 0) + 1

        # Most common criteria
        criteria_counts: Dict[str, int] = {}
        for c in period_classifications:
            for criterion in c.criteria_met:
                criteria_counts[criterion] = criteria_counts.get(criterion, 0) + 1

        return {
            "period_start": period_start,
            "period_end": period_end,
            "total_classifications": len(period_classifications),
            "by_type": by_type,
            "major_incidents": by_type[IncidentClassificationType.MAJOR.value],
            "pending_reviews": sum(1 for c in period_classifications if c.requires_review),
            "notifications_required": sum(
                1 for c in period_classifications if c.requires_notification
            ),
            "common_triggers": dict(sorted(
                trigger_counts.items(), key=lambda x: x[1], reverse=True
            )[:5]),
            "common_criteria": dict(sorted(
                criteria_counts.items(), key=lambda x: x[1], reverse=True
            )[:5]),
        }

    def export_classification(
        self,
        classification_id: str,
    ) -> Dict[str, Any]:
        """
        Export a classification for regulatory reporting.

        Args:
            classification_id: Classification ID

        Returns:
            Export data dictionary
        """
        with self._lock:
            if classification_id not in self._classifications:
                raise ValueError(f"Classification {classification_id} not found")

            result = self._classifications[classification_id]

        return {
            "export_date": datetime.now(timezone.utc).isoformat(),
            "article_reference": "Article 18",
            "classification": asdict(result),
            "thresholds_used": asdict(self.thresholds),
        }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_classification(
        self,
        result: IncidentClassificationResult,
        event_type: str = "classification",
    ) -> None:
        """Log a classification event."""
        if not self.config.log_all_classifications:
            return

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "classification_id": result.classification_id,
            "incident_id": result.incident_id,
            "classification": result.classification.value,
            "is_major": result.is_major,
            "criteria_count": result.criteria_count,
            "criteria_met": result.criteria_met,
            "major_triggers": [t.value for t in result.major_triggers],
            "requires_notification": result.requires_notification,
            "classified_by": result.classified_by,
        }

        log_file = self._log_path / f"classification_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log classification: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_incident_classification(
    config: Optional[IncidentClassificationConfig] = None,
) -> DORAIncidentClassification:
    """
    Create a DORAIncidentClassification instance.

    Args:
        config: Optional configuration

    Returns:
        Configured DORAIncidentClassification instance
    """
    return DORAIncidentClassification(config=config)


def get_default_thresholds() -> ClassificationThresholds:
    """
    Get default classification thresholds per CDR 2024/1772.

    Returns:
        Default ClassificationThresholds
    """
    return ClassificationThresholds()


def get_classification_criteria() -> List[str]:
    """
    Get list of classification criteria names.

    Returns:
        List of criterion names
    """
    return [
        "client_impact",
        "duration",
        "geographic_spread",
        "data_loss",
        "critical_services",
        "economic_impact",
        "reputational_impact",
    ]


def create_client_impact_assessment(**kwargs: Any) -> ClientImpactAssessment:
    """Create a ClientImpactAssessment with keyword arguments."""
    return ClientImpactAssessment(**kwargs)


def create_duration_assessment(**kwargs: Any) -> DurationAssessment:
    """Create a DurationAssessment with keyword arguments."""
    return DurationAssessment(**kwargs)


def create_economic_impact_assessment(**kwargs: Any) -> EconomicImpactAssessment:
    """Create an EconomicImpactAssessment with keyword arguments."""
    return EconomicImpactAssessment(**kwargs)


def create_data_loss_assessment(**kwargs: Any) -> DataLossAssessment:
    """Create a DataLossAssessment with keyword arguments."""
    return DataLossAssessment(**kwargs)


def create_critical_service_assessment(**kwargs: Any) -> CriticalServiceAssessment:
    """Create a CriticalServiceAssessment with keyword arguments."""
    return CriticalServiceAssessment(**kwargs)
