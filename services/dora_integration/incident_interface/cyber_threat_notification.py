# -*- coding: utf-8 -*-
"""
DORA Cyber Threat Notification Module (Article 19(4)).

Regulation (EU) 2022/2554 Article 19(4) provides for voluntary notification
of significant cyber threats to competent authorities when deemed relevant
to the financial system, service users, or clients.

Key aspects:
    - VOLUNTARY notification (not mandatory)
    - For significant cyber threats
    - When relevant to financial system, users, or clients
    - Allows collective defense through information sharing

Integration Layer Context:
    For ICT Third-Party Providers:
    - We assess and track cyber threats affecting our services
    - We notify CLIENTS about significant threats
    - Clients may choose to voluntarily report to NCAs
    - We provide threat data packages for client's voluntary reporting

This module provides:
    1. CyberThreat assessment and classification
    2. Threat significance evaluation
    3. Voluntary notification workflow
    4. Threat intelligence integration
    5. Client threat notification

References:
    - Article 19(4) DORA: https://www.digital-operational-resilience-act.com/Article_19.html
    - Article 45 DORA: Information sharing arrangements
    - ENISA Threat Landscape reports
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
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class ThreatCategory(Enum):
    """Cyber threat categories based on ENISA taxonomy."""

    MALWARE = "malware"
    RANSOMWARE = "ransomware"
    PHISHING = "phishing"
    DENIAL_OF_SERVICE = "denial_of_service"
    WEB_BASED_ATTACK = "web_based_attack"
    WEB_APPLICATION_ATTACK = "web_application_attack"
    SPAM = "spam"
    BOTNETS = "botnets"
    DATA_BREACH = "data_breach"
    IDENTITY_THEFT = "identity_theft"
    INFORMATION_LEAKAGE = "information_leakage"
    INSIDER_THREAT = "insider_threat"
    PHYSICAL_MANIPULATION = "physical_manipulation"
    CYBER_ESPIONAGE = "cyber_espionage"
    SUPPLY_CHAIN_ATTACK = "supply_chain_attack"
    CRYPTOJACKING = "cryptojacking"
    ZERO_DAY = "zero_day"
    APT = "apt"  # Advanced Persistent Threat
    OTHER = "other"


class ThreatActorType(Enum):
    """Threat actor types."""

    NATION_STATE = "nation_state"
    CYBERCRIMINAL = "cybercriminal"
    HACKTIVIST = "hacktivist"
    INSIDER = "insider"
    COMPETITOR = "competitor"
    SCRIPT_KIDDIE = "script_kiddie"
    TERRORIST = "terrorist"
    UNKNOWN = "unknown"


class ThreatSeverity(Enum):
    """Threat severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatStatus(Enum):
    """Threat status."""

    IDENTIFIED = "identified"
    UNDER_ANALYSIS = "under_analysis"
    CONFIRMED = "confirmed"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


class ThreatSignificance(Enum):
    """Threat significance for notification decision."""

    NOT_SIGNIFICANT = "not_significant"
    POTENTIALLY_SIGNIFICANT = "potentially_significant"
    SIGNIFICANT = "significant"
    HIGHLY_SIGNIFICANT = "highly_significant"


class NotificationStatus(Enum):
    """Notification status."""

    NOT_NOTIFIED = "not_notified"
    PENDING_DECISION = "pending_decision"
    APPROVED_FOR_NOTIFICATION = "approved_for_notification"
    NOTIFIED = "notified"
    NOTIFICATION_DECLINED = "notification_declined"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class ThreatIndicator:
    """
    Indicator of Compromise (IoC) or threat indicator.
    """

    indicator_id: str = ""
    indicator_type: str = ""  # ip, domain, hash, url, email, etc.
    indicator_value: str = ""
    description: str = ""

    # Context
    first_seen: str = ""
    last_seen: str = ""
    confidence: str = "medium"  # low, medium, high
    source: str = ""

    # Classification
    threat_types: List[ThreatCategory] = field(default_factory=list)
    malware_families: List[str] = field(default_factory=list)

    # Status
    is_active: bool = True
    verified: bool = False
    false_positive: bool = False

    def __post_init__(self):
        if not self.indicator_id:
            self.indicator_id = f"IOC-{uuid.uuid4().hex[:8].upper()}"
        if not self.first_seen:
            self.first_seen = datetime.now(timezone.utc).isoformat()


@dataclass
class CyberThreat:
    """
    Cyber threat record per Article 19(4).
    """

    threat_id: str = ""
    title: str = ""
    description: str = ""

    # Classification
    category: ThreatCategory = ThreatCategory.OTHER
    severity: ThreatSeverity = ThreatSeverity.MEDIUM
    status: ThreatStatus = ThreatStatus.IDENTIFIED

    # Actor information
    threat_actor_type: ThreatActorType = ThreatActorType.UNKNOWN
    threat_actor_name: str = ""
    threat_actor_aliases: List[str] = field(default_factory=list)
    attribution_confidence: str = ""  # low, medium, high

    # Technical details
    attack_vector: str = ""
    attack_pattern: str = ""  # MITRE ATT&CK reference
    indicators: List[ThreatIndicator] = field(default_factory=list)
    exploited_vulnerabilities: List[str] = field(default_factory=list)  # CVE IDs

    # Target information
    targeted_sectors: List[str] = field(default_factory=list)
    targeted_technologies: List[str] = field(default_factory=list)
    targeted_regions: List[str] = field(default_factory=list)

    # Impact assessment
    potential_impact: str = ""
    affected_ict_services: List[str] = field(default_factory=list)
    affects_critical_functions: bool = False

    # Timing
    detected_at: str = ""
    first_observed: str = ""
    last_observed: str = ""

    # Source
    source: str = ""  # internal, isac, vendor, cert, etc.
    source_reference: str = ""
    tlp_marking: str = "TLP:AMBER"  # TLP:CLEAR, TLP:GREEN, TLP:AMBER, TLP:RED

    # Significance assessment
    significance: ThreatSignificance = ThreatSignificance.NOT_SIGNIFICANT
    significance_rationale: str = ""
    significance_assessed_at: str = ""
    significance_assessed_by: str = ""

    # Notification decision
    notification_status: NotificationStatus = NotificationStatus.NOT_NOTIFIED
    notification_decision_at: str = ""
    notification_decision_by: str = ""
    notification_id: str = ""

    # Mitigation
    mitigation_actions: List[str] = field(default_factory=list)
    mitigation_status: str = ""
    containment_achieved: bool = False

    # Related items
    related_incidents: List[str] = field(default_factory=list)
    related_threats: List[str] = field(default_factory=list)

    # Metadata
    created_at: str = ""
    updated_at: str = ""
    created_by: str = ""
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.threat_id:
            self.threat_id = (
                f"THR-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
            )
        if not self.detected_at:
            self.detected_at = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class ThreatSignificanceAssessment:
    """
    Significance assessment for notification decision.
    """

    assessment_id: str = ""
    threat_id: str = ""

    # Assessment criteria per Article 19(4)
    impacts_critical_functions: bool = False
    impacts_other_financial_entities: bool = False
    impacts_third_parties: bool = False
    impacts_clients: bool = False
    impacts_financial_system_stability: bool = False

    # Scale of potential impact
    potential_affected_entities: int = 0
    potential_affected_clients: int = 0
    geographic_scope: List[str] = field(default_factory=list)

    # Likelihood
    exploitation_likelihood: str = ""  # low, medium, high, very_high
    attack_sophistication: str = ""  # low, medium, high
    active_exploitation_observed: bool = False

    # Mitigation availability
    mitigations_available: bool = False
    patches_available: bool = False
    workarounds_available: bool = False

    # Overall significance
    significance_score: int = 0  # 0-100
    significance: ThreatSignificance = ThreatSignificance.NOT_SIGNIFICANT
    rationale: str = ""

    # Recommendation
    recommend_notification: bool = False
    notification_urgency: str = ""  # standard, urgent
    notification_scope: str = ""  # nca_only, sector_wide, eu_wide

    # Assessment metadata
    assessed_at: str = ""
    assessed_by: str = ""
    approved_by: str = ""
    approved_at: str = ""

    def __post_init__(self):
        if not self.assessment_id:
            self.assessment_id = f"TSA-{uuid.uuid4().hex[:8].upper()}"
        if not self.assessed_at:
            self.assessed_at = datetime.now(timezone.utc).isoformat()


@dataclass
class ThreatNotification:
    """
    Voluntary cyber threat notification per Article 19(4).
    """

    notification_id: str = ""
    threat_id: str = ""

    # Notification content
    title: str = ""
    summary: str = ""
    detailed_description: str = ""

    # Threat information
    threat_category: ThreatCategory = ThreatCategory.OTHER
    threat_severity: ThreatSeverity = ThreatSeverity.MEDIUM
    threat_actor_type: ThreatActorType = ThreatActorType.UNKNOWN

    # Indicators shared
    indicators_shared: List[Dict[str, Any]] = field(default_factory=list)
    tlp_marking: str = "TLP:AMBER"

    # Impact assessment
    affected_services: List[str] = field(default_factory=list)
    potential_impact: str = ""
    geographic_scope: List[str] = field(default_factory=list)

    # Mitigations
    mitigations_applied: List[str] = field(default_factory=list)
    recommended_mitigations: List[str] = field(default_factory=list)

    # Reporting entity
    reporting_entity_lei: str = ""
    reporting_entity_name: str = ""
    contact_person_name: str = ""
    contact_person_email: str = ""
    contact_person_phone: str = ""

    # Submission
    submitted_to: List[str] = field(default_factory=list)  # Authority IDs
    submitted_at: str = ""
    submitted_by: str = ""
    submission_method: str = ""

    # Acknowledgement
    acknowledgement_received: bool = False
    acknowledgement_reference: str = ""
    acknowledgement_datetime: str = ""

    # Status
    status: NotificationStatus = NotificationStatus.PENDING_DECISION

    # Metadata
    created_at: str = ""

    def __post_init__(self):
        if not self.notification_id:
            self.notification_id = (
                f"TNOTIF-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
            )
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CyberThreatNotificationConfig:
    """Configuration for cyber threat notification."""

    # Entity information
    entity_lei: str = ""
    entity_name: str = ""

    # Notification settings
    auto_assess_significance: bool = True
    require_approval_for_notification: bool = True
    approval_roles: List[str] = field(default_factory=list)

    # Significance thresholds
    high_significance_score_threshold: int = 70
    significance_score_for_recommendation: int = 50

    # Primary NCA for notifications
    primary_nca_id: str = ""
    primary_nca_name: str = ""

    # Logging
    log_all_threats: bool = True
    log_path: str = "logs/dora/cyber_threats"

    # Callbacks
    notification_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# Main Class Implementation
# =============================================================================


class CyberThreatNotificationService:
    """
    DORA Article 19(4) Cyber Threat Notification Service.

    Provides:
    - Cyber threat recording and tracking
    - Significance assessment
    - Voluntary notification workflow
    - Indicator management

    Integration Layer Purpose:
        For ICT providers, this service tracks threats affecting our
        services and helps prepare data for client notifications.
        Clients may then choose to voluntarily notify their NCAs.

    Usage:
        config = CyberThreatNotificationConfig(
            entity_lei="549300EXAMPLE0000",
            entity_name="Example Investment Firm",
        )
        service = CyberThreatNotificationService(config)

        # Record a threat
        threat = service.record_threat(
            title="New Ransomware Campaign",
            description="Ransomware targeting financial sector",
            category=ThreatCategory.RANSOMWARE,
            severity=ThreatSeverity.HIGH,
        )

        # Assess significance
        assessment = service.assess_significance(
            threat_id=threat.threat_id,
            impacts_critical_functions=True,
            impacts_other_financial_entities=True,
        )

        # If significant, notify
        if assessment.recommend_notification:
            notification = service.create_notification(threat.threat_id)
            service.submit_notification(notification.notification_id)
    """

    def __init__(self, config: Optional[CyberThreatNotificationConfig] = None):
        """Initialize cyber threat notification service."""
        self.config = config or CyberThreatNotificationConfig()

        # Data stores
        self._threats: Dict[str, CyberThreat] = {}
        self._indicators: Dict[str, ThreatIndicator] = {}
        self._assessments: Dict[str, ThreatSignificanceAssessment] = {}
        self._notifications: Dict[str, ThreatNotification] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("CyberThreatNotificationService initialized")

    # =========================================================================
    # Threat Management
    # =========================================================================

    def record_threat(
        self,
        title: str,
        description: str,
        category: ThreatCategory = ThreatCategory.OTHER,
        severity: ThreatSeverity = ThreatSeverity.MEDIUM,
        threat_actor_type: ThreatActorType = ThreatActorType.UNKNOWN,
        threat_actor_name: str = "",
        attack_vector: str = "",
        indicators: Optional[List[ThreatIndicator]] = None,
        exploited_vulnerabilities: Optional[List[str]] = None,
        targeted_sectors: Optional[List[str]] = None,
        targeted_technologies: Optional[List[str]] = None,
        potential_impact: str = "",
        source: str = "",
        tlp_marking: str = "TLP:AMBER",
        created_by: str = "",
        tags: Optional[List[str]] = None,
    ) -> CyberThreat:
        """
        Record a new cyber threat.

        Args:
            title: Threat title
            description: Threat description
            category: Threat category
            severity: Threat severity
            threat_actor_type: Type of threat actor
            threat_actor_name: Known threat actor name
            attack_vector: Attack vector
            indicators: List of threat indicators
            exploited_vulnerabilities: CVE IDs
            targeted_sectors: Targeted sectors
            targeted_technologies: Targeted technologies
            potential_impact: Potential impact description
            source: Threat source
            tlp_marking: TLP marking
            created_by: Creator
            tags: Tags

        Returns:
            Created CyberThreat
        """
        threat = CyberThreat(
            title=title,
            description=description,
            category=category,
            severity=severity,
            threat_actor_type=threat_actor_type,
            threat_actor_name=threat_actor_name,
            attack_vector=attack_vector,
            indicators=indicators or [],
            exploited_vulnerabilities=exploited_vulnerabilities or [],
            targeted_sectors=targeted_sectors or [],
            targeted_technologies=targeted_technologies or [],
            potential_impact=potential_impact,
            source=source,
            tlp_marking=tlp_marking,
            created_by=created_by,
            tags=tags or [],
        )

        with self._lock:
            self._threats[threat.threat_id] = threat

            # Store indicators separately for lookup
            for indicator in threat.indicators:
                self._indicators[indicator.indicator_id] = indicator

        self._log_event(
            "threat_recorded",
            {
                "threat_id": threat.threat_id,
                "title": title,
                "category": category.value,
                "severity": severity.value,
            },
        )

        logger.info(f"Cyber threat recorded: {threat.threat_id} - {title}")

        # Auto-assess if configured
        if self.config.auto_assess_significance:
            self._auto_assess_significance(threat)

        return threat

    def update_threat(
        self,
        threat_id: str,
        **updates: Any,
    ) -> CyberThreat:
        """
        Update a threat record.

        Args:
            threat_id: Threat ID
            **updates: Fields to update

        Returns:
            Updated CyberThreat
        """
        with self._lock:
            if threat_id not in self._threats:
                raise ValueError(f"Threat {threat_id} not found")

            threat = self._threats[threat_id]

            for key, value in updates.items():
                if hasattr(threat, key):
                    setattr(threat, key, value)

            threat.updated_at = datetime.now(timezone.utc).isoformat()

        return threat

    def add_indicator(
        self,
        threat_id: str,
        indicator: ThreatIndicator,
    ) -> CyberThreat:
        """
        Add an indicator to a threat.

        Args:
            threat_id: Threat ID
            indicator: Indicator to add

        Returns:
            Updated CyberThreat
        """
        with self._lock:
            if threat_id not in self._threats:
                raise ValueError(f"Threat {threat_id} not found")

            threat = self._threats[threat_id]
            threat.indicators.append(indicator)
            self._indicators[indicator.indicator_id] = indicator
            threat.updated_at = datetime.now(timezone.utc).isoformat()

        return threat

    def get_threat(self, threat_id: str) -> Optional[CyberThreat]:
        """Get threat by ID."""
        with self._lock:
            return self._threats.get(threat_id)

    def get_active_threats(self) -> List[CyberThreat]:
        """Get all active threats."""
        with self._lock:
            return [
                t
                for t in self._threats.values()
                if t.status not in (ThreatStatus.RESOLVED, ThreatStatus.FALSE_POSITIVE)
            ]

    def get_threats_by_severity(
        self,
        severity: ThreatSeverity,
    ) -> List[CyberThreat]:
        """Get threats by severity."""
        with self._lock:
            return [t for t in self._threats.values() if t.severity == severity]

    def get_threats_by_category(
        self,
        category: ThreatCategory,
    ) -> List[CyberThreat]:
        """Get threats by category."""
        with self._lock:
            return [t for t in self._threats.values() if t.category == category]

    # =========================================================================
    # Significance Assessment
    # =========================================================================

    def assess_significance(
        self,
        threat_id: str,
        impacts_critical_functions: bool = False,
        impacts_other_financial_entities: bool = False,
        impacts_third_parties: bool = False,
        impacts_clients: bool = False,
        impacts_financial_system_stability: bool = False,
        potential_affected_entities: int = 0,
        potential_affected_clients: int = 0,
        geographic_scope: Optional[List[str]] = None,
        exploitation_likelihood: str = "medium",
        active_exploitation_observed: bool = False,
        mitigations_available: bool = False,
        assessed_by: str = "",
    ) -> ThreatSignificanceAssessment:
        """
        Assess threat significance for notification decision.

        Per Article 19(4), significance is based on relevance to:
        - Financial system
        - Service users
        - Clients

        Args:
            threat_id: Threat ID
            impacts_critical_functions: Affects critical/important functions
            impacts_other_financial_entities: Affects other FIs
            impacts_third_parties: Affects third parties
            impacts_clients: Affects clients
            impacts_financial_system_stability: Systemic risk
            potential_affected_entities: Number of entities
            potential_affected_clients: Number of clients
            geographic_scope: Affected regions
            exploitation_likelihood: Likelihood of exploitation
            active_exploitation_observed: Active exploitation seen
            mitigations_available: Mitigations available
            assessed_by: Assessor

        Returns:
            ThreatSignificanceAssessment
        """
        with self._lock:
            if threat_id not in self._threats:
                raise ValueError(f"Threat {threat_id} not found")

            threat = self._threats[threat_id]

        assessment = ThreatSignificanceAssessment(
            threat_id=threat_id,
            impacts_critical_functions=impacts_critical_functions,
            impacts_other_financial_entities=impacts_other_financial_entities,
            impacts_third_parties=impacts_third_parties,
            impacts_clients=impacts_clients,
            impacts_financial_system_stability=impacts_financial_system_stability,
            potential_affected_entities=potential_affected_entities,
            potential_affected_clients=potential_affected_clients,
            geographic_scope=geographic_scope or [],
            exploitation_likelihood=exploitation_likelihood,
            active_exploitation_observed=active_exploitation_observed,
            mitigations_available=mitigations_available,
            assessed_by=assessed_by,
        )

        # Calculate significance score
        assessment.significance_score = self._calculate_significance_score(assessment, threat)

        # Determine significance level
        if assessment.significance_score >= 80:
            assessment.significance = ThreatSignificance.HIGHLY_SIGNIFICANT
            assessment.recommend_notification = True
            assessment.notification_urgency = "urgent"
        elif assessment.significance_score >= 60:
            assessment.significance = ThreatSignificance.SIGNIFICANT
            assessment.recommend_notification = True
            assessment.notification_urgency = "standard"
        elif assessment.significance_score >= 40:
            assessment.significance = ThreatSignificance.POTENTIALLY_SIGNIFICANT
            assessment.recommend_notification = False
        else:
            assessment.significance = ThreatSignificance.NOT_SIGNIFICANT
            assessment.recommend_notification = False

        assessment.rationale = self._build_significance_rationale(assessment, threat)

        with self._lock:
            self._assessments[assessment.assessment_id] = assessment
            threat.significance = assessment.significance
            threat.significance_rationale = assessment.rationale
            threat.significance_assessed_at = assessment.assessed_at
            threat.significance_assessed_by = assessed_by

        self._log_event(
            "significance_assessed",
            {
                "threat_id": threat_id,
                "assessment_id": assessment.assessment_id,
                "significance": assessment.significance.value,
                "recommend_notification": assessment.recommend_notification,
            },
        )

        return assessment

    def _calculate_significance_score(
        self,
        assessment: ThreatSignificanceAssessment,
        threat: CyberThreat,
    ) -> int:
        """Calculate significance score (0-100)."""
        score = 0

        # Impact factors (up to 50 points)
        if assessment.impacts_financial_system_stability:
            score += 20
        if assessment.impacts_critical_functions:
            score += 15
        if assessment.impacts_other_financial_entities:
            score += 10
        if assessment.impacts_clients:
            score += 5

        # Scale factors (up to 20 points)
        if assessment.potential_affected_entities > 100:
            score += 10
        elif assessment.potential_affected_entities > 10:
            score += 5

        if len(assessment.geographic_scope) > 5:
            score += 10
        elif len(assessment.geographic_scope) > 1:
            score += 5

        # Likelihood factors (up to 20 points)
        likelihood_scores = {"very_high": 15, "high": 10, "medium": 5, "low": 2}
        score += likelihood_scores.get(assessment.exploitation_likelihood, 5)

        if assessment.active_exploitation_observed:
            score += 5

        # Severity factor (up to 10 points)
        severity_scores = {
            ThreatSeverity.CRITICAL: 10,
            ThreatSeverity.HIGH: 7,
            ThreatSeverity.MEDIUM: 4,
            ThreatSeverity.LOW: 1,
        }
        score += severity_scores.get(threat.severity, 4)

        return min(100, score)

    def _build_significance_rationale(
        self,
        assessment: ThreatSignificanceAssessment,
        threat: CyberThreat,
    ) -> str:
        """Build human-readable significance rationale."""
        parts = [
            f"Significance Assessment for threat: {threat.title}",
            f"Score: {assessment.significance_score}/100",
            f"Level: {assessment.significance.value}",
            "",
            "Impact factors:",
        ]

        if assessment.impacts_financial_system_stability:
            parts.append("  - Potential impact on financial system stability")
        if assessment.impacts_critical_functions:
            parts.append("  - Affects critical/important functions")
        if assessment.impacts_other_financial_entities:
            parts.append("  - May affect other financial entities")
        if assessment.impacts_clients:
            parts.append("  - Affects clients")

        parts.append("")
        parts.append(f"Exploitation likelihood: {assessment.exploitation_likelihood}")
        if assessment.active_exploitation_observed:
            parts.append("  - Active exploitation observed!")

        parts.append("")
        if assessment.recommend_notification:
            parts.append(f"Recommendation: NOTIFY ({assessment.notification_urgency})")
        else:
            parts.append("Recommendation: No notification required")

        return "\n".join(parts)

    def _auto_assess_significance(self, threat: CyberThreat) -> None:
        """Perform automatic significance assessment based on threat attributes."""
        impacts_critical = threat.affects_critical_functions
        high_severity = threat.severity in (ThreatSeverity.HIGH, ThreatSeverity.CRITICAL)

        # Only auto-assess if likely significant
        if impacts_critical or high_severity:
            self.assess_significance(
                threat_id=threat.threat_id,
                impacts_critical_functions=impacts_critical,
                assessed_by="auto_assessment",
            )

    def get_assessment(
        self,
        assessment_id: str,
    ) -> Optional[ThreatSignificanceAssessment]:
        """Get assessment by ID."""
        with self._lock:
            return self._assessments.get(assessment_id)

    def get_assessment_for_threat(
        self,
        threat_id: str,
    ) -> Optional[ThreatSignificanceAssessment]:
        """Get most recent assessment for a threat."""
        with self._lock:
            for assessment in reversed(list(self._assessments.values())):
                if assessment.threat_id == threat_id:
                    return assessment
        return None

    # =========================================================================
    # Notification Workflow
    # =========================================================================

    def create_notification(
        self,
        threat_id: str,
        summary: str = "",
        detailed_description: str = "",
        mitigations_applied: Optional[List[str]] = None,
        recommended_mitigations: Optional[List[str]] = None,
        contact_person_name: str = "",
        contact_person_email: str = "",
        contact_person_phone: str = "",
        include_indicators: bool = True,
    ) -> ThreatNotification:
        """
        Create a voluntary threat notification.

        Args:
            threat_id: Threat ID to notify about
            summary: Summary for notification
            detailed_description: Detailed description
            mitigations_applied: Mitigations applied
            recommended_mitigations: Recommended mitigations
            contact_person_name: Contact name
            contact_person_email: Contact email
            contact_person_phone: Contact phone
            include_indicators: Whether to include IoCs

        Returns:
            ThreatNotification
        """
        with self._lock:
            if threat_id not in self._threats:
                raise ValueError(f"Threat {threat_id} not found")

            threat = self._threats[threat_id]

        # Prepare indicators for sharing
        indicators_shared = []
        if include_indicators:
            for ind in threat.indicators:
                if not ind.false_positive:
                    indicators_shared.append(
                        {
                            "type": ind.indicator_type,
                            "value": ind.indicator_value,
                            "description": ind.description,
                            "confidence": ind.confidence,
                        }
                    )

        notification = ThreatNotification(
            threat_id=threat_id,
            title=threat.title,
            summary=summary or threat.description[:500],
            detailed_description=detailed_description or threat.description,
            threat_category=threat.category,
            threat_severity=threat.severity,
            threat_actor_type=threat.threat_actor_type,
            indicators_shared=indicators_shared,
            tlp_marking=threat.tlp_marking,
            affected_services=threat.affected_ict_services,
            potential_impact=threat.potential_impact,
            geographic_scope=threat.targeted_regions,
            mitigations_applied=mitigations_applied or threat.mitigation_actions,
            recommended_mitigations=recommended_mitigations or [],
            reporting_entity_lei=self.config.entity_lei,
            reporting_entity_name=self.config.entity_name,
            contact_person_name=contact_person_name,
            contact_person_email=contact_person_email,
            contact_person_phone=contact_person_phone,
        )

        with self._lock:
            self._notifications[notification.notification_id] = notification
            threat.notification_status = NotificationStatus.PENDING_DECISION

        self._log_event(
            "notification_created",
            {
                "notification_id": notification.notification_id,
                "threat_id": threat_id,
            },
        )

        return notification

    def approve_notification(
        self,
        notification_id: str,
        approved_by: str,
    ) -> ThreatNotification:
        """
        Approve a notification for submission.

        Args:
            notification_id: Notification ID
            approved_by: Approver

        Returns:
            Updated ThreatNotification
        """
        with self._lock:
            if notification_id not in self._notifications:
                raise ValueError(f"Notification {notification_id} not found")

            notification = self._notifications[notification_id]
            notification.status = NotificationStatus.APPROVED_FOR_NOTIFICATION

            # Update threat
            if notification.threat_id in self._threats:
                threat = self._threats[notification.threat_id]
                threat.notification_status = NotificationStatus.APPROVED_FOR_NOTIFICATION
                threat.notification_decision_at = datetime.now(timezone.utc).isoformat()
                threat.notification_decision_by = approved_by

        self._log_event(
            "notification_approved",
            {
                "notification_id": notification_id,
                "approved_by": approved_by,
            },
        )

        return notification

    def decline_notification(
        self,
        notification_id: str,
        declined_by: str,
        reason: str = "",
    ) -> ThreatNotification:
        """
        Decline to submit a notification.

        Args:
            notification_id: Notification ID
            declined_by: Decliner
            reason: Reason for declining

        Returns:
            Updated ThreatNotification
        """
        with self._lock:
            if notification_id not in self._notifications:
                raise ValueError(f"Notification {notification_id} not found")

            notification = self._notifications[notification_id]
            notification.status = NotificationStatus.NOTIFICATION_DECLINED

            # Update threat
            if notification.threat_id in self._threats:
                threat = self._threats[notification.threat_id]
                threat.notification_status = NotificationStatus.NOTIFICATION_DECLINED
                threat.notification_decision_at = datetime.now(timezone.utc).isoformat()
                threat.notification_decision_by = declined_by

        return notification

    def submit_notification(
        self,
        notification_id: str,
        authority_ids: Optional[List[str]] = None,
        submitted_by: str = "",
        submission_method: str = "portal",
    ) -> ThreatNotification:
        """
        Submit notification to competent authority.

        Args:
            notification_id: Notification ID
            authority_ids: Target authority IDs
            submitted_by: Submitter
            submission_method: Submission method

        Returns:
            Updated ThreatNotification
        """
        with self._lock:
            if notification_id not in self._notifications:
                raise ValueError(f"Notification {notification_id} not found")

            notification = self._notifications[notification_id]

            if self.config.require_approval_for_notification:
                if notification.status != NotificationStatus.APPROVED_FOR_NOTIFICATION:
                    raise ValueError("Notification must be approved before submission")

            now = datetime.now(timezone.utc).isoformat()

            notification.submitted_to = authority_ids or [self.config.primary_nca_id]
            notification.submitted_at = now
            notification.submitted_by = submitted_by
            notification.submission_method = submission_method
            notification.status = NotificationStatus.NOTIFIED

            # Update threat
            if notification.threat_id in self._threats:
                threat = self._threats[notification.threat_id]
                threat.notification_status = NotificationStatus.NOTIFIED
                threat.notification_id = notification_id

        # Trigger callback
        if self.config.notification_callback:
            try:
                self.config.notification_callback(
                    "threat_notified",
                    {
                        "notification_id": notification_id,
                        "threat_id": notification.threat_id,
                        "submitted_to": notification.submitted_to,
                    },
                )
            except Exception as e:
                logger.error(f"Notification callback failed: {e}")

        self._log_event(
            "notification_submitted",
            {
                "notification_id": notification_id,
                "submitted_to": notification.submitted_to,
            },
        )

        logger.info(f"Threat notification submitted: {notification_id}")
        return notification

    def get_notification(
        self,
        notification_id: str,
    ) -> Optional[ThreatNotification]:
        """Get notification by ID."""
        with self._lock:
            return self._notifications.get(notification_id)

    def get_pending_notifications(self) -> List[ThreatNotification]:
        """Get notifications pending decision or submission."""
        with self._lock:
            return [
                n
                for n in self._notifications.values()
                if n.status
                in (
                    NotificationStatus.PENDING_DECISION,
                    NotificationStatus.APPROVED_FOR_NOTIFICATION,
                )
            ]

    # =========================================================================
    # Indicator Management
    # =========================================================================

    def create_indicator(
        self,
        indicator_type: str,
        indicator_value: str,
        description: str = "",
        threat_types: Optional[List[ThreatCategory]] = None,
        confidence: str = "medium",
        source: str = "",
    ) -> ThreatIndicator:
        """
        Create a standalone threat indicator.

        Args:
            indicator_type: Type (ip, domain, hash, url, etc.)
            indicator_value: Value
            description: Description
            threat_types: Associated threat types
            confidence: Confidence level
            source: Source of indicator

        Returns:
            Created ThreatIndicator
        """
        indicator = ThreatIndicator(
            indicator_type=indicator_type,
            indicator_value=indicator_value,
            description=description,
            threat_types=threat_types or [],
            confidence=confidence,
            source=source,
        )

        with self._lock:
            self._indicators[indicator.indicator_id] = indicator

        return indicator

    def search_indicators(
        self,
        indicator_type: Optional[str] = None,
        indicator_value: Optional[str] = None,
    ) -> List[ThreatIndicator]:
        """
        Search for indicators.

        Args:
            indicator_type: Filter by type
            indicator_value: Filter by value (partial match)

        Returns:
            List of matching indicators
        """
        with self._lock:
            results = []
            for ind in self._indicators.values():
                if indicator_type and ind.indicator_type != indicator_type:
                    continue
                if indicator_value and indicator_value.lower() not in ind.indicator_value.lower():
                    continue
                results.append(ind)
        return results

    # =========================================================================
    # Statistics and Export
    # =========================================================================

    def get_threat_statistics(
        self,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get threat statistics for a period."""
        if not period_end:
            period_end = datetime.now(timezone.utc).isoformat()
        if not period_start:
            period_start = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

        start_dt = datetime.fromisoformat(period_start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(period_end.replace("Z", "+00:00"))

        with self._lock:
            period_threats = [
                t
                for t in self._threats.values()
                if start_dt <= datetime.fromisoformat(t.created_at.replace("Z", "+00:00")) <= end_dt
            ]

        by_category = {}
        for cat in ThreatCategory:
            count = sum(1 for t in period_threats if t.category == cat)
            if count > 0:
                by_category[cat.value] = count

        by_severity = {}
        for sev in ThreatSeverity:
            by_severity[sev.value] = sum(1 for t in period_threats if t.severity == sev)

        notified = sum(
            1 for t in period_threats if t.notification_status == NotificationStatus.NOTIFIED
        )

        return {
            "period_start": period_start,
            "period_end": period_end,
            "total_threats": len(period_threats),
            "by_category": by_category,
            "by_severity": by_severity,
            "significant_threats": sum(
                1
                for t in period_threats
                if t.significance
                in (
                    ThreatSignificance.SIGNIFICANT,
                    ThreatSignificance.HIGHLY_SIGNIFICANT,
                )
            ),
            "notifications_submitted": notified,
            "total_indicators": len(self._indicators),
        }

    def export_threat(self, threat_id: str) -> Dict[str, Any]:
        """Export threat data."""
        with self._lock:
            if threat_id not in self._threats:
                raise ValueError(f"Threat {threat_id} not found")
            return {
                "export_date": datetime.now(timezone.utc).isoformat(),
                "article_reference": "Article 19(4)",
                "threat": asdict(self._threats[threat_id]),
            }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a cyber threat event."""
        if not self.config.log_all_threats:
            return

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"cyber_threats_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log cyber threat event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================


def create_cyber_threat_notification_service(
    config: Optional[CyberThreatNotificationConfig] = None,
) -> CyberThreatNotificationService:
    """Create a CyberThreatNotificationService instance."""
    return CyberThreatNotificationService(config=config)


def get_threat_categories() -> List[ThreatCategory]:
    """Get list of threat categories."""
    return list(ThreatCategory)


def get_threat_severities() -> List[ThreatSeverity]:
    """Get list of threat severities."""
    return list(ThreatSeverity)
