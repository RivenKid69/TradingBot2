# -*- coding: utf-8 -*-
"""
DORA ICT Identification Module (Article 8).

Regulation (EU) 2022/2554 Article 8 defines identification requirements:
    - Identify, classify and document all ICT assets
    - Identify sources of ICT risk
    - Assess cyber threats and ICT vulnerabilities
    - Identify ICT-supported business functions and dependencies
    - Document information assets and ICT third-party dependencies

This module provides:
    1. ICTAssetInventory - Asset identification and classification
    2. RiskSourceIdentification - ICT risk source identification
    3. ThreatVulnerabilityAssessment - Threat and vulnerability assessment
    4. DependencyMapping - ICT dependencies and interconnections
    5. DORAICTIdentification - Main identification orchestrator

References:
    - Article 8 DORA: https://www.digital-operational-resilience-act.com/Article_8.html
    - RTS on ICT Risk Management: CDR 2024/1774
    - NIST SP 800-30: Risk Assessment Guide
    - ISO 27001:2022 Asset Management
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


class AssetType(Enum):
    """ICT asset types per DORA Article 8."""

    HARDWARE = "hardware"
    SOFTWARE = "software"
    DATA = "data"
    NETWORK = "network"
    PERSONNEL = "personnel"
    FACILITY = "facility"
    SERVICE = "service"
    PROCESS = "process"


class AssetClassification(Enum):
    """Asset classification levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class RiskSourceCategory(Enum):
    """ICT risk source categories."""

    EXTERNAL_THREAT = "external_threat"
    INTERNAL_THREAT = "internal_threat"
    ENVIRONMENTAL = "environmental"
    TECHNICAL = "technical"
    OPERATIONAL = "operational"
    THIRD_PARTY = "third_party"
    REGULATORY = "regulatory"


class ThreatCategory(Enum):
    """Cyber threat categories."""

    MALWARE = "malware"
    RANSOMWARE = "ransomware"
    PHISHING = "phishing"
    DDOS = "ddos"
    APT = "apt"  # Advanced Persistent Threat
    INSIDER_THREAT = "insider_threat"
    SUPPLY_CHAIN = "supply_chain"
    ZERO_DAY = "zero_day"
    SOCIAL_ENGINEERING = "social_engineering"
    DATA_BREACH = "data_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


class VulnerabilitySeverity(Enum):
    """Vulnerability severity levels (CVSS aligned)."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DependencyType(Enum):
    """Types of ICT dependencies."""

    INFRASTRUCTURE = "infrastructure"
    APPLICATION = "application"
    DATA = "data"
    NETWORK = "network"
    THIRD_PARTY = "third_party"
    PERSONNEL = "personnel"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class ICTAsset:
    """
    ICT Asset record per DORA Article 8(1).

    Documents ICT assets including information assets supporting
    business functions.
    """

    asset_id: str = ""
    name: str = ""
    description: str = ""
    asset_type: AssetType = AssetType.SOFTWARE
    classification: AssetClassification = AssetClassification.INTERNAL

    # Location and ownership
    location: str = ""
    owner: str = ""
    custodian: str = ""
    business_unit: str = ""

    # Technical details
    version: str = ""
    vendor: str = ""
    model: str = ""
    serial_number: str = ""
    ip_address: str = ""
    hostname: str = ""

    # Lifecycle
    acquisition_date: Optional[str] = None
    end_of_life_date: Optional[str] = None
    last_updated: str = ""

    # Criticality
    criticality_level: str = ""  # "critical", "high", "medium", "low"
    supports_critical_function: bool = False
    critical_function_ids: List[str] = field(default_factory=list)

    # Risk profile
    inherent_risk_level: str = ""
    control_effectiveness: str = ""
    residual_risk_level: str = ""

    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    depended_by: List[str] = field(default_factory=list)

    # Third-party
    is_third_party: bool = False
    third_party_provider_id: str = ""

    # Status
    is_active: bool = True

    def __post_init__(self):
        if not self.asset_id:
            self.asset_id = f"ASSET-{uuid.uuid4().hex[:8].upper()}"
        if not self.last_updated:
            self.last_updated = datetime.now(timezone.utc).isoformat()


@dataclass
class RiskSource:
    """
    ICT Risk Source per DORA Article 8(2).

    Documents sources of ICT risk that could affect operations.
    """

    source_id: str = ""
    name: str = ""
    description: str = ""
    category: RiskSourceCategory = RiskSourceCategory.EXTERNAL_THREAT

    # Assessment
    likelihood: int = 3  # 1-5
    impact: int = 3  # 1-5
    risk_level: str = ""

    # Affected assets
    affected_asset_ids: List[str] = field(default_factory=list)
    affected_function_ids: List[str] = field(default_factory=list)

    # Trends
    trend: str = ""  # "increasing", "stable", "decreasing"
    emerging: bool = False

    # Mitigation
    existing_controls: List[str] = field(default_factory=list)
    residual_risk_level: str = ""

    # Monitoring
    monitoring_frequency: str = ""
    last_assessed: str = ""
    next_assessment: Optional[str] = None

    # Status
    is_active: bool = True

    def __post_init__(self):
        if not self.source_id:
            self.source_id = f"RSRC-{uuid.uuid4().hex[:8].upper()}"
        if not self.last_assessed:
            self.last_assessed = datetime.now(timezone.utc).isoformat()

    def calculate_risk_score(self) -> int:
        """Calculate risk score."""
        return self.likelihood * self.impact


@dataclass
class CyberThreat:
    """
    Cyber Threat record per DORA Article 8(3).

    Documents cyber threats relevant to the entity.
    """

    threat_id: str = ""
    name: str = ""
    description: str = ""
    category: ThreatCategory = ThreatCategory.MALWARE

    # Threat actor
    threat_actor_type: str = ""  # "nation_state", "criminal", "hacktivist", "insider"
    threat_actor_name: str = ""
    motivation: str = ""

    # Capabilities
    sophistication_level: str = ""  # "low", "medium", "high", "advanced"
    attack_vectors: List[str] = field(default_factory=list)
    ttps: List[str] = field(default_factory=list)  # Tactics, Techniques, Procedures

    # Relevance
    relevance_to_entity: str = ""  # "high", "medium", "low"
    industry_targeting: bool = False
    geographic_targeting: List[str] = field(default_factory=list)

    # Impact
    potential_impact: str = ""
    affected_asset_types: List[AssetType] = field(default_factory=list)

    # Intelligence
    intelligence_source: str = ""
    confidence_level: str = ""  # "confirmed", "high", "medium", "low"
    first_observed: Optional[str] = None
    last_observed: Optional[str] = None

    # Status
    is_active: bool = True

    def __post_init__(self):
        if not self.threat_id:
            self.threat_id = f"THR-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class ICTVulnerability:
    """
    ICT Vulnerability record per DORA Article 8(3).

    Documents vulnerabilities in ICT systems.
    """

    vulnerability_id: str = ""
    name: str = ""
    description: str = ""

    # CVE information
    cve_id: str = ""
    cwe_id: str = ""

    # Severity (CVSS aligned)
    severity: VulnerabilitySeverity = VulnerabilitySeverity.MEDIUM
    cvss_score: float = 0.0
    cvss_vector: str = ""

    # Affected systems
    affected_asset_ids: List[str] = field(default_factory=list)
    affected_systems: List[str] = field(default_factory=list)
    affected_versions: List[str] = field(default_factory=list)

    # Exploitation
    exploitability: str = ""  # "none", "poc", "functional", "high", "widespread"
    exploit_in_wild: bool = False
    exploit_code_maturity: str = ""

    # Status
    status: str = ""  # "open", "in_remediation", "mitigated", "accepted", "closed"
    discovery_date: str = ""
    target_remediation_date: Optional[str] = None
    actual_remediation_date: Optional[str] = None

    # Remediation
    remediation_plan: str = ""
    compensating_controls: List[str] = field(default_factory=list)
    remediation_owner: str = ""

    # Risk acceptance
    risk_accepted: bool = False
    acceptance_justification: str = ""
    accepted_by: str = ""
    acceptance_expiry: Optional[str] = None

    def __post_init__(self):
        if not self.vulnerability_id:
            self.vulnerability_id = f"VUL-{uuid.uuid4().hex[:8].upper()}"
        if not self.discovery_date:
            self.discovery_date = datetime.now(timezone.utc).isoformat()


@dataclass
class ICTDependency:
    """
    ICT Dependency record per DORA Article 8(4).

    Documents dependencies between ICT assets, functions, and providers.
    """

    dependency_id: str = ""
    source_id: str = ""  # The asset/function that depends
    source_type: str = ""  # "asset", "function", "system"
    target_id: str = ""  # The asset/function being depended on
    target_type: str = ""

    # Dependency details
    dependency_type: DependencyType = DependencyType.APPLICATION
    description: str = ""
    criticality: str = ""  # "critical", "high", "medium", "low"

    # Impact analysis
    failure_impact: str = ""
    max_acceptable_downtime_hours: float = 0.0
    alternative_available: bool = False
    alternative_id: str = ""

    # Third-party
    is_third_party: bool = False
    third_party_provider_id: str = ""

    # Mapping
    data_flows: List[str] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.dependency_id:
            self.dependency_id = f"DEP-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class BusinessFunction:
    """
    Business Function supported by ICT.

    Documents the mapping between business functions and ICT assets.
    """

    function_id: str = ""
    name: str = ""
    description: str = ""
    business_unit: str = ""

    # Classification
    is_critical: bool = False
    is_important: bool = False
    criticality_rationale: str = ""

    # Recovery requirements
    rto_hours: float = 4.0
    rpo_hours: float = 1.0
    max_tolerable_downtime_hours: float = 24.0

    # Supporting ICT
    supporting_asset_ids: List[str] = field(default_factory=list)
    supporting_system_ids: List[str] = field(default_factory=list)
    third_party_dependencies: List[str] = field(default_factory=list)

    # Ownership
    business_owner: str = ""
    technical_owner: str = ""

    # Status
    is_active: bool = True
    last_assessed: str = ""

    def __post_init__(self):
        if not self.function_id:
            self.function_id = f"FUNC-{uuid.uuid4().hex[:8].upper()}"
        if not self.last_assessed:
            self.last_assessed = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ICTIdentificationConfig:
    """Configuration for ICT Identification."""

    # Assessment frequencies
    asset_review_frequency_months: int = 12
    risk_source_review_frequency_months: int = 6
    threat_assessment_frequency_months: int = 3
    vulnerability_scan_frequency_days: int = 30

    # Classification defaults
    default_asset_classification: AssetClassification = AssetClassification.INTERNAL
    default_criticality: str = "medium"

    # Vulnerability management
    critical_vuln_remediation_days: int = 14
    high_vuln_remediation_days: int = 30
    medium_vuln_remediation_days: int = 90
    low_vuln_remediation_days: int = 180

    # Dependency mapping
    map_transitive_dependencies: bool = True
    max_dependency_depth: int = 5

    # Logging
    log_all_events: bool = True
    log_path: str = "logs/dora/ict_identification"

    # Callbacks
    alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# Main ICT Identification Class
# =============================================================================


class DORAICTIdentification:
    """
    DORA Article 8 compliant ICT Identification.

    Manages:
    - ICT asset inventory and classification
    - Risk source identification
    - Threat and vulnerability assessment
    - Dependency mapping
    - Business function to ICT mapping

    Usage:
        config = ICTIdentificationConfig()
        identification = DORAICTIdentification(config)

        # Register an asset
        asset = identification.register_asset(
            name="Trading Database",
            asset_type=AssetType.SOFTWARE,
            classification=AssetClassification.RESTRICTED,
        )

        # Identify a risk source
        risk_source = identification.identify_risk_source(
            name="DDoS Attack",
            category=RiskSourceCategory.EXTERNAL_THREAT,
        )

        # Record a vulnerability
        vuln = identification.record_vulnerability(
            name="SQL Injection in Trading API",
            severity=VulnerabilitySeverity.HIGH,
        )
    """

    def __init__(self, config: Optional[ICTIdentificationConfig] = None):
        """Initialize ICT Identification."""
        self.config = config or ICTIdentificationConfig()

        # Data stores
        self._assets: Dict[str, ICTAsset] = {}
        self._risk_sources: Dict[str, RiskSource] = {}
        self._threats: Dict[str, CyberThreat] = {}
        self._vulnerabilities: Dict[str, ICTVulnerability] = {}
        self._dependencies: Dict[str, ICTDependency] = {}
        self._functions: Dict[str, BusinessFunction] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("DORAICTIdentification initialized")

    # =========================================================================
    # Asset Inventory (Article 8(1))
    # =========================================================================

    def register_asset(
        self,
        name: str,
        asset_type: AssetType,
        classification: Optional[AssetClassification] = None,
        description: str = "",
        owner: str = "",
        location: str = "",
        criticality_level: str = "",
        supports_critical_function: bool = False,
        critical_function_ids: Optional[List[str]] = None,
        is_third_party: bool = False,
        third_party_provider_id: str = "",
    ) -> ICTAsset:
        """
        Register an ICT asset.

        Per Article 8(1), entities must identify, classify and document
        all ICT assets including information assets.

        Args:
            name: Asset name
            asset_type: Type of asset
            classification: Classification level
            description: Asset description
            owner: Asset owner
            location: Asset location
            criticality_level: Criticality level
            supports_critical_function: Whether asset supports critical functions
            critical_function_ids: IDs of critical functions supported
            is_third_party: Whether asset is provided by third party
            third_party_provider_id: Third party provider ID

        Returns:
            Registered ICTAsset
        """
        asset = ICTAsset(
            name=name,
            asset_type=asset_type,
            classification=classification or self.config.default_asset_classification,
            description=description,
            owner=owner,
            location=location,
            criticality_level=criticality_level or self.config.default_criticality,
            supports_critical_function=supports_critical_function,
            critical_function_ids=critical_function_ids or [],
            is_third_party=is_third_party,
            third_party_provider_id=third_party_provider_id,
        )

        with self._lock:
            self._assets[asset.asset_id] = asset

        self._log_event(
            "asset_registered",
            {
                "asset_id": asset.asset_id,
                "name": name,
                "type": asset_type.value,
                "classification": asset.classification.value,
            },
        )

        logger.info(f"Asset registered: {name} ({asset_type.value})")
        return asset

    def get_asset(self, asset_id: str) -> Optional[ICTAsset]:
        """Get asset by ID."""
        with self._lock:
            return self._assets.get(asset_id)

    def get_assets_by_type(self, asset_type: AssetType) -> List[ICTAsset]:
        """Get all assets of a specific type."""
        with self._lock:
            return [a for a in self._assets.values() if a.asset_type == asset_type and a.is_active]

    def get_assets_by_classification(
        self,
        classification: AssetClassification,
    ) -> List[ICTAsset]:
        """Get all assets with a specific classification."""
        with self._lock:
            return [
                a
                for a in self._assets.values()
                if a.classification == classification and a.is_active
            ]

    def get_critical_assets(self) -> List[ICTAsset]:
        """Get all critical assets."""
        with self._lock:
            return [
                a
                for a in self._assets.values()
                if (a.criticality_level == "critical" or a.supports_critical_function)
                and a.is_active
            ]

    def get_third_party_assets(self) -> List[ICTAsset]:
        """Get all third-party assets."""
        with self._lock:
            return [a for a in self._assets.values() if a.is_third_party and a.is_active]

    def update_asset(
        self,
        asset_id: str,
        **updates: Any,
    ) -> Optional[ICTAsset]:
        """Update an asset."""
        with self._lock:
            if asset_id not in self._assets:
                return None

            asset = self._assets[asset_id]
            for key, value in updates.items():
                if hasattr(asset, key):
                    setattr(asset, key, value)

            asset.last_updated = datetime.now(timezone.utc).isoformat()

        return asset

    # =========================================================================
    # Risk Source Identification (Article 8(2))
    # =========================================================================

    def identify_risk_source(
        self,
        name: str,
        category: RiskSourceCategory,
        description: str = "",
        likelihood: int = 3,
        impact: int = 3,
        affected_asset_ids: Optional[List[str]] = None,
        affected_function_ids: Optional[List[str]] = None,
        existing_controls: Optional[List[str]] = None,
        trend: str = "stable",
        emerging: bool = False,
    ) -> RiskSource:
        """
        Identify a source of ICT risk.

        Per Article 8(2), entities must identify sources of ICT risk.

        Args:
            name: Risk source name
            category: Risk category
            description: Description
            likelihood: Likelihood score (1-5)
            impact: Impact score (1-5)
            affected_asset_ids: Affected asset IDs
            affected_function_ids: Affected function IDs
            existing_controls: Existing control measures
            trend: Risk trend
            emerging: Whether this is an emerging risk

        Returns:
            Identified RiskSource
        """
        risk_source = RiskSource(
            name=name,
            category=category,
            description=description,
            likelihood=min(5, max(1, likelihood)),
            impact=min(5, max(1, impact)),
            affected_asset_ids=affected_asset_ids or [],
            affected_function_ids=affected_function_ids or [],
            existing_controls=existing_controls or [],
            trend=trend,
            emerging=emerging,
        )

        # Calculate risk level
        score = risk_source.calculate_risk_score()
        if score <= 4:
            risk_source.risk_level = "low"
        elif score <= 9:
            risk_source.risk_level = "medium"
        elif score <= 16:
            risk_source.risk_level = "high"
        else:
            risk_source.risk_level = "critical"

        with self._lock:
            self._risk_sources[risk_source.source_id] = risk_source

        self._log_event(
            "risk_source_identified",
            {
                "source_id": risk_source.source_id,
                "name": name,
                "category": category.value,
                "risk_level": risk_source.risk_level,
            },
        )

        logger.info(f"Risk source identified: {name} ({risk_source.risk_level})")
        return risk_source

    def get_risk_source(self, source_id: str) -> Optional[RiskSource]:
        """Get risk source by ID."""
        with self._lock:
            return self._risk_sources.get(source_id)

    def get_risk_sources_by_category(
        self,
        category: RiskSourceCategory,
    ) -> List[RiskSource]:
        """Get risk sources by category."""
        with self._lock:
            return [
                r for r in self._risk_sources.values() if r.category == category and r.is_active
            ]

    def get_high_risk_sources(self) -> List[RiskSource]:
        """Get high and critical risk sources."""
        with self._lock:
            return [
                r
                for r in self._risk_sources.values()
                if r.risk_level in ("high", "critical") and r.is_active
            ]

    def get_emerging_risks(self) -> List[RiskSource]:
        """Get emerging risk sources."""
        with self._lock:
            return [r for r in self._risk_sources.values() if r.emerging and r.is_active]

    # =========================================================================
    # Threat Assessment (Article 8(3))
    # =========================================================================

    def record_threat(
        self,
        name: str,
        category: ThreatCategory,
        description: str = "",
        threat_actor_type: str = "",
        sophistication_level: str = "medium",
        attack_vectors: Optional[List[str]] = None,
        relevance_to_entity: str = "medium",
        industry_targeting: bool = False,
        potential_impact: str = "",
        intelligence_source: str = "",
        confidence_level: str = "medium",
    ) -> CyberThreat:
        """
        Record a cyber threat.

        Per Article 8(3), entities must assess cyber threats.

        Args:
            name: Threat name
            category: Threat category
            description: Threat description
            threat_actor_type: Type of threat actor
            sophistication_level: Sophistication level
            attack_vectors: Attack vectors used
            relevance_to_entity: Relevance to the entity
            industry_targeting: Whether targeting the industry
            potential_impact: Potential impact
            intelligence_source: Intelligence source
            confidence_level: Confidence in the intelligence

        Returns:
            Recorded CyberThreat
        """
        threat = CyberThreat(
            name=name,
            category=category,
            description=description,
            threat_actor_type=threat_actor_type,
            sophistication_level=sophistication_level,
            attack_vectors=attack_vectors or [],
            relevance_to_entity=relevance_to_entity,
            industry_targeting=industry_targeting,
            potential_impact=potential_impact,
            intelligence_source=intelligence_source,
            confidence_level=confidence_level,
            first_observed=datetime.now(timezone.utc).isoformat(),
        )

        with self._lock:
            self._threats[threat.threat_id] = threat

        self._log_event(
            "threat_recorded",
            {
                "threat_id": threat.threat_id,
                "name": name,
                "category": category.value,
                "relevance": relevance_to_entity,
            },
        )

        logger.info(f"Threat recorded: {name}")
        return threat

    def get_threat(self, threat_id: str) -> Optional[CyberThreat]:
        """Get threat by ID."""
        with self._lock:
            return self._threats.get(threat_id)

    def get_threats_by_category(self, category: ThreatCategory) -> List[CyberThreat]:
        """Get threats by category."""
        with self._lock:
            return [t for t in self._threats.values() if t.category == category and t.is_active]

    def get_high_relevance_threats(self) -> List[CyberThreat]:
        """Get threats with high relevance to the entity."""
        with self._lock:
            return [
                t for t in self._threats.values() if t.relevance_to_entity == "high" and t.is_active
            ]

    # =========================================================================
    # Vulnerability Management (Article 8(3))
    # =========================================================================

    def record_vulnerability(
        self,
        name: str,
        severity: VulnerabilitySeverity,
        description: str = "",
        cve_id: str = "",
        cvss_score: float = 0.0,
        affected_asset_ids: Optional[List[str]] = None,
        affected_systems: Optional[List[str]] = None,
        exploitability: str = "none",
        exploit_in_wild: bool = False,
        remediation_owner: str = "",
    ) -> ICTVulnerability:
        """
        Record an ICT vulnerability.

        Per Article 8(3), entities must assess ICT vulnerabilities.

        Args:
            name: Vulnerability name
            severity: Severity level
            description: Vulnerability description
            cve_id: CVE identifier
            cvss_score: CVSS score
            affected_asset_ids: Affected asset IDs
            affected_systems: Affected system names
            exploitability: Exploitability level
            exploit_in_wild: Whether exploit exists in the wild
            remediation_owner: Owner of remediation

        Returns:
            Recorded ICTVulnerability
        """
        # Determine remediation target date based on severity
        target_days = {
            VulnerabilitySeverity.CRITICAL: self.config.critical_vuln_remediation_days,
            VulnerabilitySeverity.HIGH: self.config.high_vuln_remediation_days,
            VulnerabilitySeverity.MEDIUM: self.config.medium_vuln_remediation_days,
            VulnerabilitySeverity.LOW: self.config.low_vuln_remediation_days,
        }.get(severity, 90)

        target_date = (datetime.now(timezone.utc) + timedelta(days=target_days)).isoformat()

        vulnerability = ICTVulnerability(
            name=name,
            severity=severity,
            description=description,
            cve_id=cve_id,
            cvss_score=cvss_score,
            affected_asset_ids=affected_asset_ids or [],
            affected_systems=affected_systems or [],
            exploitability=exploitability,
            exploit_in_wild=exploit_in_wild,
            status="open",
            target_remediation_date=target_date,
            remediation_owner=remediation_owner,
        )

        with self._lock:
            self._vulnerabilities[vulnerability.vulnerability_id] = vulnerability

        # Alert for critical/high with exploit in wild
        if (
            severity in (VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH)
            and exploit_in_wild
        ):
            self._alert_critical_vulnerability(vulnerability)

        self._log_event(
            "vulnerability_recorded",
            {
                "vulnerability_id": vulnerability.vulnerability_id,
                "name": name,
                "severity": severity.value,
                "cve_id": cve_id,
            },
        )

        logger.info(f"Vulnerability recorded: {name} ({severity.value})")
        return vulnerability

    def update_vulnerability_status(
        self,
        vulnerability_id: str,
        status: str,
        remediation_details: str = "",
    ) -> Optional[ICTVulnerability]:
        """Update vulnerability status."""
        with self._lock:
            if vulnerability_id not in self._vulnerabilities:
                return None

            vuln = self._vulnerabilities[vulnerability_id]
            vuln.status = status

            if status == "closed" or status == "mitigated":
                vuln.actual_remediation_date = datetime.now(timezone.utc).isoformat()

            if remediation_details:
                vuln.remediation_plan = remediation_details

        return vuln

    def accept_vulnerability_risk(
        self,
        vulnerability_id: str,
        justification: str,
        accepted_by: str,
        expiry_days: int = 90,
    ) -> Optional[ICTVulnerability]:
        """Accept risk for a vulnerability."""
        with self._lock:
            if vulnerability_id not in self._vulnerabilities:
                return None

            vuln = self._vulnerabilities[vulnerability_id]
            vuln.risk_accepted = True
            vuln.acceptance_justification = justification
            vuln.accepted_by = accepted_by
            vuln.acceptance_expiry = (
                datetime.now(timezone.utc) + timedelta(days=expiry_days)
            ).isoformat()
            vuln.status = "accepted"

        self._log_event(
            "vulnerability_risk_accepted",
            {
                "vulnerability_id": vulnerability_id,
                "accepted_by": accepted_by,
                "expiry_days": expiry_days,
            },
        )

        return vuln

    def get_vulnerability(self, vulnerability_id: str) -> Optional[ICTVulnerability]:
        """Get vulnerability by ID."""
        with self._lock:
            return self._vulnerabilities.get(vulnerability_id)

    def get_open_vulnerabilities(self) -> List[ICTVulnerability]:
        """Get all open vulnerabilities."""
        with self._lock:
            return [v for v in self._vulnerabilities.values() if v.status == "open"]

    def get_overdue_vulnerabilities(self) -> List[ICTVulnerability]:
        """Get vulnerabilities past their remediation deadline."""
        now = datetime.now(timezone.utc)

        with self._lock:
            overdue = []
            for vuln in self._vulnerabilities.values():
                if vuln.status not in ("open", "in_remediation"):
                    continue
                if vuln.target_remediation_date:
                    target = datetime.fromisoformat(
                        vuln.target_remediation_date.replace("Z", "+00:00")
                    )
                    if now > target:
                        overdue.append(vuln)

            return overdue

    def get_critical_vulnerabilities(self) -> List[ICTVulnerability]:
        """Get critical and high severity open vulnerabilities."""
        with self._lock:
            return [
                v
                for v in self._vulnerabilities.values()
                if v.severity in (VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH)
                and v.status == "open"
            ]

    def _alert_critical_vulnerability(self, vuln: ICTVulnerability) -> None:
        """Alert on critical vulnerability."""
        if self.config.alert_callback:
            try:
                self.config.alert_callback("critical_vulnerability", asdict(vuln))
            except Exception as e:
                logger.error(f"Vulnerability alert callback failed: {e}")

        logger.warning(
            f"Critical vulnerability with exploit in wild: "
            f"{vuln.vulnerability_id} - {vuln.name}"
        )

    # =========================================================================
    # Dependency Mapping (Article 8(4))
    # =========================================================================

    def map_dependency(
        self,
        source_id: str,
        source_type: str,
        target_id: str,
        target_type: str,
        dependency_type: DependencyType,
        criticality: str = "medium",
        description: str = "",
        failure_impact: str = "",
        max_acceptable_downtime_hours: float = 0.0,
        is_third_party: bool = False,
        third_party_provider_id: str = "",
    ) -> ICTDependency:
        """
        Map an ICT dependency.

        Per Article 8(4), entities must map dependencies between ICT assets,
        functions, and third-party providers.

        Args:
            source_id: Source asset/function ID
            source_type: Source type ("asset", "function", "system")
            target_id: Target asset/function ID
            target_type: Target type
            dependency_type: Type of dependency
            criticality: Dependency criticality
            description: Dependency description
            failure_impact: Impact if dependency fails
            max_acceptable_downtime_hours: Max acceptable downtime
            is_third_party: Whether dependency is on third party
            third_party_provider_id: Third party provider ID

        Returns:
            Mapped ICTDependency
        """
        dependency = ICTDependency(
            source_id=source_id,
            source_type=source_type,
            target_id=target_id,
            target_type=target_type,
            dependency_type=dependency_type,
            criticality=criticality,
            description=description,
            failure_impact=failure_impact,
            max_acceptable_downtime_hours=max_acceptable_downtime_hours,
            is_third_party=is_third_party,
            third_party_provider_id=third_party_provider_id,
        )

        with self._lock:
            self._dependencies[dependency.dependency_id] = dependency

            # Update asset dependencies if applicable
            if source_type == "asset" and source_id in self._assets:
                asset = self._assets[source_id]
                if target_id not in asset.depends_on:
                    asset.depends_on.append(target_id)

            if target_type == "asset" and target_id in self._assets:
                asset = self._assets[target_id]
                if source_id not in asset.depended_by:
                    asset.depended_by.append(source_id)

        self._log_event(
            "dependency_mapped",
            {
                "dependency_id": dependency.dependency_id,
                "source_id": source_id,
                "target_id": target_id,
                "type": dependency_type.value,
            },
        )

        return dependency

    def get_dependencies_for(
        self,
        entity_id: str,
        direction: str = "both",
    ) -> List[ICTDependency]:
        """
        Get dependencies for an entity.

        Args:
            entity_id: Entity ID (asset or function)
            direction: "upstream", "downstream", or "both"

        Returns:
            List of dependencies
        """
        with self._lock:
            results = []

            for dep in self._dependencies.values():
                if direction in ("both", "upstream"):
                    if dep.source_id == entity_id:
                        results.append(dep)

                if direction in ("both", "downstream"):
                    if dep.target_id == entity_id:
                        results.append(dep)

            return results

    def get_third_party_dependencies(self) -> List[ICTDependency]:
        """Get all third-party dependencies."""
        with self._lock:
            return [d for d in self._dependencies.values() if d.is_third_party]

    def get_critical_dependencies(self) -> List[ICTDependency]:
        """Get critical dependencies."""
        with self._lock:
            return [d for d in self._dependencies.values() if d.criticality == "critical"]

    def analyze_dependency_chain(
        self,
        entity_id: str,
        max_depth: Optional[int] = None,
    ) -> List[List[str]]:
        """
        Analyze dependency chains for an entity.

        Returns list of dependency chains (paths) from entity to all dependencies.
        """
        max_depth = max_depth or self.config.max_dependency_depth
        chains: List[List[str]] = []
        visited: Set[str] = set()

        def _traverse(current_id: str, current_chain: List[str], depth: int) -> None:
            if depth >= max_depth or current_id in visited:
                return

            visited.add(current_id)

            deps = self.get_dependencies_for(current_id, direction="upstream")
            if not deps:
                if len(current_chain) > 1:
                    chains.append(current_chain.copy())
                return

            for dep in deps:
                new_chain = current_chain + [dep.target_id]
                chains.append(new_chain.copy())
                _traverse(dep.target_id, new_chain, depth + 1)

        _traverse(entity_id, [entity_id], 0)
        return chains

    # =========================================================================
    # Business Function Mapping
    # =========================================================================

    def register_business_function(
        self,
        name: str,
        description: str = "",
        business_unit: str = "",
        is_critical: bool = False,
        is_important: bool = False,
        criticality_rationale: str = "",
        rto_hours: float = 4.0,
        rpo_hours: float = 1.0,
        supporting_asset_ids: Optional[List[str]] = None,
        supporting_system_ids: Optional[List[str]] = None,
        business_owner: str = "",
        technical_owner: str = "",
    ) -> BusinessFunction:
        """
        Register a business function.

        Args:
            name: Function name
            description: Function description
            business_unit: Business unit
            is_critical: Whether function is critical
            is_important: Whether function is important
            criticality_rationale: Rationale for criticality
            rto_hours: Recovery time objective
            rpo_hours: Recovery point objective
            supporting_asset_ids: Supporting asset IDs
            supporting_system_ids: Supporting system IDs
            business_owner: Business owner
            technical_owner: Technical owner

        Returns:
            Registered BusinessFunction
        """
        function = BusinessFunction(
            name=name,
            description=description,
            business_unit=business_unit,
            is_critical=is_critical,
            is_important=is_important,
            criticality_rationale=criticality_rationale,
            rto_hours=rto_hours,
            rpo_hours=rpo_hours,
            supporting_asset_ids=supporting_asset_ids or [],
            supporting_system_ids=supporting_system_ids or [],
            business_owner=business_owner,
            technical_owner=technical_owner,
        )

        with self._lock:
            self._functions[function.function_id] = function

        self._log_event(
            "function_registered",
            {
                "function_id": function.function_id,
                "name": name,
                "is_critical": is_critical,
            },
        )

        return function

    def get_business_function(self, function_id: str) -> Optional[BusinessFunction]:
        """Get business function by ID."""
        with self._lock:
            return self._functions.get(function_id)

    def get_critical_functions(self) -> List[BusinessFunction]:
        """Get all critical business functions."""
        with self._lock:
            return [f for f in self._functions.values() if f.is_critical and f.is_active]

    def get_important_functions(self) -> List[BusinessFunction]:
        """Get all important business functions."""
        with self._lock:
            return [f for f in self._functions.values() if f.is_important and f.is_active]

    def get_functions_for_asset(self, asset_id: str) -> List[BusinessFunction]:
        """Get functions that depend on a specific asset."""
        with self._lock:
            return [
                f
                for f in self._functions.values()
                if asset_id in f.supporting_asset_ids and f.is_active
            ]

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_identification_summary(self) -> Dict[str, Any]:
        """Get comprehensive identification summary."""
        with self._lock:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "assets": {
                    "total": len(self._assets),
                    "active": sum(1 for a in self._assets.values() if a.is_active),
                    "critical": len(self.get_critical_assets()),
                    "third_party": len(self.get_third_party_assets()),
                    "by_type": {
                        t.value: sum(1 for a in self._assets.values() if a.asset_type == t)
                        for t in AssetType
                    },
                    "by_classification": {
                        c.value: sum(1 for a in self._assets.values() if a.classification == c)
                        for c in AssetClassification
                    },
                },
                "risk_sources": {
                    "total": len(self._risk_sources),
                    "active": sum(1 for r in self._risk_sources.values() if r.is_active),
                    "high_critical": len(self.get_high_risk_sources()),
                    "emerging": len(self.get_emerging_risks()),
                    "by_category": {
                        c.value: sum(1 for r in self._risk_sources.values() if r.category == c)
                        for c in RiskSourceCategory
                    },
                },
                "threats": {
                    "total": len(self._threats),
                    "active": sum(1 for t in self._threats.values() if t.is_active),
                    "high_relevance": len(self.get_high_relevance_threats()),
                    "by_category": {
                        c.value: sum(1 for t in self._threats.values() if t.category == c)
                        for c in ThreatCategory
                    },
                },
                "vulnerabilities": {
                    "total": len(self._vulnerabilities),
                    "open": len(self.get_open_vulnerabilities()),
                    "overdue": len(self.get_overdue_vulnerabilities()),
                    "critical_high": len(self.get_critical_vulnerabilities()),
                    "by_severity": {
                        s.value: sum(1 for v in self._vulnerabilities.values() if v.severity == s)
                        for s in VulnerabilitySeverity
                    },
                },
                "dependencies": {
                    "total": len(self._dependencies),
                    "third_party": len(self.get_third_party_dependencies()),
                    "critical": len(self.get_critical_dependencies()),
                },
                "functions": {
                    "total": len(self._functions),
                    "critical": len(self.get_critical_functions()),
                    "important": len(self.get_important_functions()),
                },
            }

    def export_asset_inventory(self) -> Dict[str, Any]:
        """Export complete asset inventory per Article 8."""
        with self._lock:
            return {
                "export_date": datetime.now(timezone.utc).isoformat(),
                "article_reference": "Article 8",
                "summary": self.get_identification_summary(),
                "assets": [asdict(a) for a in self._assets.values()],
                "risk_sources": [asdict(r) for r in self._risk_sources.values()],
                "threats": [asdict(t) for t in self._threats.values()],
                "vulnerabilities": [asdict(v) for v in self._vulnerabilities.values()],
                "dependencies": [asdict(d) for d in self._dependencies.values()],
                "functions": [asdict(f) for f in self._functions.values()],
            }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event."""
        if not self.config.log_all_events:
            return

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = (
            self._log_path / f"ict_identification_events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================


def create_ict_identification(
    config: Optional[ICTIdentificationConfig] = None,
) -> DORAICTIdentification:
    """
    Create a DORAICTIdentification instance.

    Args:
        config: Optional configuration

    Returns:
        Configured DORAICTIdentification instance
    """
    return DORAICTIdentification(config=config)
