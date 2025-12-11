# -*- coding: utf-8 -*-
"""
DORA Concentration Risk Module (Article 29).

ICT Provider Perspective (Why This Module Exists):
==================================================
As an ICT Third-Party Service Provider (Art. 30), this module serves THREE purposes:

    1. OUR SUBCONTRACTOR CONCENTRATION: Assess and manage OUR concentration risk
       on cloud providers, data centers, and other subcontractors. Per Art. 30(2)(a),
       we must manage subcontracting arrangements responsibly. If we depend heavily
       on AWS or Google Cloud, we need to understand and mitigate that risk.

    2. CLIENT TRANSPARENCY: Provide clients with data about their concentration
       risk ON US. Financial entities (Art. 29) must assess concentration risk
       including "the degree of substitutability" of their ICT providers.
       We help them by providing:
       - Our substitutability metrics
       - Our geographic footprint
       - Our subcontractor dependencies
       - Exit strategy feasibility data

    3. B2B ANALYTICS ADDON: Enterprise clients can use this module to analyze
       their overall ICT concentration across all providers (not just us).
       This is a value-added feature for their Art. 29 compliance.

Regulatory Context:
    Article 29 DORA requires FINANCIAL ENTITIES (not ICT providers) to assess
    concentration risk. However, as an ICT provider we:
    - Must provide data to support client assessments (Art. 30(3)(e))
    - Should manage our own supply chain concentration (best practice)
    - Can offer concentration analytics as enterprise B2B feature

Key Concentration Risk Types:
    1. Provider concentration - Single provider for multiple services
    2. Geographic concentration - Data/services in limited locations
    3. Service concentration - Critical functions on few providers
    4. Technology concentration - Limited technology diversity
    5. Subcontractor concentration - Common subcontractors in supply chains

This module provides:
    1. OUR subcontractor concentration analysis
    2. Client dependency metrics (how much they depend on us)
    3. Substitutability scoring and exit feasibility data
    4. Geographic and technology diversity reporting
    5. Concentration risk data export for client ROI submissions

Architecture Context:
    This module is part of the INTEGRATION layer (services/dora_integration/),
    providing client-facing interfaces for DORA compliance. It is NOT part of
    the archived Financial Entity modules (services/archive/dora_financial_entity/).

References:
    - Article 29 DORA: https://www.digital-operational-resilience-act.com/Article_29.html
    - Article 30(3)(e) DORA: Cooperation with FE due diligence
    - RTS on ICT Risk Management: CDR 2024/1774
    - ESAs Guidelines on ICT Concentration Risk
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
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class ConcentrationType(Enum):
    """Types of ICT concentration risk."""
    PROVIDER = "provider"  # Single provider for multiple services
    GEOGRAPHIC = "geographic"  # Data/services concentrated geographically
    SERVICE = "service"  # Critical services on limited providers
    TECHNOLOGY = "technology"  # Limited technology diversity
    SUBCONTRACTOR = "subcontractor"  # Common subcontractors in chains
    SECTOR = "sector"  # Concentration within financial sector
    CLOUD = "cloud"  # Cloud service concentration


class RiskLevel(Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MitigationStatus(Enum):
    """Mitigation measure status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"
    NOT_APPLICABLE = "not_applicable"


class AssessmentScope(Enum):
    """Assessment scope level."""
    ENTITY = "entity"
    SUB_CONSOLIDATED = "sub_consolidated"
    CONSOLIDATED = "consolidated"


class SubstitutabilityLevel(Enum):
    """Provider substitutability classification."""
    EASILY_SUBSTITUTABLE = "easily_substitutable"
    SUBSTITUTABLE_WITH_EFFORT = "substitutable_with_effort"
    DIFFICULT_TO_SUBSTITUTE = "difficult_to_substitute"
    NOT_SUBSTITUTABLE = "not_substitutable"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ProviderDependency:
    """
    Provider dependency information.
    """
    provider_id: str = ""
    provider_name: str = ""
    provider_country: str = ""
    is_ctpp: bool = False

    # Services provided
    services: List[str] = field(default_factory=list)
    critical_functions_supported: List[str] = field(default_factory=list)

    # Dependency metrics
    services_count: int = 0
    critical_services_count: int = 0
    revenue_dependency_pct: float = 0.0
    transaction_volume_pct: float = 0.0

    # Substitutability
    substitutability: SubstitutabilityLevel = SubstitutabilityLevel.SUBSTITUTABLE_WITH_EFFORT
    alternatives_count: int = 0
    alternatives: List[str] = field(default_factory=list)

    # Data location
    data_processing_countries: List[str] = field(default_factory=list)
    data_storage_countries: List[str] = field(default_factory=list)

    # Subcontracting
    known_subcontractors: List[str] = field(default_factory=list)


@dataclass
class ConcentrationMetric:
    """
    Single concentration metric.
    """
    metric_id: str = ""
    metric_name: str = ""
    metric_type: ConcentrationType = ConcentrationType.PROVIDER
    dimension: str = ""  # What is being measured

    # Values
    current_value: float = 0.0
    threshold_warning: float = 0.0
    threshold_critical: float = 0.0
    unit: str = ""  # percentage, count, etc.

    # Status
    status: str = ""  # normal, warning, critical
    last_calculated: str = ""

    # Details
    contributing_factors: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self):
        if not self.metric_id:
            self.metric_id = f"MET-{uuid.uuid4().hex[:8].upper()}"
        if not self.last_calculated:
            self.last_calculated = datetime.now(timezone.utc).isoformat()

        # Determine status
        if self.current_value >= self.threshold_critical:
            self.status = "critical"
        elif self.current_value >= self.threshold_warning:
            self.status = "warning"
        else:
            self.status = "normal"


@dataclass
class ConcentrationRisk:
    """
    Identified concentration risk.
    """
    risk_id: str = ""
    risk_name: str = ""
    risk_type: ConcentrationType = ConcentrationType.PROVIDER
    description: str = ""

    # Risk assessment
    likelihood: int = 3  # 1-5
    impact: int = 3  # 1-5
    risk_level: RiskLevel = RiskLevel.MEDIUM

    # Affected items
    affected_providers: List[str] = field(default_factory=list)
    affected_services: List[str] = field(default_factory=list)
    affected_functions: List[str] = field(default_factory=list)
    affected_countries: List[str] = field(default_factory=list)

    # Concentration details
    concentration_factor: float = 0.0  # e.g., % of services
    related_metrics: List[str] = field(default_factory=list)

    # Impact analysis
    business_impact: str = ""
    regulatory_impact: str = ""
    financial_impact_estimate_eur: float = 0.0

    # Mitigation
    mitigation_measures: List[str] = field(default_factory=list)
    mitigation_status: MitigationStatus = MitigationStatus.NOT_STARTED
    mitigation_owner: str = ""
    mitigation_deadline: Optional[str] = None

    # Tracking
    identified_date: str = ""
    last_reviewed: str = ""
    review_frequency_months: int = 6

    def __post_init__(self):
        if not self.risk_id:
            self.risk_id = f"CCR-{uuid.uuid4().hex[:8].upper()}"
        if not self.identified_date:
            self.identified_date = datetime.now(timezone.utc).isoformat()
        if not self.last_reviewed:
            self.last_reviewed = self.identified_date

        # Calculate risk level
        score = self.likelihood * self.impact
        if score <= 4:
            self.risk_level = RiskLevel.LOW
        elif score <= 9:
            self.risk_level = RiskLevel.MEDIUM
        elif score <= 16:
            self.risk_level = RiskLevel.HIGH
        else:
            self.risk_level = RiskLevel.CRITICAL


@dataclass
class MitigationMeasure:
    """
    Concentration risk mitigation measure.
    """
    measure_id: str = ""
    risk_id: str = ""  # Related risk
    measure_name: str = ""
    description: str = ""

    # Type
    measure_type: str = ""  # diversify, alternative, exit_plan, monitoring, acceptance

    # Actions
    actions: List[Dict[str, Any]] = field(default_factory=list)

    # Status
    status: MitigationStatus = MitigationStatus.NOT_STARTED
    progress_pct: float = 0.0

    # Ownership
    owner: str = ""
    team: str = ""

    # Timeline
    planned_start: Optional[str] = None
    planned_end: Optional[str] = None
    actual_start: Optional[str] = None
    actual_end: Optional[str] = None

    # Effectiveness
    expected_risk_reduction: str = ""  # Qualitative
    actual_risk_reduction: str = ""

    # Cost
    estimated_cost_eur: float = 0.0
    actual_cost_eur: float = 0.0

    def __post_init__(self):
        if not self.measure_id:
            self.measure_id = f"MIT-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class ConcentrationAssessment:
    """
    Comprehensive concentration risk assessment.
    """
    assessment_id: str = ""
    assessment_date: str = ""
    assessed_by: str = ""
    scope: AssessmentScope = AssessmentScope.ENTITY

    # Entity information (for consolidated)
    entities_in_scope: List[str] = field(default_factory=list)

    # Metrics calculated
    metrics: List[str] = field(default_factory=list)  # Metric IDs

    # Risks identified
    risks_identified: List[str] = field(default_factory=list)  # Risk IDs
    new_risks_count: int = 0
    existing_risks_updated: int = 0

    # Summary by type
    risks_by_type: Dict[str, int] = field(default_factory=dict)
    risks_by_level: Dict[str, int] = field(default_factory=dict)

    # Overall assessment
    overall_concentration_level: RiskLevel = RiskLevel.MEDIUM
    key_concentration_areas: List[str] = field(default_factory=list)

    # Recommendations
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    required_actions: List[Dict[str, Any]] = field(default_factory=list)

    # Approval
    reviewed_by: str = ""
    review_date: Optional[str] = None
    approved_by: str = ""
    approval_date: Optional[str] = None

    def __post_init__(self):
        if not self.assessment_id:
            self.assessment_id = f"CCA-{uuid.uuid4().hex[:8].upper()}"
        if not self.assessment_date:
            self.assessment_date = datetime.now(timezone.utc).isoformat()


@dataclass
class DependencyMap:
    """
    Provider dependency map for visualization and analysis.
    """
    map_id: str = ""
    generated_date: str = ""

    # Dependencies
    providers: Dict[str, ProviderDependency] = field(default_factory=dict)
    total_providers: int = 0

    # Geographic distribution
    countries: Dict[str, List[str]] = field(default_factory=dict)  # country -> provider_ids

    # Function mapping
    functions_to_providers: Dict[str, List[str]] = field(default_factory=dict)

    # Concentration scores
    provider_hhi: float = 0.0  # Herfindahl-Hirschman Index
    geographic_hhi: float = 0.0
    function_coverage: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.map_id:
            self.map_id = f"DMP-{uuid.uuid4().hex[:8].upper()}"
        if not self.generated_date:
            self.generated_date = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ConcentrationRiskConfig:
    """Configuration for Concentration Risk management."""

    # Thresholds
    provider_concentration_warning_pct: float = 30.0
    provider_concentration_critical_pct: float = 50.0
    geographic_concentration_warning_pct: float = 40.0
    geographic_concentration_critical_pct: float = 60.0
    single_function_dependency_warning: int = 1
    single_function_dependency_critical: int = 1

    # Limits
    max_critical_functions_per_provider: int = 3
    min_providers_per_critical_function: int = 2
    max_single_country_data_pct: float = 70.0

    # Assessment
    assessment_frequency_months: int = 6
    hhi_threshold_concentrated: float = 2500  # Standard HHI threshold

    # Notifications
    notify_on_threshold_breach: bool = True
    notify_on_new_critical_risk: bool = True

    # Documentation
    log_all_events: bool = True
    log_path: str = "logs/dora/concentration_risk"

    # Callbacks
    escalation_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    notification_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# Main Implementation
# =============================================================================

class DORAConcentrationRisk:
    """
    DORA Article 29 compliant ICT Concentration Risk Management.

    Implements comprehensive concentration risk assessment and monitoring:
    - Provider dependency analysis
    - Geographic concentration assessment
    - Critical function distribution
    - Substitutability analysis
    - Risk identification and mitigation
    - Concentration metrics calculation

    Per Article 29:
    - Assess concentration at entity, sub-consolidated, and consolidated levels
    - Consider benefits vs risks of diversification
    - Identify concentration at time of contracting
    - Evaluate systemic implications

    Usage:
        config = ConcentrationRiskConfig()
        manager = DORAConcentrationRisk(config)

        # Add provider dependencies
        manager.add_provider_dependency(
            provider_id="PRV-001",
            provider_name="Cloud Provider",
            services=["compute", "storage"],
            critical_functions=["order_execution"],
        )

        # Perform assessment
        assessment = manager.perform_concentration_assessment(
            scope=AssessmentScope.ENTITY,
        )

        # Get metrics
        metrics = manager.calculate_concentration_metrics()
    """

    def __init__(self, config: Optional[ConcentrationRiskConfig] = None):
        """Initialize Concentration Risk Management."""
        self.config = config or ConcentrationRiskConfig()

        # Data stores
        self._dependencies: Dict[str, ProviderDependency] = {}
        self._metrics: Dict[str, ConcentrationMetric] = {}
        self._risks: Dict[str, ConcentrationRisk] = {}
        self._mitigations: Dict[str, MitigationMeasure] = {}
        self._assessments: Dict[str, ConcentrationAssessment] = {}

        # Indexes
        self._providers_by_function: Dict[str, Set[str]] = defaultdict(set)
        self._providers_by_country: Dict[str, Set[str]] = defaultdict(set)
        self._services_by_provider: Dict[str, Set[str]] = defaultdict(set)

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("DORAConcentrationRisk initialized")

    # =========================================================================
    # Provider Dependency Management
    # =========================================================================

    def add_provider_dependency(
        self,
        provider_id: str,
        provider_name: str,
        services: List[str],
        provider_country: str = "",
        is_ctpp: bool = False,
        critical_functions: Optional[List[str]] = None,
        revenue_dependency_pct: float = 0.0,
        transaction_volume_pct: float = 0.0,
        substitutability: SubstitutabilityLevel = SubstitutabilityLevel.SUBSTITUTABLE_WITH_EFFORT,
        alternatives: Optional[List[str]] = None,
        data_processing_countries: Optional[List[str]] = None,
        data_storage_countries: Optional[List[str]] = None,
        subcontractors: Optional[List[str]] = None,
    ) -> ProviderDependency:
        """
        Add or update provider dependency information.

        Args:
            provider_id: Provider identifier
            provider_name: Provider name
            services: Services provided
            provider_country: Provider country
            is_ctpp: Is designated CTPP
            critical_functions: Critical functions supported
            revenue_dependency_pct: Revenue dependency percentage
            transaction_volume_pct: Transaction volume percentage
            substitutability: Substitutability level
            alternatives: Alternative providers
            data_processing_countries: Data processing locations
            data_storage_countries: Data storage locations
            subcontractors: Known subcontractors

        Returns:
            ProviderDependency
        """
        dependency = ProviderDependency(
            provider_id=provider_id,
            provider_name=provider_name,
            provider_country=provider_country.upper() if provider_country else "",
            is_ctpp=is_ctpp,
            services=services,
            critical_functions_supported=critical_functions or [],
            services_count=len(services),
            critical_services_count=len(critical_functions or []),
            revenue_dependency_pct=revenue_dependency_pct,
            transaction_volume_pct=transaction_volume_pct,
            substitutability=substitutability,
            alternatives_count=len(alternatives or []),
            alternatives=alternatives or [],
            data_processing_countries=[c.upper() for c in (data_processing_countries or [])],
            data_storage_countries=[c.upper() for c in (data_storage_countries or [])],
            known_subcontractors=subcontractors or [],
        )

        with self._lock:
            self._dependencies[provider_id] = dependency

            # Update indexes
            for func in (critical_functions or []):
                self._providers_by_function[func].add(provider_id)

            for country in (data_processing_countries or []) + (data_storage_countries or []):
                self._providers_by_country[country.upper()].add(provider_id)

            for service in services:
                self._services_by_provider[provider_id].add(service)

        self._log_event("dependency_added", {
            "provider_id": provider_id,
            "provider_name": provider_name,
            "services_count": len(services),
            "critical_functions": critical_functions or [],
        })

        return dependency

    def get_provider_dependency(
        self,
        provider_id: str,
    ) -> Optional[ProviderDependency]:
        """Get provider dependency."""
        with self._lock:
            return self._dependencies.get(provider_id)

    def get_all_dependencies(self) -> List[ProviderDependency]:
        """Get all provider dependencies."""
        with self._lock:
            return list(self._dependencies.values())

    def get_providers_for_function(
        self,
        function_name: str,
    ) -> List[ProviderDependency]:
        """Get all providers supporting a function."""
        with self._lock:
            provider_ids = self._providers_by_function.get(function_name, set())
            return [
                self._dependencies[pid]
                for pid in provider_ids
                if pid in self._dependencies
            ]

    def get_providers_in_country(
        self,
        country: str,
    ) -> List[ProviderDependency]:
        """Get all providers with data in a country."""
        with self._lock:
            provider_ids = self._providers_by_country.get(country.upper(), set())
            return [
                self._dependencies[pid]
                for pid in provider_ids
                if pid in self._dependencies
            ]

    # =========================================================================
    # Concentration Metrics
    # =========================================================================

    def calculate_concentration_metrics(self) -> List[ConcentrationMetric]:
        """
        Calculate all concentration metrics.

        Returns list of calculated metrics.
        """
        metrics = []

        with self._lock:
            dependencies = list(self._dependencies.values())
            if not dependencies:
                return metrics

            # 1. Provider concentration (HHI)
            provider_hhi = self._calculate_provider_hhi()
            metrics.append(ConcentrationMetric(
                metric_name="Provider Concentration (HHI)",
                metric_type=ConcentrationType.PROVIDER,
                dimension="service_distribution",
                current_value=provider_hhi,
                threshold_warning=1500,
                threshold_critical=self.config.hhi_threshold_concentrated,
                unit="hhi_points",
                notes="Herfindahl-Hirschman Index for provider concentration",
            ))

            # 2. Top provider dependency
            top_provider = self._get_top_provider()
            if top_provider:
                top_pct = (top_provider.services_count / max(1, sum(
                    d.services_count for d in dependencies
                ))) * 100
                metrics.append(ConcentrationMetric(
                    metric_name=f"Top Provider Dependency ({top_provider.provider_name})",
                    metric_type=ConcentrationType.PROVIDER,
                    dimension="single_provider",
                    current_value=top_pct,
                    threshold_warning=self.config.provider_concentration_warning_pct,
                    threshold_critical=self.config.provider_concentration_critical_pct,
                    unit="percentage",
                    contributing_factors=[{
                        "provider": top_provider.provider_name,
                        "services": top_provider.services_count,
                    }],
                ))

            # 3. Geographic concentration
            geo_metrics = self._calculate_geographic_concentration()
            metrics.extend(geo_metrics)

            # 4. Critical function provider count
            func_metrics = self._calculate_function_concentration()
            metrics.extend(func_metrics)

            # 5. CTPP concentration
            ctpp_count = sum(1 for d in dependencies if d.is_ctpp)
            if ctpp_count > 0:
                ctpp_pct = (ctpp_count / len(dependencies)) * 100
                metrics.append(ConcentrationMetric(
                    metric_name="CTPP Provider Ratio",
                    metric_type=ConcentrationType.PROVIDER,
                    dimension="ctpp_dependency",
                    current_value=ctpp_pct,
                    threshold_warning=30.0,
                    threshold_critical=50.0,
                    unit="percentage",
                    notes="Percentage of providers that are designated CTPPs",
                ))

            # 6. Substitutability analysis
            non_substitutable = sum(
                1 for d in dependencies
                if d.substitutability in [
                    SubstitutabilityLevel.DIFFICULT_TO_SUBSTITUTE,
                    SubstitutabilityLevel.NOT_SUBSTITUTABLE
                ]
            )
            non_sub_pct = (non_substitutable / len(dependencies)) * 100
            metrics.append(ConcentrationMetric(
                metric_name="Non-Substitutable Providers",
                metric_type=ConcentrationType.PROVIDER,
                dimension="substitutability",
                current_value=non_sub_pct,
                threshold_warning=20.0,
                threshold_critical=40.0,
                unit="percentage",
            ))

            # 7. Subcontractor concentration
            sub_metrics = self._calculate_subcontractor_concentration()
            metrics.extend(sub_metrics)

            # Store metrics
            for m in metrics:
                self._metrics[m.metric_id] = m

        self._log_event("metrics_calculated", {
            "metrics_count": len(metrics),
            "critical_count": sum(1 for m in metrics if m.status == "critical"),
            "warning_count": sum(1 for m in metrics if m.status == "warning"),
        })

        return metrics

    def _calculate_provider_hhi(self) -> float:
        """Calculate Herfindahl-Hirschman Index for providers."""
        dependencies = list(self._dependencies.values())
        if not dependencies:
            return 0.0

        total_services = sum(d.services_count for d in dependencies)
        if total_services == 0:
            return 0.0

        hhi = sum(
            ((d.services_count / total_services) * 100) ** 2
            for d in dependencies
        )
        return hhi

    def _get_top_provider(self) -> Optional[ProviderDependency]:
        """Get provider with most services."""
        dependencies = list(self._dependencies.values())
        if not dependencies:
            return None
        return max(dependencies, key=lambda d: d.services_count)

    def _calculate_geographic_concentration(self) -> List[ConcentrationMetric]:
        """Calculate geographic concentration metrics."""
        metrics = []

        # Count data by country
        country_counts: Dict[str, int] = defaultdict(int)
        for dep in self._dependencies.values():
            for country in dep.data_processing_countries + dep.data_storage_countries:
                country_counts[country] += 1

        if not country_counts:
            return metrics

        total = sum(country_counts.values())

        # Top country concentration
        if country_counts:
            top_country = max(country_counts.items(), key=lambda x: x[1])
            top_pct = (top_country[1] / total) * 100
            metrics.append(ConcentrationMetric(
                metric_name=f"Geographic Concentration ({top_country[0]})",
                metric_type=ConcentrationType.GEOGRAPHIC,
                dimension="top_country",
                current_value=top_pct,
                threshold_warning=self.config.geographic_concentration_warning_pct,
                threshold_critical=self.config.geographic_concentration_critical_pct,
                unit="percentage",
                contributing_factors=[
                    {"country": c, "count": v}
                    for c, v in sorted(country_counts.items(), key=lambda x: -x[1])[:5]
                ],
            ))

        # EU vs non-EU concentration
        eu_countries = self._get_eu_countries()
        eu_count = sum(v for k, v in country_counts.items() if k in eu_countries)
        eu_pct = (eu_count / total) * 100 if total > 0 else 0
        metrics.append(ConcentrationMetric(
            metric_name="EU Data Concentration",
            metric_type=ConcentrationType.GEOGRAPHIC,
            dimension="eu_vs_non_eu",
            current_value=eu_pct,
            threshold_warning=0,  # Not necessarily a warning
            threshold_critical=0,
            unit="percentage",
            notes="Percentage of data in EU countries",
        ))

        return metrics

    def _calculate_function_concentration(self) -> List[ConcentrationMetric]:
        """Calculate critical function concentration metrics."""
        metrics = []

        for func, provider_ids in self._providers_by_function.items():
            count = len(provider_ids)
            providers = [
                self._dependencies[pid].provider_name
                for pid in provider_ids
                if pid in self._dependencies
            ]

            # Check if below minimum
            is_warning = count <= self.config.single_function_dependency_warning
            is_critical = count <= self.config.single_function_dependency_critical

            metrics.append(ConcentrationMetric(
                metric_name=f"Function Provider Count: {func}",
                metric_type=ConcentrationType.SERVICE,
                dimension="function_coverage",
                current_value=count,
                threshold_warning=self.config.min_providers_per_critical_function,
                threshold_critical=1,  # Single point of failure
                unit="count",
                status="critical" if count <= 1 else ("warning" if count < 2 else "normal"),
                contributing_factors=[{"providers": providers}],
            ))

        return metrics

    def _calculate_subcontractor_concentration(self) -> List[ConcentrationMetric]:
        """Calculate subcontractor concentration metrics."""
        metrics = []

        # Count subcontractor appearances
        subcontractor_counts: Dict[str, int] = defaultdict(int)
        for dep in self._dependencies.values():
            for sub in dep.known_subcontractors:
                subcontractor_counts[sub] += 1

        if not subcontractor_counts:
            return metrics

        # Top subcontractor
        if subcontractor_counts:
            top_sub = max(subcontractor_counts.items(), key=lambda x: x[1])
            total_providers = len(self._dependencies)
            top_pct = (top_sub[1] / total_providers) * 100 if total_providers > 0 else 0

            if top_pct > 10:  # Only report if significant
                metrics.append(ConcentrationMetric(
                    metric_name=f"Common Subcontractor ({top_sub[0]})",
                    metric_type=ConcentrationType.SUBCONTRACTOR,
                    dimension="shared_subcontractor",
                    current_value=top_pct,
                    threshold_warning=20.0,
                    threshold_critical=40.0,
                    unit="percentage",
                    notes="Percentage of providers using this subcontractor",
                ))

        return metrics

    def get_metric(
        self,
        metric_id: str,
    ) -> Optional[ConcentrationMetric]:
        """Get metric by ID."""
        with self._lock:
            return self._metrics.get(metric_id)

    def get_critical_metrics(self) -> List[ConcentrationMetric]:
        """Get metrics in critical status."""
        with self._lock:
            return [m for m in self._metrics.values() if m.status == "critical"]

    def get_warning_metrics(self) -> List[ConcentrationMetric]:
        """Get metrics in warning status."""
        with self._lock:
            return [m for m in self._metrics.values() if m.status == "warning"]

    # =========================================================================
    # Risk Identification and Assessment
    # =========================================================================

    def identify_concentration_risks(self) -> List[ConcentrationRisk]:
        """
        Identify concentration risks based on current dependencies and metrics.

        Automatically identifies risks from calculated metrics.
        """
        risks = []

        with self._lock:
            # Ensure metrics are current
            self.calculate_concentration_metrics()

            # Identify risks from critical metrics
            for metric in self._metrics.values():
                if metric.status in ["critical", "warning"]:
                    risk = self._create_risk_from_metric(metric)
                    if risk:
                        risks.append(risk)

            # Check for single points of failure
            spof_risks = self._identify_single_points_of_failure()
            risks.extend(spof_risks)

            # Check for geographic concentration
            geo_risks = self._identify_geographic_risks()
            risks.extend(geo_risks)

            # Store risks
            for risk in risks:
                self._risks[risk.risk_id] = risk

        self._log_event("risks_identified", {
            "total_risks": len(risks),
            "critical_risks": sum(1 for r in risks if r.risk_level == RiskLevel.CRITICAL),
        })

        return risks

    def _create_risk_from_metric(
        self,
        metric: ConcentrationMetric,
    ) -> Optional[ConcentrationRisk]:
        """Create a risk from a metric."""
        if metric.status == "normal":
            return None

        likelihood = 4 if metric.status == "critical" else 3
        impact = 4 if metric.status == "critical" else 3

        risk = ConcentrationRisk(
            risk_name=f"Concentration Risk: {metric.metric_name}",
            risk_type=metric.metric_type,
            description=f"Concentration metric '{metric.metric_name}' is in {metric.status} status. "
                       f"Current value: {metric.current_value:.1f} {metric.unit}",
            likelihood=likelihood,
            impact=impact,
            concentration_factor=metric.current_value,
            related_metrics=[metric.metric_id],
        )

        return risk

    def _identify_single_points_of_failure(self) -> List[ConcentrationRisk]:
        """Identify single points of failure."""
        risks = []

        for func, provider_ids in self._providers_by_function.items():
            if len(provider_ids) == 1:
                provider_id = list(provider_ids)[0]
                provider = self._dependencies.get(provider_id)

                risk = ConcentrationRisk(
                    risk_name=f"Single Point of Failure: {func}",
                    risk_type=ConcentrationType.SERVICE,
                    description=f"Critical function '{func}' depends on single provider "
                               f"'{provider.provider_name if provider else provider_id}'",
                    likelihood=4,
                    impact=5,
                    affected_providers=[provider_id],
                    affected_functions=[func],
                    business_impact="Service unavailability if provider fails",
                    regulatory_impact="Potential DORA compliance breach",
                )
                risks.append(risk)

        return risks

    def _identify_geographic_risks(self) -> List[ConcentrationRisk]:
        """Identify geographic concentration risks."""
        risks = []

        # Count data by country
        country_data: Dict[str, List[str]] = defaultdict(list)
        for dep in self._dependencies.values():
            for country in set(dep.data_processing_countries + dep.data_storage_countries):
                country_data[country].append(dep.provider_id)

        total_entries = sum(len(v) for v in country_data.values())
        if total_entries == 0:
            return risks

        for country, providers in country_data.items():
            pct = (len(providers) / total_entries) * 100
            if pct > self.config.geographic_concentration_critical_pct:
                risk = ConcentrationRisk(
                    risk_name=f"Geographic Concentration: {country}",
                    risk_type=ConcentrationType.GEOGRAPHIC,
                    description=f"{pct:.1f}% of data processing/storage in {country}",
                    likelihood=3,
                    impact=4,
                    affected_providers=providers,
                    affected_countries=[country],
                    concentration_factor=pct,
                    business_impact="Regional disruption affects majority of services",
                )
                risks.append(risk)

        return risks

    def add_concentration_risk(
        self,
        risk_name: str,
        risk_type: ConcentrationType,
        description: str,
        likelihood: int = 3,
        impact: int = 3,
        affected_providers: Optional[List[str]] = None,
        affected_functions: Optional[List[str]] = None,
        mitigation_owner: str = "",
    ) -> ConcentrationRisk:
        """Manually add a concentration risk."""
        risk = ConcentrationRisk(
            risk_name=risk_name,
            risk_type=risk_type,
            description=description,
            likelihood=likelihood,
            impact=impact,
            affected_providers=affected_providers or [],
            affected_functions=affected_functions or [],
            mitigation_owner=mitigation_owner,
        )

        with self._lock:
            self._risks[risk.risk_id] = risk

        return risk

    def get_risk(
        self,
        risk_id: str,
    ) -> Optional[ConcentrationRisk]:
        """Get risk by ID."""
        with self._lock:
            return self._risks.get(risk_id)

    def get_all_risks(self) -> List[ConcentrationRisk]:
        """Get all concentration risks."""
        with self._lock:
            return list(self._risks.values())

    def get_risks_by_type(
        self,
        risk_type: ConcentrationType,
    ) -> List[ConcentrationRisk]:
        """Get risks by type."""
        with self._lock:
            return [r for r in self._risks.values() if r.risk_type == risk_type]

    def get_risks_by_level(
        self,
        level: RiskLevel,
    ) -> List[ConcentrationRisk]:
        """Get risks by level."""
        with self._lock:
            return [r for r in self._risks.values() if r.risk_level == level]

    # =========================================================================
    # Mitigation Management
    # =========================================================================

    def add_mitigation_measure(
        self,
        risk_id: str,
        measure_name: str,
        measure_type: str,
        description: str = "",
        actions: Optional[List[Dict[str, Any]]] = None,
        owner: str = "",
        planned_start: Optional[str] = None,
        planned_end: Optional[str] = None,
        estimated_cost_eur: float = 0.0,
    ) -> Optional[MitigationMeasure]:
        """Add a mitigation measure for a concentration risk."""
        with self._lock:
            if risk_id not in self._risks:
                return None

            measure = MitigationMeasure(
                risk_id=risk_id,
                measure_name=measure_name,
                measure_type=measure_type,
                description=description,
                actions=actions or [],
                owner=owner,
                planned_start=planned_start,
                planned_end=planned_end,
                estimated_cost_eur=estimated_cost_eur,
            )

            self._mitigations[measure.measure_id] = measure
            self._risks[risk_id].mitigation_measures.append(measure.measure_id)

        return measure

    def update_mitigation_status(
        self,
        measure_id: str,
        status: MitigationStatus,
        progress_pct: float = 0.0,
        actual_cost_eur: Optional[float] = None,
    ) -> Optional[MitigationMeasure]:
        """Update mitigation measure status."""
        with self._lock:
            if measure_id not in self._mitigations:
                return None

            measure = self._mitigations[measure_id]
            measure.status = status
            measure.progress_pct = progress_pct

            if status == MitigationStatus.IN_PROGRESS and not measure.actual_start:
                measure.actual_start = datetime.now(timezone.utc).isoformat()

            if status in [MitigationStatus.IMPLEMENTED, MitigationStatus.VERIFIED]:
                measure.actual_end = datetime.now(timezone.utc).isoformat()
                measure.progress_pct = 100.0

            if actual_cost_eur is not None:
                measure.actual_cost_eur = actual_cost_eur

        return measure

    def get_mitigations_for_risk(
        self,
        risk_id: str,
    ) -> List[MitigationMeasure]:
        """Get mitigation measures for a risk."""
        with self._lock:
            risk = self._risks.get(risk_id)
            if not risk:
                return []
            return [
                self._mitigations[mid]
                for mid in risk.mitigation_measures
                if mid in self._mitigations
            ]

    def suggest_mitigation_measures(
        self,
        risk_id: str,
    ) -> List[Dict[str, Any]]:
        """Suggest mitigation measures for a risk."""
        with self._lock:
            risk = self._risks.get(risk_id)
            if not risk:
                return []

            suggestions = []

            if risk.risk_type == ConcentrationType.PROVIDER:
                suggestions.extend([
                    {
                        "type": "diversify",
                        "name": "Provider Diversification",
                        "description": "Identify and onboard additional providers",
                        "expected_effort": "high",
                    },
                    {
                        "type": "exit_plan",
                        "name": "Exit Strategy Enhancement",
                        "description": "Develop detailed exit plan for concentrated provider",
                        "expected_effort": "medium",
                    },
                ])

            if risk.risk_type == ConcentrationType.GEOGRAPHIC:
                suggestions.extend([
                    {
                        "type": "diversify",
                        "name": "Geographic Diversification",
                        "description": "Distribute data across multiple regions",
                        "expected_effort": "high",
                    },
                    {
                        "type": "monitoring",
                        "name": "Regional Monitoring",
                        "description": "Implement monitoring for geographic region issues",
                        "expected_effort": "low",
                    },
                ])

            if risk.risk_type == ConcentrationType.SERVICE:
                suggestions.extend([
                    {
                        "type": "alternative",
                        "name": "Alternative Provider Setup",
                        "description": "Configure standby with alternative provider",
                        "expected_effort": "high",
                    },
                    {
                        "type": "monitoring",
                        "name": "Enhanced SLA Monitoring",
                        "description": "Implement strict SLA monitoring",
                        "expected_effort": "low",
                    },
                ])

            # Generic suggestions
            suggestions.append({
                "type": "acceptance",
                "name": "Risk Acceptance (with controls)",
                "description": "Accept risk with enhanced monitoring and contingency",
                "expected_effort": "low",
            })

            return suggestions

    # =========================================================================
    # Concentration Assessment
    # =========================================================================

    def perform_concentration_assessment(
        self,
        scope: AssessmentScope = AssessmentScope.ENTITY,
        assessed_by: str = "",
        entities_in_scope: Optional[List[str]] = None,
    ) -> ConcentrationAssessment:
        """
        Perform comprehensive concentration risk assessment.

        Args:
            scope: Assessment scope level
            assessed_by: Assessor name
            entities_in_scope: Entities in scope (for consolidated)

        Returns:
            ConcentrationAssessment
        """
        with self._lock:
            # Calculate metrics
            metrics = self.calculate_concentration_metrics()

            # Identify risks
            risks = self.identify_concentration_risks()

            # Create assessment
            assessment = ConcentrationAssessment(
                assessed_by=assessed_by,
                scope=scope,
                entities_in_scope=entities_in_scope or [],
                metrics=[m.metric_id for m in metrics],
                risks_identified=[r.risk_id for r in risks],
                new_risks_count=len(risks),
            )

            # Summarize by type
            for r in risks:
                type_key = r.risk_type.value
                assessment.risks_by_type[type_key] = assessment.risks_by_type.get(type_key, 0) + 1

                level_key = r.risk_level.value
                assessment.risks_by_level[level_key] = assessment.risks_by_level.get(level_key, 0) + 1

            # Determine overall level
            if assessment.risks_by_level.get("critical", 0) > 0:
                assessment.overall_concentration_level = RiskLevel.CRITICAL
            elif assessment.risks_by_level.get("high", 0) > 0:
                assessment.overall_concentration_level = RiskLevel.HIGH
            elif assessment.risks_by_level.get("medium", 0) > 0:
                assessment.overall_concentration_level = RiskLevel.MEDIUM
            else:
                assessment.overall_concentration_level = RiskLevel.LOW

            # Key concentration areas
            critical_metrics = self.get_critical_metrics()
            assessment.key_concentration_areas = [
                m.metric_name for m in critical_metrics
            ]

            # Generate recommendations
            assessment.recommendations = self._generate_recommendations(risks, metrics)

            # Store assessment
            self._assessments[assessment.assessment_id] = assessment

        self._log_event("assessment_completed", {
            "assessment_id": assessment.assessment_id,
            "scope": scope.value,
            "risks_count": len(risks),
            "overall_level": assessment.overall_concentration_level.value,
        })

        return assessment

    def _generate_recommendations(
        self,
        risks: List[ConcentrationRisk],
        metrics: List[ConcentrationMetric],
    ) -> List[Dict[str, Any]]:
        """Generate assessment recommendations."""
        recommendations = []

        # Check for critical risks
        critical_risks = [r for r in risks if r.risk_level == RiskLevel.CRITICAL]
        if critical_risks:
            recommendations.append({
                "priority": "critical",
                "recommendation": f"Address {len(critical_risks)} critical concentration risks immediately",
                "risks": [r.risk_name for r in critical_risks],
            })

        # Check for single points of failure
        spof_risks = [r for r in risks if "Single Point of Failure" in r.risk_name]
        if spof_risks:
            recommendations.append({
                "priority": "high",
                "recommendation": "Eliminate single points of failure for critical functions",
                "affected_functions": [r.affected_functions for r in spof_risks],
            })

        # Check for geographic concentration
        geo_risks = [r for r in risks if r.risk_type == ConcentrationType.GEOGRAPHIC]
        if geo_risks:
            recommendations.append({
                "priority": "medium",
                "recommendation": "Review and diversify geographic distribution of data",
                "countries": list(set(c for r in geo_risks for c in r.affected_countries)),
            })

        # Check for limited alternatives
        non_substitutable = [
            d for d in self._dependencies.values()
            if d.substitutability in [
                SubstitutabilityLevel.DIFFICULT_TO_SUBSTITUTE,
                SubstitutabilityLevel.NOT_SUBSTITUTABLE
            ]
        ]
        if non_substitutable:
            recommendations.append({
                "priority": "medium",
                "recommendation": "Develop alternatives for non-substitutable providers",
                "providers": [d.provider_name for d in non_substitutable],
            })

        return recommendations

    def get_assessment(
        self,
        assessment_id: str,
    ) -> Optional[ConcentrationAssessment]:
        """Get assessment by ID."""
        with self._lock:
            return self._assessments.get(assessment_id)

    def get_latest_assessment(self) -> Optional[ConcentrationAssessment]:
        """Get most recent assessment."""
        with self._lock:
            if not self._assessments:
                return None
            return max(
                self._assessments.values(),
                key=lambda a: a.assessment_date
            )

    # =========================================================================
    # Dependency Mapping
    # =========================================================================

    def generate_dependency_map(self) -> DependencyMap:
        """Generate comprehensive dependency map."""
        with self._lock:
            dep_map = DependencyMap()
            dep_map.providers = dict(self._dependencies)
            dep_map.total_providers = len(self._dependencies)

            # Geographic distribution
            for country, provider_ids in self._providers_by_country.items():
                dep_map.countries[country] = list(provider_ids)

            # Function mapping
            for func, provider_ids in self._providers_by_function.items():
                dep_map.functions_to_providers[func] = list(provider_ids)
                dep_map.function_coverage[func] = len(provider_ids)

            # Calculate HHI
            dep_map.provider_hhi = self._calculate_provider_hhi()

            # Geographic HHI
            if dep_map.countries:
                total = sum(len(v) for v in dep_map.countries.values())
                if total > 0:
                    dep_map.geographic_hhi = sum(
                        ((len(v) / total) * 100) ** 2
                        for v in dep_map.countries.values()
                    )

            return dep_map

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_concentration_status(self) -> Dict[str, Any]:
        """Get comprehensive concentration risk status."""
        with self._lock:
            all_risks = list(self._risks.values())
            all_metrics = list(self._metrics.values())

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dependencies": {
                    "total_providers": len(self._dependencies),
                    "ctpp_providers": sum(
                        1 for d in self._dependencies.values() if d.is_ctpp
                    ),
                    "critical_function_providers": sum(
                        1 for d in self._dependencies.values()
                        if d.critical_services_count > 0
                    ),
                },
                "metrics": {
                    "total": len(all_metrics),
                    "critical": len(self.get_critical_metrics()),
                    "warning": len(self.get_warning_metrics()),
                    "normal": sum(1 for m in all_metrics if m.status == "normal"),
                },
                "risks": {
                    "total": len(all_risks),
                    "by_level": {
                        level.value: sum(1 for r in all_risks if r.risk_level == level)
                        for level in RiskLevel
                    },
                    "by_type": {
                        ct.value: sum(1 for r in all_risks if r.risk_type == ct)
                        for ct in ConcentrationType
                        if sum(1 for r in all_risks if r.risk_type == ct) > 0
                    },
                },
                "mitigations": {
                    "total": len(self._mitigations),
                    "in_progress": sum(
                        1 for m in self._mitigations.values()
                        if m.status == MitigationStatus.IN_PROGRESS
                    ),
                    "implemented": sum(
                        1 for m in self._mitigations.values()
                        if m.status in [MitigationStatus.IMPLEMENTED, MitigationStatus.VERIFIED]
                    ),
                },
                "compliance_indicators": {
                    "no_critical_concentration": len(self.get_risks_by_level(RiskLevel.CRITICAL)) == 0,
                    "no_single_points_of_failure": all(
                        len(pids) > 1
                        for pids in self._providers_by_function.values()
                    ),
                    "all_risks_have_mitigation": all(
                        len(r.mitigation_measures) > 0
                        for r in all_risks
                        if r.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
                    ),
                },
            }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _get_eu_countries(self) -> Set[str]:
        """Get set of EU country codes."""
        return {
            "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
            "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
            "PL", "PT", "RO", "SK", "SI", "ES", "SE"
        }

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

        log_file = self._log_path / f"concentration_events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_concentration_risk(
    config: Optional[ConcentrationRiskConfig] = None,
) -> DORAConcentrationRisk:
    """
    Create a DORAConcentrationRisk instance.

    Args:
        config: Optional configuration

    Returns:
        Configured DORAConcentrationRisk instance
    """
    return DORAConcentrationRisk(config=config)


def get_concentration_types() -> List[ConcentrationType]:
    """Get all concentration types."""
    return list(ConcentrationType)


def get_substitutability_levels() -> List[SubstitutabilityLevel]:
    """Get all substitutability levels."""
    return list(SubstitutabilityLevel)
