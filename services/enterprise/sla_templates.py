# -*- coding: utf-8 -*-
"""
Enterprise SLA Templates Service.

DORA Phase 3 Block 3.7: Enterprise SLA templates

Provides enterprise-grade SLA template management:
- Pre-defined SLA templates per DORA requirements
- Customizable service level metrics
- Penalty and credit calculations
- SLA compliance tracking

DORA References:
    - Art. 30(2)(e): Service level descriptions with targets
    - Art. 30(3)(a): Service levels for critical functions
    - Art. 30(2)(f): Incident assistance obligations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


# =============================================================================
# Enums
# =============================================================================


class SLACategory(Enum):
    """SLA categories."""

    AVAILABILITY = "availability"
    PERFORMANCE = "performance"
    SUPPORT = "support"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    DATA = "data"


class SLAMetricType(Enum):
    """SLA metric types."""

    PERCENTAGE = "percentage"  # e.g., uptime %
    TIME = "time"  # e.g., response time in ms
    COUNT = "count"  # e.g., incidents per month
    DURATION = "duration"  # e.g., hours to resolve


class PenaltyType(Enum):
    """SLA penalty types."""

    SERVICE_CREDIT = "service_credit"  # % of monthly fee
    FIXED_AMOUNT = "fixed_amount"  # Fixed monetary amount
    EXTENDED_SERVICE = "extended_service"  # Additional free service
    PRIORITY_SUPPORT = "priority_support"  # Enhanced support


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SLAMetric:
    """SLA metric definition."""

    metric_id: str
    name: str
    description: str
    category: SLACategory
    metric_type: SLAMetricType
    unit: str
    measurement_method: str
    measurement_frequency: str  # hourly, daily, monthly
    exclusions: list[str] = field(default_factory=list)


@dataclass
class SLATarget:
    """SLA target value."""

    target_id: str
    metric_id: str
    tier: str  # standard, professional, enterprise
    target_value: float
    minimum_value: float  # Threshold for penalty
    stretch_value: float | None = None  # For bonus credits
    notes: str = ""


@dataclass
class SLAPenalty:
    """SLA penalty definition."""

    penalty_id: str
    target_id: str
    threshold: float  # Value at which penalty applies
    penalty_type: PenaltyType
    penalty_value: float  # Amount or percentage
    cap_percentage: float = 100.0  # Maximum penalty as % of monthly fee
    description: str = ""


@dataclass
class SLAViolation:
    """SLA violation record."""

    violation_id: str
    sla_id: str
    metric_id: str
    target_id: str
    period_start: datetime
    period_end: datetime
    target_value: float
    actual_value: float
    shortfall: float
    penalty_applicable: bool
    penalty_amount: float
    credited: bool = False
    credited_at: datetime | None = None
    notes: str = ""


@dataclass
class EnterpriseSLA:
    """Enterprise SLA definition."""

    sla_id: str
    name: str
    description: str
    version: str
    tier: str  # standard, professional, enterprise
    effective_from: datetime
    effective_until: datetime | None = None
    metrics: list[SLAMetric] = field(default_factory=list)
    targets: list[SLATarget] = field(default_factory=list)
    penalties: list[SLAPenalty] = field(default_factory=list)
    violations: list[SLAViolation] = field(default_factory=list)
    dora_articles: list[str] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""

    def get_target_for_metric(self, metric_id: str) -> SLATarget | None:
        """Get target for a specific metric."""
        for target in self.targets:
            if target.metric_id == metric_id:
                return target
        return None

    def get_penalties_for_target(self, target_id: str) -> list[SLAPenalty]:
        """Get penalties for a specific target."""
        return [p for p in self.penalties if p.target_id == target_id]


@dataclass
class SLATemplateConfig:
    """SLA template service configuration."""

    default_measurement_frequency: str = "monthly"
    max_penalty_cap_percent: float = 100.0
    auto_calculate_penalties: bool = True
    retention_months: int = 84  # 7 years for DORA


# =============================================================================
# Default SLA Templates
# =============================================================================


def get_default_sla_templates() -> list[dict[str, Any]]:
    """Get default enterprise SLA templates per DORA requirements."""
    return [
        # Standard Tier
        {
            "name": "Standard Service Level Agreement",
            "tier": "standard",
            "description": "Standard SLA for basic service tier",
            "dora_articles": ["Art. 30(2)(e)"],
            "metrics": [
                {
                    "name": "Monthly Uptime",
                    "category": SLACategory.AVAILABILITY,
                    "metric_type": SLAMetricType.PERCENTAGE,
                    "unit": "percent",
                    "measurement_method": "Automated monitoring",
                    "target_value": 99.5,
                    "minimum_value": 99.0,
                },
                {
                    "name": "API Response Time (P95)",
                    "category": SLACategory.PERFORMANCE,
                    "metric_type": SLAMetricType.TIME,
                    "unit": "milliseconds",
                    "measurement_method": "APM monitoring",
                    "target_value": 500,
                    "minimum_value": 1000,
                },
                {
                    "name": "Support Response Time",
                    "category": SLACategory.SUPPORT,
                    "metric_type": SLAMetricType.DURATION,
                    "unit": "hours",
                    "measurement_method": "Ticket system",
                    "target_value": 24,
                    "minimum_value": 48,
                },
            ],
        },
        # Professional Tier
        {
            "name": "Professional Service Level Agreement",
            "tier": "professional",
            "description": "Professional SLA with enhanced commitments",
            "dora_articles": ["Art. 30(2)(e)", "Art. 30(2)(f)"],
            "metrics": [
                {
                    "name": "Monthly Uptime",
                    "category": SLACategory.AVAILABILITY,
                    "metric_type": SLAMetricType.PERCENTAGE,
                    "unit": "percent",
                    "measurement_method": "Automated monitoring",
                    "target_value": 99.9,
                    "minimum_value": 99.5,
                },
                {
                    "name": "API Response Time (P95)",
                    "category": SLACategory.PERFORMANCE,
                    "metric_type": SLAMetricType.TIME,
                    "unit": "milliseconds",
                    "measurement_method": "APM monitoring",
                    "target_value": 200,
                    "minimum_value": 500,
                },
                {
                    "name": "Critical Incident Response",
                    "category": SLACategory.SUPPORT,
                    "metric_type": SLAMetricType.DURATION,
                    "unit": "minutes",
                    "measurement_method": "Incident tracking",
                    "target_value": 30,
                    "minimum_value": 60,
                },
                {
                    "name": "Incident Notification Time",
                    "category": SLACategory.SUPPORT,
                    "metric_type": SLAMetricType.DURATION,
                    "unit": "minutes",
                    "measurement_method": "Notification logs",
                    "target_value": 30,
                    "minimum_value": 60,
                },
            ],
        },
        # Enterprise Tier
        {
            "name": "Enterprise Service Level Agreement",
            "tier": "enterprise",
            "description": "Enterprise SLA with maximum commitments per DORA Art. 30(3)(a)",
            "dora_articles": ["Art. 30(2)(e)", "Art. 30(2)(f)", "Art. 30(3)(a)"],
            "metrics": [
                {
                    "name": "Monthly Uptime",
                    "category": SLACategory.AVAILABILITY,
                    "metric_type": SLAMetricType.PERCENTAGE,
                    "unit": "percent",
                    "measurement_method": "Automated monitoring with multi-region verification",
                    "target_value": 99.99,
                    "minimum_value": 99.9,
                },
                {
                    "name": "API Response Time (P95)",
                    "category": SLACategory.PERFORMANCE,
                    "metric_type": SLAMetricType.TIME,
                    "unit": "milliseconds",
                    "measurement_method": "APM monitoring",
                    "target_value": 100,
                    "minimum_value": 200,
                },
                {
                    "name": "API Response Time (P99)",
                    "category": SLACategory.PERFORMANCE,
                    "metric_type": SLAMetricType.TIME,
                    "unit": "milliseconds",
                    "measurement_method": "APM monitoring",
                    "target_value": 200,
                    "minimum_value": 500,
                },
                {
                    "name": "Critical Incident Response",
                    "category": SLACategory.SUPPORT,
                    "metric_type": SLAMetricType.DURATION,
                    "unit": "minutes",
                    "measurement_method": "Incident tracking",
                    "target_value": 15,
                    "minimum_value": 30,
                },
                {
                    "name": "Incident Notification Time",
                    "category": SLACategory.SUPPORT,
                    "metric_type": SLAMetricType.DURATION,
                    "unit": "minutes",
                    "measurement_method": "Notification logs",
                    "target_value": 15,
                    "minimum_value": 30,
                },
                {
                    "name": "RTO (Recovery Time Objective)",
                    "category": SLACategory.AVAILABILITY,
                    "metric_type": SLAMetricType.DURATION,
                    "unit": "hours",
                    "measurement_method": "DR test results",
                    "target_value": 4,
                    "minimum_value": 8,
                },
                {
                    "name": "RPO (Recovery Point Objective)",
                    "category": SLACategory.DATA,
                    "metric_type": SLAMetricType.DURATION,
                    "unit": "hours",
                    "measurement_method": "Backup verification",
                    "target_value": 1,
                    "minimum_value": 4,
                },
                {
                    "name": "Security Incident Response",
                    "category": SLACategory.SECURITY,
                    "metric_type": SLAMetricType.DURATION,
                    "unit": "minutes",
                    "measurement_method": "Security incident tracking",
                    "target_value": 15,
                    "minimum_value": 30,
                },
            ],
        },
    ]


# =============================================================================
# Main Service Class
# =============================================================================


class EnterpriseSLAService:
    """
    Enterprise SLA Templates Service.

    Manages enterprise SLA templates per DORA Art. 30(2)(e) and Art. 30(3)(a).
    """

    def __init__(self, config: SLATemplateConfig | None = None) -> None:
        """Initialize enterprise SLA service."""
        self.config = config or SLATemplateConfig()
        self._slas: dict[str, EnterpriseSLA] = {}
        self._initialize_templates()

    def _initialize_templates(self) -> None:
        """Initialize default SLA templates."""
        for template in get_default_sla_templates():
            sla = self._create_sla_from_template(template)
            self._slas[sla.sla_id] = sla

    def _create_sla_from_template(self, template: dict[str, Any]) -> EnterpriseSLA:
        """Create SLA from template definition."""
        sla = EnterpriseSLA(
            sla_id=str(uuid4()),
            name=template["name"],
            description=template["description"],
            version="1.0",
            tier=template["tier"],
            effective_from=datetime.utcnow(),
            dora_articles=template.get("dora_articles", []),
        )

        # Create metrics and targets
        for metric_def in template.get("metrics", []):
            metric = SLAMetric(
                metric_id=str(uuid4()),
                name=metric_def["name"],
                description=metric_def.get("description", ""),
                category=metric_def["category"],
                metric_type=metric_def["metric_type"],
                unit=metric_def["unit"],
                measurement_method=metric_def["measurement_method"],
                measurement_frequency=self.config.default_measurement_frequency,
            )
            sla.metrics.append(metric)

            target = SLATarget(
                target_id=str(uuid4()),
                metric_id=metric.metric_id,
                tier=template["tier"],
                target_value=metric_def["target_value"],
                minimum_value=metric_def["minimum_value"],
            )
            sla.targets.append(target)

            # Create penalty for this target
            penalty = SLAPenalty(
                penalty_id=str(uuid4()),
                target_id=target.target_id,
                threshold=target.minimum_value,
                penalty_type=PenaltyType.SERVICE_CREDIT,
                penalty_value=10.0,  # 10% service credit
                cap_percentage=self.config.max_penalty_cap_percent,
                description=f"Service credit for {metric.name} breach",
            )
            sla.penalties.append(penalty)

        return sla

    # =========================================================================
    # SLA Management
    # =========================================================================

    def get_sla(self, sla_id: str) -> EnterpriseSLA | None:
        """Get SLA by ID."""
        return self._slas.get(sla_id)

    def get_sla_by_tier(self, tier: str) -> EnterpriseSLA | None:
        """Get SLA by tier."""
        for sla in self._slas.values():
            if sla.tier == tier and sla.is_active:
                return sla
        return None

    def list_slas(self, tier: str | None = None, active_only: bool = True) -> list[EnterpriseSLA]:
        """List SLAs with optional filters."""
        slas = list(self._slas.values())

        if tier:
            slas = [s for s in slas if s.tier == tier]
        if active_only:
            slas = [s for s in slas if s.is_active]

        return slas

    def create_custom_sla(
        self,
        name: str,
        description: str,
        tier: str,
        metrics: list[dict[str, Any]],
        created_by: str,
    ) -> EnterpriseSLA:
        """Create a custom SLA."""
        template = {
            "name": name,
            "description": description,
            "tier": tier,
            "metrics": metrics,
            "dora_articles": ["Art. 30(2)(e)"],
        }
        sla = self._create_sla_from_template(template)
        sla.created_by = created_by
        self._slas[sla.sla_id] = sla
        return sla

    # =========================================================================
    # Metrics and Targets
    # =========================================================================

    def add_metric(
        self,
        sla_id: str,
        name: str,
        category: SLACategory,
        metric_type: SLAMetricType,
        unit: str,
        measurement_method: str,
        target_value: float,
        minimum_value: float,
    ) -> SLAMetric:
        """Add metric to an SLA."""
        sla = self._slas.get(sla_id)
        if not sla:
            raise ValueError(f"SLA not found: {sla_id}")

        metric = SLAMetric(
            metric_id=str(uuid4()),
            name=name,
            description="",
            category=category,
            metric_type=metric_type,
            unit=unit,
            measurement_method=measurement_method,
            measurement_frequency=self.config.default_measurement_frequency,
        )
        sla.metrics.append(metric)

        target = SLATarget(
            target_id=str(uuid4()),
            metric_id=metric.metric_id,
            tier=sla.tier,
            target_value=target_value,
            minimum_value=minimum_value,
        )
        sla.targets.append(target)

        return metric

    def update_target(
        self,
        sla_id: str,
        target_id: str,
        target_value: float | None = None,
        minimum_value: float | None = None,
    ) -> SLATarget | None:
        """Update target values."""
        sla = self._slas.get(sla_id)
        if not sla:
            return None

        for target in sla.targets:
            if target.target_id == target_id:
                if target_value is not None:
                    target.target_value = target_value
                if minimum_value is not None:
                    target.minimum_value = minimum_value
                return target

        return None

    # =========================================================================
    # Violation Tracking
    # =========================================================================

    def record_violation(
        self,
        sla_id: str,
        metric_id: str,
        period_start: datetime,
        period_end: datetime,
        actual_value: float,
    ) -> SLAViolation | None:
        """Record an SLA violation."""
        sla = self._slas.get(sla_id)
        if not sla:
            return None

        target = sla.get_target_for_metric(metric_id)
        if not target:
            return None

        # Determine if this is a violation
        metric = next((m for m in sla.metrics if m.metric_id == metric_id), None)
        if not metric:
            return None

        is_violation = False
        shortfall = 0.0

        # For percentage metrics (higher is better)
        if metric.metric_type == SLAMetricType.PERCENTAGE:
            if actual_value < target.minimum_value:
                is_violation = True
                shortfall = target.minimum_value - actual_value

        # For time/duration metrics (lower is better)
        elif metric.metric_type in (SLAMetricType.TIME, SLAMetricType.DURATION):
            if actual_value > target.minimum_value:
                is_violation = True
                shortfall = actual_value - target.minimum_value

        if not is_violation:
            return None

        # Calculate penalty
        penalty_amount = 0.0
        penalties = sla.get_penalties_for_target(target.target_id)
        for penalty in penalties:
            if penalty.penalty_type == PenaltyType.SERVICE_CREDIT:
                penalty_amount = penalty.penalty_value  # Percentage

        violation = SLAViolation(
            violation_id=str(uuid4()),
            sla_id=sla_id,
            metric_id=metric_id,
            target_id=target.target_id,
            period_start=period_start,
            period_end=period_end,
            target_value=target.target_value,
            actual_value=actual_value,
            shortfall=shortfall,
            penalty_applicable=True,
            penalty_amount=penalty_amount,
        )
        sla.violations.append(violation)
        return violation

    def get_violations(
        self,
        sla_id: str,
        credited_only: bool | None = None,
    ) -> list[SLAViolation]:
        """Get violations for an SLA."""
        sla = self._slas.get(sla_id)
        if not sla:
            return []

        violations = sla.violations
        if credited_only is not None:
            violations = [v for v in violations if v.credited == credited_only]

        return violations

    def credit_violation(self, sla_id: str, violation_id: str) -> bool:
        """Mark violation as credited."""
        sla = self._slas.get(sla_id)
        if not sla:
            return False

        for violation in sla.violations:
            if violation.violation_id == violation_id:
                violation.credited = True
                violation.credited_at = datetime.utcnow()
                return True

        return False

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_sla_summary(self, sla_id: str) -> dict[str, Any]:
        """Get summary of an SLA."""
        sla = self._slas.get(sla_id)
        if not sla:
            return {}

        total_violations = len(sla.violations)
        credited_violations = sum(1 for v in sla.violations if v.credited)
        total_penalty_percent = sum(v.penalty_amount for v in sla.violations if not v.credited)

        return {
            "sla_id": sla.sla_id,
            "name": sla.name,
            "tier": sla.tier,
            "version": sla.version,
            "is_active": sla.is_active,
            "metrics_count": len(sla.metrics),
            "targets_count": len(sla.targets),
            "dora_articles": sla.dora_articles,
            "violations": {
                "total": total_violations,
                "credited": credited_violations,
                "pending_credit": total_violations - credited_violations,
                "total_penalty_percent": total_penalty_percent,
            },
            "effective_from": sla.effective_from.isoformat(),
        }

    def get_all_templates_summary(self) -> list[dict[str, Any]]:
        """Get summary of all SLA templates."""
        return [self.get_sla_summary(sla_id) for sla_id in self._slas.keys()]


# =============================================================================
# Factory Functions
# =============================================================================


def create_enterprise_sla(
    max_penalty_cap_percent: float = 100.0,
    auto_calculate_penalties: bool = True,
    **kwargs: Any,
) -> EnterpriseSLAService:
    """Create enterprise SLA service instance."""
    config = SLATemplateConfig(
        max_penalty_cap_percent=max_penalty_cap_percent,
        auto_calculate_penalties=auto_calculate_penalties,
        **kwargs,
    )
    return EnterpriseSLAService(config)


def get_enterprise_sla_templates() -> list[dict[str, Any]]:
    """Get default enterprise SLA templates."""
    return get_default_sla_templates()
