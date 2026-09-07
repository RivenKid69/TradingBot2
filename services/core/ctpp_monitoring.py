# -*- coding: utf-8 -*-
"""
CTPP Risk Monitoring (Block 2.14).

Implements automated Critical Third-Party Provider risk monitoring:
- CTPP designation tracking
- Risk indicator monitoring
- Concentration risk assessment
- Regulatory update alerts

DORA References:
    - Article 31-44: CTPP Oversight Framework
    - Article 29: ICT Concentration Risk
    - RTS CDR 2024/1774: Third-party risk monitoring
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class CTPPRiskLevel(Enum):
    """CTPP risk levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MonitoringStatus(Enum):
    """Monitoring status."""

    ACTIVE = "active"
    PAUSED = "paused"
    PENDING_REVIEW = "pending_review"
    INACTIVE = "inactive"


class RiskIndicator(Enum):
    """Risk indicator types."""

    DESIGNATION_STATUS = "designation_status"
    CONCENTRATION_LEVEL = "concentration_level"
    SERVICE_DEPENDENCY = "service_dependency"
    GEOGRAPHIC_CONCENTRATION = "geographic_concentration"
    REGULATORY_ACTION = "regulatory_action"
    INCIDENT_FREQUENCY = "incident_frequency"
    SLA_PERFORMANCE = "sla_performance"
    FINANCIAL_STABILITY = "financial_stability"


# Known designated CTPPs as of 2025
DESIGNATED_CTPPS = [
    {
        "name": "Microsoft (Azure, M365, LinkedIn)",
        "designation_date": "2024-11-28",
        "lead_overseer": "EBA",
        "services": ["Azure Cloud", "Microsoft 365", "LinkedIn"],
    },
    {
        "name": "Amazon Web Services (AWS)",
        "designation_date": "2024-11-28",
        "lead_overseer": "EBA",
        "services": ["AWS Cloud Services"],
    },
    {
        "name": "Google (GCP)",
        "designation_date": "2024-11-28",
        "lead_overseer": "ESMA",
        "services": ["Google Cloud Platform"],
    },
    {
        "name": "Alibaba Cloud",
        "designation_date": "2025-Expected",
        "lead_overseer": "TBD",
        "services": ["Cloud Services"],
    },
    {
        "name": "IBM (Kyndryl, Promontory)",
        "designation_date": "2025-Expected",
        "lead_overseer": "TBD",
        "services": ["Infrastructure Services", "Consulting"],
    },
    {
        "name": "Salesforce",
        "designation_date": "2025-Expected",
        "lead_overseer": "TBD",
        "services": ["CRM Services"],
    },
    {
        "name": "Oracle",
        "designation_date": "2025-Expected",
        "lead_overseer": "TBD",
        "services": ["Database Services", "Cloud Infrastructure"],
    },
    {
        "name": "ServiceNow",
        "designation_date": "2025-Expected",
        "lead_overseer": "TBD",
        "services": ["IT Service Management"],
    },
]


@dataclass
class CTPPRiskAssessment:
    """CTPP risk assessment."""

    assessment_id: str = ""
    provider_name: str = ""
    provider_id: str = ""

    # Designation
    is_designated_ctpp: bool = False
    designation_date: str = ""
    lead_overseer: str = ""

    # Risk scoring
    overall_risk_level: CTPPRiskLevel = CTPPRiskLevel.MEDIUM
    concentration_risk_score: float = 0.0
    dependency_risk_score: float = 0.0
    operational_risk_score: float = 0.0

    # Services
    services_used: List[str] = field(default_factory=list)
    critical_functions_supported: List[str] = field(default_factory=list)

    # Concentration
    dependency_percentage: float = 0.0  # % of operations depending on this CTPP
    substitutability: str = "medium"  # low, medium, high
    exit_complexity: str = "medium"

    # Monitoring
    monitoring_status: MonitoringStatus = MonitoringStatus.ACTIVE
    last_assessment_date: str = ""
    next_review_date: str = ""

    # Issues
    active_issues: List[Dict[str, Any]] = field(default_factory=list)
    regulatory_actions: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.assessment_id:
            self.assessment_id = f"CTPPA-{uuid.uuid4().hex[:8].upper()}"
        if not self.last_assessment_date:
            self.last_assessment_date = datetime.now(timezone.utc).isoformat()


@dataclass
class RiskMetric:
    """Risk metric measurement."""

    metric_id: str = ""
    provider_id: str = ""
    indicator: RiskIndicator = RiskIndicator.CONCENTRATION_LEVEL
    value: float = 0.0
    threshold_warning: float = 50.0
    threshold_critical: float = 75.0
    status: str = "normal"  # normal, warning, critical
    measured_at: str = ""

    def __post_init__(self):
        if not self.metric_id:
            self.metric_id = f"MTRC-{uuid.uuid4().hex[:8].upper()}"
        if not self.measured_at:
            self.measured_at = datetime.now(timezone.utc).isoformat()

        # Determine status
        if self.value >= self.threshold_critical:
            self.status = "critical"
        elif self.value >= self.threshold_warning:
            self.status = "warning"
        else:
            self.status = "normal"


@dataclass
class RiskAlert:
    """CTPP risk alert."""

    alert_id: str = ""
    provider_name: str = ""
    provider_id: str = ""
    alert_type: str = ""
    severity: CTPPRiskLevel = CTPPRiskLevel.MEDIUM
    title: str = ""
    description: str = ""
    recommended_action: str = ""
    created_at: str = ""
    acknowledged_at: str = ""
    resolved_at: str = ""
    status: str = "open"  # open, acknowledged, resolved

    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = f"CTPALT-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class CTPPMonitoringConfig:
    """Configuration for CTPPRiskMonitor."""

    auto_check_designations: bool = True
    concentration_warning_threshold: float = 30.0
    concentration_critical_threshold: float = 50.0
    assessment_frequency_days: int = 90
    log_all_events: bool = True
    log_path: str = "logs/core/ctpp_monitoring"
    alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


class CTPPRiskMonitor:
    """CTPP Risk Monitor."""

    def __init__(self, config: Optional[CTPPMonitoringConfig] = None):
        self.config = config or CTPPMonitoringConfig()
        self._assessments: Dict[str, CTPPRiskAssessment] = {}
        self._metrics: Dict[str, List[RiskMetric]] = {}
        self._alerts: Dict[str, RiskAlert] = {}
        self._ctpp_registry: List[Dict[str, Any]] = DESIGNATED_CTPPS.copy()
        self._lock = threading.RLock()
        logger.info("CTPPRiskMonitor initialized")

    def register_provider(
        self,
        provider_name: str,
        services_used: List[str],
        critical_functions: Optional[List[str]] = None,
        dependency_percentage: float = 0.0,
    ) -> CTPPRiskAssessment:
        """Register a provider for monitoring."""
        # Check if designated CTPP
        is_ctpp = False
        ctpp_info = None
        for ctpp in self._ctpp_registry:
            if provider_name.lower() in ctpp["name"].lower():
                is_ctpp = True
                ctpp_info = ctpp
                break

        assessment = CTPPRiskAssessment(
            provider_name=provider_name,
            provider_id=f"PROV-{uuid.uuid4().hex[:8].upper()}",
            is_designated_ctpp=is_ctpp,
            designation_date=ctpp_info["designation_date"] if ctpp_info else "",
            lead_overseer=ctpp_info["lead_overseer"] if ctpp_info else "",
            services_used=services_used,
            critical_functions_supported=critical_functions or [],
            dependency_percentage=dependency_percentage,
        )

        # Calculate risk scores
        assessment.concentration_risk_score = min(dependency_percentage * 2, 100)
        assessment.dependency_risk_score = len(critical_functions or []) * 10
        assessment.operational_risk_score = 50 if is_ctpp else 30

        # Determine overall risk
        avg_score = (
            assessment.concentration_risk_score
            + assessment.dependency_risk_score
            + assessment.operational_risk_score
        ) / 3

        if avg_score >= 75:
            assessment.overall_risk_level = CTPPRiskLevel.CRITICAL
        elif avg_score >= 50:
            assessment.overall_risk_level = CTPPRiskLevel.HIGH
        elif avg_score >= 25:
            assessment.overall_risk_level = CTPPRiskLevel.MEDIUM
        else:
            assessment.overall_risk_level = CTPPRiskLevel.LOW

        # Set next review date
        next_review = datetime.now(timezone.utc) + timedelta(
            days=self.config.assessment_frequency_days
        )
        assessment.next_review_date = next_review.isoformat()

        with self._lock:
            self._assessments[assessment.provider_id] = assessment
            self._metrics[assessment.provider_id] = []

        # Generate alerts if needed
        self._check_and_generate_alerts(assessment)

        return assessment

    def record_metric(
        self,
        provider_id: str,
        indicator: RiskIndicator,
        value: float,
    ) -> Optional[RiskMetric]:
        """Record a risk metric."""
        with self._lock:
            if provider_id not in self._assessments:
                return None

            metric = RiskMetric(
                provider_id=provider_id,
                indicator=indicator,
                value=value,
                threshold_warning=self.config.concentration_warning_threshold,
                threshold_critical=self.config.concentration_critical_threshold,
            )

            self._metrics[provider_id].append(metric)

            # Check for alerts
            if metric.status in ("warning", "critical"):
                assessment = self._assessments[provider_id]
                self._create_alert(
                    provider_name=assessment.provider_name,
                    provider_id=provider_id,
                    alert_type=f"{indicator.value}_threshold",
                    severity=(
                        CTPPRiskLevel.CRITICAL
                        if metric.status == "critical"
                        else CTPPRiskLevel.HIGH
                    ),
                    title=f"{indicator.value} threshold exceeded",
                    description=f"Value {value} exceeds {metric.status} threshold",
                    recommended_action="Review concentration risk mitigation measures",
                )

        return metric

    def _check_and_generate_alerts(self, assessment: CTPPRiskAssessment) -> None:
        """Check and generate alerts for an assessment."""
        # CTPP designation alert
        if assessment.is_designated_ctpp:
            self._create_alert(
                provider_name=assessment.provider_name,
                provider_id=assessment.provider_id,
                alert_type="ctpp_designation",
                severity=CTPPRiskLevel.HIGH,
                title=f"Using designated CTPP: {assessment.provider_name}",
                description=f"Provider is designated as CTPP under DORA. Lead overseer: {assessment.lead_overseer}",
                recommended_action="Review CTPP compliance requirements per Articles 31-44",
            )

        # Concentration risk alert
        if assessment.dependency_percentage >= self.config.concentration_critical_threshold:
            self._create_alert(
                provider_name=assessment.provider_name,
                provider_id=assessment.provider_id,
                alert_type="concentration_risk",
                severity=CTPPRiskLevel.CRITICAL,
                title=f"High concentration risk: {assessment.provider_name}",
                description=f"Dependency at {assessment.dependency_percentage}% exceeds critical threshold",
                recommended_action="Develop concentration risk mitigation plan per Article 29",
            )

    def _create_alert(
        self,
        provider_name: str,
        provider_id: str,
        alert_type: str,
        severity: CTPPRiskLevel,
        title: str,
        description: str,
        recommended_action: str,
    ) -> RiskAlert:
        """Create a risk alert."""
        alert = RiskAlert(
            provider_name=provider_name,
            provider_id=provider_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            description=description,
            recommended_action=recommended_action,
        )

        with self._lock:
            self._alerts[alert.alert_id] = alert

        if self.config.alert_callback:
            self.config.alert_callback("ctpp_risk_alert", asdict(alert))

        return alert

    def get_assessment(self, provider_id: str) -> Optional[CTPPRiskAssessment]:
        """Get provider assessment."""
        with self._lock:
            return self._assessments.get(provider_id)

    def get_all_assessments(self) -> List[CTPPRiskAssessment]:
        """Get all assessments."""
        with self._lock:
            return list(self._assessments.values())

    def get_ctpp_providers(self) -> List[CTPPRiskAssessment]:
        """Get designated CTPP providers."""
        with self._lock:
            return [a for a in self._assessments.values() if a.is_designated_ctpp]

    def get_high_risk_providers(self) -> List[CTPPRiskAssessment]:
        """Get high/critical risk providers."""
        with self._lock:
            return [
                a
                for a in self._assessments.values()
                if a.overall_risk_level in (CTPPRiskLevel.HIGH, CTPPRiskLevel.CRITICAL)
            ]

    def get_open_alerts(self) -> List[RiskAlert]:
        """Get open alerts."""
        with self._lock:
            return [a for a in self._alerts.values() if a.status == "open"]

    def acknowledge_alert(self, alert_id: str) -> Optional[RiskAlert]:
        """Acknowledge an alert."""
        with self._lock:
            if alert_id not in self._alerts:
                return None
            alert = self._alerts[alert_id]
            alert.status = "acknowledged"
            alert.acknowledged_at = datetime.now(timezone.utc).isoformat()
        return alert

    def resolve_alert(self, alert_id: str) -> Optional[RiskAlert]:
        """Resolve an alert."""
        with self._lock:
            if alert_id not in self._alerts:
                return None
            alert = self._alerts[alert_id]
            alert.status = "resolved"
            alert.resolved_at = datetime.now(timezone.utc).isoformat()
        return alert

    def check_designation_updates(self) -> List[Dict[str, Any]]:
        """Check for new CTPP designations."""
        updates = []
        for ctpp in self._ctpp_registry:
            if "Expected" not in ctpp.get("designation_date", ""):
                updates.append(
                    {
                        "provider": ctpp["name"],
                        "designation_date": ctpp["designation_date"],
                        "lead_overseer": ctpp["lead_overseer"],
                        "status": "designated",
                    }
                )
        return updates

    def get_concentration_summary(self) -> Dict[str, Any]:
        """Get concentration risk summary."""
        with self._lock:
            assessments = list(self._assessments.values())

        if not assessments:
            return {"status": "no_providers"}

        total_dependency = sum(a.dependency_percentage for a in assessments)
        ctpp_count = sum(1 for a in assessments if a.is_designated_ctpp)
        high_risk_count = sum(
            1
            for a in assessments
            if a.overall_risk_level in (CTPPRiskLevel.HIGH, CTPPRiskLevel.CRITICAL)
        )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_providers": len(assessments),
            "ctpp_providers": ctpp_count,
            "high_risk_providers": high_risk_count,
            "total_dependency_percent": round(total_dependency, 2),
            "concentration_risk": (
                "high" if total_dependency > 150 else "medium" if total_dependency > 100 else "low"
            ),
            "dora_compliance": {
                "article_29": "monitored",
                "articles_31_44": "applicable" if ctpp_count > 0 else "not_applicable",
            },
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get CTPP monitoring summary."""
        concentration = self.get_concentration_summary()
        open_alerts = self.get_open_alerts()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "concentration": concentration,
            "alerts": {
                "open": len(open_alerts),
                "by_severity": {
                    level.value: sum(1 for a in open_alerts if a.severity == level)
                    for level in CTPPRiskLevel
                },
            },
            "ctpp_registry": {
                "designated": len(
                    [
                        c
                        for c in self._ctpp_registry
                        if "Expected" not in c.get("designation_date", "")
                    ]
                ),
                "expected": len(
                    [c for c in self._ctpp_registry if "Expected" in c.get("designation_date", "")]
                ),
            },
        }


def create_ctpp_risk_monitor(
    config: Optional[CTPPMonitoringConfig] = None,
) -> CTPPRiskMonitor:
    """Create a CTPPRiskMonitor instance."""
    return CTPPRiskMonitor(config=config)
