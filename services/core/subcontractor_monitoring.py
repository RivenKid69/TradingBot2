# -*- coding: utf-8 -*-
"""
Subcontractor Status Monitoring (Block 2.11).

Implements subcontractor health and status monitoring:
- Real-time status tracking
- SLA compliance monitoring
- Incident correlation
- Alert integration

DORA References:
    - Article 28: Third-Party ICT Risk
    - Article 30(2)(b): Subcontracting requirements
    - Article 30(3): Subcontractor chain monitoring
    - RTS CDR 2024/1774: Third-party monitoring
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


class SubcontractorHealthStatus(Enum):
    """Subcontractor health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"


class MonitoringFrequency(Enum):
    """Monitoring frequency."""
    REAL_TIME = "real_time"
    MINUTE = "minute"
    FIVE_MINUTES = "five_minutes"
    HOURLY = "hourly"
    DAILY = "daily"


class AlertThreshold(Enum):
    """Alert threshold levels."""
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SubcontractorStatus:
    """Subcontractor status record."""
    subcontractor_id: str = ""
    name: str = ""
    service_type: str = ""

    # Status
    health_status: SubcontractorHealthStatus = SubcontractorHealthStatus.UNKNOWN
    last_check: str = ""
    last_healthy: str = ""

    # Metrics
    uptime_percent_30d: float = 100.0
    response_time_ms: float = 0.0
    error_rate_percent: float = 0.0

    # SLA
    sla_target_uptime: float = 99.9
    sla_compliant: bool = True

    # Contact
    status_page_url: str = ""
    support_contact: str = ""

    # Risk
    risk_level: str = "low"  # low, medium, high, critical
    is_critical_provider: bool = False

    def __post_init__(self):
        if not self.subcontractor_id:
            self.subcontractor_id = f"SUB-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class HealthCheckResult:
    """Health check result."""
    check_id: str = ""
    subcontractor_id: str = ""
    timestamp: str = ""

    # Results
    status: SubcontractorHealthStatus = SubcontractorHealthStatus.UNKNOWN
    response_time_ms: float = 0.0
    success: bool = True

    # Details
    check_type: str = ""  # api, status_page, synthetic
    endpoint_checked: str = ""
    error_message: str = ""

    def __post_init__(self):
        if not self.check_id:
            self.check_id = f"CHK-{uuid.uuid4().hex[:8].upper()}"
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class StatusReport:
    """Subcontractor status report."""
    report_id: str = ""
    generated_at: str = ""
    reporting_period: str = ""

    # Summary
    total_subcontractors: int = 0
    healthy_count: int = 0
    degraded_count: int = 0
    unhealthy_count: int = 0

    # SLA
    sla_compliant_count: int = 0
    sla_breaches: List[Dict[str, Any]] = field(default_factory=list)

    # Incidents
    incidents_count: int = 0
    incidents_by_subcontractor: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.report_id:
            self.report_id = f"SRPT-{uuid.uuid4().hex[:8].upper()}"
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()


@dataclass
class SubcontractorMonitoringConfig:
    """Configuration for SubcontractorMonitor."""
    default_check_frequency: MonitoringFrequency = MonitoringFrequency.FIVE_MINUTES
    warning_threshold_uptime: float = 99.5
    critical_threshold_uptime: float = 99.0
    response_time_warning_ms: float = 1000.0
    response_time_critical_ms: float = 5000.0
    log_all_events: bool = True
    log_path: str = "logs/core/subcontractor_monitoring"
    alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


class SubcontractorMonitor:
    """Subcontractor Status Monitor."""

    def __init__(self, config: Optional[SubcontractorMonitoringConfig] = None):
        self.config = config or SubcontractorMonitoringConfig()
        self._subcontractors: Dict[str, SubcontractorStatus] = {}
        self._health_checks: List[HealthCheckResult] = []
        self._alerts: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        logger.info("SubcontractorMonitor initialized")

    def register_subcontractor(
        self,
        name: str,
        service_type: str,
        status_page_url: str = "",
        sla_target_uptime: float = 99.9,
        is_critical: bool = False,
    ) -> SubcontractorStatus:
        """Register a subcontractor for monitoring."""
        status = SubcontractorStatus(
            name=name,
            service_type=service_type,
            status_page_url=status_page_url,
            sla_target_uptime=sla_target_uptime,
            is_critical_provider=is_critical,
            risk_level="high" if is_critical else "medium",
        )
        with self._lock:
            self._subcontractors[status.subcontractor_id] = status
        return status

    def record_health_check(
        self,
        subcontractor_id: str,
        status: SubcontractorHealthStatus,
        response_time_ms: float = 0.0,
        check_type: str = "api",
        error_message: str = "",
    ) -> HealthCheckResult:
        """Record a health check result."""
        check = HealthCheckResult(
            subcontractor_id=subcontractor_id,
            status=status,
            response_time_ms=response_time_ms,
            success=status in (SubcontractorHealthStatus.HEALTHY, SubcontractorHealthStatus.DEGRADED),
            check_type=check_type,
            error_message=error_message,
        )

        with self._lock:
            self._health_checks.append(check)

            # Update subcontractor status
            if subcontractor_id in self._subcontractors:
                sub = self._subcontractors[subcontractor_id]
                sub.health_status = status
                sub.last_check = check.timestamp
                sub.response_time_ms = response_time_ms

                if status == SubcontractorHealthStatus.HEALTHY:
                    sub.last_healthy = check.timestamp

                # Recalculate uptime
                sub.uptime_percent_30d = self._calculate_uptime(subcontractor_id)
                sub.sla_compliant = sub.uptime_percent_30d >= sub.sla_target_uptime

                # Check for alerts
                self._check_alerts(sub, check)

        return check

    def _calculate_uptime(self, subcontractor_id: str) -> float:
        """Calculate 30-day uptime percentage."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

        recent_checks = [
            c for c in self._health_checks
            if c.subcontractor_id == subcontractor_id and c.timestamp > cutoff
        ]

        if not recent_checks:
            return 100.0

        healthy = sum(1 for c in recent_checks if c.success)
        return round(healthy / len(recent_checks) * 100, 2)

    def _check_alerts(self, sub: SubcontractorStatus, check: HealthCheckResult) -> None:
        """Check and trigger alerts if needed."""
        alerts_to_send = []

        # Unhealthy status
        if check.status == SubcontractorHealthStatus.UNHEALTHY:
            alerts_to_send.append({
                "type": "subcontractor_unhealthy",
                "severity": "critical" if sub.is_critical_provider else "high",
                "subcontractor": sub.name,
                "message": f"Subcontractor {sub.name} is unhealthy: {check.error_message}",
            })

        # SLA breach
        if not sub.sla_compliant:
            alerts_to_send.append({
                "type": "sla_breach",
                "severity": "critical",
                "subcontractor": sub.name,
                "message": f"SLA breach: {sub.name} uptime {sub.uptime_percent_30d}% < target {sub.sla_target_uptime}%",
            })

        # Response time
        if check.response_time_ms > self.config.response_time_critical_ms:
            alerts_to_send.append({
                "type": "high_latency",
                "severity": "warning",
                "subcontractor": sub.name,
                "message": f"High latency for {sub.name}: {check.response_time_ms}ms",
            })

        for alert in alerts_to_send:
            alert["timestamp"] = datetime.now(timezone.utc).isoformat()
            self._alerts.append(alert)

            if self.config.alert_callback:
                self.config.alert_callback(alert["type"], alert)

    def get_subcontractor_status(self, subcontractor_id: str) -> Optional[SubcontractorStatus]:
        """Get subcontractor status."""
        with self._lock:
            return self._subcontractors.get(subcontractor_id)

    def get_all_statuses(self) -> List[SubcontractorStatus]:
        """Get all subcontractor statuses."""
        with self._lock:
            return list(self._subcontractors.values())

    def get_unhealthy_subcontractors(self) -> List[SubcontractorStatus]:
        """Get unhealthy subcontractors."""
        with self._lock:
            return [
                s for s in self._subcontractors.values()
                if s.health_status in (SubcontractorHealthStatus.UNHEALTHY, SubcontractorHealthStatus.DEGRADED)
            ]

    def generate_report(self, period: str = "last_30_days") -> StatusReport:
        """Generate status report."""
        with self._lock:
            subs = list(self._subcontractors.values())

        healthy = sum(1 for s in subs if s.health_status == SubcontractorHealthStatus.HEALTHY)
        degraded = sum(1 for s in subs if s.health_status == SubcontractorHealthStatus.DEGRADED)
        unhealthy = sum(1 for s in subs if s.health_status == SubcontractorHealthStatus.UNHEALTHY)
        sla_compliant = sum(1 for s in subs if s.sla_compliant)

        sla_breaches = [
            {"subcontractor": s.name, "uptime": s.uptime_percent_30d, "target": s.sla_target_uptime}
            for s in subs if not s.sla_compliant
        ]

        return StatusReport(
            reporting_period=period,
            total_subcontractors=len(subs),
            healthy_count=healthy,
            degraded_count=degraded,
            unhealthy_count=unhealthy,
            sla_compliant_count=sla_compliant,
            sla_breaches=sla_breaches,
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        report = self.generate_report()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "subcontractors": {
                "total": report.total_subcontractors,
                "healthy": report.healthy_count,
                "degraded": report.degraded_count,
                "unhealthy": report.unhealthy_count,
            },
            "sla_compliance": {
                "compliant": report.sla_compliant_count,
                "breaches": len(report.sla_breaches),
            },
            "alerts_24h": len([
                a for a in self._alerts
                if a["timestamp"] > (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
            ]),
            "dora_compliance": {
                "article_28": "monitored",
                "article_30": "compliant" if report.unhealthy_count == 0 else "attention_required",
            },
        }


def create_subcontractor_monitor(
    config: Optional[SubcontractorMonitoringConfig] = None,
) -> SubcontractorMonitor:
    """Create a SubcontractorMonitor instance."""
    return SubcontractorMonitor(config=config)
