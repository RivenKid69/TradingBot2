# -*- coding: utf-8 -*-
"""
Per-Client Metrics and Dashboards Service.

DORA Phase 3 Block 3.3: Per-client metrics and dashboards

Provides enterprise-grade per-client monitoring capabilities:
- Real-time metric collection and aggregation
- Customizable client dashboards
- Alert rules and thresholds
- SLA compliance tracking

DORA References:
    - Art. 30(2)(e): Service level descriptions with quantitative/qualitative targets
    - Art. 11: Response and recovery monitoring
    - Art. 10: Detection and monitoring requirements
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4


# =============================================================================
# Enums
# =============================================================================


class MetricType(Enum):
    """Types of metrics tracked per client."""

    # Availability metrics
    UPTIME = "uptime"
    AVAILABILITY = "availability"
    DOWNTIME = "downtime"

    # Performance metrics
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"

    # Usage metrics
    API_CALLS = "api_calls"
    DATA_PROCESSED = "data_processed"
    STORAGE_USED = "storage_used"
    BANDWIDTH = "bandwidth"

    # Security metrics
    SECURITY_EVENTS = "security_events"
    FAILED_AUTH = "failed_auth"
    BLOCKED_REQUESTS = "blocked_requests"

    # SLA metrics
    SLA_COMPLIANCE = "sla_compliance"
    MTTR = "mttr"  # Mean Time to Resolve
    MTTD = "mttd"  # Mean Time to Detect
    INCIDENTS = "incidents"


class MetricPeriod(Enum):
    """Metric aggregation periods."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class AlertThreshold(Enum):
    """Alert threshold types."""

    ABOVE = "above"
    BELOW = "below"
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CHANGE_PERCENT = "change_percent"
    ANOMALY = "anomaly"


class DashboardType(Enum):
    """Dashboard types for different audiences."""

    EXECUTIVE = "executive"  # High-level KPIs
    OPERATIONAL = "operational"  # Detailed operations
    TECHNICAL = "technical"  # Technical metrics
    COMPLIANCE = "compliance"  # Regulatory compliance
    SLA = "sla"  # SLA tracking


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class MetricDataPoint:
    """Single metric data point."""

    timestamp: datetime
    value: float
    unit: str
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "unit": self.unit,
            "tags": self.tags,
        }


@dataclass
class ClientMetric:
    """Client-specific metric definition and data."""

    metric_id: str
    client_id: str
    metric_type: MetricType
    name: str
    description: str
    unit: str
    data_points: list[MetricDataPoint] = field(default_factory=list)
    aggregation_period: MetricPeriod = MetricPeriod.HOUR
    retention_days: int = 365
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_data_point(
        self, value: float, timestamp: datetime | None = None, tags: dict[str, str] | None = None
    ) -> None:
        """Add a data point to the metric."""
        self.data_points.append(
            MetricDataPoint(
                timestamp=timestamp or datetime.utcnow(),
                value=value,
                unit=self.unit,
                tags=tags or {},
            )
        )

    def get_latest(self) -> MetricDataPoint | None:
        """Get the latest data point."""
        if not self.data_points:
            return None
        return max(self.data_points, key=lambda x: x.timestamp)

    def get_average(self, period: timedelta | None = None) -> float | None:
        """Calculate average value over a period."""
        if not self.data_points:
            return None

        points = self.data_points
        if period:
            cutoff = datetime.utcnow() - period
            points = [p for p in points if p.timestamp >= cutoff]

        if not points:
            return None
        return sum(p.value for p in points) / len(points)

    def get_min(self, period: timedelta | None = None) -> float | None:
        """Get minimum value over a period."""
        if not self.data_points:
            return None

        points = self.data_points
        if period:
            cutoff = datetime.utcnow() - period
            points = [p for p in points if p.timestamp >= cutoff]

        if not points:
            return None
        return min(p.value for p in points)

    def get_max(self, period: timedelta | None = None) -> float | None:
        """Get maximum value over a period."""
        if not self.data_points:
            return None

        points = self.data_points
        if period:
            cutoff = datetime.utcnow() - period
            points = [p for p in points if p.timestamp >= cutoff]

        if not points:
            return None
        return max(p.value for p in points)

    def get_percentile(self, percentile: float, period: timedelta | None = None) -> float | None:
        """Calculate percentile value over a period."""
        if not self.data_points:
            return None

        points = self.data_points
        if period:
            cutoff = datetime.utcnow() - period
            points = [p for p in points if p.timestamp >= cutoff]

        if not points:
            return None

        sorted_values = sorted(p.value for p in points)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]


@dataclass
class AlertRule:
    """Alert rule definition for metrics."""

    rule_id: str
    metric_id: str
    client_id: str
    name: str
    description: str
    threshold_type: AlertThreshold
    threshold_value: float
    comparison_period: MetricPeriod = MetricPeriod.HOUR
    enabled: bool = True
    severity: str = "warning"  # info, warning, critical
    notification_channels: list[str] = field(default_factory=list)
    cooldown_minutes: int = 15
    last_triggered: datetime | None = None

    def should_trigger(self, current_value: float, previous_value: float | None = None) -> bool:
        """Evaluate if the alert should trigger."""
        if not self.enabled:
            return False

        # Check cooldown
        if self.last_triggered:
            cooldown_end = self.last_triggered + timedelta(minutes=self.cooldown_minutes)
            if datetime.utcnow() < cooldown_end:
                return False

        if self.threshold_type == AlertThreshold.ABOVE:
            return current_value > self.threshold_value
        elif self.threshold_type == AlertThreshold.BELOW:
            return current_value < self.threshold_value
        elif self.threshold_type == AlertThreshold.EQUALS:
            return current_value == self.threshold_value
        elif self.threshold_type == AlertThreshold.NOT_EQUALS:
            return current_value != self.threshold_value
        elif self.threshold_type == AlertThreshold.CHANGE_PERCENT:
            if previous_value is None or previous_value == 0:
                return False
            change = abs((current_value - previous_value) / previous_value) * 100
            return change > self.threshold_value
        return False


@dataclass
class MetricAlert:
    """Generated metric alert."""

    alert_id: str
    rule_id: str
    metric_id: str
    client_id: str
    triggered_at: datetime
    current_value: float
    threshold_value: float
    severity: str
    message: str
    acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    resolved: bool = False
    resolved_at: datetime | None = None

    def acknowledge(self, user: str) -> None:
        """Acknowledge the alert."""
        self.acknowledged = True
        self.acknowledged_by = user
        self.acknowledged_at = datetime.utcnow()

    def resolve(self) -> None:
        """Resolve the alert."""
        self.resolved = True
        self.resolved_at = datetime.utcnow()


@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""

    widget_id: str
    widget_type: str  # chart, gauge, table, counter
    title: str
    metric_ids: list[str]
    visualization: str  # line, bar, pie, area
    time_range: MetricPeriod
    refresh_interval_seconds: int = 60
    position: dict[str, int] = field(default_factory=dict)  # x, y, width, height
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClientDashboard:
    """Client-specific dashboard configuration."""

    dashboard_id: str
    client_id: str
    name: str
    description: str
    dashboard_type: DashboardType
    widgets: list[DashboardWidget] = field(default_factory=list)
    layout: str = "grid"
    refresh_interval_seconds: int = 60
    is_default: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"

    def add_widget(self, widget: DashboardWidget) -> None:
        """Add a widget to the dashboard."""
        self.widgets.append(widget)

    def remove_widget(self, widget_id: str) -> bool:
        """Remove a widget from the dashboard."""
        for i, widget in enumerate(self.widgets):
            if widget.widget_id == widget_id:
                self.widgets.pop(i)
                return True
        return False


@dataclass
class ClientMetricsConfig:
    """Client metrics service configuration."""

    default_retention_days: int = 365
    max_data_points_per_metric: int = 100000
    default_aggregation_period: MetricPeriod = MetricPeriod.HOUR
    alert_check_interval_seconds: int = 60
    dashboard_refresh_interval_seconds: int = 60
    enable_anomaly_detection: bool = True
    anomaly_sensitivity: float = 2.0  # Standard deviations


# =============================================================================
# Main Service Class
# =============================================================================


class ClientMetricsService:
    """
    Per-Client Metrics and Dashboards Service.

    Provides enterprise-grade per-client monitoring capabilities
    per DORA Art. 30(2)(e) service level requirements.
    """

    def __init__(self, config: ClientMetricsConfig | None = None) -> None:
        """Initialize client metrics service."""
        self.config = config or ClientMetricsConfig()
        self._metrics: dict[str, ClientMetric] = {}
        self._alert_rules: dict[str, AlertRule] = {}
        self._alerts: dict[str, MetricAlert] = {}
        self._dashboards: dict[str, ClientDashboard] = {}

    # =========================================================================
    # Metric Management
    # =========================================================================

    def create_metric(
        self,
        client_id: str,
        metric_type: MetricType,
        name: str,
        description: str,
        unit: str,
        aggregation_period: MetricPeriod | None = None,
        retention_days: int | None = None,
    ) -> ClientMetric:
        """Create a new client metric."""
        metric_id = str(uuid4())
        metric = ClientMetric(
            metric_id=metric_id,
            client_id=client_id,
            metric_type=metric_type,
            name=name,
            description=description,
            unit=unit,
            aggregation_period=aggregation_period or self.config.default_aggregation_period,
            retention_days=retention_days or self.config.default_retention_days,
        )
        self._metrics[metric_id] = metric
        return metric

    def get_metric(self, metric_id: str) -> ClientMetric | None:
        """Get metric by ID."""
        return self._metrics.get(metric_id)

    def list_metrics(
        self,
        client_id: str | None = None,
        metric_type: MetricType | None = None,
    ) -> list[ClientMetric]:
        """List metrics with optional filters."""
        metrics = list(self._metrics.values())

        if client_id:
            metrics = [m for m in metrics if m.client_id == client_id]
        if metric_type:
            metrics = [m for m in metrics if m.metric_type == metric_type]

        return metrics

    def record_metric(
        self,
        metric_id: str,
        value: float,
        timestamp: datetime | None = None,
        tags: dict[str, str] | None = None,
    ) -> bool:
        """Record a metric data point."""
        metric = self._metrics.get(metric_id)
        if not metric:
            return False

        metric.add_data_point(value, timestamp, tags)

        # Trim old data points if exceeding limit
        if len(metric.data_points) > self.config.max_data_points_per_metric:
            cutoff = datetime.utcnow() - timedelta(days=metric.retention_days)
            metric.data_points = [p for p in metric.data_points if p.timestamp >= cutoff]

        # Check alert rules
        self._check_alerts(metric)

        return True

    def get_metric_summary(
        self,
        metric_id: str,
        period: timedelta | None = None,
    ) -> dict[str, Any] | None:
        """Get metric summary statistics."""
        metric = self._metrics.get(metric_id)
        if not metric:
            return None

        latest = metric.get_latest()
        return {
            "metric_id": metric_id,
            "name": metric.name,
            "type": metric.metric_type.value,
            "unit": metric.unit,
            "latest_value": latest.value if latest else None,
            "latest_timestamp": latest.timestamp.isoformat() if latest else None,
            "average": metric.get_average(period),
            "min": metric.get_min(period),
            "max": metric.get_max(period),
            "p50": metric.get_percentile(50, period),
            "p95": metric.get_percentile(95, period),
            "p99": metric.get_percentile(99, period),
            "data_points_count": len(metric.data_points),
        }

    # =========================================================================
    # Alert Management
    # =========================================================================

    def create_alert_rule(
        self,
        metric_id: str,
        name: str,
        description: str,
        threshold_type: AlertThreshold,
        threshold_value: float,
        severity: str = "warning",
        notification_channels: list[str] | None = None,
    ) -> AlertRule:
        """Create an alert rule for a metric."""
        metric = self._metrics.get(metric_id)
        if not metric:
            raise ValueError(f"Metric not found: {metric_id}")

        rule_id = str(uuid4())
        rule = AlertRule(
            rule_id=rule_id,
            metric_id=metric_id,
            client_id=metric.client_id,
            name=name,
            description=description,
            threshold_type=threshold_type,
            threshold_value=threshold_value,
            severity=severity,
            notification_channels=notification_channels or [],
        )
        self._alert_rules[rule_id] = rule
        return rule

    def get_alert_rule(self, rule_id: str) -> AlertRule | None:
        """Get alert rule by ID."""
        return self._alert_rules.get(rule_id)

    def list_alert_rules(self, client_id: str | None = None) -> list[AlertRule]:
        """List alert rules with optional client filter."""
        rules = list(self._alert_rules.values())
        if client_id:
            rules = [r for r in rules if r.client_id == client_id]
        return rules

    def _check_alerts(self, metric: ClientMetric) -> None:
        """Check alert rules for a metric."""
        rules = [r for r in self._alert_rules.values() if r.metric_id == metric.metric_id]

        latest = metric.get_latest()
        if not latest:
            return

        for rule in rules:
            previous = metric.get_average(timedelta(hours=1))
            if rule.should_trigger(latest.value, previous):
                self._trigger_alert(rule, latest.value)

    def _trigger_alert(self, rule: AlertRule, current_value: float) -> MetricAlert:
        """Trigger an alert."""
        alert_id = str(uuid4())
        alert = MetricAlert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            metric_id=rule.metric_id,
            client_id=rule.client_id,
            triggered_at=datetime.utcnow(),
            current_value=current_value,
            threshold_value=rule.threshold_value,
            severity=rule.severity,
            message=f"Alert: {rule.name} - Current value {current_value} {rule.threshold_type.value} threshold {rule.threshold_value}",
        )

        rule.last_triggered = datetime.utcnow()
        self._alerts[alert_id] = alert
        return alert

    def get_alert(self, alert_id: str) -> MetricAlert | None:
        """Get alert by ID."""
        return self._alerts.get(alert_id)

    def list_alerts(
        self,
        client_id: str | None = None,
        acknowledged: bool | None = None,
        resolved: bool | None = None,
    ) -> list[MetricAlert]:
        """List alerts with optional filters."""
        alerts = list(self._alerts.values())

        if client_id:
            alerts = [a for a in alerts if a.client_id == client_id]
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]

        return alerts

    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert."""
        alert = self._alerts.get(alert_id)
        if not alert:
            return False
        alert.acknowledge(user)
        return True

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        alert = self._alerts.get(alert_id)
        if not alert:
            return False
        alert.resolve()
        return True

    # =========================================================================
    # Dashboard Management
    # =========================================================================

    def create_dashboard(
        self,
        client_id: str,
        name: str,
        description: str,
        dashboard_type: DashboardType,
        created_by: str = "system",
    ) -> ClientDashboard:
        """Create a new dashboard for a client."""
        dashboard_id = str(uuid4())
        dashboard = ClientDashboard(
            dashboard_id=dashboard_id,
            client_id=client_id,
            name=name,
            description=description,
            dashboard_type=dashboard_type,
            created_by=created_by,
        )
        self._dashboards[dashboard_id] = dashboard
        return dashboard

    def get_dashboard(self, dashboard_id: str) -> ClientDashboard | None:
        """Get dashboard by ID."""
        return self._dashboards.get(dashboard_id)

    def list_dashboards(
        self,
        client_id: str | None = None,
        dashboard_type: DashboardType | None = None,
    ) -> list[ClientDashboard]:
        """List dashboards with optional filters."""
        dashboards = list(self._dashboards.values())

        if client_id:
            dashboards = [d for d in dashboards if d.client_id == client_id]
        if dashboard_type:
            dashboards = [d for d in dashboards if d.dashboard_type == dashboard_type]

        return dashboards

    def add_widget_to_dashboard(
        self,
        dashboard_id: str,
        widget_type: str,
        title: str,
        metric_ids: list[str],
        visualization: str = "line",
        time_range: MetricPeriod = MetricPeriod.DAY,
    ) -> DashboardWidget:
        """Add a widget to a dashboard."""
        dashboard = self._dashboards.get(dashboard_id)
        if not dashboard:
            raise ValueError(f"Dashboard not found: {dashboard_id}")

        widget = DashboardWidget(
            widget_id=str(uuid4()),
            widget_type=widget_type,
            title=title,
            metric_ids=metric_ids,
            visualization=visualization,
            time_range=time_range,
        )
        dashboard.add_widget(widget)
        return widget

    def create_default_dashboard(self, client_id: str) -> ClientDashboard:
        """Create a default SLA dashboard for a client."""
        dashboard = self.create_dashboard(
            client_id=client_id,
            name="SLA Compliance Dashboard",
            description="Default dashboard showing SLA compliance metrics",
            dashboard_type=DashboardType.SLA,
        )
        dashboard.is_default = True

        # Create default metrics if they don't exist
        uptime_metric = self.create_metric(
            client_id=client_id,
            metric_type=MetricType.UPTIME,
            name="Service Uptime",
            description="Service uptime percentage",
            unit="percent",
        )

        latency_metric = self.create_metric(
            client_id=client_id,
            metric_type=MetricType.LATENCY,
            name="API Latency",
            description="API response latency",
            unit="ms",
        )

        error_metric = self.create_metric(
            client_id=client_id,
            metric_type=MetricType.ERROR_RATE,
            name="Error Rate",
            description="API error rate",
            unit="percent",
        )

        # Add default widgets
        self.add_widget_to_dashboard(
            dashboard_id=dashboard.dashboard_id,
            widget_type="gauge",
            title="Current Uptime",
            metric_ids=[uptime_metric.metric_id],
            visualization="gauge",
        )

        self.add_widget_to_dashboard(
            dashboard_id=dashboard.dashboard_id,
            widget_type="chart",
            title="Latency Over Time",
            metric_ids=[latency_metric.metric_id],
            visualization="line",
        )

        self.add_widget_to_dashboard(
            dashboard_id=dashboard.dashboard_id,
            widget_type="chart",
            title="Error Rate",
            metric_ids=[error_metric.metric_id],
            visualization="area",
        )

        return dashboard

    # =========================================================================
    # SLA Compliance
    # =========================================================================

    def calculate_sla_compliance(
        self,
        client_id: str,
        period: timedelta = timedelta(days=30),
    ) -> dict[str, Any]:
        """
        Calculate SLA compliance for a client.

        Per DORA Art. 30(2)(e): Service level descriptions with
        quantitative and qualitative performance targets.
        """
        metrics = self.list_metrics(client_id=client_id)

        uptime_metrics = [m for m in metrics if m.metric_type == MetricType.UPTIME]
        latency_metrics = [m for m in metrics if m.metric_type == MetricType.LATENCY]
        error_metrics = [m for m in metrics if m.metric_type == MetricType.ERROR_RATE]

        avg_uptime = None
        if uptime_metrics:
            uptimes = [
                m.get_average(period) for m in uptime_metrics if m.get_average(period) is not None
            ]
            avg_uptime = sum(uptimes) / len(uptimes) if uptimes else None

        avg_latency = None
        p95_latency = None
        if latency_metrics:
            latencies = [
                m.get_average(period) for m in latency_metrics if m.get_average(period) is not None
            ]
            avg_latency = sum(latencies) / len(latencies) if latencies else None
            p95_values = [
                m.get_percentile(95, period)
                for m in latency_metrics
                if m.get_percentile(95, period) is not None
            ]
            p95_latency = max(p95_values) if p95_values else None

        avg_error_rate = None
        if error_metrics:
            errors = [
                m.get_average(period) for m in error_metrics if m.get_average(period) is not None
            ]
            avg_error_rate = sum(errors) / len(errors) if errors else None

        # Determine overall compliance
        compliance_score = 100.0
        violations = []

        if avg_uptime is not None and avg_uptime < 99.9:
            compliance_score -= (99.9 - avg_uptime) * 10
            violations.append(f"Uptime {avg_uptime:.2f}% below 99.9% target")

        if p95_latency is not None and p95_latency > 200:
            compliance_score -= min(20, (p95_latency - 200) / 10)
            violations.append(f"P95 latency {p95_latency:.0f}ms above 200ms target")

        if avg_error_rate is not None and avg_error_rate > 0.1:
            compliance_score -= min(20, avg_error_rate * 100)
            violations.append(f"Error rate {avg_error_rate:.2f}% above 0.1% target")

        return {
            "client_id": client_id,
            "period_days": period.days,
            "calculated_at": datetime.utcnow().isoformat(),
            "metrics": {
                "uptime_percent": avg_uptime,
                "avg_latency_ms": avg_latency,
                "p95_latency_ms": p95_latency,
                "error_rate_percent": avg_error_rate,
            },
            "compliance_score": max(0, compliance_score),
            "is_compliant": compliance_score >= 95,
            "violations": violations,
            "sla_targets": {
                "uptime_percent": 99.9,
                "p95_latency_ms": 200,
                "error_rate_percent": 0.1,
            },
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_client_metrics(
    default_retention_days: int = 365,
    enable_anomaly_detection: bool = True,
    **kwargs: Any,
) -> ClientMetricsService:
    """Create client metrics service instance."""
    config = ClientMetricsConfig(
        default_retention_days=default_retention_days,
        enable_anomaly_detection=enable_anomaly_detection,
        **kwargs,
    )
    return ClientMetricsService(config)
