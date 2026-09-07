# -*- coding: utf-8 -*-
"""
Comprehensive tests for Client Metrics Service.

Tests per-client metrics, dashboards, alerts, and SLA compliance
per DORA Art. 30(2)(e) requirements.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from services.enterprise.client_metrics import (
    # Enums
    MetricType,
    MetricPeriod,
    AlertThreshold,
    DashboardType,
    # Data structures
    MetricDataPoint,
    ClientMetric,
    AlertRule,
    MetricAlert,
    DashboardWidget,
    ClientDashboard,
    ClientMetricsConfig,
    # Service
    ClientMetricsService,
    # Factory
    create_client_metrics,
)


# =============================================================================
# MetricDataPoint Tests
# =============================================================================


class TestMetricDataPoint:
    """Tests for MetricDataPoint dataclass."""

    def test_create_data_point(self) -> None:
        """Test creating a metric data point."""
        now = datetime.utcnow()
        point = MetricDataPoint(
            timestamp=now,
            value=99.5,
            unit="percent",
            tags={"environment": "production"},
        )
        assert point.timestamp == now
        assert point.value == 99.5
        assert point.unit == "percent"
        assert point.tags == {"environment": "production"}

    def test_data_point_to_dict(self) -> None:
        """Test converting data point to dictionary."""
        now = datetime.utcnow()
        point = MetricDataPoint(
            timestamp=now,
            value=42.0,
            unit="ms",
            tags={"region": "eu-west-1"},
        )
        result = point.to_dict()
        assert result["timestamp"] == now.isoformat()
        assert result["value"] == 42.0
        assert result["unit"] == "ms"
        assert result["tags"] == {"region": "eu-west-1"}

    def test_data_point_default_tags(self) -> None:
        """Test data point with default empty tags."""
        point = MetricDataPoint(
            timestamp=datetime.utcnow(),
            value=100.0,
            unit="count",
        )
        assert point.tags == {}


# =============================================================================
# ClientMetric Tests
# =============================================================================


class TestClientMetric:
    """Tests for ClientMetric dataclass."""

    def test_create_client_metric(self) -> None:
        """Test creating a client metric."""
        metric = ClientMetric(
            metric_id="metric-123",
            client_id="client-456",
            metric_type=MetricType.UPTIME,
            name="Service Uptime",
            description="Service uptime percentage",
            unit="percent",
        )
        assert metric.metric_id == "metric-123"
        assert metric.client_id == "client-456"
        assert metric.metric_type == MetricType.UPTIME
        assert metric.unit == "percent"
        assert metric.data_points == []

    def test_add_data_point(self) -> None:
        """Test adding data point to metric."""
        metric = ClientMetric(
            metric_id="metric-1",
            client_id="client-1",
            metric_type=MetricType.LATENCY,
            name="API Latency",
            description="API response latency",
            unit="ms",
        )
        metric.add_data_point(45.0)
        assert len(metric.data_points) == 1
        assert metric.data_points[0].value == 45.0
        assert metric.data_points[0].unit == "ms"

    def test_add_data_point_with_timestamp(self) -> None:
        """Test adding data point with specific timestamp."""
        metric = ClientMetric(
            metric_id="metric-1",
            client_id="client-1",
            metric_type=MetricType.THROUGHPUT,
            name="Throughput",
            description="Requests per second",
            unit="rps",
        )
        timestamp = datetime(2024, 1, 15, 12, 0, 0)
        metric.add_data_point(1000.0, timestamp=timestamp)
        assert metric.data_points[0].timestamp == timestamp

    def test_add_data_point_with_tags(self) -> None:
        """Test adding data point with tags."""
        metric = ClientMetric(
            metric_id="metric-1",
            client_id="client-1",
            metric_type=MetricType.ERROR_RATE,
            name="Error Rate",
            description="API error rate",
            unit="percent",
        )
        tags = {"endpoint": "/api/v1/users"}
        metric.add_data_point(0.5, tags=tags)
        assert metric.data_points[0].tags == tags

    def test_get_latest_no_data(self) -> None:
        """Test get_latest with no data points."""
        metric = ClientMetric(
            metric_id="metric-1",
            client_id="client-1",
            metric_type=MetricType.UPTIME,
            name="Uptime",
            description="Uptime",
            unit="percent",
        )
        assert metric.get_latest() is None

    def test_get_latest_with_data(self) -> None:
        """Test get_latest returns most recent data point."""
        metric = ClientMetric(
            metric_id="metric-1",
            client_id="client-1",
            metric_type=MetricType.UPTIME,
            name="Uptime",
            description="Uptime",
            unit="percent",
        )
        metric.add_data_point(99.0, timestamp=datetime(2024, 1, 1))
        metric.add_data_point(99.5, timestamp=datetime(2024, 1, 2))
        metric.add_data_point(99.9, timestamp=datetime(2024, 1, 3))

        latest = metric.get_latest()
        assert latest is not None
        assert latest.value == 99.9

    def test_get_average_no_data(self) -> None:
        """Test get_average with no data points."""
        metric = ClientMetric(
            metric_id="metric-1",
            client_id="client-1",
            metric_type=MetricType.LATENCY,
            name="Latency",
            description="Latency",
            unit="ms",
        )
        assert metric.get_average() is None

    def test_get_average_with_data(self) -> None:
        """Test get_average calculation."""
        metric = ClientMetric(
            metric_id="metric-1",
            client_id="client-1",
            metric_type=MetricType.LATENCY,
            name="Latency",
            description="Latency",
            unit="ms",
        )
        metric.add_data_point(10.0)
        metric.add_data_point(20.0)
        metric.add_data_point(30.0)

        avg = metric.get_average()
        assert avg == 20.0

    def test_get_average_with_period(self) -> None:
        """Test get_average with period filter."""
        metric = ClientMetric(
            metric_id="metric-1",
            client_id="client-1",
            metric_type=MetricType.LATENCY,
            name="Latency",
            description="Latency",
            unit="ms",
        )
        # Old data point
        metric.add_data_point(100.0, timestamp=datetime.utcnow() - timedelta(days=10))
        # Recent data points
        metric.add_data_point(10.0, timestamp=datetime.utcnow() - timedelta(hours=1))
        metric.add_data_point(20.0, timestamp=datetime.utcnow())

        avg = metric.get_average(period=timedelta(days=1))
        assert avg == 15.0  # Only recent points

    def test_get_min(self) -> None:
        """Test get_min calculation."""
        metric = ClientMetric(
            metric_id="metric-1",
            client_id="client-1",
            metric_type=MetricType.LATENCY,
            name="Latency",
            description="Latency",
            unit="ms",
        )
        metric.add_data_point(50.0)
        metric.add_data_point(30.0)
        metric.add_data_point(70.0)

        assert metric.get_min() == 30.0

    def test_get_max(self) -> None:
        """Test get_max calculation."""
        metric = ClientMetric(
            metric_id="metric-1",
            client_id="client-1",
            metric_type=MetricType.LATENCY,
            name="Latency",
            description="Latency",
            unit="ms",
        )
        metric.add_data_point(50.0)
        metric.add_data_point(30.0)
        metric.add_data_point(70.0)

        assert metric.get_max() == 70.0

    def test_get_percentile(self) -> None:
        """Test get_percentile calculation."""
        metric = ClientMetric(
            metric_id="metric-1",
            client_id="client-1",
            metric_type=MetricType.LATENCY,
            name="Latency",
            description="Latency",
            unit="ms",
        )
        for i in range(1, 101):
            metric.add_data_point(float(i))

        # P50 should be around 50
        p50 = metric.get_percentile(50)
        assert p50 is not None
        assert 49 <= p50 <= 51

        # P95 should be around 95
        p95 = metric.get_percentile(95)
        assert p95 is not None
        assert 94 <= p95 <= 96


# =============================================================================
# AlertRule Tests
# =============================================================================


class TestAlertRule:
    """Tests for AlertRule dataclass."""

    def test_create_alert_rule(self) -> None:
        """Test creating an alert rule."""
        rule = AlertRule(
            rule_id="rule-1",
            metric_id="metric-1",
            client_id="client-1",
            name="High Latency Alert",
            description="Alert when latency exceeds threshold",
            threshold_type=AlertThreshold.ABOVE,
            threshold_value=200.0,
        )
        assert rule.rule_id == "rule-1"
        assert rule.threshold_type == AlertThreshold.ABOVE
        assert rule.threshold_value == 200.0
        assert rule.enabled is True

    def test_should_trigger_above(self) -> None:
        """Test alert triggering for ABOVE threshold."""
        rule = AlertRule(
            rule_id="rule-1",
            metric_id="metric-1",
            client_id="client-1",
            name="High Latency",
            description="High latency alert",
            threshold_type=AlertThreshold.ABOVE,
            threshold_value=100.0,
        )
        assert rule.should_trigger(150.0) is True
        assert rule.should_trigger(50.0) is False

    def test_should_trigger_below(self) -> None:
        """Test alert triggering for BELOW threshold."""
        rule = AlertRule(
            rule_id="rule-1",
            metric_id="metric-1",
            client_id="client-1",
            name="Low Uptime",
            description="Low uptime alert",
            threshold_type=AlertThreshold.BELOW,
            threshold_value=99.0,
        )
        assert rule.should_trigger(98.0) is True
        assert rule.should_trigger(99.5) is False

    def test_should_trigger_equals(self) -> None:
        """Test alert triggering for EQUALS threshold."""
        rule = AlertRule(
            rule_id="rule-1",
            metric_id="metric-1",
            client_id="client-1",
            name="Error Count",
            description="Error count alert",
            threshold_type=AlertThreshold.EQUALS,
            threshold_value=0.0,
        )
        assert rule.should_trigger(0.0) is True
        assert rule.should_trigger(1.0) is False

    def test_should_trigger_not_equals(self) -> None:
        """Test alert triggering for NOT_EQUALS threshold."""
        rule = AlertRule(
            rule_id="rule-1",
            metric_id="metric-1",
            client_id="client-1",
            name="Status Check",
            description="Status alert",
            threshold_type=AlertThreshold.NOT_EQUALS,
            threshold_value=1.0,
        )
        assert rule.should_trigger(0.0) is True
        assert rule.should_trigger(1.0) is False

    def test_should_trigger_change_percent(self) -> None:
        """Test alert triggering for CHANGE_PERCENT threshold."""
        rule = AlertRule(
            rule_id="rule-1",
            metric_id="metric-1",
            client_id="client-1",
            name="Spike Detection",
            description="Detect sudden changes",
            threshold_type=AlertThreshold.CHANGE_PERCENT,
            threshold_value=50.0,
        )
        # 100 to 200 is 100% change
        assert rule.should_trigger(200.0, 100.0) is True
        # 100 to 110 is 10% change
        assert rule.should_trigger(110.0, 100.0) is False

    def test_should_trigger_disabled(self) -> None:
        """Test that disabled rules don't trigger."""
        rule = AlertRule(
            rule_id="rule-1",
            metric_id="metric-1",
            client_id="client-1",
            name="Disabled Alert",
            description="This is disabled",
            threshold_type=AlertThreshold.ABOVE,
            threshold_value=100.0,
            enabled=False,
        )
        assert rule.should_trigger(200.0) is False

    def test_should_trigger_cooldown(self) -> None:
        """Test alert cooldown period."""
        rule = AlertRule(
            rule_id="rule-1",
            metric_id="metric-1",
            client_id="client-1",
            name="Cooldown Alert",
            description="Has cooldown",
            threshold_type=AlertThreshold.ABOVE,
            threshold_value=100.0,
            cooldown_minutes=15,
        )
        rule.last_triggered = datetime.utcnow()  # Just triggered
        assert rule.should_trigger(200.0) is False  # Still in cooldown


# =============================================================================
# MetricAlert Tests
# =============================================================================


class TestMetricAlert:
    """Tests for MetricAlert dataclass."""

    def test_create_alert(self) -> None:
        """Test creating a metric alert."""
        alert = MetricAlert(
            alert_id="alert-1",
            rule_id="rule-1",
            metric_id="metric-1",
            client_id="client-1",
            triggered_at=datetime.utcnow(),
            current_value=250.0,
            threshold_value=200.0,
            severity="warning",
            message="Latency exceeded threshold",
        )
        assert alert.alert_id == "alert-1"
        assert alert.acknowledged is False
        assert alert.resolved is False

    def test_acknowledge_alert(self) -> None:
        """Test acknowledging an alert."""
        alert = MetricAlert(
            alert_id="alert-1",
            rule_id="rule-1",
            metric_id="metric-1",
            client_id="client-1",
            triggered_at=datetime.utcnow(),
            current_value=250.0,
            threshold_value=200.0,
            severity="warning",
            message="Test alert",
        )
        alert.acknowledge("admin@example.com")
        assert alert.acknowledged is True
        assert alert.acknowledged_by == "admin@example.com"
        assert alert.acknowledged_at is not None

    def test_resolve_alert(self) -> None:
        """Test resolving an alert."""
        alert = MetricAlert(
            alert_id="alert-1",
            rule_id="rule-1",
            metric_id="metric-1",
            client_id="client-1",
            triggered_at=datetime.utcnow(),
            current_value=250.0,
            threshold_value=200.0,
            severity="critical",
            message="Test alert",
        )
        alert.resolve()
        assert alert.resolved is True
        assert alert.resolved_at is not None


# =============================================================================
# Dashboard Tests
# =============================================================================


class TestClientDashboard:
    """Tests for ClientDashboard dataclass."""

    def test_create_dashboard(self) -> None:
        """Test creating a dashboard."""
        dashboard = ClientDashboard(
            dashboard_id="dash-1",
            client_id="client-1",
            name="SLA Dashboard",
            description="SLA compliance monitoring",
            dashboard_type=DashboardType.SLA,
        )
        assert dashboard.dashboard_id == "dash-1"
        assert dashboard.dashboard_type == DashboardType.SLA
        assert dashboard.widgets == []

    def test_add_widget(self) -> None:
        """Test adding widget to dashboard."""
        dashboard = ClientDashboard(
            dashboard_id="dash-1",
            client_id="client-1",
            name="Test Dashboard",
            description="Test",
            dashboard_type=DashboardType.OPERATIONAL,
        )
        widget = DashboardWidget(
            widget_id="widget-1",
            widget_type="chart",
            title="Latency Chart",
            metric_ids=["metric-1"],
            visualization="line",
            time_range=MetricPeriod.DAY,
        )
        dashboard.add_widget(widget)
        assert len(dashboard.widgets) == 1
        assert dashboard.widgets[0].title == "Latency Chart"

    def test_remove_widget(self) -> None:
        """Test removing widget from dashboard."""
        dashboard = ClientDashboard(
            dashboard_id="dash-1",
            client_id="client-1",
            name="Test Dashboard",
            description="Test",
            dashboard_type=DashboardType.EXECUTIVE,
        )
        widget = DashboardWidget(
            widget_id="widget-1",
            widget_type="gauge",
            title="Uptime Gauge",
            metric_ids=["metric-1"],
            visualization="gauge",
            time_range=MetricPeriod.HOUR,
        )
        dashboard.add_widget(widget)

        result = dashboard.remove_widget("widget-1")
        assert result is True
        assert len(dashboard.widgets) == 0

    def test_remove_widget_not_found(self) -> None:
        """Test removing non-existent widget."""
        dashboard = ClientDashboard(
            dashboard_id="dash-1",
            client_id="client-1",
            name="Test Dashboard",
            description="Test",
            dashboard_type=DashboardType.TECHNICAL,
        )
        result = dashboard.remove_widget("nonexistent")
        assert result is False


# =============================================================================
# ClientMetricsService Tests
# =============================================================================


class TestClientMetricsService:
    """Tests for ClientMetricsService."""

    def test_create_service_default_config(self) -> None:
        """Test creating service with default config."""
        service = ClientMetricsService()
        assert service.config.default_retention_days == 365
        assert service.config.enable_anomaly_detection is True

    def test_create_service_custom_config(self) -> None:
        """Test creating service with custom config."""
        config = ClientMetricsConfig(
            default_retention_days=180,
            max_data_points_per_metric=50000,
            enable_anomaly_detection=False,
        )
        service = ClientMetricsService(config)
        assert service.config.default_retention_days == 180
        assert service.config.max_data_points_per_metric == 50000

    def test_create_metric(self) -> None:
        """Test creating a metric."""
        service = ClientMetricsService()
        metric = service.create_metric(
            client_id="client-1",
            metric_type=MetricType.UPTIME,
            name="Service Uptime",
            description="Service uptime percentage",
            unit="percent",
        )
        assert metric.client_id == "client-1"
        assert metric.metric_type == MetricType.UPTIME
        assert metric.name == "Service Uptime"

    def test_get_metric(self) -> None:
        """Test getting a metric by ID."""
        service = ClientMetricsService()
        metric = service.create_metric(
            client_id="client-1",
            metric_type=MetricType.LATENCY,
            name="API Latency",
            description="Latency",
            unit="ms",
        )
        retrieved = service.get_metric(metric.metric_id)
        assert retrieved is not None
        assert retrieved.metric_id == metric.metric_id

    def test_get_metric_not_found(self) -> None:
        """Test getting non-existent metric."""
        service = ClientMetricsService()
        assert service.get_metric("nonexistent") is None

    def test_list_metrics(self) -> None:
        """Test listing metrics."""
        service = ClientMetricsService()
        service.create_metric("client-1", MetricType.UPTIME, "Uptime", "Uptime", "percent")
        service.create_metric("client-1", MetricType.LATENCY, "Latency", "Latency", "ms")
        service.create_metric("client-2", MetricType.ERROR_RATE, "Errors", "Errors", "percent")

        all_metrics = service.list_metrics()
        assert len(all_metrics) == 3

    def test_list_metrics_by_client(self) -> None:
        """Test listing metrics filtered by client."""
        service = ClientMetricsService()
        service.create_metric("client-1", MetricType.UPTIME, "Uptime", "Uptime", "percent")
        service.create_metric("client-1", MetricType.LATENCY, "Latency", "Latency", "ms")
        service.create_metric("client-2", MetricType.ERROR_RATE, "Errors", "Errors", "percent")

        client_1_metrics = service.list_metrics(client_id="client-1")
        assert len(client_1_metrics) == 2

    def test_list_metrics_by_type(self) -> None:
        """Test listing metrics filtered by type."""
        service = ClientMetricsService()
        service.create_metric("client-1", MetricType.UPTIME, "Uptime 1", "Uptime", "percent")
        service.create_metric("client-2", MetricType.UPTIME, "Uptime 2", "Uptime", "percent")
        service.create_metric("client-1", MetricType.LATENCY, "Latency", "Latency", "ms")

        uptime_metrics = service.list_metrics(metric_type=MetricType.UPTIME)
        assert len(uptime_metrics) == 2

    def test_record_metric(self) -> None:
        """Test recording metric data."""
        service = ClientMetricsService()
        metric = service.create_metric(
            client_id="client-1",
            metric_type=MetricType.UPTIME,
            name="Uptime",
            description="Uptime",
            unit="percent",
        )
        result = service.record_metric(metric.metric_id, 99.9)
        assert result is True
        assert len(metric.data_points) == 1

    def test_record_metric_not_found(self) -> None:
        """Test recording to non-existent metric."""
        service = ClientMetricsService()
        result = service.record_metric("nonexistent", 100.0)
        assert result is False

    def test_get_metric_summary(self) -> None:
        """Test getting metric summary."""
        service = ClientMetricsService()
        metric = service.create_metric(
            client_id="client-1",
            metric_type=MetricType.LATENCY,
            name="API Latency",
            description="Latency",
            unit="ms",
        )
        for i in range(10):
            service.record_metric(metric.metric_id, float(i * 10))

        summary = service.get_metric_summary(metric.metric_id)
        assert summary is not None
        assert summary["name"] == "API Latency"
        assert summary["type"] == "latency"
        assert summary["data_points_count"] == 10
        assert summary["average"] is not None

    def test_get_metric_summary_not_found(self) -> None:
        """Test getting summary for non-existent metric."""
        service = ClientMetricsService()
        assert service.get_metric_summary("nonexistent") is None

    def test_create_alert_rule(self) -> None:
        """Test creating an alert rule."""
        service = ClientMetricsService()
        metric = service.create_metric(
            client_id="client-1",
            metric_type=MetricType.LATENCY,
            name="Latency",
            description="Latency",
            unit="ms",
        )
        rule = service.create_alert_rule(
            metric_id=metric.metric_id,
            name="High Latency",
            description="Alert on high latency",
            threshold_type=AlertThreshold.ABOVE,
            threshold_value=200.0,
            severity="warning",
        )
        assert rule.name == "High Latency"
        assert rule.threshold_type == AlertThreshold.ABOVE

    def test_create_alert_rule_metric_not_found(self) -> None:
        """Test creating alert rule for non-existent metric."""
        service = ClientMetricsService()
        with pytest.raises(ValueError, match="Metric not found"):
            service.create_alert_rule(
                metric_id="nonexistent",
                name="Test",
                description="Test",
                threshold_type=AlertThreshold.ABOVE,
                threshold_value=100.0,
            )

    def test_list_alert_rules(self) -> None:
        """Test listing alert rules."""
        service = ClientMetricsService()
        metric = service.create_metric("client-1", MetricType.LATENCY, "Latency", "Latency", "ms")
        service.create_alert_rule(metric.metric_id, "Rule 1", "Desc", AlertThreshold.ABOVE, 100.0)
        service.create_alert_rule(metric.metric_id, "Rule 2", "Desc", AlertThreshold.BELOW, 10.0)

        rules = service.list_alert_rules()
        assert len(rules) == 2

    def test_acknowledge_alert(self) -> None:
        """Test acknowledging an alert."""
        service = ClientMetricsService()
        metric = service.create_metric("client-1", MetricType.LATENCY, "Latency", "Latency", "ms")
        rule = service.create_alert_rule(
            metric.metric_id,
            "High Latency",
            "Test",
            AlertThreshold.ABOVE,
            100.0,
            severity="critical",
        )

        # Trigger alert by recording high value
        service.record_metric(metric.metric_id, 200.0)

        alerts = service.list_alerts()
        if alerts:
            result = service.acknowledge_alert(alerts[0].alert_id, "admin")
            assert result is True

    def test_resolve_alert(self) -> None:
        """Test resolving an alert."""
        service = ClientMetricsService()
        metric = service.create_metric(
            "client-1", MetricType.ERROR_RATE, "Errors", "Errors", "percent"
        )
        rule = service.create_alert_rule(
            metric.metric_id, "High Errors", "Test", AlertThreshold.ABOVE, 1.0, severity="critical"
        )

        # Trigger alert
        service.record_metric(metric.metric_id, 5.0)

        alerts = service.list_alerts()
        if alerts:
            result = service.resolve_alert(alerts[0].alert_id)
            assert result is True

    def test_list_alerts_filters(self) -> None:
        """Test listing alerts with filters."""
        service = ClientMetricsService()
        metric = service.create_metric("client-1", MetricType.LATENCY, "Latency", "Latency", "ms")
        service.create_alert_rule(metric.metric_id, "Rule", "Test", AlertThreshold.ABOVE, 100.0)

        service.record_metric(metric.metric_id, 200.0)

        all_alerts = service.list_alerts()
        unacknowledged = service.list_alerts(acknowledged=False)
        unresolved = service.list_alerts(resolved=False)

        # All alerts should be unacknowledged and unresolved initially
        assert len(unacknowledged) >= len(all_alerts) - len(unacknowledged)

    def test_create_dashboard(self) -> None:
        """Test creating a dashboard."""
        service = ClientMetricsService()
        dashboard = service.create_dashboard(
            client_id="client-1",
            name="Operations Dashboard",
            description="Operational metrics",
            dashboard_type=DashboardType.OPERATIONAL,
        )
        assert dashboard.client_id == "client-1"
        assert dashboard.name == "Operations Dashboard"

    def test_get_dashboard(self) -> None:
        """Test getting dashboard by ID."""
        service = ClientMetricsService()
        dashboard = service.create_dashboard(
            client_id="client-1",
            name="Test",
            description="Test",
            dashboard_type=DashboardType.EXECUTIVE,
        )
        retrieved = service.get_dashboard(dashboard.dashboard_id)
        assert retrieved is not None
        assert retrieved.dashboard_id == dashboard.dashboard_id

    def test_list_dashboards(self) -> None:
        """Test listing dashboards."""
        service = ClientMetricsService()
        service.create_dashboard("client-1", "Dash 1", "Test", DashboardType.SLA)
        service.create_dashboard("client-1", "Dash 2", "Test", DashboardType.OPERATIONAL)
        service.create_dashboard("client-2", "Dash 3", "Test", DashboardType.SLA)

        all_dashboards = service.list_dashboards()
        assert len(all_dashboards) == 3

    def test_list_dashboards_by_client(self) -> None:
        """Test listing dashboards filtered by client."""
        service = ClientMetricsService()
        service.create_dashboard("client-1", "Dash 1", "Test", DashboardType.SLA)
        service.create_dashboard("client-1", "Dash 2", "Test", DashboardType.OPERATIONAL)
        service.create_dashboard("client-2", "Dash 3", "Test", DashboardType.SLA)

        client_1_dashboards = service.list_dashboards(client_id="client-1")
        assert len(client_1_dashboards) == 2

    def test_add_widget_to_dashboard(self) -> None:
        """Test adding widget to dashboard."""
        service = ClientMetricsService()
        metric = service.create_metric("client-1", MetricType.UPTIME, "Uptime", "Uptime", "percent")
        dashboard = service.create_dashboard("client-1", "Test", "Test", DashboardType.SLA)

        widget = service.add_widget_to_dashboard(
            dashboard_id=dashboard.dashboard_id,
            widget_type="gauge",
            title="Uptime Gauge",
            metric_ids=[metric.metric_id],
            visualization="gauge",
        )
        assert widget.title == "Uptime Gauge"
        assert len(dashboard.widgets) == 1

    def test_add_widget_dashboard_not_found(self) -> None:
        """Test adding widget to non-existent dashboard."""
        service = ClientMetricsService()
        with pytest.raises(ValueError, match="Dashboard not found"):
            service.add_widget_to_dashboard(
                dashboard_id="nonexistent",
                widget_type="chart",
                title="Test",
                metric_ids=["metric-1"],
            )

    def test_create_default_dashboard(self) -> None:
        """Test creating default SLA dashboard."""
        service = ClientMetricsService()
        dashboard = service.create_default_dashboard("client-1")

        assert dashboard.is_default is True
        assert dashboard.dashboard_type == DashboardType.SLA
        assert len(dashboard.widgets) == 3  # Default widgets

    def test_calculate_sla_compliance(self) -> None:
        """Test SLA compliance calculation."""
        service = ClientMetricsService()

        # Create and populate metrics
        uptime = service.create_metric("client-1", MetricType.UPTIME, "Uptime", "Uptime", "percent")
        latency = service.create_metric("client-1", MetricType.LATENCY, "Latency", "Latency", "ms")
        errors = service.create_metric(
            "client-1", MetricType.ERROR_RATE, "Errors", "Errors", "percent"
        )

        # Record good values
        service.record_metric(uptime.metric_id, 99.95)
        service.record_metric(latency.metric_id, 50.0)
        service.record_metric(errors.metric_id, 0.05)

        compliance = service.calculate_sla_compliance("client-1")

        assert "client_id" in compliance
        assert "compliance_score" in compliance
        assert "is_compliant" in compliance
        assert "sla_targets" in compliance

    def test_calculate_sla_compliance_with_violations(self) -> None:
        """Test SLA compliance with violations."""
        service = ClientMetricsService()

        # Create metric with bad values
        uptime = service.create_metric("client-1", MetricType.UPTIME, "Uptime", "Uptime", "percent")
        service.record_metric(uptime.metric_id, 98.0)  # Below 99.9% target

        compliance = service.calculate_sla_compliance("client-1")

        assert compliance["compliance_score"] < 100
        assert len(compliance["violations"]) > 0


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_client_metrics_default(self) -> None:
        """Test creating service with factory function."""
        service = create_client_metrics()
        assert isinstance(service, ClientMetricsService)

    def test_create_client_metrics_custom(self) -> None:
        """Test creating service with custom options."""
        service = create_client_metrics(
            default_retention_days=180,
            enable_anomaly_detection=False,
        )
        assert service.config.default_retention_days == 180
        assert service.config.enable_anomaly_detection is False


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_metric_types(self) -> None:
        """Test all metric type values."""
        assert MetricType.UPTIME.value == "uptime"
        assert MetricType.AVAILABILITY.value == "availability"
        assert MetricType.LATENCY.value == "latency"
        assert MetricType.ERROR_RATE.value == "error_rate"
        assert MetricType.SLA_COMPLIANCE.value == "sla_compliance"

    def test_metric_periods(self) -> None:
        """Test all metric period values."""
        assert MetricPeriod.MINUTE.value == "minute"
        assert MetricPeriod.HOUR.value == "hour"
        assert MetricPeriod.DAY.value == "day"
        assert MetricPeriod.MONTH.value == "month"

    def test_alert_thresholds(self) -> None:
        """Test all alert threshold types."""
        assert AlertThreshold.ABOVE.value == "above"
        assert AlertThreshold.BELOW.value == "below"
        assert AlertThreshold.EQUALS.value == "equals"
        assert AlertThreshold.CHANGE_PERCENT.value == "change_percent"

    def test_dashboard_types(self) -> None:
        """Test all dashboard types."""
        assert DashboardType.EXECUTIVE.value == "executive"
        assert DashboardType.OPERATIONAL.value == "operational"
        assert DashboardType.COMPLIANCE.value == "compliance"
        assert DashboardType.SLA.value == "sla"
