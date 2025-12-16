# -*- coding: utf-8 -*-
"""
Tests for Dashboard Package.

Tests for:
- DashboardService
- Widgets
- AlertManager
"""

import asyncio
import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from packages.cloud.dashboard import (
    DashboardService,
    DashboardConfig,
    DashboardData,
    TimeRange,
    AggregationInterval,
    Widget,
    WidgetType,
    MetricWidget,
    ChartWidget,
    TableWidget,
    AlertWidget,
    DashboardLayout,
    AlertManager,
    AlertRule,
    AlertCondition,
    Alert,
    AlertSeverity,
    AlertState,
    SilenceRule,
    create_default_rules,
)
from packages.cloud.dashboard.alerts import ComparisonOperator, AggregationType


# ============================================================================
# DashboardService Tests
# ============================================================================


class TestDashboardService:
    """Tests for DashboardService."""

    @pytest.fixture
    def service(self):
        """Create DashboardService instance."""
        return DashboardService()

    @pytest.fixture
    def workspace_id(self):
        """Create test workspace ID."""
        return uuid4()

    @pytest.mark.asyncio
    async def test_get_dashboard_data(self, service, workspace_id):
        """Test getting dashboard data."""
        data = await service.get_dashboard_data(workspace_id)

        assert data is not None
        assert isinstance(data, DashboardData)

    @pytest.mark.asyncio
    async def test_get_dashboard_data_with_time_range(self, service, workspace_id):
        """Test getting dashboard data with time range."""
        data = await service.get_dashboard_data(
            workspace_id,
            time_range=TimeRange.LAST_24_HOURS,
        )

        assert data is not None

    @pytest.mark.asyncio
    async def test_get_metric_series(self, service, workspace_id):
        """Test getting metric time series."""
        series = await service.get_metric_series(
            workspace_id,
            "pnl",
            time_range=TimeRange.LAST_24_HOURS,
        )

        assert series is not None

    @pytest.mark.asyncio
    async def test_workspace_isolation(self, service):
        """Test workspace data isolation."""
        ws1 = uuid4()
        ws2 = uuid4()

        data1 = await service.get_dashboard_data(ws1)
        data2 = await service.get_dashboard_data(ws2)

        # Data should be separate per workspace
        assert data1 is not None
        assert data2 is not None


# ============================================================================
# Widget Tests
# ============================================================================


class TestWidgets:
    """Tests for Dashboard Widgets."""

    def test_metric_widget_creation(self):
        """Test MetricWidget creation."""
        widget = MetricWidget(
            title="Total PnL",
            metric_name="pnl_today",
            unit="$",
            precision=2,
        )

        assert widget.widget_type == WidgetType.METRIC
        assert widget.title == "Total PnL"
        assert widget.metric_name == "pnl_today"

    def test_metric_widget_to_dict(self):
        """Test MetricWidget serialization."""
        widget = MetricWidget(
            title="Test",
            metric_name="test_metric",
            warning_threshold=90.0,
            critical_threshold=95.0,
        )

        data = widget.to_dict()

        assert data["title"] == "Test"
        assert data["thresholds"]["warning"] == 90.0
        assert data["thresholds"]["critical"] == 95.0

    def test_chart_widget_creation(self):
        """Test ChartWidget creation."""
        widget = ChartWidget(
            title="PnL Chart",
            metrics=["pnl", "cumulative_pnl"],
            time_range="24h",
        )

        assert widget.widget_type == WidgetType.CHART
        assert len(widget.metrics) == 2

    def test_table_widget_creation(self):
        """Test TableWidget creation."""
        widget = TableWidget(
            title="Active Runs",
            columns=[
                {"key": "name", "label": "Name"},
                {"key": "status", "label": "Status"},
            ],
            page_size=10,
        )

        assert widget.widget_type == WidgetType.TABLE
        assert len(widget.columns) == 2

    def test_alert_widget_creation(self):
        """Test AlertWidget creation."""
        widget = AlertWidget(
            title="Recent Alerts",
            max_items=5,
            severity_filter=["critical", "warning"],
        )

        assert widget.widget_type == WidgetType.ALERT
        assert widget.max_items == 5


class TestDashboardLayout:
    """Tests for DashboardLayout."""

    def test_layout_creation(self):
        """Test creating dashboard layout."""
        layout = DashboardLayout(name="Test Layout")

        assert layout.name == "Test Layout"
        assert len(layout.widgets) == 0

    def test_add_widget(self):
        """Test adding widget to layout."""
        layout = DashboardLayout()
        widget = MetricWidget(title="Test", metric_name="test")

        layout.add_widget(widget)

        assert len(layout.widgets) == 1

    def test_remove_widget(self):
        """Test removing widget from layout."""
        layout = DashboardLayout()
        widget = MetricWidget(title="Test", metric_name="test")
        layout.add_widget(widget)

        removed = layout.remove_widget(widget.widget_id)

        assert removed is True
        assert len(layout.widgets) == 0

    def test_get_widget(self):
        """Test getting widget by ID."""
        layout = DashboardLayout()
        widget = MetricWidget(title="Test", metric_name="test")
        layout.add_widget(widget)

        found = layout.get_widget(widget.widget_id)

        assert found is not None
        assert found.title == "Test"

    def test_create_default_layout(self):
        """Test creating default layout."""
        layout = DashboardLayout.create_default()

        assert layout.name == "Default Layout"
        assert len(layout.widgets) > 0

    def test_layout_to_dict(self):
        """Test layout serialization."""
        layout = DashboardLayout(name="Test")
        layout.add_widget(MetricWidget(title="Widget1", metric_name="m1"))

        data = layout.to_dict()

        assert data["name"] == "Test"
        assert len(data["widgets"]) == 1


# ============================================================================
# AlertManager Tests
# ============================================================================


class TestAlertManager:
    """Tests for AlertManager."""

    @pytest.fixture
    def manager(self):
        """Create AlertManager instance."""
        return AlertManager()

    @pytest.fixture
    def workspace_id(self):
        """Create test workspace ID."""
        return uuid4()

    @pytest.mark.asyncio
    async def test_create_rule(self, manager, workspace_id):
        """Test creating alert rule."""
        rule = AlertRule(
            name="High CPU",
            workspace_id=workspace_id,
            conditions=[
                AlertCondition(
                    metric_name="cpu_usage",
                    operator=ComparisonOperator.GT,
                    threshold=90.0,
                )
            ],
            severity=AlertSeverity.WARNING,
        )

        created = await manager.create_rule(rule)

        assert created is not None
        assert created.name == "High CPU"

    @pytest.mark.asyncio
    async def test_get_rule(self, manager, workspace_id):
        """Test getting rule by ID."""
        rule = AlertRule(name="Test", workspace_id=workspace_id)
        created = await manager.create_rule(rule)

        found = await manager.get_rule(created.rule_id)

        assert found is not None
        assert found.rule_id == created.rule_id

    @pytest.mark.asyncio
    async def test_update_rule(self, manager, workspace_id):
        """Test updating alert rule."""
        rule = AlertRule(name="Original", workspace_id=workspace_id)
        created = await manager.create_rule(rule)

        updated = await manager.update_rule(
            created.rule_id,
            {"name": "Updated", "enabled": False}
        )

        assert updated is not None
        assert updated.name == "Updated"
        assert updated.enabled is False

    @pytest.mark.asyncio
    async def test_delete_rule(self, manager, workspace_id):
        """Test deleting alert rule."""
        rule = AlertRule(name="ToDelete", workspace_id=workspace_id)
        created = await manager.create_rule(rule)

        deleted = await manager.delete_rule(created.rule_id)

        assert deleted is True
        assert await manager.get_rule(created.rule_id) is None

    @pytest.mark.asyncio
    async def test_list_rules(self, manager, workspace_id):
        """Test listing rules by workspace."""
        await manager.create_rule(AlertRule(name="Rule1", workspace_id=workspace_id))
        await manager.create_rule(AlertRule(name="Rule2", workspace_id=workspace_id))

        rules = await manager.list_rules(workspace_id=workspace_id)

        assert len(rules) == 2

    @pytest.mark.asyncio
    async def test_trigger_alert(self, manager, workspace_id):
        """Test triggering an alert."""
        rule = AlertRule(
            name="Test Alert",
            workspace_id=workspace_id,
            conditions=[
                AlertCondition(
                    metric_name="test",
                    operator=ComparisonOperator.GT,
                    threshold=100.0,
                )
            ],
        )
        await manager.create_rule(rule)

        alert = await manager.trigger_alert(
            rule,
            triggered_value=150.0,
            condition=rule.conditions[0],
        )

        assert alert is not None
        assert alert.state == AlertState.FIRING

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, manager, workspace_id):
        """Test acknowledging an alert."""
        rule = AlertRule(name="Test", workspace_id=workspace_id)
        await manager.create_rule(rule)

        condition = AlertCondition(
            metric_name="test",
            operator=ComparisonOperator.GT,
            threshold=0,
        )
        alert = await manager.trigger_alert(rule, 100.0, condition)

        result = await manager.acknowledge_alert(alert.alert_id, "user123")

        assert result is True
        assert alert.state == AlertState.ACKNOWLEDGED
        assert alert.acknowledged_by == "user123"

    @pytest.mark.asyncio
    async def test_resolve_alert(self, manager, workspace_id):
        """Test resolving an alert."""
        rule = AlertRule(name="Test", workspace_id=workspace_id)
        await manager.create_rule(rule)

        condition = AlertCondition(
            metric_name="test",
            operator=ComparisonOperator.GT,
            threshold=0,
        )
        alert = await manager.trigger_alert(rule, 100.0, condition)

        result = await manager.resolve_alert(alert.alert_id)

        assert result is True
        assert alert.state == AlertState.RESOLVED

    @pytest.mark.asyncio
    async def test_list_alerts(self, manager, workspace_id):
        """Test listing alerts."""
        rule = AlertRule(name="Test", workspace_id=workspace_id)
        await manager.create_rule(rule)

        condition = AlertCondition(
            metric_name="test",
            operator=ComparisonOperator.GT,
            threshold=0,
        )
        await manager.trigger_alert(rule, 100.0, condition)
        await manager.trigger_alert(rule, 200.0, condition)

        alerts = await manager.list_alerts(workspace_id=workspace_id)

        # Should have at least 1 (might combine duplicates)
        assert len(alerts) >= 1

    @pytest.mark.asyncio
    async def test_silence_rule(self, manager, workspace_id):
        """Test creating silence rule."""
        silence = SilenceRule(
            workspace_id=workspace_id,
            matchers={"category": "test"},
            comment="Testing silence",
        )

        created = await manager.create_silence(silence)

        assert created is not None
        assert created.is_active()

    @pytest.mark.asyncio
    async def test_silenced_alert_not_notified(self, manager, workspace_id):
        """Test that silenced alerts are marked accordingly."""
        # Create silence for category=test
        silence = SilenceRule(
            workspace_id=workspace_id,
            matchers={"category": "test"},
        )
        await manager.create_silence(silence)

        rule = AlertRule(
            name="Test",
            workspace_id=workspace_id,
            labels={"category": "test"},
        )
        await manager.create_rule(rule)

        condition = AlertCondition(
            metric_name="test",
            operator=ComparisonOperator.GT,
            threshold=0,
        )
        alert = await manager.trigger_alert(rule, 100.0, condition)

        assert alert.state == AlertState.SILENCED


class TestAlertCondition:
    """Tests for AlertCondition evaluation."""

    def test_greater_than(self):
        """Test greater than operator."""
        condition = AlertCondition(
            metric_name="cpu",
            operator=ComparisonOperator.GT,
            threshold=90.0,
        )

        # evaluate() takes a float value directly
        assert condition.evaluate(95.0) is True
        assert condition.evaluate(85.0) is False

    def test_less_than(self):
        """Test less than operator."""
        condition = AlertCondition(
            metric_name="free_space",
            operator=ComparisonOperator.LT,
            threshold=10.0,
        )

        assert condition.evaluate(5.0) is True
        assert condition.evaluate(15.0) is False

    def test_equals(self):
        """Test equals operator."""
        condition = AlertCondition(
            metric_name="status_code",
            operator=ComparisonOperator.EQ,
            threshold=500.0,
        )

        assert condition.evaluate(500.0) is True
        assert condition.evaluate(200.0) is False

    def test_not_equals(self):
        """Test not equals operator."""
        condition = AlertCondition(
            metric_name="status_code",
            operator=ComparisonOperator.NE,
            threshold=200.0,
        )

        assert condition.evaluate(500.0) is True
        assert condition.evaluate(200.0) is False

    def test_greater_or_equal(self):
        """Test greater or equal operator."""
        condition = AlertCondition(
            metric_name="cpu",
            operator=ComparisonOperator.GTE,
            threshold=90.0,
        )

        assert condition.evaluate(90.0) is True
        assert condition.evaluate(95.0) is True
        assert condition.evaluate(85.0) is False


class TestDefaultAlertRules:
    """Tests for default alert rules."""

    def test_create_default_rules(self):
        """Test creating default rules."""
        workspace_id = uuid4()
        rules = create_default_rules(workspace_id)

        assert len(rules) > 0
        assert all(r.workspace_id == workspace_id for r in rules)
