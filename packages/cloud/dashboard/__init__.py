# -*- coding: utf-8 -*-
"""
Dashboard Package - CLOUD ZONE ONLY.

Telemetry monitoring, visualization, and alerting dashboard.

Design Doc Reference:
    - Section 4.1: "Telemetry Ingestion & Monitoring"
    - Section 13: "Telemetry: уровни чувствительности"

Components:
    - dashboard_service: Dashboard data aggregation
    - widgets: Dashboard widgets for visualization
    - alerts: Alert management and routing
    - api: REST API endpoints for dashboard
"""

from .dashboard_service import (
    DashboardService,
    DashboardConfig,
    DashboardData,
    TimeRange,
    AggregationInterval,
)
from .widgets import (
    Widget,
    WidgetType,
    MetricWidget,
    ChartWidget,
    TableWidget,
    AlertWidget,
    DashboardLayout,
)
from .alerts import (
    AlertManager,
    AlertRule,
    AlertCondition,
    Alert,
    AlertSeverity,
    AlertState,
    SilenceRule,
    create_default_rules,
)

__all__ = [
    "DashboardService",
    "DashboardConfig",
    "DashboardData",
    "TimeRange",
    "AggregationInterval",
    "Widget",
    "WidgetType",
    "MetricWidget",
    "ChartWidget",
    "TableWidget",
    "AlertWidget",
    "DashboardLayout",
    "AlertManager",
    "AlertRule",
    "AlertCondition",
    "Alert",
    "AlertSeverity",
    "AlertState",
    "SilenceRule",
    "create_default_rules",
]
