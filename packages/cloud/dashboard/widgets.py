# -*- coding: utf-8 -*-
"""
Dashboard Widgets - CLOUD ZONE ONLY.

Widget definitions for dashboard visualization.

Design Doc Reference:
    - Section 4.1: "Telemetry Ingestion & Monitoring"
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Final, List, Optional
from uuid import UUID


# ============================================================================
# Enums
# ============================================================================


class WidgetType(str, Enum):
    """Widget types."""

    METRIC = "metric"
    CHART = "chart"
    TABLE = "table"
    ALERT = "alert"
    STATUS = "status"
    TEXT = "text"
    GAUGE = "gauge"
    MAP = "map"


class ChartType(str, Enum):
    """Chart visualization types."""

    LINE = "line"
    BAR = "bar"
    AREA = "area"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"


# ============================================================================
# Base Widget
# ============================================================================


@dataclass
class Widget:
    """Base widget class."""

    widget_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    widget_type: WidgetType = WidgetType.METRIC
    title: str = ""
    description: str = ""

    # Layout
    x: int = 0  # Grid column
    y: int = 0  # Grid row
    width: int = 4  # Columns to span (out of 12)
    height: int = 2  # Rows to span

    # Data source
    metric_name: Optional[str] = None
    query: Optional[str] = None
    refresh_interval: int = 60  # seconds

    # Styling
    theme: str = "default"
    custom_css: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "widget_id": self.widget_id,
            "widget_type": self.widget_type.value,
            "title": self.title,
            "description": self.description,
            "layout": {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
            },
            "metric_name": self.metric_name,
            "query": self.query,
            "refresh_interval": self.refresh_interval,
        }


# ============================================================================
# Specific Widgets
# ============================================================================


@dataclass
class MetricWidget(Widget):
    """
    Single metric display widget.

    Shows a single value with optional trend indicator.
    """

    widget_type: WidgetType = field(default=WidgetType.METRIC, init=False)

    # Display options
    unit: str = ""
    precision: int = 2
    show_trend: bool = True
    trend_period: str = "24h"

    # Thresholds for coloring
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    threshold_direction: str = "above"  # above or below

    # Value formatting
    format_type: str = "number"  # number, percent, currency, duration

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "unit": self.unit,
                "precision": self.precision,
                "show_trend": self.show_trend,
                "trend_period": self.trend_period,
                "thresholds": {
                    "warning": self.warning_threshold,
                    "critical": self.critical_threshold,
                    "direction": self.threshold_direction,
                },
                "format_type": self.format_type,
            }
        )
        return base


@dataclass
class ChartWidget(Widget):
    """
    Chart widget for time series visualization.

    Supports line, bar, area, pie charts.
    """

    widget_type: WidgetType = field(default=WidgetType.CHART, init=False)

    # Chart options
    chart_type: ChartType = ChartType.LINE
    metrics: List[str] = field(default_factory=list)
    time_range: str = "24h"
    aggregation: str = "1h"

    # Axis options
    y_axis_label: str = ""
    y_axis_min: Optional[float] = None
    y_axis_max: Optional[float] = None
    show_legend: bool = True
    show_grid: bool = True

    # Series styling
    colors: List[str] = field(default_factory=list)
    fill_opacity: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "chart_type": self.chart_type.value,
                "metrics": self.metrics,
                "time_range": self.time_range,
                "aggregation": self.aggregation,
                "y_axis": {
                    "label": self.y_axis_label,
                    "min": self.y_axis_min,
                    "max": self.y_axis_max,
                },
                "show_legend": self.show_legend,
                "show_grid": self.show_grid,
                "colors": self.colors,
                "fill_opacity": self.fill_opacity,
            }
        )
        return base


@dataclass
class TableWidget(Widget):
    """
    Table widget for tabular data display.

    Shows lists of items with sorting and filtering.
    """

    widget_type: WidgetType = field(default=WidgetType.TABLE, init=False)

    # Column definitions
    columns: List[Dict[str, Any]] = field(default_factory=list)
    # Each column: {"key": "name", "label": "Name", "sortable": True, "width": 100}

    # Table options
    page_size: int = 10
    sortable: bool = True
    filterable: bool = True
    show_pagination: bool = True

    # Row styling
    row_highlight_field: Optional[str] = None
    row_highlight_rules: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "columns": self.columns,
                "page_size": self.page_size,
                "sortable": self.sortable,
                "filterable": self.filterable,
                "show_pagination": self.show_pagination,
                "row_highlight": {
                    "field": self.row_highlight_field,
                    "rules": self.row_highlight_rules,
                },
            }
        )
        return base


@dataclass
class AlertWidget(Widget):
    """
    Alert feed widget.

    Shows recent alerts with filtering by severity.
    """

    widget_type: WidgetType = field(default=WidgetType.ALERT, init=False)

    # Filter options
    severity_filter: List[str] = field(default_factory=lambda: ["critical", "warning", "info"])
    max_items: int = 10
    show_resolved: bool = False

    # Display options
    show_timestamp: bool = True
    show_source: bool = True
    compact_mode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "severity_filter": self.severity_filter,
                "max_items": self.max_items,
                "show_resolved": self.show_resolved,
                "show_timestamp": self.show_timestamp,
                "show_source": self.show_source,
                "compact_mode": self.compact_mode,
            }
        )
        return base


# ============================================================================
# Dashboard Layout
# ============================================================================


@dataclass
class DashboardLayout:
    """
    Dashboard layout configuration.

    Defines grid-based widget arrangement.
    """

    layout_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Default Layout"

    # Grid configuration
    columns: int = 12  # Total columns
    row_height: int = 100  # Pixels per row
    margin: int = 10  # Pixels between widgets

    # Widgets
    widgets: List[Widget] = field(default_factory=list)

    # Breakpoints for responsive design
    breakpoints: Dict[str, int] = field(
        default_factory=lambda: {
            "lg": 1200,
            "md": 996,
            "sm": 768,
            "xs": 480,
        }
    )

    def add_widget(self, widget: Widget) -> None:
        """Add widget to layout."""
        self.widgets.append(widget)

    def remove_widget(self, widget_id: str) -> bool:
        """Remove widget by ID."""
        for i, w in enumerate(self.widgets):
            if w.widget_id == widget_id:
                self.widgets.pop(i)
                return True
        return False

    def get_widget(self, widget_id: str) -> Optional[Widget]:
        """Get widget by ID."""
        for w in self.widgets:
            if w.widget_id == widget_id:
                return w
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layout_id": self.layout_id,
            "name": self.name,
            "grid": {
                "columns": self.columns,
                "row_height": self.row_height,
                "margin": self.margin,
            },
            "widgets": [w.to_dict() for w in self.widgets],
            "breakpoints": self.breakpoints,
        }

    @classmethod
    def create_default(cls) -> "DashboardLayout":
        """Create default dashboard layout."""
        layout = cls(name="Default Layout")

        # Add default widgets
        layout.add_widget(
            MetricWidget(
                title="Total PnL Today",
                metric_name="pnl_today",
                x=0,
                y=0,
                width=3,
                height=2,
                format_type="currency",
                warning_threshold=-1000,
                critical_threshold=-5000,
                threshold_direction="below",
            )
        )

        layout.add_widget(
            MetricWidget(
                title="Active Runs",
                metric_name="active_runs",
                x=3,
                y=0,
                width=3,
                height=2,
            )
        )

        layout.add_widget(
            MetricWidget(
                title="Online Agents",
                metric_name="online_agents",
                x=6,
                y=0,
                width=3,
                height=2,
            )
        )

        layout.add_widget(
            MetricWidget(
                title="Trades Today",
                metric_name="trades_today",
                x=9,
                y=0,
                width=3,
                height=2,
            )
        )

        layout.add_widget(
            ChartWidget(
                title="PnL Over Time",
                metrics=["pnl"],
                chart_type=ChartType.LINE,
                x=0,
                y=2,
                width=8,
                height=4,
                time_range="24h",
                aggregation="1h",
            )
        )

        layout.add_widget(
            AlertWidget(
                title="Recent Alerts",
                x=8,
                y=2,
                width=4,
                height=4,
                max_items=5,
            )
        )

        layout.add_widget(
            TableWidget(
                title="Active Runs",
                query="runs",
                x=0,
                y=6,
                width=12,
                height=4,
                columns=[
                    {"key": "strategy_name", "label": "Strategy", "sortable": True},
                    {"key": "agent_id", "label": "Agent", "sortable": True},
                    {"key": "status", "label": "Status", "sortable": True},
                    {"key": "pnl_today", "label": "PnL Today", "sortable": True},
                    {"key": "trade_count_today", "label": "Trades", "sortable": True},
                ],
            )
        )

        return layout
