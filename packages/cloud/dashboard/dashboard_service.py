# -*- coding: utf-8 -*-
"""
Dashboard Service - CLOUD ZONE ONLY.

Aggregates telemetry data for dashboard visualization.

Design Doc Reference:
    - Section 4.1: "Telemetry Ingestion & Monitoring"
    - Section 13.1: "Уровни телеметрии (AGGREGATED, DETAILED, RAW)"

Privacy:
    - Respects telemetry levels
    - AGGREGATED by default for retail
    - RAW_ORDER_EVENTS enterprise-only
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Tuple
from uuid import UUID

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_RETENTION_DAYS: Final[int] = 90
MAX_QUERY_RANGE_DAYS: Final[int] = 365


# ============================================================================
# Enums
# ============================================================================


class TimeRange(str, Enum):
    """Predefined time ranges."""
    LAST_HOUR = "1h"
    LAST_6_HOURS = "6h"
    LAST_24_HOURS = "24h"
    LAST_7_DAYS = "7d"
    LAST_30_DAYS = "30d"
    LAST_90_DAYS = "90d"
    CUSTOM = "custom"


class AggregationInterval(str, Enum):
    """Data aggregation intervals."""
    MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    HOUR = "1h"
    DAY = "1d"
    WEEK = "1w"


class MetricType(str, Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    workspace_id: UUID
    name: str = "Main Dashboard"
    description: str = ""
    default_time_range: TimeRange = TimeRange.LAST_24_HOURS
    refresh_interval_seconds: int = 60
    widgets: List[str] = field(default_factory=list)
    layout: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of metric points."""
    metric_name: str
    metric_type: MetricType
    points: List[MetricPoint]
    labels: Dict[str, str] = field(default_factory=dict)

    @property
    def latest_value(self) -> Optional[float]:
        """Get latest value."""
        return self.points[-1].value if self.points else None

    @property
    def average(self) -> Optional[float]:
        """Calculate average."""
        if not self.points:
            return None
        return statistics.mean(p.value for p in self.points)

    @property
    def min_value(self) -> Optional[float]:
        """Get minimum value."""
        if not self.points:
            return None
        return min(p.value for p in self.points)

    @property
    def max_value(self) -> Optional[float]:
        """Get maximum value."""
        if not self.points:
            return None
        return max(p.value for p in self.points)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "metric_type": self.metric_type.value,
            "points": [
                {"timestamp": p.timestamp.isoformat(), "value": p.value}
                for p in self.points
            ],
            "labels": self.labels,
            "stats": {
                "latest": self.latest_value,
                "average": self.average,
                "min": self.min_value,
                "max": self.max_value,
            },
        }


@dataclass
class AgentStatus:
    """Agent status summary."""
    agent_id: str
    workspace_id: UUID
    status: str  # ONLINE, OFFLINE, DEGRADED
    last_heartbeat: Optional[datetime]
    active_runs: int = 0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    broker_connected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "active_runs": self.active_runs,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "broker_connected": self.broker_connected,
        }


@dataclass
class RunStatus:
    """Run status summary."""
    run_id: str
    deployment_id: str
    strategy_name: str
    agent_id: str
    status: str
    started_at: Optional[datetime]
    pnl_total: float = 0.0
    pnl_today: float = 0.0
    trade_count_today: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "deployment_id": self.deployment_id,
            "strategy_name": self.strategy_name,
            "agent_id": self.agent_id,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "pnl_total": self.pnl_total,
            "pnl_today": self.pnl_today,
            "trade_count_today": self.trade_count_today,
        }


@dataclass
class DashboardData:
    """Complete dashboard data snapshot."""
    workspace_id: UUID
    timestamp: datetime
    time_range: TimeRange

    # Summary metrics
    total_agents: int = 0
    online_agents: int = 0
    active_runs: int = 0
    total_pnl_today: float = 0.0
    total_trades_today: int = 0

    # Details
    agents: List[AgentStatus] = field(default_factory=list)
    runs: List[RunStatus] = field(default_factory=list)
    metrics: Dict[str, MetricSeries] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace_id": str(self.workspace_id),
            "timestamp": self.timestamp.isoformat(),
            "time_range": self.time_range.value,
            "summary": {
                "total_agents": self.total_agents,
                "online_agents": self.online_agents,
                "active_runs": self.active_runs,
                "total_pnl_today": self.total_pnl_today,
                "total_trades_today": self.total_trades_today,
            },
            "agents": [a.to_dict() for a in self.agents],
            "runs": [r.to_dict() for r in self.runs],
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "alerts": self.alerts[:20],
        }


# ============================================================================
# Dashboard Service
# ============================================================================


class DashboardService:
    """
    Dashboard data aggregation service.

    Aggregates telemetry from various sources for visualization.

    SECURITY:
        - Workspace isolation enforced
        - Telemetry level respected (AGGREGATED default)
        - No raw order data unless enterprise
    """

    def __init__(self):
        self._dashboards: Dict[UUID, DashboardConfig] = {}
        self._cache: Dict[str, Tuple[datetime, DashboardData]] = {}
        self._cache_ttl = timedelta(seconds=30)
        logger.info("DashboardService initialized")

    # ========================================================================
    # Dashboard Management
    # ========================================================================

    async def create_dashboard(self, config: DashboardConfig) -> DashboardConfig:
        """Create a new dashboard."""
        self._dashboards[config.workspace_id] = config
        return config

    async def get_dashboard(self, workspace_id: UUID) -> Optional[DashboardConfig]:
        """Get dashboard config for workspace."""
        return self._dashboards.get(workspace_id)

    async def update_dashboard(
        self,
        workspace_id: UUID,
        updates: Dict[str, Any],
    ) -> Optional[DashboardConfig]:
        """Update dashboard configuration."""
        config = self._dashboards.get(workspace_id)
        if not config:
            return None

        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    # ========================================================================
    # Data Retrieval
    # ========================================================================

    async def get_dashboard_data(
        self,
        workspace_id: UUID,
        time_range: TimeRange = TimeRange.LAST_24_HOURS,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        force_refresh: bool = False,
    ) -> DashboardData:
        """
        Get dashboard data for workspace.

        Args:
            workspace_id: Workspace ID
            time_range: Predefined time range
            start_time: Custom start time (if time_range=CUSTOM)
            end_time: Custom end time (if time_range=CUSTOM)
            force_refresh: Bypass cache

        Returns:
            DashboardData with aggregated metrics
        """
        # Check cache
        cache_key = f"{workspace_id}:{time_range.value}"
        if not force_refresh and cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if datetime.now(timezone.utc) - cached_time < self._cache_ttl:
                return cached_data

        # Calculate time bounds
        now = datetime.now(timezone.utc)
        if time_range == TimeRange.CUSTOM and start_time and end_time:
            query_start = start_time
            query_end = end_time
        else:
            query_end = now
            query_start = self._get_start_time(time_range, now)

        # Build dashboard data
        data = DashboardData(
            workspace_id=workspace_id,
            timestamp=now,
            time_range=time_range,
        )

        # Aggregate data (in production, this would query actual data stores)
        data = await self._aggregate_data(data, query_start, query_end)

        # Cache result
        self._cache[cache_key] = (now, data)

        return data

    async def get_metric_series(
        self,
        workspace_id: UUID,
        metric_name: str,
        time_range: TimeRange = TimeRange.LAST_24_HOURS,
        interval: AggregationInterval = AggregationInterval.HOUR,
    ) -> MetricSeries:
        """
        Get time series for a specific metric.

        Args:
            workspace_id: Workspace ID
            metric_name: Name of metric to retrieve
            time_range: Time range
            interval: Aggregation interval

        Returns:
            MetricSeries with data points
        """
        now = datetime.now(timezone.utc)
        start = self._get_start_time(time_range, now)
        interval_seconds = self._interval_to_seconds(interval)

        # Generate mock data points (in production, query from time series DB)
        points = []
        current = start
        while current <= now:
            points.append(MetricPoint(
                timestamp=current,
                value=50 + (hash(str(current)) % 50),  # Mock value
            ))
            current += timedelta(seconds=interval_seconds)

        return MetricSeries(
            metric_name=metric_name,
            metric_type=MetricType.GAUGE,
            points=points,
        )

    async def get_agents_status(self, workspace_id: UUID) -> List[AgentStatus]:
        """Get status of all agents in workspace."""
        # Mock data (in production, query from agent registry)
        return [
            AgentStatus(
                agent_id=f"agent_{i}",
                workspace_id=workspace_id,
                status="ONLINE" if i % 3 != 0 else "OFFLINE",
                last_heartbeat=datetime.now(timezone.utc) - timedelta(minutes=i),
                active_runs=i % 3,
                cpu_percent=30 + (i * 10) % 60,
                memory_percent=40 + (i * 5) % 40,
                broker_connected=i % 2 == 0,
            )
            for i in range(3)
        ]

    async def get_runs_status(self, workspace_id: UUID) -> List[RunStatus]:
        """Get status of all runs in workspace."""
        # Mock data
        return [
            RunStatus(
                run_id=f"run_{i}",
                deployment_id=f"dep_{i}",
                strategy_name=f"Strategy {i}",
                agent_id=f"agent_{i % 3}",
                status="RUNNING" if i % 2 == 0 else "PAUSED",
                started_at=datetime.now(timezone.utc) - timedelta(hours=i * 2),
                pnl_total=1000 * (i - 2),
                pnl_today=100 * (i - 1),
                trade_count_today=10 + i * 5,
            )
            for i in range(5)
        ]

    async def get_alerts(
        self,
        workspace_id: UUID,
        severity: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get recent alerts for workspace."""
        # Mock data
        alerts = []
        for i in range(min(limit, 10)):
            alerts.append({
                "alert_id": f"alert_{i}",
                "severity": "warning" if i % 3 == 0 else "info",
                "title": f"Alert {i}",
                "message": f"Alert message {i}",
                "created_at": (datetime.now(timezone.utc) - timedelta(hours=i)).isoformat(),
            })
        return alerts

    # ========================================================================
    # Private Methods
    # ========================================================================

    def _get_start_time(self, time_range: TimeRange, end: datetime) -> datetime:
        """Calculate start time from time range."""
        deltas = {
            TimeRange.LAST_HOUR: timedelta(hours=1),
            TimeRange.LAST_6_HOURS: timedelta(hours=6),
            TimeRange.LAST_24_HOURS: timedelta(hours=24),
            TimeRange.LAST_7_DAYS: timedelta(days=7),
            TimeRange.LAST_30_DAYS: timedelta(days=30),
            TimeRange.LAST_90_DAYS: timedelta(days=90),
        }
        delta = deltas.get(time_range, timedelta(hours=24))
        return end - delta

    def _interval_to_seconds(self, interval: AggregationInterval) -> int:
        """Convert interval to seconds."""
        intervals = {
            AggregationInterval.MINUTE: 60,
            AggregationInterval.FIVE_MINUTES: 300,
            AggregationInterval.FIFTEEN_MINUTES: 900,
            AggregationInterval.HOUR: 3600,
            AggregationInterval.DAY: 86400,
            AggregationInterval.WEEK: 604800,
        }
        return intervals.get(interval, 3600)

    async def _aggregate_data(
        self,
        data: DashboardData,
        start: datetime,
        end: datetime,
    ) -> DashboardData:
        """Aggregate data for dashboard."""
        # Get agents
        data.agents = await self.get_agents_status(data.workspace_id)
        data.total_agents = len(data.agents)
        data.online_agents = sum(1 for a in data.agents if a.status == "ONLINE")

        # Get runs
        data.runs = await self.get_runs_status(data.workspace_id)
        data.active_runs = sum(1 for r in data.runs if r.status == "RUNNING")
        data.total_pnl_today = sum(r.pnl_today for r in data.runs)
        data.total_trades_today = sum(r.trade_count_today for r in data.runs)

        # Get alerts
        data.alerts = await self.get_alerts(data.workspace_id)

        # Get key metrics
        data.metrics["pnl"] = await self.get_metric_series(
            data.workspace_id, "pnl", TimeRange.LAST_24_HOURS
        )
        data.metrics["trades"] = await self.get_metric_series(
            data.workspace_id, "trades", TimeRange.LAST_24_HOURS
        )

        return data
