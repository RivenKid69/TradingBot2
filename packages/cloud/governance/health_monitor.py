# -*- coding: utf-8 -*-
"""
Health Monitor Service.

CCEA Phase 8 Implementation.

This service provides health monitoring:
    - Agent health tracking
    - Dashboard data aggregation
    - State monitoring (online/offline, run state, halted)
    - Health alerts

Design Doc Reference:
    - Phase 8 (4.1/3.1): "Dashboards health and states"
    - "Agent online/offline, run state, halted reasons"

CLOUD ZONE ONLY.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, Final, List, Optional, Set
from uuid import uuid4


# ============================================================================
# Constants
# ============================================================================

# Agent is considered offline after this many seconds without heartbeat
OFFLINE_THRESHOLD_SECONDS: Final[int] = 120

# Warning threshold
WARNING_THRESHOLD_SECONDS: Final[int] = 60

# Health check interval
HEALTH_CHECK_INTERVAL_SECONDS: Final[int] = 30


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AgentStatus(Enum):
    """Agent connection status."""

    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class RunStatus(Enum):
    """Run status."""

    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    HALTED = "halted"
    STARTING = "starting"
    UNKNOWN = "unknown"


@dataclass
class AgentHealth:
    """
    Health status for a single agent.

    Attributes:
        agent_id: Agent identifier
        workspace_id: Workspace
        status: Connection status
        health: Overall health
        last_heartbeat: Last heartbeat time
        run_status: Current run status
        halt_reason: Reason if halted
        metrics: Health metrics
    """

    agent_id: str = ""
    workspace_id: str = ""
    status: AgentStatus = AgentStatus.UNKNOWN
    health: HealthStatus = HealthStatus.UNKNOWN
    last_heartbeat: Optional[datetime] = None
    run_status: RunStatus = RunStatus.UNKNOWN
    halt_reason: Optional[str] = None
    error_message: Optional[str] = None

    # Metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    error_count: int = 0
    order_count: int = 0
    latency_ms: float = 0.0

    # Timestamps
    status_changed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "workspace_id": self.workspace_id,
            "status": self.status.value,
            "health": self.health.value,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "run_status": self.run_status.value,
            "halt_reason": self.halt_reason,
            "error_message": self.error_message,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "error_count": self.error_count,
            "order_count": self.order_count,
            "latency_ms": self.latency_ms,
            "status_changed_at": (
                self.status_changed_at.isoformat() if self.status_changed_at else None
            ),
            "updated_at": self.updated_at.isoformat(),
        }

    @property
    def seconds_since_heartbeat(self) -> float:
        """Get seconds since last heartbeat."""
        if not self.last_heartbeat:
            return float("inf")
        return (datetime.utcnow() - self.last_heartbeat).total_seconds()


@dataclass
class HealthDashboard:
    """
    Health dashboard data.

    Aggregated health information for display.
    """

    workspace_id: str = ""
    generated_at: datetime = field(default_factory=datetime.utcnow)

    # Counts
    total_agents: int = 0
    online_agents: int = 0
    offline_agents: int = 0
    degraded_agents: int = 0

    # Health breakdown
    healthy_count: int = 0
    warning_count: int = 0
    critical_count: int = 0

    # Run status breakdown
    running_count: int = 0
    paused_count: int = 0
    stopped_count: int = 0
    halted_count: int = 0

    # Issues
    halted_agents: List[Dict[str, Any]] = field(default_factory=list)
    recent_errors: List[Dict[str, Any]] = field(default_factory=list)
    recent_alerts: List[Dict[str, Any]] = field(default_factory=list)

    # Metrics summary
    avg_cpu_usage: float = 0.0
    avg_memory_usage: float = 0.0
    total_orders_today: int = 0
    avg_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workspace_id": self.workspace_id,
            "generated_at": self.generated_at.isoformat(),
            "total_agents": self.total_agents,
            "online_agents": self.online_agents,
            "offline_agents": self.offline_agents,
            "degraded_agents": self.degraded_agents,
            "healthy_count": self.healthy_count,
            "warning_count": self.warning_count,
            "critical_count": self.critical_count,
            "running_count": self.running_count,
            "paused_count": self.paused_count,
            "stopped_count": self.stopped_count,
            "halted_count": self.halted_count,
            "halted_agents": self.halted_agents,
            "recent_errors": self.recent_errors,
            "recent_alerts": self.recent_alerts,
            "avg_cpu_usage": self.avg_cpu_usage,
            "avg_memory_usage": self.avg_memory_usage,
            "total_orders_today": self.total_orders_today,
            "avg_latency_ms": self.avg_latency_ms,
        }


@dataclass
class HealthEvent:
    """Health-related event."""

    id: str = field(default_factory=lambda: str(uuid4()))
    agent_id: str = ""
    workspace_id: str = ""
    event_type: str = ""
    severity: HealthStatus = HealthStatus.WARNING
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class HealthMonitorService:
    """
    Health monitoring service for agents.

    Tracks agent health and provides dashboard data.

    Usage:
        monitor = HealthMonitorService()

        # Update agent health
        monitor.update_agent_health(
            agent_id="agent-123",
            workspace_id="workspace-456",
            heartbeat_time=datetime.utcnow(),
            run_status=RunStatus.RUNNING,
            metrics={"cpu_usage": 45.0},
        )

        # Get dashboard
        dashboard = monitor.get_dashboard("workspace-456")

        # Check for issues
        issues = monitor.get_health_issues("workspace-456")

    MONITORING:
        - Agent connection status (online/offline)
        - Run state tracking
        - Health metrics aggregation
        - Alert generation
    """

    def __init__(
        self,
        on_status_change: Optional[Callable[[AgentHealth, AgentStatus, AgentStatus], None]] = None,
        on_health_event: Optional[Callable[[HealthEvent], None]] = None,
    ):
        """
        Initialize health monitor.

        Args:
            on_status_change: Callback for status changes (agent, old_status, new_status)
            on_health_event: Callback for health events
        """
        self._on_status_change = on_status_change
        self._on_health_event = on_health_event
        self._agents: Dict[str, AgentHealth] = {}
        self._events: List[HealthEvent] = []
        self._lock = threading.Lock()

    def update_agent_health(
        self,
        agent_id: str,
        workspace_id: str,
        heartbeat_time: Optional[datetime] = None,
        run_status: Optional[RunStatus] = None,
        halt_reason: Optional[str] = None,
        error_message: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> AgentHealth:
        """
        Update agent health information.

        Args:
            agent_id: Agent identifier
            workspace_id: Workspace
            heartbeat_time: Time of heartbeat
            run_status: Current run status
            halt_reason: Reason if halted
            error_message: Error if any
            metrics: Health metrics

        Returns:
            Updated AgentHealth
        """
        with self._lock:
            health = self._agents.get(agent_id)
            old_status = health.status if health else AgentStatus.UNKNOWN

            if not health:
                health = AgentHealth(
                    agent_id=agent_id,
                    workspace_id=workspace_id,
                )
                self._agents[agent_id] = health

            # Update heartbeat
            if heartbeat_time:
                health.last_heartbeat = heartbeat_time

            # Update run status
            if run_status:
                health.run_status = run_status

            # Update halt reason
            if halt_reason:
                health.halt_reason = halt_reason
                health.run_status = RunStatus.HALTED

            # Update error
            if error_message:
                health.error_message = error_message
                health.error_count += 1

            # Update metrics
            if metrics:
                if "cpu_usage" in metrics:
                    health.cpu_usage = metrics["cpu_usage"]
                if "memory_usage" in metrics:
                    health.memory_usage = metrics["memory_usage"]
                if "latency_ms" in metrics:
                    health.latency_ms = metrics["latency_ms"]
                if "order_count" in metrics:
                    health.order_count = int(metrics["order_count"])

            # Calculate status
            health.status = self._calculate_status(health)
            health.health = self._calculate_health(health)
            health.updated_at = datetime.utcnow()

            # Detect status change
            if health.status != old_status:
                health.status_changed_at = datetime.utcnow()
                self._emit_status_change(health, old_status, health.status)

            # Check for health events
            self._check_health_events(health)

            return health

    def record_heartbeat(self, agent_id: str, workspace_id: str) -> AgentHealth:
        """Record heartbeat from agent."""
        return self.update_agent_health(
            agent_id=agent_id,
            workspace_id=workspace_id,
            heartbeat_time=datetime.utcnow(),
        )

    def mark_agent_halted(
        self,
        agent_id: str,
        workspace_id: str,
        reason: str,
    ) -> AgentHealth:
        """Mark agent as halted."""
        health = self.update_agent_health(
            agent_id=agent_id,
            workspace_id=workspace_id,
            run_status=RunStatus.HALTED,
            halt_reason=reason,
        )

        # Generate event
        event = HealthEvent(
            agent_id=agent_id,
            workspace_id=workspace_id,
            event_type="agent_halted",
            severity=HealthStatus.CRITICAL,
            message=f"Agent halted: {reason}",
            details={"reason": reason},
        )
        self._record_event(event)

        return health

    def get_agent_health(self, agent_id: str) -> Optional[AgentHealth]:
        """Get health for specific agent."""
        with self._lock:
            return self._agents.get(agent_id)

    def get_workspace_agents(self, workspace_id: str) -> List[AgentHealth]:
        """Get all agents for a workspace."""
        with self._lock:
            return [h for h in self._agents.values() if h.workspace_id == workspace_id]

    def get_dashboard(self, workspace_id: str) -> HealthDashboard:
        """
        Get health dashboard for workspace.

        Args:
            workspace_id: Workspace identifier

        Returns:
            HealthDashboard
        """
        with self._lock:
            agents = [h for h in self._agents.values() if h.workspace_id == workspace_id]

            # Refresh status for all agents
            for agent in agents:
                agent.status = self._calculate_status(agent)
                agent.health = self._calculate_health(agent)

            dashboard = HealthDashboard(workspace_id=workspace_id)
            dashboard.total_agents = len(agents)

            # Status counts
            for agent in agents:
                if agent.status == AgentStatus.ONLINE:
                    dashboard.online_agents += 1
                elif agent.status == AgentStatus.OFFLINE:
                    dashboard.offline_agents += 1
                elif agent.status == AgentStatus.DEGRADED:
                    dashboard.degraded_agents += 1

            # Health counts
            for agent in agents:
                if agent.health == HealthStatus.HEALTHY:
                    dashboard.healthy_count += 1
                elif agent.health == HealthStatus.WARNING:
                    dashboard.warning_count += 1
                elif agent.health == HealthStatus.CRITICAL:
                    dashboard.critical_count += 1

            # Run status counts
            for agent in agents:
                if agent.run_status == RunStatus.RUNNING:
                    dashboard.running_count += 1
                elif agent.run_status == RunStatus.PAUSED:
                    dashboard.paused_count += 1
                elif agent.run_status == RunStatus.STOPPED:
                    dashboard.stopped_count += 1
                elif agent.run_status == RunStatus.HALTED:
                    dashboard.halted_count += 1
                    dashboard.halted_agents.append(
                        {
                            "agent_id": agent.agent_id,
                            "reason": agent.halt_reason,
                            "since": (
                                agent.status_changed_at.isoformat()
                                if agent.status_changed_at
                                else None
                            ),
                        }
                    )

            # Metrics averages
            if agents:
                dashboard.avg_cpu_usage = sum(a.cpu_usage for a in agents) / len(agents)
                dashboard.avg_memory_usage = sum(a.memory_usage for a in agents) / len(agents)
                dashboard.total_orders_today = sum(a.order_count for a in agents)
                latencies = [a.latency_ms for a in agents if a.latency_ms > 0]
                if latencies:
                    dashboard.avg_latency_ms = sum(latencies) / len(latencies)

            # Recent events
            workspace_events = [e for e in self._events if e.workspace_id == workspace_id]
            recent_errors = [
                {
                    "agent_id": e.agent_id,
                    "message": e.message,
                    "timestamp": e.timestamp.isoformat(),
                }
                for e in sorted(workspace_events, key=lambda x: x.timestamp, reverse=True)[:10]
                if e.severity in (HealthStatus.WARNING, HealthStatus.CRITICAL)
            ]
            dashboard.recent_errors = recent_errors

            return dashboard

    def get_health_issues(self, workspace_id: str) -> List[Dict[str, Any]]:
        """
        Get current health issues for workspace.

        Args:
            workspace_id: Workspace identifier

        Returns:
            List of issues
        """
        issues = []
        agents = self.get_workspace_agents(workspace_id)

        for agent in agents:
            # Offline agents
            if agent.status == AgentStatus.OFFLINE:
                issues.append(
                    {
                        "type": "agent_offline",
                        "severity": "critical",
                        "agent_id": agent.agent_id,
                        "message": f"Agent {agent.agent_id} is offline",
                        "since": (
                            agent.status_changed_at.isoformat() if agent.status_changed_at else None
                        ),
                    }
                )

            # Halted agents
            if agent.run_status == RunStatus.HALTED:
                issues.append(
                    {
                        "type": "agent_halted",
                        "severity": "critical",
                        "agent_id": agent.agent_id,
                        "message": f"Agent {agent.agent_id} is halted: {agent.halt_reason}",
                        "reason": agent.halt_reason,
                    }
                )

            # High resource usage
            if agent.cpu_usage > 90:
                issues.append(
                    {
                        "type": "high_cpu",
                        "severity": "warning",
                        "agent_id": agent.agent_id,
                        "message": f"Agent {agent.agent_id} CPU usage at {agent.cpu_usage:.1f}%",
                        "value": agent.cpu_usage,
                    }
                )

            if agent.memory_usage > 90:
                issues.append(
                    {
                        "type": "high_memory",
                        "severity": "warning",
                        "agent_id": agent.agent_id,
                        "message": f"Agent {agent.agent_id} memory usage at {agent.memory_usage:.1f}%",
                        "value": agent.memory_usage,
                    }
                )

            # High latency
            if agent.latency_ms > 1000:
                issues.append(
                    {
                        "type": "high_latency",
                        "severity": "warning",
                        "agent_id": agent.agent_id,
                        "message": f"Agent {agent.agent_id} latency at {agent.latency_ms:.0f}ms",
                        "value": agent.latency_ms,
                    }
                )

        return issues

    def _calculate_status(self, health: AgentHealth) -> AgentStatus:
        """Calculate agent connection status."""
        seconds = health.seconds_since_heartbeat

        if seconds == float("inf"):
            return AgentStatus.UNKNOWN
        elif seconds > OFFLINE_THRESHOLD_SECONDS:
            return AgentStatus.OFFLINE
        elif seconds > WARNING_THRESHOLD_SECONDS:
            return AgentStatus.DEGRADED
        else:
            return AgentStatus.ONLINE

    def _calculate_health(self, health: AgentHealth) -> HealthStatus:
        """Calculate overall health status."""
        # Critical conditions
        if health.status == AgentStatus.OFFLINE:
            return HealthStatus.CRITICAL
        if health.run_status == RunStatus.HALTED:
            return HealthStatus.CRITICAL
        if health.error_count > 10:
            return HealthStatus.CRITICAL

        # Warning conditions
        if health.status == AgentStatus.DEGRADED:
            return HealthStatus.WARNING
        if health.cpu_usage > 80:
            return HealthStatus.WARNING
        if health.memory_usage > 80:
            return HealthStatus.WARNING
        if health.latency_ms > 500:
            return HealthStatus.WARNING
        if health.error_count > 0:
            return HealthStatus.WARNING

        return HealthStatus.HEALTHY

    def _emit_status_change(
        self,
        health: AgentHealth,
        old_status: AgentStatus,
        new_status: AgentStatus,
    ) -> None:
        """Emit status change callback."""
        if self._on_status_change:
            try:
                self._on_status_change(health, old_status, new_status)
            except Exception:
                pass

        # Generate event
        severity = (
            HealthStatus.CRITICAL if new_status == AgentStatus.OFFLINE else HealthStatus.WARNING
        )
        event = HealthEvent(
            agent_id=health.agent_id,
            workspace_id=health.workspace_id,
            event_type="status_change",
            severity=severity,
            message=f"Agent status changed from {old_status.value} to {new_status.value}",
            details={"old_status": old_status.value, "new_status": new_status.value},
        )
        self._record_event(event)

    def _check_health_events(self, health: AgentHealth) -> None:
        """Check for health-related events."""
        # High CPU
        if health.cpu_usage > 90:
            event = HealthEvent(
                agent_id=health.agent_id,
                workspace_id=health.workspace_id,
                event_type="high_cpu",
                severity=HealthStatus.WARNING,
                message=f"CPU usage at {health.cpu_usage:.1f}%",
                details={"cpu_usage": health.cpu_usage},
            )
            self._record_event(event)

        # High memory
        if health.memory_usage > 90:
            event = HealthEvent(
                agent_id=health.agent_id,
                workspace_id=health.workspace_id,
                event_type="high_memory",
                severity=HealthStatus.WARNING,
                message=f"Memory usage at {health.memory_usage:.1f}%",
                details={"memory_usage": health.memory_usage},
            )
            self._record_event(event)

    def _record_event(self, event: HealthEvent) -> None:
        """Record health event."""
        self._events.append(event)

        # Keep only recent events
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self._events = [e for e in self._events if e.timestamp > cutoff]

        if self._on_health_event:
            try:
                self._on_health_event(event)
            except Exception:
                pass

    def get_events(
        self,
        workspace_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[HealthEvent]:
        """Get health events."""
        with self._lock:
            events = list(self._events)

            if workspace_id:
                events = [e for e in events if e.workspace_id == workspace_id]
            if agent_id:
                events = [e for e in events if e.agent_id == agent_id]

            return sorted(events, key=lambda x: x.timestamp, reverse=True)[:limit]
