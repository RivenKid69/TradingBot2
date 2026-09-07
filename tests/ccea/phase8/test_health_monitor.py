# -*- coding: utf-8 -*-
"""
Tests for HealthMonitorService.

CCEA Phase 8 - Agent health monitoring tests.
"""

import pytest
from datetime import datetime, timedelta

from packages.cloud.governance.health_monitor import (
    HealthMonitorService,
    AgentHealth,
    HealthDashboard,
    HealthEvent,
    HealthStatus,
    AgentStatus,
    RunStatus,
    OFFLINE_THRESHOLD_SECONDS,
    WARNING_THRESHOLD_SECONDS,
)


class TestHealthMonitorServiceBasic:
    """Basic health monitor tests."""

    def test_create_monitor(self):
        """Test creating health monitor."""
        monitor = HealthMonitorService()
        assert monitor is not None

    def test_create_with_callbacks(self):
        """Test creating with callbacks."""
        status_changes = []
        events = []

        monitor = HealthMonitorService(
            on_status_change=lambda h, o, n: status_changes.append((h, o, n)),
            on_health_event=lambda e: events.append(e),
        )

        assert monitor._on_status_change is not None
        assert monitor._on_health_event is not None


class TestAgentHealthUpdate:
    """Agent health update tests."""

    def test_update_agent_health(self):
        """Test updating agent health."""
        monitor = HealthMonitorService()

        health = monitor.update_agent_health(
            agent_id="agent-123",
            workspace_id="ws-456",
            heartbeat_time=datetime.utcnow(),
            run_status=RunStatus.RUNNING,
        )

        assert health.agent_id == "agent-123"
        assert health.workspace_id == "ws-456"
        assert health.run_status == RunStatus.RUNNING
        assert health.status == AgentStatus.ONLINE

    def test_update_with_metrics(self):
        """Test updating with metrics."""
        monitor = HealthMonitorService()

        health = monitor.update_agent_health(
            agent_id="agent-123",
            workspace_id="ws-456",
            heartbeat_time=datetime.utcnow(),
            metrics={
                "cpu_usage": 45.5,
                "memory_usage": 60.0,
                "latency_ms": 100.0,
            },
        )

        assert health.cpu_usage == 45.5
        assert health.memory_usage == 60.0
        assert health.latency_ms == 100.0

    def test_record_heartbeat(self):
        """Test recording heartbeat."""
        monitor = HealthMonitorService()

        health = monitor.record_heartbeat("agent-123", "ws-456")

        assert health.last_heartbeat is not None
        assert health.status == AgentStatus.ONLINE


class TestAgentStatusCalculation:
    """Agent status calculation tests."""

    def test_status_online(self):
        """Test status is online with recent heartbeat."""
        monitor = HealthMonitorService()

        health = monitor.update_agent_health(
            agent_id="agent-123",
            workspace_id="ws-456",
            heartbeat_time=datetime.utcnow(),
        )

        assert health.status == AgentStatus.ONLINE

    def test_status_degraded(self):
        """Test status is degraded with delayed heartbeat."""
        monitor = HealthMonitorService()

        # Heartbeat older than warning threshold
        old_time = datetime.utcnow() - timedelta(seconds=WARNING_THRESHOLD_SECONDS + 10)

        health = monitor.update_agent_health(
            agent_id="agent-123",
            workspace_id="ws-456",
            heartbeat_time=old_time,
        )

        assert health.status == AgentStatus.DEGRADED

    def test_status_offline(self):
        """Test status is offline with very old heartbeat."""
        monitor = HealthMonitorService()

        # Heartbeat older than offline threshold
        old_time = datetime.utcnow() - timedelta(seconds=OFFLINE_THRESHOLD_SECONDS + 10)

        health = monitor.update_agent_health(
            agent_id="agent-123",
            workspace_id="ws-456",
            heartbeat_time=old_time,
        )

        assert health.status == AgentStatus.OFFLINE


class TestHealthStatusCalculation:
    """Health status calculation tests."""

    def test_health_healthy(self):
        """Test health is healthy under normal conditions."""
        monitor = HealthMonitorService()

        health = monitor.update_agent_health(
            agent_id="agent-123",
            workspace_id="ws-456",
            heartbeat_time=datetime.utcnow(),
            run_status=RunStatus.RUNNING,
            metrics={"cpu_usage": 30, "memory_usage": 40},
        )

        assert health.health == HealthStatus.HEALTHY

    def test_health_warning_high_cpu(self):
        """Test health is warning with high CPU."""
        monitor = HealthMonitorService()

        health = monitor.update_agent_health(
            agent_id="agent-123",
            workspace_id="ws-456",
            heartbeat_time=datetime.utcnow(),
            metrics={"cpu_usage": 85},  # Above 80%
        )

        assert health.health == HealthStatus.WARNING

    def test_health_warning_high_memory(self):
        """Test health is warning with high memory."""
        monitor = HealthMonitorService()

        health = monitor.update_agent_health(
            agent_id="agent-123",
            workspace_id="ws-456",
            heartbeat_time=datetime.utcnow(),
            metrics={"memory_usage": 85},  # Above 80%
        )

        assert health.health == HealthStatus.WARNING

    def test_health_critical_offline(self):
        """Test health is critical when offline."""
        monitor = HealthMonitorService()

        old_time = datetime.utcnow() - timedelta(seconds=OFFLINE_THRESHOLD_SECONDS + 10)

        health = monitor.update_agent_health(
            agent_id="agent-123",
            workspace_id="ws-456",
            heartbeat_time=old_time,
        )

        assert health.health == HealthStatus.CRITICAL

    def test_health_critical_halted(self):
        """Test health is critical when halted."""
        monitor = HealthMonitorService()

        health = monitor.update_agent_health(
            agent_id="agent-123",
            workspace_id="ws-456",
            heartbeat_time=datetime.utcnow(),
            run_status=RunStatus.HALTED,
            halt_reason="Risk limit exceeded",
        )

        assert health.health == HealthStatus.CRITICAL


class TestMarkAgentHalted:
    """Mark agent halted tests."""

    def test_mark_halted(self):
        """Test marking agent as halted."""
        monitor = HealthMonitorService()

        health = monitor.mark_agent_halted(
            agent_id="agent-123",
            workspace_id="ws-456",
            reason="Risk limit exceeded",
        )

        assert health.run_status == RunStatus.HALTED
        assert health.halt_reason == "Risk limit exceeded"
        assert health.health == HealthStatus.CRITICAL


class TestAgentQueries:
    """Agent query tests."""

    def test_get_agent_health(self):
        """Test getting agent health."""
        monitor = HealthMonitorService()

        monitor.update_agent_health("agent-123", "ws-456", datetime.utcnow())

        health = monitor.get_agent_health("agent-123")

        assert health is not None
        assert health.agent_id == "agent-123"

    def test_get_workspace_agents(self):
        """Test getting all agents for workspace."""
        monitor = HealthMonitorService()

        monitor.update_agent_health("agent-1", "ws-456", datetime.utcnow())
        monitor.update_agent_health("agent-2", "ws-456", datetime.utcnow())
        monitor.update_agent_health("agent-3", "ws-other", datetime.utcnow())

        agents = monitor.get_workspace_agents("ws-456")

        assert len(agents) == 2


class TestDashboard:
    """Dashboard tests."""

    def test_get_dashboard(self):
        """Test getting health dashboard."""
        monitor = HealthMonitorService()

        # Add some agents
        monitor.update_agent_health("agent-1", "ws-456", datetime.utcnow(), RunStatus.RUNNING)
        monitor.update_agent_health("agent-2", "ws-456", datetime.utcnow(), RunStatus.PAUSED)

        dashboard = monitor.get_dashboard("ws-456")

        assert dashboard.workspace_id == "ws-456"
        assert dashboard.total_agents == 2
        assert dashboard.running_count == 1
        assert dashboard.paused_count == 1

    def test_dashboard_counts_status(self):
        """Test dashboard counts agent status."""
        monitor = HealthMonitorService()

        # Online agent
        monitor.update_agent_health("agent-1", "ws-456", datetime.utcnow())

        # Offline agent
        old_time = datetime.utcnow() - timedelta(seconds=OFFLINE_THRESHOLD_SECONDS + 10)
        monitor.update_agent_health("agent-2", "ws-456", old_time)

        dashboard = monitor.get_dashboard("ws-456")

        assert dashboard.online_agents == 1
        assert dashboard.offline_agents == 1

    def test_dashboard_halted_agents(self):
        """Test dashboard includes halted agents."""
        monitor = HealthMonitorService()

        monitor.mark_agent_halted("agent-1", "ws-456", "Risk limit")

        dashboard = monitor.get_dashboard("ws-456")

        assert dashboard.halted_count == 1
        assert len(dashboard.halted_agents) == 1
        assert dashboard.halted_agents[0]["reason"] == "Risk limit"

    def test_dashboard_metrics_averages(self):
        """Test dashboard calculates metric averages."""
        monitor = HealthMonitorService()

        monitor.update_agent_health(
            "agent-1",
            "ws-456",
            datetime.utcnow(),
            metrics={"cpu_usage": 40, "memory_usage": 50},
        )
        monitor.update_agent_health(
            "agent-2",
            "ws-456",
            datetime.utcnow(),
            metrics={"cpu_usage": 60, "memory_usage": 70},
        )

        dashboard = monitor.get_dashboard("ws-456")

        assert dashboard.avg_cpu_usage == 50.0  # (40 + 60) / 2
        assert dashboard.avg_memory_usage == 60.0  # (50 + 70) / 2


class TestHealthIssues:
    """Health issues tests."""

    def test_get_health_issues_offline(self):
        """Test getting offline agent issues."""
        monitor = HealthMonitorService()

        old_time = datetime.utcnow() - timedelta(seconds=OFFLINE_THRESHOLD_SECONDS + 10)
        monitor.update_agent_health("agent-1", "ws-456", old_time)

        issues = monitor.get_health_issues("ws-456")

        assert len(issues) >= 1
        assert any(i["type"] == "agent_offline" for i in issues)

    def test_get_health_issues_halted(self):
        """Test getting halted agent issues."""
        monitor = HealthMonitorService()

        monitor.mark_agent_halted("agent-1", "ws-456", "Risk limit")

        issues = monitor.get_health_issues("ws-456")

        assert any(i["type"] == "agent_halted" for i in issues)

    def test_get_health_issues_high_cpu(self):
        """Test getting high CPU issues."""
        monitor = HealthMonitorService()

        monitor.update_agent_health(
            "agent-1",
            "ws-456",
            datetime.utcnow(),
            metrics={"cpu_usage": 95},
        )

        issues = monitor.get_health_issues("ws-456")

        assert any(i["type"] == "high_cpu" for i in issues)

    def test_get_health_issues_high_latency(self):
        """Test getting high latency issues."""
        monitor = HealthMonitorService()

        monitor.update_agent_health(
            "agent-1",
            "ws-456",
            datetime.utcnow(),
            metrics={"latency_ms": 2000},  # 2 seconds
        )

        issues = monitor.get_health_issues("ws-456")

        assert any(i["type"] == "high_latency" for i in issues)


class TestHealthEvents:
    """Health events tests."""

    def test_status_change_event(self):
        """Test event on status change."""
        events = []
        monitor = HealthMonitorService(on_health_event=lambda e: events.append(e))

        # First update - unknown to online
        monitor.update_agent_health("agent-1", "ws-456", datetime.utcnow())

        assert len(events) >= 1
        assert any(e.event_type == "status_change" for e in events)

    def test_get_events(self):
        """Test getting events."""
        monitor = HealthMonitorService()

        monitor.mark_agent_halted("agent-1", "ws-456", "Test")

        events = monitor.get_events(workspace_id="ws-456")

        assert len(events) >= 1

    def test_events_filtered_by_agent(self):
        """Test events filtered by agent."""
        monitor = HealthMonitorService()

        monitor.update_agent_health("agent-1", "ws-456", datetime.utcnow())
        monitor.update_agent_health("agent-2", "ws-456", datetime.utcnow())

        events = monitor.get_events(agent_id="agent-1")

        assert all(e.agent_id == "agent-1" for e in events)


class TestStatusChangeCallback:
    """Status change callback tests."""

    def test_callback_on_status_change(self):
        """Test callback is called on status change."""
        changes = []
        monitor = HealthMonitorService(
            on_status_change=lambda h, old, new: changes.append((old, new))
        )

        monitor.update_agent_health("agent-1", "ws-456", datetime.utcnow())

        # Unknown -> Online
        assert len(changes) >= 1
        assert changes[0] == (AgentStatus.UNKNOWN, AgentStatus.ONLINE)


class TestAgentHealthSerialization:
    """Agent health serialization tests."""

    def test_to_dict(self):
        """Test agent health serialization."""
        health = AgentHealth(
            agent_id="agent-123",
            workspace_id="ws-456",
            status=AgentStatus.ONLINE,
            health=HealthStatus.HEALTHY,
        )

        data = health.to_dict()

        assert data["agent_id"] == "agent-123"
        assert data["status"] == "online"
        assert data["health"] == "healthy"

    def test_dashboard_to_dict(self):
        """Test dashboard serialization."""
        dashboard = HealthDashboard(
            workspace_id="ws-456",
            total_agents=5,
            online_agents=4,
        )

        data = dashboard.to_dict()

        assert data["workspace_id"] == "ws-456"
        assert data["total_agents"] == 5
        assert data["online_agents"] == 4


class TestSecondsSinceHeartbeat:
    """Seconds since heartbeat tests."""

    def test_seconds_since_heartbeat(self):
        """Test seconds since heartbeat calculation."""
        health = AgentHealth(
            agent_id="agent-123",
            last_heartbeat=datetime.utcnow() - timedelta(seconds=30),
        )

        seconds = health.seconds_since_heartbeat

        assert 29 <= seconds <= 31

    def test_seconds_since_no_heartbeat(self):
        """Test seconds since heartbeat when never heartbeated."""
        health = AgentHealth(
            agent_id="agent-123",
            last_heartbeat=None,
        )

        seconds = health.seconds_since_heartbeat

        assert seconds == float("inf")
