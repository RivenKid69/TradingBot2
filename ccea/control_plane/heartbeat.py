# -*- coding: utf-8 -*-
"""
CCEA Heartbeat Service.

Handles agent heartbeat monitoring:
- Track agent online/offline status
- Monitor health metrics
- Detect stale agents
- Alert on anomalies

Security:
- Heartbeats are authenticated
- No sensitive data in health metrics
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional
import threading

from ccea.models.protocol import (
    HeartbeatMessage,
    AgentState,
    AgentHealth,
    DeploymentState,
    RunState,
)


logger = logging.getLogger(__name__)


# ============================================================================
# Agent Status
# ============================================================================

class AgentStatus(str, Enum):
    """Agent online status."""
    ONLINE = "ONLINE"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"
    UNKNOWN = "UNKNOWN"


@dataclass
class AgentStatusInfo:
    """Agent status information."""
    agent_id: str
    status: AgentStatus
    last_heartbeat: Optional[datetime] = None
    deployment_state: Optional[DeploymentState] = None
    run_state: Optional[RunState] = None
    agent_version: Optional[str] = None
    uptime_seconds: Optional[int] = None
    health: Optional[AgentHealth] = None
    consecutive_missed: int = 0
    alerts: List[str] = field(default_factory=list)


# ============================================================================
# Heartbeat Store Interface
# ============================================================================

class HeartbeatStore(ABC):
    """Abstract heartbeat store interface."""

    @abstractmethod
    def save(self, agent_id: str, heartbeat: HeartbeatMessage) -> None:
        """Save a heartbeat."""
        pass

    @abstractmethod
    def get_last(self, agent_id: str) -> Optional[HeartbeatMessage]:
        """Get last heartbeat for agent."""
        pass

    @abstractmethod
    def get_status(self, agent_id: str) -> AgentStatusInfo:
        """Get agent status info."""
        pass

    @abstractmethod
    def list_all(self) -> List[AgentStatusInfo]:
        """List all agent statuses."""
        pass


class InMemoryHeartbeatStore(HeartbeatStore):
    """In-memory heartbeat store for development/testing."""

    def __init__(self):
        self._heartbeats: Dict[str, HeartbeatMessage] = {}
        self._statuses: Dict[str, AgentStatusInfo] = {}
        self._lock = threading.Lock()

    def save(self, agent_id: str, heartbeat: HeartbeatMessage) -> None:
        with self._lock:
            self._heartbeats[agent_id] = heartbeat

    def get_last(self, agent_id: str) -> Optional[HeartbeatMessage]:
        with self._lock:
            return self._heartbeats.get(agent_id)

    def get_status(self, agent_id: str) -> AgentStatusInfo:
        with self._lock:
            return self._statuses.get(agent_id, AgentStatusInfo(
                agent_id=agent_id,
                status=AgentStatus.UNKNOWN,
            ))

    def set_status(self, status: AgentStatusInfo) -> None:
        with self._lock:
            self._statuses[status.agent_id] = status

    def list_all(self) -> List[AgentStatusInfo]:
        with self._lock:
            return list(self._statuses.values())


# ============================================================================
# Heartbeat Service
# ============================================================================

class HeartbeatService:
    """
    Agent heartbeat monitoring service.

    Handles:
    - Processing heartbeat messages
    - Tracking agent status
    - Detecting offline agents
    - Health alerting
    """

    def __init__(
        self,
        store: Optional[HeartbeatStore] = None,
        heartbeat_interval: int = 30,
        offline_threshold: int = 3,
        degraded_threshold: int = 2,
    ):
        """
        Initialize heartbeat service.

        Args:
            store: Heartbeat storage backend
            heartbeat_interval: Expected interval in seconds
            offline_threshold: Missed heartbeats before offline
            degraded_threshold: Missed heartbeats before degraded
        """
        self.store = store or InMemoryHeartbeatStore()
        self.heartbeat_interval = heartbeat_interval
        self.offline_threshold = offline_threshold
        self.degraded_threshold = degraded_threshold
        self._lock = threading.Lock()

    def process_heartbeat(self, heartbeat: HeartbeatMessage) -> AgentStatusInfo:
        """
        Process incoming heartbeat.

        Args:
            heartbeat: Heartbeat message from agent

        Returns:
            Updated agent status
        """
        agent_id = heartbeat.agent_id

        # Save heartbeat
        self.store.save(agent_id, heartbeat)

        # Update status
        status = AgentStatusInfo(
            agent_id=agent_id,
            status=AgentStatus.ONLINE,
            last_heartbeat=heartbeat.timestamp,
            deployment_state=heartbeat.state.deployment_state if heartbeat.state else None,
            run_state=heartbeat.state.run_state if heartbeat.state else None,
            agent_version=heartbeat.state.agent_version if heartbeat.state else None,
            uptime_seconds=heartbeat.state.uptime_seconds if heartbeat.state else None,
            health=heartbeat.health,
            consecutive_missed=0,
            alerts=[],
        )

        # Check health metrics for alerts
        if heartbeat.health:
            status.alerts = self._check_health_alerts(heartbeat.health)
            if status.alerts:
                status.status = AgentStatus.DEGRADED

        # Store status
        if isinstance(self.store, InMemoryHeartbeatStore):
            self.store.set_status(status)

        logger.debug(
            "Heartbeat processed",
            extra={
                "agent_id": agent_id,
                "status": status.status.value,
                "alerts": status.alerts,
            }
        )

        return status

    def check_agent_status(self, agent_id: str) -> AgentStatusInfo:
        """
        Check current agent status.

        Args:
            agent_id: Agent to check

        Returns:
            Current agent status
        """
        last_heartbeat = self.store.get_last(agent_id)

        if last_heartbeat is None:
            return AgentStatusInfo(
                agent_id=agent_id,
                status=AgentStatus.UNKNOWN,
            )

        # Calculate time since last heartbeat
        now = datetime.utcnow()
        last_time = last_heartbeat.timestamp
        elapsed = (now - last_time).total_seconds()
        missed = int(elapsed / self.heartbeat_interval)

        # Determine status
        if missed >= self.offline_threshold:
            status = AgentStatus.OFFLINE
        elif missed >= self.degraded_threshold:
            status = AgentStatus.DEGRADED
        else:
            status = AgentStatus.ONLINE

        return AgentStatusInfo(
            agent_id=agent_id,
            status=status,
            last_heartbeat=last_time,
            deployment_state=last_heartbeat.state.deployment_state if last_heartbeat.state else None,
            run_state=last_heartbeat.state.run_state if last_heartbeat.state else None,
            agent_version=last_heartbeat.state.agent_version if last_heartbeat.state else None,
            uptime_seconds=last_heartbeat.state.uptime_seconds if last_heartbeat.state else None,
            health=last_heartbeat.health,
            consecutive_missed=missed,
        )

    def get_all_statuses(self) -> List[AgentStatusInfo]:
        """Get status of all known agents."""
        return self.store.list_all()

    def get_offline_agents(self) -> List[AgentStatusInfo]:
        """Get list of offline agents."""
        all_statuses = self.store.list_all()
        offline = []

        for status in all_statuses:
            current = self.check_agent_status(status.agent_id)
            if current.status == AgentStatus.OFFLINE:
                offline.append(current)

        return offline

    def _check_health_alerts(self, health: AgentHealth) -> List[str]:
        """Check health metrics for alerts."""
        alerts = []

        if health.cpu_percent is not None and health.cpu_percent > 90:
            alerts.append(f"High CPU: {health.cpu_percent:.1f}%")

        if health.memory_percent is not None and health.memory_percent > 90:
            alerts.append(f"High memory: {health.memory_percent:.1f}%")

        if health.disk_percent is not None and health.disk_percent > 95:
            alerts.append(f"Disk nearly full: {health.disk_percent:.1f}%")

        if health.broker_connected is False:
            alerts.append("Broker disconnected")

        if health.data_feed_ok is False:
            alerts.append("Data feed error")

        return alerts


# ============================================================================
# Heartbeat Monitor (Background Task)
# ============================================================================

class HeartbeatMonitor:
    """
    Background monitor for heartbeat checking.

    Periodically checks all agents and generates alerts.
    """

    def __init__(
        self,
        heartbeat_service: HeartbeatService,
        check_interval: int = 30,
    ):
        """
        Initialize heartbeat monitor.

        Args:
            heartbeat_service: Heartbeat service to use
            check_interval: Check interval in seconds
        """
        self.service = heartbeat_service
        self.check_interval = check_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._alerts_callback = None

    def set_alerts_callback(self, callback) -> None:
        """Set callback for alerts."""
        self._alerts_callback = callback

    def start(self) -> None:
        """Start background monitoring."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

        logger.info("Heartbeat monitor started")

    def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

        logger.info("Heartbeat monitor stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        import time

        while self._running:
            try:
                offline = self.service.get_offline_agents()

                if offline and self._alerts_callback:
                    for agent in offline:
                        self._alerts_callback({
                            "type": "agent_offline",
                            "agent_id": agent.agent_id,
                            "last_heartbeat": agent.last_heartbeat.isoformat() if agent.last_heartbeat else None,
                            "missed_count": agent.consecutive_missed,
                        })

            except Exception as e:
                logger.exception("Error in heartbeat monitor")

            time.sleep(self.check_interval)
