# -*- coding: utf-8 -*-
"""
services/healthcheck.py
Healthcheck service for live trading monitoring.

This module provides health monitoring for live trading services including:
- Exchange connection status
- Data feed freshness
- Order execution latency
- Position reconciliation status
- Risk guard status
- System resources (memory, CPU)

Usage:
    from services.healthcheck import (
        HealthcheckService,
        HealthStatus,
        create_healthcheck,
    )

    # Create healthcheck service
    healthcheck = create_healthcheck(config={
        "stale_data_threshold_sec": 60,
        "high_latency_threshold_ms": 1000,
    })

    # Register components to monitor
    healthcheck.register_exchange("binance", exchange_client)
    healthcheck.register_data_feed("market_data", data_feed)

    # Check health
    status = healthcheck.check_health()
    if not status.is_healthy:
        logger.error(f"System unhealthy: {status.get_failures()}")

    # Start background monitoring
    await healthcheck.start_background_checks(interval_sec=30)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================


class ComponentStatus(Enum):
    """Status of a monitored component."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Type of monitored component."""

    EXCHANGE = "exchange"
    DATA_FEED = "data_feed"
    EXECUTOR = "executor"
    RISK_GUARD = "risk_guard"
    POSITION_SYNC = "position_sync"
    SYSTEM = "system"


# =============================================================================
# Protocols for Monitorable Components
# =============================================================================


class ExchangeClient(Protocol):
    """Protocol for exchange clients that can be health-checked."""

    def ping(self) -> bool:
        """Check if exchange is reachable."""
        ...

    def get_server_time(self) -> Optional[int]:
        """Get exchange server time in milliseconds."""
        ...


class DataFeed(Protocol):
    """Protocol for data feeds that can be health-checked."""

    def get_last_update_time(self) -> Optional[float]:
        """Get timestamp of last data update."""
        ...

    def is_connected(self) -> bool:
        """Check if feed is connected."""
        ...


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ComponentHealth:
    """Health status of a single component."""

    name: str
    component_type: ComponentType
    status: ComponentStatus
    message: str
    latency_ms: Optional[float] = None
    last_check: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.component_type.value,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "last_check": self.last_check,
            "details": self.details,
        }


@dataclass
class HealthStatus:
    """Overall health status of the system."""

    status: ComponentStatus
    components: List[ComponentHealth] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    system_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        """Return True if all components are healthy or degraded."""
        return self.status in (ComponentStatus.HEALTHY, ComponentStatus.DEGRADED)

    def get_failures(self) -> List[ComponentHealth]:
        """Get list of unhealthy components."""
        return [c for c in self.components if c.status == ComponentStatus.UNHEALTHY]

    def get_warnings(self) -> List[ComponentHealth]:
        """Get list of degraded components."""
        return [c for c in self.components if c.status == ComponentStatus.DEGRADED]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "is_healthy": self.is_healthy,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "components": [c.to_dict() for c in self.components],
            "system_info": self.system_info,
            "failures": len(self.get_failures()),
            "warnings": len(self.get_warnings()),
        }


@dataclass
class HealthcheckConfig:
    """Configuration for healthcheck service."""

    # Data feed freshness
    stale_data_threshold_sec: float = 60.0
    stale_data_warning_sec: float = 30.0

    # Latency thresholds
    high_latency_threshold_ms: float = 1000.0
    warning_latency_threshold_ms: float = 500.0

    # Exchange ping timeout
    exchange_ping_timeout_sec: float = 10.0

    # Clock drift tolerance
    max_clock_drift_ms: float = 5000.0

    # System resource thresholds
    max_memory_percent: float = 90.0
    warning_memory_percent: float = 80.0
    max_cpu_percent: float = 95.0
    warning_cpu_percent: float = 85.0

    # Background check interval
    background_check_interval_sec: float = 30.0

    # Alert callbacks
    alert_on_unhealthy: bool = True
    alert_on_degraded: bool = False


# =============================================================================
# Healthcheck Service
# =============================================================================


class HealthcheckService:
    """
    Service for monitoring health of trading system components.

    Features:
        - Exchange connectivity monitoring
        - Data feed freshness checks
        - Latency monitoring
        - System resource monitoring
        - Background health checks
        - Alert callbacks
    """

    def __init__(self, config: Optional[HealthcheckConfig] = None):
        """
        Initialize healthcheck service.

        Args:
            config: Healthcheck configuration.
        """
        self.config = config or HealthcheckConfig()
        self._exchanges: Dict[str, Any] = {}
        self._data_feeds: Dict[str, Any] = {}
        self._executors: Dict[str, Any] = {}
        self._risk_guards: Dict[str, Any] = {}
        self._custom_checks: Dict[str, Callable[[], ComponentHealth]] = {}
        self._background_task: Optional[asyncio.Task] = None
        self._alert_callbacks: List[Callable[[HealthStatus], None]] = []
        self._last_status: Optional[HealthStatus] = None
        self._logger = logging.getLogger(f"{__name__}.HealthcheckService")

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def register_exchange(self, name: str, client: Any) -> None:
        """Register an exchange client for monitoring."""
        self._exchanges[name] = client
        self._logger.info(f"Registered exchange: {name}")

    def register_data_feed(self, name: str, feed: Any) -> None:
        """Register a data feed for monitoring."""
        self._data_feeds[name] = feed
        self._logger.info(f"Registered data feed: {name}")

    def register_executor(self, name: str, executor: Any) -> None:
        """Register an executor for monitoring."""
        self._executors[name] = executor
        self._logger.info(f"Registered executor: {name}")

    def register_risk_guard(self, name: str, guard: Any) -> None:
        """Register a risk guard for monitoring."""
        self._risk_guards[name] = guard
        self._logger.info(f"Registered risk guard: {name}")

    def register_custom_check(
        self,
        name: str,
        check_fn: Callable[[], ComponentHealth],
    ) -> None:
        """
        Register a custom health check function.

        Args:
            name: Name of the check.
            check_fn: Function that returns ComponentHealth.
        """
        self._custom_checks[name] = check_fn
        self._logger.info(f"Registered custom check: {name}")

    def add_alert_callback(
        self,
        callback: Callable[[HealthStatus], None],
    ) -> None:
        """Add a callback to be called when health status changes."""
        self._alert_callbacks.append(callback)

    # -------------------------------------------------------------------------
    # Health Checks
    # -------------------------------------------------------------------------

    def check_exchange(self, name: str, client: Any) -> ComponentHealth:
        """Check health of an exchange connection."""
        start_time = time.time()

        try:
            # Try to ping exchange
            ping_fn = getattr(client, "ping", None)
            if ping_fn and callable(ping_fn):
                ping_result = ping_fn()
                if not ping_result:
                    return ComponentHealth(
                        name=name,
                        component_type=ComponentType.EXCHANGE,
                        status=ComponentStatus.UNHEALTHY,
                        message="Exchange ping failed",
                    )

            latency_ms = (time.time() - start_time) * 1000

            # Check server time drift
            get_time_fn = getattr(client, "get_server_time", None)
            if get_time_fn and callable(get_time_fn):
                server_time = get_time_fn()
                if server_time:
                    local_time = int(time.time() * 1000)
                    drift_ms = abs(local_time - server_time)

                    if drift_ms > self.config.max_clock_drift_ms:
                        return ComponentHealth(
                            name=name,
                            component_type=ComponentType.EXCHANGE,
                            status=ComponentStatus.DEGRADED,
                            message=f"Clock drift {drift_ms}ms exceeds threshold",
                            latency_ms=latency_ms,
                            details={"clock_drift_ms": drift_ms},
                        )

            # Check latency
            if latency_ms > self.config.high_latency_threshold_ms:
                return ComponentHealth(
                    name=name,
                    component_type=ComponentType.EXCHANGE,
                    status=ComponentStatus.DEGRADED,
                    message=f"High latency: {latency_ms:.0f}ms",
                    latency_ms=latency_ms,
                )

            status = ComponentStatus.HEALTHY
            if latency_ms > self.config.warning_latency_threshold_ms:
                status = ComponentStatus.DEGRADED

            return ComponentHealth(
                name=name,
                component_type=ComponentType.EXCHANGE,
                status=status,
                message=f"Connected, latency {latency_ms:.0f}ms",
                latency_ms=latency_ms,
            )

        except Exception as e:
            return ComponentHealth(
                name=name,
                component_type=ComponentType.EXCHANGE,
                status=ComponentStatus.UNHEALTHY,
                message=f"Error: {str(e)[:100]}",
                latency_ms=(time.time() - start_time) * 1000,
            )

    def check_data_feed(self, name: str, feed: Any) -> ComponentHealth:
        """Check health of a data feed."""
        try:
            # Check if connected
            is_connected_fn = getattr(feed, "is_connected", None)
            if is_connected_fn and callable(is_connected_fn):
                if not is_connected_fn():
                    return ComponentHealth(
                        name=name,
                        component_type=ComponentType.DATA_FEED,
                        status=ComponentStatus.UNHEALTHY,
                        message="Feed disconnected",
                    )

            # Check data freshness
            get_last_update_fn = getattr(feed, "get_last_update_time", None)
            if get_last_update_fn and callable(get_last_update_fn):
                last_update = get_last_update_fn()
                if last_update:
                    age_sec = time.time() - last_update

                    if age_sec > self.config.stale_data_threshold_sec:
                        return ComponentHealth(
                            name=name,
                            component_type=ComponentType.DATA_FEED,
                            status=ComponentStatus.UNHEALTHY,
                            message=f"Stale data: {age_sec:.0f}s old",
                            details={"age_sec": age_sec},
                        )

                    if age_sec > self.config.stale_data_warning_sec:
                        return ComponentHealth(
                            name=name,
                            component_type=ComponentType.DATA_FEED,
                            status=ComponentStatus.DEGRADED,
                            message=f"Data aging: {age_sec:.0f}s old",
                            details={"age_sec": age_sec},
                        )

            return ComponentHealth(
                name=name,
                component_type=ComponentType.DATA_FEED,
                status=ComponentStatus.HEALTHY,
                message="Feed healthy",
            )

        except Exception as e:
            return ComponentHealth(
                name=name,
                component_type=ComponentType.DATA_FEED,
                status=ComponentStatus.UNHEALTHY,
                message=f"Error: {str(e)[:100]}",
            )

    def check_system_resources(self) -> ComponentHealth:
        """Check system resource usage."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)

            details = {
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "cpu_percent": cpu_percent,
            }

            # Check memory
            if memory.percent > self.config.max_memory_percent:
                return ComponentHealth(
                    name="system_resources",
                    component_type=ComponentType.SYSTEM,
                    status=ComponentStatus.UNHEALTHY,
                    message=f"Memory critical: {memory.percent:.1f}%",
                    details=details,
                )

            # Check CPU
            if cpu_percent > self.config.max_cpu_percent:
                return ComponentHealth(
                    name="system_resources",
                    component_type=ComponentType.SYSTEM,
                    status=ComponentStatus.UNHEALTHY,
                    message=f"CPU critical: {cpu_percent:.1f}%",
                    details=details,
                )

            # Check warnings
            if (
                memory.percent > self.config.warning_memory_percent
                or cpu_percent > self.config.warning_cpu_percent
            ):
                return ComponentHealth(
                    name="system_resources",
                    component_type=ComponentType.SYSTEM,
                    status=ComponentStatus.DEGRADED,
                    message=f"Resources elevated: CPU {cpu_percent:.0f}%, MEM {memory.percent:.0f}%",
                    details=details,
                )

            return ComponentHealth(
                name="system_resources",
                component_type=ComponentType.SYSTEM,
                status=ComponentStatus.HEALTHY,
                message=f"CPU {cpu_percent:.0f}%, MEM {memory.percent:.0f}%",
                details=details,
            )

        except ImportError:
            return ComponentHealth(
                name="system_resources",
                component_type=ComponentType.SYSTEM,
                status=ComponentStatus.UNKNOWN,
                message="psutil not installed",
            )
        except Exception as e:
            return ComponentHealth(
                name="system_resources",
                component_type=ComponentType.SYSTEM,
                status=ComponentStatus.UNKNOWN,
                message=f"Error: {str(e)[:100]}",
            )

    def check_health(self) -> HealthStatus:
        """
        Run all health checks and return overall status.

        Returns:
            HealthStatus with all component statuses.
        """
        components: List[ComponentHealth] = []

        # Check exchanges
        for name, client in self._exchanges.items():
            components.append(self.check_exchange(name, client))

        # Check data feeds
        for name, feed in self._data_feeds.items():
            components.append(self.check_data_feed(name, feed))

        # Check system resources
        components.append(self.check_system_resources())

        # Run custom checks
        for name, check_fn in self._custom_checks.items():
            try:
                components.append(check_fn())
            except Exception as e:
                components.append(
                    ComponentHealth(
                        name=name,
                        component_type=ComponentType.SYSTEM,
                        status=ComponentStatus.UNHEALTHY,
                        message=f"Check failed: {str(e)[:100]}",
                    )
                )

        # Determine overall status
        unhealthy_count = sum(1 for c in components if c.status == ComponentStatus.UNHEALTHY)
        degraded_count = sum(1 for c in components if c.status == ComponentStatus.DEGRADED)

        if unhealthy_count > 0:
            overall_status = ComponentStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = ComponentStatus.DEGRADED
        else:
            overall_status = ComponentStatus.HEALTHY

        status = HealthStatus(
            status=overall_status,
            components=components,
            system_info=self._get_system_info(),
        )

        self._last_status = status
        return status

    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information."""
        import platform

        return {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
        }

    # -------------------------------------------------------------------------
    # Background Monitoring
    # -------------------------------------------------------------------------

    async def start_background_checks(
        self,
        interval_sec: Optional[float] = None,
    ) -> None:
        """
        Start background health monitoring.

        Args:
            interval_sec: Check interval in seconds (default from config).
        """
        if self._background_task is not None:
            self._logger.warning("Background checks already running")
            return

        interval = interval_sec or self.config.background_check_interval_sec
        self._background_task = asyncio.create_task(self._background_check_loop(interval))
        self._logger.info(f"Started background health checks (interval: {interval}s)")

    async def stop_background_checks(self) -> None:
        """Stop background health monitoring."""
        if self._background_task is not None:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None
            self._logger.info("Stopped background health checks")

    async def _background_check_loop(self, interval_sec: float) -> None:
        """Background health check loop."""
        while True:
            try:
                status = self.check_health()

                # Trigger alerts if needed
                if status.status == ComponentStatus.UNHEALTHY:
                    if self.config.alert_on_unhealthy:
                        self._trigger_alerts(status)
                elif status.status == ComponentStatus.DEGRADED:
                    if self.config.alert_on_degraded:
                        self._trigger_alerts(status)

                await asyncio.sleep(interval_sec)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Background check error: {e}")
                await asyncio.sleep(interval_sec)

    def _trigger_alerts(self, status: HealthStatus) -> None:
        """Trigger alert callbacks."""
        for callback in self._alert_callbacks:
            try:
                callback(status)
            except Exception as e:
                self._logger.error(f"Alert callback error: {e}")

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def last_status(self) -> Optional[HealthStatus]:
        """Get last health status."""
        return self._last_status

    @property
    def is_running(self) -> bool:
        """Check if background monitoring is running."""
        return self._background_task is not None and not self._background_task.done()


# =============================================================================
# Factory Functions
# =============================================================================


def create_healthcheck(config: Optional[Dict[str, Any]] = None) -> HealthcheckService:
    """
    Create a healthcheck service with configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Configured HealthcheckService.
    """
    if config:
        hc_config = HealthcheckConfig(**config)
    else:
        hc_config = HealthcheckConfig()

    return HealthcheckService(config=hc_config)


def create_healthcheck_from_yaml(yaml_path: str) -> HealthcheckService:
    """
    Create a healthcheck service from YAML config file.

    Args:
        yaml_path: Path to YAML configuration file.

    Returns:
        Configured HealthcheckService.
    """
    import yaml
    from pathlib import Path

    with open(Path(yaml_path), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    healthcheck_config = config.get("healthcheck", {})
    return create_healthcheck(healthcheck_config)


# =============================================================================
# HTTP Endpoint Handler (for external monitoring)
# =============================================================================


def create_health_endpoint_handler(
    healthcheck: HealthcheckService,
) -> Callable[[], Dict[str, Any]]:
    """
    Create a handler function for HTTP health endpoint.

    Usage:
        handler = create_health_endpoint_handler(healthcheck)

        # In your web framework:
        @app.get("/health")
        def health():
            return handler()

    Args:
        healthcheck: HealthcheckService instance.

    Returns:
        Handler function that returns health status dict.
    """

    def handler() -> Dict[str, Any]:
        status = healthcheck.check_health()
        return status.to_dict()

    return handler


def create_readiness_endpoint_handler(
    healthcheck: HealthcheckService,
) -> Callable[[], Dict[str, Any]]:
    """
    Create a handler for Kubernetes readiness probe.

    Returns:
        Handler function that returns readiness status.
    """

    def handler() -> Dict[str, Any]:
        status = healthcheck.check_health()
        return {
            "ready": status.is_healthy,
            "status": status.status.value,
        }

    return handler


def create_liveness_endpoint_handler() -> Callable[[], Dict[str, Any]]:
    """
    Create a handler for Kubernetes liveness probe.

    Simple liveness check - just confirms the process is responding.

    Returns:
        Handler function that returns liveness status.
    """

    def handler() -> Dict[str, Any]:
        return {
            "alive": True,
            "timestamp": time.time(),
        }

    return handler
