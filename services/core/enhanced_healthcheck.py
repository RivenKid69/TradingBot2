# -*- coding: utf-8 -*-
"""
Enhanced Healthcheck System (Block 2.2).

Implements Kubernetes-compatible health endpoints:
- /health - Full health check with dependencies
- /ready - Readiness probe (can accept traffic)
- /live - Liveness probe (process is alive)

DORA References:
    - Article 10: Detection of Anomalous Activities
    - Article 11: Response and Recovery
    - RTS CDR 2024/1774: ICT Risk Management Framework

Best Practices:
    - Kubernetes Health Checks Best Practices
    - CNCF Guidelines for Cloud Native Applications
    - Google SRE Book: Implementing Health Checks
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class ProbeType(Enum):
    """Health probe types."""
    LIVENESS = "liveness"      # Process is running
    READINESS = "readiness"    # Ready to accept traffic
    STARTUP = "startup"        # Initial startup check
    HEALTH = "health"          # Full health check


class DependencyType(Enum):
    """Dependency types."""
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    EXTERNAL_API = "external_api"
    FILE_SYSTEM = "file_system"
    INTERNAL_SERVICE = "internal_service"
    THIRD_PARTY = "third_party"


class DependencyStatus(Enum):
    """Dependency health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ReadinessCondition(Enum):
    """Readiness conditions."""
    ALL_DEPENDENCIES = "all_dependencies"
    CRITICAL_DEPENDENCIES = "critical_dependencies"
    BASIC_CHECKS = "basic_checks"


# =============================================================================
# Protocols
# =============================================================================

class DependencyChecker(Protocol):
    """Protocol for dependency health checkers."""

    def check(self) -> DependencyStatus:
        """Check dependency health."""
        ...

    def get_latency_ms(self) -> float:
        """Get check latency in milliseconds."""
        ...


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class HealthProbe:
    """Health probe configuration."""
    probe_id: str = ""
    name: str = ""
    probe_type: ProbeType = ProbeType.HEALTH

    # Timing
    timeout_seconds: float = 10.0
    interval_seconds: float = 30.0

    # Thresholds
    failure_threshold: int = 3
    success_threshold: int = 1

    # Status
    is_enabled: bool = True
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_check_time: str = ""
    last_status: DependencyStatus = DependencyStatus.UNKNOWN

    def __post_init__(self):
        if not self.probe_id:
            self.probe_id = f"PROBE-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class DependencyCheck:
    """Dependency health check result."""
    check_id: str = ""
    dependency_name: str = ""
    dependency_type: DependencyType = DependencyType.INTERNAL_SERVICE

    # Status
    status: DependencyStatus = DependencyStatus.UNKNOWN
    is_critical: bool = True

    # Metrics
    latency_ms: float = 0.0
    check_time: str = ""

    # Details
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def __post_init__(self):
        if not self.check_id:
            self.check_id = f"CHK-{uuid.uuid4().hex[:8].upper()}"
        if not self.check_time:
            self.check_time = datetime.now(timezone.utc).isoformat()


@dataclass
class LivenessResult:
    """Liveness probe result."""
    alive: bool = True
    timestamp: str = ""
    uptime_seconds: float = 0.0
    process_id: int = 0
    memory_mb: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReadinessResult:
    """Readiness probe result."""
    ready: bool = True
    timestamp: str = ""
    condition: str = ""

    # Dependencies
    dependencies_checked: int = 0
    dependencies_healthy: int = 0
    dependencies_unhealthy: int = 0

    # Details
    unhealthy_dependencies: List[str] = field(default_factory=list)
    message: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HealthResult:
    """Full health check result."""
    healthy: bool = True
    status: str = "healthy"  # healthy, degraded, unhealthy
    timestamp: str = ""

    # Components
    components: List[DependencyCheck] = field(default_factory=list)

    # Summary
    total_checks: int = 0
    healthy_checks: int = 0
    degraded_checks: int = 0
    unhealthy_checks: int = 0

    # System info
    version: str = ""
    uptime_seconds: float = 0.0
    system_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "healthy": self.healthy,
            "status": self.status,
            "timestamp": self.timestamp,
            "summary": {
                "total_checks": self.total_checks,
                "healthy": self.healthy_checks,
                "degraded": self.degraded_checks,
                "unhealthy": self.unhealthy_checks,
            },
            "components": [asdict(c) for c in self.components],
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
            "system_info": self.system_info,
        }
        return result


@dataclass
class EnhancedHealthcheckConfig:
    """Configuration for EnhancedHealthcheck."""
    # Version info
    service_version: str = "1.0.0"
    service_name: str = "quantitative-research-platform"

    # Timing
    default_timeout_seconds: float = 10.0
    liveness_timeout_seconds: float = 5.0
    readiness_timeout_seconds: float = 15.0

    # Readiness condition
    readiness_condition: ReadinessCondition = ReadinessCondition.CRITICAL_DEPENDENCIES

    # Thresholds
    latency_warning_ms: float = 500.0
    latency_critical_ms: float = 2000.0

    # Memory thresholds
    memory_warning_percent: float = 80.0
    memory_critical_percent: float = 95.0

    # Cache health results
    cache_duration_seconds: float = 5.0

    # Background monitoring
    enable_background_monitoring: bool = True
    background_interval_seconds: float = 30.0

    # Alerting
    alert_on_unhealthy: bool = True
    alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# Built-in Checkers
# =============================================================================

class DatabaseChecker:
    """Database health checker."""

    def __init__(self, connection_string: str = "", timeout: float = 5.0):
        self.connection_string = connection_string
        self.timeout = timeout
        self._latency_ms = 0.0

    def check(self) -> DependencyStatus:
        """Check database connectivity."""
        start = time.time()
        try:
            # Simulate database ping
            time.sleep(0.01)  # In real implementation, execute ping query
            self._latency_ms = (time.time() - start) * 1000

            if self._latency_ms < 100:
                return DependencyStatus.HEALTHY
            elif self._latency_ms < 500:
                return DependencyStatus.DEGRADED
            else:
                return DependencyStatus.UNHEALTHY
        except Exception:
            self._latency_ms = (time.time() - start) * 1000
            return DependencyStatus.UNHEALTHY

    def get_latency_ms(self) -> float:
        return self._latency_ms


class CacheChecker:
    """Cache health checker (Redis/Memcached)."""

    def __init__(self, host: str = "localhost", port: int = 6379):
        self.host = host
        self.port = port
        self._latency_ms = 0.0

    def check(self) -> DependencyStatus:
        """Check cache connectivity."""
        start = time.time()
        try:
            # Simulate cache ping
            time.sleep(0.005)
            self._latency_ms = (time.time() - start) * 1000
            return DependencyStatus.HEALTHY
        except Exception:
            self._latency_ms = (time.time() - start) * 1000
            return DependencyStatus.UNHEALTHY

    def get_latency_ms(self) -> float:
        return self._latency_ms


class ExternalAPIChecker:
    """External API health checker."""

    def __init__(self, url: str, timeout: float = 10.0, name: str = ""):
        self.url = url
        self.timeout = timeout
        self.name = name or url
        self._latency_ms = 0.0

    def check(self) -> DependencyStatus:
        """Check external API availability."""
        start = time.time()
        try:
            # Simulate API call
            time.sleep(0.02)
            self._latency_ms = (time.time() - start) * 1000

            if self._latency_ms < 200:
                return DependencyStatus.HEALTHY
            elif self._latency_ms < 1000:
                return DependencyStatus.DEGRADED
            else:
                return DependencyStatus.UNHEALTHY
        except Exception:
            self._latency_ms = (time.time() - start) * 1000
            return DependencyStatus.UNHEALTHY

    def get_latency_ms(self) -> float:
        return self._latency_ms


# =============================================================================
# Main Class
# =============================================================================

class EnhancedHealthcheck:
    """
    Enhanced Healthcheck Service per DORA Article 10.

    Provides Kubernetes-compatible health endpoints:
    - /health - Full health check
    - /ready - Readiness probe
    - /live - Liveness probe

    Features:
    - Dependency health monitoring
    - Latency tracking
    - Background monitoring
    - Alerting integration

    Usage:
        config = EnhancedHealthcheckConfig()
        healthcheck = EnhancedHealthcheck(config)

        # Register dependencies
        healthcheck.register_dependency(
            name="primary-db",
            dependency_type=DependencyType.DATABASE,
            checker=DatabaseChecker("postgres://..."),
            is_critical=True,
        )

        # Get health status
        health = healthcheck.health()
        ready = healthcheck.ready()
        live = healthcheck.live()

        # Start background monitoring
        await healthcheck.start_monitoring()
    """

    def __init__(self, config: Optional[EnhancedHealthcheckConfig] = None):
        """Initialize Enhanced Healthcheck."""
        self.config = config or EnhancedHealthcheckConfig()

        # Dependencies
        self._dependencies: Dict[str, Dict[str, Any]] = {}
        self._custom_checks: Dict[str, Callable[[], DependencyCheck]] = {}

        # Probes
        self._probes: Dict[ProbeType, HealthProbe] = {}

        # Caching
        self._last_health: Optional[HealthResult] = None
        self._last_health_time: float = 0.0

        # Timing
        self._start_time = time.time()

        # Thread safety
        self._lock = threading.RLock()

        # Background monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_running = False

        # Initialize default probes
        self._init_probes()

        logger.info("EnhancedHealthcheck initialized")

    def _init_probes(self) -> None:
        """Initialize default probes."""
        self._probes[ProbeType.LIVENESS] = HealthProbe(
            name="Liveness",
            probe_type=ProbeType.LIVENESS,
            timeout_seconds=self.config.liveness_timeout_seconds,
        )
        self._probes[ProbeType.READINESS] = HealthProbe(
            name="Readiness",
            probe_type=ProbeType.READINESS,
            timeout_seconds=self.config.readiness_timeout_seconds,
        )
        self._probes[ProbeType.HEALTH] = HealthProbe(
            name="Health",
            probe_type=ProbeType.HEALTH,
            timeout_seconds=self.config.default_timeout_seconds,
        )

    # =========================================================================
    # Dependency Registration
    # =========================================================================

    def register_dependency(
        self,
        name: str,
        dependency_type: DependencyType,
        checker: Any,
        is_critical: bool = True,
        timeout_seconds: float = 10.0,
    ) -> None:
        """
        Register a dependency for health checking.

        Args:
            name: Dependency name
            dependency_type: Type of dependency
            checker: Checker object with check() method
            is_critical: Whether this is a critical dependency
            timeout_seconds: Check timeout
        """
        with self._lock:
            self._dependencies[name] = {
                "name": name,
                "type": dependency_type,
                "checker": checker,
                "is_critical": is_critical,
                "timeout": timeout_seconds,
                "last_status": DependencyStatus.UNKNOWN,
                "last_check": None,
                "consecutive_failures": 0,
            }

        logger.info(f"Registered dependency: {name} ({dependency_type.value})")

    def register_custom_check(
        self,
        name: str,
        check_fn: Callable[[], DependencyCheck],
    ) -> None:
        """
        Register a custom health check function.

        Args:
            name: Check name
            check_fn: Function returning DependencyCheck
        """
        with self._lock:
            self._custom_checks[name] = check_fn

        logger.info(f"Registered custom check: {name}")

    def unregister_dependency(self, name: str) -> bool:
        """Unregister a dependency."""
        with self._lock:
            if name in self._dependencies:
                del self._dependencies[name]
                return True
            return False

    # =========================================================================
    # Health Endpoints
    # =========================================================================

    def live(self) -> LivenessResult:
        """
        Liveness probe endpoint (/live).

        Returns True if the process is alive and responding.
        Kubernetes will restart the pod if this fails.

        Returns:
            LivenessResult with liveness status
        """
        import os
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            memory_mb = 0.0

        return LivenessResult(
            alive=True,
            uptime_seconds=time.time() - self._start_time,
            process_id=os.getpid(),
            memory_mb=round(memory_mb, 2),
        )

    def ready(self) -> ReadinessResult:
        """
        Readiness probe endpoint (/ready).

        Returns True if the service is ready to accept traffic.
        Kubernetes will stop sending traffic if this fails.

        Returns:
            ReadinessResult with readiness status
        """
        with self._lock:
            dependencies = list(self._dependencies.values())

        checks = []
        for dep in dependencies:
            check = self._check_dependency(dep)
            checks.append(check)

        # Apply readiness condition
        condition = self.config.readiness_condition

        if condition == ReadinessCondition.ALL_DEPENDENCIES:
            healthy = all(c.status == DependencyStatus.HEALTHY for c in checks)
        elif condition == ReadinessCondition.CRITICAL_DEPENDENCIES:
            critical_checks = [c for c in checks if c.is_critical]
            healthy = all(c.status == DependencyStatus.HEALTHY for c in critical_checks)
        else:  # BASIC_CHECKS
            healthy = True  # Basic process check passed

        unhealthy = [c.dependency_name for c in checks if c.status == DependencyStatus.UNHEALTHY]

        return ReadinessResult(
            ready=healthy,
            condition=condition.value,
            dependencies_checked=len(checks),
            dependencies_healthy=sum(1 for c in checks if c.status == DependencyStatus.HEALTHY),
            dependencies_unhealthy=len(unhealthy),
            unhealthy_dependencies=unhealthy,
            message="Ready to accept traffic" if healthy else f"Unhealthy dependencies: {', '.join(unhealthy)}",
        )

    def health(self, force_refresh: bool = False) -> HealthResult:
        """
        Full health check endpoint (/health).

        Returns comprehensive health status of all dependencies.

        Args:
            force_refresh: Force refresh, ignoring cache

        Returns:
            HealthResult with full health status
        """
        # Check cache
        if not force_refresh and self._last_health:
            cache_age = time.time() - self._last_health_time
            if cache_age < self.config.cache_duration_seconds:
                return self._last_health

        with self._lock:
            dependencies = list(self._dependencies.values())
            custom_checks = dict(self._custom_checks)

        # Check dependencies
        checks: List[DependencyCheck] = []

        for dep in dependencies:
            check = self._check_dependency(dep)
            checks.append(check)

        # Run custom checks
        for name, check_fn in custom_checks.items():
            try:
                check = check_fn()
                checks.append(check)
            except Exception as e:
                checks.append(DependencyCheck(
                    dependency_name=name,
                    dependency_type=DependencyType.INTERNAL_SERVICE,
                    status=DependencyStatus.UNHEALTHY,
                    is_critical=False,
                    error=str(e),
                ))

        # Calculate summary
        healthy_count = sum(1 for c in checks if c.status == DependencyStatus.HEALTHY)
        degraded_count = sum(1 for c in checks if c.status == DependencyStatus.DEGRADED)
        unhealthy_count = sum(1 for c in checks if c.status == DependencyStatus.UNHEALTHY)

        # Determine overall status
        if unhealthy_count > 0:
            # Check if any critical dependency is unhealthy
            critical_unhealthy = any(
                c.status == DependencyStatus.UNHEALTHY and c.is_critical
                for c in checks
            )
            if critical_unhealthy:
                overall_status = "unhealthy"
                overall_healthy = False
            else:
                overall_status = "degraded"
                overall_healthy = True
        elif degraded_count > 0:
            overall_status = "degraded"
            overall_healthy = True
        else:
            overall_status = "healthy"
            overall_healthy = True

        # Get system info
        system_info = self._get_system_info()

        result = HealthResult(
            healthy=overall_healthy,
            status=overall_status,
            components=checks,
            total_checks=len(checks),
            healthy_checks=healthy_count,
            degraded_checks=degraded_count,
            unhealthy_checks=unhealthy_count,
            version=self.config.service_version,
            uptime_seconds=round(time.time() - self._start_time, 2),
            system_info=system_info,
        )

        # Update cache
        with self._lock:
            self._last_health = result
            self._last_health_time = time.time()

        # Alert if unhealthy
        if not overall_healthy and self.config.alert_on_unhealthy:
            self._send_alert("health_unhealthy", result.to_dict())

        return result

    def _check_dependency(self, dep: Dict[str, Any]) -> DependencyCheck:
        """Check a single dependency."""
        start = time.time()
        checker = dep.get("checker")

        try:
            if checker and hasattr(checker, "check"):
                status = checker.check()
                latency_ms = checker.get_latency_ms() if hasattr(checker, "get_latency_ms") else 0.0
            else:
                # No checker, assume healthy
                status = DependencyStatus.HEALTHY
                latency_ms = (time.time() - start) * 1000

            # Update dependency status
            with self._lock:
                if dep["name"] in self._dependencies:
                    self._dependencies[dep["name"]]["last_status"] = status
                    self._dependencies[dep["name"]]["last_check"] = datetime.now(timezone.utc).isoformat()

                    if status == DependencyStatus.UNHEALTHY:
                        self._dependencies[dep["name"]]["consecutive_failures"] += 1
                    else:
                        self._dependencies[dep["name"]]["consecutive_failures"] = 0

            return DependencyCheck(
                dependency_name=dep["name"],
                dependency_type=dep["type"],
                status=status,
                is_critical=dep.get("is_critical", True),
                latency_ms=round(latency_ms, 2),
                message=f"{dep['name']} is {status.value}",
            )

        except Exception as e:
            # Update failure count
            with self._lock:
                if dep["name"] in self._dependencies:
                    self._dependencies[dep["name"]]["consecutive_failures"] += 1
                    self._dependencies[dep["name"]]["last_status"] = DependencyStatus.UNHEALTHY

            return DependencyCheck(
                dependency_name=dep["name"],
                dependency_type=dep["type"],
                status=DependencyStatus.UNHEALTHY,
                is_critical=dep.get("is_critical", True),
                latency_ms=round((time.time() - start) * 1000, 2),
                error=str(e),
            )

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        import platform
        import os

        info = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "pid": os.getpid(),
        }

        try:
            import psutil
            info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            info["memory_percent"] = round(memory.percent, 1)
            info["memory_available_gb"] = round(memory.available / (1024 ** 3), 2)
        except ImportError:
            pass

        return info

    # =========================================================================
    # Background Monitoring
    # =========================================================================

    async def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._is_running:
            logger.warning("Monitoring already running")
            return

        self._is_running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started background health monitoring")

    async def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        self._is_running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
        logger.info("Stopped background health monitoring")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._is_running:
            try:
                # Run health check
                self.health(force_refresh=True)

                await asyncio.sleep(self.config.background_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.config.background_interval_seconds)

    # =========================================================================
    # HTTP Handlers
    # =========================================================================

    def create_handlers(self) -> Dict[str, Callable[[], Dict[str, Any]]]:
        """
        Create HTTP handlers for health endpoints.

        Returns:
            Dictionary of endpoint handlers

        Usage:
            handlers = healthcheck.create_handlers()

            @app.get("/health")
            def health_handler():
                return handlers["health"]()

            @app.get("/ready")
            def ready_handler():
                return handlers["ready"]()

            @app.get("/live")
            def live_handler():
                return handlers["live"]()
        """
        return {
            "health": lambda: self.health().to_dict(),
            "ready": lambda: self.ready().to_dict(),
            "live": lambda: self.live().to_dict(),
        }

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_dependency_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific dependency."""
        with self._lock:
            if name not in self._dependencies:
                return None

            dep = self._dependencies[name]
            return {
                "name": name,
                "type": dep["type"].value,
                "is_critical": dep["is_critical"],
                "status": dep["last_status"].value if dep["last_status"] else "unknown",
                "last_check": dep["last_check"],
                "consecutive_failures": dep["consecutive_failures"],
            }

    def get_all_dependencies(self) -> List[Dict[str, Any]]:
        """Get status of all dependencies."""
        results = []
        with self._lock:
            for name in self._dependencies:
                status = self.get_dependency_status(name)
                if status:
                    results.append(status)
        return results

    @property
    def uptime(self) -> timedelta:
        """Get service uptime."""
        return timedelta(seconds=time.time() - self._start_time)

    @property
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self.health().healthy

    @property
    def is_ready(self) -> bool:
        """Check if service is ready."""
        return self.ready().ready

    def _send_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        """Send an alert."""
        if self.config.alert_callback:
            try:
                self.config.alert_callback(alert_type, data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        logger.warning(f"Health alert: {alert_type}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_enhanced_healthcheck(
    config: Optional[EnhancedHealthcheckConfig] = None,
) -> EnhancedHealthcheck:
    """
    Create an EnhancedHealthcheck instance.

    Args:
        config: Optional configuration

    Returns:
        Configured EnhancedHealthcheck instance
    """
    return EnhancedHealthcheck(config=config)
