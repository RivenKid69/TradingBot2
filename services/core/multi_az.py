# -*- coding: utf-8 -*-
"""
Multi-AZ Deployment Support (Block 2.9).

Implements Multi-Availability Zone deployment:
- Zone configuration and management
- Failover orchestration
- Health monitoring per zone
- Load balancing support

DORA References:
    - Article 11: Response and Recovery
    - Article 12: Backup (geographically diverse locations)
    - Article 15: ICT Business Continuity
    - RTS CDR 2024/1774: Geographic redundancy requirements
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AvailabilityZone(Enum):
    """Availability zones."""

    EU_WEST_1A = "eu-west-1a"
    EU_WEST_1B = "eu-west-1b"
    EU_WEST_1C = "eu-west-1c"
    EU_CENTRAL_1A = "eu-central-1a"
    EU_CENTRAL_1B = "eu-central-1b"
    US_EAST_1A = "us-east-1a"
    US_EAST_1B = "us-east-1b"


class DeploymentStrategy(Enum):
    """Deployment strategies."""

    ACTIVE_ACTIVE = "active_active"
    ACTIVE_PASSIVE = "active_passive"
    ACTIVE_STANDBY = "active_standby"


class FailoverMode(Enum):
    """Failover modes."""

    AUTOMATIC = "automatic"
    MANUAL = "manual"
    SEMI_AUTOMATIC = "semi_automatic"


class ZoneStatus(Enum):
    """Zone health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


@dataclass
class ZoneConfig:
    """Availability zone configuration."""

    zone_id: str = ""
    zone: AvailabilityZone = AvailabilityZone.EU_WEST_1A
    region: str = "eu-west-1"
    is_primary: bool = False
    weight: int = 100  # Load balancing weight
    services: List[str] = field(default_factory=list)
    endpoints: Dict[str, str] = field(default_factory=dict)
    is_active: bool = True

    def __post_init__(self):
        if not self.zone_id:
            self.zone_id = f"ZONE-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class DeploymentConfig:
    """Multi-AZ deployment configuration."""

    deployment_id: str = ""
    name: str = ""
    strategy: DeploymentStrategy = DeploymentStrategy.ACTIVE_ACTIVE
    zones: List[ZoneConfig] = field(default_factory=list)
    minimum_zones: int = 2
    failover_mode: FailoverMode = FailoverMode.AUTOMATIC
    health_check_interval_seconds: int = 30
    failover_threshold: int = 3

    def __post_init__(self):
        if not self.deployment_id:
            self.deployment_id = f"DEPL-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class FailoverConfig:
    """Failover configuration."""

    config_id: str = ""
    mode: FailoverMode = FailoverMode.AUTOMATIC
    health_check_failures_threshold: int = 3
    failback_enabled: bool = True
    failback_delay_seconds: int = 300
    notification_targets: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.config_id:
            self.config_id = f"FOVCFG-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class ZoneHealthStatus:
    """Zone health status record."""

    zone_id: str = ""
    status: ZoneStatus = ZoneStatus.HEALTHY
    last_check: str = ""
    latency_ms: float = 0.0
    error_rate_percent: float = 0.0
    active_connections: int = 0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    issues: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.last_check:
            self.last_check = datetime.now(timezone.utc).isoformat()


@dataclass
class MultiAZConfig:
    """Configuration for MultiAZManager."""

    default_strategy: DeploymentStrategy = DeploymentStrategy.ACTIVE_ACTIVE
    health_check_interval_seconds: int = 30
    failover_threshold: int = 3
    minimum_healthy_zones: int = 2
    log_all_events: bool = True
    log_path: str = "logs/core/multi_az"
    alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


class MultiAZManager:
    """Multi-AZ Deployment Manager."""

    def __init__(self, config: Optional[MultiAZConfig] = None):
        self.config = config or MultiAZConfig()
        self._deployments: Dict[str, DeploymentConfig] = {}
        self._zones: Dict[str, ZoneConfig] = {}
        self._health: Dict[str, ZoneHealthStatus] = {}
        self._failover_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        logger.info("MultiAZManager initialized")

    def register_zone(
        self,
        zone: AvailabilityZone,
        region: str,
        is_primary: bool = False,
        services: Optional[List[str]] = None,
        weight: int = 100,
    ) -> ZoneConfig:
        """Register an availability zone."""
        zone_config = ZoneConfig(
            zone=zone,
            region=region,
            is_primary=is_primary,
            services=services or [],
            weight=weight,
        )
        with self._lock:
            self._zones[zone_config.zone_id] = zone_config
            self._health[zone_config.zone_id] = ZoneHealthStatus(zone_id=zone_config.zone_id)
        return zone_config

    def create_deployment(
        self,
        name: str,
        strategy: DeploymentStrategy,
        zone_ids: List[str],
        failover_mode: FailoverMode = FailoverMode.AUTOMATIC,
    ) -> Optional[DeploymentConfig]:
        """Create a multi-AZ deployment."""
        with self._lock:
            zones = [self._zones[zid] for zid in zone_ids if zid in self._zones]

        if len(zones) < 2:
            logger.error("Multi-AZ deployment requires at least 2 zones")
            return None

        deployment = DeploymentConfig(
            name=name,
            strategy=strategy,
            zones=zones,
            failover_mode=failover_mode,
        )

        with self._lock:
            self._deployments[deployment.deployment_id] = deployment

        return deployment

    def update_zone_health(
        self,
        zone_id: str,
        status: ZoneStatus,
        latency_ms: float = 0.0,
        error_rate_percent: float = 0.0,
    ) -> None:
        """Update zone health status."""
        with self._lock:
            if zone_id in self._health:
                health = self._health[zone_id]
                health.status = status
                health.latency_ms = latency_ms
                health.error_rate_percent = error_rate_percent
                health.last_check = datetime.now(timezone.utc).isoformat()

                # Check for failover trigger
                if status == ZoneStatus.UNHEALTHY:
                    self._check_failover(zone_id)

    def _check_failover(self, unhealthy_zone_id: str) -> None:
        """Check if failover is needed."""
        with self._lock:
            for deployment in self._deployments.values():
                zone_ids = [z.zone_id for z in deployment.zones]
                if unhealthy_zone_id in zone_ids:
                    healthy_zones = [
                        zid
                        for zid in zone_ids
                        if self._health.get(zid, ZoneHealthStatus()).status == ZoneStatus.HEALTHY
                    ]

                    if len(healthy_zones) >= deployment.minimum_zones:
                        logger.info(f"Sufficient healthy zones for deployment {deployment.name}")
                    else:
                        logger.warning(f"Insufficient healthy zones for {deployment.name}")
                        if self.config.alert_callback:
                            self.config.alert_callback(
                                "failover_warning",
                                {
                                    "deployment": deployment.name,
                                    "unhealthy_zone": unhealthy_zone_id,
                                    "healthy_zones": healthy_zones,
                                },
                            )

    def trigger_failover(
        self,
        deployment_id: str,
        from_zone_id: str,
        reason: str = "",
    ) -> Dict[str, Any]:
        """Trigger manual failover."""
        with self._lock:
            if deployment_id not in self._deployments:
                return {"success": False, "error": "Deployment not found"}

            deployment = self._deployments[deployment_id]
            zone_ids = [z.zone_id for z in deployment.zones]

            if from_zone_id not in zone_ids:
                return {"success": False, "error": "Zone not in deployment"}

            # Find target zone
            healthy_zones = [
                zid
                for zid in zone_ids
                if zid != from_zone_id
                and self._health.get(zid, ZoneHealthStatus()).status == ZoneStatus.HEALTHY
            ]

            if not healthy_zones:
                return {"success": False, "error": "No healthy target zones"}

            target_zone = healthy_zones[0]

            # Record failover
            failover_record = {
                "deployment_id": deployment_id,
                "from_zone": from_zone_id,
                "to_zone": target_zone,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._failover_history.append(failover_record)

        return {
            "success": True,
            "from_zone": from_zone_id,
            "to_zone": target_zone,
            "timestamp": failover_record["timestamp"],
        }

    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment status."""
        with self._lock:
            if deployment_id not in self._deployments:
                return None

            deployment = self._deployments[deployment_id]
            zone_statuses = []

            for zone in deployment.zones:
                health = self._health.get(zone.zone_id, ZoneHealthStatus())
                zone_statuses.append(
                    {
                        "zone_id": zone.zone_id,
                        "zone": zone.zone.value,
                        "is_primary": zone.is_primary,
                        "is_active": zone.is_active,
                        "status": health.status.value,
                        "latency_ms": health.latency_ms,
                    }
                )

        healthy_count = sum(1 for z in zone_statuses if z["status"] == "healthy")
        overall_status = "healthy" if healthy_count >= deployment.minimum_zones else "degraded"

        return {
            "deployment_id": deployment_id,
            "name": deployment.name,
            "strategy": deployment.strategy.value,
            "overall_status": overall_status,
            "zones": zone_statuses,
            "healthy_zones": healthy_count,
            "minimum_zones": deployment.minimum_zones,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get multi-AZ summary."""
        with self._lock:
            zones = list(self._zones.values())
            deployments = list(self._deployments.values())
            health_statuses = list(self._health.values())

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "zones": {
                "total": len(zones),
                "healthy": sum(1 for h in health_statuses if h.status == ZoneStatus.HEALTHY),
                "by_region": {},
            },
            "deployments": {
                "total": len(deployments),
                "by_strategy": {
                    s.value: sum(1 for d in deployments if d.strategy == s)
                    for s in DeploymentStrategy
                },
            },
            "failovers_30d": len(
                [
                    f
                    for f in self._failover_history
                    if f["timestamp"] > (datetime.now(timezone.utc).isoformat()[:10])
                ]
            ),
        }


def create_multi_az_manager(
    config: Optional[MultiAZConfig] = None,
) -> MultiAZManager:
    """Create a MultiAZManager instance."""
    return MultiAZManager(config=config)
