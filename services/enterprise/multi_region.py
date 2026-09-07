# -*- coding: utf-8 -*-
"""
Multi-Region Deployment Service.

DORA Phase 3 Block 3.9: Multi-region deployment

Provides multi-region deployment capabilities for enterprise resilience:
- Region configuration and management
- Cross-region replication
- Automatic failover
- Geographic data residency

DORA References:
    - Art. 11: Response and recovery (geographic redundancy)
    - Art. 12: Backup and restoration
    - Art. 30(2)(b): Data location provisions
    - Art. 30(3)(a): Service level descriptions for critical functions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4


# =============================================================================
# Enums
# =============================================================================


class Region(Enum):
    """Available deployment regions."""

    # Europe
    EU_WEST_1 = "eu-west-1"  # Ireland
    EU_WEST_2 = "eu-west-2"  # London
    EU_CENTRAL_1 = "eu-central-1"  # Frankfurt
    EU_NORTH_1 = "eu-north-1"  # Stockholm

    # US (for non-EU data)
    US_EAST_1 = "us-east-1"  # Virginia
    US_WEST_1 = "us-west-1"  # California

    # Asia Pacific
    AP_SOUTHEAST_1 = "ap-southeast-1"  # Singapore
    AP_NORTHEAST_1 = "ap-northeast-1"  # Tokyo


class RegionStatus(Enum):
    """Region operational status."""

    ACTIVE = "active"  # Serving traffic
    STANDBY = "standby"  # Ready for failover
    DEGRADED = "degraded"  # Partial functionality
    MAINTENANCE = "maintenance"  # Planned maintenance
    OFFLINE = "offline"  # Not operational


class ReplicationMode(Enum):
    """Data replication modes."""

    SYNCHRONOUS = "synchronous"  # Strong consistency
    ASYNCHRONOUS = "asynchronous"  # Eventual consistency
    SEMI_SYNCHRONOUS = "semi_synchronous"  # Hybrid


class FailoverStatus(Enum):
    """Failover operation status."""

    IDLE = "idle"  # No failover in progress
    INITIATED = "initiated"  # Failover started
    IN_PROGRESS = "in_progress"  # Failover underway
    VALIDATING = "validating"  # Validating failover
    COMPLETED = "completed"  # Failover complete
    FAILED = "failed"  # Failover failed
    ROLLED_BACK = "rolled_back"  # Failover rolled back


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class RegionConfig:
    """Region configuration."""

    region: Region
    endpoint: str
    is_primary: bool
    data_residency: str  # EU, US, APAC, etc.
    availability_zones: list[str]
    max_connections: int = 10000
    storage_class: str = "standard"
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True


@dataclass
class RegionHealth:
    """Region health metrics."""

    region: Region
    status: RegionStatus
    latency_ms: float
    availability_percent: float
    error_rate_percent: float
    active_connections: int
    cpu_utilization_percent: float
    memory_utilization_percent: float
    storage_utilization_percent: float
    last_check: datetime
    issues: list[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        """Check if region is healthy."""
        return (
            self.status == RegionStatus.ACTIVE
            and self.availability_percent >= 99.0
            and self.error_rate_percent < 1.0
            and self.cpu_utilization_percent < 90.0
            and self.memory_utilization_percent < 90.0
        )


@dataclass
class ReplicationStatus:
    """Cross-region replication status."""

    source_region: Region
    target_region: Region
    mode: ReplicationMode
    lag_seconds: float
    bytes_pending: int
    is_healthy: bool
    last_sync: datetime
    sync_errors: int = 0

    @property
    def lag_status(self) -> str:
        """Get lag status description."""
        if self.lag_seconds < 1:
            return "excellent"
        elif self.lag_seconds < 5:
            return "good"
        elif self.lag_seconds < 30:
            return "acceptable"
        else:
            return "degraded"


@dataclass
class FailoverPlan:
    """Failover plan configuration."""

    plan_id: str
    name: str
    source_region: Region
    target_region: Region
    trigger_conditions: list[str]  # e.g., ["availability < 95%", "latency > 1000ms"]
    auto_failover: bool
    min_healthy_checks: int = 3
    cooldown_minutes: int = 30
    rollback_timeout_minutes: int = 60
    notification_channels: list[str] = field(default_factory=list)
    pre_failover_checks: list[str] = field(default_factory=list)
    post_failover_checks: list[str] = field(default_factory=list)


@dataclass
class FailoverEvent:
    """Failover event record."""

    event_id: str
    plan_id: str
    source_region: Region
    target_region: Region
    trigger: str
    initiated_at: datetime
    initiated_by: str  # "auto" or user ID
    status: FailoverStatus
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    error_message: str | None = None
    rollback_initiated: bool = False

    def complete(self) -> None:
        """Mark failover as completed."""
        self.status = FailoverStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.duration_seconds = (self.completed_at - self.initiated_at).total_seconds()

    def fail(self, error: str) -> None:
        """Mark failover as failed."""
        self.status = FailoverStatus.FAILED
        self.error_message = error
        self.completed_at = datetime.utcnow()


@dataclass
class RegionDeployment:
    """Deployment configuration for a region."""

    deployment_id: str
    region: Region
    client_id: str | None  # None for shared deployment
    version: str
    deployed_at: datetime
    deployed_by: str
    status: str = "active"  # active, inactive, pending
    instances: int = 2
    auto_scaling: bool = True
    min_instances: int = 2
    max_instances: int = 10


@dataclass
class MultiRegionConfig:
    """Multi-region service configuration."""

    primary_region: Region
    secondary_regions: list[Region]
    replication_mode: ReplicationMode
    auto_failover_enabled: bool = True
    health_check_interval_seconds: int = 30
    failover_threshold_checks: int = 3
    eu_data_residency: bool = True  # Restrict EU client data to EU regions


# =============================================================================
# Main Service Class
# =============================================================================


class MultiRegionService:
    """
    Multi-Region Deployment Service.

    Provides multi-region deployment capabilities per DORA resilience requirements.
    """

    def __init__(self, config: MultiRegionConfig) -> None:
        """Initialize multi-region service."""
        self.config = config
        self._region_configs: dict[Region, RegionConfig] = {}
        self._health_status: dict[Region, RegionHealth] = {}
        self._replication_status: dict[str, ReplicationStatus] = {}
        self._failover_plans: dict[str, FailoverPlan] = {}
        self._failover_events: list[FailoverEvent] = []
        self._deployments: dict[str, RegionDeployment] = {}
        self._initialize_regions()

    def _initialize_regions(self) -> None:
        """Initialize region configurations."""
        # Initialize primary region
        self._region_configs[self.config.primary_region] = RegionConfig(
            region=self.config.primary_region,
            endpoint=f"https://{self.config.primary_region.value}.api.platform.com",
            is_primary=True,
            data_residency=self._get_data_residency(self.config.primary_region),
            availability_zones=[
                f"{self.config.primary_region.value}-a",
                f"{self.config.primary_region.value}-b",
            ],
        )

        # Initialize secondary regions
        for region in self.config.secondary_regions:
            self._region_configs[region] = RegionConfig(
                region=region,
                endpoint=f"https://{region.value}.api.platform.com",
                is_primary=False,
                data_residency=self._get_data_residency(region),
                availability_zones=[f"{region.value}-a", f"{region.value}-b"],
            )

    def _get_data_residency(self, region: Region) -> str:
        """Get data residency classification for a region."""
        if region.value.startswith("eu-"):
            return "EU"
        elif region.value.startswith("us-"):
            return "US"
        elif region.value.startswith("ap-"):
            return "APAC"
        return "GLOBAL"

    # =========================================================================
    # Region Management
    # =========================================================================

    def get_region_config(self, region: Region) -> RegionConfig | None:
        """Get region configuration."""
        return self._region_configs.get(region)

    def list_regions(self, data_residency: str | None = None) -> list[RegionConfig]:
        """List all configured regions."""
        regions = list(self._region_configs.values())
        if data_residency:
            regions = [r for r in regions if r.data_residency == data_residency]
        return regions

    def get_eu_regions(self) -> list[RegionConfig]:
        """Get EU-only regions for DORA compliance."""
        return self.list_regions(data_residency="EU")

    def set_region_status(self, region: Region, status: RegionStatus) -> bool:
        """Set region operational status."""
        if region not in self._region_configs:
            return False

        if region not in self._health_status:
            self._health_status[region] = RegionHealth(
                region=region,
                status=status,
                latency_ms=0,
                availability_percent=100.0,
                error_rate_percent=0.0,
                active_connections=0,
                cpu_utilization_percent=0.0,
                memory_utilization_percent=0.0,
                storage_utilization_percent=0.0,
                last_check=datetime.utcnow(),
            )
        else:
            self._health_status[region].status = status
            self._health_status[region].last_check = datetime.utcnow()

        return True

    # =========================================================================
    # Health Monitoring
    # =========================================================================

    def update_health(
        self,
        region: Region,
        latency_ms: float,
        availability_percent: float,
        error_rate_percent: float,
        active_connections: int,
        cpu_utilization_percent: float,
        memory_utilization_percent: float,
        storage_utilization_percent: float,
    ) -> RegionHealth:
        """Update region health metrics."""
        # Determine status based on metrics
        status = RegionStatus.ACTIVE
        issues = []

        if availability_percent < 99.0:
            status = RegionStatus.DEGRADED
            issues.append(f"Low availability: {availability_percent}%")
        if error_rate_percent > 1.0:
            status = RegionStatus.DEGRADED
            issues.append(f"High error rate: {error_rate_percent}%")
        if latency_ms > 500:
            issues.append(f"High latency: {latency_ms}ms")
        if cpu_utilization_percent > 80:
            issues.append(f"High CPU: {cpu_utilization_percent}%")

        health = RegionHealth(
            region=region,
            status=status,
            latency_ms=latency_ms,
            availability_percent=availability_percent,
            error_rate_percent=error_rate_percent,
            active_connections=active_connections,
            cpu_utilization_percent=cpu_utilization_percent,
            memory_utilization_percent=memory_utilization_percent,
            storage_utilization_percent=storage_utilization_percent,
            last_check=datetime.utcnow(),
            issues=issues,
        )
        self._health_status[region] = health

        # Check if auto-failover should be triggered
        if self.config.auto_failover_enabled and not health.is_healthy:
            self._check_auto_failover(region)

        return health

    def get_health(self, region: Region) -> RegionHealth | None:
        """Get region health status."""
        return self._health_status.get(region)

    def get_all_health(self) -> dict[Region, RegionHealth]:
        """Get all region health statuses."""
        return self._health_status.copy()

    # =========================================================================
    # Replication
    # =========================================================================

    def configure_replication(
        self,
        source_region: Region,
        target_region: Region,
        mode: ReplicationMode | None = None,
    ) -> ReplicationStatus:
        """Configure replication between regions."""
        key = f"{source_region.value}->{target_region.value}"
        status = ReplicationStatus(
            source_region=source_region,
            target_region=target_region,
            mode=mode or self.config.replication_mode,
            lag_seconds=0.0,
            bytes_pending=0,
            is_healthy=True,
            last_sync=datetime.utcnow(),
        )
        self._replication_status[key] = status
        return status

    def get_replication_status(
        self,
        source_region: Region,
        target_region: Region,
    ) -> ReplicationStatus | None:
        """Get replication status between regions."""
        key = f"{source_region.value}->{target_region.value}"
        return self._replication_status.get(key)

    def update_replication_status(
        self,
        source_region: Region,
        target_region: Region,
        lag_seconds: float,
        bytes_pending: int,
    ) -> ReplicationStatus | None:
        """Update replication status."""
        key = f"{source_region.value}->{target_region.value}"
        status = self._replication_status.get(key)
        if not status:
            return None

        status.lag_seconds = lag_seconds
        status.bytes_pending = bytes_pending
        status.is_healthy = lag_seconds < 30 and bytes_pending < 1_000_000
        status.last_sync = datetime.utcnow()
        return status

    # =========================================================================
    # Failover
    # =========================================================================

    def create_failover_plan(
        self,
        name: str,
        source_region: Region,
        target_region: Region,
        auto_failover: bool = True,
        trigger_conditions: list[str] | None = None,
    ) -> FailoverPlan:
        """Create a failover plan."""
        plan = FailoverPlan(
            plan_id=str(uuid4()),
            name=name,
            source_region=source_region,
            target_region=target_region,
            trigger_conditions=trigger_conditions or ["availability < 95%"],
            auto_failover=auto_failover,
        )
        self._failover_plans[plan.plan_id] = plan
        return plan

    def get_failover_plan(self, plan_id: str) -> FailoverPlan | None:
        """Get failover plan by ID."""
        return self._failover_plans.get(plan_id)

    def list_failover_plans(self) -> list[FailoverPlan]:
        """List all failover plans."""
        return list(self._failover_plans.values())

    def initiate_failover(
        self,
        plan_id: str,
        trigger: str,
        initiated_by: str = "manual",
    ) -> FailoverEvent:
        """Initiate a failover."""
        plan = self._failover_plans.get(plan_id)
        if not plan:
            raise ValueError(f"Failover plan not found: {plan_id}")

        event = FailoverEvent(
            event_id=str(uuid4()),
            plan_id=plan_id,
            source_region=plan.source_region,
            target_region=plan.target_region,
            trigger=trigger,
            initiated_at=datetime.utcnow(),
            initiated_by=initiated_by,
            status=FailoverStatus.INITIATED,
        )

        # Simulate failover process
        event.status = FailoverStatus.IN_PROGRESS

        # Update region statuses
        self.set_region_status(plan.source_region, RegionStatus.STANDBY)
        self.set_region_status(plan.target_region, RegionStatus.ACTIVE)

        # Update primary region config
        if plan.source_region in self._region_configs:
            self._region_configs[plan.source_region].is_primary = False
        if plan.target_region in self._region_configs:
            self._region_configs[plan.target_region].is_primary = True

        event.complete()
        self._failover_events.append(event)
        return event

    def _check_auto_failover(self, region: Region) -> None:
        """Check if auto-failover should be triggered."""
        for plan in self._failover_plans.values():
            if plan.source_region == region and plan.auto_failover:
                # Check if we've had enough unhealthy checks
                recent_events = [
                    e
                    for e in self._failover_events
                    if e.source_region == region
                    and e.initiated_at
                    > datetime.utcnow() - timedelta(minutes=plan.cooldown_minutes)
                ]
                if not recent_events:
                    # Trigger auto-failover
                    self.initiate_failover(
                        plan.plan_id,
                        trigger="auto_failover_health_check",
                        initiated_by="auto",
                    )
                    break

    def get_failover_events(
        self,
        plan_id: str | None = None,
        limit: int = 100,
    ) -> list[FailoverEvent]:
        """Get failover events."""
        events = self._failover_events
        if plan_id:
            events = [e for e in events if e.plan_id == plan_id]
        return events[-limit:]

    # =========================================================================
    # Deployment
    # =========================================================================

    def deploy_to_region(
        self,
        region: Region,
        version: str,
        deployed_by: str,
        client_id: str | None = None,
        instances: int = 2,
    ) -> RegionDeployment:
        """Deploy to a specific region."""
        if region not in self._region_configs:
            raise ValueError(f"Region not configured: {region}")

        deployment = RegionDeployment(
            deployment_id=str(uuid4()),
            region=region,
            client_id=client_id,
            version=version,
            deployed_at=datetime.utcnow(),
            deployed_by=deployed_by,
            instances=instances,
        )
        self._deployments[deployment.deployment_id] = deployment
        return deployment

    def get_deployment(self, deployment_id: str) -> RegionDeployment | None:
        """Get deployment by ID."""
        return self._deployments.get(deployment_id)

    def list_deployments(
        self,
        region: Region | None = None,
        client_id: str | None = None,
    ) -> list[RegionDeployment]:
        """List deployments."""
        deployments = list(self._deployments.values())
        if region:
            deployments = [d for d in deployments if d.region == region]
        if client_id:
            deployments = [d for d in deployments if d.client_id == client_id]
        return deployments

    # =========================================================================
    # Status Summary
    # =========================================================================

    def get_status_summary(self) -> dict[str, Any]:
        """Get overall multi-region status summary."""
        healthy_regions = sum(1 for h in self._health_status.values() if h.is_healthy)
        total_regions = len(self._region_configs)

        primary = next(
            (r for r, c in self._region_configs.items() if c.is_primary),
            None,
        )

        return {
            "primary_region": primary.value if primary else None,
            "total_regions": total_regions,
            "healthy_regions": healthy_regions,
            "degraded_regions": total_regions - healthy_regions,
            "replication_count": len(self._replication_status),
            "active_failover_plans": len(self._failover_plans),
            "recent_failovers": len(
                [
                    e
                    for e in self._failover_events
                    if e.initiated_at > datetime.utcnow() - timedelta(hours=24)
                ]
            ),
            "auto_failover_enabled": self.config.auto_failover_enabled,
            "eu_data_residency": self.config.eu_data_residency,
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_multi_region_service(
    primary_region: Region = Region.EU_CENTRAL_1,
    secondary_regions: list[Region] | None = None,
    replication_mode: ReplicationMode = ReplicationMode.ASYNCHRONOUS,
    auto_failover_enabled: bool = True,
    **kwargs: Any,
) -> MultiRegionService:
    """Create multi-region service instance."""
    config = MultiRegionConfig(
        primary_region=primary_region,
        secondary_regions=secondary_regions or [Region.EU_WEST_1],
        replication_mode=replication_mode,
        auto_failover_enabled=auto_failover_enabled,
        **kwargs,
    )
    return MultiRegionService(config)
