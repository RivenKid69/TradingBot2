# -*- coding: utf-8 -*-
"""
Multi-Region Replication - CLOUD ZONE ONLY.

Cross-region artifact and data replication for enterprise deployments.

Design Doc Reference:
    - Section 4.4: "Artifact Registry"
    - Section 13.4: "Data Residency"
    - Enterprise high availability requirements

Features:
    - Asynchronous cross-region replication
    - Conflict resolution
    - Regional failover
    - Data residency compliance
    - Consistency guarantees
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

REPLICATION_LAG_THRESHOLD_SECONDS: Final[int] = 300  # 5 minutes
HEALTH_CHECK_INTERVAL_SECONDS: Final[int] = 30
SYNC_BATCH_SIZE: Final[int] = 100
MAX_RETRY_ATTEMPTS: Final[int] = 3


# ============================================================================
# Enums
# ============================================================================


class RegionStatus(str, Enum):
    """Region health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    SYNCING = "syncing"


class ReplicationMode(str, Enum):
    """Replication modes."""

    ASYNC = "async"  # Asynchronous replication (default)
    SYNC = "sync"  # Synchronous (write to all before ack)
    SEMI_SYNC = "semi_sync"  # Wait for at least one replica


class ConflictResolution(str, Enum):
    """Conflict resolution strategies."""

    LAST_WRITE_WINS = "last_write_wins"
    FIRST_WRITE_WINS = "first_write_wins"
    MANUAL = "manual"
    MERGE = "merge"


class ReplicationState(str, Enum):
    """Replication state for an item."""

    PENDING = "pending"
    REPLICATING = "replicating"
    REPLICATED = "replicated"
    FAILED = "failed"
    CONFLICT = "conflict"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class Region:
    """Represents a geographic region."""

    region_id: str
    name: str
    display_name: str

    # Location
    continent: str = ""
    country: str = ""
    city: str = ""

    # Network
    endpoint: str = ""
    internal_endpoint: str = ""

    # Status
    status: RegionStatus = RegionStatus.HEALTHY
    is_primary: bool = False
    is_writable: bool = True

    # Capabilities
    supports_artifacts: bool = True
    supports_telemetry: bool = True
    supports_configs: bool = True

    # Data residency
    data_residency_zones: List[str] = field(default_factory=list)
    # e.g., ["EU", "GDPR"] for EU region

    # Timestamps
    last_health_check: Optional[datetime] = None
    last_sync_completed: Optional[datetime] = None

    # Metrics
    replication_lag_seconds: float = 0.0
    pending_replications: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "name": self.name,
            "display_name": self.display_name,
            "continent": self.continent,
            "country": self.country,
            "endpoint": self.endpoint,
            "status": self.status.value,
            "is_primary": self.is_primary,
            "is_writable": self.is_writable,
            "data_residency_zones": self.data_residency_zones,
            "last_health_check": (
                self.last_health_check.isoformat() if self.last_health_check else None
            ),
            "replication_lag_seconds": self.replication_lag_seconds,
            "pending_replications": self.pending_replications,
        }


@dataclass
class ReplicationItem:
    """Item to be replicated across regions."""

    item_id: str = field(default_factory=lambda: str(uuid4()))
    item_type: str = ""  # artifact, config, metadata
    item_key: str = ""  # unique identifier
    content_hash: str = ""  # SHA256 of content

    # Source
    source_region: str = ""

    # State per region
    region_states: Dict[str, ReplicationState] = field(default_factory=dict)
    region_timestamps: Dict[str, datetime] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1
    size_bytes: int = 0

    # Conflict info
    has_conflict: bool = False
    conflict_regions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "item_key": self.item_key,
            "content_hash": self.content_hash,
            "source_region": self.source_region,
            "region_states": {k: v.value for k, v in self.region_states.items()},
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "has_conflict": self.has_conflict,
        }


@dataclass
class ReplicationEvent:
    """Event in the replication log."""

    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: str = ""  # created, updated, deleted, replicated
    item_id: str = ""
    item_key: str = ""
    source_region: str = ""
    target_region: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "item_id": self.item_id,
            "item_key": self.item_key,
            "source_region": self.source_region,
            "target_region": self.target_region,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass
class ReplicationConfig:
    """Multi-region replication configuration."""

    # Regions
    primary_region: str = "us-east-1"
    replica_regions: List[str] = field(default_factory=list)

    # Mode
    replication_mode: ReplicationMode = ReplicationMode.ASYNC
    conflict_resolution: ConflictResolution = ConflictResolution.LAST_WRITE_WINS

    # Performance
    batch_size: int = SYNC_BATCH_SIZE
    max_concurrent_replications: int = 10
    retry_attempts: int = MAX_RETRY_ATTEMPTS
    retry_delay_seconds: int = 5

    # Health
    health_check_interval_seconds: int = HEALTH_CHECK_INTERVAL_SECONDS
    lag_threshold_seconds: int = REPLICATION_LAG_THRESHOLD_SECONDS

    # Data residency
    enforce_data_residency: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_region": self.primary_region,
            "replica_regions": self.replica_regions,
            "replication_mode": self.replication_mode.value,
            "conflict_resolution": self.conflict_resolution.value,
            "batch_size": self.batch_size,
            "enforce_data_residency": self.enforce_data_residency,
        }


@dataclass
class ReplicationStats:
    """Replication statistics."""

    total_items: int = 0
    replicated_items: int = 0
    pending_items: int = 0
    failed_items: int = 0
    conflict_items: int = 0

    # Per region
    region_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Timing
    avg_replication_time_ms: float = 0.0
    p95_replication_time_ms: float = 0.0
    max_lag_seconds: float = 0.0

    # Throughput
    items_per_second: float = 0.0
    bytes_per_second: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_items": self.total_items,
            "replicated_items": self.replicated_items,
            "pending_items": self.pending_items,
            "failed_items": self.failed_items,
            "conflict_items": self.conflict_items,
            "region_stats": self.region_stats,
            "avg_replication_time_ms": self.avg_replication_time_ms,
            "max_lag_seconds": self.max_lag_seconds,
        }


# ============================================================================
# Multi-Region Replication Manager
# ============================================================================


class MultiRegionReplicator:
    """
    Manages cross-region artifact and data replication.

    Provides:
    - Asynchronous replication to multiple regions
    - Conflict detection and resolution
    - Regional failover support
    - Data residency compliance

    Usage:
        config = ReplicationConfig(
            primary_region="us-east-1",
            replica_regions=["eu-west-1", "ap-southeast-1"],
        )
        replicator = MultiRegionReplicator(config)
        await replicator.start()

        # Replicate an item
        await replicator.replicate("artifact", "sha256:abc...", content_bytes)
    """

    def __init__(
        self,
        config: Optional[ReplicationConfig] = None,
        on_conflict: Optional[Callable[[ReplicationItem], None]] = None,
        on_failover: Optional[Callable[[str, str], None]] = None,
    ):
        self.config = config or ReplicationConfig()
        self._on_conflict = on_conflict
        self._on_failover = on_failover

        # Regions
        self._regions: Dict[str, Region] = {}
        self._primary_region: Optional[str] = None

        # Items
        self._items: Dict[str, ReplicationItem] = {}
        self._pending_queue: asyncio.Queue = asyncio.Queue()

        # Events
        self._events: List[ReplicationEvent] = []
        self._max_events = 10000

        # State
        self._running = False
        self._lock = asyncio.Lock()

        # Background tasks
        self._replication_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None

        logger.info("MultiRegionReplicator initialized")

    # ========================================================================
    # Region Management
    # ========================================================================

    def register_region(self, region: Region) -> None:
        """
        Register a region for replication.

        Args:
            region: Region configuration
        """
        self._regions[region.region_id] = region

        if region.is_primary:
            self._primary_region = region.region_id

        logger.info(f"Registered region: {region.name} ({region.region_id})")

    def get_region(self, region_id: str) -> Optional[Region]:
        """Get region by ID."""
        return self._regions.get(region_id)

    def list_regions(self, status: Optional[RegionStatus] = None) -> List[Region]:
        """List all regions, optionally filtered by status."""
        regions = list(self._regions.values())
        if status:
            regions = [r for r in regions if r.status == status]
        return regions

    def get_primary_region(self) -> Optional[Region]:
        """Get the primary region."""
        if self._primary_region:
            return self._regions.get(self._primary_region)
        return None

    async def promote_region(self, region_id: str) -> bool:
        """
        Promote a region to primary.

        Args:
            region_id: Region to promote

        Returns:
            True if successful
        """
        async with self._lock:
            region = self._regions.get(region_id)
            if not region:
                return False

            old_primary = self._primary_region

            # Demote current primary
            if old_primary and old_primary in self._regions:
                self._regions[old_primary].is_primary = False

            # Promote new primary
            region.is_primary = True
            self._primary_region = region_id

            logger.warning(f"Region promoted to primary: {region_id} (was: {old_primary})")

            # Trigger failover callback
            if self._on_failover and old_primary:
                try:
                    self._on_failover(old_primary, region_id)
                except Exception as e:
                    logger.error(f"Failover callback error: {e}")

        return True

    # ========================================================================
    # Replication Operations
    # ========================================================================

    async def replicate(
        self,
        item_type: str,
        item_key: str,
        content: bytes,
        source_region: Optional[str] = None,
        target_regions: Optional[List[str]] = None,
    ) -> ReplicationItem:
        """
        Replicate an item to target regions.

        Args:
            item_type: Type of item (artifact, config, etc.)
            item_key: Unique key for the item
            content: Item content
            source_region: Source region (defaults to primary)
            target_regions: Target regions (defaults to all replicas)

        Returns:
            ReplicationItem with status
        """
        source = source_region or self._primary_region
        targets = target_regions or self.config.replica_regions

        # Create or update item
        content_hash = hashlib.sha256(content).hexdigest()

        existing = self._items.get(item_key)
        if existing:
            # Check for conflict
            if existing.content_hash != content_hash and existing.source_region != source:
                return await self._handle_conflict(existing, content_hash, source)

            existing.content_hash = content_hash
            existing.updated_at = datetime.now(timezone.utc)
            existing.version += 1
            item = existing
        else:
            item = ReplicationItem(
                item_type=item_type,
                item_key=item_key,
                content_hash=content_hash,
                source_region=source or "",
                size_bytes=len(content),
            )
            self._items[item_key] = item

        # Mark source as replicated
        item.region_states[source or ""] = ReplicationState.REPLICATED
        item.region_timestamps[source or ""] = datetime.now(timezone.utc)

        # Queue replication to targets
        for target in targets:
            if target != source:
                # Check data residency
                if self.config.enforce_data_residency:
                    if not self._check_data_residency(item, target):
                        continue

                item.region_states[target] = ReplicationState.PENDING
                await self._pending_queue.put((item, target, content))

        # Log event
        self._log_event(
            ReplicationEvent(
                event_type="created" if not existing else "updated",
                item_id=item.item_id,
                item_key=item_key,
                source_region=source or "",
            )
        )

        return item

    async def get_item(
        self,
        item_key: str,
        preferred_region: Optional[str] = None,
    ) -> Optional[ReplicationItem]:
        """
        Get replication item status.

        Args:
            item_key: Item key
            preferred_region: Preferred region to check first

        Returns:
            ReplicationItem or None
        """
        return self._items.get(item_key)

    async def delete_item(
        self,
        item_key: str,
        cascade: bool = True,
    ) -> bool:
        """
        Delete item and optionally propagate to all regions.

        Args:
            item_key: Item key
            cascade: Delete from all regions

        Returns:
            True if deleted
        """
        item = self._items.pop(item_key, None)
        if not item:
            return False

        if cascade:
            # Queue deletion to all regions
            for region_id in item.region_states.keys():
                self._log_event(
                    ReplicationEvent(
                        event_type="deleted",
                        item_id=item.item_id,
                        item_key=item_key,
                        target_region=region_id,
                    )
                )

        return True

    # ========================================================================
    # Conflict Handling
    # ========================================================================

    async def _handle_conflict(
        self,
        item: ReplicationItem,
        new_hash: str,
        new_source: str,
    ) -> ReplicationItem:
        """Handle replication conflict."""
        item.has_conflict = True
        if new_source not in item.conflict_regions:
            item.conflict_regions.append(new_source)

        if self.config.conflict_resolution == ConflictResolution.LAST_WRITE_WINS:
            # Update with new content
            item.content_hash = new_hash
            item.source_region = new_source
            item.has_conflict = False
            item.conflict_regions = []
            logger.info(f"Conflict resolved (last-write-wins): {item.item_key}")

        elif self.config.conflict_resolution == ConflictResolution.FIRST_WRITE_WINS:
            # Keep existing
            logger.info(f"Conflict resolved (first-write-wins): {item.item_key}")

        elif self.config.conflict_resolution == ConflictResolution.MANUAL:
            # Mark for manual resolution
            item.region_states[new_source] = ReplicationState.CONFLICT
            logger.warning(f"Conflict requires manual resolution: {item.item_key}")

            if self._on_conflict:
                try:
                    self._on_conflict(item)
                except Exception as e:
                    logger.error(f"Conflict callback error: {e}")

        return item

    async def resolve_conflict(
        self,
        item_key: str,
        winning_region: str,
        content: bytes,
    ) -> bool:
        """
        Manually resolve a conflict.

        Args:
            item_key: Item key
            winning_region: Region whose content should win
            content: Content to use

        Returns:
            True if resolved
        """
        item = self._items.get(item_key)
        if not item or not item.has_conflict:
            return False

        item.content_hash = hashlib.sha256(content).hexdigest()
        item.source_region = winning_region
        item.has_conflict = False
        item.conflict_regions = []
        item.version += 1

        # Re-queue for replication
        for region_id in self._regions:
            if region_id != winning_region:
                item.region_states[region_id] = ReplicationState.PENDING
                await self._pending_queue.put((item, region_id, content))

        logger.info(f"Conflict manually resolved: {item_key} (winner: {winning_region})")
        return True

    # ========================================================================
    # Data Residency
    # ========================================================================

    def _check_data_residency(self, item: ReplicationItem, target_region: str) -> bool:
        """Check if item can be replicated to target region."""
        target = self._regions.get(target_region)
        if not target:
            return False

        # Get item's residency requirements from metadata (if any)
        # In production, this would check the item's workspace/tenant settings

        return True

    def set_item_residency(
        self,
        item_key: str,
        allowed_zones: List[str],
    ) -> bool:
        """
        Set data residency restrictions for an item.

        Args:
            item_key: Item key
            allowed_zones: List of allowed residency zones

        Returns:
            True if set
        """
        item = self._items.get(item_key)
        if not item:
            return False

        # Remove from regions not in allowed zones
        for region_id, region in self._regions.items():
            if not any(zone in region.data_residency_zones for zone in allowed_zones):
                if region_id in item.region_states:
                    del item.region_states[region_id]

        return True

    # ========================================================================
    # Background Processing
    # ========================================================================

    async def start(self) -> None:
        """Start replication services."""
        if self._running:
            return

        self._running = True

        # Start replication worker
        self._replication_task = asyncio.create_task(self._replication_worker())

        # Start health check
        self._health_check_task = asyncio.create_task(self._health_check_worker())

        logger.info("MultiRegionReplicator started")

    async def stop(self) -> None:
        """Stop replication services."""
        self._running = False

        if self._replication_task:
            self._replication_task.cancel()
            try:
                await self._replication_task
            except asyncio.CancelledError:
                pass

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        logger.info("MultiRegionReplicator stopped")

    async def _replication_worker(self) -> None:
        """Background worker for processing replication queue."""
        while self._running:
            try:
                # Get batch from queue
                batch: List[Tuple[ReplicationItem, str, bytes]] = []

                try:
                    item_data = await asyncio.wait_for(
                        self._pending_queue.get(),
                        timeout=1.0,
                    )
                    batch.append(item_data)

                    # Try to get more items for batch
                    while len(batch) < self.config.batch_size:
                        try:
                            item_data = self._pending_queue.get_nowait()
                            batch.append(item_data)
                        except asyncio.QueueEmpty:
                            break

                except asyncio.TimeoutError:
                    continue

                # Process batch
                for item, target_region, content in batch:
                    await self._replicate_to_region(item, target_region, content)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Replication worker error: {e}")

    async def _replicate_to_region(
        self,
        item: ReplicationItem,
        target_region: str,
        content: bytes,
    ) -> bool:
        """Replicate item to a specific region."""
        item.region_states[target_region] = ReplicationState.REPLICATING
        start_time = time.perf_counter()

        for attempt in range(self.config.retry_attempts):
            try:
                # In production: call region's API to store content
                # await self._regions[target_region].store(item.item_key, content)

                # Simulate success
                await asyncio.sleep(0.01)  # Simulated network latency

                item.region_states[target_region] = ReplicationState.REPLICATED
                item.region_timestamps[target_region] = datetime.now(timezone.utc)

                self._log_event(
                    ReplicationEvent(
                        event_type="replicated",
                        item_id=item.item_id,
                        item_key=item.item_key,
                        source_region=item.source_region,
                        target_region=target_region,
                        success=True,
                    )
                )

                # Update region metrics
                region = self._regions.get(target_region)
                if region:
                    region.pending_replications = max(0, region.pending_replications - 1)
                    region.last_sync_completed = datetime.now(timezone.utc)

                return True

            except Exception as e:
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay_seconds)
                else:
                    item.region_states[target_region] = ReplicationState.FAILED

                    self._log_event(
                        ReplicationEvent(
                            event_type="replicated",
                            item_id=item.item_id,
                            item_key=item.item_key,
                            source_region=item.source_region,
                            target_region=target_region,
                            success=False,
                            error_message=str(e),
                        )
                    )

                    logger.error(f"Replication failed: {item.item_key} -> {target_region}: {e}")
                    return False

        return False

    async def _health_check_worker(self) -> None:
        """Background worker for region health checks."""
        while self._running:
            try:
                for region_id, region in self._regions.items():
                    try:
                        # In production: ping region endpoint
                        # healthy = await self._ping_region(region)

                        region.last_health_check = datetime.now(timezone.utc)

                        # Calculate replication lag
                        lag = self._calculate_lag(region_id)
                        region.replication_lag_seconds = lag

                        if lag > self.config.lag_threshold_seconds:
                            region.status = RegionStatus.DEGRADED
                        else:
                            region.status = RegionStatus.HEALTHY

                    except Exception as e:
                        region.status = RegionStatus.UNHEALTHY
                        logger.warning(f"Health check failed for {region_id}: {e}")

                await asyncio.sleep(self.config.health_check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Health check worker error: {e}")

    def _calculate_lag(self, region_id: str) -> float:
        """Calculate replication lag for a region."""
        max_lag = 0.0
        now = datetime.now(timezone.utc)

        for item in self._items.values():
            if region_id in item.region_timestamps:
                source_time = item.region_timestamps.get(item.source_region)
                target_time = item.region_timestamps.get(region_id)

                if source_time and target_time:
                    lag = (target_time - source_time).total_seconds()
                    max_lag = max(max_lag, lag)
            elif region_id in item.region_states:
                if item.region_states[region_id] == ReplicationState.PENDING:
                    # Item waiting to replicate
                    created = item.created_at
                    lag = (now - created).total_seconds()
                    max_lag = max(max_lag, lag)

        return max_lag

    # ========================================================================
    # Statistics & Events
    # ========================================================================

    def get_stats(self) -> ReplicationStats:
        """Get replication statistics."""
        stats = ReplicationStats()

        for item in self._items.values():
            stats.total_items += 1

            if item.has_conflict:
                stats.conflict_items += 1

            for region_id, state in item.region_states.items():
                if region_id not in stats.region_stats:
                    stats.region_stats[region_id] = {
                        "pending": 0,
                        "replicated": 0,
                        "failed": 0,
                    }

                if state == ReplicationState.PENDING:
                    stats.pending_items += 1
                    stats.region_stats[region_id]["pending"] += 1
                elif state == ReplicationState.REPLICATED:
                    stats.replicated_items += 1
                    stats.region_stats[region_id]["replicated"] += 1
                elif state == ReplicationState.FAILED:
                    stats.failed_items += 1
                    stats.region_stats[region_id]["failed"] += 1

        # Calculate max lag
        for region in self._regions.values():
            stats.max_lag_seconds = max(stats.max_lag_seconds, region.replication_lag_seconds)

        return stats

    def _log_event(self, event: ReplicationEvent) -> None:
        """Log a replication event."""
        self._events.append(event)

        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events // 2 :]

    def get_events(
        self,
        limit: int = 100,
        item_key: Optional[str] = None,
        region: Optional[str] = None,
    ) -> List[ReplicationEvent]:
        """Get recent replication events."""
        events = self._events

        if item_key:
            events = [e for e in events if e.item_key == item_key]

        if region:
            events = [e for e in events if e.source_region == region or e.target_region == region]

        return list(reversed(events))[:limit]


# ============================================================================
# Default Regions
# ============================================================================


def create_default_regions() -> List[Region]:
    """Create default region configurations."""
    return [
        Region(
            region_id="us-east-1",
            name="us-east-1",
            display_name="US East (N. Virginia)",
            continent="North America",
            country="US",
            city="Virginia",
            endpoint="https://us-east-1.registry.ccea.io",
            is_primary=True,
            data_residency_zones=["US", "GLOBAL"],
        ),
        Region(
            region_id="us-west-2",
            name="us-west-2",
            display_name="US West (Oregon)",
            continent="North America",
            country="US",
            city="Oregon",
            endpoint="https://us-west-2.registry.ccea.io",
            data_residency_zones=["US", "GLOBAL"],
        ),
        Region(
            region_id="eu-west-1",
            name="eu-west-1",
            display_name="EU (Ireland)",
            continent="Europe",
            country="IE",
            city="Dublin",
            endpoint="https://eu-west-1.registry.ccea.io",
            data_residency_zones=["EU", "GDPR", "GLOBAL"],
        ),
        Region(
            region_id="eu-central-1",
            name="eu-central-1",
            display_name="EU (Frankfurt)",
            continent="Europe",
            country="DE",
            city="Frankfurt",
            endpoint="https://eu-central-1.registry.ccea.io",
            data_residency_zones=["EU", "GDPR", "GLOBAL"],
        ),
        Region(
            region_id="ap-southeast-1",
            name="ap-southeast-1",
            display_name="Asia Pacific (Singapore)",
            continent="Asia",
            country="SG",
            city="Singapore",
            endpoint="https://ap-southeast-1.registry.ccea.io",
            data_residency_zones=["APAC", "GLOBAL"],
        ),
    ]
