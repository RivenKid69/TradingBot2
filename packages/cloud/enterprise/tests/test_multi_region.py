# -*- coding: utf-8 -*-
"""
Tests for Multi-Region Replication.

Tests for:
- MultiRegionReplicator
- Region management
- Replication operations
- Conflict resolution
"""

import asyncio
import pytest
from uuid import uuid4

from packages.cloud.enterprise import (
    MultiRegionReplicator,
    Region,
    RegionStatus,
    ReplicationConfig,
    ReplicationMode,
    ReplicationItem,
    ReplicationStats,
    ConflictResolution,
    create_default_regions,
)
from packages.cloud.enterprise.multi_region import ReplicationState


# ============================================================================
# Region Tests
# ============================================================================


class TestRegion:
    """Tests for Region class."""

    def test_region_creation(self):
        """Test Region creation."""
        region = Region(
            region_id="us-east-1",
            name="us-east-1",
            display_name="US East (N. Virginia)",
            endpoint="https://us-east-1.example.com",
        )

        assert region.region_id == "us-east-1"
        assert region.status == RegionStatus.HEALTHY

    def test_region_with_data_residency(self):
        """Test Region with data residency zones."""
        region = Region(
            region_id="eu-west-1",
            name="eu-west-1",
            display_name="EU (Ireland)",
            data_residency_zones=["EU", "GDPR"],
        )

        assert "EU" in region.data_residency_zones
        assert "GDPR" in region.data_residency_zones

    def test_region_to_dict(self):
        """Test Region serialization."""
        region = Region(
            region_id="ap-southeast-1",
            name="ap-southeast-1",
            display_name="Asia Pacific",
            is_primary=True,
        )

        data = region.to_dict()

        assert data["region_id"] == "ap-southeast-1"
        assert data["is_primary"] is True


class TestCreateDefaultRegions:
    """Tests for create_default_regions function."""

    def test_creates_multiple_regions(self):
        """Test that default regions are created."""
        regions = create_default_regions()

        assert len(regions) > 0
        assert any(r.region_id == "us-east-1" for r in regions)
        assert any(r.region_id == "eu-west-1" for r in regions)

    def test_one_primary_region(self):
        """Test that exactly one region is primary."""
        regions = create_default_regions()

        primary_count = sum(1 for r in regions if r.is_primary)
        assert primary_count == 1

    def test_all_regions_healthy(self):
        """Test that all default regions are healthy."""
        regions = create_default_regions()

        assert all(r.status == RegionStatus.HEALTHY for r in regions)


# ============================================================================
# ReplicationConfig Tests
# ============================================================================


class TestReplicationConfig:
    """Tests for ReplicationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ReplicationConfig()

        assert config.primary_region == "us-east-1"
        assert config.replication_mode == ReplicationMode.ASYNC
        assert config.conflict_resolution == ConflictResolution.LAST_WRITE_WINS

    def test_custom_config(self):
        """Test custom configuration."""
        config = ReplicationConfig(
            primary_region="eu-west-1",
            replica_regions=["us-east-1", "ap-southeast-1"],
            replication_mode=ReplicationMode.SEMI_SYNC,
            conflict_resolution=ConflictResolution.MANUAL,
        )

        assert config.primary_region == "eu-west-1"
        assert len(config.replica_regions) == 2
        assert config.replication_mode == ReplicationMode.SEMI_SYNC

    def test_config_to_dict(self):
        """Test config serialization."""
        config = ReplicationConfig(
            primary_region="us-west-2",
            batch_size=50,
        )

        data = config.to_dict()

        assert data["primary_region"] == "us-west-2"
        assert data["batch_size"] == 50


# ============================================================================
# MultiRegionReplicator Tests
# ============================================================================


class TestMultiRegionReplicator:
    """Tests for MultiRegionReplicator."""

    @pytest.fixture
    def replicator(self):
        """Create MultiRegionReplicator instance."""
        config = ReplicationConfig(
            primary_region="us-east-1",
            replica_regions=["eu-west-1"],
        )
        return MultiRegionReplicator(config)

    @pytest.fixture
    def regions(self):
        """Create test regions."""
        return [
            Region(
                region_id="us-east-1",
                name="us-east-1",
                display_name="US East",
                is_primary=True,
            ),
            Region(
                region_id="eu-west-1",
                name="eu-west-1",
                display_name="EU West",
            ),
        ]

    def test_register_region(self, replicator, regions):
        """Test registering regions."""
        for region in regions:
            replicator.register_region(region)

        assert len(replicator._regions) == 2
        assert replicator._primary_region == "us-east-1"

    def test_get_region(self, replicator, regions):
        """Test getting region by ID."""
        for region in regions:
            replicator.register_region(region)

        region = replicator.get_region("us-east-1")

        assert region is not None
        assert region.region_id == "us-east-1"

    def test_list_regions(self, replicator, regions):
        """Test listing all regions."""
        for region in regions:
            replicator.register_region(region)

        all_regions = replicator.list_regions()

        assert len(all_regions) == 2

    def test_list_regions_by_status(self, replicator, regions):
        """Test listing regions by status."""
        for region in regions:
            replicator.register_region(region)

        # Make one region degraded
        replicator._regions["eu-west-1"].status = RegionStatus.DEGRADED

        healthy = replicator.list_regions(status=RegionStatus.HEALTHY)
        degraded = replicator.list_regions(status=RegionStatus.DEGRADED)

        assert len(healthy) == 1
        assert len(degraded) == 1

    def test_get_primary_region(self, replicator, regions):
        """Test getting primary region."""
        for region in regions:
            replicator.register_region(region)

        primary = replicator.get_primary_region()

        assert primary is not None
        assert primary.is_primary is True
        assert primary.region_id == "us-east-1"

    @pytest.mark.asyncio
    async def test_promote_region(self, replicator, regions):
        """Test promoting region to primary."""
        for region in regions:
            replicator.register_region(region)

        result = await replicator.promote_region("eu-west-1")

        assert result is True
        assert replicator._primary_region == "eu-west-1"
        assert replicator._regions["us-east-1"].is_primary is False
        assert replicator._regions["eu-west-1"].is_primary is True

    @pytest.mark.asyncio
    async def test_replicate_item(self, replicator, regions):
        """Test replicating an item."""
        for region in regions:
            replicator.register_region(region)

        item = await replicator.replicate(
            item_type="artifact",
            item_key="sha256:abc123",
            content=b"test content",
            source_region="us-east-1",
        )

        assert item is not None
        assert item.item_key == "sha256:abc123"
        assert item.source_region == "us-east-1"

    @pytest.mark.asyncio
    async def test_get_item(self, replicator, regions):
        """Test getting replication item."""
        for region in regions:
            replicator.register_region(region)

        await replicator.replicate(
            item_type="config",
            item_key="config-123",
            content=b"config data",
        )

        item = await replicator.get_item("config-123")

        assert item is not None
        assert item.item_type == "config"

    @pytest.mark.asyncio
    async def test_delete_item(self, replicator, regions):
        """Test deleting replication item."""
        for region in regions:
            replicator.register_region(region)

        await replicator.replicate(
            item_type="artifact",
            item_key="to-delete",
            content=b"content",
        )

        deleted = await replicator.delete_item("to-delete")

        assert deleted is True
        assert await replicator.get_item("to-delete") is None

    @pytest.mark.asyncio
    async def test_start_stop(self, replicator):
        """Test starting and stopping replicator."""
        await replicator.start()
        assert replicator._running is True

        await replicator.stop()
        assert replicator._running is False

    def test_get_stats(self, replicator, regions):
        """Test getting replication stats."""
        for region in regions:
            replicator.register_region(region)

        stats = replicator.get_stats()

        assert isinstance(stats, ReplicationStats)
        assert stats.total_items >= 0

    def test_get_events(self, replicator):
        """Test getting replication events."""
        events = replicator.get_events(limit=10)

        assert isinstance(events, list)


# ============================================================================
# Conflict Resolution Tests
# ============================================================================


class TestConflictResolution:
    """Tests for conflict resolution strategies."""

    @pytest.fixture
    def replicator_last_write(self):
        """Create replicator with last-write-wins resolution."""
        config = ReplicationConfig(
            conflict_resolution=ConflictResolution.LAST_WRITE_WINS,
        )
        return MultiRegionReplicator(config)

    @pytest.fixture
    def replicator_first_write(self):
        """Create replicator with first-write-wins resolution."""
        config = ReplicationConfig(
            conflict_resolution=ConflictResolution.FIRST_WRITE_WINS,
        )
        return MultiRegionReplicator(config)

    @pytest.fixture
    def replicator_manual(self):
        """Create replicator with manual resolution."""
        config = ReplicationConfig(
            conflict_resolution=ConflictResolution.MANUAL,
        )
        return MultiRegionReplicator(config)

    @pytest.mark.asyncio
    async def test_last_write_wins(self, replicator_last_write):
        """Test last-write-wins conflict resolution."""
        replicator = replicator_last_write

        regions = [
            Region(region_id="r1", name="r1", display_name="R1", is_primary=True),
            Region(region_id="r2", name="r2", display_name="R2"),
        ]
        for r in regions:
            replicator.register_region(r)

        # First write
        await replicator.replicate(
            item_type="artifact",
            item_key="conflict-key",
            content=b"first content",
            source_region="r1",
        )

        # Second write from different region (should win)
        item = await replicator.replicate(
            item_type="artifact",
            item_key="conflict-key",
            content=b"second content",
            source_region="r2",
        )

        # Conflict resolved - should have second content hash
        assert item.has_conflict is False

    @pytest.mark.asyncio
    async def test_manual_resolution(self, replicator_manual):
        """Test manual conflict resolution."""
        replicator = replicator_manual

        regions = [
            Region(region_id="r1", name="r1", display_name="R1", is_primary=True),
            Region(region_id="r2", name="r2", display_name="R2"),
        ]
        for r in regions:
            replicator.register_region(r)

        # First write
        await replicator.replicate(
            item_type="artifact",
            item_key="manual-conflict",
            content=b"first",
            source_region="r1",
        )

        # Second write causing conflict
        item = await replicator.replicate(
            item_type="artifact",
            item_key="manual-conflict",
            content=b"second",
            source_region="r2",
        )

        # Should have conflict
        assert item.has_conflict is True
        assert ReplicationState.CONFLICT in item.region_states.values()

    @pytest.mark.asyncio
    async def test_resolve_conflict_manually(self, replicator_manual):
        """Test manually resolving a conflict."""
        replicator = replicator_manual

        regions = [
            Region(region_id="r1", name="r1", display_name="R1", is_primary=True),
            Region(region_id="r2", name="r2", display_name="R2"),
        ]
        for r in regions:
            replicator.register_region(r)

        # Create conflict
        await replicator.replicate(
            item_type="artifact",
            item_key="to-resolve",
            content=b"content1",
            source_region="r1",
        )
        await replicator.replicate(
            item_type="artifact",
            item_key="to-resolve",
            content=b"content2",
            source_region="r2",
        )

        # Resolve manually
        resolved = await replicator.resolve_conflict(
            "to-resolve",
            winning_region="r1",
            content=b"content1",
        )

        assert resolved is True
        item = await replicator.get_item("to-resolve")
        assert item.has_conflict is False


# ============================================================================
# ReplicationItem Tests
# ============================================================================


class TestReplicationItem:
    """Tests for ReplicationItem."""

    def test_item_creation(self):
        """Test ReplicationItem creation."""
        item = ReplicationItem(
            item_type="artifact",
            item_key="sha256:abc",
            content_hash="def123",
            source_region="us-east-1",
        )

        assert item.item_type == "artifact"
        assert item.version == 1

    def test_item_to_dict(self):
        """Test ReplicationItem serialization."""
        item = ReplicationItem(
            item_type="config",
            item_key="config-1",
            source_region="eu-west-1",
        )

        data = item.to_dict()

        assert data["item_type"] == "config"
        assert data["source_region"] == "eu-west-1"


# ============================================================================
# ReplicationStats Tests
# ============================================================================


class TestReplicationStats:
    """Tests for ReplicationStats."""

    def test_stats_creation(self):
        """Test ReplicationStats creation."""
        stats = ReplicationStats(
            total_items=100,
            replicated_items=95,
            pending_items=5,
        )

        assert stats.total_items == 100
        assert stats.replicated_items == 95

    def test_stats_to_dict(self):
        """Test ReplicationStats serialization."""
        stats = ReplicationStats(
            total_items=50,
            failed_items=2,
            max_lag_seconds=5.5,
        )

        data = stats.to_dict()

        assert data["total_items"] == 50
        assert data["failed_items"] == 2
        assert data["max_lag_seconds"] == 5.5
