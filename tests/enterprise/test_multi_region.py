# -*- coding: utf-8 -*-
"""
Comprehensive tests for Multi-Region Deployment Service.

Tests multi-region capabilities per DORA Art. 11 resilience requirements.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from services.enterprise.multi_region import (
    # Enums
    Region,
    RegionStatus,
    ReplicationMode,
    FailoverStatus,
    # Data structures
    RegionConfig,
    RegionHealth,
    ReplicationStatus,
    FailoverPlan,
    FailoverEvent,
    RegionDeployment,
    MultiRegionConfig,
    # Service
    MultiRegionService,
    # Factory
    create_multi_region_service,
)


# =============================================================================
# RegionHealth Tests
# =============================================================================


class TestRegionHealth:
    """Tests for RegionHealth dataclass."""

    def test_create_region_health(self) -> None:
        """Test creating region health metrics."""
        health = RegionHealth(
            region=Region.EU_CENTRAL_1,
            status=RegionStatus.ACTIVE,
            latency_ms=50.0,
            availability_percent=99.95,
            error_rate_percent=0.1,
            active_connections=5000,
            cpu_utilization_percent=45.0,
            memory_utilization_percent=60.0,
            storage_utilization_percent=30.0,
            last_check=datetime.utcnow(),
        )
        assert health.region == Region.EU_CENTRAL_1
        assert health.status == RegionStatus.ACTIVE

    def test_is_healthy_true(self) -> None:
        """Test is_healthy returns True for healthy region."""
        health = RegionHealth(
            region=Region.EU_CENTRAL_1,
            status=RegionStatus.ACTIVE,
            latency_ms=50.0,
            availability_percent=99.95,
            error_rate_percent=0.1,
            active_connections=5000,
            cpu_utilization_percent=45.0,
            memory_utilization_percent=60.0,
            storage_utilization_percent=30.0,
            last_check=datetime.utcnow(),
        )
        assert health.is_healthy is True

    def test_is_healthy_false_status(self) -> None:
        """Test is_healthy returns False for non-active status."""
        health = RegionHealth(
            region=Region.EU_CENTRAL_1,
            status=RegionStatus.DEGRADED,
            latency_ms=50.0,
            availability_percent=99.95,
            error_rate_percent=0.1,
            active_connections=5000,
            cpu_utilization_percent=45.0,
            memory_utilization_percent=60.0,
            storage_utilization_percent=30.0,
            last_check=datetime.utcnow(),
        )
        assert health.is_healthy is False

    def test_is_healthy_false_availability(self) -> None:
        """Test is_healthy returns False for low availability."""
        health = RegionHealth(
            region=Region.EU_CENTRAL_1,
            status=RegionStatus.ACTIVE,
            latency_ms=50.0,
            availability_percent=98.0,  # Below 99%
            error_rate_percent=0.1,
            active_connections=5000,
            cpu_utilization_percent=45.0,
            memory_utilization_percent=60.0,
            storage_utilization_percent=30.0,
            last_check=datetime.utcnow(),
        )
        assert health.is_healthy is False

    def test_is_healthy_false_high_cpu(self) -> None:
        """Test is_healthy returns False for high CPU."""
        health = RegionHealth(
            region=Region.EU_CENTRAL_1,
            status=RegionStatus.ACTIVE,
            latency_ms=50.0,
            availability_percent=99.95,
            error_rate_percent=0.1,
            active_connections=5000,
            cpu_utilization_percent=95.0,  # Above 90%
            memory_utilization_percent=60.0,
            storage_utilization_percent=30.0,
            last_check=datetime.utcnow(),
        )
        assert health.is_healthy is False


# =============================================================================
# ReplicationStatus Tests
# =============================================================================


class TestReplicationStatus:
    """Tests for ReplicationStatus dataclass."""

    def test_create_replication_status(self) -> None:
        """Test creating replication status."""
        status = ReplicationStatus(
            source_region=Region.EU_CENTRAL_1,
            target_region=Region.EU_WEST_1,
            mode=ReplicationMode.ASYNCHRONOUS,
            lag_seconds=0.5,
            bytes_pending=1000,
            is_healthy=True,
            last_sync=datetime.utcnow(),
        )
        assert status.source_region == Region.EU_CENTRAL_1
        assert status.target_region == Region.EU_WEST_1

    def test_lag_status_excellent(self) -> None:
        """Test lag status for excellent latency."""
        status = ReplicationStatus(
            source_region=Region.EU_CENTRAL_1,
            target_region=Region.EU_WEST_1,
            mode=ReplicationMode.SYNCHRONOUS,
            lag_seconds=0.5,
            bytes_pending=0,
            is_healthy=True,
            last_sync=datetime.utcnow(),
        )
        assert status.lag_status == "excellent"

    def test_lag_status_good(self) -> None:
        """Test lag status for good latency."""
        status = ReplicationStatus(
            source_region=Region.EU_CENTRAL_1,
            target_region=Region.EU_WEST_1,
            mode=ReplicationMode.ASYNCHRONOUS,
            lag_seconds=3.0,
            bytes_pending=0,
            is_healthy=True,
            last_sync=datetime.utcnow(),
        )
        assert status.lag_status == "good"

    def test_lag_status_acceptable(self) -> None:
        """Test lag status for acceptable latency."""
        status = ReplicationStatus(
            source_region=Region.EU_CENTRAL_1,
            target_region=Region.EU_WEST_1,
            mode=ReplicationMode.ASYNCHRONOUS,
            lag_seconds=15.0,
            bytes_pending=0,
            is_healthy=True,
            last_sync=datetime.utcnow(),
        )
        assert status.lag_status == "acceptable"

    def test_lag_status_degraded(self) -> None:
        """Test lag status for degraded latency."""
        status = ReplicationStatus(
            source_region=Region.EU_CENTRAL_1,
            target_region=Region.EU_WEST_1,
            mode=ReplicationMode.ASYNCHRONOUS,
            lag_seconds=60.0,
            bytes_pending=0,
            is_healthy=False,
            last_sync=datetime.utcnow(),
        )
        assert status.lag_status == "degraded"


# =============================================================================
# FailoverEvent Tests
# =============================================================================


class TestFailoverEvent:
    """Tests for FailoverEvent dataclass."""

    def test_create_failover_event(self) -> None:
        """Test creating failover event."""
        event = FailoverEvent(
            event_id="event-1",
            plan_id="plan-1",
            source_region=Region.EU_CENTRAL_1,
            target_region=Region.EU_WEST_1,
            trigger="manual",
            initiated_at=datetime.utcnow(),
            initiated_by="admin",
            status=FailoverStatus.INITIATED,
        )
        assert event.event_id == "event-1"
        assert event.status == FailoverStatus.INITIATED

    def test_complete_failover(self) -> None:
        """Test completing a failover event."""
        event = FailoverEvent(
            event_id="event-1",
            plan_id="plan-1",
            source_region=Region.EU_CENTRAL_1,
            target_region=Region.EU_WEST_1,
            trigger="auto",
            initiated_at=datetime.utcnow() - timedelta(seconds=30),
            initiated_by="auto",
            status=FailoverStatus.IN_PROGRESS,
        )
        event.complete()
        assert event.status == FailoverStatus.COMPLETED
        assert event.completed_at is not None
        assert event.duration_seconds is not None
        assert event.duration_seconds >= 30

    def test_fail_failover(self) -> None:
        """Test failing a failover event."""
        event = FailoverEvent(
            event_id="event-1",
            plan_id="plan-1",
            source_region=Region.EU_CENTRAL_1,
            target_region=Region.EU_WEST_1,
            trigger="auto",
            initiated_at=datetime.utcnow(),
            initiated_by="auto",
            status=FailoverStatus.IN_PROGRESS,
        )
        event.fail("Connection timeout")
        assert event.status == FailoverStatus.FAILED
        assert event.error_message == "Connection timeout"


# =============================================================================
# MultiRegionService Tests
# =============================================================================


class TestMultiRegionService:
    """Tests for MultiRegionService."""

    def test_create_service(self) -> None:
        """Test creating multi-region service."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[Region.EU_WEST_1],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
        )
        service = MultiRegionService(config)
        assert service.config.primary_region == Region.EU_CENTRAL_1

    def test_regions_initialized(self) -> None:
        """Test that regions are initialized."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[Region.EU_WEST_1, Region.EU_WEST_2],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
        )
        service = MultiRegionService(config)

        primary_config = service.get_region_config(Region.EU_CENTRAL_1)
        assert primary_config is not None
        assert primary_config.is_primary is True

        secondary_config = service.get_region_config(Region.EU_WEST_1)
        assert secondary_config is not None
        assert secondary_config.is_primary is False

    def test_list_regions(self) -> None:
        """Test listing all regions."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[Region.EU_WEST_1, Region.EU_WEST_2],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
        )
        service = MultiRegionService(config)

        regions = service.list_regions()
        assert len(regions) == 3

    def test_list_regions_by_residency(self) -> None:
        """Test listing regions by data residency."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[Region.EU_WEST_1, Region.US_EAST_1],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
        )
        service = MultiRegionService(config)

        eu_regions = service.list_regions(data_residency="EU")
        assert len(eu_regions) == 2

    def test_get_eu_regions(self) -> None:
        """Test getting EU-only regions."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[Region.EU_WEST_1, Region.US_EAST_1],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
        )
        service = MultiRegionService(config)

        eu_regions = service.get_eu_regions()
        assert len(eu_regions) == 2
        for region in eu_regions:
            assert region.data_residency == "EU"

    def test_set_region_status(self) -> None:
        """Test setting region status."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[Region.EU_WEST_1],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
        )
        service = MultiRegionService(config)

        result = service.set_region_status(Region.EU_CENTRAL_1, RegionStatus.MAINTENANCE)
        assert result is True

        health = service.get_health(Region.EU_CENTRAL_1)
        assert health is not None
        assert health.status == RegionStatus.MAINTENANCE

    def test_set_region_status_unknown_region(self) -> None:
        """Test setting status for unknown region."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
        )
        service = MultiRegionService(config)

        result = service.set_region_status(Region.US_EAST_1, RegionStatus.ACTIVE)
        assert result is False

    def test_update_health(self) -> None:
        """Test updating region health."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[Region.EU_WEST_1],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
            auto_failover_enabled=False,  # Disable for this test
        )
        service = MultiRegionService(config)

        health = service.update_health(
            region=Region.EU_CENTRAL_1,
            latency_ms=50.0,
            availability_percent=99.95,
            error_rate_percent=0.1,
            active_connections=5000,
            cpu_utilization_percent=45.0,
            memory_utilization_percent=60.0,
            storage_utilization_percent=30.0,
        )
        assert health.is_healthy is True

    def test_update_health_degraded(self) -> None:
        """Test updating health with degraded metrics."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[Region.EU_WEST_1],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
            auto_failover_enabled=False,
        )
        service = MultiRegionService(config)

        health = service.update_health(
            region=Region.EU_CENTRAL_1,
            latency_ms=50.0,
            availability_percent=97.0,  # Below threshold
            error_rate_percent=2.0,  # Above threshold
            active_connections=5000,
            cpu_utilization_percent=45.0,
            memory_utilization_percent=60.0,
            storage_utilization_percent=30.0,
        )
        assert health.status == RegionStatus.DEGRADED
        assert len(health.issues) > 0

    def test_get_all_health(self) -> None:
        """Test getting all region health statuses."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[Region.EU_WEST_1],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
            auto_failover_enabled=False,
        )
        service = MultiRegionService(config)

        service.update_health(Region.EU_CENTRAL_1, 50.0, 99.9, 0.1, 5000, 45.0, 60.0, 30.0)

        all_health = service.get_all_health()
        assert Region.EU_CENTRAL_1 in all_health

    def test_configure_replication(self) -> None:
        """Test configuring replication."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[Region.EU_WEST_1],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
        )
        service = MultiRegionService(config)

        status = service.configure_replication(
            source_region=Region.EU_CENTRAL_1,
            target_region=Region.EU_WEST_1,
        )
        assert status.source_region == Region.EU_CENTRAL_1
        assert status.target_region == Region.EU_WEST_1
        assert status.mode == ReplicationMode.ASYNCHRONOUS

    def test_configure_replication_custom_mode(self) -> None:
        """Test configuring replication with custom mode."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[Region.EU_WEST_1],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
        )
        service = MultiRegionService(config)

        status = service.configure_replication(
            source_region=Region.EU_CENTRAL_1,
            target_region=Region.EU_WEST_1,
            mode=ReplicationMode.SYNCHRONOUS,
        )
        assert status.mode == ReplicationMode.SYNCHRONOUS

    def test_get_replication_status(self) -> None:
        """Test getting replication status."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[Region.EU_WEST_1],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
        )
        service = MultiRegionService(config)

        service.configure_replication(Region.EU_CENTRAL_1, Region.EU_WEST_1)

        status = service.get_replication_status(Region.EU_CENTRAL_1, Region.EU_WEST_1)
        assert status is not None

    def test_update_replication_status(self) -> None:
        """Test updating replication status."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[Region.EU_WEST_1],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
        )
        service = MultiRegionService(config)

        service.configure_replication(Region.EU_CENTRAL_1, Region.EU_WEST_1)

        status = service.update_replication_status(
            source_region=Region.EU_CENTRAL_1,
            target_region=Region.EU_WEST_1,
            lag_seconds=2.5,
            bytes_pending=50000,
        )
        assert status is not None
        assert status.lag_seconds == 2.5
        assert status.is_healthy is True

    def test_create_failover_plan(self) -> None:
        """Test creating a failover plan."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[Region.EU_WEST_1],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
        )
        service = MultiRegionService(config)

        plan = service.create_failover_plan(
            name="EU Failover",
            source_region=Region.EU_CENTRAL_1,
            target_region=Region.EU_WEST_1,
            auto_failover=True,
        )
        assert plan.name == "EU Failover"
        assert plan.auto_failover is True

    def test_get_failover_plan(self) -> None:
        """Test getting failover plan by ID."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[Region.EU_WEST_1],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
        )
        service = MultiRegionService(config)

        plan = service.create_failover_plan("Test Plan", Region.EU_CENTRAL_1, Region.EU_WEST_1)

        retrieved = service.get_failover_plan(plan.plan_id)
        assert retrieved is not None
        assert retrieved.plan_id == plan.plan_id

    def test_list_failover_plans(self) -> None:
        """Test listing failover plans."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[Region.EU_WEST_1, Region.EU_WEST_2],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
        )
        service = MultiRegionService(config)

        service.create_failover_plan("Plan 1", Region.EU_CENTRAL_1, Region.EU_WEST_1)
        service.create_failover_plan("Plan 2", Region.EU_CENTRAL_1, Region.EU_WEST_2)

        plans = service.list_failover_plans()
        assert len(plans) == 2

    def test_initiate_failover(self) -> None:
        """Test initiating a failover."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[Region.EU_WEST_1],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
        )
        service = MultiRegionService(config)

        plan = service.create_failover_plan("EU Failover", Region.EU_CENTRAL_1, Region.EU_WEST_1)

        event = service.initiate_failover(
            plan_id=plan.plan_id,
            trigger="manual_test",
            initiated_by="admin",
        )
        assert event.status == FailoverStatus.COMPLETED

        # Verify region statuses changed
        primary_config = service.get_region_config(Region.EU_CENTRAL_1)
        target_config = service.get_region_config(Region.EU_WEST_1)
        assert primary_config is not None
        assert target_config is not None
        assert primary_config.is_primary is False
        assert target_config.is_primary is True

    def test_initiate_failover_plan_not_found(self) -> None:
        """Test initiating failover with invalid plan."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[Region.EU_WEST_1],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
        )
        service = MultiRegionService(config)

        with pytest.raises(ValueError, match="Failover plan not found"):
            service.initiate_failover("nonexistent", "test")

    def test_get_failover_events(self) -> None:
        """Test getting failover events."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[Region.EU_WEST_1],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
        )
        service = MultiRegionService(config)

        plan = service.create_failover_plan("EU Failover", Region.EU_CENTRAL_1, Region.EU_WEST_1)
        service.initiate_failover(plan.plan_id, "test")

        events = service.get_failover_events()
        assert len(events) >= 1

    def test_deploy_to_region(self) -> None:
        """Test deploying to a region."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[Region.EU_WEST_1],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
        )
        service = MultiRegionService(config)

        deployment = service.deploy_to_region(
            region=Region.EU_CENTRAL_1,
            version="1.0.0",
            deployed_by="ci_pipeline",
            instances=4,
        )
        assert deployment.version == "1.0.0"
        assert deployment.instances == 4

    def test_deploy_to_region_not_configured(self) -> None:
        """Test deploying to unconfigured region."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
        )
        service = MultiRegionService(config)

        with pytest.raises(ValueError, match="Region not configured"):
            service.deploy_to_region(Region.US_EAST_1, "1.0.0", "admin")

    def test_get_deployment(self) -> None:
        """Test getting deployment by ID."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[Region.EU_WEST_1],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
        )
        service = MultiRegionService(config)

        deployment = service.deploy_to_region(Region.EU_CENTRAL_1, "1.0.0", "admin")

        retrieved = service.get_deployment(deployment.deployment_id)
        assert retrieved is not None

    def test_list_deployments(self) -> None:
        """Test listing deployments."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[Region.EU_WEST_1],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
        )
        service = MultiRegionService(config)

        service.deploy_to_region(Region.EU_CENTRAL_1, "1.0.0", "admin")
        service.deploy_to_region(Region.EU_WEST_1, "1.0.0", "admin")

        all_deployments = service.list_deployments()
        assert len(all_deployments) == 2

    def test_list_deployments_by_region(self) -> None:
        """Test listing deployments by region."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[Region.EU_WEST_1],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
        )
        service = MultiRegionService(config)

        service.deploy_to_region(Region.EU_CENTRAL_1, "1.0.0", "admin")
        service.deploy_to_region(Region.EU_CENTRAL_1, "1.1.0", "admin")
        service.deploy_to_region(Region.EU_WEST_1, "1.0.0", "admin")

        central_deployments = service.list_deployments(region=Region.EU_CENTRAL_1)
        assert len(central_deployments) == 2

    def test_get_status_summary(self) -> None:
        """Test getting status summary."""
        config = MultiRegionConfig(
            primary_region=Region.EU_CENTRAL_1,
            secondary_regions=[Region.EU_WEST_1],
            replication_mode=ReplicationMode.ASYNCHRONOUS,
            auto_failover_enabled=True,
            eu_data_residency=True,
        )
        service = MultiRegionService(config)

        summary = service.get_status_summary()

        assert summary["primary_region"] == Region.EU_CENTRAL_1.value
        assert summary["total_regions"] == 2
        assert summary["auto_failover_enabled"] is True
        assert summary["eu_data_residency"] is True


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_multi_region_service_default(self) -> None:
        """Test creating service with factory function."""
        service = create_multi_region_service()
        assert isinstance(service, MultiRegionService)
        assert service.config.primary_region == Region.EU_CENTRAL_1

    def test_create_multi_region_service_custom(self) -> None:
        """Test creating service with custom options."""
        service = create_multi_region_service(
            primary_region=Region.EU_WEST_1,
            secondary_regions=[Region.EU_WEST_2],
            replication_mode=ReplicationMode.SYNCHRONOUS,
            auto_failover_enabled=False,
        )
        assert service.config.primary_region == Region.EU_WEST_1
        assert service.config.replication_mode == ReplicationMode.SYNCHRONOUS
        assert service.config.auto_failover_enabled is False


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_region_values(self) -> None:
        """Test all region values."""
        assert Region.EU_WEST_1.value == "eu-west-1"
        assert Region.EU_CENTRAL_1.value == "eu-central-1"
        assert Region.US_EAST_1.value == "us-east-1"
        assert Region.AP_SOUTHEAST_1.value == "ap-southeast-1"

    def test_region_status_values(self) -> None:
        """Test all region status values."""
        assert RegionStatus.ACTIVE.value == "active"
        assert RegionStatus.STANDBY.value == "standby"
        assert RegionStatus.DEGRADED.value == "degraded"
        assert RegionStatus.MAINTENANCE.value == "maintenance"
        assert RegionStatus.OFFLINE.value == "offline"

    def test_replication_mode_values(self) -> None:
        """Test all replication mode values."""
        assert ReplicationMode.SYNCHRONOUS.value == "synchronous"
        assert ReplicationMode.ASYNCHRONOUS.value == "asynchronous"
        assert ReplicationMode.SEMI_SYNCHRONOUS.value == "semi_synchronous"

    def test_failover_status_values(self) -> None:
        """Test all failover status values."""
        assert FailoverStatus.IDLE.value == "idle"
        assert FailoverStatus.INITIATED.value == "initiated"
        assert FailoverStatus.IN_PROGRESS.value == "in_progress"
        assert FailoverStatus.COMPLETED.value == "completed"
        assert FailoverStatus.FAILED.value == "failed"
