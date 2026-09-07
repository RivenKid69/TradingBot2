# -*- coding: utf-8 -*-
"""
Tests for ResourceQuotaManager.

CCEA Phase 10: Per-tenant resource quotas.
"""

import pytest
from datetime import datetime, timedelta

from packages.cloud.research.sandbox.resource_quota import (
    ResourceQuotaManager,
    TenantQuota,
    QuotaUsage,
    QuotaCheckResult,
    QuotaExceededError,
    QuotaTier,
    QuotaResource,
    create_quota_manager,
    DEFAULT_CPU_HOURS_DAILY,
    DEFAULT_MEMORY_GB_HOURS_DAILY,
    DEFAULT_MAX_CONCURRENT_JOBS,
    PREMIUM_CPU_HOURS_DAILY,
    ENTERPRISE_CPU_HOURS_DAILY,
)


class TestTenantQuota:
    """Tests for TenantQuota."""

    def test_default_quota(self):
        """Test default quota values."""
        quota = TenantQuota(tenant_id="tenant-123")

        assert quota.tenant_id == "tenant-123"
        assert quota.tier == QuotaTier.FREE
        assert quota.cpu_hours_daily == DEFAULT_CPU_HOURS_DAILY
        assert quota.memory_gb_hours_daily == DEFAULT_MEMORY_GB_HOURS_DAILY
        assert quota.max_concurrent_jobs == DEFAULT_MAX_CONCURRENT_JOBS
        assert quota.is_active is True

    def test_premium_quota(self):
        """Test premium quota values."""
        quota = TenantQuota(
            tenant_id="tenant-123",
            tier=QuotaTier.PREMIUM,
            cpu_hours_daily=PREMIUM_CPU_HOURS_DAILY,
        )

        assert quota.tier == QuotaTier.PREMIUM
        assert quota.cpu_hours_daily == PREMIUM_CPU_HOURS_DAILY

    def test_quota_to_dict(self):
        """Test quota serialization."""
        quota = TenantQuota(tenant_id="tenant-123")
        data = quota.to_dict()

        assert data["tenant_id"] == "tenant-123"
        assert "cpu_hours_daily" in data
        assert "tier" in data


class TestQuotaUsage:
    """Tests for QuotaUsage."""

    def test_default_usage(self):
        """Test default usage values."""
        usage = QuotaUsage(tenant_id="tenant-123")

        assert usage.tenant_id == "tenant-123"
        assert usage.cpu_hours_used == 0.0
        assert usage.memory_gb_hours_used == 0.0
        assert usage.concurrent_jobs == 0
        assert usage.jobs_today == 0

    def test_usage_to_dict(self):
        """Test usage serialization."""
        usage = QuotaUsage(
            tenant_id="tenant-123",
            cpu_hours_used=1.5,
        )
        data = usage.to_dict()

        assert data["tenant_id"] == "tenant-123"
        assert data["cpu_hours_used"] == 1.5


class TestQuotaCheckResult:
    """Tests for QuotaCheckResult."""

    def test_allowed_result(self):
        """Test allowed result."""
        result = QuotaCheckResult(
            allowed=True,
            message="Quota check passed",
        )

        assert result.allowed is True

    def test_denied_result(self):
        """Test denied result."""
        result = QuotaCheckResult(
            allowed=False,
            resource=QuotaResource.CPU_HOURS,
            requested=5.0,
            available=2.0,
            message="CPU quota exceeded",
        )

        assert result.allowed is False
        assert result.resource == QuotaResource.CPU_HOURS
        assert result.requested == 5.0
        assert result.available == 2.0


class TestResourceQuotaManager:
    """Tests for ResourceQuotaManager."""

    def test_manager_creation(self):
        """Test manager creation."""
        manager = ResourceQuotaManager()

        stats = manager.get_stats()
        assert stats["quota_checks"] == 0

    def test_set_quota_free_tier(self):
        """Test setting free tier quota."""
        manager = ResourceQuotaManager()

        quota = manager.set_quota("tenant-123", QuotaTier.FREE)

        assert quota.tenant_id == "tenant-123"
        assert quota.tier == QuotaTier.FREE
        assert quota.cpu_hours_daily == DEFAULT_CPU_HOURS_DAILY

    def test_set_quota_premium_tier(self):
        """Test setting premium tier quota."""
        manager = ResourceQuotaManager()

        quota = manager.set_quota("tenant-123", QuotaTier.PREMIUM)

        assert quota.tier == QuotaTier.PREMIUM
        assert quota.cpu_hours_daily == PREMIUM_CPU_HOURS_DAILY

    def test_set_quota_enterprise_tier(self):
        """Test setting enterprise tier quota."""
        manager = ResourceQuotaManager()

        quota = manager.set_quota("tenant-123", QuotaTier.ENTERPRISE)

        assert quota.tier == QuotaTier.ENTERPRISE
        assert quota.cpu_hours_daily == ENTERPRISE_CPU_HOURS_DAILY

    def test_set_custom_quota(self):
        """Test setting custom quota."""
        manager = ResourceQuotaManager()

        custom_quota = TenantQuota(
            cpu_hours_daily=100.0,
            max_concurrent_jobs=50,
        )

        quota = manager.set_quota("tenant-123", custom_quota=custom_quota)

        assert quota.tier == QuotaTier.CUSTOM
        assert quota.cpu_hours_daily == 100.0
        assert quota.max_concurrent_jobs == 50

    def test_get_quota(self):
        """Test getting quota."""
        manager = ResourceQuotaManager()
        manager.set_quota("tenant-123", QuotaTier.PREMIUM)

        quota = manager.get_quota("tenant-123")

        assert quota is not None
        assert quota.tier == QuotaTier.PREMIUM

    def test_get_quota_nonexistent(self):
        """Test getting non-existent quota."""
        manager = ResourceQuotaManager()

        quota = manager.get_quota("nonexistent")

        assert quota is None

    def test_get_usage(self):
        """Test getting usage."""
        manager = ResourceQuotaManager()
        manager.set_quota("tenant-123", QuotaTier.FREE)

        usage = manager.get_usage("tenant-123")

        assert usage is not None
        assert usage.tenant_id == "tenant-123"
        assert usage.cpu_hours_used == 0.0

    def test_check_quota_allowed(self):
        """Test quota check - allowed."""
        manager = ResourceQuotaManager()
        manager.set_quota("tenant-123", QuotaTier.FREE)

        result = manager.check_quota(
            tenant_id="tenant-123",
            cpu=1.0,
            memory_mb=1024,
            duration_seconds=1800,  # 30 minutes
        )

        assert result.allowed is True

    def test_check_quota_denied_cpu(self):
        """Test quota check - denied by CPU or duration."""
        manager = ResourceQuotaManager()
        manager.set_quota("tenant-123", QuotaTier.FREE)  # 2 CPU hours/day, 1 hour max duration

        result = manager.check_quota(
            tenant_id="tenant-123",
            cpu=1.0,
            memory_mb=1024,
            duration_seconds=10800,  # 3 hours - exceeds both 2 hour quota AND 1 hour max duration
        )

        assert result.allowed is False
        # Duration check may fire first since FREE tier has 1 hour max duration
        assert result.resource in (QuotaResource.CPU_HOURS, QuotaResource.JOB_DURATION)

    def test_check_quota_denied_concurrent(self):
        """Test quota check - denied by concurrent limit."""
        manager = ResourceQuotaManager()
        manager.set_quota("tenant-123", QuotaTier.FREE)  # 2 concurrent jobs

        # Start 2 jobs
        manager.start_job("tenant-123", "job-1", cpu=1.0, memory_mb=1024)
        manager.start_job("tenant-123", "job-2", cpu=1.0, memory_mb=1024)

        # Third job should be denied
        result = manager.check_quota(
            tenant_id="tenant-123",
            cpu=1.0,
            memory_mb=1024,
            duration_seconds=3600,
        )

        assert result.allowed is False
        assert result.resource == QuotaResource.CONCURRENT_JOBS

    def test_check_quota_denied_per_job_cpu(self):
        """Test quota check - denied by per-job CPU limit."""
        manager = ResourceQuotaManager()
        manager.set_quota("tenant-123", QuotaTier.FREE)  # max 4 CPU per job

        result = manager.check_quota(
            tenant_id="tenant-123",
            cpu=8.0,  # Exceeds per-job limit
            memory_mb=1024,
            duration_seconds=3600,
        )

        assert result.allowed is False
        assert "per job" in result.message.lower()

    def test_check_quota_denied_per_job_memory(self):
        """Test quota check - denied by per-job memory limit."""
        manager = ResourceQuotaManager()
        manager.set_quota("tenant-123", QuotaTier.FREE)  # max 8GB per job

        result = manager.check_quota(
            tenant_id="tenant-123",
            cpu=1.0,
            memory_mb=16384,  # 16GB - exceeds limit
            duration_seconds=3600,
        )

        assert result.allowed is False
        assert "memory" in result.message.lower()

    def test_check_quota_denied_duration(self):
        """Test quota check - denied by duration limit."""
        manager = ResourceQuotaManager()
        manager.set_quota("tenant-123", QuotaTier.FREE)  # 1 hour max

        result = manager.check_quota(
            tenant_id="tenant-123",
            cpu=1.0,
            memory_mb=1024,
            duration_seconds=7200,  # 2 hours
        )

        assert result.allowed is False
        assert result.resource == QuotaResource.JOB_DURATION

    def test_start_job(self):
        """Test starting a job."""
        manager = ResourceQuotaManager()
        manager.set_quota("tenant-123", QuotaTier.FREE)

        result = manager.start_job("tenant-123", "job-456", cpu=1.0, memory_mb=1024)

        assert result is True

        usage = manager.get_usage("tenant-123")
        assert usage.concurrent_jobs == 1
        assert "job-456" in usage.active_job_ids
        assert usage.jobs_today == 1

    def test_update_usage(self):
        """Test updating usage."""
        manager = ResourceQuotaManager()
        manager.set_quota("tenant-123", QuotaTier.FREE)
        manager.start_job("tenant-123", "job-456", cpu=1.0, memory_mb=1024)

        result = manager.update_usage(
            tenant_id="tenant-123",
            job_id="job-456",
            cpu_seconds=3600,  # 1 hour
            memory_mb_seconds=1024 * 3600,
        )

        assert result is True

        usage = manager.get_usage("tenant-123")
        assert usage.cpu_hours_used == 1.0

    def test_end_job(self):
        """Test ending a job."""
        manager = ResourceQuotaManager()
        manager.set_quota("tenant-123", QuotaTier.FREE)
        manager.start_job("tenant-123", "job-456", cpu=1.0, memory_mb=1024)

        result = manager.end_job("tenant-123", "job-456")

        assert result is True

        usage = manager.get_usage("tenant-123")
        assert usage.concurrent_jobs == 0
        assert "job-456" not in usage.active_job_ids

    def test_suspend_tenant(self):
        """Test suspending tenant."""
        manager = ResourceQuotaManager()
        manager.set_quota("tenant-123", QuotaTier.FREE)

        result = manager.suspend_tenant("tenant-123", "Abuse detected")

        assert result is True

        quota = manager.get_quota("tenant-123")
        assert quota.is_active is False
        assert quota.suspension_reason == "Abuse detected"

    def test_suspended_tenant_blocked(self):
        """Test suspended tenant is blocked."""
        manager = ResourceQuotaManager()
        manager.set_quota("tenant-123", QuotaTier.FREE)
        manager.suspend_tenant("tenant-123", "Abuse detected")

        result = manager.check_quota(
            tenant_id="tenant-123",
            cpu=1.0,
            memory_mb=1024,
            duration_seconds=3600,
        )

        assert result.allowed is False
        assert "suspended" in result.message.lower()

    def test_unsuspend_tenant(self):
        """Test unsuspending tenant."""
        manager = ResourceQuotaManager()
        manager.set_quota("tenant-123", QuotaTier.FREE)
        manager.suspend_tenant("tenant-123", "Abuse detected")

        result = manager.unsuspend_tenant("tenant-123")

        assert result is True

        quota = manager.get_quota("tenant-123")
        assert quota.is_active is True

    def test_get_usage_percentage(self):
        """Test getting usage percentage."""
        manager = ResourceQuotaManager()
        manager.set_quota("tenant-123", QuotaTier.FREE)  # 2 CPU hours
        manager.start_job("tenant-123", "job-456", cpu=1.0, memory_mb=1024)
        manager.update_usage(
            tenant_id="tenant-123",
            job_id="job-456",
            cpu_seconds=3600,  # 1 hour = 50% of 2 hours
        )

        percentages = manager.get_usage_percentage("tenant-123")

        assert percentages["cpu_hours"] == 50.0

    def test_get_remaining(self):
        """Test getting remaining quota."""
        manager = ResourceQuotaManager()
        manager.set_quota("tenant-123", QuotaTier.FREE)  # 2 CPU hours
        manager.start_job("tenant-123", "job-456", cpu=1.0, memory_mb=1024)
        manager.update_usage(
            tenant_id="tenant-123",
            job_id="job-456",
            cpu_seconds=3600,  # 1 hour used
        )

        remaining = manager.get_remaining("tenant-123")

        assert remaining["cpu_hours"] == 1.0  # 2 - 1 = 1

    def test_quota_warning_callback(self):
        """Test quota warning callback."""
        warnings = []

        def on_warning(tenant_id, resource, pct):
            warnings.append((tenant_id, resource, pct))

        manager = ResourceQuotaManager(on_quota_warning=on_warning)
        manager.set_quota("tenant-123", QuotaTier.FREE)  # 2 CPU hours
        manager.start_job("tenant-123", "job-456", cpu=1.0, memory_mb=1024)

        # Use 85% of quota
        manager.update_usage(
            tenant_id="tenant-123",
            job_id="job-456",
            cpu_seconds=6120,  # 1.7 hours = 85%
        )

        assert len(warnings) == 1
        assert warnings[0][0] == "tenant-123"
        assert warnings[0][1] == QuotaResource.CPU_HOURS

    def test_cleanup_stale_jobs(self):
        """Test cleaning up stale jobs."""
        manager = ResourceQuotaManager()
        manager.set_quota("tenant-123", QuotaTier.FREE)
        manager.start_job("tenant-123", "job-456", cpu=1.0, memory_mb=1024)

        # Manually make job stale
        manager._active_jobs["job-456"]["start_time"] = datetime.utcnow() - timedelta(hours=48)

        cleaned = manager.cleanup_stale_jobs(max_age_hours=24.0)

        assert cleaned == 1

        usage = manager.get_usage("tenant-123")
        assert usage.concurrent_jobs == 0

    def test_get_stats(self):
        """Test getting manager stats."""
        manager = ResourceQuotaManager()
        manager.set_quota("tenant-123", QuotaTier.FREE)
        manager.check_quota("tenant-123", cpu=1.0, memory_mb=1024, duration_seconds=3600)

        stats = manager.get_stats()

        assert stats["quota_checks"] == 1
        assert stats["tenants_with_quota"] == 1


class TestQuotaTier:
    """Tests for QuotaTier enum."""

    def test_tier_values(self):
        """Test all tiers exist."""
        assert QuotaTier.FREE
        assert QuotaTier.PREMIUM
        assert QuotaTier.ENTERPRISE
        assert QuotaTier.CUSTOM


class TestQuotaResource:
    """Tests for QuotaResource enum."""

    def test_resource_values(self):
        """Test all resources exist."""
        assert QuotaResource.CPU_HOURS
        assert QuotaResource.MEMORY_GB_HOURS
        assert QuotaResource.CONCURRENT_JOBS
        assert QuotaResource.JOB_DURATION
        assert QuotaResource.STORAGE_MB
        assert QuotaResource.NETWORK_BYTES


class TestCreateQuotaManager:
    """Tests for create_quota_manager factory."""

    def test_create_with_defaults(self):
        """Test creating manager with defaults."""
        manager = create_quota_manager()

        assert manager is not None

    def test_create_with_warning_callback(self):
        """Test creating manager with warning callback."""

        def on_warning(tenant_id, resource, pct):
            pass

        manager = create_quota_manager(on_warning=on_warning)

        assert manager._on_quota_warning is not None
