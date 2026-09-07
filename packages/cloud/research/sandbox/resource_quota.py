# -*- coding: utf-8 -*-
"""
Resource Quota Manager - Per-tenant resource quotas.

CCEA Phase 10: Quotas CPU/RAM/time.

Design Doc 15.3/5.3:
- Quotas CPU/RAM/time
- Tenant isolation at execution level
- Prevent resource abuse

This module manages resource quotas:
- Per-tenant CPU, memory, time limits
- Concurrent job limits
- Daily/monthly execution quotas
- Storage quotas
- Network transfer quotas

Quota Enforcement:
- Pre-job validation (reject if quota exceeded)
- Real-time usage tracking
- Graceful degradation on quota exhaustion
- Automatic quota refresh (daily/monthly)

Reference:
- Kubernetes Resource Quotas: https://kubernetes.io/docs/concepts/policy/resource-quotas/
- AWS Service Quotas: https://docs.aws.amazon.com/servicequotas/
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, Final, List, Optional, Set
from uuid import uuid4


# ============================================================================
# Constants
# ============================================================================

# Default quotas (free tier)
DEFAULT_CPU_HOURS_DAILY: Final[float] = 2.0  # 2 CPU-hours/day
DEFAULT_MEMORY_GB_HOURS_DAILY: Final[float] = 4.0  # 4 GB-hours/day
DEFAULT_MAX_CONCURRENT_JOBS: Final[int] = 2
DEFAULT_MAX_JOB_DURATION_SECONDS: Final[int] = 3600  # 1 hour
DEFAULT_STORAGE_MB: Final[int] = 1024  # 1 GB
DEFAULT_NETWORK_GB_DAILY: Final[float] = 1.0  # 1 GB/day

# Premium quotas
PREMIUM_CPU_HOURS_DAILY: Final[float] = 24.0
PREMIUM_MEMORY_GB_HOURS_DAILY: Final[float] = 48.0
PREMIUM_MAX_CONCURRENT_JOBS: Final[int] = 10
PREMIUM_MAX_JOB_DURATION_SECONDS: Final[int] = 86400  # 24 hours
PREMIUM_STORAGE_MB: Final[int] = 10240  # 10 GB
PREMIUM_NETWORK_GB_DAILY: Final[float] = 10.0

# Enterprise quotas (high limits)
ENTERPRISE_CPU_HOURS_DAILY: Final[float] = 1000.0
ENTERPRISE_MEMORY_GB_HOURS_DAILY: Final[float] = 2000.0
ENTERPRISE_MAX_CONCURRENT_JOBS: Final[int] = 100
ENTERPRISE_MAX_JOB_DURATION_SECONDS: Final[int] = 604800  # 7 days
ENTERPRISE_STORAGE_MB: Final[int] = 102400  # 100 GB
ENTERPRISE_NETWORK_GB_DAILY: Final[float] = 100.0


class QuotaTier(Enum):
    """Quota tier levels."""

    FREE = auto()
    PREMIUM = auto()
    ENTERPRISE = auto()
    CUSTOM = auto()


class QuotaResource(Enum):
    """Types of quota resources."""

    CPU_HOURS = auto()
    MEMORY_GB_HOURS = auto()
    CONCURRENT_JOBS = auto()
    JOB_DURATION = auto()
    STORAGE_MB = auto()
    NETWORK_BYTES = auto()
    JOBS_PER_DAY = auto()
    JOBS_PER_MONTH = auto()


class QuotaExceededError(Exception):
    """Exception raised when quota is exceeded."""

    def __init__(
        self,
        resource: QuotaResource,
        requested: float,
        available: float,
        message: str = "",
    ):
        self.resource = resource
        self.requested = requested
        self.available = available
        self.message = (
            message
            or f"Quota exceeded for {resource.name}: requested {requested}, available {available}"
        )
        super().__init__(self.message)


@dataclass
class TenantQuota:
    """
    Quota definition for a tenant.
    """

    tenant_id: str = ""
    tier: QuotaTier = QuotaTier.FREE

    # Compute quotas (daily)
    cpu_hours_daily: float = DEFAULT_CPU_HOURS_DAILY
    memory_gb_hours_daily: float = DEFAULT_MEMORY_GB_HOURS_DAILY

    # Concurrent limits
    max_concurrent_jobs: int = DEFAULT_MAX_CONCURRENT_JOBS
    max_job_duration_seconds: int = DEFAULT_MAX_JOB_DURATION_SECONDS

    # Storage quotas
    storage_mb: int = DEFAULT_STORAGE_MB

    # Network quotas (daily)
    network_gb_daily: float = DEFAULT_NETWORK_GB_DAILY

    # Job count quotas
    jobs_per_day: int = 100
    jobs_per_month: int = 2000

    # Custom per-job limits
    max_cpu_per_job: float = 4.0  # cores
    max_memory_per_job_mb: int = 8192  # 8 GB
    max_disk_per_job_mb: int = 4096  # 4 GB

    # Status
    is_active: bool = True
    suspended_at: Optional[datetime] = None
    suspension_reason: str = ""

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "tier": self.tier.name,
            "cpu_hours_daily": self.cpu_hours_daily,
            "memory_gb_hours_daily": self.memory_gb_hours_daily,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "max_job_duration_seconds": self.max_job_duration_seconds,
            "storage_mb": self.storage_mb,
            "network_gb_daily": self.network_gb_daily,
            "jobs_per_day": self.jobs_per_day,
            "jobs_per_month": self.jobs_per_month,
            "is_active": self.is_active,
        }


@dataclass
class QuotaUsage:
    """
    Current quota usage for a tenant.
    """

    tenant_id: str = ""

    # Period tracking
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_type: str = "daily"  # daily, monthly

    # Compute usage
    cpu_hours_used: float = 0.0
    memory_gb_hours_used: float = 0.0

    # Current state
    concurrent_jobs: int = 0
    active_job_ids: Set[str] = field(default_factory=set)

    # Storage usage
    storage_mb_used: int = 0

    # Network usage
    network_bytes_used: int = 0

    # Job counts
    jobs_today: int = 0
    jobs_this_month: int = 0

    # Timestamps
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "period_start": self.period_start.isoformat(),
            "cpu_hours_used": self.cpu_hours_used,
            "memory_gb_hours_used": self.memory_gb_hours_used,
            "concurrent_jobs": self.concurrent_jobs,
            "storage_mb_used": self.storage_mb_used,
            "network_bytes_used": self.network_bytes_used,
            "jobs_today": self.jobs_today,
            "jobs_this_month": self.jobs_this_month,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class QuotaCheckResult:
    """Result of a quota check."""

    allowed: bool = False
    resource: Optional[QuotaResource] = None
    requested: float = 0.0
    available: float = 0.0
    used: float = 0.0
    limit: float = 0.0
    message: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "resource": self.resource.name if self.resource else None,
            "requested": self.requested,
            "available": self.available,
            "used": self.used,
            "limit": self.limit,
            "message": self.message,
            "warnings": self.warnings,
        }


class ResourceQuotaManager:
    """
    Manage resource quotas for tenants.

    Tracks and enforces:
    - CPU hours (daily/monthly)
    - Memory usage (GB-hours)
    - Concurrent jobs
    - Job duration limits
    - Storage quotas
    - Network transfer limits

    Usage:
        manager = ResourceQuotaManager()

        # Set quota for tenant
        manager.set_quota(tenant_id, QuotaTier.PREMIUM)

        # Check before starting job
        result = manager.check_quota(tenant_id, cpu=2.0, memory_mb=4096, duration=3600)
        if not result.allowed:
            raise QuotaExceededError(result.resource, result.requested, result.available)

        # Start job and track
        manager.start_job(tenant_id, job_id, cpu=2.0, memory_mb=4096)

        # Update usage during job
        manager.update_usage(tenant_id, job_id, cpu_seconds=60, memory_mb_seconds=4096*60)

        # End job
        manager.end_job(tenant_id, job_id)

        # Get usage report
        usage = manager.get_usage(tenant_id)
    """

    def __init__(
        self,
        on_quota_warning: Optional[Callable[[str, QuotaResource, float], None]] = None,
        on_quota_exceeded: Optional[Callable[[str, QuotaResource, float], None]] = None,
    ):
        """
        Initialize quota manager.

        Args:
            on_quota_warning: Callback when usage reaches warning level (80%)
            on_quota_exceeded: Callback when quota is exceeded
        """
        self._on_quota_warning = on_quota_warning
        self._on_quota_exceeded = on_quota_exceeded

        # Storage
        self._quotas: Dict[str, TenantQuota] = {}
        self._usage: Dict[str, QuotaUsage] = {}
        self._lock = threading.RLock()

        # Warning threshold
        self._warning_threshold = 0.8  # 80%

        # Job tracking (job_id -> {tenant_id, start_time, cpu, memory})
        self._active_jobs: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self._stats = {
            "quota_checks": 0,
            "quota_exceeded": 0,
            "jobs_started": 0,
            "jobs_completed": 0,
        }

    def set_quota(
        self,
        tenant_id: str,
        tier: QuotaTier = QuotaTier.FREE,
        custom_quota: Optional[TenantQuota] = None,
    ) -> TenantQuota:
        """
        Set quota for a tenant.

        Args:
            tenant_id: Tenant identifier
            tier: Quota tier (FREE, PREMIUM, ENTERPRISE)
            custom_quota: Custom quota definition

        Returns:
            Created/updated TenantQuota
        """
        with self._lock:
            if custom_quota:
                quota = custom_quota
                quota.tenant_id = tenant_id
                quota.tier = QuotaTier.CUSTOM
            else:
                quota = self._create_tier_quota(tenant_id, tier)

            self._quotas[tenant_id] = quota

            # Initialize usage if not exists
            if tenant_id not in self._usage:
                self._usage[tenant_id] = QuotaUsage(tenant_id=tenant_id)

            return quota

    def _create_tier_quota(self, tenant_id: str, tier: QuotaTier) -> TenantQuota:
        """Create quota based on tier."""
        if tier == QuotaTier.FREE:
            return TenantQuota(
                tenant_id=tenant_id,
                tier=tier,
                cpu_hours_daily=DEFAULT_CPU_HOURS_DAILY,
                memory_gb_hours_daily=DEFAULT_MEMORY_GB_HOURS_DAILY,
                max_concurrent_jobs=DEFAULT_MAX_CONCURRENT_JOBS,
                max_job_duration_seconds=DEFAULT_MAX_JOB_DURATION_SECONDS,
                storage_mb=DEFAULT_STORAGE_MB,
                network_gb_daily=DEFAULT_NETWORK_GB_DAILY,
            )
        elif tier == QuotaTier.PREMIUM:
            return TenantQuota(
                tenant_id=tenant_id,
                tier=tier,
                cpu_hours_daily=PREMIUM_CPU_HOURS_DAILY,
                memory_gb_hours_daily=PREMIUM_MEMORY_GB_HOURS_DAILY,
                max_concurrent_jobs=PREMIUM_MAX_CONCURRENT_JOBS,
                max_job_duration_seconds=PREMIUM_MAX_JOB_DURATION_SECONDS,
                storage_mb=PREMIUM_STORAGE_MB,
                network_gb_daily=PREMIUM_NETWORK_GB_DAILY,
                max_cpu_per_job=8.0,
                max_memory_per_job_mb=16384,
            )
        elif tier == QuotaTier.ENTERPRISE:
            return TenantQuota(
                tenant_id=tenant_id,
                tier=tier,
                cpu_hours_daily=ENTERPRISE_CPU_HOURS_DAILY,
                memory_gb_hours_daily=ENTERPRISE_MEMORY_GB_HOURS_DAILY,
                max_concurrent_jobs=ENTERPRISE_MAX_CONCURRENT_JOBS,
                max_job_duration_seconds=ENTERPRISE_MAX_JOB_DURATION_SECONDS,
                storage_mb=ENTERPRISE_STORAGE_MB,
                network_gb_daily=ENTERPRISE_NETWORK_GB_DAILY,
                max_cpu_per_job=32.0,
                max_memory_per_job_mb=65536,
                jobs_per_day=10000,
                jobs_per_month=100000,
            )
        else:
            return TenantQuota(tenant_id=tenant_id, tier=tier)

    def get_quota(self, tenant_id: str) -> Optional[TenantQuota]:
        """Get quota for a tenant."""
        with self._lock:
            return self._quotas.get(tenant_id)

    def get_usage(self, tenant_id: str) -> Optional[QuotaUsage]:
        """Get current usage for a tenant."""
        with self._lock:
            usage = self._usage.get(tenant_id)
            if usage:
                # Reset if new period
                usage = self._check_period_reset(tenant_id, usage)
            return usage

    def _check_period_reset(self, tenant_id: str, usage: QuotaUsage) -> QuotaUsage:
        """Check and reset usage if new period."""
        now = datetime.utcnow()
        period_start = usage.period_start

        # Daily reset
        if now.date() > period_start.date():
            usage.period_start = datetime(now.year, now.month, now.day)
            usage.cpu_hours_used = 0.0
            usage.memory_gb_hours_used = 0.0
            usage.network_bytes_used = 0
            usage.jobs_today = 0

            # Monthly reset
            if now.month != period_start.month:
                usage.jobs_this_month = 0

            usage.last_updated = now

        return usage

    def check_quota(
        self,
        tenant_id: str,
        cpu: float = 1.0,
        memory_mb: int = 1024,
        duration_seconds: int = 3600,
        storage_mb: int = 0,
        network_bytes: int = 0,
    ) -> QuotaCheckResult:
        """
        Check if job can be started within quota.

        Args:
            tenant_id: Tenant identifier
            cpu: CPU cores requested
            memory_mb: Memory in MB
            duration_seconds: Expected duration
            storage_mb: Storage needed
            network_bytes: Network transfer expected

        Returns:
            QuotaCheckResult indicating if allowed
        """
        with self._lock:
            self._stats["quota_checks"] += 1

            result = QuotaCheckResult()
            warnings = []

            # Get quota and usage
            quota = self._quotas.get(tenant_id)
            if not quota:
                result.message = "No quota defined for tenant"
                return result

            if not quota.is_active:
                result.message = f"Tenant quota suspended: {quota.suspension_reason}"
                return result

            usage = self._usage.get(tenant_id)
            if not usage:
                usage = QuotaUsage(tenant_id=tenant_id)
                self._usage[tenant_id] = usage
            else:
                usage = self._check_period_reset(tenant_id, usage)

            # Check per-job limits
            if cpu > quota.max_cpu_per_job:
                result.resource = QuotaResource.CPU_HOURS
                result.requested = cpu
                result.available = quota.max_cpu_per_job
                result.message = f"CPU per job exceeds limit: {cpu} > {quota.max_cpu_per_job}"
                return result

            if memory_mb > quota.max_memory_per_job_mb:
                result.resource = QuotaResource.MEMORY_GB_HOURS
                result.requested = memory_mb
                result.available = quota.max_memory_per_job_mb
                result.message = (
                    f"Memory per job exceeds limit: {memory_mb}MB > {quota.max_memory_per_job_mb}MB"
                )
                return result

            if duration_seconds > quota.max_job_duration_seconds:
                result.resource = QuotaResource.JOB_DURATION
                result.requested = duration_seconds
                result.available = quota.max_job_duration_seconds
                result.message = f"Duration exceeds limit: {duration_seconds}s > {quota.max_job_duration_seconds}s"
                return result

            # Check concurrent jobs
            if usage.concurrent_jobs >= quota.max_concurrent_jobs:
                result.resource = QuotaResource.CONCURRENT_JOBS
                result.requested = 1
                result.available = 0
                result.used = usage.concurrent_jobs
                result.limit = quota.max_concurrent_jobs
                result.message = f"Concurrent job limit reached: {usage.concurrent_jobs}/{quota.max_concurrent_jobs}"
                return result

            # Check CPU hours
            estimated_cpu_hours = (cpu * duration_seconds) / 3600
            cpu_available = quota.cpu_hours_daily - usage.cpu_hours_used
            if estimated_cpu_hours > cpu_available:
                result.resource = QuotaResource.CPU_HOURS
                result.requested = estimated_cpu_hours
                result.available = cpu_available
                result.used = usage.cpu_hours_used
                result.limit = quota.cpu_hours_daily
                result.message = f"CPU hours quota exceeded: need {estimated_cpu_hours:.2f}, available {cpu_available:.2f}"
                self._stats["quota_exceeded"] += 1
                return result

            # Warn if close to limit
            if usage.cpu_hours_used / quota.cpu_hours_daily > self._warning_threshold:
                warnings.append(
                    f"CPU hours at {usage.cpu_hours_used/quota.cpu_hours_daily*100:.0f}% of daily limit"
                )

            # Check memory hours
            estimated_memory_hours = (memory_mb / 1024 * duration_seconds) / 3600
            memory_available = quota.memory_gb_hours_daily - usage.memory_gb_hours_used
            if estimated_memory_hours > memory_available:
                result.resource = QuotaResource.MEMORY_GB_HOURS
                result.requested = estimated_memory_hours
                result.available = memory_available
                result.used = usage.memory_gb_hours_used
                result.limit = quota.memory_gb_hours_daily
                result.message = f"Memory hours quota exceeded: need {estimated_memory_hours:.2f}, available {memory_available:.2f}"
                self._stats["quota_exceeded"] += 1
                return result

            # Check storage
            if storage_mb > 0 and usage.storage_mb_used + storage_mb > quota.storage_mb:
                result.resource = QuotaResource.STORAGE_MB
                result.requested = storage_mb
                result.available = quota.storage_mb - usage.storage_mb_used
                result.used = usage.storage_mb_used
                result.limit = quota.storage_mb
                result.message = f"Storage quota exceeded"
                self._stats["quota_exceeded"] += 1
                return result

            # Check network
            if network_bytes > 0:
                network_available = (
                    quota.network_gb_daily * 1024 * 1024 * 1024
                ) - usage.network_bytes_used
                if network_bytes > network_available:
                    result.resource = QuotaResource.NETWORK_BYTES
                    result.requested = network_bytes
                    result.available = network_available
                    result.message = f"Network quota exceeded"
                    self._stats["quota_exceeded"] += 1
                    return result

            # Check job count
            if usage.jobs_today >= quota.jobs_per_day:
                result.resource = QuotaResource.JOBS_PER_DAY
                result.requested = 1
                result.available = 0
                result.used = usage.jobs_today
                result.limit = quota.jobs_per_day
                result.message = f"Daily job limit reached: {usage.jobs_today}/{quota.jobs_per_day}"
                self._stats["quota_exceeded"] += 1
                return result

            if usage.jobs_this_month >= quota.jobs_per_month:
                result.resource = QuotaResource.JOBS_PER_MONTH
                result.requested = 1
                result.available = 0
                result.used = usage.jobs_this_month
                result.limit = quota.jobs_per_month
                result.message = (
                    f"Monthly job limit reached: {usage.jobs_this_month}/{quota.jobs_per_month}"
                )
                self._stats["quota_exceeded"] += 1
                return result

            # All checks passed
            result.allowed = True
            result.warnings = warnings
            result.message = "Quota check passed"

            return result

    def start_job(
        self,
        tenant_id: str,
        job_id: str,
        cpu: float = 1.0,
        memory_mb: int = 1024,
    ) -> bool:
        """
        Record job start and reserve quota.

        Args:
            tenant_id: Tenant identifier
            job_id: Job identifier
            cpu: CPU cores allocated
            memory_mb: Memory allocated

        Returns:
            True if job started successfully
        """
        with self._lock:
            usage = self._usage.get(tenant_id)
            if not usage:
                return False

            # Track active job
            self._active_jobs[job_id] = {
                "tenant_id": tenant_id,
                "start_time": datetime.utcnow(),
                "cpu": cpu,
                "memory_mb": memory_mb,
                "cpu_seconds_used": 0.0,
                "memory_mb_seconds_used": 0.0,
            }

            # Update usage
            usage.concurrent_jobs += 1
            usage.active_job_ids.add(job_id)
            usage.jobs_today += 1
            usage.jobs_this_month += 1
            usage.last_updated = datetime.utcnow()

            self._stats["jobs_started"] += 1

            return True

    def update_usage(
        self,
        tenant_id: str,
        job_id: str,
        cpu_seconds: float = 0.0,
        memory_mb_seconds: float = 0.0,
        network_bytes: int = 0,
        storage_mb_delta: int = 0,
    ) -> bool:
        """
        Update resource usage during job execution.

        Args:
            tenant_id: Tenant identifier
            job_id: Job identifier
            cpu_seconds: CPU seconds consumed since last update
            memory_mb_seconds: Memory-MB-seconds consumed
            network_bytes: Network bytes transferred
            storage_mb_delta: Change in storage usage

        Returns:
            True if update successful
        """
        with self._lock:
            usage = self._usage.get(tenant_id)
            job = self._active_jobs.get(job_id)

            if not usage or not job:
                return False

            # Update job tracking
            job["cpu_seconds_used"] += cpu_seconds
            job["memory_mb_seconds_used"] += memory_mb_seconds

            # Update usage
            usage.cpu_hours_used += cpu_seconds / 3600
            usage.memory_gb_hours_used += (memory_mb_seconds / 1024) / 3600
            usage.network_bytes_used += network_bytes
            usage.storage_mb_used += storage_mb_delta
            usage.last_updated = datetime.utcnow()

            # Check for warnings
            quota = self._quotas.get(tenant_id)
            if quota and self._on_quota_warning:
                if usage.cpu_hours_used / quota.cpu_hours_daily > self._warning_threshold:
                    self._on_quota_warning(
                        tenant_id,
                        QuotaResource.CPU_HOURS,
                        usage.cpu_hours_used / quota.cpu_hours_daily,
                    )

            return True

    def end_job(self, tenant_id: str, job_id: str) -> bool:
        """
        Record job completion and release quota.

        Args:
            tenant_id: Tenant identifier
            job_id: Job identifier

        Returns:
            True if job ended successfully
        """
        with self._lock:
            usage = self._usage.get(tenant_id)
            job = self._active_jobs.get(job_id)

            if not usage:
                return False

            # Update concurrent count
            if job:
                del self._active_jobs[job_id]

            usage.concurrent_jobs = max(0, usage.concurrent_jobs - 1)
            usage.active_job_ids.discard(job_id)
            usage.last_updated = datetime.utcnow()

            self._stats["jobs_completed"] += 1

            return True

    def suspend_tenant(self, tenant_id: str, reason: str) -> bool:
        """
        Suspend a tenant's quota (block new jobs).

        Args:
            tenant_id: Tenant identifier
            reason: Suspension reason

        Returns:
            True if suspended
        """
        with self._lock:
            quota = self._quotas.get(tenant_id)
            if not quota:
                return False

            quota.is_active = False
            quota.suspended_at = datetime.utcnow()
            quota.suspension_reason = reason
            quota.updated_at = datetime.utcnow()

            return True

    def unsuspend_tenant(self, tenant_id: str) -> bool:
        """
        Unsuspend a tenant's quota.

        Args:
            tenant_id: Tenant identifier

        Returns:
            True if unsuspended
        """
        with self._lock:
            quota = self._quotas.get(tenant_id)
            if not quota:
                return False

            quota.is_active = True
            quota.suspended_at = None
            quota.suspension_reason = ""
            quota.updated_at = datetime.utcnow()

            return True

    def get_usage_percentage(
        self,
        tenant_id: str,
    ) -> Dict[str, float]:
        """
        Get usage as percentage of quota.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Dictionary of resource -> percentage
        """
        with self._lock:
            quota = self._quotas.get(tenant_id)
            usage = self._usage.get(tenant_id)

            if not quota or not usage:
                return {}

            return {
                "cpu_hours": (
                    (usage.cpu_hours_used / quota.cpu_hours_daily * 100)
                    if quota.cpu_hours_daily > 0
                    else 0
                ),
                "memory_gb_hours": (
                    (usage.memory_gb_hours_used / quota.memory_gb_hours_daily * 100)
                    if quota.memory_gb_hours_daily > 0
                    else 0
                ),
                "concurrent_jobs": (
                    (usage.concurrent_jobs / quota.max_concurrent_jobs * 100)
                    if quota.max_concurrent_jobs > 0
                    else 0
                ),
                "storage": (
                    (usage.storage_mb_used / quota.storage_mb * 100) if quota.storage_mb > 0 else 0
                ),
                "network": (
                    (usage.network_bytes_used / (quota.network_gb_daily * 1024**3) * 100)
                    if quota.network_gb_daily > 0
                    else 0
                ),
                "jobs_today": (
                    (usage.jobs_today / quota.jobs_per_day * 100) if quota.jobs_per_day > 0 else 0
                ),
            }

    def get_remaining(
        self,
        tenant_id: str,
    ) -> Dict[str, Any]:
        """
        Get remaining quota.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Dictionary of resource -> remaining
        """
        with self._lock:
            quota = self._quotas.get(tenant_id)
            usage = self._usage.get(tenant_id)

            if not quota or not usage:
                return {}

            return {
                "cpu_hours": quota.cpu_hours_daily - usage.cpu_hours_used,
                "memory_gb_hours": quota.memory_gb_hours_daily - usage.memory_gb_hours_used,
                "concurrent_jobs_available": quota.max_concurrent_jobs - usage.concurrent_jobs,
                "storage_mb": quota.storage_mb - usage.storage_mb_used,
                "network_gb": quota.network_gb_daily - (usage.network_bytes_used / 1024**3),
                "jobs_today": quota.jobs_per_day - usage.jobs_today,
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get quota manager statistics."""
        with self._lock:
            return {
                **self._stats,
                "tenants_with_quota": len(self._quotas),
                "active_jobs": len(self._active_jobs),
            }

    def cleanup_stale_jobs(self, max_age_hours: float = 24.0) -> int:
        """
        Cleanup stale job tracking entries.

        Args:
            max_age_hours: Maximum age for active jobs

        Returns:
            Number of jobs cleaned up
        """
        with self._lock:
            now = datetime.utcnow()
            stale_cutoff = now - timedelta(hours=max_age_hours)
            cleaned = 0

            stale_job_ids = [
                job_id
                for job_id, job in self._active_jobs.items()
                if job["start_time"] < stale_cutoff
            ]

            for job_id in stale_job_ids:
                job = self._active_jobs[job_id]
                self.end_job(job["tenant_id"], job_id)
                cleaned += 1

            return cleaned


def create_quota_manager(
    on_warning: Optional[Callable[[str, QuotaResource, float], None]] = None,
) -> ResourceQuotaManager:
    """
    Create a quota manager with default settings.

    Args:
        on_warning: Callback for quota warnings

    Returns:
        Configured ResourceQuotaManager
    """
    return ResourceQuotaManager(on_quota_warning=on_warning)
