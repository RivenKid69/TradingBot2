# -*- coding: utf-8 -*-
"""
Research Job Models - Database models for cloud research jobs.

CCEA Phase 10: Data models for research job management.

Design Doc 15.3/5.3:
- Sandbox for research jobs
- Quotas CPU/RAM/time
- Tenant isolation

This module defines SQLAlchemy ORM models for:
- ResearchJob: Job execution record
- JobQuota: Tenant quota configuration
- QuotaUsage: Current usage tracking
- EgressPolicy: Network egress rules
- AbuseIncident: Abuse detection records
- JobArtifact: Job input/output artifacts
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Dict, Final, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB, ARRAY
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import Mapped, mapped_column, relationship

# Import base from existing models
from packages.cloud.control_plane.models import (
    Base,
    TenantMixin,
    TimestampMixin,
    SoftDeleteMixin,
    PortableJSON,
    PortableStringArray100,
)


# ============================================================================
# Enums
# ============================================================================


class ResearchJobState(str, enum.Enum):
    """Research job lifecycle state."""

    PENDING = "pending"
    VALIDATING = "validating"
    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"
    CANCELLED = "cancelled"


class JobTerminationReason(str, enum.Enum):
    """Reason for job termination."""

    COMPLETED = "completed"
    TIMEOUT = "timeout"
    OOM = "oom"
    ABUSE_DETECTED = "abuse_detected"
    QUOTA_EXCEEDED = "quota_exceeded"
    USER_CANCELLED = "user_cancelled"
    SYSTEM_ERROR = "system_error"
    EGRESS_VIOLATION = "egress_violation"
    TENANT_VIOLATION = "tenant_violation"


class IsolationLevel(str, enum.Enum):
    """Sandbox isolation level."""

    NONE = "none"
    PROCESS = "process"
    CONTAINER = "container"
    GVISOR = "gvisor"
    MICROVM = "microvm"


class QuotaTier(str, enum.Enum):
    """Quota tier for tenants."""

    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class AbuseType(str, enum.Enum):
    """Type of detected abuse."""

    CRYPTOCURRENCY_MINING = "cryptocurrency_mining"
    PORT_SCANNING = "port_scanning"
    NETWORK_SCANNING = "network_scanning"
    BOTNET_C2 = "botnet_c2"
    DATA_EXFILTRATION = "data_exfiltration"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    MALWARE_EXECUTION = "malware_execution"
    REVERSE_SHELL = "reverse_shell"


class AlertSeverity(str, enum.Enum):
    """Alert severity level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# Research Job Model
# ============================================================================


class ResearchJob(Base, TenantMixin, TimestampMixin, SoftDeleteMixin):
    """
    Research job execution record.

    Tracks the complete lifecycle of a cloud research job including
    configuration, execution state, resource usage, and results.
    """

    __tablename__ = "research_jobs"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Creator
    user_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Job identity
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tags: Mapped[Optional[List[str]]] = mapped_column(PortableStringArray100, nullable=True)

    # State
    state: Mapped[str] = mapped_column(
        String(50),
        default=ResearchJobState.PENDING.value,
        index=True,
    )
    state_changed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    termination_reason: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Configuration
    isolation_level: Mapped[str] = mapped_column(
        String(50),
        default=IsolationLevel.CONTAINER.value,
    )
    cpu_limit: Mapped[float] = mapped_column(Float, default=1.0)
    memory_limit_mb: Mapped[int] = mapped_column(Integer, default=1024)
    timeout_seconds: Mapped[int] = mapped_column(Integer, default=3600)
    disk_limit_mb: Mapped[int] = mapped_column(Integer, default=512)

    # Network
    network_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    egress_allowlist: Mapped[Optional[List[str]]] = mapped_column(
        PortableStringArray100, nullable=True
    )

    # Execution environment
    docker_image: Mapped[str] = mapped_column(String(255), default="python:3.11-slim")
    entrypoint: Mapped[str] = mapped_column(String(255), default="main.py")
    env_vars: Mapped[Optional[Dict]] = mapped_column(PortableJSON, nullable=True)

    # Timing
    queued_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Sandbox
    sandbox_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Results
    exit_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    stdout_artifact_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )
    stderr_artifact_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    # Resource usage
    cpu_seconds_used: Mapped[float] = mapped_column(Float, default=0.0)
    memory_mb_seconds_used: Mapped[float] = mapped_column(Float, default=0.0)
    peak_memory_mb: Mapped[float] = mapped_column(Float, default=0.0)
    network_bytes_in: Mapped[int] = mapped_column(Integer, default=0)
    network_bytes_out: Mapped[int] = mapped_column(Integer, default=0)

    # Security
    alerts_count: Mapped[int] = mapped_column(Integer, default=0)
    egress_violations_count: Mapped[int] = mapped_column(Integer, default=0)

    # Extra metadata
    extra_metadata: Mapped[Optional[Dict]] = mapped_column(PortableJSON, nullable=True)

    # Relationships
    artifacts: Mapped[List["JobArtifact"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
    )
    abuse_incidents: Mapped[List["AbuseIncident"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_research_job_state", "state"),
        Index("ix_research_job_user", "user_id"),
        Index("ix_research_job_created", "created_at"),
    )


# ============================================================================
# Quota Models
# ============================================================================


class JobQuota(Base, TenantMixin, TimestampMixin):
    """
    Tenant quota configuration for research jobs.

    Defines resource limits and usage quotas per tenant.
    """

    __tablename__ = "job_quotas"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Tier
    tier: Mapped[str] = mapped_column(
        String(50),
        default=QuotaTier.FREE.value,
    )

    # Compute quotas (daily)
    cpu_hours_daily: Mapped[float] = mapped_column(Float, default=2.0)
    memory_gb_hours_daily: Mapped[float] = mapped_column(Float, default=4.0)

    # Concurrent limits
    max_concurrent_jobs: Mapped[int] = mapped_column(Integer, default=2)
    max_job_duration_seconds: Mapped[int] = mapped_column(Integer, default=3600)

    # Per-job limits
    max_cpu_per_job: Mapped[float] = mapped_column(Float, default=4.0)
    max_memory_per_job_mb: Mapped[int] = mapped_column(Integer, default=8192)
    max_disk_per_job_mb: Mapped[int] = mapped_column(Integer, default=4096)

    # Storage quota
    storage_mb: Mapped[int] = mapped_column(Integer, default=1024)

    # Network quota (daily)
    network_gb_daily: Mapped[float] = mapped_column(Float, default=1.0)

    # Job count limits
    jobs_per_day: Mapped[int] = mapped_column(Integer, default=100)
    jobs_per_month: Mapped[int] = mapped_column(Integer, default=2000)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    suspended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    suspension_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Expiration
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (UniqueConstraint("workspace_id", name="uq_job_quota_workspace"),)


class QuotaUsageRecord(Base, TenantMixin, TimestampMixin):
    """
    Quota usage tracking record.

    Tracks daily/monthly resource consumption.
    """

    __tablename__ = "quota_usage_records"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Period
    period_type: Mapped[str] = mapped_column(String(20), default="daily")
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Compute usage
    cpu_hours_used: Mapped[float] = mapped_column(Float, default=0.0)
    memory_gb_hours_used: Mapped[float] = mapped_column(Float, default=0.0)

    # Storage
    storage_mb_used: Mapped[int] = mapped_column(Integer, default=0)

    # Network
    network_bytes_used: Mapped[int] = mapped_column(Integer, default=0)

    # Job counts
    jobs_run: Mapped[int] = mapped_column(Integer, default=0)
    jobs_completed: Mapped[int] = mapped_column(Integer, default=0)
    jobs_failed: Mapped[int] = mapped_column(Integer, default=0)
    jobs_terminated: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        Index("ix_quota_usage_period", "workspace_id", "period_type", "period_start"),
    )


# ============================================================================
# Egress Policy Model
# ============================================================================


class EgressPolicyRecord(Base, TenantMixin, TimestampMixin):
    """
    Egress policy configuration for tenants.

    Defines allowed network destinations and rules.
    """

    __tablename__ = "egress_policies"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Policy identity
    name: Mapped[str] = mapped_column(String(255), default="default")
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Allowlist
    allowed_destinations: Mapped[Optional[List[str]]] = mapped_column(
        PortableStringArray100, nullable=True
    )
    allowed_ports: Mapped[Optional[List[int]]] = mapped_column(PortableJSON, nullable=True)

    # Mode
    allowlist_only: Mapped[bool] = mapped_column(Boolean, default=True)
    log_all_attempts: Mapped[bool] = mapped_column(Boolean, default=True)

    # Rate limits
    max_requests_per_minute: Mapped[int] = mapped_column(Integer, default=60)
    max_bytes_per_minute: Mapped[int] = mapped_column(Integer, default=50 * 1024 * 1024)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (
        UniqueConstraint("workspace_id", "name", name="uq_egress_policy_workspace_name"),
    )


class EgressViolationRecord(Base, TenantMixin, TimestampMixin):
    """
    Record of egress policy violations.
    """

    __tablename__ = "egress_violations"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Source
    job_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("research_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Violation details
    destination: Mapped[str] = mapped_column(String(255), nullable=False)
    port: Mapped[int] = mapped_column(Integer, nullable=False)
    protocol: Mapped[str] = mapped_column(String(20), default="TCP")
    bytes_attempted: Mapped[int] = mapped_column(Integer, default=0)

    # Rule info
    matched_rule_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    action_taken: Mapped[str] = mapped_column(String(50), default="deny")
    reason: Mapped[str] = mapped_column(Text, nullable=False)

    # Risk assessment
    risk_score: Mapped[float] = mapped_column(Float, default=0.0)
    is_suspicious: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (
        Index("ix_egress_violation_job", "job_id"),
        Index("ix_egress_violation_time", "created_at"),
    )


# ============================================================================
# Abuse Incident Model
# ============================================================================


class AbuseIncident(Base, TenantMixin, TimestampMixin):
    """
    Record of detected abuse.
    """

    __tablename__ = "abuse_incidents"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Source
    job_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("research_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Detection
    abuse_type: Mapped[str] = mapped_column(String(50), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), default=AlertSeverity.MEDIUM.value)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    # Details
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[Optional[Dict]] = mapped_column(PortableJSON, nullable=True)

    # Detection metadata
    detector_name: Mapped[str] = mapped_column(String(100), nullable=False)
    detection_method: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Action
    action_taken: Mapped[str] = mapped_column(String(100), default="alert")
    job_terminated: Mapped[bool] = mapped_column(Boolean, default=False)

    # Risk score
    risk_score: Mapped[float] = mapped_column(Float, default=0.0)

    # Relationships
    job: Mapped["ResearchJob"] = relationship(back_populates="abuse_incidents")

    __table_args__ = (
        Index("ix_abuse_incident_job", "job_id"),
        Index("ix_abuse_incident_type", "abuse_type"),
        Index("ix_abuse_incident_severity", "severity"),
    )


# ============================================================================
# Job Artifact Model
# ============================================================================


class JobArtifact(Base, TenantMixin, TimestampMixin):
    """
    Job input/output artifacts.

    Stores code, input data, and output files for jobs.
    """

    __tablename__ = "job_artifacts"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Job reference
    job_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("research_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Artifact identity
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    artifact_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # code, input, output, stdout, stderr

    # Storage
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    digest: Mapped[str] = mapped_column(String(128), nullable=False)  # sha256

    # Content type
    content_type: Mapped[str] = mapped_column(String(100), default="application/octet-stream")

    # Expiration
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    job: Mapped["ResearchJob"] = relationship(back_populates="artifacts")

    __table_args__ = (
        Index("ix_job_artifact_job", "job_id"),
        Index("ix_job_artifact_type", "artifact_type"),
        UniqueConstraint("job_id", "name", name="uq_job_artifact_job_name"),
    )


# ============================================================================
# Helper Functions
# ============================================================================


def create_research_job(
    workspace_id: UUID,
    user_id: UUID,
    name: str,
    cpu_limit: float = 1.0,
    memory_limit_mb: int = 1024,
    timeout_seconds: int = 3600,
    **kwargs,
) -> ResearchJob:
    """Create a new research job."""
    return ResearchJob(
        workspace_id=workspace_id,
        user_id=user_id,
        name=name,
        cpu_limit=cpu_limit,
        memory_limit_mb=memory_limit_mb,
        timeout_seconds=timeout_seconds,
        **kwargs,
    )


def create_job_quota(
    workspace_id: UUID,
    tier: QuotaTier = QuotaTier.FREE,
    **kwargs,
) -> JobQuota:
    """Create a job quota for a workspace."""
    # Set defaults based on tier
    defaults = {
        QuotaTier.FREE: {
            "cpu_hours_daily": 2.0,
            "memory_gb_hours_daily": 4.0,
            "max_concurrent_jobs": 2,
            "max_job_duration_seconds": 3600,
            "storage_mb": 1024,
            "network_gb_daily": 1.0,
            "jobs_per_day": 100,
        },
        QuotaTier.PREMIUM: {
            "cpu_hours_daily": 24.0,
            "memory_gb_hours_daily": 48.0,
            "max_concurrent_jobs": 10,
            "max_job_duration_seconds": 86400,
            "storage_mb": 10240,
            "network_gb_daily": 10.0,
            "jobs_per_day": 1000,
        },
        QuotaTier.ENTERPRISE: {
            "cpu_hours_daily": 1000.0,
            "memory_gb_hours_daily": 2000.0,
            "max_concurrent_jobs": 100,
            "max_job_duration_seconds": 604800,
            "storage_mb": 102400,
            "network_gb_daily": 100.0,
            "jobs_per_day": 10000,
        },
    }

    tier_defaults = defaults.get(tier, defaults[QuotaTier.FREE])
    tier_defaults.update(kwargs)

    return JobQuota(
        workspace_id=workspace_id,
        tier=tier.value,
        **tier_defaults,
    )
