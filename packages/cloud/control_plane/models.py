# -*- coding: utf-8 -*-
"""
Cloud Control Plane Data Models - CLOUD ZONE ONLY.

Phase 6 Implementation: Multi-tenant data model for Cloud Control Plane.

This module defines SQLAlchemy ORM models for:
- Organization/Workspace/User/Roles/Permissions (RBAC)
- Strategy/StrategyVersion/Build/Artifact
- Agent with trust state and enrollment
- Deployment/Run lifecycle
- Command/Approval workflow
- Telemetry/Alerts
- Data Retention/Access Audit
- Immutable Config Blobs

SECURITY CRITICAL:
    - All models enforce tenant boundary via workspace_id
    - Postgres RLS recommended for production
    - Access audit required for sensitive operations

Design Doc Reference:
    - Phase 6: "Cloud Control Plane: модель данных, RBAC, trust/revoke, blobs"
    - Multi-tenant isolation via workspace_id
"""

from __future__ import annotations

import enum
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, Final, FrozenSet, List, Optional, Set
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    LargeBinary,
    String,
    Table,
    Text,
    UniqueConstraint,
    CheckConstraint,
    event,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB, ARRAY

# Portable JSON type: uses JSONB for PostgreSQL, JSON for other databases (e.g., SQLite)
PortableJSON = JSON().with_variant(JSONB, "postgresql")

# Portable Array types: uses JSON for SQLite, ARRAY for PostgreSQL
# SQLite doesn't support ARRAY type, so we use JSON as base and ARRAY as PostgreSQL variant
PortableStringArray50 = JSON().with_variant(ARRAY(String(50)), "postgresql")
PortableStringArray100 = JSON().with_variant(ARRAY(String(100)), "postgresql")
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    validates,
)


# ============================================================================
# Constants
# ============================================================================

# Default TTL for enrollment tokens
DEFAULT_ENROLLMENT_TOKEN_TTL_HOURS: Final[int] = 24

# Maximum allowed enrollment token TTL
MAX_ENROLLMENT_TOKEN_TTL_HOURS: Final[int] = 168  # 7 days


# Trust states for agents
class TrustState(str, enum.Enum):
    """Agent trust state."""

    ENROLLED = "enrolled"
    REVOKED = "revoked"
    PENDING = "pending"
    SUSPENDED = "suspended"


# Deployment states (Design Doc 11.1)
class DeploymentState(str, enum.Enum):
    """
    Deployment lifecycle state per Design Doc 11.1.

    State Machine:
        CREATED → PENDING_APPROVAL → APPROVED → DEPLOYING → DEPLOYED
                                                              ↓
        HALTED ← (kill switch) ← RUNNING ← ────────────────────
           ↓
        SUSPENDED ← (user request)
           ↓
        REVOKED/RETIRED (terminal)
    """

    CREATED = "created"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    SUSPENDED = "suspended"
    HALTED = "halted"  # Design Doc 9.4, 11.1: kill switch triggered
    REVOKED = "revoked"  # Trust revoked
    FAILED = "failed"
    RETIRED = "retired"


# Halt reason (Design Doc 9.4)
class HaltReason(str, enum.Enum):
    """
    Reason for HALTED state per Design Doc 9.4.

    Kill switch triggers include:
    - max daily loss
    - broker errors burst
    - latency spike
    - order spam
    - state divergence
    - data feed invalid
    """

    MAX_DAILY_LOSS = "max_daily_loss"
    BROKER_ERROR_BURST = "broker_error_burst"
    LATENCY_SPIKE = "latency_spike"
    ORDER_SPAM = "order_spam"
    STATE_DIVERGENCE = "state_divergence"
    DATA_FEED_INVALID = "data_feed_invalid"
    MANUAL_KILL_SWITCH = "manual_kill_switch"
    RECONCILIATION_FAILURE = "reconciliation_failure"
    TIME_SYNC_DRIFT = "time_sync_drift"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    UNKNOWN = "unknown"


# Run states (Design Doc 11.2)
class RunState(str, enum.Enum):
    """
    Run lifecycle state per Design Doc 11.2.

    State Machine:
        CREATED → PENDING_APPROVAL → APPROVED → STARTING → RUNNING
                                                              ↓
        HALTED ← (kill switch) ← ─────────────────────────────┤
           ↓                                                  ↓
        STOPPED ← (user stop) ← PAUSED ← (user pause) ←──────┘
           ↓
        COMPLETED/FAILED (terminal)
    """

    CREATED = "created"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    HALTED = "halted"  # Design Doc 9.4, 11.2: kill switch triggered
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    COMPLETED = "completed"


# Build states
class BuildState(str, enum.Enum):
    """Build lifecycle state."""

    PENDING = "pending"
    BUILDING = "building"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Command status
class CommandStatus(str, enum.Enum):
    """Command lifecycle status."""

    PENDING = "pending"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"
    EXPIRED = "expired"


# Change classification
class ChangeClass(str, enum.Enum):
    """Change classification for approval workflow."""

    OPERATIONAL = "operational"
    TRADING_IMPACTING = "trading_impacting"
    SECURITY_SENSITIVE = "security_sensitive"
    DATA_SENSITIVE = "data_sensitive"


# Telemetry level
class TelemetryLevel(str, enum.Enum):
    """Telemetry detail level."""

    AGGREGATED = "aggregated"  # Default for retail
    DETAILED_NON_SENSITIVE = "detailed_non_sensitive"
    RAW_ORDER_EVENTS = "raw_order_events"  # Enterprise only, opt-in


# Alert severity
class AlertSeverity(str, enum.Enum):
    """Alert severity level."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# Access audit action types
class AuditAction(str, enum.Enum):
    """Audit action types."""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    APPROVE = "approve"
    REJECT = "reject"
    REVOKE = "revoke"
    ENROLL = "enroll"
    EXPORT = "export"
    BREAK_GLASS = "break_glass"


# ============================================================================
# Base Model with Tenant Isolation
# ============================================================================


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class TenantMixin:
    """
    Mixin to enforce tenant boundary via workspace_id.

    All tenant-scoped models MUST include this mixin.
    Postgres RLS policies should be created for production use.
    """

    @declared_attr
    def workspace_id(cls) -> Mapped[UUID]:
        return mapped_column(
            PGUUID(as_uuid=True),
            ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class SoftDeleteMixin:
    """Mixin for soft delete support."""

    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )

    @hybrid_property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None


# ============================================================================
# RBAC Models: Organization, Workspace, User, Role, Permission
# ============================================================================


class Organization(Base, TimestampMixin, SoftDeleteMixin):
    """
    Top-level organization entity.

    Organizations contain multiple workspaces and manage billing/compliance.
    """

    __tablename__ = "organizations"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[Optional[str]] = mapped_column(
        String(63), unique=True, nullable=True
    )  # Auto-generated if not provided
    display_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Organization settings
    settings: Mapped[Optional[Dict]] = mapped_column(PortableJSON, nullable=True, default=dict)

    # Enterprise features
    mfa_required: Mapped[bool] = mapped_column(Boolean, default=False)
    sso_provider: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    sso_config: Mapped[Optional[Dict]] = mapped_column(PortableJSON, nullable=True)

    # Billing and compliance
    billing_email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    billing_tier: Mapped[str] = mapped_column(String(50), default="free")

    # Relationships
    workspaces: Mapped[List["Workspace"]] = relationship(
        back_populates="organization",
        cascade="all, delete-orphan",
    )
    users: Mapped[List["User"]] = relationship(
        back_populates="organization",
        cascade="all, delete-orphan",
    )


class Workspace(Base, TimestampMixin, SoftDeleteMixin):
    """
    Workspace for tenant isolation.

    All tenant-scoped resources reference workspace_id for RLS.
    """

    __tablename__ = "workspaces"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    organization_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[Optional[str]] = mapped_column(
        String(63), nullable=True
    )  # Auto-generated if not provided
    display_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Workspace settings
    settings: Mapped[Optional[Dict]] = mapped_column(PortableJSON, nullable=True)

    # Data residency (for GDPR compliance)
    data_residency_region: Mapped[str] = mapped_column(String(50), default="us-east-1")

    # Relationships
    organization: Mapped["Organization"] = relationship(back_populates="workspaces")

    __table_args__ = (
        UniqueConstraint("organization_id", "slug", name="uq_workspace_org_slug"),
        Index("ix_workspace_org_id", "organization_id"),
    )


class Permission(Base, TimestampMixin):
    """
    Permission definition.

    Permissions define specific actions on resources.
    """

    __tablename__ = "permissions"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Resource and action (nullable for simpler permission creation like "org:write")
    resource: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, default="*")
    action: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, default="*")

    __table_args__ = (UniqueConstraint("resource", "action", name="uq_permission_resource_action"),)


# Role-Permission association table
role_permissions = Table(
    "role_permissions",
    Base.metadata,
    Column(
        "role_id",
        PGUUID(as_uuid=True),
        ForeignKey("roles.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "permission_id",
        PGUUID(as_uuid=True),
        ForeignKey("permissions.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class Role(Base, TimestampMixin, SoftDeleteMixin):
    """
    Role definition with permissions.

    Roles belong to an organization and can optionally be scoped to a workspace.
    If workspace_id is None, the role applies to all workspaces in the organization.
    """

    __tablename__ = "roles"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Organization scope (roles belong to an organization)
    organization_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Optional workspace scope (None = org-wide role)
    workspace_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # System roles cannot be deleted
    is_system_role: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    permissions: Mapped[List["Permission"]] = relationship(
        secondary=role_permissions,
    )
    users: Mapped[List["User"]] = relationship(
        secondary="user_roles",
        back_populates="roles",
    )

    __table_args__ = (
        UniqueConstraint(
            "organization_id", "workspace_id", "name", name="uq_role_org_workspace_name"
        ),
    )


# User-Role association table
# Note: Workspace scoping is done via Role.workspace_id, not on the association
user_roles = Table(
    "user_roles",
    Base.metadata,
    Column(
        "user_id",
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "role_id",
        PGUUID(as_uuid=True),
        ForeignKey("roles.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class User(Base, TimestampMixin, SoftDeleteMixin):
    """
    User account.

    Users belong to an organization and have roles per workspace.
    """

    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    organization_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    default_workspace_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Identity
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    display_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Authentication
    password_hash: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    mfa_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    mfa_secret: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # MFA enforcement (Design Doc 4.1)
    mfa_backup_codes_hash: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="Hashed backup codes for MFA recovery"
    )
    mfa_verified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When MFA was successfully verified first time",
    )
    failed_mfa_attempts: Mapped[int] = mapped_column(
        Integer, default=0, comment="Failed MFA attempts for lockout"
    )
    mfa_locked_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="MFA lockout expiry time"
    )

    # SSO
    sso_subject: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    @hybrid_property
    def last_login(self) -> Optional[datetime]:
        """Alias for last_login_at for API compatibility."""
        return self.last_login_at

    @last_login.setter
    def last_login(self, value: Optional[datetime]) -> None:
        """Setter for last_login alias."""
        self.last_login_at = value

    # Relationships
    organization: Mapped["Organization"] = relationship(back_populates="users")
    roles: Mapped[List["Role"]] = relationship(
        secondary="user_roles",
        back_populates="users",
    )

    __table_args__ = (
        Index("ix_user_org_id", "organization_id"),
        Index("ix_user_email", "email"),
    )


# ============================================================================
# Strategy and Artifact Models
# ============================================================================


class Strategy(Base, TenantMixin, TimestampMixin, SoftDeleteMixin):
    """
    Strategy definition.

    Strategies are versioned and contain multiple versions.
    """

    __tablename__ = "strategies"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Repository reference
    git_repo: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Metadata
    tags: Mapped[Optional[List[str]]] = mapped_column(PortableStringArray50, nullable=True)
    extra_metadata: Mapped[Optional[Dict]] = mapped_column(PortableJSON, nullable=True)

    # Relationships
    versions: Mapped[List["StrategyVersion"]] = relationship(
        back_populates="strategy",
        cascade="all, delete-orphan",
    )

    __table_args__ = (UniqueConstraint("workspace_id", "name", name="uq_strategy_workspace_name"),)


class StrategyVersion(Base, TenantMixin, TimestampMixin):
    """
    Strategy version.

    Each version represents a specific commit/tag of the strategy.
    """

    __tablename__ = "strategy_versions"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    strategy_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("strategies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Version info
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    git_sha: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    git_tag: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Status
    is_latest: Mapped[bool] = mapped_column(Boolean, default=False)
    is_deprecated: Mapped[bool] = mapped_column(Boolean, default=False)

    # Changelog and metadata
    changelog: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    extra_metadata: Mapped[Optional[Dict]] = mapped_column(PortableJSON, nullable=True)

    # Relationships
    strategy: Mapped["Strategy"] = relationship(back_populates="versions")
    builds: Mapped[List["Build"]] = relationship(
        back_populates="strategy_version",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("strategy_id", "version", name="uq_strategyversion_strategy_version"),
    )


class Build(Base, TenantMixin, TimestampMixin):
    """
    Build record.

    Represents a CI/CD build that produces artifacts.
    """

    __tablename__ = "builds"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    strategy_version_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("strategy_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Build info
    build_number: Mapped[int] = mapped_column(Integer, nullable=False)
    builder_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, default="system")

    # CI/CD reference
    ci_job_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    ci_pipeline_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Status
    status: Mapped[str] = mapped_column(String(50), default="pending")
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Logs
    logs_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Error handling
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Extra metadata
    extra_metadata: Mapped[Optional[Dict]] = mapped_column(PortableJSON, nullable=True)

    # Relationships
    strategy_version: Mapped["StrategyVersion"] = relationship(back_populates="builds")
    artifacts: Mapped[List["Artifact"]] = relationship(
        back_populates="build",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("strategy_version_id", "build_number", name="uq_build_version_number"),
    )


class Artifact(Base, TenantMixin, TimestampMixin):
    """
    Built artifact.

    Contains digest, signature, SBOM reference, and provenance.
    """

    __tablename__ = "artifacts"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    build_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("builds.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Artifact identity
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    format: Mapped[str] = mapped_column(String(50), nullable=False)  # oci_image, zip_bundle, wheel

    # Content addressing
    digest: Mapped[str] = mapped_column(String(128), nullable=False, index=True)  # sha256:...
    size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Security
    signature_ref: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True
    )  # Signature reference
    signature_algorithm: Mapped[str] = mapped_column(
        String(50), default="ed25519"
    )  # Design Doc: Ed25519 default
    is_signed: Mapped[bool] = mapped_column(Boolean, default=False)

    # SBOM
    sbom_ref: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    sbom_digest: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    # Provenance (SLSA)
    provenance: Mapped[Optional[Dict]] = mapped_column(PortableJSON, nullable=True)

    # Classification
    change_class: Mapped[str] = mapped_column(
        String(50),
        default=ChangeClass.TRADING_IMPACTING.value,
    )

    # Registry
    registry_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Extra metadata
    extra_metadata: Mapped[Optional[Dict]] = mapped_column(PortableJSON, nullable=True)

    # Relationships
    build: Mapped["Build"] = relationship(back_populates="artifacts")

    __table_args__ = (
        UniqueConstraint("digest", name="uq_artifact_digest"),
        Index("ix_artifact_digest", "digest"),
    )


# ============================================================================
# Agent Models
# ============================================================================


class AgentEnrollmentToken(Base, TenantMixin, TimestampMixin):
    """
    Agent enrollment token with TTL.

    One-time tokens for agent enrollment process.
    """

    __tablename__ = "agent_enrollment_tokens"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Token identity
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Token value (hashed for storage)
    token_hash: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)

    # Capabilities granted by this token
    capabilities: Mapped[Optional[List[str]]] = mapped_column(PortableStringArray100, nullable=True)

    # TTL
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Usage tracking
    is_used: Mapped[bool] = mapped_column(Boolean, default=False)
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    used_by_agent_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Creator
    created_by_user_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Metadata
    extra_metadata: Mapped[Optional[Dict]] = mapped_column(PortableJSON, nullable=True)

    @hybrid_property
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at

    __table_args__ = (Index("ix_enrollment_token_expires", "expires_at"),)


class Agent(Base, TenantMixin, TimestampMixin, SoftDeleteMixin):
    """
    Agent registration.

    Represents an enrolled agent with trust state and capabilities.
    """

    __tablename__ = "agents"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Identity
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    agent_version: Mapped[str] = mapped_column(String(50), nullable=False)

    # Cryptographic identity
    public_key: Mapped[str] = mapped_column(Text, nullable=False)  # PEM format
    key_algorithm: Mapped[str] = mapped_column(String(50), default="ed25519")

    # Trust state
    trust_state: Mapped[str] = mapped_column(
        String(50),
        default=TrustState.PENDING.value,
    )
    trust_state_changed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    revocation_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Capabilities
    capabilities: Mapped[Optional[List[str]]] = mapped_column(PortableStringArray100, nullable=True)

    # Health and status
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_heartbeat_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    health_status: Mapped[Optional[Dict]] = mapped_column(PortableJSON, nullable=True)

    # Enterprise features
    attestation: Mapped[Optional[Dict]] = mapped_column(PortableJSON, nullable=True)

    # Device info
    device_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    os_info: Mapped[Optional[Dict]] = mapped_column(PortableJSON, nullable=True)

    # Key rotation tracking (Design Doc 15.2)
    key_version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        server_default="1",
        comment="Key version for rotation tracking",
    )
    previous_public_key: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Previous public key for grace period during rotation",
    )
    key_rotated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp of last key rotation",
    )
    key_rotation_grace_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Until when previous key is still valid",
    )

    # Relationships
    deployments: Mapped[List["Deployment"]] = relationship(
        back_populates="agent",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("workspace_id", "name", name="uq_agent_workspace_name"),
        Index("ix_agent_trust_state", "trust_state"),
        Index("ix_agent_last_seen", "last_seen_at"),
    )


# ============================================================================
# Deployment and Run Models
# ============================================================================


class Deployment(Base, TenantMixin, TimestampMixin, SoftDeleteMixin):
    """
    Deployment of an artifact to an agent.

    State machine follows Design Doc 11.1.
    """

    __tablename__ = "deployments"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # References
    agent_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    artifact_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("artifacts.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    # Config blob reference
    config_blob_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("config_blobs.id", ondelete="RESTRICT"),
        nullable=True,
    )

    # State
    state: Mapped[str] = mapped_column(
        String(50),
        default=DeploymentState.CREATED.value,
    )
    state_changed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Scheduling
    desired_state: Mapped[str] = mapped_column(
        String(50),
        default=DeploymentState.DEPLOYED.value,
    )

    # Halt info (Design Doc 9.4, 11.1)
    halt_reason: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )
    halt_details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    halted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Metadata
    extra_metadata: Mapped[Optional[Dict]] = mapped_column(PortableJSON, nullable=True)

    # Relationships
    agent: Mapped["Agent"] = relationship(back_populates="deployments")
    artifact: Mapped["Artifact"] = relationship()
    config_blob: Mapped[Optional["ConfigBlob"]] = relationship()
    runs: Mapped[List["Run"]] = relationship(
        back_populates="deployment",
        cascade="all, delete-orphan",
    )

    __table_args__ = (Index("ix_deployment_agent_state", "agent_id", "state"),)


class Run(Base, TenantMixin, TimestampMixin):
    """
    Run instance of a deployment.

    State machine follows Design Doc 11.2.
    """

    __tablename__ = "runs"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Reference
    deployment_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("deployments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # State
    state: Mapped[str] = mapped_column(
        String(50),
        default=RunState.CREATED.value,
    )
    state_changed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    stopped_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Mode
    is_paper_trading: Mapped[bool] = mapped_column(Boolean, default=True)

    # Error handling
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_code: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Halt info (Design Doc 9.4, 11.2)
    halt_reason: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )
    halt_details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    halted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Metrics summary
    metrics_summary: Mapped[Optional[Dict]] = mapped_column(PortableJSON, nullable=True)

    # Relationships
    deployment: Mapped["Deployment"] = relationship(back_populates="runs")

    __table_args__ = (Index("ix_run_deployment_state", "deployment_id", "state"),)


# ============================================================================
# Command and Approval Models
# ============================================================================


class Command(Base, TenantMixin, TimestampMixin):
    """
    Command sent from Cloud to Agent.

    Includes idempotency key, payload reference, and approval requirement.
    """

    __tablename__ = "commands"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Idempotency
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)

    # Target
    agent_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    deployment_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("deployments.id", ondelete="SET NULL"),
        nullable=True,
    )
    run_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Command type
    command_type: Mapped[str] = mapped_column(String(100), nullable=False)

    # Payload reference (digest)
    payload_ref: Mapped[str] = mapped_column(String(128), nullable=False)  # sha256 digest

    # Classification
    change_class: Mapped[str] = mapped_column(
        String(50),
        default=ChangeClass.OPERATIONAL.value,
    )
    requires_approval: Mapped[bool] = mapped_column(Boolean, default=False)

    # Status
    status: Mapped[str] = mapped_column(
        String(50),
        default=CommandStatus.PENDING.value,
    )

    # Timing
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    executed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Result
    result: Mapped[Optional[Dict]] = mapped_column(PortableJSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    approvals: Mapped[List["ApprovalRecord"]] = relationship(
        back_populates="command",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint(
            "workspace_id", "idempotency_key", name="uq_command_workspace_idempotency"
        ),
        Index("ix_command_agent_status", "agent_id", "status"),
        Index("ix_command_idempotency", "idempotency_key"),
    )


class ApprovalRecord(Base, TenantMixin, TimestampMixin):
    """
    Approval record for commands.

    Includes evidence hash and attestation for audit trail.

    Design Doc 6.2, 12.2: Structured approval evidence with immutable blob references.
    All digests link to actual stored blobs for full audit trail.
    """

    __tablename__ = "approval_records"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Command reference
    command_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("commands.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Approval decision
    approved: Mapped[bool] = mapped_column(Boolean, nullable=False)
    approved_by: Mapped[str] = mapped_column(String(255), nullable=False)  # "local" or user ID

    # Evidence
    evidence_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    attestation: Mapped[Optional[Dict]] = mapped_column(PortableJSON, nullable=True)

    # Reason
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Diff shown at approval time (structured, links to blobs)
    diff_summary: Mapped[Optional[Dict]] = mapped_column(PortableJSON, nullable=True)

    # Design Doc 6.2, 12.2: Immutable blob references for structured evidence
    # These link to actual ConfigBlob/Artifact digests for audit verification
    config_blob_digest: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True, comment="ConfigBlob digest at approval time"
    )
    manifest_digest: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True, comment="Artifact manifest digest at approval time"
    )
    previous_state_digest: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True, comment="Previous configuration state digest"
    )
    new_state_digest: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True, comment="New configuration state digest"
    )

    # Relationships
    command: Mapped["Command"] = relationship(back_populates="approvals")

    __table_args__ = (Index("ix_approval_command", "command_id"),)


# ============================================================================
# Immutable Config Blob Model
# ============================================================================


class ConfigBlob(Base, TenantMixin, TimestampMixin):
    """
    Immutable configuration blob.

    Configs are stored by digest; any change creates new digest.
    Cloud stores only desired state (not secrets).
    """

    __tablename__ = "config_blobs"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Content addressing
    digest: Mapped[str] = mapped_column(String(128), nullable=False, index=True)  # sha256:...

    # Content (JSONB for queryability)
    content: Mapped[Dict] = mapped_column(PortableJSON, nullable=False)

    # Size
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)

    # Classification
    config_type: Mapped[str] = mapped_column(
        String(100), nullable=False
    )  # strategy, risk, execution, etc.

    # Schema version
    schema_version: Mapped[str] = mapped_column(String(20), default="1.0.0")

    # Creator
    created_by_user_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    __table_args__ = (
        UniqueConstraint("workspace_id", "digest", name="uq_configblob_workspace_digest"),
        Index("ix_configblob_digest", "digest"),
        Index("ix_configblob_type", "config_type"),
    )


# ============================================================================
# Telemetry and Alert Models
# ============================================================================


class TelemetryEvent(Base, TenantMixin, TimestampMixin):
    """
    Telemetry event from agent.

    Events are classified by level (aggregated/detailed/raw).
    """

    __tablename__ = "telemetry_events"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Source
    agent_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    run_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Event info
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    event_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Level
    telemetry_level: Mapped[str] = mapped_column(
        String(50),
        default=TelemetryLevel.AGGREGATED.value,
    )

    # Payload (redacted)
    payload: Mapped[Dict] = mapped_column(PortableJSON, nullable=False)

    # Redaction confirmation
    redaction_applied: Mapped[bool] = mapped_column(Boolean, default=True)
    redaction_version: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    __table_args__ = (
        Index("ix_telemetry_agent_time", "agent_id", "event_timestamp"),
        Index("ix_telemetry_type", "event_type"),
    )


class Alert(Base, TenantMixin, TimestampMixin):
    """
    Alert generated from telemetry or system events.
    """

    __tablename__ = "alerts"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Source
    agent_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    run_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Alert info
    alert_type: Mapped[str] = mapped_column(String(100), nullable=False)
    severity: Mapped[str] = mapped_column(
        String(20),
        default=AlertSeverity.INFO.value,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)

    # Status
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Context
    context: Mapped[Optional[Dict]] = mapped_column(PortableJSON, nullable=True)

    __table_args__ = (
        Index("ix_alert_severity", "severity"),
        Index("ix_alert_unresolved", "resolved", "severity"),
    )


# ============================================================================
# Data Retention and Access Audit Models
# ============================================================================


class DataRetentionPolicy(Base, TenantMixin, TimestampMixin):
    """
    Data retention policy per workspace.

    Defines retention periods for different data types.
    """

    __tablename__ = "data_retention_policies"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Data type
    data_type: Mapped[str] = mapped_column(String(100), nullable=False)

    # Retention
    retention_days: Mapped[int] = mapped_column(Integer, nullable=False)

    # Auto-purge
    auto_purge_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_purge_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # GDPR/DSAR
    dsar_export_enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (
        UniqueConstraint("workspace_id", "data_type", name="uq_retention_workspace_type"),
    )


class AccessAudit(Base, TenantMixin, TimestampMixin):
    """
    Access audit log.

    Records all access to sensitive data and break-glass events.
    """

    __tablename__ = "access_audits"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Actor
    user_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    agent_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="SET NULL"),
        nullable=True,
    )
    actor_type: Mapped[str] = mapped_column(String(50), nullable=False)  # user, agent, system
    actor_ip: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Action
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Break-glass
    is_break_glass: Mapped[bool] = mapped_column(Boolean, default=False)
    break_glass_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    break_glass_evidence_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    # Result
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Context
    request_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    context: Mapped[Optional[Dict]] = mapped_column(PortableJSON, nullable=True)

    __table_args__ = (
        Index("ix_audit_actor", "actor_type", "user_id"),
        Index("ix_audit_resource", "resource_type", "resource_id"),
        Index("ix_audit_break_glass", "is_break_glass"),
        Index("ix_audit_time", "created_at"),
    )


# ============================================================================
# Governance Models (Design Doc Phase 10: DB-backed governance)
# ============================================================================


class ResidencyPolicyType(str, enum.Enum):
    """Residency mode."""

    CLOUD = "cloud"
    ENTERPRISE_LOCAL = "enterprise_local"
    HYBRID = "hybrid"


class DSARRequestStatus(str, enum.Enum):
    """DSAR request status."""

    PENDING = "pending"
    VERIFIED = "verified"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    EXPIRED = "expired"


class DSARRequestType(str, enum.Enum):
    """DSAR request type."""

    ACCESS = "access"
    ERASURE = "erasure"
    PORTABILITY = "portability"
    RECTIFICATION = "rectification"
    RESTRICTION = "restriction"


class GovernancePolicy(Base, TenantMixin, TimestampMixin):
    """
    Governance policy record - source of truth for residency, retention, and other policies.

    Design Doc Phase 10: DB-backed governance policies.
    """

    __tablename__ = "governance_policies"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Policy type
    policy_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # residency, retention, etc.

    # Data type this policy applies to (for retention)
    data_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Policy configuration (JSON)
    config: Mapped[Dict] = mapped_column(PortableJSON, nullable=False, default=dict)

    # Status
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Creator
    created_by_user_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    __table_args__ = (
        UniqueConstraint("workspace_id", "policy_type", "data_type", name="uq_governance_policy"),
        Index("ix_governance_policy_type", "policy_type"),
    )


class DSARRequest(Base, TenantMixin, TimestampMixin):
    """
    DSAR (Data Subject Access Request) record.

    Design Doc Phase 10: DB-backed DSAR workflow.
    """

    __tablename__ = "dsar_requests"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Subject
    user_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Request details
    request_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(
        String(50),
        default=DSARRequestStatus.PENDING.value,
    )

    # Data categories
    data_categories: Mapped[Optional[List[str]]] = mapped_column(
        PortableStringArray100, nullable=True
    )
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Identity verification
    verification_method: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    verified_by: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Deadlines (GDPR: 30 days, extendable to 90)
    deadline: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    extended_deadline: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Processing
    processed_by: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Export info (for ACCESS/PORTABILITY)
    export_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    export_checksum: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    # Statistics
    records_processed: Mapped[int] = mapped_column(Integer, default=0)
    records_deleted: Mapped[int] = mapped_column(Integer, default=0)

    # Notes
    notes: Mapped[Optional[Dict]] = mapped_column(PortableJSON, nullable=True)

    __table_args__ = (
        Index("ix_dsar_user", "user_id"),
        Index("ix_dsar_status", "status"),
        Index("ix_dsar_deadline", "deadline"),
    )


class LegalHold(Base, TenantMixin, TimestampMixin):
    """
    Legal hold on data - prevents automatic purge.

    Design Doc Phase 10: DB-backed legal holds.
    """

    __tablename__ = "legal_holds"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # What's on hold
    data_type: Mapped[str] = mapped_column(String(100), nullable=False)

    # Hold details
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    hold_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    released_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    released_by: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Creator
    created_by_user_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    __table_args__ = (
        UniqueConstraint("workspace_id", "data_type", name="uq_legal_hold_workspace_type"),
        Index("ix_legal_hold_active", "is_active"),
    )


class BreakGlassRequest(Base, TenantMixin, TimestampMixin):
    """
    Break-glass access request for emergency access.

    Design Doc Phase 10: DB-backed break-glass audit.
    """

    __tablename__ = "break_glass_requests"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Requester
    requested_by: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Request details
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    scope: Mapped[str] = mapped_column(String(100), nullable=False)  # what's being accessed
    resource_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Duration
    valid_until: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Approval
    approved: Mapped[bool] = mapped_column(Boolean, default=False)
    approved_by: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Evidence
    evidence_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    # Revocation
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_by: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    __table_args__ = (
        Index("ix_break_glass_requester", "requested_by"),
        Index("ix_break_glass_valid", "valid_until"),
    )


class GovernanceAuditLog(Base, TenantMixin, TimestampMixin):
    """
    Governance audit log - tracks all governance operations.

    Design Doc Phase 10: Comprehensive audit trail.
    """

    __tablename__ = "governance_audit_logs"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Action
    action: Mapped[str] = mapped_column(String(100), nullable=False)

    # Actor
    actor_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    actor_type: Mapped[str] = mapped_column(String(50), default="user")  # user, system, scheduler

    # Details
    details: Mapped[Dict] = mapped_column(PortableJSON, nullable=False, default=dict)

    # Related entities
    policy_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("governance_policies.id", ondelete="SET NULL"),
        nullable=True,
    )
    dsar_request_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("dsar_requests.id", ondelete="SET NULL"),
        nullable=True,
    )

    __table_args__ = (
        Index("ix_governance_audit_action", "action"),
        Index("ix_governance_audit_time", "created_at"),
    )


# ============================================================================
# Experiments Tracking (Design Doc - Feature Flags & A/B Testing)
# ============================================================================


class ExperimentState(str, enum.Enum):
    """Experiment lifecycle state."""

    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class Experiment(Base, TenantMixin, TimestampMixin, SoftDeleteMixin):
    """
    Experiment definition for feature flags and A/B testing.

    Tracks experiments across deployments with variant assignments.
    """

    __tablename__ = "experiments"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Identity
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # State
    state: Mapped[str] = mapped_column(
        String(50),
        default=ExperimentState.DRAFT.value,
    )

    # Configuration
    hypothesis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metric_keys: Mapped[Optional[List[str]]] = mapped_column(
        PortableStringArray100,
        nullable=True,
        comment="Metrics to track for this experiment",
    )
    start_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    end_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Variants configuration
    variants: Mapped[Optional[Dict]] = mapped_column(
        PortableJSON,
        nullable=True,
        comment="Variant definitions: {variant_id: {name, weight, config}}",
    )
    default_variant: Mapped[str] = mapped_column(String(50), default="control")

    # Traffic allocation
    traffic_percentage: Mapped[int] = mapped_column(
        Integer,
        default=100,
        comment="Percentage of eligible traffic included in experiment",
    )

    # Results
    results_summary: Mapped[Optional[Dict]] = mapped_column(
        PortableJSON,
        nullable=True,
        comment="Summary statistics and results",
    )
    winner_variant: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Relationships
    assignments: Mapped[List["ExperimentAssignment"]] = relationship(
        back_populates="experiment",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("workspace_id", "name", name="uq_experiment_workspace_name"),
        Index("ix_experiment_state", "state"),
    )


class ExperimentAssignment(Base, TenantMixin, TimestampMixin):
    """
    Experiment variant assignment for a specific entity (deployment, user, etc.).

    Tracks which variant an entity is assigned to.
    """

    __tablename__ = "experiment_assignments"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Experiment reference
    experiment_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Entity being assigned (polymorphic)
    entity_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Type of entity: deployment, agent, user",
    )
    entity_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        comment="ID of the assigned entity",
    )

    # Assignment
    variant: Mapped[str] = mapped_column(String(50), nullable=False)
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )

    # Tracking
    first_exposure_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When entity was first exposed to variant",
    )
    conversion_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When conversion event occurred (if any)",
    )
    metrics: Mapped[Optional[Dict]] = mapped_column(
        PortableJSON,
        nullable=True,
        comment="Collected metrics for this assignment",
    )

    # Relationships
    experiment: Mapped["Experiment"] = relationship(back_populates="assignments")

    __table_args__ = (
        UniqueConstraint(
            "experiment_id", "entity_type", "entity_id", name="uq_experiment_assignment"
        ),
        Index("ix_experiment_assignment_entity", "entity_type", "entity_id"),
    )


# ============================================================================
# Helper Functions
# ============================================================================


def generate_enrollment_token() -> str:
    """Generate a secure enrollment token."""
    return secrets.token_urlsafe(32)


def compute_token_hash(token: str) -> str:
    """Compute hash of enrollment token for storage."""
    import hashlib

    return hashlib.sha256(token.encode()).hexdigest()


def create_enrollment_token_expiry(ttl_hours: int = DEFAULT_ENROLLMENT_TOKEN_TTL_HOURS) -> datetime:
    """Create expiry datetime for enrollment token."""
    if ttl_hours > MAX_ENROLLMENT_TOKEN_TTL_HOURS:
        ttl_hours = MAX_ENROLLMENT_TOKEN_TTL_HOURS
    return datetime.utcnow() + timedelta(hours=ttl_hours)


# ============================================================================
# Postgres RLS Setup SQL (to be executed during migration)
# ============================================================================

POSTGRES_RLS_SETUP = """
-- Enable RLS on tenant-scoped tables
ALTER TABLE strategies ENABLE ROW LEVEL SECURITY;
ALTER TABLE strategy_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE builds ENABLE ROW LEVEL SECURITY;
ALTER TABLE artifacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE agents ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_enrollment_tokens ENABLE ROW LEVEL SECURITY;
ALTER TABLE deployments ENABLE ROW LEVEL SECURITY;
ALTER TABLE runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE commands ENABLE ROW LEVEL SECURITY;
ALTER TABLE approval_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE config_blobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE telemetry_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE data_retention_policies ENABLE ROW LEVEL SECURITY;
ALTER TABLE access_audits ENABLE ROW LEVEL SECURITY;
ALTER TABLE roles ENABLE ROW LEVEL SECURITY;

-- Create policies (example for strategies table)
-- Application must set current_setting('app.current_workspace_id')
CREATE POLICY workspace_isolation ON strategies
    USING (workspace_id = current_setting('app.current_workspace_id')::uuid);

-- Repeat for other tables...
"""
