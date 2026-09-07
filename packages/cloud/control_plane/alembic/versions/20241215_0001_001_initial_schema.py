"""Initial schema for CCEA Cloud Control Plane.

Revision ID: 001
Revises: None
Create Date: 2024-12-15

Phase 7 (WI-CLOUD-04): Creates all tables for multi-tenant control plane.
Includes: Organizations, Workspaces, Users, Roles, Permissions, Strategies,
Agents, Deployments, Runs, Commands, ConfigBlobs, Telemetry, Alerts.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _is_postgresql() -> bool:
    """Check if we're running on PostgreSQL."""
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    """Create all tables for CCEA Cloud Control Plane."""

    # Use JSONB for PostgreSQL, JSON for others (SQLite)
    json_type = postgresql.JSONB() if _is_postgresql() else sa.JSON()
    uuid_type = postgresql.UUID(as_uuid=True) if _is_postgresql() else sa.String(36)
    array_type_50 = postgresql.ARRAY(sa.String(50)) if _is_postgresql() else sa.JSON()
    array_type_100 = postgresql.ARRAY(sa.String(100)) if _is_postgresql() else sa.JSON()

    # ============================================================================
    # Organizations
    # ============================================================================
    op.create_table(
        "organizations",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(63), unique=True, nullable=True),
        sa.Column("display_name", sa.String(255), nullable=True),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("settings", json_type, nullable=True),
        sa.Column("mfa_required", sa.Boolean(), default=False),
        sa.Column("sso_provider", sa.String(50), nullable=True),
        sa.Column("sso_config", json_type, nullable=True),
        sa.Column("billing_email", sa.String(255), nullable=True),
        sa.Column("billing_tier", sa.String(50), default="free"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )

    # ============================================================================
    # Workspaces
    # ============================================================================
    op.create_table(
        "workspaces",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column(
            "organization_id",
            uuid_type,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(63), nullable=True),
        sa.Column("display_name", sa.String(255), nullable=True),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("settings", json_type, nullable=True),
        sa.Column("data_residency_region", sa.String(50), default="us-east-1"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("organization_id", "slug", name="uq_workspace_org_slug"),
    )
    op.create_index("ix_workspace_org_id", "workspaces", ["organization_id"])

    # ============================================================================
    # Permissions
    # ============================================================================
    op.create_table(
        "permissions",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column("name", sa.String(100), unique=True, nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("resource", sa.String(100), nullable=True, default="*"),
        sa.Column("action", sa.String(50), nullable=True, default="*"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("resource", "action", name="uq_permission_resource_action"),
    )

    # ============================================================================
    # Roles
    # ============================================================================
    op.create_table(
        "roles",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "organization_id",
            uuid_type,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "workspace_id",
            uuid_type,
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=True,
            index=True,
        ),
        sa.Column("is_system_role", sa.Boolean(), default=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint(
            "organization_id", "workspace_id", "name", name="uq_role_org_workspace_name"
        ),
    )

    # Role-Permission association
    op.create_table(
        "role_permissions",
        sa.Column(
            "role_id", uuid_type, sa.ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True
        ),
        sa.Column(
            "permission_id",
            uuid_type,
            sa.ForeignKey("permissions.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )

    # ============================================================================
    # Users
    # ============================================================================
    op.create_table(
        "users",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column(
            "organization_id",
            uuid_type,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "default_workspace_id",
            uuid_type,
            sa.ForeignKey("workspaces.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("display_name", sa.String(255), nullable=True),
        sa.Column("password_hash", sa.String(255), nullable=True),
        sa.Column("mfa_enabled", sa.Boolean(), default=False),
        sa.Column("mfa_secret", sa.String(255), nullable=True),
        sa.Column("sso_subject", sa.String(255), nullable=True),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_user_org_id", "users", ["organization_id"])
    op.create_index("ix_user_email", "users", ["email"])

    # User-Role association
    op.create_table(
        "user_roles",
        sa.Column(
            "user_id", uuid_type, sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
        ),
        sa.Column(
            "role_id", uuid_type, sa.ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True
        ),
    )

    # ============================================================================
    # Strategies
    # ============================================================================
    op.create_table(
        "strategies",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column(
            "workspace_id",
            uuid_type,
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("git_repo", sa.String(500), nullable=True),
        sa.Column("tags", array_type_50, nullable=True),
        sa.Column("extra_metadata", json_type, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("workspace_id", "name", name="uq_strategy_workspace_name"),
    )

    # ============================================================================
    # Strategy Versions
    # ============================================================================
    op.create_table(
        "strategy_versions",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column(
            "workspace_id",
            uuid_type,
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "strategy_id",
            uuid_type,
            sa.ForeignKey("strategies.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("git_sha", sa.String(40), nullable=True),
        sa.Column("git_tag", sa.String(100), nullable=True),
        sa.Column("is_latest", sa.Boolean(), default=False),
        sa.Column("is_deprecated", sa.Boolean(), default=False),
        sa.Column("changelog", sa.Text(), nullable=True),
        sa.Column("extra_metadata", json_type, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("strategy_id", "version", name="uq_strategyversion_strategy_version"),
    )

    # ============================================================================
    # Builds
    # ============================================================================
    op.create_table(
        "builds",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column(
            "workspace_id",
            uuid_type,
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "strategy_version_id",
            uuid_type,
            sa.ForeignKey("strategy_versions.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("build_number", sa.Integer(), nullable=False),
        sa.Column("builder_id", sa.String(255), nullable=True, default="system"),
        sa.Column("ci_job_id", sa.String(255), nullable=True),
        sa.Column("ci_pipeline_url", sa.String(500), nullable=True),
        sa.Column("status", sa.String(50), default="pending"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("logs_url", sa.String(500), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("extra_metadata", json_type, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("strategy_version_id", "build_number", name="uq_build_version_number"),
    )

    # ============================================================================
    # Artifacts
    # ============================================================================
    op.create_table(
        "artifacts",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column(
            "workspace_id",
            uuid_type,
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "build_id",
            uuid_type,
            sa.ForeignKey("builds.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("format", sa.String(50), nullable=False),
        sa.Column("digest", sa.String(128), nullable=False, index=True),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("signature_ref", sa.String(500), nullable=True),
        sa.Column("signature_algorithm", sa.String(50), default="sigstore"),
        sa.Column("is_signed", sa.Boolean(), default=False),
        sa.Column("sbom_ref", sa.String(500), nullable=True),
        sa.Column("sbom_digest", sa.String(128), nullable=True),
        sa.Column("provenance", json_type, nullable=True),
        sa.Column("change_class", sa.String(50), default="trading_impacting"),
        sa.Column("registry_url", sa.String(500), nullable=True),
        sa.Column("extra_metadata", json_type, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("digest", name="uq_artifact_digest"),
    )
    op.create_index("ix_artifact_digest", "artifacts", ["digest"])

    # ============================================================================
    # Agents
    # ============================================================================
    op.create_table(
        "agents",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column(
            "workspace_id",
            uuid_type,
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("agent_version", sa.String(50), nullable=False),
        sa.Column("public_key", sa.Text(), nullable=False),
        sa.Column("key_algorithm", sa.String(50), default="ed25519"),
        sa.Column("trust_state", sa.String(50), default="pending"),
        sa.Column("trust_state_changed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revocation_reason", sa.Text(), nullable=True),
        sa.Column("capabilities", array_type_100, nullable=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("health_status", json_type, nullable=True),
        sa.Column("attestation", json_type, nullable=True),
        sa.Column("device_id", sa.String(255), nullable=True),
        sa.Column("os_info", json_type, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("workspace_id", "name", name="uq_agent_workspace_name"),
    )
    op.create_index("ix_agent_trust_state", "agents", ["trust_state"])
    op.create_index("ix_agent_last_seen", "agents", ["last_seen_at"])

    # ============================================================================
    # Agent Enrollment Tokens
    # ============================================================================
    op.create_table(
        "agent_enrollment_tokens",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column(
            "workspace_id",
            uuid_type,
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("token_hash", sa.String(128), unique=True, nullable=False),
        sa.Column("capabilities", array_type_100, nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_used", sa.Boolean(), default=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "used_by_agent_id",
            uuid_type,
            sa.ForeignKey("agents.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_by_user_id",
            uuid_type,
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("extra_metadata", json_type, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_enrollment_token_expires", "agent_enrollment_tokens", ["expires_at"])

    # ============================================================================
    # Config Blobs
    # ============================================================================
    op.create_table(
        "config_blobs",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column(
            "workspace_id",
            uuid_type,
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("digest", sa.String(128), nullable=False, index=True),
        sa.Column("content", json_type, nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("config_type", sa.String(100), nullable=False),
        sa.Column("schema_version", sa.String(20), default="1.0.0"),
        sa.Column(
            "created_by_user_id",
            uuid_type,
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("workspace_id", "digest", name="uq_configblob_workspace_digest"),
    )
    op.create_index("ix_configblob_digest", "config_blobs", ["digest"])
    op.create_index("ix_configblob_type", "config_blobs", ["config_type"])

    # ============================================================================
    # Deployments
    # ============================================================================
    op.create_table(
        "deployments",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column(
            "workspace_id",
            uuid_type,
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "agent_id",
            uuid_type,
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "artifact_id",
            uuid_type,
            sa.ForeignKey("artifacts.id", ondelete="RESTRICT"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "config_blob_id",
            uuid_type,
            sa.ForeignKey("config_blobs.id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column("state", sa.String(50), default="created"),
        sa.Column("state_changed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("desired_state", sa.String(50), default="deployed"),
        sa.Column("extra_metadata", json_type, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_deployment_agent_state", "deployments", ["agent_id", "state"])

    # ============================================================================
    # Runs
    # ============================================================================
    op.create_table(
        "runs",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column(
            "workspace_id",
            uuid_type,
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "deployment_id",
            uuid_type,
            sa.ForeignKey("deployments.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("state", sa.String(50), default="created"),
        sa.Column("state_changed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stopped_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_paper_trading", sa.Boolean(), default=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("error_code", sa.String(50), nullable=True),
        sa.Column("metrics_summary", json_type, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_run_deployment_state", "runs", ["deployment_id", "state"])

    # ============================================================================
    # Commands
    # ============================================================================
    op.create_table(
        "commands",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column(
            "workspace_id",
            uuid_type,
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("idempotency_key", sa.String(255), nullable=False),
        sa.Column(
            "agent_id",
            uuid_type,
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "deployment_id",
            uuid_type,
            sa.ForeignKey("deployments.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "run_id", uuid_type, sa.ForeignKey("runs.id", ondelete="SET NULL"), nullable=True
        ),
        sa.Column("command_type", sa.String(100), nullable=False),
        sa.Column("payload_ref", sa.String(128), nullable=False),
        sa.Column("change_class", sa.String(50), default="operational"),
        sa.Column("requires_approval", sa.Boolean(), default=False),
        sa.Column("status", sa.String(50), default="pending"),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("result", json_type, nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "workspace_id", "idempotency_key", name="uq_command_workspace_idempotency"
        ),
    )
    op.create_index("ix_command_agent_status", "commands", ["agent_id", "status"])
    op.create_index("ix_command_idempotency", "commands", ["idempotency_key"])

    # ============================================================================
    # Approval Records
    # ============================================================================
    op.create_table(
        "approval_records",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column(
            "workspace_id",
            uuid_type,
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "command_id",
            uuid_type,
            sa.ForeignKey("commands.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("approved", sa.Boolean(), nullable=False),
        sa.Column("approved_by", sa.String(255), nullable=False),
        sa.Column("evidence_hash", sa.String(128), nullable=True),
        sa.Column("attestation", json_type, nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("diff_summary", json_type, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_approval_command", "approval_records", ["command_id"])

    # ============================================================================
    # Telemetry Events
    # ============================================================================
    op.create_table(
        "telemetry_events",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column(
            "workspace_id",
            uuid_type,
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "agent_id",
            uuid_type,
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "run_id",
            uuid_type,
            sa.ForeignKey("runs.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("event_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("telemetry_level", sa.String(50), default="aggregated"),
        sa.Column("payload", json_type, nullable=False),
        sa.Column("redaction_applied", sa.Boolean(), default=True),
        sa.Column("redaction_version", sa.String(20), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_telemetry_agent_time", "telemetry_events", ["agent_id", "event_timestamp"])
    op.create_index("ix_telemetry_type", "telemetry_events", ["event_type"])

    # ============================================================================
    # Alerts
    # ============================================================================
    op.create_table(
        "alerts",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column(
            "workspace_id",
            uuid_type,
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "agent_id",
            uuid_type,
            sa.ForeignKey("agents.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column(
            "run_id", uuid_type, sa.ForeignKey("runs.id", ondelete="SET NULL"), nullable=True
        ),
        sa.Column("alert_type", sa.String(100), nullable=False),
        sa.Column("severity", sa.String(20), default="info"),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("acknowledged", sa.Boolean(), default=False),
        sa.Column("acknowledged_by", sa.String(255), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved", sa.Boolean(), default=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("context", json_type, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_alert_severity", "alerts", ["severity"])
    op.create_index("ix_alert_unresolved", "alerts", ["resolved", "severity"])

    # ============================================================================
    # Data Retention Policies
    # ============================================================================
    op.create_table(
        "data_retention_policies",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column(
            "workspace_id",
            uuid_type,
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("data_type", sa.String(100), nullable=False),
        sa.Column("retention_days", sa.Integer(), nullable=False),
        sa.Column("auto_purge_enabled", sa.Boolean(), default=True),
        sa.Column("last_purge_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dsar_export_enabled", sa.Boolean(), default=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("workspace_id", "data_type", name="uq_retention_workspace_type"),
    )

    # ============================================================================
    # Access Audits
    # ============================================================================
    op.create_table(
        "access_audits",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column(
            "workspace_id",
            uuid_type,
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "user_id", uuid_type, sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True
        ),
        sa.Column(
            "agent_id", uuid_type, sa.ForeignKey("agents.id", ondelete="SET NULL"), nullable=True
        ),
        sa.Column("actor_type", sa.String(50), nullable=False),
        sa.Column("actor_ip", sa.String(50), nullable=True),
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("resource_type", sa.String(100), nullable=False),
        sa.Column("resource_id", sa.String(255), nullable=True),
        sa.Column("is_break_glass", sa.Boolean(), default=False),
        sa.Column("break_glass_reason", sa.Text(), nullable=True),
        sa.Column("break_glass_evidence_hash", sa.String(128), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("request_id", sa.String(255), nullable=True),
        sa.Column("context", json_type, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_audit_actor", "access_audits", ["actor_type", "user_id"])
    op.create_index("ix_audit_resource", "access_audits", ["resource_type", "resource_id"])
    op.create_index("ix_audit_break_glass", "access_audits", ["is_break_glass"])
    op.create_index("ix_audit_time", "access_audits", ["created_at"])


def downgrade() -> None:
    """Drop all tables in reverse order."""
    # Drop tables in reverse order of dependencies
    op.drop_table("access_audits")
    op.drop_table("data_retention_policies")
    op.drop_table("alerts")
    op.drop_table("telemetry_events")
    op.drop_table("approval_records")
    op.drop_table("commands")
    op.drop_table("runs")
    op.drop_table("deployments")
    op.drop_table("config_blobs")
    op.drop_table("agent_enrollment_tokens")
    op.drop_table("agents")
    op.drop_table("artifacts")
    op.drop_table("builds")
    op.drop_table("strategy_versions")
    op.drop_table("strategies")
    op.drop_table("user_roles")
    op.drop_table("users")
    op.drop_table("role_permissions")
    op.drop_table("roles")
    op.drop_table("permissions")
    op.drop_table("workspaces")
    op.drop_table("organizations")
