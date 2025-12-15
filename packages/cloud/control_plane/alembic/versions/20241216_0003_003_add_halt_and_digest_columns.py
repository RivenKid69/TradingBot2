"""Add halt_* columns to deployments/runs and digest columns to approval_records.

Revision ID: 003
Revises: 002
Create Date: 2024-12-16

Design Doc Compliance:
    - Section 6.2: Deployment/Run fields for halt tracking
    - Section 9.4: Kill switch halt reasons and details
    - Section 11.1-11.2: State machine with HALTED state
    - Section 12.2: Immutable blob references for audit trail

This migration adds:
1. deployments.halt_reason - HaltReason enum value
2. deployments.halt_details - JSON with full halt context
3. deployments.halted_at - Timestamp when halt occurred
4. runs.halt_reason - HaltReason enum value
5. runs.halt_details - JSON with full halt context
6. runs.halted_at - Timestamp when halt occurred
7. approval_records.config_blob_digest - Config blob at approval time
8. approval_records.manifest_digest - Manifest digest at approval time
9. approval_records.previous_state_digest - State before change
10. approval_records.new_state_digest - State after change
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _is_postgresql() -> bool:
    """Check if we're running on PostgreSQL."""
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    """Add halt_* and digest columns per Design Doc."""

    # Use JSONB for PostgreSQL, JSON for others (SQLite)
    json_type = postgresql.JSONB() if _is_postgresql() else sa.JSON()

    # ============================================================================
    # Deployments - Add halt tracking columns (Design Doc 9.4, 11.1)
    # ============================================================================
    op.add_column(
        "deployments",
        sa.Column(
            "halt_reason",
            sa.String(50),
            nullable=True,
            comment="HaltReason enum value per Design Doc 9.4"
        )
    )
    op.add_column(
        "deployments",
        sa.Column(
            "halt_details",
            json_type,
            nullable=True,
            comment="Full halt context: trigger_source, metrics, recovery_action"
        )
    )
    op.add_column(
        "deployments",
        sa.Column(
            "halted_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Timestamp when halt occurred"
        )
    )

    # Index for querying halted deployments
    op.create_index(
        "ix_deployment_halt_reason",
        "deployments",
        ["halt_reason"],
        postgresql_where=sa.text("halt_reason IS NOT NULL")
    )

    # ============================================================================
    # Runs - Add halt tracking columns (Design Doc 9.4, 11.2)
    # ============================================================================
    op.add_column(
        "runs",
        sa.Column(
            "halt_reason",
            sa.String(50),
            nullable=True,
            comment="HaltReason enum value per Design Doc 9.4"
        )
    )
    op.add_column(
        "runs",
        sa.Column(
            "halt_details",
            json_type,
            nullable=True,
            comment="Full halt context: trigger_source, metrics, recovery_action"
        )
    )
    op.add_column(
        "runs",
        sa.Column(
            "halted_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Timestamp when halt occurred"
        )
    )

    # Index for querying halted runs
    op.create_index(
        "ix_run_halt_reason",
        "runs",
        ["halt_reason"],
        postgresql_where=sa.text("halt_reason IS NOT NULL")
    )

    # ============================================================================
    # Approval Records - Add immutable digest columns (Design Doc 12.2)
    # ============================================================================
    op.add_column(
        "approval_records",
        sa.Column(
            "config_blob_digest",
            sa.String(128),
            nullable=True,
            comment="ConfigBlob digest at approval time for audit verification"
        )
    )
    op.add_column(
        "approval_records",
        sa.Column(
            "manifest_digest",
            sa.String(128),
            nullable=True,
            comment="Artifact manifest digest at approval time"
        )
    )
    op.add_column(
        "approval_records",
        sa.Column(
            "previous_state_digest",
            sa.String(128),
            nullable=True,
            comment="Digest of state before change (for diff)"
        )
    )
    op.add_column(
        "approval_records",
        sa.Column(
            "new_state_digest",
            sa.String(128),
            nullable=True,
            comment="Digest of state after change (for diff)"
        )
    )

    # Indexes for digest lookups
    op.create_index(
        "ix_approval_config_digest",
        "approval_records",
        ["config_blob_digest"],
        postgresql_where=sa.text("config_blob_digest IS NOT NULL")
    )
    op.create_index(
        "ix_approval_manifest_digest",
        "approval_records",
        ["manifest_digest"],
        postgresql_where=sa.text("manifest_digest IS NOT NULL")
    )

    # ============================================================================
    # Agents - Add key rotation tracking (Design Doc 15.2)
    # ============================================================================
    op.add_column(
        "agents",
        sa.Column(
            "key_version",
            sa.Integer(),
            nullable=True,
            default=1,
            server_default="1",
            comment="Key version for rotation tracking"
        )
    )
    op.add_column(
        "agents",
        sa.Column(
            "previous_public_key",
            sa.Text(),
            nullable=True,
            comment="Previous public key for grace period during rotation"
        )
    )
    op.add_column(
        "agents",
        sa.Column(
            "key_rotated_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Timestamp of last key rotation"
        )
    )
    op.add_column(
        "agents",
        sa.Column(
            "key_rotation_grace_until",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Until when previous key is still valid"
        )
    )

    # ============================================================================
    # Users - Add MFA backup codes and lockout (Design Doc 4.1)
    # ============================================================================
    op.add_column(
        "users",
        sa.Column(
            "mfa_backup_codes_hash",
            sa.Text(),
            nullable=True,
            comment="Hashed backup codes for MFA recovery"
        )
    )
    op.add_column(
        "users",
        sa.Column(
            "mfa_verified_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="When MFA was successfully verified first time"
        )
    )
    op.add_column(
        "users",
        sa.Column(
            "failed_mfa_attempts",
            sa.Integer(),
            nullable=True,
            default=0,
            server_default="0",
            comment="Failed MFA attempts for lockout"
        )
    )
    op.add_column(
        "users",
        sa.Column(
            "mfa_locked_until",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="MFA lockout expiry time"
        )
    )


def downgrade() -> None:
    """Remove added columns."""
    # Users MFA columns
    op.drop_column("users", "mfa_locked_until")
    op.drop_column("users", "failed_mfa_attempts")
    op.drop_column("users", "mfa_verified_at")
    op.drop_column("users", "mfa_backup_codes_hash")

    # Agents key rotation columns
    op.drop_column("agents", "key_rotation_grace_until")
    op.drop_column("agents", "key_rotated_at")
    op.drop_column("agents", "previous_public_key")
    op.drop_column("agents", "key_version")

    # Approval records digest columns
    op.drop_index("ix_approval_manifest_digest", table_name="approval_records")
    op.drop_index("ix_approval_config_digest", table_name="approval_records")
    op.drop_column("approval_records", "new_state_digest")
    op.drop_column("approval_records", "previous_state_digest")
    op.drop_column("approval_records", "manifest_digest")
    op.drop_column("approval_records", "config_blob_digest")

    # Runs halt columns
    op.drop_index("ix_run_halt_reason", table_name="runs")
    op.drop_column("runs", "halted_at")
    op.drop_column("runs", "halt_details")
    op.drop_column("runs", "halt_reason")

    # Deployments halt columns
    op.drop_index("ix_deployment_halt_reason", table_name="deployments")
    op.drop_column("deployments", "halted_at")
    op.drop_column("deployments", "halt_details")
    op.drop_column("deployments", "halt_reason")
