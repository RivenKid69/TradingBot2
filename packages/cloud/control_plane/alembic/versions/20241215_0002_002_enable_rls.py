"""Enable Row Level Security for tenant isolation.

Revision ID: 002
Revises: 001
Create Date: 2024-12-15

Phase 7 (WI-CLOUD-04): Implements PostgreSQL Row Level Security policies
for multi-tenant data isolation. Each tenant can only access their own data.

Security Model:
- All tenant-scoped tables have RLS enabled
- Policies use app.current_workspace_id session variable
- Application MUST set this variable before queries
- Superusers bypass RLS for administrative operations

Reference:
- Design Doc CCEA Cloud.txt: Multi-tenant isolation
- SOC2/ISO27001: Data segregation requirements
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Tables that require RLS (tenant-scoped)
TENANT_SCOPED_TABLES = [
    "strategies",
    "strategy_versions",
    "builds",
    "artifacts",
    "agents",
    "agent_enrollment_tokens",
    "deployments",
    "runs",
    "commands",
    "approval_records",
    "config_blobs",
    "telemetry_events",
    "alerts",
    "data_retention_policies",
    "access_audits",
]

# Tables with special policies (org-scoped or cross-tenant)
ORG_SCOPED_TABLES = [
    "roles",  # Roles can be org-wide or workspace-scoped
]


def _is_postgresql() -> bool:
    """Check if we're running on PostgreSQL."""
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    """Enable RLS on tenant-scoped tables and create isolation policies."""

    # RLS is only supported on PostgreSQL
    if not _is_postgresql():
        # For SQLite and other databases, skip RLS
        # Tenant isolation must be enforced at application level
        return

    # Create helper function for checking workspace access
    op.execute(
        """
        CREATE OR REPLACE FUNCTION ccea_get_current_workspace_id()
        RETURNS uuid
        LANGUAGE sql
        STABLE
        AS $$
            SELECT NULLIF(current_setting('app.current_workspace_id', true), '')::uuid;
        $$;
    """
    )

    # Create helper function to check if user is admin/superuser
    op.execute(
        """
        CREATE OR REPLACE FUNCTION ccea_is_admin()
        RETURNS boolean
        LANGUAGE sql
        STABLE
        AS $$
            SELECT COALESCE(current_setting('app.is_admin', true)::boolean, false);
        $$;
    """
    )

    # Enable RLS on all tenant-scoped tables
    for table in TENANT_SCOPED_TABLES:
        # Enable RLS
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;")

        # Force RLS even for table owner (security best practice)
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY;")

        # Create policy for workspace isolation
        # Policy name: {table}_workspace_isolation
        op.execute(
            f"""
            CREATE POLICY {table}_workspace_isolation ON {table}
            FOR ALL
            USING (
                ccea_is_admin()
                OR workspace_id = ccea_get_current_workspace_id()
            )
            WITH CHECK (
                ccea_is_admin()
                OR workspace_id = ccea_get_current_workspace_id()
            );
        """
        )

    # Special handling for org-scoped tables (roles)
    for table in ORG_SCOPED_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY;")

        # Roles need org-level isolation, with optional workspace scoping
        op.execute(
            f"""
            CREATE POLICY {table}_org_isolation ON {table}
            FOR ALL
            USING (
                ccea_is_admin()
                OR organization_id = (
                    SELECT organization_id
                    FROM workspaces
                    WHERE id = ccea_get_current_workspace_id()
                )
            )
            WITH CHECK (
                ccea_is_admin()
                OR organization_id = (
                    SELECT organization_id
                    FROM workspaces
                    WHERE id = ccea_get_current_workspace_id()
                )
            );
        """
        )

    # Create helper view for tenant-safe command polling
    # This view pre-filters commands by workspace for agent polling
    op.execute(
        """
        CREATE OR REPLACE VIEW pending_commands_view AS
        SELECT
            c.id,
            c.workspace_id,
            c.idempotency_key,
            c.agent_id,
            c.deployment_id,
            c.run_id,
            c.command_type,
            c.payload_ref,
            c.change_class,
            c.requires_approval,
            c.status,
            c.expires_at,
            c.created_at
        FROM commands c
        WHERE c.status IN ('pending', 'sent', 'approved')
        AND (c.expires_at IS NULL OR c.expires_at > NOW())
        AND c.workspace_id = ccea_get_current_workspace_id();
    """
    )

    # Create index on commands for efficient agent polling
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_commands_agent_pending
        ON commands (agent_id, status)
        WHERE status IN ('pending', 'sent', 'approved');
    """
    )


def downgrade() -> None:
    """Disable RLS and remove policies."""

    if not _is_postgresql():
        return

    # Drop view
    op.execute("DROP VIEW IF EXISTS pending_commands_view;")

    # Drop index
    op.execute("DROP INDEX IF EXISTS ix_commands_agent_pending;")

    # Disable RLS and drop policies for org-scoped tables
    for table in ORG_SCOPED_TABLES:
        op.execute(f"DROP POLICY IF EXISTS {table}_org_isolation ON {table};")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY;")

    # Disable RLS and drop policies for tenant-scoped tables
    for table in TENANT_SCOPED_TABLES:
        op.execute(f"DROP POLICY IF EXISTS {table}_workspace_isolation ON {table};")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY;")

    # Drop helper functions
    op.execute("DROP FUNCTION IF EXISTS ccea_is_admin();")
    op.execute("DROP FUNCTION IF EXISTS ccea_get_current_workspace_id();")
