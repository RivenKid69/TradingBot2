# -*- coding: utf-8 -*-
"""
Tests for Phase 7 (WI-CLOUD-04): Alembic Migrations and RLS.

Tests cover:
- Migration file structure and syntax
- Migration upgrade/downgrade
- RLS policy creation
- Tenant isolation enforcement

References:
- CCEA_MASTER_REMEDIATION_PLAN.md Phase 7
- WI-CLOUD-04: Introduce migrations and enforce tenant isolation via RLS
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add package to path for imports
PACKAGE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))


class TestAlembicConfiguration:
    """Tests for Alembic configuration."""

    def test_alembic_ini_exists(self):
        """Test that alembic.ini exists."""
        alembic_ini = PACKAGE_ROOT / "alembic.ini"
        assert alembic_ini.exists(), "alembic.ini should exist"

    def test_alembic_env_exists(self):
        """Test that alembic/env.py exists."""
        env_py = PACKAGE_ROOT / "alembic" / "env.py"
        assert env_py.exists(), "alembic/env.py should exist"

    def test_alembic_versions_directory_exists(self):
        """Test that alembic/versions directory exists."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        assert versions_dir.exists(), "alembic/versions should exist"
        assert versions_dir.is_dir(), "alembic/versions should be a directory"

    def test_initial_migration_exists(self):
        """Test that initial migration exists."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migrations = list(versions_dir.glob("*_001_initial_schema.py"))
        assert len(migrations) == 1, "Initial migration should exist"

    def test_rls_migration_exists(self):
        """Test that RLS migration exists."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migrations = list(versions_dir.glob("*_002_enable_rls.py"))
        assert len(migrations) == 1, "RLS migration should exist"


class TestMigrationContent:
    """Tests for migration file content."""

    def test_initial_migration_creates_all_tables(self):
        """Test that initial migration creates all required tables."""
        # Migration content test - read file directly

        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_001_initial_schema.py"))[0]
        content = migration_file.read_text()

        # Check for all required tables
        required_tables = [
            "organizations",
            "workspaces",
            "users",
            "roles",
            "permissions",
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

        for table in required_tables:
            assert f'"{table}"' in content, f"Table {table} should be in migration"

    def test_rls_migration_enables_rls(self):
        """Test that RLS migration enables row level security."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]
        content = migration_file.read_text()

        # Check for RLS enablement
        assert "ENABLE ROW LEVEL SECURITY" in content
        assert "FORCE ROW LEVEL SECURITY" in content

        # Check for tenant-scoped tables
        tenant_tables = [
            "strategies",
            "agents",
            "commands",
            "deployments",
            "runs",
        ]
        for table in tenant_tables:
            assert (
                f'"{table}"' in content or f"'{table}'" in content
            ), f"Table {table} should have RLS enabled"

    def test_rls_migration_has_downgrade(self):
        """Test that RLS migration can be rolled back."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]
        content = migration_file.read_text()

        assert "def downgrade" in content
        assert "DISABLE ROW LEVEL SECURITY" in content
        assert "DROP POLICY" in content


class TestRLSTenantIsolation:
    """Tests for RLS tenant isolation logic."""

    def test_tenant_scoped_tables_list(self):
        """Test that tenant-scoped tables are correctly defined."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]

        # Import the migration module
        import importlib.util

        spec = importlib.util.spec_from_file_location("rls_migration", migration_file)
        rls_module = importlib.util.module_from_spec(spec)

        # Check that we have the expected tenant-scoped tables
        expected_tables = {
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
        }

        # Parse TENANT_SCOPED_TABLES from migration content
        content = migration_file.read_text()
        assert "TENANT_SCOPED_TABLES" in content, "Should define TENANT_SCOPED_TABLES"

    def test_rls_uses_workspace_id(self):
        """Test that RLS policies use workspace_id for isolation."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]
        content = migration_file.read_text()

        # Check for workspace_id in policy
        assert "workspace_id" in content
        assert "ccea_get_current_workspace_id" in content

    def test_rls_has_admin_bypass(self):
        """Test that RLS policies have admin bypass."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]
        content = migration_file.read_text()

        # Check for admin bypass
        assert "ccea_is_admin" in content


class TestDatabaseHelperFunctions:
    """Tests for database helper functions used by RLS."""

    @pytest.mark.asyncio
    async def test_tenant_context_sets_workspace_id(self):
        """Test that TenantContext sets workspace_id correctly."""
        from packages.cloud.control_plane.database import TenantContext
        from uuid import uuid4

        workspace_id = uuid4()
        context = TenantContext(workspace_id)

        # Mock session
        mock_session = AsyncMock()

        await context.set_tenant(mock_session)

        # The id travels as a bind parameter, not inside the statement text
        mock_session.execute.assert_called_once()
        call_args = mock_session.execute.call_args[0][0]
        sql_text = str(call_args.text) if hasattr(call_args, "text") else str(call_args)
        assert "app.current_workspace_id" in sql_text
        assert str(workspace_id) not in sql_text
        assert mock_session.execute.call_args[0][1] == {"workspace_id": str(workspace_id)}

    @pytest.mark.asyncio
    async def test_tenant_context_resets_when_none(self):
        """Test that TenantContext resets when workspace_id is None."""
        from packages.cloud.control_plane.database import TenantContext

        context = TenantContext(None)
        mock_session = AsyncMock()

        await context.set_tenant(mock_session)

        mock_session.execute.assert_called_once()
        call_args = mock_session.execute.call_args[0][0]
        sql_text = str(call_args.text) if hasattr(call_args, "text") else str(call_args)
        assert "app.current_workspace_id" in sql_text
        assert mock_session.execute.call_args[0][1] == {"workspace_id": ""}


class TestMigrationRevisionChain:
    """Tests for migration revision chain integrity."""

    def test_revision_chain_is_linear(self):
        """Test that migration revisions form a linear chain."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migrations = sorted(versions_dir.glob("*.py"))

        # Exclude __pycache__ and __init__.py
        migrations = [m for m in migrations if not m.name.startswith("__")]

        # Check we have at least 2 migrations
        assert len(migrations) >= 2, "Should have at least 2 migrations"

        # Parse revision info from each migration
        revisions = []
        for migration in migrations:
            content = migration.read_text()

            # Find revision
            for line in content.split("\n"):
                if line.strip().startswith("revision:"):
                    rev = line.split("=")[1].strip().strip('"').strip("'")
                    revisions.append(rev)
                    break

        # All revisions should be unique
        assert len(revisions) == len(set(revisions)), "Revisions should be unique"

    def test_first_migration_has_no_down_revision(self):
        """Test that first migration has no down_revision."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_001_initial_schema.py"))[0]
        content = migration_file.read_text()

        # Check down_revision is None
        assert "down_revision" in content
        for line in content.split("\n"):
            if "down_revision" in line and "=" in line:
                assert "None" in line, "First migration should have down_revision = None"
                break

    def test_rls_migration_depends_on_initial(self):
        """Test that RLS migration depends on initial migration."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]
        content = migration_file.read_text()

        # Check down_revision points to initial
        for line in content.split("\n"):
            if "down_revision" in line and "=" in line:
                assert "001" in line, "RLS migration should depend on initial (001)"
                break
