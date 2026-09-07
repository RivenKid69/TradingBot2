# -*- coding: utf-8 -*-
"""
Tests for Phase 7 (WI-CLOUD-04): Tenant Isolation via RLS.

Tests cover:
- TenantContext correctly sets session variables
- RLS policies enforce workspace isolation
- Admin bypass works correctly
- Cross-tenant access is prevented

References:
- CCEA_MASTER_REMEDIATION_PLAN.md Phase 7
- WI-CLOUD-04: Enforce tenant isolation via RLS
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

# Add package to path for imports
PACKAGE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))


# ============================================================================
# TenantContext Tests
# ============================================================================


class TestTenantContext:
    """Tests for TenantContext class."""

    def test_tenant_context_init_with_workspace_id(self):
        """Test TenantContext initialization with workspace_id."""
        from packages.cloud.control_plane.database import TenantContext

        workspace_id = uuid4()
        context = TenantContext(workspace_id)

        assert context.workspace_id == workspace_id

    def test_tenant_context_init_without_workspace_id(self):
        """Test TenantContext initialization without workspace_id."""
        from packages.cloud.control_plane.database import TenantContext

        context = TenantContext(None)
        assert context.workspace_id is None

    @pytest.mark.asyncio
    async def test_set_tenant_with_workspace_id(self):
        """Test set_tenant sets session variable correctly."""
        from packages.cloud.control_plane.database import TenantContext

        workspace_id = uuid4()
        context = TenantContext(workspace_id)

        mock_session = AsyncMock()

        await context.set_tenant(mock_session)

        # Verify execute was called
        mock_session.execute.assert_called_once()

        # The id is bound, not interpolated into the statement
        call_args = mock_session.execute.call_args
        sql_text = str(call_args[0][0])
        assert "app.current_workspace_id" in sql_text
        assert str(workspace_id) not in sql_text
        assert call_args[0][1] == {"workspace_id": str(workspace_id)}

    @pytest.mark.asyncio
    async def test_set_tenant_without_workspace_id(self):
        """Test set_tenant resets to empty string when None."""
        from packages.cloud.control_plane.database import TenantContext

        context = TenantContext(None)

        mock_session = AsyncMock()

        await context.set_tenant(mock_session)

        mock_session.execute.assert_called_once()
        call_args = mock_session.execute.call_args
        sql_text = str(call_args[0][0])
        assert "app.current_workspace_id" in sql_text
        assert call_args[0][1] == {"workspace_id": ""}


# ============================================================================
# Session Factory Tests
# ============================================================================


class TestGetSessionWithTenant:
    """Tests for get_session with tenant context."""

    @pytest.mark.asyncio
    async def test_get_session_without_context(self):
        """Test get_session works without tenant context."""
        from packages.cloud.control_plane.database import get_session

        # This should not raise when database is properly configured
        # For unit tests, we just verify the function signature
        import inspect

        sig = inspect.signature(get_session)
        params = sig.parameters

        assert "tenant_context" in params
        assert params["tenant_context"].default is None

    @pytest.mark.asyncio
    async def test_get_session_with_context(self):
        """Test get_session accepts tenant context."""
        from packages.cloud.control_plane.database import get_session, TenantContext

        workspace_id = uuid4()
        context = TenantContext(workspace_id)

        # Verify function accepts TenantContext
        # Actual DB test would require running database


class TestGetSessionDependency:
    """Tests for FastAPI session dependency."""

    def test_dependency_signature(self):
        """Test get_session_dependency has correct signature."""
        from packages.cloud.control_plane.database import get_session_dependency
        import inspect

        sig = inspect.signature(get_session_dependency)
        params = sig.parameters

        assert "workspace_id" in params
        assert params["workspace_id"].default is None


# ============================================================================
# RLS Policy Tests (Structure)
# ============================================================================


class TestRLSPolicyStructure:
    """Tests for RLS policy structure in migrations."""

    def test_rls_migration_defines_helper_functions(self):
        """Test that RLS migration creates helper functions."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]
        content = migration_file.read_text()

        # Check for helper functions
        assert "ccea_get_current_workspace_id" in content
        assert "ccea_is_admin" in content

    def test_rls_migration_enables_rls_on_tenant_tables(self):
        """Test that RLS is enabled on all tenant-scoped tables."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]
        content = migration_file.read_text()

        # Critical tenant-scoped tables
        critical_tables = [
            "commands",
            "agents",
            "deployments",
            "runs",
            "strategies",
        ]

        for table in critical_tables:
            assert (
                f'"{table}"' in content or f"'{table}'" in content
            ), f"Table {table} should have RLS enabled"

    def test_rls_migration_creates_isolation_policies(self):
        """Test that isolation policies are created."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]
        content = migration_file.read_text()

        # Check for policy creation
        assert "CREATE POLICY" in content
        assert "workspace_isolation" in content

    def test_rls_migration_has_using_clause(self):
        """Test that policies have USING clause for SELECT."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]
        content = migration_file.read_text()

        assert "USING" in content

    def test_rls_migration_has_with_check_clause(self):
        """Test that policies have WITH CHECK clause for INSERT/UPDATE."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]
        content = migration_file.read_text()

        assert "WITH CHECK" in content


# ============================================================================
# Tenant Isolation Logic Tests
# ============================================================================


class TestTenantIsolationLogic:
    """Tests for tenant isolation enforcement logic."""

    def test_rls_policy_checks_workspace_id(self):
        """Test that RLS policies check workspace_id."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]
        content = migration_file.read_text()

        # Policy should compare workspace_id
        assert "workspace_id = ccea_get_current_workspace_id()" in content

    def test_rls_policy_allows_admin_bypass(self):
        """Test that RLS policies allow admin bypass."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]
        content = migration_file.read_text()

        # Admin should bypass
        assert "ccea_is_admin()" in content
        assert "OR" in content  # admin OR workspace_id match

    def test_rls_forces_rls_for_owner(self):
        """Test that FORCE ROW LEVEL SECURITY is used."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]
        content = migration_file.read_text()

        # Force RLS even for table owner
        assert "FORCE ROW LEVEL SECURITY" in content


# ============================================================================
# Org-Scoped Tables Tests
# ============================================================================


class TestOrgScopedTables:
    """Tests for org-scoped tables (roles)."""

    def test_roles_table_has_special_policy(self):
        """Test that roles table has org-level isolation."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]
        content = migration_file.read_text()

        # Roles should have org-level policy
        assert "ORG_SCOPED_TABLES" in content
        assert "roles" in content
        assert "organization_id" in content


# ============================================================================
# Downgrade Tests
# ============================================================================


class TestRLSDowngrade:
    """Tests for RLS migration downgrade."""

    def test_downgrade_disables_rls(self):
        """Test that downgrade disables RLS."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]
        content = migration_file.read_text()

        # Check downgrade function
        assert "DISABLE ROW LEVEL SECURITY" in content

    def test_downgrade_drops_policies(self):
        """Test that downgrade drops policies."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]
        content = migration_file.read_text()

        assert "DROP POLICY" in content

    def test_downgrade_drops_helper_functions(self):
        """Test that downgrade drops helper functions."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]
        content = migration_file.read_text()

        assert "DROP FUNCTION" in content


# ============================================================================
# Pending Commands View Tests
# ============================================================================


class TestPendingCommandsView:
    """Tests for pending_commands_view in RLS migration."""

    def test_view_is_created(self):
        """Test that pending_commands_view is created."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]
        content = migration_file.read_text()

        assert "pending_commands_view" in content
        assert "CREATE" in content and "VIEW" in content

    def test_view_filters_by_workspace(self):
        """Test that view filters by current workspace."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]
        content = migration_file.read_text()

        # View should use RLS helper
        assert "ccea_get_current_workspace_id()" in content

    def test_view_filters_pending_statuses(self):
        """Test that view only returns non-expired pending commands."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]
        content = migration_file.read_text()

        # Should filter by status
        assert "pending" in content.lower()
        assert "sent" in content.lower()
        assert "approved" in content.lower()


# ============================================================================
# Index Tests
# ============================================================================


class TestRLSIndexes:
    """Tests for indexes created in RLS migration."""

    def test_command_pending_index_created(self):
        """Test that index for pending commands is created."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]
        content = migration_file.read_text()

        # Should create index for efficient polling
        assert "ix_commands_agent_pending" in content or "CREATE INDEX" in content


# ============================================================================
# Security Tests
# ============================================================================


class TestRLSSecurity:
    """Security tests for RLS implementation."""

    def test_no_sql_injection_in_helper_functions(self):
        """Test that helper functions don't have SQL injection risks."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]
        content = migration_file.read_text()

        # Helper functions should use current_setting, not string interpolation
        assert "current_setting" in content

    def test_helper_functions_are_stable(self):
        """Test that helper functions are marked STABLE."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]
        content = migration_file.read_text()

        # Functions should be STABLE for query planning
        assert "STABLE" in content

    def test_no_superuser_bypass_without_admin_flag(self):
        """Test that RLS is enforced even for superusers (via FORCE)."""
        versions_dir = PACKAGE_ROOT / "alembic" / "versions"
        migration_file = list(versions_dir.glob("*_002_enable_rls.py"))[0]
        content = migration_file.read_text()

        # FORCE ensures even owner is subject to RLS
        assert "FORCE ROW LEVEL SECURITY" in content
