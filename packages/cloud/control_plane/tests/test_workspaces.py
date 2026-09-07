# -*- coding: utf-8 -*-
"""Tests for Workspaces Router."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import User

from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import (
    Agent,
    Artifact,
    Build,
    Deployment,
    DeploymentState,
    Organization,
    Strategy,
    StrategyVersion,
    Workspace,
)

pytestmark = pytest.mark.asyncio


class TestListWorkspaces:
    """Tests for GET /workspaces endpoint."""

    async def test_list_workspaces_user(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_workspace: Workspace,
    ) -> None:
        """User can list workspaces in their organization."""
        response = await client.get(
            "/api/v1/workspaces",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data

    async def test_list_workspaces_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_workspace: Workspace,
    ) -> None:
        """Superuser can list all workspaces."""
        response = await client.get(
            "/api/v1/workspaces",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1

    async def test_list_workspaces_superuser_filter_by_org(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_organization: Organization,
        sample_workspace: Workspace,
    ) -> None:
        """Superuser can filter workspaces by organization."""
        response = await client.get(
            f"/api/v1/workspaces?organization_id={sample_organization.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        # All items should belong to the specified org
        for item in data["items"]:
            assert item["organization_id"] == str(sample_organization.id)

    async def test_list_workspaces_pagination(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
    ) -> None:
        """Pagination works correctly."""
        # Create organization with multiple workspaces
        org = Organization(name="ws-list-org", display_name="WS List Org")
        db_session.add(org)
        await db_session.commit()
        await db_session.refresh(org)

        for i in range(5):
            ws = Workspace(name=f"ws-page-{i}", organization_id=org.id)
            db_session.add(ws)
        await db_session.commit()

        response = await client.get(
            f"/api/v1/workspaces?organization_id={org.id}&page=1&page_size=2",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 2
        assert len(data["items"]) <= 2

    async def test_list_workspaces_includes_counts(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
    ) -> None:
        """Workspace response includes agent and deployment counts."""
        # Create workspace with agents and deployments
        org = Organization(name="ws-counts-org", display_name="Counts Org")
        db_session.add(org)
        await db_session.commit()
        await db_session.refresh(org)

        ws = Workspace(name="ws-with-counts", organization_id=org.id)
        db_session.add(ws)
        await db_session.commit()
        await db_session.refresh(ws)

        # Add agent
        agent = Agent(
            name="test-agent",
            workspace_id=ws.id,
            public_key="a" * 64,
            agent_version="1.0.0",
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        # Add strategy first (for deployment)
        strategy = Strategy(
            name="test-strategy",
            workspace_id=ws.id,
            description="Test strategy for workspace counts",
        )
        db_session.add(strategy)
        await db_session.commit()
        await db_session.refresh(strategy)

        # Create strategy version
        strategy_version = StrategyVersion(
            workspace_id=ws.id,  # Required field
            strategy_id=strategy.id,
            version="1.0.0",
            git_sha="sha256:wstest123",
        )
        db_session.add(strategy_version)
        await db_session.commit()
        await db_session.refresh(strategy_version)

        # Create build
        build = Build(
            workspace_id=ws.id,  # Required field
            strategy_version_id=strategy_version.id,
            build_number=1,
            status="completed",
        )
        db_session.add(build)
        await db_session.commit()
        await db_session.refresh(build)

        # Create artifact
        artifact = Artifact(
            workspace_id=ws.id,  # Required field
            build_id=build.id,
            name="ws-test.whl",
            format="wheel",
            digest="sha256:wsartifact123",
            size_bytes=256000,
        )
        db_session.add(artifact)
        await db_session.commit()
        await db_session.refresh(artifact)

        # Add deployment
        deployment = Deployment(
            workspace_id=ws.id,
            agent_id=agent.id,
            artifact_id=artifact.id,
            state=DeploymentState.DEPLOYED.value,
        )
        db_session.add(deployment)
        await db_session.commit()

        response = await client.get(
            f"/api/v1/workspaces?organization_id={org.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()

        # Find our workspace
        found_ws = None
        for item in data["items"]:
            if item["name"] == "ws-with-counts":
                found_ws = item
                break

        assert found_ws is not None
        assert found_ws["agent_count"] == 1
        assert found_ws["deployment_count"] == 1

    async def test_list_workspaces_unauthenticated(
        self,
        client: AsyncClient,
    ) -> None:
        """Unauthenticated request returns 401."""
        response = await client.get("/api/v1/workspaces")
        assert response.status_code == 401


class TestCreateWorkspace:
    """Tests for POST /workspaces endpoint."""

    async def test_create_workspace_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_organization: Organization,
    ) -> None:
        """Superuser can create a workspace."""
        response = await client.post(
            "/api/v1/workspaces",
            headers=superuser_headers,
            json={
                "name": "new-workspace",
                "display_name": "New Workspace",
                "organization_id": str(sample_organization.id),
                "settings": {"key": "value"},
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "new-workspace"
        assert data["display_name"] == "New Workspace"
        assert data["organization_id"] == str(sample_organization.id)
        assert data["settings"] == {"key": "value"}
        assert data["is_active"] is True
        assert data["agent_count"] == 0
        assert data["deployment_count"] == 0

    async def test_create_workspace_with_permission(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        org_id,
        sample_user: "User",  # Ensure user exists in DB
    ) -> None:
        """User with workspace:create permission can create workspace."""
        from ..models import Permission, Role, User
        from ..routers.auth import create_access_token
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        # Reload user in current session to avoid greenlet issues
        result = await db_session.execute(
            select(User).where(User.id == sample_user.id).options(selectinload(User.roles))
        )
        user = result.scalar_one()

        # Create permission and role
        perm = Permission(name="workspace:create", description="Create WS")
        db_session.add(perm)
        await db_session.commit()
        await db_session.refresh(perm)

        role = Role(
            name="ws-creator-test",
            description="WS Creator",
            organization_id=org_id,
        )
        role.permissions.append(perm)
        db_session.add(role)
        await db_session.commit()
        await db_session.refresh(role)

        user.roles.append(role)
        await db_session.commit()

        token = create_access_token(
            user_id=user.id,
            email=user.email,
            org_id=org_id,
            workspace_id=None,
            is_superuser=False,
            permissions=["workspace:create"],
        )

        response = await client.post(
            "/api/v1/workspaces",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "name": "perm-workspace",
                "organization_id": str(org_id),
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "perm-workspace"

    async def test_create_workspace_without_permission_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        org_id,
    ) -> None:
        """User without permission cannot create workspace."""
        response = await client.post(
            "/api/v1/workspaces",
            headers=auth_headers,
            json={
                "name": "forbidden-ws",
                "organization_id": str(org_id),
            },
        )

        assert response.status_code == 403
        assert "workspace:create" in response.json()["detail"]

    async def test_create_workspace_other_org_forbidden(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        org_id,
        sample_user: "User",  # Ensure user exists in DB
    ) -> None:
        """Cannot create workspace in another organization."""
        from ..models import Permission, Role, User
        from ..routers.auth import create_access_token
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        # Create another org
        other_org = Organization(name="other-org-ws", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        # Reload user in current session to avoid greenlet issues
        result = await db_session.execute(
            select(User).where(User.id == sample_user.id).options(selectinload(User.roles))
        )
        user = result.scalar_one()

        perm = Permission(name="workspace:create2", description="Create WS 2")
        db_session.add(perm)
        await db_session.commit()
        await db_session.refresh(perm)

        role = Role(
            name="ws-creator-test2",
            description="WS Creator 2",
            organization_id=org_id,
        )
        role.permissions.append(perm)
        db_session.add(role)
        await db_session.commit()
        await db_session.refresh(role)

        user.roles.append(role)
        await db_session.commit()

        token = create_access_token(
            user_id=user.id,
            email=user.email,
            org_id=org_id,
            workspace_id=None,
            is_superuser=False,
            permissions=["workspace:create"],
        )

        response = await client.post(
            "/api/v1/workspaces",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "name": "cross-org-ws",
                "organization_id": str(other_org.id),
            },
        )

        assert response.status_code == 403
        assert "another organization" in response.json()["detail"]

    async def test_create_workspace_org_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Cannot create workspace in non-existent organization."""
        response = await client.post(
            "/api/v1/workspaces",
            headers=superuser_headers,
            json={
                "name": "orphan-ws",
                "organization_id": str(uuid4()),
            },
        )

        assert response.status_code == 404
        assert "Organization not found" in response.json()["detail"]

    async def test_create_workspace_duplicate_name(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_organization: Organization,
        sample_workspace: Workspace,
    ) -> None:
        """Cannot create workspace with duplicate name in same org."""
        response = await client.post(
            "/api/v1/workspaces",
            headers=superuser_headers,
            json={
                "name": sample_workspace.name,
                "organization_id": str(sample_organization.id),
            },
        )

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    async def test_create_workspace_same_name_different_org(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_workspace: Workspace,
    ) -> None:
        """Can create workspace with same name in different org."""
        other_org = Organization(name="other-org-samename", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        response = await client.post(
            "/api/v1/workspaces",
            headers=superuser_headers,
            json={
                "name": sample_workspace.name,
                "organization_id": str(other_org.id),
            },
        )

        assert response.status_code == 201

    async def test_create_workspace_empty_name(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_organization: Organization,
    ) -> None:
        """Cannot create workspace with empty name."""
        response = await client.post(
            "/api/v1/workspaces",
            headers=superuser_headers,
            json={
                "name": "",
                "organization_id": str(sample_organization.id),
            },
        )

        assert response.status_code == 422


class TestGetWorkspace:
    """Tests for GET /workspaces/{workspace_id} endpoint."""

    async def test_get_workspace_user(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_workspace: Workspace,
    ) -> None:
        """User can get workspace in their organization."""
        response = await client.get(
            f"/api/v1/workspaces/{sample_workspace.id}",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_workspace.id)
        assert data["name"] == sample_workspace.name
        assert "agent_count" in data
        assert "deployment_count" in data

    async def test_get_workspace_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_workspace: Workspace,
    ) -> None:
        """Superuser can get any workspace."""
        response = await client.get(
            f"/api/v1/workspaces/{sample_workspace.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_workspace.id)

    async def test_get_workspace_other_org_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
    ) -> None:
        """Cannot get workspace from another organization."""
        other_org = Organization(name="other-org-getws", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        other_ws = Workspace(name="other-ws", organization_id=other_org.id)
        db_session.add(other_ws)
        await db_session.commit()
        await db_session.refresh(other_ws)

        response = await client.get(
            f"/api/v1/workspaces/{other_ws.id}",
            headers=auth_headers,
        )

        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]

    async def test_get_workspace_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Non-existent workspace returns 404."""
        response = await client.get(
            f"/api/v1/workspaces/{uuid4()}",
            headers=superuser_headers,
        )

        assert response.status_code == 404
        assert "Workspace not found" in response.json()["detail"]

    async def test_get_workspace_unauthenticated(
        self,
        client: AsyncClient,
        sample_workspace: Workspace,
    ) -> None:
        """Unauthenticated request returns 401."""
        response = await client.get(
            f"/api/v1/workspaces/{sample_workspace.id}",
        )

        assert response.status_code == 401


class TestUpdateWorkspace:
    """Tests for PATCH /workspaces/{workspace_id} endpoint."""

    async def test_update_workspace_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_workspace: Workspace,
    ) -> None:
        """Superuser can update any workspace."""
        response = await client.patch(
            f"/api/v1/workspaces/{sample_workspace.id}",
            headers=superuser_headers,
            json={
                "display_name": "Updated Workspace",
                "settings": {"updated": True},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["display_name"] == "Updated Workspace"
        assert data["settings"] == {"updated": True}

    async def test_update_workspace_with_permission(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        org_id,
        sample_user: "User",  # Ensure user exists in DB
        workspace_id,
    ) -> None:
        """User with workspace:write permission can update workspace."""
        from ..models import Permission, Role, User
        from ..routers.auth import create_access_token
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        # Reload user in current session to avoid greenlet issues
        result = await db_session.execute(
            select(User).where(User.id == sample_user.id).options(selectinload(User.roles))
        )
        user = result.scalar_one()

        perm = Permission(name="workspace:write", description="Write WS")
        db_session.add(perm)
        await db_session.commit()
        await db_session.refresh(perm)

        role = Role(
            name="ws-writer-test",
            description="WS Writer",
            organization_id=org_id,
        )
        role.permissions.append(perm)
        db_session.add(role)
        await db_session.commit()
        await db_session.refresh(role)

        user.roles.append(role)
        await db_session.commit()

        token = create_access_token(
            user_id=user.id,
            email=user.email,
            org_id=org_id,
            workspace_id=None,
            is_superuser=False,
            permissions=["workspace:write"],
        )

        response = await client.patch(
            f"/api/v1/workspaces/{workspace_id}",
            headers={"Authorization": f"Bearer {token}"},
            json={"display_name": "Permission Updated"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["display_name"] == "Permission Updated"

    async def test_update_workspace_without_permission_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        workspace_id,
    ) -> None:
        """User without permission cannot update workspace."""
        response = await client.patch(
            f"/api/v1/workspaces/{workspace_id}",
            headers=auth_headers,
            json={"display_name": "Should Fail"},
        )

        assert response.status_code == 403
        assert "workspace:write" in response.json()["detail"]

    async def test_update_workspace_other_org_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
    ) -> None:
        """Cannot update workspace from another organization."""
        other_org = Organization(name="other-org-updatews", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        other_ws = Workspace(name="other-update-ws", organization_id=other_org.id)
        db_session.add(other_ws)
        await db_session.commit()
        await db_session.refresh(other_ws)

        response = await client.patch(
            f"/api/v1/workspaces/{other_ws.id}",
            headers=auth_headers,
            json={"display_name": "Hacked"},
        )

        assert response.status_code == 403

    async def test_update_workspace_name(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_organization: Organization,
    ) -> None:
        """Can update workspace name."""
        ws = Workspace(name="name-update-ws", organization_id=sample_organization.id)
        db_session.add(ws)
        await db_session.commit()
        await db_session.refresh(ws)

        response = await client.patch(
            f"/api/v1/workspaces/{ws.id}",
            headers=superuser_headers,
            json={"name": "updated-name-ws"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "updated-name-ws"

    async def test_update_workspace_duplicate_name(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_organization: Organization,
    ) -> None:
        """Cannot update to duplicate name in same org."""
        ws1 = Workspace(name="ws-dup-1", organization_id=sample_organization.id)
        ws2 = Workspace(name="ws-dup-2", organization_id=sample_organization.id)
        db_session.add(ws1)
        db_session.add(ws2)
        await db_session.commit()
        await db_session.refresh(ws2)

        response = await client.patch(
            f"/api/v1/workspaces/{ws2.id}",
            headers=superuser_headers,
            json={"name": "ws-dup-1"},
        )

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    async def test_update_workspace_deactivate(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_organization: Organization,
    ) -> None:
        """Can deactivate workspace."""
        ws = Workspace(name="deactivate-ws", organization_id=sample_organization.id)
        db_session.add(ws)
        await db_session.commit()
        await db_session.refresh(ws)
        assert ws.is_active is True

        response = await client.patch(
            f"/api/v1/workspaces/{ws.id}",
            headers=superuser_headers,
            json={"is_active": False},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["is_active"] is False

    async def test_update_workspace_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Updating non-existent workspace returns 404."""
        response = await client.patch(
            f"/api/v1/workspaces/{uuid4()}",
            headers=superuser_headers,
            json={"display_name": "Not Found"},
        )

        assert response.status_code == 404


class TestDeleteWorkspace:
    """Tests for DELETE /workspaces/{workspace_id} endpoint."""

    async def test_delete_workspace_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_organization: Organization,
    ) -> None:
        """Superuser can delete workspace."""
        ws = Workspace(name="delete-ws", organization_id=sample_organization.id)
        db_session.add(ws)
        await db_session.commit()
        await db_session.refresh(ws)
        assert ws.is_active is True

        response = await client.delete(
            f"/api/v1/workspaces/{ws.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 204

        # Verify soft deleted
        await db_session.refresh(ws)
        assert ws.is_active is False

    async def test_delete_workspace_with_permission(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        org_id,
        sample_user: "User",  # Ensure user exists in DB
    ) -> None:
        """User with workspace:delete permission can delete workspace."""
        from ..models import Permission, Role, User
        from ..routers.auth import create_access_token
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        # Create workspace to delete
        ws = Workspace(name="perm-delete-ws", organization_id=org_id)
        db_session.add(ws)
        await db_session.commit()
        await db_session.refresh(ws)

        # Reload user in current session to avoid greenlet issues
        result = await db_session.execute(
            select(User).where(User.id == sample_user.id).options(selectinload(User.roles))
        )
        user = result.scalar_one()

        perm = Permission(name="workspace:delete", description="Delete WS")
        db_session.add(perm)
        await db_session.commit()
        await db_session.refresh(perm)

        role = Role(
            name="ws-deleter-test",
            description="WS Deleter",
            organization_id=org_id,
        )
        role.permissions.append(perm)
        db_session.add(role)
        await db_session.commit()
        await db_session.refresh(role)

        user.roles.append(role)
        await db_session.commit()

        token = create_access_token(
            user_id=user.id,
            email=user.email,
            org_id=org_id,
            workspace_id=None,
            is_superuser=False,
            permissions=["workspace:delete"],
        )

        response = await client.delete(
            f"/api/v1/workspaces/{ws.id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 204

    async def test_delete_workspace_without_permission_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_workspace: Workspace,
    ) -> None:
        """User without permission cannot delete workspace."""
        response = await client.delete(
            f"/api/v1/workspaces/{sample_workspace.id}",
            headers=auth_headers,
        )

        assert response.status_code == 403
        assert "workspace:delete" in response.json()["detail"]

    async def test_delete_workspace_other_org_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
    ) -> None:
        """Cannot delete workspace from another organization."""
        other_org = Organization(name="other-org-deletews", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        other_ws = Workspace(name="other-delete-ws", organization_id=other_org.id)
        db_session.add(other_ws)
        await db_session.commit()
        await db_session.refresh(other_ws)

        response = await client.delete(
            f"/api/v1/workspaces/{other_ws.id}",
            headers=auth_headers,
        )

        assert response.status_code == 403

    async def test_delete_workspace_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Deleting non-existent workspace returns 404."""
        response = await client.delete(
            f"/api/v1/workspaces/{uuid4()}",
            headers=superuser_headers,
        )

        assert response.status_code == 404

    async def test_delete_workspace_unauthenticated(
        self,
        client: AsyncClient,
        sample_workspace: Workspace,
    ) -> None:
        """Unauthenticated request returns 401."""
        response = await client.delete(
            f"/api/v1/workspaces/{sample_workspace.id}",
        )

        assert response.status_code == 401
