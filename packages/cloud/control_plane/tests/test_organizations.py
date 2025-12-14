# -*- coding: utf-8 -*-
"""Tests for Organizations Router."""

from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Organization, Workspace

pytestmark = pytest.mark.asyncio


class TestListOrganizations:
    """Tests for GET /organizations endpoint."""

    async def test_list_organizations_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_organization: Organization,
    ) -> None:
        """Superuser can list all organizations."""
        response = await client.get(
            "/api/v1/organizations",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert data["total"] >= 1

    async def test_list_organizations_regular_user_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ) -> None:
        """Regular user cannot list all organizations."""
        response = await client.get(
            "/api/v1/organizations",
            headers=auth_headers,
        )

        assert response.status_code == 403
        assert "Superuser privileges required" in response.json()["detail"]

    async def test_list_organizations_unauthenticated(
        self,
        client: AsyncClient,
    ) -> None:
        """Unauthenticated request returns 401."""
        response = await client.get("/api/v1/organizations")

        assert response.status_code == 401

    async def test_list_organizations_with_pagination(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
    ) -> None:
        """Pagination works correctly."""
        # Create multiple organizations
        for i in range(5):
            org = Organization(name=f"list-page-org-{i}", display_name=f"Org {i}")
            db_session.add(org)
        await db_session.commit()

        # Test with pagination
        response = await client.get(
            "/api/v1/organizations?page=1&page_size=2",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 2
        assert len(data["items"]) <= 2

    async def test_list_organizations_includes_workspace_count(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
    ) -> None:
        """Organization response includes workspace count."""
        # Create organization with workspaces
        org = Organization(name="org-with-workspaces", display_name="Org With WS")
        db_session.add(org)
        await db_session.commit()
        await db_session.refresh(org)

        ws1 = Workspace(name="ws1", organization_id=org.id)
        ws2 = Workspace(name="ws2", organization_id=org.id)
        db_session.add(ws1)
        db_session.add(ws2)
        await db_session.commit()

        response = await client.get(
            "/api/v1/organizations",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()

        # Find our organization
        found_org = None
        for item in data["items"]:
            if item["name"] == "org-with-workspaces":
                found_org = item
                break

        assert found_org is not None
        assert found_org["workspace_count"] == 2


class TestCreateOrganization:
    """Tests for POST /organizations endpoint."""

    async def test_create_organization_success(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Superuser can create an organization."""
        response = await client.post(
            "/api/v1/organizations",
            headers=superuser_headers,
            json={
                "name": "new-org",
                "display_name": "New Organization",
                "settings": {"key": "value"},
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "new-org"
        assert data["display_name"] == "New Organization"
        assert data["settings"] == {"key": "value"}
        assert data["is_active"] is True
        assert data["workspace_count"] == 0

    async def test_create_organization_minimal(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Create organization with minimal fields."""
        response = await client.post(
            "/api/v1/organizations",
            headers=superuser_headers,
            json={"name": "minimal-org"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "minimal-org"
        assert data["display_name"] is None
        assert data["settings"] == {}

    async def test_create_organization_regular_user_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ) -> None:
        """Regular user cannot create organization."""
        response = await client.post(
            "/api/v1/organizations",
            headers=auth_headers,
            json={"name": "forbidden-org"},
        )

        assert response.status_code == 403
        assert "Superuser privileges required" in response.json()["detail"]

    async def test_create_organization_duplicate_name(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_organization: Organization,
    ) -> None:
        """Cannot create organization with duplicate name."""
        response = await client.post(
            "/api/v1/organizations",
            headers=superuser_headers,
            json={"name": sample_organization.name},
        )

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    async def test_create_organization_empty_name(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Cannot create organization with empty name."""
        response = await client.post(
            "/api/v1/organizations",
            headers=superuser_headers,
            json={"name": ""},
        )

        assert response.status_code == 422

    async def test_create_organization_missing_name(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Name is required."""
        response = await client.post(
            "/api/v1/organizations",
            headers=superuser_headers,
            json={},
        )

        assert response.status_code == 422


class TestGetOrganization:
    """Tests for GET /organizations/{org_id} endpoint."""

    async def test_get_own_organization(
        self,
        client: AsyncClient,
        auth_headers: dict,
        org_id,
    ) -> None:
        """User can get their own organization."""
        response = await client.get(
            f"/api/v1/organizations/{org_id}",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(org_id)

    async def test_get_organization_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_organization: Organization,
    ) -> None:
        """Superuser can get any organization."""
        response = await client.get(
            f"/api/v1/organizations/{sample_organization.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_organization.id)
        assert data["name"] == sample_organization.name
        assert "workspace_count" in data

    async def test_get_organization_different_org_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
    ) -> None:
        """User cannot get a different organization."""
        # Create a different organization
        other_org = Organization(name="other-org-view", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        response = await client.get(
            f"/api/v1/organizations/{other_org.id}",
            headers=auth_headers,
        )

        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]

    async def test_get_organization_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Non-existent organization returns 404."""
        response = await client.get(
            f"/api/v1/organizations/{uuid4()}",
            headers=superuser_headers,
        )

        assert response.status_code == 404
        assert "Organization not found" in response.json()["detail"]

    async def test_get_organization_unauthenticated(
        self,
        client: AsyncClient,
        sample_organization: Organization,
    ) -> None:
        """Unauthenticated request returns 401."""
        response = await client.get(
            f"/api/v1/organizations/{sample_organization.id}",
        )

        assert response.status_code == 401


class TestUpdateOrganization:
    """Tests for PATCH /organizations/{org_id} endpoint."""

    async def test_update_organization_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_organization: Organization,
    ) -> None:
        """Superuser can update any organization."""
        response = await client.patch(
            f"/api/v1/organizations/{sample_organization.id}",
            headers=superuser_headers,
            json={
                "display_name": "Updated Display Name",
                "settings": {"updated": True},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["display_name"] == "Updated Display Name"
        assert data["settings"] == {"updated": True}

    async def test_update_organization_with_permission(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        org_id,
        user_id,
    ) -> None:
        """User with org:write permission can update their organization."""
        from ..models import Permission, Role, User

        # Get the user and add org:write permission
        from sqlalchemy import select
        result = await db_session.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one()

        # Create role with org:write permission
        perm = Permission(name="org:write", description="Write org")
        db_session.add(perm)
        await db_session.commit()
        await db_session.refresh(perm)

        role = Role(
            name="org-admin-test",
            description="Org Admin",
            organization_id=org_id,
        )
        role.permissions.append(perm)
        db_session.add(role)
        await db_session.commit()
        await db_session.refresh(role)

        user.roles.append(role)
        await db_session.commit()

        # Create auth headers with permissions
        from ..routers.auth import create_access_token
        token = create_access_token(
            user_id=user.id,
            email=user.email,
            org_id=org_id,
            workspace_id=None,
            is_superuser=False,
            permissions=["org:write"],
        )

        response = await client.patch(
            f"/api/v1/organizations/{org_id}",
            headers={"Authorization": f"Bearer {token}"},
            json={"display_name": "Permission Updated"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["display_name"] == "Permission Updated"

    async def test_update_organization_without_permission_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        org_id,
    ) -> None:
        """User without org:write permission cannot update."""
        response = await client.patch(
            f"/api/v1/organizations/{org_id}",
            headers=auth_headers,
            json={"display_name": "Should Fail"},
        )

        assert response.status_code == 403
        assert "org:write" in response.json()["detail"]

    async def test_update_organization_different_org_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
    ) -> None:
        """User cannot update a different organization."""
        other_org = Organization(name="other-org-update", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        response = await client.patch(
            f"/api/v1/organizations/{other_org.id}",
            headers=auth_headers,
            json={"display_name": "Hacked"},
        )

        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]

    async def test_update_organization_name(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
    ) -> None:
        """Can update organization name."""
        org = Organization(name="name-update-test", display_name="Name Test")
        db_session.add(org)
        await db_session.commit()
        await db_session.refresh(org)

        response = await client.patch(
            f"/api/v1/organizations/{org.id}",
            headers=superuser_headers,
            json={"name": "name-updated"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "name-updated"

    async def test_update_organization_duplicate_name(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_organization: Organization,
    ) -> None:
        """Cannot update to duplicate name."""
        org = Organization(name="another-org-to-update", display_name="Another")
        db_session.add(org)
        await db_session.commit()
        await db_session.refresh(org)

        response = await client.patch(
            f"/api/v1/organizations/{org.id}",
            headers=superuser_headers,
            json={"name": sample_organization.name},
        )

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    async def test_update_organization_deactivate(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
    ) -> None:
        """Superuser can deactivate organization."""
        org = Organization(name="deactivate-org", display_name="Deactivate")
        db_session.add(org)
        await db_session.commit()
        await db_session.refresh(org)
        assert org.is_active is True

        response = await client.patch(
            f"/api/v1/organizations/{org.id}",
            headers=superuser_headers,
            json={"is_active": False},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["is_active"] is False

    async def test_update_organization_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Updating non-existent organization returns 404."""
        response = await client.patch(
            f"/api/v1/organizations/{uuid4()}",
            headers=superuser_headers,
            json={"display_name": "Not Found"},
        )

        assert response.status_code == 404


class TestDeleteOrganization:
    """Tests for DELETE /organizations/{org_id} endpoint."""

    async def test_delete_organization_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
    ) -> None:
        """Superuser can delete (soft) an organization."""
        org = Organization(name="delete-org", display_name="Delete Me")
        db_session.add(org)
        await db_session.commit()
        await db_session.refresh(org)
        assert org.is_active is True

        response = await client.delete(
            f"/api/v1/organizations/{org.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 204

        # Verify soft deleted
        await db_session.refresh(org)
        assert org.is_active is False

    async def test_delete_organization_regular_user_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_organization: Organization,
    ) -> None:
        """Regular user cannot delete organization."""
        response = await client.delete(
            f"/api/v1/organizations/{sample_organization.id}",
            headers=auth_headers,
        )

        assert response.status_code == 403
        assert "Superuser privileges required" in response.json()["detail"]

    async def test_delete_organization_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Deleting non-existent organization returns 404."""
        response = await client.delete(
            f"/api/v1/organizations/{uuid4()}",
            headers=superuser_headers,
        )

        assert response.status_code == 404
        assert "Organization not found" in response.json()["detail"]

    async def test_delete_organization_unauthenticated(
        self,
        client: AsyncClient,
        sample_organization: Organization,
    ) -> None:
        """Unauthenticated request returns 401."""
        response = await client.delete(
            f"/api/v1/organizations/{sample_organization.id}",
        )

        assert response.status_code == 401
