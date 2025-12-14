# -*- coding: utf-8 -*-
"""Tests for Users Router."""

from __future__ import annotations

import hashlib
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Organization, Permission, Role, User, Workspace

pytestmark = pytest.mark.asyncio


def hash_password(password: str) -> str:
    """Hash password with SHA256 (mirrors users.py)."""
    return hashlib.sha256(password.encode()).hexdigest()


class TestListUsers:
    """Tests for GET /users endpoint."""

    async def test_list_users_in_own_org(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_user: User,
    ) -> None:
        """User can list users in their organization."""
        response = await client.get(
            "/api/v1/users",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert data["total"] >= 1

        # Should see at least the sample user
        emails = [item["email"] for item in data["items"]]
        assert sample_user.email in emails

    async def test_list_users_superuser_all(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_user: User,
    ) -> None:
        """Superuser can list all users."""
        response = await client.get(
            "/api/v1/users",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1

    async def test_list_users_superuser_filter_by_org(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        org_id,
    ) -> None:
        """Superuser can filter by organization."""
        # Create another org with a user
        other_org = Organization(name="other-org-list", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        other_user = User(
            email="otherlist@example.com",
            password_hash=hash_password("password123"),
            organization_id=other_org.id,
        )
        db_session.add(other_user)
        await db_session.commit()

        # Filter by original org
        response = await client.get(
            f"/api/v1/users?organization_id={org_id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()

        # Should not include the other org user
        org_ids = [item["organization_id"] for item in data["items"]]
        assert str(other_org.id) not in org_ids

    async def test_list_users_pagination(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_organization: Organization,
    ) -> None:
        """Pagination works correctly."""
        # Create multiple users
        for i in range(5):
            user = User(
                email=f"paguser{i}@example.com",
                password_hash=hash_password("password123"),
                organization_id=sample_organization.id,
            )
            db_session.add(user)
        await db_session.commit()

        response = await client.get(
            "/api/v1/users?page=1&page_size=2",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 2
        assert len(data["items"]) <= 2

    async def test_list_users_includes_roles(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_organization: Organization,
    ) -> None:
        """User response includes roles."""
        # Create role and user with role
        role = Role(
            name="test-role-list",
            description="Test Role",
            organization_id=sample_organization.id,
        )
        db_session.add(role)
        await db_session.commit()
        await db_session.refresh(role)

        user = User(
            email="userwithrole@example.com",
            password_hash=hash_password("password123"),
            organization_id=sample_organization.id,
        )
        user.roles.append(role)
        db_session.add(user)
        await db_session.commit()

        response = await client.get(
            "/api/v1/users",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()

        # Find user with role
        found = None
        for item in data["items"]:
            if item["email"] == "userwithrole@example.com":
                found = item
                break

        assert found is not None
        assert "roles" in found
        assert len(found["roles"]) == 1
        assert found["roles"][0]["name"] == "test-role-list"

    async def test_list_users_unauthenticated(
        self,
        client: AsyncClient,
    ) -> None:
        """Unauthenticated request returns 401."""
        response = await client.get("/api/v1/users")
        assert response.status_code == 401


class TestCreateUser:
    """Tests for POST /users endpoint."""

    async def test_create_user_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_organization: Organization,
    ) -> None:
        """Superuser can create a user."""
        response = await client.post(
            "/api/v1/users",
            headers=superuser_headers,
            json={
                "email": "newuser@example.com",
                "password": "password123",
                "display_name": "New User",
                "organization_id": str(sample_organization.id),
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert data["display_name"] == "New User"
        assert data["organization_id"] == str(sample_organization.id)
        assert data["is_active"] is True
        assert "roles" in data

    async def test_create_user_with_permission(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        org_id,
        user_id,
    ) -> None:
        """User with user:create permission can create user."""
        from sqlalchemy import select

        from ..routers.auth import create_access_token

        # Get user and add permission
        result = await db_session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one()

        perm = Permission(name="user:create", description="Create users")
        db_session.add(perm)
        await db_session.commit()
        await db_session.refresh(perm)

        role = Role(
            name="user-creator",
            description="Can create users",
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
            permissions=["user:create"],
        )

        response = await client.post(
            "/api/v1/users",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "email": "permcreated@example.com",
                "password": "password123",
                "organization_id": str(org_id),
            },
        )

        assert response.status_code == 201
        assert response.json()["email"] == "permcreated@example.com"

    async def test_create_user_without_permission_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_organization: Organization,
    ) -> None:
        """User without permission cannot create user."""
        response = await client.post(
            "/api/v1/users",
            headers=auth_headers,
            json={
                "email": "shouldfail@example.com",
                "password": "password123",
                "organization_id": str(sample_organization.id),
            },
        )

        assert response.status_code == 403
        assert "user:create" in response.json()["detail"]

    async def test_create_user_other_org_forbidden(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        org_id,
        user_id,
    ) -> None:
        """Cannot create user in another organization."""
        from ..routers.auth import create_access_token

        # Create another org
        other_org = Organization(name="other-org-create", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        # Even with permission, cannot create in other org
        from sqlalchemy import select

        result = await db_session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one()

        token = create_access_token(
            user_id=user.id,
            email=user.email,
            org_id=org_id,
            workspace_id=None,
            is_superuser=False,
            permissions=["user:create"],
        )

        response = await client.post(
            "/api/v1/users",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "email": "otherorg@example.com",
                "password": "password123",
                "organization_id": str(other_org.id),
            },
        )

        assert response.status_code == 403
        assert "another organization" in response.json()["detail"]

    async def test_create_user_org_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Creating user in non-existent organization returns 404."""
        response = await client.post(
            "/api/v1/users",
            headers=superuser_headers,
            json={
                "email": "noorg@example.com",
                "password": "password123",
                "organization_id": str(uuid4()),
            },
        )

        assert response.status_code == 404
        assert "Organization not found" in response.json()["detail"]

    async def test_create_user_duplicate_email(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_organization: Organization,
        sample_user: User,
    ) -> None:
        """Cannot create user with duplicate email."""
        response = await client.post(
            "/api/v1/users",
            headers=superuser_headers,
            json={
                "email": sample_user.email,
                "password": "password123",
                "organization_id": str(sample_organization.id),
            },
        )

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    async def test_create_user_with_workspace(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_organization: Organization,
        sample_workspace: Workspace,
    ) -> None:
        """Can create user with default workspace."""
        response = await client.post(
            "/api/v1/users",
            headers=superuser_headers,
            json={
                "email": "withworkspace@example.com",
                "password": "password123",
                "organization_id": str(sample_organization.id),
                "default_workspace_id": str(sample_workspace.id),
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["default_workspace_id"] == str(sample_workspace.id)

    async def test_create_user_workspace_wrong_org(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_organization: Organization,
    ) -> None:
        """Cannot set workspace from different organization."""
        # Create another org with workspace
        other_org = Organization(name="other-org-ws", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        other_ws = Workspace(name="other-ws", organization_id=other_org.id)
        db_session.add(other_ws)
        await db_session.commit()
        await db_session.refresh(other_ws)

        response = await client.post(
            "/api/v1/users",
            headers=superuser_headers,
            json={
                "email": "wrongws@example.com",
                "password": "password123",
                "organization_id": str(sample_organization.id),
                "default_workspace_id": str(other_ws.id),
            },
        )

        assert response.status_code == 404
        assert "Workspace not found" in response.json()["detail"]

    async def test_create_user_with_roles(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_organization: Organization,
    ) -> None:
        """Can create user with roles."""
        role = Role(
            name="test-role-create",
            description="Test Role",
            organization_id=sample_organization.id,
        )
        db_session.add(role)
        await db_session.commit()
        await db_session.refresh(role)

        response = await client.post(
            "/api/v1/users",
            headers=superuser_headers,
            json={
                "email": "withrole@example.com",
                "password": "password123",
                "organization_id": str(sample_organization.id),
                "role_ids": [str(role.id)],
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert len(data["roles"]) == 1
        assert data["roles"][0]["name"] == "test-role-create"

    async def test_create_user_role_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_organization: Organization,
    ) -> None:
        """Cannot create user with non-existent role."""
        response = await client.post(
            "/api/v1/users",
            headers=superuser_headers,
            json={
                "email": "badrole@example.com",
                "password": "password123",
                "organization_id": str(sample_organization.id),
                "role_ids": [str(uuid4())],
            },
        )

        assert response.status_code == 404
        assert "roles not found" in response.json()["detail"]

    async def test_create_user_short_password(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_organization: Organization,
    ) -> None:
        """Cannot create user with short password."""
        response = await client.post(
            "/api/v1/users",
            headers=superuser_headers,
            json={
                "email": "shortpwd@example.com",
                "password": "short",
                "organization_id": str(sample_organization.id),
            },
        )

        assert response.status_code == 422


class TestGetUser:
    """Tests for GET /users/{user_id} endpoint."""

    async def test_get_self(
        self,
        client: AsyncClient,
        auth_headers: dict,
        user_id,
    ) -> None:
        """User can get self."""
        response = await client.get(
            f"/api/v1/users/{user_id}",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(user_id)
        assert "email" in data
        assert "roles" in data

    async def test_get_other_user_with_permission(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        org_id,
        user_id,
        sample_organization: Organization,
    ) -> None:
        """User with user:read permission can get other user in same org."""
        from sqlalchemy import select

        from ..routers.auth import create_access_token

        # Create another user in same org
        other_user = User(
            email="otheruserget@example.com",
            password_hash=hash_password("password123"),
            organization_id=org_id,
        )
        db_session.add(other_user)
        await db_session.commit()
        await db_session.refresh(other_user)

        # Get original user and add permission
        result = await db_session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one()

        perm = Permission(name="user:read", description="Read users")
        db_session.add(perm)
        await db_session.commit()
        await db_session.refresh(perm)

        role = Role(
            name="user-reader",
            description="Can read users",
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
            permissions=["user:read"],
        )

        response = await client.get(
            f"/api/v1/users/{other_user.id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert response.json()["id"] == str(other_user.id)

    async def test_get_user_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_user: User,
    ) -> None:
        """Superuser can get any user."""
        response = await client.get(
            f"/api/v1/users/{sample_user.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_user.id)

    async def test_get_other_user_without_permission_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
        org_id,
    ) -> None:
        """User without permission cannot get other user."""
        # Create another user
        other_user = User(
            email="cannotget@example.com",
            password_hash=hash_password("password123"),
            organization_id=org_id,
        )
        db_session.add(other_user)
        await db_session.commit()
        await db_session.refresh(other_user)

        response = await client.get(
            f"/api/v1/users/{other_user.id}",
            headers=auth_headers,
        )

        assert response.status_code == 403
        assert "user:read" in response.json()["detail"]

    async def test_get_user_other_org_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
    ) -> None:
        """Cannot get user in different organization."""
        # Create another org with user
        other_org = Organization(name="other-org-get", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        other_user = User(
            email="otherorguserget@example.com",
            password_hash=hash_password("password123"),
            organization_id=other_org.id,
        )
        db_session.add(other_user)
        await db_session.commit()
        await db_session.refresh(other_user)

        response = await client.get(
            f"/api/v1/users/{other_user.id}",
            headers=auth_headers,
        )

        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]

    async def test_get_user_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Non-existent user returns 404."""
        response = await client.get(
            f"/api/v1/users/{uuid4()}",
            headers=superuser_headers,
        )

        assert response.status_code == 404
        assert "User not found" in response.json()["detail"]

    async def test_get_user_unauthenticated(
        self,
        client: AsyncClient,
        sample_user: User,
    ) -> None:
        """Unauthenticated request returns 401."""
        response = await client.get(f"/api/v1/users/{sample_user.id}")
        assert response.status_code == 401


class TestUpdateUser:
    """Tests for PATCH /users/{user_id} endpoint."""

    async def test_update_self(
        self,
        client: AsyncClient,
        auth_headers: dict,
        user_id,
    ) -> None:
        """User can update self."""
        response = await client.patch(
            f"/api/v1/users/{user_id}",
            headers=auth_headers,
            json={"display_name": "Updated Self"},
        )

        assert response.status_code == 200
        assert response.json()["display_name"] == "Updated Self"

    async def test_update_user_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_user: User,
    ) -> None:
        """Superuser can update any user."""
        response = await client.patch(
            f"/api/v1/users/{sample_user.id}",
            headers=superuser_headers,
            json={"display_name": "Superuser Updated"},
        )

        assert response.status_code == 200
        assert response.json()["display_name"] == "Superuser Updated"

    async def test_update_other_user_with_permission(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        org_id,
        user_id,
    ) -> None:
        """User with user:write permission can update other user."""
        from sqlalchemy import select

        from ..routers.auth import create_access_token

        # Create another user in same org
        other_user = User(
            email="otheruserupdate@example.com",
            password_hash=hash_password("password123"),
            organization_id=org_id,
        )
        db_session.add(other_user)
        await db_session.commit()
        await db_session.refresh(other_user)

        # Get original user
        result = await db_session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one()

        perm = Permission(name="user:write", description="Write users")
        db_session.add(perm)
        await db_session.commit()
        await db_session.refresh(perm)

        role = Role(
            name="user-writer-update",
            description="Can write users",
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
            permissions=["user:write"],
        )

        response = await client.patch(
            f"/api/v1/users/{other_user.id}",
            headers={"Authorization": f"Bearer {token}"},
            json={"display_name": "Permission Updated"},
        )

        assert response.status_code == 200
        assert response.json()["display_name"] == "Permission Updated"

    async def test_update_other_user_without_permission_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
        org_id,
    ) -> None:
        """User without permission cannot update other user."""
        other_user = User(
            email="cannotupdate@example.com",
            password_hash=hash_password("password123"),
            organization_id=org_id,
        )
        db_session.add(other_user)
        await db_session.commit()
        await db_session.refresh(other_user)

        response = await client.patch(
            f"/api/v1/users/{other_user.id}",
            headers=auth_headers,
            json={"display_name": "Should Fail"},
        )

        assert response.status_code == 403
        assert "user:write" in response.json()["detail"]

    async def test_update_user_other_org_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
    ) -> None:
        """Cannot update user in different organization."""
        other_org = Organization(name="other-org-update", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        other_user = User(
            email="otherorguserupdate@example.com",
            password_hash=hash_password("password123"),
            organization_id=other_org.id,
        )
        db_session.add(other_user)
        await db_session.commit()
        await db_session.refresh(other_user)

        response = await client.patch(
            f"/api/v1/users/{other_user.id}",
            headers=auth_headers,
            json={"display_name": "Hacked"},
        )

        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]

    async def test_update_user_email(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_organization: Organization,
    ) -> None:
        """Can update user email."""
        user = User(
            email="oldemail@example.com",
            password_hash=hash_password("password123"),
            organization_id=sample_organization.id,
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        response = await client.patch(
            f"/api/v1/users/{user.id}",
            headers=superuser_headers,
            json={"email": "newemail@example.com"},
        )

        assert response.status_code == 200
        assert response.json()["email"] == "newemail@example.com"

    async def test_update_user_email_duplicate(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_organization: Organization,
        sample_user: User,
    ) -> None:
        """Cannot update to duplicate email."""
        user = User(
            email="toupdate@example.com",
            password_hash=hash_password("password123"),
            organization_id=sample_organization.id,
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        response = await client.patch(
            f"/api/v1/users/{user.id}",
            headers=superuser_headers,
            json={"email": sample_user.email},
        )

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    async def test_update_user_password(
        self,
        client: AsyncClient,
        auth_headers: dict,
        user_id,
    ) -> None:
        """Can update password."""
        response = await client.patch(
            f"/api/v1/users/{user_id}",
            headers=auth_headers,
            json={"password": "newpassword123"},
        )

        assert response.status_code == 200

    async def test_update_user_workspace(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_user: User,
        sample_workspace: Workspace,
    ) -> None:
        """Can update default workspace."""
        response = await client.patch(
            f"/api/v1/users/{sample_user.id}",
            headers=superuser_headers,
            json={"default_workspace_id": str(sample_workspace.id)},
        )

        assert response.status_code == 200
        assert response.json()["default_workspace_id"] == str(sample_workspace.id)

    async def test_update_user_workspace_wrong_org(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_user: User,
    ) -> None:
        """Cannot set workspace from different organization."""
        other_org = Organization(name="other-org-ws-update", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        other_ws = Workspace(name="other-ws-update", organization_id=other_org.id)
        db_session.add(other_ws)
        await db_session.commit()
        await db_session.refresh(other_ws)

        response = await client.patch(
            f"/api/v1/users/{sample_user.id}",
            headers=superuser_headers,
            json={"default_workspace_id": str(other_ws.id)},
        )

        assert response.status_code == 404
        assert "Workspace not found" in response.json()["detail"]

    async def test_update_user_deactivate_requires_permission(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
        org_id,
    ) -> None:
        """Changing is_active requires permission."""
        other_user = User(
            email="todeactivate@example.com",
            password_hash=hash_password("password123"),
            organization_id=org_id,
        )
        db_session.add(other_user)
        await db_session.commit()
        await db_session.refresh(other_user)

        response = await client.patch(
            f"/api/v1/users/{other_user.id}",
            headers=auth_headers,
            json={"is_active": False},
        )

        assert response.status_code == 403

    async def test_update_user_deactivate_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_organization: Organization,
    ) -> None:
        """Superuser can deactivate user."""
        user = User(
            email="superdeactivate@example.com",
            password_hash=hash_password("password123"),
            organization_id=sample_organization.id,
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        response = await client.patch(
            f"/api/v1/users/{user.id}",
            headers=superuser_headers,
            json={"is_active": False},
        )

        assert response.status_code == 200
        assert response.json()["is_active"] is False

    async def test_update_user_roles_requires_permission(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
        org_id,
    ) -> None:
        """Changing roles requires user:assign_role permission."""
        other_user = User(
            email="torole@example.com",
            password_hash=hash_password("password123"),
            organization_id=org_id,
        )
        db_session.add(other_user)
        await db_session.commit()
        await db_session.refresh(other_user)

        role = Role(
            name="target-role",
            description="Target Role",
            organization_id=org_id,
        )
        db_session.add(role)
        await db_session.commit()
        await db_session.refresh(role)

        response = await client.patch(
            f"/api/v1/users/{other_user.id}",
            headers=auth_headers,
            json={"role_ids": [str(role.id)]},
        )

        assert response.status_code == 403
        assert "user:assign_role" in response.json()["detail"]

    async def test_update_user_roles_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_organization: Organization,
    ) -> None:
        """Superuser can assign roles."""
        user = User(
            email="assignrole@example.com",
            password_hash=hash_password("password123"),
            organization_id=sample_organization.id,
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        role = Role(
            name="assigned-role",
            description="Assigned Role",
            organization_id=sample_organization.id,
        )
        db_session.add(role)
        await db_session.commit()
        await db_session.refresh(role)

        response = await client.patch(
            f"/api/v1/users/{user.id}",
            headers=superuser_headers,
            json={"role_ids": [str(role.id)]},
        )

        assert response.status_code == 200
        assert len(response.json()["roles"]) == 1
        assert response.json()["roles"][0]["name"] == "assigned-role"

    async def test_update_user_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Updating non-existent user returns 404."""
        response = await client.patch(
            f"/api/v1/users/{uuid4()}",
            headers=superuser_headers,
            json={"display_name": "Not Found"},
        )

        assert response.status_code == 404


class TestDeleteUser:
    """Tests for DELETE /users/{user_id} endpoint."""

    async def test_delete_user_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_organization: Organization,
    ) -> None:
        """Superuser can delete user."""
        user = User(
            email="todelete@example.com",
            password_hash=hash_password("password123"),
            organization_id=sample_organization.id,
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        assert user.is_active is True

        response = await client.delete(
            f"/api/v1/users/{user.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 204

        # Verify soft deleted
        await db_session.refresh(user)
        assert user.is_active is False

    async def test_delete_user_with_permission(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        org_id,
        user_id,
    ) -> None:
        """User with user:delete permission can delete user."""
        from sqlalchemy import select

        from ..routers.auth import create_access_token

        # Create user to delete
        to_delete = User(
            email="permdelete@example.com",
            password_hash=hash_password("password123"),
            organization_id=org_id,
        )
        db_session.add(to_delete)
        await db_session.commit()
        await db_session.refresh(to_delete)

        # Get original user and add permission
        result = await db_session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one()

        perm = Permission(name="user:delete", description="Delete users")
        db_session.add(perm)
        await db_session.commit()
        await db_session.refresh(perm)

        role = Role(
            name="user-deleter",
            description="Can delete users",
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
            permissions=["user:delete"],
        )

        response = await client.delete(
            f"/api/v1/users/{to_delete.id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 204

    async def test_delete_self_forbidden(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        superuser_id,
    ) -> None:
        """Cannot delete yourself."""
        response = await client.delete(
            f"/api/v1/users/{superuser_id}",
            headers=superuser_headers,
        )

        assert response.status_code == 400
        assert "Cannot delete yourself" in response.json()["detail"]

    async def test_delete_user_without_permission_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
        org_id,
    ) -> None:
        """User without permission cannot delete user."""
        to_delete = User(
            email="cannotdelete@example.com",
            password_hash=hash_password("password123"),
            organization_id=org_id,
        )
        db_session.add(to_delete)
        await db_session.commit()
        await db_session.refresh(to_delete)

        response = await client.delete(
            f"/api/v1/users/{to_delete.id}",
            headers=auth_headers,
        )

        assert response.status_code == 403
        assert "user:delete" in response.json()["detail"]

    async def test_delete_user_other_org_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
    ) -> None:
        """Cannot delete user in different organization."""
        other_org = Organization(name="other-org-delete", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        other_user = User(
            email="otherorguserdelete@example.com",
            password_hash=hash_password("password123"),
            organization_id=other_org.id,
        )
        db_session.add(other_user)
        await db_session.commit()
        await db_session.refresh(other_user)

        response = await client.delete(
            f"/api/v1/users/{other_user.id}",
            headers=auth_headers,
        )

        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]

    async def test_delete_user_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Deleting non-existent user returns 404."""
        response = await client.delete(
            f"/api/v1/users/{uuid4()}",
            headers=superuser_headers,
        )

        assert response.status_code == 404
        assert "User not found" in response.json()["detail"]

    async def test_delete_user_unauthenticated(
        self,
        client: AsyncClient,
        sample_user: User,
    ) -> None:
        """Unauthenticated request returns 401."""
        response = await client.delete(f"/api/v1/users/{sample_user.id}")
        assert response.status_code == 401
