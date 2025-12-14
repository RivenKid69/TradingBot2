# -*- coding: utf-8 -*-
"""Tests for Strategies Router."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import (
    Artifact,
    Build,
    ChangeClass,
    Organization,
    Permission,
    Role,
    Strategy,
    StrategyVersion,
    User,
    Workspace,
)

pytestmark = pytest.mark.asyncio


# =============================================================================
# Test Fixtures for Strategies Module
# =============================================================================


@pytest.fixture
async def sample_strategy(
    db_session: AsyncSession,
    sample_workspace: Workspace,
) -> Strategy:
    """Create a sample strategy for testing."""
    strategy = Strategy(
        name="test-strategy",
        description="Test strategy for unit tests",
        workspace_id=sample_workspace.id,
        git_repo="https://github.com/test/strategy.git",
        tags=["test", "ml"],
        metadata={"framework": "pytorch"},
    )
    db_session.add(strategy)
    await db_session.commit()
    await db_session.refresh(strategy)
    return strategy


@pytest.fixture
async def sample_version(
    db_session: AsyncSession,
    sample_strategy: Strategy,
    sample_workspace: Workspace,
) -> StrategyVersion:
    """Create a sample strategy version for testing."""
    version = StrategyVersion(
        strategy_id=sample_strategy.id,
        workspace_id=sample_workspace.id,
        version="1.0.0",
        git_sha="a" * 40,
        git_tag="v1.0.0",
        changelog="Initial release",
        is_latest=True,
    )
    db_session.add(version)
    await db_session.commit()
    await db_session.refresh(version)
    return version


@pytest.fixture
async def sample_build(
    db_session: AsyncSession,
    sample_version: StrategyVersion,
    sample_workspace: Workspace,
) -> Build:
    """Create a sample build for testing."""
    build = Build(
        strategy_version_id=sample_version.id,
        workspace_id=sample_workspace.id,
        build_number=1,
        builder_id="builder-001",
        status="success",
        started_at=datetime.now(timezone.utc),
    )
    db_session.add(build)
    await db_session.commit()
    await db_session.refresh(build)
    return build


@pytest.fixture
async def sample_artifact(
    db_session: AsyncSession,
    sample_build: Build,
    sample_workspace: Workspace,
) -> Artifact:
    """Create a sample artifact for testing."""
    artifact = Artifact(
        build_id=sample_build.id,
        workspace_id=sample_workspace.id,
        name="model.pt",
        format="pytorch",
        digest="b" * 64,
        size_bytes=1024000,
        registry_url="registry.example.com/model:v1.0.0",
        change_class=ChangeClass.OPERATIONAL,
        is_signed=False,
    )
    db_session.add(artifact)
    await db_session.commit()
    await db_session.refresh(artifact)
    return artifact


# =============================================================================
# Strategy Tests
# =============================================================================


class TestListStrategies:
    """Tests for GET /strategies endpoint."""

    async def test_list_strategies_in_workspace(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_strategy: Strategy,
    ) -> None:
        """User can list strategies in their workspace."""
        response = await client.get(
            "/api/v1/strategies",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert data["total"] >= 1

        # Find our strategy
        found = False
        for item in data["items"]:
            if item["name"] == "test-strategy":
                found = True
                assert item["workspace_id"] is not None
                break
        assert found

    async def test_list_strategies_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_strategy: Strategy,
    ) -> None:
        """Superuser can list all strategies."""
        response = await client.get(
            "/api/v1/strategies",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1

    async def test_list_strategies_superuser_filter_workspace(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_strategy: Strategy,
        workspace_id,
    ) -> None:
        """Superuser can filter by workspace_id."""
        response = await client.get(
            f"/api/v1/strategies?workspace_id={workspace_id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert item["workspace_id"] == str(workspace_id)

    async def test_list_strategies_unauthenticated(
        self,
        client: AsyncClient,
    ) -> None:
        """Unauthenticated request returns 401."""
        response = await client.get("/api/v1/strategies")
        assert response.status_code == 401

    async def test_list_strategies_includes_version_count(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_strategy: Strategy,
        sample_version: StrategyVersion,
    ) -> None:
        """Response includes version count."""
        response = await client.get(
            "/api/v1/strategies",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        for item in data["items"]:
            if item["name"] == "test-strategy":
                assert item["version_count"] == 1
                assert item["latest_version"] == "1.0.0"
                break


class TestCreateStrategy:
    """Tests for POST /strategies endpoint."""

    async def test_create_strategy_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_workspace: Workspace,
    ) -> None:
        """Superuser can create a strategy."""
        response = await client.post(
            "/api/v1/strategies",
            headers=superuser_headers,
            json={
                "name": "new-strategy",
                "description": "A new strategy",
                "workspace_id": str(sample_workspace.id),
                "git_repo": "https://github.com/test/new-strategy.git",
                "tags": ["new"],
                "metadata": {"type": "ml"},
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "new-strategy"
        assert data["description"] == "A new strategy"
        assert data["workspace_id"] == str(sample_workspace.id)
        assert data["tags"] == ["new"]
        assert data["is_active"] is True
        assert data["version_count"] == 0

    async def test_create_strategy_with_permission(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_user: User,
        sample_workspace: Workspace,
        sample_organization: Organization,
    ) -> None:
        """User with strategy:create can create strategy."""
        from ..routers.auth import create_access_token

        # Create permission and role
        perm = Permission(name="strategy:create", description="Create strategy")
        db_session.add(perm)
        await db_session.commit()

        role = Role(name="strat-creator", description="Strategy Creator", organization_id=sample_organization.id)
        role.permissions.append(perm)
        db_session.add(role)
        await db_session.commit()

        # Refresh user to access roles relationship
        await db_session.refresh(sample_user, attribute_names=["roles"])
        sample_user.roles.append(role)
        await db_session.commit()

        token = create_access_token(
            user_id=sample_user.id,
            email=sample_user.email,
            org_id=sample_organization.id,
            workspace_id=sample_workspace.id,
            is_superuser=False,
            permissions=["strategy:create"],
        )

        response = await client.post(
            "/api/v1/strategies",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "name": "user-created-strategy",
                "workspace_id": str(sample_workspace.id),
            },
        )

        assert response.status_code == 201
        assert response.json()["name"] == "user-created-strategy"

    async def test_create_strategy_without_permission_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        workspace_id,
    ) -> None:
        """User without strategy:create cannot create strategy."""
        response = await client.post(
            "/api/v1/strategies",
            headers=auth_headers,
            json={
                "name": "forbidden-strategy",
                "workspace_id": str(workspace_id),
            },
        )

        assert response.status_code == 403
        assert "strategy:create" in response.json()["detail"]

    async def test_create_strategy_other_workspace_forbidden(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        user_id,
        workspace_id,
        org_id,
    ) -> None:
        """Cannot create strategy in another workspace."""
        from sqlalchemy import select
        from ..routers.auth import create_access_token

        # Create another workspace
        other_ws = Workspace(name="other-ws-for-strat", organization_id=org_id)
        db_session.add(other_ws)
        await db_session.commit()
        await db_session.refresh(other_ws)

        # User with permission but for different workspace
        perm = Permission(name="strategy:create:other", description="Create")
        db_session.add(perm)
        await db_session.commit()

        token = create_access_token(
            user_id=user_id,
            email="test@example.com",
            org_id=org_id,
            workspace_id=workspace_id,
            is_superuser=False,
            permissions=["strategy:create"],
        )

        response = await client.post(
            "/api/v1/strategies",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "name": "cross-ws-strategy",
                "workspace_id": str(other_ws.id),  # Different workspace
            },
        )

        assert response.status_code == 403
        assert "another workspace" in response.json()["detail"]

    async def test_create_strategy_workspace_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Creating strategy in non-existent workspace returns 404."""
        response = await client.post(
            "/api/v1/strategies",
            headers=superuser_headers,
            json={
                "name": "orphan-strategy",
                "workspace_id": str(uuid4()),
            },
        )

        assert response.status_code == 404
        assert "Workspace not found" in response.json()["detail"]

    async def test_create_strategy_duplicate_name(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_strategy: Strategy,
        workspace_id,
    ) -> None:
        """Cannot create strategy with duplicate name in same workspace."""
        response = await client.post(
            "/api/v1/strategies",
            headers=superuser_headers,
            json={
                "name": sample_strategy.name,
                "workspace_id": str(workspace_id),
            },
        )

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]


class TestGetStrategy:
    """Tests for GET /strategies/{strategy_id} endpoint."""

    async def test_get_strategy_success(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_strategy: Strategy,
    ) -> None:
        """User can get strategy in their workspace."""
        response = await client.get(
            f"/api/v1/strategies/{sample_strategy.id}",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_strategy.id)
        assert data["name"] == sample_strategy.name

    async def test_get_strategy_not_found(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ) -> None:
        """Non-existent strategy returns 404."""
        response = await client.get(
            f"/api/v1/strategies/{uuid4()}",
            headers=auth_headers,
        )

        assert response.status_code == 404
        assert "Strategy not found" in response.json()["detail"]

    async def test_get_strategy_other_workspace_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
    ) -> None:
        """Cannot get strategy from another workspace."""
        # Create strategy in different organization
        other_org = Organization(name="other-org-strat", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        other_ws = Workspace(name="other-ws-strat", organization_id=other_org.id)
        db_session.add(other_ws)
        await db_session.commit()
        await db_session.refresh(other_ws)

        other_strategy = Strategy(
            name="other-strategy",
            workspace_id=other_ws.id,
        )
        db_session.add(other_strategy)
        await db_session.commit()
        await db_session.refresh(other_strategy)

        response = await client.get(
            f"/api/v1/strategies/{other_strategy.id}",
            headers=auth_headers,
        )

        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]


class TestUpdateStrategy:
    """Tests for PATCH /strategies/{strategy_id} endpoint."""

    async def test_update_strategy_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_strategy: Strategy,
    ) -> None:
        """Superuser can update strategy."""
        response = await client.patch(
            f"/api/v1/strategies/{sample_strategy.id}",
            headers=superuser_headers,
            json={
                "name": "updated-name",
                "description": "Updated description",
                "tags": ["updated"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "updated-name"
        assert data["description"] == "Updated description"
        assert data["tags"] == ["updated"]

    async def test_update_strategy_with_permission(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_strategy: Strategy,
        sample_user: User,
        sample_workspace: Workspace,
        sample_organization: Organization,
    ) -> None:
        """User with strategy:write can update strategy."""
        from ..routers.auth import create_access_token

        perm = Permission(name="strategy:write", description="Write strategy")
        db_session.add(perm)
        await db_session.commit()

        role = Role(name="strat-writer", description="Writer", organization_id=sample_organization.id)
        role.permissions.append(perm)
        db_session.add(role)
        await db_session.commit()

        # Refresh user to access roles relationship
        await db_session.refresh(sample_user, attribute_names=["roles"])
        sample_user.roles.append(role)
        await db_session.commit()

        token = create_access_token(
            user_id=sample_user.id,
            email=sample_user.email,
            org_id=sample_organization.id,
            workspace_id=sample_workspace.id,
            is_superuser=False,
            permissions=["strategy:write"],
        )

        response = await client.patch(
            f"/api/v1/strategies/{sample_strategy.id}",
            headers={"Authorization": f"Bearer {token}"},
            json={"description": "Permission updated"},
        )

        assert response.status_code == 200
        assert response.json()["description"] == "Permission updated"

    async def test_update_strategy_without_permission_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_strategy: Strategy,
    ) -> None:
        """User without strategy:write cannot update strategy."""
        response = await client.patch(
            f"/api/v1/strategies/{sample_strategy.id}",
            headers=auth_headers,
            json={"name": "should-fail"},
        )

        assert response.status_code == 403
        assert "strategy:write" in response.json()["detail"]

    async def test_update_strategy_duplicate_name(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_strategy: Strategy,
        db_session: AsyncSession,
        sample_workspace: Workspace,
    ) -> None:
        """Cannot update to duplicate name."""
        # Create another strategy
        another = Strategy(name="another-strategy", workspace_id=sample_workspace.id)
        db_session.add(another)
        await db_session.commit()

        response = await client.patch(
            f"/api/v1/strategies/{sample_strategy.id}",
            headers=superuser_headers,
            json={"name": "another-strategy"},
        )

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]


class TestDeleteStrategy:
    """Tests for DELETE /strategies/{strategy_id} endpoint."""

    async def test_delete_strategy_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_workspace: Workspace,
    ) -> None:
        """Superuser can soft delete strategy."""
        # Create strategy to delete
        strategy = Strategy(name="delete-me", workspace_id=sample_workspace.id)
        db_session.add(strategy)
        await db_session.commit()
        await db_session.refresh(strategy)

        response = await client.delete(
            f"/api/v1/strategies/{strategy.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 204

        # Verify soft deleted
        await db_session.refresh(strategy)
        assert strategy.is_active is False

    async def test_delete_strategy_without_permission_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_strategy: Strategy,
    ) -> None:
        """User without strategy:delete cannot delete strategy."""
        response = await client.delete(
            f"/api/v1/strategies/{sample_strategy.id}",
            headers=auth_headers,
        )

        assert response.status_code == 403
        assert "strategy:delete" in response.json()["detail"]


# =============================================================================
# Strategy Version Tests
# =============================================================================


class TestListStrategyVersions:
    """Tests for GET /strategies/{strategy_id}/versions endpoint."""

    async def test_list_versions(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_strategy: Strategy,
        sample_version: StrategyVersion,
    ) -> None:
        """User can list versions of their strategy."""
        response = await client.get(
            f"/api/v1/strategies/{sample_strategy.id}/versions",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert any(v["version"] == "1.0.0" for v in data["items"])

    async def test_list_versions_include_deprecated(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_strategy: Strategy,
        db_session: AsyncSession,
        sample_workspace: Workspace,
    ) -> None:
        """Can include deprecated versions in listing."""
        # Create deprecated version
        deprecated = StrategyVersion(
            strategy_id=sample_strategy.id,
            workspace_id=sample_workspace.id,
            version="0.9.0",
            git_sha="c" * 40,
            is_deprecated=True,
        )
        db_session.add(deprecated)
        await db_session.commit()

        # Without include_deprecated
        response = await client.get(
            f"/api/v1/strategies/{sample_strategy.id}/versions",
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert not any(v["version"] == "0.9.0" for v in response.json()["items"])

        # With include_deprecated
        response = await client.get(
            f"/api/v1/strategies/{sample_strategy.id}/versions?include_deprecated=true",
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert any(v["version"] == "0.9.0" for v in response.json()["items"])

    async def test_list_versions_strategy_not_found(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ) -> None:
        """Listing versions of non-existent strategy returns 404."""
        response = await client.get(
            f"/api/v1/strategies/{uuid4()}/versions",
            headers=auth_headers,
        )

        assert response.status_code == 404
        assert "Strategy not found" in response.json()["detail"]


class TestCreateStrategyVersion:
    """Tests for POST /strategies/{strategy_id}/versions endpoint."""

    async def test_create_version_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_strategy: Strategy,
    ) -> None:
        """Superuser can create a new version."""
        response = await client.post(
            f"/api/v1/strategies/{sample_strategy.id}/versions",
            headers=superuser_headers,
            json={
                "version": "2.0.0",
                "git_sha": "d" * 40,
                "git_tag": "v2.0.0",
                "changelog": "Major release",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["version"] == "2.0.0"
        assert data["is_latest"] is True
        assert data["is_deprecated"] is False

    async def test_create_version_sets_previous_not_latest(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_strategy: Strategy,
        sample_version: StrategyVersion,
        db_session: AsyncSession,
    ) -> None:
        """Creating new version sets previous version to not latest."""
        assert sample_version.is_latest is True

        response = await client.post(
            f"/api/v1/strategies/{sample_strategy.id}/versions",
            headers=superuser_headers,
            json={
                "version": "2.0.0",
                "git_sha": "e" * 40,
            },
        )

        assert response.status_code == 201
        assert response.json()["is_latest"] is True

        # Verify previous version is no longer latest
        await db_session.refresh(sample_version)
        assert sample_version.is_latest is False

    async def test_create_version_duplicate(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_strategy: Strategy,
        sample_version: StrategyVersion,
    ) -> None:
        """Cannot create duplicate version."""
        response = await client.post(
            f"/api/v1/strategies/{sample_strategy.id}/versions",
            headers=superuser_headers,
            json={
                "version": sample_version.version,
                "git_sha": "f" * 40,
            },
        )

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    async def test_create_version_invalid_format(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_strategy: Strategy,
    ) -> None:
        """Version must match semver pattern."""
        response = await client.post(
            f"/api/v1/strategies/{sample_strategy.id}/versions",
            headers=superuser_headers,
            json={
                "version": "invalid",
                "git_sha": "g" * 40,
            },
        )

        assert response.status_code == 422


class TestGetStrategyVersion:
    """Tests for GET /strategies/{strategy_id}/versions/{version_id} endpoint."""

    async def test_get_version_success(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_strategy: Strategy,
        sample_version: StrategyVersion,
    ) -> None:
        """User can get version by ID."""
        response = await client.get(
            f"/api/v1/strategies/{sample_strategy.id}/versions/{sample_version.id}",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_version.id)
        assert data["version"] == "1.0.0"

    async def test_get_version_not_found(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_strategy: Strategy,
    ) -> None:
        """Non-existent version returns 404."""
        response = await client.get(
            f"/api/v1/strategies/{sample_strategy.id}/versions/{uuid4()}",
            headers=auth_headers,
        )

        assert response.status_code == 404


class TestUpdateStrategyVersion:
    """Tests for PATCH /strategies/{strategy_id}/versions/{version_id} endpoint."""

    async def test_update_version_deprecate(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_strategy: Strategy,
        sample_version: StrategyVersion,
    ) -> None:
        """Can deprecate a version."""
        response = await client.patch(
            f"/api/v1/strategies/{sample_strategy.id}/versions/{sample_version.id}",
            headers=superuser_headers,
            json={"is_deprecated": True},
        )

        assert response.status_code == 200
        assert response.json()["is_deprecated"] is True

    async def test_update_version_changelog(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_strategy: Strategy,
        sample_version: StrategyVersion,
    ) -> None:
        """Can update version changelog."""
        response = await client.patch(
            f"/api/v1/strategies/{sample_strategy.id}/versions/{sample_version.id}",
            headers=superuser_headers,
            json={"changelog": "Updated changelog"},
        )

        assert response.status_code == 200
        assert response.json()["changelog"] == "Updated changelog"


# =============================================================================
# Build Tests
# =============================================================================


class TestListBuilds:
    """Tests for GET /strategies/{strategy_id}/versions/{version_id}/builds endpoint."""

    async def test_list_builds(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_strategy: Strategy,
        sample_version: StrategyVersion,
        sample_build: Build,
    ) -> None:
        """User can list builds for a version."""
        response = await client.get(
            f"/api/v1/strategies/{sample_strategy.id}/versions/{sample_version.id}/builds",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert any(b["build_number"] == 1 for b in data["items"])

    async def test_list_builds_filter_status(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_strategy: Strategy,
        sample_version: StrategyVersion,
        sample_build: Build,
    ) -> None:
        """Can filter builds by status."""
        response = await client.get(
            f"/api/v1/strategies/{sample_strategy.id}/versions/{sample_version.id}/builds?status_filter=success",
            headers=auth_headers,
        )

        assert response.status_code == 200
        for build in response.json()["items"]:
            assert build["status"] == "success"


class TestCreateBuild:
    """Tests for POST /strategies/{strategy_id}/versions/{version_id}/builds endpoint."""

    async def test_create_build_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_strategy: Strategy,
        sample_version: StrategyVersion,
    ) -> None:
        """Superuser can create a build."""
        response = await client.post(
            f"/api/v1/strategies/{sample_strategy.id}/versions/{sample_version.id}/builds",
            headers=superuser_headers,
            json={
                "builder_id": "builder-002",
                "ci_job_id": "job-123",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["builder_id"] == "builder-002"
        assert data["status"] == "pending"
        assert data["build_number"] >= 1

    async def test_create_build_auto_increment_build_number(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_strategy: Strategy,
        sample_version: StrategyVersion,
        sample_build: Build,
    ) -> None:
        """Build number auto-increments."""
        response = await client.post(
            f"/api/v1/strategies/{sample_strategy.id}/versions/{sample_version.id}/builds",
            headers=superuser_headers,
            json={},
        )

        assert response.status_code == 201
        assert response.json()["build_number"] == sample_build.build_number + 1

    async def test_create_build_deprecated_version_error(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_strategy: Strategy,
        sample_version: StrategyVersion,
        db_session: AsyncSession,
    ) -> None:
        """Cannot build deprecated version."""
        sample_version.is_deprecated = True
        await db_session.commit()

        response = await client.post(
            f"/api/v1/strategies/{sample_strategy.id}/versions/{sample_version.id}/builds",
            headers=superuser_headers,
            json={},
        )

        assert response.status_code == 400
        assert "deprecated" in response.json()["detail"]

    async def test_create_build_without_permission_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_strategy: Strategy,
        sample_version: StrategyVersion,
    ) -> None:
        """User without build:create cannot create build."""
        response = await client.post(
            f"/api/v1/strategies/{sample_strategy.id}/versions/{sample_version.id}/builds",
            headers=auth_headers,
            json={},
        )

        assert response.status_code == 403
        assert "build:create" in response.json()["detail"]


class TestGetBuild:
    """Tests for GET /strategies/{strategy_id}/versions/{version_id}/builds/{build_id} endpoint."""

    async def test_get_build_success(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_strategy: Strategy,
        sample_version: StrategyVersion,
        sample_build: Build,
    ) -> None:
        """User can get build by ID."""
        response = await client.get(
            f"/api/v1/strategies/{sample_strategy.id}/versions/{sample_version.id}/builds/{sample_build.id}",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_build.id)
        assert data["build_number"] == 1

    async def test_get_build_not_found(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_strategy: Strategy,
        sample_version: StrategyVersion,
    ) -> None:
        """Non-existent build returns 404."""
        response = await client.get(
            f"/api/v1/strategies/{sample_strategy.id}/versions/{sample_version.id}/builds/{uuid4()}",
            headers=auth_headers,
        )

        assert response.status_code == 404


class TestUpdateBuild:
    """Tests for PATCH /strategies/{strategy_id}/versions/{version_id}/builds/{build_id} endpoint."""

    async def test_update_build_status(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_strategy: Strategy,
        sample_version: StrategyVersion,
        sample_build: Build,
    ) -> None:
        """Superuser can update build status."""
        response = await client.patch(
            f"/api/v1/strategies/{sample_strategy.id}/versions/{sample_version.id}/builds/{sample_build.id}",
            headers=superuser_headers,
            json={
                "status": "failed",
                "logs_url": "https://logs.example.com/build/1",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["logs_url"] == "https://logs.example.com/build/1"

    async def test_update_build_invalid_status(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_strategy: Strategy,
        sample_version: StrategyVersion,
        sample_build: Build,
    ) -> None:
        """Invalid status returns 422."""
        response = await client.patch(
            f"/api/v1/strategies/{sample_strategy.id}/versions/{sample_version.id}/builds/{sample_build.id}",
            headers=superuser_headers,
            json={"status": "invalid-status"},
        )

        assert response.status_code == 422


# =============================================================================
# Artifact Tests
# =============================================================================


class TestListArtifacts:
    """Tests for GET /strategies/{strategy_id}/versions/{version_id}/builds/{build_id}/artifacts endpoint."""

    async def test_list_artifacts(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_strategy: Strategy,
        sample_version: StrategyVersion,
        sample_build: Build,
        sample_artifact: Artifact,
    ) -> None:
        """User can list artifacts for a build."""
        response = await client.get(
            f"/api/v1/strategies/{sample_strategy.id}/versions/{sample_version.id}/builds/{sample_build.id}/artifacts",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert any(a["name"] == "model.pt" for a in data["items"])


class TestCreateArtifact:
    """Tests for POST /strategies/{strategy_id}/versions/{version_id}/builds/{build_id}/artifacts endpoint."""

    async def test_create_artifact_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_strategy: Strategy,
        sample_version: StrategyVersion,
        sample_build: Build,
    ) -> None:
        """Superuser can create an artifact."""
        response = await client.post(
            f"/api/v1/strategies/{sample_strategy.id}/versions/{sample_version.id}/builds/{sample_build.id}/artifacts",
            headers=superuser_headers,
            json={
                "name": "config.yaml",
                "format": "yaml",
                "digest": "h" * 64,
                "size_bytes": 1024,
                "registry_url": "registry.example.com/config:v1",
                "change_class": "OPERATIONAL",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "config.yaml"
        assert data["format"] == "yaml"
        assert data["is_signed"] is False

    async def test_create_artifact_with_signature(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_strategy: Strategy,
        sample_version: StrategyVersion,
        sample_build: Build,
    ) -> None:
        """Artifact with signature_ref is marked as signed."""
        response = await client.post(
            f"/api/v1/strategies/{sample_strategy.id}/versions/{sample_version.id}/builds/{sample_build.id}/artifacts",
            headers=superuser_headers,
            json={
                "name": "signed-model.pt",
                "format": "pytorch",
                "digest": "i" * 64,
                "size_bytes": 2048,
                "registry_url": "registry.example.com/signed:v1",
                "signature_ref": "sigstore://signed-model.sig",
                "change_class": "SECURITY_SENSITIVE",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["is_signed"] is True
        assert data["change_class"] == "security_sensitive"

    async def test_create_artifact_duplicate_digest(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_strategy: Strategy,
        sample_version: StrategyVersion,
        sample_build: Build,
        sample_artifact: Artifact,
    ) -> None:
        """Cannot create artifact with duplicate digest in same build."""
        response = await client.post(
            f"/api/v1/strategies/{sample_strategy.id}/versions/{sample_version.id}/builds/{sample_build.id}/artifacts",
            headers=superuser_headers,
            json={
                "name": "duplicate.pt",
                "format": "pytorch",
                "digest": sample_artifact.digest,
                "size_bytes": 1024,
                "registry_url": "registry.example.com/dup:v1",
            },
        )

        assert response.status_code == 409
        assert "digest already exists" in response.json()["detail"]

    async def test_create_artifact_without_permission_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_strategy: Strategy,
        sample_version: StrategyVersion,
        sample_build: Build,
    ) -> None:
        """User without artifact:create cannot create artifact."""
        response = await client.post(
            f"/api/v1/strategies/{sample_strategy.id}/versions/{sample_version.id}/builds/{sample_build.id}/artifacts",
            headers=auth_headers,
            json={
                "name": "forbidden.pt",
                "format": "pytorch",
                "digest": "j" * 64,
                "size_bytes": 512,
                "registry_url": "registry.example.com/forbidden:v1",
            },
        )

        assert response.status_code == 403
        assert "artifact:create" in response.json()["detail"]


class TestGetArtifact:
    """Tests for GET /strategies/{strategy_id}/versions/{version_id}/builds/{build_id}/artifacts/{artifact_id} endpoint."""

    async def test_get_artifact_success(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_strategy: Strategy,
        sample_version: StrategyVersion,
        sample_build: Build,
        sample_artifact: Artifact,
    ) -> None:
        """User can get artifact by ID."""
        response = await client.get(
            f"/api/v1/strategies/{sample_strategy.id}/versions/{sample_version.id}/builds/{sample_build.id}/artifacts/{sample_artifact.id}",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_artifact.id)
        assert data["name"] == "model.pt"

    async def test_get_artifact_not_found(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_strategy: Strategy,
        sample_version: StrategyVersion,
        sample_build: Build,
    ) -> None:
        """Non-existent artifact returns 404."""
        response = await client.get(
            f"/api/v1/strategies/{sample_strategy.id}/versions/{sample_version.id}/builds/{sample_build.id}/artifacts/{uuid4()}",
            headers=auth_headers,
        )

        assert response.status_code == 404
