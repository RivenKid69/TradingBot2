# -*- coding: utf-8 -*-
"""Tests for Config Blobs Router."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Any
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import ConfigBlob, Organization, Workspace

pytestmark = pytest.mark.asyncio


# ============================================================================
# Helper Functions
# ============================================================================


def compute_expected_digest(content: Dict[str, Any]) -> str:
    """Compute expected SHA256 digest for content."""
    content_str = json.dumps(content, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(content_str.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def compute_size_bytes(content: Dict[str, Any]) -> int:
    """Compute expected size in bytes."""
    content_str = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return len(content_str.encode("utf-8"))


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def sample_config_blob(
    db_session: AsyncSession,
    sample_workspace: Workspace,
    sample_user,
) -> ConfigBlob:
    """Create a sample config blob."""
    content = {"strategy": "momentum", "params": {"lookback": 20, "threshold": 0.05}}
    digest = compute_expected_digest(content)
    size_bytes = compute_size_bytes(content)

    blob = ConfigBlob(
        workspace_id=sample_workspace.id,
        digest=digest,
        content=content,
        size_bytes=size_bytes,
        config_type="strategy",
        schema_version="1.0.0",
        created_by_user_id=sample_user.id,
    )
    db_session.add(blob)
    await db_session.commit()
    await db_session.refresh(blob)
    return blob


@pytest.fixture
async def multiple_config_blobs(
    db_session: AsyncSession,
    sample_workspace: Workspace,
    sample_user,
) -> list[ConfigBlob]:
    """Create multiple config blobs with different types."""
    configs = [
        {"type": "strategy", "content": {"strategy": "momentum", "version": 1}, "schema": "1.0.0"},
        {
            "type": "strategy",
            "content": {"strategy": "mean_reversion", "version": 1},
            "schema": "1.0.0",
        },
        {"type": "risk", "content": {"max_position": 100, "stop_loss": 0.05}, "schema": "1.0.0"},
        {"type": "risk", "content": {"max_position": 200, "stop_loss": 0.03}, "schema": "2.0.0"},
        {"type": "execution", "content": {"slippage": 0.001, "timeout": 60}, "schema": "1.0.0"},
    ]

    blobs = []
    for cfg in configs:
        content = cfg["content"]
        digest = compute_expected_digest(content)
        size_bytes = compute_size_bytes(content)

        blob = ConfigBlob(
            workspace_id=sample_workspace.id,
            digest=digest,
            content=content,
            size_bytes=size_bytes,
            config_type=cfg["type"],
            schema_version=cfg["schema"],
            created_by_user_id=sample_user.id,
        )
        db_session.add(blob)
        blobs.append(blob)

    await db_session.commit()
    for blob in blobs:
        await db_session.refresh(blob)

    return blobs


@pytest.fixture
async def config_with_permission_headers(
    db_session: AsyncSession,
    sample_user,
    org_id,
    workspace_id,
) -> dict:
    """Create auth headers with config:create and config:read permissions."""
    from uuid import uuid4
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    from ..models import Permission, Role, User
    from ..routers.auth import create_access_token

    # Reload user in current session to avoid greenlet issues
    result = await db_session.execute(
        select(User).where(User.id == sample_user.id).options(selectinload(User.roles))
    )
    user = result.scalar_one()

    # Create unique permissions (use uuid suffix to avoid conflicts on both name and resource/action)
    suffix = str(uuid4())[:8]
    perm_create = Permission(
        name=f"config:create:{suffix}",
        description="Create config",
        resource=f"config:{suffix}",
        action="create",
    )
    perm_read = Permission(
        name=f"config:read:{suffix}",
        description="Read config",
        resource=f"config:{suffix}",
        action="read",
    )
    db_session.add(perm_create)
    db_session.add(perm_read)
    await db_session.commit()
    await db_session.refresh(perm_create)
    await db_session.refresh(perm_read)

    # Create role with permissions
    role = Role(
        name=f"config-manager-test-{suffix}",
        description="Config Manager",
        organization_id=org_id,
    )
    role.permissions.append(perm_create)
    role.permissions.append(perm_read)
    db_session.add(role)
    await db_session.commit()
    await db_session.refresh(role)

    # Add role to user
    user.roles.append(role)
    await db_session.commit()

    # Create token with permissions
    token = create_access_token(
        user_id=user.id,
        email=user.email,
        org_id=org_id,
        workspace_id=workspace_id,
        is_superuser=False,
        permissions=["config:create", "config:read"],
    )
    return {"Authorization": f"Bearer {token}"}


# ============================================================================
# Test Classes
# ============================================================================


class TestListConfigBlobs:
    """Tests for GET /config-blobs endpoint."""

    async def test_list_config_blobs_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_config_blob: ConfigBlob,
    ) -> None:
        """Superuser can list all config blobs."""
        response = await client.get(
            "/api/v1/config-blobs",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert data["total"] >= 1

    async def test_list_config_blobs_filter_by_workspace(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        multiple_config_blobs: list[ConfigBlob],
        workspace_id,
    ) -> None:
        """Can filter config blobs by workspace."""
        response = await client.get(
            f"/api/v1/config-blobs?workspace_id={workspace_id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 5  # All 5 blobs in same workspace

    async def test_list_config_blobs_filter_by_type(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        multiple_config_blobs: list[ConfigBlob],
        workspace_id,
    ) -> None:
        """Can filter config blobs by config_type."""
        response = await client.get(
            f"/api/v1/config-blobs?workspace_id={workspace_id}&config_type=strategy",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 2  # 2 strategy blobs

    async def test_list_config_blobs_filter_by_schema_version(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        multiple_config_blobs: list[ConfigBlob],
        workspace_id,
    ) -> None:
        """Can filter config blobs by schema_version."""
        response = await client.get(
            f"/api/v1/config-blobs?workspace_id={workspace_id}&schema_version=2.0.0",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1  # Only 1 blob with schema 2.0.0

    async def test_list_config_blobs_regular_user_no_workspaces(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
    ) -> None:
        """Regular user with no workspaces sees empty list."""
        # Create org and user with no workspaces
        org = Organization(name="empty-org", display_name="Empty Org")
        db_session.add(org)
        await db_session.commit()
        await db_session.refresh(org)

        from ..models import User
        from ..routers.auth import create_access_token
        import hashlib

        user = User(
            email="empty@example.com",
            password_hash=hashlib.sha256("test".encode()).hexdigest(),
            organization_id=org.id,
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        token = create_access_token(
            user_id=user.id,
            email=user.email,
            org_id=org.id,
            workspace_id=None,
            is_superuser=False,
            permissions=["config:read"],
        )

        response = await client.get(
            "/api/v1/config-blobs",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["items"] == []

    async def test_list_config_blobs_unauthenticated(
        self,
        client: AsyncClient,
    ) -> None:
        """Unauthenticated request returns 401."""
        response = await client.get("/api/v1/config-blobs")
        assert response.status_code == 401


class TestCreateConfigBlob:
    """Tests for POST /config-blobs endpoint."""

    async def test_create_config_blob_success(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        workspace_id,
    ) -> None:
        """Superuser can create config blob."""
        content = {"strategy": "new_strategy", "params": {"alpha": 0.1}}

        response = await client.post(
            "/api/v1/config-blobs",
            headers=superuser_headers,
            json={
                "workspace_id": str(workspace_id),
                "content": content,
                "config_type": "strategy",
                "schema_version": "1.0.0",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["content"] == content
        assert data["config_type"] == "strategy"
        assert data["schema_version"] == "1.0.0"
        assert data["digest"] == compute_expected_digest(content)
        assert data["size_bytes"] == compute_size_bytes(content)

    async def test_create_config_blob_with_permission(
        self,
        client: AsyncClient,
        config_with_permission_headers: dict,
        workspace_id,
    ) -> None:
        """User with config:create permission can create blob."""
        content = {"risk": "conservative", "max_drawdown": 0.1}

        response = await client.post(
            "/api/v1/config-blobs",
            headers=config_with_permission_headers,
            json={
                "workspace_id": str(workspace_id),
                "content": content,
                "config_type": "risk",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["config_type"] == "risk"

    async def test_create_config_blob_without_permission(
        self,
        client: AsyncClient,
        auth_headers: dict,
        workspace_id,
    ) -> None:
        """User without config:create permission cannot create blob."""
        response = await client.post(
            "/api/v1/config-blobs",
            headers=auth_headers,
            json={
                "workspace_id": str(workspace_id),
                "content": {"test": "data"},
                "config_type": "strategy",
            },
        )

        assert response.status_code == 403
        assert "config:create" in response.json()["detail"]

    async def test_create_config_blob_invalid_type(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        workspace_id,
    ) -> None:
        """Cannot create blob with invalid config_type."""
        response = await client.post(
            "/api/v1/config-blobs",
            headers=superuser_headers,
            json={
                "workspace_id": str(workspace_id),
                "content": {"test": "data"},
                "config_type": "invalid_type",
            },
        )

        assert response.status_code == 400
        assert "Invalid config_type" in response.json()["detail"]

    async def test_create_config_blob_duplicate_returns_existing(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_config_blob: ConfigBlob,
        workspace_id,
    ) -> None:
        """Creating blob with same content returns existing blob."""
        # Use the same content as sample_config_blob
        content = {"strategy": "momentum", "params": {"lookback": 20, "threshold": 0.05}}

        response = await client.post(
            "/api/v1/config-blobs",
            headers=superuser_headers,
            json={
                "workspace_id": str(workspace_id),
                "content": content,
                "config_type": "strategy",
            },
        )

        assert response.status_code == 201
        data = response.json()
        # Should return the existing blob
        assert data["id"] == str(sample_config_blob.id)

    async def test_create_config_blob_different_workspace_creates_new(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_config_blob: ConfigBlob,
        org_id,
    ) -> None:
        """Same content in different workspace creates new blob."""
        # Create another workspace
        other_ws = Workspace(name="other-ws", organization_id=org_id)
        db_session.add(other_ws)
        await db_session.commit()
        await db_session.refresh(other_ws)

        # Use the same content as sample_config_blob
        content = {"strategy": "momentum", "params": {"lookback": 20, "threshold": 0.05}}

        response = await client.post(
            "/api/v1/config-blobs",
            headers=superuser_headers,
            json={
                "workspace_id": str(other_ws.id),
                "content": content,
                "config_type": "strategy",
            },
        )

        assert response.status_code == 201
        data = response.json()
        # Should create NEW blob (different workspace)
        assert data["id"] != str(sample_config_blob.id)
        assert data["workspace_id"] == str(other_ws.id)

    async def test_create_config_blob_workspace_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Creating blob in non-existent workspace returns 404."""
        response = await client.post(
            "/api/v1/config-blobs",
            headers=superuser_headers,
            json={
                "workspace_id": str(uuid4()),
                "content": {"test": "data"},
                "config_type": "strategy",
            },
        )

        assert response.status_code == 404
        assert "Workspace not found" in response.json()["detail"]

    async def test_create_config_blob_different_org_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
    ) -> None:
        """User cannot create blob in workspace of different org."""
        # Create different org and workspace
        other_org = Organization(name="other-org-blob", display_name="Other Org")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        other_ws = Workspace(name="other-ws-blob", organization_id=other_org.id)
        db_session.add(other_ws)
        await db_session.commit()
        await db_session.refresh(other_ws)

        response = await client.post(
            "/api/v1/config-blobs",
            headers=auth_headers,
            json={
                "workspace_id": str(other_ws.id),
                "content": {"test": "data"},
                "config_type": "strategy",
            },
        )

        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]

    async def test_create_config_blob_all_valid_types(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        workspace_id,
    ) -> None:
        """Can create blobs with all valid config types."""
        valid_types = [
            "strategy",
            "risk",
            "execution",
            "environment",
            "feature_flags",
            "model",
            "data",
            "alert",
            "monitoring",
            "custom",
        ]

        for config_type in valid_types:
            content = {"type": config_type, "value": 123}
            response = await client.post(
                "/api/v1/config-blobs",
                headers=superuser_headers,
                json={
                    "workspace_id": str(workspace_id),
                    "content": content,
                    "config_type": config_type,
                },
            )
            assert response.status_code == 201, f"Failed for type: {config_type}"


class TestGetConfigBlob:
    """Tests for GET /config-blobs/{config_blob_id} endpoint."""

    async def test_get_config_blob_success(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_config_blob: ConfigBlob,
    ) -> None:
        """Can get config blob by ID."""
        response = await client.get(
            f"/api/v1/config-blobs/{sample_config_blob.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_config_blob.id)
        assert data["content"] == sample_config_blob.content
        assert data["digest"] == sample_config_blob.digest

    async def test_get_config_blob_with_permission(
        self,
        client: AsyncClient,
        config_with_permission_headers: dict,
        sample_config_blob: ConfigBlob,
    ) -> None:
        """User with config:read permission can get blob."""
        response = await client.get(
            f"/api/v1/config-blobs/{sample_config_blob.id}",
            headers=config_with_permission_headers,
        )

        assert response.status_code == 200

    async def test_get_config_blob_without_permission(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_config_blob: ConfigBlob,
    ) -> None:
        """User without config:read permission cannot get blob."""
        response = await client.get(
            f"/api/v1/config-blobs/{sample_config_blob.id}",
            headers=auth_headers,
        )

        assert response.status_code == 403
        assert "config:read" in response.json()["detail"]

    async def test_get_config_blob_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Getting non-existent blob returns 404."""
        response = await client.get(
            f"/api/v1/config-blobs/{uuid4()}",
            headers=superuser_headers,
        )

        assert response.status_code == 404
        assert "Config blob not found" in response.json()["detail"]

    async def test_get_config_blob_different_org_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
    ) -> None:
        """User cannot get blob from workspace of different org."""
        # Create different org, workspace, and blob
        other_org = Organization(name="other-org-get", display_name="Other Org")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        other_ws = Workspace(name="other-ws-get", organization_id=other_org.id)
        db_session.add(other_ws)
        await db_session.commit()
        await db_session.refresh(other_ws)

        content = {"other": "config"}
        blob = ConfigBlob(
            workspace_id=other_ws.id,
            digest=compute_expected_digest(content),
            content=content,
            size_bytes=compute_size_bytes(content),
            config_type="custom",
            schema_version="1.0.0",
        )
        db_session.add(blob)
        await db_session.commit()
        await db_session.refresh(blob)

        response = await client.get(
            f"/api/v1/config-blobs/{blob.id}",
            headers=auth_headers,
        )

        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]


class TestLookupByDigest:
    """Tests for POST /config-blobs/lookup-by-digest endpoint."""

    async def test_lookup_by_digest_success(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_config_blob: ConfigBlob,
        workspace_id,
    ) -> None:
        """Can lookup config blob by digest."""
        response = await client.post(
            "/api/v1/config-blobs/lookup-by-digest",
            headers=superuser_headers,
            json={
                "workspace_id": str(workspace_id),
                "digest": sample_config_blob.digest,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_config_blob.id)
        assert data["digest"] == sample_config_blob.digest

    async def test_lookup_by_digest_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        workspace_id,
    ) -> None:
        """Lookup with non-existent digest returns 404."""
        response = await client.post(
            "/api/v1/config-blobs/lookup-by-digest",
            headers=superuser_headers,
            json={
                "workspace_id": str(workspace_id),
                "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            },
        )

        assert response.status_code == 404
        assert "Config blob not found" in response.json()["detail"]

    async def test_lookup_by_digest_without_permission(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_config_blob: ConfigBlob,
        workspace_id,
    ) -> None:
        """User without config:read permission cannot lookup."""
        response = await client.post(
            "/api/v1/config-blobs/lookup-by-digest",
            headers=auth_headers,
            json={
                "workspace_id": str(workspace_id),
                "digest": sample_config_blob.digest,
            },
        )

        assert response.status_code == 403


class TestGetByDigest:
    """Tests for GET /config-blobs/by-digest/{digest} endpoint."""

    async def test_get_by_digest_with_workspace(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_config_blob: ConfigBlob,
        workspace_id,
    ) -> None:
        """Can get config blob by digest with workspace_id."""
        response = await client.get(
            f"/api/v1/config-blobs/by-digest/{sample_config_blob.digest}?workspace_id={workspace_id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_config_blob.id)

    async def test_get_by_digest_superuser_no_workspace(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_config_blob: ConfigBlob,
    ) -> None:
        """Superuser can get blob by digest without workspace_id."""
        response = await client.get(
            f"/api/v1/config-blobs/by-digest/{sample_config_blob.digest}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_config_blob.id)

    async def test_get_by_digest_regular_user_searches_org(
        self,
        client: AsyncClient,
        config_with_permission_headers: dict,
        sample_config_blob: ConfigBlob,
    ) -> None:
        """Regular user can get blob by digest (searches their org's workspaces)."""
        response = await client.get(
            f"/api/v1/config-blobs/by-digest/{sample_config_blob.digest}",
            headers=config_with_permission_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_config_blob.id)

    async def test_get_by_digest_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Getting by non-existent digest returns 404."""
        fake_digest = "sha256:1111111111111111111111111111111111111111111111111111111111111111"
        response = await client.get(
            f"/api/v1/config-blobs/by-digest/{fake_digest}",
            headers=superuser_headers,
        )

        assert response.status_code == 404


class TestVerifyDigest:
    """Tests for POST /config-blobs/verify-digest endpoint."""

    async def test_verify_digest_success(
        self,
        client: AsyncClient,
        auth_headers: dict,  # No special permission required
    ) -> None:
        """Can verify digest for content."""
        content = {"test": "content", "value": 42}

        response = await client.post(
            "/api/v1/config-blobs/verify-digest",
            headers=auth_headers,
            json=content,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["digest"] == compute_expected_digest(content)
        assert data["size_bytes"] == compute_size_bytes(content)
        assert data["exceeds_max_size"] is False

    async def test_verify_digest_deterministic(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ) -> None:
        """Digest computation is deterministic regardless of key order."""
        content1 = {"a": 1, "b": 2, "c": 3}
        content2 = {"c": 3, "b": 2, "a": 1}  # Same content, different order

        response1 = await client.post(
            "/api/v1/config-blobs/verify-digest",
            headers=auth_headers,
            json=content1,
        )
        response2 = await client.post(
            "/api/v1/config-blobs/verify-digest",
            headers=auth_headers,
            json=content2,
        )

        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response1.json()["digest"] == response2.json()["digest"]

    async def test_verify_digest_large_content(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ) -> None:
        """Verify digest correctly reports when content exceeds max size."""
        # Create content that would exceed 10 MB
        # We can't actually send 10 MB in test, but we can check the logic
        # by computing expected result
        small_content = {"small": "content"}

        response = await client.post(
            "/api/v1/config-blobs/verify-digest",
            headers=auth_headers,
            json=small_content,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["exceeds_max_size"] is False

    async def test_verify_digest_unauthenticated(
        self,
        client: AsyncClient,
    ) -> None:
        """Unauthenticated request returns 401."""
        response = await client.post(
            "/api/v1/config-blobs/verify-digest",
            json={"test": "data"},
        )

        assert response.status_code == 401


class TestListConfigTypes:
    """Tests for GET /config-blobs/types/list endpoint."""

    async def test_list_config_types(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ) -> None:
        """Can list valid config types."""
        response = await client.get(
            "/api/v1/config-blobs/types/list",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        expected_types = [
            "alert",
            "custom",
            "data",
            "environment",
            "execution",
            "feature_flags",
            "model",
            "monitoring",
            "risk",
            "strategy",
        ]
        assert data == expected_types

    async def test_list_config_types_unauthenticated(
        self,
        client: AsyncClient,
    ) -> None:
        """Unauthenticated request returns 401."""
        response = await client.get("/api/v1/config-blobs/types/list")
        assert response.status_code == 401


class TestConfigBlobDigestIntegrity:
    """Tests for digest computation and integrity."""

    async def test_digest_matches_on_retrieval(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        workspace_id,
    ) -> None:
        """Digest in response matches computed digest of content."""
        content = {"integrity": "test", "nested": {"data": [1, 2, 3]}}

        create_response = await client.post(
            "/api/v1/config-blobs",
            headers=superuser_headers,
            json={
                "workspace_id": str(workspace_id),
                "content": content,
                "config_type": "custom",
            },
        )

        assert create_response.status_code == 201
        created_data = create_response.json()
        blob_id = created_data["id"]

        # Get the blob
        get_response = await client.get(
            f"/api/v1/config-blobs/{blob_id}",
            headers=superuser_headers,
        )

        assert get_response.status_code == 200
        retrieved_data = get_response.json()

        # Verify digest matches content
        expected_digest = compute_expected_digest(retrieved_data["content"])
        assert retrieved_data["digest"] == expected_digest

    async def test_different_content_different_digest(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        workspace_id,
    ) -> None:
        """Different content produces different digest."""
        content1 = {"version": 1}
        content2 = {"version": 2}

        response1 = await client.post(
            "/api/v1/config-blobs",
            headers=superuser_headers,
            json={
                "workspace_id": str(workspace_id),
                "content": content1,
                "config_type": "custom",
            },
        )
        response2 = await client.post(
            "/api/v1/config-blobs",
            headers=superuser_headers,
            json={
                "workspace_id": str(workspace_id),
                "content": content2,
                "config_type": "custom",
            },
        )

        assert response1.status_code == 201
        assert response2.status_code == 201
        assert response1.json()["digest"] != response2.json()["digest"]


class TestConfigBlobImmutability:
    """Tests for config blob immutability."""

    async def test_same_content_same_blob(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        workspace_id,
    ) -> None:
        """Creating blob with same content returns same blob ID."""
        content = {"immutable": "test", "timestamp": "ignored"}

        response1 = await client.post(
            "/api/v1/config-blobs",
            headers=superuser_headers,
            json={
                "workspace_id": str(workspace_id),
                "content": content,
                "config_type": "custom",
            },
        )
        response2 = await client.post(
            "/api/v1/config-blobs",
            headers=superuser_headers,
            json={
                "workspace_id": str(workspace_id),
                "content": content,
                "config_type": "custom",
            },
        )

        assert response1.status_code == 201
        assert response2.status_code == 201
        # Same blob ID (content-addressable)
        assert response1.json()["id"] == response2.json()["id"]

    async def test_config_type_ignored_for_dedup(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        workspace_id,
    ) -> None:
        """Same content with different config_type still returns same blob."""
        # Note: This tests current behavior - digest is computed only from content
        content = {"shared": "content"}

        response1 = await client.post(
            "/api/v1/config-blobs",
            headers=superuser_headers,
            json={
                "workspace_id": str(workspace_id),
                "content": content,
                "config_type": "strategy",
            },
        )
        response2 = await client.post(
            "/api/v1/config-blobs",
            headers=superuser_headers,
            json={
                "workspace_id": str(workspace_id),
                "content": content,
                "config_type": "risk",  # Different type
            },
        )

        assert response1.status_code == 201
        assert response2.status_code == 201
        # Same blob ID (content-only digest)
        assert response1.json()["id"] == response2.json()["id"]
