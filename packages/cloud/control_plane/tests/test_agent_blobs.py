# -*- coding: utf-8 -*-
"""
Tests for Agent Blob/Artifact Access Router.

Design Doc 12.2 + 22.2: Agent-authenticated endpoints for blob and artifact access.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Dict
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import (
    Agent,
    Artifact,
    Build,
    ConfigBlob,
    Organization,
    Strategy,
    StrategyVersion,
    TrustState,
    User,
    Workspace,
)


class TestConfigBlobByDigest:
    """Tests for config blob lookup by digest."""

    @pytest_asyncio.fixture
    async def config_blob_with_known_digest(
        self,
        db_session: AsyncSession,
        sample_workspace: Workspace,
        sample_user: User,
    ) -> ConfigBlob:
        """Create a config blob with known digest."""
        content = {"strategy_params": {"lookback": 20, "threshold": 0.5}}
        content_str = json.dumps(content, sort_keys=True, separators=(",", ":"))
        digest = f"sha256:{hashlib.sha256(content_str.encode()).hexdigest()}"

        blob = ConfigBlob(
            workspace_id=sample_workspace.id,
            digest=digest,
            content=content,
            size_bytes=len(content_str.encode()),
            config_type="strategy",
            schema_version="1.0.0",
            created_by_user_id=sample_user.id,
        )
        db_session.add(blob)
        await db_session.commit()
        await db_session.refresh(blob)
        return blob

    @pytest.mark.asyncio
    async def test_get_config_blob_by_digest_success(
        self,
        client: AsyncClient,
        config_blob_with_known_digest: ConfigBlob,
        sample_agent: Agent,  # Required to create agent in DB
        agent_headers: Dict[str, str],
    ):
        """Agent should be able to retrieve config blob by digest."""
        digest = config_blob_with_known_digest.digest

        response = await client.get(
            f"/api/v1/agent/config-blobs/by-digest/{digest}",
            headers=agent_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["digest"] == digest
        assert data["content"] == config_blob_with_known_digest.content
        assert data["config_type"] == "strategy"

    @pytest.mark.asyncio
    async def test_get_config_blob_not_found(
        self,
        client: AsyncClient,
        sample_agent: Agent,  # Required to create agent in DB
        agent_headers: Dict[str, str],
    ):
        """Should return 404 for unknown digest."""
        response = await client.get(
            "/api/v1/agent/config-blobs/by-digest/sha256:nonexistent123",
            headers=agent_headers,
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_config_blob_requires_auth(
        self,
        client: AsyncClient,
        config_blob_with_known_digest: ConfigBlob,
    ):
        """Should require agent authentication."""
        digest = config_blob_with_known_digest.digest

        response = await client.get(
            f"/api/v1/agent/config-blobs/by-digest/{digest}",
            # No auth headers
        )

        assert response.status_code == 401


class TestArtifactDownloadUrl:
    """Tests for artifact download URL generation."""

    @pytest_asyncio.fixture
    async def artifact_with_known_digest(
        self,
        db_session: AsyncSession,
        sample_build: Build,
        sample_workspace: Workspace,
    ) -> Artifact:
        """Create an artifact with known digest."""
        artifact = Artifact(
            build_id=sample_build.id,
            workspace_id=sample_workspace.id,
            name="strategy-1.0.0.tar.gz",
            format="tar.gz",
            digest="sha256:artifact_test_digest_123",
            size_bytes=1024000,
            is_signed=True,
            signature_algorithm="ed25519",
            signature_ref="sigstore://fulcio.example.com/abc123",
        )
        db_session.add(artifact)
        await db_session.commit()
        await db_session.refresh(artifact)
        return artifact

    @pytest.mark.asyncio
    async def test_get_artifact_download_url_success(
        self,
        client: AsyncClient,
        artifact_with_known_digest: Artifact,
        sample_agent: Agent,  # Required to create agent in DB
        agent_headers: Dict[str, str],
    ):
        """Agent should be able to get download URL for artifact."""
        digest = artifact_with_known_digest.digest

        response = await client.get(
            f"/api/v1/agent/artifacts/by-digest/{digest}",
            headers=agent_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["digest"] == digest
        assert data["artifact_name"] == "strategy-1.0.0.tar.gz"
        assert "download_url" in data
        assert "download_url_expires_at" in data

        # Download URL should have signature
        assert "sig=" in data["download_url"]
        assert "expires=" in data["download_url"]

    @pytest.mark.asyncio
    async def test_artifact_download_url_has_expiry(
        self,
        client: AsyncClient,
        artifact_with_known_digest: Artifact,
        sample_agent: Agent,  # Required to create agent in DB
        agent_headers: Dict[str, str],
    ):
        """Download URL should expire in ~5 minutes."""
        digest = artifact_with_known_digest.digest

        response = await client.get(
            f"/api/v1/agent/artifacts/by-digest/{digest}",
            headers=agent_headers,
        )

        data = response.json()
        expires_at = datetime.fromisoformat(data["download_url_expires_at"].replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)

        # Should expire within 5-6 minutes
        delta = expires_at - now
        assert timedelta(minutes=4) < delta < timedelta(minutes=6)

    @pytest.mark.asyncio
    async def test_artifact_with_signature_info(
        self,
        client: AsyncClient,
        artifact_with_known_digest: Artifact,
        sample_agent: Agent,  # Required to create agent in DB
        agent_headers: Dict[str, str],
    ):
        """Artifact response should include signature info for verification."""
        digest = artifact_with_known_digest.digest

        response = await client.get(
            f"/api/v1/agent/artifacts/by-digest/{digest}",
            headers=agent_headers,
        )

        data = response.json()
        assert "signature_info" in data
        if data["signature_info"]:
            assert "algorithm" in data["signature_info"]
            assert "signature_ref" in data["signature_info"]
            assert "signed_digest" in data["signature_info"]

    @pytest.mark.asyncio
    async def test_artifact_not_found(
        self,
        client: AsyncClient,
        sample_agent: Agent,  # Required to create agent in DB
        agent_headers: Dict[str, str],
    ):
        """Should return 404 for unknown artifact digest."""
        response = await client.get(
            "/api/v1/agent/artifacts/by-digest/sha256:nonexistent",
            headers=agent_headers,
        )

        assert response.status_code == 404


class TestPayloadRefResolution:
    """Tests for payload reference resolution."""

    @pytest_asyncio.fixture
    async def config_blob_for_ref(
        self,
        db_session: AsyncSession,
        sample_workspace: Workspace,
        sample_user: User,
    ) -> ConfigBlob:
        """Create a config blob for ref resolution testing."""
        content = {"param1": "value1", "param2": 42}
        content_str = json.dumps(content, sort_keys=True, separators=(",", ":"))
        digest = f"sha256:{hashlib.sha256(content_str.encode()).hexdigest()}"

        blob = ConfigBlob(
            workspace_id=sample_workspace.id,
            digest=digest,
            content=content,
            size_bytes=len(content_str.encode()),
            config_type="strategy",
            schema_version="1.0.0",
            created_by_user_id=sample_user.id,
        )
        db_session.add(blob)
        await db_session.commit()
        await db_session.refresh(blob)
        return blob

    @pytest.mark.asyncio
    async def test_resolve_config_ref(
        self,
        client: AsyncClient,
        config_blob_for_ref: ConfigBlob,
        sample_agent: Agent,  # Required to create agent in DB
        agent_headers: Dict[str, str],
    ):
        """Should resolve config: prefixed payload ref."""
        digest = config_blob_for_ref.digest
        payload_ref = f"config:{digest}"

        response = await client.get(
            "/api/v1/agent/payload-refs/resolve",
            params={"payload_ref": payload_ref},
            headers=agent_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["resolved"]
        assert data["ref_type"] == "config"
        assert data["content"] == config_blob_for_ref.content
        assert data["digest"] == digest

    @pytest.mark.asyncio
    async def test_resolve_artifact_ref(
        self,
        client: AsyncClient,
        artifact_with_known_digest: Artifact,
        sample_agent: Agent,  # Required to create agent in DB
        agent_headers: Dict[str, str],
    ):
        """Should resolve artifact: prefixed payload ref."""
        digest = artifact_with_known_digest.digest
        payload_ref = f"artifact:{digest}"

        response = await client.get(
            "/api/v1/agent/payload-refs/resolve",
            params={"payload_ref": payload_ref},
            headers=agent_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["resolved"]
        assert data["ref_type"] == "artifact"
        assert "download_url" in data
        assert data["download_url"] is not None

    @pytest.mark.asyncio
    async def test_resolve_bare_digest_as_artifact(
        self,
        client: AsyncClient,
        artifact_with_known_digest: Artifact,
        sample_agent: Agent,  # Required to create agent in DB
        agent_headers: Dict[str, str],
    ):
        """Bare digest (no prefix) should be treated as artifact ref."""
        digest = artifact_with_known_digest.digest

        response = await client.get(
            "/api/v1/agent/payload-refs/resolve",
            params={"payload_ref": digest},  # No prefix
            headers=agent_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["ref_type"] == "artifact"

    @pytest.mark.asyncio
    async def test_resolve_not_found(
        self,
        client: AsyncClient,
        sample_agent: Agent,  # Required to create agent in DB
        agent_headers: Dict[str, str],
    ):
        """Should return resolved=False for unknown ref."""
        response = await client.get(
            "/api/v1/agent/payload-refs/resolve",
            params={"payload_ref": "config:sha256:notfound"},
            headers=agent_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert not data["resolved"]

    @pytest_asyncio.fixture
    async def artifact_with_known_digest(
        self,
        db_session: AsyncSession,
        sample_build: Build,
        sample_workspace: Workspace,
    ) -> Artifact:
        """Create an artifact with known digest."""
        artifact = Artifact(
            build_id=sample_build.id,
            workspace_id=sample_workspace.id,
            name="strategy.tar.gz",
            format="tar.gz",
            digest="sha256:test_artifact_digest",
            size_bytes=2048,
        )
        db_session.add(artifact)
        await db_session.commit()
        await db_session.refresh(artifact)
        return artifact


class TestBatchPayloadRefResolution:
    """Tests for batch payload reference resolution."""

    @pytest.mark.asyncio
    async def test_batch_resolve_multiple_refs(
        self,
        client: AsyncClient,
        config_blob_for_ref: ConfigBlob,
        sample_agent: Agent,  # Required to create agent in DB
        agent_headers: Dict[str, str],
    ):
        """Should resolve multiple refs in one request."""
        refs = [
            f"config:{config_blob_for_ref.digest}",
            "artifact:sha256:unknown",  # Will not resolve
        ]

        response = await client.post(
            "/api/v1/agent/payload-refs/batch-resolve",
            json={"refs": refs},
            headers=agent_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["resolutions"]) == 2
        assert data["resolutions"][0]["resolved"]
        assert not data["resolutions"][1]["resolved"]

    @pytest_asyncio.fixture
    async def config_blob_for_ref(
        self,
        db_session: AsyncSession,
        sample_workspace: Workspace,
        sample_user: User,
    ) -> ConfigBlob:
        """Create a config blob for ref resolution testing."""
        content = {"test": True}
        content_str = json.dumps(content, sort_keys=True, separators=(",", ":"))
        digest = f"sha256:{hashlib.sha256(content_str.encode()).hexdigest()}"

        blob = ConfigBlob(
            workspace_id=sample_workspace.id,
            digest=digest,
            content=content,
            size_bytes=len(content_str.encode()),
            config_type="test",
            schema_version="1.0.0",
            created_by_user_id=sample_user.id,
        )
        db_session.add(blob)
        await db_session.commit()
        await db_session.refresh(blob)
        return blob


class TestAgentWorkspaceIsolation:
    """Tests for workspace isolation in agent blob access."""

    @pytest_asyncio.fixture
    async def other_workspace(
        self,
        db_session: AsyncSession,
        sample_organization: Organization,
    ) -> Workspace:
        """Create another workspace."""
        workspace = Workspace(
            organization_id=sample_organization.id,
            name="other-workspace",
            slug="other-workspace",
        )
        db_session.add(workspace)
        await db_session.commit()
        await db_session.refresh(workspace)
        return workspace

    @pytest_asyncio.fixture
    async def blob_in_other_workspace(
        self,
        db_session: AsyncSession,
        other_workspace: Workspace,
        sample_user: User,
    ) -> ConfigBlob:
        """Create a config blob in the other workspace."""
        content = {"secret": "data"}
        content_str = json.dumps(content, sort_keys=True, separators=(",", ":"))
        digest = f"sha256:{hashlib.sha256(content_str.encode()).hexdigest()}"

        blob = ConfigBlob(
            workspace_id=other_workspace.id,
            digest=digest,
            content=content,
            size_bytes=len(content_str.encode()),
            config_type="strategy",
            schema_version="1.0.0",
            created_by_user_id=sample_user.id,
        )
        db_session.add(blob)
        await db_session.commit()
        await db_session.refresh(blob)
        return blob

    @pytest.mark.asyncio
    async def test_cannot_access_other_workspace_blob(
        self,
        client: AsyncClient,
        blob_in_other_workspace: ConfigBlob,
        sample_agent: Agent,  # Required to create agent in DB
        agent_headers: Dict[str, str],
    ):
        """Agent should not be able to access blobs from other workspaces."""
        # Agent is in sample_workspace, blob is in other_workspace
        response = await client.get(
            f"/api/v1/agent/config-blobs/by-digest/{blob_in_other_workspace.digest}",
            headers=agent_headers,
        )

        # Should not find it (404)
        assert response.status_code == 404
