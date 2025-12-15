# -*- coding: utf-8 -*-
"""Tests for Agents Router."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import (
    Agent,
    AgentEnrollmentToken,
    Artifact,
    Build,
    Deployment,
    DeploymentState,
    Organization,
    Permission,
    Role,
    Strategy,
    StrategyVersion,
    TrustState,
    User,
    Workspace,
)

pytestmark = pytest.mark.asyncio


def hash_password(password: str) -> str:
    """Hash password with SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()


class TestCreateEnrollmentToken:
    """Tests for POST /agents/enrollment-tokens endpoint."""

    async def test_create_enrollment_token_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_workspace: Workspace,
    ) -> None:
        """Superuser can create enrollment token."""
        response = await client.post(
            "/api/v1/agents/enrollment-tokens",
            headers=superuser_headers,
            json={
                "workspace_id": str(sample_workspace.id),
                "expires_in_hours": 24,
                "description": "Test token",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert "token" in data  # Raw token shown only once
        assert data["workspace_id"] == str(sample_workspace.id)
        assert data["description"] == "Test token"
        assert data["is_used"] is False

    async def test_create_enrollment_token_with_permission(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_user: User,
        sample_workspace: Workspace,
        org_id,
    ) -> None:
        """User with agent:enroll permission can create token."""
        # Create permission and role
        perm = Permission(name="agent:enroll", description="Enroll agents", resource="agent", action="enroll")
        db_session.add(perm)
        await db_session.commit()
        await db_session.refresh(perm)

        role = Role(
            name="agent-manager",
            description="Agent Manager",
            organization_id=org_id,
        )
        role.permissions.append(perm)
        db_session.add(role)
        await db_session.commit()
        await db_session.refresh(role)

        # Refresh user to load roles relationship for async context
        await db_session.refresh(sample_user, attribute_names=["roles"])
        sample_user.roles.append(role)
        await db_session.commit()

        # Create token with permission
        from ..routers.auth import create_access_token

        token = create_access_token(
            user_id=sample_user.id,
            email=sample_user.email,
            org_id=org_id,
            workspace_id=sample_workspace.id,
            is_superuser=False,
            permissions=["agent:enroll"],
        )

        response = await client.post(
            "/api/v1/agents/enrollment-tokens",
            headers={"Authorization": f"Bearer {token}"},
            json={"workspace_id": str(sample_workspace.id)},
        )

        assert response.status_code == 201

    async def test_create_enrollment_token_without_permission_forbidden(
        self,
        client: AsyncClient,
        sample_user: User,
        sample_workspace: Workspace,
        org_id,
    ) -> None:
        """User without agent:enroll permission cannot create token."""
        # Create token WITHOUT agent:enroll permission
        from ..dependencies import create_access_token

        token = create_access_token(
            user_id=sample_user.id,
            email=sample_user.email,
            org_id=org_id,
            workspace_id=sample_workspace.id,
            is_superuser=False,
            permissions=["config:read"],  # No agent:enroll permission
        )

        response = await client.post(
            "/api/v1/agents/enrollment-tokens",
            headers={"Authorization": f"Bearer {token}"},
            json={"workspace_id": str(sample_workspace.id)},
        )

        assert response.status_code == 403
        assert "agent:enroll" in response.json()["detail"]

    async def test_create_enrollment_token_other_org_forbidden(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_user: User,
        org_id,
    ) -> None:
        """Cannot create token for workspace in different org."""
        # Create another organization
        other_org = Organization(name="other-org-token", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        # Create workspace in other org
        other_ws = Workspace(name="other-ws", organization_id=other_org.id)
        db_session.add(other_ws)
        await db_session.commit()
        await db_session.refresh(other_ws)

        # Create token with permission but for wrong org
        from ..routers.auth import create_access_token

        token = create_access_token(
            user_id=sample_user.id,
            email=sample_user.email,
            org_id=org_id,
            workspace_id=None,
            is_superuser=False,
            permissions=["agent:enroll"],
        )

        response = await client.post(
            "/api/v1/agents/enrollment-tokens",
            headers={"Authorization": f"Bearer {token}"},
            json={"workspace_id": str(other_ws.id)},
        )

        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]

    async def test_create_enrollment_token_workspace_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Cannot create token for non-existent workspace."""
        response = await client.post(
            "/api/v1/agents/enrollment-tokens",
            headers=superuser_headers,
            json={"workspace_id": str(uuid4())},
        )

        assert response.status_code == 404
        assert "Workspace not found" in response.json()["detail"]

    async def test_create_enrollment_token_custom_expiry(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_workspace: Workspace,
    ) -> None:
        """Token with custom expiry is created correctly."""
        response = await client.post(
            "/api/v1/agents/enrollment-tokens",
            headers=superuser_headers,
            json={
                "workspace_id": str(sample_workspace.id),
                "expires_in_hours": 48,
            },
        )

        assert response.status_code == 201
        data = response.json()
        # Parse datetime and ensure it's timezone-aware (UTC)
        expires_at_str = data["expires_at"].replace("Z", "+00:00")
        expires_at = datetime.fromisoformat(expires_at_str)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        # Should expire in ~48 hours
        delta = expires_at - now
        assert 47 < delta.total_seconds() / 3600 < 49

    async def test_create_enrollment_token_unauthenticated(
        self,
        client: AsyncClient,
        sample_workspace: Workspace,
    ) -> None:
        """Unauthenticated request returns 401."""
        response = await client.post(
            "/api/v1/agents/enrollment-tokens",
            json={"workspace_id": str(sample_workspace.id)},
        )

        assert response.status_code == 401


class TestListEnrollmentTokens:
    """Tests for GET /agents/enrollment-tokens endpoint."""

    async def test_list_enrollment_tokens_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_workspace: Workspace,
        sample_user: User,
    ) -> None:
        """Superuser can list all tokens."""
        # Create some tokens
        for i in range(3):
            token = AgentEnrollmentToken(
                workspace_id=sample_workspace.id,
                name=f"test-token-{i}",
                token_hash=f"hash{i}",
                expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
                created_by_user_id=sample_user.id,
            )
            db_session.add(token)
        await db_session.commit()

        response = await client.get(
            f"/api/v1/agents/enrollment-tokens?workspace_id={sample_workspace.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert data["total"] >= 3

    async def test_list_enrollment_tokens_filter_by_workspace(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_workspace: Workspace,
        sample_user: User,
        org_id,
    ) -> None:
        """List tokens filtered by workspace."""
        # Create another workspace
        other_ws = Workspace(name="other-ws-list", organization_id=org_id)
        db_session.add(other_ws)
        await db_session.commit()
        await db_session.refresh(other_ws)

        # Create tokens in both workspaces
        token1 = AgentEnrollmentToken(
            workspace_id=sample_workspace.id,
            name="token-ws1",
            token_hash="hash-ws1",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
            created_by_user_id=sample_user.id,
        )
        token2 = AgentEnrollmentToken(
            workspace_id=other_ws.id,
            name="token-ws2",
            token_hash="hash-ws2",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
            created_by_user_id=sample_user.id,
        )
        db_session.add_all([token1, token2])
        await db_session.commit()

        response = await client.get(
            f"/api/v1/agents/enrollment-tokens?workspace_id={sample_workspace.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert item["workspace_id"] == str(sample_workspace.id)

    async def test_list_enrollment_tokens_include_used(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_workspace: Workspace,
        sample_user: User,
    ) -> None:
        """Include used tokens when flag is set."""
        # Create used and unused tokens
        unused = AgentEnrollmentToken(
            workspace_id=sample_workspace.id,
            name="token-unused",
            token_hash="hash-unused",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
            created_by_user_id=sample_user.id,
            is_used=False,
        )
        used = AgentEnrollmentToken(
            workspace_id=sample_workspace.id,
            name="token-used",
            token_hash="hash-used",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
            created_by_user_id=sample_user.id,
            is_used=True,
            used_at=datetime.now(timezone.utc),
        )
        db_session.add_all([unused, used])
        await db_session.commit()

        # Without include_used
        response1 = await client.get(
            f"/api/v1/agents/enrollment-tokens?workspace_id={sample_workspace.id}",
            headers=superuser_headers,
        )
        data1 = response1.json()
        used_count1 = sum(1 for item in data1["items"] if item["is_used"])

        # With include_used
        response2 = await client.get(
            f"/api/v1/agents/enrollment-tokens?workspace_id={sample_workspace.id}&include_used=true",
            headers=superuser_headers,
        )
        data2 = response2.json()
        used_count2 = sum(1 for item in data2["items"] if item["is_used"])

        assert used_count2 >= used_count1

    async def test_list_enrollment_tokens_non_superuser_requires_workspace_id(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ) -> None:
        """Non-superuser must provide workspace_id."""
        response = await client.get(
            "/api/v1/agents/enrollment-tokens",
            headers=auth_headers,
        )

        assert response.status_code == 400
        assert "workspace_id required" in response.json()["detail"]

    async def test_list_enrollment_tokens_other_org_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
    ) -> None:
        """Cannot list tokens in workspace from different org."""
        # Create workspace in another org
        other_org = Organization(name="other-org-list", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        other_ws = Workspace(name="other-ws-forbidden", organization_id=other_org.id)
        db_session.add(other_ws)
        await db_session.commit()
        await db_session.refresh(other_ws)

        response = await client.get(
            f"/api/v1/agents/enrollment-tokens?workspace_id={other_ws.id}",
            headers=auth_headers,
        )

        assert response.status_code == 403


class TestDeleteEnrollmentToken:
    """Tests for DELETE /agents/enrollment-tokens/{token_id} endpoint."""

    async def test_delete_enrollment_token_success(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_workspace: Workspace,
        sample_user: User,
    ) -> None:
        """Superuser can delete unused token."""
        token = AgentEnrollmentToken(
            workspace_id=sample_workspace.id,
            name="token-to-delete",
            token_hash="hash-to-delete",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
            created_by_user_id=sample_user.id,
        )
        db_session.add(token)
        await db_session.commit()
        await db_session.refresh(token)

        response = await client.delete(
            f"/api/v1/agents/enrollment-tokens/{token.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 204

        # Verify deleted
        result = await db_session.execute(
            select(AgentEnrollmentToken).where(AgentEnrollmentToken.id == token.id)
        )
        assert result.scalar_one_or_none() is None

    async def test_delete_enrollment_token_used_forbidden(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_workspace: Workspace,
        sample_user: User,
    ) -> None:
        """Cannot delete used token."""
        token = AgentEnrollmentToken(
            workspace_id=sample_workspace.id,
            name="token-used-delete",
            token_hash="hash-used-delete",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
            created_by_user_id=sample_user.id,
            is_used=True,
            used_at=datetime.now(timezone.utc),
        )
        db_session.add(token)
        await db_session.commit()
        await db_session.refresh(token)

        response = await client.delete(
            f"/api/v1/agents/enrollment-tokens/{token.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 400
        assert "Cannot delete used" in response.json()["detail"]

    async def test_delete_enrollment_token_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Delete non-existent token returns 404."""
        response = await client.delete(
            f"/api/v1/agents/enrollment-tokens/{uuid4()}",
            headers=superuser_headers,
        )

        assert response.status_code == 404

    async def test_delete_enrollment_token_other_org_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
        sample_user: User,
    ) -> None:
        """Cannot delete token in workspace from different org."""
        # Create workspace in another org
        other_org = Organization(name="other-org-del", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        other_ws = Workspace(name="other-ws-del", organization_id=other_org.id)
        db_session.add(other_ws)
        await db_session.commit()
        await db_session.refresh(other_ws)

        token = AgentEnrollmentToken(
            workspace_id=other_ws.id,
            name="token-other-org",
            token_hash="hash-other-org",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
            created_by_user_id=sample_user.id,
        )
        db_session.add(token)
        await db_session.commit()
        await db_session.refresh(token)

        response = await client.delete(
            f"/api/v1/agents/enrollment-tokens/{token.id}",
            headers=auth_headers,
        )

        assert response.status_code == 403


class TestListAgents:
    """Tests for GET /agents endpoint."""

    async def test_list_agents_in_workspace(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
        sample_workspace: Workspace,
    ) -> None:
        """User can list agents in their org's workspace."""
        # Create agents
        for i in range(3):
            agent = Agent(
                name=f"agent-{i}",
                workspace_id=sample_workspace.id,
                trust_state=TrustState.ENROLLED,
                public_key="a" * 64,
                agent_version="1.0.0",
            )
            db_session.add(agent)
        await db_session.commit()

        response = await client.get(
            f"/api/v1/agents?workspace_id={sample_workspace.id}",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert data["total"] >= 3

    async def test_list_agents_filter_by_trust_state(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
        sample_workspace: Workspace,
    ) -> None:
        """Filter agents by trust state."""
        # Create agents with different states
        enrolled = Agent(
            name="enrolled-agent",
            workspace_id=sample_workspace.id,
            trust_state=TrustState.ENROLLED,
            public_key="b" * 64,
            agent_version="1.0.0",
        )
        revoked = Agent(
            name="revoked-agent",
            workspace_id=sample_workspace.id,
            trust_state=TrustState.REVOKED,
            public_key="c" * 64,
            agent_version="1.0.0",
        )
        db_session.add_all([enrolled, revoked])
        await db_session.commit()

        response = await client.get(
            f"/api/v1/agents?workspace_id={sample_workspace.id}&trust_state=enrolled",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert item["trust_state"] == "enrolled"

    async def test_list_agents_all_org_workspaces(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
        sample_workspace: Workspace,
        org_id,
    ) -> None:
        """Non-superuser without workspace_id sees agents from all org workspaces."""
        # Create another workspace in same org
        ws2 = Workspace(name="ws2-agents", organization_id=org_id)
        db_session.add(ws2)
        await db_session.commit()
        await db_session.refresh(ws2)

        # Create agents in both workspaces
        agent1 = Agent(
            name="agent-ws1",
            workspace_id=sample_workspace.id,
            trust_state=TrustState.ENROLLED,
            public_key="d" * 64,
            agent_version="1.0.0",
        )
        agent2 = Agent(
            name="agent-ws2",
            workspace_id=ws2.id,
            trust_state=TrustState.ENROLLED,
            public_key="e" * 64,
            agent_version="1.0.0",
        )
        db_session.add_all([agent1, agent2])
        await db_session.commit()

        response = await client.get(
            "/api/v1/agents",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        workspace_ids = {item["workspace_id"] for item in data["items"]}
        assert str(sample_workspace.id) in workspace_ids or str(ws2.id) in workspace_ids

    async def test_list_agents_other_org_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
    ) -> None:
        """Cannot list agents in workspace from different org."""
        # Create workspace in another org
        other_org = Organization(name="other-org-agents", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        other_ws = Workspace(name="other-ws-agents", organization_id=other_org.id)
        db_session.add(other_ws)
        await db_session.commit()
        await db_session.refresh(other_ws)

        response = await client.get(
            f"/api/v1/agents?workspace_id={other_ws.id}",
            headers=auth_headers,
        )

        assert response.status_code == 403

    async def test_list_agents_includes_deployment_count(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_workspace: Workspace,
    ) -> None:
        """Agent response includes active deployment count."""
        agent = Agent(
            name="agent-with-deploys",
            workspace_id=sample_workspace.id,
            trust_state=TrustState.ENROLLED,
            public_key="f" * 64,
            agent_version="1.0.0",
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        # Create strategy -> version -> build -> artifact chain
        strategy = Strategy(
            name="test-strategy",
            workspace_id=sample_workspace.id,
        )
        db_session.add(strategy)
        await db_session.flush()

        version = StrategyVersion(
            strategy_id=strategy.id,
            version="1.0.0",
            workspace_id=sample_workspace.id,
        )
        db_session.add(version)
        await db_session.flush()

        build = Build(
            strategy_version_id=version.id,
            build_number=1,
            status="completed",
            workspace_id=sample_workspace.id,
        )
        db_session.add(build)
        await db_session.flush()

        artifact = Artifact(
            build_id=build.id,
            name="test-artifact",
            format="oci_image",
            digest="sha256:" + "a" * 64,
            workspace_id=sample_workspace.id,
        )
        db_session.add(artifact)
        await db_session.flush()

        # Create deployment
        deploy = Deployment(
            agent_id=agent.id,
            artifact_id=artifact.id,
            workspace_id=sample_workspace.id,
            state=DeploymentState.DEPLOYED,
        )
        db_session.add(deploy)
        await db_session.commit()

        response = await client.get(
            f"/api/v1/agents?workspace_id={sample_workspace.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        found = False
        for item in data["items"]:
            if item["id"] == str(agent.id):
                assert item["active_deployments"] == 1
                found = True
        assert found


class TestGetAgent:
    """Tests for GET /agents/{agent_id} endpoint."""

    async def test_get_agent_success(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
        sample_workspace: Workspace,
    ) -> None:
        """Get agent by ID."""
        agent = Agent(
            name="get-agent-test",
            workspace_id=sample_workspace.id,
            trust_state=TrustState.ENROLLED,
            public_key="g" * 64,
            agent_version="1.0.0",
            capabilities=["trading", "backtest"],
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        response = await client.get(
            f"/api/v1/agents/{agent.id}",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(agent.id)
        assert data["name"] == "get-agent-test"
        assert data["trust_state"] == "enrolled"
        assert data["capabilities"] == ["trading", "backtest"]

    async def test_get_agent_not_found(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ) -> None:
        """Get non-existent agent returns 404."""
        response = await client.get(
            f"/api/v1/agents/{uuid4()}",
            headers=auth_headers,
        )

        assert response.status_code == 404
        assert "Agent not found" in response.json()["detail"]

    async def test_get_agent_other_org_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
    ) -> None:
        """Cannot get agent in workspace from different org."""
        # Create workspace in another org
        other_org = Organization(name="other-org-get", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        other_ws = Workspace(name="other-ws-get", organization_id=other_org.id)
        db_session.add(other_ws)
        await db_session.commit()
        await db_session.refresh(other_ws)

        agent = Agent(
            name="other-org-agent",
            workspace_id=other_ws.id,
            trust_state=TrustState.ENROLLED,
            public_key="h" * 64,
            agent_version="1.0.0",
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        response = await client.get(
            f"/api/v1/agents/{agent.id}",
            headers=auth_headers,
        )

        assert response.status_code == 403

    async def test_get_agent_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_workspace: Workspace,
    ) -> None:
        """Superuser can get any agent."""
        agent = Agent(
            name="superuser-get-agent",
            workspace_id=sample_workspace.id,
            trust_state=TrustState.ENROLLED,
            public_key="i" * 64,
            agent_version="1.0.0",
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        response = await client.get(
            f"/api/v1/agents/{agent.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200


class TestUpdateAgent:
    """Tests for PATCH /agents/{agent_id} endpoint."""

    async def test_update_agent_name_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_workspace: Workspace,
    ) -> None:
        """Superuser can update agent name."""
        agent = Agent(
            name="original-name",
            workspace_id=sample_workspace.id,
            trust_state=TrustState.ENROLLED,
            public_key="j" * 64,
            agent_version="1.0.0",
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        response = await client.patch(
            f"/api/v1/agents/{agent.id}",
            headers=superuser_headers,
            json={"name": "updated-name"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "updated-name"

    async def test_update_agent_with_permission(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_user: User,
        sample_workspace: Workspace,
        org_id,
    ) -> None:
        """User with agent:write can update agent."""
        # Create permission and role
        perm = Permission(name="agent:write", description="Write agents", resource="agent", action="write")
        db_session.add(perm)
        await db_session.commit()
        await db_session.refresh(perm)

        role = Role(
            name="agent-writer",
            description="Agent Writer",
            organization_id=org_id,
        )
        role.permissions.append(perm)
        db_session.add(role)
        await db_session.commit()
        await db_session.refresh(role)

        # Refresh user to load roles relationship for async context
        await db_session.refresh(sample_user, attribute_names=["roles"])
        sample_user.roles.append(role)
        await db_session.commit()

        agent = Agent(
            name="writable-agent",
            workspace_id=sample_workspace.id,
            trust_state=TrustState.ENROLLED,
            public_key="k" * 64,
            agent_version="1.0.0",
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        from ..routers.auth import create_access_token

        token = create_access_token(
            user_id=sample_user.id,
            email=sample_user.email,
            org_id=org_id,
            workspace_id=sample_workspace.id,
            is_superuser=False,
            permissions=["agent:write"],
        )

        response = await client.patch(
            f"/api/v1/agents/{agent.id}",
            headers={"Authorization": f"Bearer {token}"},
            json={"capabilities": ["new-capability"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["capabilities"] == ["new-capability"]

    async def test_update_agent_without_permission_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
        sample_workspace: Workspace,
    ) -> None:
        """User without agent:write cannot update agent."""
        agent = Agent(
            name="protected-agent",
            workspace_id=sample_workspace.id,
            trust_state=TrustState.ENROLLED,
            public_key="l" * 64,
            agent_version="1.0.0",
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        response = await client.patch(
            f"/api/v1/agents/{agent.id}",
            headers=auth_headers,
            json={"name": "hacked-name"},
        )

        assert response.status_code == 403
        assert "agent:write" in response.json()["detail"]

    async def test_update_agent_trust_state_requires_trust_permission(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_user: User,
        sample_workspace: Workspace,
        org_id,
    ) -> None:
        """Updating trust_state requires agent:trust permission."""
        # Create agent:write permission only
        perm = Permission(name="agent:write", description="Write agents", resource="agent", action="write")
        db_session.add(perm)
        await db_session.commit()
        await db_session.refresh(perm)

        role = Role(
            name="agent-writer-only",
            description="Agent Writer Only",
            organization_id=org_id,
        )
        role.permissions.append(perm)
        db_session.add(role)
        await db_session.commit()
        await db_session.refresh(role)

        # Refresh user to load roles relationship for async context
        await db_session.refresh(sample_user, attribute_names=["roles"])
        sample_user.roles.append(role)
        await db_session.commit()

        agent = Agent(
            name="trust-protected-agent",
            workspace_id=sample_workspace.id,
            trust_state=TrustState.ENROLLED,
            public_key="m" * 64,
            agent_version="1.0.0",
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        from ..routers.auth import create_access_token

        token = create_access_token(
            user_id=sample_user.id,
            email=sample_user.email,
            org_id=org_id,
            workspace_id=sample_workspace.id,
            is_superuser=False,
            permissions=["agent:write"],
        )

        response = await client.patch(
            f"/api/v1/agents/{agent.id}",
            headers={"Authorization": f"Bearer {token}"},
            json={"trust_state": "suspended"},
        )

        assert response.status_code == 403
        assert "agent:trust" in response.json()["detail"]

    async def test_update_agent_trust_state_with_permission(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_user: User,
        sample_workspace: Workspace,
        org_id,
    ) -> None:
        """User with agent:trust can update trust_state."""
        # Create permissions
        write_perm = Permission(name="agent:write", description="Write agents", resource="agent", action="write2")
        trust_perm = Permission(name="agent:trust", description="Trust agents", resource="agent", action="trust")
        db_session.add_all([write_perm, trust_perm])
        await db_session.commit()

        role = Role(
            name="agent-trust-manager",
            description="Agent Trust Manager",
            organization_id=org_id,
        )
        role.permissions.extend([write_perm, trust_perm])
        db_session.add(role)
        await db_session.commit()
        await db_session.refresh(role)

        # Refresh user to load roles relationship for async context
        await db_session.refresh(sample_user, attribute_names=["roles"])
        sample_user.roles.append(role)
        await db_session.commit()

        agent = Agent(
            name="trust-managed-agent",
            workspace_id=sample_workspace.id,
            trust_state=TrustState.ENROLLED,
            public_key="n" * 64,
            agent_version="1.0.0",
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        from ..routers.auth import create_access_token

        token = create_access_token(
            user_id=sample_user.id,
            email=sample_user.email,
            org_id=org_id,
            workspace_id=sample_workspace.id,
            is_superuser=False,
            permissions=["agent:write", "agent:trust"],
        )

        response = await client.patch(
            f"/api/v1/agents/{agent.id}",
            headers={"Authorization": f"Bearer {token}"},
            json={"trust_state": "suspended"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["trust_state"] == "suspended"

    async def test_update_agent_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Update non-existent agent returns 404."""
        response = await client.patch(
            f"/api/v1/agents/{uuid4()}",
            headers=superuser_headers,
            json={"name": "ghost"},
        )

        assert response.status_code == 404

    async def test_update_agent_other_org_forbidden(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_user: User,
        org_id,
    ) -> None:
        """Cannot update agent in workspace from different org."""
        # Create workspace in another org
        other_org = Organization(name="other-org-update", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        other_ws = Workspace(name="other-ws-update", organization_id=other_org.id)
        db_session.add(other_ws)
        await db_session.commit()
        await db_session.refresh(other_ws)

        agent = Agent(
            name="other-org-update-agent",
            workspace_id=other_ws.id,
            trust_state=TrustState.ENROLLED,
            public_key="o" * 64,
            agent_version="1.0.0",
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        from ..routers.auth import create_access_token

        token = create_access_token(
            user_id=sample_user.id,
            email=sample_user.email,
            org_id=org_id,
            workspace_id=None,
            is_superuser=False,
            permissions=["agent:write", "agent:trust"],
        )

        response = await client.patch(
            f"/api/v1/agents/{agent.id}",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "hacked"},
        )

        assert response.status_code == 403


class TestRevokeAgent:
    """Tests for POST /agents/{agent_id}/revoke endpoint."""

    async def test_revoke_agent_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_workspace: Workspace,
    ) -> None:
        """Superuser can revoke agent."""
        agent = Agent(
            name="revoke-agent-test",
            workspace_id=sample_workspace.id,
            trust_state=TrustState.ENROLLED,
            public_key="p" * 64,
            agent_version="1.0.0",
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        response = await client.post(
            f"/api/v1/agents/{agent.id}/revoke",
            headers=superuser_headers,
            json={"reason": "Compromised credentials"},
        )

        assert response.status_code == 204

        # Verify revoked
        await db_session.refresh(agent)
        assert agent.trust_state == TrustState.REVOKED
        assert agent.revocation_reason == "Compromised credentials"

    async def test_revoke_agent_with_permission(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_user: User,
        sample_workspace: Workspace,
        org_id,
    ) -> None:
        """User with agent:trust can revoke agent."""
        # Create permission
        perm = Permission(name="agent:trust", description="Trust agents", resource="agent", action="trust")
        db_session.add(perm)
        await db_session.commit()
        await db_session.refresh(perm)

        role = Role(
            name="agent-trust-revoker",
            description="Agent Revoker",
            organization_id=org_id,
        )
        role.permissions.append(perm)
        db_session.add(role)
        await db_session.commit()
        await db_session.refresh(role)

        # Refresh user to load roles relationship for async context
        await db_session.refresh(sample_user, attribute_names=["roles"])
        sample_user.roles.append(role)
        await db_session.commit()

        agent = Agent(
            name="revokable-agent",
            workspace_id=sample_workspace.id,
            trust_state=TrustState.ENROLLED,
            public_key="q" * 64,
            agent_version="1.0.0",
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        from ..routers.auth import create_access_token

        token = create_access_token(
            user_id=sample_user.id,
            email=sample_user.email,
            org_id=org_id,
            workspace_id=sample_workspace.id,
            is_superuser=False,
            permissions=["agent:trust"],
        )

        response = await client.post(
            f"/api/v1/agents/{agent.id}/revoke",
            headers={"Authorization": f"Bearer {token}"},
            json={"reason": "Policy violation"},
        )

        assert response.status_code == 204

    async def test_revoke_agent_without_permission_forbidden(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_user: User,
        sample_workspace: Workspace,
        org_id,
    ) -> None:
        """User without agent:trust cannot revoke agent."""
        agent = Agent(
            name="unrevokable-agent",
            workspace_id=sample_workspace.id,
            trust_state=TrustState.ENROLLED,
            public_key="r" * 64,
            agent_version="1.0.0",
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        # Create token WITHOUT agent:trust permission
        from ..dependencies import create_access_token

        token = create_access_token(
            user_id=sample_user.id,
            email=sample_user.email,
            org_id=org_id,
            workspace_id=sample_workspace.id,
            is_superuser=False,
            permissions=["config:read"],  # No agent:trust permission
        )

        response = await client.post(
            f"/api/v1/agents/{agent.id}/revoke",
            headers={"Authorization": f"Bearer {token}"},
            json={"reason": "Attempt to revoke"},
        )

        assert response.status_code == 403
        assert "agent:trust" in response.json()["detail"]

    async def test_revoke_agent_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Revoke non-existent agent returns 404."""
        response = await client.post(
            f"/api/v1/agents/{uuid4()}/revoke",
            headers=superuser_headers,
            json={"reason": "Ghost agent"},
        )

        assert response.status_code == 404

    async def test_revoke_agent_other_org_forbidden(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_user: User,
        org_id,
    ) -> None:
        """Cannot revoke agent in workspace from different org."""
        # Create workspace in another org
        other_org = Organization(name="other-org-revoke", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        other_ws = Workspace(name="other-ws-revoke", organization_id=other_org.id)
        db_session.add(other_ws)
        await db_session.commit()
        await db_session.refresh(other_ws)

        agent = Agent(
            name="other-org-revoke-agent",
            workspace_id=other_ws.id,
            trust_state=TrustState.ENROLLED,
            public_key="s" * 64,
            agent_version="1.0.0",
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        from ..routers.auth import create_access_token

        token = create_access_token(
            user_id=sample_user.id,
            email=sample_user.email,
            org_id=org_id,
            workspace_id=None,
            is_superuser=False,
            permissions=["agent:trust"],
        )

        response = await client.post(
            f"/api/v1/agents/{agent.id}/revoke",
            headers={"Authorization": f"Bearer {token}"},
            json={"reason": "Hack attempt"},
        )

        assert response.status_code == 403


class TestReinstateAgent:
    """Tests for POST /agents/{agent_id}/reinstate endpoint."""

    async def test_reinstate_agent_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_workspace: Workspace,
    ) -> None:
        """Superuser can reinstate revoked agent."""
        agent = Agent(
            name="reinstate-agent-test",
            workspace_id=sample_workspace.id,
            trust_state=TrustState.REVOKED,
            public_key="t" * 64,
            agent_version="1.0.0",
            revocation_reason="Old reason",
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        response = await client.post(
            f"/api/v1/agents/{agent.id}/reinstate",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["trust_state"] == "enrolled"

        # Verify revocation reason cleared
        await db_session.refresh(agent)
        assert agent.revocation_reason is None

    async def test_reinstate_suspended_agent(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_workspace: Workspace,
    ) -> None:
        """Can reinstate suspended agent."""
        agent = Agent(
            name="suspended-agent-test",
            workspace_id=sample_workspace.id,
            trust_state=TrustState.SUSPENDED,
            public_key="u" * 64,
            agent_version="1.0.0",
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        response = await client.post(
            f"/api/v1/agents/{agent.id}/reinstate",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["trust_state"] == "enrolled"

    async def test_reinstate_already_enrolled_error(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_workspace: Workspace,
    ) -> None:
        """Cannot reinstate already enrolled agent."""
        agent = Agent(
            name="enrolled-agent-test",
            workspace_id=sample_workspace.id,
            trust_state=TrustState.ENROLLED,
            public_key="v" * 64,
            agent_version="1.0.0",
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        response = await client.post(
            f"/api/v1/agents/{agent.id}/reinstate",
            headers=superuser_headers,
        )

        assert response.status_code == 400
        assert "already enrolled" in response.json()["detail"]

    async def test_reinstate_agent_with_permission(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_user: User,
        sample_workspace: Workspace,
        org_id,
    ) -> None:
        """User with agent:trust can reinstate agent."""
        # Create permission
        perm = Permission(name="agent:trust", description="Trust agents", resource="agent", action="trust")
        db_session.add(perm)
        await db_session.commit()
        await db_session.refresh(perm)

        role = Role(
            name="agent-trust-reinstater",
            description="Agent Reinstater",
            organization_id=org_id,
        )
        role.permissions.append(perm)
        db_session.add(role)
        await db_session.commit()
        await db_session.refresh(role)

        # Refresh user to load roles relationship for async context
        await db_session.refresh(sample_user, attribute_names=["roles"])
        sample_user.roles.append(role)
        await db_session.commit()

        agent = Agent(
            name="reinstatable-agent",
            workspace_id=sample_workspace.id,
            trust_state=TrustState.REVOKED,
            public_key="w" * 64,
            agent_version="1.0.0",
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        from ..routers.auth import create_access_token

        token = create_access_token(
            user_id=sample_user.id,
            email=sample_user.email,
            org_id=org_id,
            workspace_id=sample_workspace.id,
            is_superuser=False,
            permissions=["agent:trust"],
        )

        response = await client.post(
            f"/api/v1/agents/{agent.id}/reinstate",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200

    async def test_reinstate_agent_without_permission_forbidden(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_user: User,
        sample_workspace: Workspace,
        org_id,
    ) -> None:
        """User without agent:trust cannot reinstate agent."""
        agent = Agent(
            name="unreinstatable-agent",
            workspace_id=sample_workspace.id,
            trust_state=TrustState.REVOKED,
            public_key="x" * 64,
            agent_version="1.0.0",
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        # Create token WITHOUT agent:trust permission
        from ..dependencies import create_access_token

        token = create_access_token(
            user_id=sample_user.id,
            email=sample_user.email,
            org_id=org_id,
            workspace_id=sample_workspace.id,
            is_superuser=False,
            permissions=["config:read"],  # No agent:trust permission
        )

        response = await client.post(
            f"/api/v1/agents/{agent.id}/reinstate",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 403
        assert "agent:trust" in response.json()["detail"]

    async def test_reinstate_agent_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Reinstate non-existent agent returns 404."""
        response = await client.post(
            f"/api/v1/agents/{uuid4()}/reinstate",
            headers=superuser_headers,
        )

        assert response.status_code == 404

    async def test_reinstate_agent_other_org_forbidden(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_user: User,
        org_id,
    ) -> None:
        """Cannot reinstate agent in workspace from different org."""
        # Create workspace in another org
        other_org = Organization(name="other-org-reinstate", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        other_ws = Workspace(name="other-ws-reinstate", organization_id=other_org.id)
        db_session.add(other_ws)
        await db_session.commit()
        await db_session.refresh(other_ws)

        agent = Agent(
            name="other-org-reinstate-agent",
            workspace_id=other_ws.id,
            trust_state=TrustState.REVOKED,
            public_key="y" * 64,
            agent_version="1.0.0",
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        from ..routers.auth import create_access_token

        token = create_access_token(
            user_id=sample_user.id,
            email=sample_user.email,
            org_id=org_id,
            workspace_id=None,
            is_superuser=False,
            permissions=["agent:trust"],
        )

        response = await client.post(
            f"/api/v1/agents/{agent.id}/reinstate",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 403


class TestAgentKeyRotation:
    """Tests for agent key rotation endpoints (Design Doc 15.2)."""

    async def test_rotate_key_success(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        superuser_headers: dict,
        sample_workspace: Workspace,
    ) -> None:
        """Superuser can rotate agent key."""
        # Create agent
        old_key = "old_public_key_" + "a" * 50
        agent = Agent(
            name="key-rotation-agent",
            workspace_id=sample_workspace.id,
            trust_state=TrustState.ENROLLED,
            public_key=old_key,
            agent_version="1.0.0",
            key_version=1,
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        new_key = "new_public_key_" + "b" * 50
        response = await client.post(
            f"/api/v1/agents/{agent.id}/rotate-key",
            headers=superuser_headers,
            json={"new_public_key": new_key},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["key_version"] == 2
        assert "previous_key_valid_until" in data

        # Verify in database
        await db_session.refresh(agent)
        assert agent.public_key == new_key
        assert agent.previous_public_key == old_key
        assert agent.key_version == 2
        assert agent.key_rotation_grace_until is not None

    async def test_rotate_key_revoked_agent_fails(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        superuser_headers: dict,
        sample_workspace: Workspace,
    ) -> None:
        """Cannot rotate key for revoked agent."""
        agent = Agent(
            name="revoked-key-agent",
            workspace_id=sample_workspace.id,
            trust_state=TrustState.REVOKED,
            public_key="a" * 64,
            agent_version="1.0.0",
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        response = await client.post(
            f"/api/v1/agents/{agent.id}/rotate-key",
            headers=superuser_headers,
            json={"new_public_key": "b" * 64},
        )

        assert response.status_code == 400
        assert "revoked" in response.json()["detail"].lower()

    async def test_complete_key_rotation(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        superuser_headers: dict,
        sample_workspace: Workspace,
    ) -> None:
        """Complete key rotation invalidates old key."""
        agent = Agent(
            name="complete-rotation-agent",
            workspace_id=sample_workspace.id,
            trust_state=TrustState.ENROLLED,
            public_key="new_key_" + "a" * 50,
            previous_public_key="old_key_" + "b" * 50,
            agent_version="1.0.0",
            key_version=2,
            key_rotation_grace_until=datetime.now(timezone.utc) + timedelta(hours=24),
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        response = await client.post(
            f"/api/v1/agents/{agent.id}/complete-key-rotation",
            headers=superuser_headers,
        )

        assert response.status_code == 200

        # Verify old key is cleared
        await db_session.refresh(agent)
        assert agent.previous_public_key is None
        assert agent.key_rotation_grace_until is None

    async def test_complete_rotation_no_rotation_in_progress(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        superuser_headers: dict,
        sample_workspace: Workspace,
    ) -> None:
        """Complete rotation fails if no rotation in progress."""
        agent = Agent(
            name="no-rotation-agent",
            workspace_id=sample_workspace.id,
            trust_state=TrustState.ENROLLED,
            public_key="a" * 64,
            previous_public_key=None,  # No rotation in progress
            agent_version="1.0.0",
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        response = await client.post(
            f"/api/v1/agents/{agent.id}/complete-key-rotation",
            headers=superuser_headers,
        )

        assert response.status_code == 400
        assert "No key rotation in progress" in response.json()["detail"]

    async def test_get_key_info(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        superuser_headers: dict,
        sample_workspace: Workspace,
    ) -> None:
        """Get key info returns current key status."""
        agent = Agent(
            name="key-info-agent",
            workspace_id=sample_workspace.id,
            trust_state=TrustState.ENROLLED,
            public_key="a" * 64,
            agent_version="1.0.0",
            key_version=1,
            key_algorithm="ed25519",
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        response = await client.get(
            f"/api/v1/agents/{agent.id}/key-info",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["key_version"] == 1
        assert data["key_algorithm"] == "ed25519"
        assert "public_key_fingerprint" in data
        assert data["rotation_in_progress"] is False
