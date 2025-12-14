# -*- coding: utf-8 -*-
"""Tests for Deployments Router."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import (
    Agent,
    Artifact,
    Build,
    BuildState,
    ConfigBlob,
    Deployment,
    DeploymentState,
    Organization,
    Run,
    RunState,
    Strategy,
    StrategyVersion,
    TrustState,
    User,
    Workspace,
)

pytestmark = pytest.mark.asyncio


# =============================================================================
# Fixtures for Deployments tests
# =============================================================================


@pytest.fixture
async def sample_strategy(db_session: AsyncSession, sample_workspace: Workspace) -> Strategy:
    """Create a sample strategy for tests."""
    strategy = Strategy(
        workspace_id=sample_workspace.id,
        name="test-strategy-deploy",
        description="Strategy for deployment tests",
    )
    db_session.add(strategy)
    await db_session.commit()
    await db_session.refresh(strategy)
    return strategy


@pytest.fixture
async def sample_version(
    db_session: AsyncSession, sample_strategy: Strategy, sample_workspace: Workspace
) -> StrategyVersion:
    """Create a sample strategy version."""
    version = StrategyVersion(
        strategy_id=sample_strategy.id,
        workspace_id=sample_workspace.id,
        version="1.0.0",
        git_sha="abc123def456",
    )
    db_session.add(version)
    await db_session.commit()
    await db_session.refresh(version)
    return version


@pytest.fixture
async def sample_build(
    db_session: AsyncSession, sample_version: StrategyVersion, sample_workspace: Workspace
) -> Build:
    """Create a sample build."""
    build = Build(
        strategy_version_id=sample_version.id,
        workspace_id=sample_workspace.id,
        build_number=1,
        status=BuildState.COMPLETED.value,
    )
    db_session.add(build)
    await db_session.commit()
    await db_session.refresh(build)
    return build


@pytest.fixture
async def sample_artifact(
    db_session: AsyncSession, sample_build: Build, sample_workspace: Workspace
) -> Artifact:
    """Create a sample artifact."""
    artifact = Artifact(
        build_id=sample_build.id,
        workspace_id=sample_workspace.id,
        name="test-artifact.tar.gz",
        format="strategy_bundle",
        digest="sha256:abc123def456",
        size_bytes=1024,
    )
    db_session.add(artifact)
    await db_session.commit()
    await db_session.refresh(artifact)
    return artifact


@pytest.fixture
async def enrolled_agent(
    db_session: AsyncSession, sample_workspace: Workspace
) -> Agent:
    """Create an enrolled agent for tests."""
    agent = Agent(
        workspace_id=sample_workspace.id,
        name="enrolled-agent-deploy",
        public_key="x" * 64,
        agent_version="1.0.0",
        trust_state=TrustState.ENROLLED,
    )
    db_session.add(agent)
    await db_session.commit()
    await db_session.refresh(agent)
    return agent


@pytest.fixture
async def sample_config_blob(
    db_session: AsyncSession, sample_workspace: Workspace
) -> ConfigBlob:
    """Create a sample config blob."""
    config_blob = ConfigBlob(
        workspace_id=sample_workspace.id,
        digest="sha256:config123",
        size_bytes=256,
        content={"key": "value"},
        config_type="strategy",
    )
    db_session.add(config_blob)
    await db_session.commit()
    await db_session.refresh(config_blob)
    return config_blob


@pytest.fixture
async def sample_deployment(
    db_session: AsyncSession,
    sample_workspace: Workspace,
    enrolled_agent: Agent,
    sample_artifact: Artifact,
) -> Deployment:
    """Create a sample deployment."""
    deployment = Deployment(
        workspace_id=sample_workspace.id,
        agent_id=enrolled_agent.id,
        artifact_id=sample_artifact.id,
        state=DeploymentState.CREATED,
        desired_state="deployed",
    )
    db_session.add(deployment)
    await db_session.commit()
    await db_session.refresh(deployment)
    return deployment


@pytest.fixture
async def deployed_deployment(
    db_session: AsyncSession,
    sample_workspace: Workspace,
    enrolled_agent: Agent,
    sample_artifact: Artifact,
) -> Deployment:
    """Create a deployment in DEPLOYED state."""
    deployment = Deployment(
        workspace_id=sample_workspace.id,
        agent_id=enrolled_agent.id,
        artifact_id=sample_artifact.id,
        state=DeploymentState.DEPLOYED,
        desired_state="deployed",
    )
    db_session.add(deployment)
    await db_session.commit()
    await db_session.refresh(deployment)
    return deployment


@pytest.fixture
async def sample_run(
    db_session: AsyncSession,
    deployed_deployment: Deployment,
    sample_workspace: Workspace,
) -> Run:
    """Create a sample run."""
    run = Run(
        deployment_id=deployed_deployment.id,
        workspace_id=sample_workspace.id,
        state=RunState.CREATED,
        is_paper_trading=True,
    )
    db_session.add(run)
    await db_session.commit()
    await db_session.refresh(run)
    return run


# =============================================================================
# Deployment List Tests
# =============================================================================


class TestListDeployments:
    """Tests for GET /deployments endpoint."""

    async def test_list_deployments_success(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_deployment: Deployment,
    ) -> None:
        """User can list deployments in their organization."""
        response = await client.get(
            "/api/v1/deployments",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert data["total"] >= 1

    async def test_list_deployments_filter_by_workspace(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_deployment: Deployment,
        workspace_id,
    ) -> None:
        """Can filter deployments by workspace."""
        response = await client.get(
            f"/api/v1/deployments?workspace_id={workspace_id}",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert item["workspace_id"] == str(workspace_id)

    async def test_list_deployments_filter_by_agent(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_deployment: Deployment,
        enrolled_agent: Agent,
    ) -> None:
        """Can filter deployments by agent."""
        response = await client.get(
            f"/api/v1/deployments?agent_id={enrolled_agent.id}",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert item["agent_id"] == str(enrolled_agent.id)

    async def test_list_deployments_exclude_retired_by_default(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
        sample_workspace: Workspace,
        enrolled_agent: Agent,
        sample_artifact: Artifact,
    ) -> None:
        """Retired deployments are excluded by default."""
        # Create a retired deployment
        retired = Deployment(
            workspace_id=sample_workspace.id,
            agent_id=enrolled_agent.id,
            artifact_id=sample_artifact.id,
            state=DeploymentState.RETIRED,
            desired_state="RETIRED",
        )
        db_session.add(retired)
        await db_session.commit()

        response = await client.get(
            "/api/v1/deployments",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        retired_items = [i for i in data["items"] if i["state"] == "retired"]
        assert len(retired_items) == 0

    async def test_list_deployments_include_retired(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
        sample_workspace: Workspace,
        enrolled_agent: Agent,
        sample_artifact: Artifact,
    ) -> None:
        """Can include retired deployments."""
        # Create a retired deployment
        retired = Deployment(
            workspace_id=sample_workspace.id,
            agent_id=enrolled_agent.id,
            artifact_id=sample_artifact.id,
            state=DeploymentState.RETIRED,
            desired_state="RETIRED",
        )
        db_session.add(retired)
        await db_session.commit()

        response = await client.get(
            "/api/v1/deployments?include_retired=true",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        retired_items = [i for i in data["items"] if i["state"] == "retired"]
        assert len(retired_items) >= 1

    async def test_list_deployments_unauthenticated(
        self,
        client: AsyncClient,
    ) -> None:
        """Unauthenticated request returns 401."""
        response = await client.get("/api/v1/deployments")
        assert response.status_code == 401


# =============================================================================
# Deployment Create Tests
# =============================================================================


class TestCreateDeployment:
    """Tests for POST /deployments endpoint."""

    async def test_create_deployment_success(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_user: User,
        sample_workspace: Workspace,
        enrolled_agent: Agent,
        sample_artifact: Artifact,
    ) -> None:
        """User with permission can create deployment."""
        from ..models import Permission, Role
        from ..routers.auth import create_access_token

        # Add deployment:create permission to user
        perm = Permission(name="deployment:create", description="Create deployments")
        db_session.add(perm)
        await db_session.commit()
        await db_session.refresh(perm)

        role = Role(
            name="deployment-creator-test",
            description="Deployment Creator",
            organization_id=sample_user.organization_id,
        )
        role.permissions.append(perm)
        db_session.add(role)
        await db_session.commit()
        await db_session.refresh(role)

        # Refresh user with roles relationship before appending
        await db_session.refresh(sample_user, attribute_names=["roles"])
        sample_user.roles.append(role)
        await db_session.commit()

        token = create_access_token(
            user_id=sample_user.id,
            email=sample_user.email,
            org_id=sample_user.organization_id,
            workspace_id=sample_workspace.id,
            is_superuser=False,
            permissions=["deployment:create"],
        )

        response = await client.post(
            "/api/v1/deployments",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "workspace_id": str(sample_workspace.id),
                "agent_id": str(enrolled_agent.id),
                "artifact_id": str(sample_artifact.id),
                "metadata": {"env": "test"},
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["agent_id"] == str(enrolled_agent.id)
        assert data["artifact_id"] == str(sample_artifact.id)
        assert data["state"] == "created"
        assert data["desired_state"] == "DEPLOYED"

    async def test_create_deployment_with_config_blob(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        workspace_id,
        enrolled_agent: Agent,
        sample_artifact: Artifact,
        sample_config_blob: ConfigBlob,
    ) -> None:
        """Can create deployment with config blob."""
        response = await client.post(
            "/api/v1/deployments",
            headers=superuser_headers,
            json={
                "workspace_id": str(workspace_id),
                "agent_id": str(enrolled_agent.id),
                "artifact_id": str(sample_artifact.id),
                "config_blob_id": str(sample_config_blob.id),
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["config_blob_id"] == str(sample_config_blob.id)

    async def test_create_deployment_agent_not_enrolled(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_workspace: Workspace,
        sample_artifact: Artifact,
    ) -> None:
        """Cannot create deployment for non-enrolled agent."""
        # Create a non-enrolled agent
        agent = Agent(
            workspace_id=sample_workspace.id,
            name="pending-agent",
            public_key="y" * 64,
            agent_version="1.0.0",
            trust_state=TrustState.PENDING,
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        response = await client.post(
            "/api/v1/deployments",
            headers=superuser_headers,
            json={
                "workspace_id": str(sample_workspace.id),
                "agent_id": str(agent.id),
                "artifact_id": str(sample_artifact.id),
            },
        )

        assert response.status_code == 400
        assert "not enrolled" in response.json()["detail"]

    async def test_create_deployment_agent_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        workspace_id,
        sample_artifact: Artifact,
    ) -> None:
        """Cannot create deployment with non-existent agent."""
        response = await client.post(
            "/api/v1/deployments",
            headers=superuser_headers,
            json={
                "workspace_id": str(workspace_id),
                "agent_id": str(uuid4()),
                "artifact_id": str(sample_artifact.id),
            },
        )

        assert response.status_code == 404
        assert "Agent not found" in response.json()["detail"]

    async def test_create_deployment_artifact_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        workspace_id,
        enrolled_agent: Agent,
    ) -> None:
        """Cannot create deployment with non-existent artifact."""
        response = await client.post(
            "/api/v1/deployments",
            headers=superuser_headers,
            json={
                "workspace_id": str(workspace_id),
                "agent_id": str(enrolled_agent.id),
                "artifact_id": str(uuid4()),
            },
        )

        assert response.status_code == 404
        assert "Artifact not found" in response.json()["detail"]

    async def test_create_deployment_config_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        workspace_id,
        enrolled_agent: Agent,
        sample_artifact: Artifact,
    ) -> None:
        """Cannot create deployment with non-existent config blob."""
        response = await client.post(
            "/api/v1/deployments",
            headers=superuser_headers,
            json={
                "workspace_id": str(workspace_id),
                "agent_id": str(enrolled_agent.id),
                "artifact_id": str(sample_artifact.id),
                "config_blob_id": str(uuid4()),
            },
        )

        assert response.status_code == 404
        assert "Config blob not found" in response.json()["detail"]

    async def test_create_deployment_without_permission(
        self,
        client: AsyncClient,
        auth_headers: dict,
        workspace_id,
        enrolled_agent: Agent,
        sample_artifact: Artifact,
    ) -> None:
        """User without permission cannot create deployment."""
        response = await client.post(
            "/api/v1/deployments",
            headers=auth_headers,
            json={
                "workspace_id": str(workspace_id),
                "agent_id": str(enrolled_agent.id),
                "artifact_id": str(sample_artifact.id),
            },
        )

        assert response.status_code == 403
        assert "deployment:create" in response.json()["detail"]


# =============================================================================
# Deployment Get Tests
# =============================================================================


class TestGetDeployment:
    """Tests for GET /deployments/{deployment_id} endpoint."""

    async def test_get_deployment_success(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_deployment: Deployment,
    ) -> None:
        """User can get deployment by ID."""
        response = await client.get(
            f"/api/v1/deployments/{sample_deployment.id}",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_deployment.id)
        assert "run_count" in data
        assert "active_run_id" in data

    async def test_get_deployment_not_found(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ) -> None:
        """Non-existent deployment returns 404."""
        response = await client.get(
            f"/api/v1/deployments/{uuid4()}",
            headers=auth_headers,
        )

        assert response.status_code == 404

    async def test_get_deployment_different_org_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
    ) -> None:
        """Cannot get deployment from different organization."""
        # Create different org with deployment
        other_org = Organization(name="other-org-deploy", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        other_ws = Workspace(name="other-ws-deploy", organization_id=other_org.id)
        db_session.add(other_ws)
        await db_session.commit()
        await db_session.refresh(other_ws)

        other_agent = Agent(
            workspace_id=other_ws.id,
            name="other-agent",
            public_key="z" * 64,
            agent_version="1.0.0",
            trust_state=TrustState.ENROLLED,
        )
        db_session.add(other_agent)

        # Create minimal artifact dependencies
        other_strategy = Strategy(
            workspace_id=other_ws.id,
            name="other-strat",
        )
        db_session.add(other_strategy)
        await db_session.commit()
        await db_session.refresh(other_strategy)

        other_version = StrategyVersion(
            strategy_id=other_strategy.id,
            workspace_id=other_ws.id,
            version="1.0.0",
            git_sha="xyz789",
        )
        db_session.add(other_version)
        await db_session.commit()
        await db_session.refresh(other_version)

        other_build = Build(
            strategy_version_id=other_version.id,
            workspace_id=other_ws.id,
            build_number=1,
            status=BuildState.COMPLETED.value,
        )
        db_session.add(other_build)
        await db_session.commit()
        await db_session.refresh(other_build)

        other_artifact = Artifact(
            build_id=other_build.id,
            workspace_id=other_ws.id,
            name="other-artifact.tar.gz",
            format="strategy_bundle",
            digest="sha256:other123",
            size_bytes=512,
        )
        db_session.add(other_artifact)
        await db_session.commit()
        await db_session.refresh(other_agent)
        await db_session.refresh(other_artifact)

        other_deployment = Deployment(
            workspace_id=other_ws.id,
            agent_id=other_agent.id,
            artifact_id=other_artifact.id,
            state=DeploymentState.CREATED,
            desired_state="deployed",
        )
        db_session.add(other_deployment)
        await db_session.commit()
        await db_session.refresh(other_deployment)

        response = await client.get(
            f"/api/v1/deployments/{other_deployment.id}",
            headers=auth_headers,
        )

        assert response.status_code == 403


# =============================================================================
# Deployment Update Tests
# =============================================================================


class TestUpdateDeployment:
    """Tests for PATCH /deployments/{deployment_id} endpoint."""

    async def test_update_deployment_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_deployment: Deployment,
    ) -> None:
        """Superuser can update deployment."""
        response = await client.patch(
            f"/api/v1/deployments/{sample_deployment.id}",
            headers=superuser_headers,
            json={
                "desired_state": "SUSPENDED",
                "metadata": {"updated": True},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["desired_state"] == "SUSPENDED"
        assert data["metadata"]["updated"] is True

    async def test_update_deployment_config_blob(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_deployment: Deployment,
        sample_config_blob: ConfigBlob,
    ) -> None:
        """Can update deployment config blob."""
        response = await client.patch(
            f"/api/v1/deployments/{sample_deployment.id}",
            headers=superuser_headers,
            json={
                "config_blob_id": str(sample_config_blob.id),
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["config_blob_id"] == str(sample_config_blob.id)

    async def test_update_deployment_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Updating non-existent deployment returns 404."""
        response = await client.patch(
            f"/api/v1/deployments/{uuid4()}",
            headers=superuser_headers,
            json={"desired_state": "SUSPENDED"},
        )

        assert response.status_code == 404

    async def test_update_deployment_without_permission(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_deployment: Deployment,
    ) -> None:
        """User without permission cannot update deployment."""
        response = await client.patch(
            f"/api/v1/deployments/{sample_deployment.id}",
            headers=auth_headers,
            json={"desired_state": "SUSPENDED"},
        )

        assert response.status_code == 403


# =============================================================================
# Deployment State Transition Tests
# =============================================================================


class TestDeploymentStateTransition:
    """Tests for POST /deployments/{deployment_id}/transition endpoint."""

    async def test_transition_created_to_approved(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_deployment: Deployment,
    ) -> None:
        """Can transition from CREATED to APPROVED."""
        response = await client.post(
            f"/api/v1/deployments/{sample_deployment.id}/transition",
            headers=superuser_headers,
            json={"target_state": "approved"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "approved"
        assert data["state_changed_at"] is not None

    async def test_transition_approved_to_deploying(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_deployment: Deployment,
    ) -> None:
        """Can transition from APPROVED to DEPLOYING."""
        # Set to APPROVED first
        sample_deployment.state = DeploymentState.APPROVED
        await db_session.commit()

        response = await client.post(
            f"/api/v1/deployments/{sample_deployment.id}/transition",
            headers=superuser_headers,
            json={"target_state": "deploying"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "deploying"

    async def test_transition_deploying_to_deployed(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_deployment: Deployment,
    ) -> None:
        """Can transition from DEPLOYING to DEPLOYED."""
        sample_deployment.state = DeploymentState.DEPLOYING
        await db_session.commit()

        response = await client.post(
            f"/api/v1/deployments/{sample_deployment.id}/transition",
            headers=superuser_headers,
            json={"target_state": "deployed"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "deployed"

    async def test_transition_invalid_state(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_deployment: Deployment,
    ) -> None:
        """Invalid state transition returns error."""
        # CREATED -> DEPLOYED directly is invalid
        response = await client.post(
            f"/api/v1/deployments/{sample_deployment.id}/transition",
            headers=superuser_headers,
            json={"target_state": "deployed"},
        )

        assert response.status_code == 400
        assert "Invalid state transition" in response.json()["detail"]

    async def test_transition_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Transitioning non-existent deployment returns 404."""
        response = await client.post(
            f"/api/v1/deployments/{uuid4()}/transition",
            headers=superuser_headers,
            json={"target_state": "approved"},
        )

        assert response.status_code == 404


# =============================================================================
# Deployment Delete Tests
# =============================================================================


class TestDeleteDeployment:
    """Tests for DELETE /deployments/{deployment_id} endpoint."""

    async def test_delete_deployment_success(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_deployment: Deployment,
    ) -> None:
        """Superuser can delete deployment."""
        response = await client.delete(
            f"/api/v1/deployments/{sample_deployment.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 204

    async def test_delete_deployment_with_active_run(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        deployed_deployment: Deployment,
    ) -> None:
        """Cannot delete deployment with active run."""
        # Create an active run
        run = Run(
            deployment_id=deployed_deployment.id,
            workspace_id=deployed_deployment.workspace_id,
            state=RunState.RUNNING,
            is_paper_trading=True,
        )
        db_session.add(run)
        await db_session.commit()

        response = await client.delete(
            f"/api/v1/deployments/{deployed_deployment.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 400
        assert "active runs" in response.json()["detail"]

    async def test_delete_deployment_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Deleting non-existent deployment returns 404."""
        response = await client.delete(
            f"/api/v1/deployments/{uuid4()}",
            headers=superuser_headers,
        )

        assert response.status_code == 404


# =============================================================================
# Run List Tests
# =============================================================================


class TestListRuns:
    """Tests for GET /deployments/{deployment_id}/runs endpoint."""

    async def test_list_runs_success(
        self,
        client: AsyncClient,
        auth_headers: dict,
        deployed_deployment: Deployment,
        sample_run: Run,
    ) -> None:
        """User can list runs for a deployment."""
        response = await client.get(
            f"/api/v1/deployments/{deployed_deployment.id}/runs",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert data["total"] >= 1

    async def test_list_runs_filter_by_state(
        self,
        client: AsyncClient,
        auth_headers: dict,
        deployed_deployment: Deployment,
        sample_run: Run,
    ) -> None:
        """Can filter runs by state."""
        response = await client.get(
            f"/api/v1/deployments/{deployed_deployment.id}/runs?state=created",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert item["state"] == "created"

    async def test_list_runs_deployment_not_found(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ) -> None:
        """Listing runs for non-existent deployment returns 404."""
        response = await client.get(
            f"/api/v1/deployments/{uuid4()}/runs",
            headers=auth_headers,
        )

        assert response.status_code == 404


# =============================================================================
# Run Create Tests
# =============================================================================


class TestCreateRun:
    """Tests for POST /deployments/{deployment_id}/runs endpoint."""

    async def test_create_run_success(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_user: User,
        sample_workspace: Workspace,
        deployed_deployment: Deployment,
    ) -> None:
        """User with permission can create run."""
        from ..models import Permission, Role
        from ..routers.auth import create_access_token

        # Add run:create permission to user
        perm = Permission(name="run:create", description="Create runs")
        db_session.add(perm)
        await db_session.commit()
        await db_session.refresh(perm)

        role = Role(
            name="run-creator-test",
            description="Run Creator",
            organization_id=sample_user.organization_id,
        )
        role.permissions.append(perm)
        db_session.add(role)
        await db_session.commit()
        await db_session.refresh(role)

        # Refresh user with roles relationship before appending
        await db_session.refresh(sample_user, attribute_names=["roles"])
        sample_user.roles.append(role)
        await db_session.commit()

        token = create_access_token(
            user_id=sample_user.id,
            email=sample_user.email,
            org_id=sample_user.organization_id,
            workspace_id=sample_workspace.id,
            is_superuser=False,
            permissions=["run:create"],
        )

        response = await client.post(
            f"/api/v1/deployments/{deployed_deployment.id}/runs",
            headers={"Authorization": f"Bearer {token}"},
            json={"is_paper_trading": True},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["deployment_id"] == str(deployed_deployment.id)
        assert data["state"] == "created"
        assert data["is_paper_trading"] is True

    async def test_create_run_deployment_not_deployed(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_deployment: Deployment,
    ) -> None:
        """Cannot create run for deployment not in DEPLOYED state."""
        response = await client.post(
            f"/api/v1/deployments/{sample_deployment.id}/runs",
            headers=superuser_headers,
            json={"is_paper_trading": True},
        )

        assert response.status_code == 400
        assert "Cannot create run" in response.json()["detail"]

    async def test_create_run_already_active_run(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        deployed_deployment: Deployment,
    ) -> None:
        """Cannot create run if deployment already has active run."""
        # Create an active run
        run = Run(
            deployment_id=deployed_deployment.id,
            workspace_id=deployed_deployment.workspace_id,
            state=RunState.RUNNING,
            is_paper_trading=True,
        )
        db_session.add(run)
        await db_session.commit()

        response = await client.post(
            f"/api/v1/deployments/{deployed_deployment.id}/runs",
            headers=superuser_headers,
            json={"is_paper_trading": True},
        )

        assert response.status_code == 409
        assert "already has an active run" in response.json()["detail"]

    async def test_create_run_deployment_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Creating run for non-existent deployment returns 404."""
        response = await client.post(
            f"/api/v1/deployments/{uuid4()}/runs",
            headers=superuser_headers,
            json={"is_paper_trading": True},
        )

        assert response.status_code == 404


# =============================================================================
# Run Get Tests
# =============================================================================


class TestGetRun:
    """Tests for GET /deployments/{deployment_id}/runs/{run_id} endpoint."""

    async def test_get_run_success(
        self,
        client: AsyncClient,
        auth_headers: dict,
        deployed_deployment: Deployment,
        sample_run: Run,
    ) -> None:
        """User can get run by ID."""
        response = await client.get(
            f"/api/v1/deployments/{deployed_deployment.id}/runs/{sample_run.id}",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_run.id)

    async def test_get_run_not_found(
        self,
        client: AsyncClient,
        auth_headers: dict,
        deployed_deployment: Deployment,
    ) -> None:
        """Non-existent run returns 404."""
        response = await client.get(
            f"/api/v1/deployments/{deployed_deployment.id}/runs/{uuid4()}",
            headers=auth_headers,
        )

        assert response.status_code == 404


# =============================================================================
# Run Update Tests
# =============================================================================


class TestUpdateRun:
    """Tests for PATCH /deployments/{deployment_id}/runs/{run_id} endpoint."""

    async def test_update_run_success(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        deployed_deployment: Deployment,
        sample_run: Run,
    ) -> None:
        """Superuser can update run."""
        response = await client.patch(
            f"/api/v1/deployments/{deployed_deployment.id}/runs/{sample_run.id}",
            headers=superuser_headers,
            json={
                "error_message": "Test error",
                "error_code": "ERR001",
                "metrics_summary": {"pnl": 100},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["error_message"] == "Test error"
        assert data["error_code"] == "ERR001"
        assert data["metrics_summary"]["pnl"] == 100

    async def test_update_run_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        deployed_deployment: Deployment,
    ) -> None:
        """Updating non-existent run returns 404."""
        response = await client.patch(
            f"/api/v1/deployments/{deployed_deployment.id}/runs/{uuid4()}",
            headers=superuser_headers,
            json={"error_message": "Test"},
        )

        assert response.status_code == 404


# =============================================================================
# Run State Transition Tests
# =============================================================================


class TestRunStateTransition:
    """Tests for POST /deployments/{deployment_id}/runs/{run_id}/transition endpoint."""

    async def test_transition_created_to_approved(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        deployed_deployment: Deployment,
        sample_run: Run,
    ) -> None:
        """Can transition from CREATED to APPROVED."""
        response = await client.post(
            f"/api/v1/deployments/{deployed_deployment.id}/runs/{sample_run.id}/transition",
            headers=superuser_headers,
            json={"target_state": "approved"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "approved"

    async def test_transition_to_running_sets_started_at(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        deployed_deployment: Deployment,
        sample_run: Run,
    ) -> None:
        """Transitioning to RUNNING sets started_at."""
        # Move through state machine
        sample_run.state = RunState.STARTING
        await db_session.commit()

        response = await client.post(
            f"/api/v1/deployments/{deployed_deployment.id}/runs/{sample_run.id}/transition",
            headers=superuser_headers,
            json={"target_state": "running"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "running"
        assert data["started_at"] is not None

    async def test_transition_to_stopped_sets_stopped_at(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        deployed_deployment: Deployment,
        sample_run: Run,
    ) -> None:
        """Transitioning to STOPPED sets stopped_at."""
        # Move through state machine
        sample_run.state = RunState.STOPPING
        await db_session.commit()

        response = await client.post(
            f"/api/v1/deployments/{deployed_deployment.id}/runs/{sample_run.id}/transition",
            headers=superuser_headers,
            json={"target_state": "stopped"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "stopped"
        assert data["stopped_at"] is not None

    async def test_transition_invalid_state(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        deployed_deployment: Deployment,
        sample_run: Run,
    ) -> None:
        """Invalid state transition returns error."""
        # CREATED -> RUNNING directly is invalid
        response = await client.post(
            f"/api/v1/deployments/{deployed_deployment.id}/runs/{sample_run.id}/transition",
            headers=superuser_headers,
            json={"target_state": "running"},
        )

        assert response.status_code == 400
        assert "Invalid state transition" in response.json()["detail"]

    async def test_transition_from_terminal_state(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        deployed_deployment: Deployment,
        sample_run: Run,
    ) -> None:
        """Cannot transition from terminal state."""
        sample_run.state = RunState.COMPLETED
        await db_session.commit()

        response = await client.post(
            f"/api/v1/deployments/{deployed_deployment.id}/runs/{sample_run.id}/transition",
            headers=superuser_headers,
            json={"target_state": "running"},
        )

        assert response.status_code == 400
        assert "Invalid state transition" in response.json()["detail"]

    async def test_transition_run_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        deployed_deployment: Deployment,
    ) -> None:
        """Transitioning non-existent run returns 404."""
        response = await client.post(
            f"/api/v1/deployments/{deployed_deployment.id}/runs/{uuid4()}/transition",
            headers=superuser_headers,
            json={"target_state": "approved"},
        )

        assert response.status_code == 404
