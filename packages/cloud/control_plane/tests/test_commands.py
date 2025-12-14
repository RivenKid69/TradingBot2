# -*- coding: utf-8 -*-
"""Tests for Commands Router."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import (
    Agent,
    ApprovalRecord,
    Artifact,
    Build,
    ChangeClass,
    Command,
    CommandStatus,
    Deployment,
    DeploymentState,
    Run,
    Strategy,
    StrategyVersion,
    TrustState,
    Workspace,
)

pytestmark = pytest.mark.asyncio


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def enrolled_agent(
    db_session: AsyncSession,
    sample_workspace: Workspace,
) -> Agent:
    """Create an enrolled agent for testing."""
    agent = Agent(
        workspace_id=sample_workspace.id,
        name="command-test-agent",
        public_key="c" * 64,
        agent_version="1.0.0",
        trust_state=TrustState.ENROLLED.value,
        capabilities=["execute"],
    )
    db_session.add(agent)
    await db_session.commit()
    await db_session.refresh(agent)
    return agent


@pytest.fixture
async def unenrolled_agent(
    db_session: AsyncSession,
    sample_workspace: Workspace,
) -> Agent:
    """Create an unenrolled (pending) agent."""
    agent = Agent(
        workspace_id=sample_workspace.id,
        name="unenrolled-agent",
        public_key="u" * 64,
        agent_version="1.0.0",
        trust_state=TrustState.PENDING.value,
        capabilities=["execute"],
    )
    db_session.add(agent)
    await db_session.commit()
    await db_session.refresh(agent)
    return agent


@pytest.fixture
async def sample_strategy(
    db_session: AsyncSession,
    sample_workspace: Workspace,
) -> Strategy:
    """Create a sample strategy for commands testing."""
    strategy = Strategy(
        workspace_id=sample_workspace.id,
        name="cmd-test-strategy",
        description="Strategy for command tests",
    )
    db_session.add(strategy)
    await db_session.commit()
    await db_session.refresh(strategy)
    return strategy


@pytest.fixture
async def sample_strategy_version(
    db_session: AsyncSession,
    sample_strategy: Strategy,
    sample_workspace: Workspace,
) -> StrategyVersion:
    """Create a sample strategy version."""
    version = StrategyVersion(
        strategy_id=sample_strategy.id,
        workspace_id=sample_workspace.id,
        version="1.0.0",
        git_sha="sha256:cmdtest123",
    )
    db_session.add(version)
    await db_session.commit()
    await db_session.refresh(version)
    return version


@pytest.fixture
async def sample_build(
    db_session: AsyncSession,
    sample_strategy_version: StrategyVersion,
    sample_workspace: Workspace,
) -> Build:
    """Create a sample build."""
    build = Build(
        strategy_version_id=sample_strategy_version.id,
        workspace_id=sample_workspace.id,
        build_number=1,
        status="completed",
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
    """Create a sample artifact."""
    artifact = Artifact(
        build_id=sample_build.id,
        workspace_id=sample_workspace.id,
        name="cmd-test.whl",
        format="wheel",
        digest="sha256:cmdartifact123",
        size_bytes=512000,
    )
    db_session.add(artifact)
    await db_session.commit()
    await db_session.refresh(artifact)
    return artifact


@pytest.fixture
async def sample_deployment(
    db_session: AsyncSession,
    sample_workspace: Workspace,
    enrolled_agent: Agent,
    sample_artifact: Artifact,
) -> Deployment:
    """Create a sample deployment for testing."""
    deployment = Deployment(
        workspace_id=sample_workspace.id,
        agent_id=enrolled_agent.id,
        artifact_id=sample_artifact.id,
        state=DeploymentState.DEPLOYED.value,
    )
    db_session.add(deployment)
    await db_session.commit()
    await db_session.refresh(deployment)
    return deployment


@pytest.fixture
async def sample_run(
    db_session: AsyncSession,
    sample_workspace: Workspace,
    sample_deployment: Deployment,
) -> Run:
    """Create a sample run for testing."""
    run = Run(
        workspace_id=sample_workspace.id,
        deployment_id=sample_deployment.id,
        state="running",
    )
    db_session.add(run)
    await db_session.commit()
    await db_session.refresh(run)
    return run


@pytest.fixture
async def sample_command(
    db_session: AsyncSession,
    sample_workspace: Workspace,
    enrolled_agent: Agent,
) -> Command:
    """Create a sample command for testing."""
    command = Command(
        workspace_id=sample_workspace.id,
        idempotency_key="test-cmd-key-001",
        agent_id=enrolled_agent.id,
        command_type="START",
        payload_ref="a" * 64,
        change_class=ChangeClass.OPERATIONAL.value,
        requires_approval=False,
        status=CommandStatus.PENDING.value,
    )
    db_session.add(command)
    await db_session.commit()
    await db_session.refresh(command)
    return command


@pytest.fixture
async def approval_required_command(
    db_session: AsyncSession,
    sample_workspace: Workspace,
    enrolled_agent: Agent,
) -> Command:
    """Create a command that requires approval."""
    command = Command(
        workspace_id=sample_workspace.id,
        idempotency_key="approval-required-key",
        agent_id=enrolled_agent.id,
        command_type="CONFIG_UPDATE",
        payload_ref="b" * 64,
        change_class=ChangeClass.OPERATIONAL.value,
        requires_approval=True,
        status=CommandStatus.PENDING_APPROVAL.value,
    )
    db_session.add(command)
    await db_session.commit()
    await db_session.refresh(command)
    return command


@pytest.fixture
async def sent_command(
    db_session: AsyncSession,
    sample_workspace: Workspace,
    enrolled_agent: Agent,
) -> Command:
    """Create a command in SENT status."""
    command = Command(
        workspace_id=sample_workspace.id,
        idempotency_key="sent-cmd-key",
        agent_id=enrolled_agent.id,
        command_type="EXECUTE",
        payload_ref="s" * 64,
        change_class=ChangeClass.OPERATIONAL.value,
        requires_approval=False,
        status=CommandStatus.SENT.value,
        sent_at=datetime.now(timezone.utc),
    )
    db_session.add(command)
    await db_session.commit()
    await db_session.refresh(command)
    return command


@pytest.fixture
async def expired_command(
    db_session: AsyncSession,
    sample_workspace: Workspace,
    enrolled_agent: Agent,
) -> Command:
    """Create an expired command."""
    command = Command(
        workspace_id=sample_workspace.id,
        idempotency_key="expired-cmd-key",
        agent_id=enrolled_agent.id,
        command_type="CHECK",
        payload_ref="e" * 64,
        change_class=ChangeClass.OPERATIONAL.value,
        requires_approval=False,
        status=CommandStatus.PENDING.value,
        expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    db_session.add(command)
    await db_session.commit()
    await db_session.refresh(command)
    return command


# ============================================================================
# Test List Commands
# ============================================================================


class TestListCommands:
    """Tests for GET /commands endpoint."""

    async def test_list_commands_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_command: Command,
    ) -> None:
        """Superuser can list all commands."""
        response = await client.get(
            "/api/v1/commands",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert data["total"] >= 1

    async def test_list_commands_filter_by_workspace(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_command: Command,
        workspace_id,
    ) -> None:
        """Filter commands by workspace."""
        response = await client.get(
            f"/api/v1/commands?workspace_id={workspace_id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        for item in data["items"]:
            assert item["workspace_id"] == str(workspace_id)

    async def test_list_commands_filter_by_status(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_command: Command,
    ) -> None:
        """Filter commands by status."""
        response = await client.get(
            "/api/v1/commands?command_status=pending",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert item["status"] == "pending"

    async def test_list_commands_filter_by_agent(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_command: Command,
        enrolled_agent: Agent,
    ) -> None:
        """Filter commands by agent."""
        response = await client.get(
            f"/api/v1/commands?agent_id={enrolled_agent.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        for item in data["items"]:
            assert item["agent_id"] == str(enrolled_agent.id)

    async def test_list_commands_filter_by_requires_approval(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        approval_required_command: Command,
    ) -> None:
        """Filter commands that require approval."""
        response = await client.get(
            "/api/v1/commands?requires_approval=true",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert item["requires_approval"] is True

    async def test_list_commands_filter_by_change_class(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_command: Command,
    ) -> None:
        """Filter commands by change class."""
        response = await client.get(
            "/api/v1/commands?change_class=operational",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert item["change_class"] == "operational"


# ============================================================================
# Test Create Command
# ============================================================================


class TestCreateCommand:
    """Tests for POST /commands endpoint."""

    async def test_create_command_success(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        enrolled_agent: Agent,
        workspace_id,
    ) -> None:
        """Superuser can create a command."""
        response = await client.post(
            f"/api/v1/commands?workspace_id={workspace_id}",
            headers=superuser_headers,
            json={
                "agent_id": str(enrolled_agent.id),
                "command_type": "REQUEST_START_RUN",
                "payload_ref": "sha256:" + "x" * 64,
                "change_class": "trading_impacting",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["agent_id"] == str(enrolled_agent.id)
        assert data["command_type"] == "REQUEST_START_RUN"
        assert data["status"] == "pending_approval"
        assert data["idempotency_key"] is not None

    async def test_create_command_with_idempotency_key(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        enrolled_agent: Agent,
        workspace_id,
    ) -> None:
        """Create command with custom idempotency key."""
        idem_key = "custom-idem-key-123"
        response = await client.post(
            f"/api/v1/commands?workspace_id={workspace_id}&idempotency_key={idem_key}",
            headers=superuser_headers,
            json={
                "agent_id": str(enrolled_agent.id),
                "command_type": "REQUEST_STOP_RUN",
                "payload_ref": "sha256:" + "y" * 64,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["idempotency_key"] == idem_key

    async def test_create_command_duplicate_idempotency_key(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_command: Command,
        enrolled_agent: Agent,
        workspace_id,
    ) -> None:
        """Cannot create command with duplicate idempotency key."""
        response = await client.post(
            f"/api/v1/commands?workspace_id={workspace_id}&idempotency_key={sample_command.idempotency_key}",
            headers=superuser_headers,
            json={
                "agent_id": str(enrolled_agent.id),
                "command_type": "REQUEST_STOP_RUN",
                "payload_ref": "sha256:" + "z" * 64,
            },
        )

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    async def test_create_command_agent_not_enrolled(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        unenrolled_agent: Agent,
        workspace_id,
    ) -> None:
        """Cannot create command for unenrolled agent."""
        response = await client.post(
            f"/api/v1/commands?workspace_id={workspace_id}",
            headers=superuser_headers,
            json={
                "agent_id": str(unenrolled_agent.id),
                "command_type": "REQUEST_STOP_RUN",
                "payload_ref": "sha256:" + "z" * 64,
            },
        )

        assert response.status_code == 400
        assert "not enrolled" in response.json()["detail"]

    async def test_create_command_agent_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_workspace: Workspace,
    ) -> None:
        """Cannot create command for non-existent agent."""
        response = await client.post(
            f"/api/v1/commands?workspace_id={sample_workspace.id}",
            headers=superuser_headers,
            json={
                "agent_id": str(uuid4()),
                "command_type": "REQUEST_STOP_RUN",
                "payload_ref": "sha256:" + "z" * 64,
            },
        )

        assert response.status_code == 404
        assert "Agent not found" in response.json()["detail"]

    async def test_create_command_with_deployment(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        enrolled_agent: Agent,
        sample_deployment: Deployment,
        workspace_id,
    ) -> None:
        """Create command with deployment reference."""
        response = await client.post(
            f"/api/v1/commands?workspace_id={workspace_id}",
            headers=superuser_headers,
            json={
                "agent_id": str(enrolled_agent.id),
                "deployment_id": str(sample_deployment.id),
                "command_type": "REQUEST_UPDATE_CONFIG",
                "payload_ref": "sha256:" + "d" * 64,
                "change_class": "trading_impacting",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["deployment_id"] == str(sample_deployment.id)

    async def test_create_command_with_run(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        enrolled_agent: Agent,
        sample_run: Run,
        workspace_id,
    ) -> None:
        """Create command with run reference."""
        response = await client.post(
            f"/api/v1/commands?workspace_id={workspace_id}",
            headers=superuser_headers,
            json={
                "agent_id": str(enrolled_agent.id),
                "run_id": str(sample_run.id),
                "command_type": "REQUEST_PAUSE_RUN",
                "payload_ref": "sha256:" + "r" * 64,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["run_id"] == str(sample_run.id)

    async def test_create_command_requires_approval(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        enrolled_agent: Agent,
        sample_workspace: Workspace,
    ) -> None:
        """Create command that requires approval."""
        response = await client.post(
            f"/api/v1/commands?workspace_id={sample_workspace.id}",
            headers=superuser_headers,
            json={
                "agent_id": str(enrolled_agent.id),
                "command_type": "REQUEST_ROTATE_AGENT_SESSION",
                "payload_ref": "sha256:" + "a" * 64,
                "change_class": "security_sensitive",
                "requires_approval": True,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["requires_approval"] is True
        assert data["change_class"] == "security_sensitive"

    async def test_create_command_workspace_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        enrolled_agent: Agent,
    ) -> None:
        """Cannot create command in non-existent workspace."""
        response = await client.post(
            f"/api/v1/commands?workspace_id={uuid4()}",
            headers=superuser_headers,
            json={
                "agent_id": str(enrolled_agent.id),
                "command_type": "REQUEST_STOP_RUN",
                "payload_ref": "sha256:" + "z" * 64,
            },
        )

        assert response.status_code == 404
        assert "Workspace not found" in response.json()["detail"]


# ============================================================================
# Test Get Command
# ============================================================================


class TestGetCommand:
    """Tests for GET /commands/{command_id} endpoint."""

    async def test_get_command_success(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_command: Command,
    ) -> None:
        """Can get command by ID."""
        response = await client.get(
            f"/api/v1/commands/{sample_command.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_command.id)
        assert data["command_type"] == sample_command.command_type
        assert "approvals" in data

    async def test_get_command_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Non-existent command returns 404."""
        response = await client.get(
            f"/api/v1/commands/{uuid4()}",
            headers=superuser_headers,
        )

        assert response.status_code == 404
        assert "Command not found" in response.json()["detail"]

    async def test_get_command_different_org_forbidden(
        self,
        client: AsyncClient,
        auth_headers: dict,
        db_session: AsyncSession,
    ) -> None:
        """User cannot get command from different organization."""
        # Create command in different workspace/org
        other_org_ws = Workspace(
            name="other-org-workspace",
            organization_id=uuid4(),
        )
        db_session.add(other_org_ws)
        await db_session.commit()
        await db_session.refresh(other_org_ws)

        agent = Agent(
            workspace_id=other_org_ws.id,
            name="other-agent",
            public_key="o" * 64,
            agent_version="1.0.0",
            trust_state=TrustState.ENROLLED.value,
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)

        command = Command(
            workspace_id=other_org_ws.id,
            idempotency_key="other-org-cmd",
            agent_id=agent.id,
            command_type="START",
            payload_ref="o" * 64,
            status=CommandStatus.PENDING.value,
        )
        db_session.add(command)
        await db_session.commit()
        await db_session.refresh(command)

        response = await client.get(
            f"/api/v1/commands/{command.id}",
            headers=auth_headers,
        )

        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]


# ============================================================================
# Test Update Command
# ============================================================================


class TestUpdateCommand:
    """Tests for PATCH /commands/{command_id} endpoint."""

    async def test_update_command_success(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_command: Command,
    ) -> None:
        """Can update command fields."""
        new_expires = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        response = await client.patch(
            f"/api/v1/commands/{sample_command.id}",
            headers=superuser_headers,
            json={
                "expires_at": new_expires,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["expires_at"] is not None

    async def test_update_command_result(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sent_command: Command,
    ) -> None:
        """Can update command result."""
        response = await client.patch(
            f"/api/v1/commands/{sent_command.id}",
            headers=superuser_headers,
            json={
                "result": {"outcome": "success", "details": "task completed"},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["result"]["outcome"] == "success"

    async def test_update_command_in_terminal_state(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_command: Command,
    ) -> None:
        """Cannot update command in terminal state."""
        # Move command to terminal state
        sample_command.status = CommandStatus.EXECUTED.value
        await db_session.commit()

        response = await client.patch(
            f"/api/v1/commands/{sample_command.id}",
            headers=superuser_headers,
            json={
                "result": {"new": "data"},
            },
        )

        assert response.status_code == 400
        assert "terminal state" in response.json()["detail"]

    async def test_update_command_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Cannot update non-existent command."""
        response = await client.patch(
            f"/api/v1/commands/{uuid4()}",
            headers=superuser_headers,
            json={"result": {"test": True}},
        )

        assert response.status_code == 404


# ============================================================================
# Test Command State Transitions
# ============================================================================


class TestCommandStateTransition:
    """Tests for POST /commands/{command_id}/transition endpoint."""

    async def test_transition_pending_to_sent(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_command: Command,
    ) -> None:
        """Can transition from PENDING to SENT."""
        response = await client.post(
            f"/api/v1/commands/{sample_command.id}/transition",
            headers=superuser_headers,
            json={"new_status": "sent"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "sent"
        assert data["sent_at"] is not None

    async def test_transition_sent_to_acknowledged(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sent_command: Command,
    ) -> None:
        """Can transition from SENT to ACKNOWLEDGED."""
        response = await client.post(
            f"/api/v1/commands/{sent_command.id}/transition",
            headers=superuser_headers,
            json={"new_status": "acknowledged"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "acknowledged"
        assert data["acknowledged_at"] is not None

    async def test_transition_acknowledged_to_executed(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sent_command: Command,
    ) -> None:
        """Can transition from ACKNOWLEDGED to EXECUTED with result."""
        # First move to ACKNOWLEDGED
        sent_command.status = CommandStatus.ACKNOWLEDGED.value
        sent_command.acknowledged_at = datetime.now(timezone.utc)
        await db_session.commit()

        response = await client.post(
            f"/api/v1/commands/{sent_command.id}/transition",
            headers=superuser_headers,
            json={
                "new_status": "executed",
                "result": {"exit_code": 0, "output": "done"},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "executed"
        assert data["executed_at"] is not None
        assert data["result"]["exit_code"] == 0

    async def test_transition_to_failed_with_error(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sent_command: Command,
    ) -> None:
        """Can transition to FAILED with error message."""
        response = await client.post(
            f"/api/v1/commands/{sent_command.id}/transition",
            headers=superuser_headers,
            json={
                "new_status": "failed",
                "error_message": "Network timeout",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["error_message"] == "Network timeout"

    async def test_invalid_transition(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_command: Command,
    ) -> None:
        """Invalid state transition returns 400."""
        response = await client.post(
            f"/api/v1/commands/{sample_command.id}/transition",
            headers=superuser_headers,
            json={"new_status": "executed"},  # Can't go from PENDING to EXECUTED
        )

        assert response.status_code == 400
        assert "Invalid transition" in response.json()["detail"]

    async def test_transition_requires_approval_validation(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        enrolled_agent: Agent,
        sample_workspace: Workspace,
    ) -> None:
        """Command requiring approval cannot go directly to SENT."""
        command = Command(
            workspace_id=sample_workspace.id,
            idempotency_key="approval-test-cmd",
            agent_id=enrolled_agent.id,
            command_type="CRITICAL",
            payload_ref="c" * 64,
            requires_approval=True,
            status=CommandStatus.PENDING.value,
        )
        db_session.add(command)
        await db_session.commit()
        await db_session.refresh(command)

        response = await client.post(
            f"/api/v1/commands/{command.id}/transition",
            headers=superuser_headers,
            json={"new_status": "sent"},
        )

        assert response.status_code == 400
        assert "requires approval" in response.json()["detail"]

    async def test_transition_expired_command(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        expired_command: Command,
    ) -> None:
        """Expired command can only transition to EXPIRED."""
        # Try to transition to SENT
        response = await client.post(
            f"/api/v1/commands/{expired_command.id}/transition",
            headers=superuser_headers,
            json={"new_status": "sent"},
        )

        assert response.status_code == 400
        assert "expired" in response.json()["detail"].lower()

        # Can transition to EXPIRED
        response = await client.post(
            f"/api/v1/commands/{expired_command.id}/transition",
            headers=superuser_headers,
            json={"new_status": "expired"},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "expired"

    async def test_transition_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Cannot transition non-existent command."""
        response = await client.post(
            f"/api/v1/commands/{uuid4()}/transition",
            headers=superuser_headers,
            json={"new_status": "sent"},
        )

        assert response.status_code == 404


# ============================================================================
# Test List Approvals
# ============================================================================


class TestListApprovals:
    """Tests for GET /commands/{command_id}/approvals endpoint."""

    async def test_list_approvals_empty(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        approval_required_command: Command,
    ) -> None:
        """List approvals returns empty list when none exist."""
        response = await client.get(
            f"/api/v1/commands/{approval_required_command.id}/approvals",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data == []

    async def test_list_approvals_with_records(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        approval_required_command: Command,
    ) -> None:
        """List approvals returns existing records."""
        # Create approval record
        approval = ApprovalRecord(
            workspace_id=approval_required_command.workspace_id,
            command_id=approval_required_command.id,
            approved=True,
            approved_by="test-approver",
            reason="Looks good",
        )
        db_session.add(approval)
        await db_session.commit()

        response = await client.get(
            f"/api/v1/commands/{approval_required_command.id}/approvals",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["approved"] is True
        assert data[0]["reason"] == "Looks good"

    async def test_list_approvals_command_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Cannot list approvals for non-existent command."""
        response = await client.get(
            f"/api/v1/commands/{uuid4()}/approvals",
            headers=superuser_headers,
        )

        assert response.status_code == 404


# ============================================================================
# Test Create Approval
# ============================================================================


class TestCreateApproval:
    """Tests for POST /commands/{command_id}/approvals endpoint."""

    async def test_create_approval_approved(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        approval_required_command: Command,
    ) -> None:
        """Create approval that approves the command."""
        response = await client.post(
            f"/api/v1/commands/{approval_required_command.id}/approvals",
            headers=superuser_headers,
            json={
                "approved": True,
                "reason": "Reviewed and approved",
                "evidence_hash": "hash123",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["approved"] is True
        assert data["reason"] == "Reviewed and approved"
        assert data["evidence_hash"] == "hash123"

        # Verify command state changed
        get_response = await client.get(
            f"/api/v1/commands/{approval_required_command.id}",
            headers=superuser_headers,
        )
        assert get_response.json()["status"] == "approved"

    async def test_create_approval_rejected(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        enrolled_agent: Agent,
        sample_workspace: Workspace,
    ) -> None:
        """Create approval that rejects the command."""
        # Create a new command requiring approval
        command = Command(
            workspace_id=sample_workspace.id,
            idempotency_key="reject-test-cmd",
            agent_id=enrolled_agent.id,
            command_type="DANGEROUS",
            payload_ref="d" * 64,
            requires_approval=True,
            status=CommandStatus.PENDING_APPROVAL.value,
        )
        db_session.add(command)
        await db_session.commit()
        await db_session.refresh(command)

        response = await client.post(
            f"/api/v1/commands/{command.id}/approvals",
            headers=superuser_headers,
            json={
                "approved": False,
                "reason": "Security concerns",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["approved"] is False

        # Verify command state changed to REJECTED
        get_response = await client.get(
            f"/api/v1/commands/{command.id}",
            headers=superuser_headers,
        )
        assert get_response.json()["status"] == "rejected"

    async def test_create_approval_command_not_pending_approval(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_command: Command,
    ) -> None:
        """Cannot approve command not in PENDING_APPROVAL state."""
        response = await client.post(
            f"/api/v1/commands/{sample_command.id}/approvals",
            headers=superuser_headers,
            json={"approved": True},
        )

        assert response.status_code == 400
        assert "not pending approval" in response.json()["detail"]

    async def test_create_approval_expired_command(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        enrolled_agent: Agent,
        sample_workspace: Workspace,
    ) -> None:
        """Cannot approve expired command."""
        command = Command(
            workspace_id=sample_workspace.id,
            idempotency_key="expired-approval-cmd",
            agent_id=enrolled_agent.id,
            command_type="LATE",
            payload_ref="l" * 64,
            requires_approval=True,
            status=CommandStatus.PENDING_APPROVAL.value,
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        db_session.add(command)
        await db_session.commit()
        await db_session.refresh(command)

        response = await client.post(
            f"/api/v1/commands/{command.id}/approvals",
            headers=superuser_headers,
            json={"approved": True},
        )

        assert response.status_code == 400
        assert "expired" in response.json()["detail"].lower()

    async def test_create_approval_command_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Cannot approve non-existent command."""
        response = await client.post(
            f"/api/v1/commands/{uuid4()}/approvals",
            headers=superuser_headers,
            json={"approved": True},
        )

        assert response.status_code == 404


# ============================================================================
# Test Get Approval
# ============================================================================


class TestGetApproval:
    """Tests for GET /commands/{command_id}/approvals/{approval_id} endpoint."""

    async def test_get_approval_success(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        approval_required_command: Command,
    ) -> None:
        """Can get specific approval."""
        approval = ApprovalRecord(
            workspace_id=approval_required_command.workspace_id,
            command_id=approval_required_command.id,
            approved=True,
            approved_by="tester",
            reason="Test approval",
            attestation={"signer": "test"},
        )
        db_session.add(approval)
        await db_session.commit()
        await db_session.refresh(approval)

        response = await client.get(
            f"/api/v1/commands/{approval_required_command.id}/approvals/{approval.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(approval.id)
        assert data["approved"] is True
        assert data["attestation"]["signer"] == "test"

    async def test_get_approval_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        approval_required_command: Command,
    ) -> None:
        """Non-existent approval returns 404."""
        response = await client.get(
            f"/api/v1/commands/{approval_required_command.id}/approvals/{uuid4()}",
            headers=superuser_headers,
        )

        assert response.status_code == 404

    async def test_get_approval_command_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Non-existent command returns 404."""
        response = await client.get(
            f"/api/v1/commands/{uuid4()}/approvals/{uuid4()}",
            headers=superuser_headers,
        )

        assert response.status_code == 404


# ============================================================================
# Test Expire Stale Commands
# ============================================================================


class TestExpireStaleCommands:
    """Tests for POST /commands/expire-stale endpoint."""

    async def test_expire_stale_commands_success(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        enrolled_agent: Agent,
        sample_workspace: Workspace,
        workspace_id,
    ) -> None:
        """Expire stale commands in workspace."""
        # Create expired commands
        for i in range(3):
            cmd = Command(
                workspace_id=workspace_id,
                idempotency_key=f"stale-cmd-{i}",
                agent_id=enrolled_agent.id,
                command_type="STALE",
                payload_ref=f"{i}" * 64,
                status=CommandStatus.PENDING.value,
                expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
            )
            db_session.add(cmd)
        await db_session.commit()

        response = await client.post(
            f"/api/v1/commands/expire-stale?workspace_id={workspace_id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["expired_count"] >= 3

    async def test_expire_stale_no_expired_commands(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_workspace: Workspace,
    ) -> None:
        """No commands to expire returns 0."""
        response = await client.post(
            f"/api/v1/commands/expire-stale?workspace_id={sample_workspace.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["expired_count"] >= 0

    async def test_expire_stale_workspace_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Non-existent workspace returns 404."""
        response = await client.post(
            f"/api/v1/commands/expire-stale?workspace_id={uuid4()}",
            headers=superuser_headers,
        )

        assert response.status_code == 404

    async def test_expire_stale_does_not_affect_terminal_states(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        enrolled_agent: Agent,
        sample_workspace: Workspace,
        workspace_id,
    ) -> None:
        """Expire does not affect commands already in terminal states."""
        # Create command in terminal state with past expiry
        cmd = Command(
            workspace_id=workspace_id,
            idempotency_key="terminal-stale-cmd",
            agent_id=enrolled_agent.id,
            command_type="DONE",
            payload_ref="t" * 64,
            status=CommandStatus.EXECUTED.value,  # Terminal state
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        db_session.add(cmd)
        await db_session.commit()
        await db_session.refresh(cmd)

        response = await client.post(
            f"/api/v1/commands/expire-stale?workspace_id={workspace_id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200

        # Verify the terminal command wasn't changed
        await db_session.refresh(cmd)
        assert cmd.status == CommandStatus.EXECUTED.value
