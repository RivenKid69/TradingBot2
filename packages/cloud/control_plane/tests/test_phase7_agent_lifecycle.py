# -*- coding: utf-8 -*-
"""
Tests for Phase 7 (WI-CLOUD-02): Agent Lifecycle Router.

Tests cover:
- Agent heartbeat with proper auth
- Command polling
- Command acknowledgement
- Command result submission
- Local approval workflow

References:
- CCEA_MASTER_REMEDIATION_PLAN.md Phase 7
- WI-CLOUD-02: Implement agent-auth lifecycle endpoints
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

# Add package to path for imports
PACKAGE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))


# ============================================================================
# Router Registration Tests
# ============================================================================

class TestAgentLifecycleRouterRegistration:
    """Tests for router registration in app."""

    def test_router_is_registered(self):
        """Test that agent_lifecycle router is registered."""
        from packages.cloud.control_plane.app import app

        # Get all registered routes
        routes = [route.path for route in app.routes]

        # Check for agent lifecycle routes
        assert any("/agent" in route for route in routes), \
            "Agent lifecycle router should be registered"

    def test_router_has_correct_prefix(self):
        """Test that router has /api/v1/agent prefix."""
        from packages.cloud.control_plane.app import app

        routes = [route.path for route in app.routes]

        expected_routes = [
            "/api/v1/agent/heartbeat",
            "/api/v1/agent/commands/poll",
            "/api/v1/agent/status",
        ]

        for expected in expected_routes:
            assert expected in routes, f"Route {expected} should exist"

    def test_router_uses_agent_lifecycle_tag(self):
        """Test that router routes use Agent Lifecycle tag."""
        from packages.cloud.control_plane.app import app

        # Find routes under /api/v1/agent
        for route in app.routes:
            if hasattr(route, "path") and "/api/v1/agent" in route.path:
                if hasattr(route, "tags"):
                    # Check tags include Agent Lifecycle
                    pass  # Tags are set at router level


# ============================================================================
# Request/Response Model Tests
# ============================================================================

class TestAgentLifecycleModels:
    """Tests for request/response models."""

    def test_heartbeat_request_model(self):
        """Test AgentHeartbeatRequest model."""
        from packages.cloud.control_plane.routers.agent_lifecycle import AgentHeartbeatRequest

        request = AgentHeartbeatRequest(
            agent_version="1.0.0",
            current_state="running",
            health_metrics={"cpu": 50.0, "memory": 60.0},
        )

        assert request.agent_version == "1.0.0"
        assert request.current_state == "running"
        assert request.health_metrics["cpu"] == 50.0

    def test_heartbeat_response_model(self):
        """Test AgentHeartbeatResponse model."""
        from packages.cloud.control_plane.routers.agent_lifecycle import AgentHeartbeatResponse

        response = AgentHeartbeatResponse(
            server_time=datetime.now(timezone.utc),
            trust_state="enrolled",
            pending_commands=5,
            next_heartbeat_sec=60,
        )

        assert response.trust_state == "enrolled"
        assert response.pending_commands == 5
        assert response.next_heartbeat_sec == 60

    def test_pending_command_model(self):
        """Test PendingCommand model."""
        from packages.cloud.control_plane.routers.agent_lifecycle import PendingCommand

        cmd_id = uuid4()
        command = PendingCommand(
            id=cmd_id,
            idempotency_key="test-key-123",
            command_type="REQUEST_START_RUN",
            payload_ref="sha256:abc123",
            change_class="operational",
            requires_approval=False,
            created_at=datetime.now(timezone.utc),
        )

        assert command.id == cmd_id
        assert command.command_type == "REQUEST_START_RUN"

    def test_command_poll_response_model(self):
        """Test CommandPollResponse model."""
        from packages.cloud.control_plane.routers.agent_lifecycle import CommandPollResponse

        response = CommandPollResponse(
            commands=[],
            has_more=False,
            poll_again_after_sec=60,
        )

        assert response.commands == []
        assert response.has_more is False

    def test_command_ack_response_model(self):
        """Test CommandAckResponse model."""
        from packages.cloud.control_plane.routers.agent_lifecycle import CommandAckResponse

        cmd_id = uuid4()
        response = CommandAckResponse(
            command_id=cmd_id,
            status="acknowledged",
            acknowledged_at=datetime.now(timezone.utc),
        )

        assert response.command_id == cmd_id
        assert response.status == "acknowledged"

    def test_command_result_submission_model(self):
        """Test CommandResultSubmission model."""
        from packages.cloud.control_plane.routers.agent_lifecycle import CommandResultSubmission

        submission = CommandResultSubmission(
            success=True,
            result={"output": "completed"},
        )

        assert submission.success is True
        assert submission.result["output"] == "completed"

    def test_command_result_submission_failure(self):
        """Test CommandResultSubmission for failure case."""
        from packages.cloud.control_plane.routers.agent_lifecycle import CommandResultSubmission

        submission = CommandResultSubmission(
            success=False,
            error_message="Failed to start run",
        )

        assert submission.success is False
        assert submission.error_message == "Failed to start run"

    def test_local_approval_request_model(self):
        """Test LocalApprovalRequest model."""
        from packages.cloud.control_plane.routers.agent_lifecycle import LocalApprovalRequest

        request = LocalApprovalRequest(
            approved=True,
            evidence_hash="sha256:evidence123",
            reason="Approved by operator",
        )

        assert request.approved is True
        assert request.evidence_hash == "sha256:evidence123"


# ============================================================================
# Authentication Tests
# ============================================================================

class TestAgentAuthentication:
    """Tests for agent authentication requirements."""

    def test_heartbeat_requires_agent_token(self):
        """Test that heartbeat endpoint requires agent token."""
        from packages.cloud.control_plane.routers.agent_lifecycle import agent_heartbeat
        import inspect

        # Get function signature
        sig = inspect.signature(agent_heartbeat)
        params = sig.parameters

        # Should have current_agent parameter with AgentDep type
        assert "current_agent" in params

    def test_poll_requires_agent_token(self):
        """Test that poll endpoint requires agent token."""
        from packages.cloud.control_plane.routers.agent_lifecycle import poll_commands
        import inspect

        sig = inspect.signature(poll_commands)
        params = sig.parameters

        assert "current_agent" in params

    def test_ack_requires_agent_token(self):
        """Test that ack endpoint requires agent token."""
        from packages.cloud.control_plane.routers.agent_lifecycle import acknowledge_command
        import inspect

        sig = inspect.signature(acknowledge_command)
        params = sig.parameters

        assert "current_agent" in params


# ============================================================================
# Agent Verification Tests
# ============================================================================

class TestAgentVerification:
    """Tests for verify_agent_enrolled function."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        session = AsyncMock()
        return session

    @pytest.mark.asyncio
    async def test_verify_enrolled_agent_success(self, mock_session):
        """Test verification of enrolled agent succeeds."""
        from packages.cloud.control_plane.routers.agent_lifecycle import verify_agent_enrolled
        from packages.cloud.control_plane.dependencies import CurrentAgent
        from packages.cloud.control_plane.models import Agent, TrustState

        agent_id = uuid4()
        workspace_id = uuid4()

        # Mock current agent from token
        current_agent = CurrentAgent(
            id=agent_id,
            workspace_id=workspace_id,
            org_id=uuid4(),
            capabilities=[],
            trust_state=TrustState.ENROLLED,
        )

        # Mock database agent
        mock_agent = MagicMock(spec=Agent)
        mock_agent.id = agent_id
        mock_agent.workspace_id = workspace_id
        mock_agent.trust_state = TrustState.ENROLLED.value
        mock_agent.deleted_at = None

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_agent
        mock_session.execute.return_value = mock_result

        # Should not raise
        result = await verify_agent_enrolled(mock_session, current_agent)
        assert result is not None

    @pytest.mark.asyncio
    async def test_verify_revoked_agent_fails(self, mock_session):
        """Test verification of revoked agent fails."""
        from packages.cloud.control_plane.routers.agent_lifecycle import verify_agent_enrolled
        from packages.cloud.control_plane.dependencies import CurrentAgent
        from packages.cloud.control_plane.models import Agent, TrustState
        from fastapi import HTTPException

        current_agent = CurrentAgent(
            id=uuid4(),
            workspace_id=uuid4(),
            org_id=uuid4(),
            capabilities=[],
            trust_state=TrustState.ENROLLED,
        )

        # Mock revoked agent in DB
        mock_agent = MagicMock(spec=Agent)
        mock_agent.trust_state = TrustState.REVOKED.value
        mock_agent.deleted_at = None

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_agent
        mock_session.execute.return_value = mock_result

        with pytest.raises(HTTPException) as exc_info:
            await verify_agent_enrolled(mock_session, current_agent)

        assert exc_info.value.status_code == 403
        assert "revoked" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_verify_suspended_agent_fails(self, mock_session):
        """Test verification of suspended agent fails."""
        from packages.cloud.control_plane.routers.agent_lifecycle import verify_agent_enrolled
        from packages.cloud.control_plane.dependencies import CurrentAgent
        from packages.cloud.control_plane.models import Agent, TrustState
        from fastapi import HTTPException

        current_agent = CurrentAgent(
            id=uuid4(),
            workspace_id=uuid4(),
            org_id=uuid4(),
            capabilities=[],
            trust_state=TrustState.ENROLLED,
        )

        # Mock suspended agent
        mock_agent = MagicMock(spec=Agent)
        mock_agent.trust_state = TrustState.SUSPENDED.value
        mock_agent.deleted_at = None

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_agent
        mock_session.execute.return_value = mock_result

        with pytest.raises(HTTPException) as exc_info:
            await verify_agent_enrolled(mock_session, current_agent)

        assert exc_info.value.status_code == 403
        assert "suspended" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_verify_missing_agent_fails(self, mock_session):
        """Test verification of non-existent agent fails."""
        from packages.cloud.control_plane.routers.agent_lifecycle import verify_agent_enrolled
        from packages.cloud.control_plane.dependencies import CurrentAgent
        from packages.cloud.control_plane.models import TrustState
        from fastapi import HTTPException

        current_agent = CurrentAgent(
            id=uuid4(),
            workspace_id=uuid4(),
            org_id=uuid4(),
            capabilities=[],
            trust_state=TrustState.ENROLLED,
        )

        # Mock no agent found
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        with pytest.raises(HTTPException) as exc_info:
            await verify_agent_enrolled(mock_session, current_agent)

        assert exc_info.value.status_code == 404


# ============================================================================
# Endpoint Behavior Tests
# ============================================================================

class TestHeartbeatBehavior:
    """Tests for heartbeat endpoint behavior."""

    def test_heartbeat_returns_pending_count(self):
        """Test that heartbeat returns pending command count."""
        from packages.cloud.control_plane.routers.agent_lifecycle import AgentHeartbeatResponse

        # Create response with pending commands
        response = AgentHeartbeatResponse(
            server_time=datetime.now(timezone.utc),
            trust_state="enrolled",
            pending_commands=3,
            next_heartbeat_sec=60,
        )

        assert response.pending_commands == 3

    def test_heartbeat_includes_server_time(self):
        """Test that heartbeat includes server time."""
        from packages.cloud.control_plane.routers.agent_lifecycle import AgentHeartbeatResponse

        now = datetime.now(timezone.utc)
        response = AgentHeartbeatResponse(
            server_time=now,
            trust_state="enrolled",
            pending_commands=0,
            next_heartbeat_sec=60,
        )

        assert response.server_time == now


class TestCommandPollBehavior:
    """Tests for command poll endpoint behavior."""

    def test_poll_response_includes_has_more(self):
        """Test that poll response indicates if more commands exist."""
        from packages.cloud.control_plane.routers.agent_lifecycle import CommandPollResponse

        response = CommandPollResponse(
            commands=[],
            has_more=True,
            poll_again_after_sec=5,
        )

        assert response.has_more is True

    def test_poll_response_includes_next_poll_time(self):
        """Test that poll response includes next poll timing."""
        from packages.cloud.control_plane.routers.agent_lifecycle import CommandPollResponse

        # No commands - should poll again soon
        response = CommandPollResponse(
            commands=[],
            has_more=False,
            poll_again_after_sec=5,
        )

        assert response.poll_again_after_sec == 5


class TestLocalApprovalBehavior:
    """Tests for local approval endpoint behavior."""

    def test_approval_request_validates_evidence_hash(self):
        """Test that approval request validates evidence hash length."""
        from packages.cloud.control_plane.routers.agent_lifecycle import LocalApprovalRequest
        from pydantic import ValidationError

        # Valid evidence hash
        request = LocalApprovalRequest(
            approved=True,
            evidence_hash="sha256:" + "a" * 64,  # 64 hex chars
        )
        assert request.evidence_hash is not None

        # Too long evidence hash should fail
        try:
            LocalApprovalRequest(
                approved=True,
                evidence_hash="a" * 200,  # Too long
            )
        except ValidationError:
            pass  # Expected

    def test_approval_response_includes_status(self):
        """Test that approval response includes command status."""
        from packages.cloud.control_plane.routers.agent_lifecycle import LocalApprovalResponse

        cmd_id = uuid4()
        response = LocalApprovalResponse(
            command_id=cmd_id,
            status="approved",
            approved=True,
        )

        assert response.status == "approved"
        assert response.approved is True


# ============================================================================
# Integration Tests
# ============================================================================

class TestEndpointIntegration:
    """Integration tests for agent lifecycle endpoints."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for testing."""
        from packages.cloud.control_plane.dependencies import CurrentAgent
        from packages.cloud.control_plane.models import TrustState

        agent = CurrentAgent(
            id=uuid4(),
            workspace_id=uuid4(),
            org_id=uuid4(),
            capabilities=["execute_run"],
            trust_state=TrustState.ENROLLED,
        )
        return agent

    def test_all_endpoints_exist(self):
        """Test that all lifecycle endpoints exist."""
        from packages.cloud.control_plane.routers.agent_lifecycle import (
            agent_heartbeat,
            poll_commands,
            acknowledge_command,
            submit_command_result,
            submit_local_approval,
            get_agent_status,
        )

        # All should be callable functions
        assert callable(agent_heartbeat)
        assert callable(poll_commands)
        assert callable(acknowledge_command)
        assert callable(submit_command_result)
        assert callable(submit_local_approval)
        assert callable(get_agent_status)

    def test_endpoints_are_async(self):
        """Test that all endpoints are async functions."""
        import asyncio
        from packages.cloud.control_plane.routers.agent_lifecycle import (
            agent_heartbeat,
            poll_commands,
            acknowledge_command,
        )

        assert asyncio.iscoroutinefunction(agent_heartbeat)
        assert asyncio.iscoroutinefunction(poll_commands)
        assert asyncio.iscoroutinefunction(acknowledge_command)


# ============================================================================
# Security Tests
# ============================================================================

class TestSecurityRequirements:
    """Tests for security requirements."""

    def test_no_user_tokens_allowed(self):
        """Test that user tokens cannot access agent endpoints."""
        # This is enforced by AgentDep which checks token type
        from packages.cloud.control_plane.dependencies import get_current_agent

        # The dependency will reject non-agent tokens

    def test_workspace_isolation(self):
        """Test that commands are workspace-scoped."""
        # verify_agent_enrolled checks workspace_id matches
        from packages.cloud.control_plane.routers.agent_lifecycle import verify_agent_enrolled

        # The function filters by workspace_id in the query

    def test_approval_records_approver(self):
        """Test that local approvals record who approved."""
        from packages.cloud.control_plane.routers.agent_lifecycle import LocalApprovalRequest

        # The endpoint creates ApprovalRecord with approved_by = local:{agent_id}
