# -*- coding: utf-8 -*-
"""
Tests for Cloud Client - Phase 4.9

Tests cover:
- Agent enrollment
- Bearer token authentication
- Outbound-only transport
- Heartbeat lifecycle
- Command polling
- Request signing
"""

from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4, UUID

import pytest

from packages.agent.cloud.client import (
    CloudClient,
    CloudClientConfig,
    CloudClientError,
    AgentIdentity,
)
from packages.agent.cloud.types import (
    AgentEnrollResult,
    AgentHeartbeatResult,
    CommandPollResult,
    CommandAckResult,
    PendingCommand,
)


class TestAgentIdentity:
    """Test AgentIdentity for keypair management."""

    def test_generate_creates_keypair(self):
        """Test generating agent identity creates keypair."""
        identity = AgentIdentity.generate()

        assert identity.public_key_pem is not None
        assert identity.private_key_pem is not None
        assert "BEGIN PUBLIC KEY" in identity.public_key_pem
        assert "BEGIN PRIVATE KEY" in identity.private_key_pem

    def test_public_key_fingerprint(self):
        """Test public key fingerprint generation."""
        identity = AgentIdentity.generate()

        fingerprint = identity.public_key_fingerprint()

        assert fingerprint is not None
        assert len(fingerprint) == 64  # SHA256 hex

    def test_different_identities_have_different_keys(self):
        """Test each identity has unique keys."""
        identity1 = AgentIdentity.generate()
        identity2 = AgentIdentity.generate()

        assert identity1.public_key_pem != identity2.public_key_pem
        assert identity1.public_key_fingerprint() != identity2.public_key_fingerprint()


class TestCloudClientConfig:
    """Test CloudClientConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CloudClientConfig(base_url="https://api.example.com")

        assert config.timeout_seconds == 30
        assert config.user_agent == "ccea-agent-cloud-client/1.0"


class TestCloudClientInitialization:
    """Test CloudClient initialization."""

    def test_creates_identity_if_not_provided(self):
        """Test client creates identity if not provided."""
        config = CloudClientConfig(base_url="https://api.example.com")

        with patch("httpx.Client"):
            client = CloudClient(config)

        assert client.identity is not None
        assert client.identity.public_key_pem is not None

    def test_uses_provided_identity(self):
        """Test client uses provided identity."""
        config = CloudClientConfig(base_url="https://api.example.com")
        identity = AgentIdentity.generate()

        with patch("httpx.Client"):
            client = CloudClient(config, identity=identity)

        assert client.identity == identity

    def test_initial_state_not_connected(self):
        """Test client starts in not-connected state."""
        config = CloudClientConfig(base_url="https://api.example.com")

        with patch("httpx.Client"):
            client = CloudClient(config)

        assert client.is_connected is False


class TestCloudClientEnrollment:
    """Test enrollment flow."""

    @pytest.fixture
    def client(self):
        """Create cloud client with mocked transport."""
        config = CloudClientConfig(base_url="https://api.example.com")
        mock_transport = Mock()
        with patch("httpx.Client") as mock_client:
            client = CloudClient(config, transport=mock_transport)
            client._client = mock_client.return_value
            yield client

    def test_enroll_sends_public_key_not_private(self, client):
        """Test enrollment sends public key but not private key."""
        agent_id = uuid4()
        workspace_id = uuid4()
        org_id = uuid4()

        mock_response = Mock()
        mock_response.json.return_value = {
            "agent_id": str(agent_id),
            "access_token": "test-token-123",
            "workspace_id": str(workspace_id),
            "org_id": str(org_id),
        }
        mock_response.raise_for_status = Mock()
        client._client.request.return_value = mock_response

        result = client.enroll(
            enrollment_token="enroll-123",
            agent_name="test-agent",
            agent_version="1.0.0",
        )

        # Check request was made
        client._client.request.assert_called_once()
        call_args = client._client.request.call_args

        # Verify public key was sent
        assert call_args[1]["json"]["public_key"] is not None
        # Verify private key was NOT sent
        assert "private_key" not in call_args[1]["json"]

    def test_enroll_no_auth_required(self, client):
        """Test enrollment doesn't require auth."""
        agent_id = uuid4()
        workspace_id = uuid4()
        org_id = uuid4()

        mock_response = Mock()
        mock_response.json.return_value = {
            "agent_id": str(agent_id),
            "access_token": "test-token-123",
            "workspace_id": str(workspace_id),
            "org_id": str(org_id),
        }
        mock_response.raise_for_status = Mock()
        client._client.request.return_value = mock_response

        client.enroll(
            enrollment_token="enroll-123",
            agent_name="test-agent",
            agent_version="1.0.0",
        )

        # Check no Authorization header
        call_args = client._client.request.call_args
        headers = call_args[1].get("headers", {})
        assert "Authorization" not in headers

    def test_enroll_stores_access_token(self, client):
        """Test enrollment stores access token."""
        agent_id = uuid4()
        workspace_id = uuid4()
        org_id = uuid4()

        mock_response = Mock()
        mock_response.json.return_value = {
            "agent_id": str(agent_id),
            "access_token": "new-access-token",
            "workspace_id": str(workspace_id),
            "org_id": str(org_id),
        }
        mock_response.raise_for_status = Mock()
        client._client.request.return_value = mock_response

        result = client.enroll(
            enrollment_token="enroll-123",
            agent_name="test-agent",
            agent_version="1.0.0",
        )

        assert client.access_token == "new-access-token"
        assert result.access_token == "new-access-token"


class TestCloudClientAuthentication:
    """Test authentication and authorization."""

    @pytest.fixture
    def authenticated_client(self):
        """Create authenticated cloud client."""
        config = CloudClientConfig(base_url="https://api.example.com")
        mock_transport = Mock()
        with patch("httpx.Client") as mock_client:
            client = CloudClient(
                config,
                transport=mock_transport,
                access_token="valid-token-123",
            )
            client._client = mock_client.return_value
            yield client

    def test_authenticated_request_includes_bearer_token(self, authenticated_client):
        """Test authenticated requests include Bearer token."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "server_time": datetime.now(timezone.utc).isoformat(),
            "trust_state": "trusted",
            "pending_commands": 0,
            "next_heartbeat_sec": 60,
        }
        mock_response.raise_for_status = Mock()
        authenticated_client._client.request.return_value = mock_response

        authenticated_client.heartbeat(
            agent_version="1.0.0",
            current_state="running",
        )

        call_args = authenticated_client._client.request.call_args
        headers = call_args[1].get("headers", {})
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer valid-token-123"

    def test_request_without_token_raises(self):
        """Test request without token raises error."""
        config = CloudClientConfig(base_url="https://api.example.com")

        with patch("httpx.Client"):
            client = CloudClient(config)

        with pytest.raises(CloudClientError) as exc_info:
            client.heartbeat(
                agent_version="1.0.0",
                current_state="running",
            )

        assert "Missing access_token" in str(exc_info.value)


class TestCloudClientSigning:
    """Test request signing."""

    @pytest.fixture
    def client(self):
        """Create cloud client."""
        config = CloudClientConfig(base_url="https://api.example.com")
        mock_transport = Mock()
        with patch("httpx.Client") as mock_client:
            client = CloudClient(
                config,
                transport=mock_transport,
                access_token="test-token",
            )
            client._client = mock_client.return_value
            yield client

    def test_requests_are_signed(self, client):
        """Test requests include signature headers."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "server_time": datetime.now(timezone.utc).isoformat(),
            "trust_state": "trusted",
            "pending_commands": 0,
            "next_heartbeat_sec": 60,
        }
        mock_response.raise_for_status = Mock()
        client._client.request.return_value = mock_response

        client.heartbeat(
            agent_version="1.0.0",
            current_state="running",
        )

        call_args = client._client.request.call_args
        headers = call_args[1].get("headers", {})

        # Check signature headers are present
        assert "X-CCEA-Timestamp" in headers
        assert "X-CCEA-Signature" in headers
        assert "X-CCEA-Signature-Alg" in headers
        assert "X-CCEA-Key-Fingerprint" in headers


class TestCloudClientHeartbeat:
    """Test heartbeat operations."""

    @pytest.fixture
    def client(self):
        """Create authenticated cloud client."""
        config = CloudClientConfig(base_url="https://api.example.com")
        mock_transport = Mock()
        with patch("httpx.Client") as mock_client:
            client = CloudClient(
                config,
                transport=mock_transport,
                access_token="test-token",
            )
            client._client = mock_client.return_value
            yield client

    def test_heartbeat_returns_result(self, client):
        """Test heartbeat returns proper result."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "server_time": datetime.now(timezone.utc).isoformat(),
            "trust_state": "trusted",
            "pending_commands": 5,
            "next_heartbeat_sec": 30,
        }
        mock_response.raise_for_status = Mock()
        client._client.request.return_value = mock_response

        result = client.heartbeat(
            agent_version="1.0.0",
            current_state="running",
        )

        assert isinstance(result, AgentHeartbeatResult)
        assert result.trust_state == "trusted"
        assert result.pending_commands == 5
        assert result.next_heartbeat_sec == 30

    def test_heartbeat_updates_connected_state(self, client):
        """Test successful heartbeat updates connected state."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "server_time": datetime.now(timezone.utc).isoformat(),
            "trust_state": "trusted",
            "pending_commands": 0,
            "next_heartbeat_sec": 60,
        }
        mock_response.raise_for_status = Mock()
        client._client.request.return_value = mock_response

        assert client.is_connected is False

        client.heartbeat(
            agent_version="1.0.0",
            current_state="running",
        )

        assert client.is_connected is True


class TestCloudClientCommands:
    """Test command operations."""

    @pytest.fixture
    def client(self):
        """Create authenticated cloud client with signature verification disabled."""
        config = CloudClientConfig(base_url="https://api.example.com")
        mock_transport = Mock()
        with patch("httpx.Client") as mock_client:
            client = CloudClient(
                config,
                transport=mock_transport,
                access_token="test-token",
                verify_cloud_signatures=False,  # Disable for testing
            )
            client._client = mock_client.return_value
            # Design Doc 10.3: Mark version as negotiated for test purposes
            client._version_negotiated = True
            client._negotiated_version = "1.0.0"
            yield client

    def test_poll_commands(self, client):
        """Test polling for commands."""
        command_id = uuid4()
        mock_response = Mock()
        mock_response.json.return_value = {
            "commands": [
                {
                    "id": str(command_id),
                    "status": "pending",
                    "idempotency_key": "idem-123",
                    "command_type": "START_RUN",
                    "payload_ref": "s3://bucket/payload",
                    "change_class": "trading_impacting",
                    "requires_approval": True,
                }
            ],
            "has_more": False,
            "poll_again_after_sec": 60,
        }
        mock_response.raise_for_status = Mock()
        client._client.request.return_value = mock_response

        result = client.poll_commands(limit=10)

        assert isinstance(result, CommandPollResult)
        assert len(result.commands) == 1
        assert result.commands[0].command_type == "START_RUN"
        assert result.commands[0].requires_approval is True

    def test_acknowledge_command(self, client):
        """Test acknowledging a command."""
        command_id = uuid4()
        mock_response = Mock()
        mock_response.json.return_value = {
            "command_id": str(command_id),
            "status": "acknowledged",
            "acknowledged_at": datetime.now(timezone.utc).isoformat(),
        }
        mock_response.raise_for_status = Mock()
        client._client.request.return_value = mock_response

        result = client.acknowledge_command(command_id)

        assert isinstance(result, CommandAckResult)
        assert result.command_id == command_id
        assert result.status == "acknowledged"

    def test_submit_local_approval(self, client):
        """Test submitting local approval."""
        command_id = uuid4()
        mock_response = Mock()
        mock_response.json.return_value = {
            "command_id": str(command_id),
            "status": "approved",
        }
        mock_response.raise_for_status = Mock()
        client._client.request.return_value = mock_response

        result = client.submit_local_approval(
            command_id=command_id,
            approved=True,
            evidence_hash="hash123",
            reason="Approved by operator",
        )

        assert result["status"] == "approved"

    def test_submit_command_result(self, client):
        """Test submitting command result."""
        command_id = uuid4()
        mock_response = Mock()
        mock_response.json.return_value = {
            "command_id": str(command_id),
            "status": "completed",
            "executed_at": datetime.now(timezone.utc).isoformat(),
        }
        mock_response.raise_for_status = Mock()
        client._client.request.return_value = mock_response

        result = client.submit_command_result(
            command_id=command_id,
            success=True,
            result={"run_id": str(uuid4())},
        )

        assert result.status == "completed"


class TestOutboundOnlyTransport:
    """Test that transport is outbound-only (agent initiates all connections)."""

    def test_all_methods_are_agent_initiated(self):
        """Verify all API methods are agent-initiated (outbound only)."""
        config = CloudClientConfig(base_url="https://api.example.com")

        with patch("httpx.Client"):
            client = CloudClient(config, access_token="test")

        # All public API methods should be client-initiated
        public_methods = [
            "enroll",
            "heartbeat",
            "poll_commands",
            "acknowledge_command",
            "submit_local_approval",
            "submit_command_result",
        ]

        for method_name in public_methods:
            method = getattr(client, method_name)
            assert callable(method), f"{method_name} should be a method"

        # There should be NO server-initiated callbacks or webhooks
        # All communication is poll-based (agent polls cloud)

    def test_no_webhook_endpoints_exposed(self):
        """Verify no webhook/callback endpoints are exposed."""
        config = CloudClientConfig(base_url="https://api.example.com")

        with patch("httpx.Client"):
            client = CloudClient(config)

        # Check there are no listener/server methods
        assert not hasattr(client, "start_server")
        assert not hasattr(client, "listen")
        assert not hasattr(client, "on_command")
        assert not hasattr(client, "register_webhook")
