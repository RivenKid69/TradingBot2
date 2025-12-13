# -*- coding: utf-8 -*-
"""
Tests for CCEA Cloud Control Plane.

Tests:
- Enrollment service
- Command service
- Blob storage
- Heartbeat service
"""

import pytest
from datetime import datetime, timedelta

from ccea.control_plane.enrollment import (
    EnrollmentService,
    InMemoryTokenStore,
    InMemoryAgentStore,
    TokenExpiredError,
    TokenInvalidError,
    TokenRevokedError,
)
from ccea.control_plane.commands import (
    CommandService,
    CommandDispatcher,
    CommandQueue,
    InMemoryCommandStore,
    ALLOWED_COMMAND_TYPES,
    APPROVAL_REQUIRED,
    SAFETY_COMMANDS,
)
from ccea.control_plane.blobs import (
    BlobStore,
    InMemoryBlobStore,
    FileBlobStore,
    ConfigBlobStore,
    BlobNotFoundError,
)
from ccea.control_plane.heartbeat import (
    HeartbeatService,
    AgentStatus,
    InMemoryHeartbeatStore,
)

from ccea.models.enrollment import (
    EnrollmentRequest,
    AgentCapability,
)
from ccea.models.protocol import (
    HeartbeatMessage,
    AgentState,
    DeploymentState,
    RunState,
    CommandStatus,
)
from ccea.crypto.keys import generate_keypair


class TestEnrollmentService:
    """Tests for EnrollmentService."""

    def test_create_token(self):
        """Test enrollment token creation."""
        service = EnrollmentService()
        token = service.create_token(
            workspace_id="ws_test",
            created_by="admin",
            ttl_hours=24,
        )

        assert token.token_id.startswith("enroll_")
        assert token.workspace_id == "ws_test"
        assert token.is_valid() is True

    def test_validate_token(self):
        """Test token validation."""
        service = EnrollmentService()
        token = service.create_token(
            workspace_id="ws_test",
            created_by="admin",
        )

        validated = service.validate_token(token.token_id)
        assert validated.token_id == token.token_id

    def test_validate_invalid_token(self):
        """Test validation of invalid token."""
        service = EnrollmentService()

        with pytest.raises(TokenInvalidError):
            service.validate_token("invalid_token")

    def test_validate_expired_token(self):
        """Test validation of expired token."""
        service = EnrollmentService()
        token = service.create_token(
            workspace_id="ws_test",
            created_by="admin",
            ttl_hours=0,  # Expires immediately
        )

        # Manually expire it
        stored = service.token_store.get(token.token_id)
        stored.expires_at = datetime.utcnow() - timedelta(hours=1)

        with pytest.raises(TokenExpiredError):
            service.validate_token(token.token_id)

    def test_enroll_agent(self):
        """Test agent enrollment."""
        service = EnrollmentService()
        token = service.create_token(
            workspace_id="ws_test",
            created_by="admin",
        )

        keypair = generate_keypair()
        request = EnrollmentRequest(
            token=token.token_id,
            public_key=keypair.get_public_key_pem(),
            agent_version="1.0.0",
        )

        response = service.enroll_agent(request)

        assert response.success is True
        assert response.agent_id.startswith("agent_")
        assert response.workspace_id == "ws_test"

    def test_enroll_with_invalid_token(self):
        """Test enrollment with invalid token."""
        service = EnrollmentService()
        keypair = generate_keypair()

        # Use a properly formatted token that doesn't exist in the service
        request = EnrollmentRequest(
            token="enroll_nonexistenttoken1234567890123456",
            public_key=keypair.get_public_key_pem(),
            agent_version="1.0.0",
        )

        response = service.enroll_agent(request)

        assert response.success is False
        assert response.error_code == "TOKEN_INVALID"

    def test_revoke_agent(self):
        """Test agent revocation."""
        service = EnrollmentService()
        token = service.create_token(
            workspace_id="ws_test",
            created_by="admin",
        )

        keypair = generate_keypair()
        request = EnrollmentRequest(
            token=token.token_id,
            public_key=keypair.get_public_key_pem(),
            agent_version="1.0.0",
        )

        response = service.enroll_agent(request)
        assert response.success is True

        result = service.revoke_agent(
            response.agent_id,
            "admin",
            "Security concern",
        )

        assert result is True

        agent = service.get_agent(response.agent_id)
        assert agent.is_active() is False

    def test_revoke_token(self):
        """Test token revocation."""
        service = EnrollmentService()
        token = service.create_token(
            workspace_id="ws_test",
            created_by="admin",
        )

        result = service.revoke_token(token.token_id, "admin")
        assert result is True

        with pytest.raises(TokenRevokedError):
            service.validate_token(token.token_id)


class TestCommandService:
    """Tests for CommandService."""

    def test_command_allowlist(self):
        """Test that only allowlisted commands exist."""
        expected = {
            "REQUEST_START_RUN",
            "REQUEST_STOP_RUN",
            "REQUEST_PAUSE_RUN",
            "REQUEST_UPGRADE_ARTIFACT",
            "REQUEST_UPDATE_CONFIG",
            "REQUEST_ROTATE_AGENT_SESSION",
            "REQUEST_EXPORT_LOGS",
        }
        assert ALLOWED_COMMAND_TYPES == expected

    def test_approval_required_commands(self):
        """Test approval required commands."""
        assert "REQUEST_START_RUN" in APPROVAL_REQUIRED
        assert "REQUEST_UPGRADE_ARTIFACT" in APPROVAL_REQUIRED
        assert "REQUEST_UPDATE_CONFIG" in APPROVAL_REQUIRED

    def test_safety_commands(self):
        """Test safety commands don't require approval."""
        assert "REQUEST_STOP_RUN" in SAFETY_COMMANDS
        assert "REQUEST_PAUSE_RUN" in SAFETY_COMMANDS

    def test_create_start_command(self):
        """Test creating a start run command."""
        service = CommandService()
        dispatcher = CommandDispatcher(service)

        record = dispatcher.request_start_run(
            agent_id="agent_test1234567890123456",
            deployment_id="deploy_123",
            artifact_digest="sha256:0000000000000000000000000000000000000000000000000000000000000000",
        )

        assert record.command_type == "REQUEST_START_RUN"
        assert record.requires_approval is True

    def test_create_stop_command_no_approval(self):
        """Test stop command doesn't require approval."""
        service = CommandService()
        dispatcher = CommandDispatcher(service)

        record = dispatcher.request_stop_run(
            agent_id="agent_test1234567890123456",
            deployment_id="deploy_123",
        )

        assert record.command_type == "REQUEST_STOP_RUN"
        assert record.requires_approval is False

    def test_command_idempotency(self):
        """Test command idempotency."""
        service = CommandService()
        dispatcher = CommandDispatcher(service)

        # Create same command twice with same idempotency key
        from ccea.models.protocol import RequestStartRunCommand
        from ccea.crypto.tokens import generate_idempotency_key

        idem_key = generate_idempotency_key()
        cmd = RequestStartRunCommand(
            deployment_id="deploy_123",
            artifact_digest="sha256:0000000000000000000000000000000000000000000000000000000000000000",
            idempotency_key=idem_key,
        )

        record1 = service.create_command("agent_test1234567890123456", cmd, idem_key)
        record2 = service.create_command("agent_test1234567890123456", cmd, idem_key)

        assert record1.command_id == record2.command_id

    def test_acknowledge_command(self):
        """Test command acknowledgment."""
        service = CommandService()
        dispatcher = CommandDispatcher(service)

        record = dispatcher.request_stop_run(
            agent_id="agent_test1234567890123456",
            deployment_id="deploy_123",
        )

        result = service.acknowledge(record.command_id)
        assert result is True

        updated = service.get(record.command_id)
        assert updated.status == CommandStatus.RECEIVED

    def test_set_approval(self):
        """Test setting approval."""
        service = CommandService()
        dispatcher = CommandDispatcher(service)

        record = dispatcher.request_start_run(
            agent_id="agent_test1234567890123456",
            deployment_id="deploy_123",
            artifact_digest="sha256:0000000000000000000000000000000000000000000000000000000000000000",
        )

        result = service.set_approval(
            record.command_id,
            approved=True,
            evidence_hash="sha256:evidence123456789012345678901234567890123456789012345678901234",
        )

        assert result is True
        updated = service.get(record.command_id)
        assert updated.status == CommandStatus.APPROVED

    def test_complete_command(self):
        """Test command completion."""
        service = CommandService()
        dispatcher = CommandDispatcher(service)

        record = dispatcher.request_stop_run(
            agent_id="agent_test1234567890123456",
            deployment_id="deploy_123",
        )

        result = service.complete(record.command_id, result={"stopped": True})
        assert result is True

        updated = service.get(record.command_id)
        assert updated.status == CommandStatus.COMPLETED


class TestCommandQueue:
    """Tests for CommandQueue."""

    def test_enqueue_and_poll(self):
        """Test command queue operations."""
        queue = CommandQueue(poll_timeout=1.0)

        from ccea.control_plane.commands import CommandRecord

        record = CommandRecord(
            command_id="cmd_123",
            idempotency_key="idem_key_123456789012",
            agent_id="agent_test1234567890123456",
            command_type="REQUEST_STOP_RUN",
            payload={},
        )

        queue.enqueue("agent_test1234567890123456", record)
        commands = queue.poll("agent_test1234567890123456", timeout=1.0)

        assert len(commands) == 1
        assert commands[0].command_id == "cmd_123"

    def test_poll_timeout_empty(self):
        """Test poll timeout returns empty."""
        queue = CommandQueue(poll_timeout=0.1)

        commands = queue.poll("agent_nonexistent12345678", timeout=0.1)
        assert len(commands) == 0


class TestBlobStore:
    """Tests for BlobStore."""

    def test_put_and_get_blob(self):
        """Test storing and retrieving blobs."""
        store = InMemoryBlobStore()

        content = b"Test blob content"
        metadata = store.put(content, workspace_id="ws_test")

        assert metadata.digest.startswith("sha256:")
        assert metadata.size_bytes == len(content)

        retrieved = store.get(metadata.digest)
        assert retrieved == content

    def test_blob_immutability(self):
        """Test that same content produces same digest."""
        store = InMemoryBlobStore()

        content = b"Immutable content"
        meta1 = store.put(content)
        meta2 = store.put(content)

        assert meta1.digest == meta2.digest

    def test_get_nonexistent_blob(self):
        """Test getting nonexistent blob."""
        store = InMemoryBlobStore()

        with pytest.raises(BlobNotFoundError):
            store.get("sha256:nonexistent1234567890123456789012345678901234567890123456789012")

    def test_blob_exists(self):
        """Test checking blob existence."""
        store = InMemoryBlobStore()

        content = b"Existence check"
        metadata = store.put(content)

        assert store.exists(metadata.digest) is True
        assert store.exists("sha256:nonexistent") is False

    def test_delete_blob(self):
        """Test deleting blob."""
        store = InMemoryBlobStore()

        content = b"Deletable content"
        metadata = store.put(content)

        result = store.delete(metadata.digest)
        assert result is True
        assert store.exists(metadata.digest) is False

    def test_config_blob_store(self):
        """Test ConfigBlobStore helper."""
        blob_store = InMemoryBlobStore()
        config_store = ConfigBlobStore(blob_store)

        config = {"setting": "value", "number": 42}
        digest = config_store.put_config(config, workspace_id="ws_test")

        assert digest.startswith("sha256:")

        retrieved = config_store.get_config(digest)
        assert retrieved == config


class TestHeartbeatService:
    """Tests for HeartbeatService."""

    def test_process_heartbeat(self):
        """Test processing heartbeat."""
        service = HeartbeatService()

        heartbeat = HeartbeatMessage(
            agent_id="agent_heartbeat123456789012",
            state=AgentState(
                deployment_state=DeploymentState.RUNNING,
                run_state=RunState.RUNNING,
            ),
        )

        status = service.process_heartbeat(heartbeat)

        assert status.status == AgentStatus.ONLINE
        assert status.agent_id == "agent_heartbeat123456789012"

    def test_check_agent_status(self):
        """Test checking agent status."""
        service = HeartbeatService(heartbeat_interval=30)

        heartbeat = HeartbeatMessage(
            agent_id="agent_status12345678901234",
            state=AgentState(),
        )

        service.process_heartbeat(heartbeat)
        status = service.check_agent_status("agent_status12345678901234")

        assert status.status == AgentStatus.ONLINE

    def test_unknown_agent_status(self):
        """Test status of unknown agent."""
        service = HeartbeatService()

        status = service.check_agent_status("agent_unknown1234567890123")
        assert status.status == AgentStatus.UNKNOWN

    def test_health_alerts(self):
        """Test health metric alerts."""
        service = HeartbeatService()

        from ccea.models.protocol import AgentHealth

        heartbeat = HeartbeatMessage(
            agent_id="agent_alerts123456789012345",
            state=AgentState(),
            health=AgentHealth(
                cpu_percent=95.0,  # High CPU
                memory_percent=92.0,  # High memory
                broker_connected=False,  # Disconnected
            ),
        )

        status = service.process_heartbeat(heartbeat)

        assert status.status == AgentStatus.DEGRADED
        assert len(status.alerts) > 0
        assert any("CPU" in a for a in status.alerts)
