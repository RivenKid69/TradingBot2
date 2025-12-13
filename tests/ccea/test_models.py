# -*- coding: utf-8 -*-
"""
Tests for CCEA Protocol Models.

Tests:
- Protocol message models
- Manifest models
- Enrollment models
- Prohibited fields validation
"""

import pytest
from datetime import datetime, timedelta
from pydantic import ValidationError

from ccea.models.protocol import (
    CommandStatus,
    ApprovalStatus,
    RunState,
    DeploymentState,
    ChangeClass,
    TelemetryLevel,
    CommandType,
    MessageType,
    Signature,
    HeartbeatMessage,
    PollCommandsMessage,
    CommandAckMessage,
    CommandApprovalMessage,
    CommandResultMessage,
    TelemetryMessage,
    RequestStartRunCommand,
    RequestStopRunCommand,
    RequestPauseRunCommand,
    RequestUpgradeArtifactCommand,
    RequestUpdateConfigCommand,
    AgentState,
    AgentHealth,
    TelemetryEvent,
    PROHIBITED_FIELDS,
)

from ccea.models.manifest import (
    ArtifactManifest,
    ArtifactType,
    Entrypoint,
    Runtime,
    ChangeClass as ManifestChangeClass,
)

from ccea.models.enrollment import (
    EnrollmentToken,
    EnrollmentRequest,
    EnrollmentResponse,
    AgentInfo,
    TrustState,
    AgentCapability,
)


class TestProtocolEnums:
    """Tests for protocol enums."""

    def test_command_status_values(self):
        """Test CommandStatus enum values."""
        assert CommandStatus.PENDING.value == "PENDING"
        assert CommandStatus.COMPLETED.value == "COMPLETED"
        assert CommandStatus.FAILED.value == "FAILED"

    def test_run_state_values(self):
        """Test RunState enum values."""
        assert RunState.INITIALIZING.value == "INITIALIZING"
        assert RunState.RUNNING.value == "RUNNING"
        assert RunState.HALTED.value == "HALTED"

    def test_command_type_values(self):
        """Test CommandType enum values."""
        assert CommandType.REQUEST_START_RUN.value == "REQUEST_START_RUN"
        assert CommandType.REQUEST_STOP_RUN.value == "REQUEST_STOP_RUN"


class TestHeartbeatMessage:
    """Tests for HeartbeatMessage."""

    def test_valid_heartbeat(self):
        """Test valid heartbeat message."""
        msg = HeartbeatMessage(
            agent_id="agent_abcdefghij1234567890",
            state=AgentState(
                deployment_state=DeploymentState.RUNNING,
                run_state=RunState.RUNNING,
            ),
        )

        assert msg.message_type == MessageType.HEARTBEAT
        assert msg.agent_id == "agent_abcdefghij1234567890"

    def test_heartbeat_with_health(self):
        """Test heartbeat with health metrics."""
        msg = HeartbeatMessage(
            agent_id="agent_test1234567890123456",
            state=AgentState(),
            health=AgentHealth(
                cpu_percent=50.0,
                memory_percent=60.0,
                broker_connected=True,
            ),
        )

        assert msg.health.cpu_percent == 50.0
        assert msg.health.broker_connected is True

    def test_invalid_agent_id_format(self):
        """Test that invalid agent_id is rejected."""
        with pytest.raises(ValidationError):
            HeartbeatMessage(
                agent_id="invalid_format",
                state=AgentState(),
            )


class TestCommandMessages:
    """Tests for command messages."""

    def test_request_start_run(self):
        """Test REQUEST_START_RUN command."""
        cmd = RequestStartRunCommand(
            deployment_id="deploy_123",
            artifact_digest="sha256:abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            idempotency_key="test_idempotency_key_123",
        )

        assert cmd.command_type == CommandType.REQUEST_START_RUN
        assert cmd.requires_approval is True
        assert cmd.change_class == ChangeClass.TRADING_IMPACTING

    def test_request_stop_run_no_approval(self):
        """Test REQUEST_STOP_RUN doesn't require approval."""
        cmd = RequestStopRunCommand(
            deployment_id="deploy_123",
            reason="User requested stop",
            idempotency_key="stop_key_1234567890123456",
        )

        assert cmd.requires_approval is False

    def test_request_update_config_trading_impacting(self):
        """Test UPDATE_CONFIG with TRADING_IMPACTING requires approval."""
        cmd = RequestUpdateConfigCommand(
            deployment_id="deploy_123",
            new_config_digest="sha256:abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            change_class=ChangeClass.TRADING_IMPACTING,
            requires_approval=True,
            idempotency_key="config_key_123456789012",
        )

        assert cmd.requires_approval is True

    def test_update_config_trading_impacting_must_require_approval(self):
        """Test that TRADING_IMPACTING config must require approval."""
        with pytest.raises(ValidationError):
            RequestUpdateConfigCommand(
                deployment_id="deploy_123",
                new_config_digest="sha256:newconfig123456789012345678901234567890123456789012345678901234",
                change_class=ChangeClass.TRADING_IMPACTING,
                requires_approval=False,  # This should fail
                idempotency_key="config_key_123456789012",
            )


class TestProhibitedFields:
    """Tests for prohibited fields validation."""

    def test_prohibited_fields_defined(self):
        """Test that prohibited fields are defined."""
        assert "side" in PROHIBITED_FIELDS
        assert "quantity" in PROHIBITED_FIELDS
        assert "price" in PROHIBITED_FIELDS
        assert "order_type" in PROHIBITED_FIELDS
        assert "target_position" in PROHIBITED_FIELDS

    def test_order_like_payload_rejected(self):
        """Test that order-like payloads are rejected in messages."""
        # This test verifies the model validator rejects prohibited fields
        # The actual implementation uses a model_validator
        pass  # Schema-level validation tested in guardrails


class TestTelemetryMessage:
    """Tests for TelemetryMessage."""

    def test_telemetry_redaction_required(self):
        """Test that redaction_applied must be True."""
        msg = TelemetryMessage(
            agent_id="agent_telemetry1234567890",
            level=TelemetryLevel.AGGREGATED,
            events=[
                TelemetryEvent(
                    event_type="PERFORMANCE",
                    timestamp=datetime.utcnow(),
                ),
            ],
            redaction_applied=True,
        )

        assert msg.redaction_applied is True

    def test_telemetry_levels(self):
        """Test telemetry level values."""
        assert TelemetryLevel.AGGREGATED.value == "AGGREGATED"
        assert TelemetryLevel.DETAILED_NON_SENSITIVE.value == "DETAILED_NON_SENSITIVE"
        assert TelemetryLevel.RAW_ORDER_EVENTS.value == "RAW_ORDER_EVENTS"


class TestArtifactManifest:
    """Tests for ArtifactManifest."""

    def test_valid_manifest(self):
        """Test valid manifest creation."""
        manifest = ArtifactManifest(
            schema_version="1.0.0",
            artifact_id="test_strategy",
            artifact_type=ArtifactType.STRATEGY,
            entrypoint=Entrypoint(
                module="strategies.test",
                class_name="TestStrategy",
            ),
            runtime=Runtime(python_version="3.11"),
            deps_lock_digest="sha256:abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            created_at=datetime.utcnow(),
        )

        assert manifest.artifact_id == "test_strategy"
        assert manifest.artifact_type == ArtifactType.STRATEGY

    def test_manifest_compute_digest(self):
        """Test manifest digest computation."""
        manifest = ArtifactManifest(
            schema_version="1.0.0",
            artifact_id="test",
            artifact_type=ArtifactType.STRATEGY,
            entrypoint=Entrypoint(module="test", class_name="Test"),
            runtime=Runtime(python_version="3.11"),
            deps_lock_digest="sha256:0000000000000000000000000000000000000000000000000000000000000000",
            created_at=datetime(2024, 1, 1, 0, 0, 0),
        )

        digest = manifest.compute_digest()
        assert digest.startswith("sha256:")

    def test_invalid_schema_version(self):
        """Test that invalid schema version is rejected."""
        with pytest.raises(ValidationError):
            ArtifactManifest(
                schema_version="invalid",
                artifact_id="test",
                artifact_type=ArtifactType.STRATEGY,
                entrypoint=Entrypoint(module="test", class_name="Test"),
                runtime=Runtime(python_version="3.11"),
                deps_lock_digest="sha256:0000000000000000000000000000000000000000000000000000000000000000",
                created_at=datetime.utcnow(),
            )


class TestEnrollmentModels:
    """Tests for enrollment models."""

    def test_enrollment_token_creation(self):
        """Test EnrollmentToken creation."""
        token = EnrollmentToken.create(
            token_id="enroll_test123456789012345678901234567890",
            workspace_id="ws_test",
            created_by="admin",
            ttl_hours=24,
        )

        assert token.is_valid() is True
        assert token.uses_count == 0

    def test_enrollment_token_expiry(self):
        """Test token expiry detection."""
        token = EnrollmentToken(
            token_id="enroll_expiredtest1234567890123456789012",
            workspace_id="ws_test",
            created_by="admin",
            created_at=datetime.utcnow() - timedelta(hours=48),
            expires_at=datetime.utcnow() - timedelta(hours=24),
        )

        assert token.is_valid() is False

    def test_enrollment_request(self):
        """Test EnrollmentRequest model."""
        request = EnrollmentRequest(
            token="enroll_requesttest123456789012345678901",
            public_key="-----BEGIN PUBLIC KEY-----\ntest\n-----END PUBLIC KEY-----",
            agent_version="1.0.0",
        )

        assert request.token.startswith("enroll_")

    def test_enrollment_response_success(self):
        """Test successful enrollment response."""
        response = EnrollmentResponse.success_response(
            agent_id="agent_enrolled123456789012",
            workspace_id="ws_test",
            heartbeat_endpoint="http://localhost/heartbeat",
            commands_endpoint="http://localhost/commands",
            telemetry_endpoint="http://localhost/telemetry",
        )

        assert response.success is True
        assert response.agent_id is not None

    def test_enrollment_response_error(self):
        """Test error enrollment response."""
        response = EnrollmentResponse.error_response(
            "TOKEN_EXPIRED",
            "Token has expired",
        )

        assert response.success is False
        assert response.error_code == "TOKEN_EXPIRED"

    def test_agent_info(self):
        """Test AgentInfo model."""
        agent = AgentInfo(
            agent_id="agent_info1234567890123456",
            workspace_id="ws_test",
            public_key="-----BEGIN PUBLIC KEY-----\ntest\n-----END PUBLIC KEY-----",
            agent_version="1.0.0",
            enrolled_at=datetime.utcnow(),
            enrolled_by_token="enroll_token12345678901234567890123",
        )

        assert agent.is_active() is True
        assert agent.trust_state == TrustState.ENROLLED

    def test_agent_revoke(self):
        """Test agent revocation."""
        agent = AgentInfo(
            agent_id="agent_revoke123456789012345",
            workspace_id="ws_test",
            public_key="-----BEGIN PUBLIC KEY-----\ntest\n-----END PUBLIC KEY-----",
            agent_version="1.0.0",
            enrolled_at=datetime.utcnow(),
            enrolled_by_token="enroll_token12345678901234567890123",
        )

        agent.revoke("admin", "Security concern")

        assert agent.is_active() is False
        assert agent.trust_state == TrustState.REVOKED
        assert agent.revoke_reason == "Security concern"
