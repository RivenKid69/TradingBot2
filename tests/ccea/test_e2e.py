# -*- coding: utf-8 -*-
"""
CCEA Phase 1 E2E Tests.

Tests the complete flow:
1. User creates deployment in cloud
2. Cloud sends REQUEST_START_RUN
3. Agent requires local approval
4. Agent pulls signed artifact by digest
5. Agent runs "hello strategy" (no broker)
6. Agent reports state/telemetry

This validates Design Doc Phase 1 requirements.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from ccea.control_plane.enrollment import EnrollmentService
from ccea.control_plane.commands import CommandService, CommandDispatcher
from ccea.control_plane.blobs import InMemoryBlobStore, ConfigBlobStore
from ccea.control_plane.heartbeat import HeartbeatService

from ccea.agent.command_handler import CommandHandler, LocalCommandStatus
from ccea.agent.approval import ApprovalManager, ApprovalStatus

from ccea.artifact.builder import build_hello_strategy
from ccea.artifact.registry import ArtifactRegistry
from ccea.artifact.signer import ArtifactSigner, SignatureVerifier

from ccea.telemetry.collector import TelemetryCollector, TelemetryLevel
from ccea.telemetry.redaction import RedactionMiddleware

from ccea.crypto.keys import generate_keypair
from ccea.crypto.tokens import generate_idempotency_key

from ccea.models.enrollment import EnrollmentRequest
from ccea.models.protocol import (
    HeartbeatMessage,
    AgentState,
    DeploymentState,
    RunState,
    CommandStatus,
)


class TestE2EPhase1:
    """End-to-end tests for CCEA Phase 1."""

    @pytest.fixture
    def cloud_services(self):
        """Setup cloud services."""
        return {
            "enrollment": EnrollmentService(),
            "commands": CommandService(),
            "blobs": InMemoryBlobStore(),
            "heartbeat": HeartbeatService(),
        }

    @pytest.fixture
    def agent_keypair(self):
        """Generate agent keypair."""
        return generate_keypair(key_id="agent_e2e_key")

    @pytest.fixture
    def signing_keypair(self):
        """Generate artifact signing keypair."""
        return generate_keypair(key_id="artifact_signer")

    def test_e2e_enrollment_flow(self, cloud_services, agent_keypair):
        """Test complete enrollment flow."""
        enrollment = cloud_services["enrollment"]

        # 1. Admin creates enrollment token
        token = enrollment.create_token(
            workspace_id="ws_e2e_test",
            created_by="admin",
            ttl_hours=24,
        )
        assert token.is_valid() is True

        # 2. Agent sends enrollment request
        request = EnrollmentRequest(
            token=token.token_id,
            public_key=agent_keypair.get_public_key_pem(),
            agent_version="1.0.0",
            hostname="e2e-test-host",
        )

        response = enrollment.enroll_agent(request)

        # 3. Verify enrollment successful
        assert response.success is True
        assert response.agent_id.startswith("agent_")
        assert response.heartbeat_endpoint is not None
        assert response.commands_endpoint is not None

        return response.agent_id

    def test_e2e_artifact_build_and_sign(self, signing_keypair):
        """Test artifact build and signing flow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # 1. Build hello strategy artifact
            package = build_hello_strategy(output_dir, signing_key=signing_keypair)

            # 2. Verify artifact properties
            assert package.artifact_id == "hello_strategy"
            assert package.digest.startswith("sha256:")
            assert package.signature is not None

            # 3. Verify manifest
            assert package.manifest.schema_version == "1.0.0"
            assert package.manifest.artifact_type.value == "strategy"

            # 4. Verify no broker requirement
            if package.manifest.live_capabilities:
                assert package.manifest.live_capabilities.requires_broker_access is False

            return package

    def test_e2e_artifact_registry(self, signing_keypair):
        """Test artifact registry flow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            registry_dir = Path(tmpdir) / "registry"

            # 1. Build artifact
            package = build_hello_strategy(output_dir, signing_key=signing_keypair)

            # 2. Push to registry
            registry = ArtifactRegistry(registry_dir)

            from ccea.artifact.signer import SignatureInfo
            sig_info = SignatureInfo(
                algorithm=signing_keypair.algorithm.value,
                signature=package.signature,
                key_id=signing_keypair.key_id,
                signed_digest=package.digest,
            )

            entry = registry.push(
                artifact_path=package.package_path,
                manifest_path=package.manifest_path,
                artifact_id=package.artifact_id,
                version=package.version,
                signature=sig_info,
                workspace_id="ws_e2e",
            )

            # 3. Verify registry entry
            assert entry.digest == package.digest
            assert registry.exists(package.digest)

            # 4. Pull artifact
            pull_path = Path(tmpdir) / "pulled.zip"
            result = registry.pull(package.digest, pull_path)

            assert result is True
            assert pull_path.exists()

    def test_e2e_command_and_approval_flow(self, cloud_services):
        """Test command dispatch and approval flow."""
        commands = cloud_services["commands"]
        dispatcher = CommandDispatcher(commands)

        # Setup agent command handler
        agent_handler = CommandHandler()
        approval_manager = ApprovalManager()

        # 1. Cloud dispatches START_RUN command
        record = dispatcher.request_start_run(
            agent_id="agent_e2ecommand123456789",
            deployment_id="deploy_e2e",
            artifact_digest="sha256:e2e1234567890123456789012345678901234567890123456789012345678901",
        )

        # 2. Verify command requires approval
        assert record.requires_approval is True
        assert record.command_type == "REQUEST_START_RUN"

        # 3. Agent receives command
        agent_record = agent_handler.handle_command({
            "command_id": record.command_id,
            "command_type": record.command_type,
            "deployment_id": "deploy_e2e",
            "artifact_digest": "sha256:e2e1234567890123456789012345678901234567890123456789012345678901",
            "idempotency_key": record.idempotency_key,
            "requires_approval": True,
        })

        # 4. Verify agent requires approval
        assert agent_record.status == LocalCommandStatus.AWAITING_APPROVAL

        # 5. Create approval request
        approval_request = approval_manager.create_request(
            command_id=record.command_id,
            command_type=record.command_type,
            deployment_id="deploy_e2e",
            artifact_digest="sha256:e2e1234567890123456789012345678901234567890123456789012345678901",
        )

        assert approval_request.status == ApprovalStatus.PENDING

        # 6. User approves locally
        approval_result = approval_manager.approve(
            approval_request.request_id,
            approver="local_user",
        )

        assert approval_result.approved is True
        assert approval_result.evidence_hash is not None

        # 7. Agent approves command
        agent_handler.approve_command(
            record.command_id,
            evidence_hash=approval_result.evidence_hash,
        )

        # 8. Verify command approved
        updated = agent_handler.get_command(record.command_id)
        assert updated.status == LocalCommandStatus.APPROVED

    def test_e2e_safety_command_no_approval(self, cloud_services):
        """Test safety commands don't require approval."""
        commands = cloud_services["commands"]
        dispatcher = CommandDispatcher(commands)

        agent_handler = CommandHandler()

        # 1. Cloud dispatches STOP command
        record = dispatcher.request_stop_run(
            agent_id="agent_safety123456789012",
            deployment_id="deploy_safety",
            reason="Emergency stop",
        )

        # 2. Verify no approval required
        assert record.requires_approval is False

        # 3. Agent handles command immediately
        agent_record = agent_handler.handle_command({
            "command_id": record.command_id,
            "command_type": record.command_type,
            "deployment_id": "deploy_safety",
            "idempotency_key": record.idempotency_key,
            "requires_approval": False,
        })

        # 4. Command should be immediately approved (safety)
        assert agent_record.status == LocalCommandStatus.APPROVED

    def test_e2e_telemetry_with_redaction(self):
        """Test telemetry collection with mandatory redaction."""
        collector = TelemetryCollector(
            agent_id="agent_telemetrye2e12345678",
            level=TelemetryLevel.AGGREGATED,
        )

        # 1. Verify redaction always applied
        assert collector.redaction_applied is True

        # 2. Collect event with sensitive data
        event = collector.collect(
            event_type="STRATEGY_ITERATION",
            data={
                "iteration": 1,
                "password": "should_be_redacted",
                "normal": "visible",
            },
        )

        # 3. Verify sensitive data redacted
        assert event.data.get("password") == "[REDACTED]"
        assert event.data.get("normal") == "visible"

        # 4. Collect performance
        perf = collector.collect_performance(
            pnl=5.0,
            win_rate=60.0,
        )

        assert perf.event_type == "PERFORMANCE_SUMMARY"

    def test_e2e_heartbeat_flow(self, cloud_services):
        """Test heartbeat monitoring flow."""
        heartbeat = cloud_services["heartbeat"]

        # 1. Agent sends heartbeat
        msg = HeartbeatMessage(
            agent_id="agent_heartbeate2e12345678",
            state=AgentState(
                deployment_state=DeploymentState.RUNNING,
                run_state=RunState.RUNNING,
                agent_version="1.0.0",
                uptime_seconds=3600,
            ),
        )

        status = heartbeat.process_heartbeat(msg)

        # 2. Verify agent online
        from ccea.control_plane.heartbeat import AgentStatus
        assert status.status == AgentStatus.ONLINE

        # 3. Check status
        checked = heartbeat.check_agent_status("agent_heartbeate2e12345678")
        assert checked.status == AgentStatus.ONLINE

    def test_e2e_prohibited_order_fields(self):
        """Test that order-like fields are always rejected."""
        agent_handler = CommandHandler()

        # 1. Try to inject order-like payload
        bad_command = {
            "command_id": "cmd_bad",
            "command_type": "REQUEST_START_RUN",
            "deployment_id": "deploy_bad",
            "idempotency_key": "bad_idem_key_1234567890",
            "side": "BUY",  # PROHIBITED
            "quantity": 100,  # PROHIBITED
        }

        record = agent_handler.handle_command(bad_command)

        # 2. Verify command filtered
        assert record.status == LocalCommandStatus.FILTERED
        assert "side" in record.filter_reason or "quantity" in record.filter_reason

    def test_e2e_idempotency(self, cloud_services):
        """Test command idempotency."""
        commands = cloud_services["commands"]
        dispatcher = CommandDispatcher(commands)

        # 1. Create command with specific idempotency key
        from ccea.models.protocol import RequestStopRunCommand

        idem_key = "e2e_idem_key_123456789012"
        cmd = RequestStopRunCommand(
            deployment_id="deploy_idem",
            idempotency_key=idem_key,
        )

        record1 = commands.create_command("agent_idem1234567890123456", cmd, idem_key)
        record2 = commands.create_command("agent_idem1234567890123456", cmd, idem_key)

        # 2. Verify same command returned
        assert record1.command_id == record2.command_id

    def test_e2e_config_blob_immutability(self, cloud_services):
        """Test config blob immutability."""
        blobs = cloud_services["blobs"]
        config_store = ConfigBlobStore(blobs)

        # 1. Store config
        config1 = {"setting": "value1", "number": 1}
        digest1 = config_store.put_config(config1, workspace_id="ws_e2e")

        # 2. Store same config again
        digest2 = config_store.put_config(config1, workspace_id="ws_e2e")

        # 3. Same config = same digest
        assert digest1 == digest2

        # 4. Different config = different digest
        config2 = {"setting": "value2", "number": 2}
        digest3 = config_store.put_config(config2, workspace_id="ws_e2e")

        assert digest1 != digest3

    def test_e2e_signature_verification(self, signing_keypair):
        """Test signature verification flow."""
        signer = ArtifactSigner.from_keypair(signing_keypair)
        verifier = SignatureVerifier()

        verifier.add_trusted_key(signing_keypair.key_id, signing_keypair.public_key)

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Build and sign artifact
            package = build_hello_strategy(Path(tmpdir), signing_key=signing_keypair)

            from ccea.artifact.signer import SignatureInfo
            sig_info = SignatureInfo(
                algorithm=signing_keypair.algorithm.value,
                signature=package.signature,
                key_id=signing_keypair.key_id,
                signed_digest=package.digest,
            )

            # 2. Verify signature
            is_valid = verifier.verify_file(
                package.package_path,
                sig_info,
                signing_keypair.key_id,
            )

            assert is_valid is True


class TestE2EPhase1Compliance:
    """Tests verifying Phase 1 Done criteria per Design Doc."""

    def test_protocol_authenticated(self):
        """Verify: Protocol is authenticated."""
        from ccea.crypto.signing import MessageSigner, MessageVerifier
        from ccea.crypto.keys import generate_keypair

        keypair = generate_keypair(key_id="auth_key")
        signer = MessageSigner(keypair.private_key, "auth_key")
        verifier = MessageVerifier()
        verifier.add_key("auth_key", keypair.public_key)

        # All messages can be signed and verified
        message = {"type": "HEARTBEAT", "agent_id": "agent_auth1234567890123456"}
        signed = signer.sign(message)

        assert verifier.verify(signed) is True

    def test_commands_idempotent(self):
        """Verify: Commands are idempotent."""
        service = CommandService()

        from ccea.models.protocol import RequestStopRunCommand

        idem_key = "idempotent_test_key_1234"
        cmd = RequestStopRunCommand(
            deployment_id="deploy_idem",
            idempotency_key=idem_key,
        )

        r1 = service.create_command("agent_idem1234567890123456", cmd, idem_key)
        r2 = service.create_command("agent_idem1234567890123456", cmd, idem_key)

        # Same idempotency key = same command
        assert r1.command_id == r2.command_id

    def test_order_payload_impossible(self):
        """Verify: Order-like payload impossible by schema/CI."""
        handler = CommandHandler()

        # All order-like fields rejected
        for field in ["side", "quantity", "qty", "price", "order_type", "target_position"]:
            cmd = {
                "command_id": f"cmd_{field}",
                "command_type": "REQUEST_START_RUN",
                "deployment_id": "deploy",
                "idempotency_key": f"key_{field}_123456789012",
                field: "test_value",
            }

            record = handler.handle_command(cmd)
            assert record.status == LocalCommandStatus.FILTERED, f"Field {field} should be filtered"

    def test_redaction_mandatory(self):
        """Verify: Telemetry redaction cannot be disabled."""
        middleware = RedactionMiddleware()

        # Cannot disable
        middleware.disable()
        assert middleware.enabled is True

        # Sensitive data always redacted
        data = {"password": "secret", "api_key": "key123"}
        redacted = middleware.process(data)

        assert redacted["password"] == "[REDACTED]"
        assert redacted["api_key"] == "[REDACTED]"
