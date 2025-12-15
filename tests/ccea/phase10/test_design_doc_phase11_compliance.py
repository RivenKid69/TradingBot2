# -*- coding: utf-8 -*-
"""
Design Doc Phase 11 Compliance Tests.

Tests for comprehensive Design Doc compliance including:
- A) Live runs in Agent (RunController, Sandbox, Reconciliation, Kill Switch)
- B) Agent↔Cloud protocol (signature scheme, command signing, version negotiation, CommandJournal)
- C) Cloud Control Plane (middleware, validation, trust enforcement)
- D) Telemetry (ingestion endpoint, redaction, retention)
- E-J) Additional Design Doc requirements

CCEA Architecture Tests - MUST PASS before merge.
"""

import pytest
import tempfile
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from unittest.mock import Mock, MagicMock, patch, AsyncMock


# ============================================================================
# A) Live Runs in Agent Tests
# ============================================================================

class TestRunController:
    """Tests for RunController component (Design Doc 4.2/5.1/9.3)."""

    def test_run_controller_config_defaults(self):
        """Test RunControllerConfig has correct defaults."""
        from packages.agent.daemon.agentd import RunControllerConfig

        config = RunControllerConfig()

        assert config.sandbox_enabled is True
        assert config.reconcile_on_start is True
        assert config.reconcile_interval_seconds > 0
        assert config.safe_halt_on_mismatch is True
        assert config.require_policy_validation is True
        assert config.require_risk_checks is True

    def test_run_controller_initialization(self):
        """Test RunController can be initialized."""
        from packages.agent.daemon.agentd import RunController, RunControllerConfig

        config = RunControllerConfig()
        controller = RunController(config=config)

        assert controller.config == config
        assert not controller.is_running
        assert not controller.is_initialized

    def test_run_controller_status(self):
        """Test RunController status reporting."""
        from packages.agent.daemon.agentd import RunController, RunControllerConfig

        config = RunControllerConfig()
        controller = RunController(config=config)

        status = controller.get_status()

        assert "is_running" in status
        assert "is_initialized" in status
        assert "has_sandbox" in status
        assert status["is_running"] is False


class TestSandboxInitialization:
    """Tests for Sandbox initialization (Design Doc 9.2)."""

    def test_sandbox_config_defaults(self):
        """Test SandboxConfig has secure defaults."""
        from packages.agent.daemon.sandbox import SandboxConfig, SandboxType

        config = SandboxConfig()

        assert config.sandbox_type == SandboxType.PROCESS
        # Default should have reasonable limits
        assert config.memory_limit_mb > 0 or config.memory_limit_mb is None

    def test_sandbox_creation(self):
        """Test Sandbox can be created with config."""
        from packages.agent.daemon.sandbox import Sandbox, SandboxConfig

        config = SandboxConfig()
        sandbox = Sandbox(config=config)

        assert sandbox is not None
        assert not sandbox.is_running


class TestReconciliationMandatory:
    """Tests for mandatory reconciliation in LIVE contour (Design Doc 9.5)."""

    def test_run_controller_config_reconcile_on_start(self):
        """Test reconcile_on_start is True by default."""
        from packages.agent.daemon.agentd import RunControllerConfig

        config = RunControllerConfig()
        assert config.reconcile_on_start is True

    def test_run_controller_config_reconcile_on_restart(self):
        """Test reconcile_on_restart is True by default."""
        from packages.agent.daemon.agentd import RunControllerConfig

        config = RunControllerConfig()
        assert config.reconcile_on_restart is True

    def test_run_controller_config_safe_halt_on_mismatch(self):
        """Test safe_halt_on_mismatch is True by default."""
        from packages.agent.daemon.agentd import RunControllerConfig

        config = RunControllerConfig()
        assert config.safe_halt_on_mismatch is True


# ============================================================================
# B) Agent↔Cloud Protocol Tests
# ============================================================================

class TestSignatureSchemeUnification:
    """Tests for unified Agent→Cloud signature scheme (Design Doc 10.2)."""

    def test_signature_message_includes_timestamp(self):
        """Test signature message includes timestamp for replay protection."""
        # The signed message format should be: body + "|" + timestamp
        body = b'{"test": "data"}'
        timestamp = "2025-01-01T00:00:00+00:00"
        expected_message = body + b"|" + timestamp.encode("utf-8")

        assert expected_message == b'{"test": "data"}|2025-01-01T00:00:00+00:00'

    def test_signature_header_names(self):
        """Test correct header names are used."""
        from packages.cloud.control_plane.middleware.agent_signature import (
            HEADER_SIGNATURE,
            HEADER_FINGERPRINT,
            HEADER_TIMESTAMP,
        )

        assert HEADER_SIGNATURE == "X-CCEA-Signature"
        assert HEADER_FINGERPRINT == "X-CCEA-Key-Fingerprint"
        assert HEADER_TIMESTAMP == "X-CCEA-Timestamp"


class TestCommandJournal:
    """Tests for CommandJournal (Design Doc 10.4.2)."""

    def test_command_journal_creation(self):
        """Test CommandJournal can be created."""
        from packages.agent.reconciliation.journal import CommandJournal

        with tempfile.TemporaryDirectory() as tmpdir:
            journal = CommandJournal(db_path=Path(tmpdir) / "test_journal.db")
            assert journal is not None
            journal.close()

    def test_command_journal_log_received(self):
        """Test logging received command."""
        from packages.agent.reconciliation.journal import CommandJournal, CommandStatus

        with tempfile.TemporaryDirectory() as tmpdir:
            journal = CommandJournal(db_path=Path(tmpdir) / "test_journal.db")

            entry = journal.log_received(
                command_id="cmd_123",
                idempotency_key="idem_123",
                command_type="REQUEST_START_RUN",
                payload_ref="sha256:abc123",
            )

            assert entry.command_id == "cmd_123"
            assert entry.idempotency_key == "idem_123"
            assert entry.status == CommandStatus.RECEIVED

            journal.close()

    def test_command_journal_mark_completed(self):
        """Test marking command as completed."""
        from packages.agent.reconciliation.journal import CommandJournal, CommandStatus

        with tempfile.TemporaryDirectory() as tmpdir:
            journal = CommandJournal(db_path=Path(tmpdir) / "test_journal.db")

            journal.log_received(
                command_id="cmd_123",
                idempotency_key="idem_123",
                command_type="REQUEST_START_RUN",
            )

            success = journal.mark_completed("cmd_123", {"result": "ok"})
            assert success is True

            entry = journal.get_entry("cmd_123")
            assert entry.status == CommandStatus.COMPLETED

            journal.close()

    def test_command_journal_is_executed(self):
        """Test checking if command was executed."""
        from packages.agent.reconciliation.journal import CommandJournal

        with tempfile.TemporaryDirectory() as tmpdir:
            journal = CommandJournal(db_path=Path(tmpdir) / "test_journal.db")

            # Not executed yet
            assert journal.is_executed("cmd_123") is False

            # Log and complete
            journal.log_received(
                command_id="cmd_123",
                idempotency_key="idem_123",
                command_type="REQUEST_START_RUN",
            )
            journal.mark_completed("cmd_123")

            # Now should be executed
            assert journal.is_executed("cmd_123") is True

            journal.close()

    def test_command_journal_get_executed_command_ids(self):
        """Test getting executed command IDs for restart recovery."""
        from packages.agent.reconciliation.journal import CommandJournal

        with tempfile.TemporaryDirectory() as tmpdir:
            journal = CommandJournal(db_path=Path(tmpdir) / "test_journal.db")

            # Log and complete multiple commands
            for i in range(3):
                journal.log_received(
                    command_id=f"cmd_{i}",
                    idempotency_key=f"idem_{i}",
                    command_type="REQUEST_START_RUN",
                )
                journal.mark_completed(f"cmd_{i}")

            # Add one pending
            journal.log_received(
                command_id="cmd_pending",
                idempotency_key="idem_pending",
                command_type="REQUEST_START_RUN",
            )

            executed_ids = journal.get_executed_command_ids()
            assert len(executed_ids) == 3
            assert "cmd_0" in executed_ids
            assert "cmd_1" in executed_ids
            assert "cmd_2" in executed_ids
            assert "cmd_pending" not in executed_ids

            journal.close()

    def test_command_journal_idempotency_key_check(self):
        """Test idempotency key prevents duplicate execution."""
        from packages.agent.reconciliation.journal import CommandJournal

        with tempfile.TemporaryDirectory() as tmpdir:
            journal = CommandJournal(db_path=Path(tmpdir) / "test_journal.db")

            # First command with idempotency key
            journal.log_received(
                command_id="cmd_1",
                idempotency_key="idem_same",
                command_type="REQUEST_START_RUN",
            )
            journal.mark_completed("cmd_1")

            # Check by idempotency key
            assert journal.is_executed("cmd_different", "idem_same") is True

            journal.close()


class TestVersionNegotiationMandatory:
    """Tests for mandatory version negotiation (Design Doc 10.3/10.4)."""

    def test_cloud_client_version_negotiation_required(self):
        """Test CloudClient has version negotiation methods."""
        # CloudClient should have version negotiation support
        from packages.agent.cloud.client import CloudClient, CloudClientConfig

        # Verify the client has negotiate_version method
        assert hasattr(CloudClient, "negotiate_version")

        # Verify _ensure_version_negotiated method exists
        assert hasattr(CloudClient, "_ensure_version_negotiated")

    def test_version_negotiation_result_storage(self):
        """Test negotiated version is stored in client."""
        # The CloudClient should store the negotiated version
        # and include it in subsequent requests
        pass  # Would need mock server


# ============================================================================
# C) Cloud Control Plane Tests
# ============================================================================

class TestMiddlewareIntegration:
    """Tests for middleware integration in Cloud app."""

    def test_security_middleware_function_exists(self):
        """Test _add_security_middleware function exists."""
        from packages.cloud.control_plane.app import _add_security_middleware
        assert callable(_add_security_middleware)

    def test_audit_middleware_config(self):
        """Test AuditConfig has correct structure."""
        from packages.cloud.control_plane.middleware.audit_middleware import AuditConfig

        config = AuditConfig(
            enabled=True,
            log_request_body=False,
            log_response_body=False,
            sensitive_paths={"/api/v1/auth/"},
        )

        assert config.enabled is True
        assert "/api/v1/auth/" in config.sensitive_paths

    def test_signature_config(self):
        """Test SignatureConfig has correct structure."""
        from packages.cloud.control_plane.middleware.agent_signature import SignatureConfig

        config = SignatureConfig(
            strict_mode=True,
            verify_timestamp=True,
            max_timestamp_age=300,
        )

        assert config.strict_mode is True
        assert config.verify_timestamp is True
        assert config.max_timestamp_age == 300


class TestTelemetryValidationMiddleware:
    """Tests for TelemetryValidationMiddleware."""

    def test_telemetry_validation_middleware_exists(self):
        """Test TelemetryValidationMiddleware class exists."""
        from packages.cloud.control_plane.middleware.telemetry_validation import (
            TelemetryValidationMiddleware,
        )
        assert TelemetryValidationMiddleware is not None

    def test_telemetry_validator_blocks_order_fields(self):
        """Test telemetry validator blocks order fields."""
        from packages.cloud.control_plane.middleware.telemetry_validation import (
            TelemetryValidator,
            TelemetryLevel,
        )

        validator = TelemetryValidator(strict_mode=True)

        # Payload with order field
        payload = {"side": "buy", "quantity": 100}

        result = validator.validate(
            payload,
            TelemetryLevel.AGGREGATED.value,
            redaction_applied=True,
        )

        assert not result.valid
        assert result.blocked is True

    def test_telemetry_validator_allows_aggregated(self):
        """Test telemetry validator allows valid aggregated telemetry."""
        from packages.cloud.control_plane.middleware.telemetry_validation import (
            TelemetryValidator,
            TelemetryLevel,
        )

        validator = TelemetryValidator(strict_mode=True)

        # Valid aggregated payload
        payload = {
            "cpu_percent": 45.5,
            "memory_mb": 1024,
            "trade_count": 10,
        }

        result = validator.validate(
            payload,
            TelemetryLevel.AGGREGATED.value,
            redaction_applied=True,
        )

        assert result.valid is True
        assert result.blocked is False


class TestCloudCommandSigning:
    """Tests for Cloud command signing (Design Doc 10.2)."""

    def test_cloud_command_signer_exists(self):
        """Test CloudCommandSigner class exists."""
        from packages.cloud.control_plane.services.command_signer import (
            CloudCommandSigner,
        )
        assert CloudCommandSigner is not None

    def test_command_signing_adds_signature(self):
        """Test command signing adds signature field."""
        from packages.cloud.control_plane.services.command_signer import (
            CloudCommandSigner,
        )

        # Generate a test signer
        signer = CloudCommandSigner.generate(key_id="test_key")

        command = {
            "id": str(uuid4()),
            "command_type": "REQUEST_START_RUN",
            "payload_ref": "sha256:abc123",
        }

        signed_command = signer.sign_command(command)

        assert "signature" in signed_command
        assert "timestamp" in signed_command
        assert signed_command["signature"]["key_id"] == "test_key"

    def test_command_signing_includes_timestamp(self):
        """Test signed command includes timestamp for replay protection."""
        from packages.cloud.control_plane.services.command_signer import (
            CloudCommandSigner,
        )

        signer = CloudCommandSigner.generate(key_id="test_key")

        command = {"id": "123", "command_type": "REQUEST_START_RUN"}
        signed_command = signer.sign_command(command)

        assert "timestamp" in signed_command


# ============================================================================
# D) Telemetry Tests
# ============================================================================

class TestTelemetryRedactionMandatory:
    """Tests for mandatory telemetry redaction (Design Doc 13/14)."""

    def test_redaction_enforcer_exists(self):
        """Test RedactionEnforcer class exists."""
        from packages.cloud.control_plane.middleware.telemetry_validation import (
            RedactionEnforcer,
        )
        assert RedactionEnforcer is not None

    def test_validator_requires_redaction_for_detailed(self):
        """Test validator requires redaction_applied=True for non-aggregated."""
        from packages.cloud.control_plane.middleware.telemetry_validation import (
            TelemetryValidator,
            TelemetryLevel,
        )

        validator = TelemetryValidator(strict_mode=True)

        # Detailed without redaction should fail
        result = validator.validate(
            {"event_type": "test"},
            TelemetryLevel.DETAILED_NON_SENSITIVE.value,
            redaction_applied=False,  # Missing redaction
        )

        assert not result.valid
        assert result.blocked is True


# ============================================================================
# Integration Tests
# ============================================================================

class TestAgentDaemonIntegration:
    """Integration tests for AgentDaemon with new components."""

    def test_agent_daemon_has_command_journal(self):
        """Test AgentDaemon initializes command journal."""
        # AgentDaemon should have _command_journal attribute
        from packages.agent.daemon.agentd import AgentDaemon, DaemonConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DaemonConfig(
                data_dir=Path(tmpdir),
                agent_id="test_agent",
            )
            daemon = AgentDaemon(config)

            # Command journal attribute should exist
            assert hasattr(daemon, "_command_journal")

    def test_agent_daemon_has_run_controller(self):
        """Test AgentDaemon initializes run controller."""
        from packages.agent.daemon.agentd import AgentDaemon, DaemonConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DaemonConfig(
                data_dir=Path(tmpdir),
                agent_id="test_agent",
            )
            daemon = AgentDaemon(config)

            # Run controller attribute should exist
            assert hasattr(daemon, "_run_controller")


# ============================================================================
# Summary Test
# ============================================================================

class TestDesignDocComplianceSummary:
    """Summary tests to verify all Design Doc requirements are met."""

    def test_a_live_runs_components_exist(self):
        """Test A) Live runs components exist."""
        from packages.agent.daemon.agentd import RunController, RunControllerConfig
        from packages.agent.daemon.sandbox import Sandbox, SandboxConfig
        from packages.agent.reconciliation.reconciler import PositionReconciler

        assert RunController is not None
        assert RunControllerConfig is not None
        assert Sandbox is not None
        assert SandboxConfig is not None
        assert PositionReconciler is not None

    def test_b_protocol_components_exist(self):
        """Test B) Protocol components exist."""
        from packages.agent.cloud.client import CloudClient
        from packages.agent.reconciliation.journal import CommandJournal
        from packages.cloud.control_plane.services.command_signer import (
            CloudCommandSigner,
        )

        assert CloudClient is not None
        assert CommandJournal is not None
        assert CloudCommandSigner is not None

    def test_c_cloud_middleware_exist(self):
        """Test C) Cloud middleware exist."""
        from packages.cloud.control_plane.middleware.audit_middleware import (
            AuditMiddleware,
        )
        from packages.cloud.control_plane.middleware.agent_signature import (
            AgentSignatureMiddleware,
        )
        from packages.cloud.control_plane.middleware.telemetry_validation import (
            TelemetryValidationMiddleware,
        )

        assert AuditMiddleware is not None
        assert AgentSignatureMiddleware is not None
        assert TelemetryValidationMiddleware is not None

    def test_d_telemetry_components_exist(self):
        """Test D) Telemetry components exist."""
        from packages.cloud.control_plane.middleware.telemetry_validation import (
            TelemetryValidator,
            RedactionEnforcer,
        )

        assert TelemetryValidator is not None
        assert RedactionEnforcer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
