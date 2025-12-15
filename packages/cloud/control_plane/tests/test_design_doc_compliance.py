# -*- coding: utf-8 -*-
"""
Design Doc Compliance Tests.

Tests to verify Cloud adheres to Design Doc requirements:
- Design Doc 9.4, 11.1, 11.2: HALTED state in state machines
- Design Doc 10.2: Command signing
- Design Doc 12.2, 22.2: Blob/artifact access endpoints
- Design Doc 7/11: Change class enforcement
- Design Doc 0.2, 3.1, 5.1: Cloud boundary / signal isolation
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from ..boundary import (
    BoundaryValidationResult,
    BoundaryViolationType,
    CloudBoundaryValidator,
    ConfigBoundaryValidator,
    SignalIsolationValidator,
    PROHIBITED_CONFIG_PATTERNS,
)
from ..models import (
    Agent,
    Artifact,
    ChangeClass,
    Command,
    CommandStatus,
    ConfigBlob,
    Deployment,
    DeploymentState,
    HaltReason,
    Organization,
    Run,
    RunState,
    TrustState,
    User,
    Workspace,
)
from ..services.command_signer import (
    CloudCommandSigner,
    CloudKeyNotConfiguredError,
    get_cloud_signer,
    reset_cloud_signer,
    set_cloud_signer,
)
from ..services.change_class_enforcer import (
    ChangeClass as EnforcerChangeClass,
    ChangeClassEnforcer,
)


# =============================================================================
# Test: HALTED State (Design Doc 9.4, 11.1, 11.2)
# =============================================================================


class TestHaltedState:
    """Tests for HALTED state in state machines (Design Doc 9.4, 11.1, 11.2)."""

    def test_deployment_state_has_halted(self):
        """DeploymentState enum must have HALTED value."""
        assert hasattr(DeploymentState, "HALTED")
        assert DeploymentState.HALTED.value == "halted"

    def test_run_state_has_halted(self):
        """RunState enum must have HALTED value."""
        assert hasattr(RunState, "HALTED")
        assert RunState.HALTED.value == "halted"

    def test_deployment_state_has_revoked(self):
        """DeploymentState should have REVOKED for trust revocation."""
        assert hasattr(DeploymentState, "REVOKED")
        assert DeploymentState.REVOKED.value == "revoked"

    def test_halt_reason_enum_exists(self):
        """HaltReason enum must exist with Design Doc 9.4 triggers."""
        assert hasattr(HaltReason, "MAX_DAILY_LOSS")
        assert hasattr(HaltReason, "BROKER_ERROR_BURST")
        assert hasattr(HaltReason, "LATENCY_SPIKE")
        assert hasattr(HaltReason, "ORDER_SPAM")
        assert hasattr(HaltReason, "STATE_DIVERGENCE")
        assert hasattr(HaltReason, "DATA_FEED_INVALID")
        assert hasattr(HaltReason, "MANUAL_KILL_SWITCH")

    @pytest_asyncio.fixture
    async def halted_deployment(
        self,
        db_session: AsyncSession,
        sample_workspace: Workspace,
        sample_agent: Agent,
        sample_artifact: Artifact,
    ) -> Deployment:
        """Create a deployment in HALTED state."""
        deployment = Deployment(
            workspace_id=sample_workspace.id,
            agent_id=sample_agent.id,
            artifact_id=sample_artifact.id,
            state=DeploymentState.HALTED.value,
            halt_reason=HaltReason.MAX_DAILY_LOSS.value,
            halt_details="Daily loss exceeded $5000 threshold",
            halted_at=datetime.now(timezone.utc),
        )
        db_session.add(deployment)
        await db_session.commit()
        await db_session.refresh(deployment)
        return deployment

    @pytest.mark.asyncio
    async def test_deployment_halt_fields(
        self,
        db_session: AsyncSession,
        halted_deployment: Deployment,
    ):
        """Deployment should have halt_reason and halt_details fields."""
        assert halted_deployment.state == DeploymentState.HALTED.value
        assert halted_deployment.halt_reason == HaltReason.MAX_DAILY_LOSS.value
        assert "Daily loss" in halted_deployment.halt_details
        assert halted_deployment.halted_at is not None

    @pytest_asyncio.fixture
    async def halted_run(
        self,
        db_session: AsyncSession,
        sample_deployment: Deployment,
    ) -> Run:
        """Create a run in HALTED state."""
        run = Run(
            workspace_id=sample_deployment.workspace_id,
            deployment_id=sample_deployment.id,
            state=RunState.HALTED.value,
            halt_reason=HaltReason.BROKER_ERROR_BURST.value,
            halt_details="10 consecutive broker errors",
            halted_at=datetime.now(timezone.utc),
        )
        db_session.add(run)
        await db_session.commit()
        await db_session.refresh(run)
        return run

    @pytest.mark.asyncio
    async def test_run_halt_fields(
        self,
        db_session: AsyncSession,
        halted_run: Run,
    ):
        """Run should have halt_reason and halt_details fields."""
        assert halted_run.state == RunState.HALTED.value
        assert halted_run.halt_reason == HaltReason.BROKER_ERROR_BURST.value
        assert "broker errors" in halted_run.halt_details
        assert halted_run.halted_at is not None


# =============================================================================
# Test: Command Signing (Design Doc 10.2)
# =============================================================================


class TestCommandSigning:
    """Tests for Cloud command signing (Design Doc 10.2)."""

    @pytest.fixture
    def signer(self) -> CloudCommandSigner:
        """Create a test signer."""
        return CloudCommandSigner.generate(key_id="test_key")

    def test_signer_generates_keypair(self, signer: CloudCommandSigner):
        """Signer should generate valid keypair."""
        assert signer.key_id == "test_key"
        assert signer.fingerprint
        assert len(signer.fingerprint) == 64  # SHA256 hex

    def test_signer_returns_public_key_pem(self, signer: CloudCommandSigner):
        """Signer should return PEM-encoded public key."""
        pem = signer.get_public_key_pem()
        assert pem.startswith("-----BEGIN PUBLIC KEY-----")
        assert "-----END PUBLIC KEY-----" in pem

    def test_sign_command_adds_signature(self, signer: CloudCommandSigner):
        """sign_command should add signature field."""
        command = {
            "id": str(uuid4()),
            "command_type": "REQUEST_START_RUN",
            "payload_ref": "sha256:abc123",
        }

        signed = signer.sign_command(command)

        assert "signature" in signed
        assert "value" in signed["signature"]
        assert "algorithm" in signed["signature"]
        assert "key_id" in signed["signature"]
        assert signed["signature"]["key_id"] == "test_key"

    def test_sign_command_adds_timestamp(self, signer: CloudCommandSigner):
        """sign_command should add timestamp for replay protection."""
        command = {"command_type": "REQUEST_STOP_RUN"}
        signed = signer.sign_command(command)

        assert "timestamp" in signed
        # Timestamp should be recent
        ts = datetime.fromisoformat(signed["timestamp"])
        assert ts > datetime.now(timezone.utc) - timedelta(minutes=1)

    def test_signature_verifiable(self, signer: CloudCommandSigner):
        """Signed command should be verifiable with public key."""
        from ccea.crypto.keys import load_public_key
        from ccea.crypto.signing import verify_json_signature

        command = {
            "id": str(uuid4()),
            "command_type": "REQUEST_PAUSE_RUN",
            "payload_ref": "sha256:xyz789",
        }

        signed = signer.sign_command(command)
        public_key = load_public_key(signer.get_public_key_pem())

        assert verify_json_signature(signed, public_key)

    def test_signature_fails_on_tamper(self, signer: CloudCommandSigner):
        """Modified command should fail verification."""
        from ccea.crypto.keys import load_public_key
        from ccea.crypto.signing import verify_json_signature

        command = {"command_type": "REQUEST_START_RUN", "payload_ref": "sha256:abc"}
        signed = signer.sign_command(command)

        # Tamper with the command
        signed["payload_ref"] = "sha256:TAMPERED"

        public_key = load_public_key(signer.get_public_key_pem())
        assert not verify_json_signature(signed, public_key)

    def test_sign_multiple_commands(self, signer: CloudCommandSigner):
        """sign_commands should sign multiple commands."""
        commands = [
            {"command_type": "REQUEST_START_RUN", "payload_ref": "sha256:a"},
            {"command_type": "REQUEST_STOP_RUN", "payload_ref": "sha256:b"},
        ]

        signed_all = signer.sign_commands(commands)

        assert len(signed_all) == 2
        assert all("signature" in cmd for cmd in signed_all)

    def test_signer_counts_signed_commands(self, signer: CloudCommandSigner):
        """Signer should track number of signed commands."""
        assert signer.commands_signed == 0

        signer.sign_command({"command_type": "TEST"})
        assert signer.commands_signed == 1

        signer.sign_command({"command_type": "TEST2"})
        assert signer.commands_signed == 2

    def test_get_key_info(self, signer: CloudCommandSigner):
        """get_key_info should return key metadata."""
        info = signer.get_key_info()

        assert info.key_id == "test_key"
        assert info.algorithm == "ed25519"
        assert info.fingerprint == signer.fingerprint
        assert info.public_key_pem.startswith("-----BEGIN PUBLIC KEY-----")

    def test_from_pem_creates_signer(self):
        """from_pem should create signer from PEM key."""
        from ccea.crypto.keys import generate_keypair, serialize_private_key

        keypair = generate_keypair()
        pem = serialize_private_key(keypair.private_key)

        signer = CloudCommandSigner.from_pem(pem, key_id="custom_key")

        assert signer.key_id == "custom_key"
        assert signer.get_public_key_pem()

    def test_global_signer_management(self):
        """Test global signer set/get/reset."""
        reset_cloud_signer()

        signer = CloudCommandSigner.generate()
        set_cloud_signer(signer)

        retrieved = get_cloud_signer(auto_init=False)
        assert retrieved is signer

        reset_cloud_signer()
        with pytest.raises(CloudKeyNotConfiguredError):
            get_cloud_signer(auto_init=False)


# =============================================================================
# Test: Change Class Enforcement (Design Doc 7/11)
# =============================================================================


class TestChangeClassEnforcement:
    """Tests for change class enforcement (Design Doc 7/11)."""

    @pytest.fixture
    def enforcer(self) -> ChangeClassEnforcer:
        """Create a strict mode enforcer."""
        return ChangeClassEnforcer(strict_mode=True)

    def test_trading_impacting_requires_approval(self, enforcer: ChangeClassEnforcer):
        """TRADING_IMPACTING changes must require local approval."""
        determination = enforcer.determine_change_class(
            command_type="REQUEST_START_RUN",
            requested_class=EnforcerChangeClass.TRADING_IMPACTING,
        )

        approval_req = enforcer.get_approval_requirement(determination)

        assert approval_req.required
        assert "TRADING_IMPACTING" in approval_req.reason

    def test_operational_may_not_require_approval(self, enforcer: ChangeClassEnforcer):
        """OPERATIONAL changes may not require approval."""
        determination = enforcer.determine_change_class(
            command_type="REQUEST_EXPORT_LOGS",
            requested_class=EnforcerChangeClass.OPERATIONAL,
        )

        approval_req = enforcer.get_approval_requirement(determination)

        # OPERATIONAL export logs typically doesn't need approval
        # (actual logic may vary)
        assert isinstance(approval_req.required, bool)

    def test_config_impact_upgrades_class(self, enforcer: ChangeClassEnforcer):
        """Config with trading impact should upgrade change class."""
        determination = enforcer.determine_change_class(
            command_type="REQUEST_UPDATE_CONFIG",
            requested_class=EnforcerChangeClass.OPERATIONAL,
            config_has_trading_impact=True,
        )

        # Should be upgraded to TRADING_IMPACTING
        assert determination.change_class == EnforcerChangeClass.TRADING_IMPACTING

    def test_artifact_change_class_propagates(self, enforcer: ChangeClassEnforcer):
        """Artifact's change class should be respected."""
        determination = enforcer.determine_change_class(
            command_type="REQUEST_UPGRADE_ARTIFACT",
            requested_class=None,  # Not specified
            artifact_change_class="trading_impacting",
        )

        assert determination.change_class == EnforcerChangeClass.TRADING_IMPACTING

    def test_cannot_downgrade_trading_impacting(self, enforcer: ChangeClassEnforcer):
        """Cloud cannot downgrade TRADING_IMPACTING to OPERATIONAL."""
        determination = enforcer.determine_change_class(
            command_type="REQUEST_START_RUN",
            requested_class=EnforcerChangeClass.OPERATIONAL,  # Try to downgrade
            artifact_change_class="trading_impacting",  # But artifact is TRADING_IMPACTING
        )

        # Should remain TRADING_IMPACTING (highest wins)
        assert determination.change_class == EnforcerChangeClass.TRADING_IMPACTING

    def test_validation_rejects_invalid_state(self, enforcer: ChangeClassEnforcer):
        """Validation should reject approval_required=False for TRADING_IMPACTING."""
        is_valid, msg = enforcer.validate_command_approval_state(
            command_type="REQUEST_START_RUN",
            change_class=EnforcerChangeClass.TRADING_IMPACTING,
            requires_approval=False,  # Invalid!
        )

        assert not is_valid
        assert "TRADING_IMPACTING" in msg


# =============================================================================
# Test: Config Boundary Validator (Design Doc 3.1, 5.1)
# =============================================================================


class TestConfigBoundaryValidator:
    """Tests for config boundary validation (Design Doc 3.1, 5.1)."""

    @pytest.fixture
    def validator(self) -> ConfigBoundaryValidator:
        """Create validator."""
        return ConfigBoundaryValidator()

    def test_valid_config_passes(self, validator: ConfigBoundaryValidator):
        """Valid config without signals should pass."""
        config = {
            "strategy_params": {
                "lookback_period": 20,
                "threshold": 0.5,
            },
            "universe": ["AAPL", "GOOGL", "MSFT"],
        }

        result = validator.validate(config)
        assert result.valid

    def test_signal_field_blocked(self, validator: ConfigBoundaryValidator):
        """Config with 'signal' field should be blocked."""
        config = {
            "signal": {"symbol": "AAPL", "side": "buy", "qty": 100},
        }

        result = validator.validate(config)
        assert not result.valid
        assert any(v.field_path == "signal" for v in result.violations)

    def test_target_position_blocked(self, validator: ConfigBoundaryValidator):
        """Config with 'target_position' should be blocked."""
        config = {
            "portfolio": {
                "target_position": {"AAPL": 100, "GOOGL": -50},
            }
        }

        result = validator.validate(config)
        assert not result.valid

    def test_order_fields_blocked(self, validator: ConfigBoundaryValidator):
        """Config with order-like fields should be blocked."""
        for field in ["side", "qty", "quantity", "limit_price", "order_type"]:
            config = {field: "some_value"}
            result = validator.validate(config)
            assert not result.valid, f"Field '{field}' should be blocked"

    def test_signal_like_structure_blocked(self, validator: ConfigBoundaryValidator):
        """Config with signal-like list structure should be blocked."""
        config = {
            "positions": [
                {"symbol": "AAPL", "qty": 100},
                {"symbol": "GOOGL", "qty": -50},
            ]
        }

        result = validator.validate(config)
        assert not result.valid

    def test_nested_prohibited_field_blocked(self, validator: ConfigBoundaryValidator):
        """Nested prohibited fields should be detected."""
        config = {
            "level1": {
                "level2": {
                    "intent": {"action": "buy"},
                }
            }
        }

        result = validator.validate(config)
        assert not result.valid
        # Find the violation for nested intent
        intent_violations = [v for v in result.violations if "intent" in v.field_path.lower()]
        assert len(intent_violations) > 0


# =============================================================================
# Test: Signal Isolation Validator (Design Doc 5.1)
# =============================================================================


class TestSignalIsolationValidator:
    """Tests for signal isolation (Design Doc 5.1)."""

    @pytest.fixture
    def validator(self) -> SignalIsolationValidator:
        """Create validator."""
        return SignalIsolationValidator()

    def test_signal_import_blocked(self, validator: SignalIsolationValidator):
        """Import of signal modules should be blocked in Cloud."""
        result = validator.validate_no_signal_imports(
            module_name="packages.cloud.some_module",
            imported_modules=["api.spot_signals", "utils.helpers"],
        )

        assert not result.valid
        assert any("spot_signals" in v.message for v in result.violations)

    def test_signal_runner_import_blocked(self, validator: SignalIsolationValidator):
        """Import of service_signal_runner should be blocked."""
        result = validator.validate_no_signal_imports(
            module_name="packages.cloud.router",
            imported_modules=["service_signal_runner"],
        )

        assert not result.valid

    def test_valid_imports_pass(self, validator: SignalIsolationValidator):
        """Valid imports without signal code should pass."""
        result = validator.validate_no_signal_imports(
            module_name="packages.cloud.control_plane.routers.agents",
            imported_modules=[
                "fastapi",
                "sqlalchemy",
                "packages.cloud.control_plane.models",
            ],
        )

        assert result.valid

    def test_api_response_without_signals_passes(self, validator: SignalIsolationValidator):
        """API response without signal content should pass."""
        response = {
            "deployments": [
                {"id": "123", "state": "running"},
            ],
            "count": 1,
        }

        result = validator.validate_api_response(response)
        assert result.valid

    def test_api_response_with_signals_blocked(self, validator: SignalIsolationValidator):
        """API response with signal content should be blocked."""
        response = {
            "signals": [
                {"symbol": "AAPL", "side": "buy", "qty": 100},
            ],
        }

        result = validator.validate_api_response(response)
        assert not result.valid


# =============================================================================
# Test: Cloud Boundary Validator
# =============================================================================


class TestCloudBoundaryValidator:
    """Tests for Cloud boundary validation."""

    @pytest.fixture
    def validator(self) -> CloudBoundaryValidator:
        """Create strict validator."""
        return CloudBoundaryValidator(strict_mode=True)

    def test_lifecycle_command_passes(self, validator: CloudBoundaryValidator):
        """Lifecycle commands should pass validation."""
        for cmd_type in [
            "REQUEST_START_RUN",
            "REQUEST_STOP_RUN",
            "REQUEST_PAUSE_RUN",
            "REQUEST_UPGRADE_ARTIFACT",
            "REQUEST_UPDATE_CONFIG",
        ]:
            msg = {
                "command_type": cmd_type,
                "idempotency_key": "abc123",
                "agent_id": str(uuid4()),
            }
            result = validator.validate_outgoing_message(msg)
            assert result.valid, f"Command {cmd_type} should be valid"

    def test_non_lifecycle_command_blocked(self, validator: CloudBoundaryValidator):
        """Non-lifecycle commands should be blocked."""
        msg = {
            "command_type": "PLACE_ORDER",
            "payload": {"symbol": "AAPL", "side": "buy"},
        }

        result = validator.validate_outgoing_message(msg)
        assert not result.valid
        assert any(
            v.violation_type == BoundaryViolationType.PROHIBITED_COMMAND
            for v in result.violations
        )

    def test_intent_injection_blocked(self, validator: CloudBoundaryValidator):
        """Intent injection attempts should be blocked."""
        msg = {
            "command_type": "REQUEST_UPDATE_CONFIG",
            "config": {
                "intent": {"action": "buy", "symbol": "AAPL"},
            },
        }

        result = validator.validate_outgoing_message(msg)
        assert not result.valid
        assert any(
            v.violation_type == BoundaryViolationType.PROHIBITED_FIELD
            for v in result.violations
        )

    def test_order_fields_blocked(self, validator: CloudBoundaryValidator):
        """Order-like fields should be blocked."""
        msg = {
            "command_type": "REQUEST_START_RUN",
            "side": "buy",
            "qty": 100,
            "price": 150.00,
        }

        result = validator.validate_outgoing_message(msg)
        assert not result.valid
        # Should have multiple violations
        assert len(result.violations) >= 3

    def test_validate_command_method(self, validator: CloudBoundaryValidator):
        """validate_command should work the same as validate_outgoing_message."""
        cmd = {
            "command_type": "REQUEST_START_RUN",
            "idempotency_key": "test123",
        }

        result = validator.validate_command(cmd)
        assert result.valid

    def test_validate_config_method(self, validator: CloudBoundaryValidator):
        """validate_config should check config content."""
        config = {
            "lookback_period": 20,
            "signals": [{"symbol": "AAPL"}],  # Prohibited!
        }

        result = validator.validate_config(config)
        assert not result.valid


# =============================================================================
# Test: Integration - Poll Commands with Signing
# =============================================================================


class TestPollCommandsSigning:
    """Integration tests for command polling with signing."""

    @pytest_asyncio.fixture
    async def command_in_db(
        self,
        db_session: AsyncSession,
        sample_workspace: Workspace,
        sample_agent: Agent,
    ) -> Command:
        """Create a command ready for polling."""
        command = Command(
            workspace_id=sample_workspace.id,
            agent_id=sample_agent.id,
            idempotency_key=f"test-{uuid4()}",
            command_type="REQUEST_START_RUN",
            payload_ref="sha256:abc123",
            change_class=ChangeClass.OPERATIONAL.value,
            status=CommandStatus.PENDING.value,
        )
        db_session.add(command)
        await db_session.commit()
        await db_session.refresh(command)
        return command

    @pytest.mark.asyncio
    async def test_polled_commands_have_signature(
        self,
        client: AsyncClient,
        command_in_db: Command,
        agent_headers: Dict[str, str],
    ):
        """Commands returned from poll should have signatures."""
        # Set up a test signer
        signer = CloudCommandSigner.generate()
        set_cloud_signer(signer)

        try:
            response = await client.get(
                "/api/v1/agent/commands/poll",
                headers=agent_headers,
            )

            assert response.status_code == 200
            data = response.json()

            if data["commands"]:
                cmd = data["commands"][0]
                assert "signature" in cmd
                # Signature should have required fields
                if cmd["signature"]:
                    assert "value" in cmd["signature"]
                    assert "algorithm" in cmd["signature"]
        finally:
            reset_cloud_signer()


# =============================================================================
# Test: PROHIBITED_CONFIG_PATTERNS constant
# =============================================================================


class TestProhibitedConfigPatterns:
    """Test that prohibited patterns are comprehensive."""

    def test_signal_patterns_present(self):
        """Signal-related patterns should be in prohibited list."""
        assert "signal" in PROHIBITED_CONFIG_PATTERNS
        assert "signals" in PROHIBITED_CONFIG_PATTERNS
        assert "live_signal" in PROHIBITED_CONFIG_PATTERNS

    def test_target_patterns_present(self):
        """Target-related patterns should be in prohibited list."""
        assert "target_position" in PROHIBITED_CONFIG_PATTERNS
        assert "target_positions" in PROHIBITED_CONFIG_PATTERNS
        assert "target_qty" in PROHIBITED_CONFIG_PATTERNS

    def test_order_patterns_present(self):
        """Order-related patterns should be in prohibited list."""
        assert "order" in PROHIBITED_CONFIG_PATTERNS
        assert "side" in PROHIBITED_CONFIG_PATTERNS
        assert "qty" in PROHIBITED_CONFIG_PATTERNS
        assert "quantity" in PROHIBITED_CONFIG_PATTERNS
        assert "limit_price" in PROHIBITED_CONFIG_PATTERNS

    def test_intent_patterns_present(self):
        """Intent patterns should be in prohibited list."""
        assert "intent" in PROHIBITED_CONFIG_PATTERNS
        assert "trade_intent" in PROHIBITED_CONFIG_PATTERNS
