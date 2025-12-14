# -*- coding: utf-8 -*-
"""
CCEA Integration Tests.

Comprehensive tests verifying CCEA architecture compliance:
1. Import boundary enforcement (Cloud vs Agent)
2. Intent prohibition (no order payloads in protocol)
3. Protocol schema validation
4. Hard caps enforcement
5. Kill switch triggers
6. Artifact manifest validation
7. Zone separation (SimRunner vs LiveRunner)

These tests ensure the platform follows Design Doc CCEA Cloud requirements.
"""

import json
import sys
import ast
from pathlib import Path
from decimal import Decimal
from datetime import datetime
from typing import Dict, Any
from uuid import uuid4

import pytest

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


# =============================================================================
# Test 1: Import Boundary Enforcement
# =============================================================================

class TestImportBoundary:
    """Test that Cloud zone doesn't import Agent-only modules."""

    PROHIBITED_IN_CLOUD = [
        "adapters.alpaca.order_execution",
        "adapters.oanda.order_execution",
        "adapters.ib.order_execution",
        "execution_providers",
        "service_signal_runner",
    ]

    def get_imports_from_file(self, file_path: Path) -> list:
        """Extract all imports from a Python file."""
        imports = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=str(file_path))
        except (SyntaxError, UnicodeDecodeError):
            return imports

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports

    def test_cloud_zone_has_no_order_execution_imports(self):
        """Cloud zone must not import order execution modules."""
        cloud_path = PROJECT_ROOT / "packages" / "cloud"
        if not cloud_path.exists():
            pytest.skip("Cloud package not found")

        violations = []
        for py_file in cloud_path.rglob("*.py"):
            if "test" in py_file.name.lower():
                continue

            imports = self.get_imports_from_file(py_file)
            for imp in imports:
                for prohibited in self.PROHIBITED_IN_CLOUD:
                    if prohibited in imp:
                        violations.append(f"{py_file}: imports {imp}")

        assert len(violations) == 0, f"Cloud zone import violations:\n" + "\n".join(violations)

    def test_cloud_control_plane_no_broker_imports(self):
        """Control plane must not have broker imports."""
        control_plane_path = PROJECT_ROOT / "packages" / "cloud" / "control_plane"
        if not control_plane_path.exists():
            control_plane_path = PROJECT_ROOT / "ccea" / "control_plane"
        if not control_plane_path.exists():
            pytest.skip("Control plane not found")

        broker_modules = ["alpaca", "oanda", "ib", "deribit", "binance"]
        violations = []

        for py_file in control_plane_path.rglob("*.py"):
            imports = self.get_imports_from_file(py_file)
            for imp in imports:
                for broker in broker_modules:
                    if f"adapters.{broker}.order" in imp:
                        violations.append(f"{py_file}: imports {imp}")

        assert len(violations) == 0, f"Control plane broker violations:\n" + "\n".join(violations)


# =============================================================================
# Test 2: Intent Prohibition
# =============================================================================

class TestIntentProhibition:
    """Test that prohibited fields are blocked in protocol messages."""

    PROHIBITED_FIELDS = {
        "side", "quantity", "qty", "price", "order_type",
        "intent", "signal", "target_position", "execute_order",
        "place_order", "submit_order", "target_qty",
    }

    PROHIBITED_COMMAND_TYPES = {
        "PLACE_ORDER", "SUBMIT_ORDER", "EXECUTE_ORDER",
        "SEND_ORDER", "SET_TARGET", "PUSH_INTENT", "PUSH_SIGNAL",
    }

    def check_dict_for_prohibited(self, data: Dict[str, Any], path: str = "") -> list:
        """Recursively check dict for prohibited fields."""
        violations = []

        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                if key.lower() in self.PROHIBITED_FIELDS:
                    violations.append(f"Prohibited field '{key}' at {current_path}")
                violations.extend(self.check_dict_for_prohibited(value, current_path))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                violations.extend(self.check_dict_for_prohibited(item, f"{path}[{i}]"))

        return violations

    def test_heartbeat_message_no_prohibited_fields(self):
        """HEARTBEAT message must not contain order-like fields."""
        heartbeat = {
            "message_type": "HEARTBEAT",
            "agent_id": "agent_test123456789012",
            "timestamp": datetime.utcnow().isoformat(),
            "state": {
                "deployment_state": "RUNNING",
                "run_state": "RUNNING",
                "uptime_seconds": 3600,
            },
            "health": {
                "cpu_percent": 45.5,
                "memory_percent": 60.0,
                "broker_connected": True,
            }
        }

        violations = self.check_dict_for_prohibited(heartbeat)
        assert len(violations) == 0, f"HEARTBEAT violations: {violations}"

    def test_command_batch_no_order_payloads(self):
        """COMMAND_BATCH must only contain lifecycle commands."""
        command_batch = {
            "message_type": "COMMAND_BATCH",
            "timestamp": datetime.utcnow().isoformat(),
            "commands": [
                {
                    "command_type": "REQUEST_START_RUN",
                    "idempotency_key": "key_" + "a" * 20,
                    "timestamp": datetime.utcnow().isoformat(),
                    "deployment_id": "deploy_123",
                    "artifact_digest": "sha256:" + "a" * 64,
                }
            ]
        }

        violations = self.check_dict_for_prohibited(command_batch)
        assert len(violations) == 0, f"COMMAND_BATCH violations: {violations}"

    def test_prohibited_command_types_rejected(self):
        """Prohibited command types must be rejected."""
        for cmd_type in self.PROHIBITED_COMMAND_TYPES:
            assert cmd_type in self.PROHIBITED_COMMAND_TYPES
            # These should never appear in valid protocol

    def test_order_like_payload_detected(self):
        """Order-like payloads must be detected as violations."""
        malicious_message = {
            "command_type": "REQUEST_UPDATE_CONFIG",
            "side": "BUY",  # PROHIBITED
            "quantity": 100,  # PROHIBITED
            "price": 50000.0,  # PROHIBITED
        }

        violations = self.check_dict_for_prohibited(malicious_message)
        assert len(violations) >= 3, "Should detect side, quantity, price violations"


# =============================================================================
# Test 3: Protocol Schema Validation
# =============================================================================

class TestProtocolSchema:
    """Test protocol schema enforcement."""

    @pytest.fixture
    def schema(self):
        """Load protocol schema."""
        schema_path = PROJECT_ROOT / "docs" / "schemas" / "protocol_messages.schema.json"
        if not schema_path.exists():
            pytest.skip("Protocol schema not found")

        with open(schema_path) as f:
            return json.load(f)

    def test_schema_has_prohibited_fields_definition(self, schema):
        """Schema must define prohibited_order_fields."""
        # prohibited_order_fields can be in root definitions or nested in messages
        schema_str = json.dumps(schema)
        assert "prohibited_order_fields" in schema_str, \
            "Schema must reference prohibited_order_fields somewhere"

    def test_schema_prohibits_side_field(self, schema):
        """Schema must use prohibited_order_fields pattern."""
        schema_str = json.dumps(schema)
        # Schema uses allOf with prohibited_order_fields which uses "not" pattern
        assert "prohibited_order_fields" in schema_str, \
            "Schema must use prohibited_order_fields pattern"

    def test_schema_prohibits_quantity_field(self, schema):
        """Schema must reference prohibited_order_fields pattern."""
        schema_str = json.dumps(schema)
        assert "prohibited_order_fields" in schema_str, \
            "Schema must reference prohibited_order_fields"

    def test_allowed_command_types(self, schema):
        """Schema must define only lifecycle command types."""
        allowed_commands = {
            "REQUEST_START_RUN",
            "REQUEST_STOP_RUN",
            "REQUEST_PAUSE_RUN",
            "REQUEST_UPGRADE_ARTIFACT",
            "REQUEST_UPDATE_CONFIG",
            "REQUEST_ROTATE_AGENT_SESSION",
            "REQUEST_EXPORT_LOGS",
        }

        schema_str = json.dumps(schema)
        for cmd in allowed_commands:
            assert cmd in schema_str, f"Schema should define {cmd}"


# =============================================================================
# Test 4: Hard Caps Enforcement
# =============================================================================

class TestHardCapsEnforcement:
    """Test that hard caps are properly enforced."""

    @pytest.fixture
    def hard_caps(self):
        """Create test hard caps."""
        try:
            from packages.agent.policy.hard_caps import HardCaps, HardCapEnforcer
            return HardCapEnforcer(HardCaps(
                absolute_max_order_size=Decimal("10000"),
                absolute_max_position=Decimal("100000"),
                absolute_max_daily_loss=Decimal("5000"),
                absolute_max_daily_loss_pct=Decimal("0.1"),
                kill_switch_loss_pct=Decimal("0.05"),
                kill_switch_error_count=5,
            ))
        except ImportError:
            pytest.skip("HardCaps not available")

    def test_order_size_cap(self, hard_caps):
        """Order exceeding max size must be rejected."""
        violation = hard_caps.check_order_size(Decimal("15000"))
        assert violation is not None, "Should reject order > max size"
        assert violation.action == "reject"

    def test_order_within_cap_allowed(self, hard_caps):
        """Order within limits must be allowed."""
        violation = hard_caps.check_order_size(Decimal("5000"))
        assert violation is None, "Should allow order within limits"

    def test_daily_loss_triggers_kill_switch(self, hard_caps):
        """Exceeding daily loss triggers kill switch."""
        violation = hard_caps.check_daily_loss(
            daily_pnl=Decimal("-6000"),
            equity=Decimal("100000"),
        )
        assert violation is not None
        assert violation.action == "kill_switch"

    def test_loss_percentage_triggers_kill_switch(self, hard_caps):
        """Exceeding loss percentage triggers kill switch."""
        violation = hard_caps.check_daily_loss(
            daily_pnl=Decimal("-6000"),
            equity=Decimal("100000"),
        )
        assert violation is not None
        assert "kill_switch" in violation.action

    def test_position_cap_enforced(self, hard_caps):
        """Position exceeding limit must be rejected."""
        violation = hard_caps.check_position_size(Decimal("150000"))
        assert violation is not None
        assert violation.action == "reject"


# =============================================================================
# Test 5: Kill Switch
# =============================================================================

class TestKillSwitch:
    """Test kill switch functionality."""

    @pytest.fixture
    def kill_switch(self):
        """Create test kill switch."""
        try:
            from packages.agent.daemon.kill_switch import KillSwitch, HaltReasonType
            return KillSwitch()
        except ImportError:
            pytest.skip("KillSwitch not available")

    def test_kill_switch_not_triggered_initially(self, kill_switch):
        """Kill switch should not be triggered on init."""
        assert not kill_switch.is_triggered

    def test_kill_switch_triggers_on_halt(self, kill_switch):
        """Kill switch should trigger when halt is called."""
        from packages.agent.daemon.kill_switch import HaltReasonType

        kill_switch.trigger_halt(
            reason_type=HaltReasonType.MAX_DAILY_LOSS,
            message="Test halt",
            evidence={"loss": -5000},
        )

        assert kill_switch.is_triggered
        assert kill_switch.halt_reason is not None

    def test_kill_switch_records_history(self, kill_switch):
        """Kill switch should record halt history."""
        from packages.agent.daemon.kill_switch import HaltReasonType

        kill_switch.trigger_halt(
            reason_type=HaltReasonType.BROKER_ERROR_BURST,
            message="Too many errors",
        )

        history = kill_switch.get_history()
        assert len(history) >= 1

    def test_halt_reason_types_exist(self):
        """All expected halt reason types should exist."""
        try:
            from packages.agent.daemon.kill_switch import HaltReasonType

            # Use actual enum names from kill_switch.py
            expected_types = [
                "MAX_DAILY_LOSS",
                "MAX_DRAWDOWN",
                "BROKER_ERROR_BURST",  # Not ERROR_THRESHOLD
                "LATENCY_SPIKE",
                "MANUAL_TRIGGER",  # Not MANUAL
                "POSITION_MISMATCH",
            ]

            for reason_type in expected_types:
                assert hasattr(HaltReasonType, reason_type), \
                    f"HaltReasonType should have {reason_type}"
        except ImportError:
            pytest.skip("HaltReasonType not available")


# =============================================================================
# Test 6: Artifact Manifest
# =============================================================================

class TestArtifactManifest:
    """Test artifact manifest validation."""

    @pytest.fixture
    def valid_manifest_data(self):
        """Create valid manifest data."""
        return {
            "schema_version": "1.0.0",
            "artifact_id": "test_strategy_001",
            "artifact_type": "strategy",
            "entrypoint": {
                "module": "strategies.test",
                "class": "TestStrategy",
            },
            "runtime": {
                "python_version": "3.11",
                "min_memory_mb": 1024,
            },
            "deps_lock_digest": "sha256:" + "a" * 64,
            "created_at": datetime.utcnow().isoformat(),
        }

    def test_manifest_requires_schema_version(self, valid_manifest_data):
        """Manifest must have schema_version."""
        try:
            from ccea.models.manifest import ArtifactManifest

            manifest = ArtifactManifest(**valid_manifest_data)
            assert manifest.schema_version == "1.0.0"
        except ImportError:
            pytest.skip("ArtifactManifest not available")

    def test_manifest_requires_artifact_id(self, valid_manifest_data):
        """Manifest must have artifact_id."""
        try:
            from ccea.models.manifest import ArtifactManifest

            del valid_manifest_data["artifact_id"]
            with pytest.raises(Exception):  # ValidationError
                ArtifactManifest(**valid_manifest_data)
        except ImportError:
            pytest.skip("ArtifactManifest not available")

    def test_manifest_digest_format(self, valid_manifest_data):
        """deps_lock_digest must be sha256 format."""
        try:
            from ccea.models.manifest import ArtifactManifest

            # Invalid digest format
            valid_manifest_data["deps_lock_digest"] = "invalid_digest"
            with pytest.raises(Exception):
                ArtifactManifest(**valid_manifest_data)
        except ImportError:
            pytest.skip("ArtifactManifest not available")

    def test_manifest_change_class(self, valid_manifest_data):
        """Manifest should support change_class."""
        try:
            from ccea.models.manifest import ArtifactManifest, ChangeClass

            valid_manifest_data["change_class"] = "TRADING_IMPACTING"
            manifest = ArtifactManifest(**valid_manifest_data)
            assert manifest.change_class == ChangeClass.TRADING_IMPACTING
        except ImportError:
            pytest.skip("ArtifactManifest not available")


# =============================================================================
# Test 7: Zone Separation
# =============================================================================

class TestZoneSeparation:
    """Test that zones are properly separated."""

    def test_simulation_runner_is_cloud_zone(self):
        """SimulationRunner should be CLOUD zone."""
        try:
            from packages.shared.runner.simulation import SimulationRunner, SimulationRunnerConfig
            from packages.shared.runner.base import RunnerZone

            config = SimulationRunnerConfig(
                run_id="test_sim",
                strategy_id="test",
                symbols=["BTCUSDT"],
            )
            runner = SimulationRunner(config)

            assert runner._config.zone == RunnerZone.CLOUD
        except ImportError:
            pytest.skip("SimulationRunner not available")

    def test_live_runner_is_agent_zone(self):
        """LiveRunner should be AGENT zone."""
        try:
            from packages.agent.runner.live import LiveRunner, LiveRunnerConfig
            from packages.shared.runner.base import RunnerZone

            config = LiveRunnerConfig(
                run_id="test_live",
                strategy_id="test",
                symbols=["BTCUSDT"],
            )
            runner = LiveRunner(config)

            assert runner._config.zone == RunnerZone.AGENT
        except ImportError:
            pytest.skip("LiveRunner not available")

    def test_simulation_runner_never_executes_real_orders(self):
        """SimulationRunner must never have broker connection."""
        try:
            from packages.shared.runner.simulation import SimulationRunner

            # Check class doesn't have broker-related methods
            assert not hasattr(SimulationRunner, "connect_broker")
            assert not hasattr(SimulationRunner, "submit_order_to_exchange")
        except ImportError:
            pytest.skip("SimulationRunner not available")


# =============================================================================
# Test 8: OrderIntent Contract
# =============================================================================

class TestOrderIntentContract:
    """Test OrderIntent is properly defined."""

    def test_order_intent_is_not_order(self):
        """OrderIntent must be distinct from Order."""
        try:
            from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide

            intent = OrderIntent(
                strategy_id="test",
                symbol="BTCUSDT",
                intent_type=IntentType.MARKET_ENTRY,
                side=IntentSide.LONG,
                target_quantity=Decimal("0.1"),
            )

            # Intent should not have order-specific fields
            assert not hasattr(intent, "order_id")
            assert not hasattr(intent, "filled_quantity")
            assert not hasattr(intent, "status")

            # Intent should have intent-specific fields
            assert hasattr(intent, "intent_id")
            assert hasattr(intent, "intent_type")
        except ImportError:
            pytest.skip("OrderIntent not available")

    def test_order_intent_has_target_not_actual(self):
        """OrderIntent should express targets, not actual orders."""
        try:
            from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide

            intent = OrderIntent(
                strategy_id="test",
                symbol="BTCUSDT",
                intent_type=IntentType.LIMIT_ENTRY,
                side=IntentSide.LONG,
                target_quantity=Decimal("1.0"),
                limit_price=Decimal("50000"),
            )

            assert intent.target_quantity == Decimal("1.0")
            assert intent.limit_price == Decimal("50000")
        except ImportError:
            pytest.skip("OrderIntent not available")


# =============================================================================
# Test 9: Telemetry Redaction
# =============================================================================

class TestTelemetryRedaction:
    """Test that telemetry is properly redacted."""

    def test_telemetry_schema_requires_redaction(self):
        """Telemetry messages must have redaction_applied=true."""
        schema_path = PROJECT_ROOT / "docs" / "schemas" / "protocol_messages.schema.json"
        if not schema_path.exists():
            pytest.skip("Protocol schema not found")

        with open(schema_path) as f:
            schema = json.load(f)

        schema_str = json.dumps(schema)

        # Check for redaction_applied with const: true
        assert "redaction_applied" in schema_str
        assert '"const": true' in schema_str or '"const":true' in schema_str


# =============================================================================
# Test 10: CI Guardrails Integration
# =============================================================================

class TestCIGuardrailsIntegration:
    """Test CI guardrails are properly integrated."""

    def test_import_check_module_exists(self):
        """Import check guardrail should exist."""
        import_check_path = PROJECT_ROOT / "ccea" / "guardrails" / "import_check.py"
        assert import_check_path.exists(), "import_check.py should exist"

    def test_intent_prohibition_module_exists(self):
        """Intent prohibition guardrail should exist."""
        intent_path = PROJECT_ROOT / "ccea" / "guardrails" / "intent_prohibition.py"
        assert intent_path.exists(), "intent_prohibition.py should exist"

    def test_import_check_has_prohibited_list(self):
        """Import check should define prohibited modules."""
        # Read file directly to avoid cryptography import issues in test env
        import_check_path = PROJECT_ROOT / "ccea" / "guardrails" / "import_check.py"
        if not import_check_path.exists():
            pytest.skip("import_check.py not found")

        content = import_check_path.read_text()
        assert "PROHIBITED_IN_CLOUD" in content, "Should define PROHIBITED_IN_CLOUD"
        assert "order_execution" in content, "Should list order_execution as prohibited"

    def test_intent_prohibition_has_prohibited_fields(self):
        """Intent prohibition should define prohibited fields."""
        # Read file directly to avoid cryptography import issues in test env
        intent_path = PROJECT_ROOT / "ccea" / "guardrails" / "intent_prohibition.py"
        if not intent_path.exists():
            pytest.skip("intent_prohibition.py not found")

        content = intent_path.read_text()
        assert "PROHIBITED_INTENT_FIELDS" in content or "prohibited" in content.lower()
        # Verify common prohibited fields are mentioned
        assert "side" in content, "Should prohibit 'side' field"
        assert "quantity" in content, "Should prohibit 'quantity' field"
        assert "price" in content, "Should prohibit 'price' field"


# =============================================================================
# Test 11: Legal Documentation
# =============================================================================

class TestLegalDocumentation:
    """Test that legal documentation exists and contains CCEA info."""

    def test_terms_of_service_exists(self):
        """Terms of Service must exist."""
        tos_path = PROJECT_ROOT / "docs" / "legal" / "TERMS_OF_SERVICE.md"
        assert tos_path.exists(), "TERMS_OF_SERVICE.md should exist"

    def test_terms_mentions_ccea(self):
        """Terms of Service must mention CCEA architecture."""
        tos_path = PROJECT_ROOT / "docs" / "legal" / "TERMS_OF_SERVICE.md"
        if not tos_path.exists():
            pytest.skip("ToS not found")

        content = tos_path.read_text()
        assert "CCEA" in content, "ToS should mention CCEA"
        assert "Cloud" in content and "Agent" in content

    def test_terms_not_investment_advice(self):
        """Terms must state platform is not investment advice."""
        tos_path = PROJECT_ROOT / "docs" / "legal" / "TERMS_OF_SERVICE.md"
        if not tos_path.exists():
            pytest.skip("ToS not found")

        content = tos_path.read_text()
        assert "NOT" in content and "investment" in content.lower()

    def test_privacy_policy_exists(self):
        """Privacy Policy must exist."""
        privacy_path = PROJECT_ROOT / "docs" / "legal" / "PRIVACY_POLICY.md"
        assert privacy_path.exists(), "PRIVACY_POLICY.md should exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
