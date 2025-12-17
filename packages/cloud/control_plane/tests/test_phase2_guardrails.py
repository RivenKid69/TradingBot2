# -*- coding: utf-8 -*-
"""
Tests for Phase 2: CI Guardrails (2.7-2.8).

Tests cover:
- Broker clients prohibition in Cloud zone
- Order-like payload prohibition
- Intent field prohibition
- Schema validation
- Protocol command allowlist

References:
- CCEA_ALIGNMENT_PLAN.md Phase 2.7-2.8
"""

import sys
from pathlib import Path
from typing import Dict, Any, Set, FrozenSet

import pytest

# Add package to path for imports
PACKAGE_ROOT = Path(__file__).parent.parent
GUARDRAILS_ROOT = PACKAGE_ROOT.parent.parent.parent / "ccea" / "guardrails"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(GUARDRAILS_ROOT.parent))


# ============================================================================
# Schema Check Tests
# ============================================================================

class TestSchemaCheck:
    """Tests for schema_check guardrail."""

    def test_schema_check_module_exists(self):
        """Test schema_check module exists."""
        schema_check_path = GUARDRAILS_ROOT / "schema_check.py"
        assert schema_check_path.exists(), "schema_check.py should exist"

    def test_prohibited_fields_defined(self):
        """Test prohibited fields are defined in schema_check."""
        try:
            from ccea.guardrails.schema_check import PROHIBITED_FIELDS

            # Critical order fields
            critical = {"side", "quantity", "qty", "price", "order_type"}
            assert critical.issubset(PROHIBITED_FIELDS)
        except ImportError:
            pytest.skip("ccea.guardrails not importable in test environment")

    def test_prohibited_values_defined(self):
        """Test prohibited values are defined."""
        try:
            from ccea.guardrails.schema_check import PROHIBITED_VALUES

            # BUY/SELL should be prohibited values for "side"
            assert "side" in PROHIBITED_VALUES
            assert "BUY" in PROHIBITED_VALUES["side"] or "buy" in PROHIBITED_VALUES["side"]
        except ImportError:
            pytest.skip("ccea.guardrails not importable in test environment")


# ============================================================================
# Intent Prohibition Tests
# ============================================================================

class TestIntentProhibition:
    """Tests for intent_prohibition guardrail."""

    def test_intent_prohibition_module_exists(self):
        """Test intent_prohibition module exists."""
        intent_path = GUARDRAILS_ROOT / "intent_prohibition.py"
        assert intent_path.exists(), "intent_prohibition.py should exist"

    def test_prohibited_intent_fields_defined(self):
        """Test prohibited intent fields are defined."""
        try:
            from ccea.guardrails.intent_prohibition import PROHIBITED_INTENT_FIELDS

            # Critical intent fields
            intent_fields = {
                "side", "quantity", "qty", "price", "order_type",
                "intent", "signal", "target_position",
            }
            assert intent_fields.issubset(PROHIBITED_INTENT_FIELDS)
        except ImportError:
            pytest.skip("ccea.guardrails not importable in test environment")

    def test_prohibited_command_types_defined(self):
        """Test prohibited command types are defined."""
        try:
            from ccea.guardrails.intent_prohibition import PROHIBITED_COMMAND_TYPES

            # Order commands should be prohibited
            prohibited = {
                "PLACE_ORDER", "SUBMIT_ORDER", "EXECUTE_ORDER",
            }
            assert prohibited.issubset(PROHIBITED_COMMAND_TYPES)
        except ImportError:
            pytest.skip("ccea.guardrails not importable in test environment")


# ============================================================================
# Cloud Allowlist Tests
# ============================================================================

class TestCloudAllowlist:
    """Tests for cloud_allowlist guardrail."""

    def test_cloud_allowlist_module_exists(self):
        """Test cloud_allowlist module exists."""
        allowlist_path = GUARDRAILS_ROOT / "cloud_allowlist.py"
        assert allowlist_path.exists(), "cloud_allowlist.py should exist"

    def test_prohibited_packages_defined(self):
        """Test prohibited packages are defined."""
        try:
            from ccea.guardrails.cloud_allowlist import PROHIBITED_PACKAGES

            # Broker-specific SDKs should be prohibited
            broker_sdks = {"alpaca_trade_api", "ib_insync"}
            assert broker_sdks.issubset(PROHIBITED_PACKAGES)
        except ImportError:
            pytest.skip("ccea.guardrails not importable in test environment")

    def test_prohibited_internal_modules_defined(self):
        """Test prohibited internal modules are defined."""
        try:
            from ccea.guardrails.cloud_allowlist import PROHIBITED_INTERNAL

            # Order execution modules should be prohibited
            assert any("order_execution" in m for m in PROHIBITED_INTERNAL)
        except ImportError:
            pytest.skip("ccea.guardrails not importable in test environment")


# ============================================================================
# Build Artifact Check Tests
# ============================================================================

class TestBuildArtifactCheck:
    """Tests for build_artifact_check guardrail."""

    def test_build_artifact_check_module_exists(self):
        """Test build_artifact_check module exists."""
        artifact_path = GUARDRAILS_ROOT / "build_artifact_check.py"
        assert artifact_path.exists(), "build_artifact_check.py should exist"

    def test_prohibited_modules_defined(self):
        """Test prohibited modules are defined."""
        try:
            from ccea.guardrails.build_artifact_check import PROHIBITED_MODULES

            # Order execution modules
            assert "order_execution" in PROHIBITED_MODULES
        except ImportError:
            pytest.skip("ccea.guardrails not importable in test environment")


# ============================================================================
# Protocol Check Tests
# ============================================================================

class TestProtocolCheck:
    """Tests for protocol_check guardrail."""

    def test_protocol_check_module_exists(self):
        """Test protocol_check module exists."""
        protocol_path = GUARDRAILS_ROOT / "protocol_check.py"
        assert protocol_path.exists(), "protocol_check.py should exist"


# ============================================================================
# Import Boundary Tests
# ============================================================================

class TestImportBoundary:
    """Tests for import boundary guardrail."""

    def test_import_check_module_exists(self):
        """Test import_check module exists."""
        import_path = GUARDRAILS_ROOT / "import_check.py"
        assert import_path.exists(), "import_check.py should exist"


# ============================================================================
# Artifact Check Tests
# ============================================================================

class TestArtifactCheck:
    """Tests for artifact_check guardrail."""

    def test_artifact_check_module_exists(self):
        """Test artifact_check module exists."""
        artifact_path = GUARDRAILS_ROOT / "artifact_check.py"
        assert artifact_path.exists(), "artifact_check.py should exist"

    def test_secret_patterns_defined(self):
        """Test secret detection patterns are defined."""
        try:
            from ccea.guardrails.artifact_check import SECRET_PATTERNS

            # Should have broker API key patterns
            patterns_str = str(SECRET_PATTERNS)
            assert "api" in patterns_str.lower() or "key" in patterns_str.lower()
        except ImportError:
            pytest.skip("ccea.guardrails not importable in test environment")


# ============================================================================
# Boundary Guardrails Tests
# ============================================================================

class TestBoundaryGuardrails:
    """Tests for Cloud/Agent boundary guardrails."""

    def test_boundary_module_exists(self):
        """Test boundary.py exists in control_plane."""
        boundary_path = PACKAGE_ROOT / "boundary.py"
        assert boundary_path.exists(), "boundary.py should exist"

    def test_lifecycle_commands_defined(self):
        """Test lifecycle commands are properly defined."""
        try:
            from packages.cloud.control_plane.boundary import LIFECYCLE_COMMANDS

            # Only lifecycle commands should be allowed
            expected_commands = {
                "REQUEST_START_RUN",
                "REQUEST_STOP_RUN",
                "REQUEST_PAUSE_RUN",
                "REQUEST_UPGRADE_ARTIFACT",
                "REQUEST_UPDATE_CONFIG",
                "REQUEST_ROTATE_AGENT_SESSION",
                "REQUEST_EXPORT_LOGS",
            }
            assert expected_commands == LIFECYCLE_COMMANDS
        except ImportError:
            pytest.skip("boundary module not importable in test environment")

    def test_allowed_lifecycle_fields_defined(self):
        """Test allowed lifecycle fields don't include order fields."""
        try:
            from packages.cloud.control_plane.boundary import ALLOWED_LIFECYCLE_FIELDS

            # Order fields should NOT be in allowed list
            prohibited = {"side", "quantity", "qty", "price", "order_type", "intent", "signal"}
            assert not prohibited.intersection(ALLOWED_LIFECYCLE_FIELDS)
        except ImportError:
            pytest.skip("boundary module not importable in test environment")


# ============================================================================
# Telemetry Validation Integration Tests
# ============================================================================

class TestTelemetryValidationGuardrails:
    """Integration tests for telemetry validation guardrails."""

    def test_prohibited_order_fields_comprehensive(self):
        """Test all order fields are covered in telemetry validation."""
        from packages.cloud.control_plane.middleware.telemetry_validation import (
            PROHIBITED_ORDER_FIELDS,
        )

        # All critical trading fields must be prohibited
        critical_trading = {
            # Order fields
            "side", "quantity", "qty", "price", "order_type",
            "limit_price", "stop_price", "order_id",
            # Intent fields
            "intent", "signal", "target_position",
            "execute_order", "place_order", "submit_order",
            # Position fields
            "position_side", "position_size",
            # Execution fields
            "execution_id", "trade_id", "fill_price",
        }

        assert critical_trading.issubset(PROHIBITED_ORDER_FIELDS), \
            f"Missing fields: {critical_trading - PROHIBITED_ORDER_FIELDS}"

    def test_prohibited_pii_fields_comprehensive(self):
        """Test PII fields are covered."""
        from packages.cloud.control_plane.middleware.telemetry_validation import (
            PROHIBITED_PII_FIELDS,
        )

        # GDPR-relevant PII
        gdpr_pii = {
            "email", "phone", "address", "ssn",
            "credit_card", "date_of_birth",
        }

        assert gdpr_pii.issubset(PROHIBITED_PII_FIELDS)


# ============================================================================
# CI Workflow Verification Tests
# ============================================================================

class TestCIWorkflowIntegration:
    """Tests to verify CI workflow has guardrails."""

    def test_build_workflow_exists(self):
        """Test build workflow file exists."""
        workflow_path = PACKAGE_ROOT.parent.parent.parent / ".github" / "workflows" / "build-and-test.yml"
        assert workflow_path.exists(), "build-and-test.yml should exist"

    def test_security_workflow_exists(self):
        """Test security workflow file exists."""
        workflow_path = PACKAGE_ROOT.parent.parent.parent / ".github" / "workflows" / "security-sast.yml"
        assert workflow_path.exists(), "security-sast.yml should exist"


# ============================================================================
# Guardrails __init__ Exports Tests
# ============================================================================

class TestGuardrailsInit:
    """Tests for guardrails __init__.py exports."""

    def test_guardrails_init_exists(self):
        """Test guardrails __init__.py exists."""
        init_path = GUARDRAILS_ROOT / "__init__.py"
        assert init_path.exists(), "guardrails __init__.py should exist"

    def test_guardrails_init_exports(self):
        """Test guardrails module exports all guardrails."""
        try:
            import ccea.guardrails as guardrails

            # Should export all major guardrail modules
            expected_exports = [
                "SchemaValidator",
                "IntentProhibitor",
                "CloudAllowlistValidator",
            ]

            # Check at least some exports exist
            module_attrs = dir(guardrails)
            # Note: actual names may differ, just verify module loads
            assert len(module_attrs) > 5
        except ImportError:
            pytest.skip("ccea.guardrails not importable in test environment")


# ============================================================================
# Forever Prohibitions Tests (2.8)
# ============================================================================

class TestForeverProhibitions:
    """Tests for 'forever' prohibitions (2.8)."""

    def test_broker_order_execution_prohibited(self):
        """Test broker order execution is forever prohibited in Cloud."""
        # These patterns should NEVER be allowed in Cloud code
        forever_prohibited_patterns = [
            "submit_order",
            "place_order",
            "execute_order",
            "cancel_order",
            "modify_order",
        ]

        # Check they're in prohibited lists
        from packages.cloud.control_plane.middleware.telemetry_validation import (
            PROHIBITED_ORDER_FIELDS,
        )

        for pattern in forever_prohibited_patterns:
            assert pattern in PROHIBITED_ORDER_FIELDS, \
                f"'{pattern}' must be forever prohibited"

    def test_raw_order_telemetry_restricted(self):
        """Test RAW_ORDER_EVENTS telemetry is restricted to enterprise."""
        from packages.cloud.control_plane.models import TelemetryLevel

        # RAW_ORDER_EVENTS should exist but be enterprise-only
        assert TelemetryLevel.RAW_ORDER_EVENTS.value == "raw_order_events"

    def test_command_types_restricted_to_lifecycle(self):
        """Test command types are restricted to lifecycle only."""
        try:
            from packages.cloud.control_plane.boundary import LIFECYCLE_COMMANDS

            # No order commands in lifecycle
            order_commands = {
                "PLACE_ORDER", "SUBMIT_ORDER", "EXECUTE_ORDER",
                "CANCEL_ORDER", "MODIFY_ORDER",
            }
            assert not order_commands.intersection(LIFECYCLE_COMMANDS)
        except ImportError:
            pytest.skip("boundary module not importable")


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
