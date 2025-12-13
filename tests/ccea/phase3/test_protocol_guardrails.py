# -*- coding: utf-8 -*-
"""
Tests for Protocol Guardrails - Intent Prohibition.

Verifies:
1. Prohibited fields are detected
2. Prohibited commands are blocked
3. Cloud boundary enforcement works
4. Runtime validation catches violations
"""

import pytest
from datetime import datetime

from ccea.guardrails.intent_prohibition import (
    PROHIBITED_INTENT_FIELDS,
    PROHIBITED_COMMAND_TYPES,
    ALLOWED_CLOUD_TO_AGENT_TYPES,
    check_dict_for_prohibited_fields,
    check_command_type,
    check_protocol_message,
    check_python_source_for_intent_injection,
    validate_cloud_api_endpoint,
    prohibit_intent_fields,
    IntentProhibitionViolation,
    IntentProhibitionResult,
    ViolationSeverity,
)
from packages.cloud.control_plane.boundary import (
    CloudBoundaryValidator,
    CloudCommandFilter,
    RuntimeBoundaryEnforcer,
    BoundaryViolationError,
    validate_cloud_message,
    assert_no_intent_fields,
    LIFECYCLE_COMMANDS,
)


class TestProhibitedFields:
    """Tests for prohibited field detection."""

    def test_prohibited_fields_constant(self):
        """Test prohibited fields are defined."""
        assert "side" in PROHIBITED_INTENT_FIELDS
        assert "quantity" in PROHIBITED_INTENT_FIELDS
        assert "price" in PROHIBITED_INTENT_FIELDS
        assert "intent" in PROHIBITED_INTENT_FIELDS
        assert "signal" in PROHIBITED_INTENT_FIELDS
        assert "target_position" in PROHIBITED_INTENT_FIELDS

    def test_check_dict_with_prohibited_field(self):
        """Test detection of prohibited field in dict."""
        data = {
            "message_type": "COMMAND",
            "side": "BUY",  # PROHIBITED
        }

        violations = check_dict_for_prohibited_fields(data)

        assert len(violations) > 0
        assert any(v.field_name == "side" for v in violations)

    def test_check_dict_with_quantity(self):
        """Test detection of quantity field."""
        data = {
            "command_type": "TEST",
            "quantity": 100,  # PROHIBITED
        }

        violations = check_dict_for_prohibited_fields(data)

        assert len(violations) > 0
        assert any(v.field_name == "quantity" for v in violations)

    def test_check_nested_prohibited_field(self):
        """Test detection of nested prohibited field."""
        data = {
            "command_type": "TEST",
            "payload": {
                "nested": {
                    "price": 50000,  # PROHIBITED - nested
                }
            }
        }

        violations = check_dict_for_prohibited_fields(data)

        assert len(violations) > 0
        assert any(v.field_name == "price" for v in violations)

    def test_check_array_with_prohibited_field(self):
        """Test detection in array items."""
        data = {
            "commands": [
                {"command_type": "GOOD"},
                {"command_type": "BAD", "side": "SELL"},  # PROHIBITED
            ]
        }

        violations = check_dict_for_prohibited_fields(data)

        assert len(violations) > 0

    def test_clean_dict_no_violations(self):
        """Test clean dict passes."""
        data = {
            "command_type": "REQUEST_START_RUN",
            "deployment_id": "dep_123",
            "artifact_digest": "sha256:abc123",
        }

        violations = check_dict_for_prohibited_fields(data)

        assert len(violations) == 0

    def test_case_insensitive_field_check(self):
        """Test case insensitivity."""
        data = {
            "SIDE": "BUY",  # Uppercase
        }

        violations = check_dict_for_prohibited_fields(data)

        # Should detect regardless of case
        assert len(violations) > 0


class TestProhibitedCommands:
    """Tests for prohibited command detection."""

    def test_prohibited_command_types(self):
        """Test prohibited commands are defined."""
        assert "PLACE_ORDER" in PROHIBITED_COMMAND_TYPES
        assert "SUBMIT_ORDER" in PROHIBITED_COMMAND_TYPES
        assert "EXECUTE_ORDER" in PROHIBITED_COMMAND_TYPES
        assert "SET_TARGET" in PROHIBITED_COMMAND_TYPES

    def test_check_prohibited_command(self):
        """Test detection of prohibited command."""
        violation = check_command_type("PLACE_ORDER")

        assert violation is not None
        assert violation.severity == ViolationSeverity.CRITICAL

    def test_check_allowed_command(self):
        """Test allowed command passes."""
        violation = check_command_type("REQUEST_START_RUN")

        assert violation is None

    def test_check_order_pattern_command(self):
        """Test detection of order-like pattern."""
        violation = check_command_type("EXECUTE_TRADE")

        assert violation is not None  # Contains "EXECUTE"

    def test_allowed_cloud_commands(self):
        """Test lifecycle commands are allowed."""
        for cmd in LIFECYCLE_COMMANDS:
            violation = check_command_type(cmd)
            assert violation is None, f"{cmd} should be allowed"


class TestProtocolMessageCheck:
    """Tests for complete protocol message checking."""

    def test_valid_lifecycle_message(self):
        """Test valid lifecycle message passes."""
        message = {
            "message_type": "COMMAND_BATCH",
            "commands": [
                {
                    "command_type": "REQUEST_START_RUN",
                    "idempotency_key": "key_12345678901234567890",
                    "deployment_id": "dep_001",
                    "artifact_digest": "sha256:abc",
                }
            ]
        }

        result = check_protocol_message(message, "test_message")

        assert result.passed is True

    def test_message_with_prohibited_command(self):
        """Test message with prohibited command fails."""
        message = {
            "command_type": "PLACE_ORDER",
            "symbol": "BTCUSDT",
        }

        result = check_protocol_message(message, "test_message")

        assert result.passed is False
        assert result.critical_count > 0

    def test_message_with_prohibited_fields(self):
        """Test message with prohibited fields fails."""
        message = {
            "command_type": "REQUEST_START_RUN",
            "side": "BUY",  # PROHIBITED
            "quantity": 100,  # PROHIBITED
        }

        result = check_protocol_message(message, "test_message")

        assert result.passed is False


class TestCloudBoundaryValidator:
    """Tests for CloudBoundaryValidator."""

    def test_valid_message_passes(self):
        """Test valid message passes validation."""
        validator = CloudBoundaryValidator()

        message = {
            "command_type": "REQUEST_START_RUN",
            "deployment_id": "dep_001",
            "artifact_digest": "sha256:abc123",
            "requires_approval": True,
        }

        result = validator.validate_outgoing_message(message)

        assert result.valid is True
        assert result.blocked is False

    def test_prohibited_command_blocked(self):
        """Test prohibited command is blocked."""
        validator = CloudBoundaryValidator()

        message = {
            "command_type": "SUBMIT_ORDER",
            "symbol": "BTC",
        }

        result = validator.validate_outgoing_message(message)

        assert result.valid is False
        assert result.blocked is True

    def test_prohibited_field_blocked(self):
        """Test prohibited field is blocked."""
        validator = CloudBoundaryValidator()

        message = {
            "command_type": "REQUEST_START_RUN",
            "side": "BUY",
        }

        result = validator.validate_outgoing_message(message)

        assert result.valid is False

    def test_command_batch_validation(self):
        """Test command batch validation."""
        validator = CloudBoundaryValidator()

        batch = {
            "message_type": "COMMAND_BATCH",
            "commands": [
                {"command_type": "REQUEST_START_RUN", "deployment_id": "d1"},
                {"command_type": "REQUEST_STOP_RUN", "deployment_id": "d2"},
            ]
        }

        result = validator.validate_command_batch(batch)

        assert result.valid is True

    def test_invalid_command_in_batch(self):
        """Test invalid command in batch fails."""
        validator = CloudBoundaryValidator()

        batch = {
            "message_type": "COMMAND_BATCH",
            "commands": [
                {"command_type": "REQUEST_START_RUN"},
                {"command_type": "EXECUTE_ORDER", "side": "BUY"},  # Invalid
            ]
        }

        result = validator.validate_command_batch(batch)

        assert result.valid is False

    def test_strict_mode_unknown_fields(self):
        """Test strict mode catches unknown fields."""
        validator = CloudBoundaryValidator(strict_mode=True)

        message = {
            "command_type": "REQUEST_START_RUN",
            "deployment_id": "d1",
            "unknown_field": "value",  # Unknown
        }

        result = validator.validate_outgoing_message(message)

        # Should have warning for unknown field
        assert len(result.violations) > 0


class TestCloudCommandFilter:
    """Tests for CloudCommandFilter."""

    def test_sanitize_removes_prohibited(self):
        """Test sanitization removes prohibited fields."""
        dirty = {
            "command_type": "REQUEST_START_RUN",
            "deployment_id": "d1",
            "side": "BUY",  # Should be removed
            "quantity": 100,  # Should be removed
        }

        clean = CloudCommandFilter.sanitize_command(dirty)

        assert "side" not in clean
        assert "quantity" not in clean
        assert clean["command_type"] == "REQUEST_START_RUN"
        assert clean["deployment_id"] == "d1"

    def test_sanitize_nested(self):
        """Test sanitization of nested fields."""
        dirty = {
            "command_type": "REQUEST_START_RUN",
            "payload": {
                "data": {
                    "price": 50000,  # Should be removed
                },
                "good_field": "value",
            }
        }

        clean = CloudCommandFilter.sanitize_command(dirty)

        assert "price" not in clean.get("payload", {}).get("data", {})

    def test_is_lifecycle_command(self):
        """Test lifecycle command check."""
        assert CloudCommandFilter.is_lifecycle_command("REQUEST_START_RUN") is True
        assert CloudCommandFilter.is_lifecycle_command("REQUEST_STOP_RUN") is True
        assert CloudCommandFilter.is_lifecycle_command("PLACE_ORDER") is False


class TestRuntimeBoundaryEnforcer:
    """Tests for RuntimeBoundaryEnforcer."""

    def test_enforce_decorator_passes_valid(self):
        """Test decorator passes valid calls."""
        enforcer = RuntimeBoundaryEnforcer()

        @enforcer.enforce
        def send_command(cmd: dict):
            return "sent"

        result = send_command({
            "command_type": "REQUEST_START_RUN",
            "deployment_id": "d1",
        })

        assert result == "sent"
        assert enforcer.stats["passed"] == 1

    def test_enforce_decorator_blocks_invalid(self):
        """Test decorator blocks invalid calls."""
        enforcer = RuntimeBoundaryEnforcer()

        @enforcer.enforce
        def send_command(cmd: dict):
            return "sent"

        with pytest.raises(BoundaryViolationError):
            send_command({
                "command_type": "REQUEST_START_RUN",
                "side": "BUY",  # Invalid
            })

        assert enforcer.stats["blocked"] == 1

    def test_validate_and_send_passes_valid(self):
        """Test validate_and_send with valid message."""
        enforcer = RuntimeBoundaryEnforcer()
        sent_messages = []

        def mock_send(msg):
            sent_messages.append(msg)
            return True

        enforcer.validate_and_send(
            {"command_type": "REQUEST_START_RUN"},
            mock_send,
        )

        assert len(sent_messages) == 1

    def test_validate_and_send_blocks_invalid(self):
        """Test validate_and_send blocks invalid message."""
        enforcer = RuntimeBoundaryEnforcer()
        sent_messages = []

        def mock_send(msg):
            sent_messages.append(msg)
            return True

        with pytest.raises(BoundaryViolationError):
            enforcer.validate_and_send(
                {"command_type": "PLACE_ORDER"},
                mock_send,
            )

        assert len(sent_messages) == 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_validate_cloud_message_valid(self):
        """Test validate_cloud_message with valid data."""
        result = validate_cloud_message({
            "command_type": "REQUEST_START_RUN",
            "deployment_id": "d1",
        })

        assert result is True

    def test_validate_cloud_message_invalid(self):
        """Test validate_cloud_message with invalid data."""
        result = validate_cloud_message({
            "command_type": "REQUEST_START_RUN",
            "side": "BUY",
        })

        assert result is False

    def test_assert_no_intent_fields_clean(self):
        """Test assert_no_intent_fields with clean data."""
        # Should not raise
        assert_no_intent_fields({
            "command_type": "REQUEST_START_RUN",
            "deployment_id": "d1",
        })

    def test_assert_no_intent_fields_dirty(self):
        """Test assert_no_intent_fields with dirty data."""
        with pytest.raises(BoundaryViolationError):
            assert_no_intent_fields({
                "command_type": "REQUEST_START_RUN",
                "intent": {"type": "market"},
            })


class TestSourceCodeCheck:
    """Tests for Python source code checking."""

    def test_clean_source_passes(self):
        """Test clean source passes."""
        source = '''
def send_command(cmd):
    return {"command_type": "REQUEST_START_RUN"}
'''
        result = check_python_source_for_intent_injection(source, "test.py")
        assert result.passed is True

    def test_dict_with_prohibited_key_detected(self):
        """Test dict literal with prohibited key is detected."""
        source = '''
def bad_function():
    return {"side": "BUY", "quantity": 100}
'''
        result = check_python_source_for_intent_injection(source, "test.py")
        assert len(result.violations) > 0

    def test_subscript_access_detected(self):
        """Test subscript access to prohibited field detected."""
        source = '''
def access_side(data):
    return data["side"]
'''
        result = check_python_source_for_intent_injection(source, "test.py")
        assert len(result.violations) > 0


class TestProhibitIntentFieldsDecorator:
    """Tests for prohibit_intent_fields decorator."""

    def test_decorator_passes_clean(self):
        """Test decorator passes clean data."""

        @prohibit_intent_fields
        def process(data: dict):
            return "processed"

        result = process({"command_type": "REQUEST_START_RUN"})
        assert result == "processed"

    def test_decorator_blocks_dirty(self):
        """Test decorator blocks dirty data."""

        @prohibit_intent_fields
        def process(data: dict):
            return "processed"

        with pytest.raises(ValueError):
            process({"side": "BUY"})

    def test_decorator_with_kwargs(self):
        """Test decorator checks kwargs."""

        @prohibit_intent_fields
        def process(command=None):
            return "processed"

        with pytest.raises(ValueError):
            process(command={"quantity": 100})
