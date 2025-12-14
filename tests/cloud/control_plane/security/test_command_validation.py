# -*- coding: utf-8 -*-
"""
Tests for Command Type Validation - WI-CLOUD-01.

Tests verify:
- Fail-closed validation: unknown command types are rejected
- Enum allowlist enforcement
- Normalization (uppercase)
- Metadata retrieval for valid commands
"""

import pytest

from packages.cloud.control_plane.security.command_validation import (
    CommandTypeValidator,
    CommandValidationResult,
    ValidationSeverity,
    validate_command_type,
    is_valid_command_type,
    assert_valid_command_type,
    ALLOWED_COMMAND_TYPES,
)


class TestCommandTypeValidator:
    """Test CommandTypeValidator class."""

    def test_allowed_command_types_not_empty(self):
        """Verify allowed command types are defined."""
        assert len(ALLOWED_COMMAND_TYPES) > 0
        assert "REQUEST_START_RUN" in ALLOWED_COMMAND_TYPES
        assert "REQUEST_STOP_RUN" in ALLOWED_COMMAND_TYPES

    def test_validate_valid_command_type(self):
        """Valid command types should pass validation."""
        validator = CommandTypeValidator()

        for cmd_type in ALLOWED_COMMAND_TYPES:
            result = validator.validate(cmd_type)
            assert result.valid, f"Expected {cmd_type} to be valid"
            assert result.command_type == cmd_type
            assert len(result.errors) == 0

    def test_validate_valid_command_type_lowercase(self):
        """Lowercase command types should be normalized and accepted."""
        validator = CommandTypeValidator()

        result = validator.validate("request_start_run")
        assert result.valid
        assert result.command_type == "REQUEST_START_RUN"

    def test_validate_valid_command_type_mixed_case(self):
        """Mixed case command types should be normalized and accepted."""
        validator = CommandTypeValidator()

        result = validator.validate("Request_Stop_Run")
        assert result.valid
        assert result.command_type == "REQUEST_STOP_RUN"

    def test_validate_unknown_command_type_rejected(self):
        """Unknown command types MUST be rejected (fail-closed)."""
        validator = CommandTypeValidator()

        unknown_types = [
            "EXECUTE_ORDER",
            "SEND_ORDER",
            "PLACE_TRADE",
            "BUY",
            "SELL",
            "MARKET_ORDER",
            "RANDOM_COMMAND",
            "NOT_A_REAL_COMMAND",
        ]

        for cmd_type in unknown_types:
            result = validator.validate(cmd_type)
            assert not result.valid, f"Expected {cmd_type} to be rejected"
            assert len(result.errors) > 0
            assert result.errors[0].severity == ValidationSeverity.CRITICAL

    def test_validate_empty_command_type_rejected(self):
        """Empty command types should be rejected."""
        validator = CommandTypeValidator()

        result = validator.validate("")
        assert not result.valid
        assert result.errors[0].message == "Command type cannot be empty"

    def test_validate_whitespace_command_type_rejected(self):
        """Whitespace-only command types should be rejected."""
        validator = CommandTypeValidator()

        result = validator.validate("   ")
        assert not result.valid

    def test_is_valid_convenience_function(self):
        """Test is_valid() convenience method."""
        validator = CommandTypeValidator()

        assert validator.is_valid("REQUEST_START_RUN")
        assert not validator.is_valid("EXECUTE_ORDER")
        assert not validator.is_valid("")

    def test_get_metadata_for_valid_command(self):
        """Metadata should be returned for valid commands."""
        validator = CommandTypeValidator()

        metadata = validator.get_metadata("REQUEST_START_RUN")
        assert metadata is not None
        assert "requires_approval_default" in metadata
        assert "change_class" in metadata

    def test_get_metadata_for_unknown_command(self):
        """Metadata should be None for unknown commands."""
        validator = CommandTypeValidator()

        metadata = validator.get_metadata("EXECUTE_ORDER")
        assert metadata is None

    def test_requires_approval_for_trading_impacting(self):
        """Trading-impacting commands should require approval by default."""
        validator = CommandTypeValidator()

        assert validator.requires_approval("REQUEST_START_RUN")
        assert validator.requires_approval("REQUEST_UPGRADE_ARTIFACT")
        assert validator.requires_approval("REQUEST_UPDATE_CONFIG")

    def test_safety_operations_do_not_require_approval(self):
        """Safety operations (stop/pause) should not require approval."""
        validator = CommandTypeValidator()

        assert not validator.requires_approval("REQUEST_STOP_RUN")
        assert not validator.requires_approval("REQUEST_PAUSE_RUN")

    def test_is_safety_operation(self):
        """Test safety operation detection."""
        validator = CommandTypeValidator()

        assert validator.is_safety_operation("REQUEST_STOP_RUN")
        assert validator.is_safety_operation("REQUEST_PAUSE_RUN")
        assert not validator.is_safety_operation("REQUEST_START_RUN")

    def test_get_allowed_types_returns_sorted_list(self):
        """get_allowed_types should return sorted list."""
        validator = CommandTypeValidator()

        allowed = validator.get_allowed_types()
        assert isinstance(allowed, list)
        assert allowed == sorted(allowed)


class TestValidateCommandTypeFunction:
    """Test validate_command_type convenience function."""

    def test_validate_command_type_valid(self):
        """Valid command types should pass."""
        result = validate_command_type("REQUEST_START_RUN")
        assert result.valid

    def test_validate_command_type_invalid(self):
        """Invalid command types should fail."""
        result = validate_command_type("SEND_ORDER")
        assert not result.valid


class TestIsValidCommandTypeFunction:
    """Test is_valid_command_type convenience function."""

    def test_is_valid_command_type_true(self):
        """Returns True for valid types."""
        assert is_valid_command_type("REQUEST_START_RUN")

    def test_is_valid_command_type_false(self):
        """Returns False for invalid types."""
        assert not is_valid_command_type("PLACE_ORDER")


class TestAssertValidCommandType:
    """Test assert_valid_command_type function."""

    def test_assert_valid_command_type_passes(self):
        """Should return normalized type for valid commands."""
        result = assert_valid_command_type("request_start_run")
        assert result == "REQUEST_START_RUN"

    def test_assert_valid_command_type_raises(self):
        """Should raise ValueError for invalid commands."""
        with pytest.raises(ValueError) as exc_info:
            assert_valid_command_type("EXECUTE_ORDER")
        assert "Unknown command type" in str(exc_info.value)


class TestOrderLikeCommandsRejected:
    """
    Test that order-like commands are rejected.

    This is critical for CCEA security: Cloud NEVER sends order commands.
    """

    def test_order_related_commands_rejected(self):
        """Any order-related command types must be rejected."""
        validator = CommandTypeValidator()

        order_like_commands = [
            "SEND_ORDER",
            "EXECUTE_ORDER",
            "PLACE_ORDER",
            "SUBMIT_ORDER",
            "CANCEL_ORDER",
            "MODIFY_ORDER",
            "ORDER",
            "BUY",
            "SELL",
            "TRADE",
            "MARKET_BUY",
            "LIMIT_SELL",
            "MARKET_ORDER",
            "LIMIT_ORDER",
            "STOP_ORDER",
        ]

        for cmd_type in order_like_commands:
            result = validator.validate(cmd_type)
            assert not result.valid, (
                f"SECURITY VIOLATION: Order-like command '{cmd_type}' was accepted! "
                "Cloud MUST NOT send order commands."
            )
