# -*- coding: utf-8 -*-
"""
Tests for CCEA Protocol Validation.

Tests verify that:
1. Command types are validated against allowlist
2. New commands require security review
3. Order-like commands are prohibited
"""

import json
import tempfile
from pathlib import Path

import pytest

from ccea.guardrails.protocol_check import (
    ALLOWED_COMMAND_TYPES,
    ALLOWED_MESSAGE_TYPES,
    APPROVAL_REQUIRED_COMMANDS,
    SAFETY_COMMANDS,
    check_protocol_changes,
    extract_command_types,
    extract_message_types,
    validate_command_type,
    validate_message_type,
    validate_protocol_allowlist,
)


class TestAllowedCommandTypes:
    """Tests for allowed command types."""

    def test_request_start_run_allowed(self):
        """REQUEST_START_RUN should be allowed."""
        assert "REQUEST_START_RUN" in ALLOWED_COMMAND_TYPES

    def test_request_stop_run_allowed(self):
        """REQUEST_STOP_RUN should be allowed."""
        assert "REQUEST_STOP_RUN" in ALLOWED_COMMAND_TYPES

    def test_request_pause_run_allowed(self):
        """REQUEST_PAUSE_RUN should be allowed."""
        assert "REQUEST_PAUSE_RUN" in ALLOWED_COMMAND_TYPES

    def test_request_upgrade_artifact_allowed(self):
        """REQUEST_UPGRADE_ARTIFACT should be allowed."""
        assert "REQUEST_UPGRADE_ARTIFACT" in ALLOWED_COMMAND_TYPES

    def test_request_update_config_allowed(self):
        """REQUEST_UPDATE_CONFIG should be allowed."""
        assert "REQUEST_UPDATE_CONFIG" in ALLOWED_COMMAND_TYPES

    def test_request_rotate_agent_session_allowed(self):
        """REQUEST_ROTATE_AGENT_SESSION should be allowed."""
        assert "REQUEST_ROTATE_AGENT_SESSION" in ALLOWED_COMMAND_TYPES

    def test_request_export_logs_allowed(self):
        """REQUEST_EXPORT_LOGS should be allowed."""
        assert "REQUEST_EXPORT_LOGS" in ALLOWED_COMMAND_TYPES

    def test_no_order_commands_in_allowlist(self):
        """No order-related commands should be in allowlist."""
        prohibited = ["PLACE_ORDER", "SUBMIT_ORDER", "EXECUTE_ORDER", "SEND_ORDER"]
        for cmd in prohibited:
            assert cmd not in ALLOWED_COMMAND_TYPES


class TestAllowedMessageTypes:
    """Tests for allowed message types."""

    def test_heartbeat_allowed(self):
        """HEARTBEAT should be allowed."""
        assert "HEARTBEAT" in ALLOWED_MESSAGE_TYPES

    def test_telemetry_allowed(self):
        """TELEMETRY should be allowed."""
        assert "TELEMETRY" in ALLOWED_MESSAGE_TYPES

    def test_command_ack_allowed(self):
        """COMMAND_ACK should be allowed."""
        assert "COMMAND_ACK" in ALLOWED_MESSAGE_TYPES


class TestApprovalRequirements:
    """Tests for approval requirements."""

    def test_start_requires_approval(self):
        """REQUEST_START_RUN should require approval."""
        assert "REQUEST_START_RUN" in APPROVAL_REQUIRED_COMMANDS

    def test_upgrade_requires_approval(self):
        """REQUEST_UPGRADE_ARTIFACT should require approval."""
        assert "REQUEST_UPGRADE_ARTIFACT" in APPROVAL_REQUIRED_COMMANDS

    def test_stop_is_safety(self):
        """REQUEST_STOP_RUN should be a safety command (no approval)."""
        assert "REQUEST_STOP_RUN" in SAFETY_COMMANDS

    def test_pause_is_safety(self):
        """REQUEST_PAUSE_RUN should be a safety command (no approval)."""
        assert "REQUEST_PAUSE_RUN" in SAFETY_COMMANDS


class TestValidateCommandType:
    """Tests for validate_command_type function."""

    def test_allowed_command_valid(self):
        """Allowed command types should be valid."""
        is_valid, reason = validate_command_type("REQUEST_START_RUN")
        assert is_valid
        assert reason == ""

    def test_unknown_command_invalid(self):
        """Unknown command types should be invalid."""
        is_valid, reason = validate_command_type("REQUEST_UNKNOWN")
        assert not is_valid
        assert "security review" in reason.lower()

    def test_place_order_prohibited(self):
        """PLACE_ORDER should be explicitly prohibited."""
        is_valid, reason = validate_command_type("PLACE_ORDER")
        assert not is_valid
        assert "PROHIBITED" in reason

    def test_submit_order_prohibited(self):
        """SUBMIT_ORDER should be explicitly prohibited."""
        is_valid, reason = validate_command_type("SUBMIT_ORDER")
        assert not is_valid
        assert "PROHIBITED" in reason

    def test_execute_signal_prohibited(self):
        """Commands containing EXECUTE should be prohibited."""
        is_valid, reason = validate_command_type("EXECUTE_SIGNAL")
        assert not is_valid
        assert "PROHIBITED" in reason

    def test_set_target_prohibited(self):
        """SET_TARGET commands should be prohibited."""
        is_valid, reason = validate_command_type("SET_TARGET_POSITION")
        assert not is_valid
        assert "PROHIBITED" in reason


class TestValidateMessageType:
    """Tests for validate_message_type function."""

    def test_heartbeat_valid(self):
        """HEARTBEAT should be valid."""
        is_valid, reason = validate_message_type("HEARTBEAT")
        assert is_valid

    def test_telemetry_valid(self):
        """TELEMETRY should be valid."""
        is_valid, reason = validate_message_type("TELEMETRY")
        assert is_valid

    def test_command_types_also_valid_as_messages(self):
        """Command types should also be valid as messages."""
        is_valid, reason = validate_message_type("REQUEST_START_RUN")
        assert is_valid

    def test_unknown_message_invalid(self):
        """Unknown message types should be invalid."""
        is_valid, reason = validate_message_type("UNKNOWN_MESSAGE")
        assert not is_valid


class TestExtractCommandTypes:
    """Tests for extract_command_types function."""

    def test_extract_from_definitions(self):
        """Should extract command types from definitions.messages."""
        schema = {
            "definitions": {
                "messages": {
                    "request_start_run": {
                        "allOf": [
                            {
                                "properties": {
                                    "command_type": {"const": "REQUEST_START_RUN"}
                                }
                            }
                        ]
                    }
                }
            }
        }

        commands = extract_command_types(schema)
        assert "REQUEST_START_RUN" in commands

    def test_empty_schema_returns_empty(self):
        """Empty schema should return empty set."""
        commands = extract_command_types({})
        assert len(commands) == 0


class TestExtractMessageTypes:
    """Tests for extract_message_types function."""

    def test_extract_message_type_const(self):
        """Should extract message types from const."""
        schema = {
            "definitions": {
                "messages": {
                    "heartbeat": {
                        "properties": {
                            "message_type": {"const": "HEARTBEAT"}
                        }
                    }
                }
            }
        }

        messages = extract_message_types(schema)
        assert "HEARTBEAT" in messages


class TestProtocolChanges:
    """Tests for check_protocol_changes function."""

    def test_no_changes_detected(self):
        """Identical schemas should have no changes."""
        schema = {
            "definitions": {
                "messages": {
                    "heartbeat": {
                        "properties": {
                            "message_type": {"const": "HEARTBEAT"}
                        }
                    }
                }
            }
        }

        result = check_protocol_changes(schema, schema)
        assert len(result.new_commands) == 0
        assert len(result.new_messages) == 0

    def test_new_command_detected(self):
        """New command should be detected."""
        base = {"definitions": {"messages": {}}}
        new = {
            "definitions": {
                "messages": {
                    "request_start_run": {
                        "allOf": [
                            {
                                "properties": {
                                    "command_type": {"const": "REQUEST_START_RUN"}
                                }
                            }
                        ]
                    }
                }
            }
        }

        result = check_protocol_changes(base, new)
        assert "REQUEST_START_RUN" in result.new_commands

    def test_new_command_requires_review(self):
        """New commands should require security review."""
        base = {"definitions": {"messages": {}}}
        new = {
            "definitions": {
                "messages": {
                    "request_new_thing": {
                        "allOf": [
                            {
                                "properties": {
                                    "command_type": {"const": "REQUEST_NEW_THING"}
                                }
                            }
                        ]
                    }
                }
            }
        }

        result = check_protocol_changes(base, new)
        assert result.requires_security_review

    def test_removed_command_detected(self):
        """Removed command should be detected."""
        base = {
            "definitions": {
                "messages": {
                    "request_start_run": {
                        "allOf": [
                            {
                                "properties": {
                                    "command_type": {"const": "REQUEST_START_RUN"}
                                }
                            }
                        ]
                    }
                }
            }
        }
        new = {"definitions": {"messages": {}}}

        result = check_protocol_changes(base, new)
        assert "REQUEST_START_RUN" in result.removed_commands

    def test_prohibited_command_fails(self):
        """Adding prohibited command should fail."""
        base = {"definitions": {"messages": {}}}
        new = {
            "definitions": {
                "messages": {
                    "place_order": {
                        "allOf": [
                            {
                                "properties": {
                                    "command_type": {"const": "PLACE_ORDER"}
                                }
                            }
                        ]
                    }
                }
            }
        }

        result = check_protocol_changes(base, new)
        assert not result.passed
        assert result.requires_security_review


class TestProtocolAllowlistValidation:
    """Tests for validate_protocol_allowlist function."""

    def test_valid_protocol_passes(self):
        """Protocol with only allowed types should pass."""
        schema = {
            "definitions": {
                "messages": {
                    "heartbeat": {
                        "properties": {
                            "message_type": {"const": "HEARTBEAT"}
                        }
                    },
                    "request_start_run": {
                        "allOf": [
                            {
                                "properties": {
                                    "command_type": {"const": "REQUEST_START_RUN"}
                                }
                            }
                        ]
                    }
                }
            }
        }

        result = validate_protocol_allowlist(schema)
        assert result.passed

    def test_invalid_command_fails(self):
        """Protocol with invalid command should fail."""
        schema = {
            "definitions": {
                "messages": {
                    "place_order": {
                        "allOf": [
                            {
                                "properties": {
                                    "command_type": {"const": "PLACE_ORDER"}
                                }
                            }
                        ]
                    }
                }
            }
        }

        result = validate_protocol_allowlist(schema)
        assert not result.passed


class TestRealProtocolSchema:
    """Tests against the actual protocol schema."""

    @pytest.fixture
    def schemas_dir(self) -> Path:
        """Get the schemas directory."""
        return Path(__file__).parent.parent.parent / "docs" / "schemas"

    def test_actual_protocol_schema_valid(self, schemas_dir: Path):
        """The actual protocol_messages.schema.json should be valid."""
        schema_path = schemas_dir / "protocol_messages.schema.json"
        if schema_path.exists():
            with open(schema_path) as f:
                schema = json.load(f)

            result = validate_protocol_allowlist(schema)
            # The schema should define only allowed commands
            assert result.passed or result.requires_security_review


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_definitions(self):
        """Schema with empty definitions should pass."""
        schema = {"definitions": {}}
        result = validate_protocol_allowlist(schema)
        assert result.passed

    def test_no_definitions(self):
        """Schema without definitions should pass."""
        schema = {"type": "object"}
        result = validate_protocol_allowlist(schema)
        assert result.passed

    def test_case_sensitivity(self):
        """Command types should be case-sensitive."""
        # REQUEST_START_RUN is valid, but request_start_run is not
        is_valid, _ = validate_command_type("REQUEST_START_RUN")
        assert is_valid

        is_valid, _ = validate_command_type("request_start_run")
        assert not is_valid
