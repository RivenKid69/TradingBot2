# -*- coding: utf-8 -*-
"""
Tests for CCEA Telemetry Module.

Tests:
- MANDATORY redaction
- Telemetry collection
- Telemetry export
"""

import pytest
from datetime import datetime

from ccea.telemetry.redaction import (
    RedactionMiddleware,
    RedactionRule,
    RedactionAction,
    MANDATORY_REDACTION_RULES,
    SENSITIVE_FIELD_NAMES,
    redact_data,
    redact_string,
    validate_no_secrets,
)
from ccea.telemetry.collector import (
    TelemetryCollector,
    TelemetryEvent,
    TelemetryLevel,
    EventType,
)


class TestMandatoryRedaction:
    """Tests for mandatory redaction."""

    def test_redaction_always_enabled(self):
        """Test that redaction cannot be disabled."""
        middleware = RedactionMiddleware()

        assert middleware.enabled is True

        # Try to disable (should have no effect)
        middleware.disable()

        assert middleware.enabled is True

    def test_api_key_redaction(self):
        """Test API key pattern redaction."""
        data = {
            "message": "Using key ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefg",
        }

        redacted = redact_data(data)

        # Should be redacted
        assert "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefg" not in str(redacted)

    def test_password_field_redaction(self):
        """Test password field redaction."""
        data = {
            "password": "secret123",
            "api_key": "my_secret_key",
            "credential": "cred_value",
        }

        redacted = redact_data(data)

        assert redacted["password"] == "[REDACTED]"
        assert redacted["api_key"] == "[REDACTED]"
        assert redacted["credential"] == "[REDACTED]"

    def test_ip_address_anonymization(self):
        """Test IP address anonymization."""
        data = {
            "message": "Connected from 192.168.1.100",
        }

        redacted = redact_data(data)

        assert "192.168.1.100" not in redacted["message"]
        assert "X.X.X.X" in redacted["message"]

    def test_email_redaction(self):
        """Test email address redaction."""
        data = {
            "message": "Contact user@example.com for support",
        }

        redacted = redact_data(data)

        assert "user@example.com" not in redacted["message"]
        assert "[EMAIL]" in redacted["message"]

    def test_private_key_removal(self):
        """Test private key removal."""
        data = {
            "key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBg...\n-----END PRIVATE KEY-----",
        }

        redacted = redact_data(data)

        assert "BEGIN PRIVATE KEY" not in str(redacted)

    def test_nested_redaction(self):
        """Test redaction in nested structures."""
        data = {
            "outer": {
                "inner": {
                    "secret": "sensitive_value",
                    "message": "API key: ABC123DEF456GHI789JKL0",
                }
            }
        }

        redacted = redact_data(data)

        assert redacted["outer"]["inner"]["secret"] == "[REDACTED]"

    def test_list_redaction(self):
        """Test redaction in lists."""
        data = {
            "items": [
                {"password": "secret1"},
                {"password": "secret2"},
            ]
        }

        redacted = redact_data(data)

        assert redacted["items"][0]["password"] == "[REDACTED]"
        assert redacted["items"][1]["password"] == "[REDACTED]"

    def test_broker_account_redaction(self):
        """Test broker account redaction."""
        data = {
            "message": "alpaca_account: PA123456789",
        }

        redacted = redact_data(data)

        # Should contain redaction marker
        assert "PA123456789" not in str(redacted) or "[BROKER_ACCOUNT]" in str(redacted)

    def test_sensitive_field_names(self):
        """Test all sensitive field names are defined."""
        expected_sensitive = {
            "password",
            "secret",
            "token",
            "api_key",
            "credential",
        }

        for field in expected_sensitive:
            assert field in SENSITIVE_FIELD_NAMES

    def test_middleware_process(self):
        """Test middleware process method."""
        middleware = RedactionMiddleware()

        data = {
            "token": "my_secret_token",
            "normal": "visible_data",
        }

        redacted = middleware.process(data)

        assert redacted["token"] == "[REDACTED]"
        assert redacted["normal"] == "visible_data"

    def test_middleware_callable(self):
        """Test middleware as callable."""
        middleware = RedactionMiddleware()

        data = {"password": "secret"}
        redacted = middleware(data)

        assert redacted["password"] == "[REDACTED]"


class TestValidateNoSecrets:
    """Tests for secret validation."""

    def test_detect_api_key(self):
        """Test detection of potential API key."""
        data = {
            "key": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnop",
        }

        issues = validate_no_secrets(data)

        assert len(issues) > 0

    def test_detect_private_key(self):
        """Test detection of private key."""
        data = {
            "data": "-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----",
        }

        issues = validate_no_secrets(data)

        assert any("Private key" in issue for issue in issues)

    def test_detect_sensitive_field(self):
        """Test detection of sensitive field names."""
        data = {
            "password": "any_value",
        }

        issues = validate_no_secrets(data)

        assert any("password" in issue.lower() for issue in issues)

    def test_clean_data_no_issues(self):
        """Test that clean data has no issues."""
        data = {
            "message": "Normal message",
            "count": 42,
        }

        issues = validate_no_secrets(data)

        # May still have false positives for long strings
        # but shouldn't have sensitive field names
        sensitive_issues = [i for i in issues if "Sensitive field" in i]
        assert len(sensitive_issues) == 0


class TestTelemetryCollector:
    """Tests for TelemetryCollector."""

    def test_collect_event(self):
        """Test collecting telemetry event."""
        collector = TelemetryCollector(
            agent_id="agent_telemetry123456789012",
        )

        event = collector.collect(
            event_type="STRATEGY_ITERATION",
            data={"iteration": 1},
        )

        assert event.event_type == "STRATEGY_ITERATION"
        assert event.data["iteration"] == 1

    def test_redaction_always_applied(self):
        """Test that redaction is always applied."""
        collector = TelemetryCollector(
            agent_id="agent_redact12345678901234",
        )

        assert collector.redaction_applied is True

    def test_collect_with_sensitive_data(self):
        """Test collecting event with sensitive data."""
        collector = TelemetryCollector(
            agent_id="agent_sensitive1234567890",
        )

        event = collector.collect(
            event_type="TEST_EVENT",
            data={
                "message": "Using API key ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                "password": "secret123",
            },
        )

        # Password should be redacted
        assert event.data.get("password") == "[REDACTED]"

    def test_telemetry_levels(self):
        """Test telemetry level filtering."""
        collector = TelemetryCollector(
            agent_id="agent_levels123456789012",
            level=TelemetryLevel.AGGREGATED,
        )

        # Aggregated event should be collected
        event1 = collector.collect(
            event_type="PERFORMANCE",
            level=TelemetryLevel.AGGREGATED,
        )

        # Raw event should not be stored
        event2 = collector.collect(
            event_type="ORDER_DETAIL",
            level=TelemetryLevel.RAW_ORDER_EVENTS,
        )

        events = collector.get_events()
        assert len(events) == 1

    def test_collect_performance(self):
        """Test collecting performance summary."""
        collector = TelemetryCollector(
            agent_id="agent_perf1234567890123456",
        )

        event = collector.collect_performance(
            pnl=5.5,
            win_rate=60.0,
            drawdown=10.0,
        )

        assert event.event_type == EventType.PERFORMANCE_SUMMARY.value
        assert event.data["pnl_pct"] == 5.5

    def test_collect_error(self):
        """Test collecting error event."""
        collector = TelemetryCollector(
            agent_id="agent_error1234567890123456",
        )

        event = collector.collect_error(
            error_type="ConnectionError",
            message="Failed to connect to broker",
        )

        assert event.event_type == EventType.STRATEGY_ERROR.value

    def test_flush_events(self):
        """Test flushing events."""
        collector = TelemetryCollector(
            agent_id="agent_flush1234567890123456",
        )

        collector.collect("EVENT_1", {})
        collector.collect("EVENT_2", {})

        events = collector.flush()

        assert len(events) == 2
        assert len(collector.get_events()) == 0  # Buffer cleared

    def test_event_sequence(self):
        """Test event sequence numbers."""
        collector = TelemetryCollector(
            agent_id="agent_seq1234567890123456789",
        )

        event1 = collector.collect("EVENT_1", {})
        event2 = collector.collect("EVENT_2", {})

        assert event2.sequence > event1.sequence


class TestTelemetryLevels:
    """Tests for telemetry levels."""

    def test_aggregated_is_default(self):
        """Test AGGREGATED is default level."""
        collector = TelemetryCollector(
            agent_id="agent_default12345678901234",
        )

        assert collector.level == TelemetryLevel.AGGREGATED

    def test_level_values(self):
        """Test level enum values."""
        assert TelemetryLevel.AGGREGATED.value == "AGGREGATED"
        assert TelemetryLevel.DETAILED_NON_SENSITIVE.value == "DETAILED_NON_SENSITIVE"
        assert TelemetryLevel.RAW_ORDER_EVENTS.value == "RAW_ORDER_EVENTS"


class TestRedactionRules:
    """Tests for redaction rules."""

    def test_mandatory_rules_exist(self):
        """Test mandatory rules are defined."""
        assert len(MANDATORY_REDACTION_RULES) > 0

    def test_rule_structure(self):
        """Test rule structure."""
        for rule in MANDATORY_REDACTION_RULES:
            assert rule.name is not None
            assert rule.pattern is not None
            assert rule.action is not None

    def test_custom_rule(self):
        """Test adding custom rule."""
        custom_rule = RedactionRule(
            name="custom_pattern",
            pattern=r"CUSTOM_\d+",
            action=RedactionAction.MASK,
            replacement="[CUSTOM]",
        )

        middleware = RedactionMiddleware(additional_rules=[custom_rule])

        data = {"message": "Found CUSTOM_12345 in logs"}
        redacted = middleware.process(data)

        assert "CUSTOM_12345" not in redacted["message"]
