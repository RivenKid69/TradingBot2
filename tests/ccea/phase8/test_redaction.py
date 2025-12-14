# -*- coding: utf-8 -*-
"""
Tests for TelemetryRedactionMiddleware.

CCEA Phase 8 - Mandatory redaction + DLP tests.
"""

import pytest
from datetime import datetime

from packages.agent.telemetry.redaction import (
    TelemetryRedactionMiddleware,
    RedactionConfig,
    RedactionPattern,
    RedactionLevel,
    REDACTED_PLACEHOLDER,
    MASKED_PLACEHOLDER,
    SECRET_FIELD_PATTERNS,
    ACCOUNT_FIELD_PATTERNS,
    PII_FIELD_PATTERNS,
    PROHIBITED_ENV_VARS,
    redact_telemetry,
    ensure_redaction_applied,
    create_default_middleware,
)


class TestRedactionMiddlewareBasic:
    """Basic redaction middleware tests."""

    def test_create_default_middleware(self):
        """Test creating default middleware."""
        middleware = create_default_middleware()
        assert middleware is not None
        assert middleware.config is not None

    def test_redaction_always_enabled(self):
        """Test that redaction cannot be disabled."""
        config = RedactionConfig()
        config._redaction_enabled = False  # Try to disable
        middleware = TelemetryRedactionMiddleware(config)

        # Should still be enabled
        assert middleware.config._redaction_enabled is True

    def test_redaction_result_always_applied(self):
        """Test that redaction result always shows applied."""
        middleware = TelemetryRedactionMiddleware()
        result = middleware.redact({"test": "data"})

        # redaction_applied should always be True
        assert result.redaction_applied is True

    def test_basic_redaction(self):
        """Test basic data passes through."""
        middleware = TelemetryRedactionMiddleware()
        data = {"status": "ok", "count": 42}
        result = middleware.redact(data)

        assert result.data["status"] == "ok"
        assert result.data["count"] == 42


class TestSecretRedaction:
    """Secret field redaction tests."""

    @pytest.mark.parametrize("field_name", [
        "api_key", "api_secret", "password", "token",
        "access_token", "refresh_token", "secret_key",
        "private_key", "credentials", "broker_key",
    ])
    def test_secret_fields_redacted(self, field_name):
        """Test that secret fields are fully redacted."""
        middleware = TelemetryRedactionMiddleware()
        data = {field_name: "super_secret_value_12345"}
        result = middleware.redact(data)

        assert result.data[field_name] == REDACTED_PLACEHOLDER
        assert result.stats.fields_redacted > 0

    def test_nested_secret_redaction(self):
        """Test secret redaction in nested structures."""
        middleware = TelemetryRedactionMiddleware()
        data = {
            "config": {
                "broker": {
                    "api_key": "secret123",
                    "api_secret": "secret456",
                }
            }
        }
        result = middleware.redact(data)

        assert result.data["config"]["broker"]["api_key"] == REDACTED_PLACEHOLDER
        assert result.data["config"]["broker"]["api_secret"] == REDACTED_PLACEHOLDER

    def test_secret_in_list_redacted(self):
        """Test secret redaction in lists."""
        middleware = TelemetryRedactionMiddleware()
        data = {
            "credentials": [
                {"api_key": "key1"},
                {"api_key": "key2"},
            ]
        }
        result = middleware.redact(data)

        # The credentials field itself should be redacted since it matches
        assert result.data["credentials"] == REDACTED_PLACEHOLDER


class TestAccountRedaction:
    """Account identifier redaction tests."""

    @pytest.mark.parametrize("field_name", [
        "account_number", "account_id", "ssn",
        "tax_id", "routing_number", "card_number",
    ])
    def test_account_fields_masked(self, field_name):
        """Test that account fields are partially masked."""
        middleware = TelemetryRedactionMiddleware()
        data = {field_name: "1234567890123456"}
        result = middleware.redact(data)

        # Should be partially masked (starts with first chars, ends with last chars)
        assert MASKED_PLACEHOLDER in result.data[field_name] or result.data[field_name] == REDACTED_PLACEHOLDER

    def test_short_account_fully_redacted(self):
        """Test short account values are fully redacted."""
        middleware = TelemetryRedactionMiddleware()
        data = {"account_id": "123"}  # Too short for partial mask
        result = middleware.redact(data)

        assert result.data["account_id"] == REDACTED_PLACEHOLDER


class TestPIIRedaction:
    """PII field redaction tests."""

    @pytest.mark.parametrize("field_name", [
        "email", "phone", "phone_number", "address",
        "first_name", "last_name", "ip_address",
    ])
    def test_pii_fields_masked(self, field_name):
        """Test that PII fields are masked."""
        middleware = TelemetryRedactionMiddleware()
        data = {field_name: "some_pii_value_here"}
        result = middleware.redact(data)

        # Should be masked or redacted
        assert MASKED_PLACEHOLDER in result.data[field_name] or result.data[field_name] == REDACTED_PLACEHOLDER


class TestValuePatternRedaction:
    """Value-based pattern redaction tests."""

    def test_aws_key_redaction(self):
        """Test AWS access key pattern redaction."""
        middleware = TelemetryRedactionMiddleware()
        data = {"message": "Found key AKIAIOSFODNN7EXAMPLE in config"}
        result = middleware.redact(data)

        assert "AKIAIOSFODNN7EXAMPLE" not in result.data["message"]

    def test_jwt_token_redaction(self):
        """Test JWT token redaction."""
        middleware = TelemetryRedactionMiddleware()
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        data = {"auth": f"Bearer {jwt}"}
        result = middleware.redact(data)

        assert jwt not in result.data["auth"]

    def test_email_pattern_redaction(self):
        """Test email pattern redaction in values."""
        middleware = TelemetryRedactionMiddleware()
        data = {"log": "User john.doe@example.com logged in"}
        result = middleware.redact(data)

        # Email should be masked
        assert "john.doe@example.com" not in result.data["log"]

    def test_credit_card_pattern_redaction(self):
        """Test credit card number redaction."""
        middleware = TelemetryRedactionMiddleware()
        data = {"transaction": "Card 4111111111111111 used"}
        result = middleware.redact(data)

        assert "4111111111111111" not in result.data["transaction"]

    def test_ssn_pattern_redaction(self):
        """Test SSN pattern redaction."""
        middleware = TelemetryRedactionMiddleware()
        data = {"info": "SSN: 123-45-6789"}
        result = middleware.redact(data)

        assert "123-45-6789" not in result.data["info"]


class TestEnvVarRedaction:
    """Environment variable redaction tests."""

    def test_env_var_fields_redacted(self):
        """Test environment variable fields are redacted."""
        middleware = TelemetryRedactionMiddleware()
        data = {
            "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
            "DATABASE_PASSWORD": "secret123",
        }
        result = middleware.redact_env_vars(data)

        assert result["AWS_ACCESS_KEY_ID"] == REDACTED_PLACEHOLDER
        assert result["DATABASE_PASSWORD"] == REDACTED_PLACEHOLDER

    def test_env_var_in_value_redacted(self):
        """Test env var references in values are redacted."""
        middleware = TelemetryRedactionMiddleware()
        data = {"config": "Using AWS_SECRET_ACCESS_KEY from env"}
        result = middleware.redact_env_vars(data)

        assert result["config"] == REDACTED_PLACEHOLDER


class TestValidation:
    """Validation tests."""

    def test_validate_no_secrets_clean_data(self):
        """Test validation passes for clean data."""
        middleware = TelemetryRedactionMiddleware()
        data = {"status": "ok", "count": 42}
        is_valid, violations = middleware.validate_no_secrets(data)

        assert is_valid is True
        assert len(violations) == 0

    def test_validate_no_secrets_with_secrets(self):
        """Test validation fails for data with secrets."""
        middleware = TelemetryRedactionMiddleware()
        data = {"api_key": "actual_secret_here"}
        is_valid, violations = middleware.validate_no_secrets(data)

        assert is_valid is False
        assert len(violations) > 0


class TestRedactionStats:
    """Redaction statistics tests."""

    def test_stats_tracking(self):
        """Test redaction statistics are tracked."""
        middleware = TelemetryRedactionMiddleware()
        data = {
            "api_key": "secret1",
            "password": "secret2",
            "status": "ok",
        }
        result = middleware.redact(data)

        assert result.stats.total_fields_processed >= 3
        assert result.stats.fields_redacted >= 2

    def test_stats_to_dict(self):
        """Test stats serialization."""
        middleware = TelemetryRedactionMiddleware()
        result = middleware.redact({"api_key": "secret"})

        stats_dict = result.stats.to_dict()
        assert "total_fields_processed" in stats_dict
        assert "fields_redacted" in stats_dict
        assert "redaction_version" in stats_dict


class TestConvenienceFunctions:
    """Convenience function tests."""

    def test_redact_telemetry_function(self):
        """Test standalone redact_telemetry function."""
        data = {"api_key": "secret", "status": "ok"}
        result = redact_telemetry(data)

        assert result.data["api_key"] == REDACTED_PLACEHOLDER
        assert result.data["status"] == "ok"
        assert result.redaction_applied is True

    def test_ensure_redaction_applied(self):
        """Test ensure_redaction_applied function."""
        data = {"api_key": "secret", "status": "ok"}
        safe_data = ensure_redaction_applied(data)

        assert safe_data["api_key"] == REDACTED_PLACEHOLDER
        assert safe_data["status"] == "ok"


class TestCustomPatterns:
    """Custom pattern tests."""

    def test_custom_pattern_field(self):
        """Test custom field patterns."""
        custom_pattern = RedactionPattern(
            name="custom",
            field_patterns=frozenset({"my_secret_field"}),
            level=RedactionLevel.FULL,
        )
        config = RedactionConfig(custom_patterns=[custom_pattern])
        middleware = TelemetryRedactionMiddleware(config)

        data = {"my_secret_field": "custom_secret"}
        result = middleware.redact(data)

        assert result.data["my_secret_field"] == REDACTED_PLACEHOLDER


class TestMaxDepth:
    """Max depth handling tests."""

    def test_max_depth_exceeded(self):
        """Test deeply nested structures are handled."""
        config = RedactionConfig(max_depth=5)
        middleware = TelemetryRedactionMiddleware(config)

        # Create deeply nested structure
        data = {"level0": {"level1": {"level2": {"level3": {"level4": {"level5": {"level6": "deep"}}}}}}}
        result = middleware.redact(data)

        # Should not crash, deep values get redacted
        assert result.data is not None


class TestMiddlewareInstance:
    """Middleware instance tracking tests."""

    def test_middleware_instance_tracked(self):
        """Test middleware instances are tracked."""
        initial_count = len(TelemetryRedactionMiddleware._instances)
        middleware = TelemetryRedactionMiddleware()

        assert len(TelemetryRedactionMiddleware._instances) == initial_count + 1
        assert TelemetryRedactionMiddleware.is_active()

    def test_config_hash(self):
        """Test config hash generation."""
        middleware = TelemetryRedactionMiddleware()

        assert middleware.config_hash is not None
        assert len(middleware.config_hash) > 0

    def test_redaction_version(self):
        """Test redaction version."""
        middleware = TelemetryRedactionMiddleware()

        assert middleware.redaction_version == "1.0.0"
