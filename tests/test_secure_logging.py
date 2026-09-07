# -*- coding: utf-8 -*-
"""
tests/test_secure_logging.py
Tests for secure logging utilities.
"""

import logging
import pytest
from io import StringIO

from services.secure_logging import (
    mask_value,
    mask_secrets,
    mask_dict,
    safe_repr,
    SecureLogFilter,
    SecureFormatter,
    configure_secure_logging,
    get_secure_logger,
    validate_api_credentials,
    get_credential_summary,
    DEFAULT_MASK,
    MASK_PREVIEW_CHARS,
)


# =============================================================================
# Test mask_value
# =============================================================================


class TestMaskValue:
    """Tests for mask_value function."""

    def test_mask_value_basic(self):
        """Test basic value masking."""
        result = mask_value("abc123def456789")
        assert result == DEFAULT_MASK
        assert "abc" not in result
        assert "123" not in result

    def test_mask_value_short_string(self):
        """Test that short strings still get masked."""
        result = mask_value("short")
        assert result == DEFAULT_MASK

    def test_mask_value_empty_string(self):
        """Test empty string handling."""
        result = mask_value("")
        assert result == DEFAULT_MASK

    def test_mask_value_none(self):
        """Test None handling."""
        result = mask_value(None)
        assert result == DEFAULT_MASK

    def test_mask_value_with_preview(self):
        """Test preview mode shows partial value."""
        secret = "abc123456789xyz"
        result = mask_value(secret, preview=True)
        # Should show first and last N chars
        assert result.startswith(secret[:MASK_PREVIEW_CHARS])
        assert result.endswith(secret[-MASK_PREVIEW_CHARS:])
        assert "***" in result

    def test_mask_value_preview_short_string(self):
        """Test preview mode with short string."""
        # Short strings should still be fully masked
        result = mask_value("short", preview=True)
        assert result == DEFAULT_MASK


# =============================================================================
# Test mask_secrets
# =============================================================================


class TestMaskSecrets:
    """Tests for mask_secrets function."""

    def test_mask_api_key_equals(self):
        """Test masking api_key=value patterns."""
        text = "Config: api_key=abc123456789def"
        result = mask_secrets(text)
        assert "abc123456789def" not in result
        assert DEFAULT_MASK in result or "api_key=" in result

    def test_mask_api_secret_equals(self):
        """Test masking api_secret=value patterns."""
        text = "api_secret=mysecretvalue12345678"
        result = mask_secrets(text)
        assert "mysecretvalue12345678" not in result

    def test_mask_secret_colon(self):
        """Test masking secret: value patterns."""
        text = 'secret: "verysecretkey12345"'
        result = mask_secrets(text)
        assert "verysecretkey12345" not in result

    def test_mask_bearer_token(self):
        """Test masking Bearer tokens."""
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        result = mask_secrets(text)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result

    def test_mask_64_char_hex(self):
        """Test masking 64-character hex strings (common for secrets)."""
        hex_secret = "a" * 64
        text = f"Key: {hex_secret}"
        result = mask_secrets(text)
        assert hex_secret not in result

    def test_preserve_non_sensitive_text(self):
        """Test that non-sensitive text is preserved."""
        text = "User logged in successfully from 192.168.1.1"
        result = mask_secrets(text)
        assert result == text

    def test_empty_text(self):
        """Test empty text handling."""
        assert mask_secrets("") == ""
        assert mask_secrets(None) is None

    def test_mask_multiple_secrets(self):
        """Test masking multiple secrets in one text."""
        text = "api_key=key123456789 api_secret=secret987654321"
        result = mask_secrets(text)
        assert "key123456789" not in result
        assert "secret987654321" not in result


# =============================================================================
# Test mask_dict
# =============================================================================


class TestMaskDict:
    """Tests for mask_dict function."""

    def test_mask_api_key_in_dict(self):
        """Test masking api_key in dictionary."""
        data = {"api_key": "secret123456789", "name": "test"}
        result = mask_dict(data)
        assert result["api_key"] == DEFAULT_MASK
        assert result["name"] == "test"

    def test_mask_api_secret_in_dict(self):
        """Test masking api_secret in dictionary."""
        data = {"api_secret": "verysecretvalue", "timeout": 30}
        result = mask_dict(data)
        assert result["api_secret"] == DEFAULT_MASK
        assert result["timeout"] == 30

    def test_mask_password_in_dict(self):
        """Test masking password in dictionary."""
        data = {"password": "mypassword123", "username": "user"}
        result = mask_dict(data)
        assert result["password"] == DEFAULT_MASK
        assert result["username"] == "user"

    def test_mask_token_in_dict(self):
        """Test masking token in dictionary."""
        data = {"access_token": "token12345678901234"}
        result = mask_dict(data)
        assert result["access_token"] == DEFAULT_MASK

    def test_mask_nested_dict(self):
        """Test masking in nested dictionaries."""
        data = {"config": {"api_key": "nestedkey12345", "endpoint": "https://api.example.com"}}
        result = mask_dict(data)
        assert result["config"]["api_key"] == DEFAULT_MASK
        assert result["config"]["endpoint"] == "https://api.example.com"

    def test_mask_deeply_nested(self):
        """Test masking in deeply nested structures."""
        data = {"level1": {"level2": {"level3": {"secret": "deeplynestedsecret"}}}}
        result = mask_dict(data)
        assert result["level1"]["level2"]["level3"]["secret"] == DEFAULT_MASK

    def test_mask_list_of_dicts(self):
        """Test masking in list of dictionaries."""
        data = {
            "exchanges": [
                {"name": "binance", "api_key": "key1234567890"},
                {"name": "alpaca", "api_secret": "secret0987654321"},
            ]
        }
        result = mask_dict(data)
        assert result["exchanges"][0]["api_key"] == DEFAULT_MASK
        assert result["exchanges"][1]["api_secret"] == DEFAULT_MASK
        assert result["exchanges"][0]["name"] == "binance"

    def test_mask_with_additional_keys(self):
        """Test masking with additional sensitive keys."""
        data = {"custom_secret": "myvalue12345678", "normal": "visible"}
        result = mask_dict(data, sensitive_keys={"custom_secret"})
        assert result["custom_secret"] == DEFAULT_MASK
        assert result["normal"] == "visible"

    def test_mask_non_string_values(self):
        """Test that non-string values are preserved."""
        data = {
            "api_key": None,
            "timeout": 30,
            "enabled": True,
            "rate": 0.5,
        }
        result = mask_dict(data)
        assert result["api_key"] is None  # None preserved
        assert result["timeout"] == 30
        assert result["enabled"] is True
        assert result["rate"] == 0.5

    def test_original_dict_unchanged(self):
        """Test that original dictionary is not modified."""
        original = {"api_key": "secret123456789"}
        result = mask_dict(original)
        assert original["api_key"] == "secret123456789"
        assert result["api_key"] == DEFAULT_MASK


# =============================================================================
# Test safe_repr
# =============================================================================


class TestSafeRepr:
    """Tests for safe_repr function."""

    def test_safe_repr_dict(self):
        """Test safe representation of dictionary."""
        data = {"api_key": "secret123456789", "name": "test"}
        result = safe_repr(data)
        assert "secret123456789" not in result
        assert "name" in result

    def test_safe_repr_string(self):
        """Test safe representation of string."""
        text = "api_key=secret123456789"
        result = safe_repr(text)
        assert "secret123456789" not in result

    def test_safe_repr_other_types(self):
        """Test safe representation of other types."""
        # Should convert to string and mask
        result = safe_repr(12345)
        assert result == "12345"


# =============================================================================
# Test SecureLogFilter
# =============================================================================


class TestSecureLogFilter:
    """Tests for SecureLogFilter class."""

    def test_filter_masks_message(self):
        """Test that filter masks secrets in message."""
        filter_obj = SecureLogFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="api_key=secret123456789",
            args=(),
            exc_info=None,
        )
        filter_obj.filter(record)
        assert "secret123456789" not in record.msg

    def test_filter_masks_dict_message(self):
        """Test that filter masks dict messages."""
        filter_obj = SecureLogFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg={"api_key": "secret123456789"},
            args=(),
            exc_info=None,
        )
        filter_obj.filter(record)
        assert record.msg.get("api_key") == DEFAULT_MASK

    def test_filter_masks_args(self):
        """Test that filter masks secrets in args."""
        filter_obj = SecureLogFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Config: %s",
            args=({"api_key": "secret123456789"},),
            exc_info=None,
        )
        filter_obj.filter(record)
        # Args should be masked
        if isinstance(record.args, tuple) and len(record.args) > 0:
            assert record.args[0].get("api_key") == DEFAULT_MASK

    def test_filter_always_returns_true(self):
        """Test that filter always allows record through."""
        filter_obj = SecureLogFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Normal message",
            args=(),
            exc_info=None,
        )
        assert filter_obj.filter(record) is True


# =============================================================================
# Test SecureFormatter
# =============================================================================


class TestSecureFormatter:
    """Tests for SecureFormatter class."""

    def test_formatter_masks_output(self):
        """Test that formatter masks secrets in output."""
        formatter = SecureFormatter(fmt="%(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="api_key=secret123456789abc",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "secret123456789abc" not in result


# =============================================================================
# Test get_secure_logger
# =============================================================================


class TestGetSecureLogger:
    """Tests for get_secure_logger function."""

    def test_get_secure_logger_creates_logger(self):
        """Test that function creates a logger."""
        logger = get_secure_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_get_secure_logger_has_filter(self):
        """Test that logger has secure filter."""
        logger = get_secure_logger("test_module_filtered")
        has_secure_filter = any(isinstance(f, SecureLogFilter) for f in logger.filters)
        assert has_secure_filter

    def test_get_secure_logger_idempotent(self):
        """Test that calling twice doesn't add duplicate filters."""
        logger1 = get_secure_logger("test_module_idem")
        filter_count1 = sum(1 for f in logger1.filters if isinstance(f, SecureLogFilter))

        logger2 = get_secure_logger("test_module_idem")
        filter_count2 = sum(1 for f in logger2.filters if isinstance(f, SecureLogFilter))

        assert filter_count1 == filter_count2 == 1


# =============================================================================
# Test configure_secure_logging
# =============================================================================


class TestConfigureSecureLogging:
    """Tests for configure_secure_logging function."""

    def test_configure_adds_filter(self):
        """Test that configure adds filter to logger."""
        logger = logging.getLogger("test_configure")
        initial_filters = len(logger.filters)
        configure_secure_logging(logger)
        assert len(logger.filters) == initial_filters + 1
        has_secure_filter = any(isinstance(f, SecureLogFilter) for f in logger.filters)
        assert has_secure_filter


# =============================================================================
# Test validate_api_credentials
# =============================================================================


class TestValidateApiCredentials:
    """Tests for validate_api_credentials function."""

    def test_valid_credentials(self):
        """Test validation with valid credentials."""
        is_valid, error = validate_api_credentials(
            api_key="a" * 20,
            api_secret="b" * 40,
        )
        assert is_valid is True
        assert error is None

    def test_missing_api_key(self):
        """Test validation with missing API key."""
        is_valid, error = validate_api_credentials(
            api_key=None,
            api_secret="b" * 40,
        )
        assert is_valid is False
        assert "API key" in error

    def test_missing_api_secret(self):
        """Test validation with missing API secret."""
        is_valid, error = validate_api_credentials(
            api_key="a" * 20,
            api_secret=None,
        )
        assert is_valid is False
        assert "API secret" in error

    def test_empty_credentials(self):
        """Test validation with empty strings."""
        is_valid, error = validate_api_credentials(
            api_key="",
            api_secret="",
        )
        assert is_valid is False

    def test_short_api_key(self):
        """Test validation with too short API key."""
        is_valid, error = validate_api_credentials(
            api_key="short",
            api_secret="b" * 40,
        )
        assert is_valid is False
        assert "too short" in error

    def test_placeholder_api_key(self):
        """Test validation with placeholder API key."""
        is_valid, error = validate_api_credentials(
            api_key="your_api_key_here123",
            api_secret="b" * 40,
        )
        assert is_valid is False
        assert "placeholder" in error

    def test_placeholder_api_secret(self):
        """Test validation with placeholder API secret."""
        is_valid, error = validate_api_credentials(
            api_key="a" * 20,
            api_secret="changeme" + "x" * 32,
        )
        assert is_valid is False
        assert "placeholder" in error

    def test_not_require_both(self):
        """Test validation when both not required."""
        is_valid, error = validate_api_credentials(
            api_key="a" * 20,
            api_secret=None,
            require_both=False,
        )
        assert is_valid is True
        assert error is None


# =============================================================================
# Test get_credential_summary
# =============================================================================


class TestGetCredentialSummary:
    """Tests for get_credential_summary function."""

    def test_summary_with_credentials(self):
        """Test summary with valid credentials."""
        summary = get_credential_summary(
            api_key="a" * 20,
            api_secret="b" * 40,
        )
        assert summary["api_key_present"] is True
        assert summary["api_key_length"] == 20
        assert summary["api_secret_present"] is True
        assert summary["api_secret_length"] == 40

    def test_summary_without_credentials(self):
        """Test summary without credentials."""
        summary = get_credential_summary(
            api_key=None,
            api_secret=None,
        )
        assert summary["api_key_present"] is False
        assert summary["api_key_length"] == 0
        assert summary["api_secret_present"] is False
        assert summary["api_secret_length"] == 0

    def test_summary_with_empty_strings(self):
        """Test summary with empty strings."""
        summary = get_credential_summary(
            api_key="",
            api_secret="   ",
        )
        assert summary["api_key_present"] is False
        assert summary["api_secret_present"] is False

    def test_summary_does_not_expose_values(self):
        """Test that summary does not contain actual values."""
        secret_key = "super_secret_key_12345"
        secret_value = "super_secret_value_67890"
        summary = get_credential_summary(
            api_key=secret_key,
            api_secret=secret_value,
        )
        summary_str = str(summary)
        assert secret_key not in summary_str
        assert secret_value not in summary_str


# =============================================================================
# Integration test
# =============================================================================


class TestSecureLoggingIntegration:
    """Integration tests for secure logging."""

    def test_logging_masks_secrets_end_to_end(self):
        """Test that logging with secure logger masks secrets."""
        # Setup logger with string handler
        logger = get_secure_logger("test_integration")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(SecureFormatter(fmt="%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Log message with secret
        secret = "mysupersecretkey123456"
        logger.info(f"Config api_key={secret}")

        # Check output
        output = stream.getvalue()
        assert secret not in output

        # Cleanup
        logger.removeHandler(handler)

    def test_logging_masks_dict_end_to_end(self):
        """Test that logging dict with secure logger masks secrets."""
        logger = get_secure_logger("test_integration_dict")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(SecureFormatter(fmt="%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Log dict with secret
        config = {"api_key": "secret123456789abc", "timeout": 30}
        logger.info("Config: %s", mask_dict(config))

        # Check output
        output = stream.getvalue()
        assert "secret123456789abc" not in output
        assert "timeout" in output

        # Cleanup
        logger.removeHandler(handler)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
