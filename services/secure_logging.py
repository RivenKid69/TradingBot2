# -*- coding: utf-8 -*-
"""
services/secure_logging.py
Secure logging utilities with API key masking.

This module provides utilities to prevent sensitive information (API keys, secrets,
tokens, passwords) from appearing in logs.

Usage:
    from services.secure_logging import (
        mask_secrets,
        SecureLogFilter,
        configure_secure_logging,
        get_secure_logger,
    )

    # Option 1: Use secure logger directly
    logger = get_secure_logger(__name__)
    logger.info(f"Config: {config}")  # API keys will be masked

    # Option 2: Configure existing logger
    configure_secure_logging(existing_logger)

    # Option 3: Mask secrets manually
    safe_text = mask_secrets(text_with_secrets)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Pattern, Set, Union


# =============================================================================
# Constants
# =============================================================================

# Default mask replacement
DEFAULT_MASK = "***REDACTED***"

# Minimum length for a value to be considered potentially sensitive
MIN_SECRET_LENGTH = 8

# Maximum characters to show at start/end of masked value (for debugging)
MASK_PREVIEW_CHARS = 4

# Patterns that identify sensitive keys in dictionaries
SENSITIVE_KEY_PATTERNS: Set[str] = {
    "api_key",
    "api_secret",
    "apikey",
    "apisecret",
    "secret",
    "password",
    "passwd",
    "token",
    "access_token",
    "refresh_token",
    "bearer",
    "authorization",
    "auth",
    "credential",
    "credentials",
    "private_key",
    "privatekey",
    "secret_key",
    "secretkey",
    "client_secret",
    "client_id",  # Sometimes sensitive
    "account_id",  # Can be sensitive
    "signature",
    "hmac",
}

# Regex patterns for detecting secrets in text
SECRET_REGEX_PATTERNS: List[Pattern[str]] = [
    # API keys (alphanumeric, 8+ chars) - more permissive
    re.compile(r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']?([a-zA-Z0-9_-]{8,})["\']?'),
    # API secrets (8+ chars)
    re.compile(
        r'(?i)(api[_-]?secret|apisecret|secret[_-]?key)\s*[=:]\s*["\']?([a-zA-Z0-9_-]{8,})["\']?'
    ),
    # Generic secrets (8+ chars)
    re.compile(r'(?i)(secret|password|passwd|token)\s*[=:]\s*["\']?([^\s"\']{8,})["\']?'),
    # Bearer tokens (20+ chars)
    re.compile(r"(?i)bearer\s+([a-zA-Z0-9_.-]{20,})"),
    # Authorization headers (16+ chars)
    re.compile(r'(?i)authorization\s*[=:]\s*["\']?([^\s"\']{16,})["\']?'),
    # Binance-style keys (64-char hex strings)
    re.compile(r"\b([a-zA-Z0-9]{64})\b"),
    # AWS-style keys
    re.compile(r"\b(AKIA[0-9A-Z]{16})\b"),
    re.compile(r"\b([a-zA-Z0-9/+=]{40})\b"),  # AWS secret key pattern
]


# =============================================================================
# Masking Functions
# =============================================================================


def mask_value(value: str, preview: bool = False) -> str:
    """
    Mask a sensitive value.

    Args:
        value: The sensitive value to mask.
        preview: If True, show first/last few characters for debugging.

    Returns:
        Masked string.

    Examples:
        >>> mask_value("abc123def456")
        '***REDACTED***'
        >>> mask_value("abc123def456", preview=True)
        'abc1***...***f456'
    """
    if not value or len(value) < MIN_SECRET_LENGTH:
        return DEFAULT_MASK

    if preview and len(value) > MASK_PREVIEW_CHARS * 2 + 6:
        return f"{value[:MASK_PREVIEW_CHARS]}***...***{value[-MASK_PREVIEW_CHARS:]}"

    return DEFAULT_MASK


def mask_secrets(
    text: str,
    additional_patterns: Optional[List[Pattern[str]]] = None,
    preview: bool = False,
) -> str:
    """
    Mask sensitive information in a text string.

    Args:
        text: Text that may contain secrets.
        additional_patterns: Extra regex patterns to match.
        preview: If True, show partial values for debugging.

    Returns:
        Text with secrets masked.

    Examples:
        >>> mask_secrets("api_key=abc123456789")
        'api_key=***REDACTED***'
    """
    if not text:
        return text

    result = text
    patterns = SECRET_REGEX_PATTERNS.copy()
    if additional_patterns:
        patterns.extend(additional_patterns)

    for pattern in patterns:

        def replacer(match: re.Match) -> str:
            groups = match.groups()
            if len(groups) >= 2:
                # Pattern has key=value structure
                key, value = groups[0], groups[1]
                return f"{key}={mask_value(value, preview)}"
            elif len(groups) == 1:
                # Pattern matches just the value
                return mask_value(groups[0], preview)
            return match.group(0)

        result = pattern.sub(replacer, result)

    return result


def mask_dict(
    data: Dict[str, Any],
    sensitive_keys: Optional[Set[str]] = None,
    preview: bool = False,
    recursive: bool = True,
) -> Dict[str, Any]:
    """
    Mask sensitive values in a dictionary.

    Args:
        data: Dictionary that may contain sensitive values.
        sensitive_keys: Additional keys to treat as sensitive.
        preview: If True, show partial values for debugging.
        recursive: If True, process nested dictionaries.

    Returns:
        Dictionary with sensitive values masked.

    Examples:
        >>> mask_dict({"api_key": "secret123", "name": "test"})
        {'api_key': '***REDACTED***', 'name': 'test'}
    """
    if not isinstance(data, dict):
        return data

    keys_to_mask = SENSITIVE_KEY_PATTERNS.copy()
    if sensitive_keys:
        keys_to_mask.update(sensitive_keys)

    result = {}
    for key, value in data.items():
        key_lower = key.lower().replace("-", "_")

        # Check if key matches sensitive patterns
        is_sensitive = any(pattern in key_lower for pattern in keys_to_mask)

        if is_sensitive and isinstance(value, str):
            result[key] = mask_value(value, preview)
        elif recursive and isinstance(value, dict):
            result[key] = mask_dict(value, sensitive_keys, preview, recursive)
        elif recursive and isinstance(value, list):
            result[key] = [
                (
                    mask_dict(item, sensitive_keys, preview, recursive)
                    if isinstance(item, dict)
                    else item
                )
                for item in value
            ]
        else:
            result[key] = value

    return result


def safe_repr(obj: Any, preview: bool = False) -> str:
    """
    Create a safe string representation of an object with secrets masked.

    Args:
        obj: Object to represent.
        preview: If True, show partial values for debugging.

    Returns:
        Safe string representation.
    """
    if isinstance(obj, dict):
        return str(mask_dict(obj, preview=preview))
    elif isinstance(obj, str):
        return mask_secrets(obj, preview=preview)
    else:
        # For other objects, convert to string and mask
        return mask_secrets(str(obj), preview=preview)


# =============================================================================
# Logging Filter
# =============================================================================


class SecureLogFilter(logging.Filter):
    """
    Logging filter that masks sensitive information in log records.

    This filter processes log messages and arguments to mask API keys,
    secrets, tokens, and other sensitive data before they are written
    to log output.

    Usage:
        logger = logging.getLogger(__name__)
        logger.addFilter(SecureLogFilter())

    Attributes:
        preview: If True, show partial values for debugging.
        additional_keys: Extra key names to treat as sensitive.
    """

    def __init__(
        self,
        name: str = "",
        preview: bool = False,
        additional_keys: Optional[Set[str]] = None,
    ):
        """
        Initialize the secure log filter.

        Args:
            name: Filter name (passed to parent).
            preview: If True, show partial secret values.
            additional_keys: Extra keys to treat as sensitive.
        """
        super().__init__(name)
        self.preview = preview
        self.additional_keys = additional_keys or set()

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter the log record by masking sensitive data.

        Args:
            record: The log record to filter.

        Returns:
            True (always passes, just modifies the record).
        """
        # Mask the main message
        if record.msg:
            if isinstance(record.msg, str):
                record.msg = mask_secrets(record.msg, preview=self.preview)
            elif isinstance(record.msg, dict):
                record.msg = mask_dict(
                    record.msg,
                    sensitive_keys=self.additional_keys,
                    preview=self.preview,
                )

        # Mask arguments
        if record.args:
            if isinstance(record.args, dict):
                record.args = mask_dict(
                    record.args,
                    sensitive_keys=self.additional_keys,
                    preview=self.preview,
                )
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    (
                        mask_secrets(str(arg), preview=self.preview)
                        if isinstance(arg, str)
                        else mask_dict(arg, preview=self.preview) if isinstance(arg, dict) else arg
                    )
                    for arg in record.args
                )

        return True


class SecureFormatter(logging.Formatter):
    """
    Logging formatter that masks sensitive information in formatted output.

    This provides an additional layer of protection by masking secrets
    in the final formatted log message.
    """

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
        preview: bool = False,
    ):
        """
        Initialize the secure formatter.

        Args:
            fmt: Format string.
            datefmt: Date format string.
            style: Format style (%, {, or $).
            preview: If True, show partial secret values.
        """
        super().__init__(fmt, datefmt, style)
        self.preview = preview

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with secrets masked.

        Args:
            record: The log record to format.

        Returns:
            Formatted string with secrets masked.
        """
        result = super().format(record)
        return mask_secrets(result, preview=self.preview)


# =============================================================================
# Logger Configuration
# =============================================================================


def configure_secure_logging(
    logger: logging.Logger,
    preview: bool = False,
    additional_keys: Optional[Set[str]] = None,
) -> None:
    """
    Configure an existing logger with secure filtering.

    Args:
        logger: Logger to configure.
        preview: If True, show partial secret values.
        additional_keys: Extra keys to treat as sensitive.

    Example:
        logger = logging.getLogger("my_module")
        configure_secure_logging(logger)
    """
    # Add filter
    secure_filter = SecureLogFilter(
        preview=preview,
        additional_keys=additional_keys,
    )
    logger.addFilter(secure_filter)

    # Update handlers with secure formatter
    for handler in logger.handlers:
        if handler.formatter:
            original_fmt = handler.formatter._fmt
            original_datefmt = handler.formatter.datefmt
            handler.setFormatter(
                SecureFormatter(
                    fmt=original_fmt,
                    datefmt=original_datefmt,
                    preview=preview,
                )
            )


def get_secure_logger(
    name: str,
    level: int = logging.INFO,
    preview: bool = False,
    additional_keys: Optional[Set[str]] = None,
) -> logging.Logger:
    """
    Get a logger configured with secure filtering.

    Args:
        name: Logger name (typically __name__).
        level: Logging level.
        preview: If True, show partial secret values.
        additional_keys: Extra keys to treat as sensitive.

    Returns:
        Configured logger with secure filtering.

    Example:
        logger = get_secure_logger(__name__)
        logger.info(f"Config: {config}")  # Secrets masked
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Add secure filter if not already present
    has_secure_filter = any(isinstance(f, SecureLogFilter) for f in logger.filters)
    if not has_secure_filter:
        logger.addFilter(
            SecureLogFilter(
                preview=preview,
                additional_keys=additional_keys,
            )
        )

    return logger


# =============================================================================
# Config Validation Helpers
# =============================================================================


def validate_api_credentials(
    api_key: Optional[str],
    api_secret: Optional[str],
    require_both: bool = True,
    min_key_length: int = 16,
    min_secret_length: int = 32,
) -> tuple[bool, Optional[str]]:
    """
    Validate API credentials without exposing them.

    Args:
        api_key: API key to validate.
        api_secret: API secret to validate.
        require_both: If True, both key and secret must be present.
        min_key_length: Minimum length for API key.
        min_secret_length: Minimum length for API secret.

    Returns:
        Tuple of (is_valid, error_message).
        error_message is None if valid.

    Example:
        is_valid, error = validate_api_credentials(key, secret)
        if not is_valid:
            logger.error(f"Invalid credentials: {error}")
    """
    errors = []

    # Check presence
    has_key = bool(api_key and api_key.strip())
    has_secret = bool(api_secret and api_secret.strip())

    if require_both:
        if not has_key:
            errors.append("API key is missing or empty")
        if not has_secret:
            errors.append("API secret is missing or empty")
    elif not has_key and not has_secret:
        errors.append("At least one of API key or secret is required")

    # Check lengths (without exposing actual values)
    if has_key and len(api_key.strip()) < min_key_length:
        errors.append(f"API key too short (minimum {min_key_length} characters)")

    if has_secret and len(api_secret.strip()) < min_secret_length:
        errors.append(f"API secret too short (minimum {min_secret_length} characters)")

    # Check for placeholder values
    placeholder_patterns = ["xxx", "your_", "placeholder", "changeme", "test123"]
    if has_key:
        key_lower = api_key.lower()
        for pattern in placeholder_patterns:
            if pattern in key_lower:
                errors.append("API key appears to be a placeholder value")
                break

    if has_secret:
        secret_lower = api_secret.lower()
        for pattern in placeholder_patterns:
            if pattern in secret_lower:
                errors.append("API secret appears to be a placeholder value")
                break

    if errors:
        return False, "; ".join(errors)

    return True, None


def get_credential_summary(
    api_key: Optional[str],
    api_secret: Optional[str],
) -> Dict[str, Any]:
    """
    Get a safe summary of credentials for logging.

    Args:
        api_key: API key.
        api_secret: API secret.

    Returns:
        Dictionary with safe credential info (lengths, presence).

    Example:
        summary = get_credential_summary(key, secret)
        logger.info(f"Credentials: {summary}")
        # Output: Credentials: {'api_key_present': True, 'api_key_length': 64, ...}
    """
    return {
        "api_key_present": bool(api_key and api_key.strip()),
        "api_key_length": len(api_key) if api_key else 0,
        "api_secret_present": bool(api_secret and api_secret.strip()),
        "api_secret_length": len(api_secret) if api_secret else 0,
    }


# =============================================================================
# Module-level secure logger
# =============================================================================

# Default secure logger for this module
_module_logger: Optional[logging.Logger] = None


def get_module_logger() -> logging.Logger:
    """Get the module-level secure logger."""
    global _module_logger
    if _module_logger is None:
        _module_logger = get_secure_logger("services.secure_logging")
    return _module_logger
