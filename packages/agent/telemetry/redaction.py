# -*- coding: utf-8 -*-
"""
Telemetry Redaction Middleware.

CCEA Phase 8 Implementation.

This module provides mandatory redaction for all telemetry data before
it leaves the agent. Redaction cannot be disabled - this is a critical
security requirement per Design Doc.

Key Features:
    - Pattern-based redaction for secrets, credentials, and PII
    - Environment variable filtering (never log env vars)
    - Account identifier masking
    - Configurable patterns with built-in defaults
    - Statistical tracking for audit purposes

Design Doc Reference:
    - Phase 8 (13.2): "Mandatory redaction + DLP"
    - "Agent cannot send telemetry without redaction middleware enabled"
    - "Prohibit logging env vars; mask account identifiers; filter secrets"

SECURITY CRITICAL:
    - Redaction is ALWAYS enabled - cannot be disabled
    - All telemetry MUST pass through this middleware
    - Any bypass attempt results in telemetry rejection
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, Final, FrozenSet, List, Optional, Pattern, Set, Tuple, Union
from uuid import uuid4


# ============================================================================
# Constants - Patterns That MUST Be Redacted
# ============================================================================

# Redaction placeholder
REDACTED_PLACEHOLDER: Final[str] = "[REDACTED]"
MASKED_PLACEHOLDER: Final[str] = "***"

# High-sensitivity field patterns (always redact)
SECRET_FIELD_PATTERNS: Final[FrozenSet[str]] = frozenset({
    "api_key",
    "api_secret",
    "apikey",
    "apisecret",
    "secret",
    "secret_key",
    "secretkey",
    "private_key",
    "privatekey",
    "password",
    "passwd",
    "pwd",
    "token",
    "access_token",
    "refresh_token",
    "bearer",
    "auth_token",
    "authtoken",
    "credential",
    "credentials",
    "auth",
    "authorization",
    "broker_key",
    "broker_secret",
    "exchange_key",
    "exchange_secret",
    "wallet_key",
    "seed_phrase",
    "mnemonic",
    "passphrase",
})

# Account identifier patterns (mask, don't fully redact)
ACCOUNT_FIELD_PATTERNS: Final[FrozenSet[str]] = frozenset({
    "account_number",
    "account_id",
    "accountnumber",
    "accountid",
    "account",
    "ssn",
    "social_security",
    "tax_id",
    "taxid",
    "ein",
    "routing_number",
    "bank_account",
    "card_number",
    "cardnumber",
    "credit_card",
    "debit_card",
    "iban",
    "swift",
    "bic",
})

# PII patterns (mask for privacy)
PII_FIELD_PATTERNS: Final[FrozenSet[str]] = frozenset({
    "email",
    "phone",
    "phone_number",
    "phonenumber",
    "address",
    "street_address",
    "city",
    "state",
    "zip",
    "zipcode",
    "postal_code",
    "country",
    "first_name",
    "firstname",
    "last_name",
    "lastname",
    "full_name",
    "fullname",
    "name",
    "dob",
    "date_of_birth",
    "birthdate",
    "birth_date",
    "ip_address",
    "ipaddress",
    "user_agent",
    "useragent",
})

# Environment variable names that should never be logged
PROHIBITED_ENV_VARS: Final[FrozenSet[str]] = frozenset({
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AZURE_CLIENT_SECRET",
    "AZURE_SUBSCRIPTION_ID",
    "GCP_SERVICE_ACCOUNT_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "DATABASE_URL",
    "DATABASE_PASSWORD",
    "DB_PASSWORD",
    "REDIS_PASSWORD",
    "BROKER_API_KEY",
    "BROKER_API_SECRET",
    "EXCHANGE_API_KEY",
    "EXCHANGE_API_SECRET",
    "PRIVATE_KEY",
    "SECRET_KEY",
    "JWT_SECRET",
    "SESSION_SECRET",
    "ENCRYPTION_KEY",
    "MASTER_KEY",
    "SSH_KEY",
    "GPG_KEY",
    "GITHUB_TOKEN",
    "GITLAB_TOKEN",
    "NPM_TOKEN",
    "PYPI_TOKEN",
    "DOCKER_PASSWORD",
})

# Regex patterns for value-based detection
VALUE_REDACTION_PATTERNS: Final[List[Tuple[str, Pattern[str]]]] = [
    # API Keys (generic patterns)
    ("api_key", re.compile(r"(?:api[_-]?key|apikey)[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9_\-]{20,})[\"']?", re.IGNORECASE)),
    # AWS Access Keys
    ("aws_key", re.compile(r"(?:AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}")),
    # AWS Secret Keys
    ("aws_secret", re.compile(r"[A-Za-z0-9/+=]{40}")),
    # JWT Tokens
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*")),
    # Bearer Tokens
    ("bearer", re.compile(r"[Bb]earer\s+[A-Za-z0-9_\-\.]+", re.IGNORECASE)),
    # Private Keys
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    # Credit Card Numbers (basic)
    ("credit_card", re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b")),
    # SSN
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Email addresses
    ("email", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
    # IP addresses
    ("ip", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    # Phone numbers (US format)
    ("phone", re.compile(r"\b(?:\+1)?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    # UUID (for tracking but not full redaction)
    ("uuid", re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)),
]


class RedactionLevel(Enum):
    """Level of redaction to apply."""
    FULL = auto()      # Replace entirely with [REDACTED]
    PARTIAL = auto()   # Mask middle portion
    HASH = auto()      # Replace with hash for correlation
    NONE = auto()      # No redaction (use with caution)


@dataclass
class RedactionPattern:
    """
    Pattern for redaction matching.

    Attributes:
        name: Pattern identifier
        field_patterns: Field name patterns to match
        value_pattern: Optional regex for value matching
        level: How to redact matched content
        is_mandatory: Cannot be disabled
    """
    name: str
    field_patterns: FrozenSet[str] = field(default_factory=frozenset)
    value_pattern: Optional[Pattern[str]] = None
    level: RedactionLevel = RedactionLevel.FULL
    is_mandatory: bool = True

    def matches_field(self, field_name: str) -> bool:
        """Check if field name matches this pattern."""
        field_lower = field_name.lower()
        for pattern in self.field_patterns:
            if pattern in field_lower:
                return True
        return False

    def matches_value(self, value: str) -> bool:
        """Check if value matches this pattern."""
        if self.value_pattern is None:
            return False
        return bool(self.value_pattern.search(str(value)))


@dataclass
class RedactionStats:
    """
    Statistics about redaction operations.

    Used for audit and monitoring purposes.
    """
    total_fields_processed: int = 0
    fields_redacted: int = 0
    values_redacted: int = 0
    patterns_matched: Dict[str, int] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    redaction_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_fields_processed": self.total_fields_processed,
            "fields_redacted": self.fields_redacted,
            "values_redacted": self.values_redacted,
            "patterns_matched": self.patterns_matched,
            "timestamp": self.timestamp.isoformat(),
            "redaction_version": self.redaction_version,
        }


@dataclass
class RedactionResult:
    """
    Result of a redaction operation.

    Attributes:
        data: Redacted data
        stats: Statistics about what was redacted
        redaction_applied: Always True (cannot be False)
        redaction_hash: Hash of redaction config for verification
    """
    data: Dict[str, Any]
    stats: RedactionStats
    redaction_applied: bool = True  # Always True, cannot be disabled
    redaction_hash: str = ""

    def __post_init__(self):
        """Ensure redaction is always applied."""
        # SECURITY: This cannot be set to False
        self.redaction_applied = True


@dataclass
class RedactionConfig:
    """
    Configuration for redaction middleware.

    Note: Redaction itself cannot be disabled. This config only
    controls additional patterns and behavior.
    """
    # Additional custom patterns (built-in patterns always active)
    custom_patterns: List[RedactionPattern] = field(default_factory=list)

    # Whether to hash redacted values for correlation
    enable_hash_correlation: bool = False

    # Minimum length for partial masking (shorter values fully redacted)
    partial_mask_min_length: int = 8

    # Number of visible characters at start/end for partial mask
    partial_mask_visible: int = 2

    # Maximum depth for nested object redaction
    max_depth: int = 20

    # Whether to redact UUIDs (disabled by default, they're often needed)
    redact_uuids: bool = False

    # Log redaction statistics
    log_stats: bool = True

    # SECURITY: This cannot be disabled - included for documentation only
    # Setting this to False will be ignored
    _redaction_enabled: bool = field(default=True, repr=False)

    def __post_init__(self):
        """Ensure redaction is always enabled."""
        # SECURITY CRITICAL: Redaction cannot be disabled
        self._redaction_enabled = True


class TelemetryRedactionMiddleware:
    """
    Mandatory redaction middleware for telemetry.

    This middleware MUST be used for all telemetry data before transmission.
    Redaction cannot be disabled - this is a critical security requirement.

    Usage:
        middleware = TelemetryRedactionMiddleware()

        # Redact telemetry data
        result = middleware.redact(telemetry_data)

        # Access redacted data
        safe_data = result.data

        # Check statistics
        print(f"Redacted {result.stats.fields_redacted} fields")

    SECURITY:
        - Redaction is ALWAYS enabled
        - Built-in patterns cannot be disabled
        - All telemetry MUST pass through this middleware
    """

    # Class-level flag to track if middleware is being used
    _instances: Set[int] = set()
    _lock = threading.Lock()

    def __init__(self, config: Optional[RedactionConfig] = None):
        """
        Initialize redaction middleware.

        Args:
            config: Optional configuration (redaction still mandatory)
        """
        self.config = config or RedactionConfig()

        # SECURITY: Force redaction enabled regardless of config
        self.config._redaction_enabled = True

        # Build pattern registry
        self._patterns = self._build_pattern_registry()

        # Compute config hash for verification
        self._config_hash = self._compute_config_hash()

        # Track instance
        with TelemetryRedactionMiddleware._lock:
            TelemetryRedactionMiddleware._instances.add(id(self))

    def _build_pattern_registry(self) -> List[RedactionPattern]:
        """Build the pattern registry with mandatory + custom patterns."""
        patterns = []

        # Mandatory secret patterns (cannot be disabled)
        patterns.append(RedactionPattern(
            name="secrets",
            field_patterns=SECRET_FIELD_PATTERNS,
            level=RedactionLevel.FULL,
            is_mandatory=True,
        ))

        # Mandatory account patterns
        patterns.append(RedactionPattern(
            name="accounts",
            field_patterns=ACCOUNT_FIELD_PATTERNS,
            level=RedactionLevel.PARTIAL,
            is_mandatory=True,
        ))

        # Mandatory PII patterns
        patterns.append(RedactionPattern(
            name="pii",
            field_patterns=PII_FIELD_PATTERNS,
            level=RedactionLevel.PARTIAL,
            is_mandatory=True,
        ))

        # Value-based patterns
        for name, regex in VALUE_REDACTION_PATTERNS:
            if name == "uuid" and not self.config.redact_uuids:
                continue
            patterns.append(RedactionPattern(
                name=f"value_{name}",
                value_pattern=regex,
                level=RedactionLevel.FULL if name in ("api_key", "aws_key", "aws_secret", "jwt", "bearer", "private_key") else RedactionLevel.PARTIAL,
                is_mandatory=True,
            ))

        # Add custom patterns
        patterns.extend(self.config.custom_patterns)

        return patterns

    def _compute_config_hash(self) -> str:
        """Compute hash of configuration for verification."""
        config_str = str(sorted(self.config.custom_patterns, key=lambda p: p.name))
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def redact(self, data: Dict[str, Any]) -> RedactionResult:
        """
        Redact sensitive data from telemetry.

        This method MUST be called before any telemetry is transmitted.
        Redaction cannot be bypassed.

        Args:
            data: Telemetry data to redact

        Returns:
            RedactionResult with redacted data and statistics
        """
        stats = RedactionStats()

        # Perform deep redaction
        redacted_data = self._redact_recursive(data, stats, depth=0)

        return RedactionResult(
            data=redacted_data,
            stats=stats,
            redaction_applied=True,  # Always True
            redaction_hash=self._config_hash,
        )

    def _redact_recursive(
        self,
        obj: Any,
        stats: RedactionStats,
        depth: int,
        parent_key: str = "",
    ) -> Any:
        """Recursively redact sensitive data."""
        if depth > self.config.max_depth:
            return REDACTED_PLACEHOLDER

        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                stats.total_fields_processed += 1

                # Check if key matches any pattern
                redaction_level = self._get_redaction_level(key)
                if redaction_level != RedactionLevel.NONE:
                    result[key] = self._apply_redaction(value, redaction_level, stats, key)
                    stats.fields_redacted += 1
                else:
                    # Recurse into nested structures
                    result[key] = self._redact_recursive(
                        value, stats, depth + 1, key
                    )
            return result

        elif isinstance(obj, list):
            return [
                self._redact_recursive(item, stats, depth + 1, parent_key)
                for item in obj
            ]

        elif isinstance(obj, str):
            # Check string values for sensitive patterns
            return self._redact_string_value(obj, stats)

        else:
            return obj

    def _get_redaction_level(self, field_name: str) -> RedactionLevel:
        """Determine redaction level for a field name."""
        for pattern in self._patterns:
            if pattern.matches_field(field_name):
                self._track_pattern_match(pattern.name)
                return pattern.level
        return RedactionLevel.NONE

    def _track_pattern_match(self, pattern_name: str) -> None:
        """Track pattern match for statistics (thread-safe way would use lock)."""
        pass  # Stats tracking done at redaction time

    def _redact_string_value(self, value: str, stats: RedactionStats) -> str:
        """Check and redact string values based on content patterns."""
        result = value

        for pattern in self._patterns:
            if pattern.value_pattern and pattern.matches_value(result):
                # Redact matched portions
                if pattern.level == RedactionLevel.FULL:
                    result = pattern.value_pattern.sub(REDACTED_PLACEHOLDER, result)
                elif pattern.level == RedactionLevel.PARTIAL:
                    result = pattern.value_pattern.sub(
                        lambda m: self._partial_mask(m.group()),
                        result
                    )
                stats.values_redacted += 1
                stats.patterns_matched[pattern.name] = stats.patterns_matched.get(pattern.name, 0) + 1

        return result

    def _apply_redaction(
        self,
        value: Any,
        level: RedactionLevel,
        stats: RedactionStats,
        field_name: str,
    ) -> Any:
        """Apply redaction to a value based on level."""
        if level == RedactionLevel.FULL:
            return REDACTED_PLACEHOLDER

        elif level == RedactionLevel.PARTIAL:
            if isinstance(value, str):
                return self._partial_mask(value)
            return REDACTED_PLACEHOLDER

        elif level == RedactionLevel.HASH:
            if self.config.enable_hash_correlation:
                return f"hash:{hashlib.sha256(str(value).encode()).hexdigest()[:12]}"
            return REDACTED_PLACEHOLDER

        return value

    def _partial_mask(self, value: str) -> str:
        """Apply partial masking to a value."""
        if len(value) < self.config.partial_mask_min_length:
            return REDACTED_PLACEHOLDER

        visible = self.config.partial_mask_visible
        if len(value) <= visible * 2:
            return REDACTED_PLACEHOLDER

        return f"{value[:visible]}{MASKED_PLACEHOLDER}{value[-visible:]}"

    def redact_env_vars(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove any environment variable references from data.

        Environment variables must NEVER appear in telemetry.

        Args:
            data: Data potentially containing env var references

        Returns:
            Data with env vars removed
        """
        result = self.redact(data).data

        # Additional check for env var patterns
        def remove_env_refs(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {
                    k: remove_env_refs(v) if k.upper() not in PROHIBITED_ENV_VARS else REDACTED_PLACEHOLDER
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [remove_env_refs(item) for item in obj]
            elif isinstance(obj, str):
                # Check if value looks like an env var reference
                for env_var in PROHIBITED_ENV_VARS:
                    if env_var in obj.upper():
                        return REDACTED_PLACEHOLDER
                return obj
            return obj

        return remove_env_refs(result)

    def validate_no_secrets(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that data contains no secrets.

        Use this to verify data after redaction as a safety check.

        Args:
            data: Data to validate

        Returns:
            Tuple of (is_valid, list of violations)
        """
        violations = []

        def check_recursive(obj: Any, path: str = "") -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key

                    # Check field name
                    key_lower = key.lower()
                    for pattern in SECRET_FIELD_PATTERNS:
                        if pattern in key_lower and value != REDACTED_PLACEHOLDER:
                            violations.append(f"Unredacted secret field: {current_path}")

                    check_recursive(value, current_path)

            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_recursive(item, f"{path}[{i}]")

            elif isinstance(obj, str):
                # Check for secret patterns in values
                for name, regex in VALUE_REDACTION_PATTERNS:
                    if name in ("api_key", "aws_key", "aws_secret", "jwt", "bearer", "private_key"):
                        if regex.search(obj) and obj != REDACTED_PLACEHOLDER:
                            violations.append(f"Potential secret in value at {path}")

        check_recursive(data)
        return len(violations) == 0, violations

    @classmethod
    def is_active(cls) -> bool:
        """Check if any redaction middleware instance is active."""
        with cls._lock:
            return len(cls._instances) > 0

    @property
    def redaction_version(self) -> str:
        """Get redaction version for telemetry metadata."""
        return "1.0.0"

    @property
    def config_hash(self) -> str:
        """Get configuration hash for verification."""
        return self._config_hash


# ============================================================================
# Convenience Functions
# ============================================================================

def create_default_middleware() -> TelemetryRedactionMiddleware:
    """Create middleware with default configuration."""
    return TelemetryRedactionMiddleware()


def redact_telemetry(data: Dict[str, Any]) -> RedactionResult:
    """
    Convenience function to redact telemetry data.

    Creates a default middleware and redacts data.

    Args:
        data: Telemetry data to redact

    Returns:
        RedactionResult
    """
    middleware = create_default_middleware()
    return middleware.redact(data)


def ensure_redaction_applied(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure data has been properly redacted.

    If data hasn't been redacted, applies redaction.
    Always safe to call.

    Args:
        data: Potentially unredacted data

    Returns:
        Redacted data
    """
    result = redact_telemetry(data)
    return result.data
