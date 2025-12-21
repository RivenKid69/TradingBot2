# -*- coding: utf-8 -*-
"""
CCEA Telemetry Redaction.

MANDATORY redaction middleware for all telemetry.
This module is designed to be non-bypassable - redaction is always active.

Security (design intent):
- Secrets are designed to be redacted on all telemetry paths
- Account identifiers are masked
- IP addresses are anonymized
- Environment variables filtered

Note: Enforcement correctness depends on CI tests and deployment validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Union


class RedactionAction(str, Enum):
    """Redaction action types."""
    MASK = "MASK"           # Replace with ***
    HASH = "HASH"           # Replace with hash
    REMOVE = "REMOVE"       # Remove entirely
    TRUNCATE = "TRUNCATE"   # Truncate to N chars
    ANONYMIZE = "ANONYMIZE" # Anonymize (e.g., IP -> X.X.X.X)


@dataclass
class RedactionRule:
    """
    Redaction rule definition.
    """
    name: str
    pattern: Union[str, Pattern]
    action: RedactionAction = RedactionAction.MASK
    replacement: str = "[REDACTED]"
    field_names: Optional[Set[str]] = None
    priority: int = 100  # Lower = higher priority

    def __post_init__(self):
        if isinstance(self.pattern, str):
            self.pattern = re.compile(self.pattern, re.IGNORECASE)


# ============================================================================
# MANDATORY Redaction Rules (CANNOT be disabled)
# ============================================================================

MANDATORY_REDACTION_RULES: List[RedactionRule] = [
    # API Keys and Secrets
    RedactionRule(
        name="api_key_pattern",
        pattern=r"[a-zA-Z0-9]{20,}",  # Long alphanumeric strings
        action=RedactionAction.MASK,
        replacement="[API_KEY]",
        priority=10,
    ),
    RedactionRule(
        name="secret_key_pattern",
        pattern=r"(?:secret|key|token|password|credential|auth)[_\-]?[a-zA-Z]*[:\s=]+['\"]?[\w\-\.]+['\"]?",
        action=RedactionAction.MASK,
        replacement="[SECRET]",
        priority=5,
    ),
    RedactionRule(
        name="bearer_token",
        pattern=r"Bearer\s+[a-zA-Z0-9\-\._~\+\/]+",
        action=RedactionAction.MASK,
        replacement="Bearer [TOKEN]",
        priority=5,
    ),

    # Account Identifiers
    RedactionRule(
        name="account_id",
        pattern=r"account[_\-]?id[:\s=]+['\"]?[\w\-]+['\"]?",
        action=RedactionAction.MASK,
        replacement="account_id=[MASKED]",
        priority=20,
    ),
    RedactionRule(
        name="broker_account",
        pattern=r"(?:alpaca|binance|oanda|ib|deribit)[_\-]?(?:account|key)[:\s=]+['\"]?[\w\-]+['\"]?",
        action=RedactionAction.MASK,
        replacement="[BROKER_ACCOUNT]",
        priority=15,
    ),

    # IP Addresses
    RedactionRule(
        name="ipv4_address",
        pattern=r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        action=RedactionAction.ANONYMIZE,
        replacement="X.X.X.X",
        priority=30,
    ),
    RedactionRule(
        name="ipv6_address",
        pattern=r"(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}",
        action=RedactionAction.ANONYMIZE,
        replacement="[IPv6]",
        priority=30,
    ),

    # Email Addresses
    RedactionRule(
        name="email",
        pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        action=RedactionAction.MASK,
        replacement="[EMAIL]",
        priority=25,
    ),

    # Private Keys
    RedactionRule(
        name="private_key",
        pattern=r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+(?:RSA\s+)?PRIVATE\s+KEY-----",
        action=RedactionAction.REMOVE,
        replacement="[PRIVATE_KEY_REMOVED]",
        priority=1,
    ),

    # Environment Variables (common secret patterns)
    RedactionRule(
        name="env_var_secret",
        pattern=r"(?:AWS|AZURE|GCP|DATABASE|DB|REDIS|MONGO|POSTGRES|MYSQL)[_\-]?(?:SECRET|KEY|PASSWORD|TOKEN|CREDENTIAL)[_\-]?[\w]*[:\s=]+['\"]?[\w\-\.]+['\"]?",
        action=RedactionAction.MASK,
        replacement="[ENV_SECRET]",
        priority=10,
    ),
]

# Field names that should always be redacted
SENSITIVE_FIELD_NAMES: Set[str] = {
    "password", "secret", "token", "api_key", "apikey", "api_secret",
    "private_key", "credential", "auth", "authorization",
    "access_key", "secret_key", "session_token", "bearer",
    "account_number", "routing_number", "ssn", "tax_id",
}


# ============================================================================
# Redaction Functions
# ============================================================================

def redact_string(
    value: str,
    rules: List[RedactionRule],
) -> str:
    """
    Apply redaction rules to a string.

    Args:
        value: String to redact
        rules: Redaction rules to apply

    Returns:
        Redacted string
    """
    # Sort rules by priority
    sorted_rules = sorted(rules, key=lambda r: r.priority)

    result = value
    for rule in sorted_rules:
        if rule.action == RedactionAction.MASK:
            result = rule.pattern.sub(rule.replacement, result)
        elif rule.action == RedactionAction.REMOVE:
            result = rule.pattern.sub("", result)
        elif rule.action == RedactionAction.ANONYMIZE:
            result = rule.pattern.sub(rule.replacement, result)
        elif rule.action == RedactionAction.TRUNCATE:
            matches = rule.pattern.findall(result)
            for match in matches:
                truncated = match[:8] + "..." if len(match) > 8 else match
                result = result.replace(match, truncated)

    return result


def redact_data(
    data: Any,
    rules: Optional[List[RedactionRule]] = None,
    sensitive_fields: Optional[Set[str]] = None,
) -> Any:
    """
    Recursively redact data structure.

    IMPORTANT: Uses MANDATORY rules by default.

    Args:
        data: Data to redact (dict, list, or string)
        rules: Additional rules (mandatory rules always applied)
        sensitive_fields: Additional sensitive field names

    Returns:
        Redacted data
    """
    # Always include mandatory rules
    all_rules = MANDATORY_REDACTION_RULES.copy()
    if rules:
        all_rules.extend(rules)

    # Always include sensitive fields
    all_sensitive = SENSITIVE_FIELD_NAMES.copy()
    if sensitive_fields:
        all_sensitive.update(sensitive_fields)

    return _redact_recursive(data, all_rules, all_sensitive)


def _redact_recursive(
    data: Any,
    rules: List[RedactionRule],
    sensitive_fields: Set[str],
    current_key: Optional[str] = None,
) -> Any:
    """Recursively redact data."""
    # Check if current key is sensitive
    if current_key and current_key.lower() in sensitive_fields:
        return "[REDACTED]"

    if isinstance(data, str):
        return redact_string(data, rules)

    elif isinstance(data, dict):
        result = {}
        for key, value in data.items():
            # Check if key is sensitive
            if key.lower() in sensitive_fields:
                result[key] = "[REDACTED]"
            else:
                result[key] = _redact_recursive(value, rules, sensitive_fields, key)
        return result

    elif isinstance(data, list):
        return [_redact_recursive(item, rules, sensitive_fields) for item in data]

    else:
        # Primitives (int, float, bool, None)
        return data


# ============================================================================
# Redaction Middleware
# ============================================================================

class RedactionMiddleware:
    """
    Mandatory redaction middleware.

    CANNOT be disabled - redaction is always applied.
    """

    # Class-level flag to prevent disabling
    _ENABLED: bool = True
    _LOCK_ENABLED: bool = True  # Prevents modification

    def __init__(
        self,
        additional_rules: Optional[List[RedactionRule]] = None,
        additional_sensitive_fields: Optional[Set[str]] = None,
    ):
        """
        Initialize middleware.

        Args:
            additional_rules: Additional redaction rules
            additional_sensitive_fields: Additional sensitive field names
        """
        # Always include mandatory rules
        self.rules = MANDATORY_REDACTION_RULES.copy()
        if additional_rules:
            self.rules.extend(additional_rules)

        self.sensitive_fields = SENSITIVE_FIELD_NAMES.copy()
        if additional_sensitive_fields:
            self.sensitive_fields.update(additional_sensitive_fields)

    @property
    def enabled(self) -> bool:
        """
        Check if redaction is enabled.

        Always returns True - redaction cannot be disabled.
        """
        return True

    def disable(self) -> None:
        """
        Attempt to disable redaction.

        IMPORTANT: This method does NOTHING.
        Redaction cannot be disabled per Design Doc.
        """
        # Intentionally does nothing
        pass

    def process(self, data: Any) -> Any:
        """
        Process data through redaction.

        Args:
            data: Data to redact

        Returns:
            Redacted data
        """
        return redact_data(data, self.rules, self.sensitive_fields)

    def __call__(self, data: Any) -> Any:
        """Allow using middleware as callable."""
        return self.process(data)

    def add_rule(self, rule: RedactionRule) -> None:
        """Add a redaction rule."""
        self.rules.append(rule)

    def add_sensitive_field(self, field_name: str) -> None:
        """Add a sensitive field name."""
        self.sensitive_fields.add(field_name.lower())


# ============================================================================
# Validation
# ============================================================================

def validate_no_secrets(data: Any) -> List[str]:
    """
    Validate that data contains no obvious secrets.

    Args:
        data: Data to validate

    Returns:
        List of potential issues found
    """
    issues = []

    def check_recursive(obj: Any, path: str = "") -> None:
        if isinstance(obj, str):
            # Check for obvious patterns
            if re.search(r"[a-zA-Z0-9]{32,}", obj):
                issues.append(f"{path}: Potential API key detected")
            if re.search(r"-----BEGIN.*KEY-----", obj):
                issues.append(f"{path}: Private key detected")
            if re.search(r"password\s*[:=]", obj, re.IGNORECASE):
                issues.append(f"{path}: Password reference detected")

        elif isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                if key.lower() in SENSITIVE_FIELD_NAMES:
                    issues.append(f"{new_path}: Sensitive field name")
                check_recursive(value, new_path)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_recursive(item, f"{path}[{i}]")

    check_recursive(data)
    return issues
