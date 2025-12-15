# -*- coding: utf-8 -*-
"""
Telemetry Validation Middleware for Cloud Control Plane.

Phase 2 Implementation (2.5): Cloud rejection of prohibited fields + mandatory redaction.

This module provides:
    - TelemetryValidator: Validates telemetry payloads before ingestion
    - RedactionEnforcer: Ensures mandatory redaction is applied
    - ProhibitedFieldScanner: Scans for forbidden fields that should never reach cloud

Design Doc Reference:
    - Phase 2.5: "Telemetry ingestion: чувствительность уровней, redaction mandatory"
    - Cloud MUST reject telemetry containing order-like or sensitive fields
    - All non-aggregated telemetry MUST have redaction_applied=True

CLOUD ZONE ONLY.

CRITICAL SECURITY:
    The Cloud MUST NOT receive or store raw trading data.
    Telemetry payloads are validated to ensure no intent/order fields leak through.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

from ..models import TelemetryLevel


# ============================================================================
# Enums
# ============================================================================

class ValidationSeverity(str, Enum):
    """Validation violation severity."""
    CRITICAL = "critical"   # Block ingestion immediately
    HIGH = "high"           # Block with detailed logging
    MEDIUM = "medium"       # Allow with warning
    LOW = "low"             # Log only


class ViolationType(str, Enum):
    """Type of validation violation."""
    PROHIBITED_FIELD = "prohibited_field"
    INTENT_INJECTION = "intent_injection"
    MISSING_REDACTION = "missing_redaction"
    INVALID_LEVEL = "invalid_level"
    RAW_ORDER_DATA = "raw_order_data"
    SENSITIVE_PII = "sensitive_pii"
    BROKER_CREDENTIALS = "broker_credentials"
    UNKNOWN_FIELD = "unknown_field"


# ============================================================================
# Constants - PROHIBITED FIELDS
# ============================================================================

# Order/Intent fields - NEVER allowed in Cloud telemetry
PROHIBITED_ORDER_FIELDS: FrozenSet[str] = frozenset({
    # Direct order fields
    "side",
    "quantity",
    "qty",
    "price",
    "order_type",
    "limit_price",
    "stop_price",
    "take_profit",
    "stop_loss",
    "order_id",
    "client_order_id",
    "filled_qty",
    "remaining_qty",
    "average_price",
    "commission",
    "fills",

    # Intent/Signal fields
    "intent",
    "signal",
    "target_position",
    "target_qty",
    "target_allocation",
    "execute_order",
    "place_order",
    "submit_order",
    "cancel_order",
    "modify_order",

    # Position fields that indicate trading activity
    "position_side",
    "position_size",
    "entry_price",
    "exit_price",
    "unrealized_pnl",
    "realized_pnl",

    # Execution details
    "execution_id",
    "trade_id",
    "fill_price",
    "fill_qty",
    "slippage",
})

# PII fields - Must be redacted
PROHIBITED_PII_FIELDS: FrozenSet[str] = frozenset({
    "email",
    "phone",
    "phone_number",
    "address",
    "ssn",
    "social_security",
    "tax_id",
    "passport",
    "driver_license",
    "credit_card",
    "card_number",
    "cvv",
    "account_number",
    "routing_number",
    "iban",
    "swift",
    "bank_account",
    "date_of_birth",
    "dob",
    "ip_address",  # Unless aggregated
    "mac_address",
    "device_id",   # Unless hashed
    "user_agent",  # Unless aggregated
    "password",    # Never store passwords
    "secret",      # Generic secret field
})

# Broker credential patterns - CRITICAL
BROKER_CREDENTIAL_PATTERNS: List[Tuple[str, str]] = [
    (r"api[_-]?key", "API key"),
    (r"api[_-]?secret", "API secret"),
    (r"secret[_-]?key", "Secret key"),
    (r"private[_-]?key", "Private key"),
    (r"access[_-]?token", "Access token"),
    (r"refresh[_-]?token", "Refresh token"),
    (r"bearer[_-]?token", "Bearer token"),
    (r"password", "Password"),
    (r"passphrase", "Passphrase"),
    (r"(binance|alpaca|deribit|oanda|interactive_brokers|ib)[_-]?(key|secret|token)", "Broker credential"),
]

# Allowed telemetry payload fields for AGGREGATED level
ALLOWED_AGGREGATED_FIELDS: FrozenSet[str] = frozenset({
    # Metrics
    "timestamp",
    "event_time",
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "median",
    "p50",
    "p90",
    "p95",
    "p99",
    "stddev",
    "variance",
    "histogram",
    "buckets",

    # System metrics
    "cpu_percent",
    "cpu_usage",
    "memory_percent",
    "memory_usage",
    "memory_mb",
    "disk_usage",
    "disk_percent",
    "network_bytes_sent",
    "network_bytes_recv",
    "latency_ms",
    "latency_p50",
    "latency_p95",
    "latency_p99",

    # Trading metrics (aggregated only)
    "trade_count",
    "order_count",
    "fill_rate",
    "win_rate",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "daily_return",
    "cumulative_return",
    "volatility",
    "alpha",
    "beta",
    "information_ratio",

    # Strategy metrics
    "strategy_id",
    "strategy_name",
    "run_id",
    "deployment_id",
    "agent_id",
    "workspace_id",
    "version",
    "status",
    "state",
    "is_paper_trading",

    # Error metrics
    "error_count",
    "error_rate",
    "warning_count",
    "success_count",
    "failure_count",
    "retry_count",
    "timeout_count",

    # Resource usage
    "active_connections",
    "queue_depth",
    "processing_time_ms",
    "request_count",
    "response_count",
    "cache_hit_rate",
    "cache_miss_rate",

    # Time windows
    "period",
    "window",
    "interval",
    "start_time",
    "end_time",
    "duration_seconds",
})

# Allowed additional fields for DETAILED_NON_SENSITIVE level
ALLOWED_DETAILED_FIELDS: FrozenSet[str] = frozenset({
    # Can include non-sensitive details
    "event_type",
    "event_name",
    "event_category",
    "source",
    "component",
    "module",
    "function",
    "level",
    "severity",
    "message",
    "description",
    "details",
    "context",
    "metadata",
    "tags",
    "labels",
    "annotations",

    # Error details (without PII)
    "error_type",
    "error_code",
    "error_class",
    "stack_trace_hash",  # Hash only, not full stack trace

    # Performance details
    "operation",
    "operation_type",
    "resource_type",
    "resource_id",
    "correlation_id",
    "trace_id",
    "span_id",
    "parent_span_id",
})


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ValidationViolation:
    """A validation violation."""
    violation_type: ViolationType
    severity: ValidationSeverity
    field_path: str
    message: str
    blocked: bool = True
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ValidationResult:
    """Result of telemetry validation."""
    valid: bool
    violations: List[ValidationViolation] = field(default_factory=list)
    blocked: bool = False
    sanitized_payload: Optional[Dict[str, Any]] = None
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def critical_violations(self) -> List[ValidationViolation]:
        """Get critical violations that caused blocking."""
        return [v for v in self.violations if v.severity == ValidationSeverity.CRITICAL]

    @property
    def high_violations(self) -> List[ValidationViolation]:
        """Get high severity violations."""
        return [v for v in self.violations if v.severity == ValidationSeverity.HIGH]


# ============================================================================
# Telemetry Validator
# ============================================================================

class TelemetryValidator:
    """
    Validates telemetry payloads before cloud ingestion.

    CRITICAL: This validator ensures no order/intent data reaches the cloud.
    It implements a defense-in-depth approach with multiple validation layers.
    """

    def __init__(self, strict_mode: bool = True):
        """
        Initialize validator.

        Args:
            strict_mode: If True, reject on any prohibited field.
                         If False, sanitize and allow (not recommended for production).
        """
        self.strict_mode = strict_mode
        self._compiled_patterns = self._compile_credential_patterns()

    def _compile_credential_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """Compile credential detection patterns."""
        return [
            (re.compile(pattern, re.IGNORECASE), desc)
            for pattern, desc in BROKER_CREDENTIAL_PATTERNS
        ]

    def validate(
        self,
        payload: Dict[str, Any],
        telemetry_level: str,
        redaction_applied: bool = True,
    ) -> ValidationResult:
        """
        Validate a telemetry payload.

        Args:
            payload: Telemetry payload to validate
            telemetry_level: Level of telemetry (AGGREGATED, DETAILED_NON_SENSITIVE, etc.)
            redaction_applied: Whether redaction has been applied

        Returns:
            ValidationResult with violations and whether to block

        CRITICAL: Always blocks on prohibited order/intent fields.
        """
        violations: List[ValidationViolation] = []
        blocked = False

        # 1. Validate redaction flag
        if telemetry_level != TelemetryLevel.AGGREGATED.value and not redaction_applied:
            violations.append(ValidationViolation(
                violation_type=ViolationType.MISSING_REDACTION,
                severity=ValidationSeverity.CRITICAL,
                field_path="redaction_applied",
                message=f"Non-aggregated telemetry requires redaction_applied=True",
                blocked=True,
            ))
            blocked = True

        # 2. Scan for prohibited order/intent fields
        order_violations = self._scan_prohibited_fields(
            payload, PROHIBITED_ORDER_FIELDS, "order/intent"
        )
        if order_violations:
            violations.extend(order_violations)
            blocked = True

        # 3. Scan for broker credentials (CRITICAL)
        credential_violations = self._scan_credential_patterns(payload)
        if credential_violations:
            violations.extend(credential_violations)
            blocked = True

        # 4. Scan for PII if not aggregated
        if telemetry_level != TelemetryLevel.AGGREGATED.value:
            pii_violations = self._scan_prohibited_fields(
                payload, PROHIBITED_PII_FIELDS, "PII"
            )
            if pii_violations:
                violations.extend(pii_violations)
                blocked = True

        # 5. Validate field allowlist based on telemetry level
        if self.strict_mode:
            allowed_fields = self._get_allowed_fields(telemetry_level)
            unknown_violations = self._check_unknown_fields(payload, allowed_fields)
            if unknown_violations:
                violations.extend(unknown_violations)
                # Unknown fields only block in strict mode for non-aggregated
                if telemetry_level != TelemetryLevel.AGGREGATED.value:
                    blocked = True

        # 6. Deep scan for intent injection
        intent_violations = self._deep_scan_intent_injection(payload)
        if intent_violations:
            violations.extend(intent_violations)
            blocked = True

        # Create sanitized payload if not blocking
        sanitized = None
        if not blocked and not self.strict_mode:
            sanitized = self._sanitize_payload(payload, telemetry_level)

        return ValidationResult(
            valid=not blocked,
            violations=violations,
            blocked=blocked,
            sanitized_payload=sanitized,
        )

    def _scan_prohibited_fields(
        self,
        data: Any,
        prohibited: FrozenSet[str],
        category: str,
        path: str = "",
    ) -> List[ValidationViolation]:
        """Recursively scan for prohibited fields."""
        violations = []

        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                key_lower = key.lower()

                # Check if key is prohibited
                if key_lower in prohibited:
                    violations.append(ValidationViolation(
                        violation_type=ViolationType.PROHIBITED_FIELD,
                        severity=ValidationSeverity.CRITICAL,
                        field_path=current_path,
                        message=f"Prohibited {category} field detected: {key}",
                        blocked=True,
                    ))

                # Recurse into nested structures
                violations.extend(
                    self._scan_prohibited_fields(value, prohibited, category, current_path)
                )

        elif isinstance(data, list):
            for i, item in enumerate(data):
                violations.extend(
                    self._scan_prohibited_fields(item, prohibited, category, f"{path}[{i}]")
                )

        return violations

    def _scan_credential_patterns(
        self,
        data: Any,
        path: str = "",
    ) -> List[ValidationViolation]:
        """Scan for broker credential patterns."""
        violations = []

        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key

                # Check key against credential patterns
                for pattern, desc in self._compiled_patterns:
                    if pattern.search(key):
                        violations.append(ValidationViolation(
                            violation_type=ViolationType.BROKER_CREDENTIALS,
                            severity=ValidationSeverity.CRITICAL,
                            field_path=current_path,
                            message=f"Potential broker credential detected: {desc}",
                            blocked=True,
                        ))

                # Also check if value looks like a credential
                if isinstance(value, str) and len(value) >= 16:
                    # Check for patterns that look like API keys
                    if self._looks_like_credential(value):
                        violations.append(ValidationViolation(
                            violation_type=ViolationType.BROKER_CREDENTIALS,
                            severity=ValidationSeverity.CRITICAL,
                            field_path=current_path,
                            message=f"Value appears to be a credential or token",
                            blocked=True,
                        ))

                # Recurse
                violations.extend(self._scan_credential_patterns(value, current_path))

        elif isinstance(data, list):
            for i, item in enumerate(data):
                violations.extend(
                    self._scan_credential_patterns(item, f"{path}[{i}]")
                )

        return violations

    def _looks_like_credential(self, value: str) -> bool:
        """Check if a string looks like a credential."""
        # Check for common credential patterns
        patterns = [
            r'^[A-Za-z0-9+/]{32,}={0,2}$',  # Base64
            r'^[a-f0-9]{32,}$',              # Hex
            r'^sk-[a-zA-Z0-9]{20,}$',        # API key pattern
            r'^pk-[a-zA-Z0-9]{20,}$',        # Public key pattern
            r'^[A-Z0-9]{20,}$',              # AWS-style key
        ]

        for pattern in patterns:
            if re.match(pattern, value):
                return True

        return False

    def _deep_scan_intent_injection(
        self,
        data: Any,
        path: str = "",
    ) -> List[ValidationViolation]:
        """
        Deep scan for intent injection attempts.

        Catches attempts to embed order-like structures in nested data.
        """
        violations = []

        if isinstance(data, dict):
            # Check for structures that look like orders
            keys_lower = {k.lower() for k in data.keys()}

            # Order-like structure detection
            order_indicators = {"side", "quantity", "price"}
            if order_indicators.issubset(keys_lower):
                violations.append(ValidationViolation(
                    violation_type=ViolationType.INTENT_INJECTION,
                    severity=ValidationSeverity.CRITICAL,
                    field_path=path or "(root)",
                    message="Detected order-like structure with side/quantity/price",
                    blocked=True,
                ))

            # Intent-like structure detection
            intent_indicators = {"signal", "target"}
            if intent_indicators.intersection(keys_lower):
                if any(k in keys_lower for k in ("position", "allocation", "order")):
                    violations.append(ValidationViolation(
                        violation_type=ViolationType.INTENT_INJECTION,
                        severity=ValidationSeverity.CRITICAL,
                        field_path=path or "(root)",
                        message="Detected trading intent structure",
                        blocked=True,
                    ))

            # Recurse
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                violations.extend(self._deep_scan_intent_injection(value, current_path))

        elif isinstance(data, list):
            for i, item in enumerate(data):
                violations.extend(
                    self._deep_scan_intent_injection(item, f"{path}[{i}]")
                )

        return violations

    def _get_allowed_fields(self, telemetry_level: str) -> FrozenSet[str]:
        """Get allowed fields for telemetry level."""
        if telemetry_level == TelemetryLevel.AGGREGATED.value:
            return ALLOWED_AGGREGATED_FIELDS
        elif telemetry_level == TelemetryLevel.DETAILED_NON_SENSITIVE.value:
            return ALLOWED_AGGREGATED_FIELDS | ALLOWED_DETAILED_FIELDS
        else:
            # RAW_ORDER_EVENTS - enterprise only, very restricted
            return ALLOWED_AGGREGATED_FIELDS

    def _check_unknown_fields(
        self,
        data: Dict[str, Any],
        allowed: FrozenSet[str],
        path: str = "",
    ) -> List[ValidationViolation]:
        """Check for unknown fields not in allowlist."""
        violations = []

        for key in data.keys():
            current_path = f"{path}.{key}" if path else key
            if key.lower() not in allowed:
                violations.append(ValidationViolation(
                    violation_type=ViolationType.UNKNOWN_FIELD,
                    severity=ValidationSeverity.MEDIUM,
                    field_path=current_path,
                    message=f"Unknown field not in allowlist: {key}",
                    blocked=False,  # Unknown fields don't block by default
                ))

        return violations

    def _sanitize_payload(
        self,
        payload: Dict[str, Any],
        telemetry_level: str,
    ) -> Dict[str, Any]:
        """
        Sanitize payload by removing prohibited fields.

        Note: This is only used when strict_mode=False (not recommended).
        """
        allowed = self._get_allowed_fields(telemetry_level)
        prohibited = PROHIBITED_ORDER_FIELDS | PROHIBITED_PII_FIELDS

        def sanitize_recursive(data: Any) -> Any:
            if isinstance(data, dict):
                return {
                    k: sanitize_recursive(v)
                    for k, v in data.items()
                    if k.lower() not in prohibited
                }
            elif isinstance(data, list):
                return [sanitize_recursive(item) for item in data]
            else:
                return data

        return sanitize_recursive(payload)


# ============================================================================
# Redaction Enforcer
# ============================================================================

class RedactionEnforcer:
    """
    Enforces mandatory redaction rules for telemetry.

    Ensures that non-aggregated telemetry has proper redaction applied.
    """

    # Fields that must be redacted/hashed
    MUST_REDACT_FIELDS: FrozenSet[str] = frozenset({
        "ip_address",
        "user_agent",
        "device_id",
        "session_id",
        "request_id",
        "correlation_id",
        "stack_trace",
        "error_message",
    })

    # Redaction markers to look for
    REDACTION_MARKERS: Set[str] = {
        "[REDACTED]",
        "[HASHED]",
        "[REMOVED]",
        "[SANITIZED]",
        "***",
        "XXX",
    }

    def check_redaction(
        self,
        payload: Dict[str, Any],
        telemetry_level: str,
    ) -> Tuple[bool, List[str]]:
        """
        Check if payload has proper redaction.

        Args:
            payload: Telemetry payload
            telemetry_level: Level of telemetry

        Returns:
            (is_properly_redacted, list_of_unredacted_fields)
        """
        if telemetry_level == TelemetryLevel.AGGREGATED.value:
            # Aggregated doesn't need field-level redaction
            return True, []

        unredacted = []
        self._check_redaction_recursive(payload, "", unredacted)

        return len(unredacted) == 0, unredacted

    def _check_redaction_recursive(
        self,
        data: Any,
        path: str,
        unredacted: List[str],
    ) -> None:
        """Recursively check for redaction."""
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                key_lower = key.lower()

                # Check if this field should be redacted
                if key_lower in self.MUST_REDACT_FIELDS:
                    if isinstance(value, str):
                        # Check for redaction markers
                        has_marker = any(
                            marker in value.upper()
                            for marker in self.REDACTION_MARKERS
                        )
                        # Or check if it's hashed (64 char hex)
                        is_hashed = bool(re.match(r'^[a-f0-9]{64}$', value))

                        if not has_marker and not is_hashed:
                            unredacted.append(current_path)

                # Recurse
                self._check_redaction_recursive(value, current_path, unredacted)

        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._check_redaction_recursive(item, f"{path}[{i}]", unredacted)


# ============================================================================
# Convenience Functions
# ============================================================================

def validate_telemetry_payload(
    payload: Dict[str, Any],
    telemetry_level: str = TelemetryLevel.AGGREGATED.value,
    redaction_applied: bool = True,
    strict_mode: bool = True,
) -> ValidationResult:
    """
    Convenience function to validate a telemetry payload.

    Args:
        payload: Telemetry payload to validate
        telemetry_level: Level of telemetry
        redaction_applied: Whether redaction has been applied
        strict_mode: Whether to use strict validation

    Returns:
        ValidationResult
    """
    validator = TelemetryValidator(strict_mode=strict_mode)
    return validator.validate(payload, telemetry_level, redaction_applied)


def assert_no_order_fields(payload: Dict[str, Any]) -> None:
    """
    Assert that payload contains no order/intent fields.

    Raises:
        ValueError: If prohibited fields are found
    """
    validator = TelemetryValidator(strict_mode=True)
    result = validator.validate(
        payload,
        TelemetryLevel.AGGREGATED.value,
        redaction_applied=True,
    )

    if not result.valid:
        violations = [
            f"{v.field_path}: {v.message}"
            for v in result.violations
            if v.violation_type in (ViolationType.PROHIBITED_FIELD, ViolationType.INTENT_INJECTION)
        ]
        if violations:
            raise ValueError(
                f"Prohibited order/intent fields detected:\n" + "\n".join(violations)
            )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    "ValidationSeverity",
    "ViolationType",
    # Data classes
    "ValidationViolation",
    "ValidationResult",
    # Classes
    "TelemetryValidator",
    "RedactionEnforcer",
    # Functions
    "validate_telemetry_payload",
    "assert_no_order_fields",
    # Constants
    "PROHIBITED_ORDER_FIELDS",
    "PROHIBITED_PII_FIELDS",
    "ALLOWED_AGGREGATED_FIELDS",
    "ALLOWED_DETAILED_FIELDS",
]
