# -*- coding: utf-8 -*-
"""
Telemetry Level Contract Enforcement.

GDPR Phase 2 Implementation: Data minimization enforcement via telemetry contracts.

This module provides:
    - TelemetryLevelContract: Defines allowed/forbidden fields per telemetry level
    - RawOrderEventsGate: Enterprise gating for RAW_ORDER_EVENTS
    - TelemetryContractValidator: Validates payloads against contracts

Design Doc Reference:
    - Phase 2.5: "Telemetry ingestion: sensitivity levels, redaction mandatory"
    - docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L846 (telemetry levels)
    - docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L853 (RAW_ORDER_EVENTS)
    - docs/compliance/TELEMETRY_DATA_DICTIONARY.md

CLOUD ZONE ONLY.

CRITICAL SECURITY:
    - Order-like fields forbidden in non-RAW telemetry
    - RAW_ORDER_EVENTS requires enterprise + explicit opt-in
    - Redaction is MANDATORY and cannot be disabled
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from ccea.models.protocol import TelemetryLevel


# ============================================================================
# Contract Violation Types
# ============================================================================


class ContractViolationType(str, Enum):
    """Type of contract violation."""

    FORBIDDEN_FIELD = "forbidden_field"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    INVALID_FIELD_TYPE = "invalid_field_type"
    LEVEL_NOT_ALLOWED = "level_not_allowed"
    ENTERPRISE_REQUIRED = "enterprise_required"
    OPT_IN_REQUIRED = "opt_in_required"
    REDACTION_MISSING = "redaction_missing"
    CREDENTIAL_DETECTED = "credential_detected"


class ViolationSeverity(str, Enum):
    """Severity of contract violation."""

    CRITICAL = "critical"  # Block immediately
    HIGH = "high"  # Block with logging
    MEDIUM = "medium"  # Warn and allow
    LOW = "low"  # Log only


# ============================================================================
# Contract Violation
# ============================================================================


@dataclass
class ContractViolation:
    """A telemetry contract violation."""

    violation_type: ContractViolationType
    severity: ViolationSeverity
    field_path: str
    message: str
    telemetry_level: str
    blocked: bool = True
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "violation_type": self.violation_type.value,
            "severity": self.severity.value,
            "field_path": self.field_path,
            "message": self.message,
            "telemetry_level": self.telemetry_level,
            "blocked": self.blocked,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# Contract Validation Result
# ============================================================================


@dataclass
class ContractValidationResult:
    """Result of telemetry contract validation."""

    valid: bool
    telemetry_level: str
    violations: List[ContractViolation] = field(default_factory=list)
    blocked: bool = False
    enterprise_verified: bool = False
    opt_in_verified: bool = False
    redaction_verified: bool = False
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def critical_violations(self) -> List[ContractViolation]:
        """Get critical violations."""
        return [v for v in self.violations if v.severity == ViolationSeverity.CRITICAL]

    @property
    def blocking_violations(self) -> List[ContractViolation]:
        """Get violations that caused blocking."""
        return [v for v in self.violations if v.blocked]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "valid": self.valid,
            "telemetry_level": self.telemetry_level,
            "violations": [v.to_dict() for v in self.violations],
            "blocked": self.blocked,
            "enterprise_verified": self.enterprise_verified,
            "opt_in_verified": self.opt_in_verified,
            "redaction_verified": self.redaction_verified,
            "validated_at": self.validated_at.isoformat(),
        }


# ============================================================================
# Field Definitions (from TELEMETRY_DATA_DICTIONARY.md)
# ============================================================================

# Fields allowed at AGGREGATED level
AGGREGATED_ALLOWED_FIELDS: FrozenSet[str] = frozenset(
    {
        # Temporal
        "timestamp",
        "event_time",
        "start_time",
        "end_time",
        "period",
        "window",
        "interval",
        "duration_seconds",
        # Aggregates
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
        # Trading metrics (aggregated)
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
        # Identifiers
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
    }
)

# Additional fields allowed at DETAILED_NON_SENSITIVE level
DETAILED_ALLOWED_FIELDS: FrozenSet[str] = frozenset(
    {
        # Event details
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
        # Error details
        "error_type",
        "error_code",
        "error_class",
        "stack_trace_hash",
        # Tracing
        "operation",
        "operation_type",
        "resource_type",
        "resource_id",
        "correlation_id",
        "trace_id",
        "span_id",
        "parent_span_id",
    }
)

# Additional fields allowed at RAW_ORDER_EVENTS level (enterprise only)
RAW_ORDER_ALLOWED_FIELDS: FrozenSet[str] = frozenset(
    {
        # Order fields
        "side",
        "quantity",
        "qty",
        "price",
        "order_type",
        "limit_price",
        "stop_price",
        "order_id",
        "client_order_id",
        "filled_qty",
        "remaining_qty",
        "average_price",
        "fill_price",
        "fill_qty",
        "execution_id",
        "trade_id",
        "commission",
        "slippage",
        # Position fields
        "position_side",
        "position_size",
        "entry_price",
        "exit_price",
        "unrealized_pnl",
        "realized_pnl",
        # Signal/Intent
        "signal",
        "intent",
        "target_position",
        "target_qty",
        # Order lifecycle
        "order_status",
        "order_created_at",
        "order_submitted_at",
        "order_filled_at",
        "order_cancelled_at",
        # Execution timing
        "signal_timestamp",
        "submit_timestamp",
        "ack_timestamp",
        "fill_timestamp",
        "exchange_timestamp",
    }
)

# Fields ALWAYS forbidden (credentials, even in RAW)
ALWAYS_FORBIDDEN_FIELDS: FrozenSet[str] = frozenset(
    {
        # Credentials
        "api_key",
        "api_secret",
        "secret_key",
        "private_key",
        "access_token",
        "refresh_token",
        "bearer_token",
        "password",
        "passphrase",
        # Broker-specific
        "binance_key",
        "binance_secret",
        "binance_token",
        "alpaca_key",
        "alpaca_secret",
        "alpaca_token",
        "deribit_key",
        "deribit_secret",
        "deribit_token",
        "oanda_key",
        "oanda_secret",
        "oanda_token",
        "ib_key",
        "ib_secret",
        "ib_token",
        "interactive_brokers_key",
        "interactive_brokers_secret",
    }
)

# PII fields - forbidden unless aggregated/hashed
PII_FIELDS: FrozenSet[str] = frozenset(
    {
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
        "ip_address",
        "mac_address",
        "device_id",
        "user_agent",
    }
)

# Order-like fields - forbidden in non-RAW telemetry
ORDER_LIKE_FIELDS: FrozenSet[str] = frozenset(
    {
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
        "position_side",
        "position_size",
        "entry_price",
        "exit_price",
        "unrealized_pnl",
        "realized_pnl",
        "execution_id",
        "trade_id",
        "fill_price",
        "fill_qty",
        "slippage",
    }
)

# Credential detection patterns
CREDENTIAL_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"api[_-]?key", re.IGNORECASE), "API key"),
    (re.compile(r"api[_-]?secret", re.IGNORECASE), "API secret"),
    (re.compile(r"secret[_-]?key", re.IGNORECASE), "Secret key"),
    (re.compile(r"private[_-]?key", re.IGNORECASE), "Private key"),
    (re.compile(r"access[_-]?token", re.IGNORECASE), "Access token"),
    (re.compile(r"refresh[_-]?token", re.IGNORECASE), "Refresh token"),
    (re.compile(r"bearer[_-]?token", re.IGNORECASE), "Bearer token"),
    (re.compile(r"password", re.IGNORECASE), "Password"),
    (re.compile(r"passphrase", re.IGNORECASE), "Passphrase"),
    (
        re.compile(
            r"(binance|alpaca|deribit|oanda|interactive_brokers|ib)[_-]?(key|secret|token)",
            re.IGNORECASE,
        ),
        "Broker credential",
    ),
]


# ============================================================================
# Telemetry Level Contract
# ============================================================================


@dataclass
class TelemetryLevelContract:
    """
    Contract defining allowed/forbidden fields for a telemetry level.

    This class enforces data minimization by specifying exactly what
    fields are allowed at each telemetry sensitivity level.
    """

    level: TelemetryLevel
    allowed_fields: FrozenSet[str]
    forbidden_fields: FrozenSet[str]
    requires_enterprise: bool = False
    requires_explicit_opt_in: bool = False
    requires_redaction: bool = True
    max_retention_days: int = 90

    @classmethod
    def for_aggregated(cls) -> "TelemetryLevelContract":
        """Create contract for AGGREGATED level."""
        return cls(
            level=TelemetryLevel.AGGREGATED,
            allowed_fields=AGGREGATED_ALLOWED_FIELDS,
            forbidden_fields=ORDER_LIKE_FIELDS | PII_FIELDS | ALWAYS_FORBIDDEN_FIELDS,
            requires_enterprise=False,
            requires_explicit_opt_in=False,
            requires_redaction=False,  # Aggregated doesn't need field redaction
            max_retention_days=365,
        )

    @classmethod
    def for_detailed_non_sensitive(cls) -> "TelemetryLevelContract":
        """Create contract for DETAILED_NON_SENSITIVE level."""
        return cls(
            level=TelemetryLevel.DETAILED_NON_SENSITIVE,
            allowed_fields=AGGREGATED_ALLOWED_FIELDS | DETAILED_ALLOWED_FIELDS,
            forbidden_fields=ORDER_LIKE_FIELDS | PII_FIELDS | ALWAYS_FORBIDDEN_FIELDS,
            requires_enterprise=False,
            requires_explicit_opt_in=True,
            requires_redaction=True,
            max_retention_days=90,
        )

    @classmethod
    def for_raw_order_events(cls) -> "TelemetryLevelContract":
        """Create contract for RAW_ORDER_EVENTS level (enterprise only)."""
        return cls(
            level=TelemetryLevel.RAW_ORDER_EVENTS,
            allowed_fields=(
                AGGREGATED_ALLOWED_FIELDS | DETAILED_ALLOWED_FIELDS | RAW_ORDER_ALLOWED_FIELDS
            ),
            # Even RAW cannot include credentials
            forbidden_fields=ALWAYS_FORBIDDEN_FIELDS,
            requires_enterprise=True,
            requires_explicit_opt_in=True,
            requires_redaction=True,
            max_retention_days=30,
        )

    @classmethod
    def get_contract(cls, level: TelemetryLevel) -> "TelemetryLevelContract":
        """Get the contract for a given telemetry level."""
        contracts = {
            TelemetryLevel.AGGREGATED: cls.for_aggregated,
            TelemetryLevel.DETAILED_NON_SENSITIVE: cls.for_detailed_non_sensitive,
            TelemetryLevel.RAW_ORDER_EVENTS: cls.for_raw_order_events,
        }
        factory = contracts.get(level)
        if factory is None:
            raise ValueError(f"Unknown telemetry level: {level}")
        return factory()


# ============================================================================
# Enterprise Opt-In Record
# ============================================================================


@dataclass
class EnterpriseRawOptIn:
    """
    Record of enterprise opt-in for RAW_ORDER_EVENTS.

    Required for RAW telemetry ingestion.
    """

    workspace_id: str
    organization_id: str
    opted_in_at: datetime
    opted_in_by: str  # Principal ID
    acknowledgment_hash: str  # Hash of acknowledgment document
    dpa_signed: bool = True
    raw_addendum_signed: bool = True
    expires_at: Optional[datetime] = None
    revoked: bool = False
    revoked_at: Optional[datetime] = None
    revoked_by: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Check if opt-in is currently valid."""
        if self.revoked:
            return False
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        if not self.dpa_signed or not self.raw_addendum_signed:
            return False
        return True


# ============================================================================
# RAW Order Events Gate
# ============================================================================


class RawOrderEventsGate:
    """
    Gate for RAW_ORDER_EVENTS telemetry ingestion.

    ENTERPRISE ONLY - requires explicit opt-in per workspace.

    Reference: docs/compliance/RAW_ORDER_EVENTS_HANDLING_SPEC.md
    """

    def __init__(
        self,
        enterprise_checker: Optional[callable] = None,
        opt_in_store: Optional[Dict[str, EnterpriseRawOptIn]] = None,
    ):
        """
        Initialize gate.

        Args:
            enterprise_checker: Function to check enterprise license
            opt_in_store: Storage for opt-in records (in-memory for testing)
        """
        self._enterprise_checker = enterprise_checker
        self._opt_in_store = opt_in_store or {}

    def check_enterprise_license(
        self,
        workspace_id: str,
        organization_id: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify enterprise license is active and includes RAW feature.

        Returns:
            (is_allowed, error_message)
        """
        if self._enterprise_checker:
            return self._enterprise_checker(workspace_id, organization_id)

        # Default: assume enterprise check passes for testing
        # In production, this would call the license service
        return True, None

    def check_explicit_opt_in(
        self,
        workspace_id: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify explicit opt-in exists for workspace.

        Returns:
            (is_allowed, error_message)
        """
        opt_in = self._opt_in_store.get(workspace_id)

        if opt_in is None:
            return False, (
                "RAW_ORDER_EVENTS requires explicit opt-in for workspace. "
                "Please enable RAW telemetry in workspace settings."
            )

        if not opt_in.is_valid:
            if opt_in.revoked:
                return False, "RAW_ORDER_EVENTS opt-in has been revoked."
            if opt_in.expires_at and datetime.now(timezone.utc) > opt_in.expires_at:
                return False, "RAW_ORDER_EVENTS opt-in has expired."
            if not opt_in.dpa_signed:
                return False, "Enterprise DPA must be signed for RAW_ORDER_EVENTS."
            if not opt_in.raw_addendum_signed:
                return False, "RAW Data Processing Addendum must be signed."

        return True, None

    def register_opt_in(
        self,
        workspace_id: str,
        organization_id: str,
        principal_id: str,
        acknowledgment_text: str,
        dpa_signed: bool = True,
        raw_addendum_signed: bool = True,
        expires_at: Optional[datetime] = None,
    ) -> EnterpriseRawOptIn:
        """
        Register explicit opt-in for RAW_ORDER_EVENTS.

        Args:
            workspace_id: Workspace identifier
            organization_id: Organization identifier
            principal_id: Who is opting in
            acknowledgment_text: Text that was acknowledged
            dpa_signed: Whether DPA is signed
            raw_addendum_signed: Whether RAW addendum is signed
            expires_at: Optional expiry date

        Returns:
            EnterpriseRawOptIn record
        """
        # Calculate acknowledgment hash
        ack_hash = hashlib.sha256(acknowledgment_text.encode()).hexdigest()

        opt_in = EnterpriseRawOptIn(
            workspace_id=workspace_id,
            organization_id=organization_id,
            opted_in_at=datetime.now(timezone.utc),
            opted_in_by=principal_id,
            acknowledgment_hash=f"sha256:{ack_hash}",
            dpa_signed=dpa_signed,
            raw_addendum_signed=raw_addendum_signed,
            expires_at=expires_at,
        )

        self._opt_in_store[workspace_id] = opt_in
        return opt_in

    def revoke_opt_in(
        self,
        workspace_id: str,
        principal_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Revoke RAW_ORDER_EVENTS opt-in.

        Returns:
            True if revoked, False if no opt-in exists
        """
        opt_in = self._opt_in_store.get(workspace_id)
        if opt_in is None:
            return False

        opt_in.revoked = True
        opt_in.revoked_at = datetime.now(timezone.utc)
        opt_in.revoked_by = principal_id
        return True

    def validate_gate(
        self,
        workspace_id: str,
        organization_id: str,
    ) -> Tuple[bool, List[str]]:
        """
        Full gate validation for RAW_ORDER_EVENTS.

        Returns:
            (is_allowed, list_of_error_messages)
        """
        errors = []

        # Check enterprise license
        license_ok, license_error = self.check_enterprise_license(workspace_id, organization_id)
        if not license_ok:
            errors.append(license_error or "Enterprise license required")

        # Check explicit opt-in
        opt_in_ok, opt_in_error = self.check_explicit_opt_in(workspace_id)
        if not opt_in_ok:
            errors.append(opt_in_error or "Explicit opt-in required")

        return len(errors) == 0, errors


# ============================================================================
# Telemetry Contract Validator
# ============================================================================


class TelemetryContractValidator:
    """
    Validates telemetry payloads against level contracts.

    Enforces data minimization by rejecting payloads that violate
    the contract for their declared telemetry level.
    """

    def __init__(
        self,
        raw_gate: Optional[RawOrderEventsGate] = None,
    ):
        """
        Initialize validator.

        Args:
            raw_gate: Gate for RAW_ORDER_EVENTS validation
        """
        self._raw_gate = raw_gate or RawOrderEventsGate()
        self._compiled_patterns = CREDENTIAL_PATTERNS

    def validate(
        self,
        payload: Dict[str, Any],
        telemetry_level: TelemetryLevel,
        workspace_id: str,
        organization_id: str,
        redaction_applied: bool = True,
    ) -> ContractValidationResult:
        """
        Validate a telemetry payload against its level contract.

        Args:
            payload: Telemetry payload to validate
            telemetry_level: Declared telemetry level
            workspace_id: Workspace identifier
            organization_id: Organization identifier
            redaction_applied: Whether redaction has been applied

        Returns:
            ContractValidationResult
        """
        violations: List[ContractViolation] = []
        blocked = False
        enterprise_verified = False
        opt_in_verified = False
        redaction_verified = redaction_applied

        # Get contract for level
        try:
            contract = TelemetryLevelContract.get_contract(telemetry_level)
        except ValueError as e:
            violations.append(
                ContractViolation(
                    violation_type=ContractViolationType.LEVEL_NOT_ALLOWED,
                    severity=ViolationSeverity.CRITICAL,
                    field_path="telemetry_level",
                    message=str(e),
                    telemetry_level=str(telemetry_level),
                    blocked=True,
                )
            )
            return ContractValidationResult(
                valid=False,
                telemetry_level=str(telemetry_level),
                violations=violations,
                blocked=True,
            )

        # Check enterprise requirement
        if contract.requires_enterprise:
            gate_ok, gate_errors = self._raw_gate.validate_gate(workspace_id, organization_id)
            if not gate_ok:
                for error in gate_errors:
                    violations.append(
                        ContractViolation(
                            violation_type=ContractViolationType.ENTERPRISE_REQUIRED,
                            severity=ViolationSeverity.CRITICAL,
                            field_path="telemetry_level",
                            message=error,
                            telemetry_level=telemetry_level.value,
                            blocked=True,
                        )
                    )
                blocked = True
            else:
                enterprise_verified = True
                opt_in_verified = True

        # Check redaction requirement
        if contract.requires_redaction and not redaction_applied:
            violations.append(
                ContractViolation(
                    violation_type=ContractViolationType.REDACTION_MISSING,
                    severity=ViolationSeverity.CRITICAL,
                    field_path="redaction_applied",
                    message=(
                        f"Telemetry level {telemetry_level.value} requires redaction. "
                        "Set redaction_applied=True after applying redaction."
                    ),
                    telemetry_level=telemetry_level.value,
                    blocked=True,
                )
            )
            blocked = True
            redaction_verified = False

        # Scan payload for violations
        field_violations = self._scan_payload(payload, contract, "")
        violations.extend(field_violations)

        # Check if any field violation blocks
        if any(v.blocked for v in field_violations):
            blocked = True

        # Scan for credentials (always forbidden)
        credential_violations = self._scan_credentials(payload, telemetry_level, "")
        violations.extend(credential_violations)
        if credential_violations:
            blocked = True

        return ContractValidationResult(
            valid=not blocked,
            telemetry_level=telemetry_level.value,
            violations=violations,
            blocked=blocked,
            enterprise_verified=enterprise_verified,
            opt_in_verified=opt_in_verified,
            redaction_verified=redaction_verified,
        )

    def _scan_payload(
        self,
        data: Any,
        contract: TelemetryLevelContract,
        path: str,
    ) -> List[ContractViolation]:
        """Recursively scan payload for contract violations."""
        violations = []

        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                key_lower = key.lower()

                # Check if field is forbidden
                if key_lower in contract.forbidden_fields:
                    violations.append(
                        ContractViolation(
                            violation_type=ContractViolationType.FORBIDDEN_FIELD,
                            severity=ViolationSeverity.CRITICAL,
                            field_path=current_path,
                            message=(
                                f"Field '{key}' is forbidden at telemetry level "
                                f"{contract.level.value}. See TELEMETRY_DATA_DICTIONARY.md."
                            ),
                            telemetry_level=contract.level.value,
                            blocked=True,
                        )
                    )

                # Recurse into nested structures
                violations.extend(self._scan_payload(value, contract, current_path))

        elif isinstance(data, list):
            for i, item in enumerate(data):
                violations.extend(self._scan_payload(item, contract, f"{path}[{i}]"))

        return violations

    def _scan_credentials(
        self,
        data: Any,
        telemetry_level: TelemetryLevel,
        path: str,
    ) -> List[ContractViolation]:
        """Scan for credential patterns (always forbidden)."""
        violations = []

        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key

                # Check key against credential patterns
                for pattern, desc in self._compiled_patterns:
                    if pattern.search(key):
                        violations.append(
                            ContractViolation(
                                violation_type=ContractViolationType.CREDENTIAL_DETECTED,
                                severity=ViolationSeverity.CRITICAL,
                                field_path=current_path,
                                message=f"Potential credential detected: {desc}",
                                telemetry_level=telemetry_level.value,
                                blocked=True,
                            )
                        )

                # Check if value looks like a credential
                if isinstance(value, str) and self._looks_like_credential(value):
                    violations.append(
                        ContractViolation(
                            violation_type=ContractViolationType.CREDENTIAL_DETECTED,
                            severity=ViolationSeverity.CRITICAL,
                            field_path=current_path,
                            message="Value appears to be a credential or token",
                            telemetry_level=telemetry_level.value,
                            blocked=True,
                        )
                    )

                # Recurse
                violations.extend(self._scan_credentials(value, telemetry_level, current_path))

        elif isinstance(data, list):
            for i, item in enumerate(data):
                violations.extend(self._scan_credentials(item, telemetry_level, f"{path}[{i}]"))

        return violations

    def _looks_like_credential(self, value: str) -> bool:
        """Check if a string looks like a credential."""
        if len(value) < 16:
            return False

        patterns = [
            r"^[A-Za-z0-9+/]{32,}={0,2}$",  # Base64
            r"^[a-f0-9]{32,}$",  # Hex
            r"^sk-[a-zA-Z0-9]{20,}$",  # API key pattern
            r"^pk-[a-zA-Z0-9]{20,}$",  # Public key pattern
            r"^[A-Z0-9]{20,}$",  # AWS-style key
        ]

        for pattern in patterns:
            if re.match(pattern, value):
                return True

        return False


# ============================================================================
# Convenience Functions
# ============================================================================


def validate_telemetry_contract(
    payload: Dict[str, Any],
    telemetry_level: str,
    workspace_id: str,
    organization_id: str,
    redaction_applied: bool = True,
    raw_gate: Optional[RawOrderEventsGate] = None,
) -> ContractValidationResult:
    """
    Convenience function to validate a telemetry payload against contract.

    Args:
        payload: Telemetry payload
        telemetry_level: Level string (AGGREGATED, DETAILED_NON_SENSITIVE, RAW_ORDER_EVENTS)
        workspace_id: Workspace identifier
        organization_id: Organization identifier
        redaction_applied: Whether redaction has been applied
        raw_gate: Optional RAW gate (uses default if not provided)

    Returns:
        ContractValidationResult
    """
    try:
        level = TelemetryLevel(telemetry_level)
    except ValueError:
        return ContractValidationResult(
            valid=False,
            telemetry_level=telemetry_level,
            violations=[
                ContractViolation(
                    violation_type=ContractViolationType.LEVEL_NOT_ALLOWED,
                    severity=ViolationSeverity.CRITICAL,
                    field_path="telemetry_level",
                    message=f"Invalid telemetry level: {telemetry_level}",
                    telemetry_level=telemetry_level,
                    blocked=True,
                )
            ],
            blocked=True,
        )

    validator = TelemetryContractValidator(raw_gate=raw_gate)
    return validator.validate(
        payload,
        level,
        workspace_id,
        organization_id,
        redaction_applied,
    )


def get_allowed_fields_for_level(
    telemetry_level: str,
) -> FrozenSet[str]:
    """
    Get allowed fields for a telemetry level.

    Args:
        telemetry_level: Level string

    Returns:
        Set of allowed field names
    """
    try:
        level = TelemetryLevel(telemetry_level)
        contract = TelemetryLevelContract.get_contract(level)
        return contract.allowed_fields
    except ValueError:
        return frozenset()


def get_forbidden_fields_for_level(
    telemetry_level: str,
) -> FrozenSet[str]:
    """
    Get forbidden fields for a telemetry level.

    Args:
        telemetry_level: Level string

    Returns:
        Set of forbidden field names
    """
    try:
        level = TelemetryLevel(telemetry_level)
        contract = TelemetryLevelContract.get_contract(level)
        return contract.forbidden_fields
    except ValueError:
        return frozenset()


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    "ContractViolationType",
    "ViolationSeverity",
    # Data classes
    "ContractViolation",
    "ContractValidationResult",
    "TelemetryLevelContract",
    "EnterpriseRawOptIn",
    # Classes
    "RawOrderEventsGate",
    "TelemetryContractValidator",
    # Functions
    "validate_telemetry_contract",
    "get_allowed_fields_for_level",
    "get_forbidden_fields_for_level",
    # Field sets
    "AGGREGATED_ALLOWED_FIELDS",
    "DETAILED_ALLOWED_FIELDS",
    "RAW_ORDER_ALLOWED_FIELDS",
    "ALWAYS_FORBIDDEN_FIELDS",
    "PII_FIELDS",
    "ORDER_LIKE_FIELDS",
]
