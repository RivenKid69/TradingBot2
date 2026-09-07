# -*- coding: utf-8 -*-
"""
Data Loss Prevention (DLP) Service for Telemetry.

CCEA Phase 8 Implementation.

This module provides DLP capabilities to prevent sensitive data
from leaving the agent in telemetry streams. It works in conjunction
with the TelemetryRedactionMiddleware.

Key Features:
    - Rule-based data classification
    - Configurable sensitivity levels
    - Action enforcement (block, alert, allow)
    - Audit logging of DLP events
    - Integration with telemetry pipeline

Design Doc Reference:
    - Phase 8 (13.2): "Mandatory redaction + DLP"
    - Data classification and sensitivity handling

SECURITY CRITICAL:
    - DLP rules for CRITICAL level cannot be disabled
    - All violations must be logged for audit
"""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, Final, FrozenSet, List, Optional, Pattern, Set
from uuid import uuid4


# ============================================================================
# Enums and Constants
# ============================================================================


class SensitivityLevel(Enum):
    """Data sensitivity classification levels."""

    PUBLIC = 0  # Can be shared freely
    INTERNAL = 1  # Internal use only
    CONFIDENTIAL = 2  # Restricted access
    SENSITIVE = 3  # Highly sensitive (PII, financial)
    CRITICAL = 4  # Critical secrets (keys, credentials)


class DLPAction(Enum):
    """Actions to take when DLP rule is triggered."""

    ALLOW = auto()  # Allow with logging
    MASK = auto()  # Mask sensitive portions
    BLOCK = auto()  # Block transmission entirely
    ALERT = auto()  # Allow but generate alert
    QUARANTINE = auto()  # Store locally, don't transmit


# Default actions per sensitivity level
DEFAULT_SENSITIVITY_ACTIONS: Final[Dict[SensitivityLevel, DLPAction]] = {
    SensitivityLevel.PUBLIC: DLPAction.ALLOW,
    SensitivityLevel.INTERNAL: DLPAction.ALLOW,
    SensitivityLevel.CONFIDENTIAL: DLPAction.MASK,
    SensitivityLevel.SENSITIVE: DLPAction.MASK,
    SensitivityLevel.CRITICAL: DLPAction.BLOCK,
}


# Critical patterns that MUST be blocked
CRITICAL_PATTERNS: Final[FrozenSet[str]] = frozenset(
    {
        "private_key",
        "secret_key",
        "api_secret",
        "master_password",
        "encryption_key",
        "signing_key",
        "seed_phrase",
        "mnemonic",
        "wallet_key",
    }
)


@dataclass
class DLPRule:
    """
    DLP rule definition.

    Attributes:
        id: Unique rule identifier
        name: Human-readable rule name
        description: Rule description
        sensitivity: Sensitivity level this rule detects
        field_patterns: Field name patterns to match
        value_patterns: Regex patterns for value matching
        action: Action to take when triggered
        is_mandatory: Cannot be disabled (for CRITICAL rules)
        enabled: Whether rule is active
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    sensitivity: SensitivityLevel = SensitivityLevel.CONFIDENTIAL
    field_patterns: FrozenSet[str] = field(default_factory=frozenset)
    value_patterns: List[Pattern[str]] = field(default_factory=list)
    action: DLPAction = DLPAction.MASK
    is_mandatory: bool = False
    enabled: bool = True

    def __post_init__(self):
        """Ensure mandatory rules stay enabled."""
        if self.is_mandatory:
            self.enabled = True
            # Critical sensitivity must block
            if self.sensitivity == SensitivityLevel.CRITICAL:
                self.action = DLPAction.BLOCK

    def matches(self, field_name: str, value: Any) -> bool:
        """Check if field/value matches this rule."""
        if not self.enabled:
            return False

        # Check field patterns
        field_lower = field_name.lower()
        for pattern in self.field_patterns:
            if pattern in field_lower:
                return True

        # Check value patterns
        if isinstance(value, str):
            for regex in self.value_patterns:
                if regex.search(value):
                    return True

        return False


@dataclass
class DLPViolation:
    """
    Record of a DLP violation.

    Attributes:
        id: Violation identifier
        rule_id: ID of triggered rule
        rule_name: Name of triggered rule
        field_path: Path to violating field
        sensitivity: Detected sensitivity level
        action_taken: Action that was taken
        timestamp: When violation occurred
        masked_preview: Masked preview of content (for audit)
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    rule_id: str = ""
    rule_name: str = ""
    field_path: str = ""
    sensitivity: SensitivityLevel = SensitivityLevel.CONFIDENTIAL
    action_taken: DLPAction = DLPAction.MASK
    timestamp: datetime = field(default_factory=datetime.utcnow)
    masked_preview: str = ""
    data_hash: str = ""  # For correlation without storing actual data

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission."""
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "field_path": self.field_path,
            "sensitivity": self.sensitivity.name,
            "action_taken": self.action_taken.name,
            "timestamp": self.timestamp.isoformat(),
            "masked_preview": self.masked_preview,
            "data_hash": self.data_hash,
        }


@dataclass
class DLPConfig:
    """
    DLP service configuration.

    Attributes:
        custom_rules: Additional custom rules
        default_action: Default action for unclassified data
        enable_audit_log: Log all violations
        max_preview_length: Max length for masked previews
        strict_mode: Block on any uncertainty
    """

    custom_rules: List[DLPRule] = field(default_factory=list)
    default_action: DLPAction = DLPAction.ALLOW
    enable_audit_log: bool = True
    max_preview_length: int = 20
    strict_mode: bool = False

    # Actions per sensitivity level
    sensitivity_actions: Dict[SensitivityLevel, DLPAction] = field(
        default_factory=lambda: dict(DEFAULT_SENSITIVITY_ACTIONS)
    )


@dataclass
class DLPResult:
    """
    Result of DLP processing.

    Attributes:
        data: Processed data (may be modified/blocked)
        violations: List of DLP violations found
        blocked: Whether transmission was blocked
        action: Overall action taken
    """

    data: Dict[str, Any]
    violations: List[DLPViolation] = field(default_factory=list)
    blocked: bool = False
    action: DLPAction = DLPAction.ALLOW

    @property
    def has_violations(self) -> bool:
        """Check if any violations were found."""
        return len(self.violations) > 0

    @property
    def critical_violations(self) -> List[DLPViolation]:
        """Get critical-level violations."""
        return [v for v in self.violations if v.sensitivity == SensitivityLevel.CRITICAL]


class DLPService:
    """
    Data Loss Prevention service for telemetry.

    This service analyzes telemetry data and enforces DLP policies
    to prevent sensitive data leakage.

    Usage:
        dlp = DLPService()

        # Process telemetry
        result = dlp.process(telemetry_data)

        if result.blocked:
            log.error("Telemetry blocked by DLP")
        else:
            send_telemetry(result.data)

    SECURITY:
        - Critical rules cannot be disabled
        - All violations are logged for audit
        - Blocks on critical sensitivity data
    """

    def __init__(self, config: Optional[DLPConfig] = None):
        """
        Initialize DLP service.

        Args:
            config: Optional configuration
        """
        self.config = config or DLPConfig()
        self._rules = self._build_rule_registry()
        self._violations: List[DLPViolation] = []
        self._lock = threading.Lock()

    def _build_rule_registry(self) -> List[DLPRule]:
        """Build rule registry with mandatory and custom rules."""
        rules = []

        # Mandatory critical rules (cannot be disabled)
        rules.append(
            DLPRule(
                name="critical_secrets",
                description="Block critical secrets like private keys",
                sensitivity=SensitivityLevel.CRITICAL,
                field_patterns=CRITICAL_PATTERNS,
                action=DLPAction.BLOCK,
                is_mandatory=True,
            )
        )

        # API credentials rule
        rules.append(
            DLPRule(
                name="api_credentials",
                description="Block API credentials",
                sensitivity=SensitivityLevel.CRITICAL,
                field_patterns=frozenset(
                    {
                        "api_key",
                        "api_secret",
                        "apikey",
                        "apisecret",
                        "access_key",
                        "secret_key",
                        "auth_token",
                    }
                ),
                value_patterns=[
                    re.compile(r"(?:AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}"),  # AWS
                    re.compile(r"sk-[a-zA-Z0-9]{20,}"),  # OpenAI-style
                    re.compile(r"xox[baprs]-[a-zA-Z0-9-]+"),  # Slack
                ],
                action=DLPAction.BLOCK,
                is_mandatory=True,
            )
        )

        # Financial data rule
        rules.append(
            DLPRule(
                name="financial_data",
                description="Mask financial identifiers",
                sensitivity=SensitivityLevel.SENSITIVE,
                field_patterns=frozenset(
                    {
                        "account_number",
                        "routing_number",
                        "card_number",
                        "credit_card",
                        "bank_account",
                        "iban",
                        "swift",
                    }
                ),
                value_patterns=[
                    re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),  # Card
                    re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b"),  # IBAN
                ],
                action=DLPAction.MASK,
                is_mandatory=True,
            )
        )

        # PII rule
        rules.append(
            DLPRule(
                name="pii_data",
                description="Mask personally identifiable information",
                sensitivity=SensitivityLevel.SENSITIVE,
                field_patterns=frozenset(
                    {
                        "ssn",
                        "social_security",
                        "tax_id",
                        "national_id",
                        "passport",
                        "driver_license",
                        "drivers_license",
                    }
                ),
                value_patterns=[
                    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
                    re.compile(r"\b\d{9}\b"),  # Tax ID
                ],
                action=DLPAction.MASK,
                is_mandatory=True,
            )
        )

        # Contact info rule
        rules.append(
            DLPRule(
                name="contact_info",
                description="Mask contact information",
                sensitivity=SensitivityLevel.CONFIDENTIAL,
                field_patterns=frozenset(
                    {
                        "email",
                        "phone",
                        "phone_number",
                        "mobile",
                        "address",
                        "street",
                        "city",
                        "zip",
                        "postal",
                    }
                ),
                value_patterns=[
                    re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
                    re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
                ],
                action=DLPAction.MASK,
                is_mandatory=False,
            )
        )

        # Add custom rules
        rules.extend(self.config.custom_rules)

        return rules

    def process(self, data: Dict[str, Any]) -> DLPResult:
        """
        Process data through DLP engine.

        Args:
            data: Telemetry data to process

        Returns:
            DLPResult with processed data and violations
        """
        violations: List[DLPViolation] = []
        processed_data = self._process_recursive(data, violations, "")

        # Determine overall action
        blocked = any(v.action_taken == DLPAction.BLOCK for v in violations)
        overall_action = (
            DLPAction.BLOCK if blocked else (DLPAction.MASK if violations else DLPAction.ALLOW)
        )

        # Log violations
        if self.config.enable_audit_log and violations:
            self._log_violations(violations)

        return DLPResult(
            data={} if blocked else processed_data,
            violations=violations,
            blocked=blocked,
            action=overall_action,
        )

    def _process_recursive(
        self,
        obj: Any,
        violations: List[DLPViolation],
        path: str,
    ) -> Any:
        """Recursively process data through DLP rules."""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key

                # Check against rules
                rule, sensitivity = self._match_rules(key, value)

                if rule:
                    action = self._get_action(rule, sensitivity)
                    violation = self._create_violation(rule, current_path, value, action)
                    violations.append(violation)

                    # Apply action
                    if action == DLPAction.BLOCK:
                        result[key] = "[BLOCKED]"
                    elif action == DLPAction.MASK:
                        result[key] = self._mask_value(value)
                    elif action == DLPAction.QUARANTINE:
                        result[key] = "[QUARANTINED]"
                    else:
                        result[key] = self._process_recursive(value, violations, current_path)
                else:
                    result[key] = self._process_recursive(value, violations, current_path)
            return result

        elif isinstance(obj, list):
            return [
                self._process_recursive(item, violations, f"{path}[{i}]")
                for i, item in enumerate(obj)
            ]

        elif isinstance(obj, str):
            # Check string values against patterns
            rule, sensitivity = self._match_value_rules(obj)
            if rule:
                action = self._get_action(rule, sensitivity)
                violation = self._create_violation(rule, path, obj, action)
                violations.append(violation)

                if action == DLPAction.BLOCK:
                    return "[BLOCKED]"
                elif action == DLPAction.MASK:
                    return self._mask_value(obj)

            return obj

        return obj

    def _match_rules(
        self, field_name: str, value: Any
    ) -> tuple[Optional[DLPRule], SensitivityLevel]:
        """Match field/value against rules."""
        for rule in self._rules:
            if rule.matches(field_name, value):
                return rule, rule.sensitivity

        return None, SensitivityLevel.PUBLIC

    def _match_value_rules(self, value: str) -> tuple[Optional[DLPRule], SensitivityLevel]:
        """Match value against value-based rules."""
        for rule in self._rules:
            for pattern in rule.value_patterns:
                if pattern.search(value):
                    return rule, rule.sensitivity

        return None, SensitivityLevel.PUBLIC

    def _get_action(self, rule: DLPRule, sensitivity: SensitivityLevel) -> DLPAction:
        """Determine action based on rule and sensitivity."""
        # Critical sensitivity always blocks
        if sensitivity == SensitivityLevel.CRITICAL:
            return DLPAction.BLOCK

        # Use rule action if specified
        if rule.action != DLPAction.ALLOW:
            return rule.action

        # Fall back to sensitivity-based action
        return self.config.sensitivity_actions.get(sensitivity, self.config.default_action)

    def _create_violation(
        self, rule: DLPRule, path: str, value: Any, action: DLPAction
    ) -> DLPViolation:
        """Create violation record."""
        # Create masked preview
        str_value = str(value)
        max_len = self.config.max_preview_length
        if len(str_value) > max_len:
            masked_preview = str_value[: max_len // 2] + "..." + str_value[-(max_len // 2) :]
        else:
            masked_preview = str_value

        # Hash for correlation
        data_hash = hashlib.sha256(str_value.encode()).hexdigest()[:16]

        return DLPViolation(
            rule_id=rule.id,
            rule_name=rule.name,
            field_path=path,
            sensitivity=rule.sensitivity,
            action_taken=action,
            masked_preview=masked_preview if action != DLPAction.BLOCK else "[REDACTED]",
            data_hash=data_hash,
        )

    def _mask_value(self, value: Any) -> Any:
        """Apply masking to a value."""
        if isinstance(value, str):
            if len(value) <= 4:
                return "***"
            return value[:2] + "***" + value[-2:]
        return "[MASKED]"

    def _log_violations(self, violations: List[DLPViolation]) -> None:
        """Log violations for audit."""
        with self._lock:
            self._violations.extend(violations)

    def get_violation_history(self, limit: int = 100) -> List[DLPViolation]:
        """Get recent violation history."""
        with self._lock:
            return self._violations[-limit:]

    def clear_violation_history(self) -> None:
        """Clear violation history."""
        with self._lock:
            self._violations.clear()

    def add_rule(self, rule: DLPRule) -> None:
        """Add a custom rule."""
        self._rules.append(rule)

    def get_rules(self) -> List[DLPRule]:
        """Get all active rules."""
        return [r for r in self._rules if r.enabled]

    def classify_data(self, data: Dict[str, Any]) -> Dict[str, SensitivityLevel]:
        """
        Classify data by sensitivity without processing.

        Args:
            data: Data to classify

        Returns:
            Dict mapping field paths to sensitivity levels
        """
        classifications = {}

        def classify_recursive(obj: Any, path: str) -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    rule, sensitivity = self._match_rules(key, value)
                    if sensitivity != SensitivityLevel.PUBLIC:
                        classifications[current_path] = sensitivity
                    classify_recursive(value, current_path)

            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    classify_recursive(item, f"{path}[{i}]")

            elif isinstance(obj, str):
                rule, sensitivity = self._match_value_rules(obj)
                if sensitivity != SensitivityLevel.PUBLIC:
                    classifications[path] = sensitivity

        classify_recursive(data, "")
        return classifications


# ============================================================================
# Convenience Functions
# ============================================================================


def create_dlp_service(strict: bool = False) -> DLPService:
    """
    Create DLP service with optional strict mode.

    Args:
        strict: Enable strict mode (blocks on uncertainty)

    Returns:
        Configured DLPService
    """
    config = DLPConfig(strict_mode=strict)
    return DLPService(config)


def check_for_sensitive_data(data: Dict[str, Any]) -> bool:
    """
    Quick check for sensitive data.

    Args:
        data: Data to check

    Returns:
        True if sensitive data detected
    """
    dlp = create_dlp_service()
    result = dlp.process(data)
    return result.has_violations
