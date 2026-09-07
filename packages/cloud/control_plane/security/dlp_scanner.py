# -*- coding: utf-8 -*-
"""
DLP/Secret Scanner - WI-CLOUD-05.

CLOUD ZONE ONLY.

Scans configuration data for secrets and sensitive information.
Cloud MUST NOT store API keys, credentials, or other secrets.

Design Doc Reference:
    - "Cloud never stores secrets"
    - "Keys remain in Agent; cloud does not receive/store them"
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Final, FrozenSet, List, Optional, Pattern, Set

from ccea.guardrails.intent_prohibition import PROHIBITED_INTENT_FIELDS


# ============================================================================
# Constants - Secret Detection Patterns
# ============================================================================


class SecretType(str, Enum):
    """Types of secrets that can be detected."""

    API_KEY = "api_key"
    API_SECRET = "api_secret"
    PRIVATE_KEY = "private_key"
    PASSWORD = "password"
    TOKEN = "token"
    CREDENTIAL = "credential"
    BROKER_KEY = "broker_key"
    EXCHANGE_KEY = "exchange_key"
    AWS_KEY = "aws_key"
    DATABASE_URL = "database_url"
    WEBHOOK_SECRET = "webhook_secret"
    JWT_SECRET = "jwt_secret"
    ENCRYPTION_KEY = "encryption_key"
    SEED_PHRASE = "seed_phrase"
    GENERIC_SECRET = "generic_secret"


class ScanSeverity(str, Enum):
    """Severity of detected secret."""

    CRITICAL = "critical"  # API keys, private keys - immediate block
    HIGH = "high"  # Passwords, tokens - block
    MEDIUM = "medium"  # Suspicious patterns - warn
    LOW = "low"  # Potential false positive - info


# Secret field names that should NEVER appear in cloud config
PROHIBITED_SECRET_FIELDS: Final[FrozenSet[str]] = frozenset(
    {
        # Broker/Exchange API keys - various naming conventions
        "api_key",
        "api_secret",
        "apikey",
        "apisecret",
        "api-key",
        "api-secret",
        "secret_key",
        "secretkey",
        "access_key",
        "accesskey",
        "broker_key",
        "broker_secret",
        "exchange_key",
        "exchange_secret",
        # Passwords and credentials
        "password",
        "passwd",
        "pwd",
        "credential",
        "credentials",
        "auth_token",
        "authtoken",
        "bearer_token",
        "token",  # Generic token field
        # Private keys
        "private_key",
        "privatekey",
        "priv_key",
        "privkey",
        "secret",
        "signing_key",
        "encryption_key",
        "master_key",
        # Database
        "db_password",
        "database_password",
        "connection_string",
        "database_url",
        # AWS/Cloud
        "aws_secret_access_key",
        "aws_access_key_id",
        "aws_key",
        # Webhooks
        "webhook_secret",
        "callback_secret",
        # JWT
        "jwt_secret",
        # Crypto
        "seed_phrase",
        "mnemonic",
        "wallet_key",
    }
)


@dataclass
class SecretPattern:
    """Pattern for detecting secrets in values."""

    name: str
    pattern: Pattern[str]
    secret_type: SecretType
    severity: ScanSeverity
    description: str

    def match(self, value: str) -> bool:
        """Check if value matches this pattern."""
        return bool(self.pattern.search(value))


# Compiled patterns for secret detection
SECRET_PATTERNS: Final[List[SecretPattern]] = [
    # RSA/EC Private Keys
    SecretPattern(
        name="private_key_pem",
        pattern=re.compile(
            r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----",
            re.IGNORECASE,
        ),
        secret_type=SecretType.PRIVATE_KEY,
        severity=ScanSeverity.CRITICAL,
        description="PEM-encoded private key detected",
    ),
    SecretPattern(
        name="ec_private_key",
        pattern=re.compile(
            r"-----BEGIN\s+EC\s+PRIVATE\s+KEY-----",
            re.IGNORECASE,
        ),
        secret_type=SecretType.PRIVATE_KEY,
        severity=ScanSeverity.CRITICAL,
        description="EC private key detected",
    ),
    # AWS Keys
    SecretPattern(
        name="aws_access_key",
        pattern=re.compile(r"AKIA[0-9A-Z]{16}"),
        secret_type=SecretType.AWS_KEY,
        severity=ScanSeverity.CRITICAL,
        description="AWS Access Key ID detected",
    ),
    SecretPattern(
        name="aws_secret_key",
        pattern=re.compile(r"[A-Za-z0-9/+=]{40}"),
        secret_type=SecretType.AWS_KEY,
        severity=ScanSeverity.MEDIUM,  # Lower severity - many false positives
        description="Potential AWS Secret Key detected",
    ),
    # Generic API Keys (common formats)
    SecretPattern(
        name="generic_api_key_long",
        pattern=re.compile(r"[a-zA-Z0-9]{32,64}"),
        secret_type=SecretType.API_KEY,
        severity=ScanSeverity.LOW,  # Many false positives
        description="Potential API key (long alphanumeric string)",
    ),
    # JWT tokens
    SecretPattern(
        name="jwt_token",
        pattern=re.compile(
            r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+",
        ),
        secret_type=SecretType.TOKEN,
        severity=ScanSeverity.HIGH,
        description="JWT token detected",
    ),
    # Database URLs with credentials
    SecretPattern(
        name="database_url",
        pattern=re.compile(
            r"(postgres|mysql|mongodb|redis)://[^:]+:[^@]+@",
            re.IGNORECASE,
        ),
        secret_type=SecretType.DATABASE_URL,
        severity=ScanSeverity.CRITICAL,
        description="Database URL with credentials detected",
    ),
    # Base64 encoded secrets (common for API secrets)
    SecretPattern(
        name="base64_secret",
        pattern=re.compile(r"^[A-Za-z0-9+/]{32,}={0,2}$"),
        secret_type=SecretType.GENERIC_SECRET,
        severity=ScanSeverity.LOW,  # Many false positives
        description="Potential base64-encoded secret",
    ),
    # Hex-encoded secrets (common for HMAC keys)
    SecretPattern(
        name="hex_secret",
        pattern=re.compile(r"^[0-9a-fA-F]{64}$"),
        secret_type=SecretType.ENCRYPTION_KEY,
        severity=ScanSeverity.MEDIUM,
        description="Potential hex-encoded secret (256-bit)",
    ),
    # Seed phrases (BIP39 mnemonic)
    SecretPattern(
        name="seed_phrase",
        pattern=re.compile(
            r"\b(abandon|ability|able|about|above|absent|absorb|abstract|absurd|abuse|"
            r"access|accident|account|accuse|achieve|acid|acoustic|acquire|across|act|"
            r"action|actor|actress|actual|adapt|add|addict|address|adjust|admit|adult|"
            r"advance|advice|aerobic|affair|afford|afraid|again|age|agent|agree|ahead|"
            r"aim|air|airport|aisle|alarm|album|alcohol|alert|alien|all|alley|allow|"
            r"almost|alone|alpha|already|also|alter|always|amateur|amazing|among|amount|"
            r"amused|analyst|anchor|ancient|anger|angle|angry|animal|ankle|announce|"
            r"annual|another|answer|antenna|antique|anxiety|any|apart|apology|appear|"
            r"apple|approve|april|arch|arctic|area|arena|argue|arm|armed|armor|army)\b",
            re.IGNORECASE,
        ),
        secret_type=SecretType.SEED_PHRASE,
        severity=ScanSeverity.MEDIUM,  # Single word match - need context
        description="Potential BIP39 seed phrase word detected",
    ),
]


# ============================================================================
# Scan Results
# ============================================================================


@dataclass
class SecretFinding:
    """A detected secret or sensitive data."""

    field_path: str
    secret_type: SecretType
    severity: ScanSeverity
    message: str
    pattern_name: str
    redacted_value: str = ""  # Redacted preview
    blocked: bool = True
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field_path": self.field_path,
            "secret_type": self.secret_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "pattern_name": self.pattern_name,
            "redacted_value": self.redacted_value,
            "blocked": self.blocked,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DLPScanResult:
    """Result of DLP scan."""

    clean: bool = True
    blocked: bool = False
    findings: List[SecretFinding] = field(default_factory=list)
    scanned_fields: int = 0
    scanned_values: int = 0
    scan_time_ms: float = 0.0

    def add_finding(self, finding: SecretFinding) -> None:
        """Add a finding."""
        self.findings.append(finding)
        if finding.blocked:
            self.clean = False
            self.blocked = True

    @property
    def critical_count(self) -> int:
        """Count of critical findings."""
        return sum(1 for f in self.findings if f.severity == ScanSeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        """Count of high severity findings."""
        return sum(1 for f in self.findings if f.severity == ScanSeverity.HIGH)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "clean": self.clean,
            "blocked": self.blocked,
            "finding_count": len(self.findings),
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "findings": [f.to_dict() for f in self.findings],
            "scanned_fields": self.scanned_fields,
            "scanned_values": self.scanned_values,
            "scan_time_ms": self.scan_time_ms,
        }


# ============================================================================
# DLP Scanner
# ============================================================================


class DLPScanner:
    """
    Data Loss Prevention scanner for config blobs.

    FAIL-CLOSED: Blocks config if secrets are detected.

    Usage:
        scanner = DLPScanner()
        result = scanner.scan(config_content)
        if not result.clean:
            raise SecurityError(f"Secrets detected: {result.findings}")
    """

    def __init__(
        self,
        *,
        prohibited_fields: Optional[FrozenSet[str]] = None,
        patterns: Optional[List[SecretPattern]] = None,
        block_on_critical: bool = True,
        block_on_high: bool = True,
        block_on_medium: bool = True,  # WI-CLOUD-05: Block on medium for security
    ):
        """
        Initialize scanner.

        Args:
            prohibited_fields: Field names that should never appear
            patterns: Secret detection patterns
            block_on_critical: Block if critical severity found
            block_on_high: Block if high severity found
            block_on_medium: Block if medium severity found (default True for security)
        """
        self._prohibited_fields = prohibited_fields or PROHIBITED_SECRET_FIELDS
        self._patterns = patterns or SECRET_PATTERNS
        self._block_on_critical = block_on_critical
        self._block_on_high = block_on_high
        self._block_on_medium = block_on_medium
        # Also include intent prohibition fields
        self._intent_fields = PROHIBITED_INTENT_FIELDS

    def scan(self, data: Dict[str, Any]) -> DLPScanResult:
        """
        Scan data for secrets and sensitive information.

        Args:
            data: Dictionary to scan

        Returns:
            DLPScanResult with findings
        """
        import time

        start_time = time.perf_counter()

        result = DLPScanResult()
        self._scan_recursive(data, "", result)

        result.scan_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    def _scan_recursive(
        self,
        obj: Any,
        path: str,
        result: DLPScanResult,
    ) -> None:
        """Recursively scan object for secrets."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                key_lower = key.lower()
                result.scanned_fields += 1

                # Check field name against prohibited list
                if key_lower in self._prohibited_fields:
                    severity = self._get_field_severity(key_lower)
                    should_block = self._should_block(severity)
                    result.add_finding(
                        SecretFinding(
                            field_path=current_path,
                            secret_type=self._guess_secret_type(key_lower),
                            severity=severity,
                            message=f"Prohibited field name '{key}' - secrets must stay in Agent",
                            pattern_name="field_name_check",
                            redacted_value=self._redact_value(value),
                            blocked=should_block,
                        )
                    )

                # Check against intent fields (order-like payloads)
                if key_lower in self._intent_fields:
                    result.add_finding(
                        SecretFinding(
                            field_path=current_path,
                            secret_type=SecretType.GENERIC_SECRET,
                            severity=ScanSeverity.CRITICAL,
                            message=f"Intent field '{key}' detected - order-like payloads forbidden",
                            pattern_name="intent_field_check",
                            redacted_value="[BLOCKED]",
                            blocked=True,
                        )
                    )

                # Recurse into value
                self._scan_recursive(value, current_path, result)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._scan_recursive(item, f"{path}[{i}]", result)

        elif isinstance(obj, str) and obj:
            result.scanned_values += 1
            # Check value against patterns
            self._scan_value(obj, path, result)

    def _scan_value(
        self,
        value: str,
        path: str,
        result: DLPScanResult,
    ) -> None:
        """Scan a string value for secrets."""
        # Skip very short values (not secrets)
        if len(value) < 8:
            return

        # Check against patterns (limit to high-confidence patterns)
        for pattern in self._patterns:
            # Skip low-severity patterns for values (too many false positives)
            if pattern.severity == ScanSeverity.LOW:
                continue

            if pattern.match(value):
                should_block = self._should_block(pattern.severity)
                result.add_finding(
                    SecretFinding(
                        field_path=path,
                        secret_type=pattern.secret_type,
                        severity=pattern.severity,
                        message=pattern.description,
                        pattern_name=pattern.name,
                        redacted_value=self._redact_value(value),
                        blocked=should_block,
                    )
                )
                # Don't report multiple patterns for same value
                break

    def _get_field_severity(self, field_name: str) -> ScanSeverity:
        """Get severity based on field name."""
        critical_fields = {
            "api_key",
            "api_secret",
            "private_key",
            "secret_key",
            "broker_key",
            "broker_secret",
            "exchange_key",
            "exchange_secret",
            "aws_secret_access_key",
            "master_key",
            "encryption_key",
        }
        high_fields = {
            "password",
            "passwd",
            "pwd",
            "credential",
            "credentials",
            "auth_token",
            "bearer_token",
            "jwt_secret",
            "database_password",
        }

        if field_name in critical_fields:
            return ScanSeverity.CRITICAL
        elif field_name in high_fields:
            return ScanSeverity.HIGH
        else:
            return ScanSeverity.MEDIUM

    def _guess_secret_type(self, field_name: str) -> SecretType:
        """Guess secret type from field name."""
        if "api" in field_name and ("key" in field_name or "secret" in field_name):
            return SecretType.API_KEY
        elif "broker" in field_name or "exchange" in field_name:
            return SecretType.BROKER_KEY
        elif "private" in field_name or "priv" in field_name:
            return SecretType.PRIVATE_KEY
        elif "password" in field_name or "passwd" in field_name or "pwd" in field_name:
            return SecretType.PASSWORD
        elif "token" in field_name:
            return SecretType.TOKEN
        elif "credential" in field_name:
            return SecretType.CREDENTIAL
        elif "aws" in field_name:
            return SecretType.AWS_KEY
        elif "database" in field_name or "db" in field_name:
            return SecretType.DATABASE_URL
        elif "jwt" in field_name:
            return SecretType.JWT_SECRET
        elif "encryption" in field_name or "master" in field_name:
            return SecretType.ENCRYPTION_KEY
        else:
            return SecretType.GENERIC_SECRET

    def _should_block(self, severity: ScanSeverity) -> bool:
        """Determine if finding should block the operation."""
        if severity == ScanSeverity.CRITICAL:
            return self._block_on_critical
        elif severity == ScanSeverity.HIGH:
            return self._block_on_high
        elif severity == ScanSeverity.MEDIUM:
            return self._block_on_medium
        return False

    def _redact_value(self, value: Any) -> str:
        """Redact a value for safe logging."""
        if not isinstance(value, str):
            return "[NON-STRING]"
        if len(value) <= 8:
            return "*" * len(value)
        # Show first 4 and last 4 chars
        return f"{value[:4]}...{value[-4:]}"


# ============================================================================
# Convenience Functions
# ============================================================================

# Singleton scanner instance
_default_scanner: Optional[DLPScanner] = None


def get_scanner() -> DLPScanner:
    """Get the default scanner instance."""
    global _default_scanner
    if _default_scanner is None:
        _default_scanner = DLPScanner()
    return _default_scanner


def scan_for_secrets(data: Dict[str, Any]) -> DLPScanResult:
    """
    Scan data for secrets using the default scanner.

    Args:
        data: Dictionary to scan

    Returns:
        DLPScanResult
    """
    return get_scanner().scan(data)


def assert_no_secrets(data: Dict[str, Any]) -> None:
    """
    Assert that data contains no secrets.

    Args:
        data: Data to check

    Raises:
        ValueError: If secrets are found
    """
    result = scan_for_secrets(data)
    if not result.clean:
        findings = [f.message for f in result.findings if f.blocked]
        raise ValueError(f"Secrets detected: {'; '.join(findings)}")
