"""
Security Baseline Service - Phase 7 GDPR Implementation

Implements GDPR Article 32 security controls:
- Encryption at rest/in transit
- Key management with rotation
- MFA enforcement for privileged roles
- Secrets management policies
- Security monitoring and logging

Reference: docs/compliance/SECURITY_PHASE7_SPEC.md
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4


# =============================================================================
# Enums and Constants
# =============================================================================

class EncryptionAlgorithm(str, Enum):
    """Supported encryption algorithms."""
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    CHACHA20_POLY1305 = "chacha20_poly1305"


class KeyDerivation(str, Enum):
    """Key derivation functions."""
    PBKDF2 = "pbkdf2"
    ARGON2ID = "argon2id"
    SCRYPT = "scrypt"


class KeyType(str, Enum):
    """Types of cryptographic keys."""
    MASTER_KEY = "master_key"
    DATA_ENCRYPTION_KEY = "data_encryption_key"
    SIGNING_KEY = "signing_key"
    HMAC_KEY = "hmac_key"
    SESSION_KEY = "session_key"


class KeyStatus(str, Enum):
    """Key lifecycle status."""
    ACTIVE = "active"
    ROTATION_PENDING = "rotation_pending"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"
    DESTROYED = "destroyed"


class KeyManagementPolicy(str, Enum):
    """Key management policies."""
    PLATFORM_MANAGED = "platform_managed"
    CUSTOMER_MANAGED = "customer_managed"
    HSM_BACKED = "hsm_backed"


class MFAMethod(str, Enum):
    """Multi-factor authentication methods."""
    TOTP = "totp"
    WEBAUTHN = "webauthn"
    SMS = "sms"  # Deprecated but supported for legacy
    HARDWARE_TOKEN = "hardware_token"
    PUSH_NOTIFICATION = "push_notification"


class MFARequirement(str, Enum):
    """MFA enforcement levels."""
    DISABLED = "disabled"
    OPTIONAL = "optional"
    REQUIRED_SENSITIVE = "required_sensitive"
    REQUIRED_ALWAYS = "required_always"


class SecretType(str, Enum):
    """Types of secrets."""
    API_KEY = "api_key"
    DATABASE_CREDENTIAL = "database_credential"
    SERVICE_ACCOUNT = "service_account"
    ENCRYPTION_KEY = "encryption_key"
    SIGNING_KEY = "signing_key"
    OAUTH_TOKEN = "oauth_token"
    WEBHOOK_SECRET = "webhook_secret"


class SecretStatus(str, Enum):
    """Secret lifecycle status."""
    ACTIVE = "active"
    EXPIRING_SOON = "expiring_soon"
    EXPIRED = "expired"
    ROTATED = "rotated"
    REVOKED = "revoked"


class StorageBackend(str, Enum):
    """Secret storage backends."""
    VAULT = "vault"
    AWS_SECRETS_MANAGER = "aws_secrets_manager"
    GCP_SECRET_MANAGER = "gcp_secret_manager"
    AZURE_KEY_VAULT = "azure_key_vault"
    SEALED_SECRETS = "sealed_secrets"


class SecurityEventType(str, Enum):
    """Types of security events."""
    KEY_CREATED = "key_created"
    KEY_ROTATED = "key_rotated"
    KEY_REVOKED = "key_revoked"
    KEY_ACCESSED = "key_accessed"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"
    MFA_CHALLENGE_SUCCESS = "mfa_challenge_success"
    MFA_CHALLENGE_FAILED = "mfa_challenge_failed"
    SECRET_CREATED = "secret_created"
    SECRET_ROTATED = "secret_rotated"
    SECRET_ACCESSED = "secret_accessed"
    SECRET_EXPIRED = "secret_expired"
    SECRET_REVOKED = "secret_revoked"
    ENCRYPTION_CONFIG_CHANGED = "encryption_config_changed"
    SECURITY_POLICY_UPDATED = "security_policy_updated"
    COMPLIANCE_CHECK_PASSED = "compliance_check_passed"
    COMPLIANCE_CHECK_FAILED = "compliance_check_failed"


class ComplianceStatus(str, Enum):
    """Compliance check status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    REQUIRES_ATTENTION = "requires_attention"
    NOT_EVALUATED = "not_evaluated"


# Constants
DEFAULT_KEY_ROTATION_DAYS = 90
MIN_KEY_ROTATION_DAYS = 30
MAX_KEY_ROTATION_DAYS = 365
MIN_SECRET_LENGTH = 16
MIN_PASSWORD_LENGTH = 12
DEFAULT_SECRET_ROTATION_DAYS = 90
PBKDF2_MIN_ITERATIONS = 100000
ARGON2_TIME_COST = 3
ARGON2_MEMORY_COST = 65536
MFA_SESSION_TIMEOUT_HOURS = 8
MFA_ELEVATED_TIMEOUT_HOURS = 1
MAX_MFA_ATTEMPTS = 5
MFA_LOCKOUT_MINUTES = 15


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EncryptionConfig:
    """Encryption configuration."""
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    key_derivation: KeyDerivation = KeyDerivation.ARGON2ID
    key_rotation_days: int = DEFAULT_KEY_ROTATION_DAYS
    at_rest_enabled: bool = True
    in_transit_min_tls: str = "1.3"
    pbkdf2_iterations: int = PBKDF2_MIN_ITERATIONS

    def __post_init__(self) -> None:
        if self.key_rotation_days < MIN_KEY_ROTATION_DAYS:
            raise ValueError(f"Key rotation must be at least {MIN_KEY_ROTATION_DAYS} days")
        if self.key_rotation_days > MAX_KEY_ROTATION_DAYS:
            raise ValueError(f"Key rotation must not exceed {MAX_KEY_ROTATION_DAYS} days")
        if self.in_transit_min_tls not in ("1.2", "1.3"):
            raise ValueError("TLS version must be 1.2 or 1.3")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm.value,
            "key_derivation": self.key_derivation.value,
            "key_rotation_days": self.key_rotation_days,
            "at_rest_enabled": self.at_rest_enabled,
            "in_transit_min_tls": self.in_transit_min_tls,
            "pbkdf2_iterations": self.pbkdf2_iterations,
        }


@dataclass
class KeyMetadata:
    """Cryptographic key metadata (never stores actual key material)."""
    key_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    key_type: KeyType = KeyType.DATA_ENCRYPTION_KEY
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    status: KeyStatus = KeyStatus.ACTIVE
    management_policy: KeyManagementPolicy = KeyManagementPolicy.PLATFORM_MANAGED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""
    last_rotated_at: Optional[datetime] = None
    rotation_due_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    version: int = 1
    key_fingerprint: str = ""  # Hash of key for verification
    hsm_key_id: Optional[str] = None  # Reference to HSM if applicable
    description: str = ""

    def __post_init__(self) -> None:
        if not self.rotation_due_at and self.status == KeyStatus.ACTIVE:
            self.rotation_due_at = self.created_at + timedelta(days=DEFAULT_KEY_ROTATION_DAYS)

    def is_rotation_due(self) -> bool:
        """Check if key rotation is due."""
        if self.rotation_due_at is None:
            return False
        return datetime.now(timezone.utc) >= self.rotation_due_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key_id": self.key_id,
            "workspace_id": self.workspace_id,
            "key_type": self.key_type.value,
            "algorithm": self.algorithm.value,
            "status": self.status.value,
            "management_policy": self.management_policy.value,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "last_rotated_at": self.last_rotated_at.isoformat() if self.last_rotated_at else None,
            "rotation_due_at": self.rotation_due_at.isoformat() if self.rotation_due_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "version": self.version,
            "key_fingerprint": self.key_fingerprint,
            "hsm_key_id": self.hsm_key_id,
            "description": self.description,
        }


@dataclass
class KeyRotationEvent:
    """Key rotation event record."""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    key_id: str = ""
    old_version: int = 0
    new_version: int = 0
    rotated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    rotated_by: str = ""
    reason: str = ""
    old_key_fingerprint: str = ""
    new_key_fingerprint: str = ""
    automatic: bool = False
    evidence_hash: str = ""

    def __post_init__(self) -> None:
        if not self.evidence_hash:
            self.evidence_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = json.dumps({
            "event_id": self.event_id,
            "key_id": self.key_id,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "rotated_at": self.rotated_at.isoformat(),
            "rotated_by": self.rotated_by,
        }, sort_keys=True)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "key_id": self.key_id,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "rotated_at": self.rotated_at.isoformat(),
            "rotated_by": self.rotated_by,
            "reason": self.reason,
            "old_key_fingerprint": self.old_key_fingerprint,
            "new_key_fingerprint": self.new_key_fingerprint,
            "automatic": self.automatic,
            "evidence_hash": self.evidence_hash,
        }


@dataclass
class MFAConfig:
    """MFA configuration for a user/workspace."""
    config_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    workspace_id: str = ""
    requirement: MFARequirement = MFARequirement.REQUIRED_SENSITIVE
    enabled_methods: Set[MFAMethod] = field(default_factory=lambda: {MFAMethod.TOTP})
    preferred_method: MFAMethod = MFAMethod.TOTP
    session_timeout_hours: int = MFA_SESSION_TIMEOUT_HOURS
    elevated_timeout_hours: int = MFA_ELEVATED_TIMEOUT_HOURS
    max_attempts: int = MAX_MFA_ATTEMPTS
    lockout_minutes: int = MFA_LOCKOUT_MINUTES
    last_verified_at: Optional[datetime] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None
    recovery_codes_generated: bool = False

    def is_locked(self) -> bool:
        """Check if MFA is currently locked due to failed attempts."""
        if self.locked_until is None:
            return False
        return datetime.now(timezone.utc) < self.locked_until

    def needs_verification(self, for_sensitive: bool = False) -> bool:
        """Check if MFA verification is needed."""
        if self.requirement == MFARequirement.DISABLED:
            return False
        if self.requirement == MFARequirement.OPTIONAL:
            return False
        if self.requirement == MFARequirement.REQUIRED_SENSITIVE and not for_sensitive:
            return False

        if self.last_verified_at is None:
            return True

        timeout = self.elevated_timeout_hours if for_sensitive else self.session_timeout_hours
        expiry = self.last_verified_at + timedelta(hours=timeout)
        return datetime.now(timezone.utc) >= expiry

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "user_id": self.user_id,
            "workspace_id": self.workspace_id,
            "requirement": self.requirement.value,
            "enabled_methods": [m.value for m in self.enabled_methods],
            "preferred_method": self.preferred_method.value,
            "session_timeout_hours": self.session_timeout_hours,
            "elevated_timeout_hours": self.elevated_timeout_hours,
            "max_attempts": self.max_attempts,
            "lockout_minutes": self.lockout_minutes,
            "last_verified_at": self.last_verified_at.isoformat() if self.last_verified_at else None,
            "failed_attempts": self.failed_attempts,
            "locked_until": self.locked_until.isoformat() if self.locked_until else None,
            "recovery_codes_generated": self.recovery_codes_generated,
        }


@dataclass
class MFAChallengeResult:
    """Result of an MFA challenge."""
    success: bool = False
    user_id: str = ""
    method: MFAMethod = MFAMethod.TOTP
    verified_at: Optional[datetime] = None
    error: Optional[str] = None
    attempts_remaining: int = MAX_MFA_ATTEMPTS
    locked_until: Optional[datetime] = None


@dataclass
class SecretPolicy:
    """Secret management policy."""
    policy_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    min_length: int = MIN_SECRET_LENGTH
    complexity_required: bool = True
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digits: bool = True
    require_special: bool = True
    rotation_days: int = DEFAULT_SECRET_ROTATION_DAYS
    expiry_warning_days: int = 14
    storage_backend: StorageBackend = StorageBackend.VAULT
    allow_plaintext_export: bool = False
    audit_all_access: bool = True
    max_versions_retained: int = 5

    def validate_secret(self, secret: str) -> tuple[bool, List[str]]:
        """Validate a secret against policy. Returns (valid, errors)."""
        errors = []

        if len(secret) < self.min_length:
            errors.append(f"Secret must be at least {self.min_length} characters")

        if self.complexity_required:
            if self.require_uppercase and not any(c.isupper() for c in secret):
                errors.append("Secret must contain at least one uppercase letter")
            if self.require_lowercase and not any(c.islower() for c in secret):
                errors.append("Secret must contain at least one lowercase letter")
            if self.require_digits and not any(c.isdigit() for c in secret):
                errors.append("Secret must contain at least one digit")
            if self.require_special:
                special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
                if not any(c in special_chars for c in secret):
                    errors.append("Secret must contain at least one special character")

        return (len(errors) == 0, errors)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "workspace_id": self.workspace_id,
            "min_length": self.min_length,
            "complexity_required": self.complexity_required,
            "require_uppercase": self.require_uppercase,
            "require_lowercase": self.require_lowercase,
            "require_digits": self.require_digits,
            "require_special": self.require_special,
            "rotation_days": self.rotation_days,
            "expiry_warning_days": self.expiry_warning_days,
            "storage_backend": self.storage_backend.value,
            "allow_plaintext_export": self.allow_plaintext_export,
            "audit_all_access": self.audit_all_access,
            "max_versions_retained": self.max_versions_retained,
        }


@dataclass
class SecretMetadata:
    """Secret metadata (never stores actual secret value)."""
    secret_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    name: str = ""
    secret_type: SecretType = SecretType.API_KEY
    status: SecretStatus = SecretStatus.ACTIVE
    storage_backend: StorageBackend = StorageBackend.VAULT
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""
    last_rotated_at: Optional[datetime] = None
    rotation_due_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0
    version: int = 1
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rotation_due_at and self.status == SecretStatus.ACTIVE:
            self.rotation_due_at = self.created_at + timedelta(days=DEFAULT_SECRET_ROTATION_DAYS)

    def is_expiring_soon(self, warning_days: int = 14) -> bool:
        """Check if secret is expiring soon."""
        if self.expires_at is None:
            return False
        warning_date = datetime.now(timezone.utc) + timedelta(days=warning_days)
        return self.expires_at <= warning_date

    def is_rotation_due(self) -> bool:
        """Check if secret rotation is due."""
        if self.rotation_due_at is None:
            return False
        return datetime.now(timezone.utc) >= self.rotation_due_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "secret_id": self.secret_id,
            "workspace_id": self.workspace_id,
            "name": self.name,
            "secret_type": self.secret_type.value,
            "status": self.status.value,
            "storage_backend": self.storage_backend.value,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "last_rotated_at": self.last_rotated_at.isoformat() if self.last_rotated_at else None,
            "rotation_due_at": self.rotation_due_at.isoformat() if self.rotation_due_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_accessed_at": self.last_accessed_at.isoformat() if self.last_accessed_at else None,
            "access_count": self.access_count,
            "version": self.version,
            "description": self.description,
            "tags": self.tags,
        }


@dataclass
class SecurityEvent:
    """Security event for audit trail."""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    event_type: SecurityEventType = SecurityEventType.KEY_ACCESSED
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    actor_id: str = ""
    actor_type: str = "user"  # user, service, system
    resource_id: str = ""
    resource_type: str = ""
    action: str = ""
    result: str = "success"  # success, failure
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    integrity_hash: str = ""

    def __post_init__(self) -> None:
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = json.dumps({
            "event_id": self.event_id,
            "workspace_id": self.workspace_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "actor_id": self.actor_id,
            "resource_id": self.resource_id,
            "action": self.action,
            "result": self.result,
        }, sort_keys=True)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "workspace_id": self.workspace_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "action": self.action,
            "result": self.result,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "integrity_hash": self.integrity_hash,
        }


@dataclass
class ComplianceCheckResult:
    """Result of a security compliance check."""
    check_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    check_name: str = ""
    category: str = ""  # encryption, key_management, mfa, secrets
    status: ComplianceStatus = ComplianceStatus.NOT_EVALUATED
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "workspace_id": self.workspace_id,
            "check_name": self.check_name,
            "category": self.category,
            "status": self.status.value,
            "checked_at": self.checked_at.isoformat(),
            "findings": self.findings,
            "recommendations": self.recommendations,
            "evidence": self.evidence,
        }


@dataclass
class SecurityBaselineStatus:
    """Overall security baseline status."""
    workspace_id: str = ""
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    overall_status: ComplianceStatus = ComplianceStatus.NOT_EVALUATED
    encryption_status: ComplianceStatus = ComplianceStatus.NOT_EVALUATED
    key_management_status: ComplianceStatus = ComplianceStatus.NOT_EVALUATED
    mfa_status: ComplianceStatus = ComplianceStatus.NOT_EVALUATED
    secrets_status: ComplianceStatus = ComplianceStatus.NOT_EVALUATED
    checks: List[ComplianceCheckResult] = field(default_factory=list)
    keys_rotation_due: int = 0
    secrets_rotation_due: int = 0
    secrets_expiring_soon: int = 0
    mfa_coverage_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "evaluated_at": self.evaluated_at.isoformat(),
            "overall_status": self.overall_status.value,
            "encryption_status": self.encryption_status.value,
            "key_management_status": self.key_management_status.value,
            "mfa_status": self.mfa_status.value,
            "secrets_status": self.secrets_status.value,
            "checks": [c.to_dict() for c in self.checks],
            "keys_rotation_due": self.keys_rotation_due,
            "secrets_rotation_due": self.secrets_rotation_due,
            "secrets_expiring_soon": self.secrets_expiring_soon,
            "mfa_coverage_percent": self.mfa_coverage_percent,
        }


# =============================================================================
# Security Baseline Service
# =============================================================================

class SecurityBaselineService:
    """
    Security Baseline Service for GDPR Article 32 compliance.

    Manages:
    - Encryption configuration
    - Key management and rotation
    - MFA enforcement
    - Secrets management policies
    - Security event logging
    - Compliance checks
    """

    def __init__(
        self,
        default_encryption: Optional[EncryptionConfig] = None,
        default_secret_policy: Optional[SecretPolicy] = None,
        default_mfa_requirement: MFARequirement = MFARequirement.REQUIRED_SENSITIVE,
        on_security_event: Optional[Callable[[SecurityEvent], None]] = None,
        on_compliance_check: Optional[Callable[[ComplianceCheckResult], None]] = None,
    ):
        """
        Initialize Security Baseline Service.

        Args:
            default_encryption: Default encryption configuration
            default_secret_policy: Default secrets policy
            default_mfa_requirement: Default MFA requirement level
            on_security_event: Callback for security events
            on_compliance_check: Callback for compliance check results
        """
        self._lock = threading.RLock()  # Re-entrant lock for nested calls

        # Configuration
        self._default_encryption = default_encryption or EncryptionConfig()
        self._default_secret_policy = default_secret_policy or SecretPolicy()
        self._default_mfa_requirement = default_mfa_requirement

        # Callbacks
        self._on_security_event = on_security_event
        self._on_compliance_check = on_compliance_check

        # Storage (in-memory, production uses database/callbacks)
        self._encryption_configs: Dict[str, EncryptionConfig] = {}  # workspace_id -> config
        self._keys: Dict[str, KeyMetadata] = {}  # key_id -> metadata
        self._key_rotations: List[KeyRotationEvent] = []
        self._mfa_configs: Dict[str, MFAConfig] = {}  # user_id -> config
        self._secret_policies: Dict[str, SecretPolicy] = {}  # workspace_id -> policy
        self._secrets: Dict[str, SecretMetadata] = {}  # secret_id -> metadata
        self._security_events: List[SecurityEvent] = []
        self._compliance_checks: List[ComplianceCheckResult] = []

    # =========================================================================
    # Encryption Configuration
    # =========================================================================

    def get_encryption_config(self, workspace_id: str) -> EncryptionConfig:
        """Get encryption configuration for a workspace."""
        with self._lock:
            return self._encryption_configs.get(workspace_id, self._default_encryption)

    def set_encryption_config(
        self,
        workspace_id: str,
        config: EncryptionConfig,
        changed_by: str,
    ) -> EncryptionConfig:
        """Set encryption configuration for a workspace."""
        with self._lock:
            old_config = self._encryption_configs.get(workspace_id)
            self._encryption_configs[workspace_id] = config

            # Log event
            event = SecurityEvent(
                workspace_id=workspace_id,
                event_type=SecurityEventType.ENCRYPTION_CONFIG_CHANGED,
                actor_id=changed_by,
                resource_id=workspace_id,
                resource_type="encryption_config",
                action="update",
                details={
                    "old_algorithm": old_config.algorithm.value if old_config else None,
                    "new_algorithm": config.algorithm.value,
                    "old_tls": old_config.in_transit_min_tls if old_config else None,
                    "new_tls": config.in_transit_min_tls,
                },
            )
            self._log_event(event)

            return config

    def validate_encryption_compliance(self, workspace_id: str) -> ComplianceCheckResult:
        """Validate encryption configuration compliance."""
        config = self.get_encryption_config(workspace_id)
        findings = []
        recommendations = []

        # Check at-rest encryption
        if not config.at_rest_enabled:
            findings.append("Encryption at rest is disabled")
            recommendations.append("Enable encryption at rest for all data stores")

        # Check TLS version
        if config.in_transit_min_tls < "1.3":
            findings.append(f"TLS version {config.in_transit_min_tls} is below recommended 1.3")
            recommendations.append("Upgrade to TLS 1.3 for improved security")

        # Check algorithm strength
        if config.algorithm == EncryptionAlgorithm.AES_256_CBC:
            findings.append("Using CBC mode instead of recommended GCM mode")
            recommendations.append("Switch to AES-256-GCM for authenticated encryption")

        # Check key rotation
        if config.key_rotation_days > 90:
            findings.append(f"Key rotation interval ({config.key_rotation_days} days) exceeds 90 days")
            recommendations.append("Reduce key rotation interval to 90 days or less")

        status = ComplianceStatus.COMPLIANT if not findings else (
            ComplianceStatus.REQUIRES_ATTENTION if len(findings) <= 2 else ComplianceStatus.NON_COMPLIANT
        )

        result = ComplianceCheckResult(
            workspace_id=workspace_id,
            check_name="encryption_compliance",
            category="encryption",
            status=status,
            findings=findings,
            recommendations=recommendations,
            evidence={
                "algorithm": config.algorithm.value,
                "at_rest_enabled": config.at_rest_enabled,
                "tls_version": config.in_transit_min_tls,
                "key_rotation_days": config.key_rotation_days,
            },
        )

        with self._lock:
            self._compliance_checks.append(result)

        if self._on_compliance_check:
            try:
                self._on_compliance_check(result)
            except Exception:
                pass

        return result

    # =========================================================================
    # Key Management
    # =========================================================================

    def create_key(
        self,
        workspace_id: str,
        key_type: KeyType,
        created_by: str,
        management_policy: KeyManagementPolicy = KeyManagementPolicy.PLATFORM_MANAGED,
        description: str = "",
        rotation_days: Optional[int] = None,
    ) -> KeyMetadata:
        """Create a new key and return its metadata."""
        config = self.get_encryption_config(workspace_id)
        rotation = rotation_days or config.key_rotation_days

        # Generate fingerprint (would be actual key hash in production)
        fingerprint = f"sha256:{secrets.token_hex(32)}"

        key = KeyMetadata(
            workspace_id=workspace_id,
            key_type=key_type,
            algorithm=config.algorithm,
            status=KeyStatus.ACTIVE,
            management_policy=management_policy,
            created_by=created_by,
            rotation_due_at=datetime.now(timezone.utc) + timedelta(days=rotation),
            key_fingerprint=fingerprint,
            description=description,
        )

        with self._lock:
            self._keys[key.key_id] = key

        # Log event
        event = SecurityEvent(
            workspace_id=workspace_id,
            event_type=SecurityEventType.KEY_CREATED,
            actor_id=created_by,
            resource_id=key.key_id,
            resource_type="key",
            action="create",
            details={
                "key_type": key_type.value,
                "algorithm": config.algorithm.value,
                "management_policy": management_policy.value,
            },
        )
        self._log_event(event)

        return key

    def get_key(self, key_id: str) -> Optional[KeyMetadata]:
        """Get key metadata by ID."""
        with self._lock:
            return self._keys.get(key_id)

    def list_keys(
        self,
        workspace_id: Optional[str] = None,
        key_type: Optional[KeyType] = None,
        status: Optional[KeyStatus] = None,
    ) -> List[KeyMetadata]:
        """List keys with optional filtering."""
        with self._lock:
            keys = list(self._keys.values())

        if workspace_id:
            keys = [k for k in keys if k.workspace_id == workspace_id]
        if key_type:
            keys = [k for k in keys if k.key_type == key_type]
        if status:
            keys = [k for k in keys if k.status == status]

        return keys

    def rotate_key(
        self,
        key_id: str,
        rotated_by: str,
        reason: str = "Scheduled rotation",
        automatic: bool = False,
    ) -> KeyRotationEvent:
        """Rotate a key and return the rotation event."""
        with self._lock:
            key = self._keys.get(key_id)
            if not key:
                raise ValueError(f"Key {key_id} not found")

            if key.status != KeyStatus.ACTIVE:
                raise ValueError(f"Cannot rotate key in {key.status.value} status")

            old_fingerprint = key.key_fingerprint
            old_version = key.version

            # Generate new fingerprint (would be actual new key in production)
            new_fingerprint = f"sha256:{secrets.token_hex(32)}"
            config = self.get_encryption_config(key.workspace_id)

            # Update key
            key.last_rotated_at = datetime.now(timezone.utc)
            key.rotation_due_at = datetime.now(timezone.utc) + timedelta(days=config.key_rotation_days)
            key.key_fingerprint = new_fingerprint
            key.version += 1

            # Create rotation event
            rotation = KeyRotationEvent(
                key_id=key_id,
                old_version=old_version,
                new_version=key.version,
                rotated_by=rotated_by,
                reason=reason,
                old_key_fingerprint=old_fingerprint,
                new_key_fingerprint=new_fingerprint,
                automatic=automatic,
            )
            self._key_rotations.append(rotation)

        # Log event
        event = SecurityEvent(
            workspace_id=key.workspace_id,
            event_type=SecurityEventType.KEY_ROTATED,
            actor_id=rotated_by,
            resource_id=key_id,
            resource_type="key",
            action="rotate",
            details={
                "old_version": old_version,
                "new_version": key.version,
                "reason": reason,
                "automatic": automatic,
            },
        )
        self._log_event(event)

        return rotation

    def revoke_key(
        self,
        key_id: str,
        revoked_by: str,
        reason: str,
    ) -> KeyMetadata:
        """Revoke a key."""
        with self._lock:
            key = self._keys.get(key_id)
            if not key:
                raise ValueError(f"Key {key_id} not found")

            key.status = KeyStatus.REVOKED

        # Log event
        event = SecurityEvent(
            workspace_id=key.workspace_id,
            event_type=SecurityEventType.KEY_REVOKED,
            actor_id=revoked_by,
            resource_id=key_id,
            resource_type="key",
            action="revoke",
            details={"reason": reason},
        )
        self._log_event(event)

        return key

    def get_keys_needing_rotation(self, workspace_id: Optional[str] = None) -> List[KeyMetadata]:
        """Get keys that are due for rotation."""
        keys = self.list_keys(workspace_id=workspace_id, status=KeyStatus.ACTIVE)
        return [k for k in keys if k.is_rotation_due()]

    def validate_key_management_compliance(self, workspace_id: str) -> ComplianceCheckResult:
        """Validate key management compliance."""
        keys = self.list_keys(workspace_id=workspace_id)
        findings = []
        recommendations = []

        # Check for keys needing rotation
        rotation_due = [k for k in keys if k.is_rotation_due() and k.status == KeyStatus.ACTIVE]
        if rotation_due:
            findings.append(f"{len(rotation_due)} key(s) are overdue for rotation")
            recommendations.append("Rotate overdue keys immediately")

        # Check for HSM backing
        platform_managed = [k for k in keys if k.management_policy == KeyManagementPolicy.PLATFORM_MANAGED]
        master_keys = [k for k in platform_managed if k.key_type == KeyType.MASTER_KEY]
        if master_keys:
            findings.append(f"{len(master_keys)} master key(s) not HSM-backed")
            recommendations.append("Consider HSM-backing for master keys")

        # Check for deprecated keys still in use
        deprecated = [k for k in keys if k.status == KeyStatus.DEPRECATED]
        if deprecated:
            findings.append(f"{len(deprecated)} deprecated key(s) still exist")
            recommendations.append("Remove or archive deprecated keys")

        status = ComplianceStatus.COMPLIANT if not findings else (
            ComplianceStatus.REQUIRES_ATTENTION if len(rotation_due) == 0 else ComplianceStatus.NON_COMPLIANT
        )

        result = ComplianceCheckResult(
            workspace_id=workspace_id,
            check_name="key_management_compliance",
            category="key_management",
            status=status,
            findings=findings,
            recommendations=recommendations,
            evidence={
                "total_keys": len(keys),
                "rotation_due": len(rotation_due),
                "deprecated": len(deprecated),
                "hsm_backed": len([k for k in keys if k.management_policy == KeyManagementPolicy.HSM_BACKED]),
            },
        )

        with self._lock:
            self._compliance_checks.append(result)

        if self._on_compliance_check:
            try:
                self._on_compliance_check(result)
            except Exception:
                pass

        return result

    # =========================================================================
    # MFA Management
    # =========================================================================

    def get_mfa_config(self, user_id: str) -> Optional[MFAConfig]:
        """Get MFA configuration for a user."""
        with self._lock:
            return self._mfa_configs.get(user_id)

    def set_mfa_config(
        self,
        user_id: str,
        workspace_id: str,
        config: MFAConfig,
        set_by: str,
    ) -> MFAConfig:
        """Set MFA configuration for a user."""
        config.user_id = user_id
        config.workspace_id = workspace_id

        with self._lock:
            old_config = self._mfa_configs.get(user_id)
            self._mfa_configs[user_id] = config

        # Log event
        event_type = SecurityEventType.MFA_ENABLED if config.requirement != MFARequirement.DISABLED else SecurityEventType.MFA_DISABLED
        event = SecurityEvent(
            workspace_id=workspace_id,
            event_type=event_type,
            actor_id=set_by,
            resource_id=user_id,
            resource_type="mfa_config",
            action="update",
            details={
                "requirement": config.requirement.value,
                "methods": [m.value for m in config.enabled_methods],
                "previous_requirement": old_config.requirement.value if old_config else None,
            },
        )
        self._log_event(event)

        return config

    def enable_mfa(
        self,
        user_id: str,
        workspace_id: str,
        methods: Set[MFAMethod],
        requirement: MFARequirement = MFARequirement.REQUIRED_ALWAYS,
        enabled_by: str = "",
    ) -> MFAConfig:
        """Enable MFA for a user."""
        if not methods:
            raise ValueError("At least one MFA method must be provided")

        if MFAMethod.SMS in methods and len(methods) == 1:
            raise ValueError("SMS-only MFA is deprecated. Add another method.")

        config = MFAConfig(
            user_id=user_id,
            workspace_id=workspace_id,
            requirement=requirement,
            enabled_methods=methods,
            preferred_method=list(methods)[0],
        )

        return self.set_mfa_config(user_id, workspace_id, config, enabled_by or user_id)

    def verify_mfa_challenge(
        self,
        user_id: str,
        method: MFAMethod,
        code: str,
        workspace_id: str,
    ) -> MFAChallengeResult:
        """
        Verify an MFA challenge.

        Note: In production, this would integrate with actual TOTP/WebAuthn verification.
        """
        with self._lock:
            config = self._mfa_configs.get(user_id)
            if not config:
                return MFAChallengeResult(
                    success=False,
                    user_id=user_id,
                    method=method,
                    error="MFA not configured for user",
                )

            if config.is_locked():
                return MFAChallengeResult(
                    success=False,
                    user_id=user_id,
                    method=method,
                    error="Account locked due to too many failed attempts",
                    locked_until=config.locked_until,
                )

            if method not in config.enabled_methods:
                return MFAChallengeResult(
                    success=False,
                    user_id=user_id,
                    method=method,
                    error=f"MFA method {method.value} not enabled for user",
                )

            # Simulate verification (in production: actual TOTP/WebAuthn check)
            # For testing, accept codes starting with "valid_"
            is_valid = code.startswith("valid_") or len(code) == 6 and code.isdigit()

            if is_valid:
                config.failed_attempts = 0
                config.last_verified_at = datetime.now(timezone.utc)
                config.locked_until = None

                result = MFAChallengeResult(
                    success=True,
                    user_id=user_id,
                    method=method,
                    verified_at=config.last_verified_at,
                    attempts_remaining=config.max_attempts,
                )

                event_type = SecurityEventType.MFA_CHALLENGE_SUCCESS
            else:
                config.failed_attempts += 1
                if config.failed_attempts >= config.max_attempts:
                    config.locked_until = datetime.now(timezone.utc) + timedelta(minutes=config.lockout_minutes)

                result = MFAChallengeResult(
                    success=False,
                    user_id=user_id,
                    method=method,
                    error="Invalid MFA code",
                    attempts_remaining=max(0, config.max_attempts - config.failed_attempts),
                    locked_until=config.locked_until,
                )

                event_type = SecurityEventType.MFA_CHALLENGE_FAILED

        # Log event
        event = SecurityEvent(
            workspace_id=workspace_id,
            event_type=event_type,
            actor_id=user_id,
            resource_id=user_id,
            resource_type="mfa",
            action="verify",
            result="success" if is_valid else "failure",
            details={
                "method": method.value,
                "attempts_remaining": result.attempts_remaining,
            },
        )
        self._log_event(event)

        return result

    def check_mfa_required(
        self,
        user_id: str,
        for_sensitive: bool = False,
    ) -> bool:
        """Check if MFA verification is required for a user."""
        config = self.get_mfa_config(user_id)
        if not config:
            # Use default requirement
            if self._default_mfa_requirement == MFARequirement.DISABLED:
                return False
            if self._default_mfa_requirement == MFARequirement.OPTIONAL:
                return False
            if self._default_mfa_requirement == MFARequirement.REQUIRED_SENSITIVE:
                return for_sensitive
            return True

        return config.needs_verification(for_sensitive)

    def validate_mfa_compliance(self, workspace_id: str, user_ids: List[str]) -> ComplianceCheckResult:
        """Validate MFA compliance for users in a workspace."""
        findings = []
        recommendations = []

        users_with_mfa = 0
        users_with_strong_mfa = 0
        users_with_sms_only = 0

        for user_id in user_ids:
            config = self.get_mfa_config(user_id)
            if config and config.requirement != MFARequirement.DISABLED:
                users_with_mfa += 1
                if MFAMethod.WEBAUTHN in config.enabled_methods or MFAMethod.HARDWARE_TOKEN in config.enabled_methods:
                    users_with_strong_mfa += 1
                if config.enabled_methods == {MFAMethod.SMS}:
                    users_with_sms_only += 1

        total_users = len(user_ids)
        coverage = (users_with_mfa / total_users * 100) if total_users > 0 else 0

        if coverage < 100:
            findings.append(f"MFA coverage is {coverage:.1f}% ({users_with_mfa}/{total_users} users)")
            recommendations.append("Enable MFA for all users")

        if users_with_sms_only > 0:
            findings.append(f"{users_with_sms_only} user(s) using SMS-only MFA (deprecated)")
            recommendations.append("Migrate SMS-only users to TOTP or WebAuthn")

        if users_with_strong_mfa < total_users / 2:
            findings.append("Less than 50% of users have phishing-resistant MFA")
            recommendations.append("Encourage adoption of WebAuthn or hardware tokens")

        status = ComplianceStatus.COMPLIANT if coverage == 100 and users_with_sms_only == 0 else (
            ComplianceStatus.REQUIRES_ATTENTION if coverage >= 80 else ComplianceStatus.NON_COMPLIANT
        )

        result = ComplianceCheckResult(
            workspace_id=workspace_id,
            check_name="mfa_compliance",
            category="mfa",
            status=status,
            findings=findings,
            recommendations=recommendations,
            evidence={
                "total_users": total_users,
                "users_with_mfa": users_with_mfa,
                "coverage_percent": coverage,
                "strong_mfa_users": users_with_strong_mfa,
                "sms_only_users": users_with_sms_only,
            },
        )

        with self._lock:
            self._compliance_checks.append(result)

        if self._on_compliance_check:
            try:
                self._on_compliance_check(result)
            except Exception:
                pass

        return result

    # =========================================================================
    # Secrets Management
    # =========================================================================

    def get_secret_policy(self, workspace_id: str) -> SecretPolicy:
        """Get secret policy for a workspace."""
        with self._lock:
            return self._secret_policies.get(workspace_id, self._default_secret_policy)

    def set_secret_policy(
        self,
        workspace_id: str,
        policy: SecretPolicy,
        set_by: str,
    ) -> SecretPolicy:
        """Set secret policy for a workspace."""
        policy.workspace_id = workspace_id

        with self._lock:
            self._secret_policies[workspace_id] = policy

        # Log event
        event = SecurityEvent(
            workspace_id=workspace_id,
            event_type=SecurityEventType.SECURITY_POLICY_UPDATED,
            actor_id=set_by,
            resource_id=policy.policy_id,
            resource_type="secret_policy",
            action="update",
            details={
                "min_length": policy.min_length,
                "rotation_days": policy.rotation_days,
                "storage_backend": policy.storage_backend.value,
            },
        )
        self._log_event(event)

        return policy

    def register_secret(
        self,
        workspace_id: str,
        name: str,
        secret_type: SecretType,
        created_by: str,
        description: str = "",
        expires_at: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> SecretMetadata:
        """Register a secret (metadata only - actual secret stored externally)."""
        policy = self.get_secret_policy(workspace_id)

        secret = SecretMetadata(
            workspace_id=workspace_id,
            name=name,
            secret_type=secret_type,
            storage_backend=policy.storage_backend,
            created_by=created_by,
            rotation_due_at=datetime.now(timezone.utc) + timedelta(days=policy.rotation_days),
            expires_at=expires_at,
            description=description,
            tags=tags or {},
        )

        with self._lock:
            self._secrets[secret.secret_id] = secret

        # Log event
        event = SecurityEvent(
            workspace_id=workspace_id,
            event_type=SecurityEventType.SECRET_CREATED,
            actor_id=created_by,
            resource_id=secret.secret_id,
            resource_type="secret",
            action="create",
            details={
                "name": name,
                "type": secret_type.value,
                "backend": policy.storage_backend.value,
            },
        )
        self._log_event(event)

        return secret

    def get_secret_metadata(self, secret_id: str) -> Optional[SecretMetadata]:
        """Get secret metadata."""
        with self._lock:
            return self._secrets.get(secret_id)

    def list_secrets(
        self,
        workspace_id: Optional[str] = None,
        secret_type: Optional[SecretType] = None,
        status: Optional[SecretStatus] = None,
    ) -> List[SecretMetadata]:
        """List secrets with optional filtering."""
        with self._lock:
            secrets = list(self._secrets.values())

        if workspace_id:
            secrets = [s for s in secrets if s.workspace_id == workspace_id]
        if secret_type:
            secrets = [s for s in secrets if s.secret_type == secret_type]
        if status:
            secrets = [s for s in secrets if s.status == status]

        return secrets

    def record_secret_access(
        self,
        secret_id: str,
        accessed_by: str,
        workspace_id: str,
        ip_address: Optional[str] = None,
    ) -> None:
        """Record an access to a secret."""
        with self._lock:
            secret = self._secrets.get(secret_id)
            if secret:
                secret.last_accessed_at = datetime.now(timezone.utc)
                secret.access_count += 1

        # Log event
        event = SecurityEvent(
            workspace_id=workspace_id,
            event_type=SecurityEventType.SECRET_ACCESSED,
            actor_id=accessed_by,
            resource_id=secret_id,
            resource_type="secret",
            action="access",
            ip_address=ip_address,
        )
        self._log_event(event)

    def rotate_secret(
        self,
        secret_id: str,
        rotated_by: str,
        reason: str = "Scheduled rotation",
    ) -> SecretMetadata:
        """Record secret rotation."""
        with self._lock:
            secret = self._secrets.get(secret_id)
            if not secret:
                raise ValueError(f"Secret {secret_id} not found")

            policy = self.get_secret_policy(secret.workspace_id)

            secret.last_rotated_at = datetime.now(timezone.utc)
            secret.rotation_due_at = datetime.now(timezone.utc) + timedelta(days=policy.rotation_days)
            secret.status = SecretStatus.ACTIVE
            secret.version += 1

        # Log event
        event = SecurityEvent(
            workspace_id=secret.workspace_id,
            event_type=SecurityEventType.SECRET_ROTATED,
            actor_id=rotated_by,
            resource_id=secret_id,
            resource_type="secret",
            action="rotate",
            details={
                "reason": reason,
                "new_version": secret.version,
            },
        )
        self._log_event(event)

        return secret

    def revoke_secret(
        self,
        secret_id: str,
        revoked_by: str,
        reason: str,
    ) -> SecretMetadata:
        """Revoke a secret."""
        with self._lock:
            secret = self._secrets.get(secret_id)
            if not secret:
                raise ValueError(f"Secret {secret_id} not found")

            secret.status = SecretStatus.REVOKED

        # Log event
        event = SecurityEvent(
            workspace_id=secret.workspace_id,
            event_type=SecurityEventType.SECRET_REVOKED,
            actor_id=revoked_by,
            resource_id=secret_id,
            resource_type="secret",
            action="revoke",
            details={"reason": reason},
        )
        self._log_event(event)

        return secret

    def get_secrets_needing_rotation(self, workspace_id: Optional[str] = None) -> List[SecretMetadata]:
        """Get secrets that are due for rotation."""
        secrets = self.list_secrets(workspace_id=workspace_id, status=SecretStatus.ACTIVE)
        return [s for s in secrets if s.is_rotation_due()]

    def get_expiring_secrets(
        self,
        workspace_id: Optional[str] = None,
        warning_days: int = 14,
    ) -> List[SecretMetadata]:
        """Get secrets that are expiring soon."""
        secrets = self.list_secrets(workspace_id=workspace_id, status=SecretStatus.ACTIVE)
        return [s for s in secrets if s.is_expiring_soon(warning_days)]

    def validate_secrets_compliance(self, workspace_id: str) -> ComplianceCheckResult:
        """Validate secrets management compliance."""
        secrets = self.list_secrets(workspace_id=workspace_id)
        policy = self.get_secret_policy(workspace_id)
        findings = []
        recommendations = []

        # Check for secrets needing rotation
        rotation_due = [s for s in secrets if s.is_rotation_due() and s.status == SecretStatus.ACTIVE]
        if rotation_due:
            findings.append(f"{len(rotation_due)} secret(s) are overdue for rotation")
            recommendations.append("Rotate overdue secrets immediately")

        # Check for expiring secrets
        expiring = [s for s in secrets if s.is_expiring_soon(policy.expiry_warning_days) and s.status == SecretStatus.ACTIVE]
        if expiring:
            findings.append(f"{len(expiring)} secret(s) expiring within {policy.expiry_warning_days} days")
            recommendations.append("Renew or rotate expiring secrets")

        # Check for revoked secrets
        revoked = [s for s in secrets if s.status == SecretStatus.REVOKED]
        if revoked:
            findings.append(f"{len(revoked)} revoked secret(s) still in inventory")
            recommendations.append("Remove revoked secrets from inventory")

        status = ComplianceStatus.COMPLIANT if not findings else (
            ComplianceStatus.REQUIRES_ATTENTION if len(rotation_due) == 0 else ComplianceStatus.NON_COMPLIANT
        )

        result = ComplianceCheckResult(
            workspace_id=workspace_id,
            check_name="secrets_compliance",
            category="secrets",
            status=status,
            findings=findings,
            recommendations=recommendations,
            evidence={
                "total_secrets": len(secrets),
                "rotation_due": len(rotation_due),
                "expiring_soon": len(expiring),
                "revoked": len(revoked),
                "storage_backend": policy.storage_backend.value,
            },
        )

        with self._lock:
            self._compliance_checks.append(result)

        if self._on_compliance_check:
            try:
                self._on_compliance_check(result)
            except Exception:
                pass

        return result

    # =========================================================================
    # Overall Security Baseline
    # =========================================================================

    def evaluate_security_baseline(
        self,
        workspace_id: str,
        user_ids: Optional[List[str]] = None,
    ) -> SecurityBaselineStatus:
        """Evaluate overall security baseline for a workspace."""
        encryption_check = self.validate_encryption_compliance(workspace_id)
        key_check = self.validate_key_management_compliance(workspace_id)
        secrets_check = self.validate_secrets_compliance(workspace_id)

        # MFA check requires user list
        if user_ids:
            mfa_check = self.validate_mfa_compliance(workspace_id, user_ids)
        else:
            mfa_check = ComplianceCheckResult(
                workspace_id=workspace_id,
                check_name="mfa_compliance",
                category="mfa",
                status=ComplianceStatus.NOT_EVALUATED,
                findings=["No user list provided for MFA evaluation"],
            )

        checks = [encryption_check, key_check, mfa_check, secrets_check]

        # Determine overall status
        statuses = [c.status for c in checks if c.status != ComplianceStatus.NOT_EVALUATED]
        if not statuses:
            overall = ComplianceStatus.NOT_EVALUATED
        elif any(s == ComplianceStatus.NON_COMPLIANT for s in statuses):
            overall = ComplianceStatus.NON_COMPLIANT
        elif any(s == ComplianceStatus.REQUIRES_ATTENTION for s in statuses):
            overall = ComplianceStatus.REQUIRES_ATTENTION
        else:
            overall = ComplianceStatus.COMPLIANT

        # Get metrics
        keys_rotation_due = len(self.get_keys_needing_rotation(workspace_id))
        secrets_rotation_due = len(self.get_secrets_needing_rotation(workspace_id))
        secrets_expiring = len(self.get_expiring_secrets(workspace_id))

        mfa_coverage = 0.0
        if user_ids and mfa_check.evidence:
            mfa_coverage = mfa_check.evidence.get("coverage_percent", 0.0)

        status = SecurityBaselineStatus(
            workspace_id=workspace_id,
            overall_status=overall,
            encryption_status=encryption_check.status,
            key_management_status=key_check.status,
            mfa_status=mfa_check.status,
            secrets_status=secrets_check.status,
            checks=checks,
            keys_rotation_due=keys_rotation_due,
            secrets_rotation_due=secrets_rotation_due,
            secrets_expiring_soon=secrets_expiring,
            mfa_coverage_percent=mfa_coverage,
        )

        # Log compliance check event
        event = SecurityEvent(
            workspace_id=workspace_id,
            event_type=(
                SecurityEventType.COMPLIANCE_CHECK_PASSED
                if overall == ComplianceStatus.COMPLIANT
                else SecurityEventType.COMPLIANCE_CHECK_FAILED
            ),
            actor_id="system",
            actor_type="system",
            resource_id=workspace_id,
            resource_type="security_baseline",
            action="evaluate",
            result="success" if overall == ComplianceStatus.COMPLIANT else "failure",
            details={
                "overall_status": overall.value,
                "encryption": encryption_check.status.value,
                "key_management": key_check.status.value,
                "mfa": mfa_check.status.value,
                "secrets": secrets_check.status.value,
            },
        )
        self._log_event(event)

        return status

    # =========================================================================
    # Event Logging and Export
    # =========================================================================

    def _log_event(self, event: SecurityEvent) -> None:
        """Log a security event."""
        with self._lock:
            self._security_events.append(event)

        if self._on_security_event:
            try:
                self._on_security_event(event)
            except Exception:
                pass

    def get_security_events(
        self,
        workspace_id: Optional[str] = None,
        event_type: Optional[SecurityEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[SecurityEvent]:
        """Get security events with filtering."""
        with self._lock:
            events = list(self._security_events)

        if workspace_id:
            events = [e for e in events if e.workspace_id == workspace_id]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        # Sort by timestamp descending and limit
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def get_key_rotations(
        self,
        workspace_id: Optional[str] = None,
        key_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[KeyRotationEvent]:
        """Get key rotation events."""
        with self._lock:
            rotations = list(self._key_rotations)

        if key_id:
            rotations = [r for r in rotations if r.key_id == key_id]
        if start_time:
            rotations = [r for r in rotations if r.rotated_at >= start_time]
        if end_time:
            rotations = [r for r in rotations if r.rotated_at <= end_time]

        # Filter by workspace if needed
        if workspace_id:
            workspace_keys = {k.key_id for k in self.list_keys(workspace_id=workspace_id)}
            rotations = [r for r in rotations if r.key_id in workspace_keys]

        return rotations

    def export_security_baseline(
        self,
        workspace_id: str,
        include_events: bool = True,
        include_keys: bool = True,
        include_secrets: bool = True,
    ) -> Dict[str, Any]:
        """Export security baseline data for evidence pack."""
        export_data: Dict[str, Any] = {
            "workspace_id": workspace_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "encryption_config": self.get_encryption_config(workspace_id).to_dict(),
            "secret_policy": self.get_secret_policy(workspace_id).to_dict(),
        }

        if include_keys:
            export_data["keys"] = [k.to_dict() for k in self.list_keys(workspace_id=workspace_id)]
            export_data["key_rotations"] = [r.to_dict() for r in self.get_key_rotations(workspace_id=workspace_id)]

        if include_secrets:
            export_data["secrets"] = [s.to_dict() for s in self.list_secrets(workspace_id=workspace_id)]

        if include_events:
            events = self.get_security_events(workspace_id=workspace_id, limit=10000)
            export_data["security_events"] = [e.to_dict() for e in events]

        # Add compliance checks
        with self._lock:
            checks = [c for c in self._compliance_checks if c.workspace_id == workspace_id]
        export_data["compliance_checks"] = [c.to_dict() for c in checks]

        # Compute export hash
        export_content = json.dumps(export_data, sort_keys=True, default=str)
        export_data["integrity_hash"] = f"sha256:{hashlib.sha256(export_content.encode()).hexdigest()}"

        return export_data
