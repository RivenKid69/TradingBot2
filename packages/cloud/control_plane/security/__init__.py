# -*- coding: utf-8 -*-
"""
Cloud Control Plane Security Module.

CLOUD ZONE ONLY.

Provides security components for Phase 5:
- Command type validation (WI-CLOUD-01)
- DLP/Secret scanning (WI-CLOUD-05)
- Production-grade authentication (WI-AUTH-01)
"""

from .command_validation import (
    CommandTypeValidator,
    validate_command_type,
    ALLOWED_COMMAND_TYPES,
)
from .dlp_scanner import (
    DLPScanner,
    DLPScanResult,
    SecretPattern,
    scan_for_secrets,
)
from .password_policy import (
    PasswordPolicy,
    PasswordPolicyViolation,
    validate_password,
    DEFAULT_PASSWORD_POLICY,
)
from .password_hasher import (
    PasswordHasher,
    hash_password,
    verify_password,
)
from .rate_limiter import (
    RateLimiter,
    RateLimitExceeded,
    AccountLockout,
)
from .jwt_revocation import (
    JTIBlocklist,
    revoke_token,
    is_token_revoked,
)

__all__ = [
    # Command validation
    "CommandTypeValidator",
    "validate_command_type",
    "ALLOWED_COMMAND_TYPES",
    # DLP
    "DLPScanner",
    "DLPScanResult",
    "SecretPattern",
    "scan_for_secrets",
    # Password policy
    "PasswordPolicy",
    "PasswordPolicyViolation",
    "validate_password",
    "DEFAULT_PASSWORD_POLICY",
    # Password hasher
    "PasswordHasher",
    "hash_password",
    "verify_password",
    # Rate limiter
    "RateLimiter",
    "RateLimitExceeded",
    "AccountLockout",
    # JWT revocation
    "JTIBlocklist",
    "revoke_token",
    "is_token_revoked",
]
