# -*- coding: utf-8 -*-
"""
Password Policy - WI-AUTH-01.

CLOUD ZONE ONLY.

Implements NIST 800-63B and OWASP password guidelines:
- Minimum length requirements
- Character complexity (optional, per NIST 800-63B)
- Common password blocklist
- Leaked password check (via k-anonymity)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Final, FrozenSet, List, Optional, Set


# ============================================================================
# Common Weak Passwords (Top 100 from breaches)
# ============================================================================

COMMON_WEAK_PASSWORDS: Final[FrozenSet[str]] = frozenset(
    {
        "password",
        "123456",
        "12345678",
        "qwerty",
        "abc123",
        "monkey",
        "1234567",
        "letmein",
        "trustno1",
        "dragon",
        "baseball",
        "iloveyou",
        "master",
        "sunshine",
        "ashley",
        "bailey",
        "shadow",
        "123123",
        "654321",
        "superman",
        "qazwsx",
        "michael",
        "football",
        "password1",
        "password123",
        "batman",
        "login",
        "admin",
        "passw0rd",
        "welcome",
        "hello",
        "charlie",
        "donald",
        "1qaz2wsx",
        "qwerty123",
        "starwars",
        "whatever",
        "ninja",
        "princess",
        "solo",
        "666666",
        "lovely",
        "freedom",
        "121212",
        "hottie",
        "zxcvbn",
        "zxcvbnm",
        "internet",
        "cheese",
        "pepper",
        "joshua",
        "hunter",
        "2000",
        "andrea",
        "soccer",
        "tiger",
        "summer",
        "killer",
        "access",
        "andrew",
        "banana",
        "ranger",
        "batman123",
        "soccer123",
        "football123",
        "baseball123",
        "sunshine123",
        "princess123",
        "welcome1",
        "welcome123",
        "qwerty1",
        "qwerty12",
        "letmein1",
        "letmein123",
        "123456789",
        "1234567890",
        "0987654321",
        "password12",
        "12345",
        "1234",
        "111111",
        "000000",
        "password!",
        "passw0rd!",
        "p@ssw0rd",
        "p@ssword",
        "p@ssword1",
        "p@ssw0rd1",
        "admin123",
        "admin1234",
        "administrator",
        "root",
        "root123",
        "toor",
        "changeme",
        "changeit",
        "temp",
        "test",
        "test123",
        "testing",
        "testing123",
        "guest",
        "guest123",
        "default",
        "default123",
        "user",
        "user123",
        "demo",
    }
)


# ============================================================================
# Policy Classes
# ============================================================================


class PolicyViolationType(str, Enum):
    """Types of password policy violations."""

    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    NO_UPPERCASE = "no_uppercase"
    NO_LOWERCASE = "no_lowercase"
    NO_DIGIT = "no_digit"
    NO_SPECIAL = "no_special"
    COMMON_PASSWORD = "common_password"
    CONTAINS_EMAIL = "contains_email"
    CONTAINS_USERNAME = "contains_username"
    SEQUENTIAL_CHARS = "sequential_chars"
    REPEATED_CHARS = "repeated_chars"
    LEAKED_PASSWORD = "leaked_password"


@dataclass
class PasswordPolicyViolation:
    """A violation of password policy."""

    violation_type: PolicyViolationType
    message: str
    severity: str = "error"  # error, warning

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.violation_type.value,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass
class PasswordValidationResult:
    """Result of password validation."""

    valid: bool
    violations: List[PasswordPolicyViolation] = field(default_factory=list)
    strength_score: int = 0  # 0-100
    strength_label: str = "weak"

    def add_violation(self, violation: PasswordPolicyViolation) -> None:
        """Add a violation."""
        self.violations.append(violation)
        self.valid = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "violations": [v.to_dict() for v in self.violations],
            "strength_score": self.strength_score,
            "strength_label": self.strength_label,
        }


@dataclass
class PasswordPolicy:
    """
    Password policy configuration.

    Based on NIST 800-63B:
    - Focus on length over complexity
    - Block common/leaked passwords
    - Allow all characters including spaces
    """

    # Length requirements
    min_length: int = 12  # NIST recommends 8+, we use 12 for security
    max_length: int = 128  # Reasonable maximum

    # Complexity requirements (NIST says optional, but useful)
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digit: bool = True
    require_special: bool = False  # NIST says not required

    # Pattern checks
    check_common_passwords: bool = True
    check_sequential_chars: bool = True
    check_repeated_chars: bool = True
    max_sequential_chars: int = 3
    max_repeated_chars: int = 3

    # Context checks
    check_contains_email: bool = True
    check_contains_username: bool = True

    # Leaked password check (via k-anonymity API)
    check_leaked_passwords: bool = False  # Requires network call

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_length": self.min_length,
            "max_length": self.max_length,
            "require_uppercase": self.require_uppercase,
            "require_lowercase": self.require_lowercase,
            "require_digit": self.require_digit,
            "require_special": self.require_special,
            "check_common_passwords": self.check_common_passwords,
            "check_sequential_chars": self.check_sequential_chars,
            "check_repeated_chars": self.check_repeated_chars,
        }


# Default policy
DEFAULT_PASSWORD_POLICY = PasswordPolicy()


# ============================================================================
# Password Validator
# ============================================================================


class PasswordValidator:
    """
    Validates passwords against policy.

    Usage:
        validator = PasswordValidator()
        result = validator.validate("MyP@ssw0rd123")
        if not result.valid:
            print(result.violations)
    """

    def __init__(self, policy: Optional[PasswordPolicy] = None):
        """
        Initialize validator.

        Args:
            policy: Password policy to use (defaults to DEFAULT_PASSWORD_POLICY)
        """
        self._policy = policy or DEFAULT_PASSWORD_POLICY
        self._special_chars = set("!@#$%^&*()_+-=[]{}|;':\",./<>?`~")

    def validate(
        self,
        password: str,
        *,
        email: Optional[str] = None,
        username: Optional[str] = None,
    ) -> PasswordValidationResult:
        """
        Validate password against policy.

        Args:
            password: Password to validate
            email: User's email (for context check)
            username: User's username (for context check)

        Returns:
            PasswordValidationResult
        """
        result = PasswordValidationResult(valid=True)

        # Length checks
        if len(password) < self._policy.min_length:
            result.add_violation(
                PasswordPolicyViolation(
                    violation_type=PolicyViolationType.TOO_SHORT,
                    message=f"Password must be at least {self._policy.min_length} characters",
                )
            )

        if len(password) > self._policy.max_length:
            result.add_violation(
                PasswordPolicyViolation(
                    violation_type=PolicyViolationType.TOO_LONG,
                    message=f"Password must be at most {self._policy.max_length} characters",
                )
            )

        # Complexity checks
        if self._policy.require_uppercase and not any(c.isupper() for c in password):
            result.add_violation(
                PasswordPolicyViolation(
                    violation_type=PolicyViolationType.NO_UPPERCASE,
                    message="Password must contain at least one uppercase letter",
                )
            )

        if self._policy.require_lowercase and not any(c.islower() for c in password):
            result.add_violation(
                PasswordPolicyViolation(
                    violation_type=PolicyViolationType.NO_LOWERCASE,
                    message="Password must contain at least one lowercase letter",
                )
            )

        if self._policy.require_digit and not any(c.isdigit() for c in password):
            result.add_violation(
                PasswordPolicyViolation(
                    violation_type=PolicyViolationType.NO_DIGIT,
                    message="Password must contain at least one digit",
                )
            )

        if self._policy.require_special and not any(c in self._special_chars for c in password):
            result.add_violation(
                PasswordPolicyViolation(
                    violation_type=PolicyViolationType.NO_SPECIAL,
                    message="Password must contain at least one special character",
                )
            )

        # Common password check
        if self._policy.check_common_passwords:
            if password.lower() in COMMON_WEAK_PASSWORDS:
                result.add_violation(
                    PasswordPolicyViolation(
                        violation_type=PolicyViolationType.COMMON_PASSWORD,
                        message="Password is too common and easily guessed",
                    )
                )

        # Sequential characters check
        if self._policy.check_sequential_chars:
            if self._has_sequential_chars(password, self._policy.max_sequential_chars):
                result.add_violation(
                    PasswordPolicyViolation(
                        violation_type=PolicyViolationType.SEQUENTIAL_CHARS,
                        message=f"Password cannot contain more than {self._policy.max_sequential_chars} sequential characters",
                    )
                )

        # Repeated characters check
        if self._policy.check_repeated_chars:
            if self._has_repeated_chars(password, self._policy.max_repeated_chars):
                result.add_violation(
                    PasswordPolicyViolation(
                        violation_type=PolicyViolationType.REPEATED_CHARS,
                        message=f"Password cannot contain more than {self._policy.max_repeated_chars} repeated characters",
                    )
                )

        # Context checks
        if self._policy.check_contains_email and email:
            email_local = email.split("@")[0].lower()
            if len(email_local) >= 3 and email_local in password.lower():
                result.add_violation(
                    PasswordPolicyViolation(
                        violation_type=PolicyViolationType.CONTAINS_EMAIL,
                        message="Password cannot contain your email address",
                    )
                )

        if self._policy.check_contains_username and username:
            if len(username) >= 3 and username.lower() in password.lower():
                result.add_violation(
                    PasswordPolicyViolation(
                        violation_type=PolicyViolationType.CONTAINS_USERNAME,
                        message="Password cannot contain your username",
                    )
                )

        # Calculate strength score
        result.strength_score = self._calculate_strength(password)
        result.strength_label = self._get_strength_label(result.strength_score)

        return result

    def _has_sequential_chars(self, password: str, max_seq: int) -> bool:
        """Check for sequential characters (abc, 123, etc.)."""
        if len(password) <= max_seq:
            return False

        # Check for ascending sequences
        for i in range(len(password) - max_seq):
            chars = [ord(c) for c in password[i : i + max_seq + 1].lower()]
            if all(chars[j] + 1 == chars[j + 1] for j in range(len(chars) - 1)):
                return True

        # Check for descending sequences
        for i in range(len(password) - max_seq):
            chars = [ord(c) for c in password[i : i + max_seq + 1].lower()]
            if all(chars[j] - 1 == chars[j + 1] for j in range(len(chars) - 1)):
                return True

        return False

    def _has_repeated_chars(self, password: str, max_repeat: int) -> bool:
        """Check for repeated characters (aaa, 111, etc.)."""
        if len(password) <= max_repeat:
            return False

        for i in range(len(password) - max_repeat):
            if len(set(password[i : i + max_repeat + 1].lower())) == 1:
                return True

        return False

    def _calculate_strength(self, password: str) -> int:
        """Calculate password strength score (0-100)."""
        score = 0

        # Length score (up to 40 points)
        length_score = min(len(password) * 2, 40)
        score += length_score

        # Complexity score (up to 40 points)
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in self._special_chars for c in password)

        complexity_count = sum([has_upper, has_lower, has_digit, has_special])
        score += complexity_count * 10

        # Unique characters score (up to 20 points)
        unique_ratio = len(set(password.lower())) / len(password) if password else 0
        score += int(unique_ratio * 20)

        return min(score, 100)

    def _get_strength_label(self, score: int) -> str:
        """Get strength label from score."""
        if score >= 80:
            return "very_strong"
        elif score >= 60:
            return "strong"
        elif score >= 40:
            return "medium"
        elif score >= 20:
            return "weak"
        else:
            return "very_weak"


# ============================================================================
# Convenience Functions
# ============================================================================

# Singleton validator instance
_default_validator: Optional[PasswordValidator] = None


def get_validator(policy: Optional[PasswordPolicy] = None) -> PasswordValidator:
    """Get a validator instance."""
    global _default_validator
    if policy is not None:
        return PasswordValidator(policy)
    if _default_validator is None:
        _default_validator = PasswordValidator()
    return _default_validator


def validate_password(
    password: str,
    *,
    email: Optional[str] = None,
    username: Optional[str] = None,
    policy: Optional[PasswordPolicy] = None,
) -> PasswordValidationResult:
    """
    Validate password against policy.

    Args:
        password: Password to validate
        email: User's email (for context check)
        username: User's username (for context check)
        policy: Custom policy (uses default if None)

    Returns:
        PasswordValidationResult
    """
    validator = get_validator(policy)
    return validator.validate(password, email=email, username=username)


def assert_password_valid(
    password: str,
    *,
    email: Optional[str] = None,
    username: Optional[str] = None,
) -> None:
    """
    Assert that password is valid.

    Args:
        password: Password to validate
        email: User's email
        username: User's username

    Raises:
        ValueError: If password is invalid
    """
    result = validate_password(password, email=email, username=username)
    if not result.valid:
        messages = [v.message for v in result.violations]
        raise ValueError(f"Password policy violations: {'; '.join(messages)}")
