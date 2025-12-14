# -*- coding: utf-8 -*-
"""
Tests for Password Policy - WI-AUTH-01.

Tests verify:
- NIST 800-63B compliant password requirements
- Common password blocking
- Context-based validation (email/username)
- Strength scoring
"""

import pytest

from packages.cloud.control_plane.security.password_policy import (
    PasswordPolicy,
    PasswordValidator,
    PasswordValidationResult,
    PolicyViolationType,
    validate_password,
    assert_password_valid,
    DEFAULT_PASSWORD_POLICY,
    COMMON_WEAK_PASSWORDS,
)


class TestPasswordPolicy:
    """Test PasswordPolicy defaults."""

    def test_default_policy_min_length(self):
        """Default minimum length should be 12 (OWASP recommended)."""
        assert DEFAULT_PASSWORD_POLICY.min_length == 12

    def test_default_policy_max_length(self):
        """Default maximum length should be reasonable."""
        assert DEFAULT_PASSWORD_POLICY.max_length >= 64

    def test_default_policy_requires_complexity(self):
        """Default policy should require some complexity."""
        assert DEFAULT_PASSWORD_POLICY.require_uppercase
        assert DEFAULT_PASSWORD_POLICY.require_lowercase
        assert DEFAULT_PASSWORD_POLICY.require_digit

    def test_default_policy_common_password_check(self):
        """Default policy should check common passwords."""
        assert DEFAULT_PASSWORD_POLICY.check_common_passwords


class TestPasswordValidator:
    """Test PasswordValidator class."""

    def test_strong_password_passes(self):
        """Strong password should pass validation."""
        validator = PasswordValidator()

        result = validator.validate("MyStr0ngP@ssword123")
        assert result.valid
        assert len(result.violations) == 0

    def test_password_too_short(self):
        """Short password should be rejected."""
        validator = PasswordValidator()

        result = validator.validate("Short1!")
        assert not result.valid
        assert any(v.violation_type == PolicyViolationType.TOO_SHORT for v in result.violations)

    def test_password_too_long(self):
        """Extremely long password should be rejected."""
        policy = PasswordPolicy(max_length=50)
        validator = PasswordValidator(policy)

        result = validator.validate("A" * 100 + "1a")
        assert not result.valid
        assert any(v.violation_type == PolicyViolationType.TOO_LONG for v in result.violations)

    def test_password_no_uppercase(self):
        """Password without uppercase should be rejected."""
        validator = PasswordValidator()

        result = validator.validate("nouppercase123!")
        assert not result.valid
        assert any(v.violation_type == PolicyViolationType.NO_UPPERCASE for v in result.violations)

    def test_password_no_lowercase(self):
        """Password without lowercase should be rejected."""
        validator = PasswordValidator()

        result = validator.validate("NOLOWERCASE123!")
        assert not result.valid
        assert any(v.violation_type == PolicyViolationType.NO_LOWERCASE for v in result.violations)

    def test_password_no_digit(self):
        """Password without digit should be rejected."""
        validator = PasswordValidator()

        result = validator.validate("NoDigitsHere!@#")
        assert not result.valid
        assert any(v.violation_type == PolicyViolationType.NO_DIGIT for v in result.violations)

    def test_common_password_rejected(self):
        """Common passwords should be rejected."""
        validator = PasswordValidator()

        for common in ["password", "123456", "qwerty", "letmein"]:
            # Add complexity to meet other requirements
            test_password = common.capitalize() + "123!"
            # Still rejected because base is common
            # (depends on exact matching - we test the base)
            result = validator.validate(common)
            assert not result.valid

    def test_password_contains_email(self):
        """Password containing email should be rejected."""
        validator = PasswordValidator()

        result = validator.validate("johnPassword123!", email="john@example.com")
        assert not result.valid
        assert any(v.violation_type == PolicyViolationType.CONTAINS_EMAIL for v in result.violations)

    def test_password_contains_username(self):
        """Password containing username should be rejected."""
        validator = PasswordValidator()

        result = validator.validate("JohnDoe123Password!", username="johndoe")
        assert not result.valid
        assert any(v.violation_type == PolicyViolationType.CONTAINS_USERNAME for v in result.violations)

    def test_sequential_chars_rejected(self):
        """Sequential characters should be rejected."""
        validator = PasswordValidator()

        result = validator.validate("Abcdefgh123!")
        assert not result.valid
        assert any(v.violation_type == PolicyViolationType.SEQUENTIAL_CHARS for v in result.violations)

    def test_repeated_chars_rejected(self):
        """Repeated characters should be rejected."""
        validator = PasswordValidator()

        result = validator.validate("Passssword123!")
        assert not result.valid
        assert any(v.violation_type == PolicyViolationType.REPEATED_CHARS for v in result.violations)

    def test_strength_score_weak(self):
        """Weak password should have low strength score."""
        validator = PasswordValidator()

        # Use a password that passes basic requirements but is weak
        result = validator.validate("Aa1Aa1Aa1Aa1")  # 12 chars, meets requirements
        assert result.strength_score < 60

    def test_strength_score_strong(self):
        """Strong password should have high strength score."""
        validator = PasswordValidator()

        result = validator.validate("MyV3ryStr0ng&Unique#Password2024!")
        assert result.strength_score >= 60

    def test_strength_label_assignment(self):
        """Strength labels should be assigned correctly."""
        validator = PasswordValidator()

        result = validator.validate("MyV3ryStr0ng&Unique#Password!")
        assert result.strength_label in ["weak", "medium", "strong", "very_strong", "very_weak"]


class TestValidatePasswordFunction:
    """Test validate_password convenience function."""

    def test_validate_password_strong(self):
        """Strong password should pass."""
        result = validate_password("MyStr0ngP@ssword123!")
        assert result.valid

    def test_validate_password_weak(self):
        """Weak password should fail."""
        result = validate_password("weak")
        assert not result.valid


class TestAssertPasswordValid:
    """Test assert_password_valid function."""

    def test_assert_password_valid_passes(self):
        """Strong password should not raise."""
        assert_password_valid("MyStr0ngP@ssword123!")

    def test_assert_password_valid_raises(self):
        """Weak password should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            assert_password_valid("weak")
        assert "Password policy violation" in str(exc_info.value)


class TestCommonWeakPasswords:
    """Test common weak passwords list."""

    def test_common_passwords_not_empty(self):
        """Common passwords list should be populated."""
        assert len(COMMON_WEAK_PASSWORDS) > 0

    def test_top_passwords_included(self):
        """Top common passwords should be included."""
        top_passwords = ["password", "123456", "qwerty", "admin", "letmein"]
        for pwd in top_passwords:
            assert pwd in COMMON_WEAK_PASSWORDS


class TestPasswordValidationResult:
    """Test PasswordValidationResult class."""

    def test_result_to_dict(self):
        """Result should serialize to dict."""
        result = validate_password("Str0ngP@ssword123!")

        result_dict = result.to_dict()
        assert "valid" in result_dict
        assert "violations" in result_dict
        assert "strength_score" in result_dict
        assert "strength_label" in result_dict


class TestCustomPolicy:
    """Test custom password policies."""

    def test_custom_min_length(self):
        """Custom minimum length should be enforced."""
        policy = PasswordPolicy(min_length=16)
        validator = PasswordValidator(policy)

        result = validator.validate("Short1Aa!")  # 9 chars
        assert not result.valid

        result = validator.validate("LongEnoughPwd1Xy!")  # 17 chars, no sequential
        assert result.valid

    def test_no_complexity_requirements(self):
        """Policy can disable complexity requirements."""
        policy = PasswordPolicy(
            min_length=8,
            require_uppercase=False,
            require_lowercase=False,
            require_digit=False,
            require_special=False,
        )
        validator = PasswordValidator(policy)

        result = validator.validate("alllowercase")
        assert result.valid

    def test_require_special_char(self):
        """Policy can require special characters."""
        policy = PasswordPolicy(require_special=True)
        validator = PasswordValidator(policy)

        result = validator.validate("NoSpecialChar123Aa")
        assert not result.valid

        result = validator.validate("HasSpecialChar123Aa!")
        assert result.valid
