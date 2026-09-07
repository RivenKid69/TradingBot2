# -*- coding: utf-8 -*-
"""
Tests for DLP (Data Loss Prevention) Service.

CCEA Phase 8 - DLP service tests.
"""

import pytest
from datetime import datetime

from packages.agent.telemetry.dlp import (
    DLPService,
    DLPConfig,
    DLPRule,
    DLPAction,
    DLPViolation,
    SensitivityLevel,
    CRITICAL_PATTERNS,
    create_dlp_service,
    check_for_sensitive_data,
)


class TestDLPServiceBasic:
    """Basic DLP service tests."""

    def test_create_dlp_service(self):
        """Test creating DLP service."""
        dlp = DLPService()
        assert dlp is not None
        assert dlp.config is not None

    def test_create_dlp_service_strict_mode(self):
        """Test creating DLP service in strict mode."""
        dlp = create_dlp_service(strict=True)
        assert dlp.config.strict_mode is True

    def test_clean_data_passes(self):
        """Test clean data passes through."""
        dlp = DLPService()
        data = {"status": "ok", "count": 42}
        result = dlp.process(data)

        assert result.blocked is False
        assert result.has_violations is False
        assert result.data["status"] == "ok"


class TestCriticalDataBlocking:
    """Critical data blocking tests."""

    @pytest.mark.parametrize(
        "field_name",
        [
            "private_key",
            "secret_key",
            "api_secret",
            "encryption_key",
            "seed_phrase",
            "mnemonic",
        ],
    )
    def test_critical_fields_blocked(self, field_name):
        """Test critical fields are blocked."""
        dlp = DLPService()
        data = {field_name: "critical_secret_data_12345"}
        result = dlp.process(data)

        assert result.blocked is True
        assert result.action == DLPAction.BLOCK
        assert len(result.critical_violations) > 0

    def test_api_credentials_blocked(self):
        """Test API credentials are blocked."""
        dlp = DLPService()
        data = {"api_key": "AKIAIOSFODNN7EXAMPLE123456"}
        result = dlp.process(data)

        assert result.blocked is True

    def test_nested_critical_data_blocked(self):
        """Test nested critical data is blocked."""
        dlp = DLPService()
        data = {"config": {"broker": {"private_key": "-----BEGIN PRIVATE KEY-----"}}}
        result = dlp.process(data)

        assert result.blocked is True


class TestSensitiveDataMasking:
    """Sensitive data masking tests."""

    @pytest.mark.parametrize(
        "field_name",
        [
            "account_number",
            "card_number",
            "bank_account",
            "iban",
            "swift",
            "routing_number",
        ],
    )
    def test_financial_data_masked(self, field_name):
        """Test financial data is masked."""
        dlp = DLPService()
        data = {field_name: "1234567890123456"}
        result = dlp.process(data)

        assert result.has_violations
        # Should be masked, not blocked
        if not result.blocked:
            assert "***" in str(result.data.get(field_name, ""))

    @pytest.mark.parametrize(
        "field_name",
        [
            "ssn",
            "social_security",
            "tax_id",
        ],
    )
    def test_pii_data_masked(self, field_name):
        """Test PII data is masked."""
        dlp = DLPService()
        data = {field_name: "123-45-6789"}
        result = dlp.process(data)

        assert result.has_violations


class TestValuePatterns:
    """Value pattern matching tests."""

    def test_aws_key_pattern_detected(self):
        """Test AWS key pattern detection."""
        dlp = DLPService()
        data = {"log": "Using key AKIAIOSFODNN7EXAMPLE"}
        result = dlp.process(data)

        assert result.has_violations

    def test_credit_card_pattern_detected(self):
        """Test credit card pattern detection."""
        dlp = DLPService()
        data = {"transaction": "Card: 4111-1111-1111-1111"}
        result = dlp.process(data)

        assert result.has_violations

    def test_email_pattern_detected(self):
        """Test email pattern detection."""
        dlp = DLPService()
        data = {"user": "john.doe@example.com"}
        result = dlp.process(data)

        assert result.has_violations

    def test_ssn_pattern_detected(self):
        """Test SSN pattern detection."""
        dlp = DLPService()
        data = {"info": "SSN is 123-45-6789"}
        result = dlp.process(data)

        assert result.has_violations


class TestDLPRules:
    """DLP rule tests."""

    def test_add_custom_rule(self):
        """Test adding custom rules."""
        dlp = DLPService()
        initial_count = len(dlp.get_rules())

        custom_rule = DLPRule(
            name="custom_rule",
            sensitivity=SensitivityLevel.CONFIDENTIAL,
            field_patterns=frozenset({"custom_field"}),
            action=DLPAction.MASK,
        )
        dlp.add_rule(custom_rule)

        assert len(dlp.get_rules()) == initial_count + 1

    def test_mandatory_rules_stay_enabled(self):
        """Test mandatory rules cannot be disabled."""
        rule = DLPRule(
            name="mandatory",
            sensitivity=SensitivityLevel.CRITICAL,
            is_mandatory=True,
            enabled=False,  # Try to disable
        )

        # Should auto-enable
        assert rule.enabled is True
        assert rule.action == DLPAction.BLOCK  # Critical must block

    def test_rule_matches_field(self):
        """Test rule field matching."""
        rule = DLPRule(
            name="test",
            field_patterns=frozenset({"secret", "key"}),
        )

        assert rule.matches("api_secret", "value") is True
        assert rule.matches("api_key", "value") is True
        assert rule.matches("status", "value") is False


class TestDataClassification:
    """Data classification tests."""

    def test_classify_data(self):
        """Test data classification without processing."""
        dlp = DLPService()
        data = {
            "status": "ok",
            "api_key": "secret123",
            "email": "test@example.com",
        }

        classifications = dlp.classify_data(data)

        assert "api_key" in classifications
        assert classifications["api_key"] == SensitivityLevel.CRITICAL

    def test_classify_nested_data(self):
        """Test nested data classification."""
        dlp = DLPService()
        data = {"config": {"secret_key": "abc123"}}

        classifications = dlp.classify_data(data)

        assert any("secret_key" in k for k in classifications.keys())


class TestViolationHistory:
    """Violation history tests."""

    def test_violation_logged(self):
        """Test violations are logged."""
        dlp = DLPService()

        # Clear history first
        dlp.clear_violation_history()

        # Process data with violation
        dlp.process({"api_key": "secret123"})

        history = dlp.get_violation_history()
        assert len(history) > 0

    def test_violation_history_limit(self):
        """Test violation history respects limit."""
        dlp = DLPService()
        dlp.clear_violation_history()

        # Generate multiple violations
        for i in range(10):
            dlp.process({f"api_key_{i}": "secret"})

        history = dlp.get_violation_history(limit=5)
        assert len(history) <= 5

    def test_clear_violation_history(self):
        """Test clearing violation history."""
        dlp = DLPService()
        dlp.process({"api_key": "secret"})
        dlp.clear_violation_history()

        history = dlp.get_violation_history()
        assert len(history) == 0


class TestDLPViolation:
    """DLP violation tests."""

    def test_violation_to_dict(self):
        """Test violation serialization."""
        violation = DLPViolation(
            rule_name="test_rule",
            field_path="config.api_key",
            sensitivity=SensitivityLevel.CRITICAL,
            action_taken=DLPAction.BLOCK,
        )

        data = violation.to_dict()

        assert data["rule_name"] == "test_rule"
        assert data["field_path"] == "config.api_key"
        assert data["sensitivity"] == "CRITICAL"
        assert data["action_taken"] == "BLOCK"


class TestDLPResult:
    """DLP result tests."""

    def test_result_properties(self):
        """Test DLP result properties."""
        dlp = DLPService()

        # Clean data
        result = dlp.process({"status": "ok"})
        assert result.has_violations is False
        assert len(result.critical_violations) == 0

        # Dirty data
        result = dlp.process({"private_key": "secret"})
        assert result.has_violations is True
        assert len(result.critical_violations) > 0


class TestConvenienceFunctions:
    """Convenience function tests."""

    def test_check_for_sensitive_data_clean(self):
        """Test check function with clean data."""
        result = check_for_sensitive_data({"status": "ok"})
        assert result is False

    def test_check_for_sensitive_data_dirty(self):
        """Test check function with sensitive data."""
        result = check_for_sensitive_data({"api_key": "secret"})
        assert result is True


class TestDLPConfig:
    """DLP configuration tests."""

    def test_custom_sensitivity_actions(self):
        """Test custom sensitivity actions."""
        config = DLPConfig(
            sensitivity_actions={
                SensitivityLevel.CONFIDENTIAL: DLPAction.BLOCK,
            }
        )
        dlp = DLPService(config)

        assert dlp.config.sensitivity_actions[SensitivityLevel.CONFIDENTIAL] == DLPAction.BLOCK

    def test_audit_log_disabled(self):
        """Test disabling audit log."""
        config = DLPConfig(enable_audit_log=False)
        dlp = DLPService(config)

        dlp.process({"api_key": "secret"})

        # Should still track internally
        assert dlp.config.enable_audit_log is False
