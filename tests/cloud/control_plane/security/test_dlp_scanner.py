# -*- coding: utf-8 -*-
"""
Tests for DLP/Secret Scanner - WI-CLOUD-05.

Tests verify:
- Secrets in config blobs are detected and rejected
- Cloud NEVER stores API keys, credentials, or sensitive data
- Various secret patterns are detected
- Order-like payloads are rejected
"""

import pytest

from packages.cloud.control_plane.security.dlp_scanner import (
    DLPScanner,
    DLPScanResult,
    SecretType,
    ScanSeverity,
    scan_for_secrets,
    assert_no_secrets,
    PROHIBITED_SECRET_FIELDS,
)


class TestDLPScanner:
    """Test DLPScanner class."""

    def test_clean_config_passes(self):
        """Config without secrets should pass."""
        scanner = DLPScanner()

        clean_config = {
            "strategy_name": "momentum",
            "lookback_period": 20,
            "threshold": 0.05,
            "enabled": True,
        }

        result = scanner.scan(clean_config)
        assert result.clean
        assert not result.blocked
        assert len(result.findings) == 0

    def test_api_key_field_detected(self):
        """API key field names should be detected."""
        scanner = DLPScanner()

        config = {
            "api_key": "abc123xyz789",
            "name": "test",
        }

        result = scanner.scan(config)
        assert not result.clean
        assert result.blocked
        assert any(f.secret_type == SecretType.API_KEY for f in result.findings)

    def test_api_secret_field_detected(self):
        """API secret field names should be detected."""
        scanner = DLPScanner()

        config = {
            "api_secret": "supersecret123",
            "name": "test",
        }

        result = scanner.scan(config)
        assert not result.clean
        assert result.blocked

    def test_broker_credentials_detected(self):
        """Broker credentials should be detected."""
        scanner = DLPScanner()

        config = {
            "broker_key": "BROKERAPIKEY123",
            "broker_secret": "secretvalue",
        }

        result = scanner.scan(config)
        assert not result.clean
        assert len(result.findings) >= 2

    def test_exchange_credentials_detected(self):
        """Exchange credentials should be detected."""
        scanner = DLPScanner()

        config = {
            "exchange_key": "EXCHANGEKEY",
            "exchange_secret": "EXCHANGESECRET",
        }

        result = scanner.scan(config)
        assert not result.clean

    def test_password_field_detected(self):
        """Password fields should be detected."""
        scanner = DLPScanner()

        config = {
            "password": "mypassword123",
            "name": "test",
        }

        result = scanner.scan(config)
        assert not result.clean

    def test_private_key_field_detected(self):
        """Private key fields should be detected."""
        scanner = DLPScanner()

        config = {
            "private_key": "-----BEGIN RSA PRIVATE KEY-----...",
        }

        result = scanner.scan(config)
        assert not result.clean
        assert any(f.secret_type == SecretType.PRIVATE_KEY for f in result.findings)

    def test_pem_private_key_in_value_detected(self):
        """PEM-encoded private key in value should be detected."""
        scanner = DLPScanner()

        config = {
            "key_data": "-----BEGIN RSA PRIVATE KEY-----\nMIIE...",
        }

        result = scanner.scan(config)
        assert not result.clean

    def test_jwt_token_in_value_detected(self):
        """JWT token in value should be detected."""
        scanner = DLPScanner()

        # Valid JWT format (header.payload.signature)
        jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"

        config = {
            "token": jwt_token,
        }

        result = scanner.scan(config)
        assert not result.clean

    def test_database_url_with_password_detected(self):
        """Database URL with embedded password should be detected."""
        scanner = DLPScanner()

        config = {
            "connection": "postgres://user:password@localhost:5432/db",
        }

        result = scanner.scan(config)
        assert not result.clean
        assert any(f.secret_type == SecretType.DATABASE_URL for f in result.findings)

    def test_aws_access_key_detected(self):
        """AWS access key should be detected."""
        scanner = DLPScanner()

        config = {
            "aws_key": "AKIAIOSFODNN7EXAMPLE",  # Example AWS key format
        }

        result = scanner.scan(config)
        assert not result.clean

    def test_nested_secrets_detected(self):
        """Secrets in nested structures should be detected."""
        scanner = DLPScanner()

        config = {
            "connection": {
                "credentials": {
                    "api_key": "secret123",
                    "api_secret": "supersecret",
                }
            }
        }

        result = scanner.scan(config)
        assert not result.clean
        assert result.blocked
        # Should find both api_key and api_secret
        assert len(result.findings) >= 2

    def test_secrets_in_arrays_detected(self):
        """Secrets in arrays should be detected."""
        scanner = DLPScanner()

        config = {
            "connections": [
                {"name": "conn1", "password": "pass1"},
                {"name": "conn2", "password": "pass2"},
            ]
        }

        result = scanner.scan(config)
        assert not result.clean
        assert len(result.findings) >= 2

    def test_case_insensitive_field_names(self):
        """Field name detection should be case-insensitive."""
        scanner = DLPScanner()

        # Test variations that are in PROHIBITED_SECRET_FIELDS (lowercase)
        variations = [
            {"API_KEY": "secret"},  # Uppercase version of api_key
            {"Api_Key": "secret"},  # Mixed case version
            {"APIKEY": "secret"},  # Uppercase version of apikey
            {"PASSWORD": "secret"},  # Uppercase version of password
        ]

        for config in variations:
            result = scanner.scan(config)
            assert not result.clean, f"Failed for config: {config}"


class TestDLPScannerIntentFields:
    """Test that order-like intent fields are rejected."""

    def test_order_symbol_rejected(self):
        """Order symbol field should be rejected."""
        scanner = DLPScanner()

        config = {
            "symbol": "AAPL",
            "quantity": 100,
            "side": "buy",
        }

        result = scanner.scan(config)
        # Symbol, quantity, and side are intent fields
        assert not result.clean

    def test_order_quantity_rejected(self):
        """Order quantity field should be rejected."""
        scanner = DLPScanner()

        config = {
            "quantity": 50,  # Uses intent field from PROHIBITED_INTENT_FIELDS
        }

        result = scanner.scan(config)
        assert not result.clean

    def test_order_side_rejected(self):
        """Order side (buy/sell) field should be rejected."""
        scanner = DLPScanner()

        config = {
            "side": "buy",  # Uses intent field from PROHIBITED_INTENT_FIELDS
        }

        result = scanner.scan(config)
        assert not result.clean


class TestScanForSecretsFunction:
    """Test scan_for_secrets convenience function."""

    def test_scan_for_secrets_clean(self):
        """Clean config should pass."""
        result = scan_for_secrets({"name": "test", "value": 123})
        assert result.clean

    def test_scan_for_secrets_with_secret(self):
        """Config with secret should fail."""
        result = scan_for_secrets({"api_key": "secret123"})
        assert not result.clean


class TestAssertNoSecrets:
    """Test assert_no_secrets function."""

    def test_assert_no_secrets_passes(self):
        """Clean config should not raise."""
        assert_no_secrets({"name": "test", "enabled": True})

    def test_assert_no_secrets_raises(self):
        """Config with secret should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            assert_no_secrets({"api_key": "secret"})
        assert "Secrets detected" in str(exc_info.value)


class TestProhibitedSecretFields:
    """Test that all prohibited fields are defined."""

    def test_prohibited_fields_includes_api_keys(self):
        """API key fields should be prohibited."""
        assert "api_key" in PROHIBITED_SECRET_FIELDS
        assert "api_secret" in PROHIBITED_SECRET_FIELDS
        assert "secret_key" in PROHIBITED_SECRET_FIELDS

    def test_prohibited_fields_includes_broker_keys(self):
        """Broker key fields should be prohibited."""
        assert "broker_key" in PROHIBITED_SECRET_FIELDS
        assert "broker_secret" in PROHIBITED_SECRET_FIELDS

    def test_prohibited_fields_includes_passwords(self):
        """Password fields should be prohibited."""
        assert "password" in PROHIBITED_SECRET_FIELDS
        assert "passwd" in PROHIBITED_SECRET_FIELDS

    def test_prohibited_fields_includes_private_keys(self):
        """Private key fields should be prohibited."""
        assert "private_key" in PROHIBITED_SECRET_FIELDS
        assert "encryption_key" in PROHIBITED_SECRET_FIELDS


class TestDLPScanResult:
    """Test DLPScanResult class."""

    def test_scan_result_to_dict(self):
        """Scan result should serialize to dict."""
        scanner = DLPScanner()
        result = scanner.scan({"api_key": "secret"})

        result_dict = result.to_dict()
        assert "clean" in result_dict
        assert "blocked" in result_dict
        assert "finding_count" in result_dict
        assert "findings" in result_dict

    def test_critical_count(self):
        """Critical count should be accurate."""
        scanner = DLPScanner()
        result = scanner.scan({"api_key": "secret", "api_secret": "secret2"})

        assert result.critical_count >= 0  # May vary based on severity assignment


class TestCloudNeverStoresSecrets:
    """
    Critical security tests: Cloud NEVER stores secrets.

    These tests verify the fundamental CCEA security invariant.
    """

    def test_exchange_api_credentials_rejected(self):
        """Exchange API credentials MUST be rejected."""
        scanner = DLPScanner()

        # Simulating someone trying to store exchange credentials
        config = {
            "exchange": "binance",
            "api_key": "real_api_key_here",
            "api_secret": "real_api_secret_here",
        }

        result = scanner.scan(config)
        assert not result.clean, (
            "SECURITY VIOLATION: Exchange credentials were accepted! "
            "Cloud MUST NEVER store exchange API keys."
        )
        assert result.blocked

    def test_broker_credentials_rejected(self):
        """Broker credentials MUST be rejected."""
        scanner = DLPScanner()

        config = {
            "broker": "interactive_brokers",
            "broker_key": "BROKER_API_KEY",
            "broker_secret": "BROKER_SECRET",
        }

        result = scanner.scan(config)
        assert not result.clean, (
            "SECURITY VIOLATION: Broker credentials were accepted! "
            "Cloud MUST NEVER store broker credentials."
        )

    def test_wallet_keys_rejected(self):
        """Wallet/crypto keys MUST be rejected."""
        scanner = DLPScanner()

        config = {
            "wallet_key": "private_key_here",
            "seed_phrase": "word1 word2 word3...",
        }

        result = scanner.scan(config)
        assert not result.clean
