# -*- coding: utf-8 -*-
"""
tests/test_adapters_config_validation.py
Tests for adapter configuration validation.
"""

import os
import pytest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from adapters.config import (
    # Config classes
    BinanceConfig,
    AlpacaConfig,
    FeeConfig,
    TradingHoursConfig,
    ExchangeConfig,
    ConfigValidationResult,
    # Validation functions
    validate_config,
    validate_config_strict,
    validate_binance_config,
    validate_alpaca_config,
    validate_fee_config,
    get_config_summary,
    # Helper functions
    _validate_url,
    _validate_timeout,
    _validate_fee_bps,
    _validate_vip_tier,
    _validate_alpaca_feed,
    # Constants
    KNOWN_BINANCE_URLS,
    KNOWN_ALPACA_URLS,
    MIN_TIMEOUT_SECONDS,
    MAX_TIMEOUT_SECONDS,
    MIN_FEE_BPS,
    MAX_FEE_BPS,
    VALID_ALPACA_FEEDS,
)


# =============================================================================
# ConfigValidationResult Tests
# =============================================================================


class TestConfigValidationResult:
    """Tests for ConfigValidationResult dataclass."""

    def test_create_valid_result(self):
        """Test creating a valid result."""
        result = ConfigValidationResult(is_valid=True)
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.info == {}

    def test_add_error(self):
        """Test adding an error."""
        result = ConfigValidationResult(is_valid=True)
        result.add_error("Test error")
        assert result.is_valid is False
        assert "Test error" in result.errors

    def test_add_warning(self):
        """Test adding a warning."""
        result = ConfigValidationResult(is_valid=True)
        result.add_warning("Test warning")
        assert result.is_valid is True  # Warnings don't invalidate
        assert "Test warning" in result.warnings

    def test_merge_results(self):
        """Test merging two results."""
        result1 = ConfigValidationResult(is_valid=True)
        result1.add_warning("Warning 1")
        result1.info["key1"] = "value1"

        result2 = ConfigValidationResult(is_valid=True)
        result2.add_error("Error 1")
        result2.add_warning("Warning 2")
        result2.info["key2"] = "value2"

        result1.merge(result2)

        assert result1.is_valid is False
        assert "Error 1" in result1.errors
        assert "Warning 1" in result1.warnings
        assert "Warning 2" in result1.warnings
        assert result1.info["key1"] == "value1"
        assert result1.info["key2"] == "value2"


# =============================================================================
# URL Validation Tests
# =============================================================================


class TestUrlValidation:
    """Tests for URL validation."""

    def test_valid_https_url(self):
        """Test valid HTTPS URL."""
        is_valid, error = _validate_url("https://api.binance.com")
        assert is_valid is True
        assert error is None

    def test_http_url_rejected(self):
        """Test HTTP URL is rejected when HTTPS required."""
        is_valid, error = _validate_url("http://api.binance.com")
        assert is_valid is False
        assert "HTTPS" in error

    def test_http_url_allowed(self):
        """Test HTTP URL allowed when HTTPS not required."""
        is_valid, error = _validate_url("http://localhost:8080", require_https=False)
        assert is_valid is True

    def test_empty_url(self):
        """Test empty URL rejected."""
        is_valid, error = _validate_url("")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_missing_scheme(self):
        """Test URL without scheme rejected."""
        is_valid, error = _validate_url("api.binance.com")
        assert is_valid is False
        assert "scheme" in error.lower()

    def test_invalid_scheme(self):
        """Test invalid scheme rejected."""
        is_valid, error = _validate_url("ftp://api.binance.com")
        assert is_valid is False
        assert "scheme" in error.lower()

    def test_missing_domain(self):
        """Test URL without domain rejected."""
        is_valid, error = _validate_url("https://")
        assert is_valid is False
        assert "domain" in error.lower()


# =============================================================================
# Timeout Validation Tests
# =============================================================================


class TestTimeoutValidation:
    """Tests for timeout validation."""

    def test_valid_timeout(self):
        """Test valid timeout."""
        is_valid, error = _validate_timeout(30)
        assert is_valid is True
        assert error is None

    def test_min_timeout(self):
        """Test minimum timeout."""
        is_valid, error = _validate_timeout(MIN_TIMEOUT_SECONDS)
        assert is_valid is True

    def test_max_timeout(self):
        """Test maximum timeout."""
        is_valid, error = _validate_timeout(MAX_TIMEOUT_SECONDS)
        assert is_valid is True

    def test_timeout_too_low(self):
        """Test timeout below minimum."""
        is_valid, error = _validate_timeout(0)
        assert is_valid is False
        assert "too low" in error.lower()

    def test_timeout_too_high(self):
        """Test timeout above maximum."""
        is_valid, error = _validate_timeout(MAX_TIMEOUT_SECONDS + 1)
        assert is_valid is False
        assert "too high" in error.lower()


# =============================================================================
# Fee Validation Tests
# =============================================================================


class TestFeeValidation:
    """Tests for fee validation."""

    def test_valid_fee(self):
        """Test valid fee."""
        is_valid, error = _validate_fee_bps(10.0, "Maker")
        assert is_valid is True
        assert error is None

    def test_zero_fee(self):
        """Test zero fee (valid)."""
        is_valid, error = _validate_fee_bps(0.0, "Maker")
        assert is_valid is True

    def test_negative_fee(self):
        """Test negative fee rejected."""
        is_valid, error = _validate_fee_bps(-1.0, "Maker")
        assert is_valid is False
        assert "negative" in error.lower()

    def test_fee_too_high(self):
        """Test fee above maximum rejected."""
        is_valid, error = _validate_fee_bps(MAX_FEE_BPS + 1, "Maker")
        assert is_valid is False
        assert "too high" in error.lower()


# =============================================================================
# VIP Tier Validation Tests
# =============================================================================


class TestVipTierValidation:
    """Tests for VIP tier validation."""

    def test_valid_tier(self):
        """Test valid VIP tier."""
        is_valid, error = _validate_vip_tier(0)
        assert is_valid is True

    def test_max_tier(self):
        """Test maximum VIP tier."""
        is_valid, error = _validate_vip_tier(9)
        assert is_valid is True

    def test_invalid_tier_negative(self):
        """Test negative tier rejected."""
        is_valid, error = _validate_vip_tier(-1)
        assert is_valid is False

    def test_invalid_tier_too_high(self):
        """Test tier above maximum rejected."""
        is_valid, error = _validate_vip_tier(10)
        assert is_valid is False


# =============================================================================
# Alpaca Feed Validation Tests
# =============================================================================


class TestAlpacaFeedValidation:
    """Tests for Alpaca feed validation."""

    def test_valid_iex_feed(self):
        """Test valid IEX feed."""
        is_valid, error = _validate_alpaca_feed("iex")
        assert is_valid is True

    def test_valid_sip_feed(self):
        """Test valid SIP feed."""
        is_valid, error = _validate_alpaca_feed("sip")
        assert is_valid is True

    def test_case_insensitive(self):
        """Test feed validation is case insensitive."""
        is_valid, error = _validate_alpaca_feed("IEX")
        assert is_valid is True

    def test_invalid_feed(self):
        """Test invalid feed rejected."""
        is_valid, error = _validate_alpaca_feed("invalid")
        assert is_valid is False
        assert "Invalid Alpaca feed" in error


# =============================================================================
# BinanceConfig Validation Tests
# =============================================================================


class TestValidateBinanceConfig:
    """Tests for Binance config validation."""

    def test_default_config_valid(self):
        """Test default config is valid."""
        config = BinanceConfig()
        result = validate_binance_config(config)
        assert result.is_valid is True

    def test_invalid_url(self):
        """Test invalid base URL."""
        config = BinanceConfig(base_url="http://invalid.com")
        result = validate_binance_config(config)
        assert result.is_valid is False
        assert any("HTTPS" in e for e in result.errors)

    def test_unknown_url_warning(self):
        """Test unknown URL generates warning."""
        config = BinanceConfig(base_url="https://unknown.binance.com")
        result = validate_binance_config(config)
        assert result.is_valid is True  # Valid but with warning
        assert any("not a known Binance URL" in w for w in result.warnings)

    def test_invalid_timeout(self):
        """Test invalid timeout."""
        config = BinanceConfig(timeout=0)
        result = validate_binance_config(config)
        assert result.is_valid is False

    def test_invalid_fee(self):
        """Test invalid fee."""
        config = BinanceConfig(maker_bps=-5.0)
        result = validate_binance_config(config)
        assert result.is_valid is False

    def test_invalid_vip_tier(self):
        """Test invalid VIP tier."""
        config = BinanceConfig(vip_tier=100)
        result = validate_binance_config(config)
        assert result.is_valid is False

    def test_api_credentials_validation(self):
        """Test API credentials validation."""
        config = BinanceConfig(api_key="short", api_secret="also_short")  # Too short
        result = validate_binance_config(config)
        assert result.is_valid is False
        assert any("API credentials" in e for e in result.errors)

    def test_valid_api_credentials(self):
        """Test valid API credentials."""
        config = BinanceConfig(
            api_key="a" * 64, api_secret="b" * 64  # 64 char key  # 64 char secret
        )
        result = validate_binance_config(config)
        # Should be valid (may have other warnings)
        assert "binance_credentials" in result.info


# =============================================================================
# AlpacaConfig Validation Tests
# =============================================================================


class TestValidateAlpacaConfig:
    """Tests for Alpaca config validation."""

    def test_default_config_invalid(self):
        """Test default config is invalid (no API key)."""
        config = AlpacaConfig()
        result = validate_alpaca_config(config)
        assert result.is_valid is False
        assert any("API credentials" in e for e in result.errors)

    def test_valid_config(self):
        """Test valid Alpaca config."""
        config = AlpacaConfig(
            api_key="a" * 20,
            api_secret="b" * 40,
            feed="iex",
            paper=False,
        )
        result = validate_alpaca_config(config)
        assert result.is_valid is True

    def test_invalid_feed(self):
        """Test invalid feed."""
        config = AlpacaConfig(
            api_key="a" * 20,
            api_secret="b" * 40,
            feed="invalid",
        )
        result = validate_alpaca_config(config)
        assert result.is_valid is False
        assert any("Invalid Alpaca feed" in e for e in result.errors)

    def test_paper_mode_warning(self):
        """Test paper mode generates warning."""
        config = AlpacaConfig(
            api_key="a" * 20,
            api_secret="b" * 40,
            paper=True,
        )
        result = validate_alpaca_config(config)
        assert any("Paper trading" in w for w in result.warnings)

    def test_negative_options_fee(self):
        """Test negative options fee rejected."""
        config = AlpacaConfig(
            api_key="a" * 20,
            api_secret="b" * 40,
            options_per_contract_fee=-1.0,
        )
        result = validate_alpaca_config(config)
        assert result.is_valid is False

    def test_high_options_fee_warning(self):
        """Test high options fee generates warning."""
        config = AlpacaConfig(
            api_key="a" * 20,
            api_secret="b" * 40,
            options_per_contract_fee=10.0,
        )
        result = validate_alpaca_config(config)
        assert result.is_valid is True
        assert any("seems high" in w for w in result.warnings)


# =============================================================================
# FeeConfig Validation Tests
# =============================================================================


class TestValidateFeeConfig:
    """Tests for fee config validation."""

    def test_default_config_valid(self):
        """Test default config is valid."""
        config = FeeConfig()
        result = validate_fee_config(config)
        assert result.is_valid is True

    def test_valid_custom_fees(self):
        """Test valid custom fees."""
        config = FeeConfig(
            custom_maker_bps=5.0,
            custom_taker_bps=8.0,
        )
        result = validate_fee_config(config)
        assert result.is_valid is True

    def test_negative_custom_fee(self):
        """Test negative custom fee rejected."""
        config = FeeConfig(custom_maker_bps=-1.0)
        result = validate_fee_config(config)
        assert result.is_valid is False


# =============================================================================
# ExchangeConfig Validation Tests
# =============================================================================


class TestValidateConfig:
    """Tests for main config validation."""

    def test_default_binance_config(self):
        """Test default Binance config."""
        config = ExchangeConfig(vendor="binance")
        result = validate_config(config)
        # Default is valid (no API key required for public endpoints)
        assert "vendor" in result.info
        assert result.info["vendor"] == "binance"

    def test_unknown_vendor(self):
        """Test unknown vendor generates error."""
        config = ExchangeConfig(vendor="unknown_exchange")
        result = validate_config(config)
        assert result.is_valid is False
        assert any("Unknown vendor" in e for e in result.errors)

    def test_unknown_market_type_warning(self):
        """Test unknown market type generates warning."""
        config = ExchangeConfig(vendor="binance", market_type="UNKNOWN_TYPE")
        result = validate_config(config)
        # Should be valid but with warning
        assert any("Unknown market type" in w for w in result.warnings)

    def test_alpaca_config_validation(self):
        """Test Alpaca config validation is triggered."""
        config = ExchangeConfig(
            vendor="alpaca",
            alpaca=AlpacaConfig(
                api_key="",
                api_secret="",
            ),
        )
        result = validate_config(config)
        assert result.is_valid is False

    def test_result_info_populated(self):
        """Test result info is populated."""
        config = ExchangeConfig(vendor="binance")
        result = validate_config(config)
        assert "vendor" in result.info
        assert "market_type" in result.info
        assert "has_secure_logging" in result.info


# =============================================================================
# validate_config_strict Tests
# =============================================================================


class TestValidateConfigStrict:
    """Tests for strict validation."""

    def test_valid_config_no_exception(self):
        """Test valid config doesn't raise."""
        config = ExchangeConfig(vendor="binance")
        # Should not raise
        validate_config_strict(config)

    def test_invalid_config_raises(self):
        """Test invalid config raises ValueError."""
        config = ExchangeConfig(vendor="unknown")
        with pytest.raises(ValueError) as exc_info:
            validate_config_strict(config)
        assert "Invalid configuration" in str(exc_info.value)


# =============================================================================
# get_config_summary Tests
# =============================================================================


class TestGetConfigSummary:
    """Tests for config summary."""

    def test_binance_summary(self):
        """Test Binance config summary."""
        config = ExchangeConfig(
            vendor="binance",
            binance=BinanceConfig(
                api_key="secret_key",
                api_secret="secret_secret",
            ),
        )
        summary = get_config_summary(config)

        assert summary["vendor"] == "binance"
        assert "binance" in summary
        assert summary["binance"]["has_api_key"] is True
        assert summary["binance"]["has_api_secret"] is True
        # Should NOT contain the actual secrets
        assert "secret_key" not in str(summary)
        assert "secret_secret" not in str(summary)

    def test_alpaca_summary(self):
        """Test Alpaca config summary."""
        config = ExchangeConfig(
            vendor="alpaca",
            alpaca=AlpacaConfig(
                api_key="alpaca_key",
                api_secret="alpaca_secret",
                paper=True,
                feed="sip",
            ),
        )
        summary = get_config_summary(config)

        assert summary["vendor"] == "alpaca"
        assert "alpaca" in summary
        assert summary["alpaca"]["paper"] is True
        assert summary["alpaca"]["feed"] == "sip"
        # Should NOT contain the actual secrets
        assert "alpaca_key" not in str(summary)

    def test_trading_hours_in_summary(self):
        """Test trading hours are in summary."""
        config = ExchangeConfig(
            trading_hours=TradingHoursConfig(
                allow_extended=True,
                use_exchange_calendar=False,
            )
        )
        summary = get_config_summary(config)

        assert "trading_hours" in summary
        assert summary["trading_hours"]["allow_extended"] is True
        assert summary["trading_hours"]["use_exchange_calendar"] is False


# =============================================================================
# Environment Variable Resolution Tests
# =============================================================================


class TestEnvVarResolution:
    """Tests for environment variable resolution."""

    def test_resolve_env_var_braces(self):
        """Test ${VAR} syntax."""
        with patch.dict(os.environ, {"TEST_API_KEY": "resolved_value"}):
            config = BinanceConfig(api_key="${TEST_API_KEY}")
            resolved = config._resolve_env(config.api_key)
            assert resolved == "resolved_value"

    def test_resolve_env_var_dollar(self):
        """Test $VAR syntax."""
        with patch.dict(os.environ, {"TEST_API_KEY": "resolved_value"}):
            config = BinanceConfig(api_key="$TEST_API_KEY")
            resolved = config._resolve_env(config.api_key)
            assert resolved == "resolved_value"

    def test_resolve_missing_env_var(self):
        """Test missing env var returns empty string."""
        config = BinanceConfig(api_key="${NONEXISTENT_VAR}")
        resolved = config._resolve_env(config.api_key)
        assert resolved == ""

    def test_resolve_literal_value(self):
        """Test literal value is unchanged."""
        config = BinanceConfig(api_key="literal_api_key")
        resolved = config._resolve_env(config.api_key)
        assert resolved == "literal_api_key"


# =============================================================================
# Integration Tests
# =============================================================================


class TestConfigValidationIntegration:
    """Integration tests for config validation."""

    def test_full_binance_config(self):
        """Test full Binance configuration validation."""
        config = ExchangeConfig(
            vendor="binance",
            market_type="CRYPTO_SPOT",
            binance=BinanceConfig(
                base_url="https://api.binance.com",
                timeout=30,
                maker_bps=9.0,
                taker_bps=10.0,
                vip_tier=1,
            ),
            fees=FeeConfig(
                include_regulatory=True,
            ),
            trading_hours=TradingHoursConfig(
                allow_extended=False,
            ),
        )

        result = validate_config(config)
        assert result.is_valid is True
        assert result.info["vendor"] == "binance"
        assert result.info["market_type"] == "CRYPTO_SPOT"

    def test_full_alpaca_config(self):
        """Test full Alpaca configuration validation."""
        config = ExchangeConfig(
            vendor="alpaca",
            market_type="EQUITY",
            alpaca=AlpacaConfig(
                api_key="a" * 20,
                api_secret="b" * 40,
                paper=False,
                feed="sip",
                extended_hours=True,
            ),
            trading_hours=TradingHoursConfig(
                allow_extended=True,
            ),
        )

        result = validate_config(config)
        assert result.is_valid is True
        assert result.info["vendor"] == "alpaca"

    def test_multiple_validation_errors(self):
        """Test config with multiple validation errors."""
        config = ExchangeConfig(
            vendor="binance",
            binance=BinanceConfig(
                base_url="http://invalid.com",  # Not HTTPS
                timeout=0,  # Too low
                maker_bps=-5.0,  # Negative
                vip_tier=100,  # Too high
            ),
        )

        result = validate_config(config)
        assert result.is_valid is False
        assert len(result.errors) >= 4  # At least 4 errors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
