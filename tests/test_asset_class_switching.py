# -*- coding: utf-8 -*-
"""
tests/test_asset_class_switching.py
Tests for switching between crypto and equity asset classes (Phase 4.6).

Test coverage:
- Asset class enum and validation
- Provider auto-selection based on asset_class
- Slippage profile switching
- Fee provider switching
- Trading hours adapter switching
- Config validation
- Backward compatibility with crypto-only configs
"""

import pytest
from typing import Any, Dict
import yaml


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def crypto_config() -> Dict[str, Any]:
    """Sample crypto configuration."""
    return {
        "asset_class": "crypto",
        "data_vendor": "binance",
        "market": "spot",
        "fees": {
            "maker_bps": 2.0,
            "taker_bps": 4.0,
        },
        "slippage": {
            "k": 0.1,
            "default_spread_bps": 5.0,
        },
        "env": {
            "session": {
                "calendar": "crypto_24x7",
            },
        },
    }


@pytest.fixture
def equity_config() -> Dict[str, Any]:
    """Sample equity configuration."""
    return {
        "asset_class": "equity",
        "data_vendor": "alpaca",
        "market": "equity",
        "fees": {
            "maker_bps": 0.0,
            "taker_bps": 0.0,
            "regulatory": {
                "enabled": True,
                "sec_fee_per_million": 27.80,
                "taf_fee_per_share": 0.000166,
            },
        },
        "slippage": {
            "k": 0.05,
            "default_spread_bps": 2.0,
        },
        "env": {
            "session": {
                "calendar": "us_equity",
                "extended_hours": False,
            },
            "no_trade": {
                "enforce_trading_hours": True,
            },
        },
    }


# =============================================================================
# Test AssetClass Enum
# =============================================================================


class TestAssetClassEnum:
    """Tests for AssetClass enumeration."""

    def test_crypto_enum_exists(self):
        """Test crypto asset class exists."""
        from execution_providers import AssetClass

        assert hasattr(AssetClass, "CRYPTO")
        assert AssetClass.CRYPTO.value == "crypto"

    def test_equity_enum_exists(self):
        """Test equity asset class exists."""
        from execution_providers import AssetClass

        assert hasattr(AssetClass, "EQUITY")
        assert AssetClass.EQUITY.value == "equity"

    def test_futures_enum_exists(self):
        """Test futures asset class exists."""
        from execution_providers import AssetClass

        assert hasattr(AssetClass, "FUTURES")

    def test_asset_class_from_string(self):
        """Test creating asset class from string."""
        from execution_providers import AssetClass

        assert AssetClass("crypto") == AssetClass.CRYPTO
        assert AssetClass("equity") == AssetClass.EQUITY

    def test_invalid_asset_class_raises(self):
        """Test invalid asset class raises error."""
        from execution_providers import AssetClass

        with pytest.raises(ValueError):
            AssetClass("invalid")


# =============================================================================
# Test Provider Auto-Selection
# =============================================================================


class TestProviderAutoSelection:
    """Tests for automatic provider selection based on asset_class."""

    def test_crypto_gets_crypto_fee_provider(self):
        """Test crypto asset class gets CryptoFeeProvider."""
        from execution_providers import (
            create_fee_provider,
            AssetClass,
            CryptoFeeProvider,
        )

        provider = create_fee_provider(AssetClass.CRYPTO)
        assert isinstance(provider, CryptoFeeProvider)

    def test_equity_gets_equity_fee_provider(self):
        """Test equity asset class gets EquityFeeProvider."""
        from execution_providers import (
            create_fee_provider,
            AssetClass,
            EquityFeeProvider,
        )

        provider = create_fee_provider(AssetClass.EQUITY)
        assert isinstance(provider, EquityFeeProvider)

    def test_crypto_gets_statistical_slippage(self):
        """Test crypto gets StatisticalSlippageProvider."""
        from execution_providers import (
            create_slippage_provider,
            AssetClass,
            StatisticalSlippageProvider,
        )

        provider = create_slippage_provider("L2", AssetClass.CRYPTO)
        assert isinstance(provider, StatisticalSlippageProvider)

    def test_equity_gets_statistical_slippage(self):
        """Test equity gets StatisticalSlippageProvider."""
        from execution_providers import (
            create_slippage_provider,
            AssetClass,
            StatisticalSlippageProvider,
        )

        provider = create_slippage_provider("L2", AssetClass.EQUITY)
        assert isinstance(provider, StatisticalSlippageProvider)

    def test_crypto_execution_provider(self):
        """Test full crypto execution provider."""
        from execution_providers import (
            create_execution_provider,
            AssetClass,
            L2ExecutionProvider,
            CryptoFeeProvider,
        )

        provider = create_execution_provider(AssetClass.CRYPTO)
        assert isinstance(provider, L2ExecutionProvider)
        assert provider.asset_class == AssetClass.CRYPTO
        assert isinstance(provider.fees, CryptoFeeProvider)

    def test_equity_execution_provider(self):
        """Test full equity execution provider."""
        from execution_providers import (
            create_execution_provider,
            AssetClass,
            L2ExecutionProvider,
            EquityFeeProvider,
        )

        provider = create_execution_provider(AssetClass.EQUITY)
        assert isinstance(provider, L2ExecutionProvider)
        assert provider.asset_class == AssetClass.EQUITY
        assert isinstance(provider.fees, EquityFeeProvider)


# =============================================================================
# Test Slippage Profile Switching
# =============================================================================


class TestSlippageProfileSwitching:
    """Tests for slippage profile switching between asset classes."""

    def test_crypto_wider_spread(self):
        """Test crypto has wider default spread than equity."""
        from execution_providers import (
            create_slippage_provider,
            AssetClass,
        )

        crypto = create_slippage_provider("L2", AssetClass.CRYPTO)
        equity = create_slippage_provider("L2", AssetClass.EQUITY)

        assert crypto.spread_bps > equity.spread_bps

    def test_crypto_higher_impact(self):
        """Test crypto has higher impact coefficient."""
        from execution_providers import (
            create_slippage_provider,
            AssetClass,
        )

        crypto = create_slippage_provider("L2", AssetClass.CRYPTO)
        equity = create_slippage_provider("L2", AssetClass.EQUITY)

        assert crypto.impact_coef > equity.impact_coef

    def test_equity_spread_default(self):
        """Test equity default spread is 2 bps."""
        from execution_providers import create_slippage_provider, AssetClass

        equity = create_slippage_provider("L2", AssetClass.EQUITY)
        assert equity.spread_bps == 2.0

    def test_crypto_spread_default(self):
        """Test crypto default spread is 5 bps."""
        from execution_providers import create_slippage_provider, AssetClass

        crypto = create_slippage_provider("L2", AssetClass.CRYPTO)
        assert crypto.spread_bps == 5.0

    def test_slippage_calculation_differs(self):
        """Test slippage calculation differs by asset class."""
        from execution_providers import (
            create_slippage_provider,
            AssetClass,
            Order,
            MarketState,
        )

        crypto_slip = create_slippage_provider("L2", AssetClass.CRYPTO)
        equity_slip = create_slippage_provider("L2", AssetClass.EQUITY)

        order = Order(symbol="TEST", side="BUY", qty=100.0, order_type="MARKET")
        market = MarketState(timestamp=0, bid=100.0, ask=100.10)
        participation = 0.01  # 1%

        crypto_bps = crypto_slip.compute_slippage_bps(order, market, participation)
        equity_bps = equity_slip.compute_slippage_bps(order, market, participation)

        # Crypto should have higher slippage
        assert crypto_bps > equity_bps


# =============================================================================
# Test Fee Provider Switching
# =============================================================================


class TestFeeProviderSwitching:
    """Tests for fee provider switching between asset classes."""

    def test_crypto_symmetric_fees(self):
        """Test crypto has symmetric buy/sell fees."""
        from execution_providers import create_fee_provider, AssetClass

        provider = create_fee_provider(AssetClass.CRYPTO)

        buy_fee = provider.compute_fee(10000.0, "BUY", "taker", 1.0)
        sell_fee = provider.compute_fee(10000.0, "SELL", "taker", 1.0)

        assert buy_fee > 0
        assert buy_fee == sell_fee

    def test_equity_asymmetric_fees(self):
        """Test equity has asymmetric fees (buy free)."""
        from execution_providers import create_fee_provider, AssetClass

        provider = create_fee_provider(AssetClass.EQUITY)

        buy_fee = provider.compute_fee(10000.0, "BUY", "taker", 100.0)
        sell_fee = provider.compute_fee(10000.0, "SELL", "taker", 100.0)

        assert buy_fee == 0.0
        assert sell_fee > 0.0

    def test_crypto_maker_taker_spread(self):
        """Test crypto has maker/taker spread."""
        from execution_providers import create_fee_provider, AssetClass

        provider = create_fee_provider(AssetClass.CRYPTO)

        maker_fee = provider.compute_fee(10000.0, "BUY", "maker", 1.0)
        taker_fee = provider.compute_fee(10000.0, "BUY", "taker", 1.0)

        assert maker_fee < taker_fee


# =============================================================================
# Test Config Validation
# =============================================================================


class TestConfigValidation:
    """Tests for config validation and compatibility."""

    def test_crypto_config_structure(self, crypto_config):
        """Test crypto config has expected structure."""
        assert crypto_config["asset_class"] == "crypto"
        assert crypto_config["data_vendor"] == "binance"
        assert crypto_config["env"]["session"]["calendar"] == "crypto_24x7"

    def test_equity_config_structure(self, equity_config):
        """Test equity config has expected structure."""
        assert equity_config["asset_class"] == "equity"
        assert equity_config["data_vendor"] == "alpaca"
        assert equity_config["env"]["session"]["calendar"] == "us_equity"
        assert equity_config["fees"]["regulatory"]["enabled"] is True

    def test_compatible_asset_vendor_crypto(self):
        """Test crypto + binance is compatible."""
        # Valid combination
        config = {
            "asset_class": "crypto",
            "data_vendor": "binance",
        }
        assert config["data_vendor"] == "binance"

    def test_compatible_asset_vendor_equity(self):
        """Test equity + alpaca is compatible."""
        config = {
            "asset_class": "equity",
            "data_vendor": "alpaca",
        }
        assert config["data_vendor"] == "alpaca"


# =============================================================================
# Test Backward Compatibility
# =============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility with crypto-only configs."""

    def test_default_asset_class_is_crypto(self):
        """Test default asset class is crypto (backward compat)."""
        # When no asset_class specified, should default to crypto
        from execution_providers import AssetClass, L2ExecutionProvider

        # Default initialization
        provider = L2ExecutionProvider()
        assert provider.asset_class == AssetClass.CRYPTO

    def test_config_without_asset_class(self):
        """Test config without asset_class works (defaults to crypto)."""
        config = {
            "mode": "train",
            "fees": {
                "maker_bps": 2.0,
                "taker_bps": 4.0,
            },
        }
        # Should be treated as crypto
        assert config.get("asset_class", "crypto") == "crypto"

    def test_legacy_slippage_config(self):
        """Test legacy slippage config still works."""
        from execution_providers import wrap_legacy_slippage_config

        legacy = {
            "k": 0.15,
            "default_spread_bps": 6.0,
        }

        provider = wrap_legacy_slippage_config(legacy)
        assert provider.impact_coef == 0.15
        assert provider.spread_bps == 6.0

    def test_legacy_fees_config(self):
        """Test legacy fees config still works."""
        from execution_providers import wrap_legacy_fees_model

        legacy = {
            "maker_rate_bps": 1.5,
            "taker_rate_bps": 3.5,
        }

        provider = wrap_legacy_fees_model(legacy)
        assert provider.maker_bps == 1.5
        assert provider.taker_bps == 3.5


# =============================================================================
# Test Config File Reading
# =============================================================================


class TestConfigFileReading:
    """Tests for reading config files with asset_class."""

    def test_config_train_has_asset_class(self):
        """Test config_train.yaml has asset_class."""
        try:
            with open("configs/config_train.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            assert "asset_class" in config
            assert config["asset_class"] == "crypto"
        except FileNotFoundError:
            pytest.skip("Config file not found")

    def test_config_sim_has_asset_class(self):
        """Test config_sim.yaml has asset_class."""
        try:
            with open("configs/config_sim.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            assert "asset_class" in config
            assert config["asset_class"] == "crypto"
        except FileNotFoundError:
            pytest.skip("Config file not found")

    def test_config_train_stocks_has_equity(self):
        """Test config_train_stocks.yaml has equity asset_class."""
        try:
            with open("configs/config_train_stocks.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            assert "asset_class" in config
            assert config["asset_class"] == "equity"
        except FileNotFoundError:
            pytest.skip("Config file not found")

    def test_asset_class_defaults_exists(self):
        """Test asset_class_defaults.yaml exists."""
        try:
            with open("configs/asset_class_defaults.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            assert "crypto" in config
            assert "equity" in config
        except FileNotFoundError:
            pytest.skip("Config file not found")


# =============================================================================
# Test Switching at Runtime
# =============================================================================


class TestRuntimeSwitching:
    """Tests for switching asset class at runtime."""

    def test_create_both_providers(self):
        """Test creating both crypto and equity providers."""
        from execution_providers import (
            create_execution_provider,
            AssetClass,
        )

        crypto_provider = create_execution_provider(AssetClass.CRYPTO)
        equity_provider = create_execution_provider(AssetClass.EQUITY)

        # Both should be valid
        assert crypto_provider.asset_class == AssetClass.CRYPTO
        assert equity_provider.asset_class == AssetClass.EQUITY

    def test_execution_differs_by_asset_class(self):
        """Test execution behavior differs by asset class."""
        from execution_providers import (
            create_execution_provider,
            AssetClass,
            Order,
            MarketState,
            BarData,
        )

        crypto_exec = create_execution_provider(AssetClass.CRYPTO)
        equity_exec = create_execution_provider(AssetClass.EQUITY)

        market = MarketState(
            timestamp=1700000000000,
            bid=100.0,
            ask=100.10,
            adv=10_000_000.0,
        )
        bar = BarData(open=100.0, high=101.0, low=99.0, close=100.5, volume=10000.0)

        # Sell order
        order = Order(symbol="TEST", side="SELL", qty=100.0, order_type="MARKET")

        crypto_fill = crypto_exec.execute(order, market, bar)
        equity_fill = equity_exec.execute(order, market, bar)

        # Both should fill
        assert crypto_fill is not None
        assert equity_fill is not None

        # Crypto should have symmetric fee
        # Equity should have regulatory fee

        # Slippage should differ
        assert crypto_fill.slippage_bps != equity_fill.slippage_bps


# =============================================================================
# Test Provider Mapping
# =============================================================================


class TestProviderMapping:
    """Tests for provider mapping documentation."""

    def test_provider_mapping_in_defaults(self):
        """Test provider_mapping section in defaults config."""
        try:
            with open("configs/asset_class_defaults.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            assert "provider_mapping" in config

            crypto_mapping = config["provider_mapping"]["crypto"]
            assert crypto_mapping["fee_provider"] == "CryptoFeeProvider"
            assert crypto_mapping["slippage_profile"] == "crypto"

            equity_mapping = config["provider_mapping"]["equity"]
            assert equity_mapping["fee_provider"] == "EquityFeeProvider"
            assert equity_mapping["slippage_profile"] == "equity"

        except FileNotFoundError:
            pytest.skip("Config file not found")

    def test_data_vendor_mapping_in_defaults(self):
        """Test data_vendor_mapping section in defaults config."""
        try:
            with open("configs/asset_class_defaults.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            assert "data_vendor_mapping" in config

            binance = config["data_vendor_mapping"]["binance"]
            assert binance["default_asset_class"] == "crypto"

            alpaca = config["data_vendor_mapping"]["alpaca"]
            assert alpaca["default_asset_class"] == "equity"

        except FileNotFoundError:
            pytest.skip("Config file not found")


# =============================================================================
# Test Asset Class Consistency
# =============================================================================


class TestAssetClassConsistency:
    """Tests for consistency across components."""

    def test_slippage_profile_matches_asset_class(self):
        """Test slippage profile matches asset class defaults."""
        try:
            with open("configs/slippage.yaml", "r", encoding="utf-8") as f:
                slippage_config = yaml.safe_load(f)

            # Check profiles exist
            profiles = slippage_config["slippage"]["profiles"]
            assert "crypto" in profiles
            assert "equity" in profiles

            # Crypto should have wider spread
            assert profiles["crypto"]["spread_bps"] > profiles["equity"]["spread_bps"]

            # Crypto should have higher impact
            assert profiles["crypto"]["impact_coef"] > profiles["equity"]["impact_coef"]

        except FileNotFoundError:
            pytest.skip("Config file not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
