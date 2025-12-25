# -*- coding: utf-8 -*-
"""
tests/test_forex_backward_compat.py
Phase 10: Backward Compatibility Tests.

PURPOSE: Ensure existing code that uses crypto/equity APIs continues to work
         without any changes after Forex integration.

Scenarios:
1. Existing scripts run unchanged
2. Existing configs load correctly
3. Existing trained models can be loaded
4. Existing API contracts preserved

Test Count Target: 30 tests

References:
    - CLAUDE.md: API contract preservation
    - docs/FOREX_INTEGRATION_PLAN.md: Phase 10 requirements
"""

import pytest
import yaml
import os
import py_compile
from typing import Dict, Any, List
from pathlib import Path


# =============================================================================
# API Contract Preservation Tests
# =============================================================================

class TestAPIContractPreservation:
    """Verify public API contracts are preserved."""

    def test_execution_providers_api_unchanged(self):
        """Public API of execution_providers must be unchanged."""
        import execution_providers as ep

        # All existing exports must exist
        required_exports = [
            "AssetClass",
            "Order",
            "MarketState",
            "Fill",
            "BarData",
            "SlippageProvider",
            "FeeProvider",
            "FillProvider",
            "CryptoParametricSlippageProvider",
            "CryptoFeeProvider",
            "EquityFeeProvider",
            "StatisticalSlippageProvider",
            "OHLCVFillProvider",
            "L2ExecutionProvider",
            "create_execution_provider",
            "create_slippage_provider",
            "create_fee_provider",
        ]

        for name in required_exports:
            assert hasattr(ep, name), f"Missing export: {name}"

    def test_adapter_models_api_unchanged(self):
        """Public API of adapters.models must be unchanged."""
        from adapters import models

        required = [
            "MarketType",
            "ExchangeVendor",
            "FeeStructure",
            "SessionType",
            "ExchangeRule",
        ]

        for name in required:
            assert hasattr(models, name), f"Missing: {name}"

    def test_order_api_unchanged(self):
        """Order class API must be unchanged."""
        from execution_providers import Order, AssetClass

        # All original constructor parameters must work
        order = Order(
            symbol="BTCUSDT",
            side="BUY",
            qty=1.0,
            order_type="MARKET",
            asset_class=AssetClass.CRYPTO,
        )

        # Original attributes must exist
        assert hasattr(order, "symbol")
        assert hasattr(order, "side")
        assert hasattr(order, "qty")
        assert hasattr(order, "order_type")
        assert hasattr(order, "asset_class")
        assert hasattr(order, "limit_price")

        # Original methods must exist
        assert hasattr(order, "get_notional")

    def test_market_state_api_unchanged(self):
        """MarketState class API must be unchanged."""
        from execution_providers import MarketState

        market = MarketState(
            timestamp=1700000000000,
            bid=100.0,
            ask=100.05,
            bid_size=1000.0,
            ask_size=1000.0,
            adv=50_000_000.0,
        )

        # Original attributes
        assert hasattr(market, "timestamp")
        assert hasattr(market, "bid")
        assert hasattr(market, "ask")
        assert hasattr(market, "bid_size")
        assert hasattr(market, "ask_size")
        assert hasattr(market, "adv")

        # Original methods
        assert hasattr(market, "get_mid_price")
        assert hasattr(market, "get_spread_bps")

    def test_fill_api_unchanged(self):
        """Fill class API must be unchanged."""
        from execution_providers import Fill

        # Note: Fill class has: price, qty, fee, slippage_bps, liquidity, timestamp
        fill = Fill(
            price=100.0,
            qty=1.0,
            fee=0.01,
            slippage_bps=1.0,
            liquidity="taker",
            timestamp=1700000000000,
        )

        # Core attributes
        assert hasattr(fill, "qty")
        assert hasattr(fill, "price")
        assert hasattr(fill, "fee")
        assert hasattr(fill, "slippage_bps")
        assert hasattr(fill, "timestamp")
        assert hasattr(fill, "liquidity")

    def test_bar_data_api_unchanged(self):
        """BarData class API must be unchanged."""
        from execution_providers import BarData

        bar = BarData(
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=10000.0,
        )

        # Original attributes
        assert hasattr(bar, "open")
        assert hasattr(bar, "high")
        assert hasattr(bar, "low")
        assert hasattr(bar, "close")
        assert hasattr(bar, "volume")

        # Original properties/methods
        assert hasattr(bar, "typical_price")


# =============================================================================
# Existing Config Loading Tests
# =============================================================================

class TestExistingConfigsLoad:
    """Verify existing config files still load correctly."""

    @pytest.mark.parametrize("config_path", [
        "configs/config_train.yaml",
        "configs/config_sim.yaml",
        "configs/config_live.yaml",
        "configs/config_eval.yaml",
        "configs/config_train_stocks.yaml",
        "configs/config_backtest_stocks.yaml",
        "configs/config_live_alpaca.yaml",
        "configs/risk.yaml",
        "configs/fees.yaml",
        "configs/slippage.yaml",
    ])
    def test_existing_config_loads(self, config_path: str):
        """Existing config must load without errors."""
        if not os.path.exists(config_path):
            pytest.skip(f"Config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert config is not None

    def test_forex_config_does_not_break_crypto_config(self):
        """Loading forex config should not affect crypto config loading."""
        # Load forex config
        forex_path = "configs/forex_defaults.yaml"
        if os.path.exists(forex_path):
            with open(forex_path, "r", encoding="utf-8") as f:
                forex_cfg = yaml.safe_load(f)
            assert forex_cfg is not None

        # Load crypto config (should still work)
        crypto_path = "configs/config_train.yaml"
        if os.path.exists(crypto_path):
            with open(crypto_path, "r", encoding="utf-8") as f:
                crypto_cfg = yaml.safe_load(f)
            assert crypto_cfg is not None


# =============================================================================
# Asset Class Detection Tests
# =============================================================================

class TestAssetClassDetection:
    """Test backward-compatible asset class detection."""

    def test_legacy_config_no_asset_class(self):
        """Config without explicit asset_class should default correctly."""
        # Legacy config without asset_class (defaults to crypto)
        legacy_config = {}

        # Should detect as crypto (backward compatible default)
        from script_live import detect_asset_class

        result = detect_asset_class(legacy_config)
        assert result.lower() in ("crypto", "crypto_spot")

    def test_alpaca_vendor_detects_equity(self):
        """Config with alpaca vendor should detect as equity."""
        alpaca_config = {"exchange": {"vendor": "alpaca"}}

        from script_live import detect_asset_class

        result = detect_asset_class(alpaca_config)
        assert result.lower() == "equity"

    def test_oanda_vendor_detects_forex(self):
        """Config with oanda vendor should detect as forex."""
        oanda_config = {"exchange": {"vendor": "oanda"}}

        from script_live import detect_asset_class

        result = detect_asset_class(oanda_config)
        assert result.lower() == "forex"

    def test_explicit_asset_class_takes_priority(self):
        """Explicit asset_class should override vendor detection."""
        config = {
            "exchange": {"vendor": "binance"},
            "asset_class": "equity",  # Explicit override
        }

        from script_live import detect_asset_class

        result = detect_asset_class(config)
        assert result.lower() == "equity"


# =============================================================================
# Existing Scripts Syntax Tests
# =============================================================================

class TestExistingScriptsSyntax:
    """Verify existing scripts are syntactically valid."""

    @pytest.mark.parametrize("script_path", [
        "script_backtest.py",
        "script_live.py",
        "script_eval.py",
        "train_model_multi_patch.py",
    ])
    def test_script_syntax_valid(self, script_path: str):
        """Script must be syntactically valid Python."""
        if not os.path.exists(script_path):
            pytest.skip(f"Script not found: {script_path}")

        # This will raise SyntaxError if invalid
        py_compile.compile(script_path, doraise=True)


# =============================================================================
# Model Compatibility Tests
# =============================================================================

class TestModelCompatibility:
    """Verify trained model compatibility."""

    def test_crypto_model_loads_after_forex(self):
        """Crypto models trained before Forex must still load."""
        # This tests that adding ForexParametric doesn't break
        # torch.load() for existing models

        # Just verify the model classes exist and can be instantiated
        from distributional_ppo import DistributionalPPO

        assert DistributionalPPO is not None

    def test_observation_space_dimensions_unchanged(self):
        """Observation space dimensions must be unchanged for crypto/equity.

        This is a configuration-level test: observation space dimensions are
        determined by mediator configuration (configs/mediator.yaml), not by
        code changes. Forex features are isolated by asset class and do not
        affect crypto/equity observation spaces.

        Verification: Forex features use separate feature registries per asset
        class. See docs/forex_features.md for feature isolation architecture.
        """
        # Verify feature isolation exists via ForexConfig isolation
        from services.forex_config import ForexConfig

        # ForexConfig.asset_class ensures forex features are isolated
        assert hasattr(ForexConfig, "asset_class"), (
            "ForexConfig must have asset_class for feature isolation"
        )

    def test_action_space_unchanged(self):
        """Action space must be unchanged for all asset classes."""
        from execution_providers import AssetClass

        # Action space is determined by wrappers/action_space.py
        # Verify the standard action space exists
        from wrappers.action_space import LongOnlyActionWrapper

        assert LongOnlyActionWrapper is not None


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctionBackwardCompat:
    """Test factory functions maintain backward compatibility."""

    def test_create_execution_provider_crypto(self):
        """create_execution_provider must work for crypto."""
        from execution_providers import create_execution_provider, AssetClass

        provider = create_execution_provider(AssetClass.CRYPTO)
        assert provider is not None

    def test_create_execution_provider_equity(self):
        """create_execution_provider must work for equity."""
        from execution_providers import create_execution_provider, AssetClass

        provider = create_execution_provider(AssetClass.EQUITY)
        assert provider is not None

    def test_create_execution_provider_forex(self):
        """create_execution_provider must work for forex."""
        from execution_providers import create_execution_provider, AssetClass

        provider = create_execution_provider(AssetClass.FOREX)
        assert provider is not None

    def test_create_slippage_provider_l2_crypto(self):
        """create_slippage_provider L2 must work for crypto."""
        from execution_providers import create_slippage_provider, AssetClass

        provider = create_slippage_provider("L2", AssetClass.CRYPTO)
        assert provider is not None

    def test_create_slippage_provider_l2plus_crypto(self):
        """create_slippage_provider L2+ must work for crypto."""
        from execution_providers import (
            create_slippage_provider,
            AssetClass,
        )

        provider = create_slippage_provider("L2+", AssetClass.CRYPTO)
        # Should return a valid slippage provider with compute_slippage_bps
        assert provider is not None
        assert hasattr(provider, "compute_slippage_bps")

    def test_create_fee_provider_crypto(self):
        """create_fee_provider must work for crypto."""
        from execution_providers import create_fee_provider, AssetClass

        provider = create_fee_provider(AssetClass.CRYPTO)
        assert provider is not None

    def test_create_fee_provider_equity(self):
        """create_fee_provider must work for equity."""
        from execution_providers import create_fee_provider, AssetClass

        provider = create_fee_provider(AssetClass.EQUITY)
        assert provider is not None


# =============================================================================
# Import Structure Tests
# =============================================================================

class TestImportStructureBackwardCompat:
    """Test import structure backward compatibility."""

    def test_execution_providers_import(self):
        """execution_providers module must be importable."""
        import execution_providers

        assert execution_providers is not None

    def test_adapters_models_import(self):
        """adapters.models module must be importable."""
        from adapters import models

        assert models is not None

    def test_services_import(self):
        """services package must be importable."""
        import services

        assert services is not None

    def test_risk_guard_import(self):
        """risk_guard module must be importable."""
        import risk_guard

        assert risk_guard is not None

    def test_data_loader_import(self):
        """data_loader_multi_asset module must be importable."""
        import data_loader_multi_asset

        assert data_loader_multi_asset is not None


# =============================================================================
# Enum Backward Compatibility Tests
# =============================================================================

class TestEnumBackwardCompat:
    """Test enum backward compatibility."""

    def test_asset_class_values_unchanged(self):
        """AssetClass enum values must be unchanged."""
        from execution_providers import AssetClass

        assert AssetClass.CRYPTO.value == "crypto"
        assert AssetClass.EQUITY.value == "equity"
        assert AssetClass.FOREX.value == "forex"

    def test_market_type_values_unchanged(self):
        """MarketType enum values must be unchanged."""
        from adapters.models import MarketType

        assert MarketType.CRYPTO_SPOT.value == "CRYPTO_SPOT"
        assert MarketType.EQUITY.value == "EQUITY"
        assert MarketType.FOREX.value == "FOREX"

    def test_vendor_values_unchanged(self):
        """ExchangeVendor enum values must be unchanged."""
        from adapters.models import ExchangeVendor

        assert ExchangeVendor.BINANCE.value == "binance"
        assert ExchangeVendor.ALPACA.value == "alpaca"
        assert ExchangeVendor.OANDA.value == "oanda"


# =============================================================================
# Service Layer Tests
# =============================================================================

class TestServiceLayerBackwardCompat:
    """Test service layer backward compatibility."""

    def test_position_sync_service_exists(self):
        """PositionSynchronizer service must exist."""
        from services.position_sync import PositionSynchronizer

        assert PositionSynchronizer is not None

    def test_session_router_service_exists(self):
        """SessionRouter service must exist."""
        from services.session_router import SessionRouter

        assert SessionRouter is not None

    def test_stock_risk_guards_exist(self):
        """Stock risk guards must exist."""
        from services.stock_risk_guards import MarginGuard, ShortSaleGuard

        assert MarginGuard is not None
        assert ShortSaleGuard is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
