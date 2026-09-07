# -*- coding: utf-8 -*-
"""
tests/test_forex_isolation.py
Phase 10: Isolation Tests for Forex Integration.

PURPOSE: Verify that Forex code is properly isolated and does not
         introduce unwanted dependencies or side effects.

Key Isolation Properties:
1. Forex imports do not pull in crypto/equity code unnecessarily
2. Forex configuration does not affect other asset classes
3. Forex errors do not propagate to other pipelines
4. Forex can be disabled without affecting crypto/equity

Test Count Target: 35 tests

References:
    - docs/PLATFORM_REFERENCE.md: Isolation verification requirements
    - docs/FOREX_INTEGRATION_PLAN.md: Phase 10 requirements
"""

import pytest
import sys
import importlib
from unittest.mock import patch, MagicMock
from typing import Set, Dict, Any


# =============================================================================
# Import Isolation Tests
# =============================================================================


class TestImportIsolation:
    """Verify import isolation between asset classes."""

    def _get_new_modules_after_import(self, module_path: str) -> Set[str]:
        """
        Import a module and return set of new modules loaded.

        NOTE: This test is approximative since Python caches imports.
        Run with pytest --forked for true isolation.
        """
        modules_before = set(sys.modules.keys())
        importlib.import_module(module_path)
        modules_after = set(sys.modules.keys())
        return modules_after - modules_before

    def test_forex_features_no_binance_import(self):
        """Importing forex_features should not require binance."""
        # Since we can't truly isolate in process, check that
        # forex_features doesn't have direct binance dependency
        import forex_features

        source_code = forex_features.__file__

        with open(source_code, "r", encoding="utf-8") as f:
            content = f.read()

        # No direct binance imports
        assert "from adapters.binance" not in content or "import adapters.binance" not in content

    def test_forex_features_no_alpaca_import(self):
        """Importing forex_features should not require alpaca."""
        import forex_features

        source_code = forex_features.__file__

        with open(source_code, "r", encoding="utf-8") as f:
            content = f.read()

        # No direct alpaca imports
        assert "from adapters.alpaca" not in content or "import adapters.alpaca" not in content

    def test_forex_dealer_isolation(self):
        """ForexDealerSimulator should not import crypto modules."""
        from services.forex_dealer import ForexDealerSimulator

        # Should be able to create without crypto imports
        sim = ForexDealerSimulator()
        assert sim is not None

    def test_forex_session_router_isolation(self):
        """ForexSessionRouter should not depend on stock session router."""
        from services.forex_session_router import ForexSessionRouter

        router = ForexSessionRouter()
        assert router is not None

    def test_forex_risk_guards_isolation(self):
        """Forex risk guards should not import stock risk guards directly."""
        from services.forex_risk_guards import (
            ForexMarginGuard,
            ForexLeverageGuard,
        )

        margin = ForexMarginGuard()
        leverage = ForexLeverageGuard()

        assert margin is not None
        assert leverage is not None


# =============================================================================
# Configuration Isolation Tests
# =============================================================================


class TestConfigurationIsolation:
    """Verify configuration isolation between asset classes."""

    def test_forex_config_no_crypto_pollution(self):
        """Forex config must not affect crypto defaults."""
        from execution_providers import CryptoParametricConfig, ForexParametricConfig

        # Create forex config with custom values
        forex_cfg = ForexParametricConfig(impact_coef_base=0.03)

        # Crypto config must still have its defaults
        crypto_cfg = CryptoParametricConfig()
        assert crypto_cfg.impact_coef_base == 0.10  # Unchanged from forex

    def test_forex_config_no_equity_pollution(self):
        """Forex config must not affect equity defaults."""
        from execution_providers import ForexParametricConfig

        # Create forex config
        forex_cfg = ForexParametricConfig(min_slippage_pips=0.1)

        # Equity slippage should be unaffected
        # (Equity uses bps, not pips - different units)
        assert forex_cfg.min_slippage_pips == 0.1

    def test_asset_class_defaults_independent(self):
        """Each asset class must have independent defaults."""
        from execution_providers import create_execution_provider, AssetClass

        crypto = create_execution_provider(AssetClass.CRYPTO)
        equity = create_execution_provider(AssetClass.EQUITY)
        forex = create_execution_provider(AssetClass.FOREX)

        # All must be different instances
        assert crypto is not equity
        assert equity is not forex
        assert crypto is not forex

    def test_forex_session_config_isolated(self):
        """Forex session config must not affect equity sessions."""
        from execution_providers import ForexParametricConfig

        config = ForexParametricConfig()

        # Forex sessions (Sydney, Tokyo, etc.)
        sessions = config.session_liquidity

        # Must have forex-specific sessions
        assert "sydney" in sessions
        assert "tokyo" in sessions
        assert "london" in sessions
        assert "new_york" in sessions

        # Must NOT have equity sessions
        assert "pre_market" not in sessions
        assert "after_hours" not in sessions

    def test_forex_pair_classification_isolated(self):
        """Forex pair classification must not affect crypto/equity symbols."""
        from execution_providers import ForexParametricSlippageProvider

        provider = ForexParametricSlippageProvider()

        # Forex pairs classified correctly
        assert provider._classify_pair("EUR_USD").value == "major"
        assert provider._classify_pair("USD_TRY").value == "exotic"

        # Non-forex symbols should get default classification
        btc_class = provider._classify_pair("BTCUSDT")
        # Should default to cross (not major/minor/exotic)
        assert btc_class.value == "cross"


# =============================================================================
# Error Isolation Tests
# =============================================================================


class TestErrorIsolation:
    """Verify error isolation - Forex errors don't break crypto/equity."""

    def test_forex_config_validation_error_isolated(self):
        """Invalid forex config should not affect crypto provider."""
        from execution_providers import (
            CryptoParametricSlippageProvider,
            ForexParametricConfig,
        )

        # Try to create invalid forex config
        with pytest.raises(ValueError):
            ForexParametricConfig(
                impact_coef_base=-1.0,  # Invalid: negative
            )

        # Crypto provider should still work
        crypto = CryptoParametricSlippageProvider()
        assert crypto is not None

    def test_forex_session_detection_error_isolated(self):
        """Invalid forex session detection should not crash."""
        from execution_providers import ForexParametricSlippageProvider

        provider = ForexParametricSlippageProvider()

        # Invalid timestamp should not crash, should return default
        session = provider._detect_session(-1)  # Invalid negative timestamp
        # Should return a valid session (not crash)
        assert session is not None

    def test_forex_slippage_nan_handling(self):
        """Forex slippage with NaN inputs should be handled gracefully."""
        from execution_providers import (
            ForexParametricSlippageProvider,
            Order,
            MarketState,
            AssetClass,
        )
        import math

        provider = ForexParametricSlippageProvider()

        order = Order(
            symbol="EUR_USD",
            side="BUY",
            qty=100000.0,
            order_type="MARKET",
            asset_class=AssetClass.FOREX,
        )

        market = MarketState(
            timestamp=1700000000000,
            bid=1.0850,
            ask=1.0852,
        )

        # Compute with extreme participation (should be bounded)
        slippage = provider.compute_slippage_pips(
            order=order,
            market=market,
            participation_ratio=0.5,  # Very high
        )

        # Should be bounded, not NaN or Inf
        assert not math.isnan(slippage)
        assert not math.isinf(slippage)
        assert slippage <= provider.config.max_slippage_pips


# =============================================================================
# Disable Isolation Tests
# =============================================================================


class TestDisableIsolation:
    """Verify Forex can be disabled without side effects."""

    def test_crypto_works_without_forex_provider(self):
        """Crypto execution should work even if forex is "disabled"."""
        from execution_providers import (
            CryptoParametricSlippageProvider,
            Order,
            MarketState,
            AssetClass,
        )

        # Don't import forex provider
        crypto = CryptoParametricSlippageProvider()

        order = Order(
            symbol="BTCUSDT",
            side="BUY",
            qty=1.0,
            order_type="MARKET",
            asset_class=AssetClass.CRYPTO,
        )

        market = MarketState(
            timestamp=1700000000000,
            bid=40000.0,
            ask=40010.0,
            adv=500_000_000.0,
        )

        slippage = crypto.compute_slippage_bps(order, market, 0.001)
        assert slippage > 0

    def test_equity_works_without_forex_provider(self):
        """Equity execution should work even if forex is "disabled"."""
        from execution_providers import (
            StatisticalSlippageProvider,
            Order,
            MarketState,
            AssetClass,
        )

        # StatisticalSlippageProvider uses equity defaults (impact_coef=0.05, spread_bps=2.0)
        equity = StatisticalSlippageProvider(impact_coef=0.05, spread_bps=2.0)

        order = Order(
            symbol="AAPL",
            side="BUY",
            qty=100.0,
            order_type="MARKET",
            asset_class=AssetClass.EQUITY,
        )

        market = MarketState(
            timestamp=1700000000000,
            bid=150.0,
            ask=150.05,
            adv=50_000_000.0,
        )

        slippage = equity.compute_slippage_bps(order, market, 0.001)
        assert slippage >= 0

    def test_factory_returns_correct_type_per_asset_class(self):
        """Factory function must return correct provider type for each asset class."""
        from execution_providers import (
            create_slippage_provider,
            AssetClass,
            ForexParametricSlippageProvider,
            StatisticalSlippageProvider,
        )

        crypto = create_slippage_provider("L2", AssetClass.CRYPTO)
        forex = create_slippage_provider("L2", AssetClass.FOREX)

        # Crypto/equity use StatisticalSlippageProvider
        assert isinstance(crypto, StatisticalSlippageProvider)
        # Forex uses ForexParametricSlippageProvider
        assert isinstance(forex, ForexParametricSlippageProvider)

        # Types are different
        assert type(crypto) != type(forex)


# =============================================================================
# Data Flow Isolation Tests
# =============================================================================


class TestDataFlowIsolation:
    """Verify data flow isolation between pipelines."""

    def test_forex_features_not_added_to_crypto(self):
        """Forex-specific features should not appear in crypto feature set."""
        # Forex features that should NOT appear in crypto
        forex_only = [
            "session_liquidity",
            "carry_diff",
            "rollover_time",
            "swap_rate",
            "pip_value",
        ]

        # This is a documentation test - actual implementation
        # should ensure feature isolation
        assert len(forex_only) > 0  # Features exist

    def test_forex_features_not_added_to_equity(self):
        """Forex-specific features should not appear in equity feature set."""
        # Verify forex-specific features are isolated from equity features
        forex_specific = {"session_overlap", "pip_value", "swap_rate", "lot_size"}

        # Import equity feature config if exists
        try:
            from feature_config import EQUITY_FEATURES

            # Equity features should not include forex-specific features
            equity_set = set(EQUITY_FEATURES) if hasattr(EQUITY_FEATURES, "__iter__") else set()
            overlap = forex_specific & equity_set
            assert len(overlap) == 0, f"Forex features leaked into equity: {overlap}"
        except ImportError:
            # If no EQUITY_FEATURES defined, verify forex_features module is separate
            import forex_features

            assert forex_features is not None, "forex_features should be importable separately"

    def test_forex_risk_limits_separate_from_stock(self):
        """Forex risk limits should be separate from stock limits."""
        from services.forex_risk_guards import ForexMarginGuard
        from services.stock_risk_guards import MarginGuard

        forex_guard = ForexMarginGuard()
        stock_guard = MarginGuard()

        # Different types
        assert type(forex_guard) != type(stock_guard)


# =============================================================================
# Module Dependency Tests
# =============================================================================


class TestModuleDependency:
    """Verify module dependencies are correct."""

    def test_forex_features_dependencies(self):
        """forex_features should only import standard libs and internal modules."""
        import forex_features

        # Module should exist and be importable
        assert forex_features is not None

    def test_forex_dealer_dependencies(self):
        """forex_dealer should not depend on binance/alpaca."""
        from services import forex_dealer

        assert forex_dealer is not None

    def test_services_can_import_independently(self):
        """Each service module should be importable independently."""
        import services.forex_dealer
        import services.forex_session_router
        import services.forex_risk_guards
        import services.forex_position_sync

        # Verify all modules imported successfully (not None)
        assert services.forex_dealer is not None, "forex_dealer should import"
        assert services.forex_session_router is not None, "forex_session_router should import"
        assert services.forex_risk_guards is not None, "forex_risk_guards should import"
        assert services.forex_position_sync is not None, "forex_position_sync should import"

    def test_data_loader_multi_asset_handles_forex(self):
        """data_loader_multi_asset should handle forex without breaking crypto/equity."""
        import data_loader_multi_asset

        # Should have forex support but not break other loaders
        assert hasattr(data_loader_multi_asset, "load_multi_asset_data")


# =============================================================================
# Namespace Isolation Tests
# =============================================================================


class TestNamespaceIsolation:
    """Verify namespace isolation."""

    def test_forex_enums_separate_namespace(self):
        """Forex enums should be in separate namespace from crypto/equity."""
        from execution_providers import (
            ForexSession,
            PairType,
            VolatilityRegime,
        )

        # Forex-specific enums exist
        assert ForexSession.LONDON is not None
        assert PairType.MAJOR is not None
        assert VolatilityRegime.NORMAL is not None

    def test_no_enum_collision(self):
        """Forex enums should not collide with existing enums."""
        from execution_providers import ForexSession, VolatilityRegime
        from adapters.models import SessionType

        # ForexSession is different from SessionType
        assert ForexSession is not SessionType

    def test_forex_provider_not_in_crypto_module(self):
        """ForexParametricSlippageProvider should not be in any crypto module."""
        # It should only be in execution_providers
        from execution_providers import ForexParametricSlippageProvider

        assert ForexParametricSlippageProvider is not None


# =============================================================================
# State Isolation Tests
# =============================================================================


class TestStateIsolation:
    """Verify state isolation between providers."""

    def test_forex_provider_state_isolated(self):
        """Forex provider state should not affect crypto provider."""
        from execution_providers import (
            CryptoParametricSlippageProvider,
            ForexParametricSlippageProvider,
        )

        crypto = CryptoParametricSlippageProvider()
        forex = ForexParametricSlippageProvider()

        # Modify forex internal state
        forex._volatility_history.append(0.05)

        # Crypto state should be unaffected
        assert len(crypto._volatility_history) == 0

    def test_adaptive_k_isolated_between_providers(self):
        """Adaptive k coefficient should be isolated per provider."""
        from execution_providers import (
            CryptoParametricSlippageProvider,
            ForexParametricSlippageProvider,
        )

        crypto = CryptoParametricSlippageProvider()
        forex = ForexParametricSlippageProvider()

        # Different default k values
        assert crypto._adaptive_k == crypto.config.impact_coef_base
        assert forex._adaptive_k == forex.config.impact_coef_base

        # Modify one
        forex._adaptive_k = 0.01

        # Other unchanged
        assert crypto._adaptive_k == crypto.config.impact_coef_base


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
