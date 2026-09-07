# -*- coding: utf-8 -*-
"""
tests/test_execution_provider_integration.py

Phase 4.3 & 4.4: Comprehensive tests for fee/slippage provider integration.

Test coverage:
- StatisticalSlippageProvider.from_config()
- StatisticalSlippageProvider.from_profile()
- CryptoFeeProvider.from_config()
- EquityFeeProvider.from_config()
- load_slippage_profile()
- create_providers_from_asset_class()
- ExecutionSimulator provider integration
- Backward compatibility (legacy mode)
- Asset class specific behavior (crypto vs equity)
- YAML profile loading
"""

import math
import pytest
from typing import Dict, Any, Optional

# Import execution_providers module
from execution_providers import (
    # Enums
    AssetClass,
    # Data classes
    MarketState,
    Order,
    # L2 Implementations
    StatisticalSlippageProvider,
    CryptoFeeProvider,
    EquityFeeProvider,
    L2ExecutionProvider,
    # Factory functions
    create_slippage_provider,
    create_fee_provider,
    create_execution_provider,
    load_slippage_profile,
    create_providers_from_asset_class,
    # Backward compatibility
    wrap_legacy_slippage_config,
    wrap_legacy_fees_model,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def equity_config() -> Dict[str, Any]:
    """Equity slippage configuration."""
    return {
        "spread_bps": 2.0,
        "impact_coef": 0.05,
        "volatility_scale": 1.0,
        "min_slippage_bps": 0.5,
        "max_slippage_bps": 200.0,
    }


@pytest.fixture
def crypto_config() -> Dict[str, Any]:
    """Crypto slippage configuration."""
    return {
        "spread_bps": 5.0,
        "impact_coef": 0.10,
        "volatility_scale": 1.0,
        "min_slippage_bps": 1.0,
        "max_slippage_bps": 500.0,
    }


@pytest.fixture
def legacy_slippage_config() -> Dict[str, Any]:
    """Legacy slippage config format."""
    return {
        "k": 0.8,
        "default_spread_bps": 3.0,
    }


@pytest.fixture
def crypto_fee_config() -> Dict[str, Any]:
    """Crypto fee configuration."""
    return {
        "maker_bps": 2.0,
        "taker_bps": 4.0,
        "discount_rate": 0.75,
        "use_discount": False,
    }


@pytest.fixture
def equity_fee_config() -> Dict[str, Any]:
    """Equity fee configuration."""
    return {
        "sec_fee_rate": 0.0000278,
        "taf_fee_rate": 0.000166,
        "taf_max_fee": 8.30,
        "include_regulatory": True,
    }


# =============================================================================
# Test StatisticalSlippageProvider.from_config()
# =============================================================================


class TestStatisticalSlippageProviderFromConfig:
    """Tests for StatisticalSlippageProvider.from_config()."""

    def test_from_config_dict(self, equity_config):
        """Test creating provider from dict config."""
        provider = StatisticalSlippageProvider.from_config(equity_config)

        assert provider.spread_bps == pytest.approx(2.0)
        assert provider.impact_coef == pytest.approx(0.05)
        assert provider.volatility_scale == pytest.approx(1.0)
        assert provider.min_slippage_bps == pytest.approx(0.5)
        assert provider.max_slippage_bps == pytest.approx(200.0)

    def test_from_config_none(self):
        """Test creating provider from None (defaults)."""
        provider = StatisticalSlippageProvider.from_config(None)

        assert provider.spread_bps == pytest.approx(5.0)  # Crypto default
        assert provider.impact_coef == pytest.approx(0.1)  # Crypto default

    def test_from_config_none_equity(self):
        """Test creating equity provider from None."""
        provider = StatisticalSlippageProvider.from_config(None, AssetClass.EQUITY)

        assert provider.spread_bps == pytest.approx(2.0)  # Equity default
        assert provider.impact_coef == pytest.approx(0.05)  # Equity default

    def test_from_config_legacy_keys(self, legacy_slippage_config):
        """Test creating provider from legacy config format."""
        provider = StatisticalSlippageProvider.from_config(legacy_slippage_config)

        assert provider.impact_coef == pytest.approx(0.8)  # From 'k'
        assert provider.spread_bps == pytest.approx(3.0)  # From 'default_spread_bps'

    def test_from_config_with_profiles(self):
        """Test creating provider from config with profiles section."""
        config = {
            "profiles": {
                "equity": {
                    "spread_bps": 1.5,
                    "impact_coef": 0.04,
                },
                "crypto": {
                    "spread_bps": 6.0,
                    "impact_coef": 0.12,
                },
            }
        }

        # Equity profile
        equity_provider = StatisticalSlippageProvider.from_config(config, AssetClass.EQUITY)
        assert equity_provider.spread_bps == pytest.approx(1.5)
        assert equity_provider.impact_coef == pytest.approx(0.04)

        # Crypto profile
        crypto_provider = StatisticalSlippageProvider.from_config(config, AssetClass.CRYPTO)
        assert crypto_provider.spread_bps == pytest.approx(6.0)
        assert crypto_provider.impact_coef == pytest.approx(0.12)


# =============================================================================
# Test StatisticalSlippageProvider.from_profile()
# =============================================================================


class TestStatisticalSlippageProviderFromProfile:
    """Tests for StatisticalSlippageProvider.from_profile()."""

    def test_equity_profile(self):
        """Test loading equity profile."""
        provider = StatisticalSlippageProvider.from_profile("equity")

        assert provider.spread_bps == pytest.approx(2.0)
        assert provider.impact_coef == pytest.approx(0.05)
        assert provider.min_slippage_bps == pytest.approx(0.5)
        assert provider.max_slippage_bps == pytest.approx(200.0)

    def test_crypto_profile(self):
        """Test loading crypto profile."""
        provider = StatisticalSlippageProvider.from_profile("crypto")

        assert provider.spread_bps == pytest.approx(5.0)
        assert provider.impact_coef == pytest.approx(0.10)
        assert provider.min_slippage_bps == pytest.approx(1.0)
        assert provider.max_slippage_bps == pytest.approx(500.0)

    def test_crypto_futures_profile(self):
        """Test loading crypto_futures profile."""
        provider = StatisticalSlippageProvider.from_profile("crypto_futures")

        assert provider.spread_bps == pytest.approx(4.0)
        assert provider.impact_coef == pytest.approx(0.09)
        assert provider.volatility_scale == pytest.approx(1.2)

    def test_default_profile(self):
        """Test loading default profile."""
        provider = StatisticalSlippageProvider.from_profile("default")

        assert provider.spread_bps == pytest.approx(5.0)
        assert provider.impact_coef == pytest.approx(0.10)

    def test_unknown_profile_fallback(self):
        """Test unknown profile falls back to default."""
        provider = StatisticalSlippageProvider.from_profile("nonexistent")

        assert provider.spread_bps == pytest.approx(5.0)  # Default
        assert provider.impact_coef == pytest.approx(0.10)  # Default

    def test_custom_profiles_override(self):
        """Test custom profiles override defaults."""
        custom_profiles = {
            "equity": {
                "spread_bps": 1.0,
                "impact_coef": 0.03,
            }
        }

        provider = StatisticalSlippageProvider.from_profile(
            "equity", profiles_config=custom_profiles
        )

        assert provider.spread_bps == pytest.approx(1.0)
        assert provider.impact_coef == pytest.approx(0.03)


# =============================================================================
# Test CryptoFeeProvider.from_config()
# =============================================================================


class TestCryptoFeeProviderFromConfig:
    """Tests for CryptoFeeProvider.from_config()."""

    def test_from_config_dict(self, crypto_fee_config):
        """Test creating provider from dict config."""
        provider = CryptoFeeProvider.from_config(crypto_fee_config)

        assert provider.maker_bps == pytest.approx(2.0)
        assert provider.taker_bps == pytest.approx(4.0)
        assert provider.discount_rate == pytest.approx(0.75)
        assert provider.use_discount is False

    def test_from_config_none(self):
        """Test creating provider from None (defaults)."""
        provider = CryptoFeeProvider.from_config(None)

        assert provider.maker_bps == pytest.approx(2.0)
        assert provider.taker_bps == pytest.approx(4.0)

    def test_from_config_legacy_keys(self):
        """Test creating provider from legacy config format."""
        config = {
            "maker_rate_bps": 1.5,
            "taker_rate_bps": 3.5,
        }
        provider = CryptoFeeProvider.from_config(config)

        assert provider.maker_bps == pytest.approx(1.5)
        assert provider.taker_bps == pytest.approx(3.5)

    def test_compute_fee_maker(self, crypto_fee_config):
        """Test fee computation for maker order."""
        provider = CryptoFeeProvider.from_config(crypto_fee_config)

        fee = provider.compute_fee(
            notional=10000.0,
            side="BUY",
            liquidity="maker",
            qty=1.0,
        )

        # 10000 * 2.0 / 10000 = $2.00
        assert fee == pytest.approx(2.0)

    def test_compute_fee_taker(self, crypto_fee_config):
        """Test fee computation for taker order."""
        provider = CryptoFeeProvider.from_config(crypto_fee_config)

        fee = provider.compute_fee(
            notional=10000.0,
            side="BUY",
            liquidity="taker",
            qty=1.0,
        )

        # 10000 * 4.0 / 10000 = $4.00
        assert fee == pytest.approx(4.0)


# =============================================================================
# Test EquityFeeProvider.from_config()
# =============================================================================


class TestEquityFeeProviderFromConfig:
    """Tests for EquityFeeProvider.from_config()."""

    def test_from_config_dict(self, equity_fee_config):
        """Test creating provider from dict config."""
        provider = EquityFeeProvider.from_config(equity_fee_config)

        assert provider.sec_fee_rate == pytest.approx(0.0000278)
        assert provider.taf_fee_rate == pytest.approx(0.000166)
        assert provider.taf_max_fee == pytest.approx(8.30)
        assert provider.include_regulatory is True

    def test_from_config_none(self):
        """Test creating provider from None (defaults)."""
        provider = EquityFeeProvider.from_config(None)

        assert provider.sec_fee_rate == pytest.approx(EquityFeeProvider.SEC_FEE_RATE)
        assert provider.taf_fee_rate == pytest.approx(EquityFeeProvider.TAF_FEE_RATE)

    def test_buy_order_no_fee(self, equity_fee_config):
        """Test that buy orders have no fee."""
        provider = EquityFeeProvider.from_config(equity_fee_config)

        fee = provider.compute_fee(
            notional=100000.0,
            side="BUY",
            liquidity="taker",
            qty=100.0,
        )

        assert fee == 0.0

    def test_sell_order_has_regulatory_fee(self, equity_fee_config):
        """Test that sell orders have regulatory fees."""
        provider = EquityFeeProvider.from_config(equity_fee_config)

        fee = provider.compute_fee(
            notional=100000.0,
            side="SELL",
            liquidity="taker",
            qty=100.0,
        )

        # SEC: 100000 * 0.0000278 = $2.78
        # TAF: 100 * 0.000166 = $0.0166
        expected = round(100000 * 0.0000278 + 100 * 0.000166, 4)
        assert fee == pytest.approx(expected, rel=1e-3)

    def test_taf_max_fee_cap(self, equity_fee_config):
        """Test TAF fee is capped at max."""
        provider = EquityFeeProvider.from_config(equity_fee_config)

        # Large order to trigger TAF cap: 100000 shares * 0.000166 = $16.6 > $8.30
        fee = provider.compute_fee(
            notional=10_000_000.0,
            side="SELL",
            liquidity="taker",
            qty=100000.0,
        )

        # SEC: 10_000_000 * 0.0000278 = $278
        # TAF: capped at $8.30
        expected = round(10_000_000 * 0.0000278 + 8.30, 4)
        assert fee == pytest.approx(expected, rel=1e-3)


# =============================================================================
# Test create_providers_from_asset_class()
# =============================================================================


class TestCreateProvidersFromAssetClass:
    """Tests for create_providers_from_asset_class()."""

    def test_equity_providers(self):
        """Test creating equity providers."""
        slippage, fees = create_providers_from_asset_class(AssetClass.EQUITY)

        assert isinstance(slippage, StatisticalSlippageProvider)
        assert isinstance(fees, EquityFeeProvider)
        assert slippage.spread_bps == pytest.approx(2.0)
        assert slippage.impact_coef == pytest.approx(0.05)

    def test_crypto_providers(self):
        """Test creating crypto providers."""
        slippage, fees = create_providers_from_asset_class(AssetClass.CRYPTO)

        assert isinstance(slippage, StatisticalSlippageProvider)
        assert isinstance(fees, CryptoFeeProvider)
        assert slippage.spread_bps == pytest.approx(5.0)
        assert slippage.impact_coef == pytest.approx(0.10)

    def test_futures_providers(self):
        """Test creating futures providers."""
        slippage, fees = create_providers_from_asset_class(AssetClass.FUTURES)

        assert isinstance(slippage, StatisticalSlippageProvider)
        assert isinstance(fees, CryptoFeeProvider)
        assert slippage.spread_bps == pytest.approx(4.0)
        assert slippage.impact_coef == pytest.approx(0.09)

    def test_custom_slippage_config(self):
        """Test custom slippage config override."""
        custom_slippage = {
            "spread_bps": 1.0,
            "impact_coef": 0.02,
        }

        slippage, fees = create_providers_from_asset_class(
            AssetClass.EQUITY, slippage_config=custom_slippage
        )

        assert slippage.spread_bps == pytest.approx(1.0)
        assert slippage.impact_coef == pytest.approx(0.02)

    def test_custom_fee_config(self):
        """Test custom fee config override."""
        custom_fee = {
            "maker_bps": 1.0,
            "taker_bps": 2.0,
        }

        slippage, fees = create_providers_from_asset_class(AssetClass.CRYPTO, fee_config=custom_fee)

        assert fees.maker_bps == pytest.approx(1.0)
        assert fees.taker_bps == pytest.approx(2.0)


# =============================================================================
# Test load_slippage_profile()
# =============================================================================


class TestLoadSlippageProfile:
    """Tests for load_slippage_profile()."""

    def test_load_equity_profile(self):
        """Test loading equity profile from YAML or defaults."""
        provider = load_slippage_profile("equity")

        assert isinstance(provider, StatisticalSlippageProvider)
        # Should get equity defaults (from YAML or hardcoded)
        assert provider.spread_bps >= 1.0
        assert provider.spread_bps <= 5.0
        assert provider.impact_coef >= 0.03
        assert provider.impact_coef <= 0.10

    def test_load_crypto_profile(self):
        """Test loading crypto profile."""
        provider = load_slippage_profile("crypto")

        assert isinstance(provider, StatisticalSlippageProvider)
        assert provider.spread_bps >= 3.0
        assert provider.spread_bps <= 10.0

    def test_load_nonexistent_profile(self):
        """Test loading nonexistent profile falls back to default."""
        provider = load_slippage_profile("nonexistent_profile")

        assert isinstance(provider, StatisticalSlippageProvider)
        # Should get default profile values
        assert provider.spread_bps == pytest.approx(5.0)


# =============================================================================
# Test Slippage Computation
# =============================================================================


class TestSlippageComputation:
    """Tests for slippage computation with different profiles."""

    def test_equity_lower_slippage(self):
        """Test that equity has lower slippage than crypto."""
        equity_provider = StatisticalSlippageProvider.from_profile("equity")
        crypto_provider = StatisticalSlippageProvider.from_profile("crypto")

        order = Order("AAPL", "BUY", 100.0, "MARKET")
        market = MarketState(0, adv=10_000_000.0)
        participation = 10000.0 / 10_000_000.0  # $10k / $10M ADV

        equity_slip = equity_provider.compute_slippage_bps(order, market, participation)
        crypto_slip = crypto_provider.compute_slippage_bps(order, market, participation)

        # Equity should have lower slippage due to tighter spreads
        assert equity_slip < crypto_slip

    def test_participation_impact(self):
        """Test that higher participation increases slippage."""
        provider = StatisticalSlippageProvider.from_profile("crypto")
        order = Order("BTCUSDT", "BUY", 1.0, "MARKET")
        market = MarketState(0, adv=100_000_000.0)

        low_participation = 1_000_000.0 / 100_000_000.0  # 1%
        high_participation = 10_000_000.0 / 100_000_000.0  # 10%

        low_slip = provider.compute_slippage_bps(order, market, low_participation)
        high_slip = provider.compute_slippage_bps(order, market, high_participation)

        # Higher participation should have more slippage
        assert high_slip > low_slip

    def test_slippage_bounds(self):
        """Test that slippage respects min/max bounds."""
        provider = StatisticalSlippageProvider(
            spread_bps=2.0,
            impact_coef=0.05,
            min_slippage_bps=5.0,
            max_slippage_bps=50.0,
        )

        order = Order("TEST", "BUY", 1.0, "MARKET")

        # Very small order - should hit min bound
        small_market = MarketState(0, adv=1_000_000_000.0)
        small_slip = provider.compute_slippage_bps(order, small_market, 0.000001)
        assert small_slip >= 5.0

        # Very large order - should hit max bound
        large_market = MarketState(0, adv=1_000.0)
        large_slip = provider.compute_slippage_bps(order, large_market, 0.5)
        assert large_slip <= 50.0


# =============================================================================
# Test Backward Compatibility
# =============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility with legacy configs."""

    def test_wrap_legacy_slippage_config(self, legacy_slippage_config):
        """Test wrapping legacy slippage config."""
        provider = wrap_legacy_slippage_config(legacy_slippage_config)

        assert isinstance(provider, StatisticalSlippageProvider)
        assert provider.impact_coef == pytest.approx(0.8)
        assert provider.spread_bps == pytest.approx(3.0)

    def test_wrap_legacy_slippage_config_none(self):
        """Test wrapping None slippage config."""
        provider = wrap_legacy_slippage_config(None)

        assert isinstance(provider, StatisticalSlippageProvider)
        assert provider.impact_coef == pytest.approx(0.1)  # Default
        assert provider.spread_bps == pytest.approx(5.0)  # Default

    def test_wrap_legacy_fees_model(self):
        """Test wrapping legacy fees model."""
        legacy_model = {
            "maker_rate_bps": 1.5,
            "taker_rate_bps": 3.5,
        }
        provider = wrap_legacy_fees_model(legacy_model)

        assert isinstance(provider, CryptoFeeProvider)
        assert provider.maker_bps == pytest.approx(1.5)
        assert provider.taker_bps == pytest.approx(3.5)

    def test_wrap_legacy_fees_model_none(self):
        """Test wrapping None fees model."""
        provider = wrap_legacy_fees_model(None)

        assert isinstance(provider, CryptoFeeProvider)
        assert provider.maker_bps == pytest.approx(2.0)  # Default
        assert provider.taker_bps == pytest.approx(4.0)  # Default


# =============================================================================
# Test L2ExecutionProvider Integration
# =============================================================================


class TestL2ExecutionProviderIntegration:
    """Tests for L2ExecutionProvider with different asset classes."""

    def test_equity_execution_provider(self):
        """Test creating equity execution provider."""
        provider = L2ExecutionProvider(asset_class=AssetClass.EQUITY)

        assert provider.asset_class == AssetClass.EQUITY
        assert isinstance(provider.slippage, StatisticalSlippageProvider)
        assert isinstance(provider.fees, EquityFeeProvider)

    def test_crypto_execution_provider(self):
        """Test creating crypto execution provider."""
        provider = L2ExecutionProvider(asset_class=AssetClass.CRYPTO)

        assert provider.asset_class == AssetClass.CRYPTO
        assert isinstance(provider.slippage, StatisticalSlippageProvider)
        assert isinstance(provider.fees, CryptoFeeProvider)

    def test_custom_providers(self):
        """Test using custom providers."""
        custom_slippage = StatisticalSlippageProvider(
            spread_bps=1.0,
            impact_coef=0.02,
        )
        custom_fees = CryptoFeeProvider(
            maker_bps=0.5,
            taker_bps=1.0,
        )

        provider = L2ExecutionProvider(
            asset_class=AssetClass.CRYPTO,
            slippage_provider=custom_slippage,
            fee_provider=custom_fees,
        )

        assert provider.slippage.spread_bps == pytest.approx(1.0)
        assert provider.fees.maker_bps == pytest.approx(0.5)

    def test_execution_cost_estimation(self):
        """Test pre-trade cost estimation."""
        provider = L2ExecutionProvider(asset_class=AssetClass.EQUITY)

        costs = provider.estimate_execution_cost(
            notional=100_000.0,
            adv=10_000_000.0,
            side="BUY",
            volatility=0.02,
        )

        assert "participation" in costs
        assert "slippage_bps" in costs
        assert "slippage_cost" in costs
        assert "fee" in costs
        assert "total_cost" in costs
        assert "total_bps" in costs

        assert costs["participation"] == pytest.approx(0.01, rel=1e-3)
        assert costs["total_cost"] > 0


# =============================================================================
# Test create_execution_provider Factory
# =============================================================================


class TestCreateExecutionProviderFactory:
    """Tests for create_execution_provider factory function."""

    def test_create_equity_provider(self):
        """Test factory creates equity provider."""
        provider = create_execution_provider(AssetClass.EQUITY)

        assert isinstance(provider, L2ExecutionProvider)
        assert provider.asset_class == AssetClass.EQUITY

    def test_create_crypto_provider(self):
        """Test factory creates crypto provider."""
        provider = create_execution_provider(AssetClass.CRYPTO)

        assert isinstance(provider, L2ExecutionProvider)
        assert provider.asset_class == AssetClass.CRYPTO

    def test_create_with_custom_providers(self):
        """Test factory with custom providers."""
        custom_slippage = StatisticalSlippageProvider(spread_bps=1.5)

        provider = create_execution_provider(
            AssetClass.EQUITY,
            slippage_provider=custom_slippage,
        )

        assert provider.slippage.spread_bps == pytest.approx(1.5)


# =============================================================================
# Test ExecutionSimulator Integration (if available)
# =============================================================================


class TestExecutionSimulatorIntegration:
    """Tests for ExecutionSimulator provider integration."""

    @pytest.fixture
    def skip_if_no_sim(self):
        """Skip if ExecutionSimulator not available."""
        try:
            from execution_sim import ExecutionSimulator

            return ExecutionSimulator
        except ImportError:
            pytest.skip("ExecutionSimulator not available")

    def test_simulator_with_equity_asset_class(self, skip_if_no_sim):
        """Test simulator with equity asset class."""
        ExecutionSimulator = skip_if_no_sim

        sim = ExecutionSimulator(
            symbol="AAPL",
            asset_class="equity",
        )

        # Check that providers were auto-created
        assert hasattr(sim, "_fee_provider")
        assert hasattr(sim, "_slippage_provider")
        assert hasattr(sim, "_asset_class_enum")

        # If providers available, check types
        if sim._fee_provider is not None:
            assert isinstance(sim._fee_provider, EquityFeeProvider)
        if sim._slippage_provider is not None:
            assert isinstance(sim._slippage_provider, StatisticalSlippageProvider)

    def test_simulator_with_crypto_asset_class(self, skip_if_no_sim):
        """Test simulator with crypto asset class."""
        ExecutionSimulator = skip_if_no_sim

        sim = ExecutionSimulator(
            symbol="BTCUSDT",
            asset_class="crypto",
        )

        # If providers available, check types
        if sim._fee_provider is not None:
            assert isinstance(sim._fee_provider, CryptoFeeProvider)

    def test_simulator_backward_compatible(self, skip_if_no_sim):
        """Test simulator is backward compatible without asset_class."""
        ExecutionSimulator = skip_if_no_sim

        # Should work without asset_class (legacy mode)
        sim = ExecutionSimulator(
            symbol="BTCUSDT",
        )

        # No providers should be auto-created in legacy mode
        assert sim._asset_class_enum is None

    def test_simulator_custom_fee_provider(self, skip_if_no_sim):
        """Test simulator with custom fee provider."""
        ExecutionSimulator = skip_if_no_sim

        custom_fees = EquityFeeProvider(
            sec_fee_rate=0.00003,
            taf_fee_rate=0.0002,
        )

        sim = ExecutionSimulator(
            symbol="AAPL",
            fee_provider=custom_fees,
        )

        assert sim._fee_provider is custom_fees

    def test_simulator_custom_slippage_provider(self, skip_if_no_sim):
        """Test simulator with custom slippage provider."""
        ExecutionSimulator = skip_if_no_sim

        custom_slippage = StatisticalSlippageProvider(
            spread_bps=1.0,
            impact_coef=0.03,
        )

        sim = ExecutionSimulator(
            symbol="AAPL",
            slippage_provider=custom_slippage,
        )

        assert sim._slippage_provider is custom_slippage


# =============================================================================
# Test Asset Class Differences
# =============================================================================


class TestAssetClassDifferences:
    """Tests verifying correct asset class specific behavior."""

    def test_equity_vs_crypto_spread(self):
        """Test that equity has tighter spreads than crypto."""
        equity_slippage, _ = create_providers_from_asset_class(AssetClass.EQUITY)
        crypto_slippage, _ = create_providers_from_asset_class(AssetClass.CRYPTO)

        assert equity_slippage.spread_bps < crypto_slippage.spread_bps

    def test_equity_vs_crypto_impact(self):
        """Test that equity has lower impact than crypto."""
        equity_slippage, _ = create_providers_from_asset_class(AssetClass.EQUITY)
        crypto_slippage, _ = create_providers_from_asset_class(AssetClass.CRYPTO)

        assert equity_slippage.impact_coef < crypto_slippage.impact_coef

    def test_equity_buy_no_fee(self):
        """Test that equity buys have no commission."""
        _, equity_fees = create_providers_from_asset_class(AssetClass.EQUITY)

        fee = equity_fees.compute_fee(
            notional=100_000.0,
            side="BUY",
            liquidity="taker",
            qty=100.0,
        )

        assert fee == 0.0

    def test_crypto_always_has_fee(self):
        """Test that crypto always has commission."""
        _, crypto_fees = create_providers_from_asset_class(AssetClass.CRYPTO)

        buy_fee = crypto_fees.compute_fee(
            notional=100_000.0,
            side="BUY",
            liquidity="taker",
            qty=1.0,
        )

        sell_fee = crypto_fees.compute_fee(
            notional=100_000.0,
            side="SELL",
            liquidity="taker",
            qty=1.0,
        )

        assert buy_fee > 0
        assert sell_fee > 0
        assert buy_fee == pytest.approx(sell_fee)

    def test_equity_sell_has_regulatory_fee(self):
        """Test that equity sells have regulatory fees."""
        _, equity_fees = create_providers_from_asset_class(AssetClass.EQUITY)

        fee = equity_fees.compute_fee(
            notional=100_000.0,
            side="SELL",
            liquidity="taker",
            qty=100.0,
        )

        assert fee > 0


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_notional_fee(self):
        """Test fee computation with zero notional."""
        provider = CryptoFeeProvider()

        fee = provider.compute_fee(
            notional=0.0,
            side="BUY",
            liquidity="taker",
            qty=0.0,
        )

        assert fee == 0.0

    def test_negative_notional_fee(self):
        """Test fee computation with negative notional."""
        provider = CryptoFeeProvider()

        fee = provider.compute_fee(
            notional=-10000.0,
            side="BUY",
            liquidity="taker",
            qty=-1.0,
        )

        # Should use abs(notional)
        assert fee > 0

    def test_zero_participation_slippage(self):
        """Test slippage with zero participation."""
        provider = StatisticalSlippageProvider()
        order = Order("TEST", "BUY", 1.0, "MARKET")
        market = MarketState(0)

        slippage = provider.compute_slippage_bps(order, market, 0.0)

        # Should return at least half-spread
        assert slippage >= 0

    def test_very_large_participation_slippage(self):
        """Test slippage with very large participation."""
        provider = StatisticalSlippageProvider(max_slippage_bps=100.0)
        order = Order("TEST", "BUY", 1.0, "MARKET")
        market = MarketState(0)

        slippage = provider.compute_slippage_bps(order, market, 0.5)  # 50% participation

        # Should be capped at max
        assert slippage <= 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
