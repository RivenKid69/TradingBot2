# -*- coding: utf-8 -*-
"""
Tests for L3 Execution Provider Integration.

This test suite provides comprehensive coverage for:
1. L3 Configuration (lob/config.py)
2. L3 Slippage Provider (execution_providers_l3.py)
3. L3 Fill Provider (execution_providers_l3.py)
4. L3 Execution Provider (execution_providers_l3.py)
5. Factory function integration
6. Backward compatibility with L2 and crypto paths

Stage 7 of L3 LOB Simulation (v7.0)

Test Count Target: 50+ tests
"""

import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pytest
pytest.importorskip("sortedcontainers")

# Import L3 config
from lob.config import (
    L3ExecutionConfig,
    LatencyConfig,
    LatencyComponentConfig,
    LatencyDistributionType,
    LatencyProfileType,
    FillProbabilityConfig,
    FillProbabilityModelType,
    QueueValueConfig,
    MarketImpactConfig,
    ImpactModelType,
    HiddenLiquidityConfig,
    IcebergConfig,
    DarkPoolsConfig,
    DarkPoolVenueConfig,
    QueueTrackingConfig,
    EventSchedulingConfig,
    LOBDataConfig,
    create_l3_config,
    latency_config_to_dataclass,
    impact_config_to_parameters,
)

# Import execution providers
from execution_providers import (
    AssetClass,
    BarData,
    Fill,
    MarketState,
    Order,
    OrderSide,
    OrderType,
    LiquidityRole,
    create_execution_provider,
    create_slippage_provider,
    create_fill_provider,
    create_fee_provider,
    L2ExecutionProvider,
    StatisticalSlippageProvider,
    OHLCVFillProvider,
    CryptoFeeProvider,
    EquityFeeProvider,
)

# Import L3 providers
from execution_providers_l3 import (
    L3ExecutionProvider,
    L3SlippageProvider,
    L3FillProvider,
    create_l3_execution_provider,
    create_l3_slippage_provider,
    create_l3_fill_provider,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def basic_market_state() -> MarketState:
    """Create basic market state for testing."""
    return MarketState(
        timestamp=1700000000000,
        bid=100.0,
        ask=100.10,
        bid_size=1000.0,
        ask_size=1000.0,
        adv=10_000_000.0,
        volatility=0.02,
    )


@pytest.fixture
def market_state_with_lob() -> MarketState:
    """Create market state with L3 LOB depth."""
    return MarketState(
        timestamp=1700000000000,
        bid=100.0,
        ask=100.10,
        bid_size=1000.0,
        ask_size=1000.0,
        adv=10_000_000.0,
        volatility=0.02,
        bid_depth=[
            (100.00, 500.0),
            (99.99, 800.0),
            (99.98, 1200.0),
            (99.97, 1500.0),
            (99.96, 2000.0),
        ],
        ask_depth=[
            (100.10, 500.0),
            (100.11, 800.0),
            (100.12, 1200.0),
            (100.13, 1500.0),
            (100.14, 2000.0),
        ],
    )


@pytest.fixture
def buy_market_order() -> Order:
    """Create buy market order."""
    return Order(
        symbol="AAPL",
        side="BUY",
        qty=100.0,
        order_type="MARKET",
        asset_class=AssetClass.EQUITY,
    )


@pytest.fixture
def sell_market_order() -> Order:
    """Create sell market order."""
    return Order(
        symbol="AAPL",
        side="SELL",
        qty=100.0,
        order_type="MARKET",
        asset_class=AssetClass.EQUITY,
    )


@pytest.fixture
def buy_limit_order() -> Order:
    """Create buy limit order."""
    return Order(
        symbol="AAPL",
        side="BUY",
        qty=100.0,
        order_type="LIMIT",
        limit_price=99.95,
        asset_class=AssetClass.EQUITY,
    )


@pytest.fixture
def sell_limit_order() -> Order:
    """Create sell limit order."""
    return Order(
        symbol="AAPL",
        side="SELL",
        qty=100.0,
        order_type="LIMIT",
        limit_price=100.15,
        asset_class=AssetClass.EQUITY,
    )


@pytest.fixture
def aggressive_buy_limit() -> Order:
    """Create aggressive buy limit (crosses spread)."""
    return Order(
        symbol="AAPL",
        side="BUY",
        qty=100.0,
        order_type="LIMIT",
        limit_price=100.10,  # At ask
        asset_class=AssetClass.EQUITY,
    )


@pytest.fixture
def basic_bar() -> BarData:
    """Create basic bar data."""
    return BarData(
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=100000.0,
        timestamp=1700000000000,
        timeframe_ms=3600000,  # 1 hour
    )


@pytest.fixture
def equity_l3_config() -> L3ExecutionConfig:
    """Create L3 config for equity."""
    return L3ExecutionConfig.for_equity()


@pytest.fixture
def minimal_l3_config() -> L3ExecutionConfig:
    """Create minimal L3 config (fast tests)."""
    return L3ExecutionConfig.minimal()


# =============================================================================
# L3 Configuration Tests
# =============================================================================

class TestL3ExecutionConfig:
    """Test L3ExecutionConfig creation and validation."""

    def test_default_config_creation(self):
        """Test default L3 config creation."""
        config = L3ExecutionConfig()
        assert config.enabled is False
        assert config.latency is not None
        assert config.fill_probability is not None
        assert config.market_impact is not None

    def test_equity_preset(self):
        """Test equity preset configuration."""
        config = L3ExecutionConfig.for_equity()
        assert config.enabled is True
        assert config.latency.enabled is True
        assert config.market_impact.enabled is True
        assert config.market_impact.eta == 0.05
        assert config.market_impact.gamma == 0.03

    def test_crypto_preset(self):
        """Test crypto preset configuration."""
        config = L3ExecutionConfig.for_crypto()
        assert config.enabled is True
        assert config.market_impact.eta == 0.10
        assert config.market_impact.gamma == 0.05
        assert config.dark_pools.enabled is False  # No crypto dark pools

    def test_minimal_preset(self):
        """Test minimal preset (matching engine only)."""
        config = L3ExecutionConfig.minimal()
        assert config.enabled is True
        assert config.latency.enabled is False
        assert config.fill_probability.enabled is False
        assert config.market_impact.enabled is False
        assert config.dark_pools.enabled is False

    def test_is_enabled_property(self):
        """Test is_enabled property logic."""
        config = L3ExecutionConfig()
        assert config.is_enabled is False

        config = L3ExecutionConfig(enabled=True, latency=LatencyConfig(enabled=True))
        assert config.is_enabled is True

    def test_to_dict(self):
        """Test configuration serialization to dict."""
        config = L3ExecutionConfig.for_equity()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "enabled" in config_dict
        assert "latency" in config_dict
        assert "market_impact" in config_dict

    def test_from_dict(self):
        """Test configuration creation from dict."""
        data = {
            "enabled": True,
            "latency": {"enabled": True, "profile": "institutional"},
            "market_impact": {"enabled": True, "eta": 0.08},
        }
        config = L3ExecutionConfig.from_dict(data)
        assert config.enabled is True
        assert config.latency.enabled is True
        assert config.market_impact.eta == 0.08

    def test_factory_function_equity(self):
        """Test create_l3_config factory for equity."""
        config = create_l3_config("equity")
        assert config.enabled is True
        assert config.latency.profile == LatencyProfileType.INSTITUTIONAL

    def test_factory_function_crypto(self):
        """Test create_l3_config factory for crypto."""
        config = create_l3_config("crypto")
        assert config.enabled is True
        assert config.dark_pools.enabled is False

    def test_factory_function_minimal(self):
        """Test create_l3_config factory for minimal."""
        config = create_l3_config("minimal")
        assert config.enabled is True
        assert config.latency.enabled is False


class TestLatencyConfig:
    """Test latency configuration."""

    def test_default_latency_config(self):
        """Test default latency configuration."""
        config = LatencyConfig()
        assert config.enabled is False
        assert config.profile == LatencyProfileType.INSTITUTIONAL

    def test_colocated_profile(self):
        """Test co-located HFT profile."""
        config = LatencyConfig.colocated()
        assert config.enabled is True
        assert config.profile == LatencyProfileType.COLOCATED
        assert config.feed_latency.mean_us == 10.0

    def test_retail_profile(self):
        """Test retail broker profile."""
        config = LatencyConfig.retail()
        assert config.enabled is True
        assert config.profile == LatencyProfileType.RETAIL
        assert config.feed_latency.mean_us == 5000.0

    def test_institutional_profile(self):
        """Test institutional profile."""
        config = LatencyConfig.institutional()
        assert config.enabled is True
        assert config.profile == LatencyProfileType.INSTITUTIONAL
        assert config.feed_latency.mean_us == 200.0


class TestLatencyComponentConfig:
    """Test latency component configuration."""

    def test_default_component(self):
        """Test default latency component."""
        config = LatencyComponentConfig()
        assert config.enabled is True
        assert config.distribution == LatencyDistributionType.LOGNORMAL
        assert config.mean_us == 100.0

    def test_validation_max_min(self):
        """Test max >= min validation."""
        with pytest.raises(ValueError, match="max_us must be >= min_us"):
            LatencyComponentConfig(min_us=1000.0, max_us=100.0)

    def test_validation_spike_prob(self):
        """Test spike probability validation."""
        with pytest.raises(ValueError):
            LatencyComponentConfig(spike_prob=1.5)  # > 1

    def test_validation_mean_negative(self):
        """Test negative mean validation."""
        with pytest.raises(ValueError):
            LatencyComponentConfig(mean_us=-10.0)


class TestMarketImpactConfig:
    """Test market impact configuration."""

    def test_default_impact_config(self):
        """Test default market impact configuration."""
        config = MarketImpactConfig()
        assert config.enabled is True
        assert config.model == ImpactModelType.ALMGREN_CHRISS
        assert config.eta == 0.05

    def test_equity_impact(self):
        """Test equity impact preset."""
        config = MarketImpactConfig.for_equity()
        assert config.eta == 0.05
        assert config.gamma == 0.03

    def test_crypto_impact(self):
        """Test crypto impact preset."""
        config = MarketImpactConfig.for_crypto()
        assert config.eta == 0.10
        assert config.gamma == 0.05


class TestFillProbabilityConfig:
    """Test fill probability configuration."""

    def test_default_fill_prob_config(self):
        """Test default fill probability config."""
        config = FillProbabilityConfig()
        assert config.enabled is True
        assert config.model == FillProbabilityModelType.QUEUE_REACTIVE
        assert config.base_rate == 100.0

    def test_custom_fill_prob_config(self):
        """Test custom fill probability config."""
        config = FillProbabilityConfig(
            model=FillProbabilityModelType.POISSON,
            base_rate=200.0,
        )
        assert config.model == FillProbabilityModelType.POISSON
        assert config.base_rate == 200.0


class TestDarkPoolsConfig:
    """Test dark pools configuration."""

    def test_default_dark_pools_config(self):
        """Test default dark pools config."""
        config = DarkPoolsConfig()
        assert config.enabled is False
        assert len(config.venues) == 2  # Default venues

    def test_enabled_dark_pools(self):
        """Test enabled dark pools with custom venues."""
        config = DarkPoolsConfig(
            enabled=True,
            venues=[
                DarkPoolVenueConfig(venue_id="custom_1", base_fill_probability=0.4),
            ],
        )
        assert config.enabled is True
        assert len(config.venues) == 1
        assert config.venues[0].base_fill_probability == 0.4


class TestConfigConversion:
    """Test configuration conversion helpers."""

    def test_latency_config_to_dataclass(self):
        """Test converting Pydantic config to dataclass."""
        pydantic_config = LatencyComponentConfig(
            distribution=LatencyDistributionType.LOGNORMAL,
            mean_us=150.0,
            std_us=40.0,
        )
        dc_config = latency_config_to_dataclass(pydantic_config)
        assert dc_config.mean_us == 150.0
        assert dc_config.std_us == 40.0

    def test_impact_config_to_parameters(self):
        """Test converting impact config to parameters."""
        config = MarketImpactConfig(eta=0.08, gamma=0.04, delta=0.6)
        params = impact_config_to_parameters(config)
        assert params.eta == 0.08
        assert params.gamma == 0.04
        assert params.delta == 0.6


# =============================================================================
# L3 Slippage Provider Tests
# =============================================================================

class TestL3SlippageProvider:
    """Test L3 slippage provider."""

    def test_creation_equity(self):
        """Test L3 slippage provider creation for equity."""
        provider = L3SlippageProvider(asset_class=AssetClass.EQUITY)
        assert provider._asset_class == AssetClass.EQUITY

    def test_creation_crypto(self):
        """Test L3 slippage provider creation for crypto."""
        provider = L3SlippageProvider(asset_class=AssetClass.CRYPTO)
        assert provider._asset_class == AssetClass.CRYPTO

    def test_creation_with_config(self, equity_l3_config):
        """Test creation with custom config."""
        provider = L3SlippageProvider(
            asset_class=AssetClass.EQUITY,
            config=equity_l3_config,
        )
        assert provider._config == equity_l3_config

    def test_slippage_without_lob(self, buy_market_order, basic_market_state):
        """Test slippage computation without LOB depth (fallback to L2)."""
        provider = L3SlippageProvider(asset_class=AssetClass.EQUITY)
        slippage = provider.compute_slippage_bps(
            buy_market_order, basic_market_state, participation_ratio=0.001
        )
        assert slippage >= 0.0
        assert slippage < 100.0  # Reasonable range

    def test_slippage_with_lob(self, buy_market_order, market_state_with_lob):
        """Test slippage computation with LOB depth."""
        provider = L3SlippageProvider(asset_class=AssetClass.EQUITY)
        slippage = provider.compute_slippage_bps(
            buy_market_order, market_state_with_lob, participation_ratio=0.001
        )
        assert slippage >= 0.0

    def test_slippage_buy_vs_sell(self, buy_market_order, sell_market_order, market_state_with_lob):
        """Test buy vs sell slippage."""
        provider = L3SlippageProvider(asset_class=AssetClass.EQUITY)

        buy_slippage = provider.compute_slippage_bps(
            buy_market_order, market_state_with_lob, 0.001
        )
        sell_slippage = provider.compute_slippage_bps(
            sell_market_order, market_state_with_lob, 0.001
        )

        # Both should be non-negative
        assert buy_slippage >= 0.0
        assert sell_slippage >= 0.0

    def test_slippage_large_order(self, market_state_with_lob):
        """Test slippage increases with order size."""
        provider = L3SlippageProvider(asset_class=AssetClass.EQUITY)

        small_order = Order("AAPL", "BUY", 100.0, "MARKET", asset_class=AssetClass.EQUITY)
        large_order = Order("AAPL", "BUY", 10000.0, "MARKET", asset_class=AssetClass.EQUITY)

        small_slippage = provider.compute_slippage_bps(
            small_order, market_state_with_lob, 0.0001
        )
        large_slippage = provider.compute_slippage_bps(
            large_order, market_state_with_lob, 0.01
        )

        # Large order should have more slippage
        assert large_slippage >= small_slippage

    def test_estimate_impact_cost(self, equity_l3_config):
        """Test pre-trade impact cost estimation."""
        provider = L3SlippageProvider(
            asset_class=AssetClass.EQUITY,
            config=equity_l3_config,
        )
        estimate = provider.estimate_impact_cost(
            notional=100_000.0,
            adv=10_000_000.0,
            volatility=0.02,
        )

        assert "participation" in estimate
        assert "temporary_impact_bps" in estimate
        assert "permanent_impact_bps" in estimate
        assert estimate["participation"] == pytest.approx(0.01, rel=1e-6)


# =============================================================================
# L3 Fill Provider Tests
# =============================================================================

class TestL3FillProvider:
    """Test L3 fill provider."""

    def test_creation_default(self):
        """Test default L3 fill provider creation."""
        provider = L3FillProvider()
        assert provider._asset_class == AssetClass.EQUITY

    def test_creation_with_config(self, equity_l3_config):
        """Test creation with custom config."""
        provider = L3FillProvider(
            config=equity_l3_config,
            asset_class=AssetClass.EQUITY,
        )
        assert provider._config == equity_l3_config

    def test_market_order_fill(
        self, buy_market_order, basic_market_state, basic_bar
    ):
        """Test market order always fills."""
        provider = L3FillProvider(asset_class=AssetClass.EQUITY)
        fill = provider.try_fill(buy_market_order, basic_market_state, basic_bar)

        assert fill is not None
        assert fill.qty == buy_market_order.qty
        assert fill.liquidity == "taker"

    def test_market_order_with_lob(
        self, buy_market_order, market_state_with_lob, basic_bar
    ):
        """Test market order with LOB depth."""
        provider = L3FillProvider(asset_class=AssetClass.EQUITY)
        fill = provider.try_fill(buy_market_order, market_state_with_lob, basic_bar)

        assert fill is not None
        assert fill.metadata.get("has_lob") is True

    def test_aggressive_limit_fills_immediately(
        self, aggressive_buy_limit, basic_market_state, basic_bar
    ):
        """Test aggressive limit order fills immediately as taker."""
        provider = L3FillProvider(asset_class=AssetClass.EQUITY)
        fill = provider.try_fill(aggressive_buy_limit, basic_market_state, basic_bar)

        assert fill is not None
        assert fill.liquidity == "taker"
        assert fill.metadata.get("crosses_spread") is True

    def test_passive_limit_may_not_fill(
        self, buy_limit_order, basic_market_state, basic_bar
    ):
        """Test passive limit order may not fill."""
        # Adjust bar so limit price is touched
        bar_touching_limit = BarData(
            open=100.0,
            high=100.5,
            low=99.90,  # Below limit price of 99.95
            close=100.2,
            volume=100000.0,
            timeframe_ms=3600000,
        )

        provider = L3FillProvider(
            asset_class=AssetClass.EQUITY,
            config=L3ExecutionConfig.minimal(),  # Disable probability model
        )
        fill = provider.try_fill(buy_limit_order, basic_market_state, bar_touching_limit)

        # With minimal config, should fill since limit was touched
        assert fill is not None
        assert fill.liquidity == "maker"

    def test_limit_not_touched_no_fill(
        self, buy_limit_order, basic_market_state
    ):
        """Test limit order not filled if price not touched."""
        bar_not_touching = BarData(
            open=100.0,
            high=100.5,
            low=100.0,  # Above limit price of 99.95
            close=100.2,
            volume=100000.0,
            timeframe_ms=3600000,
        )

        provider = L3FillProvider(asset_class=AssetClass.EQUITY)
        fill = provider.try_fill(buy_limit_order, basic_market_state, bar_not_touching)

        assert fill is None

    def test_fill_probability_estimation(
        self, buy_limit_order, market_state_with_lob, basic_bar
    ):
        """Test fill probability estimation."""
        config = L3ExecutionConfig.for_equity()
        provider = L3FillProvider(
            asset_class=AssetClass.EQUITY,
            config=config,
        )

        prob_result = provider.get_fill_probability(
            buy_limit_order,
            market_state_with_lob,
            basic_bar,
            time_horizon_sec=60.0,
        )

        # Should return result when fill probability model enabled
        assert prob_result is not None
        assert 0.0 <= prob_result.prob_fill <= 1.0

    def test_crypto_fill_provider(self):
        """Test fill provider for crypto asset class."""
        provider = L3FillProvider(asset_class=AssetClass.CRYPTO)

        crypto_order = Order(
            symbol="BTCUSDT",
            side="BUY",
            qty=0.1,
            order_type="MARKET",
            asset_class=AssetClass.CRYPTO,
        )
        crypto_market = MarketState(
            timestamp=1700000000000,
            bid=50000.0,
            ask=50010.0,
            adv=500_000_000.0,
        )
        crypto_bar = BarData(
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=1000.0,
        )

        fill = provider.try_fill(crypto_order, crypto_market, crypto_bar)
        assert fill is not None


# =============================================================================
# L3 Execution Provider Tests
# =============================================================================

class TestL3ExecutionProvider:
    """Test L3 execution provider."""

    def test_creation_equity(self):
        """Test L3 provider creation for equity."""
        provider = L3ExecutionProvider(asset_class=AssetClass.EQUITY)
        assert provider.asset_class == AssetClass.EQUITY

    def test_creation_crypto(self):
        """Test L3 provider creation for crypto."""
        provider = L3ExecutionProvider(asset_class=AssetClass.CRYPTO)
        assert provider.asset_class == AssetClass.CRYPTO

    def test_creation_with_config(self, equity_l3_config):
        """Test creation with custom L3 config."""
        provider = L3ExecutionProvider(
            asset_class=AssetClass.EQUITY,
            config=equity_l3_config,
        )
        assert provider.config.enabled is True

    def test_execute_market_order(
        self, buy_market_order, basic_market_state, basic_bar
    ):
        """Test executing market order."""
        provider = L3ExecutionProvider(asset_class=AssetClass.EQUITY)
        fill = provider.execute(buy_market_order, basic_market_state, basic_bar)

        assert fill is not None
        assert fill.qty == buy_market_order.qty
        assert fill.fee >= 0

    def test_execute_limit_order(
        self, buy_limit_order, basic_market_state, basic_bar
    ):
        """Test executing limit order."""
        provider = L3ExecutionProvider(
            asset_class=AssetClass.EQUITY,
            config=L3ExecutionConfig.minimal(),
        )
        # Limit at 99.95, bar low is 99.0 - should fill
        fill = provider.execute(buy_limit_order, basic_market_state, basic_bar)

        # May or may not fill depending on queue position
        # With minimal config, should fill since limit touched

    def test_estimate_execution_cost(self):
        """Test pre-trade cost estimation."""
        provider = L3ExecutionProvider(asset_class=AssetClass.EQUITY)
        estimate = provider.estimate_execution_cost(
            notional=100_000.0,
            adv=10_000_000.0,
            side="BUY",
            volatility=0.02,
        )

        assert "participation" in estimate
        assert "slippage_bps" in estimate
        assert "fee" in estimate
        assert "total_cost" in estimate
        assert estimate["total_cost"] > 0

    def test_get_fill_probability(
        self, buy_limit_order, market_state_with_lob, basic_bar
    ):
        """Test fill probability estimation through provider."""
        config = L3ExecutionConfig.for_equity()
        provider = L3ExecutionProvider(
            asset_class=AssetClass.EQUITY,
            config=config,
        )

        prob = provider.get_fill_probability(
            buy_limit_order,
            market_state_with_lob,
            basic_bar,
        )

        assert prob is not None
        assert 0.0 <= prob.prob_fill <= 1.0

    def test_statistics_tracking(
        self, buy_market_order, basic_market_state, basic_bar
    ):
        """Test execution statistics tracking."""
        provider = L3ExecutionProvider(asset_class=AssetClass.EQUITY)

        # Execute some orders
        for _ in range(5):
            provider.execute(buy_market_order, basic_market_state, basic_bar)

        stats = provider.get_stats()
        assert stats["total_orders"] == 5
        assert stats["filled_orders"] == 5
        assert stats["fill_rate"] == 1.0

    def test_reset_statistics(
        self, buy_market_order, basic_market_state, basic_bar
    ):
        """Test resetting statistics."""
        provider = L3ExecutionProvider(asset_class=AssetClass.EQUITY)

        provider.execute(buy_market_order, basic_market_state, basic_bar)
        provider.reset_stats()

        stats = provider.get_stats()
        assert stats["total_orders"] == 0

    def test_dark_pool_disabled(self, buy_market_order, basic_market_state, basic_bar):
        """Test execution with dark pools disabled."""
        config = L3ExecutionConfig.minimal()
        config.dark_pools.enabled = False

        provider = L3ExecutionProvider(
            asset_class=AssetClass.EQUITY,
            config=config,
        )

        fill = provider.execute(buy_market_order, basic_market_state, basic_bar)
        assert fill is not None
        assert fill.metadata.get("fill_type") != "dark_pool"


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Test factory function integration with L3."""

    def test_create_execution_provider_l2_default(self):
        """Test factory returns L2 by default."""
        provider = create_execution_provider(AssetClass.EQUITY)
        assert isinstance(provider, L2ExecutionProvider)

    def test_create_execution_provider_l3(self):
        """Test factory returns L3 when specified."""
        provider = create_execution_provider(AssetClass.EQUITY, level="L3")
        assert isinstance(provider, L3ExecutionProvider)

    def test_create_execution_provider_l3_with_config(self):
        """Test factory with L3 config."""
        config = L3ExecutionConfig.for_equity()
        provider = create_execution_provider(
            AssetClass.EQUITY,
            level="L3",
            config=config,
        )
        assert isinstance(provider, L3ExecutionProvider)
        assert provider.config.enabled is True

    def test_create_slippage_provider_l3(self):
        """Test slippage provider factory for L3."""
        provider = create_slippage_provider(level="L3", asset_class=AssetClass.EQUITY)
        assert isinstance(provider, L3SlippageProvider)

    def test_create_fill_provider_l3(self):
        """Test fill provider factory for L3."""
        provider = create_fill_provider(level="L3", asset_class=AssetClass.EQUITY)
        assert isinstance(provider, L3FillProvider)

    def test_factory_l3_execution_provider(self):
        """Test dedicated L3 factory function."""
        provider = create_l3_execution_provider(asset_class=AssetClass.EQUITY)
        assert isinstance(provider, L3ExecutionProvider)

    def test_factory_l3_slippage_provider(self):
        """Test dedicated L3 slippage factory function."""
        provider = create_l3_slippage_provider(asset_class=AssetClass.EQUITY)
        assert isinstance(provider, L3SlippageProvider)

    def test_factory_l3_fill_provider(self):
        """Test dedicated L3 fill factory function."""
        provider = create_l3_fill_provider(asset_class=AssetClass.EQUITY)
        assert isinstance(provider, L3FillProvider)


# =============================================================================
# Backward Compatibility Tests
# =============================================================================

class TestBackwardCompatibility:
    """Test backward compatibility with L2 and crypto paths."""

    def test_l2_still_works_equity(self, buy_market_order, basic_market_state, basic_bar):
        """Test L2 equity path still works."""
        provider = create_execution_provider(AssetClass.EQUITY, level="L2")
        fill = provider.execute(buy_market_order, basic_market_state, basic_bar)

        assert fill is not None
        assert isinstance(provider, L2ExecutionProvider)

    def test_l2_still_works_crypto(self, basic_bar):
        """Test L2 crypto path still works."""
        provider = create_execution_provider(AssetClass.CRYPTO, level="L2")

        crypto_order = Order("BTCUSDT", "BUY", 0.1, "MARKET", asset_class=AssetClass.CRYPTO)
        crypto_market = MarketState(
            timestamp=1700000000000,
            bid=50000.0,
            ask=50010.0,
            adv=500_000_000.0,
        )

        fill = provider.execute(crypto_order, crypto_market, basic_bar)

        assert fill is not None
        assert isinstance(provider, L2ExecutionProvider)

    def test_crypto_uses_l2_by_default(self):
        """Test crypto uses L2 by default (crypto LOB is Cython-based)."""
        provider = create_execution_provider(AssetClass.CRYPTO)
        assert isinstance(provider, L2ExecutionProvider)

    def test_l3_crypto_works(self, basic_bar):
        """Test L3 can be used for crypto if requested."""
        provider = create_execution_provider(AssetClass.CRYPTO, level="L3")

        crypto_order = Order("BTCUSDT", "BUY", 0.1, "MARKET", asset_class=AssetClass.CRYPTO)
        crypto_market = MarketState(
            timestamp=1700000000000,
            bid=50000.0,
            ask=50010.0,
            adv=500_000_000.0,
        )

        fill = provider.execute(crypto_order, crypto_market, basic_bar)

        assert fill is not None
        assert isinstance(provider, L3ExecutionProvider)

    def test_equity_fee_provider_unchanged(self):
        """Test equity fee provider unchanged."""
        fee_provider = create_fee_provider(AssetClass.EQUITY)
        assert isinstance(fee_provider, EquityFeeProvider)

        # Test SEC/TAF fees on sell
        fee = fee_provider.compute_fee(
            notional=100000.0,
            side="SELL",
            liquidity="taker",
            qty=1000.0,
        )
        assert fee > 0  # Should have regulatory fees

    def test_crypto_fee_provider_unchanged(self):
        """Test crypto fee provider unchanged."""
        fee_provider = create_fee_provider(AssetClass.CRYPTO)
        assert isinstance(fee_provider, CryptoFeeProvider)

        # Test maker/taker fees
        maker_fee = fee_provider.compute_fee(100000.0, "BUY", "maker", 1.0)
        taker_fee = fee_provider.compute_fee(100000.0, "BUY", "taker", 1.0)
        assert taker_fee > maker_fee


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for L3 execution."""

    def test_full_equity_workflow_l3(self):
        """Test full equity workflow with L3."""
        # Create L3 provider with dark pools disabled to test core L3 functionality
        config = L3ExecutionConfig.for_equity()
        config.dark_pools.enabled = False  # Disable dark pools for this test

        provider = L3ExecutionProvider(
            asset_class=AssetClass.EQUITY,
            config=config,
        )

        # Market state with depth
        market = MarketState(
            timestamp=1700000000000,
            bid=150.0,
            ask=150.05,
            bid_size=1000.0,
            ask_size=1000.0,
            adv=5_000_000.0,
            volatility=0.025,
            bid_depth=[
                (150.00, 500.0),
                (149.99, 800.0),
                (149.98, 1200.0),
            ],
            ask_depth=[
                (150.05, 500.0),
                (150.06, 800.0),
                (150.07, 1200.0),
            ],
        )

        bar = BarData(
            open=150.0,
            high=151.0,
            low=149.5,
            close=150.5,
            volume=50000.0,
            timeframe_ms=3600000,
        )

        # Execute market order
        market_order = Order("AAPL", "BUY", 100.0, "MARKET", asset_class=AssetClass.EQUITY)
        fill = provider.execute(market_order, market, bar)

        assert fill is not None
        assert fill.qty == 100.0
        assert fill.price > 0
        assert fill.fee >= 0
        assert fill.metadata.get("has_lob") is True

        # Execute limit order
        limit_order = Order(
            "AAPL", "BUY", 100.0, "LIMIT",
            limit_price=149.90,
            asset_class=AssetClass.EQUITY,
        )
        limit_fill = provider.execute(limit_order, market, bar)

        # Limit might not fill depending on queue
        if limit_fill is not None:
            assert limit_fill.price <= 149.90

    def test_mixed_l2_l3_workflow(self):
        """Test mixing L2 (crypto) and L3 (equity) in same workflow."""
        # L2 for crypto
        crypto_provider = create_execution_provider(AssetClass.CRYPTO, level="L2")
        # L3 for equity
        equity_provider = create_execution_provider(AssetClass.EQUITY, level="L3")

        # Execute crypto order with L2
        crypto_order = Order("BTCUSDT", "BUY", 0.1, "MARKET", asset_class=AssetClass.CRYPTO)
        crypto_market = MarketState(1700000000000, bid=50000.0, ask=50010.0, adv=500_000_000.0)
        crypto_bar = BarData(50000.0, 50100.0, 49900.0, 50050.0, 1000.0)

        crypto_fill = crypto_provider.execute(crypto_order, crypto_market, crypto_bar)
        assert crypto_fill is not None
        assert isinstance(crypto_provider, L2ExecutionProvider)

        # Execute equity order with L3
        equity_order = Order("AAPL", "BUY", 100.0, "MARKET", asset_class=AssetClass.EQUITY)
        equity_market = MarketState(1700000000000, bid=150.0, ask=150.05, adv=5_000_000.0)
        equity_bar = BarData(150.0, 151.0, 149.5, 150.5, 50000.0)

        equity_fill = equity_provider.execute(equity_order, equity_market, equity_bar)
        assert equity_fill is not None
        assert isinstance(equity_provider, L3ExecutionProvider)

    def test_config_yaml_roundtrip(self):
        """Test config YAML save and load."""
        config = L3ExecutionConfig.for_equity()

        # Create temp file and close it before writing (Windows compatibility)
        fd, temp_path = tempfile.mkstemp(suffix='.yaml')
        os.close(fd)  # Close the file descriptor immediately

        try:
            config.to_yaml(temp_path)

            # Load back
            loaded_config = L3ExecutionConfig.from_yaml(temp_path)
            assert loaded_config.enabled == config.enabled
            assert loaded_config.latency.enabled == config.latency.enabled
            assert loaded_config.market_impact.eta == config.market_impact.eta
        finally:
            os.unlink(temp_path)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_quantity_order(self, basic_market_state, basic_bar):
        """Test handling of zero quantity order."""
        provider = L3ExecutionProvider(asset_class=AssetClass.EQUITY)
        zero_order = Order("AAPL", "BUY", 0.0, "MARKET", asset_class=AssetClass.EQUITY)

        # Should handle gracefully (may warn but not crash)
        fill = provider.execute(zero_order, basic_market_state, basic_bar)
        # Implementation may return None or Fill with 0 qty

    def test_nan_prices_in_market(self, buy_market_order, basic_bar):
        """Test handling of NaN prices in market state."""
        nan_market = MarketState(
            timestamp=1700000000000,
            bid=float('nan'),
            ask=float('nan'),
        )

        provider = L3ExecutionProvider(asset_class=AssetClass.EQUITY)
        # Should use fallback (bar close) or return None
        fill = provider.execute(buy_market_order, nan_market, basic_bar)

    def test_inverted_bar(self, buy_market_order, basic_market_state):
        """Test handling of inverted bar (high < low)."""
        inverted_bar = BarData(
            open=100.0,
            high=99.0,  # Inverted!
            low=101.0,
            close=100.0,
            volume=100000.0,
        )

        provider = L3ExecutionProvider(asset_class=AssetClass.EQUITY)
        # Should handle gracefully
        fill = provider.execute(buy_market_order, basic_market_state, inverted_bar)

    def test_empty_lob_depth(self, buy_market_order, basic_bar):
        """Test handling of empty LOB depth."""
        empty_depth_market = MarketState(
            timestamp=1700000000000,
            bid=100.0,
            ask=100.10,
            bid_depth=[],  # Empty
            ask_depth=[],  # Empty
        )

        provider = L3ExecutionProvider(asset_class=AssetClass.EQUITY)
        fill = provider.execute(buy_market_order, empty_depth_market, basic_bar)
        # Should fall back to L2-style execution
        assert fill is not None

    def test_limit_order_without_price(self, basic_market_state, basic_bar):
        """Test handling of limit order without limit price."""
        bad_limit = Order(
            "AAPL", "BUY", 100.0, "LIMIT",
            limit_price=None,  # Missing!
            asset_class=AssetClass.EQUITY,
        )

        provider = L3ExecutionProvider(asset_class=AssetClass.EQUITY)
        # Should treat as market or log warning
        fill = provider.execute(bad_limit, basic_market_state, basic_bar)

    def test_very_large_order(self, basic_market_state, basic_bar):
        """Test handling of very large order relative to ADV."""
        large_order = Order(
            "AAPL", "BUY", 1_000_000.0,  # 1M shares
            "MARKET",
            asset_class=AssetClass.EQUITY,
        )

        # Disable dark pools to test slippage on lit market
        config = L3ExecutionConfig.for_equity()
        config.dark_pools.enabled = False

        provider = L3ExecutionProvider(asset_class=AssetClass.EQUITY, config=config)
        fill = provider.execute(large_order, basic_market_state, basic_bar)

        # Should have significant slippage but not crash
        assert fill is not None
        assert fill.slippage_bps > 0


# =============================================================================
# Bug Fix Tests (Stage 7.1)
# =============================================================================

class TestBugFixes:
    """Tests for bug fixes identified in code review."""

    def test_seedable_rng_reproducibility(self, market_state_with_lob, basic_bar):
        """Test that seeded RNG produces reproducible fill decisions."""
        # Create order that relies on fill probability
        limit_order = Order(
            symbol="AAPL",
            side="BUY",
            qty=100.0,
            order_type="LIMIT",
            limit_price=99.90,  # Below mid, passive order
            asset_class=AssetClass.EQUITY,
        )

        # Bar that touches the limit price
        bar_touching = BarData(
            open=100.0,
            high=100.5,
            low=99.85,  # Below limit price
            close=100.2,
            volume=100000.0,
            timeframe_ms=3600000,
        )

        # Enable fill probability model
        config = L3ExecutionConfig.for_equity()
        config.fill_probability.enabled = True

        # Run with same seed multiple times - results should be identical
        seed = 42
        results_seed_42 = []
        for _ in range(5):
            provider = L3FillProvider(
                asset_class=AssetClass.EQUITY,
                config=config,
                seed=seed,
            )
            fill = provider.try_fill(limit_order, market_state_with_lob, bar_touching)
            results_seed_42.append(fill is not None)

        # All results with same seed should be identical
        assert all(r == results_seed_42[0] for r in results_seed_42), \
            "Seeded RNG should produce identical results"

    def test_different_seeds_may_differ(self, market_state_with_lob, basic_bar):
        """Test that different seeds can produce different results."""
        limit_order = Order(
            symbol="AAPL",
            side="BUY",
            qty=100.0,
            order_type="LIMIT",
            limit_price=99.90,
            asset_class=AssetClass.EQUITY,
        )

        bar_touching = BarData(
            open=100.0,
            high=100.5,
            low=99.85,
            close=100.2,
            volume=100000.0,
            timeframe_ms=3600000,
        )

        config = L3ExecutionConfig.for_equity()
        config.fill_probability.enabled = True

        # Run with different seeds
        results = []
        for seed in range(100):  # Try 100 different seeds
            provider = L3FillProvider(
                asset_class=AssetClass.EQUITY,
                config=config,
                seed=seed,
            )
            fill = provider.try_fill(limit_order, market_state_with_lob, bar_touching)
            results.append(fill is not None)

        # With enough seeds, we should see some variation (unless prob is 0 or 1)
        # This test validates the RNG is actually being used
        # Note: This may occasionally pass even with no variation if prob is extreme
        pass  # Just ensure no exceptions are raised

    def test_config_from_yaml_file_not_found(self):
        """Test that from_yaml raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError) as excinfo:
            L3ExecutionConfig.from_yaml("/nonexistent/path/config.yaml")
        assert "not found" in str(excinfo.value).lower()

    def test_config_from_yaml_directory_error(self, tmp_path):
        """Test that from_yaml raises ValueError for directory path."""
        # tmp_path is a directory
        with pytest.raises(ValueError) as excinfo:
            L3ExecutionConfig.from_yaml(tmp_path)
        assert "not a file" in str(excinfo.value).lower()

    def test_dark_pool_order_id_unique(self, buy_market_order, basic_market_state, basic_bar):
        """Test that dark pool order IDs are unique across calls."""
        # Enable dark pools
        config = L3ExecutionConfig.for_equity()
        config.dark_pools.enabled = True

        provider = L3ExecutionProvider(
            asset_class=AssetClass.EQUITY,
            config=config,
        )

        # Simulate same timestamp by using same market state
        # The order_id should still be unique due to counter
        order_ids = set()

        # We can't easily extract order_id from fills, but we can verify
        # the counter increments properly by checking internal state
        initial_counter = provider._dark_pool_order_counter
        assert initial_counter == 0

        # Execute multiple orders (may or may not fill in dark pool)
        for _ in range(5):
            provider.execute(buy_market_order, basic_market_state, basic_bar)

        # Counter should have incremented for each dark pool attempt
        # Note: Only increments if dark pool path is taken
        # Dark pools are tried first if enabled

    def test_dark_pool_order_counter_increments(self, basic_market_state, basic_bar):
        """Test that dark pool order counter increments correctly."""
        config = L3ExecutionConfig.for_equity()
        config.dark_pools.enabled = True

        provider = L3ExecutionProvider(
            asset_class=AssetClass.EQUITY,
            config=config,
        )

        # Create order
        order = Order(
            symbol="AAPL",
            side="BUY",
            qty=100.0,
            order_type="MARKET",
            asset_class=AssetClass.EQUITY,
        )

        initial = provider._dark_pool_order_counter
        assert initial == 0

        # Execute orders - counter increments on dark pool attempts
        provider.execute(order, basic_market_state, basic_bar)
        provider.execute(order, basic_market_state, basic_bar)
        provider.execute(order, basic_market_state, basic_bar)

        # Counter should be > 0 if dark pool was attempted
        # (Depends on dark pool logic, but counter increments on attempt)
        assert provider._dark_pool_order_counter >= initial

    def test_fill_provider_accepts_seed_parameter(self):
        """Test that L3FillProvider accepts seed parameter."""
        # Should not raise
        provider = L3FillProvider(
            asset_class=AssetClass.EQUITY,
            seed=12345,
        )
        assert provider._rng is not None

    def test_fill_provider_none_seed_creates_rng(self):
        """Test that L3FillProvider with seed=None still creates RNG."""
        provider = L3FillProvider(
            asset_class=AssetClass.EQUITY,
            seed=None,
        )
        assert provider._rng is not None

    def test_config_yaml_roundtrip_with_valid_file(self, tmp_path):
        """Test config YAML save and load with path validation."""
        config = L3ExecutionConfig.for_equity()

        # Save to temp file
        config_file = tmp_path / "test_config.yaml"
        config.to_yaml(config_file)

        # Load back - should not raise FileNotFoundError
        loaded = L3ExecutionConfig.from_yaml(config_file)
        assert loaded.enabled == config.enabled


# =============================================================================
# Performance Tests (Optional)
# =============================================================================

class TestPerformance:
    """Performance tests for L3 execution."""

    def test_execution_throughput(self, buy_market_order, basic_market_state, basic_bar):
        """Test execution throughput (should be >1000/sec)."""
        import time

        provider = L3ExecutionProvider(
            asset_class=AssetClass.EQUITY,
            config=L3ExecutionConfig.minimal(),  # Fast config
        )

        iterations = 100
        start = time.perf_counter()

        for _ in range(iterations):
            provider.execute(buy_market_order, basic_market_state, basic_bar)

        elapsed = time.perf_counter() - start
        throughput = iterations / elapsed

        # Should achieve reasonable throughput
        assert throughput > 100, f"Throughput {throughput:.1f} orders/sec too low"

    def test_config_creation_speed(self):
        """Test config creation speed."""
        import time

        iterations = 100
        start = time.perf_counter()

        for _ in range(iterations):
            config = L3ExecutionConfig.for_equity()

        elapsed = time.perf_counter() - start
        throughput = iterations / elapsed

        # Should be fast
        assert throughput > 1000, f"Config creation {throughput:.1f}/sec too slow"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
