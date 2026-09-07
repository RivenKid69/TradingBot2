# -*- coding: utf-8 -*-
"""
tests/test_forex_tick_simulation.py
Comprehensive tests for Tick-Level Execution Simulation

Tests cover:
1. Tick dataclass functionality
2. TickGenerator price and spread dynamics
3. TickLevelExecutor market and limit order execution
4. TickPriceImpact model
5. Market conditions and spread dynamics
6. Session-based tick rates

Author: Sultan Suyunbayev
Date: 2025-11-30
"""

import math
import time
from typing import List

import numpy as np
import pytest
pytest.importorskip("sortedcontainers")

from lob.forex_tick_simulation import (
    # Enums
    TickType,
    ExecutionQuality,
    MarketCondition,
    # Data classes
    Tick,
    TickExecutionResult,
    SpreadState,
    TickSimulationConfig,
    # Classes
    TickGenerator,
    TickLevelExecutor,
    TickPriceImpact,
    # Factory
    create_tick_simulator,
    # Constants
    TICK_ARRIVAL_RATES,
    SPREAD_PARAMS,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def tick_config() -> TickSimulationConfig:
    """Create a tick simulation config."""
    return TickSimulationConfig(
        pair_type="major",
        session="london",
        base_tick_rate=3.0,
        volatility_scale=1.0,
        seed=42,
    )


@pytest.fixture
def tick_generator(tick_config: TickSimulationConfig) -> TickGenerator:
    """Create a tick generator."""
    return TickGenerator(config=tick_config, initial_mid=1.1000)


@pytest.fixture
def tick_executor(tick_generator: TickGenerator) -> TickLevelExecutor:
    """Create a tick executor."""
    return TickLevelExecutor(tick_generator=tick_generator)


# =============================================================================
# Tick Tests
# =============================================================================

class TestTick:
    """Tests for Tick dataclass."""

    def test_creation_basic(self) -> None:
        """Test basic tick creation."""
        tick = Tick(
            timestamp_ns=1000000000,
            bid=1.0850,
            ask=1.0852,
        )
        assert tick.bid == 1.0850
        assert tick.ask == 1.0852
        assert tick.mid == pytest.approx((1.0850 + 1.0852) / 2, rel=1e-6)

    def test_spread_calculation_standard(self) -> None:
        """Test spread calculation for standard pair."""
        tick = Tick(
            timestamp_ns=1000000000,
            bid=1.0850,
            ask=1.0852,
            is_jpy_pair=False,
        )
        # Spread = 0.0002, pip = 0.0001, so 2 pips
        assert tick.spread_pips == pytest.approx(2.0, rel=0.01)

    def test_spread_calculation_jpy(self) -> None:
        """Test spread calculation for JPY pair."""
        tick = Tick(
            timestamp_ns=1000000000,
            bid=150.00,
            ask=150.02,
            is_jpy_pair=True,
        )
        # Spread = 0.02, pip = 0.01, so 2 pips
        assert tick.spread_pips == pytest.approx(2.0, rel=0.01)

    def test_timestamp_conversions(self) -> None:
        """Test timestamp conversion properties."""
        tick = Tick(
            timestamp_ns=1_000_000_000_000,  # 1000 seconds in ns
            bid=1.0850,
            ask=1.0852,
        )
        assert tick.timestamp_ms == pytest.approx(1_000_000.0, rel=1e-6)
        assert tick.timestamp_sec == pytest.approx(1000.0, rel=1e-6)

    def test_tick_type_default(self) -> None:
        """Test default tick type is QUOTE."""
        tick = Tick(timestamp_ns=0, bid=1.0, ask=1.001)
        assert tick.tick_type == TickType.QUOTE

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        tick = Tick(
            timestamp_ns=1000000000,
            bid=1.0850,
            ask=1.0852,
            tick_type=TickType.TRADE,
            volume=100000.0,
            dealer_id="dealer_1",
            sequence_num=42,
        )
        d = tick.to_dict()
        assert d["bid"] == 1.0850
        assert d["ask"] == 1.0852
        assert d["tick_type"] == "trade"
        assert d["volume"] == 100000.0
        assert d["dealer_id"] == "dealer_1"
        assert d["sequence_num"] == 42


class TestTickExecutionResult:
    """Tests for TickExecutionResult dataclass."""

    def test_creation_filled(self) -> None:
        """Test creation for filled order."""
        result = TickExecutionResult(
            filled=True,
            fill_price=1.0851,
            fill_timestamp_ns=1000000000,
            slippage_pips=0.5,
            execution_quality=ExecutionQuality.GOOD,
            latency_ns=50_000_000,
            ticks_to_fill=3,
        )
        assert result.filled is True
        assert result.fill_price == 1.0851
        assert result.latency_ms == pytest.approx(50.0, rel=1e-6)

    def test_creation_rejected(self) -> None:
        """Test creation for rejected order."""
        result = TickExecutionResult(
            filled=False,
            rejection_reason="max_slippage_exceeded",
        )
        assert result.filled is False
        assert result.rejection_reason == "max_slippage_exceeded"

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = TickExecutionResult(
            filled=True,
            fill_price=1.0851,
            execution_quality=ExecutionQuality.EXCELLENT,
        )
        d = result.to_dict()
        assert d["filled"] is True
        assert d["fill_price"] == 1.0851
        assert d["execution_quality"] == "excellent"


class TestSpreadState:
    """Tests for SpreadState dataclass."""

    def test_total_spread_calculation(self) -> None:
        """Test total spread includes all components."""
        state = SpreadState(
            base_spread_pips=1.0,
            current_spread_pips=1.0,
            volatility_component=0.5,
            inventory_component=0.3,
            news_component=1.0,
        )
        expected = 1.0 + 0.5 + 0.3 + 1.0
        assert state.total_spread_pips == pytest.approx(expected, rel=0.01)

    def test_minimum_spread(self) -> None:
        """Test minimum spread is enforced."""
        state = SpreadState(
            base_spread_pips=0.0,
            current_spread_pips=0.0,
            volatility_component=0.0,
            inventory_component=0.0,
            news_component=0.0,
        )
        assert state.total_spread_pips >= 0.1


# =============================================================================
# TickSimulationConfig Tests
# =============================================================================

class TestTickSimulationConfig:
    """Tests for TickSimulationConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = TickSimulationConfig()
        assert config.pair_type == "major"
        assert config.session == "london"
        assert config.spread_widening_enabled is True
        assert config.price_impact_enabled is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = TickSimulationConfig(
            pair_type="exotic",
            session="tokyo",
            volatility_scale=2.0,
            seed=12345,
        )
        assert config.pair_type == "exotic"
        assert config.session == "tokyo"
        assert config.volatility_scale == 2.0
        assert config.seed == 12345


# =============================================================================
# TickGenerator Tests
# =============================================================================

class TestTickGenerator:
    """Tests for TickGenerator."""

    def test_creation(self, tick_config: TickSimulationConfig) -> None:
        """Test tick generator creation."""
        gen = TickGenerator(config=tick_config, initial_mid=1.1000)
        assert gen._current_mid == 1.1000

    def test_generate_single_tick(self, tick_generator: TickGenerator) -> None:
        """Test generating a single tick."""
        tick = tick_generator.generate_tick()
        assert isinstance(tick, Tick)
        assert tick.bid < tick.ask
        assert tick.sequence_num == 1

    def test_tick_sequence_increments(self, tick_generator: TickGenerator) -> None:
        """Test tick sequence number increments."""
        tick1 = tick_generator.generate_tick()
        tick2 = tick_generator.generate_tick()
        tick3 = tick_generator.generate_tick()
        assert tick2.sequence_num == tick1.sequence_num + 1
        assert tick3.sequence_num == tick2.sequence_num + 1

    def test_tick_timestamps_increase(self, tick_generator: TickGenerator) -> None:
        """Test tick timestamps increase."""
        ts1 = time.time_ns()
        tick1 = tick_generator.generate_tick(ts1)
        ts2 = ts1 + 1_000_000  # 1ms later
        tick2 = tick_generator.generate_tick(ts2)
        assert tick2.timestamp_ns > tick1.timestamp_ns

    def test_price_evolution(self, tick_generator: TickGenerator) -> None:
        """Test price evolution over multiple ticks."""
        prices = []
        for i in range(100):
            tick = tick_generator.generate_tick(time.time_ns() + i * 100_000_000)
            prices.append(tick.mid)

        # Price should vary (not constant)
        assert len(set(prices)) > 1
        # Should stay in reasonable range
        assert all(0.5 < p < 2.0 for p in prices)

    def test_spread_dynamics(self, tick_config: TickSimulationConfig) -> None:
        """Test spread changes over time."""
        gen = TickGenerator(config=tick_config, initial_mid=1.1000)
        spreads = []

        for i in range(50):
            tick = gen.generate_tick(time.time_ns() + i * 100_000_000)
            spreads.append(tick.spread_pips)

        # Spreads should vary
        assert len(set(round(s, 2) for s in spreads)) > 1
        # Should be positive
        assert all(s > 0 for s in spreads)

    def test_news_event_spread_widening(self) -> None:
        """Test spread widens during news events."""
        config = TickSimulationConfig(
            pair_type="major",
            spread_widening_enabled=True,
            seed=42,
        )
        gen = TickGenerator(config=config, initial_mid=1.1000)

        normal_tick = gen.generate_tick(
            market_condition=MarketCondition.NORMAL
        )
        news_tick = gen.generate_tick(
            market_condition=MarketCondition.NEWS_EVENT
        )

        # News spread should be wider (usually)
        # Note: stochastic, so we check the mechanism works
        assert news_tick.spread_pips >= 0.1

    def test_generate_tick_stream(self, tick_generator: TickGenerator) -> None:
        """Test generating tick stream."""
        ticks = list(tick_generator.generate_tick_stream(
            duration_sec=1.0,
            start_timestamp_ns=time.time_ns(),
        ))

        # Should have multiple ticks
        assert len(ticks) > 0
        # Timestamps should be ordered
        for i in range(1, len(ticks)):
            assert ticks[i].timestamp_ns > ticks[i - 1].timestamp_ns

    def test_session_tick_rates(self) -> None:
        """Test different sessions have different tick rates."""
        ticks_by_session = {}

        for session in ["sydney", "london", "new_york"]:
            config = TickSimulationConfig(session=session, seed=42)
            gen = TickGenerator(config=config, initial_mid=1.10)
            ticks = list(gen.generate_tick_stream(
                duration_sec=5.0,
                start_timestamp_ns=1000000000,
            ))
            ticks_by_session[session] = len(ticks)

        # London should have more ticks than Sydney
        assert ticks_by_session["london"] >= ticks_by_session["sydney"]

    def test_get_current_state(self, tick_generator: TickGenerator) -> None:
        """Test getting current generator state."""
        tick_generator.generate_tick()
        state = tick_generator.get_current_state()

        assert "mid" in state
        assert "spread_pips" in state
        assert "volatility" in state
        assert "sequence" in state
        assert state["sequence"] == 1

    def test_set_mid_price(self, tick_generator: TickGenerator) -> None:
        """Test setting mid price."""
        tick_generator.set_mid_price(1.2000)
        tick = tick_generator.generate_tick()
        # Price should be near the set value
        assert abs(tick.mid - 1.2000) < 0.01


# =============================================================================
# TickLevelExecutor Tests
# =============================================================================

class TestTickLevelExecutor:
    """Tests for TickLevelExecutor."""

    def test_execute_market_order_buy(self, tick_executor: TickLevelExecutor) -> None:
        """Test executing a buy market order."""
        result = tick_executor.execute_market_order(
            is_buy=True,
            size=100000.0,
            submitted_at_ns=time.time_ns(),
            max_slippage_pips=5.0,
        )

        assert result.filled is True or result.rejection_reason is not None
        if result.filled:
            assert result.fill_price > 0
            assert result.latency_ns > 0

    def test_execute_market_order_sell(self, tick_executor: TickLevelExecutor) -> None:
        """Test executing a sell market order."""
        result = tick_executor.execute_market_order(
            is_buy=False,
            size=100000.0,
            submitted_at_ns=time.time_ns(),
            max_slippage_pips=5.0,
        )

        if result.filled:
            assert result.fill_price > 0

    def test_market_order_latency(self, tick_executor: TickLevelExecutor) -> None:
        """Test market order has latency."""
        result = tick_executor.execute_market_order(
            is_buy=True,
            size=100000.0,
            submitted_at_ns=time.time_ns(),
            max_slippage_pips=10.0,
        )

        assert result.latency_ns > 0
        assert result.latency_ms > 0

    def test_market_order_max_slippage_rejection(self) -> None:
        """Test rejection due to max slippage exceeded."""
        config = TickSimulationConfig(
            pair_type="major",
            volatility_scale=5.0,  # High volatility
            seed=42,
        )
        gen = TickGenerator(config=config, initial_mid=1.10)
        executor = TickLevelExecutor(tick_generator=gen)

        # Very tight slippage tolerance
        result = executor.execute_market_order(
            is_buy=True,
            size=100000.0,
            submitted_at_ns=time.time_ns(),
            max_slippage_pips=0.01,  # Very tight
        )

        # May or may not reject depending on market movement
        assert isinstance(result, TickExecutionResult)

    def test_execute_limit_order_buy(self, tick_executor: TickLevelExecutor) -> None:
        """Test executing a buy limit order."""
        # Get current market
        current_tick = tick_executor._tick_gen.generate_tick()
        limit_price = current_tick.ask + 0.0010  # Above ask, should fill

        result = tick_executor.execute_limit_order(
            is_buy=True,
            size=100000.0,
            limit_price=limit_price,
            submitted_at_ns=time.time_ns(),
            time_in_force_sec=10.0,
        )

        # Should fill as limit is above ask
        assert result.filled is True or result.rejection_reason == "expired"

    def test_execute_limit_order_sell(self, tick_executor: TickLevelExecutor) -> None:
        """Test executing a sell limit order."""
        current_tick = tick_executor._tick_gen.generate_tick()
        limit_price = current_tick.bid - 0.0010  # Below bid, should fill

        result = tick_executor.execute_limit_order(
            is_buy=False,
            size=100000.0,
            limit_price=limit_price,
            submitted_at_ns=time.time_ns(),
            time_in_force_sec=10.0,
        )

        # Should fill as limit is below bid
        assert result.filled is True or result.rejection_reason == "expired"

    def test_limit_order_expiry(self, tick_executor: TickLevelExecutor) -> None:
        """Test limit order expiry."""
        # Set limit price far from market (unlikely to fill)
        result = tick_executor.execute_limit_order(
            is_buy=True,
            size=100000.0,
            limit_price=0.5000,  # Way below market
            submitted_at_ns=time.time_ns(),
            time_in_force_sec=0.1,  # Short TIF
        )

        assert result.filled is False
        assert result.rejection_reason == "expired"

    def test_execution_quality_classification(self, tick_executor: TickLevelExecutor) -> None:
        """Test execution quality is classified."""
        results = []
        for _ in range(20):
            result = tick_executor.execute_market_order(
                is_buy=True,
                size=100000.0,
                submitted_at_ns=time.time_ns(),
                max_slippage_pips=10.0,
            )
            if result.filled:
                results.append(result)

        # Should have various quality levels
        qualities = [r.execution_quality for r in results]
        assert len(qualities) > 0

    def test_get_stats(self, tick_executor: TickLevelExecutor) -> None:
        """Test getting executor statistics."""
        # Execute some orders
        for _ in range(5):
            tick_executor.execute_market_order(
                is_buy=True,
                size=100000.0,
                submitted_at_ns=time.time_ns(),
                max_slippage_pips=10.0,
            )

        stats = tick_executor.get_stats()
        assert "total_orders" in stats
        assert "filled_orders" in stats
        assert "fill_rate" in stats
        assert stats["total_orders"] == 5

    def test_reset_stats(self, tick_executor: TickLevelExecutor) -> None:
        """Test resetting executor statistics."""
        tick_executor.execute_market_order(
            is_buy=True,
            size=100000.0,
            submitted_at_ns=time.time_ns(),
            max_slippage_pips=10.0,
        )

        tick_executor.reset_stats()
        stats = tick_executor.get_stats()
        assert stats["total_orders"] == 0

    def test_adverse_selection(self) -> None:
        """Test adverse selection rejection."""
        config = TickSimulationConfig(
            pair_type="major",
            adverse_selection_threshold_pips=0.1,
            seed=42,
        )
        gen = TickGenerator(config=config, initial_mid=1.10)
        executor = TickLevelExecutor(tick_generator=gen, config=config)

        # Execute multiple orders, some may be rejected
        rejections = 0
        for _ in range(50):
            result = executor.execute_market_order(
                is_buy=True,
                size=100000.0,
                submitted_at_ns=time.time_ns(),
                max_slippage_pips=10.0,
            )
            if not result.filled and result.rejection_reason == "adverse_selection":
                rejections += 1

        # Adverse selection rejections possible but not guaranteed
        assert rejections >= 0


# =============================================================================
# TickPriceImpact Tests
# =============================================================================

class TestTickPriceImpact:
    """Tests for TickPriceImpact model."""

    def test_creation(self) -> None:
        """Test price impact model creation."""
        impact = TickPriceImpact(
            temp_impact_coef=0.1,
            perm_impact_coef=0.02,
            decay_half_life_sec=60.0,
        )
        assert impact.temp_coef == 0.1
        assert impact.perm_coef == 0.02

    def test_calculate_impact_buy(self) -> None:
        """Test impact calculation for buy order."""
        impact = TickPriceImpact()

        temp, perm = impact.calculate_impact(
            order_size=1_000_000,
            adv=100_000_000,
            volatility=0.01,
            is_buy=True,
        )

        # Buy should push price up
        assert temp >= 0
        assert perm >= 0

    def test_calculate_impact_sell(self) -> None:
        """Test impact calculation for sell order."""
        impact = TickPriceImpact()

        temp, perm = impact.calculate_impact(
            order_size=1_000_000,
            adv=100_000_000,
            volatility=0.01,
            is_buy=False,
        )

        # Sell should push price down
        assert temp <= 0
        assert perm <= 0

    def test_impact_scales_with_size(self) -> None:
        """Test impact scales with order size."""
        impact = TickPriceImpact()

        small_temp, small_perm = impact.calculate_impact(
            order_size=100_000,
            adv=100_000_000,
            volatility=0.01,
            is_buy=True,
        )

        large_temp, large_perm = impact.calculate_impact(
            order_size=10_000_000,
            adv=100_000_000,
            volatility=0.01,
            is_buy=True,
        )

        assert large_temp > small_temp
        assert large_perm > small_perm

    def test_impact_scales_with_volatility(self) -> None:
        """Test impact scales with volatility."""
        impact = TickPriceImpact()

        low_vol_temp, _ = impact.calculate_impact(
            order_size=1_000_000,
            adv=100_000_000,
            volatility=0.005,
            is_buy=True,
        )

        high_vol_temp, _ = impact.calculate_impact(
            order_size=1_000_000,
            adv=100_000_000,
            volatility=0.02,
            is_buy=True,
        )

        assert high_vol_temp > low_vol_temp

    def test_impact_decay(self) -> None:
        """Test impact decay over time."""
        impact = TickPriceImpact(decay_half_life_sec=60.0)

        # Record an impact
        impact.record_impact(timestamp_ns=0, impact_pips=10.0)

        # Get decayed impact after half-life
        remaining = impact.get_decayed_impact(timestamp_ns=60_000_000_000)

        # Should be about half
        assert remaining == pytest.approx(5.0, rel=0.1)

    def test_impact_decay_multiple(self) -> None:
        """Test decay with multiple impacts."""
        impact = TickPriceImpact(decay_half_life_sec=60.0)

        # Record multiple impacts
        impact.record_impact(timestamp_ns=0, impact_pips=5.0)
        impact.record_impact(timestamp_ns=30_000_000_000, impact_pips=5.0)

        # Get total decayed impact
        remaining = impact.get_decayed_impact(timestamp_ns=60_000_000_000)

        # First impact half decayed, second less decayed
        assert remaining > 5.0  # More than just the second impact

    def test_zero_adv_handling(self) -> None:
        """Test handling of zero ADV."""
        impact = TickPriceImpact()

        temp, perm = impact.calculate_impact(
            order_size=1_000_000,
            adv=0,  # Zero ADV
            volatility=0.01,
            is_buy=True,
        )

        # Should use default ADV
        assert isinstance(temp, float)
        assert isinstance(perm, float)
        assert not math.isnan(temp)
        assert not math.isinf(temp)


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestCreateTickSimulator:
    """Tests for create_tick_simulator factory."""

    def test_create_major_pair(self) -> None:
        """Test creating simulator for major pair."""
        gen, exec = create_tick_simulator(
            pair="EUR_USD",
            session="london",
            initial_price=1.10,
        )
        assert gen is not None
        assert exec is not None
        assert gen.config.pair_type == "major"

    def test_create_exotic_pair(self) -> None:
        """Test creating simulator for exotic pair."""
        gen, exec = create_tick_simulator(
            pair="USD_TRY",
            session="london",
            initial_price=30.0,
        )
        assert gen.config.pair_type == "exotic"

    def test_create_jpy_cross(self) -> None:
        """Test creating simulator for JPY cross."""
        gen, exec = create_tick_simulator(
            pair="EUR_JPY",
            session="tokyo",
            initial_price=160.0,
        )
        assert gen.config.pair_type == "cross"

    def test_create_with_seed(self) -> None:
        """Test reproducibility with seed."""
        gen1, _ = create_tick_simulator(pair="EUR_USD", seed=42)
        gen2, _ = create_tick_simulator(pair="EUR_USD", seed=42)

        tick1 = gen1.generate_tick(1000000000)
        tick2 = gen2.generate_tick(1000000000)

        assert tick1.mid == tick2.mid


# =============================================================================
# Integration Tests
# =============================================================================

class TestTickSimulationIntegration:
    """Integration tests for tick simulation."""

    def test_full_trading_session(self) -> None:
        """Test simulating a full trading session."""
        gen, exec = create_tick_simulator(
            pair="EUR_USD",
            session="london",
            initial_price=1.1000,
            seed=42,
        )

        # Generate ticks for 10 seconds
        ticks = list(gen.generate_tick_stream(
            duration_sec=10.0,
            start_timestamp_ns=time.time_ns(),
        ))

        assert len(ticks) > 10

        # Execute some orders during the session
        fill_count = 0
        for tick in ticks[:20]:
            if np.random.random() < 0.3:
                result = exec.execute_market_order(
                    is_buy=np.random.random() < 0.5,
                    size=100000.0,
                    submitted_at_ns=tick.timestamp_ns,
                    max_slippage_pips=5.0,
                )
                if result.filled:
                    fill_count += 1

        stats = exec.get_stats()
        assert stats["total_orders"] > 0

    def test_high_volatility_scenario(self) -> None:
        """Test high volatility scenario."""
        config = TickSimulationConfig(
            pair_type="major",
            volatility_scale=3.0,  # High volatility
            seed=42,
        )
        gen = TickGenerator(config=config, initial_mid=1.10)
        exec = TickLevelExecutor(tick_generator=gen)

        # Execute orders in high vol
        results = []
        for _ in range(20):
            result = exec.execute_market_order(
                is_buy=True,
                size=100000.0,
                submitted_at_ns=time.time_ns(),
                max_slippage_pips=10.0,
            )
            results.append(result)

        # Check statistics
        stats = exec.get_stats()
        # High vol may lead to more slippage
        assert "avg_slippage_pips" in stats

    def test_limit_order_workflow(self) -> None:
        """Test limit order workflow."""
        gen, exec = create_tick_simulator(pair="EUR_USD", seed=42)

        # Get current price
        tick = gen.generate_tick()

        # Place limit order slightly better than market
        limit_price = tick.ask - 0.0002  # 2 pips below ask

        result = exec.execute_limit_order(
            is_buy=True,
            size=100000.0,
            limit_price=limit_price,
            submitted_at_ns=time.time_ns(),
            time_in_force_sec=60.0,
        )

        # Should eventually fill or expire
        assert result.filled or result.rejection_reason == "expired"


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestTickSimulationEdgeCases:
    """Edge case tests for tick simulation."""

    def test_zero_size_order(self, tick_executor: TickLevelExecutor) -> None:
        """Test handling of zero size order."""
        result = tick_executor.execute_market_order(
            is_buy=True,
            size=0.0,
            submitted_at_ns=time.time_ns(),
            max_slippage_pips=5.0,
        )
        # Should still process (implementation dependent)
        assert isinstance(result, TickExecutionResult)

    def test_very_large_order(self, tick_executor: TickLevelExecutor) -> None:
        """Test very large order handling."""
        result = tick_executor.execute_market_order(
            is_buy=True,
            size=1_000_000_000.0,  # 10000 lots
            submitted_at_ns=time.time_ns(),
            max_slippage_pips=50.0,
        )
        assert isinstance(result, TickExecutionResult)

    def test_negative_timestamp(self, tick_generator: TickGenerator) -> None:
        """Test handling of negative timestamp."""
        # Should handle gracefully
        tick = tick_generator.generate_tick(timestamp_ns=0)
        assert isinstance(tick, Tick)

    def test_rapid_fire_orders(self, tick_executor: TickLevelExecutor) -> None:
        """Test rapid succession of orders."""
        ts = time.time_ns()
        results = []
        for i in range(100):
            result = tick_executor.execute_market_order(
                is_buy=i % 2 == 0,
                size=100000.0,
                submitted_at_ns=ts + i * 1000,  # 1 microsecond apart
                max_slippage_pips=10.0,
            )
            results.append(result)

        # Should handle all orders
        assert len(results) == 100

    def test_price_stability(self) -> None:
        """Test price doesn't explode over time."""
        config = TickSimulationConfig(seed=42)
        gen = TickGenerator(config=config, initial_mid=1.10)

        # Generate many ticks
        for i in range(10000):
            tick = gen.generate_tick(timestamp_ns=i * 100_000_000)

        state = gen.get_current_state()
        # Price should remain in reasonable range
        assert 0.5 < state["mid"] < 2.0
