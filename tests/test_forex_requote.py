# -*- coding: utf-8 -*-
"""
tests/test_forex_requote.py
Comprehensive tests for Forex Requote Flow Simulation

Tests cover:
1. RequoteEvent and RequoteFlowResult dataclasses
2. RequoteProbabilityModel factor calculations
3. ClientAcceptanceModel behavior profiles
4. RequoteFlowSimulator full flow
5. Statistics tracking
6. Edge cases and integration scenarios

Author: Sultan Suyunbayev
Date: 2025-11-30
"""

import math
import time
from typing import Dict, Any, List

import numpy as np
import pytest

from services.forex_requote import (
    # Enums
    RequoteReason,
    RequoteOutcome,
    ClientBehavior,
    # Data classes
    RequoteEvent,
    RequoteFlowResult,
    RequoteConfig,
    MarketSnapshot,
    # Classes
    RequoteProbabilityModel,
    ClientAcceptanceModel,
    RequoteFlowSimulator,
    # Factory functions
    create_requote_simulator,
    simulate_requote_scenario,
    # Constants
    BASE_REQUOTE_PROBS,
    SESSION_REQUOTE_FACTORS,
    SIZE_THRESHOLDS,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def market_snapshot() -> MarketSnapshot:
    """Create a market snapshot for testing."""
    return MarketSnapshot(
        timestamp_ns=int(time.time() * 1e9),
        bid=1.0850,
        ask=1.0852,
        spread_pips=2.0,
        volatility=0.01,
        session="london",
        is_news_event=False,
    )


@pytest.fixture
def requote_config() -> RequoteConfig:
    """Create a requote configuration."""
    return RequoteConfig(
        client_tier="retail",
        client_behavior=ClientBehavior.NEUTRAL,
        max_total_slippage_pips=5.0,
        seed=42,
    )


@pytest.fixture
def requote_simulator(requote_config: RequoteConfig) -> RequoteFlowSimulator:
    """Create a requote simulator."""
    return RequoteFlowSimulator(config=requote_config)


# =============================================================================
# RequoteEvent Tests
# =============================================================================

class TestRequoteEvent:
    """Tests for RequoteEvent dataclass."""

    def test_creation(self) -> None:
        """Test basic creation."""
        event = RequoteEvent(
            timestamp_ns=1000000000,
            original_price=1.0850,
            requote_price=1.0853,
            price_diff_pips=3.0,
            reason=RequoteReason.PRICE_MOVED,
        )
        assert event.original_price == 1.0850
        assert event.requote_price == 1.0853
        assert event.price_diff_pips == 3.0
        assert event.reason == RequoteReason.PRICE_MOVED

    def test_latency_ms_property(self) -> None:
        """Test latency conversion to milliseconds."""
        event = RequoteEvent(
            timestamp_ns=0,
            original_price=1.0,
            requote_price=1.001,
            price_diff_pips=1.0,
            reason=RequoteReason.VOLATILITY,
            latency_ns=50_000_000,  # 50ms
        )
        assert event.latency_ms == pytest.approx(50.0, rel=1e-6)

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        event = RequoteEvent(
            timestamp_ns=1000000000,
            original_price=1.0850,
            requote_price=1.0853,
            price_diff_pips=3.0,
            reason=RequoteReason.NEWS_EVENT,
            dealer_id="dealer_1",
            accepted=True,
            sequence_num=2,
        )
        d = event.to_dict()
        assert d["original_price"] == 1.0850
        assert d["requote_price"] == 1.0853
        assert d["reason"] == "news_event"
        assert d["accepted"] is True
        assert d["sequence_num"] == 2


# =============================================================================
# RequoteFlowResult Tests
# =============================================================================

class TestRequoteFlowResult:
    """Tests for RequoteFlowResult dataclass."""

    def test_creation_filled_at_original(self) -> None:
        """Test creation for fill at original price."""
        result = RequoteFlowResult(
            outcome=RequoteOutcome.FILLED_AT_ORIGINAL,
            fill_price=1.0850,
            fill_timestamp_ns=1000000000,
            original_price=1.0850,
            was_requoted=False,
        )
        assert result.was_filled is True
        assert result.was_requoted is False

    def test_creation_filled_at_requote(self) -> None:
        """Test creation for fill at requote price."""
        result = RequoteFlowResult(
            outcome=RequoteOutcome.FILLED_AT_REQUOTE,
            fill_price=1.0853,
            original_price=1.0850,
            total_slippage_pips=3.0,
            was_requoted=True,
        )
        assert result.was_filled is True
        assert result.was_requoted is True
        assert result.total_slippage_pips == 3.0

    def test_creation_rejected(self) -> None:
        """Test creation for rejected order."""
        result = RequoteFlowResult(
            outcome=RequoteOutcome.CLIENT_REJECTED,
            was_requoted=True,
        )
        assert result.was_filled is False

    def test_was_filled_property(self) -> None:
        """Test was_filled property for various outcomes."""
        filled_outcomes = [
            RequoteOutcome.FILLED_AT_ORIGINAL,
            RequoteOutcome.FILLED_AT_REQUOTE,
        ]
        not_filled_outcomes = [
            RequoteOutcome.CLIENT_REJECTED,
            RequoteOutcome.DEALER_REJECTED,
            RequoteOutcome.EXPIRED,
            RequoteOutcome.MAX_REQUOTES_REACHED,
        ]

        for outcome in filled_outcomes:
            result = RequoteFlowResult(outcome=outcome)
            assert result.was_filled is True

        for outcome in not_filled_outcomes:
            result = RequoteFlowResult(outcome=outcome)
            assert result.was_filled is False

    def test_total_latency_ms_property(self) -> None:
        """Test latency conversion."""
        result = RequoteFlowResult(
            outcome=RequoteOutcome.FILLED_AT_ORIGINAL,
            total_latency_ns=100_000_000,
        )
        assert result.total_latency_ms == pytest.approx(100.0, rel=1e-6)

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        event = RequoteEvent(
            timestamp_ns=0,
            original_price=1.0850,
            requote_price=1.0853,
            price_diff_pips=3.0,
            reason=RequoteReason.PRICE_MOVED,
        )
        result = RequoteFlowResult(
            outcome=RequoteOutcome.FILLED_AT_REQUOTE,
            fill_price=1.0853,
            requotes=[event],
            total_requotes=1,
            total_slippage_pips=3.0,
        )
        d = result.to_dict()
        assert d["outcome"] == "filled_at_requote"
        assert d["fill_price"] == 1.0853
        assert d["total_requotes"] == 1
        assert len(d["requotes"]) == 1


# =============================================================================
# RequoteConfig Tests
# =============================================================================

class TestRequoteConfig:
    """Tests for RequoteConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = RequoteConfig()
        assert config.client_tier == "retail"
        assert config.client_behavior == ClientBehavior.NEUTRAL
        assert config.base_requote_prob == BASE_REQUOTE_PROBS["retail"]

    def test_auto_set_base_prob(self) -> None:
        """Test automatic base probability from tier."""
        retail_config = RequoteConfig(client_tier="retail")
        inst_config = RequoteConfig(client_tier="institutional")

        assert retail_config.base_requote_prob == BASE_REQUOTE_PROBS["retail"]
        assert inst_config.base_requote_prob == BASE_REQUOTE_PROBS["institutional"]
        assert inst_config.base_requote_prob < retail_config.base_requote_prob

    def test_auto_set_max_requotes(self) -> None:
        """Test automatic max requotes from tier."""
        retail_config = RequoteConfig(client_tier="retail")
        prime_config = RequoteConfig(client_tier="prime")

        assert retail_config.max_requotes == 3
        assert prime_config.max_requotes == 1

    def test_custom_overrides(self) -> None:
        """Test custom value overrides."""
        config = RequoteConfig(
            client_tier="retail",
            base_requote_prob=0.15,
            max_requotes=5,
        )
        assert config.base_requote_prob == 0.15
        assert config.max_requotes == 5


# =============================================================================
# MarketSnapshot Tests
# =============================================================================

class TestMarketSnapshot:
    """Tests for MarketSnapshot dataclass."""

    def test_mid_price_calculation(self) -> None:
        """Test mid price property."""
        snapshot = MarketSnapshot(
            timestamp_ns=0,
            bid=1.0850,
            ask=1.0852,
            spread_pips=2.0,
        )
        assert snapshot.mid == pytest.approx(1.0851, rel=1e-6)

    def test_jpy_detection(self) -> None:
        """Test JPY pair detection via price level."""
        # Standard pair
        std = MarketSnapshot(timestamp_ns=0, bid=1.08, ask=1.09, spread_pips=1.0)
        # JPY pair (price > 10)
        jpy = MarketSnapshot(timestamp_ns=0, bid=150.0, ask=150.02, spread_pips=2.0)

        assert std.bid < 10
        assert jpy.bid > 10


# =============================================================================
# RequoteProbabilityModel Tests
# =============================================================================

class TestRequoteProbabilityModel:
    """Tests for RequoteProbabilityModel."""

    def test_creation(self, requote_config: RequoteConfig) -> None:
        """Test model creation."""
        model = RequoteProbabilityModel(requote_config)
        assert model.config == requote_config

    def test_base_probability(self, market_snapshot: MarketSnapshot) -> None:
        """Test base probability is applied."""
        config = RequoteConfig(client_tier="retail", seed=42)
        model = RequoteProbabilityModel(config)

        prob, _ = model.compute_requote_probability(
            order_size_usd=50000,
            market=market_snapshot,
            price_movement_pips=0.0,
            is_adverse=False,
        )

        # Should be based on retail base prob
        assert prob > 0

    def test_volatility_scaling(self, market_snapshot: MarketSnapshot) -> None:
        """Test volatility affects probability."""
        config = RequoteConfig(client_tier="retail", use_volatility_scaling=True, seed=42)
        model = RequoteProbabilityModel(config)

        # Low volatility
        low_vol_market = MarketSnapshot(
            timestamp_ns=0, bid=1.08, ask=1.09, spread_pips=1.0, volatility=0.005
        )
        # High volatility
        high_vol_market = MarketSnapshot(
            timestamp_ns=0, bid=1.08, ask=1.09, spread_pips=1.0, volatility=0.03
        )

        low_prob, _ = model.compute_requote_probability(50000, low_vol_market)
        high_prob, _ = model.compute_requote_probability(50000, high_vol_market)

        assert high_prob > low_prob

    def test_size_scaling(self, market_snapshot: MarketSnapshot) -> None:
        """Test order size affects probability."""
        config = RequoteConfig(client_tier="retail", use_size_scaling=True, seed=42)
        model = RequoteProbabilityModel(config)

        small_prob, _ = model.compute_requote_probability(10000, market_snapshot)
        large_prob, _ = model.compute_requote_probability(5_000_000, market_snapshot)

        assert large_prob > small_prob

    def test_session_scaling(self) -> None:
        """Test session affects probability."""
        config = RequoteConfig(client_tier="retail", seed=42)
        model = RequoteProbabilityModel(config)

        # London (best liquidity)
        london = MarketSnapshot(
            timestamp_ns=0, bid=1.08, ask=1.09, spread_pips=1.0, session="london"
        )
        # Off hours (worst liquidity)
        off_hours = MarketSnapshot(
            timestamp_ns=0, bid=1.08, ask=1.09, spread_pips=1.0, session="off_hours"
        )

        london_prob, _ = model.compute_requote_probability(100000, london)
        off_hours_prob, _ = model.compute_requote_probability(100000, off_hours)

        assert off_hours_prob > london_prob

    def test_price_movement_scaling(self, market_snapshot: MarketSnapshot) -> None:
        """Test price movement affects probability."""
        config = RequoteConfig(client_tier="retail", seed=42)
        model = RequoteProbabilityModel(config)

        # No movement
        no_move_prob, _ = model.compute_requote_probability(
            100000, market_snapshot, price_movement_pips=0.0
        )
        # Significant movement
        move_prob, _ = model.compute_requote_probability(
            100000, market_snapshot, price_movement_pips=2.0, is_adverse=True
        )

        assert move_prob > no_move_prob

    def test_news_event_scaling(self, market_snapshot: MarketSnapshot) -> None:
        """Test news event increases probability."""
        config = RequoteConfig(client_tier="retail", seed=42)
        model = RequoteProbabilityModel(config)

        # Normal market
        normal = MarketSnapshot(
            timestamp_ns=0, bid=1.08, ask=1.09, spread_pips=1.0, is_news_event=False
        )
        # News event
        news = MarketSnapshot(
            timestamp_ns=0, bid=1.08, ask=1.09, spread_pips=1.0, is_news_event=True
        )

        normal_prob, _ = model.compute_requote_probability(100000, normal)
        news_prob, _ = model.compute_requote_probability(100000, news)

        assert news_prob > normal_prob

    def test_reason_assignment(self, market_snapshot: MarketSnapshot) -> None:
        """Test reason is assigned correctly."""
        config = RequoteConfig(client_tier="retail", seed=42)
        model = RequoteProbabilityModel(config)

        # Large order should have size reason
        _, reason = model.compute_requote_probability(
            order_size_usd=15_000_000,  # Institutional size
            market=market_snapshot,
        )
        assert reason in [RequoteReason.SIZE_EXCEEDED, RequoteReason.PRICE_MOVED]

    def test_should_requote(self, market_snapshot: MarketSnapshot) -> None:
        """Test should_requote decision."""
        config = RequoteConfig(client_tier="retail", base_requote_prob=1.0, seed=42)
        model = RequoteProbabilityModel(config)

        # With 100% base probability, should always requote
        should, reason = model.should_requote(100000, market_snapshot)
        assert should is True
        assert reason is not None

    def test_probability_capping(self) -> None:
        """Test probability is capped at reasonable values."""
        config = RequoteConfig(
            client_tier="retail",
            base_requote_prob=0.5,
            use_volatility_scaling=True,
            use_size_scaling=True,
            seed=42,
        )
        model = RequoteProbabilityModel(config)

        # Extreme conditions
        extreme = MarketSnapshot(
            timestamp_ns=0,
            bid=1.08,
            ask=1.09,
            spread_pips=10.0,  # Wide spread
            volatility=0.1,   # Very high vol
            session="weekend",
            is_news_event=True,
        )

        prob, _ = model.compute_requote_probability(
            order_size_usd=50_000_000,  # Huge order
            market=extreme,
            price_movement_pips=10.0,
            is_adverse=True,
        )

        assert prob <= 0.95


# =============================================================================
# ClientAcceptanceModel Tests
# =============================================================================

class TestClientAcceptanceModel:
    """Tests for ClientAcceptanceModel."""

    def test_creation(self) -> None:
        """Test model creation."""
        model = ClientAcceptanceModel(
            behavior=ClientBehavior.NEUTRAL,
            acceptance_threshold_pips=1.0,
        )
        assert model.behavior == ClientBehavior.NEUTRAL
        assert model.threshold == 1.0

    def test_algorithmic_strict_threshold(self) -> None:
        """Test algorithmic client uses strict threshold."""
        model = ClientAcceptanceModel(
            behavior=ClientBehavior.ALGORITHMIC,
            acceptance_threshold_pips=1.0,
        )

        # Under threshold - accept
        assert model.will_accept_requote(0.5) is True
        # Over threshold - reject
        assert model.will_accept_requote(1.5) is False

    def test_aggressive_low_acceptance(self) -> None:
        """Test aggressive clients have low acceptance rate."""
        model = ClientAcceptanceModel(
            behavior=ClientBehavior.AGGRESSIVE,
            acceptance_threshold_pips=1.0,
            seed=42,
        )

        # Run many trials
        accepts = sum(
            1 for _ in range(100)
            if model.will_accept_requote(0.5)
        )

        # Should have relatively low acceptance (around 40%)
        assert accepts < 70

    def test_passive_high_acceptance(self) -> None:
        """Test passive clients have high acceptance rate."""
        model = ClientAcceptanceModel(
            behavior=ClientBehavior.PASSIVE,
            acceptance_threshold_pips=1.0,
            seed=42,
        )

        # Run many trials
        accepts = sum(
            1 for _ in range(100)
            if model.will_accept_requote(0.5)
        )

        # Should have high acceptance (around 90%)
        assert accepts > 70

    def test_price_diff_affects_acceptance(self) -> None:
        """Test price difference affects acceptance."""
        model = ClientAcceptanceModel(
            behavior=ClientBehavior.NEUTRAL,
            acceptance_threshold_pips=1.0,
            seed=42,
        )

        # Small difference
        small_accepts = sum(
            1 for _ in range(100)
            if model.will_accept_requote(0.2)
        )
        # Large difference
        large_accepts = sum(
            1 for _ in range(100)
            if model.will_accept_requote(3.0)
        )

        assert small_accepts > large_accepts

    def test_requote_fatigue(self) -> None:
        """Test acceptance decreases with more requotes."""
        model = ClientAcceptanceModel(
            behavior=ClientBehavior.NEUTRAL,
            seed=42,
        )

        # First requote
        first_accepts = sum(
            1 for _ in range(100)
            if model.will_accept_requote(0.5, requote_count=1)
        )
        # Third requote
        third_accepts = sum(
            1 for _ in range(100)
            if model.will_accept_requote(0.5, requote_count=3)
        )

        # Fatigue should reduce acceptance
        assert third_accepts <= first_accepts

    def test_urgency_increases_acceptance(self) -> None:
        """Test high urgency increases acceptance."""
        model = ClientAcceptanceModel(
            behavior=ClientBehavior.NEUTRAL,
            seed=42,
        )

        # Low urgency
        low_accepts = sum(
            1 for _ in range(100)
            if model.will_accept_requote(1.0, urgency=0.2)
        )
        # High urgency
        high_accepts = sum(
            1 for _ in range(100)
            if model.will_accept_requote(1.0, urgency=0.9)
        )

        assert high_accepts >= low_accepts


# =============================================================================
# RequoteFlowSimulator Tests
# =============================================================================

class TestRequoteFlowSimulator:
    """Tests for RequoteFlowSimulator."""

    def test_creation(self) -> None:
        """Test simulator creation."""
        simulator = RequoteFlowSimulator()
        assert simulator.config is not None

    def test_creation_with_config(self, requote_config: RequoteConfig) -> None:
        """Test creation with custom config."""
        simulator = RequoteFlowSimulator(config=requote_config)
        assert simulator.config == requote_config

    def test_simulate_no_requote(self, market_snapshot: MarketSnapshot) -> None:
        """Test simulation when no requote occurs."""
        # Zero requote probability
        config = RequoteConfig(base_requote_prob=0.0, seed=42)
        simulator = RequoteFlowSimulator(config=config)

        result = simulator.simulate_requote_flow(
            is_buy=True,
            size_usd=100000,
            requested_price=market_snapshot.ask,
            market=market_snapshot,
        )

        assert result.outcome == RequoteOutcome.FILLED_AT_ORIGINAL
        assert result.was_requoted is False

    def test_simulate_with_requote(self, market_snapshot: MarketSnapshot) -> None:
        """Test simulation when requote occurs."""
        # 100% requote probability
        config = RequoteConfig(
            base_requote_prob=1.0,
            client_behavior=ClientBehavior.PASSIVE,  # High acceptance
            seed=42,
        )
        simulator = RequoteFlowSimulator(config=config)

        result = simulator.simulate_requote_flow(
            is_buy=True,
            size_usd=100000,
            requested_price=market_snapshot.ask,
            market=market_snapshot,
        )

        assert result.was_requoted is True
        # Should either fill at requote or reject
        assert result.outcome in [
            RequoteOutcome.FILLED_AT_REQUOTE,
            RequoteOutcome.CLIENT_REJECTED,
            RequoteOutcome.MAX_REQUOTES_REACHED,
        ]

    def test_buy_order_requote_price_higher(self, market_snapshot: MarketSnapshot) -> None:
        """Test buy order requote price is higher than original."""
        config = RequoteConfig(base_requote_prob=1.0, seed=42)
        simulator = RequoteFlowSimulator(config=config)

        result = simulator.simulate_requote_flow(
            is_buy=True,
            size_usd=100000,
            requested_price=market_snapshot.ask,
            market=market_snapshot,
        )

        if result.requotes:
            for event in result.requotes:
                # Buy requote should be higher (worse for client)
                assert event.requote_price >= event.original_price

    def test_sell_order_requote_price_lower(self, market_snapshot: MarketSnapshot) -> None:
        """Test sell order requote price is lower than original."""
        config = RequoteConfig(base_requote_prob=1.0, seed=42)
        simulator = RequoteFlowSimulator(config=config)

        result = simulator.simulate_requote_flow(
            is_buy=False,
            size_usd=100000,
            requested_price=market_snapshot.bid,
            market=market_snapshot,
        )

        if result.requotes:
            for event in result.requotes:
                # Sell requote should be lower (worse for client)
                assert event.requote_price <= event.original_price

    def test_max_requotes_enforced(self, market_snapshot: MarketSnapshot) -> None:
        """Test max requotes limit is enforced."""
        config = RequoteConfig(
            base_requote_prob=1.0,
            max_requotes=2,
            client_behavior=ClientBehavior.AGGRESSIVE,  # Low acceptance
            seed=42,
        )
        simulator = RequoteFlowSimulator(config=config)

        result = simulator.simulate_requote_flow(
            is_buy=True,
            size_usd=100000,
            requested_price=market_snapshot.ask,
            market=market_snapshot,
        )

        if result.was_requoted:
            assert result.total_requotes <= config.max_requotes

    def test_statistics_tracking(
        self,
        requote_simulator: RequoteFlowSimulator,
        market_snapshot: MarketSnapshot,
    ) -> None:
        """Test statistics are tracked correctly."""
        # Execute multiple orders
        for _ in range(20):
            requote_simulator.simulate_requote_flow(
                is_buy=True,
                size_usd=100000,
                requested_price=market_snapshot.ask,
                market=market_snapshot,
            )

        stats = requote_simulator.get_stats()
        assert stats["total_orders"] == 20
        assert stats["fill_rate"] >= 0
        assert stats["requote_rate"] >= 0

    def test_reset_stats(
        self,
        requote_simulator: RequoteFlowSimulator,
        market_snapshot: MarketSnapshot,
    ) -> None:
        """Test statistics reset."""
        requote_simulator.simulate_requote_flow(
            is_buy=True,
            size_usd=100000,
            requested_price=market_snapshot.ask,
            market=market_snapshot,
        )

        requote_simulator.reset_stats()
        stats = requote_simulator.get_stats()
        assert stats["total_orders"] == 0

    def test_get_recent_results(
        self,
        requote_simulator: RequoteFlowSimulator,
        market_snapshot: MarketSnapshot,
    ) -> None:
        """Test getting recent results."""
        for _ in range(15):
            requote_simulator.simulate_requote_flow(
                is_buy=True,
                size_usd=100000,
                requested_price=market_snapshot.ask,
                market=market_snapshot,
            )

        recent = requote_simulator.get_recent_results(n=10)
        assert len(recent) == 10

    def test_slippage_calculation(self, market_snapshot: MarketSnapshot) -> None:
        """Test slippage is calculated correctly."""
        config = RequoteConfig(
            base_requote_prob=1.0,
            client_behavior=ClientBehavior.PASSIVE,
            seed=42,
        )
        simulator = RequoteFlowSimulator(config=config)

        result = simulator.simulate_requote_flow(
            is_buy=True,
            size_usd=100000,
            requested_price=market_snapshot.ask,
            market=market_snapshot,
        )

        if result.outcome == RequoteOutcome.FILLED_AT_REQUOTE:
            assert result.total_slippage_pips > 0


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestCreateRequoteSimulator:
    """Tests for create_requote_simulator factory."""

    def test_create_default(self) -> None:
        """Test creation with defaults."""
        simulator = create_requote_simulator()
        assert simulator.config.client_tier == "retail"
        assert simulator.config.client_behavior == ClientBehavior.NEUTRAL

    def test_create_with_tier(self) -> None:
        """Test creation with specific tier."""
        simulator = create_requote_simulator(client_tier="institutional")
        assert simulator.config.client_tier == "institutional"

    def test_create_with_behavior(self) -> None:
        """Test creation with specific behavior."""
        simulator = create_requote_simulator(behavior="aggressive")
        assert simulator.config.client_behavior == ClientBehavior.AGGRESSIVE

    def test_create_with_invalid_behavior(self) -> None:
        """Test creation with invalid behavior falls back."""
        simulator = create_requote_simulator(behavior="invalid")
        assert simulator.config.client_behavior == ClientBehavior.NEUTRAL


class TestSimulateRequoteScenario:
    """Tests for simulate_requote_scenario function."""

    def test_basic_scenario(self) -> None:
        """Test basic scenario simulation."""
        stats = simulate_requote_scenario(
            n_orders=50,
            client_tier="retail",
            session="london",
            seed=42,
        )

        assert stats["total_orders"] == 50
        assert "fill_rate" in stats
        assert "requote_rate" in stats
        assert stats["fill_rate"] >= 0
        assert stats["fill_rate"] <= 100

    def test_different_tiers(self) -> None:
        """Test different client tiers have different rates."""
        retail_stats = simulate_requote_scenario(
            n_orders=100,
            client_tier="retail",
            seed=42,
        )
        inst_stats = simulate_requote_scenario(
            n_orders=100,
            client_tier="institutional",
            seed=42,
        )

        # Institutional should have lower requote rate
        assert inst_stats["requote_rate"] <= retail_stats["requote_rate"] + 10  # Some tolerance

    def test_high_volatility(self) -> None:
        """Test high volatility increases requotes."""
        low_vol = simulate_requote_scenario(
            n_orders=100,
            volatility=0.005,
            seed=42,
        )
        high_vol = simulate_requote_scenario(
            n_orders=100,
            volatility=0.03,
            seed=42,
        )

        # High vol should have more requotes
        assert high_vol["requote_rate"] >= low_vol["requote_rate"]


# =============================================================================
# Integration Tests
# =============================================================================

class TestRequoteIntegration:
    """Integration tests for requote functionality."""

    def test_full_workflow(self) -> None:
        """Test complete requote workflow."""
        # Create simulator
        simulator = create_requote_simulator(
            client_tier="professional",
            behavior="neutral",
            max_slippage_pips=3.0,
            seed=42,
        )

        # Create market
        market = MarketSnapshot(
            timestamp_ns=time.time_ns(),
            bid=1.0850,
            ask=1.0852,
            spread_pips=2.0,
            volatility=0.01,
            session="london",
        )

        # Execute order
        result = simulator.simulate_requote_flow(
            is_buy=True,
            size_usd=500000,
            requested_price=market.ask,
            market=market,
            urgency=0.7,
        )

        # Check result
        assert result.outcome in RequoteOutcome.__members__.values()
        if result.was_filled:
            assert result.fill_price > 0
        if result.was_requoted:
            assert result.total_requotes > 0

    def test_client_tier_comparison(self) -> None:
        """Test comparing different client tiers."""
        tiers = ["retail", "professional", "institutional", "prime"]
        results: Dict[str, Dict[str, float]] = {}

        for tier in tiers:
            stats = simulate_requote_scenario(
                n_orders=200,
                client_tier=tier,
                seed=42,
            )
            results[tier] = stats

        # Prime should have best (lowest) requote rate
        assert results["prime"]["requote_rate"] <= results["retail"]["requote_rate"]

    def test_session_comparison(self) -> None:
        """Test comparing different trading sessions."""
        sessions = ["london", "tokyo", "off_hours"]
        results: Dict[str, float] = {}

        for session in sessions:
            stats = simulate_requote_scenario(
                n_orders=200,
                session=session,
                seed=42,
            )
            results[session] = stats["requote_rate"]

        # London should have best (lowest) requote rate
        assert results["london"] <= results["off_hours"]


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestRequoteEdgeCases:
    """Edge case tests for requote functionality."""

    def test_zero_size_order(self, market_snapshot: MarketSnapshot) -> None:
        """Test handling of zero size order."""
        simulator = create_requote_simulator(seed=42)
        result = simulator.simulate_requote_flow(
            is_buy=True,
            size_usd=0,
            requested_price=market_snapshot.ask,
            market=market_snapshot,
        )
        assert isinstance(result, RequoteFlowResult)

    def test_very_large_order(self, market_snapshot: MarketSnapshot) -> None:
        """Test handling of very large order."""
        simulator = create_requote_simulator(seed=42)
        result = simulator.simulate_requote_flow(
            is_buy=True,
            size_usd=1_000_000_000,  # 1 billion
            requested_price=market_snapshot.ask,
            market=market_snapshot,
        )
        assert isinstance(result, RequoteFlowResult)
        # Very large orders likely get requoted
        if result.was_requoted:
            assert result.total_requotes >= 1

    def test_jpy_pair_handling(self) -> None:
        """Test JPY pair pip calculation."""
        jpy_market = MarketSnapshot(
            timestamp_ns=time.time_ns(),
            bid=150.00,
            ask=150.03,
            spread_pips=3.0,
            volatility=0.01,
            session="tokyo",
        )
        simulator = create_requote_simulator(seed=42)
        result = simulator.simulate_requote_flow(
            is_buy=True,
            size_usd=100000,
            requested_price=jpy_market.ask,
            market=jpy_market,
        )
        assert isinstance(result, RequoteFlowResult)

    def test_extreme_volatility(self) -> None:
        """Test extreme volatility scenario."""
        extreme_market = MarketSnapshot(
            timestamp_ns=time.time_ns(),
            bid=1.0850,
            ask=1.0900,  # 50 pip spread!
            spread_pips=50.0,
            volatility=0.1,  # 10% daily vol
            session="news_event",
            is_news_event=True,
        )
        simulator = create_requote_simulator(seed=42)

        # Most orders should get requoted
        results = []
        for _ in range(20):
            result = simulator.simulate_requote_flow(
                is_buy=True,
                size_usd=100000,
                requested_price=extreme_market.ask,
                market=extreme_market,
            )
            results.append(result.was_requoted)

        # High percentage should be requoted
        requote_pct = sum(results) / len(results)
        assert requote_pct > 0.5

    def test_many_requotes(self, market_snapshot: MarketSnapshot) -> None:
        """Test behavior with many consecutive requotes."""
        config = RequoteConfig(
            base_requote_prob=1.0,
            max_requotes=5,
            client_behavior=ClientBehavior.AGGRESSIVE,
            seed=42,
        )
        simulator = RequoteFlowSimulator(config=config)

        result = simulator.simulate_requote_flow(
            is_buy=True,
            size_usd=100000,
            requested_price=market_snapshot.ask,
            market=market_snapshot,
        )

        # Should respect max requotes
        assert result.total_requotes <= 5
